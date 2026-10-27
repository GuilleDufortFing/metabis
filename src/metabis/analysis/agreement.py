"""Agreement between two comparison results: their effects and their calls.

Comparing effect sizes and significant calls between two analyses is how one
tool is checked against another on the same cells (Amiss, Jabbari and Lum,
bioRxiv 2024) and a finding is checked in another cohort (Xu et al., Cell Rep Med
2026). Two subpopulations' responses to one condition are set side by side the
same way to separate shared from specific responses (Crowell et al., Nat Commun
2020). The view is descriptive: it does not test whether two responses differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .comparison_artifact import GroupComparisonArtifact
from .statistical_analysis import StatisticalAnalysis

__all__ = [
    "AGREEMENT_CLASSES",
    "Agreement",
    "ResultEffects",
    "compare_results",
]

AGREEMENT_CLASSES: dict[str, str] = {
    "both": "Significant in both, same sign",
    "opposite": "Significant in both, opposite signs",
    "first": "Significant in the first only",
    "second": "Significant in the second only",
    "neither": "Significant in neither",
}
_THRESHOLD_COLUMNS = {
    "pval": "adjusted_pval",
    "cohens_d": "cohens_d",
    "log2_fc": "log2_fc",
    "mean_difference": "mean_difference",
}


@dataclass(frozen=True)
class ResultEffects:
    """One result's Cohen's d and significant calls, by feature ID."""

    label: str
    effects: pd.Series
    significant: frozenset[str]

    @classmethod
    def from_artifact(
        cls, label: str, artifact: GroupComparisonArtifact, *, for_genes: bool = False
    ) -> "ResultEffects":
        """Read a stored result whose significance is already decided."""
        results = artifact.get_test_results(for_genes=for_genes)
        if results is None:
            raise ValueError(f"{label} has no {'gene' if for_genes else 'reaction'} results")
        effects = results["cohens_d"].astype(float)
        effects.index = effects.index.map(str)
        return cls(
            label=label,
            effects=effects,
            significant=frozenset(artifact.get_significant_elements(for_genes=for_genes)),
        )

    @classmethod
    def from_table(
        cls, label: str, table: pd.DataFrame, thresholds: Mapping[str, float | None]
    ) -> "ResultEffects":
        """Read an exported table and decide its calls with ``thresholds``.

        The table needs ``feature_id``, ``cohens_d`` and ``adjusted_pval``, as a
        Metabis export has. Both results are then judged by the same thresholds.
        """
        missing = [
            column for column in ("feature_id", "cohens_d", "adjusted_pval") if column not in table
        ]
        if missing:
            raise ValueError(f"The file has no column {', '.join(missing)}")
        frame = table.drop_duplicates("feature_id").copy()
        frame.index = frame["feature_id"].astype(str)
        for column in _THRESHOLD_COLUMNS.values():
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        unusable = [
            metric
            for metric, value in thresholds.items()
            if value is not None and _THRESHOLD_COLUMNS[metric] not in frame
        ]
        if unusable:
            raise ValueError(
                "The file has no column for the threshold on "
                + ", ".join(_THRESHOLD_COLUMNS[metric] for metric in unusable)
            )
        mask, _info = StatisticalAnalysis().determine_significant_elements(
            frame,
            pval_threshold=thresholds.get("pval"),
            cohens_d_threshold=thresholds.get("cohens_d"),
            log2_fc_threshold=thresholds.get("log2_fc"),
            mean_difference_threshold=thresholds.get("mean_difference"),
        )
        return cls(
            label=label,
            effects=frame["cohens_d"].astype(float),
            significant=frozenset(frame.index[mask.fillna(False).to_numpy(dtype=bool)]),
        )


@dataclass(frozen=True)
class Agreement:
    """Two results feature by feature, with their rank correlation and counts."""

    first: str
    second: str
    table: pd.DataFrame
    spearman: float
    shared: int
    counts: Mapping[str, int]


def compare_results(first: ResultEffects, second: ResultEffects) -> Agreement:
    """Join two results on their shared features and classify each feature.

    Spearman's rho is computed over features with a finite effect in both; it
    measures whether the effects line up, not whether they are equal.
    """
    shared = first.effects.index.intersection(second.effects.index)
    table = pd.DataFrame(
        {
            "feature_id": shared,
            "first_effect": first.effects.reindex(shared).to_numpy(),
            "second_effect": second.effects.reindex(shared).to_numpy(),
            "first_significant": shared.isin(list(first.significant)),
            "second_significant": shared.isin(list(second.significant)),
        }
    )
    same_sign = np.sign(table["first_effect"]) == np.sign(table["second_effect"])
    both = table["first_significant"] & table["second_significant"]
    table["status"] = np.select(
        [
            both & same_sign,
            both & ~same_sign,
            table["first_significant"],
            table["second_significant"],
        ],
        ["both", "opposite", "first", "second"],
        "neither",
    )
    finite = np.isfinite(table["first_effect"]) & np.isfinite(table["second_effect"])
    rho = (
        float(spearmanr(table.loc[finite, "first_effect"], table.loc[finite, "second_effect"])[0])
        if int(finite.sum()) >= 3
        else float("nan")
    )
    counts = {key: int((table["status"] == key).sum()) for key in AGREEMENT_CLASSES}
    return Agreement(
        first=first.label,
        second=second.label,
        table=table,
        spearman=rho,
        shared=len(table),
        counts=counts,
    )

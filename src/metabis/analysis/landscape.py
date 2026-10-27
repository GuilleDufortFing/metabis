"""Every group against the rest: Delta by subsystem and group, and marker reactions.

Comparing each group with all remaining samples and reading the results side by
side is the layout of Xu et al. (Cell Rep Med 2026, Figs. 3A-C); markers found
one group against the rest are standard in single-cell tools (Hao et al., Cell
2021). A group-by-pathway table is the core figure of single-cell metabolic
studies (Xiao, Dai and Locasale, Nat Commun 2019).

Each group's Delta is the overview's own: a :class:`SubsystemAnalysis` reads the
group's stored comparison, so a heatmap cell and the overview's bar for the same
result are the same number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .comparison_artifact import ComparisonThresholds, GroupComparisonArtifact
from .statistical_analysis import StatisticalAnalysis
from .subsystem_analysis import SubsystemAnalysis

__all__ = [
    "DEFAULT_LANDSCAPE_DELTA",
    "Landscape",
    "LandscapeRule",
    "build_landscape",
    "marker_reactions",
    "select_landscape_subsystems",
    "subsystem_delta",
    "with_thresholds",
]

# Metabis's default Delta: the signs of the significant reactions' Cohen's d,
# summed and divided by every reaction of the subsystem.
DEFAULT_LANDSCAPE_DELTA: dict[str, Any] = {
    "metric": "cohens_d",
    "transform": "sign",
    "numerator_filter": "significant",
    "denominator_filter": "all",
}

LandscapeRule = Literal["spread", "category", "chosen"]


def _decide(artifact, for_genes, thresholds, keep_reversed):
    results = artifact.get_test_results(for_genes=for_genes)
    analyzer = StatisticalAnalysis(test_type=artifact.test, correction_type=artifact.correction)
    mask, _info = analyzer.determine_significant_elements(
        results,
        pval_threshold=thresholds.get("pval"),
        cohens_d_threshold=thresholds.get("cohens_d"),
        log2_fc_threshold=thresholds.get("log2_fc"),
        mean_difference_threshold=thresholds.get("mean_difference"),
        keep_reversed=keep_reversed,
    )
    results["is_significant"] = mask.to_numpy()
    significant = tuple(str(item) for item in results.index[mask.to_numpy()])
    return results, ComparisonThresholds.capture(thresholds), significant


def with_thresholds(
    artifact: GroupComparisonArtifact,
    thresholds: Mapping[str, float | None],
    gene_thresholds: Mapping[str, float | None] | None = None,
    *,
    keep_reversed: bool = False,
) -> GroupComparisonArtifact:
    """The artifact with its significance decided at the given thresholds.

    The copy is detached: it keeps the source revisions and is read, never
    installed. Each mapping takes ``pval``, ``cohens_d``, ``log2_fc`` and
    ``mean_difference`` to a value, or to ``None`` to leave one unset. Gene
    results keep their current calls unless ``gene_thresholds`` is given.
    """
    reactions, reaction_thresholds, significant_reactions = _decide(
        artifact, False, thresholds, keep_reversed
    )
    genes = artifact.get_test_results(for_genes=True)
    gene_state = (genes, artifact.gene_thresholds, tuple(artifact.significant_genes))
    if genes is not None and gene_thresholds is not None:
        gene_state = _decide(artifact, True, gene_thresholds, keep_reversed)
    return artifact.with_threshold_results(
        revisions=artifact.metadata.source.revisions,
        reaction_results=reactions,
        reaction_thresholds=reaction_thresholds,
        significant_reactions=significant_reactions,
        gene_results=gene_state[0],
        gene_thresholds=gene_state[1],
        significant_genes=gene_state[2],
        keep_reversed=keep_reversed,
    )


@dataclass(frozen=True)
class _DeltaInputs:
    """What a SubsystemAnalysis reads off its workspace, with no colour registry.

    A detached analysis must not register colour scales, or the landscape would
    repaint the overview of whichever result is in view.
    """

    metabolic_model: Any
    reaction_data: Any
    expression_data: Any = None
    color_manager: Any = None

    def get_group_comparison_artifact(self):
        return None


def subsystem_delta(
    metabolic_model,
    reaction_data,
    artifact: GroupComparisonArtifact,
    *,
    delta_config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Delta, size and significant count of every subsystem for one comparison.

    Rows are subsystem IDs; columns ``name``, ``category``, ``reactions``,
    ``significant`` and ``delta``. Subsystems with no reaction in the data are
    left out.
    """
    analysis = SubsystemAnalysis(
        _DeltaInputs(metabolic_model, reaction_data),
        default_delta_config=dict(delta_config or DEFAULT_LANDSCAPE_DELTA),
    )
    analysis.set_group_comparison(artifact)
    rows = {}
    for subsystem in metabolic_model.get_subsystems() or []:
        stats = analysis.get_subsystem_statistics(subsystem, for_genes=False)
        if not stats.get("is_in_model"):
            continue
        rows[str(subsystem)] = {
            "name": str(metabolic_model.get_subsystem_name(subsystem) or subsystem),
            "category": str(metabolic_model.get_subsystem_category(subsystem) or ""),
            "reactions": int(stats.get("total_elements", 0)),
            "significant": int(stats.get("significant_elements", 0)),
            "delta": float(stats.get("delta_subsystem_score") or 0.0),
        }
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=["name", "category", "reactions", "significant", "delta"]
    )


@dataclass(frozen=True)
class Landscape:
    """Delta for every subsystem (rows) and group (columns), with each row's facts.

    ``significant`` counts each subsystem's significant reactions per group;
    ``significant_reactions`` counts them over the whole model per group.
    """

    delta: pd.DataFrame
    significant: pd.DataFrame
    names: pd.Series
    categories: pd.Series
    reactions: pd.Series
    significant_reactions: pd.Series


def build_landscape(
    metabolic_model,
    reaction_data,
    artifacts: Mapping[str, GroupComparisonArtifact],
    *,
    delta_config: Mapping[str, Any] | None = None,
) -> Landscape:
    """Delta by subsystem for each group's comparison, side by side.

    ``artifacts`` maps a group's label to its comparison against the rest, with
    significance already decided (see :func:`with_thresholds`).
    """
    if not artifacts:
        raise ValueError("The landscape needs at least one comparison")
    tables = {
        label: subsystem_delta(
            metabolic_model, reaction_data, artifact, delta_config=delta_config
        )
        for label, artifact in artifacts.items()
    }
    first = next(iter(tables.values()))
    delta = pd.DataFrame({label: table["delta"] for label, table in tables.items()})
    delta = delta.reindex(first.index).fillna(0.0)
    significant = pd.DataFrame(
        {label: table["significant"] for label, table in tables.items()}
    ).reindex(first.index).fillna(0).astype(int)
    return Landscape(
        delta=delta,
        significant=significant,
        names=first["name"],
        categories=first["category"],
        reactions=first["reactions"],
        significant_reactions=pd.Series(
            {label: len(artifact.significant_reactions) for label, artifact in artifacts.items()}
        ),
    )


def select_landscape_subsystems(
    landscape: Landscape,
    *,
    rule: LandscapeRule = "spread",
    top: int = 25,
    min_reactions: int = 5,
    category: str | None = None,
    chosen: Sequence[str] = (),
) -> list[str]:
    """The subsystems the heatmap shows, ordered by category and then name.

    ``spread`` removes each group's mean Delta over the eligible subsystems, then
    keeps the ``top`` subsystems whose remaining values vary most across groups.
    Without that step a group that is higher everywhere (epithelium against
    immune cells) fills the ranking with its overall level. ``category`` keeps
    every eligible subsystem of one category; ``chosen`` keeps the given IDs.
    Subsystems with fewer than ``min_reactions`` reactions swing furthest on
    chance and are left out of the first two rules.
    """
    if top < 1:
        raise ValueError("Show at least one subsystem")
    if min_reactions < 1:
        raise ValueError("The minimum size is at least one reaction")
    eligible = landscape.reactions.index[landscape.reactions >= min_reactions]
    if rule == "spread":
        block = landscape.delta.loc[eligible]
        if block.empty:
            return []
        centred = block - block.mean(axis=0)
        spread = centred.var(axis=1, ddof=1) if block.shape[1] > 1 else centred.abs().iloc[:, 0]
        picked = list(spread.sort_values(ascending=False, kind="mergesort").index[:top])
    elif rule == "category":
        if not category:
            raise ValueError("Choose a category")
        picked = [item for item in eligible if landscape.categories[item] == category]
    elif rule == "chosen":
        unknown = [item for item in chosen if item not in landscape.delta.index]
        if unknown:
            raise ValueError(f"No reactions in the data for {', '.join(unknown)}")
        picked = list(dict.fromkeys(chosen))
    else:
        raise ValueError(f"Unknown subsystem rule {rule!r}")
    return sorted(picked, key=lambda item: (landscape.categories[item], landscape.names[item]))


def marker_reactions(artifacts: Mapping[str, GroupComparisonArtifact]) -> pd.DataFrame:
    """Reactions that mark one group: significant, higher, and highest there.

    A reaction is a group's marker when it is significant and higher in that
    group than in the rest, and the group's Cohen's d is the largest of all
    groups. Markers are ranked by the margin over the next group's d, so a wide
    lead comes first. ``artifacts`` carry significance already decided.
    """
    effects = pd.DataFrame(
        {
            label: artifact.get_test_results(for_genes=False)["cohens_d"].astype(float)
            for label, artifact in artifacts.items()
        }
    )
    pvalues = pd.DataFrame(
        {
            label: artifact.get_test_results(for_genes=False)["adjusted_pval"].astype(float)
            for label, artifact in artifacts.items()
        }
    ).reindex(effects.index)
    labels = {}
    for artifact in artifacts.values():
        labels.update(dict(artifact.reaction_display_labels))
    filled = effects.fillna(-np.inf)
    best = filled.idxmax(axis=1)
    rows = []
    for label, artifact in artifacts.items():
        significant = set(artifact.significant_reactions)
        mine = [
            feature
            for feature in effects.index
            if feature in significant and effects.at[feature, label] > 0 and best[feature] == label
        ]
        for feature in mine:
            others = filled.loc[feature].drop(label)
            next_group = str(others.idxmax()) if len(others) else ""
            next_effect = float(others.max()) if len(others) else float("nan")
            rows.append(
                {
                    "group": label,
                    "feature_id": str(feature),
                    "display_label": labels.get(str(feature), str(feature)),
                    "cohens_d": float(effects.at[feature, label]),
                    "adjusted_pval": float(pvalues.at[feature, label]),
                    "next_group": next_group,
                    "next_cohens_d": next_effect,
                    "margin": float(effects.at[feature, label]) - next_effect,
                }
            )
    table = pd.DataFrame(
        rows,
        columns=[
            "group",
            "feature_id",
            "display_label",
            "cohens_d",
            "adjusted_pval",
            "next_group",
            "next_cohens_d",
            "margin",
        ],
    )
    order = {label: position for position, label in enumerate(artifacts)}
    return (
        table.assign(_order=table["group"].map(order))
        .sort_values(["_order", "margin"], ascending=[True, False], kind="mergesort")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

"""Immutable comparison artifact and canonical tabular adapters.

This module is deliberately independent of Bokeh and live analysis managers.
The artifact owns exact comparison membership and already-computed scientific
values; adapters only reshape those values for interchange.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import cached_property
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from metabis.analysis_workspace import RevisionKind, WorkspaceRevisions

from .artifacts import (
    AnalysisArtifactMetadata,
    ArtifactDimension,
    ArtifactKind,
    ArtifactProvenance,
    PlainTable,
)
from .comparison_membership import ComparisonMembership, ComparisonMode


@dataclass(frozen=True)
class ComparisonThresholds:
    """Threshold values used to classify one feature family."""

    pval: float | None
    cohens_d: float | None
    log2_fc: float | None
    mean_difference: float | None

    @classmethod
    def capture(cls, values: Mapping[str, float | None]) -> "ComparisonThresholds":
        return cls(
            pval=values.get("pval"),
            cohens_d=values.get("cohens_d"),
            log2_fc=values.get("log2_fc"),
            mean_difference=values.get("mean_difference"),
        )

    def as_mapping(self) -> Mapping[str, float | None]:
        return MappingProxyType(
            {
                "pval": self.pval,
                "cohens_d": self.cohens_d,
                "log2_fc": self.log2_fc,
                "mean_difference": self.mean_difference,
            }
        )


@dataclass(frozen=True)
class MetricSummary:
    """Mean and standard deviation used for a named metric scale."""

    name: str
    std: float
    mean: float


@dataclass(frozen=True)
class GroupComparisonArtifact:
    """Portable comparison result with exact membership and full-precision tables."""

    metadata: AnalysisArtifactMetadata
    membership: ComparisonMembership
    test: str
    correction: str
    reaction_results: PlainTable
    gene_results: PlainTable | None
    reaction_metrics: tuple[MetricSummary, ...]
    gene_metrics: tuple[MetricSummary, ...]
    significant_reactions: tuple[str, ...]
    significant_genes: tuple[str, ...]
    reaction_thresholds: ComparisonThresholds
    gene_thresholds: ComparisonThresholds | None
    keep_reversed: bool = False
    reaction_display_labels: tuple[tuple[str, str], ...] = ()
    gene_display_labels: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.metadata.kind is not ArtifactKind.GROUP_COMPARISON:
            raise ValueError("Comparison artifact metadata must have group_comparison kind")
        if self.metadata.method != self.test:
            raise ValueError("Comparison artifact method must match its statistical test")
        identity = self.metadata.scientific_identity
        expected_identity = {
            "membership_fingerprint": self.membership.fingerprint,
            "mode": self.membership.mode.value,
            "target_base_group_ids": self.membership.target_base_group_ids,
            "baseline_base_group_ids": self.membership.baseline_base_group_ids,
            "target_samples": self.membership.target_samples,
            "baseline_samples": self.membership.baseline_samples,
            "universe": self.membership.universe,
            "excluded_base_group_ids": self.membership.excluded_base_group_ids,
        }
        for name, expected in expected_identity.items():
            if identity.get(name) != expected:
                raise ValueError(
                    f"Comparison metadata scientific identity does not match {name}"
                )
        if self.metadata.configuration.get("correction") != self.correction:
            raise ValueError("Comparison metadata correction must match the artifact")
        if self.metadata.configuration.get("keep_reversed") is not self.keep_reversed:
            raise ValueError("Comparison keep_reversed must match artifact metadata")
        dimensions = {
            dimension.name: dimension.size for dimension in self.metadata.dimensions
        }
        expected_dimensions = {
            "samples": len(self.membership.universe),
            "target_samples": len(self.membership.target_samples),
            "baseline_samples": len(self.membership.baseline_samples),
            "reactions": len(self.reaction_results.index),
            "genes": len(self.gene_results.index) if self.gene_results is not None else 0,
        }
        if dimensions != expected_dimensions:
            raise ValueError("Comparison artifact dimensions do not match its payload")
        if self.gene_results is None and (
            self.gene_metrics
            or self.significant_genes
            or self.gene_thresholds is not None
            or self.gene_display_labels
        ):
            raise ValueError("Gene comparison fields require a gene result table")
        if self.gene_results is not None and self.gene_thresholds is None:
            raise ValueError("Gene comparison results require gene thresholds")
        if tuple(sorted(self.significant_reactions)) != self.significant_reactions:
            raise ValueError("Significant reaction IDs must be sorted")
        if tuple(sorted(self.significant_genes)) != self.significant_genes:
            raise ValueError("Significant gene IDs must be sorted")
        _validate_labels(self.reaction_display_labels)
        _validate_labels(self.gene_display_labels)

    def results_frame(
        self, feature_type: Literal["reaction", "gene"]
    ) -> pd.DataFrame:
        """Return a detached full scientific result frame."""
        if feature_type == "reaction":
            return self.reaction_results.to_frame()
        if feature_type == "gene":
            if self.gene_results is None:
                raise ValueError("This comparison has no gene results")
            return self.gene_results.to_frame()
        raise ValueError(f"Unsupported comparison feature type: {feature_type}")

    def export_frame(
        self, feature_type: Literal["reaction", "gene"]
    ) -> pd.DataFrame:
        """Return the canonical, unformatted publication table."""
        return comparison_export_frame(self, feature_type)

    def with_threshold_results(
        self,
        *,
        revisions: WorkspaceRevisions,
        reaction_results: pd.DataFrame,
        reaction_thresholds: ComparisonThresholds,
        significant_reactions: tuple[str, ...],
        gene_results: pd.DataFrame | None,
        gene_thresholds: ComparisonThresholds | None,
        significant_genes: tuple[str, ...],
        keep_reversed: bool,
    ) -> "GroupComparisonArtifact":
        """Replace threshold-dependent state after one workspace transaction."""
        source = ArtifactProvenance(
            workspace_id=self.metadata.source.workspace_id,
            revisions=revisions,
            dependencies=self.metadata.source.dependencies
            | frozenset({RevisionKind.THRESHOLDS}),
            source_fingerprints=self.metadata.source.source_fingerprints,
        )
        configuration = dict(self.metadata.configuration)
        configuration["reaction_thresholds"] = dict(
            reaction_thresholds.as_mapping()
        )
        configuration["gene_thresholds"] = (
            dict(gene_thresholds.as_mapping())
            if gene_thresholds is not None
            else None
        )
        configuration["keep_reversed"] = keep_reversed
        dimensions = tuple(
            ArtifactDimension(
                dimension.name,
                (
                    len(reaction_results)
                    if dimension.name == "reactions"
                    else len(gene_results)
                    if dimension.name == "genes" and gene_results is not None
                    else dimension.size
                ),
                dimension.label,
            )
            for dimension in self.metadata.dimensions
        )
        metadata = AnalysisArtifactMetadata(
            kind=self.metadata.kind,
            source=source,
            method=self.metadata.method,
            configuration=configuration,
            scientific_identity=self.metadata.scientific_identity,
            dimensions=dimensions,
            annotations=self.metadata.annotations,
            warnings=self.metadata.warnings,
            software_version=self.metadata.software_version,
        )
        return GroupComparisonArtifact(
            metadata=metadata,
            membership=self.membership,
            test=self.test,
            correction=self.correction,
            reaction_results=PlainTable.capture(reaction_results),
            gene_results=(
                PlainTable.capture(gene_results) if gene_results is not None else None
            ),
            reaction_metrics=self.reaction_metrics,
            gene_metrics=self.gene_metrics,
            significant_reactions=tuple(sorted(significant_reactions)),
            significant_genes=tuple(sorted(significant_genes)),
            reaction_thresholds=reaction_thresholds,
            gene_thresholds=gene_thresholds,
            keep_reversed=keep_reversed,
            reaction_display_labels=self.reaction_display_labels,
            gene_display_labels=self.gene_display_labels,
        )

    # ------------------------------------------------------------------
    # Read-only projection for presentation and hover.
    #
    # These reshape the artifact's own immutable values for the readers that
    # once held a live ``GroupComparison`` (AM7 deletes that view). Frames and
    # significant-ID sets are materialized once per artifact and cached, so a
    # per-element hover lookup does not rebuild a table (MR-10). ``get_test_results``
    # returns a detached copy because some plot readers mutate what they receive.
    # ------------------------------------------------------------------

    @property
    def adjusted_for(self) -> tuple[str, ...]:
        """The sample-sheet columns the model adjusted for, as labelled."""
        return tuple(self.metadata.configuration.get("adjusted_for") or ())

    @property
    def group_id_1(self) -> str:
        """The canonical ID of the left comparison group."""
        return " + ".join(self.membership.target_base_group_ids)

    @property
    def group_id_2(self) -> str:
        """The right group's canonical ID, or the one-vs-rest display label."""
        return (
            " + ".join(self.membership.baseline_base_group_ids)
            or self.membership.baseline_label
        )

    @cached_property
    def _reaction_frame(self) -> pd.DataFrame:
        return self.reaction_results.to_frame()

    @cached_property
    def _gene_frame(self) -> pd.DataFrame | None:
        return self.gene_results.to_frame() if self.gene_results is not None else None

    @cached_property
    def _significant_reaction_set(self) -> frozenset[str]:
        return frozenset(self.significant_reactions)

    @cached_property
    def _significant_gene_set(self) -> frozenset[str]:
        return frozenset(self.significant_genes)

    def get_test_results(self, for_genes: bool = False) -> pd.DataFrame | None:
        """A detached full-precision result frame, or ``None`` for absent genes."""
        frame = self._gene_frame if for_genes else self._reaction_frame
        return frame.copy() if frame is not None else None

    def get_significant_elements(self, for_genes: bool = False) -> frozenset[str]:
        """The set of significant feature IDs for one family.

        The artifact's own frozen set, shared rather than copied: every subsystem
        summary holds it, and a copy each would cost a whole family's IDs per
        subsystem.
        """
        return self._significant_gene_set if for_genes else self._significant_reaction_set

    def is_significant(self, element_id: str, for_genes: bool = False) -> bool:
        """Whether one feature is significant under the installed thresholds."""
        target = self._significant_gene_set if for_genes else self._significant_reaction_set
        return element_id in target

    def get_thresholds(self, for_genes: bool = False) -> dict:
        """Threshold values in the mapping presentation and hover consume."""
        thresholds = self.gene_thresholds if for_genes else self.reaction_thresholds
        pval = thresholds.pval if thresholds is not None else None
        neg_log = float(-np.log10(pval)) if pval is not None else None
        return {
            "adjusted_pval": pval,
            "pval": pval,
            "neg_log_adjusted_pval": neg_log,
            "neg_log_pval": neg_log,
            "cohens_d": thresholds.cohens_d if thresholds is not None else None,
            "log2_fc": thresholds.log2_fc if thresholds is not None else None,
            "mean_difference": (
                thresholds.mean_difference if thresholds is not None else None
            ),
        }

    def get_metrics_stats(self, for_genes: bool = False) -> dict:
        """Effect-size normalization stats keyed by metric name."""
        metrics = self.gene_metrics if for_genes else self.reaction_metrics
        return {metric.name: {"mean": metric.mean, "std": metric.std} for metric in metrics}

    def get_statistical_info(
        self, element_id: str, for_genes: bool = False
    ) -> dict[str, Any]:
        """Per-element statistics in the shape ``DataProvider`` and hover expect."""
        results = self._gene_frame if for_genes else self._reaction_frame
        if results is None:
            raise ValueError(
                "No test results found. Run the test first with run_test()."
            )
        if element_id not in results.index:
            return {}
        row = results.loc[element_id]
        return {
            "target_label": self.membership.target_label,
            "baseline_label": self.membership.baseline_label,
            "cohens_d": float(row.get("cohens_d", 0)),
            "log2_fc": float(row.get("log2_fc", 0)),
            "mean_difference": float(row.get("mean_difference", 0)),
            "pval": float(row.get("pval", 1.0)),
            "adjusted_pval": float(row.get("adjusted_pval", 1.0)),
            "reversed": bool(row.get("reversed", False)),
            "test": self.test,
            "max_cohens_d": (
                float(abs(results["cohens_d"]).max())
                if "cohens_d" in results.columns
                else 1.0
            ),
            "is_significant": bool(row.get("is_significant", False)),
        }


# Columns a reading the other way negates, and the pairs it exchanges.
_SIGNED_COLUMNS = ("cohens_d", "log2_fc", "mean_difference")
_PAIRED_COLUMNS = (("group_1_mean", "group_2_mean"), ("n_group_1", "n_group_2"), ("group_1", "group_2"))
_SIGNED_METRICS = frozenset(_SIGNED_COLUMNS)
# The neutral rows the test kernel leaves untested keep a zero statistic.
_UNTESTED_STATUS = "neutral_constant"


def _swapped_results(results: pd.DataFrame, test: str) -> pd.DataFrame:
    """One family's results as a run with the sides exchanged computes them.

    Every effect is Target minus Baseline, so it changes sign; the p-values do
    not, the tests being two-sided. The t statistics change sign, and the
    rank-sum U becomes the other side's, n1 n2 - U.
    """
    swapped = results.copy()
    for column in _SIGNED_COLUMNS:
        if column in swapped.columns:
            swapped[column] = -swapped[column]
    for first, second in _PAIRED_COLUMNS:
        if first in swapped.columns and second in swapped.columns:
            swapped[first], swapped[second] = results[second], results[first]
    if "test_stat" in swapped.columns:
        statistic = results["test_stat"].astype(float)
        if test == "wilcoxon":
            tested = results.get("calculation_status", pd.Series("ok", index=results.index)) != _UNTESTED_STATUS
            pairs = results["n_group_1"].astype(float) * results["n_group_2"].astype(float)
            swapped["test_stat"] = statistic.where(~tested, pairs - statistic)
        else:
            swapped["test_stat"] = -statistic
    swapped.attrs = dict(results.attrs)
    return swapped


def _swapped_metrics(metrics: tuple[MetricSummary, ...]) -> tuple[MetricSummary, ...]:
    return tuple(
        MetricSummary(metric.name, metric.std, -metric.mean if metric.name in _SIGNED_METRICS else metric.mean)
        for metric in metrics
    )


def swapped_comparison(artifact: GroupComparisonArtifact) -> GroupComparisonArtifact:
    """The same comparison read the other way: its Baseline becomes its Target.

    Only a comparison of two named sides has another way to read it. The numbers
    are those a run with the sides exchanged gives, so nothing is tested again:
    effects change sign, the sides' means and counts change places, p-values and
    significance stay.
    """
    membership = artifact.membership
    if membership.mode is not ComparisonMode.PAIRWISE:
        raise ValueError("Only a comparison of two sides can be read the other way")
    swapped = replace(
        membership,
        target_base_group_ids=membership.baseline_base_group_ids,
        baseline_base_group_ids=membership.target_base_group_ids,
        target_label=membership.baseline_label,
        baseline_label=membership.target_label,
        target_samples=membership.baseline_samples,
        baseline_samples=membership.target_samples,
    )
    identity = dict(artifact.metadata.scientific_identity)
    identity.update(
        membership_fingerprint=swapped.fingerprint,
        target_base_group_ids=swapped.target_base_group_ids,
        baseline_base_group_ids=swapped.baseline_base_group_ids,
        target_samples=swapped.target_samples,
        baseline_samples=swapped.baseline_samples,
    )
    sizes = {"target_samples": len(swapped.target_samples), "baseline_samples": len(swapped.baseline_samples)}
    metadata = replace(
        artifact.metadata,
        scientific_identity=identity,
        dimensions=tuple(
            ArtifactDimension(dimension.name, sizes.get(dimension.name, dimension.size), dimension.label)
            for dimension in artifact.metadata.dimensions
        ),
    )
    genes = artifact.gene_results
    return replace(
        artifact,
        metadata=metadata,
        membership=swapped,
        reaction_results=PlainTable.capture(_swapped_results(artifact.reaction_results.to_frame(), artifact.test)),
        gene_results=(
            PlainTable.capture(_swapped_results(genes.to_frame(), artifact.test)) if genes is not None else None
        ),
        reaction_metrics=_swapped_metrics(artifact.reaction_metrics),
        gene_metrics=_swapped_metrics(artifact.gene_metrics),
    )


def comparison_export_frame(
    artifact: GroupComparisonArtifact,
    feature_type: Literal["reaction", "gene"],
) -> pd.DataFrame:
    """Adapt an artifact to its canonical full-precision feature table.

    Numeric values are not rounded or string-formatted. Every row carries the
    calculation method, correction, thresholds, significance decision, and the
    exact membership fingerprint needed to join it to exported membership.
    """
    results = artifact.results_frame(feature_type)
    thresholds = (
        artifact.reaction_thresholds
        if feature_type == "reaction"
        else artifact.gene_thresholds
    )
    if thresholds is None:  # protected by the artifact invariant
        raise ValueError(f"This comparison has no {feature_type} thresholds")
    label_pairs = (
        artifact.reaction_display_labels
        if feature_type == "reaction"
        else artifact.gene_display_labels
    )
    labels = dict(label_pairs)
    feature_ids = tuple(map(str, results.index))

    export = results.copy()
    for legacy_id in ("reaction_id", "gene_id"):
        if legacy_id in export.columns:
            export = export.drop(columns=[legacy_id])

    # The statistical kernel retains its compact group_1/group_2 schema, but the
    # public artifact adapter must make the signed contract unambiguous. Target
    # is the first analysis input and Baseline is the second.
    export = export.rename(
        columns={
            "group_1_mean": "target_mean",
            "group_2_mean": "baseline_mean",
            "n_group_1": "n_target",
            "n_group_2": "n_baseline",
        }
    )
    export = export.drop(columns=["group_1", "group_2"], errors="ignore")
    export.insert(0, "feature_id", feature_ids)
    export.insert(1, "display_label", [labels.get(item, item) for item in feature_ids])
    export.insert(2, "feature_type", feature_type)
    export.insert(3, "target_label", artifact.membership.target_label)
    export.insert(4, "baseline_label", artifact.membership.baseline_label)
    export.insert(
        5,
        "target_base_group_ids",
        json.dumps(list(artifact.membership.target_base_group_ids)),
    )
    export.insert(
        6,
        "baseline_base_group_ids",
        json.dumps(list(artifact.membership.baseline_base_group_ids)),
    )
    export.insert(7, "effect_direction", "target_minus_baseline")
    export["test"] = artifact.test
    export["adjusted_for"] = ", ".join(artifact.adjusted_for)
    export["correction"] = artifact.correction
    export["threshold_pval"] = thresholds.pval
    export["threshold_cohens_d"] = thresholds.cohens_d
    export["threshold_log2_fc"] = thresholds.log2_fc
    export["threshold_mean_difference"] = thresholds.mean_difference
    export["keep_reversed"] = artifact.keep_reversed
    export["membership_fingerprint"] = artifact.membership.fingerprint
    return export.reset_index(drop=True)


def _validate_labels(labels: tuple[tuple[str, str], ...]) -> None:
    if not isinstance(labels, tuple) or not all(
        isinstance(item, tuple)
        and len(item) == 2
        and all(isinstance(value, str) for value in item)
        for item in labels
    ):
        raise TypeError("Display labels must be canonical (feature ID, label) tuples")
    ids = [item[0] for item in labels]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("Display label feature IDs must be unique and sorted")


__all__ = [
    "ComparisonThresholds",
    "GroupComparisonArtifact",
    "MetricSummary",
    "comparison_export_frame",
    "swapped_comparison",
]

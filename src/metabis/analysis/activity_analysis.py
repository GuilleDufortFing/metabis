"""Normalized single-group activity summaries for pathway visualization.

For a selected group ``g`` and measured feature ``f``, the activity score is the
mean or median of the already-normalized input values in that group. No feature
z-score, reference center, reference spread, p-value, or effect size is computed.

The result is descriptive evidence on the uploaded input's normalized scale. It
is not a measured physical flux and does not establish regulation or statistical
significance. Non-finite group summaries remain unavailable and are audited rather
than being turned into a synthetic neutral value.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import pandas as pd

from metabis.analysis.artifacts import (
    AnalysisArtifactMetadata,
    ArtifactDimension,
    ArtifactKind,
    PlainTable,
)
from metabis.analysis_workspace import WorkspaceProvenance

if TYPE_CHECKING:
    from metabis.domain.expression_matrix import ExpressionMatrix

__all__ = [
    "ACTIVITY_METHOD",
    "ActivityOverlay",
    "GroupSummary",
    "ActivityConfig",
    "FeatureActivityResult",
    "SubsystemActivityRow",
    "ActivityContext",
    "ActivityFeatureResults",
    "ActivityArtifact",
    "calculate_feature_activity",
    "calculate_feature_activity_from_matrix",
    "calculate_subsystem_activity",
    "resolve_display_direction",
    "displayed_feature_id",
]

ACTIVITY_METHOD = "single_group_normalized_activity"
"""The calculation an activity artifact records as its method."""

class ActivityOverlay(str, Enum):
    """Which measured data paints the graph."""

    REACTION = "reaction"
    GENE = "gene"


class GroupSummary(str, Enum):
    """How the samples of the selected group are collapsed into one value."""

    MEAN = "mean"
    MEDIAN = "median"


SUMMARY_DESCRIPTIONS: Mapping[GroupSummary, str] = {
    GroupSummary.MEAN: "mean across the group's samples",
    GroupSummary.MEDIAN: "median across the group's samples",
}


@dataclass(frozen=True)
class ActivityConfig:
    """Every scientific input that changes a single-group activity result.

    The reaction/gene overlay is deliberately absent.  It is presentation state:
    one calculation and one artifact carry every measured family for this group.

    ``sample_ids`` is empty for a base group, whose membership is resolved from the
    Analysis Workspace.  A comparison-local Target or Baseline population is not a
    persisted ``SampleGroup``, so it carries its own frozen membership here and
    ``group_id`` is that population's comparison label.
    """

    group_id: str
    summary: GroupSummary = GroupSummary.MEAN
    minimum_subsystem_coverage: float = 0.1
    sample_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.group_id).strip():
            raise ValueError("Activity mode requires a group_id")
        if not 0.0 <= self.minimum_subsystem_coverage <= 1.0:
            raise ValueError("minimum_subsystem_coverage must be between 0 and 1")
        if isinstance(self.sample_ids, str):
            raise TypeError("sample_ids must be a sequence of IDs, not a string")
        samples = tuple(str(sample_id) for sample_id in self.sample_ids)
        if len(set(samples)) != len(samples):
            raise ValueError(
                "sample_ids must not repeat a sample; a population summarizes each "
                "sample once"
            )
        object.__setattr__(self, "sample_ids", samples)

    @property
    def has_explicit_population(self) -> bool:
        """True when this config carries its own frozen sample membership.

        Callers use this to decide whether ``group_id`` names a base group they may
        resolve, or labels a comparison-local population that only this config knows.
        """
        return bool(self.sample_ids)

    @property
    def population_fingerprint(self) -> str:
        """Stable digest of an explicit population's exact membership.

        The population is a set, so the digest is order-independent; two labels
        cannot share a cache entry unless they summarize the same samples.
        """
        if not self.sample_ids:
            return ""
        serialized = json.dumps(sorted(self.sample_ids), separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @property
    def metric_key(self) -> str:
        """Stable identity for the dual-family calculation cache.

        Every field that changes a score is part of the key. Switching the overview
        overlay does not change this key because it does not change the calculation.
        """
        parts = [
            "activity",
            str(self.group_id),
            self.summary.value,
            f"coverage={self.minimum_subsystem_coverage:g}",
        ]
        if self.sample_ids:
            # Two comparisons can label their populations identically, so only
            # exact membership separates their cached scores.
            parts.append(f"population={self.population_fingerprint}")
        return "|".join(parts)

    def metric_key_for(self, overlay: ActivityOverlay | str) -> str:
        """Family-specific color/hover key under this calculation identity."""
        family = ActivityOverlay(overlay)
        return f"{self.metric_key}|family={family.value}"

    def describe(self, overlay: ActivityOverlay | str) -> str:
        """One human-readable sentence naming the statistic being colored."""
        family = ActivityOverlay(overlay)
        element = (
            "reaction activity scores"
            if family is ActivityOverlay.REACTION
            else "gene expression"
        )
        return (
            f"Normalized {SUMMARY_DESCRIPTIONS[self.summary]} of {element} "
            f"in '{self.group_id}'. Values retain the normalized input scale."
        )


@dataclass(frozen=True)
class FeatureActivityResult:
    """Per-feature activity scores for one group under one config."""

    group_id: str
    overlay: str
    activity_score: pd.Series
    sample_count: int
    reference_sample_count: int
    config: ActivityConfig
    non_finite_feature_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def score(self, feature_id: str) -> float | None:
        """Return the normalized activity score, or None when unavailable."""
        if feature_id not in self.activity_score.index:
            return None
        value = self.activity_score.loc[feature_id]
        return None if pd.isna(value) else float(value)


@dataclass(frozen=True)
class SubsystemActivityRow:
    """Activity of one subsystem, with the coverage it was computed from."""

    subsystem_id: str
    score: float | None
    total_model_reactions: int
    measured_reactions: int
    coverage: float
    positive_count: int
    negative_count: int

    @property
    def is_reportable(self) -> bool:
        return self.score is not None


SUBSYSTEM_ACTIVITY_COLUMNS = (
    "subsystem",
    "activity_score",
    "total_model_features",
    "measured_features",
    "coverage",
    "positive_count",
    "negative_count",
)


COLOR_SATURATION_QUANTILE = 0.98
"""Quantile of activity scores that saturates the sequential colour palette.

Clipping the extreme tail is what lets the bulk of the data use the whole palette.
Without it, one reaction ten times more active than any other compresses every
remaining reaction into the neutral centre colour, and the map reads as though
nothing is active anywhere.
"""

PLOT_SATURATION_QUANTILE = 0.98
"""Quantile of the plotted values that bounds the ranking plot's axis.

The same problem in the other view: a single outlier stretches the axis until the
rest of the points collapse onto the zero line. Kept equal to
``COLOR_SATURATION_QUANTILE`` so that a point saturating the palette is also a
point pinned to the edge of the axis.
"""

REFERENCE_SAMPLE_LIMIT = 1500
"""Cap on the reference distribution carried for hover plots.

A hover violin is built for every glyph of a rendered graph, so the background
distribution must be sampled once per context rather than re-derived from tens of
thousands of features on every hover.
"""


def sample_reference_distribution(
    values: pd.Series, limit: int = REFERENCE_SAMPLE_LIMIT
) -> tuple[float, ...]:
    """Subsample a distribution deterministically, preserving its shape.

    Takes evenly spaced order statistics rather than a random draw: the quantiles
    of the subsample match the quantiles of the full distribution, and the same
    data always yields the same plot.
    """
    finite = values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return ()
    if finite.size <= limit:
        return tuple(float(value) for value in finite)

    ordered = np.sort(finite)
    positions = np.linspace(0, ordered.size - 1, limit).round().astype(int)
    return tuple(float(value) for value in ordered[positions])


# ----------------------------------------------------------------------------
# Colour meaning shared by the live compute context and the portable artifact.
#
# These depend only on the config (or a subsystem-indexed frame), so both the
# ``ActivityContext`` the worker computes and the ``ActivityArtifact`` the
# workspace installs answer them from one implementation. AM7 deletes the
# installed context, and its presentation readers move onto the artifact; sharing
# the meaning here keeps the two answers identical while both exist.
# ----------------------------------------------------------------------------


def activity_colors_element_type(
    overlay: ActivityOverlay | str, element_type: str
) -> bool:
    """Only the overlay's own glyphs and subsystems carry activity color."""
    family = ActivityOverlay(overlay)
    return element_type in {family.value, "subsystem", "subsystem_genes"}


def activity_binning(config: "ActivityConfig") -> dict[str, object]:
    """Center/spread the color bins are built from.

    The spread is *asymmetrically saturating*, not a standard deviation. Activity
    scores are bounded magnitudes and usually right-skewed. Scaling both sides by
    the longer upper tail would place the white endpoint below the attainable
    data, making the least-active measured features start yellow. Each half is
    therefore stretched over the data that exists on that side of the median;
    clipping the extreme tails keeps one outlier from flattening everything else.
    """
    return {
        "center_method": "median",
        "spread_method": "asymmetric_saturating",
        "saturation_quantile": COLOR_SATURATION_QUANTILE,
    }


def activity_legend_text(
    config: "ActivityConfig", overlay: ActivityOverlay | str
) -> str:
    """The color meaning for activity mode.

    Deliberately never says "up-regulated"/"down-regulated": nothing is being
    compared, so there is no regulation to report. The ramp runs from low
    through typical to high, a level against the family's typical value.
    """
    family = ActivityOverlay(overlay)
    element = "reactions" if family is ActivityOverlay.REACTION else "genes"
    return (
        f"Coloring {element} by their normalized {config.summary.value} in "
        f"'{config.group_id}'. Gold = high, grey = typical, purple = low within "
        f"this {element[:-1]} family; white and dashed = no data."
    )


def subsystem_activity_row(
    subsystems: pd.DataFrame, subsystem_id: str
) -> "SubsystemActivityRow | None":
    """One subsystem's activity row from a subsystem-indexed frame, or None."""
    if subsystems.empty or subsystem_id not in subsystems.index:
        return None
    row = subsystems.loc[subsystem_id]
    score = row["activity_score"]
    return SubsystemActivityRow(
        subsystem_id=subsystem_id,
        score=None if pd.isna(score) else float(score),
        total_model_reactions=int(row["total_model_features"]),
        measured_reactions=int(row["measured_features"]),
        coverage=float(row["coverage"]),
        positive_count=int(row["positive_count"]),
        negative_count=int(row["negative_count"]),
    )


@dataclass(frozen=True)
class ActivityContext:
    """One dual-family activity calculation and the data it used.

    Carrying the revisions the results were computed from lets callers detect a
    stale context after the data or the groups change, instead of coloring a new
    dataset with an old group's scores.
    """

    config: ActivityConfig
    reaction_features: FeatureActivityResult
    reaction_subsystems: pd.DataFrame
    gene_features: FeatureActivityResult | None = None
    gene_subsystems: pd.DataFrame | None = None
    dataset_revision: int = 0
    group_revision: int = 0
    reaction_reference_summary: tuple[float, ...] = ()
    gene_reference_summary: tuple[float, ...] = ()

    @property
    def metric_key(self) -> str:
        return self.config.metric_key

    @property
    def group_id(self) -> str:
        return self.config.group_id

    @property
    def features(self) -> FeatureActivityResult:
        """Compatibility projection of the required reaction family."""
        return self.reaction_features

    @property
    def subsystems(self) -> pd.DataFrame:
        """Compatibility projection of the required reaction family."""
        return self.reaction_subsystems

    @property
    def reference_summary(self) -> tuple[float, ...]:
        """Compatibility projection of the required reaction family."""
        return self.reaction_reference_summary

    def has_family(self, overlay: ActivityOverlay | str) -> bool:
        family = ActivityOverlay(overlay)
        return family is ActivityOverlay.REACTION or self.gene_features is not None

    def features_for(self, overlay: ActivityOverlay | str) -> FeatureActivityResult:
        family = ActivityOverlay(overlay)
        if family is ActivityOverlay.REACTION:
            return self.reaction_features
        if self.gene_features is None:
            raise ValueError("Gene activity evidence is unavailable")
        return self.gene_features

    def subsystems_for(self, overlay: ActivityOverlay | str) -> pd.DataFrame:
        family = ActivityOverlay(overlay)
        if family is ActivityOverlay.REACTION:
            return self.reaction_subsystems
        if self.gene_subsystems is None:
            raise ValueError("Gene activity evidence is unavailable")
        return self.gene_subsystems

    def reference_summary_for(
        self, overlay: ActivityOverlay | str
    ) -> tuple[float, ...]:
        family = ActivityOverlay(overlay)
        if family is ActivityOverlay.REACTION:
            return self.reaction_reference_summary
        if self.gene_features is None:
            raise ValueError("Gene activity evidence is unavailable")
        return self.gene_reference_summary

    def colors_element_type(
        self, overlay: ActivityOverlay | str, element_type: str
    ) -> bool:
        """Only the overlay's own glyphs and subsystems carry activity color."""
        return self.has_family(overlay) and activity_colors_element_type(
            overlay, element_type
        )

    def binning(self) -> dict[str, object]:
        """Center/spread the color bins are built from (see ``activity_binning``)."""
        return activity_binning(self.config)

    def subsystem_row(
        self, overlay: ActivityOverlay | str, subsystem_id: str
    ) -> SubsystemActivityRow | None:
        return subsystem_activity_row(self.subsystems_for(overlay), subsystem_id)

    def legend_text(self, overlay: ActivityOverlay | str) -> str:
        """The color meaning for activity mode (see ``activity_legend_text``)."""
        return activity_legend_text(self.config, overlay)


ACTIVITY_FEATURE_EXPORT_COLUMNS = (
    "artifact_id",
    "group_id",
    "overlay",
    "feature_id",
    "activity_score",
    "is_non_finite",
)

ACTIVITY_SUBSYSTEM_EXPORT_COLUMNS = (
    "artifact_id",
    "group_id",
    "overlay",
    *SUBSYSTEM_ACTIVITY_COLUMNS,
)


@dataclass(frozen=True)
class ActivityFeatureResults:
    """Deeply immutable capture of per-feature activity values and audit flags."""

    group_id: str
    overlay: str
    table: PlainTable
    sample_count: int
    reference_sample_count: int
    config: ActivityConfig
    non_finite_feature_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @classmethod
    def capture(cls, result: FeatureActivityResult) -> "ActivityFeatureResults":
        """Detach every feature value and audit flag from a live result."""
        non_finite = set(result.non_finite_feature_ids)
        rows: list[dict[str, object]] = []
        for feature_key in result.activity_score.index:
            feature_id = str(feature_key)
            rows.append(
                {
                    "feature_id": feature_id,
                    "activity_score": result.activity_score.get(feature_key),
                    "is_non_finite": feature_id in non_finite,
                }
            )
        columns = ACTIVITY_FEATURE_EXPORT_COLUMNS[3:]
        return cls(
            group_id=result.group_id,
            overlay=result.overlay,
            table=PlainTable.capture(pd.DataFrame(rows, columns=list(columns))),
            sample_count=result.sample_count,
            reference_sample_count=result.reference_sample_count,
            config=result.config,
            non_finite_feature_ids=tuple(result.non_finite_feature_ids),
            warnings=tuple(result.warnings),
        )

    def to_frame(self) -> pd.DataFrame:
        """Return a detached canonical feature-result table."""
        return self.table.to_frame()


@dataclass(frozen=True)
class ActivityArtifact:
    """Portable activity result with exact inputs and immutable tabular state.

    ``ActivityContext`` is the compute-side value used to build this artifact.
    The artifact is the authoritative installed and portable result: it owns the
    exact selected/reference membership, provenance, and presentation projections.
    """

    metadata: AnalysisArtifactMetadata
    selected_sample_ids: tuple[str, ...]
    reference_sample_ids: tuple[str, ...]
    config: ActivityConfig
    reaction_features: ActivityFeatureResults
    reaction_subsystem_table: PlainTable
    gene_features: ActivityFeatureResults | None = None
    gene_subsystem_table: PlainTable | None = None
    reaction_reference_summary: tuple[float, ...] = ()
    gene_reference_summary: tuple[float, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, AnalysisArtifactMetadata):
            raise TypeError("metadata must be AnalysisArtifactMetadata")
        if self.metadata.kind is not ArtifactKind.ACTIVITY:
            raise ValueError("ActivityArtifact metadata must have activity kind")
        if not isinstance(self.config, ActivityConfig):
            raise TypeError("config must be ActivityConfig")
        expected_configuration = {
            "summary": self.config.summary.value,
            "minimum_subsystem_coverage": self.config.minimum_subsystem_coverage,
        }
        if dict(self.metadata.configuration) != expected_configuration:
            raise ValueError("Activity artifact configuration does not match metadata")
        if not isinstance(self.selected_sample_ids, tuple) or not all(
            isinstance(sample_id, str) for sample_id in self.selected_sample_ids
        ):
            raise TypeError("selected_sample_ids must be a tuple of strings")
        if not isinstance(self.reference_sample_ids, tuple) or not all(
            isinstance(sample_id, str) for sample_id in self.reference_sample_ids
        ):
            raise TypeError("reference_sample_ids must be a tuple of strings")
        if len(set(self.selected_sample_ids)) != len(self.selected_sample_ids):
            raise ValueError("selected_sample_ids must be unique")
        if len(set(self.reference_sample_ids)) != len(self.reference_sample_ids):
            raise ValueError("reference_sample_ids must be unique")
        if not set(self.selected_sample_ids) <= set(self.reference_sample_ids):
            raise ValueError("selected samples must belong to the reference universe")
        identity = self.metadata.scientific_identity
        if identity.get("group_id") != self.config.group_id:
            raise ValueError("Activity group identity does not match metadata")
        if tuple(identity.get("selected_sample_ids", ())) != self.selected_sample_ids:
            raise ValueError("Activity selected samples do not match metadata")
        if tuple(identity.get("reference_sample_ids", ())) != self.reference_sample_ids:
            raise ValueError("Activity reference samples do not match metadata")
        if not isinstance(self.reaction_features, ActivityFeatureResults):
            raise TypeError("reaction_features must be ActivityFeatureResults")
        if not isinstance(self.reaction_subsystem_table, PlainTable):
            raise TypeError("reaction_subsystem_table must be a PlainTable")
        if (self.gene_features is None) != (self.gene_subsystem_table is None):
            raise ValueError(
                "gene_features and gene_subsystem_table must be present together"
            )
        for overlay, features in self._captured_families():
            if features.config != self.config:
                raise ValueError("Activity feature config does not match the artifact")
            if features.group_id != self.config.group_id:
                raise ValueError("Activity feature group does not match the artifact")
            if features.overlay != overlay.value:
                raise ValueError("Activity feature family does not match its slot")
            if features.sample_count != len(self.selected_sample_ids):
                raise ValueError(
                    "Activity selected sample count does not match features"
                )
            if features.reference_sample_count != len(self.reference_sample_ids):
                raise ValueError(
                    "Activity reference sample count does not match features"
                )
        identity_families = tuple(identity.get("available_families", ()))
        if identity_families != tuple(
            overlay.value for overlay, _features in self._captured_families()
        ):
            raise ValueError("Activity family identity does not match metadata")
        for name, values in (
            ("reaction_reference_summary", self.reaction_reference_summary),
            ("gene_reference_summary", self.gene_reference_summary),
        ):
            if not isinstance(values, tuple) or not all(
                isinstance(value, float) for value in values
            ):
                raise TypeError(f"{name} must be a tuple of floats")
        if self.gene_features is None and self.gene_reference_summary:
            raise ValueError("Gene reference summary requires gene features")
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(warning, str) for warning in self.warnings
        ):
            raise TypeError("warnings must be a tuple of strings")
        if self.metadata.warnings != self.warnings:
            raise ValueError("Activity warnings do not match metadata")
        dimensions = {
            dimension.name: dimension.size for dimension in self.metadata.dimensions
        }
        expected_dimensions = {
            "reaction_features": len(self.reaction_features.table.rows),
            "reaction_subsystems": len(self.reaction_subsystem_table.rows),
            "gene_features": (
                len(self.gene_features.table.rows) if self.gene_features else 0
            ),
            "gene_subsystems": (
                len(self.gene_subsystem_table.rows)
                if self.gene_subsystem_table is not None
                else 0
            ),
            "selected_samples": len(self.selected_sample_ids),
            "reference_samples": len(self.reference_sample_ids),
        }
        if dimensions != expected_dimensions:
            raise ValueError(
                "Activity artifact dimensions do not match its scientific payload"
            )

    def _captured_families(
        self,
    ) -> tuple[tuple[ActivityOverlay, ActivityFeatureResults], ...]:
        families = [(ActivityOverlay.REACTION, self.reaction_features)]
        if self.gene_features is not None:
            families.append((ActivityOverlay.GENE, self.gene_features))
        return tuple(families)

    @classmethod
    def build(
        cls,
        context: ActivityContext,
        *,
        provenance: WorkspaceProvenance,
        source_fingerprints: Mapping[str, str],
        selected_sample_ids: Sequence[str],
        reference_sample_ids: Sequence[str],
        warnings: Sequence[str] | None = None,
    ) -> "ActivityArtifact":
        """Project a live activity context into its portable artifact.

        Input identity is read from the Analysis Workspace, which paid for it once
        at load; recomputing it here would hash the full matrix on every overlay
        (ADR-0007, MR-10).
        """
        selected = tuple(map(str, selected_sample_ids))
        reference = tuple(map(str, reference_sample_ids))
        computed_warnings = tuple(context.reaction_features.warnings)
        if context.gene_features is not None:
            computed_warnings += tuple(context.gene_features.warnings)
        resolved_warnings = tuple(
            dict.fromkeys(computed_warnings if warnings is None else warnings)
        )
        config = context.config
        families = [ActivityOverlay.REACTION.value]
        if context.gene_features is not None:
            families.append(ActivityOverlay.GENE.value)
        metadata = AnalysisArtifactMetadata.from_workspace(
            kind=ArtifactKind.ACTIVITY,
            provenance=provenance,
            source_fingerprints=source_fingerprints,
            method=ACTIVITY_METHOD,
            configuration={
                "summary": config.summary.value,
                "minimum_subsystem_coverage": config.minimum_subsystem_coverage,
            },
            scientific_identity={
                "group_id": config.group_id,
                "selected_sample_ids": selected,
                "reference_sample_ids": reference,
                "available_families": families,
            },
            dimensions=(
                ArtifactDimension(
                    "reaction_features", len(context.reaction_features.activity_score)
                ),
                ArtifactDimension(
                    "reaction_subsystems", len(context.reaction_subsystems)
                ),
                ArtifactDimension(
                    "gene_features",
                    len(context.gene_features.activity_score)
                    if context.gene_features is not None
                    else 0,
                ),
                ArtifactDimension(
                    "gene_subsystems",
                    len(context.gene_subsystems)
                    if context.gene_subsystems is not None
                    else 0,
                ),
                ArtifactDimension("selected_samples", len(selected)),
                ArtifactDimension("reference_samples", len(reference)),
            ),
            warnings=resolved_warnings,
        )
        return cls.from_context(
            context,
            metadata=metadata,
            selected_sample_ids=selected,
            reference_sample_ids=reference,
            warnings=resolved_warnings,
        )

    @classmethod
    def from_context(
        cls,
        context: ActivityContext,
        *,
        metadata: AnalysisArtifactMetadata,
        selected_sample_ids: Sequence[str],
        reference_sample_ids: Sequence[str],
        warnings: Sequence[str] | None = None,
    ) -> "ActivityArtifact":
        """Capture a live context without retaining its mutable DataFrame."""
        return cls(
            metadata=metadata,
            selected_sample_ids=tuple(map(str, selected_sample_ids)),
            reference_sample_ids=tuple(map(str, reference_sample_ids)),
            config=context.config,
            reaction_features=ActivityFeatureResults.capture(
                context.reaction_features
            ),
            reaction_subsystem_table=PlainTable.capture(
                context.reaction_subsystems
            ),
            gene_features=(
                ActivityFeatureResults.capture(context.gene_features)
                if context.gene_features is not None
                else None
            ),
            gene_subsystem_table=(
                PlainTable.capture(context.gene_subsystems)
                if context.gene_subsystems is not None
                else None
            ),
            reaction_reference_summary=tuple(
                map(float, context.reaction_reference_summary)
            ),
            gene_reference_summary=tuple(map(float, context.gene_reference_summary)),
            warnings=tuple(
                dict.fromkeys(
                    (
                        tuple(context.reaction_features.warnings)
                        + (
                            tuple(context.gene_features.warnings)
                            if context.gene_features is not None
                            else ()
                        )
                    )
                    if warnings is None
                    else warnings
                )
            ),
        )

    # ------------------------------------------------------------------
    # Read-only projection for presentation, hover, and graph colouring.
    #
    # These reshape the artifact's own immutable values for the readers that
    # once held a live ``ActivityContext`` (AM7 deletes the installed view).
    # The rehydrated feature Series and subsystem frame are materialized once
    # per artifact and cached, so a per-glyph colour or hover lookup rebuilds no
    # table (MR-10). The colour-meaning methods delegate to the module helpers
    # the compute context shares, so an installed artifact and the context it was
    # projected from answer identically.
    # ------------------------------------------------------------------

    @property
    def group_id(self) -> str:
        """The canonical ID of the group this activity view scores."""
        return self.config.group_id

    @property
    def features(self) -> ActivityFeatureResults:
        """Compatibility projection of the required reaction family."""
        return self.reaction_features

    @property
    def subsystem_table(self) -> PlainTable:
        """Compatibility projection of the required reaction family."""
        return self.reaction_subsystem_table

    @property
    def reference_summary(self) -> tuple[float, ...]:
        """Compatibility projection of the required reaction family."""
        return self.reaction_reference_summary

    @property
    def metric_key(self) -> str:
        """The dual-family calculation identity (not a color-distribution key)."""
        return self.config.metric_key

    @property
    def available_families(self) -> tuple[ActivityOverlay, ...]:
        families = [ActivityOverlay.REACTION]
        if self.gene_features is not None:
            families.append(ActivityOverlay.GENE)
        return tuple(families)

    def has_family(self, overlay: ActivityOverlay | str) -> bool:
        return ActivityOverlay(overlay) in self.available_families

    def metric_key_for(self, overlay: ActivityOverlay | str) -> str:
        if not self.has_family(overlay):
            raise ValueError(f"{ActivityOverlay(overlay).value.title()} evidence unavailable")
        return self.config.metric_key_for(overlay)

    def colors_element_type(
        self, overlay: ActivityOverlay | str, element_type: str
    ) -> bool:
        """Whether one element type carries this view's activity colour."""
        return self.has_family(overlay) and activity_colors_element_type(
            overlay, element_type
        )

    def binning(self) -> dict[str, object]:
        """Center/spread the colour bins are built from (see ``activity_binning``)."""
        return activity_binning(self.config)

    def legend_text(self, overlay: ActivityOverlay | str) -> str:
        """The colour meaning for activity mode (see ``activity_legend_text``)."""
        return activity_legend_text(self.config, overlay)

    def subsystems_for(self, overlay: ActivityOverlay | str) -> pd.DataFrame:
        family = ActivityOverlay(overlay)
        if family is ActivityOverlay.REACTION:
            return self.reaction_subsystems
        if self.gene_subsystem_table is None:
            raise ValueError("Gene activity evidence is unavailable")
        return self.gene_subsystems

    def subsystem_row(
        self, overlay: ActivityOverlay | str, subsystem_id: str
    ) -> SubsystemActivityRow | None:
        """One subsystem's activity row, or ``None`` when it was not measured."""
        return subsystem_activity_row(self.subsystems_for(overlay), subsystem_id)

    @cached_property
    def reaction_subsystems(self) -> pd.DataFrame:
        return self.reaction_subsystem_table.to_frame()

    @cached_property
    def gene_subsystems(self) -> pd.DataFrame:
        if self.gene_subsystem_table is None:
            return pd.DataFrame(columns=list(SUBSYSTEM_ACTIVITY_COLUMNS))
        return self.gene_subsystem_table.to_frame()

    def _build_feature_result(
        self, features: ActivityFeatureResults
    ) -> FeatureActivityResult:
        """Rehydrate the per-feature scores into a fresh live result.

        Each call returns independent Series, so a caller that mutates one
        materialization cannot corrupt another (or the cached hot-path one).
        """
        feature_frame = features.to_frame()
        feature_ids = feature_frame["feature_id"].map(str)

        def feature_series(column: str) -> pd.Series:
            return pd.Series(
                feature_frame[column].to_numpy(copy=True),
                index=pd.Index(feature_ids, name="feature_id"),
                name=column,
            )

        return FeatureActivityResult(
            group_id=features.group_id,
            overlay=features.overlay,
            activity_score=feature_series("activity_score"),
            sample_count=features.sample_count,
            reference_sample_count=features.reference_sample_count,
            config=self.config,
            non_finite_feature_ids=features.non_finite_feature_ids,
            warnings=features.warnings,
        )

    @cached_property
    def reaction_feature_result(self) -> FeatureActivityResult:
        return self._build_feature_result(self.reaction_features)

    @property
    def feature_result(self) -> FeatureActivityResult:
        """Compatibility projection of the required reaction family."""
        return self.reaction_feature_result

    @property
    def subsystems(self) -> pd.DataFrame:
        """Compatibility projection of the required reaction family."""
        return self.reaction_subsystems

    @cached_property
    def gene_feature_result(self) -> FeatureActivityResult | None:
        if self.gene_features is None:
            return None
        return self._build_feature_result(self.gene_features)

    def feature_result_for(
        self, overlay: ActivityOverlay | str
    ) -> FeatureActivityResult:
        """The requested family's scores, materialized once and cached.

        The presentation readers only read these Series (scores, indices), so one
        shared materialization spares a per-glyph rebuild (MR-10).
        """
        family = ActivityOverlay(overlay)
        if family is ActivityOverlay.REACTION:
            return self.reaction_feature_result
        if self.gene_feature_result is None:
            raise ValueError("Gene activity evidence is unavailable")
        return self.gene_feature_result

    def reference_summary_for(
        self, overlay: ActivityOverlay | str
    ) -> tuple[float, ...]:
        family = ActivityOverlay(overlay)
        if family is ActivityOverlay.REACTION:
            return self.reaction_reference_summary
        if self.gene_features is None:
            raise ValueError("Gene activity evidence is unavailable")
        return self.gene_reference_summary

    def display_direction(self, reaction_id: str) -> str:
        """Displayed direction of a reversible reaction under this view's summary."""
        return resolve_display_direction(
            self.reaction_feature_result.activity_score, reaction_id
        )

    def feature_frame(
        self, overlay: ActivityOverlay | str | None = None
    ) -> pd.DataFrame:
        """Canonical feature table for one family, or all measured families."""
        families = (
            (ActivityOverlay(overlay),)
            if overlay is not None
            else self.available_families
        )
        rows: list[dict[str, object]] = []
        for family in families:
            features = (
                self.reaction_features
                if family is ActivityOverlay.REACTION
                else self.gene_features
            )
            if features is None:
                continue
            rows.extend(
                {
                    "artifact_id": self.metadata.artifact_id,
                    "group_id": self.config.group_id,
                    "overlay": family.value,
                    **row.to_dict(),
                }
                for _, row in features.to_frame().iterrows()
            )
        return pd.DataFrame(rows, columns=list(ACTIVITY_FEATURE_EXPORT_COLUMNS))

    def subsystem_frame(
        self, overlay: ActivityOverlay | str | None = None
    ) -> pd.DataFrame:
        """Canonical subsystem table for one family, or all measured families."""
        families = (
            (ActivityOverlay(overlay),)
            if overlay is not None
            else self.available_families
        )
        rows: list[dict[str, object]] = []
        for family in families:
            subsystem_frame = self.subsystems_for(family)
            rows.extend(
                {
                    "artifact_id": self.metadata.artifact_id,
                    "group_id": self.config.group_id,
                    "overlay": family.value,
                    **{
                        column: row[column]
                        for column in SUBSYSTEM_ACTIVITY_COLUMNS
                    },
                }
                for _, row in subsystem_frame.iterrows()
            )
        return pd.DataFrame(rows, columns=list(ACTIVITY_SUBSYSTEM_EXPORT_COLUMNS))


def _summarize_group(frame: pd.DataFrame, summary: GroupSummary) -> pd.Series:
    if summary is GroupSummary.MEAN:
        return frame.mean(axis=0)
    return frame.median(axis=0)


def _assemble_activity_result(
    config: ActivityConfig,
    activity_score: pd.Series,
    *,
    overlay: ActivityOverlay,
    sample_count: int,
    reference_sample_count: int,
) -> FeatureActivityResult:
    """Assemble a normalized group summary without reference standardization."""
    score = activity_score.astype(float)
    finite = np.isfinite(score.to_numpy(dtype=float))
    non_finite = tuple(str(feature) for feature in score.index[~finite])
    warnings = (
        (
            f"{len(non_finite)} {overlay.value} feature(s) have a non-finite "
            "group summary and "
            "remain unavailable"
        ),
    ) if non_finite else ()

    return FeatureActivityResult(
        group_id=config.group_id,
        overlay=overlay.value,
        activity_score=score.where(finite),
        sample_count=sample_count,
        reference_sample_count=reference_sample_count,
        config=config,
        non_finite_feature_ids=non_finite,
        warnings=tuple(warnings),
    )


def calculate_feature_activity(
    frame: pd.DataFrame,
    group_samples: Sequence[str],
    config: ActivityConfig,
    *,
    overlay: ActivityOverlay = ActivityOverlay.REACTION,
) -> FeatureActivityResult:
    """Summarize normalized feature values for one selected group.

    Args:
        frame: samples x features matrix, already aligned and preprocessed.
        group_samples: samples belonging to ``config.group_id``.
        config: the frozen activity context.

    Returns:
        FeatureActivityResult whose activity scores are the selected summaries.

    This is the dense public entry point; :func:`calculate_feature_activity_from_matrix`
    is its storage-invariant twin that reads the same numbers through the matrix
    seam without densifying a sparse matrix whole (AM6 slice 6). Both delegate to
    one shared assembly core, so they cannot drift.
    """
    if frame is None or frame.empty:
        raise ValueError("Activity scoring requires a non-empty samples x features matrix")

    present = [str(sample) for sample in group_samples if str(sample) in frame.index]
    if not present:
        raise ValueError(
            f"Group '{config.group_id}' has no samples in the aligned matrix "
            f"({len(frame.index)} samples available)"
        )

    numeric = frame.astype(float)
    activity_score = _summarize_group(numeric.loc[present], config.summary)
    return _assemble_activity_result(
        config,
        activity_score,
        overlay=overlay,
        sample_count=len(present),
        reference_sample_count=int(len(numeric.index)),
    )


def calculate_feature_activity_from_matrix(
    matrix: "ExpressionMatrix",
    group_samples: Sequence[str],
    config: ActivityConfig,
    *,
    overlay: ActivityOverlay = ActivityOverlay.REACTION,
) -> FeatureActivityResult:
    """Storage-invariant activity scoring through the expression-matrix seam.

    The bounded twin of :func:`calculate_feature_activity`. It reads only the
    selected group's rows through ``select_samples`` and never materializes the
    whole matrix dense. The result is identical to the dense function over the
    same values.
    """
    sample_ids = matrix.sample_ids
    if not sample_ids or not matrix.feature_ids:
        raise ValueError("Activity scoring requires a non-empty samples x features matrix")

    known = set(sample_ids)
    present = [str(sample) for sample in group_samples if str(sample) in known]
    if not present:
        raise ValueError(
            f"Group '{config.group_id}' has no samples in the aligned matrix "
            f"({len(sample_ids)} samples available)"
        )

    activity_score = _summarize_group(
        matrix.select_samples(present).astype(float), config.summary
    )
    return _assemble_activity_result(
        config,
        activity_score,
        overlay=overlay,
        sample_count=len(present),
        reference_sample_count=int(len(sample_ids)),
    )


def resolve_display_direction(activity_score: pd.Series, reaction_id: str) -> str:
    """Pick the direction shown for a reversible reaction in the selected group.

    Mirrors the comparison-mode rule: the forward direction wins ties, and the
    backward direction is only shown when its magnitude exceeds the forward
    summary. Both values stay available in the result for hovers and exports.
    """
    base_id = reaction_id[:-5] if reaction_id.endswith("_back") else reaction_id
    backward_id = f"{base_id}_back"
    if backward_id not in activity_score.index:
        return "forward"

    forward = activity_score.get(base_id, np.nan)
    backward = activity_score.get(backward_id, np.nan)
    forward = 0.0 if pd.isna(forward) else float(forward)
    backward = 0.0 if pd.isna(backward) else float(backward)
    return "forward" if forward >= abs(backward) else "backward"


def displayed_feature_id(activity_score: pd.Series, reaction_id: str) -> str:
    """Return the reaction column whose value is displayed for this group."""
    base_id = reaction_id[:-5] if reaction_id.endswith("_back") else reaction_id
    if resolve_display_direction(activity_score, base_id) == "backward":
        return f"{base_id}_back"
    return base_id


def calculate_subsystem_activity(
    feature_result: FeatureActivityResult,
    subsystem_features: Mapping[str, Sequence[str]],
    *,
    use_main_direction: bool = True,
) -> pd.DataFrame:
    """Aggregate member feature scores into a subsystem activity table.

    Each subsystem's score is the **median** of its measured member scores, a
    robust statistic that a handful of extreme reactions cannot dominate. A
    subsystem whose measured coverage falls below
    ``config.minimum_subsystem_coverage`` reports ``None`` instead of a score,
    so the graph can gray it out rather than imply a measurement.

    Args:
        feature_result: per-feature scores for the selected group.
        subsystem_features: subsystem id -> member feature ids from the model.
        use_main_direction: for reactions, score the direction the group
            actually displays instead of always scoring the forward column.
    """
    scores = feature_result.activity_score
    minimum_coverage = feature_result.config.minimum_subsystem_coverage
    is_reaction_overlay = feature_result.overlay == ActivityOverlay.REACTION.value

    rows: list[SubsystemActivityRow] = []
    for subsystem_id, members in subsystem_features.items():
        member_ids = [str(member) for member in members]
        total = len(member_ids)

        measured_values: list[float] = []
        for member in member_ids:
            feature_id = member
            if is_reaction_overlay and use_main_direction:
                feature_id = displayed_feature_id(feature_result.activity_score, member)
            if feature_id not in scores.index:
                continue
            value = scores.loc[feature_id]
            if pd.isna(value):
                continue
            measured_values.append(float(value))

        measured = len(measured_values)
        coverage = (measured / total) if total else 0.0
        meets_coverage = bool(measured) and coverage >= minimum_coverage

        rows.append(
            SubsystemActivityRow(
                subsystem_id=str(subsystem_id),
                score=float(np.median(measured_values)) if meets_coverage else None,
                total_model_reactions=total,
                measured_reactions=measured,
                coverage=coverage,
                positive_count=sum(1 for value in measured_values if value > 0),
                negative_count=sum(1 for value in measured_values if value < 0),
            )
        )

    frame = pd.DataFrame(
        [
            {
                "subsystem": row.subsystem_id,
                "activity_score": row.score,
                "total_model_features": row.total_model_reactions,
                "measured_features": row.measured_reactions,
                "coverage": row.coverage,
                "positive_count": row.positive_count,
                "negative_count": row.negative_count,
            }
            for row in rows
        ],
        columns=list(SUBSYSTEM_ACTIVITY_COLUMNS),
    )
    if frame.empty:
        return frame
    return frame.set_index("subsystem", drop=False).sort_values(
        "activity_score", ascending=False, na_position="last", kind="mergesort"
    )

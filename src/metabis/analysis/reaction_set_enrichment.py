"""Pure reaction-set over-representation analysis over subsystem annotations.

The engine consumes a frozen comparison artifact and plain subsystem annotation
records.  It does not access managers, renderers, or Bokeh, and it never treats
the descriptive subsystem delta score as an inferential result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

from ..domain.metabolic_model import MetabolicModel
from .artifacts import (
    AnalysisArtifactMetadata,
    ArtifactDimension,
    ArtifactKind,
    ArtifactProvenance,
    PlainTable,
    fingerprint_json,
)
from .comparison_artifact import GroupComparisonArtifact
from .comparison_membership import ComparisonMembership
from .reaction_direction import (
    DEFAULT_REACTION_DIRECTION_CRITERION,
    ReactionDirectionCriterion,
    ReactionDirectionMap,
    build_reaction_direction_map,
    to_base_reaction_id,
)


class EnrichmentDirection(str, Enum):
    """Which significant reaction signs enter the enrichment query."""

    EITHER = "either"
    UP = "up"
    DOWN = "down"


@dataclass(frozen=True)
class ReactionSetEnrichmentConfig:
    """Every choice that changes reaction-set enrichment output."""

    direction: EnrichmentDirection = EnrichmentDirection.EITHER
    direction_metric: Literal["cohens_d", "log2_fc", "mean_difference"] = "cohens_d"
    correction: Literal["fdr_bh", "holm"] = "fdr_bh"
    minimum_measured_set_size: int = 3
    adjusted_pvalue_threshold: float = 0.05
    significance_fingerprint: str = ""
    reaction_direction_criterion: ReactionDirectionCriterion = (
        DEFAULT_REACTION_DIRECTION_CRITERION
    )

    def __post_init__(self) -> None:
        if not isinstance(self.direction, EnrichmentDirection):
            raise TypeError("direction must be an EnrichmentDirection")
        if self.direction_metric not in {"cohens_d", "log2_fc", "mean_difference"}:
            raise ValueError("Unsupported enrichment direction metric")
        if self.correction not in {"fdr_bh", "holm"}:
            raise ValueError("Enrichment correction must be 'fdr_bh' or 'holm'")
        if (
            isinstance(self.minimum_measured_set_size, bool)
            or not isinstance(self.minimum_measured_set_size, int)
            or self.minimum_measured_set_size < 1
        ):
            raise ValueError("minimum_measured_set_size must be a positive integer")
        if not 0.0 < self.adjusted_pvalue_threshold <= 1.0:
            raise ValueError("adjusted_pvalue_threshold must be in (0, 1]")
        if not isinstance(
            self.reaction_direction_criterion, ReactionDirectionCriterion
        ):
            raise TypeError(
                "reaction_direction_criterion must be a ReactionDirectionCriterion"
            )


@dataclass(frozen=True)
class SubsystemAnnotation:
    """One named, potentially overlapping subsystem reaction set."""

    subsystem_id: str
    subsystem_name: str
    reaction_ids: tuple[str, ...]
    category: str | None = None

    def __post_init__(self) -> None:
        if not self.subsystem_id or not self.subsystem_name:
            raise ValueError("Subsystem annotations require canonical ID and name")
        normalized = tuple(sorted({to_base_reaction_id(item) for item in self.reaction_ids}))
        object.__setattr__(self, "reaction_ids", normalized)


def build_subsystem_annotations(
    metabolic_model: MetabolicModel,
) -> tuple[SubsystemAnnotation, ...]:
    """Project a Metabolic Model into deterministic enrichment annotations.

    This adapter owns the model's subsystem-entity layout so callers never need
    to traverse ``model.subsystems`` or reconstruct the canonical ID, display
    name, category, and reaction-set mapping in presentation code.
    """
    if not isinstance(metabolic_model, MetabolicModel):
        raise TypeError("metabolic_model must be a MetabolicModel")

    return tuple(
        SubsystemAnnotation(
            subsystem_id=str(subsystem.id),
            subsystem_name=str(subsystem.name),
            reaction_ids=tuple(map(str, subsystem.reaction_ids)),
            category=str(subsystem.category) if subsystem.category else None,
        )
        for subsystem in sorted(
            metabolic_model.subsystems.get_all(),
            key=lambda item: str(item.id),
        )
    )


@dataclass(frozen=True)
class ReactionSetUniverse:
    """Exact background, query, and overlapping annotation sets used by ORA."""

    membership: ComparisonMembership
    tested_reaction_ids: frozenset[str]
    query_reaction_ids: frozenset[str]
    subsystem_reaction_ids: Mapping[str, frozenset[str]]
    subsystem_annotations: tuple[SubsystemAnnotation, ...]
    reaction_direction_map: ReactionDirectionMap
    unmapped_query_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.membership, ComparisonMembership):
            raise TypeError("membership must be ComparisonMembership")
        if not self.query_reaction_ids <= self.tested_reaction_ids:
            raise ValueError("Enrichment query must be a subset of the tested universe")
        if self.reaction_direction_map.membership_fingerprint != self.membership.fingerprint:
            raise ValueError("Reaction direction map belongs to different membership")
        frozen = {
            str(key): frozenset(map(str, value))
            for key, value in sorted(self.subsystem_reaction_ids.items())
        }
        object.__setattr__(self, "subsystem_reaction_ids", MappingProxyType(frozen))


ENRICHMENT_RESULT_COLUMNS = (
    "artifact_id",
    "subsystem_id",
    "subsystem_name",
    "category",
    "universe_size",
    "subsystem_size",
    "query_size",
    "overlap_size",
    "overlap_reaction_ids",
    "overlap_source_columns",
    "enrichment_ratio",
    "pval",
    "adjusted_pval",
    "direction",
    "direction_metric",
    "correction",
    "is_significant",
)


@dataclass(frozen=True)
class ReactionSetEnrichmentArtifact:
    """Portable enrichment result and the exact universe that produced it."""

    metadata: AnalysisArtifactMetadata
    config: ReactionSetEnrichmentConfig
    universe: ReactionSetUniverse
    results: PlainTable
    warnings: tuple[str, ...]
    explanation: str

    def __post_init__(self) -> None:
        if self.metadata.kind is not ArtifactKind.REACTION_SET_ENRICHMENT:
            raise ValueError("Enrichment artifact metadata has the wrong kind")
        if self.metadata.warnings != self.warnings:
            raise ValueError("Enrichment warnings do not match common metadata")
        if self.metadata.scientific_identity.get("membership_fingerprint") != (
            self.universe.membership.fingerprint
        ):
            raise ValueError("Enrichment membership does not match common metadata")
        if tuple(self.results.columns) != ENRICHMENT_RESULT_COLUMNS:
            raise ValueError("Enrichment results do not have the canonical schema")

    def result_frame(self) -> pd.DataFrame:
        """Return a detached full-precision canonical result table."""
        return self.results.to_frame()


def _annotation_manifest(
    annotations: Sequence[SubsystemAnnotation],
) -> list[dict[str, object]]:
    return [
        {
            "subsystem_id": item.subsystem_id,
            "subsystem_name": item.subsystem_name,
            "category": item.category,
            "reaction_ids": list(item.reaction_ids),
        }
        for item in sorted(annotations, key=lambda value: value.subsystem_id)
    ]


def build_reaction_set_universe(
    comparison: GroupComparisonArtifact,
    subsystem_annotations: Sequence[SubsystemAnnotation],
    config: ReactionSetEnrichmentConfig,
) -> ReactionSetUniverse:
    """Build the unique tested/mapped base universe and directional query."""
    if not isinstance(comparison, GroupComparisonArtifact):
        raise TypeError("comparison must be a GroupComparisonArtifact")
    annotations = tuple(sorted(subsystem_annotations, key=lambda item: item.subsystem_id))
    if len({item.subsystem_id for item in annotations}) != len(annotations):
        raise ValueError("Subsystem annotation IDs must be unique")
    results = comparison.results_frame("reaction")
    if config.direction_metric not in results.columns:
        raise ValueError(
            f"Reaction results do not contain direction metric '{config.direction_metric}'"
        )

    direction_map = build_reaction_direction_map(
        comparison, config.reaction_direction_criterion
    )
    annotated = {
        reaction_id
        for annotation in annotations
        for reaction_id in annotation.reaction_ids
    }
    tested_base = {entry.base_reaction_id for entry in direction_map.entries}
    tested_mapped = frozenset(tested_base & annotated)

    significant = set(map(str, comparison.significant_reactions))
    directional: set[str] = set()
    for entry in direction_map.entries:
        reaction_id = entry.source_column
        base_id = entry.base_reaction_id
        if reaction_id not in significant:
            continue
        row = results.loc[reaction_id]
        metric = row.get(config.direction_metric)
        if pd.isna(metric):
            continue
        if config.direction is EnrichmentDirection.UP and not float(metric) > 0:
            continue
        if config.direction is EnrichmentDirection.DOWN and not float(metric) < 0:
            continue
        directional.add(base_id)

    query = frozenset(directional & tested_mapped)
    unselectable_significant = {
        to_base_reaction_id(reaction_id)
        for reaction_id in significant
        if to_base_reaction_id(reaction_id) not in tested_base
    }
    unmapped = tuple(
        sorted((directional - tested_mapped) | unselectable_significant)
    )
    subsystem_sets = {
        item.subsystem_id: frozenset(item.reaction_ids) & tested_mapped
        for item in annotations
    }
    return ReactionSetUniverse(
        membership=comparison.membership,
        tested_reaction_ids=tested_mapped,
        query_reaction_ids=query,
        subsystem_reaction_ids=subsystem_sets,
        subsystem_annotations=annotations,
        reaction_direction_map=direction_map,
        unmapped_query_ids=unmapped,
    )


def run_reaction_set_enrichment(
    comparison: GroupComparisonArtifact,
    subsystem_annotations: Sequence[SubsystemAnnotation],
    config: ReactionSetEnrichmentConfig = ReactionSetEnrichmentConfig(),
) -> ReactionSetEnrichmentArtifact:
    """Run one-sided hypergeometric ORA and attach portable provenance."""
    universe = build_reaction_set_universe(comparison, subsystem_annotations, config)
    warnings: list[str] = []
    if universe.unmapped_query_ids:
        warnings.append(
            f"{len(universe.unmapped_query_ids)} significant base reaction(s) were "
            "not in the tested/mapped universe"
        )
    if not universe.query_reaction_ids:
        warnings.append("The directional significant-reaction query is empty")

    annotation_by_id = {
        item.subsystem_id: item for item in universe.subsystem_annotations
    }
    eligible = [
        subsystem_id
        for subsystem_id, reaction_ids in universe.subsystem_reaction_ids.items()
        if len(reaction_ids) >= config.minimum_measured_set_size
    ]
    excluded_count = len(universe.subsystem_reaction_ids) - len(eligible)
    if excluded_count:
        warnings.append(
            f"{excluded_count} subsystem(s) had fewer than "
            f"{config.minimum_measured_set_size} measured reactions and were excluded"
        )

    N = len(universe.tested_reaction_ids)
    n = len(universe.query_reaction_ids)
    rows: list[dict[str, object]] = []
    pvalues: list[float] = []
    for subsystem_id in eligible:
        annotation = annotation_by_id[subsystem_id]
        subsystem = universe.subsystem_reaction_ids[subsystem_id]
        overlap = tuple(sorted(universe.query_reaction_ids & subsystem))
        overlap_sources = tuple(
            universe.reaction_direction_map.source_column_for(base_id)
            for base_id in overlap
        )
        K = len(subsystem)
        k = len(overlap)
        pvalue = float(hypergeom.sf(k - 1, N, K, n)) if n else 1.0
        ratio = ((k / n) / (K / N)) if n and K and N else None
        pvalues.append(pvalue)
        rows.append(
            {
                "subsystem_id": subsystem_id,
                "subsystem_name": annotation.subsystem_name,
                "category": annotation.category,
                "universe_size": N,
                "subsystem_size": K,
                "query_size": n,
                "overlap_size": k,
                "overlap_reaction_ids": json.dumps(list(overlap), separators=(",", ":")),
                "overlap_source_columns": json.dumps(
                    list(overlap_sources), separators=(",", ":")
                ),
                "enrichment_ratio": ratio,
                "pval": pvalue,
            }
        )

    adjusted = (
        multipletests(pvalues, method=config.correction)[1]
        if pvalues
        else []
    )
    for row, adjusted_pvalue in zip(rows, adjusted):
        row["adjusted_pval"] = float(adjusted_pvalue)
        row["direction"] = config.direction.value
        row["direction_metric"] = config.direction_metric
        row["correction"] = config.correction
        row["is_significant"] = bool(
            adjusted_pvalue <= config.adjusted_pvalue_threshold
        )

    annotations_manifest = _annotation_manifest(universe.subsystem_annotations)
    annotation_fingerprint = fingerprint_json(annotations_manifest)
    threshold_identity = {
        "comparison_artifact_id": comparison.metadata.artifact_id,
        "reaction_thresholds": dict(comparison.reaction_thresholds.as_mapping()),
        "significant_reaction_ids": list(comparison.significant_reactions),
        "keep_reversed": comparison.keep_reversed,
    }
    significance_fingerprint = config.significance_fingerprint or fingerprint_json(
        threshold_identity
    )
    threshold_fingerprint = fingerprint_json(threshold_identity)
    source_fingerprints = dict(comparison.metadata.source.source_fingerprints)
    source_fingerprints["subsystem_annotations"] = annotation_fingerprint
    source = ArtifactProvenance(
        workspace_id=comparison.metadata.source.workspace_id,
        revisions=comparison.metadata.source.revisions,
        dependencies=comparison.metadata.source.dependencies,
        source_fingerprints=source_fingerprints,
    )
    metadata = AnalysisArtifactMetadata(
        kind=ArtifactKind.REACTION_SET_ENRICHMENT,
        source=source,
        method="one_sided_hypergeometric_over_representation",
        configuration={
            "direction": config.direction.value,
            "direction_metric": config.direction_metric,
            "correction": config.correction,
            "minimum_measured_set_size": config.minimum_measured_set_size,
            "adjusted_pvalue_threshold": config.adjusted_pvalue_threshold,
            "reaction_direction_criterion": (
                config.reaction_direction_criterion.value
            ),
            "reaction_direction_fingerprint": (
                universe.reaction_direction_map.fingerprint
            ),
            "universe_rule": "unique tested and subsystem-mapped base reactions",
        },
        scientific_identity={
            "comparison_artifact_id": comparison.metadata.artifact_id,
            "membership_fingerprint": comparison.membership.fingerprint,
            "target_base_group_ids": comparison.membership.target_base_group_ids,
            "baseline_base_group_ids": comparison.membership.baseline_base_group_ids,
            "target_samples": comparison.membership.target_samples,
            "baseline_samples": comparison.membership.baseline_samples,
            "significance_fingerprint": significance_fingerprint,
            "comparison_threshold_fingerprint": threshold_fingerprint,
            "subsystem_annotation_fingerprint": annotation_fingerprint,
            "tested_reaction_ids": sorted(universe.tested_reaction_ids),
            "query_reaction_ids": sorted(universe.query_reaction_ids),
            "reaction_direction_fingerprint": (
                universe.reaction_direction_map.fingerprint
            ),
        },
        dimensions=(
            ArtifactDimension("eligible_subsystems", len(rows)),
            ArtifactDimension("query_reactions", n),
            ArtifactDimension("tested_reactions", N),
        ),
        annotations={
            "subsystems_overlap": True,
            "topology_aware": False,
            "subsystem_source": annotations_manifest,
            "comparison_threshold_identity": threshold_identity,
            "comparison_test": comparison.test,
            "comparison_correction": comparison.correction,
            "reaction_direction_audit": [
                {
                    "base_reaction_id": entry.base_reaction_id,
                    "source_column": entry.source_column,
                    "selected_direction": entry.selected_direction.value,
                    "target_mean_activity": entry.target_mean_activity,
                    "baseline_mean_activity": entry.baseline_mean_activity,
                    "criterion": entry.criterion.value,
                    "reason": entry.reason,
                }
                for entry in universe.reaction_direction_map.entries
                if entry.base_reaction_id in universe.tested_reaction_ids
            ],
        },
        warnings=tuple(warnings),
    )
    for row in rows:
        row["artifact_id"] = metadata.artifact_id
    frame = pd.DataFrame(rows, columns=list(ENRICHMENT_RESULT_COLUMNS))
    explanation = (
        f"Tested {len(rows)} eligible overlapping subsystem annotation set(s) "
        f"against N={N} unique mapped base reactions with query n={n}."
    )
    return ReactionSetEnrichmentArtifact(
        metadata=metadata,
        config=config,
        universe=universe,
        results=PlainTable.capture(frame),
        warnings=tuple(warnings),
        explanation=explanation,
    )


__all__ = [
    "ENRICHMENT_RESULT_COLUMNS",
    "EnrichmentDirection",
    "ReactionSetEnrichmentArtifact",
    "ReactionSetEnrichmentConfig",
    "ReactionSetUniverse",
    "SubsystemAnnotation",
    "build_reaction_set_universe",
    "build_subsystem_annotations",
    "run_reaction_set_enrichment",
    "to_base_reaction_id",
]

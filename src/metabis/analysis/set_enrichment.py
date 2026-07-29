"""Shared paired-direction over-representation analysis for feature sets.

Feature-family adapters own mapping and significance decisions.  This module
owns the scientific operation they share: one hypergeometric test for each
eligible subsystem and Target/Baseline direction, followed by one joint
multiple-testing correction across both direction families.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal, Mapping

import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

from ..analysis_workspace import RevisionKind
from ..domain.metabolic_model import MetabolicModel
from .artifacts import (
    AnalysisArtifactMetadata,
    ArtifactDimension,
    ArtifactKind,
    ArtifactProvenance,
    PlainTable,
    fingerprint_frame,
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


class SetEnrichmentFeatureType(str, Enum):
    """Feature family whose tested universe is analysed."""

    REACTION = "reaction"
    GENE = "gene"


class SetEnrichmentDirection(str, Enum):
    """Target-minus-Baseline direction represented by one query."""

    TARGET_HIGHER = "target_higher"
    TARGET_LOWER = "target_lower"


@dataclass(frozen=True)
class SetEnrichmentConfig:
    """Choices shared by reaction- and gene-set ORA."""

    correction: Literal["fdr_bh", "holm"] = "fdr_bh"
    minimum_measured_set_size: int = 3
    adjusted_pvalue_threshold: float = 0.05
    direction_metric: Literal["cohens_d", "log2_fc", "mean_difference"] = "cohens_d"
    reaction_direction_criterion: ReactionDirectionCriterion | None = None

    def __post_init__(self) -> None:
        if self.correction not in {"fdr_bh", "holm"}:
            raise ValueError("Set-enrichment correction must be 'fdr_bh' or 'holm'")
        if (
            isinstance(self.minimum_measured_set_size, bool)
            or not isinstance(self.minimum_measured_set_size, int)
            or self.minimum_measured_set_size < 1
        ):
            raise ValueError("minimum_measured_set_size must be a positive integer")
        if not 0.0 < self.adjusted_pvalue_threshold <= 1.0:
            raise ValueError("adjusted_pvalue_threshold must be in (0, 1]")
        if self.direction_metric not in {
            "cohens_d",
            "log2_fc",
            "mean_difference",
        }:
            raise ValueError("Unsupported set-enrichment direction metric")
        if self.reaction_direction_criterion is not None and not isinstance(
            self.reaction_direction_criterion, ReactionDirectionCriterion
        ):
            raise TypeError(
                "reaction_direction_criterion must be a "
                "ReactionDirectionCriterion or None"
            )


@dataclass(frozen=True)
class SetEnrichmentAnnotation:
    """One named, potentially overlapping subsystem feature set."""

    subsystem_id: str
    subsystem_name: str
    feature_ids: tuple[str, ...]
    category: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.subsystem_id, str) or not self.subsystem_id:
            raise ValueError("Set annotations require a canonical subsystem ID")
        if not isinstance(self.subsystem_name, str) or not self.subsystem_name:
            raise ValueError("Set annotations require a subsystem name")
        normalized = _feature_ids(self.feature_ids, field_name="feature_ids")
        object.__setattr__(self, "feature_ids", tuple(sorted(normalized)))


@dataclass(frozen=True)
class SetEnrichmentUniverse:
    """Exact tested universe, paired queries, and measured subsystem sets."""

    feature_type: SetEnrichmentFeatureType
    tested_feature_ids: frozenset[str]
    target_higher_query_ids: frozenset[str]
    target_lower_query_ids: frozenset[str]
    subsystem_annotations: tuple[SetEnrichmentAnnotation, ...]
    unmapped_target_higher_ids: tuple[str, ...] = ()
    unmapped_target_lower_ids: tuple[str, ...] = ()
    subsystem_feature_ids: Mapping[str, frozenset[str]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.feature_type, SetEnrichmentFeatureType):
            raise TypeError("feature_type must be a SetEnrichmentFeatureType")
        tested = _feature_ids(self.tested_feature_ids, field_name="tested_feature_ids")
        target_higher = _feature_ids(
            self.target_higher_query_ids,
            field_name="target_higher_query_ids",
        )
        target_lower = _feature_ids(
            self.target_lower_query_ids,
            field_name="target_lower_query_ids",
        )
        if not target_higher <= tested or not target_lower <= tested:
            raise ValueError(
                "Each set-enrichment query must be a subset of the universe"
            )
        overlap = target_higher & target_lower
        if overlap:
            sample = ", ".join(sorted(overlap)[:5])
            raise ValueError(
                "Target-higher and Target-lower queries must be disjoint; "
                f"overlap includes {sample}"
            )

        annotations = tuple(
            sorted(self.subsystem_annotations, key=lambda item: item.subsystem_id)
        )
        if not all(isinstance(item, SetEnrichmentAnnotation) for item in annotations):
            raise TypeError(
                "subsystem_annotations must contain SetEnrichmentAnnotation values"
            )
        annotation_ids = tuple(item.subsystem_id for item in annotations)
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError("Subsystem annotation IDs must be unique")

        object.__setattr__(self, "tested_feature_ids", frozenset(tested))
        object.__setattr__(self, "target_higher_query_ids", frozenset(target_higher))
        object.__setattr__(self, "target_lower_query_ids", frozenset(target_lower))
        object.__setattr__(self, "subsystem_annotations", annotations)
        object.__setattr__(
            self,
            "unmapped_target_higher_ids",
            tuple(
                sorted(
                    _feature_ids(
                        self.unmapped_target_higher_ids,
                        field_name="unmapped_target_higher_ids",
                    )
                )
            ),
        )
        object.__setattr__(
            self,
            "unmapped_target_lower_ids",
            tuple(
                sorted(
                    _feature_ids(
                        self.unmapped_target_lower_ids,
                        field_name="unmapped_target_lower_ids",
                    )
                )
            ),
        )
        object.__setattr__(
            self,
            "subsystem_feature_ids",
            MappingProxyType(
                {
                    annotation.subsystem_id: (
                        frozenset(annotation.feature_ids) & frozenset(tested)
                    )
                    for annotation in annotations
                }
            ),
        )

    def query_ids(self, direction: SetEnrichmentDirection) -> frozenset[str]:
        """Return the immutable query for one signed direction."""
        if direction is SetEnrichmentDirection.TARGET_HIGHER:
            return self.target_higher_query_ids
        if direction is SetEnrichmentDirection.TARGET_LOWER:
            return self.target_lower_query_ids
        raise TypeError("direction must be a SetEnrichmentDirection")


SET_ENRICHMENT_RESULT_COLUMNS = (
    "subsystem_id",
    "subsystem_name",
    "category",
    "feature_type",
    "direction",
    "universe_size",
    "subsystem_size",
    "query_size",
    "overlap_size",
    "overlap_feature_ids",
    "enrichment_ratio",
    "signed_log2_enrichment_ratio",
    "pval",
    "adjusted_pval",
    "correction",
    "correction_family_size",
    "is_significant",
)


@dataclass(frozen=True)
class SetEnrichmentResult:
    """Renderer-independent paired ORA result produced by the shared kernel."""

    config: SetEnrichmentConfig
    universe: SetEnrichmentUniverse
    results: PlainTable
    warnings: tuple[str, ...]
    explanation: str

    def __post_init__(self) -> None:
        if not isinstance(self.config, SetEnrichmentConfig):
            raise TypeError("config must be a SetEnrichmentConfig")
        if not isinstance(self.universe, SetEnrichmentUniverse):
            raise TypeError("universe must be a SetEnrichmentUniverse")
        if tuple(self.results.columns) != SET_ENRICHMENT_RESULT_COLUMNS:
            raise ValueError("Set-enrichment results do not have the canonical schema")

    def result_frame(self) -> pd.DataFrame:
        """Return a detached full-precision canonical result table."""
        return self.results.to_frame()


SET_ENRICHMENT_ARTIFACT_RESULT_COLUMNS = (
    "artifact_id",
    *SET_ENRICHMENT_RESULT_COLUMNS,
)


@dataclass(frozen=True)
class SetEnrichmentArtifact:
    """Portable paired-direction ORA result for one measured feature family."""

    metadata: AnalysisArtifactMetadata
    membership: ComparisonMembership
    config: SetEnrichmentConfig
    universe: SetEnrichmentUniverse
    results: PlainTable
    warnings: tuple[str, ...]
    explanation: str
    reaction_direction_map: ReactionDirectionMap | None = None

    def __post_init__(self) -> None:
        if self.metadata.kind is not ArtifactKind.SET_ENRICHMENT:
            raise ValueError("Set-enrichment artifact metadata has the wrong kind")
        if self.metadata.warnings != self.warnings:
            raise ValueError("Set-enrichment warnings do not match common metadata")
        identity = self.metadata.scientific_identity
        if identity.get("membership_fingerprint") != self.membership.fingerprint:
            raise ValueError("Set-enrichment membership does not match metadata")
        if identity.get("feature_type") != self.universe.feature_type.value:
            raise ValueError("Set-enrichment feature type does not match metadata")
        if tuple(self.results.columns) != SET_ENRICHMENT_RESULT_COLUMNS:
            raise ValueError("Set-enrichment results do not have the canonical schema")
        if self.universe.feature_type is SetEnrichmentFeatureType.REACTION:
            if self.reaction_direction_map is None:
                raise ValueError("Reaction-set enrichment requires a direction map")
            if self.config.reaction_direction_criterion is None:
                raise ValueError(
                    "Reaction-set enrichment requires a direction criterion"
                )
            if (
                self.reaction_direction_map.membership_fingerprint
                != self.membership.fingerprint
            ):
                raise ValueError(
                    "Reaction direction map belongs to different membership"
                )
            if (
                self.reaction_direction_map.criterion
                is not self.config.reaction_direction_criterion
            ):
                raise ValueError("Reaction direction map uses a different criterion")
        elif self.reaction_direction_map is not None:
            raise ValueError(
                "Gene-set enrichment cannot carry a reaction direction map"
            )

    @property
    def feature_type(self) -> SetEnrichmentFeatureType:
        """Feature family represented by this artifact."""
        return self.universe.feature_type

    def result_frame(self) -> pd.DataFrame:
        """Return the portable full-precision table with artifact identity."""
        frame = self.results.to_frame()
        frame.insert(0, "artifact_id", self.metadata.artifact_id)
        return frame.loc[:, list(SET_ENRICHMENT_ARTIFACT_RESULT_COLUMNS)]

    def require_current_comparison(self, comparison: GroupComparisonArtifact) -> None:
        """Reject installation against changed membership or family evidence."""
        if not isinstance(comparison, GroupComparisonArtifact):
            raise TypeError("comparison must be a GroupComparisonArtifact")
        if comparison.membership != self.membership:
            raise ValueError("Set enrichment belongs to a different comparison")
        current = _comparison_family_identity(
            comparison,
            self.feature_type,
            self.config.direction_metric,
        )
        identity = self.metadata.scientific_identity
        if (
            identity.get("comparison_feature_fingerprint")
            != current["comparison_feature_fingerprint"]
        ):
            raise ValueError("Set-enrichment comparison evidence is not current")
        if (
            identity.get("comparison_threshold_fingerprint")
            != current["comparison_threshold_fingerprint"]
        ):
            raise ValueError("Set-enrichment thresholds are not current")


def run_set_enrichment(
    universe: SetEnrichmentUniverse,
    config: SetEnrichmentConfig = SetEnrichmentConfig(),
) -> SetEnrichmentResult:
    """Run paired one-sided hypergeometric ORA with joint family correction."""
    if not isinstance(universe, SetEnrichmentUniverse):
        raise TypeError("universe must be a SetEnrichmentUniverse")
    if not isinstance(config, SetEnrichmentConfig):
        raise TypeError("config must be a SetEnrichmentConfig")

    warnings = _warnings(universe, config)
    annotation_by_id = {
        item.subsystem_id: item for item in universe.subsystem_annotations
    }
    eligible = tuple(
        subsystem_id
        for subsystem_id, feature_ids in universe.subsystem_feature_ids.items()
        if len(feature_ids) >= config.minimum_measured_set_size
    )
    correction_family_size = len(eligible) * len(SetEnrichmentDirection)
    universe_size = len(universe.tested_feature_ids)
    rows: list[dict[str, object]] = []
    pvalues: list[float] = []

    for subsystem_id in eligible:
        annotation = annotation_by_id[subsystem_id]
        subsystem = universe.subsystem_feature_ids[subsystem_id]
        subsystem_size = len(subsystem)
        for direction in SetEnrichmentDirection:
            query = universe.query_ids(direction)
            query_size = len(query)
            overlap = tuple(sorted(query & subsystem))
            overlap_size = len(overlap)
            pvalue = (
                float(
                    hypergeom.sf(
                        overlap_size - 1,
                        universe_size,
                        subsystem_size,
                        query_size,
                    )
                )
                if query_size
                else 1.0
            )
            ratio = (
                (overlap_size / query_size) / (subsystem_size / universe_size)
                if query_size and subsystem_size and universe_size
                else None
            )
            rows.append(
                {
                    "subsystem_id": subsystem_id,
                    "subsystem_name": annotation.subsystem_name,
                    "category": annotation.category,
                    "feature_type": universe.feature_type.value,
                    "direction": direction.value,
                    "universe_size": universe_size,
                    "subsystem_size": subsystem_size,
                    "query_size": query_size,
                    "overlap_size": overlap_size,
                    "overlap_feature_ids": json.dumps(
                        list(overlap), separators=(",", ":")
                    ),
                    "enrichment_ratio": ratio,
                    "signed_log2_enrichment_ratio": _signed_visual(ratio, direction),
                    "pval": pvalue,
                }
            )
            pvalues.append(pvalue)

    adjusted = multipletests(pvalues, method=config.correction)[1] if pvalues else ()
    for row, adjusted_pvalue in zip(rows, adjusted):
        row["adjusted_pval"] = float(adjusted_pvalue)
        row["correction"] = config.correction
        row["correction_family_size"] = correction_family_size
        row["is_significant"] = bool(
            adjusted_pvalue <= config.adjusted_pvalue_threshold
        )

    frame = pd.DataFrame(rows, columns=list(SET_ENRICHMENT_RESULT_COLUMNS))
    explanation = (
        f"Tested {len(eligible)} eligible overlapping subsystem annotation set(s) "
        f"in both Target directions against N={universe_size} unique mapped "
        f"{universe.feature_type.value} features; jointly corrected "
        f"{correction_family_size} hypotheses."
    )
    return SetEnrichmentResult(
        config=config,
        universe=universe,
        results=PlainTable.capture(frame),
        warnings=tuple(warnings),
        explanation=explanation,
    )


def build_set_enrichment_annotations(
    metabolic_model: MetabolicModel,
    feature_type: SetEnrichmentFeatureType,
) -> tuple[SetEnrichmentAnnotation, ...]:
    """Project Human1 reaction/GPR membership into deterministic feature sets."""
    if not isinstance(metabolic_model, MetabolicModel):
        raise TypeError("metabolic_model must be a MetabolicModel")
    if not isinstance(feature_type, SetEnrichmentFeatureType):
        raise TypeError("feature_type must be a SetEnrichmentFeatureType")

    annotations: list[SetEnrichmentAnnotation] = []
    for subsystem in sorted(
        metabolic_model.subsystems.get_all(), key=lambda item: str(item.id)
    ):
        if feature_type is SetEnrichmentFeatureType.REACTION:
            feature_ids = tuple(
                sorted(
                    {
                        to_base_reaction_id(str(reaction_id))
                        for reaction_id in subsystem.reaction_ids
                    }
                )
            )
        else:
            # ``gene_ids`` is populated from each Human1 reaction's GPR membership
            # by ``build_subsystem_content_relationships`` during model loading.
            feature_ids = tuple(sorted(map(str, subsystem.gene_ids)))
        annotations.append(
            SetEnrichmentAnnotation(
                subsystem_id=str(subsystem.id),
                subsystem_name=str(subsystem.name),
                feature_ids=feature_ids,
                category=str(subsystem.category) if subsystem.category else None,
            )
        )
    return tuple(annotations)


def build_gene_set_annotations(
    metabolic_model: MetabolicModel,
) -> tuple[SetEnrichmentAnnotation, ...]:
    """Build deterministic Human1 gene sets derived through reaction GPRs."""
    return build_set_enrichment_annotations(
        metabolic_model, SetEnrichmentFeatureType.GENE
    )


def build_comparison_set_enrichment_universe(
    comparison: GroupComparisonArtifact,
    subsystem_annotations: Iterable[SetEnrichmentAnnotation],
    feature_type: SetEnrichmentFeatureType,
    config: SetEnrichmentConfig,
) -> tuple[SetEnrichmentUniverse, ReactionDirectionMap | None]:
    """Build one exact mapped universe and its two signed significance queries."""
    if not isinstance(comparison, GroupComparisonArtifact):
        raise TypeError("comparison must be a GroupComparisonArtifact")
    if not isinstance(feature_type, SetEnrichmentFeatureType):
        raise TypeError("feature_type must be a SetEnrichmentFeatureType")
    if not isinstance(config, SetEnrichmentConfig):
        raise TypeError("config must be a SetEnrichmentConfig")
    annotations = tuple(subsystem_annotations)
    if not all(isinstance(item, SetEnrichmentAnnotation) for item in annotations):
        raise TypeError(
            "subsystem_annotations must contain SetEnrichmentAnnotation values"
        )

    family: Literal["reaction", "gene"] = (
        "reaction" if feature_type is SetEnrichmentFeatureType.REACTION else "gene"
    )
    results = comparison.results_frame(family)
    if config.direction_metric not in results.columns:
        raise ValueError(
            f"{family.title()} results do not contain direction metric "
            f"'{config.direction_metric}'"
        )
    annotated_ids = {
        feature_id
        for annotation in annotations
        for feature_id in annotation.feature_ids
    }
    reaction_map: ReactionDirectionMap | None = None
    directional: dict[SetEnrichmentDirection, set[str]] = {
        direction: set() for direction in SetEnrichmentDirection
    }

    if feature_type is SetEnrichmentFeatureType.REACTION:
        criterion = config.reaction_direction_criterion
        if criterion is None:
            raise ValueError(
                "Reaction-set enrichment requires reaction_direction_criterion"
            )
        reaction_map = build_reaction_direction_map(comparison, criterion)
        selectable = {entry.base_reaction_id for entry in reaction_map.entries}
        tested_mapped = frozenset(selectable & annotated_ids)
        significant = set(map(str, comparison.significant_reactions))
        for entry in reaction_map.entries:
            if entry.source_column not in significant:
                continue
            metric = results.loc[entry.source_column].get(config.direction_metric)
            direction = _metric_direction(metric)
            if direction is not None:
                directional[direction].add(entry.base_reaction_id)
    else:
        if config.reaction_direction_criterion is not None:
            raise ValueError(
                "reaction_direction_criterion applies only to reaction-set enrichment"
            )
        tested = set(map(str, results.index))
        tested_mapped = frozenset(tested & annotated_ids)
        significant = set(map(str, comparison.significant_genes))
        for gene_id in sorted(significant & tested):
            metric = results.loc[gene_id].get(config.direction_metric)
            direction = _metric_direction(metric)
            if direction is not None:
                directional[direction].add(gene_id)

    target_higher = frozenset(
        directional[SetEnrichmentDirection.TARGET_HIGHER] & tested_mapped
    )
    target_lower = frozenset(
        directional[SetEnrichmentDirection.TARGET_LOWER] & tested_mapped
    )
    universe = SetEnrichmentUniverse(
        feature_type=feature_type,
        tested_feature_ids=tested_mapped,
        target_higher_query_ids=target_higher,
        target_lower_query_ids=target_lower,
        subsystem_annotations=annotations,
        unmapped_target_higher_ids=tuple(
            sorted(directional[SetEnrichmentDirection.TARGET_HIGHER] - tested_mapped)
        ),
        unmapped_target_lower_ids=tuple(
            sorted(directional[SetEnrichmentDirection.TARGET_LOWER] - tested_mapped)
        ),
    )
    return universe, reaction_map


def run_comparison_set_enrichment(
    comparison: GroupComparisonArtifact,
    subsystem_annotations: Iterable[SetEnrichmentAnnotation],
    feature_type: SetEnrichmentFeatureType,
    config: SetEnrichmentConfig = SetEnrichmentConfig(),
) -> SetEnrichmentArtifact:
    """Build, calculate, and package paired ORA for one comparison family."""
    annotations = tuple(subsystem_annotations)
    universe, reaction_map = build_comparison_set_enrichment_universe(
        comparison,
        annotations,
        feature_type,
        config,
    )
    result = run_set_enrichment(universe, config)
    annotation_manifest = _annotation_manifest(annotations)
    annotation_fingerprint = fingerprint_json(annotation_manifest)
    family_identity = _comparison_family_identity(
        comparison, feature_type, config.direction_metric
    )
    source_fingerprints = dict(comparison.metadata.source.source_fingerprints)
    source_fingerprints["subsystem_annotations"] = annotation_fingerprint
    source_fingerprints[f"comparison_{feature_type.value}_results"] = str(
        family_identity["comparison_feature_fingerprint"]
    )
    # Relevant family thresholds are verified transactionally from scientific
    # identity. Excluding the workspace's shared threshold revision lets a gene
    # artifact survive a reaction-only threshold change, and vice versa.
    source = ArtifactProvenance(
        workspace_id=comparison.metadata.source.workspace_id,
        revisions=comparison.metadata.source.revisions,
        dependencies=comparison.metadata.source.dependencies
        - frozenset({RevisionKind.THRESHOLDS}),
        source_fingerprints=source_fingerprints,
    )
    direction_fingerprint = reaction_map.fingerprint if reaction_map else None
    metadata = AnalysisArtifactMetadata(
        kind=ArtifactKind.SET_ENRICHMENT,
        source=source,
        method="paired_one_sided_hypergeometric_over_representation",
        configuration={
            "feature_type": feature_type.value,
            "direction_metric": config.direction_metric,
            "correction": config.correction,
            "minimum_measured_set_size": config.minimum_measured_set_size,
            "adjusted_pvalue_threshold": config.adjusted_pvalue_threshold,
            "reaction_direction_criterion": (
                config.reaction_direction_criterion.value
                if config.reaction_direction_criterion is not None
                else None
            ),
            "reaction_direction_fingerprint": direction_fingerprint,
            "universe_rule": (
                "unique selected Human1-mapped base reactions"
                if feature_type is SetEnrichmentFeatureType.REACTION
                else "unique tested genes mapped through Human1 reaction GPRs"
            ),
        },
        scientific_identity={
            "feature_type": feature_type.value,
            "membership_fingerprint": comparison.membership.fingerprint,
            "target_base_group_ids": comparison.membership.target_base_group_ids,
            "baseline_base_group_ids": comparison.membership.baseline_base_group_ids,
            "target_samples": comparison.membership.target_samples,
            "baseline_samples": comparison.membership.baseline_samples,
            **family_identity,
            "subsystem_annotation_fingerprint": annotation_fingerprint,
            "tested_feature_ids": sorted(universe.tested_feature_ids),
            "target_higher_query_ids": sorted(universe.target_higher_query_ids),
            "target_lower_query_ids": sorted(universe.target_lower_query_ids),
            "reaction_direction_fingerprint": direction_fingerprint,
        },
        dimensions=(
            ArtifactDimension("eligible_subsystems", len(result.result_frame()) // 2),
            ArtifactDimension("correction_hypotheses", len(result.result_frame())),
            ArtifactDimension(
                "target_higher_query_features",
                len(universe.target_higher_query_ids),
            ),
            ArtifactDimension(
                "target_lower_query_features", len(universe.target_lower_query_ids)
            ),
            ArtifactDimension("tested_features", len(universe.tested_feature_ids)),
        ),
        annotations={
            "subsystems_overlap": True,
            "topology_aware": False,
            "subsystem_source": annotation_manifest,
            "mapping_rule": (
                "Human1 subsystem genes derived from reaction GPR membership"
                if feature_type is SetEnrichmentFeatureType.GENE
                else "Human1 base-reaction subsystem membership"
            ),
            "comparison_threshold_identity": family_identity[
                "comparison_threshold_identity"
            ],
            "comparison_test": comparison.test,
            "comparison_correction": comparison.correction,
            "reaction_direction_audit": _reaction_direction_audit(
                reaction_map, universe.tested_feature_ids
            ),
        },
        warnings=result.warnings,
    )
    return SetEnrichmentArtifact(
        metadata=metadata,
        membership=comparison.membership,
        config=config,
        universe=universe,
        results=result.results,
        warnings=result.warnings,
        explanation=result.explanation,
        reaction_direction_map=reaction_map,
    )


def run_gene_set_enrichment(
    comparison: GroupComparisonArtifact,
    subsystem_annotations: Iterable[SetEnrichmentAnnotation],
    config: SetEnrichmentConfig = SetEnrichmentConfig(),
) -> SetEnrichmentArtifact:
    """Run paired gene-set over-representation analysis over mapped Human1 genes."""
    return run_comparison_set_enrichment(
        comparison,
        subsystem_annotations,
        SetEnrichmentFeatureType.GENE,
        config,
    )


def run_paired_reaction_set_enrichment(
    comparison: GroupComparisonArtifact,
    subsystem_annotations: Iterable[SetEnrichmentAnnotation],
    config: SetEnrichmentConfig = SetEnrichmentConfig(
        reaction_direction_criterion=DEFAULT_REACTION_DIRECTION_CRITERION
    ),
) -> SetEnrichmentArtifact:
    """Run paired reaction-set over-representation over selected base reactions."""
    return run_comparison_set_enrichment(
        comparison,
        subsystem_annotations,
        SetEnrichmentFeatureType.REACTION,
        config,
    )


def _comparison_family_identity(
    comparison: GroupComparisonArtifact,
    feature_type: SetEnrichmentFeatureType,
    direction_metric: str,
) -> dict[str, object]:
    family: Literal["reaction", "gene"] = (
        "reaction" if feature_type is SetEnrichmentFeatureType.REACTION else "gene"
    )
    results = comparison.results_frame(family)
    thresholds = (
        comparison.reaction_thresholds
        if feature_type is SetEnrichmentFeatureType.REACTION
        else comparison.gene_thresholds
    )
    if thresholds is None:
        raise ValueError(f"This comparison has no {family} thresholds")
    significant = (
        comparison.significant_reactions
        if feature_type is SetEnrichmentFeatureType.REACTION
        else comparison.significant_genes
    )
    threshold_identity = {
        "feature_type": family,
        "thresholds": dict(thresholds.as_mapping()),
        "significant_feature_ids": list(significant),
    }
    return {
        "comparison_feature_fingerprint": fingerprint_frame(results),
        "comparison_threshold_fingerprint": fingerprint_json(threshold_identity),
        "comparison_threshold_identity": threshold_identity,
        "direction_metric": direction_metric,
    }


def _annotation_manifest(
    annotations: Iterable[SetEnrichmentAnnotation],
) -> list[dict[str, object]]:
    return [
        {
            "subsystem_id": item.subsystem_id,
            "subsystem_name": item.subsystem_name,
            "category": item.category,
            "feature_ids": list(item.feature_ids),
        }
        for item in sorted(annotations, key=lambda value: value.subsystem_id)
    ]


def _metric_direction(value: Any) -> SetEnrichmentDirection | None:
    if pd.isna(value):
        return None
    numeric = float(value)
    if numeric > 0:
        return SetEnrichmentDirection.TARGET_HIGHER
    if numeric < 0:
        return SetEnrichmentDirection.TARGET_LOWER
    return None


def _reaction_direction_audit(
    direction_map: ReactionDirectionMap | None,
    tested_feature_ids: frozenset[str],
) -> list[dict[str, object]]:
    if direction_map is None:
        return []
    return [
        {
            "base_reaction_id": entry.base_reaction_id,
            "source_column": entry.source_column,
            "selected_direction": entry.selected_direction.value,
            "target_mean_activity": entry.target_mean_activity,
            "baseline_mean_activity": entry.baseline_mean_activity,
            "criterion": entry.criterion.value,
            "reason": entry.reason,
        }
        for entry in direction_map.entries
        if entry.base_reaction_id in tested_feature_ids
    ]


def _warnings(
    universe: SetEnrichmentUniverse, config: SetEnrichmentConfig
) -> list[str]:
    warnings: list[str] = []
    feature_label = universe.feature_type.value
    if universe.unmapped_target_higher_ids:
        warnings.append(
            f"{len(universe.unmapped_target_higher_ids)} Target-higher "
            f"{feature_label} feature(s) were not in the tested/mapped universe"
        )
    if universe.unmapped_target_lower_ids:
        warnings.append(
            f"{len(universe.unmapped_target_lower_ids)} Target-lower "
            f"{feature_label} feature(s) were not in the tested/mapped universe"
        )
    if not universe.target_higher_query_ids:
        warnings.append(f"The Target-higher {feature_label} query is empty")
    if not universe.target_lower_query_ids:
        warnings.append(f"The Target-lower {feature_label} query is empty")
    excluded = sum(
        len(feature_ids) < config.minimum_measured_set_size
        for feature_ids in universe.subsystem_feature_ids.values()
    )
    if excluded:
        warnings.append(
            f"{excluded} subsystem(s) had fewer than "
            f"{config.minimum_measured_set_size} measured {feature_label} features "
            "and were excluded"
        )
    return warnings


def _signed_visual(
    enrichment_ratio: float | None, direction: SetEnrichmentDirection
) -> float:
    if enrichment_ratio is None or enrichment_ratio <= 1.0:
        return 0.0
    magnitude = math.log2(enrichment_ratio)
    return (
        magnitude if direction is SetEnrichmentDirection.TARGET_HIGHER else -magnitude
    )


def _feature_ids(values: Iterable[str], *, field_name: str) -> set[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a collection of feature IDs")
    normalized = {str(value) for value in values}
    if any(not value for value in normalized):
        raise ValueError(f"{field_name} contains an empty feature ID")
    return normalized


__all__ = [
    "SET_ENRICHMENT_ARTIFACT_RESULT_COLUMNS",
    "SET_ENRICHMENT_RESULT_COLUMNS",
    "SetEnrichmentAnnotation",
    "SetEnrichmentArtifact",
    "SetEnrichmentConfig",
    "SetEnrichmentDirection",
    "SetEnrichmentFeatureType",
    "SetEnrichmentResult",
    "SetEnrichmentUniverse",
    "build_comparison_set_enrichment_universe",
    "build_gene_set_annotations",
    "build_set_enrichment_annotations",
    "run_comparison_set_enrichment",
    "run_gene_set_enrichment",
    "run_paired_reaction_set_enrichment",
    "run_set_enrichment",
]

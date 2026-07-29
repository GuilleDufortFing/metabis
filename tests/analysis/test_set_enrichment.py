"""Hand-computed contracts for the shared paired-direction ORA kernel."""

import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

from metabis import AnalysisWorkspace, GroupComparisonSpec, MetabolicModel
from metabis.analysis.artifacts import (
    AnalysisArtifactMetadata,
    ArtifactDimension,
    ArtifactKind,
    ArtifactProvenance,
    PlainTable,
)
from metabis.analysis.comparison_artifact import (
    ComparisonThresholds,
    GroupComparisonArtifact,
    MetricSummary,
)
from metabis.analysis.comparison_membership import ComparisonMembership
from metabis.analysis.reaction_direction import ReactionDirectionCriterion
from metabis.analysis.set_enrichment import (
    SET_ENRICHMENT_ARTIFACT_RESULT_COLUMNS,
    SET_ENRICHMENT_RESULT_COLUMNS,
    SetEnrichmentAnnotation,
    SetEnrichmentConfig,
    SetEnrichmentDirection,
    SetEnrichmentFeatureType,
    SetEnrichmentUniverse,
    build_comparison_set_enrichment_universe,
    build_gene_set_annotations,
    run_gene_set_enrichment,
    run_paired_reaction_set_enrichment,
    run_set_enrichment,
)
from metabis.analysis_workspace import RevisionKind, WorkspaceRevisions


def _universe(feature_type=SetEnrichmentFeatureType.REACTION):
    prefix = "R" if feature_type is SetEnrichmentFeatureType.REACTION else "G"
    return SetEnrichmentUniverse(
        feature_type=feature_type,
        tested_feature_ids=frozenset(f"{prefix}{index}" for index in range(1, 6)),
        target_higher_query_ids=frozenset({f"{prefix}1", f"{prefix}2"}),
        target_lower_query_ids=frozenset({f"{prefix}4"}),
        subsystem_annotations=(
            SetEnrichmentAnnotation(
                "s2",
                "Subsystem two",
                (f"{prefix}3", f"{prefix}4", f"{prefix}5", "UNTESTED"),
                "other",
            ),
            SetEnrichmentAnnotation(
                "s1",
                "Subsystem one",
                (f"{prefix}1", f"{prefix}2", f"{prefix}3"),
                "core",
            ),
        ),
    )


@pytest.mark.parametrize(
    "feature_type",
    [SetEnrichmentFeatureType.REACTION, SetEnrichmentFeatureType.GENE],
)
def test_one_kernel_produces_hand_computable_reaction_and_gene_ora(feature_type):
    result = run_set_enrichment(
        _universe(feature_type),
        SetEnrichmentConfig(minimum_measured_set_size=2),
    )
    frame = result.result_frame()
    indexed = frame.set_index(["subsystem_id", "direction"])
    prefix = "R" if feature_type is SetEnrichmentFeatureType.REACTION else "G"

    target = indexed.loc[("s1", SetEnrichmentDirection.TARGET_HIGHER.value)]
    baseline = indexed.loc[("s2", SetEnrichmentDirection.TARGET_LOWER.value)]
    assert (target.universe_size, target.subsystem_size) == (5, 3)
    assert (target.query_size, target.overlap_size) == (2, 2)
    assert target.pval == pytest.approx(hypergeom.sf(1, 5, 3, 2))
    assert target.enrichment_ratio == pytest.approx((2 / 2) / (3 / 5))
    assert target.signed_log2_enrichment_ratio == pytest.approx(math.log2(5 / 3))
    assert json.loads(target.overlap_feature_ids) == [f"{prefix}1", f"{prefix}2"]

    assert (baseline.query_size, baseline.overlap_size) == (1, 1)
    assert baseline.pval == pytest.approx(hypergeom.sf(0, 5, 3, 1))
    assert baseline.enrichment_ratio == pytest.approx((1 / 1) / (3 / 5))
    assert baseline.signed_log2_enrichment_ratio == pytest.approx(-math.log2(5 / 3))
    assert set(frame.feature_type) == {feature_type.value}


@pytest.mark.parametrize("correction", ["fdr_bh", "holm"])
def test_both_direction_families_are_corrected_jointly(correction):
    result = run_set_enrichment(
        _universe(),
        SetEnrichmentConfig(correction=correction, minimum_measured_set_size=2),
    )
    frame = result.result_frame()

    assert len(frame) == 4
    assert set(frame.correction_family_size) == {4}
    expected = multipletests(frame.pval.to_numpy(), method=correction)[1]
    np.testing.assert_allclose(frame.adjusted_pval, expected)


def test_signed_visual_has_a_true_zero_for_no_over_representation():
    frame = run_set_enrichment(
        _universe(), SetEnrichmentConfig(minimum_measured_set_size=2)
    ).result_frame()
    not_over_represented = frame[frame.enrichment_ratio <= 1.0]

    assert not_over_represented.signed_log2_enrichment_ratio.tolist() == [0.0, 0.0]


def test_empty_direction_query_retains_paired_rows_and_stable_schema():
    source = _universe(SetEnrichmentFeatureType.GENE)
    universe = SetEnrichmentUniverse(
        feature_type=source.feature_type,
        tested_feature_ids=source.tested_feature_ids,
        target_higher_query_ids=frozenset(),
        target_lower_query_ids=source.target_lower_query_ids,
        subsystem_annotations=source.subsystem_annotations,
    )
    result = run_set_enrichment(
        universe, SetEnrichmentConfig(minimum_measured_set_size=2)
    )
    frame = result.result_frame()
    empty_rows = frame[frame.direction == SetEnrichmentDirection.TARGET_HIGHER.value]

    assert tuple(frame.columns) == SET_ENRICHMENT_RESULT_COLUMNS
    assert len(empty_rows) == 2
    assert empty_rows.query_size.tolist() == [0, 0]
    assert empty_rows.enrichment_ratio.isna().all()
    assert empty_rows.signed_log2_enrichment_ratio.tolist() == [0.0, 0.0]
    assert empty_rows.pval.tolist() == [1.0, 1.0]
    assert any(
        "Target-higher gene query is empty" in warning for warning in result.warnings
    )


def test_queries_must_be_disjoint_subsets_of_the_tested_universe():
    annotation = SetEnrichmentAnnotation("s", "Set", ("G1", "G2"))
    with pytest.raises(ValueError, match="subset of the universe"):
        SetEnrichmentUniverse(
            feature_type=SetEnrichmentFeatureType.GENE,
            tested_feature_ids=frozenset({"G1"}),
            target_higher_query_ids=frozenset({"G2"}),
            target_lower_query_ids=frozenset(),
            subsystem_annotations=(annotation,),
        )
    with pytest.raises(ValueError, match="must be disjoint"):
        SetEnrichmentUniverse(
            feature_type=SetEnrichmentFeatureType.GENE,
            tested_feature_ids=frozenset({"G1", "G2"}),
            target_higher_query_ids=frozenset({"G1"}),
            target_lower_query_ids=frozenset({"G1"}),
            subsystem_annotations=(annotation,),
        )


def test_no_eligible_subsystems_returns_stable_empty_schema():
    result = run_set_enrichment(
        _universe(), SetEnrichmentConfig(minimum_measured_set_size=6)
    )

    assert result.result_frame().empty
    assert tuple(result.result_frame().columns) == SET_ENRICHMENT_RESULT_COLUMNS
    assert "jointly corrected 0 hypotheses" in result.explanation
    assert any("2 subsystem(s)" in warning for warning in result.warnings)


def _comparison_artifact() -> GroupComparisonArtifact:
    membership = ComparisonMembership.resolve(
        GroupComparisonSpec.pairwise(("target",), ("baseline",)),
        group_samples={"target": ("t1", "t2"), "baseline": ("b1", "b2")},
        aligned_sample_ids=("t1", "t2", "b1", "b2"),
        labels={"target": "Target", "baseline": "Baseline"},
    )
    reaction_results = pd.DataFrame(
        {
            "cohens_d": [2.0, -3.0, -1.0],
            "log2_fc": [1.0, -2.0, -0.5],
            "mean_difference": [4.0, -5.0, -1.0],
            "group_1_mean": [8.0, 2.0, 1.0],
            "group_2_mean": [1.0, 9.0, 2.0],
            "adjusted_pval": [0.01, 0.01, 0.02],
            "is_significant": [True, True, True],
        },
        index=["R1", "R1_back", "R2"],
    )
    gene_results = pd.DataFrame(
        {
            "cohens_d": [2.0, -1.5, 0.0, 3.0],
            "log2_fc": [1.0, -0.8, 0.0, 2.0],
            "mean_difference": [4.0, -2.0, 0.0, 6.0],
            "group_1_mean": [8.0, 2.0, 4.0, 9.0],
            "group_2_mean": [4.0, 4.0, 4.0, 3.0],
            "adjusted_pval": [0.01, 0.02, 1.0, 0.01],
            "is_significant": [True, True, False, True],
        },
        index=["G1", "G2", "G3", "G_UNMAPPED"],
    )
    metadata = AnalysisArtifactMetadata(
        kind=ArtifactKind.GROUP_COMPARISON,
        source=ArtifactProvenance(
            workspace_id="workspace",
            revisions=WorkspaceRevisions(dataset=2, groups=3, thresholds=4),
            dependencies=frozenset(
                {RevisionKind.DATASET, RevisionKind.GROUPS, RevisionKind.THRESHOLDS}
            ),
            source_fingerprints={"dataset": "sha256:dataset"},
        ),
        method="ttest",
        configuration={"correction": "fdr_bh", "keep_reversed": False},
        scientific_identity={
            "membership_fingerprint": membership.fingerprint,
            "mode": membership.mode.value,
            "target_base_group_ids": membership.target_base_group_ids,
            "baseline_base_group_ids": membership.baseline_base_group_ids,
            "target_samples": membership.target_samples,
            "baseline_samples": membership.baseline_samples,
            "universe": membership.universe,
            "excluded_base_group_ids": membership.excluded_base_group_ids,
        },
        dimensions=(
            ArtifactDimension("samples", 4),
            ArtifactDimension("target_samples", 2),
            ArtifactDimension("baseline_samples", 2),
            ArtifactDimension("reactions", 3),
            ArtifactDimension("genes", 4),
        ),
    )
    thresholds = ComparisonThresholds(0.05, 0.5, None, None)
    return GroupComparisonArtifact(
        metadata=metadata,
        membership=membership,
        test="ttest",
        correction="fdr_bh",
        reaction_results=PlainTable.capture(reaction_results),
        gene_results=PlainTable.capture(gene_results),
        reaction_metrics=(MetricSummary("cohens_d", 1.0, 0.0),),
        gene_metrics=(MetricSummary("cohens_d", 1.0, 0.0),),
        significant_reactions=("R1", "R1_back", "R2"),
        significant_genes=("G1", "G2", "G_UNMAPPED"),
        reaction_thresholds=thresholds,
        gene_thresholds=thresholds,
    )


def _gene_annotations() -> tuple[SetEnrichmentAnnotation, ...]:
    return (
        SetEnrichmentAnnotation("s1", "Set one", ("G1", "G3"), "core"),
        SetEnrichmentAnnotation("s2", "Set two", ("G2", "G3"), "other"),
    )


def test_gene_adapter_builds_unique_mapped_paired_queries_and_portable_artifact():
    comparison = _comparison_artifact()
    config = SetEnrichmentConfig(minimum_measured_set_size=1)
    universe, direction_map = build_comparison_set_enrichment_universe(
        comparison,
        _gene_annotations(),
        SetEnrichmentFeatureType.GENE,
        config,
    )

    assert direction_map is None
    assert universe.tested_feature_ids == frozenset({"G1", "G2", "G3"})
    assert universe.target_higher_query_ids == frozenset({"G1"})
    assert universe.target_lower_query_ids == frozenset({"G2"})
    assert universe.unmapped_target_higher_ids == ("G_UNMAPPED",)

    artifact = run_gene_set_enrichment(comparison, _gene_annotations(), config)
    frame = artifact.result_frame()
    assert artifact.metadata.kind is ArtifactKind.SET_ENRICHMENT
    assert artifact.feature_type is SetEnrichmentFeatureType.GENE
    assert tuple(frame.columns) == SET_ENRICHMENT_ARTIFACT_RESULT_COLUMNS
    assert len(frame) == 4
    assert set(frame.correction_family_size) == {4}
    assert set(frame.artifact_id) == {artifact.metadata.artifact_id}
    assert RevisionKind.THRESHOLDS not in artifact.metadata.source.dependencies
    assert artifact.metadata.annotations["mapping_rule"] == (
        "Human1 subsystem genes derived from reaction GPR membership"
    )
    assert artifact.metadata.annotations["subsystems_overlap"] is True
    assert artifact.metadata.annotations["topology_aware"] is False


def test_reaction_family_adapter_retains_selected_source_direction_audit():
    comparison = _comparison_artifact()
    annotations = (SetEnrichmentAnnotation("s", "Set", ("R1", "R2"), "core"),)
    artifact = run_paired_reaction_set_enrichment(
        comparison,
        annotations,
        SetEnrichmentConfig(
            minimum_measured_set_size=1,
            reaction_direction_criterion=(
                ReactionDirectionCriterion.DOMINANT_IN_TARGET
            ),
        ),
    )

    assert artifact.universe.tested_feature_ids == frozenset({"R1", "R2"})
    assert artifact.reaction_direction_map is not None
    assert artifact.reaction_direction_map.source_column_for("R1") == "R1"
    audit = artifact.metadata.annotations["reaction_direction_audit"]
    assert {row["base_reaction_id"] for row in audit} == {"R1", "R2"}
    assert (
        next(row for row in audit if row["base_reaction_id"] == "R1")["source_column"]
        == "R1"
    )


def _workspace_with_comparison(
    comparison: GroupComparisonArtifact,
) -> AnalysisWorkspace:
    workspace = AnalysisWorkspace()
    workspace._workspace_id = comparison.metadata.source.workspace_id
    revisions = comparison.metadata.source.revisions
    workspace.dataset_revision = revisions.dataset
    workspace.group_revision = revisions.groups
    workspace.threshold_revision = revisions.thresholds
    provenance = workspace.capture_analysis_snapshot(
        comparison.metadata.source.dependencies
    ).provenance
    workspace.install_group_comparison_artifact(provenance, comparison)
    return workspace


def test_workspace_installs_both_families_and_invalidates_only_the_changed_family():
    comparison = _comparison_artifact()
    gene = run_gene_set_enrichment(
        comparison,
        _gene_annotations(),
        SetEnrichmentConfig(minimum_measured_set_size=1),
    )
    reaction = run_paired_reaction_set_enrichment(
        comparison,
        (SetEnrichmentAnnotation("s", "Set", ("R1", "R2")),),
        SetEnrichmentConfig(
            minimum_measured_set_size=1,
            reaction_direction_criterion=(
                ReactionDirectionCriterion.DOMINANT_IN_TARGET
            ),
        ),
    )
    workspace = _workspace_with_comparison(comparison)
    # The reaction artifact above was built under Target dominance, so the
    # workspace has to be on that criterion for it to install as current; the
    # workspace default is Baseline dominance.
    workspace.set_reaction_direction_criterion(
        ReactionDirectionCriterion.DOMINANT_IN_TARGET
    )
    provenance = workspace.capture_analysis_snapshot(
        gene.metadata.source.dependencies
    ).provenance
    workspace.install_set_enrichment_artifact(provenance, gene)
    workspace.install_set_enrichment_artifact(provenance, reaction)

    workspace.mark_thresholds_changed(SetEnrichmentFeatureType.REACTION)

    assert workspace.get_set_enrichment_artifact("reaction") is None
    assert workspace.get_set_enrichment_artifact("gene") is gene
    workspace.set_reaction_direction_criterion(
        ReactionDirectionCriterion.DOMINANT_IN_BASELINE
    )
    assert workspace.get_set_enrichment_artifact("gene") is gene

    workspace.mark_thresholds_changed(SetEnrichmentFeatureType.GENE)
    assert workspace.get_set_enrichment_artifact("gene") is None


def test_portable_gene_identity_ignores_reaction_thresholds_but_rejects_gene_changes():
    comparison = _comparison_artifact()
    gene = run_gene_set_enrichment(
        comparison,
        _gene_annotations(),
        SetEnrichmentConfig(minimum_measured_set_size=1),
    )
    reaction_frame = comparison.results_frame("reaction")
    reaction_frame.loc[:, "is_significant"] = [False, False, True]
    reaction_changed = comparison.with_threshold_results(
        revisions=replace(comparison.metadata.source.revisions, thresholds=5),
        reaction_results=reaction_frame,
        reaction_thresholds=ComparisonThresholds(0.01, 2.0, None, None),
        significant_reactions=("R2",),
        gene_results=comparison.results_frame("gene"),
        gene_thresholds=comparison.gene_thresholds,
        significant_genes=comparison.significant_genes,
        keep_reversed=False,
    )

    gene.require_current_comparison(reaction_changed)

    gene_frame = reaction_changed.results_frame("gene")
    gene_frame.loc[:, "is_significant"] = [False, True, False, False]
    gene_changed = reaction_changed.with_threshold_results(
        revisions=replace(reaction_changed.metadata.source.revisions, thresholds=6),
        reaction_results=reaction_changed.results_frame("reaction"),
        reaction_thresholds=reaction_changed.reaction_thresholds,
        significant_reactions=reaction_changed.significant_reactions,
        gene_results=gene_frame,
        gene_thresholds=ComparisonThresholds(0.01, 1.0, None, None),
        significant_genes=("G2",),
        keep_reversed=False,
    )

    with pytest.raises(ValueError, match="comparison evidence|thresholds"):
        gene.require_current_comparison(gene_changed)


@pytest.mark.integration
def test_human1_gene_annotations_are_deterministic_gpr_derived_sets():
    model = MetabolicModel(use_cache=True)
    first = build_gene_set_annotations(model)
    second = build_gene_set_annotations(model)
    glycolysis = next(
        item for item in first if item.subsystem_id == "glycolysis_gluconeogenesis"
    )

    assert first == second
    assert first == tuple(sorted(first, key=lambda item: item.subsystem_id))
    assert glycolysis.subsystem_name == "Glycolysis / Gluconeogenesis"
    assert glycolysis.feature_ids == tuple(
        sorted(model.get_genes_for_subsystem(glycolysis.subsystem_id))
    )
    assert glycolysis.feature_ids

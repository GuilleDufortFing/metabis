import ast
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

from metabis import (
    AnalysisWorkspace,
    GroupComparison,
    GroupComparisonSpec,
    GroupDefinition,
    MetabolicModel,
)
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
    comparison_export_frame,
)
from metabis.analysis.comparison_membership import (
    ComparisonMembership,
)
from metabis.analysis.reaction_direction import ReactionDirectionCriterion
from metabis.analysis.reaction_set_enrichment import (
    ENRICHMENT_RESULT_COLUMNS,
    EnrichmentDirection,
    ReactionSetEnrichmentConfig,
    SubsystemAnnotation,
    build_reaction_set_universe,
    build_subsystem_annotations,
    run_reaction_set_enrichment,
)
from metabis.analysis_workspace import (
    RevisionKind,
    StaleAnalysisResultError,
    WorkspaceRevisions,
)


def _comparison(
    metrics,
    *,
    significant,
    thresholds=ComparisonThresholds(0.05, None, None, None),
):
    membership = ComparisonMembership.resolve(
        GroupComparisonSpec.pairwise(("case",), ("control",)),
        group_samples={"case": ("c1", "c2"), "control": ("k1", "k2")},
        aligned_sample_ids=("c1", "c2", "k1", "k2"),
        labels={"case": "Cases", "control": "Controls"},
    )
    frame = pd.DataFrame.from_dict(metrics, orient="index")
    for column in ("cohens_d", "log2_fc", "mean_difference"):
        if column not in frame:
            frame[column] = frame.get("cohens_d", 0.0)
    if "group_1_mean" not in frame:
        frame["group_1_mean"] = 0.0
    if "group_2_mean" not in frame:
        frame["group_2_mean"] = 0.0
    frame["is_significant"] = [index in significant for index in frame.index]
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
            ArtifactDimension("samples", len(membership.universe)),
            ArtifactDimension("target_samples", len(membership.target_samples)),
            ArtifactDimension("baseline_samples", len(membership.baseline_samples)),
            ArtifactDimension("reactions", len(frame)),
            ArtifactDimension("genes", 0),
        ),
    )
    return GroupComparisonArtifact(
        metadata=metadata,
        membership=membership,
        test="ttest",
        correction="fdr_bh",
        reaction_results=PlainTable.capture(frame),
        gene_results=None,
        reaction_metrics=(MetricSummary("cohens_d", 1.0, 0.0),),
        gene_metrics=(),
        significant_reactions=tuple(sorted(significant)),
        significant_genes=(),
        reaction_thresholds=thresholds,
        gene_thresholds=None,
    )


def _annotations():
    return (
        SubsystemAnnotation("s1", "Subsystem one", ("R1", "R2", "R3"), "core"),
        SubsystemAnnotation("s2", "Subsystem two", ("R3", "R4", "R5"), "core"),
        SubsystemAnnotation("small", "Small", ("R5",), "other"),
    )


def _five_reaction_comparison(significant=("R1", "R2")):
    return _comparison(
        {
            "R1": {"cohens_d": 2.0},
            "R2": {"cohens_d": 1.0},
            "R3": {"cohens_d": -0.5},
            "R4": {"cohens_d": -2.0},
            "R5": {"cohens_d": 0.25},
            "UNMAPPED": {"cohens_d": 3.0},
        },
        significant=significant,
    )


def test_hand_computable_hypergeometric_and_ratio():
    artifact = run_reaction_set_enrichment(
        _five_reaction_comparison(),
        _annotations(),
        ReactionSetEnrichmentConfig(minimum_measured_set_size=2),
    )
    row = artifact.result_frame().set_index("subsystem_id").loc["s1"]

    assert (row.universe_size, row.subsystem_size, row.query_size, row.overlap_size) == (
        5,
        3,
        2,
        2,
    )
    assert row.pval == pytest.approx(hypergeom.sf(1, 5, 3, 2))
    assert row.enrichment_ratio == pytest.approx((2 / 2) / (3 / 5))
    assert json.loads(row.overlap_reaction_ids) == ["R1", "R2"]


def test_back_directions_collapse_after_direction_filter_without_double_count():
    comparison = _comparison(
        {
            "R1": {"cohens_d": 2.0},
            "R1_back": {"cohens_d": -3.0},
            "R2": {"cohens_d": 1.0},
        },
        significant=("R1", "R1_back", "R2"),
    )
    annotations = (SubsystemAnnotation("s", "Set", ("R1", "R1_back", "R2")),)

    universe = build_reaction_set_universe(
        comparison,
        annotations,
        ReactionSetEnrichmentConfig(minimum_measured_set_size=1),
    )

    assert universe.tested_reaction_ids == frozenset({"R1", "R2"})
    assert universe.query_reaction_ids == frozenset({"R1", "R2"})


def test_only_the_selected_exact_direction_can_enter_the_query():
    comparison = _comparison(
        {
            "R1": {
                "cohens_d": 2.0,
                "group_1_mean": 8.0,
                "group_2_mean": 1.0,
            },
            "R1_back": {
                "cohens_d": -3.0,
                "group_1_mean": 2.0,
                "group_2_mean": 9.0,
            },
        },
        significant=("R1_back",),
    )
    annotations = (SubsystemAnnotation("s", "Set", ("R1",)),)

    target = build_reaction_set_universe(
        comparison,
        annotations,
        ReactionSetEnrichmentConfig(
            minimum_measured_set_size=1,
            reaction_direction_criterion=(
                ReactionDirectionCriterion.DOMINANT_IN_TARGET
            ),
        ),
    )
    baseline = build_reaction_set_universe(
        comparison,
        annotations,
        ReactionSetEnrichmentConfig(
            minimum_measured_set_size=1,
            reaction_direction_criterion=(
                ReactionDirectionCriterion.DOMINANT_IN_BASELINE
            ),
        ),
    )

    assert target.reaction_direction_map.source_column_for("R1") == "R1"
    assert target.query_reaction_ids == frozenset()
    assert baseline.reaction_direction_map.source_column_for("R1") == "R1_back"
    assert baseline.query_reaction_ids == frozenset({"R1"})


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        (EnrichmentDirection.EITHER, {"R1", "R3", "R4"}),
        (EnrichmentDirection.UP, {"R1"}),
        (EnrichmentDirection.DOWN, {"R3", "R4"}),
    ],
)
def test_direction_uses_explicit_metric_sign_after_significance(direction, expected):
    comparison = _five_reaction_comparison(significant=("R1", "R3", "R4"))

    universe = build_reaction_set_universe(
        comparison,
        _annotations(),
        ReactionSetEnrichmentConfig(
            direction=direction,
            direction_metric="cohens_d",
            minimum_measured_set_size=1,
        ),
    )

    assert universe.query_reaction_ids == frozenset(expected)


def test_overlapping_annotations_test_the_shared_reaction_in_each_set():
    artifact = run_reaction_set_enrichment(
        _five_reaction_comparison(significant=("R3",)),
        _annotations(),
        ReactionSetEnrichmentConfig(minimum_measured_set_size=2),
    )
    rows = artifact.result_frame().set_index("subsystem_id")

    assert rows.loc["s1", "overlap_size"] == 1
    assert rows.loc["s2", "overlap_size"] == 1
    assert artifact.metadata.annotations["subsystems_overlap"] is True
    assert artifact.metadata.annotations["topology_aware"] is False


@pytest.mark.parametrize("correction", ["fdr_bh", "holm"])
def test_multiple_testing_matches_statsmodels(correction):
    artifact = run_reaction_set_enrichment(
        _five_reaction_comparison(),
        _annotations(),
        ReactionSetEnrichmentConfig(
            correction=correction,
            minimum_measured_set_size=1,
        ),
    )
    frame = artifact.result_frame()

    expected = multipletests(frame.pval.to_numpy(), method=correction)[1]
    np.testing.assert_allclose(frame.adjusted_pval, expected)


def test_minimum_measured_size_excludes_only_ineligible_subsystems():
    artifact = run_reaction_set_enrichment(
        _five_reaction_comparison(),
        _annotations(),
        ReactionSetEnrichmentConfig(minimum_measured_set_size=2),
    )

    assert artifact.result_frame().subsystem_id.tolist() == ["s1", "s2"]
    assert any("fewer than 2" in warning for warning in artifact.warnings)


def test_empty_query_keeps_all_eligible_rows_full_schema_and_explanation():
    artifact = run_reaction_set_enrichment(
        _five_reaction_comparison(significant=()),
        _annotations(),
        ReactionSetEnrichmentConfig(minimum_measured_set_size=2),
    )
    frame = artifact.result_frame()

    assert tuple(frame.columns) == ENRICHMENT_RESULT_COLUMNS
    assert frame.subsystem_id.tolist() == ["s1", "s2"]
    assert set(frame.query_size) == {0}
    assert set(frame.overlap_size) == {0}
    assert set(frame.pval) == {1.0}
    assert not frame.is_significant.any()
    assert "query is empty" in " ".join(artifact.warnings)
    assert "query n=0" in artifact.explanation


def test_no_eligible_subsystems_returns_stable_empty_schema():
    artifact = run_reaction_set_enrichment(
        _five_reaction_comparison(),
        _annotations(),
        ReactionSetEnrichmentConfig(minimum_measured_set_size=6),
    )

    assert artifact.result_frame().empty
    assert tuple(artifact.result_frame().columns) == ENRICHMENT_RESULT_COLUMNS
    assert any("3 subsystem(s)" in warning for warning in artifact.warnings)


def test_unmapped_or_untested_significant_ids_are_audited_not_counted():
    comparison = _five_reaction_comparison(significant=("R1", "UNMAPPED", "ABSENT"))
    universe = build_reaction_set_universe(
        comparison,
        _annotations(),
        ReactionSetEnrichmentConfig(minimum_measured_set_size=1),
    )

    assert universe.query_reaction_ids == frozenset({"R1"})
    assert universe.unmapped_query_ids == ("ABSENT", "UNMAPPED")


def test_provenance_binds_membership_thresholds_and_subsystem_source_content():
    comparison = _five_reaction_comparison()
    baseline = run_reaction_set_enrichment(
        comparison,
        _annotations(),
        ReactionSetEnrichmentConfig(minimum_measured_set_size=2),
    )
    renamed = run_reaction_set_enrichment(
        comparison,
        (
            SubsystemAnnotation("s1", "Renamed set", ("R1", "R2", "R3"), "core"),
            *_annotations()[1:],
        ),
        ReactionSetEnrichmentConfig(minimum_measured_set_size=2),
    )
    threshold_changed = run_reaction_set_enrichment(
        _comparison(
            comparison.results_frame("reaction").to_dict(orient="index"),
            significant=("R1", "R2"),
            thresholds=ComparisonThresholds(0.01, None, None, None),
        ),
        _annotations(),
        ReactionSetEnrichmentConfig(minimum_measured_set_size=2),
    )

    identity = baseline.metadata.scientific_identity
    assert identity["membership_fingerprint"] == comparison.membership.fingerprint
    assert identity["comparison_artifact_id"] == comparison.metadata.artifact_id
    assert "subsystem_annotations" in baseline.metadata.source.source_fingerprints
    assert baseline.metadata.artifact_id != renamed.metadata.artifact_id
    assert baseline.metadata.artifact_id != threshold_changed.metadata.artifact_id


def test_enrichment_module_has_no_bokeh_or_server_dependency():
    path = Path(__file__).parents[2] / "src/metabis/analysis/reaction_set_enrichment.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )

    assert not any(name.startswith(("bokeh", "bokeh_server")) for name in imports)


def _manager_with_comparison(comparison):
    manager = AnalysisWorkspace()
    manager._workspace_id = comparison.metadata.source.workspace_id
    revisions = comparison.metadata.source.revisions
    manager.dataset_revision = revisions.dataset
    manager.metadata_revision = revisions.metadata
    manager.group_revision = revisions.groups
    manager.preprocessing_revision = revisions.preprocessing
    manager.threshold_revision = revisions.thresholds
    provenance = manager.capture_analysis_snapshot(
        comparison.metadata.source.dependencies
    ).provenance
    manager.install_group_comparison_artifact(
        provenance,
        comparison,
    )
    return manager, provenance


def test_data_manager_installs_and_gets_actual_enrichment_artifact():
    comparison = _five_reaction_comparison()
    artifact = run_reaction_set_enrichment(
        comparison,
        _annotations(),
        ReactionSetEnrichmentConfig(minimum_measured_set_size=2),
    )
    manager, provenance = _manager_with_comparison(comparison)

    returned = manager.install_reaction_set_enrichment_artifact(provenance, artifact)

    assert returned is artifact
    assert manager.get_reaction_set_enrichment_artifact() is artifact


def test_enrichment_install_is_transactional_for_a_foreign_comparison():
    comparison = _five_reaction_comparison()
    installed = run_reaction_set_enrichment(
        comparison,
        _annotations(),
        ReactionSetEnrichmentConfig(minimum_measured_set_size=2),
    )
    foreign_comparison = replace(
        comparison,
        metadata=replace(
            comparison.metadata,
            configuration={
                **dict(comparison.metadata.configuration),
                "scientific_fixture": "foreign",
            },
        ),
    )
    foreign = run_reaction_set_enrichment(
        foreign_comparison,
        _annotations(),
        ReactionSetEnrichmentConfig(minimum_measured_set_size=2),
    )
    manager, provenance = _manager_with_comparison(comparison)
    manager.install_reaction_set_enrichment_artifact(provenance, installed)

    with pytest.raises(ValueError, match="different comparison"):
        manager.install_reaction_set_enrichment_artifact(provenance, foreign)

    assert manager.get_reaction_set_enrichment_artifact() is installed


def test_stale_enrichment_install_is_rejected_without_state_change():
    comparison = _five_reaction_comparison()
    artifact = run_reaction_set_enrichment(
        comparison,
        _annotations(),
        ReactionSetEnrichmentConfig(minimum_measured_set_size=2),
    )
    manager, provenance = _manager_with_comparison(comparison)
    manager.mark_thresholds_changed()

    with pytest.raises(StaleAnalysisResultError, match="thresholds revision"):
        manager.install_reaction_set_enrichment_artifact(provenance, artifact)

    assert manager.get_reaction_set_enrichment_artifact() is None


@pytest.mark.parametrize(
    "transition",
    (
        "mark_dataset_changed",
        "mark_metadata_changed",
        "mark_groups_changed",
        "mark_preprocessing_changed",
        "mark_thresholds_changed",
    ),
)
def test_enrichment_is_cleared_for_every_declared_source_transition(transition):
    comparison = _five_reaction_comparison()
    artifact = run_reaction_set_enrichment(
        comparison,
        _annotations(),
        ReactionSetEnrichmentConfig(minimum_measured_set_size=2),
    )
    manager, provenance = _manager_with_comparison(comparison)
    manager.install_reaction_set_enrichment_artifact(provenance, artifact)

    getattr(manager, transition)()

    assert manager.get_reaction_set_enrichment_artifact() is None


class _FrameReactionData:
    """Frame-backed Reaction Data accepted by the public dataset writer."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def get_reaction_df(self) -> pd.DataFrame:
        return self.frame

    def get_sample_ids(self) -> list[str]:
        return list(map(str, self.frame.index))


@pytest.fixture(scope="module")
def human1_model() -> MetabolicModel:
    return MetabolicModel(use_cache=True)


@pytest.mark.integration
def test_public_enrichment_path_from_loaded_workspace_to_export(human1_model):
    """A real model and loaded workspace drive producer, install, and exports."""
    annotations = build_subsystem_annotations(human1_model)
    glycolysis = next(
        item
        for item in annotations
        if item.subsystem_id == "glycolysis_gluconeogenesis"
    )
    assert glycolysis.subsystem_name == "Glycolysis / Gluconeogenesis"
    assert glycolysis.category == "Central Carbon & Bioenergetics"
    assert glycolysis.reaction_ids == tuple(
        sorted(human1_model.get_reactions_for_subsystem(glycolysis.subsystem_id))
    )
    assert len(glycolysis.reaction_ids) == 44

    source_annotation = next(
        annotation for annotation in annotations if len(annotation.reaction_ids) >= 3
    )
    reaction_ids = source_annotation.reaction_ids[:3]
    samples = ("case_1", "case_2", "case_3", "control_1", "control_2", "control_3")
    reaction_data = _FrameReactionData(
        pd.DataFrame(
            {
                reaction_id: [
                    5.0 + offset,
                    5.2 + offset,
                    4.8 + offset,
                    1.0 + offset,
                    1.1 + offset,
                    0.9 + offset,
                ]
                for offset, reaction_id in enumerate(reaction_ids)
            },
            index=samples,
        )
    )
    workspace = AnalysisWorkspace(metabolic_model=human1_model)
    workspace.replace_reaction_data(reaction_data, metabolic_model=human1_model)
    workspace.group_manager.define_group(
        "case",
        samples[:3],
        definition=GroupDefinition("custom"),
        all_sample_ids=samples,
    )
    workspace.group_manager.define_group(
        "control",
        samples[3:],
        definition=GroupDefinition("custom"),
        all_sample_ids=samples,
    )

    membership = workspace.group_manager.resolve_comparison_membership(
        GroupComparisonSpec.pairwise(("case",), ("control",)),
        samples,
    )
    comparison = GroupComparison(membership=membership, test="ttest")
    comparison.run_test(
        workspace,
        test="ttest",
        correction="fdr_bh",
        register_distributions=False,
    )
    dependencies = frozenset(
        {
            RevisionKind.DATASET,
            RevisionKind.METADATA,
            RevisionKind.GROUPS,
            RevisionKind.PREPROCESSING,
            RevisionKind.THRESHOLDS,
        }
    )
    snapshot = workspace.capture_analysis_snapshot(dependencies)
    comparison_artifact = comparison.build_artifact(
        provenance=snapshot.provenance,
        source_fingerprints=workspace.source_fingerprints,
    )
    workspace.install_group_comparison_artifact(
        snapshot.provenance,
        comparison_artifact,
    )

    enrichment = run_reaction_set_enrichment(
        comparison_artifact,
        annotations,
        ReactionSetEnrichmentConfig(minimum_measured_set_size=1),
    )
    installed = workspace.install_reaction_set_enrichment_artifact(
        snapshot.provenance,
        enrichment,
    )
    comparison_export = comparison_export_frame(comparison_artifact, "reaction")
    enrichment_export = installed.result_frame()

    assert annotations == tuple(sorted(annotations, key=lambda item: item.subsystem_id))
    assert all(isinstance(item, SubsystemAnnotation) for item in annotations)
    assert source_annotation.subsystem_id in set(enrichment_export["subsystem_id"])
    assert set(comparison_export["feature_id"]) == set(reaction_ids)
    assert set(comparison_export["membership_fingerprint"]) == {
        membership.fingerprint
    }
    assert installed is workspace.get_reaction_set_enrichment_artifact()
    assert installed.metadata.source.source_fingerprints["dataset"] == (
        workspace.source_fingerprints["dataset"]
    )

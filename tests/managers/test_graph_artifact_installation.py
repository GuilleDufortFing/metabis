"""Workspace transactions for worker-produced graph analysis artifacts."""

import pandas as pd
import pytest

from metabis import (
    ActivityConfig,
    ActivityContext,
    AnalysisWorkspace,
    AnalysisWorkspaceMode,
    ExperimentAnalysis,
    FeatureActivityResult,
)
from metabis.analysis.activity_analysis import ActivityArtifact
from metabis.analysis.artifacts import (
    AnalysisArtifactMetadata,
    ArtifactDimension,
    ArtifactKind,
)
from metabis.analysis.subsystem_analysis import SubsystemDeltaArtifact
from metabis.analysis_workspace import RevisionKind, StaleAnalysisResultError
from metabis.managers.group_manager import GroupDefinition


class _ReactionData:
    def __init__(self):
        self.frame = pd.DataFrame(
            {"R1": [1.0, 3.0], "R2": [4.0, 2.0]},
            index=["A_1", "A_2"],
        )

    def get_reaction_df(self):
        return self.frame

    def get_sample_ids(self):
        return list(self.frame.index)


class _Comparison:
    group_id_1 = "A"
    group_id_2 = "B"


def _activity_manager():
    reaction_data = _ReactionData()
    manager = AnalysisWorkspace(reaction_data=reaction_data)
    manager.group_manager.define_group(
        "A",
        reaction_data.get_sample_ids(),
        definition=GroupDefinition("custom"),
        all_sample_ids=reaction_data.get_sample_ids(),
    )
    manager.mark_groups_changed()
    analysis = ExperimentAnalysis(
        manager,
        auto_run=False,
        register_distributions=False,
    )
    manager._set_experiment_analysis(analysis)
    return manager, analysis


def _activity_context(manager):
    config = ActivityConfig(group_id="A")
    scores = pd.Series({"R1": 1.0, "R2": -1.0})
    features = FeatureActivityResult(
        group_id="A",
        overlay="reaction",
        activity_score=scores,
        sample_count=2,
        reference_sample_count=2,
        config=config,
    )
    revisions = manager.get_workspace_revisions()
    return ActivityContext(
        config=config,
        reaction_features=features,
        reaction_subsystems=pd.DataFrame(),
        dataset_revision=revisions.dataset,
        group_revision=revisions.groups,
    )


def _activity_provenance(manager):
    return manager.capture_analysis_snapshot(
        {
            RevisionKind.DATASET,
            RevisionKind.GROUPS,
            RevisionKind.PREPROCESSING,
        }
    ).provenance


def _activity_artifact(manager, provenance):
    """Build the activity artifact the worker would, from a hand-built context.

    AM7 deleted the mutable activity context installed beside the artifact, so
    the workspace installs only this artifact; tests drive that seam.
    """
    context = _activity_context(manager)
    config = context.config
    metadata = AnalysisArtifactMetadata.from_workspace(
        kind=ArtifactKind.ACTIVITY,
        provenance=provenance,
        source_fingerprints=manager.source_fingerprints,
        method="single_group_normalized_activity",
        configuration={
            "summary": config.summary.value,
            "minimum_subsystem_coverage": config.minimum_subsystem_coverage,
        },
        scientific_identity={
            "group_id": "A",
            "selected_sample_ids": ["A_1", "A_2"],
            "reference_sample_ids": ["A_1", "A_2"],
            "available_families": ["reaction"],
        },
        dimensions=(
            ArtifactDimension("reaction_features", 2),
            ArtifactDimension("reaction_subsystems", 0),
            ArtifactDimension("gene_features", 0),
            ArtifactDimension("gene_subsystems", 0),
            ArtifactDimension("selected_samples", 2),
            ArtifactDimension("reference_samples", 2),
        ),
    )
    return ActivityArtifact.from_context(
        context,
        metadata=metadata,
        selected_sample_ids=("A_1", "A_2"),
        reference_sample_ids=("A_1", "A_2"),
    )


def _activity_state(manager, analysis):
    return (
        manager.activity_artifact,
        manager.data_provider.activity_artifact,
        manager.group_manager.get_active_group(),
        manager.mode,
        manager.color_manager.get_mode(),
        tuple(analysis._activity_artifacts.items()),
        tuple(manager.cache_manager._hover_cache.items()),
        frozenset(manager.color_manager._distributions),
        manager.visualization_revision,
    )


def test_matching_activity_install_synchronizes_all_workspace_owners():
    manager, analysis = _activity_manager()
    provenance = _activity_provenance(manager)
    artifact = _activity_artifact(manager, provenance)
    manager.cache_manager.cache_hover("old", "R1", "stale")
    visualization_revision = manager.visualization_revision

    returned = manager.install_activity_artifact(provenance, artifact)

    assert returned is artifact
    assert analysis.get_activity_artifact(artifact.config) is artifact
    assert manager.activity_artifact is artifact
    assert manager.data_provider.activity_artifact is artifact
    assert manager.get_active_group() == "A"
    assert manager.get_mode() is AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS
    assert manager.color_manager.get_mode() is AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS
    assert manager.cache_manager._hover_cache == {}
    assert (
            AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS,
            "reaction",
            artifact.metric_key_for("reaction"),
    ) in manager.color_manager._distributions
    assert manager.visualization_revision == visualization_revision


def test_stale_activity_install_changes_nothing():
    manager, analysis = _activity_manager()
    provenance = _activity_provenance(manager)
    artifact = _activity_artifact(manager, provenance)
    manager.mark_groups_changed()
    before = _activity_state(manager, analysis)

    with pytest.raises(StaleAnalysisResultError, match="groups revision"):
        manager.install_activity_artifact(provenance, artifact)

    assert _activity_state(manager, analysis) == before


def test_different_workspace_activity_install_changes_nothing():
    source, _source_analysis = _activity_manager()
    provenance = _activity_provenance(source)
    artifact = _activity_artifact(source, provenance)
    target, target_analysis = _activity_manager()
    before = _activity_state(target, target_analysis)

    with pytest.raises(StaleAnalysisResultError, match="different workspace"):
        target.install_activity_artifact(provenance, artifact)

    assert _activity_state(target, target_analysis) == before


def test_stale_dataset_activity_install_is_rejected_before_any_write():
    manager, analysis = _activity_manager()
    provenance = _activity_provenance(manager)
    artifact = _activity_artifact(manager, provenance)
    manager.mark_dataset_changed()
    before = _activity_state(manager, analysis)

    with pytest.raises(StaleAnalysisResultError, match="dataset revision"):
        manager.install_activity_artifact(provenance, artifact)

    assert _activity_state(manager, analysis) == before


def test_activity_presentation_failure_leaves_scientific_state_coherent(
    monkeypatch,
):
    manager, analysis = _activity_manager()
    provenance = _activity_provenance(manager)
    artifact = _activity_artifact(manager, provenance)
    visualization_revision = manager.visualization_revision
    monkeypatch.setattr(
        manager.color_manager,
        "register_distribution",
        lambda **_registration: (_ for _ in ()).throw(
            RuntimeError("color presentation failed")
        ),
    )

    with pytest.raises(RuntimeError, match="color presentation failed"):
        manager.install_activity_artifact(provenance, artifact)

    assert analysis.get_activity_artifact(artifact.config) is artifact
    assert manager.activity_artifact is artifact
    assert manager.data_provider.activity_artifact is artifact
    assert manager.get_active_group() == "A"
    assert manager.get_mode() is AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS
    assert manager.visualization_revision == visualization_revision


def _delta_artifact():
    return SubsystemDeltaArtifact(
        base_stats={("Glycolysis", False): {"is_in_model": True}},
        delta_stats={
            ("Glycolysis", False): {
                "delta_subsystem_score": 1.25,
                "delta_subsystem_meta": {"computed": True},
            }
        },
        config={"metric": "cohens_d"},
        distributions=(("subsystem", (("Glycolysis", 1.25),)),),
    )


def _comparison_manager():
    comparison = _Comparison()
    manager = AnalysisWorkspace()
    manager.group_comparison_artifact = comparison
    manager.data_provider.group_comparison_artifact = comparison
    return manager, comparison


def test_delta_artifact_install_refreshes_scientific_cache_mode_and_presentation():
    manager, comparison = _comparison_manager()
    artifact = _delta_artifact()
    provenance = manager.capture_analysis_snapshot(
        {RevisionKind.DATASET, RevisionKind.GROUPS, RevisionKind.THRESHOLDS}
    ).provenance
    manager.mode = AnalysisWorkspaceMode.NO_EXPERIMENT
    manager.cache_manager.cache_hover("old", "R1", "stale")
    visualization_revision = manager.visualization_revision

    returned = manager.install_subsystem_delta_artifact(provenance, artifact)

    assert returned is artifact
    assert manager.group_comparison_artifact is comparison
    assert manager.data_provider.group_comparison_artifact is comparison
    installed = manager.subsystem_analysis.get_delta_artifact()
    assert installed.base_stats == artifact.base_stats
    assert installed.delta_stats == artifact.delta_stats
    assert installed.config == artifact.config
    assert manager.get_mode() is AnalysisWorkspaceMode.GROUP_COMPARISON
    assert manager.cache_manager._hover_cache == {}
    assert (
        AnalysisWorkspaceMode.GROUP_COMPARISON,
        "subsystem",
        "cohens_d",
    ) in manager.color_manager._distributions
    assert manager.visualization_revision == visualization_revision


def test_delta_install_requires_a_current_comparison_before_any_write():
    manager, _comparison = _comparison_manager()
    artifact = _delta_artifact()
    provenance = manager.capture_analysis_snapshot().provenance
    manager.data_provider.group_comparison_artifact = object()
    before = manager.subsystem_analysis.get_delta_artifact()

    with pytest.raises(ValueError, match="provider mirror is not current"):
        manager.install_subsystem_delta_artifact(provenance, artifact)

    after = manager.subsystem_analysis.get_delta_artifact()
    assert after.base_stats == before.base_stats
    assert after.delta_stats == before.delta_stats
    assert after.config == before.config


def test_delta_presentation_failure_preserves_installed_scientific_artifact(
    monkeypatch,
):
    manager, _comparison = _comparison_manager()
    artifact = _delta_artifact()
    provenance = manager.capture_analysis_snapshot().provenance
    monkeypatch.setattr(
        manager.color_manager,
        "register_distribution",
        lambda **_registration: (_ for _ in ()).throw(
            RuntimeError("delta presentation failed")
        ),
    )

    with pytest.raises(RuntimeError, match="delta presentation failed"):
        manager.install_subsystem_delta_artifact(provenance, artifact)

    installed = manager.subsystem_analysis.get_delta_artifact()
    assert installed.base_stats == artifact.base_stats
    assert installed.delta_stats == artifact.delta_stats
    assert installed.config == artifact.config
    assert manager.get_mode() is AnalysisWorkspaceMode.GROUP_COMPARISON

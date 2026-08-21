"""Public rehydration contracts for precomputed experiment analysis."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from metabis import (
    ActivityConfig,
    ActivityContext,
    ExperimentAnalysis,
    FeatureActivityResult,
    ReductionKey,
    ReductionResult,
)
from metabis.analysis.dimensionality_reduction import ReductionArtifact
from metabis.analysis.experiment_analysis import ExperimentSummarySnapshot
from metabis.analysis_workspace import (
    RevisionKind,
    WorkspaceProvenance,
    WorkspaceRevisions,
)


class _RecordingColorManager:
    def __init__(self):
        self.registrations = []

    def register_distribution(self, **registration):
        self.registrations.append(registration)


class _AnalysisWorkspace:
    def __init__(self):
        self.color_manager = _RecordingColorManager()

    def get_group_ids(self):
        return ["A"]


def _reduction(method="PCA", data_type="reactions"):
    key = ReductionKey(
        method=method,
        data_type=data_type,
        n_variable_features=2,
        n_components=2,
        random_state=42,
        preprocessing_revision="1",
        sample_revision="1",
    )
    metadata = {}
    if method == "PCA":
        metadata = {
            "explained_variance_ratio": np.array([0.75, 0.25]),
            "components": np.array([[0.6, -0.8], [0.8, 0.6]]),
        }
    return ReductionResult(
        key=key,
        transformed_data=np.array([[1.0, 2.0]]),
        sample_ids=("sample-1",),
        feature_names=("feature-1", "feature-2"),
        metadata=metadata,
    )


def _reduction_artifact(method="PCA", data_type="reactions") -> ReductionArtifact:
    provenance = WorkspaceProvenance(
        workspace_id="workspace",
        revisions=WorkspaceRevisions(dataset=1, preprocessing=1),
        dependencies=frozenset({RevisionKind.DATASET, RevisionKind.PREPROCESSING}),
        reaction_frame=None,
        expression_matrix=None,
        group_memberships=(),
    )
    return _reduction(method, data_type).build_artifact(
        provenance=provenance,
        source_fingerprints={"dataset": "sha256:frame"},
    )


def _summaries():
    return {
        "reaction_averages": pd.DataFrame({"R1": [1.0]}, index=["A"]),
        "reaction_directions": pd.DataFrame({"R1": ["forward"]}, index=["A"]),
        "subsystem_averages": pd.DataFrame({"S1": [2.0]}, index=["A"]),
        "gene_averages": pd.DataFrame({"G1": [3.0]}, index=["A"]),
    }


def _activity_context():
    config = ActivityConfig(group_id="A")
    scores = pd.Series({"R1": 1.5})
    features = FeatureActivityResult(
        group_id="A",
        overlay="reaction",
        activity_score=scores,
        sample_count=2,
        reference_sample_count=4,
        config=config,
    )
    return ActivityContext(
        config=config,
        reaction_features=features,
        reaction_subsystems=pd.DataFrame(),
        dataset_revision=3,
        group_revision=5,
    )


def test_factory_rehydrates_summaries_and_reductions_without_presentation_writes():
    manager = _AnalysisWorkspace()
    summaries = _summaries()
    artifact = _reduction_artifact()

    analysis = ExperimentAnalysis.from_precomputed_results(
        manager,
        **summaries,
        reductions=(artifact,),
    )

    installed_summaries = analysis.get_summary_frames()
    expected_summaries = (
        summaries["reaction_averages"],
        summaries["reaction_directions"],
        summaries["subsystem_averages"],
        summaries["gene_averages"],
    )
    assert all(
        installed is expected
        for installed, expected in zip(installed_summaries, expected_summaries)
    )
    assert analysis.get_reduction_artifact("PCA", "reactions") is artifact
    assert analysis.get_reduction_result("PCA", "reactions").key == artifact.key
    assert manager.color_manager.registrations == []


def test_presentation_registration_is_explicit_and_uses_every_available_summary():
    manager = _AnalysisWorkspace()
    analysis = ExperimentAnalysis.from_precomputed_results(
        manager,
        **_summaries(),
    )

    analysis.register_presentation_distributions()

    assert {
        registration["element_type"]
        for registration in manager.color_manager.registrations
    } == {"reaction", "subsystem", "gene"}


@pytest.mark.parametrize("method", ("PCA", "T-SNE", "UMAP"))
def test_public_reduction_install_owns_the_artifact_and_typed_result(method):
    analysis = ExperimentAnalysis(_AnalysisWorkspace(), auto_run=False)
    artifact = _reduction_artifact(method=method)

    returned = analysis.install_reduction_artifact(artifact)

    assert returned is artifact
    assert analysis.get_reduction_artifact(method, "reactions") is artifact
    # The typed result cache keeps the coordinates by reference.
    result = analysis.get_reduction_result(method, "reactions")
    assert result.key == artifact.key
    assert np.shares_memory(result.transformed_data, artifact.coordinates)


def test_reduction_install_keeps_the_compute_cache_read_only_to_callers():
    analysis = ExperimentAnalysis(_AnalysisWorkspace(), auto_run=False)
    artifact = _reduction_artifact()
    analysis.install_reduction_artifact(artifact)

    with pytest.raises(TypeError):
        analysis.dim_reduction.results[artifact.key] = artifact.to_compute_result()


def test_summary_getter_rejects_an_incomplete_analysis():
    analysis = ExperimentAnalysis(_AnalysisWorkspace(), auto_run=False)

    with pytest.raises(ValueError, match="not complete"):
        analysis.get_summary_frames()


def test_complete_summary_snapshot_roundtrips_without_presentation_writes():
    summaries = _summaries()
    source = ExperimentAnalysis.from_precomputed_results(_AnalysisWorkspace(), **summaries)

    snapshot = source.snapshot_summaries()
    manager = _AnalysisWorkspace()
    restored = ExperimentAnalysis.from_summary_snapshot(manager, snapshot)

    assert snapshot.completed is True
    assert restored.snapshot_summaries().completed is True
    assert all(
        restored_frame is snapshot_frame
        for restored_frame, snapshot_frame in zip(
            restored.get_summary_frames(),
            (
                snapshot.reaction_averages,
                snapshot.reaction_directions,
                snapshot.subsystem_averages,
                snapshot.gene_averages,
            ),
        )
    )
    assert manager.color_manager.registrations == []


def test_incomplete_summary_snapshot_roundtrips_and_remains_incomplete():
    source = ExperimentAnalysis(_AnalysisWorkspace(), auto_run=False)

    snapshot = source.snapshot_summaries()
    restored = ExperimentAnalysis.from_summary_snapshot(_AnalysisWorkspace(), snapshot)

    assert snapshot == ExperimentSummarySnapshot(None, None, None, None, False)
    assert restored.snapshot_summaries() == snapshot
    with pytest.raises(ValueError, match="not complete"):
        restored.get_summary_frames()


def test_completed_summary_snapshot_rejects_missing_required_frames():
    with pytest.raises(ValueError, match="requires reaction averages"):
        ExperimentSummarySnapshot(None, None, None, None, True)


def test_summary_snapshot_is_frozen():
    snapshot = ExperimentAnalysis(_AnalysisWorkspace(), auto_run=False).snapshot_summaries()

    with pytest.raises(FrozenInstanceError):
        snapshot.completed = True


def test_activity_install_owns_cache_and_optional_presentation_registration():
    manager = _AnalysisWorkspace()
    analysis = ExperimentAnalysis(
        manager,
        auto_run=False,
        register_distributions=False,
    )
    context = _activity_context()

    returned = analysis.install_activity_context(
        context,
        register_presentation=False,
    )

    assert returned is context
    assert analysis._activity_cache[context.metric_key] is context
    assert manager.color_manager.registrations == []

    analysis.install_activity_context(context, register_presentation=True)

    assert len(manager.color_manager.registrations) == 1
    registration = manager.color_manager.registrations[0]
    assert registration["element_type"] == "reaction"
    assert registration["metric_name"] == context.config.metric_key_for("reaction")
    pd.testing.assert_series_equal(
        registration["data"],
        context.features.activity_score,
    )

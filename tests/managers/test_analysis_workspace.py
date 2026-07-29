"""Contracts for immutable Analysis Workspace snapshots and provenance."""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

import metabis
from metabis import AnalysisWorkspace, GroupDefinition
from metabis.analysis_workspace import (
    ALL_REVISION_KINDS,
    AnalysisSnapshot,
    RevisionKind,
    StableFrameReference,
    StaleAnalysisResultError,
    WorkspaceProvenance,
    WorkspaceRevisions,
)
from metabis.domain.expression_matrix import DenseExpressionMatrix


class _ReactionData:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def get_reaction_df(self) -> pd.DataFrame:
        return self.frame

    def get_sample_ids(self) -> list[str]:
        return list(map(str, self.frame.index))


class _ExpressionData:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def get_expression_df(self) -> pd.DataFrame:
        return self.frame

    def get_feature_ids(self):
        return [str(column) for column in self.frame.columns]

    def matrix(self) -> DenseExpressionMatrix:
        return DenseExpressionMatrix(self.frame)


def _manager() -> AnalysisWorkspace:
    reaction_frame = pd.DataFrame(
        {"R1": [1.0, 2.0], "R2": [3.0, 4.0]},
        index=["A_1", "B_1"],
    )
    expression_frame = pd.DataFrame(
        {"G1": [5.0, 6.0]},
        index=["A_1", "B_1"],
    )
    manager = AnalysisWorkspace(
        reaction_data=_ReactionData(reaction_frame),
        expression_data=_ExpressionData(expression_frame),
    )
    manager.group_manager.define_group(
        "A",
        ["A_1"],
        definition=GroupDefinition("custom"),
        all_sample_ids=list(reaction_frame.index),
    )
    manager.group_manager.define_group(
        "B",
        ["B_1"],
        definition=GroupDefinition("custom"),
        all_sample_ids=list(reaction_frame.index),
    )
    manager.mark_groups_changed()
    return manager


def _widen_group_a(manager: AnalysisWorkspace) -> None:
    """Drift group A's membership without bumping the workspace group revision."""
    manager.group_manager.define_group(
        "A",
        ["A_1", "B_1"],
        definition=GroupDefinition("custom"),
        all_sample_ids=["A_1", "B_1"],
        replace=True,
    )


def test_snapshot_keeps_matrix_references_without_copying_and_detaches_groups(
    monkeypatch,
):
    manager = _manager()

    with monkeypatch.context() as capture_patch:
        capture_patch.setattr(
            pd.DataFrame,
            "copy",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("snapshot capture copied an input matrix")
            ),
        )
        snapshot = manager.capture_analysis_snapshot()

    assert snapshot.reaction_frame.read() is manager.reaction_data.frame
    # Expression is captured through the storage-neutral matrix() seam; the dense
    # adapter still holds the frame by reference, so no copy was made.
    assert (
        snapshot.expression_matrix.read().to_dense_frame()
        is manager.expression_data.frame
    )
    assert snapshot.group_memberships == (
        ("A", ("A_1",)),
        ("B", ("B_1",)),
    )

    _widen_group_a(manager)

    assert snapshot.get_group_samples("A") == ("A_1",)


def test_snapshot_records_all_six_revisions_and_defaults_to_all_dependencies():
    manager = _manager()
    manager.mark_dataset_changed()
    manager.mark_metadata_changed()
    manager.mark_groups_changed()
    manager.mark_preprocessing_changed()
    manager.mark_thresholds_changed()
    manager.mark_visualization_changed()

    provenance = manager.capture_analysis_snapshot().provenance

    assert provenance.revisions == WorkspaceRevisions(
        dataset=1,
        metadata=1,
        groups=2,
        preprocessing=1,
        thresholds=1,
        visualization=1,
    )
    assert provenance.dependencies == ALL_REVISION_KINDS


@pytest.mark.parametrize("kind", list(RevisionKind))
def test_each_declared_revision_dependency_rejects_a_change(kind):
    manager = _manager()
    provenance = manager.capture_analysis_snapshot({kind}).provenance
    marker = {
        RevisionKind.DATASET: manager.mark_dataset_changed,
        RevisionKind.METADATA: manager.mark_metadata_changed,
        RevisionKind.GROUPS: manager.mark_groups_changed,
        RevisionKind.PREPROCESSING: manager.mark_preprocessing_changed,
        RevisionKind.THRESHOLDS: manager.mark_thresholds_changed,
        RevisionKind.VISUALIZATION: manager.mark_visualization_changed,
    }[kind]

    marker()

    with pytest.raises(StaleAnalysisResultError, match=kind.value):
        manager.require_current(provenance)


def test_undeclared_revision_change_does_not_invalidate_result():
    manager = _manager()
    provenance = manager.capture_analysis_snapshot(
        {RevisionKind.DATASET, RevisionKind.PREPROCESSING}
    ).provenance

    manager.mark_thresholds_changed()
    manager.mark_visualization_changed()

    manager.require_current(provenance)


def test_different_workspace_is_rejected_even_with_shared_inputs_and_revisions():
    manager = _manager()
    other = AnalysisWorkspace(
        reaction_data=manager.reaction_data,
        expression_data=manager.expression_data,
    )
    provenance = manager.capture_analysis_snapshot(set()).provenance

    with pytest.raises(StaleAnalysisResultError, match="different workspace"):
        other.require_current(provenance)


def test_replacing_a_frame_is_rejected_without_relying_on_a_revision_bump():
    manager = _manager()
    provenance = manager.capture_analysis_snapshot({RevisionKind.DATASET}).provenance
    manager.reaction_data.frame = manager.reaction_data.frame.copy()

    with pytest.raises(StaleAnalysisResultError, match="input frame changed"):
        manager.require_current(provenance)


def test_changing_a_captured_frame_shape_is_rejected():
    manager = _manager()
    snapshot = manager.capture_analysis_snapshot({RevisionKind.DATASET})
    manager.reaction_data.frame.loc["C_1"] = [7.0, 8.0]

    with pytest.raises(StaleAnalysisResultError, match="input frame changed"):
        snapshot.reaction_frame.read()
    with pytest.raises(StaleAnalysisResultError, match="input frame changed"):
        manager.require_current(snapshot.provenance)


def test_group_membership_is_validated_only_when_groups_are_a_dependency():
    manager = _manager()
    group_provenance = manager.capture_analysis_snapshot(
        {RevisionKind.GROUPS}
    ).provenance
    dataset_provenance = manager.capture_analysis_snapshot(
        {RevisionKind.DATASET}
    ).provenance
    _widen_group_a(manager)

    with pytest.raises(StaleAnalysisResultError, match="Group memberships changed"):
        manager.require_current(group_provenance)
    manager.require_current(dataset_provenance)


def test_workspace_contract_types_are_frozen():
    manager = _manager()
    snapshot = manager.capture_analysis_snapshot()

    values = (
        WorkspaceRevisions(),
        snapshot.reaction_frame,
        snapshot.provenance,
        snapshot,
    )
    assert isinstance(snapshot, AnalysisSnapshot)
    assert isinstance(snapshot.reaction_frame, StableFrameReference)
    assert isinstance(snapshot.provenance, WorkspaceProvenance)

    for value in values:
        with pytest.raises(FrozenInstanceError):
            value.unexpected = True

    with pytest.raises(AttributeError):
        manager.workspace_id = "replacement"


def test_workspace_types_do_not_expand_the_root_public_interface():
    assert not hasattr(metabis, "WorkspaceRevisions")
    assert not hasattr(metabis, "WorkspaceProvenance")
    assert not hasattr(metabis, "AnalysisSnapshot")

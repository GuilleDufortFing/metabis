"""Source fingerprints are computed once per input, never per run (ADR-0007, MR-10).

The first AM5 server slice opened every graph render by SHA-256ing the whole
reaction matrix, the whole expression matrix, and every model entity: about
1.5-2 s added to each render, recomputed identically each time, and invisible to
a deterministic suite that only counts tests. These tests make that regression
class visible. The load-time counts below are the budget; the run-time count is
zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metabis import AnalysisWorkspace
from metabis.analysis import source_identity
from metabis.analysis.source_identity import SourceIdentityCache, model_identity
from metabis.analysis_workspace import RevisionKind
from metabis.domain.h5ad_embedding_input import (
    H5adEmbeddingCoordinates,
    H5adEmbeddingInput,
)
from metabis.domain.sample_metadata import SampleMetadata


class _ReactionData:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def get_reaction_df(self) -> pd.DataFrame:
        return self.frame

    def get_sample_ids(self):
        return list(self.frame.index)


class _ExpressionData:
    def __init__(
        self, frame: pd.DataFrame, embedding_input: H5adEmbeddingInput | None = None
    ) -> None:
        self.frame = frame
        self.embedding_input = embedding_input

    def get_expression_df(self) -> pd.DataFrame:
        return self.frame

    def get_feature_ids(self):
        return [str(column) for column in self.frame.columns]

    def get_sample_ids(self):
        return list(self.frame.index)


class _Collection:
    def __init__(self, values) -> None:
        self._values = list(values)

    def get_all(self):
        return list(self._values)


class _Model:
    def __init__(self, reactions=("R1", "R2"), subsystems=("Glycolysis",)) -> None:
        self.reactions = _Collection(reactions)
        self.subsystems = _Collection(subsystems)


def _reaction_data(value: float = 1.0) -> _ReactionData:
    return _ReactionData(
        pd.DataFrame({"R1": [value, 2.0], "R2": [3.0, 4.0]}, index=["s1", "s2"])
    )


def _expression_data(value: float = 5.0) -> _ExpressionData:
    return _ExpressionData(
        pd.DataFrame({"G1": [value, 6.0], "G2": [7.0, 8.0]}, index=["s1", "s2"])
    )


def _embedding_input(first_value: float) -> H5adEmbeddingInput:
    return H5adEmbeddingInput(
        sample_ids=("s1", "s2"),
        feature_ids=("G1", "G2"),
        coordinates=(
            H5adEmbeddingCoordinates(
                "X_umap",
                "UMAP",
                np.array([[first_value, 0.0], [1.0, 1.0]]),
            ),
        ),
    )


def test_dataset_identity_includes_embedding_content_beside_expression_x():
    frame = _expression_data().frame
    first = AnalysisWorkspace(metabolic_model=_Model())
    first.replace_expression_data(
        _ExpressionData(frame.copy(), _embedding_input(first_value=0.0))
    )
    changed = AnalysisWorkspace(metabolic_model=_Model())
    changed.replace_expression_data(
        _ExpressionData(frame.copy(), _embedding_input(first_value=9.0))
    )

    assert first.source_fingerprints["expression"] == changed.source_fingerprints[
        "expression"
    ]
    assert first.source_fingerprints["embeddings"] != changed.source_fingerprints[
        "embeddings"
    ]
    assert first.source_fingerprints["dataset"] != changed.source_fingerprints[
        "dataset"
    ]


class _FrameFingerprintCounter:
    """Count every full-frame fingerprint, wherever it is triggered from."""

    def __init__(self, monkeypatch) -> None:
        self.calls = 0
        original = source_identity.fingerprint_frame

        def counting(frame):
            self.calls += 1
            return original(frame)

        monkeypatch.setattr(source_identity, "fingerprint_frame", counting)


def _loaded_workspace() -> AnalysisWorkspace:
    """A session workspace as the app builds one: constructed empty, then loaded."""
    manager = AnalysisWorkspace(metabolic_model=_Model())
    manager.replace_reaction_data(_reaction_data(), metabolic_model=_Model())
    manager.replace_expression_data(_expression_data())
    return manager


def test_loading_a_dataset_fingerprints_each_input_exactly_once(monkeypatch):
    counter = _FrameFingerprintCounter(monkeypatch)

    manager = _loaded_workspace()

    # One reaction frame, one expression frame. Not one per component per read.
    assert counter.calls == 2
    assert set(manager.source_fingerprints) == {
        "dataset",
        "model",
        "reactions",
        "expression",
    }


def test_constructing_a_workspace_over_existing_frames_hashes_nothing(monkeypatch):
    """Constructing a workspace is not loading data into one.

    The server builds a detached ``AnalysisWorkspace`` per graph job as a worker facade
    over frames the session already owns. If construction hashed its inputs, that
    facade would put a full-matrix hash on every single render -- which is exactly
    the regression that got the first AM5 server slice reverted.
    """
    counter = _FrameFingerprintCounter(monkeypatch)

    AnalysisWorkspace(
        metabolic_model=_Model(),
        reaction_data=_reaction_data(),
        expression_data=_expression_data(),
    )

    assert counter.calls == 0


def test_reading_identity_and_capturing_snapshots_never_fingerprints_again(monkeypatch):
    manager = _loaded_workspace()
    counter = _FrameFingerprintCounter(monkeypatch)

    for _ in range(25):
        manager.capture_analysis_snapshot([RevisionKind.DATASET])
        fingerprints = manager.source_fingerprints

    assert counter.calls == 0, (
        "An analysis run re-fingerprinted a full input matrix. Dataset identity "
        "does not change between runs; read it from the workspace."
    )
    assert fingerprints["dataset"].startswith("sha256:")


def test_no_analysis_run_may_fingerprint_a_full_matrix(monkeypatch):
    """The hard constraint, asserted the way a render would violate it."""
    manager = _loaded_workspace()

    def forbidden(frame):
        raise AssertionError(
            "A run fingerprinted a full matrix; see ADR-0007 'When source "
            "fingerprints may be computed'"
        )

    monkeypatch.setattr(source_identity, "fingerprint_frame", forbidden)

    snapshot = manager.capture_analysis_snapshot([RevisionKind.DATASET])
    manager.require_current(snapshot.provenance)

    assert manager.source_fingerprints["dataset"]


def test_replacing_one_input_re_identifies_only_that_input(monkeypatch):
    model = _Model()
    manager = AnalysisWorkspace(
        metabolic_model=model,
        reaction_data=_reaction_data(),
        expression_data=_expression_data(),
    )
    before = dict(manager.source_fingerprints)
    counter = _FrameFingerprintCounter(monkeypatch)

    manager.replace_expression_data(_expression_data(value=99.0))
    after = manager.source_fingerprints

    assert counter.calls == 1, "Replacing expression data re-hashed another input"
    assert after["expression"] != before["expression"]
    assert after["reactions"] == before["reactions"]
    assert after["model"] == before["model"]
    assert after["dataset"] != before["dataset"]


def test_reinstalling_the_same_input_object_keeps_its_fingerprint(monkeypatch):
    reaction_data = _reaction_data()
    model = _Model()
    manager = AnalysisWorkspace(metabolic_model=model, reaction_data=reaction_data)
    before = dict(manager.source_fingerprints)
    counter = _FrameFingerprintCounter(monkeypatch)

    manager.replace_reaction_data(reaction_data, metabolic_model=model)

    assert counter.calls == 0
    assert manager.source_fingerprints["dataset"] == before["dataset"]


def test_dataset_identity_reflects_content_not_object_identity():
    first = AnalysisWorkspace(metabolic_model=_Model(), reaction_data=_reaction_data())
    same_content = AnalysisWorkspace(metabolic_model=_Model(), reaction_data=_reaction_data())
    other_content = AnalysisWorkspace(
        metabolic_model=_Model(), reaction_data=_reaction_data(value=42.0)
    )

    assert (
        first.source_fingerprints["dataset"]
        == same_content.source_fingerprints["dataset"]
    )
    assert (
        first.source_fingerprints["dataset"]
        != other_content.source_fingerprints["dataset"]
    )


def test_dataset_identity_is_invariant_to_sparse_or_dense_storage():
    """Storage mode does not move identity (decision M-08).

    The AM6 sparse path will make a workflow hand the cache a sparse frame where
    it used to be dense. If the dataset fingerprint moved with storage, every
    installed artifact would go stale and every cache miss on a re-load of the
    same numbers -- the reverted-slice failure mode, at the provenance seam. So a
    sparse expression frame and its numerically identical dense twin must produce
    the same dataset fingerprint.
    """
    dense_frame = pd.DataFrame(
        {"G1": [0.0, 6.0], "G2": [7.0, 0.0]}, index=["s1", "s2"]
    )
    sparse_frame = dense_frame.astype(pd.SparseDtype("float64", 0.0))
    assert any(isinstance(dtype, pd.SparseDtype) for dtype in sparse_frame.dtypes)

    dense = AnalysisWorkspace(
        metabolic_model=_Model(), expression_data=_ExpressionData(dense_frame)
    )
    sparse = AnalysisWorkspace(
        metabolic_model=_Model(), expression_data=_ExpressionData(sparse_frame)
    )

    assert (
        sparse.source_fingerprints["dataset"] == dense.source_fingerprints["dataset"]
    )
    # And a genuine value change still diverges, so the invariance is not vacuous.
    changed_frame = dense_frame.copy()
    changed_frame.iloc[0, 0] = 1.0
    changed = AnalysisWorkspace(
        metabolic_model=_Model(), expression_data=_ExpressionData(changed_frame)
    )
    assert (
        changed.source_fingerprints["dataset"] != dense.source_fingerprints["dataset"]
    )


def test_an_empty_workspace_still_carries_a_canonical_dataset_fingerprint():
    manager = AnalysisWorkspace()

    fingerprints = manager.source_fingerprints

    assert fingerprints["dataset"].startswith("sha256:")
    assert set(fingerprints) == {"dataset"}


def test_resetting_the_dataset_drops_the_measured_input_identities():
    manager = AnalysisWorkspace(
        metabolic_model=_Model(),
        reaction_data=_reaction_data(),
        expression_data=_expression_data(),
    )
    loaded = manager.source_fingerprints["dataset"]

    manager.reset_dataset()

    assert set(manager.source_fingerprints) == {"dataset", "model"}
    assert manager.source_fingerprints["dataset"] != loaded


def test_sample_metadata_is_identified_but_is_not_part_of_dataset_identity():
    manager = AnalysisWorkspace(
        metabolic_model=_Model(),
        reaction_data=_reaction_data(),
    )
    dataset_before = manager.source_fingerprints["dataset"]

    manager.update_sample_metadata(
        SampleMetadata(pd.DataFrame({"tissue": ["gut", "liver"]}, index=["s1", "s2"]))
    )
    fingerprints = manager.source_fingerprints

    # Labels changed; the dataset did not stop being the same dataset.
    assert fingerprints["dataset"] == dataset_before
    assert fingerprints["metadata"].startswith("sha256:")


def test_metadata_identity_tracks_its_table_and_clears_with_it():
    manager = AnalysisWorkspace(metabolic_model=_Model(), reaction_data=_reaction_data())
    manager.update_sample_metadata(
        SampleMetadata(pd.DataFrame({"tissue": ["gut", "liver"]}, index=["s1", "s2"]))
    )
    first = manager.source_fingerprints["metadata"]

    manager.update_sample_metadata(
        SampleMetadata(pd.DataFrame({"tissue": ["gut", "blood"]}, index=["s1", "s2"]))
    )
    second = manager.source_fingerprints["metadata"]

    manager.clear_sample_metadata()

    assert first != second
    assert "metadata" not in manager.source_fingerprints


def test_source_fingerprints_are_read_only():
    manager = AnalysisWorkspace(metabolic_model=_Model(), reaction_data=_reaction_data())

    with pytest.raises(TypeError):
        manager.source_fingerprints["dataset"] = "sha256:forged"


def test_model_identity_is_canonical_across_entity_order():
    forward = model_identity(_Model(reactions=("R1", "R2", "R3")))
    reversed_order = model_identity(_Model(reactions=("R3", "R2", "R1")))

    assert forward == reversed_order


def test_the_cache_recomputes_only_when_the_object_behind_a_component_changes():
    cache = SourceIdentityCache()
    reaction_data = _reaction_data()

    cache.identify_dataset(
        metabolic_model=None, reaction_data=reaction_data, expression_data=None
    )
    first = cache.fingerprints["dataset"]
    cache.identify_dataset(
        metabolic_model=None, reaction_data=reaction_data, expression_data=None
    )

    assert cache.fingerprints["dataset"] == first
    assert "expression" not in cache.fingerprints


def test_a_metadata_transition_does_not_re_identify_the_matrices(monkeypatch):
    """A new label column does not make the matrices a new dataset."""
    manager = _loaded_workspace()
    counter = _FrameFingerprintCounter(monkeypatch)

    manager.update_sample_metadata(
        SampleMetadata(pd.DataFrame({"tissue": ["gut", "liver"]}, index=["s1", "s2"]))
    )

    assert counter.calls == 1, "A metadata change re-hashed a full data matrix"

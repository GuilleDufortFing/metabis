"""A computed reduction must be able to become its artifact (AM5, MR-12).

`ReductionArtifact` shipped with an install seam, coordinate/loading/variance
frame adapters, and a compute-result projection *out* of the artifact -- but
nothing in production ever built one, so the only way in was a test fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metabis import AnalysisWorkspace
from metabis.analysis import artifacts, source_identity
from metabis.analysis.artifacts import ArtifactKind
from metabis.analysis.dimensionality_reduction import (
    DimensionalityReduction,
    ReductionArtifact,
    ReductionKey,
    ReductionResult,
    pca_variance_frame,
    reduction_coordinates_frame,
)
from metabis.analysis_workspace import RevisionKind
from metabis.domain.expression_matrix import DenseExpressionMatrix


def _key(method: str = "PCA") -> ReductionKey:
    return ReductionKey(
        method=method,
        data_type="reaction",
        n_variable_features=2,
        n_components=2,
        random_state=0,
        preprocessing_revision="1",
        sample_revision="1",
    )


def _result(key: ReductionKey | None = None) -> ReductionResult:
    key = key or _key()
    return ReductionResult(
        key=key,
        transformed_data=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        sample_ids=("s1", "s2", "s3"),
        feature_names=("R1", "R2"),
        metadata={
            "pca_model": object(),  # a fitted estimator: must not reach the artifact
            "explained_variance_ratio": np.array([0.7, 0.3]),
            "components": np.array([[0.6, 0.8], [0.8, -0.6]]),
            "n_variable_features_used": 2,
            "data_source_name": "reaction data",
            "n_components_retained": 2,
        },
    )


def _provenance(manager: AnalysisWorkspace):
    return manager.capture_analysis_snapshot(
        [RevisionKind.DATASET, RevisionKind.PREPROCESSING]
    ).provenance


def test_a_computed_reduction_builds_its_artifact():
    manager = AnalysisWorkspace()
    result = _result()

    artifact = result.build_artifact(
        provenance=_provenance(manager),
        source_fingerprints=manager.source_fingerprints,
    )

    assert isinstance(artifact, ReductionArtifact)
    assert artifact.metadata.kind is ArtifactKind.REDUCTION
    assert artifact.metadata.method == "PCA"
    assert artifact.sample_ids == ("s1", "s2", "s3")
    assert artifact.n_components_retained == 2
    assert artifact.data_source_name == "reaction data"
    assert list(reduction_coordinates_frame(artifact)["sample_id"]) == [
        "s1",
        "s2",
        "s3",
    ]
    assert list(pca_variance_frame(artifact)["explained_variance_ratio"]) == [0.7, 0.3]


def test_the_artifact_carries_the_workspace_identity_of_its_real_inputs():
    manager = AnalysisWorkspace()

    artifact = _result().build_artifact(
        provenance=_provenance(manager),
        source_fingerprints=manager.source_fingerprints,
    )

    source = artifact.metadata.source
    assert source.source_fingerprints["dataset"] == manager.source_fingerprints["dataset"]
    assert source.source_fingerprints["dataset"].startswith("sha256:")
    assert source.workspace_id == manager.workspace_id


def test_the_fitted_estimator_does_not_travel_on_the_artifact():
    """An artifact carries computed values, not a live sklearn model."""
    artifact = _result().build_artifact(
        provenance=_provenance(AnalysisWorkspace()),
        source_fingerprints=AnalysisWorkspace().source_fingerprints,
    )

    assert "pca_model" not in dict(artifact.params)
    assert artifact.pca is not None
    assert artifact.pca.loadings.shape == (2, 2)


def test_building_an_artifact_never_fingerprints_a_full_matrix(monkeypatch):
    manager = AnalysisWorkspace()
    provenance = _provenance(manager)
    fingerprints = manager.source_fingerprints

    def forbidden(_frame):
        raise AssertionError(
            "The reduction producer fingerprinted a full matrix; read identity "
            "from the workspace instead (ADR-0007)"
        )

    monkeypatch.setattr(source_identity, "fingerprint_frame", forbidden)
    monkeypatch.setattr(artifacts, "fingerprint_frame", forbidden)

    artifact = _result().build_artifact(
        provenance=provenance, source_fingerprints=fingerprints
    )

    assert artifact.metadata.source.source_fingerprints["dataset"].startswith("sha256:")


def test_the_round_trip_back_to_the_compute_result_is_faithful():
    """The calculator's typed result must survive its artifact projection."""
    original = _result()

    artifact = original.build_artifact(
        provenance=_provenance(AnalysisWorkspace()),
        source_fingerprints=AnalysisWorkspace().source_fingerprints,
    )
    restored = artifact.to_compute_result()

    assert restored.key == original.key
    assert restored.sample_ids == original.sample_ids
    assert restored.feature_names == original.feature_names
    np.testing.assert_array_equal(
        restored.transformed_data, original.transformed_data
    )
    assert restored.metadata["n_components_retained"] == 2
    assert restored.metadata["data_source_name"] == "reaction data"


@pytest.mark.parametrize("method", ["T-SNE", "UMAP"])
def test_method_parameters_reach_the_artifact_for_every_method(method):
    key = ReductionKey(
        method=method,
        data_type="reaction",
        n_variable_features=2,
        n_components=2,
        random_state=0,
        preprocessing_revision="1",
        sample_revision="1",
        n_neighbors=15 if method == "UMAP" else None,
        min_dist=0.1 if method == "UMAP" else None,
        perplexity=30.0 if method == "T-SNE" else None,
    )
    result = _result(key)
    result.metadata.pop("explained_variance_ratio")
    result.metadata.pop("components")

    artifact = result.build_artifact(
        provenance=_provenance(AnalysisWorkspace()),
        source_fingerprints=AnalysisWorkspace().source_fingerprints,
    )

    parameters = dict(artifact.params)
    assert artifact.pca is None
    assert parameters["n_components"] == 2
    if method == "UMAP":
        assert parameters["n_neighbors"] == 15 and parameters["min_dist"] == 0.1
    else:
        assert parameters["perplexity"] == 30.0


@pytest.mark.parametrize("method", ["PCA", "T-SNE", "UMAP"])
def test_every_method_a_real_calculation_produces_can_become_an_artifact(method):
    """Drive the calculator, not a fixture: only PCA records a component count.

    The fixtures above hand the producer a metadata dictionary shaped like PCA's.
    A real t-SNE or UMAP result has never carried `n_components_retained` -- the
    number is a fact of the coordinates -- so a producer that trusted the
    metadata could only build the one artifact its tests handed it (MR-12).
    """
    frame = pd.DataFrame(
        np.arange(24, dtype=float).reshape(6, 4) ** 1.7,
        index=[f"s{index}" for index in range(6)],
        columns=[f"R{index}" for index in range(4)],
    )
    calculator = DimensionalityReduction()
    common = {
        "matrix": DenseExpressionMatrix(frame),
        "data_type": "reactions",
        "data_source_name": "reaction data",
        "n_variable_features": 4,
        "preprocessing_revision": "1",
        "sample_revision": "1",
    }
    if method == "PCA":
        outcome = calculator.calculate_pca(**common)
    elif method == "T-SNE":
        outcome = calculator.calculate_tsne(n_components=2, **common)
    else:
        outcome = calculator.calculate_umap(n_components=2, **common)
    assert outcome["status"] == "success"

    artifact = outcome["result"].build_artifact(
        provenance=_provenance(AnalysisWorkspace()),
        source_fingerprints=AnalysisWorkspace().source_fingerprints,
    )

    assert artifact.key.method == method
    assert artifact.n_components_retained == artifact.coordinates.shape[1]
    assert artifact.n_variable_features_used == len(artifact.feature_names)
    assert (artifact.pca is not None) == (method == "PCA")

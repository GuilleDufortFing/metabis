from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from metabis.analysis.artifacts import (
    AnalysisArtifactMetadata,
    ArtifactDimension,
    ArtifactKind,
    ArtifactProvenance,
)
from metabis.analysis.dimensionality_reduction import (
    PCAArtifacts,
    ReductionArtifact,
    ReductionKey,
    pca_loadings_frame,
    pca_variance_frame,
    reduction_coordinates_frame,
)
from metabis.analysis_workspace import RevisionKind, WorkspaceRevisions


def _pca_artifact(coordinates=None, variance=None, loadings=None) -> ReductionArtifact:
    key = ReductionKey(
        method="PCA",
        data_type="reactions",
        n_variable_features=2,
        n_components=2,
        random_state=42,
        preprocessing_revision="dataset=1|preprocessing=2",
        sample_revision="1",
    )
    metadata = AnalysisArtifactMetadata(
        kind=ArtifactKind.REDUCTION,
        source=ArtifactProvenance(
            workspace_id="workspace",
            revisions=WorkspaceRevisions(dataset=1, preprocessing=2),
            dependencies=frozenset(
                {RevisionKind.DATASET, RevisionKind.PREPROCESSING}
            ),
            source_fingerprints={"dataset": "sha256:exact-frame"},
        ),
        method="PCA",
        configuration={
            "data_type": "reactions",
            "parameters": {
                "n_components": 2,
                "n_variable_features": 2,
                "random_state": 42,
            },
            "preprocessing_revision": "dataset=1|preprocessing=2",
            "sample_revision": "1",
        },
        scientific_identity={
            "sample_ids": ["s1", "s2"],
            "feature_ids": ["R_CANONICAL", "R2"],
            "source": "computed",
        },
        dimensions=(
            ArtifactDimension("samples", 2),
            ArtifactDimension("features", 2),
            ArtifactDimension("components", 2),
        ),
    )
    return ReductionArtifact(
        metadata=metadata,
        key=key,
        coordinates=(
            np.array([[0.12345678901234566, 2.0], [3.0, 4.0]])
            if coordinates is None
            else coordinates
        ),
        sample_ids=("s1", "s2"),
        feature_names=("R_CANONICAL", "R2"),
        data_source_name="reaction data",
        n_variable_features_used=2,
        n_components_retained=2,
        pca=PCAArtifacts(
            explained_variance_ratio=(
                np.array([0.7000000000000001, 0.2999999999999999])
                if variance is None
                else variance
            ),
            loadings=(
                np.array([[0.5, -0.5], [0.25, 0.75]])
                if loadings is None
                else loadings
            ),
        ),
        params=(
            ("n_components", 2),
            ("n_variable_features", 2),
            ("random_state", 42),
        ),
    )


def test_reduction_artifact_deeply_owns_read_only_numeric_arrays():
    coordinates = np.array([[1.0, 2.0], [3.0, 4.0]])
    variance = np.array([0.6, 0.4])
    loadings = np.array([[0.1, 0.2], [0.3, 0.4]])

    artifact = _pca_artifact(coordinates, variance, loadings)
    coordinates[0, 0] = 999
    variance[0] = 999
    loadings[0, 0] = 999

    assert artifact.coordinates[0, 0] == 1.0
    assert artifact.pca.explained_variance_ratio[0] == 0.6
    assert artifact.pca.loadings[0, 0] == 0.1
    assert artifact.coordinates.flags.writeable is False
    assert artifact.pca.explained_variance_ratio.flags.writeable is False
    assert artifact.pca.loadings.flags.writeable is False


def test_reduction_adapters_preserve_canonical_ids_and_full_precision():
    artifact = _pca_artifact()

    coordinates = reduction_coordinates_frame(artifact)
    loadings = pca_loadings_frame(artifact)
    variance = pca_variance_frame(artifact)

    assert coordinates["sample_id"].tolist() == ["s1", "s2"]
    assert coordinates.loc[0, "component_1"] == 0.12345678901234566
    assert loadings["feature_id"].tolist() == ["R_CANONICAL", "R2"]
    assert variance.loc[0, "explained_variance_ratio"] == 0.7000000000000001
    assert variance.loc[1, "cumulative_explained_variance_ratio"] == 1.0


def test_compute_result_projection_reuses_artifact_arrays_for_the_typed_cache():
    """The artifact->result adapter keeps the typed reduction cache populated.

    `install_reduction_artifact` stores this result by ``ReductionKey`` so a
    worker-local ``_reuse`` and ``get_reduction_result(s)`` still find it; it
    must not copy the artifact's arrays.
    """
    artifact = _pca_artifact()

    result = artifact.to_compute_result()

    assert result.feature_names == ("R_CANONICAL", "R2")
    assert np.shares_memory(result.transformed_data, artifact.coordinates)
    assert result.metadata["components"] is artifact.pca.loadings


def test_reduction_rejects_tampered_configuration_and_dimensions():
    artifact = _pca_artifact()
    configuration = dict(artifact.metadata.configuration)
    configuration["data_type"] = "expression"

    with pytest.raises(ValueError, match="configuration"):
        replace(
            artifact,
            metadata=replace(artifact.metadata, configuration=configuration),
        )

    dimensions = tuple(
        ArtifactDimension(item.name, item.size + (1 if item.name == "samples" else 0))
        for item in artifact.metadata.dimensions
    )
    with pytest.raises(ValueError, match="dimensions"):
        replace(artifact, metadata=replace(artifact.metadata, dimensions=dimensions))

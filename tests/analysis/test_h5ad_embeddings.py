"""Imported h5ad embeddings are detached, aligned, and provenance-bearing.

The load-bearing property is what these tests spend most of their assertions on:
**``X`` is never read**. An embedding that forced a sparse matrix to densify
would defeat the purpose of importing it, so the fixtures below give the AnnData
an ``X`` that raises if anything touches its values.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from metabis import AnalysisWorkspace
from metabis.analysis.artifacts import RevisionKind
from metabis.analysis.h5ad_embeddings import (
    EMBEDDING_METHODS,
    EmbeddingAlignmentError,
    UnknownEmbeddingError,
    build_h5ad_reduction_artifact,
    discover_h5ad_embeddings,
)
from metabis.analysis_workspace import (
    StableFrameReference,
    WorkspaceProvenance,
    WorkspaceRevisions,
)
from metabis.domain.expression_data import ExpressionData
from metabis.domain.h5ad_embedding_input import capture_h5ad_embedding_input
from metabis.viz.plot_reduction_artifact import (
    ReductionFeaturePresentation,
    ReductionPlotPresentation,
    ReductionSamplePresentation,
    plot_pca_biplot_artifact,
    plot_pca_loadings_artifact,
    plot_pca_variance_artifact,
)

SAMPLES = ["s1", "s2", "s3", "s4"]
GENES = ["G1", "G2", "G3", "G4", "G5"]


def _adata(
    *,
    obsm: dict | None = None,
    loadings: np.ndarray | None = None,
    variance: np.ndarray | None = None,
    sample_names=SAMPLES,
    sparse: bool = False,
) -> ad.AnnData:
    values = np.arange(len(sample_names) * len(GENES), dtype=float).reshape(
        len(sample_names), len(GENES)
    )
    matrix = sp.csr_matrix(values) if sparse else values
    adata = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=pd.Index(sample_names, name=None)),
        var=pd.DataFrame(index=pd.Index(GENES, name=None)),
    )
    for key, coordinates in (obsm or {}).items():
        adata.obsm[key] = np.asarray(coordinates, dtype=float)
    if loadings is not None:
        adata.varm["PCs"] = np.asarray(loadings, dtype=float)
    if variance is not None:
        adata.uns["pca"] = {"variance_ratio": np.asarray(variance, dtype=float)}
    return adata


def _pca_coordinates(n_samples: int = 4, n_components: int = 3) -> np.ndarray:
    return np.arange(n_samples * n_components, dtype=float).reshape(
        n_samples, n_components
    )


def _provenance() -> WorkspaceProvenance:
    return WorkspaceProvenance(
        workspace_id="workspace-r02",
        revisions=WorkspaceRevisions(dataset=1, preprocessing=2),
        dependencies=frozenset({RevisionKind.DATASET, RevisionKind.PREPROCESSING}),
        reaction_frame=StableFrameReference.capture(
            pd.DataFrame({"R1": [1.0]}, index=["s1"])
        ),
        expression_matrix=None,
        group_memberships=(),
    )


def _artifact(adata, key="X_pca", *, sample_ids=SAMPLES, **overrides):
    embedding_input = capture_h5ad_embedding_input(adata)
    assert embedding_input is not None
    return build_h5ad_reduction_artifact(
        embedding_input,
        key,
        sample_ids=sample_ids,
        provenance=_provenance(),
        source_fingerprints={"dataset": "sha256:r02"},
        **overrides,
    )


def _presentation(artifact):
    return ReductionPlotPresentation(
        samples=tuple(
            ReductionSamplePresentation(sample_id, sample_id)
            for sample_id in artifact.sample_ids
        ),
        groups=(),
        features=tuple(
            ReductionFeaturePresentation(feature_id, feature_id)
            for feature_id in artifact.feature_names
        ),
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_it_detects_each_documented_obsm_key_with_its_method_and_shape():
    adata = _adata(
        obsm={
            "X_pca": _pca_coordinates(),
            "X_umap": np.zeros((4, 2)),
            "X_tsne": np.zeros((4, 2)),
        }
    )

    detected = discover_h5ad_embeddings(adata)

    assert [embedding.key for embedding in detected] == list(EMBEDDING_METHODS)
    assert [embedding.method for embedding in detected] == ["PCA", "UMAP", "T-SNE"]
    by_key = {embedding.key: embedding for embedding in detected}
    assert (by_key["X_pca"].n_samples, by_key["X_pca"].n_components) == (4, 3)
    assert by_key["X_umap"].n_components == 2


def test_a_file_carrying_no_embedding_reports_none_rather_than_failing():
    assert discover_h5ad_embeddings(_adata()) == ()


def test_an_unrecognised_obsm_key_is_not_offered():
    """``obsm`` is free-form; a key Metabis cannot name it must not guess at."""
    adata = _adata(obsm={"X_pca": _pca_coordinates(), "X_something": np.zeros((4, 2))})

    assert [embedding.key for embedding in discover_h5ad_embeddings(adata)] == ["X_pca"]


def test_loadings_and_variance_are_reported_only_when_the_file_carries_them():
    without = discover_h5ad_embeddings(_adata(obsm={"X_pca": _pca_coordinates()}))[0]
    with_both = discover_h5ad_embeddings(
        _adata(
            obsm={"X_pca": _pca_coordinates()},
            loadings=np.ones((5, 3)),
            variance=np.array([0.5, 0.3, 0.2]),
        )
    )[0]

    assert not without.has_loadings and not without.has_explained_variance
    assert not without.supports_pca_surfaces
    assert with_both.supports_pca_surfaces
    assert with_both.n_loading_features == 5


def test_loadings_that_do_not_cover_every_component_are_not_used():
    """Pairing coordinate axis 3 with a loading vector that stops at 2 is a guess."""
    adata = _adata(
        obsm={"X_pca": _pca_coordinates(n_components=3)},
        loadings=np.ones((5, 2)),
        variance=np.array([0.5, 0.3, 0.2]),
    )

    assert not discover_h5ad_embeddings(adata)[0].has_loadings


def test_a_non_pca_embedding_never_claims_pca_surfaces():
    adata = _adata(
        obsm={"X_umap": np.zeros((4, 2))},
        loadings=np.ones((5, 3)),
        variance=np.array([0.5, 0.3, 0.2]),
    )
    umap = discover_h5ad_embeddings(adata)[0]

    assert not umap.supports_pca_surfaces
    assert "defined for PCA" in umap.unavailable_reason()


def test_a_complete_pca_embedding_has_no_reason_to_state():
    """The case the live app exposed: every guard passed and it still complained.

    ``unavailable_reason`` guarded each *missing* combination and then fell
    through to the "without explained variance" sentence, so a file that stored
    loadings *and* variance was told its variance was missing. Nothing else
    caught it -- the artifact carried its PCA payload correctly and the card drew
    fine; only the note under the plot was wrong.
    """
    adata = _adata(
        obsm={"X_pca": _pca_coordinates()},
        loadings=np.ones((5, 3)),
        variance=np.array([0.5, 0.3, 0.2]),
    )
    embedding = discover_h5ad_embeddings(adata)[0]

    assert embedding.supports_pca_surfaces
    assert embedding.unavailable_reason() is None


@pytest.mark.parametrize(
    ("loadings", "variance", "expected"),
    (
        (None, None, "without PCA loadings"),
        (np.ones((5, 3)), None, "without explained variance"),
        (None, np.array([0.5, 0.3, 0.2]), "without PCA loadings"),
    ),
)
def test_an_absent_pca_surface_states_its_reason(loadings, variance, expected):
    """The server shows this instead of an empty card."""
    adata = _adata(
        obsm={"X_pca": _pca_coordinates()}, loadings=loadings, variance=variance
    )

    assert expected in discover_h5ad_embeddings(adata)[0].unavailable_reason()


# ---------------------------------------------------------------------------
# X is never read
# ---------------------------------------------------------------------------


class _ExplodingMatrix:
    """An ``X`` that fails on any attempt to read its values."""

    shape = (len(SAMPLES), len(GENES))
    dtype = np.dtype(float)

    def __getattr__(self, name):
        raise AssertionError(f"the embedding import read X.{name}")

    def __array__(self, *_args, **_kwargs):
        raise AssertionError("the embedding import densified X")

    def __getitem__(self, _item):
        raise AssertionError("the embedding import indexed X")


class _FakeAnnData:
    """The narrow surface the reader is allowed to touch."""

    def __init__(self, adata):
        self.obs_names = adata.obs_names
        self.var_names = adata.var_names
        self.obsm = adata.obsm
        self.varm = adata.varm
        self.uns = adata.uns
        self.X = _ExplodingMatrix()


def test_neither_discovery_nor_import_reads_the_expression_matrix():
    source = _adata(
        obsm={"X_pca": _pca_coordinates(), "X_umap": np.zeros((4, 2))},
        loadings=np.ones((5, 3)),
        variance=np.array([0.5, 0.3, 0.2]),
        sparse=True,
    )
    guarded = _FakeAnnData(source)

    detected = discover_h5ad_embeddings(guarded)
    artifact = _artifact(guarded)

    assert [embedding.key for embedding in detected] == ["X_pca", "X_umap"]
    assert artifact.coordinates.shape == (4, 3)


def test_the_load_writer_capture_is_detached_read_only_and_mutation_isolated():
    coordinates = _pca_coordinates()
    adata = _adata(
        obsm={"X_pca": coordinates},
        loadings=np.ones((5, 3)),
        variance=np.array([0.5, 0.3, 0.2]),
    )
    embedding_input = capture_h5ad_embedding_input(adata)
    assert embedding_input is not None
    captured = embedding_input.coordinate("X_pca")
    assert captured is not None

    adata.obsm["X_pca"][0, 0] = 999.0
    adata.varm["PCs"][0, 0] = 888.0
    adata.uns["pca"]["variance_ratio"][0] = 0.0

    assert captured.values[0, 0] == 0.0
    assert embedding_input.pca_loadings[0, 0] == 1.0
    assert embedding_input.pca_explained_variance[0] == 0.5
    assert captured.values.flags.writeable is False
    assert embedding_input.pca_loadings.flags.writeable is False
    assert embedding_input.pca_explained_variance.flags.writeable is False


def test_embedding_content_changes_artifact_identity_even_when_x_is_identical():
    first = _artifact(_adata(obsm={"X_umap": np.zeros((4, 2))}), "X_umap")
    changed_coordinates = np.zeros((4, 2))
    changed_coordinates[0, 0] = 1.0
    second = _artifact(
        _adata(obsm={"X_umap": changed_coordinates}),
        "X_umap",
    )

    assert (
        first.metadata.scientific_identity["embedding_fingerprint"]
        != second.metadata.scientific_identity["embedding_fingerprint"]
    )
    assert first.metadata.artifact_id != second.metadata.artifact_id


def test_expression_load_writer_captures_and_identifies_embedding_content(tmp_path):
    path = tmp_path / "embedding-input.h5ad"
    _adata(
        obsm={"X_umap": np.zeros((4, 2))},
        sparse=True,
    ).write_h5ad(path)

    expression = ExpressionData(
        path=str(path),
        gene_list=GENES,
        data_format="h5ad",
    )
    manager = AnalysisWorkspace()
    manager.replace_expression_data(expression)

    assert expression.embedding_input is not None
    assert expression.embedding_input.coordinate("X_umap") is not None
    assert (
        manager.source_fingerprints["embeddings"]
        == expression.embedding_input.content_fingerprint
    )


def test_a_sparse_file_is_imported_without_densifying_it():
    adata = _adata(obsm={"X_umap": np.zeros((4, 2))}, sparse=True)

    artifact = _artifact(adata, "X_umap")

    assert sp.issparse(adata.X), "the import replaced the sparse X with a dense one"
    assert artifact.coordinates.shape == (4, 2)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def test_rows_are_selected_by_obs_name_in_the_workspace_order():
    coordinates = _pca_coordinates()
    adata = _adata(obsm={"X_pca": coordinates})

    artifact = _artifact(adata, sample_ids=["s3", "s1"])

    assert artifact.sample_ids == ("s3", "s1")
    np.testing.assert_array_equal(artifact.coordinates[0], coordinates[2])
    np.testing.assert_array_equal(artifact.coordinates[1], coordinates[0])


def test_extra_observations_in_the_file_are_simply_not_plotted():
    adata = _adata(
        obsm={"X_umap": np.zeros((5, 2))}, sample_names=[*SAMPLES, "not_aligned"]
    )

    artifact = _artifact(adata, "X_umap", sample_ids=SAMPLES)

    assert artifact.sample_ids == tuple(SAMPLES)
    assert artifact.coordinates.shape == (4, 2)


def test_a_sample_missing_from_obs_names_is_a_named_error():
    adata = _adata(obsm={"X_umap": np.zeros((4, 2))})

    with pytest.raises(EmbeddingAlignmentError, match="no row in the h5ad file"):
        _artifact(adata, "X_umap", sample_ids=[*SAMPLES, "absent"])


def test_duplicate_obs_names_are_a_named_error_rather_than_a_silent_first_match():
    adata = _adata(obsm={"X_umap": np.zeros((4, 2))}, sample_names=["s1", "s1", "s3", "s4"])

    with pytest.raises(EmbeddingAlignmentError, match="duplicate obs_names"):
        _artifact(adata, "X_umap", sample_ids=["s1", "s3"])


def test_an_absent_key_names_what_the_file_does_offer():
    adata = _adata(obsm={"X_umap": np.zeros((4, 2))})

    with pytest.raises(UnknownEmbeddingError, match="X_umap"):
        _artifact(adata, "X_pca")


# ---------------------------------------------------------------------------
# The artifact and its provenance
# ---------------------------------------------------------------------------


def test_the_artifact_records_that_nothing_was_computed():
    adata = _adata(obsm={"X_pca": _pca_coordinates()})

    artifact = _artifact(adata)

    assert artifact.source == "h5ad"
    assert artifact.was_computed is False
    identity = artifact.metadata.scientific_identity
    assert identity["source"] == "h5ad"
    assert identity["h5ad_key"] == "X_pca"
    assert identity["h5ad_shape"] == (4, 3)
    assert "imported" in identity["computation"]


def test_an_imported_pca_without_varm_still_builds_and_offers_no_pca_payload():
    """The gate the server reads: coordinates plot, the PCA cards do not."""
    artifact = _artifact(_adata(obsm={"X_pca": _pca_coordinates()}))

    assert artifact.key.method == "PCA"
    assert artifact.pca is None
    assert artifact.feature_names == ()
    assert artifact.n_variable_features_used == 0


def test_pca_loadings_without_variance_support_loadings_and_biplot_payloads():
    loadings = np.arange(15, dtype=float).reshape(5, 3)
    artifact = _artifact(
        _adata(obsm={"X_pca": _pca_coordinates()}, loadings=loadings)
    )

    assert artifact.pca is not None
    assert artifact.pca.loadings is not None
    assert artifact.pca.explained_variance_ratio is None
    np.testing.assert_array_equal(artifact.pca.loadings, loadings.T)
    assert artifact.feature_names == tuple(GENES)
    assert plot_pca_loadings_artifact(artifact, _presentation(artifact)) is not None
    assert plot_pca_biplot_artifact(artifact, _presentation(artifact)) is not None
    with pytest.raises(ValueError, match="explained variance"):
        plot_pca_variance_artifact(artifact, _presentation(artifact))


def test_pca_variance_without_loadings_supports_only_the_variance_payload():
    variance = np.array([0.5, 0.3, 0.2])
    artifact = _artifact(
        _adata(obsm={"X_pca": _pca_coordinates()}, variance=variance)
    )

    assert artifact.pca is not None
    assert artifact.pca.loadings is None
    np.testing.assert_array_equal(artifact.pca.explained_variance_ratio, variance)
    assert artifact.feature_names == ()
    assert plot_pca_variance_artifact(artifact, _presentation(artifact)) is not None
    with pytest.raises(ValueError, match="loadings"):
        plot_pca_loadings_artifact(artifact, _presentation(artifact))


def test_varm_loadings_are_transposed_onto_the_artifact_component_axis():
    loadings = np.arange(15, dtype=float).reshape(5, 3)  # features x components
    adata = _adata(
        obsm={"X_pca": _pca_coordinates()},
        loadings=loadings,
        variance=np.array([0.5, 0.3, 0.2]),
    )

    artifact = _artifact(adata)

    assert artifact.pca is not None
    assert artifact.pca.loadings.shape == (3, 5)  # components x features
    np.testing.assert_array_equal(artifact.pca.loadings, loadings.T)
    np.testing.assert_allclose(
        artifact.pca.explained_variance_ratio, [0.5, 0.3, 0.2]
    )
    assert artifact.feature_names == tuple(GENES)


def test_extra_stored_components_are_trimmed_to_the_embedding_width():
    adata = _adata(
        obsm={"X_pca": _pca_coordinates(n_components=2)},
        loadings=np.ones((5, 6)),
        variance=np.full(6, 1 / 6),
    )

    artifact = _artifact(adata)

    assert artifact.n_components_retained == 2
    assert artifact.pca.loadings.shape == (2, 5)
    assert artifact.pca.explained_variance_ratio.shape == (2,)


def test_the_imported_artifact_survives_the_round_trip_through_a_compute_result():
    """Installation and the companion cards both go through this projection."""
    adata = _adata(
        obsm={"X_pca": _pca_coordinates()},
        loadings=np.ones((5, 3)),
        variance=np.array([0.5, 0.3, 0.2]),
    )
    artifact = _artifact(adata)

    result = artifact.to_compute_result()

    assert result.source == "h5ad"
    np.testing.assert_array_equal(result.transformed_data, artifact.coordinates)

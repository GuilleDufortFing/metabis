"""Build reduction artifacts from detached h5ad embedding input.

The h5ad load writer copies recognized ``obsm`` coordinates and relevant
``varm``/``uns`` PCA metadata into an immutable input value. Analysis workers
consume that value, never a live ``AnnData`` object and never ``X``. Imported
coordinates are aligned by canonical sample identifier and retain a content
fingerprint in scientific identity and provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..domain.h5ad_embedding_input import (
    H5AD_EMBEDDING_METHODS,
    H5adEmbeddingInput,
    capture_h5ad_embedding_input,
)
from .artifacts import AnalysisArtifactMetadata, ArtifactDimension, ArtifactKind
from .dimensionality_reduction import PCAArtifacts, ReductionArtifact, ReductionKey

__all__ = [
    "EMBEDDING_METHODS",
    "DetectedEmbedding",
    "EmbeddingAlignmentError",
    "UnknownEmbeddingError",
    "build_h5ad_reduction_artifact",
    "discover_h5ad_embeddings",
]

EMBEDDING_METHODS: Mapping[str, str] = H5AD_EMBEDDING_METHODS
"""The ``obsm`` keys Metabis imports, and the method each one names.

Deliberately a closed set. ``obsm`` is a free-form store, and a key Metabis does
not recognise carries no promise about what its columns mean; offering it as
"the UMAP" because it happens to be two-dimensional would be a guess presented
as provenance.
"""

PCA_LOADINGS_KEY = "PCs"
"""Where scanpy stores PCA loadings: ``varm["PCs"]``, features by components."""


class UnknownEmbeddingError(KeyError):
    """The requested ``obsm`` key is absent or is not one Metabis imports."""


class EmbeddingAlignmentError(ValueError):
    """The embedding's rows cannot be aligned to the workspace's samples."""


@dataclass(frozen=True)
class DetectedEmbedding:
    """One importable embedding found in an ``h5ad`` file.

    ``has_loadings`` and ``has_explained_variance`` are what the server gates the
    loadings plot, the biplot, and the explained-variance card on: an imported
    UMAP has neither, and an imported PCA has them only if whoever wrote the file
    kept them.
    """

    key: str
    method: str
    n_samples: int
    n_components: int
    has_loadings: bool = False
    has_explained_variance: bool = False
    n_loading_features: int = 0

    @property
    def supports_pca_surfaces(self) -> bool:
        """Whether all loadings, biplot, and variance cards can be drawn."""
        return self.has_loadings and self.has_explained_variance

    def unavailable_reason(self) -> str | None:
        """Describe exactly which PCA-only surfaces are unavailable.

        Loadings and the biplot require ``varm["PCs"]``. The explained-variance
        card independently requires ``uns["pca"]["variance_ratio"]``.
        """
        if self.supports_pca_surfaces:
            return None
        if self.method != "PCA":
            return (
                f"{self.method} produces coordinates only. Loadings, a biplot and "
                "explained variance are defined for PCA."
            )
        if not self.has_loadings and not self.has_explained_variance:
            return (
                f"The file stores {self.key} without PCA loadings "
                f"(varm['{PCA_LOADINGS_KEY}']) or explained variance "
                "(uns['pca']['variance_ratio']), so only the coordinates were "
                "imported."
            )
        if not self.has_loadings:
            return (
                f"The file stores {self.key} without PCA loadings "
                f"(varm['{PCA_LOADINGS_KEY}']), so the loadings heatmap and the "
                "biplot cannot be drawn from it; explained variance remains "
                "available."
            )
        return (
            f"The file stores {self.key} without explained variance "
            "(uns['pca']['variance_ratio']), so the explained-variance card is "
            "unavailable; the loadings heatmap and biplot remain available."
        )


def _embedding_input(source) -> H5adEmbeddingInput | None:
    """Accept the detached boundary value, with AnnData capture for direct callers."""
    if isinstance(source, H5adEmbeddingInput):
        return source
    return capture_h5ad_embedding_input(source)


def discover_h5ad_embeddings(source) -> tuple[DetectedEmbedding, ...]:
    """Report the embeddings a detached h5ad input offers.

    The result is ordered by :data:`EMBEDDING_METHODS`, so the offered list is
    stable across files.

    Args:
        source: A :class:`H5adEmbeddingInput`. An AnnData object is also accepted
            for direct library use and is detached immediately without reading
            ``X``.

    Returns:
        One :class:`DetectedEmbedding` per recognised key that holds a
        two-dimensional array with at least two components. An empty tuple means
        the file carries nothing importable, which is not an error.
    """
    embedding_input = _embedding_input(source)
    if embedding_input is None:
        return ()

    variance = embedding_input.pca_explained_variance
    loadings = embedding_input.pca_loadings

    detected: list[DetectedEmbedding] = []
    for item in embedding_input.coordinates:
        key, method, coordinates = item.key, item.method, item.values
        n_components = int(coordinates.shape[1])
        is_pca = method == "PCA"
        # Loadings must cover every component the embedding actually has, or the
        # pairing between a coordinate axis and its loading vector is a guess.
        has_loadings = bool(
            is_pca and loadings is not None and loadings.shape[1] >= n_components
        )
        has_variance = bool(
            is_pca and variance is not None and variance.size >= n_components
        )
        detected.append(
            DetectedEmbedding(
                key=key,
                method=method,
                n_samples=int(coordinates.shape[0]),
                n_components=n_components,
                has_loadings=has_loadings,
                has_explained_variance=has_variance,
                n_loading_features=int(loadings.shape[0]) if has_loadings else 0,
            )
        )
    return tuple(detected)


def _aligned_rows(
    embedding_input: H5adEmbeddingInput, sample_ids: Sequence[str]
) -> np.ndarray:
    """Positions in captured sample IDs for each requested sample, or an error."""
    observations = list(embedding_input.sample_ids)
    positions = {name: index for index, name in enumerate(observations)}
    if len(positions) != len(observations):
        raise EmbeddingAlignmentError(
            "The h5ad file has duplicate obs_names, so an embedding row cannot be "
            "attributed to one sample. De-duplicate the observations before import."
        )
    missing = [str(sample_id) for sample_id in sample_ids if str(sample_id) not in positions]
    if missing:
        preview = ", ".join(missing[:5])
        more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise EmbeddingAlignmentError(
            f"{len(missing)} aligned sample(s) have no row in the h5ad file's "
            f"obs_names, so the imported embedding cannot be plotted against the "
            f"current sample set: {preview}{more}"
        )
    return np.array([positions[str(sample_id)] for sample_id in sample_ids], dtype=int)


def build_h5ad_reduction_artifact(
    source,
    key: str,
    *,
    sample_ids: Sequence[str],
    provenance: Any,
    source_fingerprints: Mapping[str, str],
    data_source_name: str = "expression data",
    data_type: str = "expression",
    preprocessing_revision: str = "",
    sample_revision: str = "",
    warnings: Sequence[str] = (),
) -> ReductionArtifact:
    """Build the artifact for one imported embedding, computing nothing.

    Rows are selected by ``obs_names`` in the order ``sample_ids`` gives, so the
    artifact's coordinates line up with the workspace's aligned sample set even
    when the file stores its observations in another order or holds extra ones.

    Args:
        source: Detached h5ad embedding input. An AnnData object is accepted for
            direct library use and detached immediately.
        key: Which ``obsm`` entry to import; must be in
            :data:`EMBEDDING_METHODS`.
        sample_ids: The workspace's aligned sample identifiers, in display order.
        provenance: The workspace provenance the artifact is stamped with.
        source_fingerprints: Input identity, as every other artifact records it.
        preprocessing_revision: Recorded for symmetry with a computed reduction.
            An imported embedding was produced upstream and does not depend on
            Metabis preprocessing, so ``""`` is the honest value.

    Raises:
        UnknownEmbeddingError: if ``key`` is absent or is not importable.
        EmbeddingAlignmentError: if the rows cannot be aligned to ``sample_ids``.
    """
    embedding_input = _embedding_input(source)
    detected = {
        embedding.key: embedding
        for embedding in discover_h5ad_embeddings(embedding_input)
    }
    embedding = detected.get(key)
    if embedding is None:
        offered = ", ".join(sorted(detected)) or "none"
        raise UnknownEmbeddingError(
            f"'{key}' is not an importable embedding in this file. Importable "
            f"keys present: {offered}."
        )

    assert embedding_input is not None
    coordinate_input = embedding_input.coordinate(key)
    assert coordinate_input is not None
    rows = _aligned_rows(embedding_input, sample_ids)
    coordinates = coordinate_input.values[rows, :]
    ordered_sample_ids = tuple(str(sample_id) for sample_id in sample_ids)

    pca: PCAArtifacts | None = None
    feature_names: tuple[str, ...] = ()
    if embedding.has_loadings or embedding.has_explained_variance:
        loadings = embedding_input.pca_loadings
        variance = embedding_input.pca_explained_variance
        if embedding.has_loadings:
            feature_names = embedding_input.feature_ids
        # ``varm`` is features x components and the artifact wants components x
        # features. Each optional payload is trimmed independently.
        pca = PCAArtifacts(
            explained_variance_ratio=(
                variance[: embedding.n_components]
                if embedding.has_explained_variance and variance is not None
                else None
            ),
            loadings=(
                loadings[:, : embedding.n_components].T
                if embedding.has_loadings and loadings is not None
                else None
            ),
        )

    reduction_key = ReductionKey(
        method=embedding.method,
        data_type=data_type,
        # An imported embedding selected no features and used no seed: these are
        # parameters of a fit that did not happen here. They are recorded as the
        # zeros they are, and ``source="h5ad"`` is what says why.
        n_variable_features=len(feature_names),
        n_components=embedding.n_components,
        random_state=0,
        preprocessing_revision=preprocessing_revision,
        sample_revision=sample_revision,
    )
    metadata = AnalysisArtifactMetadata.from_workspace(
        kind=ArtifactKind.REDUCTION,
        provenance=provenance,
        source_fingerprints=source_fingerprints,
        method=embedding.method,
        configuration={
            "data_type": data_type,
            "parameters": {
                "n_components": embedding.n_components,
                "n_variable_features": len(feature_names),
                "random_state": 0,
            },
            "preprocessing_revision": preprocessing_revision,
            "sample_revision": sample_revision,
        },
        scientific_identity={
            "sample_ids": ordered_sample_ids,
            "feature_ids": feature_names,
            "source": "h5ad",
            # Imported coordinates state their h5ad origin and that no fit ran.
            "h5ad_key": key,
            "h5ad_shape": (embedding.n_samples, embedding.n_components),
            "embedding_fingerprint": embedding_input.fingerprint_for(key),
            "computation": "none - coordinates imported from the h5ad file",
        },
        dimensions=(
            ArtifactDimension("samples", len(ordered_sample_ids)),
            ArtifactDimension("features", len(feature_names)),
            ArtifactDimension("components", embedding.n_components),
        ),
        warnings=tuple(warnings),
    )
    return ReductionArtifact(
        metadata=metadata,
        key=reduction_key,
        coordinates=coordinates,
        sample_ids=ordered_sample_ids,
        feature_names=feature_names,
        source="h5ad",
        data_source_name=data_source_name,
        n_variable_features_used=len(feature_names),
        n_components_retained=embedding.n_components,
        pca=pca,
        params=(
            ("n_components", embedding.n_components),
            ("n_variable_features", len(feature_names)),
            ("random_state", 0),
        ),
    )

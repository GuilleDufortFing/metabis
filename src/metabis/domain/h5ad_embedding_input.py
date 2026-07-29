"""Detached, immutable embedding content captured from an h5ad load."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

H5AD_EMBEDDING_METHODS: Mapping[str, str] = {
    "X_pca": "PCA",
    "X_umap": "UMAP",
    "X_tsne": "T-SNE",
}


def _owned_read_only_array(values) -> np.ndarray:
    array = np.array(values, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def _update_text(digest, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _update_array(digest, values: np.ndarray | None) -> None:
    if values is None:
        digest.update(b"none\0")
        return
    array = np.ascontiguousarray(values, dtype="<f8")
    digest.update(b"array\0")
    digest.update(len(array.shape).to_bytes(2, "little"))
    for size in array.shape:
        digest.update(int(size).to_bytes(8, "little"))
    digest.update(array.tobytes(order="C"))


@dataclass(frozen=True)
class H5adEmbeddingCoordinates:
    """One recognized h5ad ``obsm`` coordinate matrix."""

    key: str
    method: str
    values: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if H5AD_EMBEDDING_METHODS.get(self.key) != self.method:
            raise ValueError(f"Unsupported h5ad embedding key/method: {self.key}")
        values = _owned_read_only_array(self.values)
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError("Embedding coordinates require at least two components")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class H5adEmbeddingInput:
    """All importable embedding content detached from a loaded AnnData object."""

    sample_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    coordinates: tuple[H5adEmbeddingCoordinates, ...]
    pca_loadings: np.ndarray | None = field(default=None, repr=False, compare=False)
    pca_explained_variance: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    content_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_ids, tuple) or not all(
            isinstance(value, str) for value in self.sample_ids
        ):
            raise TypeError("Embedding sample IDs must be a tuple of strings")
        if not isinstance(self.feature_ids, tuple) or not all(
            isinstance(value, str) for value in self.feature_ids
        ):
            raise TypeError("Embedding feature IDs must be a tuple of strings")
        if not self.coordinates:
            raise ValueError("Embedding input requires at least one coordinate matrix")
        if len({item.key for item in self.coordinates}) != len(self.coordinates):
            raise ValueError("Embedding coordinate keys must be unique")
        for item in self.coordinates:
            if item.values.shape[0] != len(self.sample_ids):
                raise ValueError(
                    f"Embedding {item.key} rows must align with sample IDs"
                )

        loadings = (
            None
            if self.pca_loadings is None
            else _owned_read_only_array(self.pca_loadings)
        )
        if loadings is not None:
            if loadings.ndim != 2 or not loadings.size:
                raise ValueError("PCA loadings must be a non-empty two-dimensional array")
            if loadings.shape[0] != len(self.feature_ids):
                raise ValueError("PCA loading rows must align with feature IDs")
        variance = (
            None
            if self.pca_explained_variance is None
            else _owned_read_only_array(self.pca_explained_variance).ravel()
        )
        if variance is not None and not variance.size:
            raise ValueError("PCA explained variance must not be empty")
        object.__setattr__(self, "pca_loadings", loadings)
        object.__setattr__(self, "pca_explained_variance", variance)
        object.__setattr__(
            self, "content_fingerprint", self._fingerprint(selected_key=None)
        )

    def coordinate(self, key: str) -> H5adEmbeddingCoordinates | None:
        """Return one detached coordinate matrix by its h5ad key."""
        return next((item for item in self.coordinates if item.key == key), None)

    def fingerprint_for(self, key: str) -> str:
        """Identity of one embedding and the PCA metadata relevant to it."""
        if self.coordinate(key) is None:
            raise KeyError(key)
        return self._fingerprint(selected_key=key)

    def _fingerprint(self, *, selected_key: str | None) -> str:
        digest = hashlib.sha256()
        digest.update(b"metabis.h5ad-embeddings.v1\0")
        digest.update(b"samples\0")
        digest.update(len(self.sample_ids).to_bytes(8, "little"))
        for value in self.sample_ids:
            _update_text(digest, value)
        selected = (
            self.coordinates
            if selected_key is None
            else (self.coordinate(selected_key),)
        )
        digest.update(b"coordinates\0")
        digest.update(len(selected).to_bytes(8, "little"))
        for item in selected:
            assert item is not None
            _update_text(digest, item.key)
            _update_text(digest, item.method)
            _update_array(digest, item.values)
        includes_pca = selected_key == "X_pca" or (
            selected_key is None and self.coordinate("X_pca") is not None
        )
        if includes_pca:
            digest.update(b"pca-metadata\0")
            digest.update(len(self.feature_ids).to_bytes(8, "little"))
            for value in self.feature_ids:
                _update_text(digest, value)
            _update_array(digest, self.pca_loadings)
            _update_array(digest, self.pca_explained_variance)
        return f"sha256:{digest.hexdigest()}"


def capture_h5ad_embedding_input(adata) -> H5adEmbeddingInput | None:
    """Copy recognized embedding content without reading ``AnnData.X``."""
    obsm = getattr(adata, "obsm", None)
    if obsm is None:
        return None
    coordinates = []
    for key, method in H5AD_EMBEDDING_METHODS.items():
        if key not in obsm:
            continue
        values = np.asarray(obsm[key], dtype=float)
        if values.ndim != 2 or values.shape[1] < 2:
            continue
        coordinates.append(H5adEmbeddingCoordinates(key, method, values))
    if not coordinates:
        return None

    loadings = None
    varm = getattr(adata, "varm", None)
    if varm is not None and "PCs" in varm:
        candidate = np.asarray(varm["PCs"], dtype=float)
        if candidate.ndim == 2 and candidate.size:
            loadings = candidate

    variance = None
    pca_uns = getattr(adata, "uns", {}).get("pca")
    if isinstance(pca_uns, Mapping) and pca_uns.get("variance_ratio") is not None:
        candidate = np.asarray(pca_uns["variance_ratio"], dtype=float).ravel()
        if candidate.size:
            variance = candidate

    return H5adEmbeddingInput(
        sample_ids=tuple(map(str, adata.obs_names)),
        feature_ids=tuple(map(str, adata.var_names)),
        coordinates=tuple(coordinates),
        pca_loadings=loadings,
        pca_explained_variance=variance,
    )


__all__ = [
    "H5AD_EMBEDDING_METHODS",
    "H5adEmbeddingCoordinates",
    "H5adEmbeddingInput",
    "capture_h5ad_embedding_input",
]

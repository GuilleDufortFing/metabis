from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from ..analysis_workspace import WorkspaceProvenance
from ..domain.expression_matrix import ExpressionMatrix
from .artifacts import AnalysisArtifactMetadata, ArtifactDimension, ArtifactKind

__all__ = [
    "DimensionalityReduction",
    "ReductionKey",
    "ReductionResult",
    "ReductionArtifact",
    "PCAArtifacts",
    "reduction_coordinates_frame",
    "pca_loadings_frame",
    "pca_variance_frame",
]


def _reduction_parameters(key: "ReductionKey") -> Tuple[Tuple[str, object], ...]:
    """Return the key's method parameters, sorted, as the artifact records them."""
    parameters: Dict[str, object] = {
        "n_components": key.n_components,
        "n_variable_features": key.n_variable_features,
        "random_state": key.random_state,
    }
    for name in ("n_neighbors", "min_dist", "perplexity"):
        value = getattr(key, name)
        if value is not None:
            parameters[name] = value
    return tuple(sorted(parameters.items()))


@dataclass(frozen=True)
class ReductionKey:
    """Complete identity of a computed dimensionality reduction."""

    method: Literal["PCA", "T-SNE", "UMAP"]
    data_type: str
    n_variable_features: int
    n_components: int
    random_state: int
    preprocessing_revision: str
    sample_revision: str
    n_neighbors: int | None = None
    min_dist: float | None = None
    perplexity: float | None = None


@dataclass
class ReductionResult:
    """Typed reduction result with coordinates and provenance."""

    key: ReductionKey
    transformed_data: np.ndarray
    sample_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    source: Literal["computed", "uploaded", "h5ad"] = "computed"
    metadata: dict[str, Any] = field(default_factory=dict)

    def build_artifact(
        self,
        *,
        provenance: "WorkspaceProvenance",
        source_fingerprints: Mapping[str, str],
        warnings: Sequence[str] = (),
    ) -> "ReductionArtifact":
        """Project this computed reduction into its portable artifact.

        Input identity is read from the Analysis Workspace, which paid for it once
        at load; recomputing it here would hash the full matrix on every reduction
        (ADR-0007, MR-10). The live `pca_model` in the compute metadata is
        deliberately left behind: an artifact carries computed values, not a
        fitted estimator.
        """
        metadata = dict(self.metadata)
        variance = metadata.get("explained_variance_ratio")
        loadings = metadata.get("components")
        pca = (
            PCAArtifacts(explained_variance_ratio=variance, loadings=loadings)
            if variance is not None or loadings is not None
            else None
        )
        # The counts are facts of the payload, so they are read from it rather
        # than from the metadata. Only PCA records a retained-component count;
        # a t-SNE or UMAP result has always carried the number in its shape.
        coordinates = np.asarray(self.transformed_data)
        n_components_retained = int(coordinates.shape[1])
        n_variable_features_used = len(self.feature_names)
        return ReductionArtifact(
            metadata=AnalysisArtifactMetadata.from_workspace(
                kind=ArtifactKind.REDUCTION,
                provenance=provenance,
                source_fingerprints=source_fingerprints,
                method=self.key.method,
                configuration={
                    "data_type": self.key.data_type,
                    "parameters": dict(_reduction_parameters(self.key)),
                    "preprocessing_revision": self.key.preprocessing_revision,
                    "sample_revision": self.key.sample_revision,
                },
                scientific_identity={
                    "sample_ids": self.sample_ids,
                    "feature_ids": self.feature_names,
                    "source": self.source,
                },
                dimensions=(
                    ArtifactDimension("samples", len(self.sample_ids)),
                    ArtifactDimension("features", n_variable_features_used),
                    ArtifactDimension("components", n_components_retained),
                ),
                warnings=tuple(warnings),
            ),
            key=self.key,
            coordinates=self.transformed_data,
            sample_ids=self.sample_ids,
            feature_names=self.feature_names,
            source=self.source,
            data_source_name=str(metadata.get("data_source_name", "data")),
            n_variable_features_used=n_variable_features_used,
            n_components_retained=n_components_retained,
            pca=pca,
            params=_reduction_parameters(self.key),
        )


@dataclass(frozen=True)
class PCAArtifacts:
    """Typed PCA-only values, retained independently when imported."""

    explained_variance_ratio: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    loadings: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.explained_variance_ratio is None and self.loadings is None:
            raise ValueError("PCA artifacts require loadings or explained variance")
        variance = (
            None
            if self.explained_variance_ratio is None
            else _owned_read_only_array(self.explained_variance_ratio)
        )
        loadings = (
            None if self.loadings is None else _owned_read_only_array(self.loadings)
        )
        if variance is not None and variance.ndim != 1:
            raise ValueError("PCA explained variance must be one-dimensional")
        if loadings is not None and loadings.ndim != 2:
            raise ValueError("PCA loadings must be two-dimensional")
        if (
            loadings is not None
            and variance is not None
            and loadings.shape[0] != len(variance)
        ):
            raise ValueError("PCA variance and loading component counts must match")
        object.__setattr__(self, "explained_variance_ratio", variance)
        object.__setattr__(self, "loadings", loadings)

    @property
    def has_loadings(self) -> bool:
        return self.loadings is not None

    @property
    def has_explained_variance(self) -> bool:
        return self.explained_variance_ratio is not None


@dataclass(frozen=True)
class ReductionArtifact:
    """Portable immutable reduction coordinates and typed method artifacts."""

    metadata: AnalysisArtifactMetadata
    key: ReductionKey
    coordinates: np.ndarray = field(repr=False, compare=False)
    sample_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    source: Literal["computed", "uploaded", "h5ad"] = "computed"
    data_source_name: str = "data"
    n_variable_features_used: int = 0
    n_components_retained: int = 0
    pca: PCAArtifacts | None = None
    params: tuple[tuple[str, object], ...] = ()

    @property
    def was_computed(self) -> bool:
        """Whether Metabis fitted these coordinates, or imported them (R-02).

        Every surface that reports where a reduction came from -- the card's
        provenance line, the exported table's header -- reads this rather than
        comparing the string itself, so "imported" cannot end up meaning
        different things in two places.
        """
        return self.source == "computed"

    def __post_init__(self) -> None:
        if self.metadata.kind is not ArtifactKind.REDUCTION:
            raise ValueError("Reduction artifact metadata must have reduction kind")
        if self.metadata.method != self.key.method:
            raise ValueError("Reduction artifact method must match its key")
        identity = self.metadata.scientific_identity
        expected_identity = {
            "sample_ids": self.sample_ids,
            "feature_ids": self.feature_names,
            "source": self.source,
        }
        for name, expected in expected_identity.items():
            if identity.get(name) != expected:
                raise ValueError(
                    f"Reduction metadata scientific identity does not match {name}"
                )
        expected_parameters = {
            "n_components": self.key.n_components,
            "n_variable_features": self.key.n_variable_features,
            "random_state": self.key.random_state,
            **(
                {"n_neighbors": self.key.n_neighbors}
                if self.key.n_neighbors is not None
                else {}
            ),
            **(
                {"min_dist": self.key.min_dist}
                if self.key.min_dist is not None
                else {}
            ),
            **(
                {"perplexity": self.key.perplexity}
                if self.key.perplexity is not None
                else {}
            ),
        }
        expected_configuration = {
            "data_type": self.key.data_type,
            "parameters": expected_parameters,
            "preprocessing_revision": self.key.preprocessing_revision,
            "sample_revision": self.key.sample_revision,
        }
        if dict(self.metadata.configuration) != expected_configuration:
            raise ValueError("Reduction metadata configuration does not match its key")
        if not isinstance(self.sample_ids, tuple) or not all(
            isinstance(sample_id, str) for sample_id in self.sample_ids
        ):
            raise TypeError("Reduction sample IDs must be a tuple of strings")
        if not isinstance(self.feature_names, tuple) or not all(
            isinstance(feature_id, str) for feature_id in self.feature_names
        ):
            raise TypeError("Reduction feature names must be a tuple of strings")
        coordinates = _owned_read_only_array(self.coordinates)
        if coordinates.ndim != 2:
            raise ValueError("Reduction coordinates must be two-dimensional")
        if coordinates.shape[0] != len(self.sample_ids):
            raise ValueError("Reduction coordinates must align with sample IDs")
        if coordinates.shape[1] != self.key.n_components:
            raise ValueError("Reduction coordinates must match key component count")
        if self.n_components_retained != coordinates.shape[1]:
            raise ValueError("Retained component count must match coordinates")
        if self.n_variable_features_used != len(self.feature_names):
            raise ValueError("Used feature count must match feature names")
        if self.key.method == "PCA" and self.source != "h5ad":
            if (
                self.pca is None
                or not self.pca.has_loadings
                or not self.pca.has_explained_variance
            ):
                raise ValueError(
                    "Computed PCA reductions require variance and loading artifacts"
                )
        if self.key.method != "PCA" and self.pca is not None:
            raise ValueError("Only PCA reductions may carry PCA artifacts")
        if self.pca is not None and self.pca.loadings is not None:
            if self.pca.loadings.shape != (
                coordinates.shape[1],
                len(self.feature_names),
            ):
                raise ValueError("PCA loadings must align with components and features")
        dimensions = {
            dimension.name: dimension.size for dimension in self.metadata.dimensions
        }
        if dimensions != {
            "samples": len(self.sample_ids),
            "features": len(self.feature_names),
            "components": coordinates.shape[1],
        }:
            raise ValueError("Reduction artifact dimensions do not match its payload")
        parameter_names = tuple(name for name, _value in self.params)
        if parameter_names != tuple(sorted(parameter_names)) or len(
            parameter_names
        ) != len(set(parameter_names)):
            raise ValueError("Reduction parameters must be sorted by name")
        if not all(
            isinstance(name, str)
            and isinstance(value, (str, bool, int, float, type(None)))
            for name, value in self.params
        ):
            raise TypeError("Reduction parameters must contain immutable scalar values")
        if dict(self.params) != expected_parameters:
            raise ValueError("Reduction parameters do not match its key")
        object.__setattr__(self, "coordinates", coordinates)

    def to_compute_result(self) -> ReductionResult:
        """Project the artifact into the calculator's typed cache value."""
        compute_metadata: dict[str, Any] = {
            "data_source_name": self.data_source_name,
            "n_variable_features_used": self.n_variable_features_used,
            "n_components_retained": self.n_components_retained,
        }
        if self.pca is not None:
            if self.pca.explained_variance_ratio is not None:
                compute_metadata["explained_variance_ratio"] = (
                    self.pca.explained_variance_ratio
                )
            if self.pca.loadings is not None:
                compute_metadata["components"] = self.pca.loadings
        if self.params:
            compute_metadata["params"] = dict(self.params)
        return ReductionResult(
            key=self.key,
            transformed_data=self.coordinates,
            sample_ids=self.sample_ids,
            feature_names=self.feature_names,
            source=self.source,
            metadata=compute_metadata,
        )


def reduction_coordinates_frame(artifact: ReductionArtifact) -> pd.DataFrame:
    """Return canonical sample coordinates with full numeric precision."""
    frame = pd.DataFrame(
        artifact.coordinates,
        columns=[f"component_{index}" for index in range(1, artifact.coordinates.shape[1] + 1)],
    )
    frame.insert(0, "sample_id", artifact.sample_ids)
    return frame


def pca_loadings_frame(artifact: ReductionArtifact) -> pd.DataFrame:
    """Return canonical feature-by-component PCA loadings."""
    if artifact.pca is None or artifact.pca.loadings is None:
        raise ValueError("PCA loadings are unavailable for this reduction")
    frame = pd.DataFrame(
        artifact.pca.loadings.T,
        columns=[f"component_{index}" for index in range(1, artifact.pca.loadings.shape[0] + 1)],
    )
    frame.insert(0, "feature_id", artifact.feature_names)
    return frame


def pca_variance_frame(artifact: ReductionArtifact) -> pd.DataFrame:
    """Return canonical PCA explained and cumulative variance."""
    if artifact.pca is None or artifact.pca.explained_variance_ratio is None:
        raise ValueError("PCA variance is unavailable for this reduction")
    variance = artifact.pca.explained_variance_ratio
    return pd.DataFrame(
        {
            "component": np.arange(1, len(variance) + 1, dtype=int),
            "explained_variance_ratio": variance,
            "cumulative_explained_variance_ratio": np.cumsum(variance),
        }
    )


def _owned_read_only_array(values: Any) -> np.ndarray:
    array = np.array(values, copy=True)
    array.setflags(write=False)
    return array


class DimensionalityReduction:
    """Handles dimensionality reduction operations (PCA, t-SNE, UMAP) for the analysis module.
    Stores results internally.
    """

    def __init__(self):
        self._results: dict[ReductionKey, ReductionResult] = {}

    def clear_results(self):
        """Clear all stored results."""
        self._results.clear()

    @property
    def results(self) -> Mapping[ReductionKey, ReductionResult]:
        """Return a read-only view of the parameterized compute-result cache."""
        return MappingProxyType(self._results)

    def cache_result(self, result: ReductionResult) -> ReductionResult:
        """Cache one typed compute result under its complete reduction key.

        Replacing an existing key moves it to the newest position, preserving
        the most-recently-installed lookup semantics without exposing a mutable
        dictionary to callers.
        """
        if not isinstance(result, ReductionResult):
            raise TypeError("result must be a ReductionResult")
        self._results.pop(result.key, None)
        self._results[result.key] = result
        return result

    def get_result(self, key: ReductionKey) -> ReductionResult | None:
        """Return the exact cached result for ``key``."""
        return self._results.get(key)

    def _reuse(self, key: ReductionKey) -> dict | None:
        cached = self._results.get(key)
        if cached is None:
            return None
        return {
            "status": "success",
            "messages": ["Reused matching dimensionality-reduction result."],
            "key": key,
            "reused": True,
            "result": cached,
        }

    def _prepare_data(
        self,
        matrix: "ExpressionMatrix",
        n_variable_features: int,
        data_source_name: str,
    ) -> Dict[str, Any]:
        """Prepare data for dimension reduction by selecting variable features.

        Reads the data through the expression-matrix seam (AM6 slice 5): the
        variance ranking is sparse-native and only the selected top-N features
        are materialized dense, so a sparse expression load never densifies the
        whole matrix to run a reduction. The dense adapter reproduces the prior
        ``df.var()`` / ``df[cols].values`` behaviour exactly, so the dense path
        (reactions, and a dense CSV load) is unchanged.

        Args:
            matrix: The expression-matrix seam over the data (samples x features).
            n_variable_features: Number of most variable features to select.
            data_source_name: Name of the data source for logging (e.g. "reaction data").

        Returns:
            Dict with keys: 'status', 'messages', 'data', 'sample_ids', 'feature_names', 'data_source_name'
        """
        results: Dict[str, Any] = {"status": "success", "messages": []}

        if matrix is None:
            return {
                "status": "error",
                "messages": [f"No data available for {data_source_name}."],
            }

        sample_ids = list(matrix.sample_ids)
        feature_ids = list(matrix.feature_ids)

        # Matches the former ``df.empty`` guard: no rows or no columns is no data.
        if not sample_ids or not feature_ids:
            return {
                "status": "error",
                "messages": [f"No data available for {data_source_name}."],
            }

        # Select most variable features -- ranking is sparse-native at the seam.
        variances = matrix.feature_variances().dropna()

        if n_variable_features <= 0 or n_variable_features >= len(variances):
            top_features = list(feature_ids)
            if 0 < n_variable_features < len(variances):
                results["messages"].append(
                    f"Using all {len(top_features)} available features as n_variable_features ({n_variable_features}) is less than total or 0."
                )
            elif n_variable_features == 0:
                results["messages"].append(
                    f"Using all {len(top_features)} available features as n_variable_features is 0."
                )
        elif variances.empty or (variances == 0).all():
            results["messages"].append(
                "Could not determine variable features (variances are zero or NaN). Using all features."
            )
            top_features = list(feature_ids)
        else:
            num_features_to_select = min(n_variable_features, len(variances))
            top_features = variances.nlargest(num_features_to_select).index.tolist()
            results["messages"].append(
                f"Selected top {len(top_features)} variable features."
            )

        # Materialize only the selected columns dense -- the bounded block the
        # sklearn estimators need, never the whole matrix.
        data_matrix = np.nan_to_num(matrix.select_features(top_features).values)

        if data_matrix.shape[0] < 2:
            return {
                "status": "error",
                "messages": [
                    f"Not enough samples ({data_matrix.shape[0]}) for reduction. Need at least 2."
                ],
            }

        results.update(
            {
                "data": data_matrix,
                "sample_ids": sample_ids,
                "feature_names": top_features,
                "data_source_name": data_source_name,
            }
        )
        return results

    def calculate_pca(
        self,
        matrix: ExpressionMatrix,
        data_type: str,
        data_source_name: str,
        n_components: Optional[int] = None,
        n_variable_features: int = 500,
        random_state: int = 42,
        preprocessing_revision: str = "0",
        sample_revision: str = "0",
        force_recompute: bool = False,
    ) -> Dict[str, Any]:
        """Calculates PCA for the specified dataframe and stores the results.

        Args:
            matrix: The expression-matrix seam over the data to analyze.
            data_type: Key to store results under (e.g. 'reactions', 'expression').
            data_source_name: Display name for the data source.
            n_components: Number of principal components to keep.
            n_variable_features: Number of most variable features to use.

        Returns:
            Dict[str, Any]: Results dictionary with status and messages.
        """
        prep_result = self._prepare_data(matrix, n_variable_features, data_source_name)
        if prep_result["status"] != "success":
            return prep_result

        data_matrix = prep_result["data"]
        sample_ids = prep_result["sample_ids"]
        top_features = prep_result["feature_names"]
        results = {"status": "success", "messages": prep_result["messages"]}

        effective_n_components = n_components
        max_possible_components = min(data_matrix.shape)
        if effective_n_components is None:
            effective_n_components = max_possible_components
        elif effective_n_components > max_possible_components:
            effective_n_components = max_possible_components
            results["messages"].append(
                f"Adjusted n_components to {effective_n_components} (min of samples, features)."
            )

        key = ReductionKey(
            method="PCA",
            data_type=data_type,
            n_variable_features=n_variable_features,
            n_components=effective_n_components,
            random_state=random_state,
            preprocessing_revision=str(preprocessing_revision),
            sample_revision=str(sample_revision),
        )
        if not force_recompute and (cached := self._reuse(key)):
            return cached

        try:
            pca_model = PCA(
                n_components=effective_n_components, random_state=random_state
            )
            transformed_data = pca_model.fit_transform(data_matrix)

            metadata = {
                "pca_model": pca_model,
                "explained_variance_ratio": pca_model.explained_variance_ratio_,
                "components": pca_model.components_,  # Loadings: (n_components, n_features)
                "n_variable_features_used": len(top_features),
                "data_source_name": data_source_name,
                "n_components_retained": pca_model.n_components_,
            }
            reduction_result = ReductionResult(
                key=key,
                transformed_data=transformed_data,
                sample_ids=tuple(sample_ids),
                feature_names=tuple(top_features),
                metadata=metadata,
            )
            self.cache_result(reduction_result)
            results.update(key=key, reused=False, result=reduction_result)
            results["messages"].append(
                f"PCA calculated for {data_source_name} using {len(top_features)} features and {transformed_data.shape[0]} samples."
            )
            results["messages"].append(
                f"Stored {pca_model.n_components_} principal components."
            )

        except Exception as e:
            results["status"] = "error"
            results["messages"].append(f"Error during PCA computation: {str(e)}")

        return results

    def calculate_tsne(
        self,
        matrix: ExpressionMatrix,
        data_type: str,
        data_source_name: str,
        n_components: int = 2,
        perplexity: float = 30.0,
        n_variable_features: int = 500,
        random_state: int = 42,
        preprocessing_revision: str = "0",
        sample_revision: str = "0",
        force_recompute: bool = False,
    ) -> Dict[str, Any]:
        """Calculates t-SNE for the specified dataframe and stores the results."""
        prep_result = self._prepare_data(matrix, n_variable_features, data_source_name)
        if prep_result["status"] != "success":
            return prep_result

        data_matrix = prep_result["data"]
        sample_ids = prep_result["sample_ids"]
        top_features = prep_result["feature_names"]
        results = {"status": "success", "messages": prep_result["messages"]}

        # Adjust perplexity if needed
        n_samples = data_matrix.shape[0]
        if n_samples - 1 < perplexity:
            perplexity = max(1, n_samples - 1)
            results["messages"].append(
                f"Adjusted perplexity to {perplexity} due to small sample size ({n_samples})."
            )

        key = ReductionKey(
            method="T-SNE",
            data_type=data_type,
            n_variable_features=n_variable_features,
            n_components=n_components,
            random_state=random_state,
            preprocessing_revision=str(preprocessing_revision),
            sample_revision=str(sample_revision),
            perplexity=perplexity,
        )
        if not force_recompute and (cached := self._reuse(key)):
            return cached

        try:
            tsne_model = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
                init="pca",
                learning_rate="auto",
            )
            transformed_data = tsne_model.fit_transform(data_matrix)

            metadata = {
                "n_variable_features_used": len(top_features),
                "data_source_name": data_source_name,
                "params": {
                    "n_components": n_components,
                    "perplexity": perplexity,
                    "random_state": random_state,
                },
            }
            reduction_result = ReductionResult(
                key=key,
                transformed_data=transformed_data,
                sample_ids=tuple(sample_ids),
                feature_names=tuple(top_features),
                metadata=metadata,
            )
            self.cache_result(reduction_result)
            results.update(key=key, reused=False, result=reduction_result)
            results["messages"].append(
                f"t-SNE calculated for {data_source_name} using {len(top_features)} features."
            )

        except Exception as e:
            results["status"] = "error"
            results["messages"].append(f"Error during t-SNE computation: {str(e)}")

        return results

    def calculate_umap(
        self,
        matrix: ExpressionMatrix,
        data_type: str,
        data_source_name: str,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_variable_features: int = 500,
        random_state: int = 42,
        preprocessing_revision: str = "0",
        sample_revision: str = "0",
        force_recompute: bool = False,
    ) -> Dict[str, Any]:
        """Calculates UMAP for the specified dataframe and stores the results."""
        prep_result = self._prepare_data(matrix, n_variable_features, data_source_name)
        if prep_result["status"] != "success":
            return prep_result

        data_matrix = prep_result["data"]
        sample_ids = prep_result["sample_ids"]
        top_features = prep_result["feature_names"]
        results = {"status": "success", "messages": prep_result["messages"]}

        # Adjust n_neighbors if needed
        n_samples = data_matrix.shape[0]
        if n_samples <= n_neighbors:
            n_neighbors = max(2, n_samples - 1)
            results["messages"].append(
                f"Adjusted n_neighbors to {n_neighbors} due to small sample size ({n_samples})."
            )

        key = ReductionKey(
            method="UMAP",
            data_type=data_type,
            n_variable_features=n_variable_features,
            n_components=n_components,
            random_state=random_state,
            preprocessing_revision=str(preprocessing_revision),
            sample_revision=str(sample_revision),
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
        if not force_recompute and (cached := self._reuse(key)):
            return cached

        try:
            umap_model = UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
            )
            transformed_data = umap_model.fit_transform(data_matrix)

            metadata = {
                "n_variable_features_used": len(top_features),
                "data_source_name": data_source_name,
                "params": {
                    "n_components": n_components,
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "random_state": random_state,
                },
            }
            reduction_result = ReductionResult(
                key=key,
                transformed_data=transformed_data,
                sample_ids=tuple(sample_ids),
                feature_names=tuple(top_features),
                metadata=metadata,
            )
            self.cache_result(reduction_result)
            results.update(key=key, reused=False, result=reduction_result)
            results["messages"].append(
                f"UMAP calculated for {data_source_name} using {len(top_features)} features."
            )

        except Exception as e:
            results["status"] = "error"
            results["messages"].append(f"Error during UMAP computation: {str(e)}")

        return results

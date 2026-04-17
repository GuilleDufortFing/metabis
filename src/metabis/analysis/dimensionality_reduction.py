from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

class DimensionalityReduction:
    """
    Handles dimensionality reduction operations (PCA, t-SNE, UMAP) for the analysis module.
    Stores results internally.
    """

    def __init__(self):
        self.pca_results = {}
        self.tsne_results = {}
        self.umap_results = {}

    def clear_results(self):
        """Clear all stored results."""
        self.pca_results.clear()
        self.tsne_results.clear()
        self.umap_results.clear()

    def _prepare_data(
        self, 
        df: pd.DataFrame, 
        n_variable_features: int, 
        data_source_name: str
    ) -> Dict[str, Any]:
        """
        Prepare data for dimension reduction by selecting variable features.
        
        Args:
            df: The dataframe containing the data (samples x features).
            n_variable_features: Number of most variable features to select.
            data_source_name: Name of the data source for logging (e.g. "reaction data").

        Returns:
            Dict with keys: 'status', 'messages', 'data', 'sample_ids', 'feature_names', 'data_source_name'
        """
        results = {"status": "success", "messages": []}

        if df is None or df.empty:
            return {
                "status": "error",
                "messages": [f"No data available for {data_source_name}."],
            }

        current_df = df.copy()
        sample_ids = current_df.index.tolist()

        # Select most variable features
        if current_df.shape[1] == 0:
            return {
                "status": "error",
                "messages": ["Dataframe has no features (columns)."],
            }

        variances = current_df.var(numeric_only=True).dropna()

        if n_variable_features <= 0 or n_variable_features >= len(variances):
            top_features = current_df.columns.tolist()
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
            top_features = current_df.columns.tolist()
        else:
            num_features_to_select = min(n_variable_features, len(variances))
            top_features = variances.nlargest(num_features_to_select).index.tolist()
            results["messages"].append(
                f"Selected top {len(top_features)} variable features."
            )

        data_matrix = current_df[top_features].values
        data_matrix = np.nan_to_num(data_matrix)

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
        df: pd.DataFrame,
        data_type: str,
        data_source_name: str,
        n_components: Optional[int] = None,
        n_variable_features: int = 500,
    ) -> Dict[str, Any]:
        """
        Calculates PCA for the specified dataframe and stores the results.

        Args:
            df: The dataframe to analyze.
            data_type: Key to store results under (e.g. 'reactions', 'expression').
            data_source_name: Display name for the data source.
            n_components: Number of principal components to keep.
            n_variable_features: Number of most variable features to use.

        Returns:
            Dict[str, Any]: Results dictionary with status and messages.
        """
        prep_result = self._prepare_data(df, n_variable_features, data_source_name)
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

        try:
            pca_model = PCA(n_components=effective_n_components, random_state=42)
            transformed_data = pca_model.fit_transform(data_matrix)

            self.pca_results[data_type] = {
                "pca_model": pca_model,
                "transformed_data": transformed_data,  # Scores
                "explained_variance_ratio": pca_model.explained_variance_ratio_,
                "components": pca_model.components_,  # Loadings: (n_components, n_features)
                "feature_names": top_features,  # Features used for PCA
                "sample_ids": sample_ids,  # Sample IDs corresponding to transformed_data rows
                "n_variable_features_used": len(top_features),
                "data_source_name": data_source_name,
                "n_components_retained": pca_model.n_components_,
            }
            results["messages"].append(
                f"PCA calculated for {data_source_name} using {len(top_features)} features and {transformed_data.shape[0]} samples."
            )
            results["messages"].append(
                f"Stored {pca_model.n_components_} principal components."
            )

        except Exception as e:
            results["status"] = "error"
            results["messages"].append(f"Error during PCA computation: {str(e)}")
            if data_type in self.pca_results:  # Clear partial results on error
                del self.pca_results[data_type]

        return results

    def calculate_tsne(
        self,
        df: pd.DataFrame,
        data_type: str,
        data_source_name: str,
        n_components: int = 2,
        perplexity: float = 30.0,
        n_variable_features: int = 500,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Calculates t-SNE for the specified dataframe and stores the results."""
        prep_result = self._prepare_data(df, n_variable_features, data_source_name)
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

        try:
            tsne_model = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
                init="pca",
                learning_rate="auto",
            )
            transformed_data = tsne_model.fit_transform(data_matrix)

            self.tsne_results[data_type] = {
                "transformed_data": transformed_data,
                "feature_names": top_features,
                "sample_ids": sample_ids,
                "n_variable_features_used": len(top_features),
                "data_source_name": data_source_name,
                "params": {
                    "n_components": n_components,
                    "perplexity": perplexity,
                    "random_state": random_state,
                },
            }
            results["messages"].append(
                f"t-SNE calculated for {data_source_name} using {len(top_features)} features."
            )

        except Exception as e:
            results["status"] = "error"
            results["messages"].append(f"Error during t-SNE computation: {str(e)}")
            if data_type in self.tsne_results:
                del self.tsne_results[data_type]

        return results

    def calculate_umap(
        self,
        df: pd.DataFrame,
        data_type: str,
        data_source_name: str,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_variable_features: int = 500,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Calculates UMAP for the specified dataframe and stores the results."""
        prep_result = self._prepare_data(df, n_variable_features, data_source_name)
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

        try:
            umap_model = UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
            )
            transformed_data = umap_model.fit_transform(data_matrix)

            self.umap_results[data_type] = {
                "transformed_data": transformed_data,
                "feature_names": top_features,
                "sample_ids": sample_ids,
                "n_variable_features_used": len(top_features),
                "data_source_name": data_source_name,
                "params": {
                    "n_components": n_components,
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "random_state": random_state,
                },
            }
            results["messages"].append(
                f"UMAP calculated for {data_source_name} using {len(top_features)} features."
            )

        except Exception as e:
            results["status"] = "error"
            results["messages"].append(f"Error during UMAP computation: {str(e)}")
            if data_type in self.umap_results:
                del self.umap_results[data_type]

        return results

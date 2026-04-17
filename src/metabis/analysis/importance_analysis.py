import random
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


class ImportanceAnalysis:
    """A class to handle all importance analysis operations for gene-reaction influence studies.

    This class encapsulates the entire workflow for analyzing how gene expression changes
    influence reaction activity changes, including model selection, feature importance
    calculation, and result formatting.
    """

    def __init__(
        self,
        model_type: str = "lasso",
        importance_type: str = "partial_r2",
        alpha: float = 0.1,
        max_iter: int = 10000,
        min_coef: float = 0.001,
        max_sample_pairs: int = 1000,
        translate_gene_func: Optional[Callable] = None,
        **config,
    ):
        """Initialize importance analysis with configuration parameters.

        Args:
            model_type (str): Type of regression model ('lasso' or 'xgboost')
            importance_type (str): Method for calculating importance ('partial_r2' or 'shap')
            alpha (float): Regularization strength
            max_iter (int): Maximum iterations for model fitting
            min_coef (float): Minimum coefficient threshold for significance
            max_sample_pairs (int): Maximum number of sample pairs to analyze
            translate_gene_func (callable, optional): Function to translate gene IDs to symbols
            **config: Additional configuration parameters
        """
        self.model_type = model_type
        self.importance_type = importance_type
        self.alpha = alpha
        self.max_iter = max_iter
        self.min_coef = min_coef
        self.max_sample_pairs = max_sample_pairs
        self.translate_gene_func = translate_gene_func

        # Additional configuration with defaults
        self.config = {
            "early_stop_r2_threshold": 0.5,
            "early_stop_delta_threshold": 0.01,
            "min_valid_pairs": 10,
            "alpha_values": [0.1, 0.01, 0.001, 0.0001, 1e-5],
            **config,
        }

        # Analysis state
        self.last_analysis_results = None
        self.analysis_summary = {}

    def configure_model(self, model_type: str, alpha: float, max_iter: int, **params):
        """Configure model parameters.

        Args:
            model_type (str): Type of regression model
            alpha (float): Regularization strength
            max_iter (int): Maximum iterations
            **params: Additional model parameters
        """
        self.model_type = model_type
        self.alpha = alpha
        self.max_iter = max_iter
        self.config.update(params)

    def configure_analysis(
        self, importance_type: str, min_coef: float, max_sample_pairs: int, **params
    ):
        """Configure analysis parameters.

        Args:
            importance_type (str): Method for calculating importance
            min_coef (float): Minimum coefficient threshold
            max_sample_pairs (int): Maximum sample pairs to analyze
            **params: Additional analysis parameters
        """
        self.importance_type = importance_type
        self.min_coef = min_coef
        self.max_sample_pairs = max_sample_pairs
        self.config.update(params)

    def calculate_pairwise_deltas(
        self,
        gene_df: pd.DataFrame,
        reaction_df: pd.DataFrame,
        reaction_id: str,
        sample_pairs: List[Tuple[str, str]],
    ) -> Tuple[List, List, List]:
        """Calculate gene and reaction deltas for sample pairs.

        Args:
            gene_df (pd.DataFrame): Gene expression data
            reaction_df (pd.DataFrame): Reaction activity data
            reaction_id (str): ID of the reaction to analyze
            sample_pairs (List[Tuple]): List of (sample1, sample2) pairs

        Returns:
            Tuple[List, List, List]: (gene_deltas, reaction_deltas, reaction_abs_deltas)
        """
        gene_deltas = []
        reaction_deltas = []
        reaction_abs_deltas = []

        for s1, s2 in sample_pairs:
            # Skip if either sample is missing in either dataset
            if (
                s1 not in gene_df.index
                or s2 not in gene_df.index
                or s1 not in reaction_df.index
                or s2 not in reaction_df.index
            ):
                continue

            # Calculate gene expression deltas for this pair
            gene_delta = gene_df.loc[s2] - gene_df.loc[s1]
            gene_deltas.append(gene_delta)

            # Calculate reaction activity delta for this pair
            reaction_delta = (
                reaction_df.loc[s2, reaction_id] - reaction_df.loc[s1, reaction_id]
            )
            reaction_deltas.append(reaction_delta)

            # Calculate absolute reaction activity delta for this pair
            reaction_abs_delta = abs(reaction_df.loc[s2, reaction_id]) - abs(
                reaction_df.loc[s1, reaction_id]
            )
            reaction_abs_deltas.append(reaction_abs_delta)

        return gene_deltas, reaction_deltas, reaction_abs_deltas

    def validate_analysis_data(
        self, gene_deltas: List, min_pairs: Optional[int] = None
    ) -> bool:
        """Validate that we have sufficient data for analysis.

        Args:
            gene_deltas (List): List of gene delta arrays
            min_pairs (int, optional): Minimum required pairs

        Returns:
            bool: True if data is sufficient for analysis
        """
        min_required = min_pairs or self.config.get("min_valid_pairs", 10)
        return len(gene_deltas) >= min_required

    def select_best_model(
        self, X: pd.DataFrame, y: pd.Series, alpha_values: Optional[List[float]] = None
    ) -> Tuple[Any, Dict]:
        """Select the best model configuration based on R² score with early stopping.

        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector
            alpha_values (List[float], optional): Alpha values to search

        Returns:
            Tuple[Any, Dict]: (best_model, best_params)
        """
        alpha_values = alpha_values or self.config.get(
            "alpha_values", [0.1, 0.01, 0.001, 0.0001, 1e-5]
        )
        early_stop_r2_threshold = self.config.get("early_stop_r2_threshold", 0.5)
        early_stop_delta_threshold = self.config.get("early_stop_delta_threshold", 0.01)

        print(f"Selecting best hyperparameter for model type: {self.model_type}")

        best_model = None
        best_r2 = -np.inf
        best_params = {}
        last_model = None
        last_params = {}
        previous_best_r2 = -np.inf

        for i, current_alpha in enumerate(alpha_values):
            # Create model parameters
            if self.model_type.lower() == "lasso":
                model_params = {
                    "alpha": current_alpha,
                    "max_iter": self.max_iter,
                    "random_state": 42,
                    "fit_intercept": True,
                }
                param_name = "alpha"
            elif self.model_type.lower() == "xgboost":
                model_params = {
                    "reg_alpha": current_alpha,
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "random_state": 42,
                    "verbosity": 0,
                }
                param_name = "reg_alpha"
            else:
                model_params = {"alpha": current_alpha, "random_state": 42}
                param_name = "alpha"

            # Create and fit model
            try:
                model = ModelFactory.create_model(self.model_type, **model_params)
                model.fit(X, y)
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)

                if np.isnan(r2):
                    print(
                        f"  Warning: R² is NaN for {param_name}={current_alpha}. Treating as -inf."
                    )
                    r2 = -np.inf
            except Exception as e:
                print(f"  Error fitting model with {param_name}={current_alpha}: {e}")
                r2 = -np.inf

            print(f"  {param_name}={current_alpha}: R²={r2:.4f}")

            # Early stopping logic
            if r2 > early_stop_r2_threshold:
                print(
                    f"  Early stopping: R² ({r2:.4f}) exceeded threshold ({early_stop_r2_threshold})."
                )
                if r2 > best_r2:
                    best_model = model
                    best_r2 = r2
                    best_params = {param_name: current_alpha}
                break

            if i > 0 and r2 > 0 and previous_best_r2 > -np.inf:
                delta_r2 = r2 - previous_best_r2
                if delta_r2 < early_stop_delta_threshold:
                    print(
                        f"  Early stopping: R² improvement ({delta_r2:.4f}) is less than threshold ({early_stop_delta_threshold})."
                    )
                    break

            # Update best model
            previous_best_r2 = best_r2
            if r2 > best_r2:
                best_model = model
                best_r2 = r2
                best_params = {param_name: current_alpha}

            # Store last model for fallback
            if current_alpha == alpha_values[-1]:
                last_model = model
                last_params = {param_name: current_alpha}

        # Fallback logic
        if best_model is None or best_r2 == -np.inf:
            print(
                f"  Warning: Could not find a model with valid R2 > -inf. Trying last alpha: {alpha_values[-1]}"
            )
            if last_model is not None:
                best_model = last_model
                best_params = last_params
                try:
                    best_r2 = r2_score(y, best_model.predict(X))
                    if np.isnan(best_r2):
                        best_r2 = -np.inf
                except Exception:
                    best_r2 = -np.inf
                print(
                    f"  Using fallback model with {list(best_params.keys())[0]}={list(best_params.values())[0]}, R2={best_r2:.4f}"
                )
            else:
                print("  Error: Failed to fit any model successfully.")
                return None, {}

        elif best_r2 <= 0:
            print(
                f"  Warning: Best model R2 ({best_r2:.4f}) is non-positive. Model may not be predictive."
            )

        print(
            f"  Selected best model with {list(best_params.keys())[0]}={list(best_params.values())[0]}, R2={best_r2:.4f}"
        )
        return best_model, best_params

    def calculate_model_metrics(
        self, X: pd.DataFrame, y: pd.Series, model: Any
    ) -> Dict[str, float]:
        """Calculate performance metrics for the model.

        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector
            model: Fitted model

        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        corr = np.corrcoef(y, y_pred)[0, 1] if len(y) > 1 else 0

        return {"r2_score": float(r2), "rmse": float(rmse), "correlation": float(corr)}

    def calculate_feature_importance(
        self, X: pd.DataFrame, y: pd.Series, model: Any, full_r2: float
    ) -> Tuple[Dict, Any]:
        """Calculate feature importance using the configured method.

        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector
            model: Fitted model
            full_r2 (float): R² of the full model

        Returns:
            Tuple[Dict, Any]: (importance_scores, raw_shap_values)
        """
        return ImportanceCalculator.calculate_importance(
            self.importance_type,
            X,
            y,
            model,
            full_r2,
            model_type=self.model_type,
            min_coef=self.min_coef,
            max_iter=self.max_iter,
        )

    def _calculate_gene_reaction_correlations(
        self, X: pd.DataFrame, y_abs: pd.Series
    ) -> Dict[str, float]:
        """Calculate pairwise gene-reaction correlations.

        Args:
            X (pd.DataFrame): Gene delta matrix
            y_abs (pd.Series): Absolute reaction deltas

        Returns:
            Dict[str, float]: Gene-reaction correlations
        """
        gene_reaction_correlations = {}

        if not y_abs.isnull().all() and y_abs.var() > 1e-9:
            for gene in X.columns:
                if not X[gene].isnull().all() and X[gene].var() > 1e-9:
                    try:
                        corr_val = X[gene].corr(y_abs)
                        gene_reaction_correlations[gene] = (
                            corr_val if not pd.isna(corr_val) else 0.0
                        )
                    except Exception as corr_err:
                        print(
                            f"  Warning: Could not calculate correlation for gene {gene}: {corr_err}"
                        )
                        gene_reaction_correlations[gene] = 0.0
                else:
                    gene_reaction_correlations[gene] = 0.0
        else:
            gene_reaction_correlations = {gene: 0.0 for gene in X.columns}

        return gene_reaction_correlations

    def format_reaction_results(
        self,
        reaction_id: str,
        importance_scores: Dict,
        metrics: Dict,
        model_params: Dict,
        gene_reaction_correlations: Dict,
        n_pairs: int = 0,
        raw_shap_values: Any = None,
    ) -> Dict:
        """Format analysis results into a standardized structure.

        Args:
            reaction_id (str): ID of the reaction
            importance_scores (Dict): Feature importance scores
            metrics (Dict): Model performance metrics
            model_params (Dict): Best model parameters
            gene_reaction_correlations (Dict): Gene-reaction correlations
            n_pairs (int): Number of sample pairs used
            raw_shap_values: Raw SHAP values (if applicable)

        Returns:
            Dict: Formatted analysis results
        """
        genes_ranked = []

        # Populate genes_ranked from the filtered importance_scores
        if self.importance_type.lower() == "partial_r2":
            for gene, scores in importance_scores.items():
                gene_react_corr = gene_reaction_correlations.get(gene, 0.0)
                genes_ranked.append(
                    (
                        gene,
                        scores["coefficient"],
                        scores["partial_r2"],
                        scores["pct_variance"],
                        gene_react_corr,
                    )
                )
            genes_ranked.sort(key=lambda x: x[2], reverse=True)

        elif self.importance_type.lower() == "shap":
            for gene, scores in importance_scores.items():
                gene_react_corr = gene_reaction_correlations.get(gene, 0.0)
                genes_ranked.append(
                    (
                        gene,
                        scores["coefficient"],
                        scores["shap_value"],
                        scores["pct_variance"],
                        gene_react_corr,
                    )
                )
            genes_ranked.sort(key=lambda x: x[2], reverse=True)

        # Add gene symbols if translation function is provided
        genes_with_symbols = []
        if self.translate_gene_func and genes_ranked:
            for (
                gene_id,
                coef_or_imp,
                importance_val,
                pct_variance,
                gene_react_corr,
            ) in genes_ranked:
                try:
                    gene_symbol = self.translate_gene_func(
                        gene_id,
                        input_format="ensmbl_id",
                        output_format="geneSymbols",
                    )
                    if gene_symbol is None:
                        gene_symbol = gene_id
                except Exception as e:
                    print(f"  Warning: Error translating gene ID {gene_id}: {e}")
                    gene_symbol = gene_id

                genes_with_symbols.append(
                    (
                        gene_id,
                        gene_symbol,
                        coef_or_imp,
                        importance_val,
                        pct_variance,
                        gene_react_corr,
                    )
                )

        # Determine the regularization parameter name used
        reg_param = list(model_params.keys())[0] if model_params else "unknown_param"
        reg_value = list(model_params.values())[0] if model_params else None

        print(
            f"Final model for {reaction_id}: {reg_param}={reg_value}, "
            f"R²={metrics['r2_score']:.3f}, significant genes: {len(genes_ranked)}"
        )

        result_dict = {
            "reaction_id": reaction_id,
            "genes_ranked": genes_ranked,
            "genes_with_symbols": genes_with_symbols
            if self.translate_gene_func
            else None,
            "r2_score": metrics["r2_score"],
            "rmse": metrics["rmse"],
            "correlation": metrics["correlation"],
            "n_pairs": n_pairs,
            "significant_genes_count": len(genes_ranked),
            "importance_type": self.importance_type,
            f"{reg_param}_used": reg_value,
        }

        if self.importance_type.lower() == "shap" and raw_shap_values is not None:
            result_dict["raw_shap_values"] = raw_shap_values

        return result_dict

    def analyze_single_reaction_influence(
        self,
        gene_df: pd.DataFrame,
        reaction_df: pd.DataFrame,
        reaction_id: str,
        group1_samples: List[str],
        group2_samples: List[str],
    ) -> Dict:
        """Analyze how gene expression deltas predict changes in reaction activity.

        Args:
            gene_df (pd.DataFrame): Gene expression data
            reaction_df (pd.DataFrame): Reaction activity data
            reaction_id (str): ID of the reaction to analyze
            group1_samples (List[str]): Sample IDs for group 1
            group2_samples (List[str]): Sample IDs for group 2

        Returns:
            Dict: Analysis results
        """
        # Create all possible sample pairs between groups
        sample_pairs = list(product(group1_samples, group2_samples))

        # Limit sample pairs if needed
        if len(sample_pairs) > self.max_sample_pairs:
            random.seed(42)
            print(
                f"  Sampling {self.max_sample_pairs} pairs from total {len(sample_pairs)} possible pairs"
            )
            sample_pairs = random.sample(sample_pairs, self.max_sample_pairs)

        # Calculate deltas for each pair
        gene_deltas, reaction_deltas, reaction_abs_deltas = (
            self.calculate_pairwise_deltas(
                gene_df, reaction_df, reaction_id, sample_pairs
            )
        )

        # Validate sufficient data
        if not self.validate_analysis_data(gene_deltas):
            print(
                f"Warning: Only {len(gene_deltas)} valid sample pairs found for {reaction_id}. Results may be unreliable."
            )
            return {
                "reaction_id": reaction_id,
                "genes_ranked": [],
                "genes_with_symbols": None,
                "r2_score": np.nan,
                "rmse": np.nan,
                "correlation": np.nan,
                "n_pairs": len(gene_deltas),
                "significant_genes_count": 0,
                "importance_type": self.importance_type,
                "error": "Insufficient valid sample pairs",
            }

        # Convert to DataFrames
        X = pd.DataFrame(gene_deltas)
        y = pd.Series(reaction_deltas, name=reaction_id)
        y_abs = pd.Series(reaction_abs_deltas, name=f"{reaction_id}")

        # Calculate gene-reaction correlations
        gene_reaction_correlations = self._calculate_gene_reaction_correlations(
            X, y_abs
        )

        # Select best model
        best_model, best_params = self.select_best_model(X, y)

        if best_model is None:
            return {
                "reaction_id": reaction_id,
                "genes_ranked": [],
                "genes_with_symbols": None,
                "r2_score": np.nan,
                "rmse": np.nan,
                "correlation": np.nan,
                "n_pairs": len(gene_deltas),
                "significant_genes_count": 0,
                "importance_type": self.importance_type,
                "error": "Model fitting failed",
            }

        # Calculate model metrics
        metrics = self.calculate_model_metrics(X, y, best_model)

        # Calculate feature importance
        importance_scores, raw_shap_values = self.calculate_feature_importance(
            X, y, best_model, metrics["r2_score"]
        )

        # Format and return results
        result = self.format_reaction_results(
            reaction_id,
            importance_scores,
            metrics,
            best_params,
            gene_reaction_correlations,
            n_pairs=len(gene_deltas),
            raw_shap_values=raw_shap_values,
        )

        return result

    def aggregate_subsystem_results(
        self, per_reaction_results: Dict, reactions: List[str]
    ) -> Dict:
        """Aggregate results across all reactions in a subsystem.

        Args:
            per_reaction_results (Dict): Results for each reaction
            reactions (List[str]): List of reaction IDs

        Returns:
            Dict: Aggregated subsystem results
        """
        all_significant_genes = {}
        metrics = {
            "r2_score": [],
            "rmse": [],
            "correlation": [],
            "significant_genes_count": [],
        }

        for reaction_id, result in per_reaction_results.items():
            # Update metrics
            metrics["r2_score"].append(result["r2_score"])
            metrics["rmse"].append(result["rmse"])
            metrics["correlation"].append(result["correlation"])
            metrics["significant_genes_count"].append(result["significant_genes_count"])

            # Track significant genes
            for gene_data in result["genes_ranked"]:
                gene_id = gene_data[0]
                coef = gene_data[1]
                importance_val = gene_data[2]

                if gene_id not in all_significant_genes:
                    all_significant_genes[gene_id] = {
                        "count": 0,
                        "total_coef": 0,
                        "total_importance": 0,
                    }

                all_significant_genes[gene_id]["count"] += 1
                all_significant_genes[gene_id]["total_coef"] += abs(coef)
                all_significant_genes[gene_id]["total_importance"] += importance_val

        # Sort genes by total importance
        sorted_genes = sorted(
            all_significant_genes.items(),
            key=lambda x: x[1]["total_importance"],
            reverse=True,
        )

        # Calculate top genes with symbols if translation function is provided
        top_genes_with_symbols = []
        if self.translate_gene_func:
            for gene_id, data in sorted_genes:
                gene_symbol = self.translate_gene_func(
                    gene_id, input_format="ensmbl_id", output_format="geneSymbols"
                )
                avg_importance = (
                    data["total_importance"] / data["count"] if data["count"] > 0 else 0
                )

                top_genes_with_symbols.append(
                    {
                        "gene_id": gene_id,
                        "gene_symbol": gene_symbol,
                        "count": data["count"],
                        "freq_pct": 100 * data["count"] / len(reactions)
                        if reactions
                        else 0,
                        "avg_abs_coef": data["total_coef"] / data["count"]
                        if data["count"] > 0
                        else 0,
                        "avg_importance": avg_importance,
                        "total_importance": data["total_importance"],
                    }
                )

        # Calculate average metrics
        avg_metrics = {k: np.mean(v) if v else np.nan for k, v in metrics.items()}

        return {
            "per_reaction": per_reaction_results,
            "top_genes": sorted_genes,
            "top_genes_with_symbols": top_genes_with_symbols
            if self.translate_gene_func
            else None,
            "avg_metrics": avg_metrics,
            "analyzed_reactions_count": len(per_reaction_results),
            "total_reactions_count": len(reactions),
        }

    def analyze_subsystem_influence(
        self,
        gene_df: pd.DataFrame,
        reaction_df: pd.DataFrame,
        reactions: List[str],
        group1_samples: List[str],
        group2_samples: List[str],
    ) -> Dict:
        """Analyze gene influence on reactions for an entire subsystem.

        Args:
            gene_df (pd.DataFrame): Gene expression data
            reaction_df (pd.DataFrame): Reaction activity data
            reactions (List[str]): List of reaction IDs to analyze
            group1_samples (List[str]): Sample IDs for group 1
            group2_samples (List[str]): Sample IDs for group 2

        Returns:
            Dict: Subsystem analysis results
        """
        results = {}

        for reaction_id in reactions:
            try:
                result = self.analyze_single_reaction_influence(
                    gene_df, reaction_df, reaction_id, group1_samples, group2_samples
                )
                results[reaction_id] = result
            except Exception as e:
                print(f"Error analyzing reaction {reaction_id}: {str(e)}")

        # Aggregate results
        aggregated_results = self.aggregate_subsystem_results(results, reactions)

        # Store for later access
        self.last_analysis_results = aggregated_results

        return aggregated_results

    def run_single_reaction_analysis(
        self,
        gene_df: pd.DataFrame,
        reaction_df: pd.DataFrame,
        reaction_id: str,
        group1_samples: List[str],
        group2_samples: List[str],
        **kwargs,
    ) -> Dict:
        """High-level interface for single reaction analysis.

        This method provides backward compatibility with the original function interface.
        """
        # Update configuration with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key in self.config:
                self.config[key] = value

        return self.analyze_single_reaction_influence(
            gene_df, reaction_df, reaction_id, group1_samples, group2_samples
        )

    def run_subsystem_analysis(
        self,
        gene_df: pd.DataFrame,
        reaction_df: pd.DataFrame,
        reactions: List[str],
        group1_samples: List[str],
        group2_samples: List[str],
        **kwargs,
    ) -> Dict:
        """High-level interface for subsystem analysis.

        This method provides backward compatibility with the original function interface.
        """
        # Update configuration with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key in self.config:
                self.config[key] = value

        return self.analyze_subsystem_influence(
            gene_df, reaction_df, reactions, group1_samples, group2_samples
        )


# 1. Define model factory and strategy classes
class ModelFactory:
    """Factory class to create different regression models."""

    @staticmethod
    def create_model(model_type, **params):
        """Create a specific type of regression model.

        Args:
            model_type: String identifier for model type ("lasso", "xgboost", etc)
            **params: Parameters for the model

        Returns:
            Model instance
        """
        if model_type.lower() == "lasso":
            from sklearn.linear_model import Lasso

            return Lasso(**params)

        elif model_type.lower() == "xgboost":
            import xgboost as xgb

            # Convert alpha to XGBoost's reg_alpha parameter if present
            if "alpha" in params:
                params["reg_alpha"] = params.pop("alpha")
            return xgb.XGBRegressor(**params)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")


class ImportanceCalculator:
    """Calculate variable importance metrics for regression models."""

    @staticmethod
    def calculate_importance(importance_type, X, y, model, full_r2, **params):
        """Calculate feature importance using the specified method.

        Args:
            importance_type: String identifier for importance metric ("partial_r2", "shap", etc)
            X: Feature matrix
            y: Target vector
            model: Fitted model
            full_r2: R² of the full model
            **params: Additional parameters for importance calculation (MUST include min_coef)

        Returns:
            Tuple: (importance_scores, raw_shap_values)
                   importance_scores: Dict mapping features to importance scores (filtered by min_coef)
                   raw_shap_values: Numpy array of raw SHAP values (or None if not applicable/error)
        """
        if importance_type.lower() == "partial_r2":
            # _calculate_partial_r2 already filters internally based on min_coef applied to raw coefficients
            # before calculating partial R2. This is acceptable as partial R2 is only meaningful for included features.
            scores = ImportanceCalculator._calculate_partial_r2(
                X, y, model, full_r2, **params
            )
            return scores, None  # No raw SHAP values for partial R2
        elif importance_type.lower() == "shap":
            # _calculate_shap filters internally based on min_coef applied to mean_abs_shap
            # It now returns both scores and raw values
            return ImportanceCalculator._calculate_shap(X, model, **params)
        else:
            raise ValueError(f"Unsupported importance type: {importance_type}")

    @staticmethod
    def _calculate_partial_r2(X, y, model, full_r2, **params):
        """Calculate partial R² for each feature by refitting the model without each feature."""
        min_coef_threshold = params.get("min_coef", 0)  # Get threshold

        # Get coefficients whose absolute value EXCEEDS the threshold
        if hasattr(model, "coef_"):
            coef_series = pd.Series(model.coef_, index=X.columns)
            # --- This is the effective filtering point for partial R2 ---
            significant_features = coef_series[abs(coef_series) > min_coef_threshold]
        else:
            # Handle non-linear models if partial R2 were adapted (currently assumes linear)
            # For now, raise error if no coef_
            raise ValueError(
                "Partial R² calculation currently requires a model with coefficients (e.g., Lasso)."
            )

        importance_scores = {}
        if significant_features.empty:
            print(
                f"  No features found with abs(coefficient) > {min_coef_threshold} for partial R2 calculation."
            )
            return importance_scores  # Return empty if no features pass threshold

        # Calculate partial R² ONLY for features passing the initial coefficient threshold
        for feature, coef in significant_features.items():
            # Create a copy of the model with same parameters
            # Ensure model_type is passed correctly if needed, assuming Lasso here based on coef_ check
            feature_model = ModelFactory.create_model(
                "lasso",  # Assuming Lasso based on coef_ check
                alpha=model.alpha,
                max_iter=params.get("max_iter", 10000),
                random_state=42,
            )

            # Fit model without this feature
            X_without_feature = X.drop(columns=[feature])
            # Handle case where X_without_feature becomes empty
            if X_without_feature.shape[1] == 0:
                r2_without_feature = 0.0  # Or handle as appropriate, maybe R2 is full_r2? Check logic. For now, 0.
            else:
                feature_model.fit(X_without_feature, y)
                r2_without_feature = r2_score(
                    y, feature_model.predict(X_without_feature)
                )

            # Calculate partial R²
            partial_r2 = max(0.0, full_r2 - r2_without_feature)
            partial_r2 = min(full_r2, partial_r2)  # Cap partial R²

            # Calculate percentage of variance explained relative to full R²
            pct_variance = (partial_r2 / full_r2 * 100) if full_r2 > 0 else 0.0

            # Store results
            importance_scores[feature] = {
                "coefficient": float(coef),
                "partial_r2": float(partial_r2),
                "pct_variance": float(pct_variance),
            }

        return importance_scores  # Filtered based on initial coef threshold

    @staticmethod
    def _calculate_shap(X, model, **params):
        """Calculate feature importance using SHAP values.

        Returns:
            Tuple: (importance_scores, raw_shap_values)
        """
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "SHAP-based importance requires the optional 'importance' extras. "
                "Install metabis with 'metabis[importance]'."
            ) from exc

        importance_scores = {}
        raw_shap_values_to_store = None  # Initialize raw SHAP values storage
        min_coef_threshold = params.get("min_coef", 0)  # Get threshold

        try:
            # ... (SHAP explainer creation) ...
            if hasattr(model, "coef_"):
                explainer = shap.LinearExplainer(model, X)
            else:
                explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list) and len(shap_values) > 0:
                # Handle cases like multi-output models if necessary, assume single output for now
                if isinstance(shap_values[0], np.ndarray):
                    shap_values = shap_values[0]
                # Add more checks if needed based on explainer outputs

            # Ensure shap_values is a 2D numpy array
            if not isinstance(shap_values, np.ndarray) or shap_values.ndim != 2:
                raise TypeError(f"Unexpected SHAP values format: {type(shap_values)}")

            # Store the raw SHAP values before calculating mean etc.
            raw_shap_values_to_store = shap_values.copy()  # Store a copy

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            total_shap = mean_abs_shap.sum()
            pct_contributions = (
                (mean_abs_shap / total_shap * 100)
                if total_shap > 0
                else np.zeros_like(mean_abs_shap)
            )

            # Get coefficients/importances for context (not for filtering here)
            if hasattr(model, "coef_"):
                coefficients = model.coef_
            elif hasattr(model, "feature_importances_"):
                coefficients = model.feature_importances_
            else:
                coefficients = np.zeros(len(X.columns))

            # --- This is the filtering point for SHAP ---
            for i, feature in enumerate(X.columns):
                # Store importance scores ONLY if mean absolute SHAP exceeds threshold
                if mean_abs_shap[i] > min_coef_threshold:
                    importance_scores[feature] = {
                        # Store original coefficient/importance for context
                        "coefficient": float(coefficients[i]),
                        "shap_value": float(mean_abs_shap[i]),
                        "pct_variance": float(pct_contributions[i]),
                    }

        except Exception as e:
            print(
                f"Error calculating SHAP values: {e}. Falling back to basic importance."
            )
            raw_shap_values_to_store = (
                None  # Ensure raw values are None on error/fallback
            )
            # Fallback uses min_coef threshold on raw values
            if hasattr(model, "coef_"):
                coefficients = model.coef_
                for i, feature in enumerate(X.columns):
                    coef = coefficients[i]
                    if abs(coef) > min_coef_threshold:  # Filter here in fallback
                        importance_scores[feature] = {
                            "coefficient": float(coef),
                            "shap_value": float(abs(coef)),  # Use abs(coef) as proxy
                            "pct_variance": 0.0,
                        }
            elif hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                for i, feature in enumerate(X.columns):
                    imp = importances[i]
                    if imp > min_coef_threshold:  # Filter here in fallback
                        importance_scores[feature] = {
                            "coefficient": float(imp),  # Use importance as proxy
                            "shap_value": float(imp),
                            "pct_variance": 0.0,
                        }

        # Return both filtered scores and the raw SHAP values (or None if error)
        return importance_scores, raw_shap_values_to_store


def _run_subsystem_influence_analysis(
    gene_df,
    reaction_df,
    reactions,
    group1_samples,
    group2_samples,
    model_type="lasso",  # Now supports "xgboost" as well
    importance_type="partial_r2",  # Now supports "shap" as well
    alpha=0.1,
    max_iter=10000,
    min_coef=0.001,
    translate_gene_func=None,
    max_sample_pairs=1000,  # New parameter
):
    """Analyze how gene expression deltas predict changes for all reactions in a subsystem."""
    # Analyze each reaction
    results = {}
    all_significant_genes = {}
    metrics = {
        "r2_score": [],
        "rmse": [],
        "correlation": [],
        "significant_genes_count": [],
    }

    for reaction_id in reactions:
        try:
            result = _analyze_single_reaction_influence(
                gene_df,
                reaction_df,
                reaction_id,
                group1_samples,
                group2_samples,
                model_type=model_type,  # Pass model_type parameter
                importance_type=importance_type,  # Pass importance_type parameter
                alpha=alpha,
                max_iter=max_iter,
                min_coef=min_coef,
                translate_gene_func=translate_gene_func,
                max_sample_pairs=max_sample_pairs,
            )

            results[reaction_id] = result

            # Update metrics
            metrics["r2_score"].append(result["r2_score"])
            metrics["rmse"].append(result["rmse"])
            metrics["correlation"].append(result["correlation"])
            metrics["significant_genes_count"].append(result["significant_genes_count"])

            # Track significant genes with partial R² values
            for gene_data in result["genes_ranked"]:
                gene_id = gene_data[0]  # gene_id is the first element
                coef = gene_data[1]  # coefficient is the second element
                partial_r2 = gene_data[2]  # partial_r2 is the third element

                if gene_id not in all_significant_genes:
                    all_significant_genes[gene_id] = {
                        "count": 0,
                        "total_coef": 0,
                        "total_partial_r2": 0,
                    }

                all_significant_genes[gene_id]["count"] += 1
                all_significant_genes[gene_id]["total_coef"] += abs(coef)
                all_significant_genes[gene_id]["total_partial_r2"] += partial_r2

        except Exception as e:
            print(f"Error analyzing reaction {reaction_id}: {str(e)}")

    # Sort genes by frequency and then by partial R² (better measure of importance)
    sorted_genes = sorted(
        all_significant_genes.items(),
        key=lambda x: (x[1]["total_partial_r2"]),
        reverse=True,
    )

    # Calculate top genes with symbols if translation function is provided
    top_genes_with_symbols = []
    if translate_gene_func:
        for gene_id, data in sorted_genes:
            gene_symbol = translate_gene_func(
                gene_id, input_format="ensmbl_id", output_format="geneSymbols"
            )
            avg_partial_r2 = (
                data["total_partial_r2"] / data["count"] if data["count"] > 0 else 0
            )

            top_genes_with_symbols.append(
                {
                    "gene_id": gene_id,
                    "gene_symbol": gene_symbol,
                    "count": data["count"],
                    "freq_pct": 100 * data["count"] / len(reactions)
                    if reactions
                    else 0,
                    "avg_abs_coef": data["total_coef"] / data["count"]
                    if data["count"] > 0
                    else 0,
                    "avg_partial_r2": avg_partial_r2,
                    "total_partial_r2": data["total_partial_r2"],
                }
            )

    # Calculate average metrics
    avg_metrics = {k: np.mean(v) if v else np.nan for k, v in metrics.items()}

    return {
        "per_reaction": results,
        "top_genes": sorted_genes,
        "top_genes_with_symbols": top_genes_with_symbols
        if translate_gene_func
        else None,
        "avg_metrics": avg_metrics,
        "analyzed_reactions_count": len(results),
        "total_reactions_count": len(reactions),
    }


# 2. The main modularized function
def _analyze_single_reaction_influence(
    gene_df,
    reaction_df,
    reaction_id,
    group1_samples,
    group2_samples,
    model_type="lasso",  # Now supports "xgboost" as well
    importance_type="partial_r2",  # Now supports "shap" as well
    alpha=0.1,
    max_iter=10000,
    min_coef=0,
    translate_gene_func=None,
    max_sample_pairs=1000,
):
    """Analyze how gene expression deltas predict changes in reaction activity.

    Args:
        gene_df: DataFrame of gene expression data
        reaction_df: DataFrame of reaction activity data
        reaction_id: ID of the reaction to analyze
        group1_samples: List of sample IDs for group 1
        group2_samples: List of sample IDs for group 2
        model_type: Regression model type ("lasso" by default)
        importance_type: Method for calculating variable importance ("partial_r2" by default)
        alpha: Regularization strength
        max_iter: Maximum iterations for model fitting
        min_coef: Minimum coefficient threshold to consider a feature relevant
        translate_gene_func: Function to translate gene IDs to symbols
        max_sample_pairs: Maximum number of sample pairs to analyze

    Returns:
        Dict containing analysis results
    """
    # Create all possible sample pairs between groups
    sample_pairs = list(product(group1_samples, group2_samples))

    # If the number of pairs exceeds the threshold, sample randomly
    if len(sample_pairs) > max_sample_pairs:
        import random

        random.seed(42)  # For reproducibility
        print(
            f"  Sampling {max_sample_pairs} pairs from total {len(sample_pairs)} possible pairs"
        )
        sample_pairs = random.sample(sample_pairs, max_sample_pairs)

    # Calculate deltas for each pair - this is independent of model type
    gene_deltas, reaction_deltas, reaction_abs_deltas = _calculate_pairwise_deltas(
        gene_df, reaction_df, reaction_id, sample_pairs
    )

    # Check if we have enough data
    if len(gene_deltas) < 10:
        print(
            f"Warning: Only {len(gene_deltas)} valid sample pairs found for {reaction_id}. Results may be unreliable."
        )
        # Return a minimal result structure indicating insufficient data
        return {
            "reaction_id": reaction_id,
            "genes_ranked": [],
            "genes_with_symbols": None,
            "r2_score": np.nan,
            "rmse": np.nan,
            "correlation": np.nan,  # Model correlation
            "n_pairs": len(gene_deltas),
            "significant_genes_count": 0,
            "importance_type": importance_type,
            "error": "Insufficient valid sample pairs",
        }

    # Convert to DataFrames
    X = pd.DataFrame(gene_deltas)
    y = pd.Series(reaction_deltas, name=reaction_id)  # Give the series a name
    y_abs = pd.Series(
        reaction_abs_deltas, name=f"{reaction_id}"
    )  # Absolute deltas for additional analysis
    # --- Calculate Pairwise Gene-Reaction Correlations ---
    gene_reaction_correlations = {}
    if not y_abs.isnull().all() and y_abs.var() > 1e-9:  # Check if y has variance
        for gene in X.columns:
            if (
                not X[gene].isnull().all() and X[gene].var() > 1e-9
            ):  # Check if gene has variance
                try:
                    # Calculate correlation, handle potential NaNs if variance is zero despite check
                    corr_val = X[gene].corr(y_abs)
                    gene_reaction_correlations[gene] = (
                        corr_val if not pd.isna(corr_val) else 0.0
                    )
                except Exception as corr_err:
                    print(
                        f"  Warning: Could not calculate correlation for gene {gene} and reaction {reaction_id}: {corr_err}"
                    )
                    gene_reaction_correlations[gene] = 0.0  # Assign 0 on error
            else:
                gene_reaction_correlations[gene] = (
                    0.0  # Assign 0 if gene has no variance
                )
    else:
        # Assign 0 if reaction has no variance
        gene_reaction_correlations = {gene: 0.0 for gene in X.columns}
    # --- End Correlation Calculation ---

    # Select best model based purely on R² score
    best_model, best_params = _select_best_regression_model(
        X,
        y,
        model_type=model_type,
        alpha_values=[  # Define the grid of hyperparameters to search
            0.1,
            0.01,
            0.001,
            0.0001,
            1e-5,
        ],
        max_iter=max_iter,
    )

    # Calculate final model metrics
    metrics = _calculate_regression_metrics(X, y, best_model)

    # Calculate feature importance using the best model.
    # Filtering by min_coef happens INSIDE calculate_importance.
    # Calculate feature importance using the best model.
    # Filtering by min_coef happens INSIDE calculate_importance.
    # ImportanceCalculator now returns a tuple: (scores, raw_shap_values)
    importance_scores, raw_shap_values = ImportanceCalculator.calculate_importance(
        importance_type,
        X,
        y,
        best_model,
        metrics["r2_score"],
        model_type=model_type,
        min_coef=min_coef,
        max_iter=max_iter,
    )

    # Format results, passing the importance_type, raw_shap_values, and gene_reaction_correlations
    result = _format_single_reaction_results(
        reaction_id,
        importance_scores,
        metrics,
        best_params,
        gene_reaction_correlations,  # Pass the new correlations
        translate_gene_func=translate_gene_func,
        n_pairs=len(gene_deltas),
        importance_type=importance_type,
        raw_shap_values=raw_shap_values,
    )

    return result


# 3. Helper functions
def _calculate_pairwise_deltas(gene_df, reaction_df, reaction_id, sample_pairs):
    """Calculate gene and reaction deltas for sample pairs."""
    gene_deltas = []
    reaction_deltas = []
    reaction_abs_deltas = []

    for s1, s2 in sample_pairs:
        # Skip if either sample is missing in either dataset
        if (
            s1 not in gene_df.index
            or s2 not in gene_df.index
            or s1 not in reaction_df.index
            or s2 not in reaction_df.index
        ):
            continue

        # Calculate gene expression deltas for this pair
        gene_delta = gene_df.loc[s2] - gene_df.loc[s1]
        gene_deltas.append(gene_delta)

        # Calculate reaction activity delta for this pair
        reaction_delta = (
            reaction_df.loc[s2, reaction_id] - reaction_df.loc[s1, reaction_id]
        )
        reaction_deltas.append(reaction_delta)

        # Calculate absolute reaction activity delta for this pair
        reaction_abs_delta = abs(reaction_df.loc[s2, reaction_id]) - abs(
            reaction_df.loc[s1, reaction_id]
        )
        reaction_abs_deltas.append(reaction_abs_delta)

    return gene_deltas, reaction_deltas, reaction_abs_deltas


def _select_best_regression_model(
    X,
    y,
    model_type="lasso",
    alpha_values=None,
    max_iter=10000,
    early_stop_r2_threshold=0.5,  # Threshold for high R2 early stop
    early_stop_delta_threshold=0.01,  # Threshold for small improvement early stop
):
    """Select the best model configuration based on R² score with early stopping."""
    print(f"Selecting best hyperparameter for model type: {model_type}")

    best_model = None
    best_r2 = -np.inf  # Initialize to ensure any valid R² is better
    best_params = {}
    last_model = None  # Store model for the last alpha value as a fallback
    last_params = {}
    previous_best_r2 = -np.inf  # Track R2 before the current iteration's update

    if alpha_values is None:
        alpha_values = [
            0.1,
            0.01,
            0.001,
            0.0001,
            1e-5,
        ]  # Default search grid

    for i, current_alpha in enumerate(alpha_values):
        # Create model parameters
        if model_type.lower() == "lasso":
            model_params = {
                "alpha": current_alpha,
                "max_iter": max_iter,
                "random_state": 42,
                "fit_intercept": True,
            }
            param_name = "alpha"
        elif model_type.lower() == "xgboost":
            model_params = {
                "reg_alpha": current_alpha,
                "n_estimators": 100,
                "learning_rate": 0.1,
                "random_state": 42,
                "verbosity": 0,
            }
            param_name = "reg_alpha"
        else:
            model_params = {"alpha": current_alpha, "random_state": 42}
            param_name = "alpha"

        # Create and fit model
        try:
            model = ModelFactory.create_model(model_type, **model_params)
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            # Handle potential NaN R2 scores gracefully
            if np.isnan(r2):
                print(
                    f"  Warning: R² is NaN for {param_name}={current_alpha}. Treating as -inf."
                )
                r2 = -np.inf
        except Exception as e:
            print(f"  Error fitting model with {param_name}={current_alpha}: {e}")
            r2 = -np.inf  # Assign negative infinity on error

        print(f"  {param_name}={current_alpha}: R²={r2:.4f}")

        # --- Early Stopping Logic ---
        # 1. Stop if R2 is already very high
        if r2 > early_stop_r2_threshold:
            print(
                f"  Early stopping: R² ({r2:.4f}) exceeded threshold ({early_stop_r2_threshold})."
            )
            # Update best model one last time if this one is the best so far
            if r2 > best_r2:
                best_model = model
                best_r2 = r2
                best_params = {param_name: current_alpha}
            break  # Exit the loop

        # 2. Stop if improvement is minimal (and R2 is positive)
        # Check only after the first iteration (i > 0)
        # Use previous_best_r2 to compare with the best score *before* this iteration
        if (
            i > 0 and r2 > 0 and previous_best_r2 > -np.inf
        ):  # Ensure previous best was valid
            delta_r2 = r2 - previous_best_r2
            if delta_r2 < early_stop_delta_threshold:
                print(
                    f"  Early stopping: R² improvement ({delta_r2:.4f}) is less than threshold ({early_stop_delta_threshold})."
                )
                # The best model is already stored from the *previous* iteration (or earlier)
                # No need to update best_model/best_params here, just break
                break  # Exit the loop

        # --- Update Best Model (if not early stopped) ---
        # Store the current best R2 *before* potentially updating it
        previous_best_r2 = best_r2
        if r2 > best_r2:
            best_model = model
            best_r2 = r2
            best_params = {param_name: current_alpha}

        # Store the last model details in case it's needed for fallback
        if current_alpha == alpha_values[-1]:
            last_model = model
            last_params = {param_name: current_alpha}
            # If the loop finishes naturally, the last valid best_r2 is already set

    # --- Fallback Logic (if no model was ever selected or all failed) ---
    if best_model is None or best_r2 == -np.inf:
        print(
            f"  Warning: Could not find a model with valid R2 > -inf. Trying last alpha: {alpha_values[-1]}"
        )
        if last_model is not None:
            best_model = last_model
            best_params = last_params
            try:
                best_r2 = r2_score(y, best_model.predict(X))
                if np.isnan(best_r2):
                    best_r2 = -np.inf  # Handle NaN in fallback too
            except Exception:
                best_r2 = -np.inf  # If prediction fails
            print(
                f"  Using fallback model with {list(best_params.keys())[0]}={list(best_params.values())[0]}, R2={best_r2:.4f}"
            )
        else:
            print("  Error: Failed to fit any model successfully.")
            return None, {}

    # --- Final Checks and Reporting ---
    elif best_r2 <= 0:
        print(
            f"  Warning: Best model R2 ({best_r2:.4f}) is non-positive. Model may not be predictive."
        )

    print(
        f"  Selected best model with {list(best_params.keys())[0]}={list(best_params.values())[0]}, R2={best_r2:.4f}"
    )
    return best_model, best_params


def _calculate_regression_metrics(X, y, model):
    """Calculate performance metrics for the model."""
    from sklearn.metrics import r2_score

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    corr = np.corrcoef(y, y_pred)[0, 1] if len(y) > 1 else 0

    return {"r2_score": float(r2), "rmse": float(rmse), "correlation": float(corr)}


def _format_single_reaction_results(
    reaction_id,
    importance_scores,
    metrics,
    model_params,
    gene_reaction_correlations,  # Add new parameter
    translate_gene_func=None,
    n_pairs=0,
    importance_type="partial_r2",
    raw_shap_values=None,
):
    """Format analysis results into a standardized structure."""
    genes_ranked = []

    # Populate genes_ranked from the filtered importance_scores
    if importance_type.lower() == "partial_r2":
        for gene, scores in importance_scores.items():
            gene_react_corr = gene_reaction_correlations.get(
                gene,
                0.0,  # Get correlation
            )
            genes_ranked.append(
                (
                    gene,
                    scores["coefficient"],
                    scores["partial_r2"],
                    scores["pct_variance"],
                    gene_react_corr,  # Add correlation to tuple
                )
            )
        # Sort by partial R2 (index 2)
        genes_ranked.sort(key=lambda x: x[2], reverse=True)

    elif importance_type.lower() == "shap":
        for gene, scores in importance_scores.items():
            gene_react_corr = gene_reaction_correlations.get(
                gene,
                0.0,  # Get correlation
            )
            genes_ranked.append(
                (
                    gene,
                    scores["coefficient"],
                    scores["shap_value"],
                    scores["pct_variance"],
                    gene_react_corr,  # Add correlation to tuple
                )
            )
        # Sort by SHAP value (index 2)
        genes_ranked.sort(key=lambda x: x[2], reverse=True)

    # Add gene symbols if translation function is provided
    genes_with_symbols = []
    if translate_gene_func and genes_ranked:
        # Adjust loop for the new tuple length (5 elements)
        for (
            gene_id,
            coef_or_imp,
            importance_val,
            pct_variance,
            gene_react_corr,
        ) in genes_ranked:
            try:
                gene_symbol = translate_gene_func(
                    gene_id,
                    input_format="ensmbl_id",
                    output_format="geneSymbols",
                )
                if gene_symbol is None:
                    gene_symbol = gene_id
            except Exception as e:
                print(f"  Warning: Error translating gene ID {gene_id}: {e}")
                gene_symbol = gene_id

            genes_with_symbols.append(
                (
                    gene_id,
                    gene_symbol,
                    coef_or_imp,
                    importance_val,
                    pct_variance,
                    gene_react_corr,
                )  # Add correlation
            )

    # Determine the regularization parameter name used in the best model
    reg_param = list(model_params.keys())[0] if model_params else "unknown_param"
    reg_value = list(model_params.values())[0] if model_params else None

    # The count len(genes_ranked) now accurately reflects the number of significant genes
    # based on the final importance metric and min_coef threshold.
    print(
        f"Final model for {reaction_id}: {reg_param}={reg_value}, "
        f"R²={metrics['r2_score']:.3f}, significant genes: {len(genes_ranked)}"
    )

    result_dict = {
        "reaction_id": reaction_id,
        "genes_ranked": genes_ranked,  # Now includes gene-reaction correlation
        "genes_with_symbols": genes_with_symbols
        if translate_gene_func
        else None,  # Now includes gene-reaction correlation
        "r2_score": metrics["r2_score"],
        "rmse": metrics["rmse"],
        "correlation": metrics[
            "correlation"
        ],  # This is still the model's prediction correlation
        "n_pairs": n_pairs,
        "significant_genes_count": len(genes_ranked),
        "importance_type": importance_type,
        f"{reg_param}_used": reg_value,
        "raw_shap_values": raw_shap_values
        if importance_type.lower() == "shap" and raw_shap_values is not None
        else None,
    }
    if result_dict.get("raw_shap_values") is None:
        result_dict.pop("raw_shap_values", None)

    return result_dict

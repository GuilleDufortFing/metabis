import time
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Tuple

import numpy as np
import pandas as pd

from .importance_analysis import (
    ImportanceAnalysis,
)
from .statistical_analysis import (
    StatisticalAnalysis,
)
from ..managers.data_manager_mode import DataManagerMode


class GroupComparison:
    def __init__(self, group_id_1: str, group_id_2: str, test: str, correction: str) -> None:
        """Initialize a group comparison between two groups with a specific statistical test.

        Args:
            group_id_1 (str): ID of the first group
            group_id_2 (str): ID of the second group
            test (str): Statistical test to use ('wilcoxon' or 'ttest')

        Returns:
            None

        Raises:
            ValueError: If the test is not supported
        """
        self.group_id_1 = group_id_1
        self.group_id_2 = group_id_2
        self.significant_reactions = set()
        self.back_significant_reactions = set()
        self.significant_genes = set()

        # Validate and store the statistical test
        supported_tests = ["wilcoxon", "ttest"]
        if test.lower() not in supported_tests:
            raise ValueError(
                f"Unsupported statistical test: {test}. Supported tests: {supported_tests}"
            )
        self.statistical_test = test.lower()

        # Initialize the StatisticalAnalysis instance
        self.stats_analyzer = StatisticalAnalysis(test_type=test, correction_type=correction)

        # Initialize the ImportanceAnalysis instance with default parameters
        self.importance_analyzer = ImportanceAnalysis()

        # Store test results as single dataframes
        self._reaction_test_results = None  # Will be a DataFrame
        self._gene_test_results = None  # Will be a DataFrame
        
        # Statistics for effect size normalization (kept for backward compatibility)
        self._reaction_metrics_stats = {
            "cohens_d": {"std": 1.0, "mean": 0.0},
            "log2_fc": {"std": 1.0, "mean": 0.0},
            "mean_difference": {"std": 0.5, "mean": 0.0},
        }
        self._gene_metrics_stats = {
            "cohens_d": {"std": 1.0, "mean": 0.0},
            "log2_fc": {"std": 1.0, "mean": 0.0},
            "mean_difference": {"std": 0.5, "mean": 0.0},
        }

    def _calculate_metrics_stats(
        self, results: pd.DataFrame, for_genes: bool = False
    ) -> None:
        """Calculate statistics for metrics normalization.

        Args:
            results (pd.DataFrame): Test results
            for_genes (bool): Whether the results are for genes (True) or reactions (False)

        Returns:
            None
        """
        # Use the StatisticalAnalysis class to calculate metrics stats
        calculated_stats = self.stats_analyzer.calculate_metrics_stats(results)

        # Update the appropriate metrics stats based on element type
        if for_genes:
            self._gene_metrics_stats.update(calculated_stats)
        else:
            self._reaction_metrics_stats.update(calculated_stats)

    def _register_distributions(
        self, data_manager, results: pd.DataFrame, element_type: str
    ) -> None:
        """Register statistical distributions with the ColorManager.

        Args:
            data_manager: The data manager instance.
            results: The results DataFrame containing metrics.
            element_type: The type of element ("reaction" or "gene").
        """
        if not hasattr(data_manager, "color_manager"):
            return

        # Calculate negative log p-values for distribution registration
        # We use a small epsilon to avoid log(0)
        epsilon = 1e-300
        if "pval" in results.columns:
            results["neg_log_pval"] = -np.log10(results["pval"].replace(0, np.nan).fillna(epsilon))
        
        if "adjusted_pval" in results.columns:
            results["neg_log_adjusted_pval"] = -np.log10(results["adjusted_pval"].replace(0, np.nan).fillna(epsilon))

        metrics = ["log2_fc", "cohens_d", "mean_difference", "pval", "adjusted_pval", "neg_log_pval", "neg_log_adjusted_pval"]

        for metric in metrics:
            if metric in results.columns:
                data_manager.color_manager.register_distribution(
                    mode=DataManagerMode.GROUP_COMPARISON,
                    element_type=element_type,
                    metric_name=metric,
                    data=results[metric],
                    center_method="zero",
                    spread_method="std",
                )

    def get_metrics_stats(self, for_genes: bool = False) -> Dict[str, Dict[str, float]]:
        """Get the metrics statistics used for normalization.

        Args:
            for_genes (bool): Whether to get gene metrics (True) or reaction metrics (False)

        Returns:
            Dict[str, Dict[str, float]]: Dictionary of metrics statistics with structure:
                {
                    "metric_name": {
                        "mean": float_value,
                        "std": float_value
                    },
                    ...
                }
        """
        # Return the appropriate metrics stats dictionary
        if for_genes:
            return self._gene_metrics_stats
        else:
            return self._reaction_metrics_stats

    def run_test(
        self,
        data_manager,
        for_genes: bool = False,
        test: str = "wilcoxon",  # default test type
        correction: str = "fdr_bh",
    ) -> None:
        """Run the statistical test on reaction or gene data.

        Args:
            data_manager (Experiment): The data_manager object containing data
            for_genes (bool): If True, run the test on gene data; otherwise on reaction data
            test (str): 'wilcoxon' or 'ttest' (default 'wilcoxon')

        Returns:
            None
        """
        group_1_samples = data_manager.get_group_samples(
            self.group_id_1, exclude_group=self.group_id_2
        )
        group_2_samples = data_manager.get_group_samples(
            self.group_id_2, exclude_group=self.group_id_1
        )

        if for_genes:
            data_df = data_manager.expression_data.get_expression_df()
        else:
            data_df = data_manager.reaction_data.get_reaction_df()

        # Use StatisticalAnalysis to perform the comparison
        results = self.stats_analyzer.run_statistical_test(
            data_df,
            group_1_samples,
            group_2_samples,
            test_type=test,
            correction_type=correction,
        )

        # Add group information
        results["group_1"] = self.group_id_1
        results["group_2"] = self.group_id_2

        # Add IDs
        if for_genes:
            results["gene_id"] = results.index
            # Add subsystems information to the results DataFrame
            results["subsystems"] = results.index.map(
                lambda gid: ", ".join(
                    data_manager.metabolic_model.get_gene_metadata(gid).get(
                        "subsystems", ["Unknown"]
                    )
                )
            )
            self._gene_test_results = results
        else:
            results["reaction_id"] = results.index
            # Add subsystems information to the results DataFrame
            results["subsystems"] = results.index.map(
                lambda rid: ", ".join(
                    data_manager.metabolic_model.get_reaction_metadata(rid).get(
                        "subsystems", ["Unknown"]
                    )
                )
            )
            self._reaction_test_results = results
        
        # Default thresholds for all significant
        p_val_threshold = 1
        cohens_d_threshold = 0.0
        log2_fc_threshold = 0.0
        mean_difference_threshold = 0.0

        self.define_significant_elements(for_genes=for_genes, 
            pval_threshold=p_val_threshold,
            cohens_d_threshold=cohens_d_threshold,
            log2_fc_threshold=log2_fc_threshold,
            mean_difference_threshold=mean_difference_threshold,
        )

        # Calculate statistics for effect-size normalization
        self._calculate_metrics_stats(results, for_genes=for_genes)

        # Register distributions for coloring
        element_type = "gene" if for_genes else "reaction"
        self._register_distributions(data_manager, results, element_type)

        

    def get_test_results(
        self, for_genes: bool = False
    ) -> Optional[pd.DataFrame]:
        """Get the test results for reactions or genes.

        Args:
            for_genes (bool): Return gene test results when True; reaction results otherwise.

        Returns:
            Optional[pd.DataFrame]: Test results for the requested element type and direction.
        """

        if for_genes:
            return self._gene_test_results
        else:
            return self._reaction_test_results

    def define_significant_elements(
        self,
        pval_threshold: Optional[float] = None,
        cohens_d_threshold: Optional[float] = None,
        log2_fc_threshold: Optional[float] = None,
        mean_difference_threshold: Optional[float] = None,
        keep_reversed: bool = False,
        for_genes: bool = False,
    ) -> Tuple[set, Dict[str, Any]]:
        """Identify significant reactions based on statistical thresholds.

        Args:
            pval_threshold (float, optional): P-value threshold for significance.
            cohens_d_threshold (float, optional): Minimum absolute Cohen's d value.
            log2_fc_threshold (float, optional): Minimum absolute log2 fold change.
            mean_difference_threshold (float, optional): Minimum absolute mean difference.
            keep_reversed (bool): Whether to always keep reactions with reversed flow
                as significant regardless of log2_fc value.
            for_genes (bool): Whether to identify significant genes instead of reactions.

        Returns:
            Tuple[set, Dict]: Tuple of (set of significant element IDs, threshold_info dict)

        Raises:
            ValueError: If reaction test hasn't been run yet
        """
        if for_genes:
            test_results = self._gene_test_results
        else:
            test_results = self._reaction_test_results

        # Check if test results exist 
        # (If there are no test results then there shouldn't be rev test results either)
        if test_results is None:
            raise ValueError(
                "No test results found. Run the test first with run_test()."
            )

        # Use StatisticalAnalysis to determine significant elements
        significant_mask, threshold_info = (
            self.stats_analyzer.determine_significant_elements(
                test_results,
                pval_threshold=pval_threshold,
                cohens_d_threshold=cohens_d_threshold,
                log2_fc_threshold=log2_fc_threshold,
                mean_difference_threshold=mean_difference_threshold,
                keep_reversed=keep_reversed,
            )
        )

        # Store the significant elements based on the type (genes or reactions)
        if for_genes:
            self.significant_genes = set(test_results[significant_mask].index)
            self._gene_test_results["is_significant"] = self._gene_test_results.index.map(
                lambda gid: gid in self.significant_genes
            )
        else:
            self.significant_reactions = set(test_results[significant_mask].index)
            self._reaction_test_results[
                "is_significant"
            ] = self._reaction_test_results.index.map(
                lambda rid: rid in self.significant_reactions
            )

        self.set_thresholds(
            pval_threshold,
            cohens_d_threshold,
            log2_fc_threshold,
            mean_difference_threshold,
            for_genes,
        )

        return set(test_results[significant_mask].index), threshold_info
    
    def is_significant(
        self,
        element_id: str,
        for_genes: bool = False,
    ) -> bool:
        """Check if a specific element (reaction or gene) is significant.

        Args:
            element_id (str): ID of the element (reaction or gene)

        Returns:
            bool: True if the element is significant, False otherwise
        """
        if for_genes:
            return element_id in self.significant_genes
        return element_id in self.significant_reactions

    def set_thresholds(
        self,
        pval_threshold=None,
        cohens_d_threshold=None,
        log2_fc_threshold=None,
        mean_difference_threshold=None,
        for_genes=False,
    ):
        """Set threshold values for significance testing.

        Args:
            pval_threshold (float, optional): P-value threshold for significance.
            cohens_d_threshold (float, optional): Minimum absolute Cohen's d value.
            log2_fc_threshold (float, optional): Minimum absolute log2 fold change.
            mean_difference_threshold (float, optional): Minimum absolute mean difference.
            for_genes (bool): Whether to set thresholds for genes (True) or reactions (False).
        """
        if for_genes:
            self.adjusted_pval_threshold_gene = pval_threshold
            self.p_val_threshold_gene = pval_threshold
            self.neg_log_adj_pval_threshold_gene = (
                -np.log10(pval_threshold) if pval_threshold is not None else None
            )
            self.neg_log_pval_threshold_gene = (
                -np.log10(pval_threshold) if pval_threshold is not None else None
            )
            self.cohens_d_threshold_gene = cohens_d_threshold
            self.log2_fc_threshold_gene = log2_fc_threshold
            self.mean_difference_threshold_gene = mean_difference_threshold
        else:
            self.adjusted_pval_threshold_reaction = pval_threshold
            self.p_val_threshold_reaction = pval_threshold
            self.neg_log_adj_pval_threshold_reaction = (
                -np.log10(pval_threshold) if pval_threshold is not None else None
            )
            self.neg_log_pval_threshold_reaction = (
                -np.log10(pval_threshold) if pval_threshold is not None else None
            )
            self.cohens_d_threshold_reaction = cohens_d_threshold
            self.log2_fc_threshold_reaction = log2_fc_threshold
            self.mean_difference_threshold_reaction = mean_difference_threshold

        

    def get_thresholds(self, for_genes=False):
        """Get threshold values for significance testing.

        Args:
            for_genes (bool): Whether to get thresholds for genes (True) or reactions (False).

        Returns:
            dict: Dictionary containing the threshold values.
        """
        if for_genes:
            return {
                "adjusted_pval": self.adjusted_pval_threshold_gene,
                "pval": self.p_val_threshold_gene,
                "neg_log_adjusted_pval": self.neg_log_adj_pval_threshold_gene,
                "neg_log_pval": self.neg_log_pval_threshold_gene,
                "cohens_d": self.cohens_d_threshold_gene,
                "log2_fc": self.log2_fc_threshold_gene,
                "mean_difference": self.mean_difference_threshold_gene,
            }
        else:
            return {
                "adjusted_pval": self.adjusted_pval_threshold_reaction,
                "pval": self.p_val_threshold_reaction,
                "neg_log_adjusted_pval": self.neg_log_adj_pval_threshold_reaction,
                "neg_log_pval": self.neg_log_pval_threshold_reaction,
                "cohens_d": self.cohens_d_threshold_reaction,
                "log2_fc": self.log2_fc_threshold_reaction,
                "mean_difference": self.mean_difference_threshold_reaction,
            }

    def get_significant_elements(self, for_genes: bool = False, reverse = False) -> set:
        """Get the set of significant reactions or genes.

        Args:
            for_genes (bool): Whether to get significant genes (True) or reactions (False)

        Returns:
            set: Set of significant reaction or gene IDs
        """
        if for_genes:
            # If significant_genes is empty, return all genes
            return self.significant_genes
        else:
            return self.significant_reactions

    def get_statistical_info(
        self, element_id: str, for_genes: bool = False
    ) -> Dict[str, Any]:
        """Get statistical information for a specific element (reaction or gene).

        Args:
            element_id (str): ID of the element (reaction or gene)
            for_genes (bool): Whether to get gene info (True) or reaction info (False)

        Returns:
            Dict[str, Any]: Statistical information

        Raises:
            ValueError: If test hasn't been run yet
        """
        # Get appropriate test results based on element type
        if for_genes:
            results = self._gene_test_results
        else:
            results = self._reaction_test_results

        if results is None:
            raise ValueError(
                "No test results found. Run the test first with run_test()."
            )

        # Use StatisticalAnalysis to get element statistical info
        return self.stats_analyzer.get_element_statistical_info(
            results, element_id, self.group_id_1, self.group_id_2
        )
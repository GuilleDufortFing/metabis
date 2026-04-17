"""Statistical analysis module for group comparisons.

This module provides statistical test functionality, effect size calculations,
and significance determination for comparing groups of samples.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests


class StatisticalAnalysis:
    """A class to handle all statistical analysis operations for group comparisons.

    This class encapsulates statistical test execution, effect size calculations,
    significance determination, and metrics normalization.
    """

    def __init__(self, test_type: str = "wilcoxon", correction_type: str = "fdr_bh"):
        """Initialize the statistical analysis with a specific test type.

        Args:
            test_type (str): Statistical test to use ('wilcoxon' or 'ttest')

        Raises:
            ValueError: If the test type is not supported
        """
        supported_tests = ["wilcoxon", "ttest"]
        if test_type.lower() not in supported_tests:
            raise ValueError(
                f"Unsupported statistical test: {test_type}. Supported tests: {supported_tests}"
            )
        self.test_type = test_type.lower()

        supported_corrections = ["bonferroni", "sidak", "holm-sidak", "holm", "simes-hochberg", "hommel", "fdr_bh", "fdr_by", "fdr_tsbh", "fdr_tsbky"]
        if correction_type.lower() not in supported_corrections:
            raise ValueError(
                f"Unsupported statistical p-value correction method: {correction_type}. Supported methods: {supported_corrections}"
            )
        self.correction_type = correction_type.lower()

        # Initialize metrics statistics for normalization
        self._default_metrics_stats = {
            "cohens_d": {"std": 1.0, "mean": 0.0},
            "log2_fc": {"std": 1.0, "mean": 0.0},
            "mean_difference": {"std": 0.5, "mean": 0.0},
        }

    def run_statistical_test(
        self,
        data_df: pd.DataFrame,
        group_1_samples: List[str],
        group_2_samples: List[str],
        test_type: Optional[str] = None,
        correction_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Run statistical test between two groups.

        Args:
            data_df (pd.DataFrame): DataFrame with features as rows and samples as columns
            group_1_samples (List[str]): List of sample IDs for group 1
            group_2_samples (List[str]): List of sample IDs for group 2
            test_type (str, optional): Override the default test type

        Returns:
            pd.DataFrame: Test results with statistical metrics
        """
        test = test_type or self.test_type
        correction = correction_type or self.correction_type

        # Use existing comparison_test function
        results = comparison_test(
            data_df,
            group_1_samples,
            group_2_samples,
            test=test,
            correction=correction
        )

        return results

    def calculate_metrics_stats(
        self, results: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for metrics normalization.

        Args:
            results (pd.DataFrame): Test results containing metrics

        Returns:
            Dict[str, Dict[str, float]]: Statistics for each metric
        """
        metrics_stats = {}

        for metric in ["cohens_d", "log2_fc", "mean_difference"]:
            if metric in results.columns:
                values = results[metric].dropna()
                if not values.empty:
                    metrics_stats[metric] = {
                        "std": float(values.std()) or 1.0,
                        "mean": float(values.mean()) or 0.0,
                    }
                else:
                    metrics_stats[metric] = self._default_metrics_stats[metric].copy()
            else:
                metrics_stats[metric] = self._default_metrics_stats[metric].copy()

        return metrics_stats

    def determine_significant_elements(
        self,
        results: pd.DataFrame,
        pval_threshold: Optional[float] = None,
        cohens_d_threshold: Optional[float] = None,
        log2_fc_threshold: Optional[float] = None,
        mean_difference_threshold: Optional[float] = None,
        keep_reversed: bool = False,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """Determine which elements are statistically significant.

        Args:
            results (pd.DataFrame): Test results
            pval_threshold (float, optional): P-value threshold
            cohens_d_threshold (float, optional): Cohen's d threshold
            log2_fc_threshold (float, optional): Log2 fold change threshold
            mean_difference_threshold (float, optional): Mean difference threshold
            keep_reversed (bool): Keep reversed reactions regardless of log2_fc

        Returns:
            Tuple[pd.Series, Dict]: (significant_mask, threshold_info)
        """
        # Initialize significant mask as all True
        significant_mask = pd.Series(True, index=results.index)

        threshold_info = {
            "applied_thresholds": {},
            "total_elements": len(results),
            "significant_count": 0,
            "failed_by_threshold": {},  # Independent count per threshold
            "passed_by_threshold": {},  # Independent count per threshold
        }

        # Apply thresholds if specified and track independent failures
        if pval_threshold is not None:
            pval_mask = results["adjusted_pval"] < pval_threshold
            threshold_info["failed_by_threshold"]["pval"] = (~pval_mask).sum()
            threshold_info["passed_by_threshold"]["pval"] = pval_mask.sum()
            significant_mask &= pval_mask
            threshold_info["applied_thresholds"]["pval"] = pval_threshold

        if cohens_d_threshold is not None:
            cohens_d_mask = abs(results["cohens_d"]) >= cohens_d_threshold
            threshold_info["failed_by_threshold"]["cohens_d"] = (~cohens_d_mask).sum()
            threshold_info["passed_by_threshold"]["cohens_d"] = cohens_d_mask.sum()
            significant_mask &= cohens_d_mask
            threshold_info["applied_thresholds"]["cohens_d"] = cohens_d_threshold

        if log2_fc_threshold is not None:
            if keep_reversed and "reversed" in results.columns:
                # Keep reversed reactions or apply threshold to non-reversed
                log2_fc_mask = results["reversed"] | (
                    abs(results["log2_fc"]) >= log2_fc_threshold
                )
            else:
                log2_fc_mask = abs(results["log2_fc"]) >= log2_fc_threshold
            threshold_info["failed_by_threshold"]["log2_fc"] = (~log2_fc_mask).sum()
            threshold_info["passed_by_threshold"]["log2_fc"] = log2_fc_mask.sum()
            significant_mask &= log2_fc_mask
            threshold_info["applied_thresholds"]["log2_fc"] = log2_fc_threshold

        if mean_difference_threshold is not None:
            mean_diff_mask = abs(results["mean_difference"]) >= mean_difference_threshold
            threshold_info["failed_by_threshold"]["mean_difference"] = (~mean_diff_mask).sum()
            threshold_info["passed_by_threshold"]["mean_difference"] = mean_diff_mask.sum()
            significant_mask &= mean_diff_mask
            threshold_info["applied_thresholds"]["mean_difference"] = (
                mean_difference_threshold
            )

        threshold_info["significant_count"] = significant_mask.sum()

        return significant_mask, threshold_info

    def get_element_statistical_info(
        self, results: pd.DataFrame, element_id: str, group_id_1: str, group_id_2: str
    ) -> Dict[str, Any]:
        """Get comprehensive statistical information for a specific element.

        Args:
            results (pd.DataFrame): Test results
            element_id (str): ID of the element to get info for
            group_id_1 (str): First group ID
            group_id_2 (str): Second group ID

        Returns:
            Dict[str, Any]: Statistical information
        """
        if element_id not in results.index:
            return {}

        row = results.loc[element_id]

        return {
            "group_id_1": group_id_1,
            "group_id_2": group_id_2,
            "cohens_d": float(row.get("cohens_d", 0)),
            "log2_fc": float(row.get("log2_fc", 0)),
            "mean_difference": float(row.get("mean_difference", 0)),
            "pval": float(row.get("pval", 1.0)),
            "adjusted_pval": float(row.get("adjusted_pval", 1.0)),
            "reversed": bool(row.get("reversed", False)),
            "test": self.test_type,
            "max_cohens_d": (
                float(abs(results["cohens_d"]).max())
                if "cohens_d" in results.columns
                else 1.0
            ),
            "is_significant": bool(row.get("is_significant", False)),
        }


def cohens_d(x, y):
    pooled_std = np.sqrt(
        ((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1))
        / (len(x) + len(y) - 2)
    )
    return (np.mean(x) - np.mean(y)) / pooled_std


def comparison_test(
    consistencies_matrix, group_A_cells, group_B_cells, test="wilcoxon", correction="fdr_bh"
):
    """Perform a two-group comparison (Wilcoxon/Mann-Whitney or t-test) between groups.

    Args:
        consistencies_matrix (pd.DataFrame): DataFrame with reaction activities
        group_A_cells (List[str]): List of cell IDs for group A
        group_B_cells (List[str]): List of cell IDs for group B
        test (str): Type of test, either 'wilcoxon' or 'ttest' (default 'wilcoxon').
        correction (str): Type of p-value correction, either 'bonferroni' or 'fdr_bh' (default 'fdr_bh')

    Returns:
                pd.DataFrame: Test results including:
                        - test statistic and p-value
                            (Mann-Whitney U for wilcoxon, Welch t statistic for ttest)
            - Cohen's d effect size
            - Log2 fold change
            - Mean difference
            - Reversed flag (True if the reaction changed direction)
            - Adjusted p-value
            - Z-scores
    """
    consistencies_matrix = consistencies_matrix.T

    group_A = consistencies_matrix.loc[:, group_A_cells]
    group_B = consistencies_matrix.loc[:, group_B_cells]

    results = pd.DataFrame(
        index=consistencies_matrix.index,
        columns=[
            "test_stat",
            "pval",
            "cohens_d",
            "log2_fc",
            "mean_difference",
            "reversed",
        ],
        dtype="float64",
    )
    results["reversed"] = False

    for elem in consistencies_matrix.index:
        A = group_A.loc[elem].to_numpy().ravel()
        B = group_B.loc[elem].to_numpy().ravel()

        # Skip if both groups have identical single values
        if (
            len(np.unique(A)) == 1
            and len(np.unique(B)) == 1
            and np.unique(A) == np.unique(B)
        ):
            results.loc[elem] = [0, 1, 0, 0, 0, False]
            continue

        # Perform the selected test
        if test == "ttest":
            stat, pval = ttest_ind(A, B, equal_var=False)
        else:  # default to 'wilcoxon' (Mann-Whitney)
            stat, pval = mannwhitneyu(B, A, alternative="two-sided")

        # Calculate group means
        mean_A = np.mean(A)
        mean_B = np.mean(B)

        # Calculate Cohen's d (sign depends on direction)
        if mean_A < 0:
            c_d = cohens_d(A, B)
            mean_diff = mean_A - mean_B
        else:
            c_d = cohens_d(B, A)
            mean_diff = mean_B - mean_A

        # Calculate log2 fold change (handling sign reversals)
        epsilon = 1e-10
        if (mean_A * mean_B) < 0 and abs(mean_A) > epsilon and abs(mean_B) > epsilon:
            results.loc[elem, "reversed"] = True
            abs_mean_A = abs(mean_A)
            abs_mean_B = abs(mean_B)
            log2_fc = abs(np.log2(abs_mean_B / abs_mean_A))
            c_d = abs(c_d)
            mean_diff = abs(mean_diff)
        else:
            results.loc[elem, "reversed"] = False
            adj_mean_A = max(abs(mean_A), epsilon)
            adj_mean_B = max(abs(mean_B), epsilon)
            log2_fc = np.log2(adj_mean_B / adj_mean_A)

        # Store results
        results.loc[elem] = [
            stat,
            pval,
            c_d,
            log2_fc,
            mean_diff,
            results.loc[elem, "reversed"],
        ]

    # Adjust p-values
    results["adjusted_pval"] = np.array(
        multipletests(results["pval"], method=correction)[1], dtype="float64"
    )

    return results
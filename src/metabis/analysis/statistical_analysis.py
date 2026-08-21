"""Statistical analysis module for group comparisons.

This module provides statistical test functionality, effect size calculations,
and significance determination for comparing groups of samples.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests

__all__ = [
    "CORRECTION_METHODS",
    "StatisticalAnalysis",
    "ComparisonInputs",
    "validate_comparison_inputs",
    "comparison_test",
    "cohens_d",
]


CORRECTION_METHODS = {
    "fdr_bh": "Benjamini-Hochberg (FDR)",
    "holm": "Holm-Bonferroni (FWER)",
}


@dataclass(frozen=True)
class ComparisonInputs:
    """Validated sample membership for a two-group comparison."""

    group_1_samples: tuple[str, ...]
    group_2_samples: tuple[str, ...]
    tested_samples: tuple[str, ...]


def validate_comparison_inputs(
    data_df: pd.DataFrame,
    group_1_samples: Sequence[str],
    group_2_samples: Sequence[str],
    minimum_group_size: int = 2,
) -> ComparisonInputs:
    """Validate membership before any SciPy function is called."""
    group_1 = tuple(group_1_samples)
    group_2 = tuple(group_2_samples)
    if not group_1 or not group_2:
        raise ValueError("Comparison groups cannot be empty")
    if len(group_1) < minimum_group_size or len(group_2) < minimum_group_size:
        raise ValueError(
            f"Each comparison group must contain at least {minimum_group_size} samples"
        )
    if len(set(group_1)) != len(group_1) or len(set(group_2)) != len(group_2):
        raise ValueError("Comparison groups cannot contain duplicate sample IDs")
    overlap = sorted(set(group_1) & set(group_2))
    if overlap:
        raise ValueError(f"Comparison groups overlap: {', '.join(overlap)}")
    unknown = sorted((set(group_1) | set(group_2)) - set(data_df.index))
    if unknown:
        raise ValueError(f"Unknown comparison samples: {', '.join(unknown)}")
    selected = data_df.loc[list(group_1 + group_2)]
    non_finite = int((~np.isfinite(selected.to_numpy(dtype=float))).sum())
    if non_finite:
        raise ValueError(f"Comparison data contains {non_finite} non-finite value(s)")
    return ComparisonInputs(group_1, group_2, group_1 + group_2)


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

        return self.attach_derived_metrics(results)

    @staticmethod
    def attach_derived_metrics(results: pd.DataFrame) -> pd.DataFrame:
        """Add the negative-log p-value metrics to a test-results frame.

        Thresholds, scatter plots, and hover read these as ordinary result
        metrics, so a computed result carries them rather than acquiring them
        from a later presentation call.

        Args:
            results (pd.DataFrame): Test results containing p-value columns

        Returns:
            pd.DataFrame: The same frame with derived metrics attached
        """
        # A small epsilon keeps -log10 finite when a p-value underflows to zero.
        epsilon = 1e-300
        for source, derived in (
            ("pval", "neg_log_pval"),
            ("adjusted_pval", "neg_log_adjusted_pval"),
        ):
            if source in results.columns:
                results[derived] = -np.log10(
                    results[source].replace(0, np.nan).fillna(epsilon)
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
    """Return Cohen's d, or NaN when pooled variance is undefined or zero."""
    if len(x) < 2 or len(y) < 2:
        return np.nan
    pooled_std = np.sqrt(
        ((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1))
        / (len(x) + len(y) - 2)
    )
    if not np.isfinite(pooled_std) or pooled_std == 0:
        return np.nan
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def _signed_log2_fold_change(target_mean: float, baseline_mean: float) -> float:
    """Return an absolute-mean fold-change with Target-minus-Baseline sign.

    Activity and expression means are normally non-negative, where this is the
    standard ``log2(target / baseline)`` calculation. Signed flux inputs do not
    have a standard fold-change ratio, so their magnitude still uses absolute
    means while their sign follows the comparison's declared effect direction.
    """
    epsilon = 1e-10
    target_magnitude = max(abs(target_mean), epsilon)
    baseline_magnitude = max(abs(baseline_mean), epsilon)
    magnitude = abs(np.log2(target_magnitude / baseline_magnitude))
    difference = target_mean - baseline_mean
    if difference == 0:
        return 0.0
    return float(np.copysign(magnitude, difference))


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
    if test not in {"wilcoxon", "ttest"}:
        raise ValueError("test must be 'wilcoxon' or 'ttest'")
    validated = validate_comparison_inputs(
        consistencies_matrix, group_A_cells, group_B_cells
    )
    # Reaction-level statistics are computed for the whole matrix at once.
    # Extracting each group as a reactions-by-samples array lets NumPy and the
    # axis-aware SciPy tests replace a per-reaction Python loop; the results are
    # numerically identical to testing one reaction at a time.
    reactions = consistencies_matrix.columns
    group_A = (
        consistencies_matrix.loc[list(validated.group_1_samples)]
        .to_numpy(dtype=float)
        .T
    )
    group_B = (
        consistencies_matrix.loc[list(validated.group_2_samples)]
        .to_numpy(dtype=float)
        .T
    )
    n_group_1 = group_A.shape[1]
    n_group_2 = group_B.shape[1]
    n_reactions = group_A.shape[0]

    mean_A = group_A.mean(axis=1)
    mean_B = group_B.mean(axis=1)
    mean_difference = mean_A - mean_B

    # A reaction whose two groups are each a single identical constant carries no
    # information. The per-reaction path skipped the test for it; we reproduce
    # that neutral result rather than feed a fully tied vector to SciPy.
    if n_reactions:
        a_constant = group_A.max(axis=1) == group_A.min(axis=1)
        b_constant = group_B.max(axis=1) == group_B.min(axis=1)
        neutral = a_constant & b_constant & (group_A[:, 0] == group_B[:, 0])
    else:
        neutral = np.zeros(0, dtype=bool)
    tested = ~neutral

    # Group A is Target and group B is Baseline; the Mann-Whitney call keeps the
    # original ``mannwhitneyu(B, A)`` orientation.
    test_stat = np.zeros(n_reactions, dtype=float)
    pval = np.ones(n_reactions, dtype=float)
    if tested.any():
        if test == "ttest":
            stat_tested, pval_tested = ttest_ind(
                group_A[tested], group_B[tested], axis=1, equal_var=False
            )
        else:  # default to 'wilcoxon' (Mann-Whitney)
            stat_tested, pval_tested = mannwhitneyu(
                group_B[tested], group_A[tested], alternative="two-sided", axis=1
            )
        test_stat[tested] = stat_tested
        pval[tested] = pval_tested

    # Cohen's d with a pooled standard deviation, matching ``cohens_d`` element
    # for element (NaN where the pooled variance is undefined or zero).
    if n_group_1 < 2 or n_group_2 < 2:
        cohens_d_values = np.full(n_reactions, np.nan)
    else:
        var_A = group_A.var(axis=1, ddof=1)
        var_B = group_B.var(axis=1, ddof=1)
        pooled_std = np.sqrt(
            ((n_group_1 - 1) * var_A + (n_group_2 - 1) * var_B)
            / (n_group_1 + n_group_2 - 2)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            cohens_d_values = mean_difference / pooled_std
        cohens_d_values[~np.isfinite(pooled_std) | (pooled_std == 0)] = np.nan

    # Signed absolute-mean fold change, vectorizing ``_signed_log2_fold_change``.
    epsilon = 1e-10
    target_magnitude = np.maximum(np.abs(mean_A), epsilon)
    baseline_magnitude = np.maximum(np.abs(mean_B), epsilon)
    magnitude = np.abs(np.log2(target_magnitude / baseline_magnitude))
    log2_fc = np.where(
        mean_difference == 0, 0.0, np.copysign(magnitude, mean_difference)
    )

    reversed_flag = (
        ((mean_A * mean_B) < 0)
        & (np.abs(mean_A) > epsilon)
        & (np.abs(mean_B) > epsilon)
    )

    # calculation_status / reason follow the same precedence as the per-reaction
    # loop: an invalid p-value wins over an undefined effect size, and a neutral
    # constant reaction overwrites either.
    calculation_status = np.full(n_reactions, "ok", dtype=object)
    reason = np.full(n_reactions, "", dtype=object)

    invalid_pval = tested & ~np.isfinite(pval)
    calculation_status[invalid_pval] = "invalid_test_result"
    reason[invalid_pval] = "scipy_returned_non_finite_pvalue"
    pval = np.where(np.isfinite(pval), pval, 1.0)
    test_stat = np.where(np.isfinite(test_stat), test_stat, 0.0)

    undefined_effect = tested & ~np.isfinite(cohens_d_values)
    calculation_status[undefined_effect & (calculation_status == "ok")] = (
        "undefined_effect_size"
    )
    reason[undefined_effect & (reason == "")] = "zero_pooled_variance"

    test_stat[neutral] = 0.0
    pval[neutral] = 1.0
    cohens_d_values[neutral] = 0.0
    log2_fc[neutral] = 0.0
    mean_difference[neutral] = 0.0
    reversed_flag[neutral] = False
    calculation_status[neutral] = "neutral_constant"
    reason[neutral] = "identical_constant_groups"

    # Correct finite values only and restore their original order.
    finite = np.isfinite(pval)
    adjusted = np.ones(n_reactions, dtype=float)
    if finite.any():
        adjusted[finite] = multipletests(pval[finite], method=correction)[1]

    results = pd.DataFrame(
        {
            "test_stat": test_stat,
            "pval": pval,
            "group_1_mean": mean_A,
            "group_2_mean": mean_B,
            "n_group_1": float(n_group_1),
            "n_group_2": float(n_group_2),
            "cohens_d": cohens_d_values,
            "log2_fc": log2_fc,
            "mean_difference": mean_difference,
            "reversed": reversed_flag.astype(bool),
            "calculation_status": calculation_status,
            "reason": reason,
            "adjusted_pval": adjusted,
        },
        index=reactions,
    )
    results.attrs["correction_method"] = correction
    results.attrs["test_method"] = test

    return results

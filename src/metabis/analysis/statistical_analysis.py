"""Statistical analysis module for group comparisons.

This module provides statistical test functionality, effect size calculations,
and significance determination for comparing groups of samples.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import special
from scipy.stats import mannwhitneyu, t as t_distribution, ttest_ind
from statsmodels.stats.multitest import multipletests

__all__ = [
    "CORRECTION_METHODS",
    "SUPPORTED_TESTS",
    "StatisticalAnalysis",
    "ComparisonInputs",
    "ModeratedFit",
    "validate_comparison_inputs",
    "comparison_test",
    "cohens_d",
    "moderated_linear_fit",
    "rank_sum_floor",
]


CORRECTION_METHODS = {
    "fdr_bh": "Benjamini-Hochberg (FDR)",
    "holm": "Holm-Bonferroni (FWER)",
}

# The two-group tests a comparison can run. ``moderated_t`` is limma's: a linear
# model per feature whose residual variance borrows strength from every feature
# (Smyth 2004), and the only one that can adjust for sample-sheet columns.
SUPPORTED_TESTS = ("wilcoxon", "ttest", "moderated_t")


def rank_sum_floor(n_target: int, n_baseline: int) -> float:
    """The smallest two-sided p-value the exact rank-sum test can return.

    Under the exact null distribution of the Mann-Whitney statistic (Mann and
    Whitney 1947) the most extreme ranking of ``n_target + n_baseline`` samples
    has probability ``1 / C(n, n_target)`` in each tail. No effect, however large,
    gives a smaller p-value, so groups this small cannot pass a correction.
    """
    if n_target < 1 or n_baseline < 1:
        raise ValueError("Both groups need at least one sample")
    return min(1.0, 2.0 / math.comb(n_target + n_baseline, n_target))


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
            test_type (str): Statistical test to use ('wilcoxon', 'ttest' or
                'moderated_t')

        Raises:
            ValueError: If the test type is not supported
        """
        if test_type.lower() not in SUPPORTED_TESTS:
            raise ValueError(
                f"Unsupported statistical test: {test_type}. Supported tests: {list(SUPPORTED_TESTS)}"
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
        covariates: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Run statistical test between two groups.

        Args:
            data_df (pd.DataFrame): DataFrame with features as rows and samples as columns
            group_1_samples (List[str]): List of sample IDs for group 1
            group_2_samples (List[str]): List of sample IDs for group 2
            test_type (str, optional): Override the default test type
            covariates (pd.DataFrame, optional): Numeric sample-sheet columns,
                indexed by sample, to adjust for; only the moderated t-test accepts them

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
            correction=correction,
            covariates=covariates,
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


def _trigamma_inverse(x: float) -> float:
    """Solve ``trigamma(y) = x`` by Newton's method, as limma's trigammaInverse."""
    if x > 1e7:
        return 1.0 / math.sqrt(x)
    if x < 1e-6:
        return 1.0 / x
    y = 0.5 + 1.0 / x
    for _ in range(50):
        tri = float(special.polygamma(1, y))
        dif = tri * (1.0 - tri / x) / float(special.polygamma(2, y))
        y += dif
        if -dif / y < 1e-8:
            break
    return y


def _prior_variance(variances: np.ndarray, df: float) -> tuple[float, float]:
    """Estimate the scaled-F prior of the residual variances (limma's fitFDist).

    Returns ``(df_prior, var_prior)``. ``df_prior`` is infinite when the observed
    variances spread no more than sampling alone explains, and every feature then
    takes the prior variance.
    """
    usable = variances[np.isfinite(variances) & (variances > -1e-15)]
    if usable.size < 2:
        return 0.0, float(usable.mean()) if usable.size else 1.0
    usable = np.maximum(usable, 0.0)
    median = float(np.median(usable))
    if median == 0:
        median = 1.0
    # Zero variances are offset away from zero, as limma does.
    usable = np.maximum(usable, 1e-5 * median)
    e = np.log(usable) - special.digamma(df / 2) + math.log(df / 2)
    mean = float(e.mean())
    spread = float(((e - mean) ** 2).sum() / (e.size - 1)) - float(
        special.polygamma(1, df / 2)
    )
    if spread > 0:
        df_prior = 2.0 * _trigamma_inverse(spread)
        var_prior = math.exp(
            mean + float(special.digamma(df_prior / 2)) - math.log(df_prior / 2)
        )
        return df_prior, var_prior
    # limma (since 2017) takes the pooled variance, the scale's MLE, here.
    return math.inf, float(usable.mean())


@dataclass(frozen=True)
class ModeratedFit:
    """One linear model per feature, tested with empirical-Bayes moderated t."""

    coefficient: np.ndarray
    t: np.ndarray
    pval: np.ndarray
    residual_variance: np.ndarray
    residual_df: int
    df_prior: float
    var_prior: float


def moderated_linear_fit(
    values: np.ndarray, design: np.ndarray, coefficient: int = 1
) -> ModeratedFit:
    """Fit ``values ~ design`` per feature and test one coefficient (limma).

    ``values`` is samples by features and ``design`` samples by parameters. Each
    feature gets an ordinary least-squares fit. Its residual variance is then
    shrunk toward a prior estimated from all features, ``(d0 s0^2 + d s^2) /
    (d0 + d)``, and the coefficient is tested with ``d + d0`` degrees of freedom
    (Smyth 2004, as limma's lmFit and eBayes without a variance trend).
    """
    values = np.asarray(values, dtype=float)
    design = np.asarray(design, dtype=float)
    n_samples, n_parameters = design.shape
    if values.shape[0] != n_samples:
        raise ValueError("The design has a different number of samples than the data")
    if np.linalg.matrix_rank(design) < n_parameters:
        raise ValueError(
            "The adjustment columns are collinear with the groups or with each other"
        )
    residual_df = n_samples - n_parameters
    if residual_df < 1:
        raise ValueError("Too few samples to fit the model: add samples or drop a column")
    unscaled = np.linalg.inv(design.T @ design)
    coefficients = unscaled @ design.T @ values
    residuals = values - design @ coefficients
    residual_variance = (residuals**2).sum(axis=0) / residual_df
    df_prior, var_prior = _prior_variance(residual_variance, float(residual_df))
    if math.isinf(df_prior):
        posterior = np.full_like(residual_variance, var_prior)
        df_total = float(residual_df * values.shape[1])
    else:
        posterior = (df_prior * var_prior + residual_df * residual_variance) / (
            df_prior + residual_df
        )
        # limma caps the total degrees of freedom at the pooled residual df.
        df_total = min(df_prior + residual_df, float(residual_df * values.shape[1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = coefficients[coefficient] / np.sqrt(
            posterior * unscaled[coefficient, coefficient]
        )
    t = np.where(np.isfinite(t), t, 0.0)
    pval = 2.0 * t_distribution.sf(np.abs(t), df_total)
    return ModeratedFit(
        coefficient=coefficients[coefficient],
        t=t,
        pval=pval,
        residual_variance=residual_variance,
        residual_df=residual_df,
        df_prior=df_prior,
        var_prior=var_prior,
    )


def _covariate_design(
    covariates: pd.DataFrame | None, samples: Sequence[str]
) -> np.ndarray:
    """Centred covariate columns for the tested samples, in their order."""
    if covariates is None or covariates.empty:
        return np.empty((len(samples), 0))
    missing = sorted(set(samples) - set(map(str, covariates.index)))
    if missing:
        raise ValueError(
            f"No adjustment value for {len(missing)} sample(s): {', '.join(missing[:5])}"
        )
    block = covariates.loc[list(samples)].to_numpy(dtype=float)
    if not np.isfinite(block).all():
        raise ValueError("Adjustment columns must be finite for every tested sample")
    constant = [
        str(name)
        for name, column in zip(covariates.columns, block.T)
        if np.ptp(column) == 0
    ]
    if constant:
        raise ValueError(
            f"Adjustment column {', '.join(constant)} has one value across these "
            "samples, so it cannot be adjusted for"
        )
    return block - block.mean(axis=0)


def comparison_test(
    consistencies_matrix,
    group_A_cells,
    group_B_cells,
    test="wilcoxon",
    correction="fdr_bh",
    covariates: pd.DataFrame | None = None,
):
    """Perform a two-group comparison between groups.

    Args:
        consistencies_matrix (pd.DataFrame): DataFrame with reaction activities
        group_A_cells (List[str]): List of cell IDs for group A
        group_B_cells (List[str]): List of cell IDs for group B
        test (str): 'wilcoxon' (rank-sum), 'ttest' (Welch) or 'moderated_t'
            (limma's linear model). Default 'wilcoxon'.
        correction (str): Type of p-value correction, either 'bonferroni' or 'fdr_bh' (default 'fdr_bh')
        covariates (pd.DataFrame, optional): Numeric columns indexed by sample to
            adjust for. Only the moderated t-test accepts them; its effect is then
            the group coefficient at equal covariate values.

    Returns:
                pd.DataFrame: Test results including:
                        - test statistic and p-value (Mann-Whitney U for
                            wilcoxon, Welch t for ttest, moderated t for moderated_t)
            - Cohen's d effect size
            - Log2 fold change
            - Mean difference
            - Reversed flag (True if the reaction changed direction)
            - Adjusted p-value
            - Z-scores
    """
    if test not in SUPPORTED_TESTS:
        raise ValueError(f"test must be one of {', '.join(SUPPORTED_TESTS)}")
    if covariates is not None and not covariates.empty and test != "moderated_t":
        raise ValueError("Only the moderated t-test can adjust for sample-sheet columns")
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
    fit: ModeratedFit | None = None
    adjusted_difference: np.ndarray | None = None
    if test == "moderated_t" and tested.any():
        # One model over every feature that varies: the prior variance is
        # estimated from all of them. Features constant across every tested
        # sample are left out, as limma's workflow filters unexpressed genes
        # before the fit (Law et al. 2016), and keep the neutral result below.
        samples = validated.tested_samples
        group = np.r_[np.ones(n_group_1), np.zeros(n_group_2)]
        adjustment = _covariate_design(covariates, samples)
        design = np.column_stack([np.ones(len(samples)), group, adjustment])
        fit = moderated_linear_fit(np.hstack([group_A, group_B])[tested].T, design)
        test_stat[tested] = fit.t
        pval[tested] = fit.pval
        if adjustment.shape[1]:
            adjusted_difference = np.zeros(n_reactions)
            adjusted_difference[tested] = fit.coefficient
    elif tested.any():
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
    # for element (NaN where the pooled variance is undefined or zero). With
    # adjustment columns the effect is the model's: the group coefficient, the
    # difference at equal covariate values, over the residual standard deviation.
    # Without them the two definitions are the same number.
    if adjusted_difference is not None:
        mean_difference = adjusted_difference
        residual_std = np.zeros(n_reactions)
        residual_std[tested] = np.sqrt(fit.residual_variance)
        with np.errstate(divide="ignore", invalid="ignore"):
            cohens_d_values = mean_difference / residual_std
        cohens_d_values[~np.isfinite(residual_std) | (residual_std == 0)] = np.nan
    elif n_group_1 < 2 or n_group_2 < 2:
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
    if fit is not None:
        results.attrs["moderation"] = {
            "residual_df": fit.residual_df,
            "df_prior": fit.df_prior,
            "var_prior": fit.var_prior,
            "covariates": (
                [str(name) for name in covariates.columns]
                if covariates is not None
                else []
            ),
        }

    return results

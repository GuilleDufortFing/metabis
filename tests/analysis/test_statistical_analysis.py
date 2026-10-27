import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests

from metabis.analysis.statistical_analysis import (
    comparison_test,
    moderated_linear_fit,
    rank_sum_floor,
)

GROUP_A = ["a1", "a2", "a3", "a4"]
GROUP_B = ["b1", "b2", "b3", "b4"]


def comparison_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_up": [1.0, 2.0, 1.5, 2.5, 6.0, 7.0, 6.5, 8.0],
            "feature_down": [9.0, 8.0, 10.0, 9.5, 3.0, 4.0, 2.0, 3.5],
        },
        index=GROUP_A + GROUP_B,
    )


def test_welch_results_match_scipy():
    frame = comparison_frame()

    result = comparison_test(frame, GROUP_A, GROUP_B, test="ttest", correction="holm")

    expected = ttest_ind(
        frame.loc[GROUP_A, "feature_up"],
        frame.loc[GROUP_B, "feature_up"],
        equal_var=False,
    )
    assert result.loc["feature_up", "test_stat"] == expected.statistic
    assert result.loc["feature_up", "pval"] == expected.pvalue


def test_mann_whitney_results_match_scipy_and_holm_matches_statsmodels():
    frame = comparison_frame()

    result = comparison_test(
        frame, GROUP_A, GROUP_B, test="wilcoxon", correction="holm"
    )

    expected = mannwhitneyu(
        frame.loc[GROUP_B, "feature_up"],
        frame.loc[GROUP_A, "feature_up"],
        alternative="two-sided",
    )
    assert result.loc["feature_up", "test_stat"] == expected.statistic
    assert result.loc["feature_up", "pval"] == expected.pvalue
    np.testing.assert_allclose(
        result["adjusted_pval"], multipletests(result["pval"], method="holm")[1]
    )


def test_identical_constant_feature_returns_finite_neutral_result():
    frame = pd.DataFrame(
        {"constant": [3.0, 3.0, 3.0, 3.0]}, index=["a1", "a2", "b1", "b2"]
    )

    result = comparison_test(frame, ["a1", "a2"], ["b1", "b2"])

    assert result.loc["constant", "pval"] == 1
    assert result.loc["constant", "adjusted_pval"] == 1
    assert result.loc["constant", "cohens_d"] == 0


def test_identical_constant_feature_after_variable_feature_is_supported():
    frame = pd.DataFrame(
        {
            "variable": [1.0, 2.0, 4.0, 5.0],
            "constant": [3.0, 3.0, 3.0, 3.0],
        },
        index=["a1", "a2", "b1", "b2"],
    )

    result = comparison_test(frame, ["a1", "a2"], ["b1", "b2"])

    assert result.loc["variable", "calculation_status"] == "ok"
    assert result.loc["constant", "calculation_status"] == "neutral_constant"
    assert result.loc["constant", "reason"] == "identical_constant_groups"


def test_different_constants_do_not_produce_infinite_effect_sizes():
    frame = pd.DataFrame(
        {"constant": [1.0, 1.0, 2.0, 2.0]}, index=["a1", "a2", "b1", "b2"]
    )
    result = comparison_test(frame, ["a1", "a2"], ["b1", "b2"], test="ttest")

    assert np.isnan(result.loc["constant", "cohens_d"])
    assert np.isfinite(result.loc["constant", "pval"])
    assert result.loc["constant", "reason"] == "zero_pooled_variance"


def test_signed_effects_are_group_1_minus_group_2_for_nonnegative_activity():
    frame = pd.DataFrame(
        {"target_higher": [10.0, 12.0, 1.0, 3.0]},
        index=["target_1", "target_2", "baseline_1", "baseline_2"],
    )

    result = comparison_test(
        frame,
        ["target_1", "target_2"],
        ["baseline_1", "baseline_2"],
    ).loc["target_higher"]

    assert result["group_1_mean"] == 11.0
    assert result["group_2_mean"] == 2.0
    assert result["mean_difference"] == 9.0
    assert result["cohens_d"] > 0
    assert result["log2_fc"] == pytest.approx(np.log2(11.0 / 2.0))


@pytest.mark.parametrize(
    ("target_values", "baseline_values", "expected_sign", "is_reversed"),
    [
        ([-3.0, -1.0], [-6.0, -4.0], 1, False),
        ([4.0, 6.0], [-3.0, -1.0], 1, True),
        ([-6.0, -4.0], [1.0, 3.0], -1, True),
    ],
)
def test_signed_inputs_keep_target_minus_baseline_direction(
    target_values, baseline_values, expected_sign, is_reversed
):
    frame = pd.DataFrame(
        {"signed": target_values + baseline_values},
        index=["target_1", "target_2", "baseline_1", "baseline_2"],
    )

    result = comparison_test(
        frame,
        ["target_1", "target_2"],
        ["baseline_1", "baseline_2"],
    ).loc["signed"]

    assert np.sign(result["mean_difference"]) == expected_sign
    assert np.sign(result["cohens_d"]) == expected_sign
    assert np.sign(result["log2_fc"]) == expected_sign
    assert bool(result["reversed"]) is is_reversed


@pytest.mark.parametrize(
    ("group_1", "group_2", "message"),
    [
        ([], ["b1", "b2"], "cannot be empty"),
        (["a1", "a2"], ["a2", "b1"], "overlap"),
        (["a1", "missing"], ["b1", "b2"], "Unknown"),
        (["a1"], ["b1", "b2"], "at least 2"),
    ],
)
def test_invalid_groups_fail_before_analysis(group_1, group_2, message):
    with pytest.raises(ValueError, match=message):
        comparison_test(comparison_frame(), group_1, group_2)


# Ten samples, eight features: three shifted in group 1, one following ``DEPTH``.
LIMMA_Y = np.array(
    [
        [1.42, 4.29, 1.97, 1.59, -0.65, -0.2, 0.48, -0.03],
        [1.81, -2.53, 2.13, 2.5, 1.49, -0.05, -0.32, 0.21],
        [1.85, 0.97, 1.33, 3.05, -1.9, -0.58, 0.33, -0.31],
        [0.35, -0.33, 1.18, 1.18, -3.26, 0.01, 0.75, -0.11],
        [0.99, 2.22, 1.74, 2.0, 1.19, 0.4, -0.17, -0.37],
        [0.19, 0.53, 0.52, 0.06, -1.44, -0.32, -1.45, 0.06],
        [0.29, -1.57, 0.65, 3.35, 1.37, 0.15, 0.8, -0.61],
        [0.34, 1.28, -0.83, 2.14, -0.55, 0.3, -0.37, -0.01],
        [0.19, -1.86, 0.28, 1.83, 1.08, -0.2, 0.91, 0.28],
        [-0.1, 1.34, 0.59, 5.05, -3.44, 0.34, 0.39, -0.04],
    ]
)
DEPTH = np.array([2.1, 2.4, 1.9, 2.6, 2.2, 1.7, 2.0, 1.5, 1.8, 2.3])
LIMMA_GROUP = np.r_[np.ones(5), np.zeros(5)]
# limma 3.x lmFit + eBayes, through its Python port inmoose 0.9.1.
LIMMA_PLAIN_T = [3.6486937977, 0.7940927818, 4.2771898634, -0.5106151064,
                 -0.0258842195, -0.5796445677, 0.3414301675, -0.2647092347]
LIMMA_PLAIN_P = [0.0047646850351296, 0.4462807601837128, 0.0017662961896316163,
                 0.6211278298471969, 0.9798785421240301, 0.5754717741582852,
                 0.7401172470027838, 0.7968163014341263]
LIMMA_DEPTH_T = [4.3035416469, 1.2388919808, 2.8115869116, -1.1583040091,
                 0.4480935044, -1.1254050805, -0.3662159098, -0.1884931556]
LIMMA_DEPTH_P = [0.0022031199365148938, 0.2481472351645825, 0.02124001382537564,
                 0.27790809063975386, 0.6651658870616479, 0.2908578033792372,
                 0.7230548028027801, 0.8548648253381157]
LIMMA_DEPTH_COEF = [1.4564759036, 1.8996686747, 1.1318975904, -1.1581927711,
                    0.6590361446, -0.3271987952, -0.2084939759, -0.0537650602]


def test_moderated_fit_matches_limma():
    design = np.column_stack([np.ones(10), LIMMA_GROUP])

    fit = moderated_linear_fit(LIMMA_Y, design)

    np.testing.assert_allclose(fit.t, LIMMA_PLAIN_T, atol=1e-9)
    np.testing.assert_allclose(fit.pval, LIMMA_PLAIN_P, rtol=1e-9)
    assert fit.df_prior == pytest.approx(1.6234361722103827)
    assert fit.var_prior == pytest.approx(0.31046243608237806)


def test_moderated_fit_with_a_covariate_matches_limma():
    design = np.column_stack(
        [np.ones(10), LIMMA_GROUP, DEPTH - DEPTH.mean()]
    )

    fit = moderated_linear_fit(LIMMA_Y, design)

    np.testing.assert_allclose(fit.t, LIMMA_DEPTH_T, atol=1e-9)
    np.testing.assert_allclose(fit.pval, LIMMA_DEPTH_P, rtol=1e-9)
    np.testing.assert_allclose(fit.coefficient, LIMMA_DEPTH_COEF, atol=1e-9)


def test_equal_variances_give_an_infinite_prior_as_limma():
    rng = np.random.default_rng(3)
    group = np.r_[np.ones(5), np.zeros(5)]
    base = rng.normal(size=10)
    columns = []
    for shift in range(20):
        column = base.copy()
        column[:5] = rng.permutation(column[:5])
        column[5:] = rng.permutation(column[5:])
        columns.append(column + (shift % 4) * group)

    fit = moderated_linear_fit(
        np.column_stack(columns), np.column_stack([np.ones(10), group])
    )

    assert fit.df_prior == np.inf
    assert fit.var_prior == pytest.approx(fit.residual_variance.mean())


def test_moderated_comparison_without_covariates_keeps_the_classic_effects():
    frame = pd.DataFrame(
        LIMMA_Y, index=[f"s{i}" for i in range(10)], columns=[f"f{i}" for i in range(8)]
    )
    target, baseline = list(frame.index[:5]), list(frame.index[5:])

    moderated = comparison_test(frame, target, baseline, test="moderated_t")
    welch = comparison_test(frame, target, baseline, test="ttest")

    np.testing.assert_allclose(moderated["test_stat"], LIMMA_PLAIN_T, atol=1e-9)
    np.testing.assert_allclose(moderated["cohens_d"], welch["cohens_d"])
    np.testing.assert_allclose(moderated["mean_difference"], welch["mean_difference"])
    assert moderated.attrs["test_method"] == "moderated_t"
    assert moderated.attrs["moderation"]["covariates"] == []


def test_adjusted_comparison_reports_the_difference_at_equal_covariates():
    index = [f"s{i}" for i in range(10)]
    frame = pd.DataFrame(LIMMA_Y, index=index, columns=[f"f{i}" for i in range(8)])
    covariates = pd.DataFrame({"depth": DEPTH}, index=index)

    result = comparison_test(
        frame, index[:5], index[5:], test="moderated_t", covariates=covariates
    )

    np.testing.assert_allclose(result["test_stat"], LIMMA_DEPTH_T, atol=1e-9)
    np.testing.assert_allclose(result["mean_difference"], LIMMA_DEPTH_COEF, atol=1e-9)
    assert result.attrs["moderation"]["covariates"] == ["depth"]
    # The raw means are still reported as measured.
    assert result.loc["f0", "group_1_mean"] == pytest.approx(LIMMA_Y[:5, 0].mean())


def test_constant_features_stay_neutral_and_out_of_the_moderated_fit():
    index = [f"s{i}" for i in range(10)]
    frame = pd.DataFrame(LIMMA_Y, index=index, columns=[f"f{i}" for i in range(8)])
    frame["constant"] = 2.0

    result = comparison_test(frame, index[:5], index[5:], test="moderated_t")

    assert result.loc["constant", "calculation_status"] == "neutral_constant"
    assert result.loc["constant", "pval"] == 1.0
    np.testing.assert_allclose(
        result.loc[[f"f{i}" for i in range(8)], "test_stat"], LIMMA_PLAIN_T, atol=1e-9
    )


@pytest.mark.parametrize(
    ("covariates", "test", "message"),
    [
        (pd.DataFrame({"depth": DEPTH}), "wilcoxon", "Only the moderated t-test"),
        (pd.DataFrame({"depth": np.ones(10)}), "moderated_t", "one value across"),
        (pd.DataFrame({"depth": DEPTH, "twice": 2 * DEPTH}), "moderated_t", "collinear"),
        (pd.DataFrame({"group": LIMMA_GROUP}), "moderated_t", "collinear"),
        (pd.DataFrame({"depth": DEPTH}).iloc[:9], "moderated_t", "No adjustment value"),
    ],
)
def test_invalid_adjustment_is_rejected(covariates, test, message):
    index = [f"s{i}" for i in range(10)]
    frame = pd.DataFrame(LIMMA_Y, index=index, columns=[f"f{i}" for i in range(8)])
    covariates = covariates.set_axis(index[: len(covariates)])

    with pytest.raises(ValueError, match=message):
        comparison_test(frame, index[:5], index[5:], test=test, covariates=covariates)


@pytest.mark.parametrize(
    ("n_target", "n_baseline", "floor"),
    [(3, 3, 0.1), (2, 2, 1 / 3), (5, 5, 2 / 252), (1, 1, 1.0)],
)
def test_rank_sum_floor_is_the_most_extreme_exact_p_value(n_target, n_baseline, floor):
    assert rank_sum_floor(n_target, n_baseline) == pytest.approx(floor)
    if n_target > 1:
        exact = mannwhitneyu(
            np.arange(n_target) + 100.0, np.arange(n_baseline), method="exact"
        ).pvalue
        assert exact == pytest.approx(floor)

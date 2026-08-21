import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests

from metabis.analysis.statistical_analysis import comparison_test

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

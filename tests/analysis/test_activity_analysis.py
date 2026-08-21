"""Hand-calculated contract for normalized single-group activity summaries."""

import numpy as np
import pandas as pd
import pytest

from metabis.analysis.activity_analysis import (
    ActivityConfig,
    ActivityOverlay,
    GroupSummary,
    calculate_feature_activity,
    calculate_subsystem_activity,
    displayed_feature_id,
    resolve_display_direction,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    """Four samples; two groups of two. Every column is hand-checkable.

    constant  : identical everywhere and remains 5 on the normalized input scale
    high_in_a : group A mean is 9
    low_in_a  : group A mean is 2
    """
    return pd.DataFrame(
        {
            "constant": [5.0, 5.0, 5.0, 5.0],
            "high_in_a": [8.0, 10.0, 2.0, 0.0],
            "low_in_a": [1.0, 3.0, 9.0, 7.0],
        },
        index=["a1", "a2", "b1", "b2"],
    )


GROUP_A = ["a1", "a2"]


def config(**overrides) -> ActivityConfig:
    base = dict(
        group_id="A",
        summary=GroupSummary.MEAN,
    )
    base.update(overrides)
    return ActivityConfig(**base)


def test_group_summary_uses_the_selected_statistic(frame):
    mean_result = calculate_feature_activity(frame, GROUP_A, config())
    median_result = calculate_feature_activity(
        frame, GROUP_A, config(summary=GroupSummary.MEDIAN)
    )

    # Group A of high_in_a is [8, 10].
    assert mean_result.score("high_in_a") == pytest.approx(9.0)
    assert median_result.score("high_in_a") == pytest.approx(9.0)
    assert mean_result.sample_count == 2
    assert mean_result.reference_sample_count == 4


def test_activity_score_is_the_normalized_group_summary_without_standardization(frame):
    result = calculate_feature_activity(frame, GROUP_A, config())

    assert result.score("high_in_a") == pytest.approx(9.0)
    assert result.score("low_in_a") == pytest.approx(2.0)
    assert result.score("constant") == pytest.approx(5.0)
    assert result.non_finite_feature_ids == ()


def test_a_group_with_no_samples_in_the_matrix_is_rejected(frame):
    with pytest.raises(ValueError, match="no samples in the aligned matrix"):
        calculate_feature_activity(frame, ["missing_sample"], config())


def test_every_scoring_input_is_part_of_the_cache_identity():
    keys = {
        config().metric_key,
        config(group_id="B").metric_key,
        config(summary=GroupSummary.MEDIAN).metric_key,
        config(minimum_subsystem_coverage=0.5).metric_key,
    }

    # Changing any one of them must produce a different key, or one context's
    # colors could be reused for another.
    assert len(keys) == 4
    assert config().metric_key_for(ActivityOverlay.REACTION) != config().metric_key_for(
        ActivityOverlay.GENE
    )


def test_a_base_group_carries_no_explicit_population():
    base = config()

    assert base.has_explicit_population is False
    assert base.sample_ids == ()
    assert base.population_fingerprint == ""
    # The key of a base group is unchanged, so existing caches stay valid.
    assert base.metric_key == "activity|A|mean|coverage=0.1"


def test_an_explicit_population_is_part_of_the_cache_identity():
    """Two comparisons can label a population identically; membership cannot."""
    labelled = config(group_id="Target: A + B", sample_ids=("a1", "a2"))
    same_label_other_members = config(group_id="Target: A + B", sample_ids=("b1", "b2"))

    assert labelled.has_explicit_population is True
    assert labelled.metric_key != same_label_other_members.metric_key
    # A base group named like the population is still a different calculation.
    assert labelled.metric_key != config(group_id="Target: A + B").metric_key


def test_an_explicit_population_is_a_set_so_its_identity_ignores_order():
    assert (
        config(group_id="P", sample_ids=("a2", "a1")).metric_key
        == config(group_id="P", sample_ids=("a1", "a2")).metric_key
    )


def test_an_explicit_population_may_not_repeat_a_sample():
    with pytest.raises(ValueError, match="must not repeat a sample"):
        config(group_id="P", sample_ids=("a1", "a1"))


def test_an_explicit_population_rejects_a_bare_string():
    with pytest.raises(TypeError, match="not a string"):
        config(group_id="P", sample_ids="a1")


def test_an_explicit_population_scores_exactly_the_samples_it_names(frame):
    """A comparison-local population is scored from the config's own membership."""
    result = calculate_feature_activity(
        frame,
        ["a1", "b1"],
        config(group_id="Target: A + B", sample_ids=("a1", "b1")),
    )

    # high_in_a over [a1, b1] is mean(8, 2).
    assert result.score("high_in_a") == pytest.approx(5.0)
    assert result.sample_count == 2


def test_reversible_direction_follows_the_selected_group_summary():
    raw = pd.Series({"MAR1": 2.0, "MAR1_back": -5.0, "MAR2": 3.0})

    assert resolve_display_direction(raw, "MAR1") == "backward"
    assert displayed_feature_id(raw, "MAR1") == "MAR1_back"
    # A reaction with no backward column is always forward.
    assert resolve_display_direction(raw, "MAR2") == "forward"
    assert displayed_feature_id(raw, "MAR2") == "MAR2"


def test_forward_wins_ties_and_back_ids_resolve_to_the_same_display():
    raw = pd.Series({"MAR1": 5.0, "MAR1_back": -5.0})

    assert resolve_display_direction(raw, "MAR1") == "forward"
    assert displayed_feature_id(raw, "MAR1_back") == "MAR1"


def _reaction_result(coverage=0.1):
    frame = pd.DataFrame(
        {
            "MAR1": [2.0, 4.0],
            "MAR1_back": [-9.0, -9.0],
            "MAR2": [1.0, 1.0],
            "MAR3": [-3.0, -1.0],
        },
        index=["a1", "a2"],
    )
    return calculate_feature_activity(
        frame,
        ["a1", "a2"],
        ActivityConfig(
            group_id="A",
            summary=GroupSummary.MEAN,
            minimum_subsystem_coverage=coverage,
        ),
    )


def test_subsystem_activity_is_the_median_of_measured_members_and_reports_coverage():
    result = _reaction_result()

    frame = calculate_subsystem_activity(
        result,
        {"glycolysis": ["MAR1", "MAR2", "MAR3"]},
    )
    row = frame.loc["glycolysis"]

    # MAR1's backward summary (-9) beats its forward summary (3), so the backward
    # value is the one the group displays. Members are then [-9, 1, -2].
    assert row["activity_score"] == pytest.approx(-2.0)
    assert row["measured_features"] == 3
    assert row["total_model_features"] == 3
    assert row["coverage"] == pytest.approx(1.0)
    assert row["positive_count"] == 1
    assert row["negative_count"] == 2


def test_a_subsystem_below_the_coverage_threshold_reports_no_score():
    result = _reaction_result(coverage=0.5)

    frame = calculate_subsystem_activity(
        result,
        {"sparse pathway": ["MAR2"] + [f"UNMEASURED{index}" for index in range(9)]},
    )
    row = frame.loc["sparse pathway"]

    # 1 of 10 model reactions measured = 10% coverage, below the 50% threshold.
    assert pd.isna(row["activity_score"])
    assert row["measured_features"] == 1
    assert row["coverage"] == pytest.approx(0.1)


def test_unmeasured_subsystem_members_are_skipped_not_treated_as_zero():
    result = _reaction_result()

    frame = calculate_subsystem_activity(
        result,
        {"partial": ["MAR2", "NOT_IN_MATRIX"]},
    )
    row = frame.loc["partial"]

    assert row["activity_score"] == pytest.approx(1.0)
    assert row["measured_features"] == 1
    assert row["coverage"] == pytest.approx(0.5)


def test_gene_overlay_aggregates_genes_without_direction_resolution():
    frame = pd.DataFrame(
        {"GENE1": [1.0, 3.0], "GENE2": [5.0, 5.0]},
        index=["a1", "a2"],
    )
    result = calculate_feature_activity(
        frame,
        ["a1", "a2"],
        ActivityConfig(
            group_id="A",
            summary=GroupSummary.MEAN,
        ),
        overlay=ActivityOverlay.GENE,
    )

    subsystems = calculate_subsystem_activity(result, {"glycolysis": ["GENE1", "GENE2"]})

    assert result.overlay == "gene"
    assert subsystems.loc["glycolysis", "activity_score"] == pytest.approx(3.5)


def test_non_finite_group_summaries_remain_unavailable_and_are_audited():
    frame = pd.DataFrame(
        {"rxn": [np.nan, np.nan, 1.0, 5.0]},
        index=["a1", "a2", "b1", "b2"],
    )

    result = calculate_feature_activity(frame, ["a1", "a2"], config())

    assert result.score("rxn") is None
    assert "rxn" in result.non_finite_feature_ids
    assert any("remain unavailable" in warning for warning in result.warnings)

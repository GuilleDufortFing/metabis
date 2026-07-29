"""Quantile normalization of flux scores (R-01, rebuttal Major 2, Supp. 1.2).

The revision lists quantile normalization among the reaction transforms Metabis
offers, defined as mapping "the values of each sample onto a common reference
distribution, defined as the mean of the sorted values across samples".

The axis is the whole point. ``RANK_PERCENTILE`` ranks one reaction *across
samples*; quantile normalization ranks one sample *across reactions*. They answer
different questions and both stay available, so several tests below exist only to
pin that the two cannot be confused for one another.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metabis import (
    ReactionPreprocessingConfig,
    ReactionTransform,
    preprocess_reactions,
    quantile_normalize,
    rank_percentile_by_feature,
)


def _frame() -> pd.DataFrame:
    """Three samples on visibly different scales, with no ties in any of them.

    Tie-free matters: with ties the mapped values are averages of reference
    entries rather than the entries themselves, so the "each sample ends up
    holding exactly the reference" assertions below need a frame without them.
    Ties get their own tests.
    """
    return pd.DataFrame(
        {
            "R1": [5.0, 4.0, 3.0],
            "R2": [2.0, 1.0, 4.0],
            "R3": [3.0, 7.0, 6.0],
            "R4": [4.0, 2.0, 8.0],
        },
        index=["s1", "s2", "s3"],
    )


def test_it_maps_every_sample_onto_the_mean_of_the_sorted_values():
    frame = _frame()

    result = quantile_normalize(frame)

    # The reference is built exactly as the supplement defines it.
    reference = np.sort(frame.to_numpy(), axis=1).mean(axis=0)
    for sample in frame.index:
        np.testing.assert_allclose(
            np.sort(result.loc[sample].to_numpy()), reference, atol=1e-12
        )


def test_every_sample_ends_with_the_same_distribution():
    result = quantile_normalize(_frame())

    summaries = {
        "mean": result.mean(axis=1).round(9).unique(),
        "std": result.std(axis=1).round(9).unique(),
        "min": result.min(axis=1).round(9).unique(),
        "max": result.max(axis=1).round(9).unique(),
    }
    for name, values in summaries.items():
        assert len(values) == 1, f"samples disagree on {name} after normalization"


def test_every_sample_keeps_the_same_total_even_when_ties_are_present():
    """The invariant that survives ties, and the one real data exercises.

    With ties the sorted rows are no longer identical -- tied reactions take a
    range mean instead of the reference entries themselves -- so "same
    distribution" is only exact on tie-free input. What holds regardless is that
    each row is a partition-average of the same reference, so every sample's total
    is the reference total. A Compass matrix is full of exact zeros, so this is
    the property that actually pins the transform on the bundled examples.
    """
    frame = pd.DataFrame(
        {
            "R1": [0.0, 0.0, 3.0],
            "R2": [0.0, 5.0, 3.0],
            "R3": [0.0, 5.0, 9.0],
            "R4": [8.0, 5.0, 1.0],
            "R5": [2.0, 1.0, 1.0],
        },
        index=["s1", "s2", "s3"],
    )

    result = quantile_normalize(frame)

    reference = np.sort(frame.to_numpy(), axis=1).mean(axis=0)
    np.testing.assert_allclose(
        result.sum(axis=1).to_numpy(), np.full(3, reference.sum()), atol=1e-12
    )
    # The tie structure differs between samples, so this frame really does
    # exercise the tie path rather than reducing to the tie-free case.
    assert (result.loc["s1"].nunique(), result.loc["s3"].nunique()) != (5, 5)


def test_it_preserves_the_ordering_of_reactions_within_each_sample():
    frame = _frame()

    result = quantile_normalize(frame)

    # Only the values are replaced; which reaction is largest in a sample is the
    # information the transform is meant to keep.
    for sample in frame.index:
        assert list(frame.loc[sample].rank()) == list(result.loc[sample].rank())


def test_ties_receive_the_average_of_the_reference_entries_they_span():
    frame = pd.DataFrame(
        {"R1": [1.0, 1.0], "R2": [1.0, 2.0], "R3": [3.0, 3.0], "R4": [4.0, 4.0]},
        index=["s1", "s2"],
    )

    result = quantile_normalize(frame)

    reference = np.sort(frame.to_numpy(), axis=1).mean(axis=0)
    # s1 is 1, 1, 3, 4: R1 and R2 tie for the two lowest sorted positions, so both
    # take the mean of the reference over exactly those two positions.
    expected_tie = reference[:2].mean()
    assert result.loc["s1", "R1"] == pytest.approx(expected_tie)
    assert result.loc["s1", "R2"] == pytest.approx(expected_tie)
    # Equal inputs must stay equal, which is the reason ties are averaged rather
    # than broken by column order.
    assert result.loc["s1", "R1"] == result.loc["s1", "R2"]


def test_it_is_a_fixed_point_of_itself():
    once = quantile_normalize(_frame())

    twice = quantile_normalize(once)

    pd.testing.assert_frame_equal(once, twice, atol=1e-12)


def test_it_preserves_the_axes_and_both_indices_exactly():
    frame = _frame()

    result = quantile_normalize(frame)

    assert result.shape == frame.shape
    assert result.index.equals(frame.index)
    assert result.columns.equals(frame.columns)


def test_a_constant_sample_stays_constant_at_the_reference_mean():
    frame = pd.DataFrame(
        {"R1": [7.0, 1.0], "R2": [7.0, 2.0], "R3": [7.0, 6.0]},
        index=["flat", "varied"],
    )

    result = quantile_normalize(frame)

    reference = np.sort(frame.to_numpy(), axis=1).mean(axis=0)
    # Every reaction of the flat sample is tied, so all take the mean of the whole
    # reference: a sample with no internal ordering cannot acquire one here.
    assert result.loc["flat"].nunique() == 1
    assert result.loc["flat", "R1"] == pytest.approx(reference.mean())


def test_a_single_sample_is_rejected_by_name_rather_than_silently_passed_through():
    frame = pd.DataFrame({"R1": [1.0], "R2": [2.0]}, index=["only"])

    with pytest.raises(ValueError, match="at least two samples"):
        quantile_normalize(frame)


def test_non_finite_input_is_rejected_through_the_shared_validation_seam():
    frame = pd.DataFrame(
        {"R1": [1.0, np.nan], "R2": [2.0, 3.0]}, index=["s1", "s2"]
    )

    with pytest.raises(ValueError, match="Quantile-normalization input"):
        quantile_normalize(frame)


def test_a_frame_with_no_reactions_returns_empty_rather_than_failing():
    frame = pd.DataFrame(index=["s1", "s2"], dtype=float)

    result = quantile_normalize(frame)

    assert result.shape == (2, 0)
    assert result.index.equals(frame.index)


# ---------------------------------------------------------------------------
# The two rank transforms are different operations on different axes
# ---------------------------------------------------------------------------


def test_it_is_not_the_rank_percentile_transform():
    frame = _frame()

    quantile = quantile_normalize(frame)
    percentile = rank_percentile_by_feature(frame)

    # Rank percentile equalizes reactions; quantile equalizes samples. If these
    # ever agree on this frame, one of them is reading the wrong axis.
    assert not np.allclose(quantile.to_numpy(), percentile.to_numpy())
    assert percentile.max(axis=0).nunique() == 1  # per-reaction scale
    assert quantile.max(axis=1).nunique() == 1  # per-sample scale


# ---------------------------------------------------------------------------
# Pipeline and provenance
# ---------------------------------------------------------------------------


def test_the_pipeline_records_the_transform_and_the_axis_it_read():
    frame = _frame()

    result, report = preprocess_reactions(
        frame, ReactionPreprocessingConfig(transform=ReactionTransform.QUANTILE)
    )

    assert report.method == "quantile"
    # The axis must be in the provenance, not left to be inferred from the name:
    # an exported result read along the wrong axis is a different claim.
    assert report.parameters["rank_axis"] == "within_sample_across_reactions"
    assert report.input_shape == frame.shape
    assert report.output_shape == frame.shape
    assert report.warnings == ()
    pd.testing.assert_frame_equal(result, quantile_normalize(frame))


def test_the_config_defaults_each_rank_transform_to_its_own_axis():
    assert (
        ReactionPreprocessingConfig(transform=ReactionTransform.QUANTILE).rank_axis
        == "within_sample_across_reactions"
    )
    assert (
        ReactionPreprocessingConfig(
            transform=ReactionTransform.RANK_PERCENTILE
        ).rank_axis
        == "within_reaction_across_samples"
    )
    # A transform that does not rank anything gains no axis.
    assert (
        ReactionPreprocessingConfig(transform=ReactionTransform.NONE).rank_axis is None
    )


def test_the_config_rejects_the_other_transform_axis():
    with pytest.raises(ValueError, match="across reactions within samples"):
        ReactionPreprocessingConfig(
            transform=ReactionTransform.QUANTILE,
            rank_axis="within_reaction_across_samples",
        )
    with pytest.raises(ValueError, match="across samples within reactions"):
        ReactionPreprocessingConfig(
            transform=ReactionTransform.RANK_PERCENTILE,
            rank_axis="within_sample_across_reactions",
        )


def test_the_low_range_filter_still_applies_after_normalization():
    frame = _frame()
    frame["flat"] = [9.0, 9.0, 9.0]

    _result, report = preprocess_reactions(
        frame,
        ReactionPreprocessingConfig(
            transform=ReactionTransform.QUANTILE,
            filter_low_variance=True,
            minimum_feature_range=0.5,
        ),
    )

    # The filter reads the *normalized* values, as it does for every other
    # transform, so what it removes is what the user would have seen. A reaction
    # constant across samples stays constant under a sample-wise transform.
    assert report.filtered_feature_ids == ("flat",)
    assert report.output_shape == (frame.shape[0], frame.shape[1] - 1)

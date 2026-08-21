import numpy as np
import pandas as pd
import pytest

from metabis import (
    COMPASS_MINIMUM_RANGE,
    ReactionPreprocessingConfig,
    ReactionTransform,
    blocked_compass_reactions,
    compass_negative_log,
    partially_blocked_compass_reactions,
    preprocess_reactions,
    rank_percentile_by_feature,
    signed_cuberoot,
)


def test_compass_activity_scores_are_non_negative_and_filter_features_not_samples():
    frame = pd.DataFrame(
        {"variable": [1.0, 2.0, 3.0], "constant": [2.0, 2.0, 2.0]},
        index=["s1", "s2", "s3"],
    )
    result, report = preprocess_reactions(
        frame,
        ReactionPreprocessingConfig(
            transform=ReactionTransform.COMPASS_NEG_LOG,
            filter_low_variance=True,
            minimum_feature_range=0.1,
        ),
    )

    assert result.index.tolist() == frame.index.tolist()
    assert result.columns.tolist() == ["variable"]
    raw_scores = -np.log1p(frame.astype(float))
    expected = raw_scores - raw_scores["variable"].min()
    np.testing.assert_allclose(result["variable"], expected["variable"])
    assert (result.to_numpy() >= 0).all()
    assert result.loc["s1", "variable"] > result.loc["s2", "variable"]
    assert result.loc["s2", "variable"] > result.loc["s3", "variable"]
    assert report.filtered_feature_ids == ("constant",)
    assert report.parameters["formula"] == "-log(x + 1) - min(-log(x + 1))"
    assert report.parameters["global_offset"] == pytest.approx(np.log(4.0))


def compass_frame() -> pd.DataFrame:
    """One blocked reaction, one that is only sometimes cheap, one costly one.

    Compass emits a penalty of exactly 0 for a reaction it could not make carry any
    flux, so "blocked" is zero in *every* sample.
    """
    return pd.DataFrame(
        {
            "blocked": [0.0, 0.0, 0.0],
            "variable": [0.5, 2.0, 7.0],
            "costly": [10.0, 12.0, 20.0],
        },
        index=["s1", "s2", "s3"],
    )


def test_blocked_reactions_are_dropped_by_default_not_scored_as_maximally_active():
    frame = compass_frame()

    result, report = preprocess_reactions(
        frame,
        ReactionPreprocessingConfig(transform=ReactionTransform.COMPASS_NEG_LOG),
    )

    # A zero penalty means Compass could not make the reaction carry flux, so it
    # is pinned to the activity floor. Blocked in every sample, it is then
    # constant there and the upstream low-range filter removes it.
    assert "blocked" not in result.columns
    assert "blocked" in report.filtered_feature_ids
    assert report.parameters["blocked_reactions"] == 1
    assert result.columns.tolist() == ["variable", "costly"]
    assert any("blocked reaction" in warning for warning in report.warnings)


def test_a_zero_penalty_is_the_blocked_sentinel_not_the_cheapest_penalty():
    # -log1p(0) = 0 is the *maximum* of the -log1p scale, so read literally a
    # reaction that cannot carry any flux would score as the most active one in
    # the dataset. Zero is a sentinel, not a point on the penalty scale.
    frame = pd.DataFrame(
        {"partly_blocked": [0.0, 4.0], "measured": [1.0, 9.0]},
        index=["s1", "s2"],
    )

    scores = compass_negative_log(frame)

    # The floor is set by the largest penalty measured anywhere, 9.
    assert scores.loc["s1", "partly_blocked"] == pytest.approx(0.0)
    assert scores.loc["s2", "measured"] == pytest.approx(0.0)
    assert scores.to_numpy().min() == pytest.approx(0.0)
    # And the blocked sample is no longer the most active value in the frame.
    assert scores.loc["s1", "partly_blocked"] < scores.loc["s1", "measured"]
    assert scores.to_numpy().max() == pytest.approx(np.log(10.0) - np.log(2.0))


def test_a_partially_blocked_reaction_is_retained_scored_at_the_floor_and_reported():
    # Blockedness cannot vary by sample inside one Compass run, so this shape
    # means the matrix was concatenated across runs. It is a real finding about
    # the input, and the range filter cannot see it: the range is large.
    frame = pd.DataFrame(
        {"merged": [0.0, 4.0, 5.0], "measured": [1.0, 2.0, 9.0]},
        index=["s1", "s2", "s3"],
    )

    result, report = preprocess_reactions(
        frame,
        ReactionPreprocessingConfig(transform=ReactionTransform.COMPASS_NEG_LOG),
    )

    assert "merged" in result.columns
    assert result.loc["s1", "merged"] == pytest.approx(0.0)
    assert report.parameters["partially_blocked_reactions"] == 1
    assert report.parameters["blocked_reactions"] == 0
    assert any("concatenated across runs" in warning for warning in report.warnings)


def _upstream_get_reaction_consistencies(
    compass_reaction_penalties: pd.DataFrame, min_range: float = 1e-3
) -> pd.DataFrame:
    """Upstream Compass postprocessing, verbatim.

    From ``compass/compass/postprocess.py`` in wagnerlab-berkeley/Compass, which
    takes reactions as rows and samples as columns.
    """
    df = -np.log(compass_reaction_penalties + 1)
    df = df[df.max(axis=1) - df.min(axis=1) >= min_range]
    df = df - df.min().min()
    return df


def test_output_is_identical_to_upstream_where_zeros_are_all_or_nothing():
    # Every acceptance claim about this transform rests on it agreeing with
    # upstream wherever upstream is defined. It is defined on data whose zeros
    # are all-or-nothing, which is what a single Compass run produces.
    rng = np.random.default_rng(20260724)
    penalties = pd.DataFrame(
        rng.gamma(shape=2.0, scale=500.0, size=(12, 40)),
        index=[f"s{i}" for i in range(12)],
        columns=[f"r{j}" for j in range(40)],
    )
    penalties.loc[:, "r0"] = 0.0
    penalties.loc[:, "r7"] = 0.0
    penalties.loc[:, "r31"] = 813.0  # invariant, but not blocked

    result, _ = preprocess_reactions(
        penalties,
        ReactionPreprocessingConfig(transform=ReactionTransform.COMPASS_NEG_LOG),
    )
    upstream = _upstream_get_reaction_consistencies(penalties.T).T

    assert result.columns.tolist() == upstream.columns.tolist()
    assert "r0" not in result.columns and "r7" not in result.columns
    assert "r31" not in result.columns
    np.testing.assert_allclose(
        result.to_numpy(), upstream.to_numpy(), rtol=0, atol=1e-12
    )


def test_the_upstream_low_range_filter_is_the_compass_default():
    config = ReactionPreprocessingConfig(transform=ReactionTransform.COMPASS_NEG_LOG)

    assert config.filter_low_variance is True
    assert config.minimum_feature_range == COMPASS_MINIMUM_RANGE

    # Other transforms are untouched: the filter is part of Compass's definition.
    plain = ReactionPreprocessingConfig(transform=ReactionTransform.SIGNED_CUBEROOT)
    assert plain.filter_low_variance is False


def test_keeping_every_reaction_stays_reachable_with_a_zero_threshold():
    frame = compass_frame()

    result, report = preprocess_reactions(
        frame,
        ReactionPreprocessingConfig(
            transform=ReactionTransform.COMPASS_NEG_LOG,
            filter_low_variance=True,
            minimum_feature_range=0.0,
        ),
    )

    assert "blocked" in result.columns
    # Retained, it sits at the floor of the scale rather than the top of it, and
    # it is constant there, so it can contribute nothing to any comparison.
    assert (result["blocked"] == 0.0).all()
    assert result["blocked"].max() == result.to_numpy().min()
    assert any("activity floor" in warning for warning in report.warnings)


def test_the_shift_is_computed_after_blocked_reactions_are_removed():
    frame = compass_frame()

    result, report = preprocess_reactions(
        frame,
        ReactionPreprocessingConfig(transform=ReactionTransform.COMPASS_NEG_LOG),
    )

    # The least active retained value is the costliest penalty, 20.
    assert report.parameters["global_offset"] == pytest.approx(np.log(21.0))
    assert result.loc["s3", "costly"] == pytest.approx(0.0)
    # The most active retained reaction is the cheapest penalty, 0.5 - and it no
    # longer shares the top of the scale with a reaction that carries no flux.
    assert result.loc["s1", "variable"] == pytest.approx(np.log(21.0) - np.log(1.5))
    assert (result.to_numpy() >= 0).all()


def test_blocked_reactions_are_identified_from_the_raw_penalties():
    assert blocked_compass_reactions(compass_frame()) == ("blocked",)
    # A reaction that is only sometimes free still carries flux somewhere.
    sometimes_free = pd.DataFrame({"r": [0.0, 3.0]}, index=["s1", "s2"])
    assert blocked_compass_reactions(sometimes_free) == ()


def test_partially_blocked_reactions_are_distinguished_from_fully_blocked_ones():
    frame = pd.DataFrame(
        {"blocked": [0.0, 0.0], "merged": [0.0, 3.0], "measured": [1.0, 3.0]},
        index=["s1", "s2"],
    )

    assert partially_blocked_compass_reactions(frame) == ("merged",)
    assert blocked_compass_reactions(frame) == ("blocked",)


@pytest.mark.parametrize(
    ("transform", "expected_zero"),
    [
        # A Compass zero is the blocked sentinel, so it lands on the floor of the
        # shifted scale rather than at its top.
        (compass_negative_log, 0.0),
        (signed_cuberoot, 0.0),
        (rank_percentile_by_feature, 1 / 3),
    ],
)
def test_transform_zero_behavior_is_defined(transform, expected_zero):
    frame = pd.DataFrame({"reaction": [0.0, 1.0, 2.0]})
    result = transform(frame)
    assert result.iloc[0, 0] == pytest.approx(expected_zero)
    assert np.isfinite(result.to_numpy()).all()


@pytest.mark.parametrize("invalid", [-0.5, -1.0, -2.0])
def test_compass_activity_score_rejects_negative_penalties(invalid):
    with pytest.raises(ValueError, match="non-negative penalty"):
        compass_negative_log(pd.DataFrame({"reaction": [0.0, invalid]}))

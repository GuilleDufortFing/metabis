"""Compass decoding is a pre-step, not a mutually exclusive scale transform.

Reading Compass penalties as activity scores answers "what do these numbers
mean"; a scale transform answers "on what scale should they be compared". The two
questions are independent, so the configuration carries them in two fields and a
user can quantile-normalize, rank, or cube-root a Compass matrix. The supported
serialized alias -- Compass as a value of ``ReactionTransform`` -- normalizes at
construction into the canonical two-stage form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metabis.domain.preprocessing import (
    COMPASS_MINIMUM_RANGE,
    ReactionPreprocessingConfig,
    ReactionTransform,
    compass_negative_log,
    describe_reaction_pipeline,
    preprocess_reactions,
    quantile_normalize,
)


@pytest.fixture
def penalties():
    """A Compass penalty matrix: non-negative, with a blocked reaction (all zero)."""
    return pd.DataFrame(
        {
            "R1": [4.0, 1.0, 9.0, 2.0],
            "R2": [0.5, 3.0, 0.25, 7.0],
            "R3": [2.0, 8.0, 1.5, 0.75],
            "BLOCKED": [0.0, 0.0, 0.0, 0.0],
        },
        index=["s1", "s2", "s3", "s4"],
    )


def test_the_serialized_compass_alias_becomes_the_canonical_pre_step(penalties):
    config = ReactionPreprocessingConfig(transform=ReactionTransform.COMPASS_NEG_LOG)

    assert config.compass_activity is True
    assert config.transform is ReactionTransform.NONE
    # The upstream low-range filter still comes with it.
    assert config.filter_low_variance is True
    assert config.minimum_feature_range == COMPASS_MINIMUM_RANGE


def test_the_serialized_alias_matches_the_canonical_configuration(penalties):
    alias_config = ReactionPreprocessingConfig(
        transform=ReactionTransform.COMPASS_NEG_LOG
    )
    canonical_config = ReactionPreprocessingConfig(compass_activity=True)

    assert alias_config == canonical_config

    alias_result, alias_report = preprocess_reactions(
        penalties,
        ReactionPreprocessingConfig(transform=ReactionTransform.COMPASS_NEG_LOG),
    )
    canonical_result, canonical_report = preprocess_reactions(
        penalties, canonical_config
    )

    pd.testing.assert_frame_equal(alias_result, canonical_result)
    assert alias_report == canonical_report
    assert canonical_report.method == "compass_neg_log"
    assert canonical_report.parameters["compass_activity"] is True


@pytest.mark.parametrize(
    "transform",
    (
        ReactionTransform.SIGNED_CUBEROOT,
        ReactionTransform.RANK_PERCENTILE,
        ReactionTransform.QUANTILE,
    ),
)
def test_every_transform_composes_with_the_compass_pre_step(penalties, transform):
    result, report = preprocess_reactions(
        penalties,
        ReactionPreprocessingConfig(compass_activity=True, transform=transform),
    )

    assert report.method == f"compass_neg_log+{transform.value}"
    assert np.isfinite(result.to_numpy()).all()
    # The blocked reaction was dropped by the Compass stage, before the transform,
    # so it cannot influence the scale the transform establishes.
    assert "BLOCKED" not in result.columns
    assert list(result.index) == list(penalties.index)


def test_the_composition_is_the_two_stages_applied_in_order(penalties):
    """Not a new calculation: decode, then transform exactly what decoding left."""
    composed, _report = preprocess_reactions(
        penalties,
        ReactionPreprocessingConfig(
            compass_activity=True, transform=ReactionTransform.QUANTILE
        ),
    )
    decoded, _decoded_report = preprocess_reactions(
        penalties, ReactionPreprocessingConfig(compass_activity=True)
    )

    pd.testing.assert_frame_equal(composed, quantile_normalize(decoded))


def test_a_transform_alone_still_ignores_the_compass_stage(penalties):
    result, report = preprocess_reactions(
        penalties, ReactionPreprocessingConfig(transform=ReactionTransform.QUANTILE)
    )

    assert report.method == "quantile"
    assert report.parameters["compass_activity"] is False
    # No decoding, so the blocked column is still a column of zeros: without the
    # Compass declaration the zeros are just values.
    pd.testing.assert_frame_equal(result, quantile_normalize(penalties))


def test_the_pre_step_alone_is_the_documented_transform(penalties):
    result, _report = preprocess_reactions(
        penalties,
        # The documented way to keep every reaction: the filter with a threshold
        # of zero, which retains everything including the blocked column.
        ReactionPreprocessingConfig(
            compass_activity=True, filter_low_variance=True, minimum_feature_range=0.0
        ),
    )

    pd.testing.assert_frame_equal(result, compass_negative_log(penalties))


@pytest.mark.parametrize(
    ("config", "expected"),
    (
        (ReactionPreprocessingConfig(), "none"),
        (ReactionPreprocessingConfig(compass_activity=True), "compass_neg_log"),
        (
            ReactionPreprocessingConfig(transform=ReactionTransform.QUANTILE),
            "quantile",
        ),
        (
            ReactionPreprocessingConfig(
                compass_activity=True, transform=ReactionTransform.QUANTILE
            ),
            "compass_neg_log+quantile",
        ),
    ),
)
def test_the_pipeline_name_states_every_stage(config, expected):
    assert describe_reaction_pipeline(config) == expected

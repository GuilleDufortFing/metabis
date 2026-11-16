"""Markers as the OSCA book's scoreMarkers ranks them: the smallest pairwise Cohen's d."""

import numpy as np
import pandas as pd
import pytest

from metabis.analysis.markers import group_means, pairwise_markers, top_markers

SAMPLES = {"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b3"], "C": ["c1", "c2", "c3"]}


def _matrix():
    # f1 is high in A only; f2 high in A and B (no group's marker against both others
    # but C's lowest); f3 high in C only; f4 flat.
    values = {
        "f1": [5, 6, 7, 1, 2, 1, 1, 2, 1],
        "f2": [5, 6, 5, 5, 6, 5, 1, 1, 2],
        "f3": [1, 2, 1, 1, 1, 2, 8, 9, 8],
        "f4": [3, 3, 3, 3, 3, 3, 3, 3, 3],
    }
    index = [s for samples in SAMPLES.values() for s in samples]
    return pd.DataFrame(values, index=index, dtype=float)


def test_a_marker_is_higher_in_its_group_than_in_every_other():
    markers = pairwise_markers(_matrix(), SAMPLES)

    mine = {group: list(rows.feature_id) for group, rows in markers.groupby("group", sort=False)}
    assert mine == {"A": ["f1"], "C": ["f3"]}
    # f2 is as high in B as in A, so it marks neither; f4 has no spread at all.
    a = markers.set_index("feature_id").loc["f1"]
    assert a.closest_group in {"B", "C"} and a.min_cohens_d > 3
    assert a["mean"] == pytest.approx(6.0)


def test_the_effect_is_cohens_d_over_the_average_standard_deviation():
    matrix = _matrix()
    markers = pairwise_markers(matrix, SAMPLES).set_index("feature_id")

    a, b = matrix.loc[SAMPLES["A"], "f1"], matrix.loc[SAMPLES["B"], "f1"]
    expected = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    against_c = (a.mean() - matrix.loc[SAMPLES["C"], "f1"].mean()) / np.sqrt(
        (a.var(ddof=1) + matrix.loc[SAMPLES["C"], "f1"].var(ddof=1)) / 2
    )
    assert markers.at["f1", "min_cohens_d"] == pytest.approx(min(expected, against_c))


def test_top_markers_keep_each_groups_strongest_and_means_follow_the_groups():
    markers = pairwise_markers(_matrix(), SAMPLES)

    assert list(top_markers(markers, 1).group) == ["A", "C"]
    means = group_means(_matrix(), SAMPLES)
    assert list(means.columns) == ["A", "B", "C"] and means.at["f3", "C"] == pytest.approx(25 / 3)
    with pytest.raises(ValueError, match="at least one marker"):
        top_markers(markers, 0)
    with pytest.raises(ValueError, match="at least two groups"):
        pairwise_markers(_matrix(), {"A": SAMPLES["A"]})

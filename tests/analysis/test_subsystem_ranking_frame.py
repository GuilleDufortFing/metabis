"""The delta ranking frame is an export schema, so its columns must be truthful.

This frame is what the server's Delta Score Results download writes, so a column
name here reaches the user's file. ``regulation_score`` named a signed difference
between two selected populations "regulation", which the flux/activity inputs do
not establish (D-63). It is ``net_direction_score``: the signed sum of the metric
over the subsystem's elements divided by its absolute sum, a net share of effect
magnitude in [-1, 1] oriented as Target minus Baseline.
"""

import pandas as pd

from metabis.analysis.subsystem_analysis import SubsystemAnalysis


class _AnalysisWorkspace:
    color_manager = None


def _analysis_with_ranking_state(net_direction_score: float) -> SubsystemAnalysis:
    analysis = SubsystemAnalysis(_AnalysisWorkspace())
    analysis._delta_stats_cache = {
        ("Glycolysis", False): {
            "delta_subsystem_score": 1.25,
            "delta_subsystem_meta": {"computed": True, "metric": "cohens_d"},
            "significant_elements": 6,
            "total_elements": 10,
            "significant_percentage": 60.0,
            "net_direction_scores": {"cohens_d": net_direction_score},
        },
    }
    return analysis


def test_the_ranking_frame_names_the_net_direction_score():
    frame = _analysis_with_ranking_state(0.632).rank_subsystems_by_delta_score()

    assert frame["net_direction_score"].tolist() == [0.632]
    assert not any("regulation" in column for column in frame.columns)


def test_a_negative_net_direction_score_survives_the_ranking_frame():
    """The sign is the orientation, so ranking by |delta| may not discard it."""
    frame = _analysis_with_ranking_state(-0.4).rank_subsystems_by_delta_score()

    assert frame["net_direction_score"].tolist() == [-0.4]


def test_the_empty_ranking_frame_declares_the_same_columns():
    """A subsystem-free comparison still downloads headers, so they must agree."""
    populated = _analysis_with_ranking_state(0.1).rank_subsystems_by_delta_score()
    empty = SubsystemAnalysis(_AnalysisWorkspace()).rank_subsystems_by_delta_score()

    assert empty.empty
    assert list(empty.columns) == list(populated.columns)


def test_the_net_direction_score_is_the_net_share_of_effect_magnitude():
    """Two elements of equal magnitude and opposite sign cancel to zero."""
    analysis = SubsystemAnalysis(_AnalysisWorkspace())
    stats_frame = pd.DataFrame({"cohens_d": [1.5, -0.5, -1.0]}, index=["R1", "R2", "R3"])

    aggregates = analysis._calculate_metric_aggregates(stats_frame, ["cohens_d"])

    assert aggregates["net_direction_scores"]["cohens_d"] == 0.0
    assert aggregates["total_metric_values"]["cohens_d"] == 0.0

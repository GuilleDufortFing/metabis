"""The direction frame must not lose a reaction measured only in reverse (D-64).

``ExperimentAnalysis._calculate_reaction_directions`` used to build its column
list from the forward columns, so a base reaction present only as ``_back``
vanished from the frame entirely: ``get_main_direction`` returned ``None``, and
colour, hover, and subsystem enrichment all fell through to a column that does
not exist. 200 base reactions over 25 subsystems in the bundled Th17 data, none
of them reachable by any threshold.

Compass produces that shape routinely - ``r0556_pos`` is exactly 0 in all 290
bundled samples (blocked) while ``r0556_neg`` ranges 13,297-15,680, so after the
low-range filter the frame holds ``MAR06414_back`` only.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from metabis.analysis.experiment_analysis import ExperimentAnalysis
from metabis.analysis.reaction_direction import to_base_reaction_id

GROUPS = ["target", "baseline"]


def directions(averages: pd.DataFrame) -> pd.DataFrame:
    """Run the direction calculation over a reaction-average frame."""
    analysis = ExperimentAnalysis.__new__(ExperimentAnalysis)
    analysis.group_ids = list(averages.index)
    analysis._reaction_avgs = averages
    analysis._reaction_directions = None
    return analysis._calculate_reaction_directions()


def averages(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns, index=GROUPS, dtype=float)


def test_a_reaction_measured_only_in_reverse_resolves_to_backward():
    frame = directions(averages(MAR06414_back=[2.1488, 2.1421]))

    assert list(frame.columns) == ["MAR06414"]
    assert frame.loc["target", "MAR06414"] == "backward"
    assert frame.loc["baseline", "MAR06414"] == "backward"


def test_a_reaction_measured_only_forward_still_resolves_to_forward():
    frame = directions(averages(MAR00001=[3.0, 4.0]))

    assert list(frame.columns) == ["MAR00001"]
    assert (frame["MAR00001"] == "forward").all()


def test_both_directions_measured_take_the_larger_mean_with_forward_winning_ties():
    frame = directions(
        averages(
            MAR00001=[5.0, 1.0],
            MAR00001_back=[2.0, 9.0],
            MAR00002=[4.0, 4.0],
            MAR00002_back=[4.0, 4.0],
        )
    )

    assert frame.loc["target", "MAR00001"] == "forward"
    assert frame.loc["baseline", "MAR00001"] == "backward"
    assert (frame["MAR00002"] == "forward").all()


def test_the_frame_is_keyed_on_base_ids_and_carries_every_measured_reaction():
    frame = directions(
        averages(
            MAR00001=[1.0, 1.0],
            MAR00001_back=[2.0, 2.0],
            MAR00002=[1.0, 1.0],
            MAR06414_back=[3.0, 3.0],
        )
    )

    assert list(frame.columns) == ["MAR00001", "MAR00002", "MAR06414"]
    assert not any(column.endswith("_back") for column in frame.columns)


def test_a_missing_mean_does_not_hide_the_direction_that_was_measured():
    frame = directions(averages(MAR06414=[np.nan, np.nan], MAR06414_back=[3.0, 3.0]))

    # A NaN forward mean reads as 0.0, so the measured reverse direction wins.
    assert (frame["MAR06414"] == "backward").all()


@pytest.mark.parametrize(
    "columns",
    [
        {"MAR00001": [5.0, 1.0], "MAR00001_back": [2.0, 9.0]},
        {"MAR06414_back": [2.1488, 2.1421]},
        {"MAR00002": [4.0, 4.0], "MAR00002_back": [4.0, 4.0]},
        {"MAR00003": [0.5, 0.5]},
    ],
)
def test_the_two_direction_resolvers_agree_on_the_same_input(columns):
    """The frame and the comparison-artifact resolver must not diverge again.

    ``build_reaction_direction_map`` already handled the reverse-only shape
    correctly while the frame did not, so Delta score and ORA saw a reaction the
    graph and hover did not. Both rules are: a lone direction wins, otherwise the
    larger mean in the named population wins, and an exact tie goes to forward.
    """
    frame = directions(averages(**columns))

    for group_index, group_id in enumerate(GROUPS):
        for base_id in frame.columns:
            forward = columns.get(base_id)
            backward = columns.get(f"{base_id}_back")
            if forward is None:
                expected = "backward"
            elif backward is None:
                expected = "forward"
            else:
                expected = (
                    "forward"
                    if forward[group_index] >= abs(backward[group_index])
                    else "backward"
                )
            assert frame.loc[group_id, base_id] == expected


def test_every_column_collapses_to_its_own_base_id():
    columns = {
        "MAR00001": [1.0, 1.0],
        "MAR00001_back": [2.0, 2.0],
        "MAR06414_back": [3.0, 3.0],
    }
    frame = directions(averages(**columns))

    assert set(frame.columns) == {to_base_reaction_id(name) for name in columns}


# --- the three consumers that fell through to the missing forward column -----


def _analysis(averages_frame: pd.DataFrame) -> ExperimentAnalysis:
    analysis = ExperimentAnalysis.__new__(ExperimentAnalysis)
    analysis.group_ids = list(averages_frame.index)
    analysis._reaction_avgs = averages_frame
    analysis._reaction_directions = None
    analysis._analysis_completed = True
    analysis._calculate_reaction_directions()
    return analysis


def test_get_main_direction_answers_for_a_reverse_only_reaction():
    analysis = _analysis(averages(MAR06414_back=[2.1488, 2.1421]))

    # It used to return None here, which is neither "forward" nor "backward", so
    # every caller fell back to the base ID and found nothing.
    assert analysis.get_main_direction("target", "MAR06414") == "backward"
    # And the `_back` spelling resolves to the same answer.
    assert analysis.get_main_direction("target", "MAR06414_back") == "backward"


def test_a_significant_reverse_only_reaction_reaches_subsystem_enrichment():
    from metabis.managers.data_provider import DataProvider

    provider = DataProvider.__new__(DataProvider)
    provider.experiment_analysis = _analysis(
        averages(MAR06414_back=[2.1488, 2.1421], MAR00001=[1.0, 1.0])
    )
    provider.group_comparison_artifact = SimpleNamespace(
        group_id_1="target",
        get_significant_elements=lambda for_genes: ["MAR06414_back", "MAR00001"],
    )

    # `MAR06414_back` was excluded unconditionally: the check requires
    # get_main_direction(...) == "backward" and it answered None.
    assert provider.get_main_direction_significant_reactions() == [
        "MAR06414",
        "MAR00001",
    ]


def test_comparison_statistics_are_found_from_the_reverse_direction_alone():
    """The same forward-keyed assumption, one layer up.

    `get_reaction_group_comparison_data` required the *forward* column's
    statistics before it would report any, so a reaction measured only in reverse
    coloured from its reverse direction but its hover said "No Experimental
    Data" beneath it.
    """
    from metabis.managers.data_provider import DataProvider

    provider = DataProvider.__new__(DataProvider)
    provider.group_comparison_artifact = SimpleNamespace(
        membership=SimpleNamespace(target_label="Th17p", baseline_label="Th17n"),
        get_statistical_info=lambda element_id, for_genes: (
            {"pval": 0.0342, "is_significant": False}
            if element_id == "MAR06414_back"
            else None
        ),
    )

    stats = provider.get_reaction_group_comparison_data("MAR06414")

    assert stats["has_comparison_data"] is True
    assert stats["forward"] is None
    assert stats["backward"] == {"pval": 0.0342, "is_significant": False}
    assert stats["group1"] == "Th17p"


def test_a_reaction_measured_in_neither_direction_still_reports_no_data():
    from metabis.managers.data_provider import DataProvider

    provider = DataProvider.__new__(DataProvider)
    provider.group_comparison_artifact = SimpleNamespace(
        membership=SimpleNamespace(target_label="Th17p", baseline_label="Th17n"),
        get_statistical_info=lambda element_id, for_genes: None,
    )

    assert provider.get_reaction_group_comparison_data("MAR00001") == {
        "has_comparison_data": False
    }


def test_the_statistics_hover_renders_a_reverse_only_reaction():
    from metabis.managers.hover.models import HoverContext
    from metabis.managers.hover.sections.statistics import StatisticsSection
    from metabis.managers.workspace_mode import AnalysisWorkspaceMode

    section = StatisticsSection("reaction")
    context = HoverContext(
        mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
        hover_type="reaction",
        element_type="reaction",
        element_id="MAR06414_back",
        main_direction="backward",
        show_stats=True,
    )
    data = {
        "is_in_model": True,
        "is_reversible": True,
        "comparison_stats": {
            "has_comparison_data": True,
            "group1": "Th17p",
            "group2": "Th17n",
            "forward": None,
            "backward": {"pval": 0.0342, "adjusted_pval": 0.074},
        },
    }

    assert section.should_include(data, context) is True
    text = section.build(data, context)
    # Only the direction that was measured, and it is named the active one.
    assert "Reverse (" in text
    assert "more active direction" in text
    assert "Forward" not in text
    assert "0.034" in text

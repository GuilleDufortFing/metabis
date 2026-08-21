"""A blocked reaction is a result, not an absence (WP10/D-65).

Compass reporting a zero penalty means it could not make the reaction carry any
flux under that model and medium. Such reactions are dropped before the
statistical comparison - correctly, since they are constant by construction - but
they then rendered exactly like a reaction that was never in the input. The
blocked set existed only as a load-time warning string and never reached a hover.
"""

import pandas as pd
import pytest

from metabis import (
    ReactionPreprocessingConfig,
    ReactionTransform,
    preprocess_reactions,
)
from metabis.managers.data_provider import DataProvider
from metabis.managers.hover.models import HoverContext
from metabis.managers.hover.sections.blocked_reaction import BlockedReactionSection
from metabis.managers.workspace_mode import AnalysisWorkspaceMode

SECTION = BlockedReactionSection()
CONTEXT = HoverContext(
    mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
    hover_type="reaction",
    element_type="reaction",
)


class _ReactionData:
    def __init__(self, report):
        self.preprocessing_report = report


def report_for(**columns):
    frame = pd.DataFrame(columns, index=["s1", "s2", "s3"], dtype=float)
    _result, report = preprocess_reactions(
        frame,
        ReactionPreprocessingConfig(transform=ReactionTransform.COMPASS_NEG_LOG),
    )
    return report


def provider_for(report) -> DataProvider:
    provider = DataProvider.__new__(DataProvider)
    provider.reaction_data = _ReactionData(report)
    return provider


# --- the report carries the set, tagged with its reason ----------------------


def test_the_report_separates_blocked_reactions_from_merely_invariant_ones():
    report = report_for(
        blocked=[0.0, 0.0, 0.0],
        invariant=[813.0, 813.0, 813.0],
        measured=[1.0, 5.0, 20.0],
    )

    assert report.blocked_feature_ids == ("blocked",)
    assert report.invariant_feature_ids == ("invariant",)
    assert set(report.filtered_feature_ids) == {"blocked", "invariant"}


def test_a_report_from_another_transform_claims_nothing_about_blocking():
    frame = pd.DataFrame({"constant": [2.0, 2.0, 2.0]}, index=["s1", "s2", "s3"])
    _result, report = preprocess_reactions(
        frame,
        ReactionPreprocessingConfig(
            transform=ReactionTransform.SIGNED_CUBEROOT,
            filter_low_variance=True,
            minimum_feature_range=0.1,
        ),
    )

    assert report.filtered_feature_ids == ("constant",)
    assert report.blocked_feature_ids == ()


# --- the provider resolves it per direction ----------------------------------


def test_the_provider_reports_which_directions_are_blocked():
    provider = provider_for(
        report_for(
            MAR06414=[0.0, 0.0, 0.0],
            MAR06414_back=[13297.0, 14000.0, 15680.0],
            MAR00001=[1.0, 5.0, 20.0],
        )
    )

    assert provider.get_blocked_directions("MAR06414") == ("forward",)
    # The `_back` spelling asks about the same reaction.
    assert provider.get_blocked_directions("MAR06414_back") == ("forward",)
    assert provider.get_blocked_directions("MAR00001") == ()


def test_an_invariant_reaction_is_not_reported_as_blocked():
    provider = provider_for(
        report_for(MAR00001=[813.0, 813.0, 813.0], MAR00002=[1.0, 5.0, 20.0])
    )

    assert provider.get_blocked_directions("MAR00001") == ()


def test_a_provider_without_a_preprocessing_report_answers_nothing():
    provider = DataProvider.__new__(DataProvider)
    provider.reaction_data = None

    assert provider.get_blocked_directions("MAR00001") == ()


def test_the_blocked_set_is_built_once_and_rebuilt_when_the_input_changes():
    """A graph render asks per reaction; the set must not be rebuilt per hover."""
    first = report_for(MAR00001=[0.0, 0.0, 0.0], MAR00002=[1.0, 5.0, 20.0])
    provider = provider_for(first)

    assert provider.get_blocked_directions("MAR00001") == ("forward",)
    built = provider._blocked_feature_ids()
    assert provider._blocked_feature_ids() is built

    # Replacing the input invalidates it rather than serving a stale answer.
    provider.reaction_data = _ReactionData(
        report_for(MAR00003=[0.0, 0.0, 0.0], MAR00002=[1.0, 5.0, 20.0])
    )
    assert provider._blocked_feature_ids() is not built
    assert provider.get_blocked_directions("MAR00001") == ()
    assert provider.get_blocked_directions("MAR00003") == ("forward",)


# --- the hover says so, scoped to this dataset -------------------------------


def test_a_fully_blocked_reaction_states_that_compass_reports_no_flux():
    text = SECTION.build(
        {
            "is_in_model": True,
            "is_reversible": False,
            "blocked_directions": ("forward",),
        },
        CONTEXT,
    )

    assert "Blocked in this dataset" in text
    # Scoped to the dataset, not asserted as a property of the model: the medium
    # and configuration of the Compass run are unknowable for an uploaded file.
    assert "Compass reports no flux for this reaction in any sample" in text
    assert "excluded from the statistical comparison" in text


def test_one_blocked_direction_of_a_reversible_reaction_names_that_direction():
    text = SECTION.build(
        {
            "is_in_model": True,
            "is_reversible": True,
            "blocked_directions": ("forward",),
        },
        CONTEXT,
    )

    assert "the forward direction of this reaction" in text
    assert "only the other direction is compared" in text


def test_both_directions_blocked_reads_as_the_whole_reaction():
    text = SECTION.build(
        {
            "is_in_model": True,
            "is_reversible": True,
            "blocked_directions": ("forward", "reverse"),
        },
        CONTEXT,
    )

    assert "no flux for this reaction in any sample" in text


@pytest.mark.parametrize(
    "data",
    [
        {"is_in_model": True, "blocked_directions": ()},
        {"is_in_model": True},
        {"is_in_model": False, "blocked_directions": ("forward",)},
    ],
)
def test_the_section_stays_out_of_every_other_hover(data):
    assert SECTION.should_include(data, CONTEXT) is False


def test_the_section_is_included_only_for_a_blocked_reaction_in_the_model():
    data = {"is_in_model": True, "blocked_directions": ("forward",)}

    assert SECTION.should_include(data, CONTEXT) is True

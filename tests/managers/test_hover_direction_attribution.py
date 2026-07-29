"""The hover must attribute the displayed direction to the population that chose it.

Two surfaces named the wrong thing once the criterion defaulted to Baseline
dominance (WP10/D-70, D-71):

* the statistics label always read "more active direction in Target <target>",
  hard-coded, about a column selected from the *Baseline* means;
* the colour reference looked its histogram marker up under the base reaction ID,
  so a reaction displayed backward marked the *forward* p-value and effect size
  and labelled them "Used for coloring".
"""

from types import SimpleNamespace

import pytest

from metabis.analysis.reaction_direction import ReactionDirectionCriterion
from metabis.managers.hover.models import HoverContext
from metabis.managers.hover.sections.color_explanation import ColorExplanationSection
from metabis.managers.hover.sections.statistics import StatisticsSection
from metabis.managers.workspace_mode import AnalysisWorkspaceMode

STATS = {
    "has_comparison_data": True,
    "group1": "Th17p",
    "group2": "Th17n",
    "forward": {"pval": 9.40e-04, "cohens_d": 0.334, "is_significant": True},
    "backward": {"pval": 0.400, "cohens_d": 0.144, "is_significant": False},
}
DATA = {"is_in_model": True, "is_reversible": True, "comparison_stats": STATS}


def _context(criterion, main_direction="backward"):
    return HoverContext(
        mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
        hover_type="reaction",
        element_type="reaction",
        element_id="MAR04139_back" if main_direction == "backward" else "MAR04139",
        main_direction=main_direction,
        direction_criterion=criterion,
        show_stats=True,
        show_color_reference=True,
    )


@pytest.mark.parametrize(
    ("criterion", "expected_population", "expected_group", "wrong_group"),
    [
        (ReactionDirectionCriterion.DOMINANT_IN_BASELINE, "Baseline", "Th17n", "Th17p"),
        (ReactionDirectionCriterion.DOMINANT_IN_TARGET, "Target", "Th17p", "Th17n"),
    ],
)
def test_the_label_names_the_population_that_selected_the_direction(
    criterion, expected_population, expected_group, wrong_group
):
    text = StatisticsSection("reaction").build(DATA, _context(criterion))

    assert f"more active direction</span> in {expected_population} " in text
    assert expected_group in text.split("more active direction")[1]
    assert f"in {wrong_group}" not in text


def test_an_unresolved_criterion_still_reads_as_target():
    """No installed map means no criterion; the wording must not become blank."""
    text = StatisticsSection("reaction").build(DATA, _context(None))

    assert "more active direction</span> in Target " in text


class _ColorManager:
    """Records the exact column the colour reference asked about."""

    def __init__(self):
        self.requested = []

    def get_distribution_details(self, mode, element_type, element_id, metric_name):
        self.requested.append((element_id, metric_name))
        # Returning no histogram short-circuits rendering; the assertion under
        # test is which column was looked up, not how it is drawn.
        return {"histogram": None, "element_value": None}

    def get_group_color(self, _group_id):
        return "black"


@pytest.mark.parametrize(
    ("main_direction", "expected_column"),
    [("backward", "MAR04139_back"), ("forward", "MAR04139")],
)
def test_the_colour_reference_reads_the_direction_that_was_coloured(
    main_direction, expected_column
):
    color_manager = _ColorManager()
    context = _context(
        ReactionDirectionCriterion.DOMINANT_IN_BASELINE, main_direction=main_direction
    )
    context.color_manager = color_manager
    context.data_provider = SimpleNamespace(group_comparison_artifact=None)

    ColorExplanationSection().build({**DATA, "reaction_id": "MAR04139"}, context)

    assert color_manager.requested, "the colour reference asked for no distribution"
    assert {column for column, _ in color_manager.requested} == {expected_column}
    assert [metric for _, metric in color_manager.requested] == [
        "neg_log_pval",
        "log2_fc",
    ]

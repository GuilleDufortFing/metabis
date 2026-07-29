"""The comparison subsystem ranking carries its gradient colour reference (R-09).

Supplementary Section 1.5 claims a colour reference for every gradient plot. The
ranking dot plot paints each element from the registered comparison distribution
of the selected metric, and until WP9 primitive #4 it showed no reference at all:
a reader could see that a dot was dark red without any way to learn how red the
rest of the distribution was.

The plot is driven through a stub workspace rather than a full analysis run,
because what is under test is the wiring -- the panel must come from *the same*
distribution the glyphs were coloured from, and must mark *each element's own*
value -- not the statistics, which their own suites cover.
"""

from __future__ import annotations

import pandas as pd
import pytest
from bokeh.models import HoverTool

from metabis.managers.color_manager import ColorManager
from metabis.managers.workspace_mode import AnalysisWorkspaceMode
from metabis.viz.plot_subsystem_ranking import plot_subsystem_comparison

METRIC = "cohens_d"
SUBSYSTEM = "Glycolysis"
CATEGORY = "Carbohydrate metabolism"
EFFECTS = {"R1": 2.4, "R2": -1.8, "R3": 0.3, "R4": 1.1, "R5": -0.6}


class _Model:
    def get_reactions_for_subsystem(self, _subsystem):
        return list(EFFECTS)

    def get_genes_for_subsystem(self, _subsystem):
        return []

    def get_subsystem_category(self, _subsystem):
        return CATEGORY

    def get_subsystem_name(self, subsystem):
        return subsystem

    def get_category_color(self, _category):
        return "#2A9D8F"

    def get_reaction_reference_data(self, reaction_id):
        return {
            "raw_id": reaction_id,
            "display_label": reaction_id,
            "external_url": "",
        }


class _Comparison:
    def __init__(self, results):
        self._results = results

    def get_test_results(self, for_genes=False):
        return self._results

    def get_significant_elements(self, for_genes=False):
        return set(self._results.index)

    def get_thresholds(self, for_genes=False):
        return {METRIC: 0.5}


class _Workspace:
    """The narrow surface ``plot_subsystem_comparison`` actually reads."""

    def __init__(self):
        self.metabolic_model = _Model()
        self.subsystem_analysis = None
        self._results = pd.DataFrame({METRIC: pd.Series(EFFECTS, dtype=float)})
        self._comparison = _Comparison(self._results)
        self.color_manager = ColorManager()
        self.color_manager.register_distribution(
            mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
            element_type="reaction",
            data=self._results[METRIC],
            metric_name=METRIC,
            center_method="zero",
            spread_method="std",
        )

    def get_group_comparison_artifact(self):
        return self._comparison

    def get_element_hover(self, element_id, element_type, **_kwargs):
        return f"<div>{element_id} statistics</div>"

    def get_subsystem_display_name(self, subsystem):
        return subsystem


@pytest.fixture
def ranking():
    stats = pd.DataFrame({"subsystem": [SUBSYSTEM], "delta_score": [1.4]})
    plot = plot_subsystem_comparison(
        data_manager=_Workspace(),
        subsystem_stats=stats,
        metric=METRIC,
        return_plot=True,
    )
    assert plot is not None
    return plot, plot.renderers[0].data_source.data


def test_every_ranked_dot_carries_the_gradient_color_reference(ranking):
    plot, source = ranking
    hover = plot.select_one(HoverTool)

    assert len(source["desc"]) == len(EFFECTS)
    assert "<strong>Color Reference:</strong>" in hover.tooltips
    assert f"{METRIC} Distribution" in hover.tooltips


def test_each_dot_marks_its_own_position_in_that_distribution(ranking):
    _plot, source = ranking
    assert list(source["color_reference_value"]) == [
        EFFECTS[element_id] for element_id in source["element_id"]
    ]

    marker_positions = list(source["color_reference_marker_x"])
    assert len(set(marker_positions)) == len(EFFECTS)


def test_the_reference_shows_the_significance_threshold_that_grays_a_dot(ranking):
    plot, _source = ranking
    hover = plot.select_one(HoverTool)

    # `_Comparison.get_thresholds` reports 0.5, and a dot below it is drawn gray;
    # the reference draws that boundary so the gray has a stated cause.
    assert "stroke-dasharray" in hover.tooltips


def test_the_distribution_svg_is_not_duplicated_in_every_point_payload(ranking):
    plot, source = ranking
    hover = plot.select_one(HoverTool)

    assert hover.tooltips.count("<svg") == 1
    assert all("<svg" not in description for description in source["desc"])

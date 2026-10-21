"""The comparison subsystem ranking hovers with its element statistics only.

The ranking dot plot paints each element from the registered comparison
distribution of the selected metric, and for a while its hover appended that
distribution as a gradient colour reference. The panel restated what the plot's
own x-axis already shows -- where a dot sits in the distribution -- while
carrying a static histogram SVG on the ``HoverTool`` of one of the heaviest
plots in the document, so it was removed.

The plot is driven through a stub workspace rather than a full analysis run,
because what is under test is the hover wiring, not the statistics.
"""

from __future__ import annotations

import pandas as pd
import pytest
from bokeh.models import HoverTool

from metabis.managers.color_manager import ColorManager
from metabis.managers.workspace_mode import AnalysisWorkspaceMode
from metabis.viz.plot_subsystem_ranking import plot_subsystem_comparison
from metabis.viz.ranking_layout import RANKING_DOTS

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
    return plot, _dots(plot).data_source.data


def _dots(layout):
    return layout.select_one({"name": RANKING_DOTS})


def _dot_hover(layout):
    dots = _dots(layout)
    (hover,) = [tool for tool in layout.select({"type": HoverTool}) if dots in (tool.renderers or [])]
    return hover


def test_every_ranked_dot_hovers_with_its_own_statistics(ranking):
    plot, source = ranking
    hover = _dot_hover(plot)

    assert hover.tooltips == "@desc{safe}"
    assert len(source["desc"]) == len(EFFECTS)


def test_the_hover_carries_no_gradient_color_reference(ranking):
    plot, source = ranking
    hover = _dot_hover(plot)

    assert "Color Reference:" not in hover.tooltips
    assert "<svg" not in hover.tooltips
    assert all("<svg" not in description for description in source["desc"])


def test_no_color_reference_payload_is_shipped_per_point(ranking):
    _plot, source = ranking

    assert "color_reference_value" not in source
    assert "color_reference_marker_x" not in source


def test_a_server_free_ranking_reads_hover_downloads_from_a_page_flag():
    stats = pd.DataFrame({"subsystem": [SUBSYSTEM], "delta_score": [1.4]})
    plot = plot_subsystem_comparison(
        data_manager=_Workspace(),
        subsystem_stats=stats,
        metric=METRIC,
        return_plot=True,
        download_flag="metabisDownloadHoverOnClick",
    )

    (callback,) = _dots(plot).data_source.selected.js_property_callbacks[
        "change:indices"
    ]
    assert callback.args["download_flag"] == "metabisDownloadHoverOnClick"
    assert "enabled_toggle" not in callback.args


def test_rows_are_named_in_a_column_beside_their_delta_score(ranking):
    layout, source = ranking
    summary, dots_panel = [child[0] for child in layout.children]

    # The names are the summary panel's y-axis; the dots' own axis is hidden.
    assert summary.yaxis[0].visible is True
    assert dots_panel.yaxis[0].visible is False
    assert summary.y_range is dots_panel.y_range
    assert summary.y_range.factors == [SUBSYSTEM, CATEGORY.upper()]
    # Each dot sits at a fixed place within its row, the same on every render.
    assert all(isinstance(y, tuple) and y[0] == SUBSYSTEM for y in source["y"])
    assert all(abs(y[1]) < 0.5 for y in source["y"])


def test_the_value_axis_is_drawn_above_and_below(ranking):
    layout, _source = ranking
    dots_panel = layout.children[1][0]

    assert len(dots_panel.above) >= 1 and len(dots_panel.below) >= 1

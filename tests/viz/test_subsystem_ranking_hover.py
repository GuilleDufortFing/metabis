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


def test_every_ranked_dot_hovers_with_its_own_statistics(ranking):
    plot, source = ranking
    hover = plot.select_one(HoverTool)

    assert hover.tooltips == "@desc{safe}"
    assert len(source["desc"]) == len(EFFECTS)


def test_the_hover_carries_no_gradient_color_reference(ranking):
    plot, source = ranking
    hover = plot.select_one(HoverTool)

    assert "Color Reference:" not in hover.tooltips
    assert "<svg" not in hover.tooltips
    assert all("<svg" not in description for description in source["desc"])


def test_no_color_reference_payload_is_shipped_per_point(ranking):
    _plot, source = ranking

    assert "color_reference_value" not in source
    assert "color_reference_marker_x" not in source

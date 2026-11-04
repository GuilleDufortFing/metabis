"""Colour bars are read from the distribution that coloured the figure."""

import pandas as pd

from metabis.managers.color_manager import ColorManager
from metabis.managers.workspace_mode import AnalysisWorkspaceMode
from metabis.viz.color_bars import (
    COLOR_BAR_LENGTH,
    KEY_ROW,
    overview_color_bar,
    registered_color_bar,
)

MODE = AnalysisWorkspaceMode.GROUP_COMPARISON


class _Workspace:
    def __init__(self, scores):
        self.color_manager = ColorManager()
        self.color_manager.register_distribution(
            mode=MODE,
            element_type="subsystem",
            data=pd.Series(scores, dtype=float),
            metric_name="cohens_d",
            center_method="zero",
            spread_method="std",
        )


def test_the_bar_samples_the_registered_ramp_through_white():
    workspace = _Workspace({"a": -0.4, "b": 0.1, "c": 0.5, "d": 0.2})
    bar = registered_color_bar(
        workspace.color_manager,
        mode=MODE,
        element_type="subsystem",
        metric_name="cohens_d",
        title="Delta score",
    )
    mapper = bar.color_mapper
    palette = list(mapper.palette)

    assert bar.title == "Delta score"
    assert mapper.low < 0 < mapper.high
    assert palette[len(palette) // 2].upper() == "#FFFFFF"
    # The ends are the registered up and down colours, at full intensity.
    assert palette[-1].upper() == workspace.color_manager.up_color.upper()
    assert palette[0].upper() == workspace.color_manager.down_color.upper()


def test_a_sign_delta_score_bar_stops_at_minus_one_and_one():
    workspace = _Workspace({"a": -0.4, "b": 0.1, "c": 0.5, "d": 0.2})
    bar = overview_color_bar(
        workspace,
        visualization_mode="group_comparison",
        delta_options={"metric": "cohens_d", "transform": "sign"},
    )

    assert (bar.color_mapper.low, bar.color_mapper.high) == (-1.0, 1.0)


def test_no_spread_draws_no_bar():
    workspace = _Workspace({"a": 0.3})

    assert (
        overview_color_bar(
            workspace,
            visualization_mode="group_comparison",
            delta_options={"metric": "cohens_d", "transform": "sign"},
        )
        is None
    )


def test_the_pathway_key_names_each_shape_and_adds_a_bar_per_family():
    from bokeh.models import ColorBar, Legend
    from bokeh.plotting import figure

    from metabis.viz.color_bars import add_pathway_key

    workspace = _Workspace({"a": -0.4, "b": 0.1, "c": 0.5, "d": 0.2})
    workspace.color_manager.register_distribution(
        mode=MODE,
        element_type="reaction",
        data=pd.Series({"r1": -1.0, "r2": 0.5, "r3": 0.8}, dtype=float),
        metric_name="cohens_d",
        center_method="zero",
        spread_method="std",
    )
    plot = figure()
    add_pathway_key(plot, workspace, visualization_mode="group_comparison", color_metric="cohens_d")

    (legend,) = [item for item in plot.below if isinstance(item, Legend)]
    assert [item.label.value for item in legend.items] == [
        "Reaction",
        "Gene",
        "Metabolite",
        "Not significant, or no data",
    ]
    # Reactions have a distribution here and genes none, so one bar: short and
    # horizontal, under the graph, for the page to set beside the others.
    bars = [item for item in plot.below if isinstance(item, ColorBar)]
    assert [bar.title for bar in bars] == ["Reactions, Cohen's d"]
    assert (bars[0].orientation, bars[0].width) == ("horizontal", COLOR_BAR_LENGTH)
    assert KEY_ROW in bars[0].tags
    assert not plot.right


def test_a_figure_scaled_to_its_width_grows_by_its_key_and_keeps_its_drawing():
    from bokeh.plotting import figure

    from metabis.viz.color_bars import (
        KEY_BARS_HEIGHT,
        KEY_SHAPES_HEIGHT,
        add_pathway_key,
    )

    workspace = _Workspace({"a": 0.1})
    workspace.color_manager.register_distribution(
        mode=MODE,
        element_type="reaction",
        data=pd.Series({"r1": -1.0, "r2": 0.5, "r3": 0.8}, dtype=float),
        metric_name="cohens_d",
        center_method="zero",
        spread_method="std",
    )
    scaled = figure(width=900, height=300, sizing_mode="scale_width", aspect_ratio=3.0)
    add_pathway_key(scaled, workspace, visualization_mode="group_comparison", color_metric="cohens_d")
    fixed = figure(width=900, height=300)
    add_pathway_key(fixed, workspace, visualization_mode="group_comparison", color_metric="cohens_d")

    # At its own width, the figure is the drawing's 300 pixels and its key's rows tall.
    assert round(900 / scaled.aspect_ratio) == 300 + KEY_SHAPES_HEIGHT + KEY_BARS_HEIGHT
    assert (fixed.height, fixed.aspect_ratio) == (300, None)


def test_the_activity_key_draws_no_data_white_and_dashed_since_grey_is_the_typical_level():
    from bokeh.models import Legend
    from bokeh.plotting import figure

    from metabis.viz.color_bars import add_pathway_key

    plot = figure()
    add_pathway_key(plot, _Workspace({"a": 0.1}), visualization_mode="activity")

    (legend,) = [item for item in plot.below if isinstance(item, Legend)]
    assert [item.label.value for item in legend.items][-1] == "No data"
    (renderer,) = legend.items[-1].renderers
    assert renderer.glyph.fill_color == "#ffffff" and renderer.glyph.line_dash == "dashed"


def test_a_figure_drawn_from_metabolic_atlas_maps_credits_them_under_its_key():
    from bokeh.models import Legend, Title
    from bokeh.plotting import figure

    from metabis.viz.atlas_paths import ATLAS_CREDIT
    from metabis.viz.color_bars import (
        CREDIT_LINE_HEIGHT,
        add_atlas_credit,
        add_pathway_key,
    )

    plot = figure(width=900, height=300, sizing_mode="scale_width", aspect_ratio=3.0)
    add_atlas_credit(plot)
    add_pathway_key(plot, _Workspace({"a": 0.1}), visualization_mode="group_comparison", color_metric="cohens_d")

    # In the figure, so an export carries it: who holds the copyright, where the
    # maps come from, their licence, that they were redrawn, and the licence the
    # figure is shared under; under the key, the figure grown by its lines.
    credit = [item for item in plot.below if isinstance(item, Title)]
    assert [line.text for line in credit] == list(ATLAS_CREDIT)
    (legend,) = [item for item in plot.below if isinstance(item, Legend)]
    assert plot.below[-len(credit):] == credit and plot.below.index(legend) < plot.below.index(credit[0])
    text = " ".join(ATLAS_CREDIT)
    for part in ("Metabolic Atlas Human-maps", "commit 8ab97f2", "© Department of Biology and Biological Engineering, Chalmers University of Technology", "Redrawn by Metabis", "shared under CC BY-SA 4.0"):
        assert part in text
    assert round(900 / plot.aspect_ratio) > 300 + CREDIT_LINE_HEIGHT * len(ATLAS_CREDIT)

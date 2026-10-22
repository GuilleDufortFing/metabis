"""Colour bars are read from the distribution that coloured the figure."""

import pandas as pd

from metabis.managers.color_manager import ColorManager
from metabis.managers.workspace_mode import AnalysisWorkspaceMode
from metabis.viz.color_bars import overview_color_bar, registered_color_bar

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
    # Reactions have a distribution here and genes none, so one bar.
    bars = [item for item in plot.right if isinstance(item, ColorBar)]
    assert [bar.title for bar in bars] == ["Reactions, Cohen's d"]

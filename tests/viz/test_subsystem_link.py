"""The network and the ranking carry their halves of the shared subsystem selection."""

import json

import pandas as pd
from bokeh.embed import json_item
from bokeh.models import ColumnDataSource, GlyphRenderer, HoverTool, TapTool
from bokeh.plotting import figure

from metabis.viz.plot_subsystem_ranking import plot_subsystem_comparison
from metabis.viz.ranking_layout import RANKING_ROWS_SOURCE
from metabis.viz.subsystem_link import FOCUS_EVENT, FOCUS_SOURCE, add_network_focus_link

from tests.viz.test_subsystem_ranking_hover import METRIC, SUBSYSTEM, _Workspace

NAVIGATE = "metabis:navigate-subsystem"


def _network():
    plot = figure()
    nodes = ColumnDataSource(
        {
            "x": [0.0, 10.0],
            "y": [0.0, 0.0],
            "radius": [2.0, 3.0],
            "element_id": ["a", "b"],
            "alpha": [1.0, 1.0],
            "is_subsystem": [True, True],
        }
    )
    renderer = plot.circle(x="x", y="y", radius="radius", source=nodes)
    plot.add_tools(HoverTool(renderers=[renderer], tooltips="@element_id"))
    return plot, nodes


def _serialized_names(model):
    return json.dumps(json_item(model))


def test_the_network_rings_the_focus_and_reports_what_is_pointed_at():
    plot, nodes = _network()

    focus = add_network_focus_link(plot)

    assert focus.name == FOCUS_SOURCE
    # No renderer draws the focus source, so it travels in the plot's tags.
    assert FOCUS_SOURCE in _serialized_names(plot)
    (callback,) = focus.js_property_callbacks["change:data"]
    assert {"nodes", "ring"} <= set(callback.args)
    (hover,) = plot.select(type=HoverTool)
    assert hover.callback.args["event"] == FOCUS_EVENT
    assert nodes.data["base_alpha"] == [1.0, 1.0]


def test_the_ranking_bands_its_rows_and_names_a_tapped_one():
    stats = pd.DataFrame({"subsystem": [SUBSYSTEM], "delta_score": [1.4]})
    layout = plot_subsystem_comparison(
        data_manager=_Workspace(),
        subsystem_stats=stats,
        metric=METRIC,
        return_plot=True,
        navigation_event=NAVIGATE,
    )
    text = _serialized_names(layout)

    assert FOCUS_SOURCE in text and RANKING_ROWS_SOURCE in text
    rows = layout.select_one({"name": RANKING_ROWS_SOURCE})
    assert rows.data == {"id": [SUBSYSTEM], "factor": [SUBSYSTEM]}
    taps = [tool for tool in layout.select({"type": TapTool}) if tool.renderers != "auto"]
    (band,) = taps[0].renderers
    (callback,) = band.data_source.selected.js_property_callbacks["change:indices"]
    assert callback.args["event"] == NAVIGATE
    # Both panels band the focused row, from the same source.
    focused = layout.select_one({"name": "metabis-ranking-focused"})
    drawn_by = [r for r in layout.select({"type": GlyphRenderer}) if getattr(r, "data_source", None) is focused]
    assert len(drawn_by) == 2


def test_only_the_networks_nodes_carry_its_subsystem_fields():
    """The names' source holds what they draw, so the nodes' source is the one found."""
    from bokeh.plotting import figure

    from metabis.viz.metabolic_visualization.rendering.bokeh_plotting import (
        NETWORK_LABELS,
        _draw_enriched_subsystems,
    )

    glyph = {
        "id": "glycolysis",
        "glyph_class": "subsystem",
        "x": 0.0,
        "y": 0.0,
        "alpha": 1.0,
        "display_label": "Glycolysis / Gluconeogenesis",
        "color": "#ffffff",
        "width": 50.0,
        "height": 50.0,
        "radius": 25.0,
        "is_subsystem": True,
    }
    plot = figure()
    _draw_enriched_subsystems(plot, [glyph])
    sources = [model for model in plot.references() if isinstance(model, ColumnDataSource)]
    (labels,) = [source for source in sources if source.name == NETWORK_LABELS]

    assert "is_subsystem" not in labels.data
    assert [source for source in sources if "is_subsystem" in source.data]

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
    # Every panel bands a search's matches from the same source, and outlines the
    # row in focus, never filling it.
    matched = layout.select_one({"name": "metabis-ranking-matched"})
    drawn_by = [r for r in layout.select({"type": GlyphRenderer}) if getattr(r, "data_source", None) is matched]
    assert len(drawn_by) == 3
    focused = layout.select_one({"name": "metabis-ranking-focused"})
    assert not [r for r in layout.select({"type": GlyphRenderer}) if getattr(r, "data_source", None) is focused]
    (outline,) = focused.js_property_callbacks["change:data"]
    assert len(outline.args["panels"]) == 3 and "metabisGradient" in outline.code


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


def _categories(plot):
    """The network's category circles and their numbered hexagons, as drawn."""
    from metabis.viz.metabolic_visualization.rendering.bokeh_plotting import (
        _draw_enriched_subsystem_categories,
    )

    glyph = {
        "id": "category:Carbohydrate Metabolism",
        "glyph_class": "subsystem_category",
        "label": "Carbohydrate Metabolism",
        "display_label": "Carbohydrate Metabolism",
        "x": 5.0,
        "y": 0.0,
        "radius": 12.0,
        "alpha": 0.4,
        "color": "#f8f8f8",
        "number": 1,
    }
    _draw_enriched_subsystem_categories(plot, [glyph])


def test_a_tap_on_a_category_selects_it_and_the_network_follows():
    from metabis.viz.subsystem_link import (
        CATEGORY_BADGE,
        CATEGORY_CIRCLES,
        CATEGORY_EVENT,
        CATEGORY_HEXES,
    )

    plot, _ = _network()
    _categories(plot)

    focus = add_network_focus_link(plot)

    circles = plot.select_one({"name": CATEGORY_CIRCLES})
    hexes = plot.select_one({"name": CATEGORY_HEXES})
    assert circles.data["category"] == ["Carbohydrate Metabolism"]
    assert hexes.data["badge"] == ["1"] and hexes.data["category"] == circles.data["category"]
    assert circles.data["base_alpha"] == [0.4]
    (callback,) = focus.js_property_callbacks["change:data"]
    assert callback.args["circles"] is circles and callback.args["hexes"] is hexes
    # The selected number takes the logo's hexagon, and the rings are outlined in
    # its gradient as segments, never filled.
    assert callback.args["badge"] is plot.select_one({"name": CATEGORY_BADGE})
    assert set(callback.args["ring"].data) == {"x0", "y0", "x1", "y1", "color"}
    assert "metabisRing" in callback.code and "tint" not in callback.code
    (tap,) = plot.js_event_callbacks["tap"]
    assert tap.args["event"] == CATEGORY_EVENT and tap.args["nodes"] is not None


def test_a_selected_categorys_subsystems_are_named_by_category():
    """The names' source says each node's category, which the selection names."""
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
        "display_label": "Glycolysis /\nGluconeogenesis",
        "name": "Glycolysis / Gluconeogenesis",
        "category": "Carbohydrate Metabolism",
        "color": "#ffffff",
        "line_dash": "dashed",
        "width": 50.0,
        "height": 50.0,
        "radius": 25.0,
        "is_subsystem": True,
    }
    plot = figure()
    nodes = _draw_enriched_subsystems(plot, [glyph]).data_source
    labels = plot.select_one({"name": NETWORK_LABELS})

    assert labels.data["category"] == ["Carbohydrate Metabolism"]
    # The full name, wrapped by the network, not the glyph's own line breaks.
    assert labels.data["label_text"] == ["Glycolysis /\nGluconeogenesis"]
    # A node with no data keeps its dashed edge a little heavier.
    assert nodes.data["line_width"] == [1.4]


def test_the_ranking_outlines_and_heads_the_selected_category():
    from metabis.viz.brand import SELECTED_HEXAGON
    from metabis.viz.ranking_layout import CHOSEN_CATEGORY, RANKING_HEADINGS_SOURCE

    stats = pd.DataFrame({"subsystem": [SUBSYSTEM], "delta_score": [1.4]})
    layout = plot_subsystem_comparison(
        data_manager=_Workspace(),
        subsystem_stats=stats,
        metric=METRIC,
        return_plot=True,
        navigation_event=NAVIGATE,
    )

    headings = layout.select_one({"name": RANKING_HEADINGS_SOURCE})
    (category,) = headings.data["category"]
    assert headings.data["factor"][0].startswith("heading:")
    chosen = layout.select_one({"name": CHOSEN_CATEGORY})
    callbacks = chosen.js_property_callbacks["change:data"]
    # The heading takes the logo's hexagon; the rows are outlined across the panels
    # by their category's edges.
    heading = next(c for c in callbacks if "badge" in c.args)
    assert heading.args["hexagon"] == SELECTED_HEXAGON
    outline = next(c for c in callbacks if "edges" in c.args)
    assert list(outline.args["edges"]) == [category]


def test_the_page_chooses_the_score_the_elements_or_both():
    from metabis.viz.ranking_layout import RANKING_PARTS, RANKING_SCORE

    stats = pd.DataFrame({"subsystem": [SUBSYSTEM], "delta_score": [1.4]})
    layout = plot_subsystem_comparison(
        data_manager=_Workspace(),
        subsystem_stats=stats,
        metric=METRIC,
        return_plot=True,
        navigation_event=NAVIGATE,
    )

    parts = layout.select_one({"name": RANKING_PARTS})
    assert parts.data == {"score": [True], "elements": [True]}
    assert RANKING_PARTS in _serialized_names(layout)
    (callback,) = parts.js_property_callbacks["change:data"]
    assert callback.args["grid"] is layout
    assert callback.args["score"] is layout.select_one({"name": RANKING_SCORE})
    # Each diamond is drawn whole: the score axis reaches past the largest score.
    score = callback.args["score"]
    assert score.x_range.end >= 1.4 * 1.25 and score.x_range.start == -score.x_range.end
    diamonds = [
        r for r in score.select({"type": GlyphRenderer}) if "score" in getattr(r.data_source, "data", {})
    ]
    (diamond,) = diamonds
    assert diamond.glyph.line_color == "#000000"

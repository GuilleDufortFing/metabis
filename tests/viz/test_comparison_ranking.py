"""Step 3's ranking of one comparison: one family's score in each row, its own events."""

from __future__ import annotations

import pandas as pd
import pytest
from bokeh.models import CustomJS, GlyphRenderer, HoverTool, Scatter, Text

from metabis.analysis.distribution_binner import DistributionBinner
from metabis.managers.hover import panel as hover_panel
from metabis.viz.color_reference import build_gradient_color_reference
from metabis.viz.plot_comparison_ranking import RankingScore, plot_comparison_ranking
from metabis.viz.ranking_layout import RANKING_NAMES, RANKING_PLOT, RankedSubsystem
from metabis.viz.subsystem_link import link_events

ENTRIES = [
    RankedSubsystem("glyc", "Glycolysis", "Carbohydrate", 0.9),
    RankedSubsystem("tca", "TCA cycle", "Energy", 0.5),
    RankedSubsystem("fao", "Fatty acid oxidation", "Lipid", 0.3),
]
SCORES = {"glyc": RankingScore(0.9, "12 of 30 significant"), "tca": RankingScore(-0.5), "fao": RankingScore(0.1)}


def _panel(layout, name):
    return next(model for model in layout.references() if getattr(model, "name", None) == name)


def _marks(plot):
    return [
        renderer
        for renderer in plot.renderers
        if isinstance(renderer, GlyphRenderer) and isinstance(renderer.glyph, Scatter) and renderer.data_source.data.get("y")
    ]


def test_each_subsystem_has_a_diamond_in_the_middle_of_its_row():
    layout = plot_comparison_ranking(ENTRIES, SCORES, events=link_events("comparison"))
    plot = _panel(layout, RANKING_PLOT)

    (mark,) = _marks(plot)

    assert mark.glyph.marker == "diamond"
    assert list(mark.data_source.data["y"]) == ["Glycolysis", "TCA cycle", "Fatty acid oxidation"]
    assert plot.title.text == "Delta score"


def test_a_subsystem_without_a_score_has_no_row():
    layout = plot_comparison_ranking(ENTRIES, {"glyc": SCORES["glyc"], "fao": SCORES["fao"]})

    (mark,) = _marks(_panel(layout, RANKING_PLOT))

    assert len(mark.data_source.data["x"]) == 2


def test_a_subsystem_not_over_represented_is_grey_at_zero():
    scores = {"glyc": RankingScore(2.1, "adjusted p 0.004"), "tca": RankingScore(None)}
    layout = plot_comparison_ranking(ENTRIES[:2], scores, kind="ora")
    plot = _panel(layout, RANKING_PLOT)

    (mark,) = _marks(plot)
    data = mark.data_source.data
    quiet = data["x"].index(0.0)

    assert data["color"][quiet] == "#c7c7c7"
    assert data["edge"][quiet] == "#949494"
    assert plot.x_range.end > 2.1


def test_the_headings_carry_no_numbers_and_the_figures_speak_the_pages_events():
    events = link_events("comparison")
    layout = plot_comparison_ranking(ENTRIES, SCORES, events=events)
    names = _panel(layout, RANKING_NAMES)

    # Names and headings only: no hexagon, no number.
    assert not [renderer for renderer in names.renderers if isinstance(renderer.glyph, Scatter)]
    texts = [renderer.glyph for renderer in names.renderers if isinstance(renderer.glyph, Text)]
    assert len(texts) == 2
    code = " ".join(
        str(model.args.get("event", "")) for model in layout.references() if isinstance(model, CustomJS)
    )
    assert events.focus in code and events.category in code and events.navigate in code
    assert "metabis:subsystem-focus" not in code


def test_a_row_hovers_and_is_coloured_as_its_node_in_the_network():
    node = hover_panel.panel(hover_panel.row("Subsystem", "Glycolysis"), hover_panel.row("Net direction", "higher"))
    scores = {**SCORES, "glyc": RankingScore(0.9, "12 of 30 significant", hover=node, color="#FFAAAA")}
    layout = plot_comparison_ranking(ENTRIES, scores)

    (mark,) = _marks(_panel(layout, RANKING_PLOT))
    data = mark.data_source.data
    glyc = data["y"].index("Glycolysis")
    tca = data["y"].index("TCA cycle")

    # The node's facts in the ranking's own panel; the rest keep the ranking's.
    assert data["hover"][glyc] == hover_panel.inner(node)
    assert data["color"][glyc] == "#FFAAAA"
    assert "Delta score" in data["hover"][tca] and data["color"][tca] != "#FFAAAA"


def test_a_colour_reference_marks_each_row_in_its_hover():
    binner = DistributionBinner(pd.Series({f"s{index}": index / 4 - 1 for index in range(9)}), center_method="zero", spread_method="std")
    histogram = [{**bar, "color": "#ff0000"} for bar in binner.get_histogram_data()]
    reference = build_gradient_color_reference(histogram, metric_label="log2(ER)")
    assert reference is not None
    layout = plot_comparison_ranking(ENTRIES[:2], {"glyc": RankingScore(1.2), "tca": RankingScore(None)}, kind="ora", reference=reference)
    plot = _panel(layout, RANKING_PLOT)

    (mark,) = _marks(plot)
    data = mark.data_source.data
    assert data["color_reference_value"] == [1.2, 0.0]
    assert data["color_reference_marker_x"] == [reference.marker_x(1.2), reference.marker_x(0.0)]
    (hover,) = [tool for tool in plot.tools if isinstance(tool, HoverTool) and mark in tool.renderers]
    assert "@color_reference_marker_x" in hover.tooltips and "@hover{safe}" in hover.tooltips


def test_a_ranking_needs_a_scored_subsystem():
    with pytest.raises(ValueError, match="No subsystem"):
        plot_comparison_ranking(ENTRIES, {})

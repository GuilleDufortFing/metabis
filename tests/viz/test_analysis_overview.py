"""The overview of an analysis of three groups or more, in the ranking's layout."""

from __future__ import annotations

import math

import pytest
from bokeh.models import BoxAnnotation, CustomJS, GlyphRenderer, HoverTool, Quad, Rect, TapTool

from metabis.viz.plot_analysis_overview import (
    OVERVIEW_COLUMNS,
    OverviewCell,
    overview_events,
    plot_analysis_overview,
)
from metabis.viz.ranking_layout import RANKING_NAMES, RANKING_PLOT, RankedSubsystem
from metabis.viz.subsystem_link import FOCUS_SOURCE

GROUPS = ["Epithelial | Healthy", "Epithelial | Inflamed", "Epithelial | Non-inflamed"]
ENTRIES = [
    RankedSubsystem("glyc", "Glycolysis", "Carbohydrate", 0.3),
    RankedSubsystem("fao", "Fatty acid oxidation", "Lipid", 0.2),
]
CELLS = {
    group: {"glyc": OverviewCell(0.5 - k * 0.4, "3 of 10 significant"), "fao": OverviewCell(None, measured=False)}
    for k, group in enumerate(GROUPS)
}


def _panel(layout, name):
    return next(model for model in layout.references() if getattr(model, "name", None) == name)


def _rects(plot):
    return [renderer for renderer in plot.renderers if isinstance(renderer, GlyphRenderer) and isinstance(renderer.glyph, Rect)]


def test_a_column_per_group_under_a_band_of_group_colours_with_its_name_reading_upward():
    colours = dict(zip(GROUPS, ["#c6dec7", "#8595e1", "#023fa5"]))
    layout = plot_analysis_overview(ENTRIES, GROUPS, CELLS, group_colors=colours, family="gene")
    plot = _panel(layout, RANKING_PLOT)
    cells, band = (renderer.data_source.data for renderer in _rects(plot))

    assert sorted(set(cells["x"])) == [0.5, 1.5, 2.5]
    assert band["color"] == ["#c6dec7", "#8595e1", "#023fa5"]
    (top,) = plot.above
    assert top.major_label_overrides[1.5] == "Epithelial | Inflamed"
    assert top.major_label_orientation == pytest.approx(math.pi / 2)
    assert not any(axis.visible for axis in plot.below)
    assert plot.title.text == "Genes: Delta score"
    # A family with no features in a subsystem is white and dashed.
    missing = [i for i, dash in enumerate(cells["dash"]) if dash == "dashed"]
    assert missing and all(cells["color"][i] == "#ffffff" for i in missing)
    # The names take their length above the rows, in both panels.
    assert _panel(layout, RANKING_NAMES).height == plot.height


def test_a_tap_anywhere_in_a_column_puts_its_comparison_in_view():
    layout = plot_analysis_overview(ENTRIES, GROUPS, CELLS, page="overview")
    plot = _panel(layout, RANKING_PLOT)

    (tap,) = plot.select({"type": TapTool})
    (target,) = tap.renderers
    assert isinstance(target.glyph, Quad)
    data = target.data_source.data
    assert data["group"] == GROUPS
    assert (data["left"][1], data["right"][1]) == (1.0, 2.0)
    # The band, two headings and two subsystems.
    assert target.glyph.bottom == 0 and target.glyph.top == 5
    events = {model.args.get("event") for model in layout.references() if isinstance(model, CustomJS)}
    assert overview_events("overview") in events


def test_the_rows_are_read_and_never_selected():
    layout = plot_analysis_overview(ENTRIES, GROUPS, CELLS)
    plot = _panel(layout, RANKING_PLOT)
    names = _panel(layout, RANKING_NAMES)

    assert not [tag for tag in plot.tags if getattr(tag, "name", None) == FOCUS_SOURCE]
    assert not names.select({"type": TapTool})
    assert not [tool for tool in names.select({"type": HoverTool}) if tool.callback is not None]
    assert not names.js_event_callbacks


def test_the_page_outlines_the_column_in_view_in_the_logos_gradient():
    layout = plot_analysis_overview(ENTRIES, GROUPS, CELLS)
    plot = _panel(layout, RANKING_PLOT)
    columns = next(tag for tag in plot.tags if getattr(tag, "name", None) == OVERVIEW_COLUMNS)

    callback = columns.js_property_callbacks["change:data"][0]

    assert callback.args["centres"] == {group: k + 0.5 for k, group in enumerate(GROUPS)}
    assert "metabisGradient" in callback.code
    assert not [box for box in plot.center if isinstance(box, BoxAnnotation) and box.line_dash == "dashed"]


def test_each_groups_note_and_its_other_side_are_in_its_hovers():
    layout = plot_analysis_overview(
        ENTRIES,
        GROUPS,
        CELLS,
        against={GROUPS[0]: "the rest of the chosen"},
        notes={GROUPS[0]: "120 significant reactions"},
    )
    cells, band = (renderer.data_source.data for renderer in _rects(_panel(layout, RANKING_PLOT)))

    assert "120 significant reactions" in band["hover"][0] and "the rest of the chosen" in band["hover"][0]
    assert "120 significant reactions" in cells["hover"][0]


def test_a_cell_can_hover_as_its_node_does_in_its_comparisons_network():
    node = "<div><strong>Net direction:</strong> 80.0% net higher in Target</div>"
    cells = {**CELLS, GROUPS[1]: {**CELLS[GROUPS[1]], "glyc": OverviewCell(0.1, "1 of 10 significant", hover=node)}}
    layout = plot_analysis_overview(ENTRIES, GROUPS, cells, against={GROUPS[1]: "the rest of the chosen"})
    data = _rects(_panel(layout, RANKING_PLOT))[0].data_source.data

    hovers = [hover for hover in data["hover"] if node in hover]
    assert len(hovers) == 1
    # Under its group, in place of the reading and its detail.
    assert GROUPS[1] in hovers[0] and "the rest of the chosen" in hovers[0] and "1 of 10" not in hovers[0]


def test_ora_cells_not_over_represented_are_grey():
    cells = {group: {"glyc": OverviewCell(None), "fao": OverviewCell(-2.5)} for group in GROUPS}
    layout = plot_analysis_overview(ENTRIES, GROUPS, cells, kind="ora")
    data = _rects(_panel(layout, RANKING_PLOT))[0].data_source.data

    assert data["color"][data["y"].index("Glycolysis")] == "#c7c7c7"
    assert data["dash"][data["y"].index("Glycolysis")] == "solid"


def test_the_overview_needs_columns():
    with pytest.raises(ValueError, match="needs a comparison"):
        plot_analysis_overview(ENTRIES, [], CELLS)


def test_features_read_their_cohens_d_under_the_group_each_is_highest_in():
    """The overview's other view: the features varying most, Cohen's d in each column."""
    headings = [f"Highest in {group}" for group in GROUPS]
    entries = [
        RankedSubsystem("R1", "ATP synthase", headings[2], 1.2),
        RankedSubsystem("R2", "Hexokinase", headings[0], 0.9),
    ]
    cells = {group: {"R1": OverviewCell(1.2 if k == 2 else 0.1), "R2": OverviewCell(None)} for k, group in enumerate(GROUPS)}
    layout = plot_analysis_overview(entries, GROUPS, cells, kind="effect", category_order=headings)
    plot = _panel(layout, RANKING_PLOT)

    assert plot.title.text == "Reactions: Cohen's d, varying most"
    # The groups' order, not the alphabet's: the first group's block at the top
    # (the factors run from the bottom up).
    factors = [factor for factor in reversed(plot.y_range.factors) if not factor.startswith(("heading", "band"))]
    assert factors == ["Hexokinase", "ATP synthase"]
    data = _rects(plot)[0].data_source.data
    assert data["color"][data["y"].index("Hexokinase")] == "#c7c7c7"
    assert "not significant" in data["hover"][data["y"].index("Hexokinase")]

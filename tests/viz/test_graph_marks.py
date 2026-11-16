"""Rings round a group's markers where the graphs draw them."""

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

from metabis import outline_elements


def _graph():
    plot = figure()
    plot.patches(
        xs="xs", ys="ys",
        source=ColumnDataSource({
            "x": [0.0, 10.0], "y": [0.0, 0.0], "element_id": ["r_MAR00001", "r_MAR00002"],
            "xs": [[0, 1, 0, -1], [10, 11, 10, 9]], "ys": [[-1, 0, 1, 0], [-1, 0, 1, 0]],
        }),
    )
    plot.rect(
        x="x", y="y", width="width", height="height",
        source=ColumnDataSource({"x": [5.0], "y": [5.0], "element_id": ["g_r_MAR00001_CD38"], "width": [4.0], "height": [1.0]}),
    )
    # A label over the reactions, which is never ringed.
    plot.text(x="x", y="y", text="element_id", source=ColumnDataSource({"x": [10.0], "y": [0.0], "element_id": ["r_MAR00002"]}))
    plot.circle(
        x="x", y="y", radius="radius",
        source=ColumnDataSource({"x": [20.0], "y": [20.0], "element_id": ["fatty_acid_oxidation"], "radius": [3.0]}),
    )
    return plot


def _rings(plot, before):
    return [renderer for renderer in plot.renderers[before:]]


def test_markers_are_ringed_round_their_own_shapes_in_the_groups_colour():
    plot = _graph()
    before = len(plot.renderers)

    ringed = outline_elements(plot, ["MAR00002_back", "CD38", "fatty_acid_oxidation", "absent"], color="#0072B2")

    assert ringed == 3
    rings = _rings(plot, before)
    assert [ring.glyph.__class__.__name__ for ring in rings] == ["Patches", "Rect", "Circle"]
    assert all(ring.glyph.line_color == "#0072B2" and ring.glyph.fill_alpha == 0 for ring in rings)
    diamond = rings[0].data_source.data
    # The reaction's diamond, larger, round the same centre.
    assert diamond["xs"][0][1] > 11 and diamond["xs"][0][3] < 9
    assert rings[2].data_source.data["radius"][0] > 3.0
    # The rings carry no element IDs, so the graphs' clicks and hovers never read them.
    assert all("element_id" not in ring.data_source.data for ring in rings)


def test_nothing_drawn_has_nothing_to_ring():
    plot = _graph()
    before = len(plot.renderers)

    assert outline_elements(plot, ["MAR09999"], color="#0072B2") == 0
    assert len(plot.renderers) == before

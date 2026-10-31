import json
import shutil
import subprocess

import pytest
from bokeh.models import GlyphRenderer, LabelSet
from bokeh.plotting import figure

from metabis.viz.metabolic_visualization.rendering.bokeh_plotting import (
    draw_graph_elements,
)


def test_label_sources_exclude_hover_payloads_used_only_by_glyphs():
    common = {
        "x": 1.0,
        "y": 2.0,
        "alpha": 1.0,
        "color": "#ffffff",
        "display_label": "Label",
        "hover": "<div>" + ("large hover payload " * 100) + "</div>",
    }
    glyphs = [
        {**common, "id": "m1", "glyph_class": "metabolite"},
        {
            **common,
            "id": "m2",
            "glyph_class": "metabolite",
            "subtype": "intermediate",
        },
        {**common, "id": "g1", "glyph_class": "gene"},
        {**common, "id": "r1", "glyph_class": "reaction"},
        {
            **common,
            "id": "c1",
            "glyph_class": "compartment",
            "width": 10,
            "height": 10,
            "line_color": "#000000",
            "x_label": 1.0,
            "y_label": 2.0,
        },
        {
            **common,
            "id": "s1",
            "glyph_class": "subsystem",
            "width": 10,
            "height": 10,
            "radius": 5,
        },
    ]
    plot = figure()

    draw_graph_elements(
        plot,
        enriched_glyphs=glyphs,
        enriched_connections=[],
        auxiliary_data={},
    )

    label_sets = list(plot.select({"type": LabelSet}))
    # The network's names are drawn twice from one source: white outlines, then the letters.
    assert len(label_sets) == 7
    assert len({id(labels.source) for labels in label_sets}) == 6
    assert all("hover" not in labels.source.data for labels in label_sets)
    assert all(labels.text in labels.source.data for labels in label_sets)

    glyph_sources = [
        renderer.data_source
        for renderer in plot.select({"type": GlyphRenderer})
        if "hover" in renderer.data_source.data
    ]
    assert len(glyph_sources) == 5
    assert all(source.data["hover"] == [common["hover"]] for source in glyph_sources)


def _zoomed(callback, spans) -> list:
    """Run a label set's zoom callback in Node for each view of ``spans`` (x, y),
    as a pan or zoom changes the plot's ranges: after each, whether its labels are
    shown and how many redraws the callback has asked for."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js runs the callback")
    script = """
    const [code, args, data, spans] = JSON.parse(process.argv[1]);
    let redraws = 0;
    const source = {data, change: {emit() { redraws += 1 }}};
    const labels = {visible: true};
    const seen = [];
    for (const [x, y] of spans) {
      const plot = {x_range: {start: 0, end: x}, y_range: {start: y, end: 0}};
      const values = {...args, source, labels, plot};
      const names = Object.keys(values);
      new Function(...names, code)(...names.map((name) => values[name]));
      seen.push([labels.visible, redraws, data.text_alpha[0]]);
    }
    console.log(JSON.stringify(seen));
    """
    args = {name: value for name, value in callback.args.items() if name not in ("source", "labels", "plot")}
    data = {key: list(callback.args["source"].data[key]) for key in ("x", "base_alpha", "text_alpha")}
    arguments = json.dumps([callback.code, args, data, spans])
    result = subprocess.run([node, "-e", script, arguments], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_labels_too_small_to_read_are_not_drawn_and_a_pan_redraws_none():
    glyphs = [
        {"x": 1.0, "y": 2.0, "alpha": 1.0, "color": "#ffffff", "display_label": "H2O", "id": "m1", "glyph_class": "metabolite"},
    ]
    # A whole compartment map in view: names far too small to read.
    plot = figure(x_range=(0, 20000), y_range=(15000, 0))
    draw_graph_elements(plot, enriched_glyphs=glyphs, enriched_connections=[], auxiliary_data={})
    (labels,) = plot.select({"type": LabelSet})
    # BokehJS draws every label of a set, transparent or not: the set is hidden instead.
    assert labels.source.data["text_alpha"] == [0.0] and labels.visible is False
    callbacks = [
        callback for callback in plot.x_range.js_property_callbacks["change:start"] if callback.args["source"] is labels.source
    ]
    assert len(callbacks) == 1 and callbacks[0].args["labels"] is labels

    seen = _zoomed(callbacks[0], [(20000, 15000), (20000, 15000), (2000, 1500), (2000, 1500), (2000, 1500), (20000, 15000)])
    # Zoomed out: hidden, one redraw when first laid out; moving the view redraws nothing.
    assert seen[:2] == [[False, 1, 0.0], [False, 1, 0.0]]
    # Zoomed in to read them: shown at full alpha, and panning there redraws nothing.
    assert seen[2:5] == [[True, 2, 1.0]] * 3
    # Out again: hidden.
    assert seen[5] == [False, 3, 0.0]

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
    assert len(label_sets) == 6
    assert all("hover" not in labels.source.data for labels in label_sets)
    assert all(labels.text in labels.source.data for labels in label_sets)

    glyph_sources = [
        renderer.data_source
        for renderer in plot.select({"type": GlyphRenderer})
        if "hover" in renderer.data_source.data
    ]
    assert len(glyph_sources) == 5
    assert all(source.data["hover"] == [common["hover"]] for source in glyph_sources)

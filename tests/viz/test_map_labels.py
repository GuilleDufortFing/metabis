"""A compartment map's names, set as Metabolic Atlas sets them: in map units, at its places and breaks."""

import json
from pathlib import Path

from bokeh.models import CustomJS, GlyphRenderer, LabelSet
from bokeh.plotting import figure

from metabis.viz.metabolic_visualization.rendering.bokeh_plotting import (
    draw_graph_elements,
)
from metabis.viz.metabolic_visualization.rendering.map_labels import (
    MAP_LABEL_MIN_PIXELS,
    MAP_LABEL_UNITS,
    MAP_NAMES_SHOWN_FROM,
    follow_breaks,
    gene_label_units,
    label_lines,
    map_label_columns,
    wrap_name,
)

COMPARTMENT_MAPS = Path(__file__).resolve().parents[2] / "src/metabis/resources/graphs/compartment/human1"


def test_a_name_breaks_where_metabolic_atlas_breaks_it():
    # Its own hyphen at a break stays; a break inside a word takes one.
    assert follow_breaks(
        "chondroitin sulfate C (GalNAc6S-GlcA) and D (GlcNAc6S-GlcA2S), precursor 1",
        ["chondroitin sulfate", "C (GalNAc6S-GlcA)", "and D (GlcNAc6S-", "GlcA2S), precursor 1"],
    ) == ["chondroitin sulfate", "C (GalNAc6S-GlcA)", "and D (GlcNAc6S-", "GlcA2S), precursor 1"]
    assert follow_breaks("17alpha-hydroxypregnenolone sulfate", ["17alpha-hydroxy-", "pregnenolone sulfate"]) == [
        "17alpha-hydroxy-",
        "pregnenolone sulfate",
    ]
    # Greek letters are their names; the model's spelling is kept.
    assert follow_breaks("5alpha-pregnane-3,20-dione", ["5α-pregnane-", "3,20-dione"]) == [
        "5alpha-pregnane-",
        "3,20-dione",
    ]
    assert follow_breaks("dihomo-gamma-linolenate", ["dihomo-δ-linolenate"]) is None


def test_a_long_name_metabolic_atlas_does_not_spell_is_wrapped_as_it_wraps_names():
    assert wrap_name("fucfucgalacglcgal14acglcgalgluside heparan sulfate") == [
        "fucfucgalacglcgal14-",
        "acglcgalgluside",
        "heparan sulfate",
    ]
    assert wrap_name("3-hydroxy-5alpha-pregnan-20-one sulfate") == ["3-hydroxy-5alpha-", "pregnan-20-one", "sulfate"]
    # Spelt otherwise, a short name stays on one line, and a long one wraps.
    assert label_lines("deoxycholate", {"lines": ["deoxy-", "cholic acid"]}) == ["deoxycholate"]
    assert label_lines("1-Eicosatrienoylglycerophosphocholine", {"lines": ["one"]}) == [
        "1-Eicosatrienoylgly-",
        "cerophosphocholine",
    ]


def test_a_metabolite_the_model_does_not_name_takes_metabolic_atlas_name():
    placed = {"lines": ["sulphate-conjugate-", "acetaminophen"]}

    assert label_lines("sulpacmp_c", placed, "sulpacmp_c") == ["sulphate-conjugate-", "acetaminophen"]
    assert label_lines("acetaminophen sulfate", placed, "sulpacmp_c") == ["acetaminophen sulfate"]


def test_a_small_metabolites_name_stands_beside_its_dot_but_a_copys_on_it():
    glyphs = [
        {"id": "ID1", "label": "MAM02039g", "x": 100.0, "y": 50.0, "display_label": "H+"},
        {"id": "ID1", "label": "MAM02039g", "x": 900.0, "y": 50.0, "display_label": "H+", "is_local_copy": True},
        {"id": "ID2", "label": "MAM01532g", "x": 0.0, "y": 0.0, "display_label": "chondroitin sulfate precursor"},
    ]
    placed = {"ID1": {"lines": ["H+"], "dx": -40.0, "dot": True}, "ID2": {"lines": ["chondroitin", "sulfate precursor"]}}

    columns = map_label_columns(glyphs, placed, metabolites=True)

    assert columns["label_x"] == [60.0, 900.0, 0.0]
    assert columns["label_text"] == ["H+", "H+", "chondroitin\nsulfate precursor"]


def _map_glyphs():
    common = dict(alpha=1.0, color="#cccccc", line_dash="solid")
    return [
        {**common, "id": "ID1", "label": "MAM02039g", "glyph_class": "metabolite", "subtype": "intermediate",
         "x": 100.0, "y": 50.0, "display_label": "H+"},
        {**common, "id": "ID2", "label": "MAM01532g", "glyph_class": "metabolite", "subtype": "main",
         "x": 0.0, "y": 0.0, "display_label": "chondroitin sulfate precursor"},
        {**common, "id": "ID3", "label": "ENSG00000109956", "glyph_class": "gene", "x": 0.0, "y": 200.0,
         "width": 60.0, "height": 30.0, "display_label": "B3GAT1"},
        {**common, "id": "MAR07335", "label": "MAR07335", "glyph_class": "reaction", "x": 200.0, "y": 0.0,
         "size": 40.0, "display_label": "MAR07335"},
    ]


def test_a_genes_name_is_as_large_as_its_box_takes():
    # A short symbol as large as a metabolite's name, a long one Metabolic Atlas's 12 units.
    assert gene_label_units("CS") == MAP_LABEL_UNITS["metabolite"]
    assert gene_label_units("ALAS1") == 16
    assert gene_label_units("SLC25A20") == MAP_LABEL_UNITS["gene"]
    # Never taller than its box.
    assert gene_label_units("CS", height=20) == 12


def test_a_maps_names_are_drawn_in_map_units_and_hidden_until_they_can_be_read():
    plot = figure(width=600, height=400)
    placed = {"ID1": {"lines": ["H+"], "dx": -40.0, "dot": True}, "ID2": {"lines": ["chondroitin", "sulfate precursor"]}}
    draw_graph_elements(plot, _map_glyphs(), [], {"atlas": {"links": {}, "labels": placed}})

    labels = {labelset.source.data["label_text"][0]: labelset for labelset in plot.select(type=LabelSet)}
    assert set(labels) == {"chondroitin\nsulfate precursor", "H+", "B3GAT1", "MAR07335"}
    for text, labelset in labels.items():
        # Only what the labels draw is sent: a map's sources hold thousands of rows.
        assert set(labelset.source.data) == {"label_x", "label_y", "label_text", "text_alpha"}
        assert (labelset.text_align, labelset.text_baseline) == ("center", "middle")
        # Hidden until the map is laid out, when its size sets theirs.
        assert labelset.visible is False
        (scale,) = [
            callback
            for callback in plot.y_range.js_property_callbacks["change:start"]
            if isinstance(callback, CustomJS) and callback.args.get("labels") is labelset
        ]
        kind = {"B3GAT1": "gene", "MAR07335": "reaction"}.get(text, "metabolite")
        assert scale.args["units"] == (gene_label_units(text) if kind == "gene" else MAP_LABEL_UNITS[kind])
        # Every kind appears at once, when a metabolite's name can be read.
        assert scale.args["shown_from"] == MAP_NAMES_SHOWN_FROM == MAP_LABEL_MIN_PIXELS / MAP_LABEL_UNITS["metabolite"]
    assert labels["H+"].source.data["label_x"] == [60.0]


def test_a_graph_without_metabolic_atlas_names_keeps_its_names_on_screen():
    plot = figure(width=600, height=400)
    draw_graph_elements(plot, _map_glyphs(), [], {"atlas": {"links": {}}})

    labelsets = list(plot.select(type=LabelSet))
    assert labelsets and all(labelset.text == "display_label" for labelset in labelsets)


def test_nothing_on_a_map_is_drawn_outside_its_frame():
    """BokehJS clips only the image, underlay and glyph levels to the frame: a gene
    box on a later level, beyond a narrowed map, was drawn over the key under it."""
    plot = figure(width=600, height=400)
    draw_graph_elements(plot, _map_glyphs(), [], {"atlas": {"links": {}, "labels": {}}})

    levels = {renderer.level for renderer in plot.renderers if isinstance(renderer, GlyphRenderer)}
    assert levels <= {"image", "underlay", "glyph"}


def test_every_compartment_map_places_its_metabolites_names():
    for layout in COMPARTMENT_MAPS.glob("*.json"):
        if layout.name.endswith(".atlas.json"):
            continue
        glyphs = json.loads(layout.read_text())["glyphs"]
        atlas = json.loads(layout.with_name(layout.stem + ".atlas.json").read_text())
        metabolites = {str(g["id"]) for g in glyphs if g["glyph_class"] == "metabolite"} | set(atlas["metabolites"])
        placed = atlas["labels"]
        # Nearly every metabolite has Metabolic Atlas's name, each one under a glyph of the layout.
        assert set(placed) <= metabolites
        assert len(placed) >= 0.95 * len(metabolites), layout.stem
        assert all(entry["lines"] and all(entry["lines"]) for entry in placed.values())

"""The packaged Human1 maps' import from Human-maps' SBGN and SVG files."""

import importlib.util
from pathlib import Path

from metabis.viz.metabolic_visualization.enrichment.connection_enrichment import (
    _nearest_ends,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "import_human_maps.py"


def _importer():
    spec = importlib.util.spec_from_file_location("import_human_maps", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# R1 turns M1 into M2 and is reversible, so the SBGN file draws M1 as a product
# too, on the reactant side's port. Metabolic Atlas draws R1 a second time, beside
# a second M2. Two touching gene boxes catalyse R1 by the first drawing, one more
# by the second.
SBGN = """<?xml version='1.0' encoding='UTF-8'?>
<sbgn xmlns="http://sbgn.org/libsbgn/0.3">
  <map id="test">
    <glyph class="compartment" id="comp1"><label text="Mitochondria"/><bbox x="0" y="0" w="10" h="10"/></glyph>
    <glyph class="simple chemical" id="met1"><label text="water"/><bbox x="93" y="93" w="14" h="14"/></glyph>
    <glyph class="simple chemical" id="met2"><label text="glucose"/><bbox x="270" y="80" w="60" h="40"/></glyph>
    <glyph class="simple chemical" id="met3"><label text="glucose"/><bbox x="870" y="80" w="60" h="40"/></glyph>
    <glyph class="process" id="process_R1">
      <label text="R1"/><bbox x="185" y="85" w="30" h="30"/>
      <port id="process_R1.1" x="180" y="100"/><port id="process_R1.2" x="220" y="100"/>
    </glyph>
    <glyph class="macromolecule" id="gene1"><label text="G1"/><bbox x="170" y="150" w="60" h="30"/></glyph>
    <glyph class="macromolecule" id="gene2"><label text="G2"/><bbox x="230" y="150" w="60" h="30"/></glyph>
    <glyph class="macromolecule" id="gene3"><label text="G1"/><bbox x="770" y="150" w="60" h="30"/></glyph>
    <arc class="production" id="process_R1_c0" source="process_R1.1" target="met1"/>
    <arc class="production" id="process_R1_p1" source="process_R1.2" target="met2"/>
    <arc class="production" id="process_R1_p2" source="process_R1.2" target="met3"/>
    <arc class="catalysis" id="process_R1_e0" source="gene1" target="process_R1"/>
  </map>
</sbgn>"""

SVG = """<svg>
  <g id="Mitochondria" class="compartment">
    <rect class="shape" x="500" y="400" width="300" height="200" rx="150" ry="150"/>
    <text y="60" transform="matrix(1,0,0,1,650,420)" font-size="72" text-anchor="middle">Mitochondria</text>
  </g>
  <g id="Extracellular" class="compartment">
    <path fill="none" stroke="#000" d="M1000 300L1000 900"/>
    <text y="60" transform="matrix(1,0,0,1,700,300)" font-size="72" text-anchor="middle">Extracellular</text>
  </g>
  <g id="Cytosol" class="compartment">
    <text y="60" transform="matrix(1,0,0,1,100,300)" font-size="72" text-anchor="middle">Cytosol</text>
  </g>
  <g id="R1" class="rea"><path class="shape" d="M0-30L30 0 0 30-30 0 0-30" transform="matrix(1,0,0,1,200,100)"/></g>
  <g id="R1" class="rea"><path class="shape" d="M0-30L30 0 0 30-30 0 0-30" transform="matrix(1,0,0,1,800,100)"/></g>
  <g class="met MAM1c R1" transform="matrix(1,0,0,1,100,100)"><circle class="shape" r="7"/></g>
  <g class="met MAM2c R1" transform="matrix(1,0,0,1,300,100)"><ellipse class="shape" rx="30" ry="20"/></g>
  <g class="met MAM2c R1" transform="matrix(1,0,0,1,900,100)"><ellipse class="shape" rx="30" ry="20"/></g>
  <g class="enz ENSG1 R1">
    <path class="shape" d="M0 0h60v30H0z" transform="matrix(1,0,0,1,170,150)"/>
  </g>
  <g class="enz ENSG2 R1">
    <path class="shape" d="M0 0h60v30H0z" transform="matrix(1,0,0,1,230,150)"/>
  </g>
  <g class="enz ENSG1 R1">
    <path class="shape" d="M0 0h60v30H0z" transform="matrix(1,0,0,1,770,150)"/>
  </g>
</svg>"""


def _layout(tmp_path):
    (tmp_path / "test.sbgn").write_text(SBGN)
    (tmp_path / "test.svg").write_text(SVG)
    return _importer().build_layout(tmp_path / "test.sbgn", tmp_path / "test.svg")


def test_a_maps_glyphs_take_the_identifiers_and_shapes_drawn_at_their_places(tmp_path):
    glyphs, _connections, report = _layout(tmp_path)
    by_id = {glyph["id"]: glyph for glyph in glyphs}

    assert [(by_id[m]["label"], by_id[m]["subtype"]) for m in ("met1", "met2", "met3")] == [
        ("MAM1c", "intermediate"),
        ("MAM2c", "main"),
        ("MAM2c", "main"),
    ]
    assert (by_id["met2"]["x"], by_id["met2"]["y"]) == (300.0, 100.0)
    assert [by_id[g]["label"] for g in ("gene1", "gene2", "gene3")] == ["ENSG1", "ENSG2", "ENSG1"]
    # Metabolic Atlas draws R1 twice: a glyph at each place, under its identifier.
    assert [(g["x"], g["y"]) for g in glyphs if g["glyph_class"] == "reaction"] == [(200.0, 100.0), (800.0, 100.0)]
    assert report["reactions"] == 1 and report["drawn_twice"] == 1 and not report["unplaced"]


def test_a_link_runs_from_the_reactant_side_whatever_its_arc_class(tmp_path):
    _glyphs, connections, _report = _layout(tmp_path)
    metabolic = {(c["source"], c["target"]) for c in connections if c["connection_class"] == "metabolic"}

    assert metabolic == {("met1", "R1"), ("R1", "met2"), ("R1", "met3")}


def test_a_reactions_touching_gene_boxes_are_one_enzyme_by_each_drawing(tmp_path):
    glyphs, connections, _report = _layout(tmp_path)
    groups = [glyph for glyph in glyphs if glyph["glyph_class"] == "enzyme_group"]

    assert [(group["enzymes"], group["glyph_ids"]) for group in groups] == [
        (["ENSG1", "ENSG2"], ["gene1", "gene2"]),
        (["ENSG1"], ["gene3"]),
    ]
    # The box round them, padded by 5.
    assert (groups[0]["x"], groups[0]["y"], groups[0]["width"], groups[0]["height"]) == (230.0, 165.0, 130.0, 40.0)
    regulatory = [(c["source"], c["target"]) for c in connections if c["connection_class"] == "regulatory"]
    assert regulatory == [(groups[0]["id"], "R1"), (groups[1]["id"], "R1")]


def test_a_compartment_is_the_box_round_its_outline_and_a_name_alone_a_point(tmp_path):
    glyphs, _connections, _report = _layout(tmp_path)
    boxes = {g["label"]: (g["x"], g["y"], g["width"], g["height"]) for g in glyphs if g["glyph_class"] == "compartment"}

    assert boxes["Mitochondria"] == (650.0, 500.0, 300.0, 200.0)
    # A membrane drawn as a line reaches out to the name beside it, whose left
    # edge sits about half its width left of its centre.
    left = 700 - 0.6 * 72 * len("Extracellular") / 2
    x, y, width, height = boxes["Extracellular"]
    assert (round(x - width / 2, 1), x + width / 2, y + height / 2) == (round(left, 1), 1000.0, 900.0)
    assert boxes["Cytosol"][2:] == (0.0, 0.0)


def test_a_link_ends_at_the_drawing_of_a_reaction_beside_it():
    first = {"id": "R1", "x": 200.0, "y": 100.0}
    second = {"id": "R1", "x": 800.0, "y": 100.0}
    group = {"id": "group", "x": 780.0, "y": 165.0}
    places = {"R1": [first, second], "group": [group]}

    assert _nearest_ends(places, "group", "R1") == (group, second)
    assert _nearest_ends(places, "group", "elsewhere") == (None, None)

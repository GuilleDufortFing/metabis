"""Metabolic Atlas's link paths: their import from Human-maps' SVG files, and their drawing."""

import importlib.util
import json
from pathlib import Path

from bokeh.plotting import figure

from metabis.viz.metabolic_visualization.rendering.bokeh_plotting import (
    _draw_atlas_links,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "import_atlas_paths.py"


def _importer():
    spec = importlib.util.spec_from_file_location("import_atlas_paths", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_path_of_straight_segments_reads_as_its_subpaths():
    polylines = _importer().polylines

    assert polylines("M10 10l5 0m-5 10h20v5Z") == [
        [(10.0, 10.0), (15.0, 10.0)],
        [(10.0, 20.0), (30.0, 20.0), (30.0, 25.0), (10.0, 20.0)],
    ]


SVG = """<svg>
<path class="fe R1" d="M110 100l40 0m-40 0l0 40"/>
<path class="feh R1" d="M150 100l-5 -3"/>
<path class="fe R2" d="M310 300l40 0"/>
<g id="tca" class="subsystem" data-bounds="0"><path class="shape" d="M0 0h100" fill="none"/></g>
<g id="R1" class="rea">
  <path class="shape" d="M0-30L30 0 0 30-30 0 0-30" stroke="#000" fill="#fff" transform="matrix(1,0,0,1,100.0,100.0)"/>
</g>
<g id="R2" class="rea">
  <path class="shape" d="M0-30L30 0 0 30-30 0 0-30" stroke="#000" fill="#fff" transform="matrix(1,0,0,1,300.0,300.0)"/>
</g>
<g id="R4" class="rea">
  <path class="shape" d="M0-30L30 0 0 30-30 0 0-30" stroke="#000" fill="#fff" transform="matrix(1,0,0,1,500.0,500.0)"/>
</g>
<g id="R5" class="rea">
  <path class="shape" d="M0-30L30 0 0 30-30 0 0-30" stroke="#000" fill="#fff" transform="matrix(1,0,0,1,700.0,700.0)"/>
</g>
<g class="met M1 R1" transform="matrix(1,0,0,1,160.0,100.0)"><circle/></g>
<g class="met Rtotal_c R1" transform="matrix(1,0,0,1,100.0,150.0)"><circle/></g>
</svg>"""


def test_a_reaction_keeps_the_paths_to_metabolites_the_layout_draws(tmp_path):
    svg = tmp_path / "lysosome.svg"
    svg.write_text(SVG)
    layout = tmp_path / "lysosome.json"
    # The layout is the SVG shifted by (5, 7), but R2 has moved and Rtotal_c is not drawn.
    layout.write_text(
        json.dumps(
            {
                "glyphs": [
                    {"id": "R1", "glyph_class": "reaction", "x": 105.0, "y": 107.0},
                    {"id": "R2", "glyph_class": "reaction", "x": 900.0, "y": 307.0},
                    {"id": "R4", "glyph_class": "reaction", "x": 505.0, "y": 507.0},
                    {"id": "R5", "glyph_class": "reaction", "x": 705.0, "y": 707.0},
                    {"id": "m1", "glyph_class": "metabolite", "x": 165.0, "y": 107.0},
                ],
                "connections": [],
            }
        )
    )
    data, report = _importer().import_map(svg, layout, "compartment")

    assert data["offset"] == [5.0, 7.0] and data["license"] == "CC BY-SA 4.0"
    # R1's path to m1 is kept, shifted; its path to Rtotal_c, which the layout does
    # not draw, is left out. R2 keeps its straight links.
    assert data["links"] == {"R1": [[115.0, 107.0, 155.0, 107.0]]}
    assert data["heads"] == {"R1": [[155.0, 107.0, 150.0, 104.0]]}
    assert data["subsystems"] == {"tca": [[5.0, 7.0, 105.0, 7.0]]}
    assert (report["routed"], report["dropped"]) == (1, 1)


def test_a_routed_reaction_is_drawn_along_its_paths_in_its_links_style():
    plot = figure()
    atlas = {"links": {"R1": [[0.0, 0.0, 10.0, 0.0, 10.0, 10.0]]}, "heads": {"R1": [[10.0, 10.0, 9.0, 8.0]]}}
    connections = [
        {"source": "m1", "target": "R1", "connection_class": "metabolic", "color": "#ff0000", "line_alpha": 0.9},
        {"source": "R1", "target": "m2", "connection_class": "metabolic", "color": "#ff0000", "line_alpha": 0.9},
        {"source": "R2", "target": "m2", "connection_class": "metabolic", "color": "#0000ff", "line_alpha": 0.9},
        {"source": "e1", "target": "R1", "connection_class": "regulatory", "color": "#7b68ee", "line_alpha": 0.9},
    ]

    remaining = _draw_atlas_links(plot, atlas, connections)

    # R1's metabolic links are drawn along the path; the rest stay straight.
    assert [(c["source"], c["target"]) for c in remaining] == [("R2", "m2"), ("e1", "R1")]
    sources = [r.data_source for r in plot.renderers]
    lines = next(s for s in sources if "glow_alpha" in s.data)
    assert lines.data["xs"] == [[0.0, 10.0, 10.0]] and lines.data["ys"] == [[0.0, 0.0, 10.0]]
    assert lines.data["color"] == ["#ff0000"] and lines.data["reaction_id"] == ["R1"]
    heads = next(s for s in sources if "glow_alpha" not in s.data)
    assert heads.data["xs"] == [[10.0, 9.0]]

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
    import math

    from bokeh.models import Line, Patches

    plot = figure()
    atlas = {
        "links": {
            "R1": [[0.0, 0.0, 10.0, 0.0, 10.0, 10.0]],
            "R3": [[20.0, 0.0, 30.0, 0.0]],
        },
        # Metabolic Atlas's arrowhead: a 10-unit base, its tip 14 above.
        "heads": {"R1": [[5.0, 0.0, 15.0, 0.0, 10.0, 14.0]]},
    }
    red = dict(connection_class="metabolic", color="#ff0000", line_alpha=0.9)
    connections = [
        {"source": "m1", "target": "R1", **red},
        {"source": "R1", "target": "m2", **red},
        {"source": "R3", "target": "m3", **red},
        {"source": "R2", "target": "m2", "connection_class": "metabolic", "color": "#0000ff", "line_alpha": 0.9},
        {"source": "e1", "target": "R1", "connection_class": "regulatory", "color": "#7b68ee", "line_alpha": 0.9},
    ]

    remaining = _draw_atlas_links(plot, atlas, connections)

    # R1's and R3's metabolic links are drawn along their paths; the rest stay straight.
    assert [(c["source"], c["target"]) for c in remaining] == [("R2", "m2"), ("e1", "R1")]
    lines = [r for r in plot.renderers if isinstance(r.glyph, Line)]
    # One line per style, not one per path: the red glow and the grey overlay.
    assert len(lines) == 2
    glow = next(r for r in lines if r.glyph.line_color == "#ff0000")
    xs = glow.data_source.data["x"]
    # Both reactions' paths in one line, broken by a gap.
    assert xs[:3] == [0.0, 10.0, 10.0] and math.isnan(xs[3]) and xs[4:] == [20.0, 30.0]
    # The arrowhead is a filled triangle over a halo in the reaction's colour, as
    # the straight links draw theirs, one triangle a row so only those in view
    # are drawn, and none while they would be too small to read.
    halo, head = [r for r in plot.renderers if isinstance(r.glyph, Patches)]
    assert halo.data_source.data["color"] == ["#ff0000"] and head.glyph.line_color is None
    assert [len(xs) for xs in head.data_source.data["xs"]] == [3]
    (hide,) = plot.x_range.js_property_callbacks["change:start"]
    assert hide.args["renderers"] == [halo, head]


def test_a_glyph_on_an_atlas_metabolite_is_a_metabolite_and_a_renamed_reaction_takes_its_paths(tmp_path):
    svg = tmp_path / "golgi.svg"
    svg.write_text(
        SVG.replace('<path class="fe R2" d="M310 300l40 0"/>', '<path class="fe MAR9" d="M710 700l40 0"/>')
        .replace('<g id="R5" class="rea">', '<g id="MAR9" class="rea">')
        + '<g class="met dolp_U_r R1" transform="matrix(1,0,0,1,300.0,100.0)"><circle/></g>'
    )
    layout = tmp_path / "golgi.json"
    layout.write_text(
        json.dumps(
            {
                "glyphs": [
                    {"id": "R1", "glyph_class": "reaction", "x": 105.0, "y": 107.0},
                    {"id": "R2", "glyph_class": "reaction", "x": 305.0, "y": 307.0},
                    {"id": "R4", "glyph_class": "reaction", "x": 505.0, "y": 507.0},
                    # Recon3D names for a reaction drawn where MAR9 is, and a metabolite.
                    {"id": "DOLPMT_U", "glyph_class": "reaction", "x": 705.0, "y": 707.0},
                    {"id": "dolp_U_r", "glyph_class": "reaction", "x": 305.0, "y": 107.0},
                    {"id": "m1", "glyph_class": "metabolite", "x": 165.0, "y": 107.0},
                ],
                "connections": [],
            }
        )
    )
    data, report = _importer().import_map(svg, layout, "compartment")

    assert data["metabolites"] == ["dolp_U_r"]
    assert data["links"]["DOLPMT_U"] == [[715.0, 707.0, 755.0, 707.0]]
    assert report["routed_by_place"] == 1


def test_a_layouts_metabolites_drawn_as_reactions_load_as_metabolites():
    from metabis.viz.metabolic_visualization.generation.utilities.cache_utils import (
        load_graph_resource,
    )

    layout = (
        Path(__file__).resolve().parents[2]
        / "src/metabis/resources/graphs/compartment/human1/endoplasmic_reticulum.json"
    )
    glyphs, _connections = load_graph_resource(str(layout))
    classes = {glyph["id"]: glyph["glyph_class"] for glyph in glyphs}

    assert classes["dolp_U_r"] == "metabolite" and classes["DOLPMT_U"] == "reaction"


def test_a_long_straight_link_ends_at_a_copy_of_its_far_node():
    from metabis.viz.metabolic_visualization.generation.data_structures.constants import (
        VisualizationConstants,
    )
    from metabis.viz.metabolic_visualization.rendering.bokeh_plotting import (
        _local_copies,
    )

    glyphs = [
        {"id": "hub", "glyph_class": "metabolite", "x": 0.0, "y": 0.0, "display_label": "dolichyl phosphate"},
        {"id": "R1", "glyph_class": "reaction", "x": 5000.0, "y": 0.0},
        {"id": "R2", "glyph_class": "reaction", "x": 0.0, "y": 5000.0},
        {"id": "R3", "glyph_class": "reaction", "x": 100.0, "y": 0.0},
    ]
    link = dict(connection_class="metabolic", color="#ff0000", line_alpha=1.0, arrowhead_xs=[1], arrowhead_ys=[1])
    connections = [
        {**link, "source": "R1", "target": "hub", "x0": 5000.0, "y0": 0.0, "x1": 0.0, "y1": 0.0},
        {**link, "source": "R2", "target": "hub", "x0": 0.0, "y0": 5000.0, "x1": 0.0, "y1": 0.0},
        {**link, "source": "R3", "target": "hub", "x0": 100.0, "y0": 0.0, "x1": 0.0, "y1": 0.0},
    ]

    glyphs, connections = _local_copies(glyphs, connections, {"links": {}})

    copies = [glyph for glyph in glyphs if glyph.get("is_local_copy")]
    stub = VisualizationConstants.LOCAL_COPY_STUB
    assert [(c["id"], c["x"], c["y"]) for c in copies] == [("hub", 5000.0 - stub, 0.0), ("hub", 0.0, 5000.0 - stub)]
    assert all(c["display_label"] == "dolichyl phosphate" for c in copies)
    # The links end at the copies, their arrowheads with them; the short one stays.
    assert all(abs(c["x1"] - c["x0"]) + abs(c["y1"] - c["y0"]) <= stub for c in connections[:2])
    assert connections[0]["arrowhead_xs"][0] == connections[0]["x1"]
    assert connections[2]["x0"] == 100.0


def test_a_reaction_is_a_diamond_taller_than_wide():
    from bokeh.models import Patches

    from metabis.viz.metabolic_visualization.rendering.bokeh_plotting import (
        _draw_enriched_reactions,
    )

    plot = figure()
    glyph = {
        "id": "MAR04370",
        "glyph_class": "reaction",
        "x": 0.0,
        "y": 0.0,
        "alpha": 1.0,
        "display_label": "MAR04370",
        "color": "#ff0000",
        "size": 40.0,
    }
    _draw_enriched_reactions(plot, [glyph])
    (diamond,) = [r for r in plot.renderers if isinstance(r.glyph, Patches)]
    xs, ys = diamond.data_source.data["xs"][0], diamond.data_source.data["ys"][0]

    assert (max(ys) - min(ys)) / (max(xs) - min(xs)) == 1.5
    # As the source keeps each reaction's place, a map still finds and frames it.
    assert diamond.data_source.data["x"] == [0.0] and "angle" in diamond.data_source.data


def test_the_logos_gradient_runs_from_its_blue_to_its_cyan():
    from metabis.viz.brand import gradient_color, gradient_segments

    assert gradient_color(0) == "#4facfe" and gradient_color(1) == "#00f2fe"
    pieces = gradient_segments([[0.0, 0.0, 300.0, 0.0]], piece=100.0)
    assert len(pieces["x0"]) == 3
    assert pieces["color"][0] != pieces["color"][-1]


def test_a_graph_keeps_every_link_drawn_while_its_view_moves():
    """Bokeh thins a glyph past 2,000 rows while the view is dragged; a map's links pass that."""
    from metabis.viz.metabolic_visualization.rendering.bokeh_plotting import (
        create_bokeh_plot,
    )

    glyph = {"id": "R1", "glyph_class": "reaction", "x": 0.0, "y": 0.0}
    plot = create_bokeh_plot(
        display_title="Map",
        width=600,
        height=400,
        aspect_ratio=None,
        sizing_mode="fixed",
        glyphs=[glyph],
        connections=[],
        auxiliary_data={},
        zoom=1.0,
        fit_view_to_bounds=True,
    )

    assert plot.lod_threshold is None


def test_an_atlas_arrowhead_becomes_the_straight_links_arrowhead_at_its_tip():
    from metabis.viz.metabolic_visualization.generation.data_structures.constants import (
        VisualizationConstants,
    )
    from metabis.viz.metabolic_visualization.rendering.bokeh_plotting import _arrowhead

    xs, ys = _arrowhead([5.0, 0.0, 15.0, 0.0, 10.0, 14.0])

    # The tip stays where Metabolic Atlas puts it, the head points away from its base.
    assert (xs[0], ys[0]) == (10.0, 14.0)
    assert all(y < 14.0 for y in ys[1:])
    length = VisualizationConstants.ARROWHEAD_SIZE
    assert all(abs(((x - 10.0) ** 2 + (y - 14.0) ** 2) ** 0.5 - length) < 1e-9 for x, y in zip(xs[1:], ys[1:]))

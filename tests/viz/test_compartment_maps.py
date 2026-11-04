"""The compartment maps: their names, their reactions, and the links into and out of them."""

import json
import shutil
import subprocess

import pytest
from bokeh.models import ColumnDataSource, HoverTool, TapTool
from bokeh.plotting import figure

from metabis import MetabolicModel
from metabis.viz.atlas_paths import ATLAS_SUFFIX, atlas_paths
from metabis.viz.click_behavior import OWN_TAP_TAG, add_click_behavior
from metabis.viz.compartment_maps import (
    ATLAS_MAPS,
    BAND_GLOW,
    COMPARTMENT_MAPS,
    MAP_BAND_SOURCE,
    MAP_ELEMENT_EVENT,
    MAP_FIND_SOURCE,
    MAP_REACTION_EVENT,
    MAPS_DIRECTORY,
    TRANSPORT_MAPS,
    add_map_bands,
    add_map_find,
    add_map_links,
    band_name,
    highlight_reactions,
    keep_map_scale,
    map_bands,
    map_ids_for_reactions,
    map_reaction_ids,
    map_reaction_names,
    map_subsystems,
    maps_for_reactions,
    resolve_map_reactions,
)
from metabis.viz.metabolic_visualization.rendering.map_labels import (
    MAP_NAMES_SHOWN_FROM,
)

TCA = "tricarboxylic_acid_cycle_and_glyoxylate_dicarboxylate_metabolism"


@pytest.fixture(scope="module")
def model():
    return MetabolicModel(use_cache=True)


def test_every_packaged_map_is_named_once():
    packaged = {path.stem for path in MAPS_DIRECTORY.glob("*.json") if not path.name.endswith(ATLAS_SUFFIX)}
    assert {entry.map_id for entry in ATLAS_MAPS} == packaged and len(ATLAS_MAPS) == len(packaged)
    # The cytosol's five maps are named by their largest pathways.
    cytosol = [entry.name for entry in COMPARTMENT_MAPS if entry.compartment == "c"]
    assert len(cytosol) == 5 and all("(" in name for name in cytosol)
    # Metabolic Atlas's twelve transport maps follow, the plasma membrane's five
    # named by their largest transporter families.
    assert {entry.group for entry in COMPARTMENT_MAPS} == {"compartment"}
    assert len(TRANSPORT_MAPS) == 12 and {entry.group for entry in TRANSPORT_MAPS} == {"transport"}
    plasma = [entry.name for entry in TRANSPORT_MAPS if entry.compartment == "e"]
    assert len(plasma) == 5 and all(name.startswith("Transport, plasma membrane") and "(" in name for name in plasma)


def test_a_maps_reactions_resolve_to_the_models_own(model):
    drawn = map_reaction_ids("mitochondria")
    resolved = resolve_map_reactions(model, "mitochondria")
    own = set(model.get_reactions())

    assert len(drawn) > 1000
    assert set(resolved.values()) <= own
    # Each carries Human-GEM's identifier, and the maps are fitted to the model's
    # release, Human-GEM 2.1.1: every reaction they draw is the model's.
    assert len(resolved) == len(drawn)


def test_a_subsystem_leads_to_the_maps_of_its_compartments(model):
    reactions = model.get_reactions_for_subsystem(TCA)
    maps = maps_for_reactions(model, reactions)

    assert maps[0][0].map_id == "mitochondria"
    assert [count for _, count in maps] == sorted((count for _, count in maps), reverse=True)
    drawn = map_ids_for_reactions(model, "mitochondria", reactions)
    assert len(drawn) >= maps[0][1]
    # And back: each reaction a map draws names its subsystem, and has a short name.
    named = map_subsystems(model, "mitochondria")
    assert {named[identifier][0] for identifier in drawn} == {TCA}
    short = map_reaction_names(model, "mitochondria")
    assert set(short) == set(resolve_map_reactions(model, "mitochondria"))
    assert sum(short[identifier] != identifier for identifier in drawn) > len(drawn) / 2


def test_a_transport_subsystem_leads_first_to_its_transport_map(model):
    maps = maps_for_reactions(model, model.get_reactions_for_subsystem("transport_mitochondrial"))

    assert [entry.map_id for entry, _count in maps[:2]] == ["transport_mitochondria", "mitochondria"]
    # The transport maps of the plasma membrane draw the extracellular transport.
    extracellular = maps_for_reactions(model, model.get_reactions_for_subsystem("transport_extracellular"))
    assert {entry.map_id for entry, _count in extracellular[:5]} == {
        f"transport_plasma_membrane_{number}" for number in range(1, 6)
    }


def test_a_transport_map_comes_from_metabolic_atlas_subsystem_maps_without_a_band():
    for entry in TRANSPORT_MAPS:
        atlas = atlas_paths(str(MAPS_DIRECTORY / f"{entry.map_id}.json"))
        assert atlas["source"].endswith(f"/subsystem/svg/{entry.map_id}.svg")
        # Its one band, "Transport reactions", lay under the whole map.
        assert map_bands(entry.map_id) == {} and atlas["labels"]


def _map_plot():
    plot = figure(x_range=(0, 1000), y_range=(1000, 0))
    reactions = ColumnDataSource(
        {
            "x": [100.0, 500.0, 900.0],
            "y": [100.0, 500.0, 900.0],
            "element_id": ["MAR1", "MAR2_back", "MAR3"],
            "display_label": ["one", "two", "three"],
            "angle": [0.0] * 3,
            "alpha": [1.0] * 3,
        }
    )
    metabolites = ColumnDataSource(
        {
            "x": [300.0],
            "y": [300.0],
            "element_id": ["m1"],
            "model_id": ["MAM01965m"],
            "display_label": ["glucose"],
            "alpha": [1.0],
            "size": [20],
            "hover": ["glucose"],
        }
    )
    # A map draws a gene beside each reaction it catalyses, labelled with its symbol.
    genes = ColumnDataSource(
        {
            "x": [110.0, 510.0, 890.0],
            "y": [110.0, 510.0, 890.0],
            "element_id": ["g1", "g2", "g3"],
            "model_id": ["ENSG00000138796", "ENSG00000084754", "ENSG00000138796"],
            "display_label": ["HADH", "HADHA", "HADH"],
            "alpha": [1.0] * 3,
            "width": [60] * 3,
            "height": [30] * 3,
            "hover": ["HADH", "HADHA", "HADH"],
        }
    )
    plot.scatter(x="x", y="y", source=reactions)
    plot.scatter(x="x", y="y", source=metabolites)
    plot.rect(x="x", y="y", width="width", height="height", source=genes)
    return plot, reactions, metabolites


def test_a_subsystems_reactions_are_framed_and_nothing_fades():
    plot, reactions, metabolites = _map_plot()
    x_range, y_range = plot.x_range, plot.y_range

    assert highlight_reactions(plot, ["MAR2", "MAR3"], padding=100) == 2
    assert reactions.data["alpha"] == [1.0, 1.0, 1.0]
    assert metabolites.data["alpha"] == [1.0]
    # Framed at the map's proportions, the y-range still running downwards, on the
    # plot's own range objects, which its names and band follow as it zooms.
    assert plot.x_range is x_range and plot.y_range is y_range
    assert (plot.x_range.start, plot.x_range.end) == (400.0, 1000.0)
    assert plot.y_range.start > plot.y_range.end
    assert highlight_reactions(_map_plot()[0], ["elsewhere"]) == 0


def test_a_tapped_reaction_offers_its_subsystem():
    plot, reactions, metabolites = _map_plot()
    add_map_links(plot, {"MAR2": ("glycolysis", "Glycolysis")})

    assert reactions.data["subsystem_id"] == ["", "glycolysis", ""]
    assert reactions.data["subsystem"] == ["", "Glycolysis", ""]
    assert plot.select({"type": TapTool})
    (callback,) = reactions.selected.js_property_callbacks["change:indices"]
    assert callback.args["event"] == MAP_REACTION_EVENT
    # The map answers every tap itself: its genes' and metabolites' show their details.
    (callback,) = metabolites.selected.js_property_callbacks["change:indices"]
    assert (callback.args["event"], callback.args["kind"]) == (MAP_ELEMENT_EVENT, "metabolite")
    assert OWN_TAP_TAG in reactions.tags and OWN_TAP_TAG in metabolites.tags


def _tap(source, index) -> dict:
    """Run a source's tap callbacks in Node, in order, as BokehJS does on a tap at ``index``."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js runs the tap")
    callbacks = [
        [callback.code, {name: None if name == "source" else value for name, value in callback.args.items()}]
        for callback in source.selected.js_property_callbacks["change:indices"]
    ]
    script = """
    const [callbacks, data, tags, index] = JSON.parse(process.argv[1]);
    const opened = [], events = [];
    globalThis.window = {dispatchEvent: (event) => events.push(event.detail)};
    globalThis.CustomEvent = class { constructor(type, init) { this.detail = init.detail } };
    globalThis.document = {createElement: () => ({click() { opened.push(this.href) }}), body: {appendChild() {}, removeChild() {}}};
    const source = {tags, data, selected: {indices: [index]}};
    for (const [code, args] of callbacks) {
      const names = Object.keys(args);
      new Function('cb_obj', ...names, code)(source.selected, ...names.map((name) => name === 'source' ? source : args[name]));
    }
    setTimeout(() => console.log(JSON.stringify({opened, events, selected: source.selected.indices})), 10);
    """
    arguments = json.dumps([callbacks, dict(source.data), list(source.tags), index])
    result = subprocess.run([node, "-e", script, arguments], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_no_tap_in_a_map_opens_a_page_each_names_what_was_tapped():
    plot, reactions, metabolites = _map_plot()
    reactions.data["external_url"] = [f"https://metabolicatlas.org/reaction/MAR{i}" for i in (1, 2, 3)]
    metabolites.data["external_url"] = ["https://metabolicatlas.org/metabolite/m1"]
    # As the API draws a map: the click behaviour first, then the map's links.
    add_click_behavior(plot, download_flag="metabisDownloadHoverOnClick")
    add_map_links(plot, {"MAR2": ("glycolysis", "Glycolysis")}, {"MAR2": "glucose → G6P"})

    tapped = _tap(reactions, 1)
    assert tapped["opened"] == []
    assert tapped["events"] == [
        {
            "element_id": "MAR2_back",
            "model_id": "MAR2_back",
            "label": "two",
            "name": "glucose → G6P",
            "subsystem_id": "glycolysis",
            "subsystem": "Glycolysis",
        }
    ]
    # Nothing is left selected, so no other reaction is drawn as unselected.
    assert tapped["selected"] == []
    # A metabolite's tap names it for its details, as a gene's does, and opens no page.
    assert _tap(metabolites, 0) == {
        "opened": [],
        "events": [{"kind": "metabolite", "element_id": "m1", "model_id": "MAM01965m", "label": "glucose"}],
        "selected": [],
    }


def test_a_find_marks_reactions_genes_and_metabolites_by_name():
    plot, reactions, metabolites = _map_plot()
    find = add_map_find(plot)

    assert find.name == MAP_FIND_SOURCE and find in plot.tags
    (callback,) = find.js_property_callbacks["change:data"]
    assert reactions in callback.args["sources"] and metabolites in callback.args["sources"]
    # A map zooms in to at least a fifth of its width; a subsystem's graph keeps its zoom.
    assert callback.args["closest"] == 160.0
    plain = figure(x_range=(0, 1000), y_range=(1000, 0))
    plain.scatter("x", "y", source=ColumnDataSource({"x": [1.0], "y": [1.0], "display_label": ["a"]}))
    (kept,) = add_map_find(plain, zoom=False).js_property_callbacks["change:data"]
    assert kept.args["closest"] == 0.0
    # The ring, a fixed size on screen, and the border round each match's shape, in
    # the logo's gradient on the frame-clipped glyph level; the ring goes once the
    # elements can be seen, as the names appear.
    ring = next(renderer for renderer in plot.renderers if type(renderer.glyph).__name__ == "Arc")
    border = next(renderer for renderer in plot.renderers if type(renderer.glyph).__name__ == "Segment")
    assert ring.glyph.radius_units == "screen" and ring.level == border.level == "glyph"
    assert ring.glyph.line_color == border.glyph.line_color == "color"
    (near,) = plot.y_range.js_property_callbacks["change:start"][-1:]
    assert near.args["rings"] is ring and near.args["shown_from"] == MAP_NAMES_SHOWN_FROM


def _find(plot, query: str, *, exact: bool = False, index: int = 0, still: bool = False) -> dict:
    """Run a map's find callback in Node on ``query``: what it reports and where it centres."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js runs the find")
    find = next(source for source in plot.tags if getattr(source, "name", None) == MAP_FIND_SOURCE)
    (callback,) = find.js_property_callbacks["change:data"]
    sources = [dict(source.data) for source in callback.args["sources"]]
    script = """
    const [code, sources, shapes, closest, pieces, event, query, exact, index, still] = JSON.parse(process.argv[1]);
    const events = [];
    globalThis.window = {dispatchEvent: (dispatched) => events.push(dispatched.detail)};
    globalThis.CustomEvent = class { constructor(type, init) { this.detail = init.detail } };
    const range = (start, end) => ({start, end, setv(values) { Object.assign(this, values) }});
    const plot = {x_range: range(0, 1000), y_range: range(1000, 0)};
    const ring = {data: null};
    const border = {data: null};
    const find = {data: {query: [query], exact: [exact], index: [index], still: [still]}};
    new Function('find', 'sources', 'shapes', 'ring', 'border', 'plot', 'closest', 'event', 'pieces', code)(
      find, sources.map((data) => ({data})), shapes, ring, border, plot, closest, event, pieces);
    console.log(JSON.stringify({found: events[0], ringed: ring.data.x.length / pieces,
      edges: border.data.x0.length, colours: new Set(border.data.color).size,
      centre: (plot.x_range.start + plot.x_range.end) / 2}));
    """
    arguments = json.dumps(
        [
            callback.code,
            sources,
            callback.args["shapes"],
            callback.args["closest"],
            callback.args["pieces"],
            callback.args["event"],
            query,
            exact,
            index,
            still,
        ]
    )
    result = subprocess.run([node, "-e", script, arguments], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_a_find_matches_exactly_when_asked_and_steps_through_its_matches():
    plot = _map_plot()[0]
    add_map_find(plot)

    # Any name that holds the query, or only the one that is it, as a gene's link asks.
    assert _find(plot, "hadh")["found"]["count"] == 3
    exact = _find(plot, "HADH", exact=True)
    assert (exact["found"]["count"], exact["ringed"]) == (2, 2)
    # Each found shape bordered in short pieces of the gradient.
    assert exact["edges"] > 8 and exact["colours"] > 4
    # Each match in turn, counting round, as Metabolic Atlas's map search steps.
    assert _find(plot, "HADH", exact=True, index=1)["centre"] == 890.0
    stepped_back = _find(plot, "HADH", exact=True, index=-1)
    assert (stepped_back["found"]["index"], stepped_back["centre"]) == (1, 890.0)
    assert _find(plot, "HADH", exact=True, index=2)["centre"] == 110.0
    # A gene or a metabolite by the model's identifier, which its label does not show.
    assert _find(plot, "ENSG00000138796", exact=True)["found"]["count"] == 2
    assert _find(plot, "MAM01965m", exact=True)["found"] == {"query": "mam01965m", "count": 1, "index": 0, "label": "glucose", "still": False}
    # A click marks every place the element is drawn, the view where it is.
    clicked = _find(plot, "ENSG00000138796", exact=True, index=1, still=True)
    assert (clicked["found"]["count"], clicked["found"]["still"], clicked["ringed"], clicked["centre"]) == (2, True, 2, 500.0)


def test_every_human1_map_carries_metabolic_atlas_paths():
    """Each packaged Human1 layout has its Atlas paths beside it, in its own frame."""
    graphs = MAPS_DIRECTORY.parent.parent
    layouts = [
        path
        for kind in ("compartment", "subsystem")
        for path in (graphs / kind / "human1").glob("*.json")
        if not path.name.endswith(ATLAS_SUFFIX)
    ]
    assert len(layouts) == 130
    atlas = atlas_paths(str(MAPS_DIRECTORY / "mitochondria.json"))
    assert atlas["license"] == "CC BY-SA 4.0"
    # Most of the map's reactions are routed, each path at least one segment.
    assert len(atlas["links"]) > 1000
    assert all(len(flat) >= 4 and len(flat) % 2 == 0 for paths in atlas["links"].values() for flat in paths)
    assert len(atlas["subsystems"]) > 50
    assert all(atlas_paths(str(path)) is not None for path in layouts)
    # A layout of the standard kind has none.
    assert atlas_paths(str(graphs / "subsystem" / "standard" / "nowhere.json")) is None


def test_one_subsystems_band_at_a_time():
    bands = map_bands("mitochondria")
    shown = band_name(bands, "tricarboxylic acid cycle and glyoxylate/dicarboxylate metabolism")
    assert shown == "Tricarboxylic acid cycle and glyoxylate/dicarboxylate metabolism"
    plot = _map_plot()[0]
    add_map_links(plot, {"MAR2": ("tca", shown)})
    hovers = list(plot.select(type=HoverTool))

    chosen = add_map_bands(plot, "mitochondria", shown)

    assert chosen.name == MAP_BAND_SOURCE and chosen.data["name"] == [shown]
    assert chosen in plot.tags
    # A soft yellow glow, no line: strokes each narrower and yellower than the one
    # under it, all opaque and drawn from one source, under everything else the
    # map draws, on the lowest level, with the links, which come after it.
    glow = plot.renderers[: len(BAND_GLOW)]
    assert [stroke.glyph.line_color for stroke in glow] == [color for _units, color in BAND_GLOW]
    assert [units for units, _color in BAND_GLOW] == sorted((units for units, _color in BAND_GLOW), reverse=True)
    assert len({stroke.data_source for stroke in glow}) == 1
    assert all(stroke.level == "image" and stroke.glyph.line_alpha == 1.0 for stroke in glow)
    band = glow[0].data_source.data
    assert band["xs"] and len(band["xs"]) == len(band["ys"])
    # Only the page changes the band: nothing redraws it under the pointer, which
    # would repaint the whole map at every reaction the pointer crossed.
    (show,) = chosen.js_property_callbacks["change:data"]
    assert show.args["chosen"] is chosen
    assert list(plot.select(type=HoverTool)) == hovers
    assert add_map_bands(_map_plot()[0], "nowhere") is None


def _resized(plot, frames, x=(0.0, 2000.0), y=(1000.0, 0.0)) -> list:
    """Run a map's scale keeper in Node as its frame takes each (width, height): its ranges after each.

    A frame given as ``{"x": [start, end]}`` is the reader panning the map there instead.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js runs the resize")
    (callback,) = plot.js_property_callbacks["change:inner_width"]
    script = """
    const [code, frames, x, y] = JSON.parse(process.argv[1]);
    const range = (start, end) => ({start, end, setv(values) { Object.assign(this, values) }});
    const plot = {x_range: range(x[0], x[1]), y_range: range(y[0], y[1])};
    const seen = [];
    for (const frame of frames) {
      if (frame.x) { plot.x_range.start = frame.x[0]; plot.x_range.end = frame.x[1] }
      else {
        [plot.inner_width, plot.inner_height] = frame;
        new Function('plot', code)(plot);
      }
      seen.push([plot.x_range.start, plot.x_range.end, plot.y_range.start, plot.y_range.end]);
    }
    console.log(JSON.stringify(seen));
    """
    arguments = json.dumps([callback.code, frames, x, y])
    result = subprocess.run([node, "-e", script, arguments], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_a_map_fits_its_plot_then_is_cut_never_squeezed():
    plot = _map_plot()[0]
    keep_map_scale(plot)

    assert set(plot.js_property_callbacks) == {"change:inner_width", "change:inner_height"}
    # First laid out in a frame of the screen's shape, the map drawn 2,000 by 1,000
    # units is shown whole and centred at one scale, 0.4 pixels a unit across and down.
    assert _resized(plot, [(800, 600)])[0] == [0, 2000, 1250, -250]
    # Two map units a pixel across and down; narrowed to 600 px at the same height by a
    # sidebar, it shows 1,200 units, its right edge kept; widened again, all of it.
    assert [frame[:2] for frame in _resized(plot, [(1000, 500), (600, 500), (1000, 500)])] == [
        [0, 2000],
        [800, 2000],
        [0, 2000],
    ]
    # A box that takes its height after the map's first layout fits it again, whole,
    # until the reader moves it: 0.25 pixels a unit when 300 px high, then 0.5.
    assert _resized(plot, [(1000, 250), (1000, 500)]) == [[-1000, 3000, 1000, 0], [0, 2000, 1000, 0]]
    assert _resized(plot, [(1000, 250), (600, 250), (1000, 500)])[2] == [0, 2000, 1000, 0]
    # Once panned, a lower frame keeps the scale, the map cut from its bottom, its top kept.
    assert _resized(plot, [(1000, 500), {"x": [100, 2100]}, (1000, 400)])[2] == [100, 2100, 800, 0]
    # A data range without its span yet is left to fit itself.
    assert _resized(plot, [(600, 500)], x=(None, None))[0][:2] == [None, None]

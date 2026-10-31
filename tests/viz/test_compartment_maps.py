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
    COMPARTMENT_MAPS,
    MAP_BAND_SOURCE,
    MAP_FIND_SOURCE,
    MAP_REACTION_EVENT,
    MAPS_DIRECTORY,
    add_map_bands,
    add_map_find,
    add_map_links,
    band_name,
    highlight_reactions,
    map_bands,
    map_ids_for_reactions,
    map_reaction_ids,
    map_reaction_names,
    map_subsystems,
    maps_for_reactions,
    resolve_map_reactions,
)

TCA = "tricarboxylic_acid_cycle_and_glyoxylate_dicarboxylate_metabolism"


@pytest.fixture(scope="module")
def model():
    return MetabolicModel(use_cache=True)


def test_every_packaged_map_is_named_once():
    packaged = {path.stem for path in MAPS_DIRECTORY.glob("*.json") if not path.name.endswith(ATLAS_SUFFIX)}
    assert {entry.map_id for entry in COMPARTMENT_MAPS} == packaged
    # The cytosol's five maps are named by their largest pathways.
    cytosol = [entry.name for entry in COMPARTMENT_MAPS if entry.compartment == "c"]
    assert len(cytosol) == 5 and all("(" in name for name in cytosol)


def test_a_maps_reactions_resolve_to_the_models_own(model):
    drawn = map_reaction_ids("mitochondria")
    resolved = resolve_map_reactions(model, "mitochondria")
    own = set(model.get_reactions())

    assert len(drawn) > 1000
    assert set(resolved.values()) <= own
    # Most carry Human1's identifiers; Recon3D's and BiGG's bring in more.
    assert sum(identifier in own for identifier in drawn) < len(resolved) <= len(drawn)


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
        {"x": [300.0], "y": [300.0], "element_id": ["m1"], "display_label": ["glucose"], "alpha": [1.0]}
    )
    plot.scatter(x="x", y="y", source=reactions)
    plot.scatter(x="x", y="y", source=metabolites)
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
    # The map answers its reactions' taps; metabolites keep the click behaviour's.
    assert OWN_TAP_TAG in reactions.tags and OWN_TAP_TAG not in metabolites.tags


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


def test_a_reaction_tap_in_a_map_opens_no_page_but_a_metabolite_tap_does():
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
            "label": "two",
            "name": "glucose → G6P",
            "subsystem_id": "glycolysis",
            "subsystem": "Glycolysis",
        }
    ]
    # Nothing is left selected, so no other reaction is drawn as unselected.
    assert tapped["selected"] == []
    assert _tap(metabolites, 0) == {
        "opened": ["https://metabolicatlas.org/metabolite/m1"],
        "events": [],
        "selected": [],
    }


def test_a_find_rings_reactions_genes_and_metabolites_by_name():
    plot, reactions, metabolites = _map_plot()
    find = add_map_find(plot)

    assert find.name == MAP_FIND_SOURCE and find in plot.tags
    (callback,) = find.js_property_callbacks["change:data"]
    assert reactions in callback.args["sources"] and metabolites in callback.args["sources"]
    assert callback.args["full_width"] == 800.0


def test_every_human1_map_carries_metabolic_atlas_paths():
    """Each packaged Human1 layout has its Atlas paths beside it, in its own frame."""
    graphs = MAPS_DIRECTORY.parent.parent
    layouts = [
        path
        for kind in ("compartment", "subsystem")
        for path in (graphs / kind / "human1").glob("*.json")
        if not path.name.endswith(ATLAS_SUFFIX)
    ]
    assert len(layouts) == 115
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
    # Outlined in the gradient over a white core, both drawn from one source,
    # under everything else the map draws.
    outline, core = plot.renderers[:2]
    assert outline.data_source is core.data_source
    # On the lowest level, with the links, which come after it, so its white core
    # never covers them.
    assert outline.level == core.level == "image"
    assert core.glyph.line_color == "#ffffff" and outline.glyph.line_color == "color"
    band = outline.data_source.data
    assert band["x0"] and len(set(band["color"])) > 3
    assert all(color.startswith("#") for color in band["color"])
    # Only the page changes the band: nothing redraws it under the pointer, which
    # would repaint the whole map at every reaction the pointer crossed.
    (show,) = chosen.js_property_callbacks["change:data"]
    assert show.args["chosen"] is chosen
    assert list(plot.select(type=HoverTool)) == hovers
    assert add_map_bands(_map_plot()[0], "nowhere") is None

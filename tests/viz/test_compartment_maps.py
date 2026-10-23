"""The compartment maps: their names, their reactions, and the links into and out of them."""

import pytest
from bokeh.models import ColumnDataSource, TapTool
from bokeh.plotting import figure

from metabis import MetabolicModel
from metabis.viz.atlas_paths import ATLAS_SUFFIX, atlas_paths
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
    # And back: each reaction a map draws names its subsystem.
    named = map_subsystems(model, "mitochondria")
    assert {named[identifier][0] for identifier in drawn} == {TCA}


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


def test_a_subsystems_reactions_stay_in_view_and_the_rest_fades():
    plot, reactions, metabolites = _map_plot()

    assert highlight_reactions(plot, ["MAR2", "MAR3"], padding=100) == 2
    assert reactions.data["alpha"][0] < 0.1 and reactions.data["alpha"][1:] == [1.0, 1.0]
    assert metabolites.data["alpha"] == [0.25]
    # Framed at the map's proportions, the y-range still running downwards.
    assert (plot.x_range.start, plot.x_range.end) == (400.0, 1000.0)
    assert plot.y_range.start > plot.y_range.end
    assert highlight_reactions(_map_plot()[0], ["elsewhere"]) == 0


def test_a_tapped_reaction_offers_its_subsystem():
    plot, reactions, _ = _map_plot()
    add_map_links(plot, {"MAR2": ("glycolysis", "Glycolysis")})

    assert reactions.data["subsystem_id"] == ["", "glycolysis", ""]
    assert reactions.data["subsystem"] == ["", "Glycolysis", ""]
    assert plot.select({"type": TapTool})
    (callback,) = reactions.selected.js_property_callbacks["change:indices"]
    assert callback.args["event"] == MAP_REACTION_EVENT


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


def test_routed_links_fade_with_their_reactions():
    plot, reactions, _ = _map_plot()
    links = ColumnDataSource(
        {
            "xs": [[100.0, 200.0], [900.0, 800.0]],
            "ys": [[100.0, 200.0], [900.0, 800.0]],
            "line_alpha": [1.0, 1.0],
            "glow_alpha": [0.7, 0.7],
            "reaction_id": ["MAR1", "MAR3"],
        }
    )
    plot.multi_line(xs="xs", ys="ys", source=links)

    highlight_reactions(plot, ["MAR3"], padding=100)

    assert links.data["line_alpha"][0] < 0.1 and links.data["line_alpha"][1] == 1.0
    assert links.data["glow_alpha"][0] < 0.1 and links.data["glow_alpha"][1] == 0.7


def test_one_subsystems_band_at_a_time():
    bands = map_bands("mitochondria")
    shown = band_name(bands, "tricarboxylic acid cycle and glyoxylate/dicarboxylate metabolism")
    assert shown == "Tricarboxylic acid cycle and glyoxylate/dicarboxylate metabolism"
    plot, reactions, _ = _map_plot()
    add_map_links(plot, {"MAR2": ("tca", shown)})

    chosen = add_map_bands(plot, "mitochondria", shown)

    assert chosen.name == MAP_BAND_SOURCE and chosen.data["name"] == [shown]
    assert chosen in plot.tags
    drawn = [r.data_source for r in plot.renderers if set(r.data_source.data) == {"xs", "ys"}]
    (band,) = drawn
    assert len(band.data["xs"]) == len(bands[shown])
    # Pointing at a reaction names its own subsystem's band.
    from bokeh.models import HoverTool

    pointing = [
        tool
        for tool in plot.select(type=HoverTool)
        if reactions in [renderer.data_source for renderer in tool.renderers]
    ]
    assert pointing and pointing[0].callback.args["chosen"] is chosen
    assert add_map_bands(_map_plot()[0], "nowhere") is None

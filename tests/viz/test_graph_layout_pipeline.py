"""What the automatic layout hands to Graphviz, and what it gets back.

`MetabisReview/graph-layout-analysis.md` verified four defects in this pipeline.
None of them raised, none of them was visible in a unit test, and three of them
were invisible in the output *scale* while distorting everything inside it. Each
is pinned here by the property its fix establishes, so a later change to the
Graphviz hand-off has to restate the same claim:

* **D1** the layout is computed over exactly the edges that get drawn;
* **D2** the argument string reaches Graphviz and changes the drawing;
* **D3** Graphviz is given geometry, in the unit it actually reads;
* **D4** a compartment boundary contains its contents.

The second half of the module pins the topology/packing redesign (§4 of the same
document): shared metabolites of high degree are split into per-reaction glyphs,
enzyme groups are demoted to satellites, and connected components are laid out
independently and shelf-packed to a target aspect ratio.
"""

import networkx as nx

from metabis.viz.metabolic_visualization.generation.data_structures.constants import (
    DefaultLayout,
    DefaultOptimization,
)
from metabis.viz.metabolic_visualization.generation.graph_building.compartment_graph import (  # noqa: E501
    create_compartment_boundary,
)
from metabis.viz.metabolic_visualization.generation.graph_building.graph_conversion_utils import (  # noqa: E501
    _calculate_standard_layout,
    _layout_graph_for_graphviz,
    _pack_component_shelves,
    calculate_network_layout,
    convert_compartment_raw_data_to_domain_objects,
    create_networkx_graph_for_layout,
    layout_compartment_components,
    split_hub_metabolites,
    update_raw_glyphs_with_positions,
)

# A hub metabolite shared by several reactions, each with its own intermediate.
# This is the shape `optimize_layout` acted on: it duplicated the hub, moved the
# hub's edges onto duplicates that never became glyphs, and re-ran Graphviz, so
# the hub was placed from a subset of its edges — or, when all of them moved, as
# a disconnected node packed off to the side of the drawing — while the renderer
# still drew every original edge from that displaced position.
HUB_REACTIONS = 6


def _hub_graph_data():
    raw_glyphs = [
        {
            "id": "m_hub",
            "glyph_type": "metabolite",
            "original_id": "MAM00001c",
            "label": "MAM00001c",
            "subtype": "main",
            "width": 60,
            "height": 40,
            "x": 0.0,
            "y": 0.0,
        }
    ]
    raw_connections = []
    for index in range(HUB_REACTIONS):
        reaction = f"r_R{index}"
        raw_glyphs.append(
            {
                "id": reaction,
                "glyph_type": "reaction",
                "original_id": f"R{index}",
                "label": f"R{index}",
                "width": 50,
                "height": 50,
                "x": 0.0,
                "y": 0.0,
            }
        )
        raw_glyphs.append(
            {
                "id": f"m_R{index}_side",
                "glyph_type": "metabolite",
                "original_id": "MAM00002c",
                "label": "MAM00002c",
                "subtype": "intermediate",
                "width": 40,
                "height": 30,
                "x": 0.0,
                "y": 0.0,
            }
        )
        raw_connections.append(
            {
                "id": f"conn_{reaction}_hub",
                "connection_type": "metabolic",
                "source": reaction,
                "target": "m_hub",
            }
        )
        raw_connections.append(
            {
                "id": f"conn_{reaction}_side",
                "connection_type": "metabolic",
                "source": reaction,
                "target": f"m_R{index}_side",
            }
        )
    return raw_glyphs, raw_connections


def test_the_layout_positions_exactly_the_nodes_it_was_given():
    """D1 — nothing may be positioned from edges the renderer does not draw.

    `optimize_layout` rewired the layout graph in place and returned positions
    for duplicate nodes that never existed as glyphs, so the positions came back
    describing a different graph from the one being drawn. Equality both ways is
    the claim: no node lost its edges, and no invented node got a position.
    """
    raw_glyphs, raw_connections = _hub_graph_data()
    layout_graph = create_networkx_graph_for_layout(raw_glyphs, raw_connections)
    drawn_edges = {(c["source"], c["target"]) for c in raw_connections}

    positions = calculate_network_layout(layout_graph, "dot")

    assert set(positions) == set(layout_graph.nodes())
    assert set(layout_graph.edges()) == drawn_edges
    assert layout_graph.degree("m_hub") == HUB_REACTIONS


def test_the_graphviz_argument_string_reaches_graphviz():
    """D2 — the arguments must change the drawing, not be silently rejected.

    `-Grankdir="LR", -Gsplines=ortho` set rankdir to `LR,`, which Graphviz
    rejects, falling back to top-to-bottom: positions were byte-identical with
    and without the string. A chain laid out left-to-right advances in x and
    holds y; the malformed value produced exactly the opposite.
    """
    chain = nx.DiGraph()
    chain.add_edge("a", "b")
    chain.add_edge("b", "c")

    positions = _calculate_standard_layout(chain, "dot")
    xs = [positions[node][0] for node in ("a", "b", "c")]
    ys = [positions[node][1] for node in ("a", "b", "c")]

    assert xs == sorted(xs) and xs[0] < xs[-1]
    assert len(set(ys)) == 1
    assert DefaultOptimization.GRAPHVIZ_RANKDIR == "LR"


def test_graphviz_is_given_geometry_in_inches_and_nothing_else():
    """D3 — node dimensions are pixels here and inches to Graphviz.

    `to_agraph` copies every NetworkX attribute into the DOT source, so a
    50-pixel reaction glyph was declared as a 50-*inch* box, 3,600 points wide,
    while `nodesep`/`ranksep` stayed at their fractional-inch defaults — rank
    separation of about 1% of node size. The gene lists and metabolite details
    were being serialised into the DOT file too, where Graphviz ignores them.
    """
    graph = nx.DiGraph()
    graph.add_node(
        "r_R1",
        type="reaction",
        label="R1",
        original_id="R1",
        width=50,
        height=50,
    )
    graph.add_node(
        "eg_r_R1",
        type="enzyme_group",
        label="Enzymes for R1",
        genes=["ALDH1A1", "ADH1B"],
        associated_reaction="r_R1",
        num_cols=1,
        width=70,
        height=65,
    )
    graph.add_edge("eg_r_R1", "r_R1")

    geometry = _layout_graph_for_graphviz(graph)

    points_per_inch = DefaultOptimization.POINTS_PER_INCH
    assert geometry.nodes["r_R1"] == {
        "label": "",
        "shape": "box",
        "fixedsize": "true",
        "width": 50 / points_per_inch,
        "height": 50 / points_per_inch,
    }
    assert geometry.nodes["eg_r_R1"]["width"] == 70 / points_per_inch
    assert set(geometry.edges()) == set(graph.edges())

    dot_source = nx.nx_agraph.to_agraph(geometry).to_string()
    for stripped in ("genes", "ALDH1A1", "original_id", "associated_reaction"):
        assert stripped not in dot_source


def test_a_compartment_boundary_contains_its_contents():
    """D4 — the box is taken from drawn extents, not from node centres.

    Measuring centres omitted half of every glyph on the edge of the
    compartment, so the boundary clipped its own contents by as much as the
    widest glyph in it. The widest glyph here is deliberately the one furthest
    right, which is the case a centre-based box gets wrong.
    """
    glyphs = [
        {"id": "a", "x": 0.0, "y": 0.0, "width": 50.0, "height": 50.0},
        {"id": "b", "x": 400.0, "y": 200.0, "width": 300.0, "height": 100.0},
    ]

    boundary = create_compartment_boundary(glyphs, boundary_id="compartment_cytosol")

    left = boundary["x"] - boundary["width"] / 2
    right = boundary["x"] + boundary["width"] / 2
    bottom = boundary["y"] - boundary["height"] / 2
    top = boundary["y"] + boundary["height"] / 2

    for glyph in glyphs:
        assert left <= glyph["x"] - glyph["width"] / 2
        assert right >= glyph["x"] + glyph["width"] / 2
        assert bottom <= glyph["y"] - glyph["height"] / 2
        assert top >= glyph["y"] + glyph["height"] / 2


def test_positions_come_back_in_the_unit_glyph_dimensions_use():
    """Layout output and glyph `width`/`height` must share one unit.

    Graphviz is told inches and answers in points, so one point out is one pixel
    of glyph space. The previous 0.03 scale existed only to shrink the 72x
    drawing the inch/pixel confusion produced; applying it to a corrected layout
    would leave 50-pixel nodes 1.5 pixels apart.
    """
    engine_positions = {"a": (0.0, 0.0), "b": (1000.0, 500.0)}
    raw_glyphs = [
        {"id": "a", "glyph_type": "reaction", "x": 0.0, "y": 0.0},
        {"id": "b", "glyph_type": "reaction", "x": 0.0, "y": 0.0},
    ]

    update_raw_glyphs_with_positions(raw_glyphs, engine_positions)

    assert DefaultLayout.X_SCALE == 1.0
    assert DefaultLayout.Y_SCALE == 1.0
    assert (raw_glyphs[1]["x"], raw_glyphs[1]["y"]) == engine_positions["b"]


# --------------------------------------------------------------------------- #
# The topology/packing redesign (graph-layout-analysis.md §4).
# --------------------------------------------------------------------------- #
def _enzyme_group_data():
    """One reaction, one shared metabolite, and the reaction's enzyme group.

    Shaped exactly like ``_create_raw_reactions_data`` emits: the enzyme group
    carries ``associated_reaction`` and a regulatory edge to its reaction.
    """
    raw_glyphs = [
        {
            "id": "r_R1",
            "glyph_type": "reaction",
            "original_id": "R1",
            "label": "R1",
            "width": 50,
            "height": 50,
            "x": 0.0,
            "y": 0.0,
        },
        {
            "id": "m_cytosol_MAM00001c",
            "glyph_type": "metabolite",
            "original_id": "MAM00001c",
            "label": "MAM00001c",
            "subtype": "main",
            "width": 60,
            "height": 40,
            "x": 0.0,
            "y": 0.0,
        },
        {
            "id": "eg_r_R1",
            "glyph_type": "enzyme_group",
            "label": "Enzymes for R1",
            "genes": ["ADH1B"],
            "width": 70,
            "height": 40,
            "num_cols": 1,
            "associated_reaction": "r_R1",
            "x": 0.0,
            "y": 0.0,
        },
    ]
    raw_connections = [
        {
            "id": "conn_r_R1_m",
            "connection_type": "metabolic",
            "source": "r_R1",
            "target": "m_cytosol_MAM00001c",
        },
        {
            "id": "regulatory_eg_r_R1",
            "connection_type": "regulatory",
            "source": "eg_r_R1",
            "target": "r_R1",
        },
    ]
    return raw_glyphs, raw_connections


def _redundant_hub_graph_data():
    """The hub fixture, plus a chain that already joins its reactions.

    ``_hub_graph_data`` is a pure star: every arm of ``m_hub`` is the only thing
    holding its reaction to the drawing. Here the reactions are additionally
    strung together by shared metabolites, so all but one of the hub's arms are
    shortcuts across a connection that exists anyway — which is the shape a
    curator duplicates.
    """
    raw_glyphs, raw_connections = _hub_graph_data()
    for index in range(HUB_REACTIONS - 1):
        link = f"m_link{index}"
        raw_glyphs.append(
            {
                "id": link,
                "glyph_type": "metabolite",
                "original_id": f"MAM0001{index}c",
                "label": f"MAM0001{index}c",
                "subtype": "main",
                "width": 60,
                "height": 40,
                "x": 0.0,
                "y": 0.0,
            }
        )
        raw_connections.append(
            {
                "id": f"conn_link{index}_in",
                "connection_type": "metabolic",
                "source": f"r_R{index}",
                "target": link,
            }
        )
        raw_connections.append(
            {
                "id": f"conn_link{index}_out",
                "connection_type": "metabolic",
                "source": link,
                "target": f"r_R{index + 1}",
            }
        )
    return raw_glyphs, raw_connections


def _core_components(glyphs, connections):
    """Connected components of the drawn metabolite↔reaction graph."""
    kinds = {
        glyph["id"]: glyph.get("glyph_type", glyph.get("glyph_class"))
        for glyph in glyphs
    }
    graph = nx.Graph()
    graph.add_nodes_from(
        node for node, kind in kinds.items() if kind in ("reaction", "metabolite")
    )
    for connection in connections:
        kind = connection.get("connection_type", connection.get("connection_class"))
        if kind != "metabolic":
            continue
        source, target = connection["source"], connection["target"]
        if source in graph and target in graph:
            graph.add_edge(source, target)
    return list(nx.connected_components(graph))


def test_a_redundant_hub_arm_is_duplicated_and_a_load_bearing_one_is_not():
    """Stage 1 — the star loses the arms it does not need, and keeps the rest.

    Duplicating *every* arm above the degree threshold is what detached the
    pathway: a metabolite stopped joining any of its reactions and the drawing
    became a field of motifs. An arm may be duplicated exactly when its endpoints
    stay connected without it, so here — where a chain already strings the
    reactions together — one arm holds ``m_hub`` to the drawing and the rest are
    shortcuts that become copies.
    """
    raw_glyphs, raw_connections = _redundant_hub_graph_data()

    glyphs, connections = split_hub_metabolites(
        raw_glyphs, raw_connections, DefaultOptimization.HUB_METABOLITE_DEGREE
    )

    ids = [glyph["id"] for glyph in glyphs]
    assert "m_hub" in ids  # the parent is never left without its own arm
    assert len(ids) == len(set(ids))

    copies = [glyph for glyph in glyphs if glyph["id"].startswith("m_hub__")]
    assert len(copies) == HUB_REACTIONS - 1
    assert all(copy["original_id"] == "MAM00001c" for copy in copies)
    assert all(copy["glyph_type"] == "metabolite" for copy in copies)

    # Each copy carries exactly its own reaction, and the parent keeps one arm.
    reactions_of = {}
    for connection in connections:
        for endpoint in ("source", "target"):
            node = connection[endpoint]
            if node == "m_hub" or node.startswith("m_hub__"):
                other = (
                    connection["target"]
                    if endpoint == "source"
                    else connection["source"]
                )
                reactions_of.setdefault(node, set()).add(other)
    assert len(reactions_of["m_hub"]) == 1
    assert all(len(seen) == 1 for node, seen in reactions_of.items() if node != "m_hub")


def test_splitting_never_detaches_anything():
    """Stage 1 leaves the drawn connectivity exactly as the model states it.

    This is the invariant the whole rule exists to hold, and the one whose absence
    was visible in the product: over a graph with redundant arms, splitting must
    change how the hub is drawn without changing what is reachable from what.
    """
    raw_glyphs, raw_connections = _redundant_hub_graph_data()
    before = _core_components(raw_glyphs, raw_connections)

    glyphs, connections = split_hub_metabolites(
        raw_glyphs, raw_connections, DefaultOptimization.HUB_METABOLITE_DEGREE
    )
    after = _core_components(glyphs, connections)

    assert len(after) == len(before) == 1
    # And the split did happen — this is not a no-op passing by accident.
    assert len(glyphs) > len(raw_glyphs)


def test_a_pool_metabolite_sheds_the_arms_above_the_cap():
    """A metabolite that is the sole path to dozens of reactions is not a junction.

    Preserving connectivity alone keeps every arm of such a pool — a shared CoA or
    carnitine — and hands the layout engine a 66-armed star, which it draws as an
    unreadable strip. Above the cap the pool keeps the arms holding the largest
    structures and duplicates the rest, which is the one place this deliberately
    detaches something. The bound comes from the curated maps: their highest
    metabolite degree is 14.
    """
    cap = DefaultOptimization.HUB_METABOLITE_DEGREE_CAP
    arms = cap + 8
    raw_glyphs = [
        {
            "id": "m_pool",
            "glyph_type": "metabolite",
            "original_id": "MAM00001c",
            "label": "MAM00001c",
            "subtype": "main",
            "width": 60,
            "height": 40,
            "x": 0.0,
            "y": 0.0,
        }
    ]
    raw_connections = []
    for index in range(arms):
        reaction = f"r_R{index}"
        raw_glyphs.append(
            {
                "id": reaction,
                "glyph_type": "reaction",
                "original_id": f"R{index}",
                "label": f"R{index}",
                "width": 50,
                "height": 50,
                "x": 0.0,
                "y": 0.0,
            }
        )
        raw_connections.append(
            {
                "id": f"conn_{reaction}_pool",
                "connection_type": "metabolic",
                "source": reaction,
                "target": "m_pool",
            }
        )
        # The first few reactions carry a partner metabolite, so their arms hold
        # more of the drawing than the bare ones and must be the arms kept.
        if index < 3:
            raw_glyphs.append(
                {
                    "id": f"m_R{index}_partner",
                    "glyph_type": "metabolite",
                    "original_id": f"MAM0002{index}c",
                    "label": f"MAM0002{index}c",
                    "subtype": "intermediate",
                    "width": 40,
                    "height": 30,
                    "x": 0.0,
                    "y": 0.0,
                }
            )
            raw_connections.append(
                {
                    "id": f"conn_{reaction}_partner",
                    "connection_type": "metabolic",
                    "source": reaction,
                    "target": f"m_R{index}_partner",
                }
            )

    glyphs, connections = split_hub_metabolites(
        raw_glyphs, raw_connections, DefaultOptimization.HUB_METABOLITE_DEGREE
    )

    kept = [
        connection
        for connection in connections
        if "m_pool" in (connection["source"], connection["target"])
    ]
    assert len(kept) == cap

    kept_reactions = {
        connection["source"]
        if connection["target"] == "m_pool"
        else connection["target"]
        for connection in kept
    }
    # The arms that hold a partner metabolite are the largest sides, so they stay.
    assert {"r_R0", "r_R1", "r_R2"} <= kept_reactions
    assert len(glyphs) == len(raw_glyphs) + (arms - cap)


def test_a_pure_star_keeps_every_arm():
    """No arm of a star is redundant, so a star is drawn whole.

    Every arm of ``m_hub`` in the plain hub fixture is the only path to its
    reaction. Duplicating them would turn one component into six, which is the
    regression this rule replaced.
    """
    raw_glyphs, raw_connections = _hub_graph_data()

    glyphs, connections = split_hub_metabolites(
        raw_glyphs, raw_connections, DefaultOptimization.HUB_METABOLITE_DEGREE
    )

    assert glyphs is raw_glyphs
    assert connections is raw_connections
    assert len(_core_components(glyphs, connections)) == 1


def test_a_low_degree_metabolite_is_left_shared():
    """Stage 1 splits only hubs — a metabolite at or below K stays one node.

    The per-reaction side metabolites in the hub fixture have degree 1, so the
    duplication budget is spent on the hub alone. Splitting everything would cost
    far more glyphs than curators use for no crossing benefit.
    """
    raw_glyphs, raw_connections = _hub_graph_data()

    glyphs, _ = split_hub_metabolites(
        raw_glyphs, raw_connections, DefaultOptimization.HUB_METABOLITE_DEGREE
    )

    ids = [glyph["id"] for glyph in glyphs]
    for index in range(HUB_REACTIONS):
        assert ids.count(f"m_R{index}_side") == 1


def test_nothing_splits_when_no_metabolite_is_a_hub():
    """A graph with no hub is returned unchanged, object-for-object."""
    raw_glyphs, raw_connections = _enzyme_group_data()

    glyphs, connections = split_hub_metabolites(
        raw_glyphs, raw_connections, DefaultOptimization.HUB_METABOLITE_DEGREE
    )

    assert glyphs is raw_glyphs
    assert connections is raw_connections


def test_enzyme_groups_are_reattached_beside_their_reaction():
    """Stage 4 — enzyme groups are satellites, placed from the reaction position.

    A large enzyme box that takes part in rank assignment distorts the metabolic
    drawing, so it is not ranked; only the space it will occupy is reserved on its
    reaction. Its x follows the reaction and its y sits above it.
    """
    raw_glyphs, raw_connections = _enzyme_group_data()

    positions = layout_compartment_components(raw_glyphs, raw_connections, "dot")

    reaction_x, reaction_y = positions["r_R1"]
    enzyme_x, enzyme_y = positions["eg_r_R1"]
    assert enzyme_x == reaction_x
    assert enzyme_y > reaction_y


def test_an_enzyme_group_lands_inside_the_space_reserved_for_it():
    """Stage 4 — the group occupies room the layout engine was told about.

    The reaction is handed to the engine inside a box tall enough for both, and
    the group is drawn at the top of that box. Reserving nothing and reattaching
    afterwards put the group outside the component extent that had been packed
    against its neighbours, which is how enzyme groups came to sit on top of other
    components' glyphs. The claim here is geometric: the pair fits within one
    reserved height, and nothing else in the drawing overlaps the group.
    """
    raw_glyphs, raw_connections = _hub_graph_data()
    # Give two of the reactions an enzyme group large enough to collide.
    for index in (0, 1):
        raw_glyphs.append(
            {
                "id": f"eg_r_R{index}",
                "glyph_type": "enzyme_group",
                "label": f"Enzymes for R{index}",
                "genes": ["G1", "G2", "G3", "G4"],
                "width": 200,
                "height": 160,
                "num_cols": 2,
                "associated_reaction": f"r_R{index}",
                "x": 0.0,
                "y": 0.0,
            }
        )
        raw_connections.append(
            {
                "id": f"regulatory_eg_r_R{index}",
                "connection_type": "regulatory",
                "source": f"eg_r_R{index}",
                "target": f"r_R{index}",
            }
        )

    glyphs, _ = convert_compartment_raw_data_to_domain_objects(
        raw_glyphs, raw_connections, "dot"
    )

    boxes = {
        glyph["id"]: (
            glyph["x"] - glyph.get("width", 0) / 2,
            glyph["y"] - glyph.get("height", 0) / 2,
            glyph["x"] + glyph.get("width", 0) / 2,
            glyph["y"] + glyph.get("height", 0) / 2,
        )
        for glyph in glyphs
        if glyph["glyph_class"] in ("reaction", "metabolite", "enzyme_group")
    }
    for group in ("eg_r_R0", "eg_r_R1"):
        left, bottom, right, top = boxes[group]
        for other, (o_left, o_bottom, o_right, o_top) in boxes.items():
            if other == group:
                continue
            overlaps = (
                left < o_right and o_left < right and bottom < o_top and o_bottom < top
            )
            assert not overlaps, f"{group} overlaps {other}"


def test_the_compartment_layout_keeps_the_original_id_of_every_glyph():
    """The whole compartment pipeline preserves identity through splitting.

    Colouring, hovers and selection resolve a metabolite by the entity behind the
    glyph, never the glyph ID, so a hub copy must still answer as its metabolite.
    """
    raw_glyphs, raw_connections = _redundant_hub_graph_data()

    glyphs, connections = convert_compartment_raw_data_to_domain_objects(
        raw_glyphs, raw_connections, "dot"
    )

    node_ids = {glyph["id"] for glyph in glyphs}
    assert len(node_ids) == len(glyphs)  # unique IDs
    for connection in connections:
        assert connection["source"] in node_ids
        assert connection["target"] in node_ids

    hub_glyphs = [
        glyph
        for glyph in glyphs
        if glyph.get("original_id") == "MAM00001c"
        and glyph.get("glyph_class") == "metabolite"
    ]
    # One parent plus a copy for each duplicated arm, all answering as MAM00001c.
    assert len(hub_glyphs) == HUB_REACTIONS


def test_components_pack_onto_more_than_one_shelf():
    """Stage 4 — many components are packed in two dimensions, not one axis.

    Graphviz spreads disconnected components along a single axis, which is what
    made the drawings letterbox bands. The shelf packer sizes a shelf for a target
    aspect ratio and wraps, so a run of identical boxes lands on several rows.
    """
    boxes = [(300.0, 300.0, index) for index in range(16)]

    offsets = _pack_component_shelves(
        boxes,
        DefaultOptimization.COMPONENT_TARGET_ASPECT,
        DefaultOptimization.COMPONENT_PACKING_GAP,
    )

    distinct_rows = {round(offset_y, 3) for _, offset_y in offsets.values()}
    assert len(distinct_rows) >= 2
    # And the packing spans a plausible aspect, not a single row or column.
    width = max(x for x, _ in offsets.values()) + 300.0
    height = max(y for _, y in offsets.values()) + 300.0
    assert 0.5 <= width / height <= 4.0

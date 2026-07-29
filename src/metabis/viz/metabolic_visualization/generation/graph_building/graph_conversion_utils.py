"""Utilities for converting raw graph data to domain objects.

This module provides generalized functions for the raw data → layout → domain objects
pipeline that can be used across different graph types.
"""

import math
from collections import Counter
from typing import Any, Dict, List, Tuple

import networkx as nx

from ..data_structures.basic_factories import (
    create_enzyme_group_glyph,
    create_gene_glyph,
    create_metabolic_connection,
    create_metabolite_glyph,
    create_reaction_glyph,
    create_subsystem_category_glyph,
    create_subsystem_connection,
    create_subsystem_glyph,
)
from ..data_structures.constants import (
    DefaultDimensions,
    DefaultEnzymeGroupLayout,
    DefaultLayout,
    DefaultOptimization,
    MetabolicSerpentineLayoutDefaults,
)
from ..data_structures.core_types import (
    Connection,
    ConnectionClass,
    Glyph,
    MetaboliteSubtype,
)
from .metabolic_serpentine_layout import (
    METABOLIC_SERPENTINE_ENGINE,
    MetabolicSerpentineLayout,
)


def convert_raw_data_to_domain_objects(
    raw_glyphs: List[Dict[str, Any]],
    raw_connections: List[Dict[str, Any]],
    layout_engine: str = "dot",
) -> Tuple[List[Glyph], List[Connection]]:
    """Convert raw graph data to domain objects through layout calculation.

    This is a generalized function that follows the pipeline:
    raw data → layout → domain objects

    Args:
        raw_glyphs: List of raw glyph data dictionaries without positions
        raw_connections: List of raw connection data dictionaries
        layout_engine: GraphViz layout algorithm to use

    Returns:
        Tuple containing:
            - glyphs: List of domain glyph objects with calculated positions
            - connections: List of domain connection objects
    """
    if not raw_glyphs:
        return [], []

    # Step 1: Create temporary NetworkX graph for layout calculation
    layout_graph = create_networkx_graph_for_layout(raw_glyphs, raw_connections)

    # Step 2: Calculate layout positions
    pos = calculate_network_layout(layout_graph, layout_engine)

    # Step 3: Update raw glyphs with calculated positions
    update_raw_glyphs_with_positions(raw_glyphs, pos)

    # Step 4: Convert raw data to domain objects
    glyphs = convert_raw_glyphs_to_domain_objects(raw_glyphs)
    connections = convert_raw_connections_to_domain_objects(raw_connections)

    return glyphs, connections


def convert_compartment_raw_data_to_domain_objects(
    raw_glyphs: List[Dict[str, Any]],
    raw_connections: List[Dict[str, Any]],
    layout_engine: str = "dot",
) -> Tuple[List[Glyph], List[Connection]]:
    """Convert one compartment's raw data to domain objects, tidily.

    This is the automatic-metabolic-map redesign (graph-layout-analysis.md §4),
    the path taken when no curated Human1 map exists. It replaces "throw the whole
    bipartite compartment at ``dot`` and hope" — which drew the shared metabolites
    as multi-armed stars packed into a single-axis letterbox — with an explicit
    pipeline:

    1. **Topology preparation.** Duplicate the *redundant* arms of a shared
       metabolite of degree > ``HUB_METABOLITE_DEGREE``, so a hub is drawn as a
       few short local arms instead of one star, without ever detaching a
       pathway. Copies keep the parent's ``original_id``, so nothing downstream
       can tell them apart from the entity they came from.
    2. **Component decomposition + per-component layered layout.** Each connected
       component is laid out on its own with ``dot`` correctly configured.
    3. **2D packing.** The component boxes are shelf-packed to a target aspect
       ratio instead of being spread along one axis.
    4. **Satellite reservation.** An enzyme group never takes part in rank
       assignment, but the space it will occupy is reserved on its reaction
       before layout, so it cannot land on a neighbouring glyph afterwards.

    The generic :func:`convert_raw_data_to_domain_objects` still serves the
    subsystem-overview network, whose nodes are subsystems, not metabolites. The
    serpentine engine carries its own topology and is delegated there unchanged.
    """
    if not raw_glyphs:
        return [], []

    if layout_engine == METABOLIC_SERPENTINE_ENGINE:
        return convert_raw_data_to_domain_objects(
            raw_glyphs, raw_connections, layout_engine
        )

    prepared_glyphs, prepared_connections = split_hub_metabolites(
        raw_glyphs, raw_connections, DefaultOptimization.HUB_METABOLITE_DEGREE
    )

    pos = layout_compartment_components(
        prepared_glyphs, prepared_connections, layout_engine
    )

    update_raw_glyphs_with_positions(prepared_glyphs, pos)

    glyphs = convert_raw_glyphs_to_domain_objects(prepared_glyphs)
    connections = convert_raw_connections_to_domain_objects(prepared_connections)

    return glyphs, connections


def split_hub_metabolites(
    raw_glyphs: List[Dict[str, Any]],
    raw_connections: List[Dict[str, Any]],
    hub_degree: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Duplicate the redundant arms of a shared metabolite, and only those.

    A "main" metabolite is drawn as a single node shared by every reaction in its
    compartment. When many reactions touch it, that node becomes a star whose arms
    cross the entire drawing, so curators duplicate it beside each reaction —
    but they do not duplicate *every* arm. They duplicate the arms that are
    shortcuts and keep the arms that are the pathway, which is why a curated map
    still reads as one connected network.

    That distinction is a graph property rather than a judgement: an incident edge
    may be duplicated exactly when it is **redundant** — when its two endpoints
    stay connected without it. Splitting only redundant arms leaves the number of
    drawn components identical to the model's own connectivity, by construction.

    Splitting *every* arm of a hub, which this function used to do, is what turned
    the automatic maps into a field of disconnected motifs: a metabolite touched
    by four reactions stopped joining any of them, and the largest connected
    component of a compartment fell to a third of the curated map's.

    The redundant set is computed as "every hub arm outside a spanning forest",
    with non-hub edges offered to the forest first. On the graphic matroid that
    greedy order is optimal: no spanning forest of the same graph keeps fewer hub
    arms, so this duplicates as much of the star as connectivity allows and no
    more.

    Connectivity is not the only thing worth keeping, though. A metabolite that is
    the sole path to dozens of reactions is a shared pool rather than a pathway
    junction: it has no redundant arm at all, so preserving connectivity alone
    hands the layout engine a 66-armed star and gets back an unreadable strip.
    Above ``HUB_METABOLITE_DEGREE_CAP`` such a metabolite therefore keeps only the
    arms holding the largest structures together and the remainder are duplicated
    — the one place this deliberately detaches something, bounded by what the
    curated maps themselves do (their highest metabolite degree is 14).

    Intermediate metabolites are already one glyph per reaction, so their degree is
    small and they are never split; the currency metabolites (H2O, ATP, …) are
    classified intermediate upstream, so the currency "floor" is already in place
    and this splits by degree on top of it.

    Each copy is a full metabolite glyph carrying the parent's ``original_id``,
    ``label`` and ``subtype``, so colouring, hovers and selection resolve it to the
    same entity. The copy IDs are derived from the ``(metabolite, reaction)`` pair,
    which is unique and stable, so a regenerated graph is byte-identical.

    Returns new glyph and connection lists; the inputs are not mutated.
    """
    core_degree: Counter = Counter()
    for connection in raw_connections:
        if connection["connection_type"] == "metabolic":
            core_degree[connection["source"]] += 1
            core_degree[connection["target"]] += 1

    hub_ids = {
        glyph["id"]
        for glyph in raw_glyphs
        if glyph["glyph_type"] == "metabolite"
        and glyph.get("subtype", "main") == "main"
        and core_degree[glyph["id"]] > hub_degree
    }

    if not hub_ids:
        return raw_glyphs, raw_connections

    redundant = _redundant_hub_arms(raw_connections, hub_ids)
    redundant |= _hub_arms_over_the_cap(
        raw_connections, hub_ids, redundant, DefaultOptimization.HUB_METABOLITE_DEGREE_CAP
    )
    if not redundant:
        return raw_glyphs, raw_connections

    hub_template = {
        glyph["id"]: glyph for glyph in raw_glyphs if glyph["id"] in hub_ids
    }

    # A hub keeps at least the arms that hold its component together, so the
    # parent glyph always survives and only the duplicated arms are appended —
    # in the deterministic order the metabolic connections introduce them.
    new_glyphs: List[Dict[str, Any]] = list(raw_glyphs)

    copy_ids: Dict[Tuple[str, str], str] = {}
    new_connections: List[Dict[str, Any]] = []
    for index, connection in enumerate(raw_connections):
        if connection["connection_type"] != "metabolic":
            new_connections.append(connection)
            continue

        source, target = connection["source"], connection["target"]
        rewired = dict(connection)
        if index in redundant:
            for endpoint in ("source", "target"):
                node = connection[endpoint]
                if node not in hub_ids:
                    continue
                reaction_node = target if node == source else source
                key = (node, reaction_node)
                if key not in copy_ids:
                    copy_id = f"{node}__{reaction_node}"
                    copy_ids[key] = copy_id
                    copy = dict(hub_template[node])
                    copy["id"] = copy_id
                    new_glyphs.append(copy)
                rewired[endpoint] = copy_ids[key]
        new_connections.append(rewired)

    return new_glyphs, new_connections


def _redundant_hub_arms(
    raw_connections: List[Dict[str, Any]],
    hub_ids: set,
) -> set:
    """Return the indices of the hub arms a spanning forest does not need.

    Union-find over the metabolite↔reaction edges, offering every non-hub edge to
    the forest before any hub arm. An edge whose endpoints are already joined is
    redundant; because the hub arms are offered last, the forest keeps the fewest
    of them any spanning forest could.
    """
    parent: Dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # path compression
            parent[node], node = root, parent[node]
        return root

    def union(left: str, right: str) -> bool:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return False
        parent[right_root] = left_root
        return True

    hub_arms: List[int] = []
    for index, connection in enumerate(raw_connections):
        if connection["connection_type"] != "metabolic":
            continue
        source, target = connection["source"], connection["target"]
        if source in hub_ids or target in hub_ids:
            hub_arms.append(index)
        else:
            union(source, target)

    redundant = set()
    for index in hub_arms:
        connection = raw_connections[index]
        if not union(connection["source"], connection["target"]):
            redundant.add(index)
    return redundant


def _hub_arms_over_the_cap(
    raw_connections: List[Dict[str, Any]],
    hub_ids: set,
    already_split: set,
    cap: int,
) -> set:
    """Return the arms a metabolite above ``cap`` gives up, largest side kept.

    Runs on the graph the redundant-arm pass leaves behind, so every arm here is
    load-bearing and dropping one really does detach what hangs off it. The arms
    are ranked by how much of the drawing they hold — the size of the structure
    reachable through that reaction without the hub — and the ``cap`` biggest are
    kept, so a pool sheds its lone reactions and keeps its pathways.
    """
    adjacency: Dict[str, List[Tuple[str, int]]] = {}
    for index, connection in enumerate(raw_connections):
        if connection["connection_type"] != "metabolic" or index in already_split:
            continue
        source, target = connection["source"], connection["target"]
        adjacency.setdefault(source, []).append((target, index))
        adjacency.setdefault(target, []).append((source, index))

    over_cap = sorted(
        hub for hub in hub_ids if len(adjacency.get(hub, ())) > cap
    )
    if not over_cap:
        return set()

    def side_size(start: str, blocked: str) -> int:
        """Nodes reachable from ``start`` without passing through ``blocked``."""
        seen = {blocked, start}
        stack = [start]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbour, _ in adjacency.get(node, ()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return size

    capped = set()
    for hub in over_cap:
        arms = adjacency.get(hub, [])
        if len(arms) <= cap:  # an earlier hub may already have shortened this one
            continue
        ranked = sorted(
            arms,
            key=lambda arm: (-side_size(arm[0], hub), arm[0]),
        )
        for _, index in ranked[cap:]:
            capped.add(index)
        adjacency[hub] = ranked[:cap]

    return capped


def layout_compartment_components(
    raw_glyphs: List[Dict[str, Any]],
    raw_connections: List[Dict[str, Any]],
    layout_engine: str,
) -> Dict[str, Tuple[float, float]]:
    """Lay out a compartment component by component, then pack the components.

    Only reactions and metabolites take part in the layout; genes and enzyme
    groups are satellites (:func:`convert_compartment_raw_data_to_domain_objects`).
    Each weakly-connected component of the metabolite↔reaction graph is laid out on
    its own with ``dot``, and the component boxes are shelf-packed to a target
    aspect ratio so the drawing is a rectangle rather than a single-axis band.

    An enzyme group still takes no part in rank assignment — it carries one edge
    and is a large box, and letting it be ranked distorts the metabolic drawing —
    but the space it will occupy is **reserved on its reaction** before layout.
    ``dot`` therefore places a reaction inside a box tall enough to hold both, and
    the group is drawn in the reserved part of that box afterwards. Reattaching
    the group blind, as this used to do, put a 100x80 box outside the component
    extent that had been packed against its neighbours, so groups landed on top of
    reactions and metabolites belonging to other components.
    """
    draw_sizes: Dict[str, Tuple[float, float]] = {}
    core_graph = nx.DiGraph()
    for glyph in raw_glyphs:
        if glyph["glyph_type"] not in ("reaction", "metabolite"):
            continue
        width = float(glyph.get("width", DefaultDimensions.DEFAULT_WIDTH))
        height = float(glyph.get("height", DefaultDimensions.DEFAULT_HEIGHT))
        draw_sizes[glyph["id"]] = (width, height)
        core_graph.add_node(glyph["id"])

    satellites, layout_sizes = _reserve_satellite_footprints(raw_glyphs, draw_sizes)
    for node in core_graph:
        width, height = layout_sizes[node]
        core_graph.nodes[node]["width"] = width
        core_graph.nodes[node]["height"] = height

    for connection in raw_connections:
        if connection["connection_type"] != "metabolic":
            continue
        source, target = connection["source"], connection["target"]
        if source in core_graph and target in core_graph:
            core_graph.add_edge(source, target)

    if not core_graph:
        return {}

    # Deterministic component order: largest first, ties broken by lowest node ID.
    components = sorted(
        nx.weakly_connected_components(core_graph),
        key=lambda component: (-len(component), min(component)),
    )

    component_layouts: List[Dict[str, Tuple[float, float]]] = []
    boxes: List[Tuple[float, float, int]] = []
    for index, nodes in enumerate(components):
        local = _layout_small_component(nodes, core_graph, layout_sizes)
        if local is None:
            local = _calculate_standard_layout(
                core_graph.subgraph(nodes), layout_engine
            )
        placed, extents = _place_within_reserved_boxes(
            local, satellites, draw_sizes, layout_sizes
        )
        left = min(placed[node][0] - extents[node][0] / 2 for node in placed)
        bottom = min(placed[node][1] - extents[node][1] / 2 for node in placed)
        right = max(placed[node][0] + extents[node][0] / 2 for node in placed)
        top = max(placed[node][1] + extents[node][1] / 2 for node in placed)
        component_layouts.append(
            {node: (x - left, y - bottom) for node, (x, y) in placed.items()}
        )
        boxes.append((right - left, top - bottom, index))

    offsets = _pack_component_shelves(
        boxes,
        DefaultOptimization.COMPONENT_TARGET_ASPECT,
        DefaultOptimization.COMPONENT_PACKING_GAP,
    )

    pos: Dict[str, Tuple[float, float]] = {}
    for index, layout in enumerate(component_layouts):
        offset_x, offset_y = offsets[index]
        for node, (x, y) in layout.items():
            pos[node] = (x + offset_x, y + offset_y)

    return pos


def _reserve_satellite_footprints(
    raw_glyphs: List[Dict[str, Any]],
    draw_sizes: Dict[str, Tuple[float, float]],
) -> Tuple[Dict[str, Tuple[str, float, float]], Dict[str, Tuple[float, float]]]:
    """Grow each reaction's layout box to hold the enzyme group above it.

    Returns the satellite of each reaction — ``(glyph_id, width, height)`` — and
    the node sizes the layout engine should be given, which are the drawn sizes
    everywhere except on a reaction that has a group.
    """
    layout_sizes = dict(draw_sizes)
    satellites: Dict[str, Tuple[str, float, float]] = {}
    gap = DefaultOptimization.ENZYME_SATELLITE_GAP

    for glyph in raw_glyphs:
        if glyph["glyph_type"] != "enzyme_group":
            continue
        reaction = glyph["associated_reaction"]
        if reaction not in draw_sizes or reaction in satellites:
            continue
        group_width = float(glyph.get("width", DefaultDimensions.DEFAULT_WIDTH))
        group_height = float(glyph.get("height", DefaultDimensions.DEFAULT_HEIGHT))
        satellites[reaction] = (glyph["id"], group_width, group_height)
        reaction_width, reaction_height = draw_sizes[reaction]
        layout_sizes[reaction] = (
            max(reaction_width, group_width),
            reaction_height + gap + group_height,
        )

    return satellites, layout_sizes


def _place_within_reserved_boxes(
    local: Dict[str, Tuple[float, float]],
    satellites: Dict[str, Tuple[str, float, float]],
    draw_sizes: Dict[str, Tuple[float, float]],
    layout_sizes: Dict[str, Tuple[float, float]],
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    """Split each reserved box into the reaction below and its group above.

    The layout engine placed a box that contains both; the reaction is drawn at
    the bottom of it and the enzyme group at the top, so the pair occupies exactly
    the space the engine kept clear and nothing spills over the component extent.
    """
    placed: Dict[str, Tuple[float, float]] = {}
    extents: Dict[str, Tuple[float, float]] = {}

    for node, (x, y) in local.items():
        reaction_height = draw_sizes[node][1]
        if node not in satellites:
            placed[node] = (x, y)
            extents[node] = draw_sizes[node]
            continue
        satellite_id, group_width, group_height = satellites[node]
        reserved_height = layout_sizes[node][1]
        placed[node] = (x, y - reserved_height / 2 + reaction_height / 2)
        extents[node] = draw_sizes[node]
        placed[satellite_id] = (x, y + reserved_height / 2 - group_height / 2)
        extents[satellite_id] = (group_width, group_height)

    return placed, extents


def _layout_small_component(
    nodes: Any,
    core_graph: nx.DiGraph,
    sizes: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[float, float]] | None:
    """Place a trivially small component without invoking Graphviz.

    A weakly-connected component of one or two nodes has only one sensible
    left-to-right drawing, so calling ``dot`` for it is pure subprocess overhead —
    and after hub splitting a subsystem can be thousands of these (every
    exchange/demand reaction is a reaction and its one metabolite). Larger
    components return ``None`` and fall through to the layout engine.
    """
    if len(nodes) == 1:
        return {next(iter(nodes)): (0.0, 0.0)}
    if len(nodes) == 2:
        source, target = next(iter(core_graph.subgraph(nodes).edges()))
        separation = (
            sizes[source][0] / 2
            + DefaultOptimization.GRAPHVIZ_RANKSEP_INCHES
            * DefaultOptimization.POINTS_PER_INCH
            + sizes[target][0] / 2
        )
        return {source: (0.0, 0.0), target: (separation, 0.0)}
    return None


def _pack_component_shelves(
    boxes: List[Tuple[float, float, int]],
    target_aspect: float,
    gap: float,
) -> Dict[int, Tuple[float, float]]:
    """Shelf-pack component bounding boxes into a roughly ``target_aspect`` rectangle.

    Boxes are placed tallest first along horizontal shelves whose width is chosen
    so the packed area comes out near the target width/height ratio; a box that
    would overrun the shelf starts a new one. The result is a dict mapping each
    box's key to the offset of its lower-left corner.

    The shelf width is sized from the space each box *actually* consumes — its
    extent plus one gap on each axis — rather than its bare area, because with
    many small components the gaps are a large share of the drawing, and ignoring
    them makes the shelf too narrow and the result a tall strip.
    """
    occupied_area = sum(
        (width + gap) * (height + gap) for width, height, _ in boxes
    )
    widest = max((width for width, _, _ in boxes), default=1.0)
    shelf_width = max(math.sqrt(occupied_area * target_aspect), widest)

    placements: Dict[int, Tuple[float, float]] = {}
    cursor_x = 0.0
    cursor_y = 0.0
    shelf_height = 0.0
    for width, height, key in sorted(boxes, key=lambda box: -box[1]):
        if cursor_x > 0.0 and cursor_x + width > shelf_width:
            cursor_x = 0.0
            cursor_y += shelf_height + gap
            shelf_height = 0.0
        placements[key] = (cursor_x, cursor_y)
        cursor_x += width + gap
        shelf_height = max(shelf_height, height)
    return placements


def create_networkx_graph_for_layout(
    raw_glyphs: List[Dict[str, Any]],
    raw_connections: List[Dict[str, Any]],
) -> nx.DiGraph:
    """Create temporary NetworkX graph from raw data for layout calculation.

    This generalized function handles both compartment graph types (excludes genes
    from layout since they're positioned within enzyme groups) and network graph
    types (includes all nodes).

    Args:
        raw_glyphs: List of raw glyph data
        raw_connections: List of raw connection data

    Returns:
        NetworkX directed graph for layout calculation
    """
    graph = nx.DiGraph()

    # Add nodes from raw glyphs
    for raw_glyph in raw_glyphs:
        # For compartment graphs, skip genes since they're positioned
        # within enzyme groups
        if raw_glyph["glyph_type"] == "gene":
            continue

        node_attrs = {
            "type": raw_glyph["glyph_type"],
            "label": raw_glyph["label"],
            "width": raw_glyph.get("width", 50),
            "height": raw_glyph.get("height", 50),
        }

        # Add type-specific attributes
        if raw_glyph["glyph_type"] == "reaction":
            node_attrs["original_id"] = raw_glyph["original_id"]
        elif raw_glyph["glyph_type"] == "metabolite":
            node_attrs["original_id"] = raw_glyph["original_id"]
            node_attrs["subtype"] = raw_glyph.get("subtype", "main")
        elif raw_glyph["glyph_type"] == "enzyme_group":
            node_attrs["genes"] = raw_glyph["genes"]
            node_attrs["associated_reaction"] = raw_glyph["associated_reaction"]
            node_attrs["num_cols"] = raw_glyph["num_cols"]
        elif raw_glyph["glyph_type"] == "subsystem":
            node_attrs["subsystem_name"] = raw_glyph["subsystem_name"]
            node_attrs["reaction_count"] = raw_glyph["reaction_count"]
            node_attrs["metabolite_count"] = raw_glyph["metabolite_count"]
        elif raw_glyph["glyph_type"] == "subsystem_category":
            node_attrs["category_name"] = raw_glyph["category_name"]
            node_attrs["subsystem_count"] = raw_glyph["subsystem_count"]
            node_attrs["radius"] = raw_glyph["radius"]

        graph.add_node(raw_glyph["id"], **node_attrs)

    # Add edges from raw connections
    for raw_connection in raw_connections:
        edge_attrs = {
            "connection_type": raw_connection["connection_type"],
        }

        # Add type-specific edge attributes
        if raw_connection["connection_type"] == "subsystem_connection":
            edge_attrs.update(
                {
                    "weight": raw_connection["weight"],
                    "shared_metabolites": raw_connection["shared_metabolites"],
                    "metabolite_details": raw_connection["metabolite_details"],
                    "direction": raw_connection["direction"],
                }
            )
        elif raw_connection["connection_type"] == "category_connection":
            edge_attrs.update(
                {
                    "weight": raw_connection["weight"],
                    "shared_metabolites": raw_connection["shared_metabolites"],
                    "subsystem_relationships": raw_connection[
                        "subsystem_relationships"
                    ],
                    "direction": raw_connection["direction"],
                }
            )

        graph.add_edge(raw_connection["source"], raw_connection["target"], **edge_attrs)

    return graph


def calculate_network_layout(
    graph: nx.DiGraph, layout_engine: str
) -> Dict[str, Tuple[float, float]]:
    """Calculate node positions, in glyph pixels, for one graph.

    Every engine returns coordinates in the same space the glyph ``width`` and
    ``height`` fields are expressed in, so a caller never has to know which one
    ran.

    Args:
        graph: NetworkX graph to calculate layout for
        layout_engine: GraphViz layout algorithm to use

    Returns:
        Dictionary mapping node IDs to (x, y) position tuples
    """
    if layout_engine == METABOLIC_SERPENTINE_ENGINE:
        try:
            positions = MetabolicSerpentineLayout().layout_graph(graph)
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(
                "Serpentine layout failed: {0}. Falling back to standard layout.".format(
                    exc
                )
            )
            layout_engine = "dot"
        else:
            scale = MetabolicSerpentineLayoutDefaults.POSITION_SCALE
            return {node: (x * scale, y * scale) for node, (x, y) in positions.items()}

    print("Using standard layout calculation...")
    return _calculate_standard_layout(graph, layout_engine)


def _layout_graph_for_graphviz(graph: nx.DiGraph) -> nx.DiGraph:
    """Return the graph as Graphviz should see it: geometry only, in inches.

    ``to_agraph`` copies every NetworkX attribute into the DOT source, so the
    attributes this graph carries for its other consumers — gene lists,
    metabolite details, subsystem relationships — were being serialised into the
    DOT file, where Graphviz ignores all of them and *misreads* two. ``width``
    and ``height`` are inches to Graphviz and pixels here, so a 50-pixel reaction
    glyph was declared as a 50-inch (3,600-point) box while ``nodesep`` and
    ``ranksep`` stayed at their fractional-inch defaults. Rank separation was
    then about 1% of node size, which is what produced the tight, extremely
    elongated bands.

    Labels are dropped as well: with ``fixedsize`` the label cannot affect
    placement, and every node here is drawn by Bokeh, not by Graphviz.
    """
    points_per_inch = DefaultOptimization.POINTS_PER_INCH
    geometry = nx.DiGraph()
    for node, data in graph.nodes(data=True):
        width = float(data.get("width", DefaultDimensions.DEFAULT_WIDTH))
        height = float(data.get("height", DefaultDimensions.DEFAULT_HEIGHT))
        geometry.add_node(
            node,
            label="",
            shape="box",
            fixedsize="true",
            width=width / points_per_inch,
            height=height / points_per_inch,
        )
    geometry.add_edges_from(graph.edges())
    return geometry


def _calculate_standard_layout(
    graph: nx.DiGraph, layout_engine: str
) -> Dict[str, Tuple[float, float]]:
    """Calculate layout with GraphViz, falling back to a spring layout.

    Args:
        graph: NetworkX graph
        layout_engine: GraphViz layout algorithm to use

    Returns:
        Dictionary mapping node IDs to (x, y) position tuples
    """
    # Try GraphViz first
    try:
        pos = nx.nx_agraph.graphviz_layout(
            _layout_graph_for_graphviz(graph),
            prog=layout_engine,
            args=DefaultOptimization.GRAPHVIZ_ARGS,
        )
    except Exception as e:
        print(f"Graphviz layout failed: {e}, falling back to spring layout")
        pos = nx.spring_layout(
            graph,
            k=DefaultOptimization.SPRING_K,
            iterations=DefaultOptimization.SPRING_ITERATIONS,
        )

    return pos


def update_raw_glyphs_with_positions(
    raw_glyphs: List[Dict[str, Any]],
    pos: Dict[str, Tuple[float, float]],
) -> None:
    """Update raw glyph data with calculated positions and handle gene positioning.

    This generalized function handles both compartment graphs (with special gene
    positioning within enzyme groups) and network graphs (basic positioning).

    Args:
        raw_glyphs: List of raw glyph data to update with positions
        pos: Dictionary mapping node IDs to (x, y) position tuples
    """
    # Scale positions
    x_scale = DefaultLayout.X_SCALE
    y_scale = DefaultLayout.Y_SCALE

    # First pass: Update main nodes (reactions, metabolites, enzyme groups, subsystems)
    enzyme_group_positions = {}
    for raw_glyph in raw_glyphs:
        if raw_glyph["id"] in pos:
            x, y = pos[raw_glyph["id"]]
            raw_glyph["x"] = x * x_scale
            raw_glyph["y"] = y * y_scale

            # Store enzyme group positions for gene positioning
            if raw_glyph["glyph_type"] == "enzyme_group":
                enzyme_group_positions[raw_glyph["id"]] = (
                    raw_glyph["x"],
                    raw_glyph["y"],
                )

    # Second pass: Position genes within enzyme groups (compartment graphs only)
    for raw_glyph in raw_glyphs:
        if raw_glyph["glyph_type"] == "gene":
            enzyme_group_id = raw_glyph["enzyme_group"]
            if enzyme_group_id in enzyme_group_positions:
                _position_gene_within_enzyme_group(
                    raw_glyph, enzyme_group_positions[enzyme_group_id], raw_glyphs
                )


def _position_gene_within_enzyme_group(
    gene_glyph: Dict[str, Any],
    enzyme_group_center: Tuple[float, float],
    raw_glyphs: List[Dict[str, Any]],
) -> None:
    """Position a gene glyph within its enzyme group.

    Args:
        gene_glyph: Raw gene glyph data to position
        enzyme_group_center: (x, y) center position of the enzyme group
        raw_glyphs: List of all raw glyphs to find enzyme group data
    """
    # Find the enzyme group data
    enzyme_group_data = None
    for glyph in raw_glyphs:
        if glyph["id"] == gene_glyph["enzyme_group"]:
            enzyme_group_data = glyph
            break

    if not enzyme_group_data:
        return

    # Extract positioning parameters
    gene_index = gene_glyph["gene_index"]
    cols = enzyme_group_data["num_cols"]
    group_width = enzyme_group_data["width"]
    group_height = enzyme_group_data["height"]

    # Calculate gene position within the group
    row = gene_index // cols
    col = gene_index % cols

    gene_width = DefaultDimensions.GENE_WIDTH
    gene_height = DefaultDimensions.GENE_HEIGHT
    padding = DefaultEnzymeGroupLayout.GENE_PADDING
    border_padding = DefaultEnzymeGroupLayout.BORDER_PADDING

    # Calculate top-left position of the enzyme group
    group_center_x, group_center_y = enzyme_group_center
    group_top_left_x = group_center_x - (group_width / 2)
    group_top_left_y = group_center_y - (group_height / 2)

    # Calculate gene position
    gene_x = (
        group_top_left_x
        + border_padding
        + col * (gene_width + padding)
        + gene_width / 2
    )
    gene_y = (
        group_top_left_y
        + border_padding
        + row * (gene_height + padding)
        + gene_height / 2
    )

    gene_glyph["x"] = gene_x
    gene_glyph["y"] = gene_y


def convert_raw_glyphs_to_domain_objects(
    raw_glyphs: List[Dict[str, Any]],
) -> List[Glyph]:
    """Convert raw glyph data to domain objects using factory functions.

    This generalized function handles all glyph types from both compartment
    graphs (reaction, metabolite, enzyme_group, gene) and network graphs (subsystem).

    Args:
        raw_glyphs: List of raw glyph data dictionaries with positions

    Returns:
        List of glyph objects created with appropriate factory functions
    """
    glyphs = []
    enzyme_group_genes = {}  # Track genes for enzyme groups

    for raw_glyph in raw_glyphs:
        if raw_glyph["glyph_type"] == "reaction":
            glyph = create_reaction_glyph(
                id=raw_glyph["id"],
                label=raw_glyph["label"],
                x=raw_glyph["x"],
                y=raw_glyph["y"],
                original_id=raw_glyph["original_id"],
                width=raw_glyph.get("width"),
                height=raw_glyph.get("height"),
            )
            glyphs.append(glyph)
        elif raw_glyph["glyph_type"] == "metabolite":
            # Map subtype from raw data
            subtype_mapping = {
                "intermediate": MetaboliteSubtype.INTERMEDIATE,
                "main": MetaboliteSubtype.MAIN,
            }
            subtype = subtype_mapping.get(
                raw_glyph.get("subtype", "main"), MetaboliteSubtype.MAIN
            )

            glyph = create_metabolite_glyph(
                id=raw_glyph["id"],
                label=raw_glyph["label"],
                x=raw_glyph["x"],
                y=raw_glyph["y"],
                original_id=raw_glyph["original_id"],
                subtype=subtype,
                width=raw_glyph.get("width"),
                height=raw_glyph.get("height"),
            )
            glyphs.append(glyph)
        elif raw_glyph["glyph_type"] == "enzyme_group":
            # Extract reaction ID from associated_reaction
            reaction_id = raw_glyph["associated_reaction"].split("_", 1)[1]
            enzyme_group_glyph = create_enzyme_group_glyph(
                id=raw_glyph["id"],
                label=raw_glyph["label"],
                x=raw_glyph["x"],
                y=raw_glyph["y"],
                reaction_id=reaction_id,
                enzymes=raw_glyph["genes"],
                glyph_ids=[],  # Will be filled when genes are processed
                width=raw_glyph.get("width", 100),
                height=raw_glyph.get("height", 100),
            )
            glyphs.append(enzyme_group_glyph)
            enzyme_group_genes[raw_glyph["id"]] = enzyme_group_glyph
        elif raw_glyph["glyph_type"] == "gene":
            glyph = create_gene_glyph(
                id=raw_glyph["id"],
                label=raw_glyph["label"],
                x=raw_glyph["x"],
                y=raw_glyph["y"],
                associated_reaction=raw_glyph["associated_reaction"],
                enzyme_group=raw_glyph["enzyme_group"],
                width=raw_glyph.get("width", DefaultDimensions.GENE_WIDTH),
                height=raw_glyph.get("height", DefaultDimensions.GENE_HEIGHT),
            )
            glyphs.append(glyph)

            # Add gene to enzyme group's glyph_ids
            enzyme_group_id = raw_glyph["enzyme_group"]
            if enzyme_group_id in enzyme_group_genes:
                enzyme_group_genes[enzyme_group_id]["glyph_ids"].append(raw_glyph["id"])
        elif raw_glyph["glyph_type"] == "subsystem":
            glyph = create_subsystem_glyph(
                id=raw_glyph["id"],
                label=raw_glyph["label"],
                x=raw_glyph["x"],
                y=raw_glyph["y"],
                subsystem_name=raw_glyph["subsystem_name"],
                reaction_count=raw_glyph["reaction_count"],
                metabolite_count=raw_glyph["metabolite_count"],
            )
            glyphs.append(glyph)
        elif raw_glyph["glyph_type"] == "subsystem_category":
            glyph = create_subsystem_category_glyph(
                id=raw_glyph["id"],
                label=raw_glyph["label"],
                x=raw_glyph["x"],
                y=raw_glyph["y"],
                radius=raw_glyph["radius"],
                subsystem_count=raw_glyph["subsystem_count"],
            )
            glyphs.append(glyph)

    return glyphs


def convert_raw_connections_to_domain_objects(
    raw_connections: List[Dict[str, Any]],
) -> List[Connection]:
    """Convert raw connection data to domain objects using factory functions.

    This generalized function handles all connection types from both compartment
    graphs (metabolic_connection, regulatory_connection) and network graphs
    (subsystem_connection).

    Args:
        raw_connections: List of raw connection data dictionaries

    Returns:
        List of connection objects created with factory functions
    """
    connections = []
    for raw_connection in raw_connections:
        if raw_connection["connection_type"] == "metabolic":
            connection = create_metabolic_connection(
                id=raw_connection["id"],
                source=raw_connection["source"],
                target=raw_connection["target"],
                connection_class=ConnectionClass.METABOLIC,
            )
            connections.append(connection)
        elif raw_connection["connection_type"] == "regulatory":
            connection = create_metabolic_connection(
                id=raw_connection["id"],
                source=raw_connection["source"],
                target=raw_connection["target"],
                connection_class=ConnectionClass.REGULATORY,
            )
            connections.append(connection)
        elif raw_connection["connection_type"] == "subsystem_connection":
            connection = create_subsystem_connection(
                id=raw_connection["id"],
                source=raw_connection["source"],
                target=raw_connection["target"],
                shared_metabolites=raw_connection["shared_metabolites"],
                weight=raw_connection["weight"],
                reaction_count=len(raw_connection.get("metabolite_details", [])),
                metabolite_flow_direction="forward",
            )
            connections.append(connection)
        elif raw_connection["connection_type"] == "category_connection":
            # For category connections, we can reuse subsystem_connection
            # since they have similar structure
            connection = create_subsystem_connection(
                id=raw_connection["id"],
                source=raw_connection["source"],
                target=raw_connection["target"],
                shared_metabolites=raw_connection["shared_metabolites"],
                weight=raw_connection["weight"],
                reaction_count=raw_connection["weight"],  # Use weight as reaction count
                metabolite_flow_direction="forward",
            )
            connections.append(connection)
    return connections

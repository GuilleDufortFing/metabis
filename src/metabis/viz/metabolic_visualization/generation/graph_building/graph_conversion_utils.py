"""Utilities for converting raw graph data to domain objects.

This module provides generalized functions for the raw data → layout → domain objects
pipeline that can be used across different graph types.
"""

import statistics
import uuid
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
)
from .metabolic_serpentine_layout import (
    METABOLIC_SERPENTINE_ENGINE,
    MetabolicSerpentineLayout,
)
from ..data_structures.core_types import (
    Connection,
    ConnectionClass,
    Glyph,
    MetaboliteSubtype,
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
    """Handle GraphViz layout calculation with fallback and optimization.

    This function automatically detects weighted graphs (specifically subsystem
    networks) and applies appropriate weight-based layout algorithms for better
    visual clustering of highly connected nodes.

    Args:
        graph: NetworkX graph to calculate layout for
        layout_engine: GraphViz layout algorithm to use

    Returns:
        Dictionary mapping node IDs to (x, y) position tuples
    """
    if layout_engine == METABOLIC_SERPENTINE_ENGINE:
        try:
            serpentine_layout = MetabolicSerpentineLayout()
            return serpentine_layout.layout_graph(graph)
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(
                "Serpentine layout failed: {0}. Falling back to standard layout.".format(
                    exc
                )
            )
            layout_engine = "dot"

    # Detect if this is a weighted subsystem graph
    # is_weighted_subsystem_graph = _detect_weighted_subsystem_graph(graph)

    # if is_weighted_subsystem_graph:
    #     print("Detected weighted subsystem graph, using weight-based layout...")
    #     pos = _calculate_weighted_layout(graph, layout_engine)
    # else:
    print("Using standard layout calculation...")
    pos = _calculate_standard_layout(graph, layout_engine)

    # Apply optimization passes (only for compartment graphs with metabolites)
    if _should_apply_optimization(graph):
        graph, pos, _ = optimize_layout(graph, pos)
        graph, pos, _ = optimize_layout(graph, pos)

    return pos


def _detect_weighted_subsystem_graph(graph: nx.DiGraph) -> bool:
    """Detect if this is a weighted subsystem graph that should use weight-based layout.

    Args:
        graph: NetworkX graph to analyze

    Returns:
        True if this is a weighted subsystem graph, False otherwise
    """
    # Check if graph has subsystem nodes and weighted edges
    has_subsystem_nodes = any(
        data.get("type") == "subsystem" for _, data in graph.nodes(data=True)
    )

    has_weighted_edges = any(
        data.get("connection_type") == "subsystem_connection" and "weight" in data
        for _, _, data in graph.edges(data=True)
    )

    return has_subsystem_nodes and has_weighted_edges


def _calculate_weighted_layout(
    graph: nx.DiGraph, layout_engine: str
) -> Dict[str, Tuple[float, float]]:
    """Calculate layout for weighted subsystem graphs using weight-aware algorithms.

    This function uses ghost connections to ensure proper spacing between unconnected
    subsystems while maintaining close positioning for highly connected ones.

    Args:
        graph: NetworkX graph with weighted edges
        layout_engine: Requested layout engine (may be overridden for weighted graphs)

    Returns:
        Dictionary mapping node IDs to (x, y) position tuples
    """
    # Extract and normalize weights for layout algorithms
    _extract_and_normalize_weights(graph)

    print("Using weighted spring layout...")
    pos = nx.spring_layout(
        graph,
        k=200,  # Larger base distance for subsystems
        iterations=20000,  # More iterations for weighted graphs
        weight="normalized_weight",  # Use normalized weights
        threshold=1e-6,  # Higher precision
        scale=70000,
        seed=42,  # Reproducible results
    )
    print(f"Spring layout successful with {len(pos)} nodes")

    return pos


def _calculate_standard_layout(
    graph: nx.DiGraph, layout_engine: str
) -> Dict[str, Tuple[float, float]]:
    """Calculate layout using standard (non-weighted) algorithms.

    Args:
        graph: NetworkX graph
        layout_engine: GraphViz layout algorithm to use

    Returns:
        Dictionary mapping node IDs to (x, y) position tuples
    """
    # Try GraphViz first
    try:
        pos = nx.nx_agraph.graphviz_layout(
            graph, prog=layout_engine, args=DefaultOptimization.GRAPHVIZ_ARGS
        )
    except Exception as e:
        print(f"Graphviz layout failed: {e}, falling back to spring layout")
        pos = nx.spring_layout(
            graph,
            k=DefaultOptimization.SPRING_K,
            iterations=DefaultOptimization.SPRING_ITERATIONS,
        )

    return pos


def _extract_and_normalize_weights(graph: nx.DiGraph) -> Dict[Tuple[str, str], float]:
    """Extract edge weights and normalize them using quadratic transformation.

    This approach addresses the outlier problem where connections with very high
    weights (like transport_reactions) dominate the linear normalization,
    causing most normal connections to have very similar normalized weights.

    The quadratic scheme ensures that:
    - Connection with 6 metabolites gets significantly higher weight than 1 metabolite
    - Outliers (>10 metabolites) don't dominate the normalization
    - Visual differences in the layout are more pronounced

    Args:
        graph: NetworkX graph with weighted edges

    Returns:
        Dictionary mapping edge tuples to normalized weights
    """
    weights = {}
    raw_weights = []

    # Extract all weights
    for source, target, data in graph.edges(data=True):
        weight = data.get("weight", 1.0)
        weights[(source, target)] = weight
        raw_weights.append(weight)

    if not raw_weights:
        return weights

    # Apply quadratic transformation with outlier resistance
    normalized_weights = {}

    # Set reasonable bounds for most biological connections (1-10 shared metabolites)
    min_expected = 1.0  # 1 metabolite
    max_expected = 5.0  # 10 metabolites as "normal maximum"

    # Target range for spring layout weights [0.1, 5.0]
    # Higher weight = stronger attraction = closer nodes
    min_layout_weight = 0.1
    max_layout_weight = 5.0

    for edge, raw_weight in weights.items():
        # Apply quadratic transformation to amplify differences
        quadratic_weight = raw_weight**1.5

        # Quadratic values for expected range
        min_quadratic = min_expected**1.5  # 1
        max_quadratic = max_expected**1.5  # 100

        # Normalize within expected range, cap outliers
        if quadratic_weight <= max_quadratic:
            # Normal case: normalize within expected range
            normalized = (quadratic_weight - min_quadratic) / (
                max_quadratic - min_quadratic
            )
        else:
            # Outlier case: cap at maximum to prevent domination
            normalized = 1.0

        # Scale to layout weight range
        weight_range = max_layout_weight - min_layout_weight
        scaled_weight = min_layout_weight + normalized * weight_range
        normalized_weights[edge] = scaled_weight

    # Add normalized weights back to graph
    for source, target, data in graph.edges(data=True):
        edge_key = (source, target)
        if edge_key in normalized_weights:
            data["normalized_weight"] = normalized_weights[edge_key]
        raw_weight = data.get("weight", 1.0)
        normalized_val = normalized_weights[edge_key]

    # Report weight distribution for analysis
    raw_weight_counts = {}
    for raw_weight in raw_weights:
        raw_weight_counts[raw_weight] = raw_weight_counts.get(raw_weight, 0) + 1

    for weight in sorted(raw_weight_counts.keys()):
        count = raw_weight_counts[weight]
        quadratic = weight**1.5
        if quadratic <= max_expected**1.5:
            normalized = (quadratic - 1) / 99
        else:
            normalized = 1.0
        weight_range = max_layout_weight - min_layout_weight
        final_weight = min_layout_weight + normalized * weight_range

    return normalized_weights


def _should_apply_optimization(graph: nx.DiGraph) -> bool:
    """Determine if optimization passes should be applied to this graph.

    Optimization (metabolite duplication) is only relevant for compartment graphs
    with reactions and metabolites, not for subsystem network graphs.

    Args:
        graph: NetworkX graph to check

    Returns:
        True if optimization should be applied, False otherwise
    """
    # Check if graph has reactions and metabolites (compartment graph)
    has_reactions = any(
        data.get("type") == "reaction" for _, data in graph.nodes(data=True)
    )
    has_metabolites = any(
        data.get("type") == "metabolite" for _, data in graph.nodes(data=True)
    )

    return has_reactions and has_metabolites


def optimize_layout(graph, pos, layout_engine="dot"):
    """Optimize graph layout by duplicating distant main metabolites.

    Args:
        graph (NetworkX DiGraph): The graph to improve
        pos (dict): Node positions dictionary
        layout_engine (str): GraphViz layout algorithm to use

    Returns:
        tuple: (Updated graph, Updated positions dictionary pos,
                Boolean indicating if changes were made)
    """
    # Calculate distances between reactions and main metabolites
    distances = []
    reaction_metabolite_edges = []

    for source, target in graph.edges():
        source_attrs = graph.nodes[source]
        target_attrs = graph.nodes[target]

        # Check if this is a reaction-metabolite edge
        if (
            source_attrs["type"] == "reaction"
            and target_attrs["type"] == "metabolite"
            and target_attrs.get("subtype") == "main"
        ):
            # Calculate Euclidean distance
            dist = (
                (pos[source][0] - pos[target][0]) ** 2
                + (pos[source][1] - pos[target][1]) ** 2
            ) ** 0.5
            distances.append(dist)
            reaction_metabolite_edges.append((source, target, dist))
        elif (
            target_attrs["type"] == "reaction"
            and source_attrs["type"] == "metabolite"
            and source_attrs.get("subtype") == "main"
        ):
            # Calculate Euclidean distance for the reverse direction
            dist = (
                (pos[source][0] - pos[target][0]) ** 2
                + (pos[source][1] - pos[target][1]) ** 2
            ) ** 0.5
            distances.append(dist)
            reaction_metabolite_edges.append((target, source, dist))

    if not distances:
        return graph, pos, False  # No edges to analyze

    # Calculate median distance
    median_distance = statistics.median(distances)
    print(f"Median reaction-metabolite distance: {median_distance:.4f}")

    # Find metabolites to duplicate (connected to multiple reactions with long edges)
    metabolites_to_check = {}
    for reaction, metabolite, dist in reaction_metabolite_edges:
        if dist > DefaultOptimization.DISTANCE_MULTIPLIER * median_distance:
            if metabolite not in metabolites_to_check:
                metabolites_to_check[metabolite] = []
            metabolites_to_check[metabolite].append(reaction)

    # Filter to only include metabolites connected to multiple reactions
    metabolites_to_duplicate = {
        met: reactions
        for met, reactions in metabolites_to_check.items()
        if len(reactions) > 1
    }

    if not metabolites_to_duplicate:
        return graph, pos, False  # No metabolites need duplication

    print(
        f"Duplicating {len(metabolites_to_duplicate)} metabolites with long connections"
    )

    # Duplicate metabolites and update graph
    for metabolite, reactions in metabolites_to_duplicate.items():
        metabolite_attrs = graph.nodes[metabolite].copy()

        # For each reaction connected to this metabolite
        for reaction in reactions:
            # Create a duplicate metabolite node
            new_node_id = f"m_{reaction}_{uuid.uuid4().hex[:6]}"

            # Copy original attributes
            graph.add_node(
                new_node_id,
                **metabolite_attrs,
                duplicated=True,
            )

            # Update edges
            if graph.has_edge(reaction, metabolite):
                graph.add_edge(reaction, new_node_id)
                graph.remove_edge(reaction, metabolite)
            elif graph.has_edge(metabolite, reaction):
                graph.add_edge(new_node_id, reaction)
                graph.remove_edge(metabolite, reaction)

    pos = nx.nx_agraph.graphviz_layout(
        graph, prog=layout_engine, args=DefaultOptimization.GRAPHVIZ_ARGS
    )
    return graph, pos, True


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

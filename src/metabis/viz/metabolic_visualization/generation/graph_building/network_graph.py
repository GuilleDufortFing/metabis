"""Metabolic network graph generation for subsystem-level visualizations.

This module provides functionality to create metabolic network visualizations where
nodes represent subsystems and edges represent shared metabolites between them.
"""

from typing import Any, Dict, List, Optional, Tuple

from ..data_structures.core_types import Connection, Glyph
from .graph_conversion_utils import convert_raw_data_to_domain_objects


def generate_subsystems_network_by_categories(
    model,
    category_filter: Optional[List[str]] = None,
    layout_engine: str = "fdp",
    category_layout_engine: str = "sfdp",
    category_spacing_multiplier: float = 0.56,
    include_inter_category_connections: bool = False,
) -> Tuple[List[Glyph], List[Connection]]:
    """Generate a complete metabolic network organized by subsystem categories.

    This function creates a comprehensive network visualization where:
    - Subsystems are grouped by their biological/functional categories
    - Each category forms a visually distinct region with a boundary
    - Categories can be arranged using graph layout or grid layout
    - Connections show metabolite sharing within and optionally between categories

    Args:
        model: MetabolicModel instance containing subsystem and reaction data
        category_filter: Optional list of specific categories to include.
                         If None, uses all categories in the model.
        layout_engine: GraphViz layout algorithm to use for subsystem positioning
                       ('dot', 'neato', 'fdp', 'sfdp', 'circo', 'twopi')
        category_layout_engine: GraphViz layout algorithm for category positioning
                              when use_graph_layout_for_categories=True (default: 'fdp')
        category_spacing_multiplier: Multiplier for spacing between categories
                                   when using graph layout (default: 2.0)
        include_inter_category_connections: Whether to include connections between
                                          different categories (default: True)
        use_graph_layout_for_categories: Whether to use graph layout (True) or
                                       grid layout (False) for categories
                                       (default: True)
        category_spacing: Minimum spacing between category boundaries for grid layout
                         (default: 300.0, used when
                         use_graph_layout_for_categories=False)
        max_categories_per_row: Maximum categories per row in grid layout
                              (default: 3, used when
                              use_graph_layout_for_categories=False)

    Returns:
        Tuple containing:
            - glyphs: List of all glyphs (subsystems and category boundaries)
            - connections: List of connections representing shared metabolites

    Raises:
        ValueError: If no valid categories are found or if all filtered
                   categories are empty

    Example:
        >>> # Generate network with graph layout and inter-category connections
        >>> glyphs, connections = generate_subsystems_network_by_categories(
        ...     model,
        ...     layout_engine='fdp',
        ...     category_layout_engine='neato',
        ...     include_inter_category_connections=True
        ... )
        >>>
        >>> # Generate network with grid layout and no inter-category connections
        >>> glyphs, connections = generate_subsystems_network_by_categories(
        ...     model,
        ...     category_filter=['Carbohydrate Metabolism', 'Lipid Metabolism'],
        ...     use_graph_layout_for_categories=False,
        ...     include_inter_category_connections=False
        ... )
        >>> print(f"Generated categorized network with {len(glyphs)} glyphs")
    """
    # Get available categories from the model
    subsystems_by_category = model.get_subsystems_by_category()
    available_categories = list(subsystems_by_category.keys())

    if not available_categories:
        raise ValueError("No subsystem categories found in the metabolic model")

    # Apply category filter if provided
    # if category_filter is not None:
    #     # Validate categories
    #     invalid_categories = set(category_filter) - set(available_categories)
    #     if invalid_categories:
    #         raise ValueError(f"Categories not found in model: {invalid_categories}")

    #     categories = category_filter
    # else:
    #     categories = available_categories

    categories = available_categories

    print(f"Generating metabolic network for {len(categories)} categories")

    # Generate individual networks for each category
    category_graphs = {}
    for category in categories:
        try:
            # Skip empty categories
            if not subsystems_by_category.get(category, []):
                continue

            # Generate network for this category
            glyphs, connections = generate_subsystems_network_for_category(
                model, category=category, layout_engine=layout_engine
            )

            # Only include categories with at least one subsystem
            if glyphs:
                category_graphs[category] = (glyphs, connections)

        except ValueError as e:
            print(f"  Error generating network for {category}: {e}")

    # Check if we have any valid categories
    if not category_graphs:
        raise ValueError("No valid category networks could be generated")

    # Extract inter-category relationships for graph-based layout
    inter_category_relationships = _extract_inter_category_relationships(
        model, categories
    )

    # Arrange category graphs using selected method
    arranged_categories = arrange_category_graphs_with_layout(
        category_graphs,
        inter_category_relationships,
        layout_engine=category_layout_engine,
        spacing_multiplier=category_spacing_multiplier,
    )

    # Collect all connections (intra-category and optionally inter-category)
    all_connections = _collect_all_connections(
        arranged_categories,
        inter_category_relationships,
        include_inter_category_connections=include_inter_category_connections,
    )

    # Combine all glyphs
    all_glyphs = []

    for category, (glyphs, connections, boundary) in arranged_categories.items():
        # Add the category boundary glyph
        all_glyphs.append(boundary)

        # Add all subsystem glyphs for this category
        all_glyphs.extend(glyphs)

    # Report final results

    return all_glyphs, all_connections


def generate_subsystems_network_for_category(
    model,
    category: str,
    layout_engine: str = "fdp",
) -> Tuple[List[Glyph], List[Connection]]:
    """Generate a metabolic network with subsystem nodes and shared metabolite edges.

    This function creates a network visualization where:
    - Nodes represent metabolic subsystems
    - Directed edges represent metabolite sharing between subsystems
    - Edge direction indicates metabolite flow (producer → consumer)
    - Edge weight represents the number of shared metabolites

    The function leverages the metabolic model's built-in subsystem neighbor
    relationships, which are pre-computed using per-subsystem metabolite
    classification (including SVG map-based extraction with fallback to
    intermediate metabolite filtering).

    Args:
        model: MetabolicModel instance containing subsystem and reaction data
        subsystem_filter: Optional list of specific subsystems to include.
                         If None, uses all subsystems in the model.
        layout_engine: GraphViz layout algorithm to use for positioning
                      ('dot', 'neato', 'fdp', 'sfdp', 'circo', 'twopi')

    Returns:
        Tuple containing:
            - glyphs: List of subsystem glyphs positioned in the network
            - connections: List of connections representing shared metabolites

    Raises:
        ValueError: If no valid subsystems are found

    Note:
        Connections always represent producer → consumer relationships:
        - Source subsystem produces metabolites that target subsystem consumes
        - This ensures biologically meaningful directional flow representation

    Example:
        >>> # Generate network for specific subsystems
        >>> glyphs, connections = generate_subsystems_network(
        ...     model,
        ...     subsystem_filter=['Glycolysis/Gluconeogenesis', 'TCA cycle'],
        ...     layout_engine='dot'
        ... )
        >>>
        >>> # Generate network for all subsystems
        >>> glyphs, connections = generate_subsystems_network(
        ...     model,
        ...     layout_engine='neato'
        ... )
        >>> print(f"Generated network with {len(glyphs)} subsystems and " +
        ...       f"{len(connections)} connections")
    """
    # Get all subsystems in the specified category
    subsystems_by_category = model.get_subsystems_by_category()

    if category not in subsystems_by_category:
        raise ValueError(f"Category '{category}' not found in model")

    category_subsystems = subsystems_by_category[category]

    if not category_subsystems:
        raise ValueError(f"No subsystems found in category '{category}'")

    # Get all neighbor relationships from the model
    all_neighbor_relationships = model.get_subsystem_neighbors()

    # Filter relationships to only include those between selected subsystems
    subsystem_set = set(category_subsystems)
    filtered_relationships = [
        rel
        for rel in all_neighbor_relationships
        if (
            rel.producing_subsystem_id in subsystem_set
            and rel.consuming_subsystem_id in subsystem_set
        )
    ]

    # Transform neighbor relationships into shared metabolite edges format
    shared_metabolite_edges = _transform_neighbor_relationships_to_edges(
        filtered_relationships
    )

    # Create raw subsystem data structures
    raw_glyphs, raw_connections = _create_raw_subsystem_data(
        model, category_subsystems, shared_metabolite_edges
    )

    # Use generalized pipeline for layout and conversion
    glyphs, connections = convert_raw_data_to_domain_objects(
        raw_glyphs, raw_connections, layout_engine
    )

    return glyphs, connections


def _transform_neighbor_relationships_to_edges(
    neighbor_relationships,
) -> List[Dict[str, Any]]:
    """Transform subsystem neighbor relationships into shared metabolite edge format.

    This function converts SubsystemNeighborRelationship objects into the format
    expected by _create_raw_subsystem_data, aggregating multiple relationships
    between the same pair of subsystems.

    Args:
        neighbor_relationships: List of SubsystemNeighborRelationship objects

    Returns:
        List of edge dictionaries in the format expected by _create_raw_subsystem_data
    """
    # Group relationships by subsystem pair
    subsystem_pair_groups = {}
    for rel in neighbor_relationships:
        pair_key = (rel.producing_subsystem_id, rel.consuming_subsystem_id)
        if pair_key not in subsystem_pair_groups:
            subsystem_pair_groups[pair_key] = []
        subsystem_pair_groups[pair_key].append(rel)
    shared_edges = []

    # Convert each group to an edge
    for pair_key, relationships in subsystem_pair_groups.items():
        producing_subsystem, consuming_subsystem = pair_key
        # Aggregate all metabolites and reaction details
        shared_metabolites = []
        metabolite_details = []

        for rel in relationships:
            shared_metabolites.append(rel.metabolite_id)
            metabolite_details.append(
                {
                    "metabolite": rel.metabolite_id,
                    "source_role": "product",  # producing subsystem produces it
                    "target_role": "reactant",  # consuming subsystem consumes it
                    "producing_reactions": rel.producing_reactions,
                    "consuming_reactions": rel.consuming_reactions,
                }
            )

        edge = {
            "source_subsystem": producing_subsystem,
            "target_subsystem": consuming_subsystem,
            "shared_metabolites": sorted(shared_metabolites),
            "direction": "forward",
            "weight": len(shared_metabolites),
            "metabolite_details": metabolite_details,
        }
        shared_edges.append(edge)

    return shared_edges


def _create_raw_subsystem_data(
    model,
    subsystems: List[str],
    shared_metabolite_edges: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create raw subsystem data structures without positions.

    This function generates raw glyph and connection data that will later be
    converted to domain objects after layout calculation.

    Args:
        model: MetabolicModel instance
        subsystems: List of subsystem names
        shared_metabolite_edges: Shared metabolite edge data

    Returns:
        Tuple of (raw_glyphs, raw_connections) as lists of dictionaries
    """
    raw_glyphs = []
    raw_connections = []

    # Create raw glyph data for each subsystem
    for subsystem in subsystems:
        # Calculate reaction count for this subsystem
        reaction_count = 0
        if model:
            try:
                reaction_ids = model.get_reactions_for_subsystem(subsystem)
                reaction_count = len(reaction_ids)
            except Exception:
                reaction_count = 1  # Default value if unable to calculate

        # Calculate metabolite count from neighbor relationships
        metabolite_count = 0
        try:
            # Get all unique metabolites involved in relationships
            # where this subsystem is either producer or consumer
            unique_metabolites = set()

            # Get relationships where this subsystem is producer
            producer_rels = model.get_subsystem_producers(subsystem)
            for rel in producer_rels:
                unique_metabolites.add(rel.metabolite_id)

            # Get relationships where this subsystem is consumer
            consumer_rels = model.get_subsystem_consumers(subsystem)
            for rel in consumer_rels:
                unique_metabolites.add(rel.metabolite_id)

            metabolite_count = len(unique_metabolites)
        except Exception:
            metabolite_count = 1  # Default value if unable to calculate

        raw_glyph = {
            "id": subsystem,
            "glyph_type": "subsystem",
            "label": subsystem,
            "subsystem_name": subsystem,
            "reaction_count": reaction_count,
            "metabolite_count": metabolite_count,
            "width": 50.0,  # Default width, will be adjusted by layout
            "height": 50.0,  # Default height, will be adjusted by layout
            # Positions will be added later by layout calculation
            "x": 0.0,
            "y": 0.0,
        }
        raw_glyphs.append(raw_glyph)

    # Create raw connection data from shared metabolite edges
    for edge_data in shared_metabolite_edges:
        source_sub = edge_data["source_subsystem"]
        target_sub = edge_data["target_subsystem"]
        connection_id = f"conn_{source_sub}_to_{target_sub}"

        raw_connection = {
            "id": connection_id,
            "connection_type": "subsystem_connection",
            "source": edge_data["source_subsystem"],
            "target": edge_data["target_subsystem"],
            "shared_metabolites": edge_data["shared_metabolites"],
            "weight": edge_data["weight"],
            "metabolite_details": edge_data["metabolite_details"],
            "direction": edge_data["direction"],
        }
        raw_connections.append(raw_connection)

    return raw_glyphs, raw_connections


def _calculate_category_boundary_radius(
    subsystem_glyphs: List[Glyph],
    center_x: float = 0.0,
    center_y: float = 0.0,
    padding: float = 50.0,
) -> Tuple[float, float, float]:
    """Calculate accurate radius and natural center for category boundary.

    This function computes the minimum circle that contains all subsystem glyphs
    and returns the radius and natural center coordinates.

    Args:
        subsystem_glyphs: List of subsystem glyphs to encompass
        center_x: Desired X coordinate for the category center
        center_y: Desired Y coordinate for the category center
        padding: Extra space around subsystems (default: 50.0)

    Returns:
        Tuple containing:
            - boundary_radius: Calculated radius including padding
            - natural_center_x: Natural center X coordinate based on glyph positions
            - natural_center_y: Natural center Y coordinate based on glyph positions

    Raises:
        ValueError: If no subsystem glyphs are provided
    """
    import math

    if not subsystem_glyphs:
        raise ValueError(
            "Cannot calculate category boundary: no subsystem glyphs provided"
        )

    # Calculate the bounding box of all subsystem glyphs
    min_x = min(glyph["x"] - glyph["width"] / 2 for glyph in subsystem_glyphs)
    max_x = max(glyph["x"] + glyph["width"] / 2 for glyph in subsystem_glyphs)
    min_y = min(glyph["y"] - glyph["height"] / 2 for glyph in subsystem_glyphs)
    max_y = max(glyph["y"] + glyph["height"] / 2 for glyph in subsystem_glyphs)

    # Calculate natural center of the bounding box
    natural_center_x = (min_x + max_x) / 2
    natural_center_y = (min_y + max_y) / 2

    # Calculate the radius needed to encompass all glyphs from the natural center
    max_distance = 0.0
    for glyph in subsystem_glyphs:
        # Distance from natural center to the farthest edge of each glyph
        distance_squared = (glyph["x"] - natural_center_x) ** 2 + (
            glyph["y"] - natural_center_y
        ) ** 2
        glyph_edge_distance = (
            math.sqrt(distance_squared) + max(glyph["width"], glyph["height"]) / 2
        )
        max_distance = max(max_distance, glyph_edge_distance)

    # Add padding to the radius
    boundary_radius = max_distance + padding

    return boundary_radius, natural_center_x, natural_center_y


def arrange_category_graphs_with_layout(
    category_graphs: Dict[str, Tuple[List[Glyph], List[Connection]]],
    inter_category_relationships: List[Dict[str, Any]],
    layout_engine: str = "fdp",
    spacing_multiplier: float = 1.0,
) -> Dict[str, Tuple[List[Glyph], List[Connection], Glyph]]:
    """Arrange multiple category graphs using graph layout algorithms.

    This function replaces the grid-based arrangement with a graph-layout-based
    approach that uses inter-category relationships to determine optimal positioning.

    Args:
        category_graphs: Dict mapping category names to (glyphs, connections) tuples
        inter_category_relationships: List of inter-category relationship dicts
        layout_engine: GraphViz layout algorithm ('fdp', 'neato', 'dot', 'circo')
        spacing_multiplier: Multiplier for spacing between categories (default: 2.0)

    Returns:
        Dict mapping category names to (glyphs, connections, boundary_glyph) tuples
        where glyphs and connections have been repositioned

    Example:
        >>> arranged = arrange_category_graphs_with_layout(
        ...     category_graphs, relationships, layout_engine='fdp'
        ... )
        >>> for category, (glyphs, connections, boundary) in arranged.items():
        ...     print(f"{category}: positioned at ({boundary['x']}, {boundary['y']})")
    """
    if not category_graphs:
        return {}

    categories = list(category_graphs.keys())

    # Build category meta-graph
    raw_category_glyphs, raw_category_connections = _build_category_meta_graph(
        categories, inter_category_relationships, category_graphs
    )

    # Use existing layout pipeline to position categories
    positioned_category_glyphs, positioned_category_connections = (
        convert_raw_data_to_domain_objects(
            raw_category_glyphs, raw_category_connections, layout_engine
        )
    )

    # Create mapping from category name to positioned glyph
    category_glyphs_map = {}
    for category_glyph in positioned_category_glyphs:
        category_name = category_glyph["label"]  # Use label instead of category_name
        # Apply spacing multiplier to increase separation
        center_x = category_glyph["x"] * spacing_multiplier
        center_y = category_glyph["y"] * spacing_multiplier

        # Store the complete positioned category glyph with scaled coordinates
        positioned_glyph = dict(category_glyph)  # Copy the glyph
        positioned_glyph["x"] = center_x
        positioned_glyph["y"] = center_y
        category_glyphs_map[category_name] = positioned_glyph

    # Arrange individual category contents at the calculated positions
    arranged_categories = {}

    for category_name in categories:
        original_glyphs, original_connections = category_graphs[category_name]
        positioned_category_glyph = category_glyphs_map[category_name]
        center_x = positioned_category_glyph["x"]
        center_y = positioned_category_glyph["y"]

        # Calculate offset to move category to target position
        current_center_x = sum(g["x"] for g in original_glyphs) / len(original_glyphs)
        current_center_y = sum(g["y"] for g in original_glyphs) / len(original_glyphs)

        offset_x = center_x - current_center_x
        offset_y = center_y - current_center_y

        # Create repositioned glyphs
        repositioned_glyphs = []
        for glyph in original_glyphs:
            # Create new glyph with updated position
            excluded_keys = ["id", "glyph_class", "label", "x", "y", "width", "height"]
            extra_attrs = {k: v for k, v in glyph.items() if k not in excluded_keys}
            new_glyph = type(glyph)(
                id=glyph["id"],
                glyph_class=glyph["glyph_class"],
                label=glyph["label"],
                x=glyph["x"] + offset_x,
                y=glyph["y"] + offset_y,
                width=glyph["width"],
                height=glyph["height"],
                **extra_attrs,
            )
            repositioned_glyphs.append(new_glyph)

        # Create repositioned connections (positions don't matter for connections)
        repositioned_connections = list(original_connections)

        arranged_categories[category_name] = (
            repositioned_glyphs,
            repositioned_connections,
            positioned_category_glyph,
        )
        # arranged_categories[category_name] = (
        #     [],
        #     [],
        #     positioned_category_glyph,
        # )

    return arranged_categories


def _extract_inter_category_relationships(
    model,
    categories: List[str],
) -> List[Dict[str, Any]]:
    """Extract relationships between different subsystem categories.

    This function analyzes all subsystem neighbor relationships to identify
    connections between different categories. If subsystem A in category C1
    has a relationship with subsystem B in category C2, this creates an
    inter-category relationship between C1 and C2.

    Args:
        model: MetabolicModel instance containing subsystem and reaction data
        categories: List of category names to analyze

    Returns:
        List of inter-category relationship dictionaries with format:
        {
            "source_category": str,
            "target_category": str,
            "subsystem_relationships": List[SubsystemNeighborRelationship],
            "connection_count": int,
            "shared_metabolites": List[str]
        }

    Example:
        >>> relationships = _extract_inter_category_relationships(model, categories)
        >>> for rel in relationships:
        ...     print(f"{rel['source_category']} → {rel['target_category']}: "
        ...           f"{rel['connection_count']} connections")
    """
    # Get category mapping for all subsystems
    subsystems_by_category = model.get_subsystems_by_category()

    # Create reverse mapping: subsystem -> category
    subsystem_to_category = {}
    for category in categories:
        if category in subsystems_by_category:
            for subsystem in subsystems_by_category[category]:
                subsystem_to_category[subsystem] = category

    # Get all neighbor relationships from the model
    all_neighbor_relationships = model.get_subsystem_neighbors()

    # Group inter-category relationships
    inter_category_groups = {}
    inter_category_count = 0

    for rel in all_neighbor_relationships:
        producer_category = subsystem_to_category.get(rel.producing_subsystem_id)
        consumer_category = subsystem_to_category.get(rel.consuming_subsystem_id)

        # Skip if either subsystem is not in our selected categories
        if not producer_category or not consumer_category:
            continue

        # Skip if both subsystems are in the same category (intra-category)
        if producer_category == consumer_category:
            continue

        # This is an inter-category relationship
        inter_category_count += 1
        category_pair = (producer_category, consumer_category)

        if category_pair not in inter_category_groups:
            inter_category_groups[category_pair] = []

        inter_category_groups[category_pair].append(rel)

    # Convert to standardized format
    inter_category_relationships = []

    for category_pair, relationships in inter_category_groups.items():
        source_category, target_category = category_pair
        # Collect all unique shared metabolites
        shared_metabolites = set()
        for rel in relationships:
            shared_metabolites.add(rel.metabolite_id)

        inter_category_rel = {
            "source_category": source_category,
            "target_category": target_category,
            "subsystem_relationships": relationships,
            "connection_count": len(relationships),
            "shared_metabolites": sorted(list(shared_metabolites)),
        }

        inter_category_relationships.append(inter_category_rel)

    return inter_category_relationships


def _build_category_meta_graph(
    categories: List[str],
    inter_category_relationships: List[Dict[str, Any]],
    category_graphs: Dict[str, Tuple[List[Glyph], List[Connection]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build a meta-graph where nodes are categories and edges are relationships.

    This function creates raw graph data structures for category-level positioning,
    where each category becomes a node in a graph layout algorithm.

    Args:
        categories: List of category names
        inter_category_relationships: List of inter-category relationship dictionaries
        category_graphs: Dict mapping category names to their (glyphs, connections)

    Returns:
        Tuple containing:
            - raw_category_glyphs: List of raw category node data for layout
            - raw_category_connections: List of raw category edge data for layout

    Example:
        >>> raw_glyphs, raw_connections = _build_category_meta_graph(
        ...     categories, relationships, category_graphs
        ... )
        >>> # These can then be passed to convert_raw_data_to_domain_objects()
    """
    raw_category_glyphs = []
    raw_category_connections = []

    # Create raw glyph data for each category
    for category in categories:
        # Get category information
        category_glyphs, category_connections = category_graphs.get(category, ([], []))

        # Calculate category statistics
        subsystem_count = len(category_glyphs)

        # Calculate accurate radius and natural center for category boundary
        if category_glyphs:
            try:
                (
                    boundary_radius,
                    natural_center_x,
                    natural_center_y,
                ) = _calculate_category_boundary_radius(
                    category_glyphs, center_x=0.0, center_y=0.0, padding=50.0
                )
            except ValueError:
                # Fallback if calculation fails
                boundary_radius = 150.0
                natural_center_x = 0.0
                natural_center_y = 0.0
        else:
            # Default values for empty categories
            boundary_radius = 150.0
            natural_center_x = 0.0
            natural_center_y = 0.0

        # Create raw category glyph data with accurate radius
        raw_category_glyph = {
            "id": f"category_{category.replace(' ', '_').lower()}",
            "glyph_type": "subsystem_category",
            "label": category,
            "category_name": category,
            "subsystem_count": subsystem_count,
            "width": boundary_radius * 2,  # Diameter for layout
            "height": boundary_radius * 2,  # Diameter for layout
            "radius": boundary_radius,
            "natural_center_x": natural_center_x,  # Store natural center for reference
            "natural_center_y": natural_center_y,
            # Positions will be calculated by layout algorithm
            "x": 0.0,
            "y": 0.0,
        }
        raw_category_glyphs.append(raw_category_glyph)


    # Create raw connection data from inter-category relationships
    for rel in inter_category_relationships:
        source_category = rel["source_category"]
        target_category = rel["target_category"]

        # Create connection ID
        source_id = f"category_{source_category.replace(' ', '_').lower()}"
        target_id = f"category_{target_category.replace(' ', '_').lower()}"
        connection_id = f"conn_{source_id}_to_{target_id}"

        raw_category_connection = {
            "id": connection_id,
            "connection_type": "category_connection",
            "source": source_id,
            "target": target_id,
            "weight": rel["connection_count"],
            "shared_metabolites": rel["shared_metabolites"],
            "subsystem_relationships": rel["subsystem_relationships"],
            "direction": "forward",
        }
        raw_category_connections.append(raw_category_connection)

    return raw_category_glyphs, raw_category_connections


def _collect_all_connections(
    arranged_categories: Dict[str, Tuple[List[Glyph], List[Connection], Glyph]],
    inter_category_relationships: List[Dict[str, Any]],
    include_inter_category_connections: bool = True,
) -> List[Connection]:
    """Collect all connections (intra-category and optionally inter-category).

    This function aggregates connections from all categories and optionally adds
    inter-category connections based on subsystem relationships.

    Args:
        arranged_categories: Dict mapping category names to
                           (glyphs, connections, boundary) tuples
        inter_category_relationships: List of inter-category relationship dicts
        include_inter_category_connections: Whether to include connections between
                                          categories (default: True)

    Returns:
        List of all connections to include in the final visualization

    Example:
        >>> # Include both intra and inter-category connections
        >>> all_connections = _collect_all_connections(
        ...     arranged_categories, relationships,
        ...     include_inter_category_connections=True
        ... )
        >>>
        >>> # Include only intra-category connections
        >>> intra_connections = _collect_all_connections(
        ...     arranged_categories, relationships,
        ...     include_inter_category_connections=False
        ... )
    """
    all_connections = []

    # Collect intra-category connections
    intra_category_count = 0
    for category, (glyphs, connections, boundary) in arranged_categories.items():
        all_connections.extend(connections)
        intra_category_count += len(connections)

    # Optionally add inter-category connections
    inter_category_count = 0
    if include_inter_category_connections:

        # Create a mapping from subsystem ID to category for quick lookup
        subsystem_to_category = {}
        for category, (glyphs, connections, boundary) in arranged_categories.items():
            for glyph in glyphs:
                if glyph["glyph_class"] == "subsystem":
                    subsystem_to_category[glyph["id"]] = category

        # Transform inter-category relationships to connections
        for rel in inter_category_relationships:
            # Convert subsystem relationships to connections
            for subsystem_rel in rel["subsystem_relationships"]:
                producer_id = subsystem_rel.producing_subsystem_id
                consumer_id = subsystem_rel.consuming_subsystem_id

                # Verify both subsystems exist in our arranged categories
                producer_exists = producer_id in subsystem_to_category
                consumer_exists = consumer_id in subsystem_to_category
                if producer_exists and consumer_exists:
                    # Create inter-category connection
                    connection_id = f"inter_conn_{producer_id}_to_{consumer_id}"

                    inter_connection = {
                        "id": connection_id,
                        "connection_type": "inter_category_connection",
                        "source": producer_id,
                        "target": consumer_id,
                        "shared_metabolites": [subsystem_rel.metabolite_id],
                        "weight": 1,  # Each subsystem relationship has weight 1
                        "metabolite_details": [
                            {
                                "metabolite": subsystem_rel.metabolite_id,
                                "source_role": "product",
                                "target_role": "reactant",
                                "producing_reactions": (
                                    subsystem_rel.producing_reactions
                                ),
                                "consuming_reactions": (
                                    subsystem_rel.consuming_reactions
                                ),
                            }
                        ],
                        "direction": "forward",
                        "source_category": rel["source_category"],
                        "target_category": rel["target_category"],
                    }
                    all_connections.append(inter_connection)
                    inter_category_count += 1


    return all_connections

"""Human1 metabolic map generation from SBGN and SVG files.

This module provides functionality to generate graph visualizations from Human1
metabolic map files, including SBGN parsing, SVG metadata extraction, enzyme
grouping, and metabolite classification.
"""

import os
from collections import defaultdict
from math import sqrt
from typing import Any, Dict, List, Tuple

from ..data_structures.basic_factories import (
    create_compartment_glyph,
    create_enzyme_group_glyph,
    create_gene_glyph,
    create_metabolic_connection,
    create_metabolite_glyph,
    create_reaction_glyph,
)
from ..data_structures.core_types import (
    Connection,
    ConnectionClass,
    Glyph,
    GlyphClass,
)
from ..data_structures.validation import validate_graph_data
from ..parsers import (
    extract_enzyme_reaction_associations,
    extract_metabolite_subtypes,
    extract_raw_sbgn_data,
    extract_reaction_id_mapping,
)
from ..utilities.graph_generation_utils import (
    GraphConfig,
    validate_config,
    validate_glyphs,
)


def generate_graph_from_h1_map(
    config: GraphConfig,
) -> Tuple[List[Glyph], List[Connection]]:
    """Generate visualization data from Human1 metabolic map files.

    This function processes SBGN and SVG files to create a complete graph
    representation with enhanced metabolite classifications and enzyme groupings.

    Args:
        config: Configuration dictionary containing:
            - sbgn_file: Path to SBGN file containing base graph structure
            - svg_file: Optional path to SVG file for additional metadata
            - hmr_mar_file: Path to HMR/MAR mapping file for ID translation

    Returns:
        Tuple containing:
            - glyphs: List of enhanced glyphs with metabolite subtypes and enzyme groups
            - connections: List of connections between glyphs

    Raises:
        ValueError: If required config parameters are missing or files don't exist

    Example:
        >>> config = {
        ...     'sbgn_file': 'human1.sbgn',
        ...     'svg_file': 'human1.svg',
        ...     'hmr_mar_file': 'hmr_mar_mapping.tsv'
        ... }
        >>> glyphs, connections = generate_graph_from_h1_map(config)
    """
    validate_config(config)
    print(f"Parsing Human1 map file: {config['sbgn_file']}")

    # Parse SBGN file with reaction ID translation using new raw data approach
    hmr_mar_file = config.get("hmr_mar_file")
    id_mapping = extract_reaction_id_mapping(hmr_mar_file) if hmr_mar_file else None

    # Extract raw data from SBGN
    raw_glyphs, raw_connections = extract_raw_sbgn_data(config["sbgn_file"], id_mapping)

    # Convert raw data to domain objects using factory functions
    glyphs = _convert_raw_glyphs_to_domain_objects(raw_glyphs)
    connections = _convert_raw_connections_to_domain_objects(raw_connections)

    # Parse SVG file if provided and process data
    if config["svg_file"] and os.path.exists(config["svg_file"]):
        # Extract specific data from SVG
        metabolite_subtypes = extract_metabolite_subtypes(config["svg_file"])
        enzyme_associations = extract_enzyme_reaction_associations(config["svg_file"])

        # Classify metabolite types using extracted subtypes
        glyphs = _classify_metabolite_types(glyphs, metabolite_subtypes)

        # Create enzyme groups and their regulatory connections
        enzyme_group_glyphs, regulatory_connections = _create_enzyme_groups(
            glyphs, enzyme_associations
        )
        glyphs.extend(enzyme_group_glyphs)
        connections.extend(regulatory_connections)
    else:
        print(f"SVG file not found: {config['svg_file']}, skipping SVG parsing")

    # Validate the generated graph data before returning
    is_valid, errors = validate_graph_data(glyphs, connections)
    if not is_valid:
        print(f"Warning: Generated graph data has validation errors: {errors}")

    return glyphs, connections


def _convert_raw_glyphs_to_domain_objects(
    raw_glyphs: List[Dict[str, Any]],
) -> List[Glyph]:
    """Convert raw glyph data to domain objects using factory functions.

    Args:
        raw_glyphs: List of raw glyph data dictionaries from SBGN parser

    Returns:
        List of glyph objects created with appropriate factory functions
    """
    glyphs = []
    for raw_glyph in raw_glyphs:
        if raw_glyph["glyph_type"] == "compartment":
            glyph_data = create_compartment_glyph(
                id=raw_glyph["id"],
                label=raw_glyph["label"],
                x=raw_glyph["x"],
                y=raw_glyph["y"],
                width=raw_glyph["width"],
                height=raw_glyph["height"],
            )
        elif raw_glyph["glyph_type"] == "metabolite":
            glyph_data = create_metabolite_glyph(
                id=raw_glyph["id"],
                label=raw_glyph["label"],
                x=raw_glyph["x"],
                y=raw_glyph["y"],
                width=raw_glyph["width"],
                height=raw_glyph["height"],
            )
        elif raw_glyph["glyph_type"] == "gene":
            glyph_data = create_gene_glyph(
                id=raw_glyph["id"],
                label=raw_glyph["label"],
                x=raw_glyph["x"],
                y=raw_glyph["y"],
                width=raw_glyph["width"],
                height=raw_glyph["height"],
                associated_reaction=None,  # No reaction association from SBGN
            )
        else:  # reaction
            glyph_data = create_reaction_glyph(
                id=raw_glyph["id"],
                label=raw_glyph["id"],  # The ID is the MAR reaction name
                x=raw_glyph["x"],
                y=raw_glyph["y"],
                original_id=raw_glyph["xml_id"],
                width=raw_glyph["width"],
                height=raw_glyph["height"],
            )

        glyphs.append(glyph_data)
    return glyphs


def _convert_raw_connections_to_domain_objects(
    raw_connections: List[Dict[str, Any]],
) -> List[Connection]:
    """Convert raw connection data to domain objects using factory functions.

    Args:
        raw_connections: List of raw connection data dictionaries from SBGN parser

    Returns:
        List of connection objects created with factory functions
    """
    connections = []
    for raw_connection in raw_connections:
        connection_data = create_metabolic_connection(
            id=raw_connection["id"],
            source=raw_connection["source"],
            target=raw_connection["target"],
        )
        connections.append(connection_data)
    return connections


def _create_reaction_position_mapping(
    glyphs: List[Glyph],
) -> Dict[str, Tuple[float, float]]:
    """Create a mapping from reaction IDs to their positions.

    Args:
        glyphs: List of glyphs to extract reaction positions from

    Returns:
        Dictionary mapping reaction IDs to (x, y) position tuples
    """
    reaction_positions = {}
    for glyph in glyphs:
        if glyph["glyph_class"] == GlyphClass.REACTION:
            reaction_positions[glyph["label"]] = (glyph["x"], glyph["y"])
    return reaction_positions


def _calculate_distance(
    point1: Tuple[float, float], point2: Tuple[float, float]
) -> float:
    """Calculate Euclidean distance between two points.

    Args:
        point1: First point as (x, y) tuple
        point2: Second point as (x, y) tuple

    Returns:
        Euclidean distance between the points
    """
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# ============================================================================
# Overlap-based Enzyme Grouping Functions
# ============================================================================


def _calculate_glyph_bounds(
    glyph: Glyph, tolerance: float = 0
) -> Tuple[float, float, float, float]:
    """Calculate bounding box of a glyph with optional tolerance.

    Args:
        glyph: Gene glyph to calculate bounds for
        tolerance: Additional padding around the glyph bounds

    Returns:
        Tuple of (min_x, max_x, min_y, max_y) bounding box coordinates
    """
    half_width = glyph["width"] / 2
    half_height = glyph["height"] / 2

    min_x = glyph["x"] - half_width - tolerance
    max_x = glyph["x"] + half_width + tolerance
    min_y = glyph["y"] - half_height - tolerance
    max_y = glyph["y"] + half_height + tolerance

    return min_x, max_x, min_y, max_y


def _find_interval_overlaps(
    intervals: List[Tuple[float, float, str]],
) -> Dict[str, set]:
    """Find all overlapping intervals using sweep line algorithm.

    Args:
        intervals: List of (start, end, glyph_id) tuples

    Returns:
        Dictionary mapping glyph_id to set of overlapping glyph_ids
    """
    if not intervals:
        return {}

    # Create events for sweep line algorithm
    events = []
    for start, end, glyph_id in intervals:
        events.append((start, "start", glyph_id))
        events.append((end, "end", glyph_id))

    # Sort events by coordinate, with 'start' events before 'end' events at same coordinate
    events.sort(key=lambda x: (x[0], x[1] == "end"))

    active_intervals = set()
    overlaps = defaultdict(set)

    for coord, event_type, glyph_id in events:
        if event_type == "start":
            # All currently active intervals overlap with this one
            for active_id in active_intervals:
                overlaps[glyph_id].add(active_id)
                overlaps[active_id].add(glyph_id)
            active_intervals.add(glyph_id)
        else:  # 'end'
            active_intervals.remove(glyph_id)

    return overlaps


def _build_overlap_graph(
    gene_glyphs: List[Glyph], tolerance: float = 2
) -> Dict[str, List[str]]:
    """Build adjacency list of overlapping gene glyphs using optimized sweep line algorithm.

    This function uses an O(n log n) sweep line algorithm instead of the naive O(n²)
    approach, making it much faster for large numbers of gene glyphs.

    Args:
        gene_glyphs: List of gene glyphs to analyze for overlaps
        tolerance: Tolerance for considering glyphs as overlapping

    Returns:
        Dictionary mapping glyph IDs to lists of overlapping glyph IDs
    """
    if not gene_glyphs:
        return {}

    # Create interval representations for X and Y dimensions
    x_intervals = []
    y_intervals = []

    for glyph in gene_glyphs:
        min_x, max_x, min_y, max_y = _calculate_glyph_bounds(glyph, tolerance)
        x_intervals.append((min_x, max_x, glyph["id"]))
        y_intervals.append((min_y, max_y, glyph["id"]))

    # Find overlaps in each dimension using sweep line algorithm
    x_overlaps = _find_interval_overlaps(x_intervals)
    y_overlaps = _find_interval_overlaps(y_intervals)

    # Build final adjacency list: intersection of X and Y overlaps
    adjacency_list = {glyph["id"]: [] for glyph in gene_glyphs}

    for glyph_id in adjacency_list:
        if glyph_id in x_overlaps and glyph_id in y_overlaps:
            # True 2D overlap = intersection of X and Y overlaps
            true_overlaps = x_overlaps[glyph_id].intersection(y_overlaps[glyph_id])
            adjacency_list[glyph_id] = list(true_overlaps)

    return adjacency_list


def _find_connected_components_dfs(
    adjacency_list: Dict[str, List[str]],
) -> List[List[str]]:
    """Use DFS to find all connected components (gene groups) in the overlap graph.

    Args:
        adjacency_list: Graph of overlapping gene glyphs

    Returns:
        List of connected components, each component is a list of glyph IDs
    """
    visited = set()
    components = []

    def dfs(node: str, current_component: List[str]):
        visited.add(node)
        current_component.append(node)

        for neighbor in adjacency_list[node]:
            if neighbor not in visited:
                dfs(neighbor, current_component)

    # Find all connected components
    for node in adjacency_list:
        if node not in visited:
            component = []
            dfs(node, component)
            components.append(component)

    return components


def _calculate_group_centroid(gene_group: List[Glyph]) -> Tuple[float, float]:
    """Calculate the center point of a gene group.

    Args:
        gene_group: List of gene glyphs in the group

    Returns:
        Tuple of (center_x, center_y) coordinates
    """
    if not gene_group:
        return 0.0, 0.0

    total_x = sum(glyph["x"] for glyph in gene_group)
    total_y = sum(glyph["y"] for glyph in gene_group)

    center_x = total_x / len(gene_group)
    center_y = total_y / len(gene_group)

    return center_x, center_y


def _get_group_associated_reactions(
    gene_group: List[Glyph], enzyme_associations: Dict[str, List[str]]
) -> List[str]:
    """Get reactions that ALL genes in the group are associated with.

    Args:
        gene_group: List of gene glyphs in the group
        enzyme_associations: Dictionary mapping gene names to reaction IDs

    Returns:
        List of reaction IDs that all genes in the group are associated with
    """
    if not gene_group:
        return []

    # Get gene names from the group
    gene_names = [glyph["label"] for glyph in gene_group]

    # Start with reactions associated with the first gene
    first_gene = gene_names[0]
    if first_gene not in enzyme_associations:
        return []

    common_reactions = set(enzyme_associations[first_gene])

    # Find intersection with reactions from all other genes
    for gene_name in gene_names[1:]:
        if gene_name not in enzyme_associations:
            return []  # If any gene has no associations, no common reactions

        gene_reactions = set(enzyme_associations[gene_name])
        common_reactions = common_reactions.intersection(gene_reactions)

    return list(common_reactions)


def _assign_with_competition(
    group_idx: int,
    gene_group: List[Glyph],
    candidate_reactions: List[str],
    reaction_positions: Dict[str, Tuple[float, float]],
    reaction_assignments: Dict[str, Tuple[int, float, List[Glyph]]],
) -> Tuple[str, float]:
    """Assign group to best available reaction, potentially replacing worse assignments.

    Args:
        group_idx: Index of the current gene group
        gene_group: List of gene glyphs in the group
        candidate_reactions: List of reaction IDs to consider
        reaction_positions: Mapping of reaction IDs to positions
        reaction_assignments: Current assignments: reaction_id -> (group_idx, distance, gene_group)

    Returns:
        Tuple of (best_reaction_id, distance) or (None, inf) if no assignment possible
    """
    if not candidate_reactions or not gene_group:
        return None, float("inf")

    group_centroid = _calculate_group_centroid(gene_group)
    best_reaction = None
    best_distance = float("inf")

    for reaction_id in candidate_reactions:
        if reaction_id not in reaction_positions:
            continue

        reaction_pos = reaction_positions[reaction_id]
        distance = _calculate_distance(group_centroid, reaction_pos)

        # Check if this reaction is unassigned or we're closer
        if reaction_id not in reaction_assignments:
            # Unassigned reaction
            if distance < best_distance:
                best_distance = distance
                best_reaction = reaction_id
        else:
            # Assigned reaction - check if we're closer
            current_group_idx, current_distance, current_group = reaction_assignments[
                reaction_id
            ]
            if distance < current_distance and distance < best_distance:
                best_distance = distance
                best_reaction = reaction_id

    return best_reaction, best_distance


def _competitive_assignment_phase(
    valid_groups: List[Tuple[int, List[Glyph], List[str]]],
    reaction_positions: Dict[str, Tuple[float, float]],
    enzyme_associations: Dict[str, List[str]],
) -> Tuple[Dict[str, Tuple[int, float, List[Glyph]]], List[Tuple[int, List[Glyph]]]]:
    """Perform competitive assignment of valid groups to reactions.

    Args:
        valid_groups: List of (group_idx, gene_group, valid_reactions) tuples
        reaction_positions: Mapping of reaction IDs to positions
        enzyme_associations: Dictionary mapping gene names to reaction IDs

    Returns:
        Tuple of (reaction_assignments, invalid_groups_after_displacement)
    """
    reaction_assignments = {}  # reaction_id -> (group_idx, distance, gene_group)
    displacement_queue = []
    invalid_groups_after_displacement = []

    # Phase 1: Initial assignments with competition
    for group_idx, gene_group, valid_reactions in valid_groups:
        best_reaction, best_distance = _assign_with_competition(
            group_idx,
            gene_group,
            valid_reactions,
            reaction_positions,
            reaction_assignments,
        )

        if best_reaction:
            # Check if we're displacing another group
            if best_reaction in reaction_assignments:
                displaced_idx, displaced_dist, displaced_group = reaction_assignments[
                    best_reaction
                ]
                displacement_queue.append((displaced_idx, displaced_group))

                gene_names = [g["label"] for g in gene_group]
                displaced_names = [g["label"] for g in displaced_group]
                print(
                    f"Group {gene_names} (d={best_distance:.1f}) displaces group {displaced_names} (d={displaced_dist:.1f}) from reaction {best_reaction}"
                )

            # Make the assignment
            reaction_assignments[best_reaction] = (group_idx, best_distance, gene_group)

    # Phase 2: Handle displaced groups
    while displacement_queue:
        displaced_idx, displaced_group = displacement_queue.pop(0)

        # Find alternative reactions for displaced group
        displaced_valid_reactions = _get_group_associated_reactions(
            displaced_group, enzyme_associations
        )

        # Try to assign to unassigned reactions first
        unassigned_reactions = [
            r for r in displaced_valid_reactions if r not in reaction_assignments
        ]

        if unassigned_reactions:
            # Assign to closest unassigned
            best_reaction, best_distance = _assign_with_competition(
                displaced_idx,
                displaced_group,
                unassigned_reactions,
                reaction_positions,
                reaction_assignments,
            )
            if best_reaction:
                reaction_assignments[best_reaction] = (
                    displaced_idx,
                    best_distance,
                    displaced_group,
                )
                displaced_names = [g["label"] for g in displaced_group]
                print(
                    f"Displaced group {displaced_names} reassigned to reaction {best_reaction} (d={best_distance:.1f})"
                )
            else:
                # No unassigned reactions available
                invalid_groups_after_displacement.append(
                    (displaced_idx, displaced_group)
                )
        else:
            # All valid reactions are assigned - add to invalid groups
            invalid_groups_after_displacement.append((displaced_idx, displaced_group))

    return reaction_assignments, invalid_groups_after_displacement


def _create_enzyme_groups_from_assignments(
    reaction_assignments: Dict[str, Tuple[int, float, List[Glyph]]],
    enzyme_group_glyphs: List[Glyph],
    regulatory_connections: List[Connection],
) -> None:
    """Create enzyme group glyphs and connections from competitive assignments.

    Args:
        reaction_assignments: Final assignments: reaction_id -> (group_idx, distance, gene_group)
        enzyme_group_glyphs: List to append new enzyme group glyphs to
        regulatory_connections: List to append new connections to
    """
    for reaction_id, (group_idx, distance, gene_group) in reaction_assignments.items():
        _create_enzyme_group_glyph_and_connection(
            group_idx,
            gene_group,
            reaction_id,
            enzyme_group_glyphs,
            regulatory_connections,
        )


def _assign_to_closest_unassigned_reaction(
    gene_group: List[Glyph],
    available_reactions: List[str],
    reaction_positions: Dict[str, Tuple[float, float]],
    assigned_reactions: set,
) -> str:
    """Assign group to closest reaction that hasn't been assigned yet.

    Args:
        gene_group: List of gene glyphs in the group
        available_reactions: List of reaction IDs to consider
        reaction_positions: Mapping of reaction IDs to positions
        assigned_reactions: Set of already assigned reaction IDs

    Returns:
        ID of the assigned reaction
    """
    if not gene_group:
        return None

    group_centroid = _calculate_group_centroid(gene_group)
    closest_reaction = None
    min_distance = float("inf")

    for reaction_id in available_reactions:
        if reaction_id not in assigned_reactions and reaction_id in reaction_positions:
            reaction_pos = reaction_positions[reaction_id]
            distance = _calculate_distance(group_centroid, reaction_pos)
            if distance < min_distance:
                min_distance = distance
                closest_reaction = reaction_id

    return closest_reaction


def _calculate_enzyme_group_bounds(
    enzyme_glyphs_list: List[Glyph],
) -> Tuple[float, float, float, float]:
    """Calculate bounding box for a group of enzyme glyphs.

    Args:
        enzyme_glyphs_list: List of enzyme glyphs to calculate bounds for

    Returns:
        Tuple of (min_x, max_x, min_y, max_y) bounding box coordinates
    """
    if not enzyme_glyphs_list:
        return 0.0, 0.0, 0.0, 0.0

    padding = 5
    min_x = min(g["x"] - g["width"] / 2 for g in enzyme_glyphs_list) - padding
    max_x = max(g["x"] + g["width"] / 2 for g in enzyme_glyphs_list) + padding
    min_y = min(g["y"] - g["height"] / 2 for g in enzyme_glyphs_list) - padding
    max_y = max(g["y"] + g["height"] / 2 for g in enzyme_glyphs_list) + padding

    return min_x, max_x, min_y, max_y


def _create_enzyme_groups(
    glyphs: List[Glyph], enzyme_associations: Dict[str, List[str]]
) -> Tuple[List[Glyph], List[Connection]]:
    """Group enzymes (genes) by spatial overlap and associate with reactions.

    This function uses a new overlap-based approach:
    1. Groups genes that spatially overlap (are "touching")
    2. For each group, finds reactions that ALL genes are associated with
    3. Assigns group to the closest valid reaction
    4. Creates enzyme group glyphs and regulatory connections

    Args:
        glyphs: List of existing glyphs including gene glyphs
        enzyme_associations: Dictionary mapping enzyme IDs to associated reaction IDs

    Returns:
        Tuple containing:
            - enzyme_group_glyphs: List of enzyme group glyphs with bounding box info
            - regulatory_connections: List of REGULATORY connections from groups
              to reactions

    Note:
        - Every gene gets an enzyme group (single genes form individual groups)
        - Groups are formed by spatial overlap, not reaction association
        - Only reactions where ALL genes in group are associated are considered
        - Fallback to closest reaction if no valid associations exist
    """
    validate_glyphs(glyphs)

    # Step 1: Extract gene glyphs and create reaction position mapping
    gene_glyphs = [glyph for glyph in glyphs if glyph["glyph_class"] == GlyphClass.GENE]
    reaction_positions = _create_reaction_position_mapping(glyphs)

    if not gene_glyphs:
        return [], []

    # Step 2: Build overlap graph and find connected components (gene groups)
    overlap_graph = _build_overlap_graph(gene_glyphs, tolerance=2)
    gene_group_ids = _find_connected_components_dfs(overlap_graph)

    # Step 3: Convert glyph ID groups to glyph objects
    gene_glyphs_by_id = {glyph["id"]: glyph for glyph in gene_glyphs}
    gene_groups = []
    for group_ids in gene_group_ids:
        group_glyphs = [gene_glyphs_by_id[glyph_id] for glyph_id in group_ids]
        gene_groups.append(group_glyphs)

    # Step 4: Assign each gene group to a reaction with competitive assignment
    enzyme_group_glyphs = []
    regulatory_connections = []

    # Categorize groups by validity
    valid_groups = []
    invalid_groups = []

    for i, gene_group in enumerate(gene_groups):
        # Find reactions that ALL genes in the group are associated with
        valid_reactions = _get_group_associated_reactions(
            gene_group, enzyme_associations
        )

        if valid_reactions:
            valid_groups.append((i, gene_group, valid_reactions))
        else:
            invalid_groups.append((i, gene_group))

    print(
        f"Found {len(valid_groups)} valid groups and {len(invalid_groups)} invalid groups"
    )

    # Phase 1: Competitive assignment for valid groups
    reaction_assignments, displaced_invalid_groups = _competitive_assignment_phase(
        valid_groups, reaction_positions, enzyme_associations
    )

    # Add displaced groups to invalid groups
    invalid_groups.extend(displaced_invalid_groups)

    # Create enzyme groups from competitive assignments
    _create_enzyme_groups_from_assignments(
        reaction_assignments, enzyme_group_glyphs, regulatory_connections
    )

    # Track assigned reactions for Phase 2
    assigned_reactions = set(reaction_assignments.keys())

    # Phase 2: Assign invalid groups to closest unassigned reactions
    for i, gene_group in invalid_groups:
        gene_names = [glyph["label"] for glyph in gene_group]
        print(
            f"Warning: Gene group {gene_names} has no common associated reactions. "
            f"Assigning to closest available reaction."
        )

        # Get all reactions from any gene in the group
        all_possible_reactions = set()
        for glyph in gene_group:
            gene_name = glyph["label"]
            if gene_name in enzyme_associations:
                all_possible_reactions.update(enzyme_associations[gene_name])

        # Try to assign to an unassigned reaction from gene associations
        assigned_reaction = None
        if all_possible_reactions:
            assigned_reaction = _assign_to_closest_unassigned_reaction(
                gene_group,
                list(all_possible_reactions),
                reaction_positions,
                assigned_reactions,
            )

        # If no unassigned reactions from associations, try all reactions
        if not assigned_reaction:
            all_reactions = list(reaction_positions.keys())
            assigned_reaction = _assign_to_closest_unassigned_reaction(
                gene_group, all_reactions, reaction_positions, assigned_reactions
            )

        # Last resort: assign to closest reaction by distance (ignoring assignments)
        if not assigned_reaction:
            group_center = _calculate_group_centroid(gene_group)
            min_distance = float("inf")
            closest_reaction = None

            for reaction_id, reaction_pos in reaction_positions.items():
                distance = _calculate_distance(group_center, reaction_pos)
                if distance < min_distance:
                    min_distance = distance
                    closest_reaction = reaction_id

            assigned_reaction = closest_reaction
            print(
                f"Warning: No unassigned reactions found for gene group {gene_names}. "
                f"Assigned to closest reaction: {assigned_reaction} (may be duplicate)"
            )

        if assigned_reaction:
            assigned_reactions.add(assigned_reaction)

        # Create enzyme group glyph and regulatory connection
        _create_enzyme_group_glyph_and_connection(
            i,
            gene_group,
            assigned_reaction,
            enzyme_group_glyphs,
            regulatory_connections,
        )

    return enzyme_group_glyphs, regulatory_connections


def _create_enzyme_group_glyph_and_connection(
    i: int,
    gene_group: List[Glyph],
    assigned_reaction: str,
    enzyme_group_glyphs: List[Glyph],
    regulatory_connections: List[Connection],
) -> None:
    """Create enzyme group glyph and regulatory connection for a gene group.

    Args:
        i: Group index
        gene_group: List of gene glyphs in the group
        assigned_reaction: Reaction ID assigned to this group
        enzyme_group_glyphs: List to append the new enzyme group glyph to
        regulatory_connections: List to append the new connection to
    """
    if assigned_reaction:
        # Calculate bounding box for the group
        min_x, max_x, min_y, max_y = _calculate_enzyme_group_bounds(gene_group)

        # Get enzyme labels for the tooltip
        enzyme_labels = [glyph["label"] for glyph in gene_group]
        glyph_ids = [glyph["id"] for glyph in gene_group]

        # Create enzyme group glyph
        enzyme_group_id = f"enzyme_group_{assigned_reaction}_{i}"
        enzyme_group_glyph = create_enzyme_group_glyph(
            id=enzyme_group_id,
            label=f"Enzymes for {assigned_reaction}",
            x=(min_x + max_x) / 2,
            y=(min_y + max_y) / 2,
            reaction_id=assigned_reaction,
            enzymes=enzyme_labels,
            glyph_ids=glyph_ids,
            width=max_x - min_x,
            height=max_y - min_y,
            outline_width=2,
            outline_dash=[3, 3],  # Dashed line
        )

        enzyme_group_glyphs.append(enzyme_group_glyph)

        # Create REGULATORY connection from enzyme group to reaction
        regulatory_connection = create_metabolic_connection(
            id=f"regulatory_{enzyme_group_id}_{assigned_reaction}",
            source=enzyme_group_id,
            target=assigned_reaction,
            connection_class=ConnectionClass.REGULATORY,
        )
        regulatory_connections.append(regulatory_connection)


def _match_glyph_to_svg_subtype(
    glyph: Glyph, metabolite_subtypes: Dict[str, str]
) -> str:
    """Match a metabolite glyph to its subtype.

    Args:
        glyph: Metabolite glyph to match
        metabolite_subtypes: Dictionary of metabolite ID to subtype mappings

    Returns:
        Subtype string ("main", "intermediate", etc.)
    """
    label = glyph["label"]

    # Try to match by label
    if label in metabolite_subtypes:
        return metabolite_subtypes[label]
    else:
        # Default to "main" if not found in subtype data
        return "main"


def _classify_metabolite_types(
    glyphs: List[Glyph], metabolite_subtypes: Dict[str, str]
) -> List[Glyph]:
    """Differentiate metabolite subtypes (main vs. intermediate) based on subtype data.

    This function enhances metabolite glyphs by adding subtype information
    from the metabolite subtypes mapping. Main metabolites are represented by
    ellipses and intermediate metabolites by circles in the original SVG.

    Args:
        glyphs: List of glyphs to enhance with subtype information
        metabolite_subtypes: Dictionary mapping metabolite IDs to subtypes

    Returns:
        Enhanced list of glyphs with metabolite subtype information

    Note:
        - Metabolites not found in subtypes data are classified as "main" by default
        - ID translation from MAM format to SBGN format is handled automatically
        - Both glyph ID and label are checked for matches
    """
    validate_glyphs(glyphs)

    # Update the glyphs with subtype information
    for glyph in glyphs:
        if glyph["glyph_class"] == GlyphClass.METABOLITE:
            glyph["subtype"] = _match_glyph_to_svg_subtype(glyph, metabolite_subtypes)

    return glyphs

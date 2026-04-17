"""Compartmentalized graph generation for metabolic visualizations.

This module provides functionality to create compartmentalized graph visualizations
where reactions are grouped by cellular compartments and arranged in a structured
layout with proper spacing and positioning.
"""

import uuid
from typing import Any, Dict, List, Tuple

from ..data_structures.basic_factories import create_compartment_glyph
from ..data_structures.constants import (
    DefaultDimensions,
    DefaultEnzymeGroupLayout,
    DefaultLayout,
)
from ..data_structures.core_types import Connection, ConnectionClass, Glyph, GlyphClass
from .graph_conversion_utils import convert_raw_data_to_domain_objects


COMPARTMENT_DISPLAY_ORDER = [
    "Cytosol",
    "Extracellular",
    "Mitochondria",
    "Endoplasmic reticulum",
    "Golgi apparatus",
    "Peroxisome",
    "Lysosome",
    "Nucleus",
]


COMPARTMENT_DISPLAY_ALIASES = {
    "c": "Cytosol",
    "cytosol": "Cytosol",
    "e": "Extracellular",
    "extracellular": "Extracellular",
    "m": "Mitochondria",
    "mitochondria": "Mitochondria",
    "r": "Endoplasmic reticulum",
    "er": "Endoplasmic reticulum",
    "endoplasmic reticulum": "Endoplasmic reticulum",
    "endoplasmic_reticulum": "Endoplasmic reticulum",
    "g": "Golgi apparatus",
    "golgi": "Golgi apparatus",
    "golgi apparatus": "Golgi apparatus",
    "golgi_apparatus": "Golgi apparatus",
    "x": "Peroxisome",
    "peroxisome": "Peroxisome",
    "l": "Lysosome",
    "lysosome": "Lysosome",
    "n": "Nucleus",
    "nucleus": "Nucleus",
}


def _normalize_compartment_display_name(compartment: str) -> str:
    """Normalize compartment IDs and aliases to canonical display labels."""
    if compartment is None:
        return ""

    normalized = str(compartment).strip()
    if not normalized:
        return ""

    lookup_key = " ".join(
        normalized.replace("_", " ").replace("-", " ").lower().split()
    )
    return COMPARTMENT_DISPLAY_ALIASES.get(lookup_key, normalized)


def create_compartmentalized_graph_from_model(
    element_name, model, layout_engine: str = "dot"
) -> Tuple[List[Glyph], List[Connection]]:
    """Create visualization data for metabolic reactions with compartments.

    This function groups reactions by their cellular compartments and creates
    separate graphs for each compartment, then arranges them in a layout with
    proper spacing and positioning.

    Args:
        model: MetabolicModel instance containing reaction metadata
        reaction_ids: List of reaction IDs to visualize
        layout_engine: GraphViz layout algorithm to use

    Returns:
        Tuple containing:
            - glyphs: List of all glyphs from all compartments
            - connections: List of all connections from all compartments

    Note:
        Compartments are arranged in a specific order: Cytosol, Extracellular,
        Mitochondria, Endoplasmic reticulum, Golgi apparatus, Peroxisome,
        Lysosome, Nucleus. They are laid out in rows with automatic wrapping.
    """
    # Get all reactions in the subsystem
    reaction_ids = model.get_reactions_for_subsystem(element_name)

    # Group reactions by compartment
    compartment_reactions = _group_reactions_by_compartment(model, reaction_ids)

    # Generate graph for each compartment
    compartment_graphs = {}
    for compartment, reactions in compartment_reactions.items():
        glyphs, connections, width, height = _create_single_compartment_graph(
            model, reactions, layout_engine
        )
        compartment_graphs[compartment] = {
            "glyphs": glyphs,
            "connections": connections,
            "width": width,
            "height": height,
        }

    # Arrange compartments in the final layout
    final_glyphs, final_connections = _arrange_compartments_in_layout(
        compartment_graphs
    )

    return final_glyphs, final_connections


def _group_reactions_by_compartment(
    model, reaction_ids: List[str]
) -> Dict[str, List[str]]:
    """Group reactions by their cellular compartments.

    Args:
        model: MetabolicModel instance containing reaction metadata
        reaction_ids: List of reaction IDs to group

    Returns:
        Dictionary mapping compartment names to lists of reaction IDs

    Example:
        >>> compartments = _group_reactions_by_compartment(model, ["R1", "R2"])
        >>> compartments
        {"Cytosol": ["R1"], "Mitochondria": ["R2"]}
    """
    compartment_reactions = {}

    for reaction_id in reaction_ids:
        meta = model.get_reaction_metadata(reaction_id)
        if meta and "compartment" in meta:
            compartment = _normalize_compartment_display_name(meta["compartment"])
            if not compartment:
                continue
            if compartment not in compartment_reactions:
                compartment_reactions[compartment] = []
            compartment_reactions[compartment].append(reaction_id)

    return compartment_reactions


def _arrange_compartments_in_layout(
    compartment_graphs: Dict[str, Dict[str, Any]],
) -> Tuple[List[Glyph], List[Connection]]:
    """Arrange compartments in a layout with proper spacing and positioning.

    This function positions compartments in rows with automatic wrapping,
    following a predefined order and applying consistent spacing.

    Args:
        compartment_graphs: Dictionary mapping compartment names to graph data

    Returns:
        Tuple containing:
            - final_glyphs: List of all positioned glyphs from all compartments
            - final_connections: List of all connections from all compartments

    Note:
        Compartments are arranged in order: Cytosol, Extracellular, Mitochondria,
        Endoplasmic reticulum, Golgi apparatus, Peroxisome, Lysosome, Nucleus.
    """
    final_glyphs = []
    final_connections = []

    # Layout configuration
    padding = DefaultLayout.COMPARTMENT_PADDING  # Fixed padding between compartments
    current_x = 0  # Starting X position
    current_y = 0  # Starting Y Position
    row_max_height = 0  # Track the tallest compartment in the current row
    row_start_x = 0  # X position to start each row
    # Maximum number of compartments per row
    max_rows = DefaultLayout.MAX_COMPARTMENTS_PER_ROW

    # Calculate width threshold for row wrapping
    remaining_compartments = [
        compartment for compartment in COMPARTMENT_DISPLAY_ORDER if compartment in compartment_graphs
    ]
    remaining_compartments.extend(
        sorted(
            compartment
            for compartment in compartment_graphs
            if compartment not in COMPARTMENT_DISPLAY_ORDER
        )
    )
    total_width = sum(compartment_graphs[c]["width"] for c in remaining_compartments)
    width_threshold = (
        total_width / max_rows if max_rows > 0 and total_width > 0 else float("inf")
    )

    # Position each compartment according to the defined order
    for compartment in remaining_compartments:
        graph_data = compartment_graphs[compartment]

        # Check if we need to start a new row
        if (
            current_x > row_start_x
            and current_x + graph_data["width"] > width_threshold
        ):
            # Move to the next row
            current_x = row_start_x
            current_y += row_max_height + padding
            row_max_height = 0

        shift_x = current_x
        shift_y = current_y

        # Apply position shifts to all glyphs in this compartment
        for glyph in graph_data["glyphs"]:
            glyph["x"] += shift_x
            glyph["y"] += shift_y
            if glyph["glyph_class"] == GlyphClass.COMPARTMENT:
                glyph["label"] = compartment

        # Update position for the next compartment
        current_x += graph_data["width"] + padding

        # Update the maximum height in this row
        row_max_height = max(row_max_height, graph_data["height"])

        # Collect glyphs and connections from this compartment
        final_glyphs.extend(graph_data["glyphs"])
        final_connections.extend(graph_data["connections"])

    return final_glyphs, final_connections


def _create_single_compartment_graph(model, reactions, layout_engine="dot"):
    """Create a graph for reactions within a single compartment.

    This function follows the new pipeline: raw data → layout → domain objects

    Args:
        model: MetabolicModel instance containing reaction metadata
        reactions: List of reaction IDs in the compartment
        layout_engine: GraphViz layout algorithm to use

    Returns:
        tuple: (glyphs, connections, width, height) where width and height
                represent the bounding box of the compartment contents
    """
    # Step 1: Generate raw data structures
    raw_glyphs, raw_connections = _create_raw_reactions_data(model, reactions)

    if not raw_glyphs:
        return [], [], 0, 0

    # Step 2-4: Convert raw data to domain objects through layout calculation
    glyphs, connections = convert_raw_data_to_domain_objects(
        raw_glyphs, raw_connections, layout_engine
    )

    # Step 5: Calculate bounding box and add compartment boundary
    if glyphs:
        compartment_glyph = create_compartment_boundary(glyphs)
        glyphs.insert(0, compartment_glyph)
        width = compartment_glyph["width"]
        height = compartment_glyph["height"]
    else:
        width = height = 0

    return glyphs, connections, width, height


def create_compartment_boundary(glyphs: List[Glyph]) -> Glyph:
    """Calculate bounding box and create compartment glyph.

    Args:
        glyphs: List of glyphs to calculate bounding box for

    Returns:
        Compartment glyph dictionary with bounding box information
    """
    if not glyphs:
        return create_compartment_glyph(
            id=f"compartment_{uuid.uuid4().hex[:8]}",
            label="",
            x=0.0,
            y=0.0,
            width=0.0,
            height=0.0,
        )

    # Calculate bounding box
    all_x = [g["x"] for g in glyphs]
    all_y = [g["y"] for g in glyphs]

    min_x = min(all_x)
    max_x = max(all_x)
    min_y = min(all_y)
    max_y = max(all_y)
    # Add padding
    width = max_x - min_x + DefaultLayout.COMPARTMENT_BOUNDARY_WIDTH_PADDING
    # Add padding
    height = max_y - min_y + DefaultLayout.COMPARTMENT_BOUNDARY_HEIGHT_PADDING

    # Create compartment glyph that surrounds all elements
    # Additional padding for the compartment
    compartment_padding = DefaultLayout.COMPARTMENT_BOUNDARY_PADDING
    compartment_glyph = create_compartment_glyph(
        id=f"compartment_{uuid.uuid4().hex[:8]}",
        label="",  # Will be set by the calling function
        x=(min_x + max_x) / 2,  # Center point
        y=(min_y + max_y) / 2,
        width=width + compartment_padding * 2,  # Extra padding
        height=height + compartment_padding * 2,
    )

    return compartment_glyph


# ============================================================================
# Raw data generation functions (new pipeline approach)
# ============================================================================
def _create_raw_reactions_data(
    model, reactions: List[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create raw reactions data using a clear separation of concerns.

    This function follows a two-phase approach:
    1. Extract all data from reactions
    2. Create glyphs by type using specialized functions

    Args:
        model: MetabolicModel instance containing reaction metadata
        reactions: List of reaction IDs

    Returns:
        Tuple of (raw_glyphs, raw_connections) where:
            - raw_glyphs: List of raw glyph dictionaries
            - raw_connections: List of raw connection dictionaries
    """
    # Phase 1: Extract all data from reactions
    extracted_data = _extract_reaction_data(model, reactions)

    # Phase 2: Create glyphs by type
    raw_glyphs = []
    raw_connections = []

    # Create reaction glyphs
    _create_raw_reaction_glyphs(extracted_data, raw_glyphs)

    # Create metabolite glyphs and connections
    _create_raw_metabolite_glyphs(extracted_data, raw_glyphs, raw_connections)

    # Create enzyme groups and regulatory connections
    _create_raw_enzyme_group_glyphs(extracted_data, raw_glyphs, raw_connections)

    # Create gene glyphs
    _create_raw_gene_glyphs(extracted_data, raw_glyphs)

    return raw_glyphs, raw_connections


# ============================================================================
# Generalized pipeline functions
# ============================================================================
def _extract_reaction_data(model, reactions: List[str]) -> Dict[str, Any]:
    """Extract all data from reactions using modern entity-based approach.

    This function leverages pre-calculated relationships in the modern MetabolicModel
    to avoid redundant parsing and improve performance.

    Args:
        model: MetabolicModel instance with structured entities
        reactions: List of reaction IDs to process

    Returns:
        Dictionary containing:
            - reaction_data: Dict mapping reaction_id -> {node_id}
            - gene_data: Dict mapping reaction_node_id -> [gene_list]
            - metabolite_data: Dict mapping metabolite -> {type, node_id, reactions}
            - node_ids: Dict mapping original_ids -> node_ids
    """
    reaction_data = {}
    gene_data = {}
    metabolite_data = {}
    node_ids = {}

    for reaction_id in reactions:
        # Try to get reaction entity first (Modern approach)
        reaction = model.get_reaction(reaction_id)

        # Create unique ID for the reaction
        if reaction_id not in node_ids:
            node_ids[reaction_id] = f"r_{reaction_id}"
        reaction_node_id = node_ids[reaction_id]

        reaction_data[reaction_id] = {
            "node_id": reaction_node_id,
        }

        # Extract genes using pre-calculated relationships (Modern approach)
        if reaction.associated_genes:
            # Use pre-parsed gene list instead of manual GPR parsing
            genes = [
                gene_id for gene_id in reaction.associated_genes if gene_id.strip()
            ]
            if genes:
                gene_data[reaction_node_id] = genes

        # Extract metabolites using pre-calculated reactants/products (Modern approach)
        metabolite_roles = [
            (reaction.reactants, "reactant"),
            (reaction.products, "product"),
        ]

        for metabolite_list, role in metabolite_roles:
            for metabolite_id in metabolite_list:
                if not metabolite_id.strip():
                    continue

                # Get metabolite entity to check if it's intermediate
                metabolite = model.get_metabolite(metabolite_id)
                if not metabolite:
                    # Skip if metabolite entity not found
                    continue

                # Use pre-calculated intermediate status
                is_intermediate = (
                    metabolite_id
                    in model.get_general_intermediate_metabolites(format="met_id")
                )

                if is_intermediate:
                    # Intermediate metabolites are reaction-specific
                    metabolite_node_id = f"m_{reaction_id}_{metabolite_id}"
                else:
                    # Main metabolites are shared
                    if metabolite_id not in node_ids:
                        node_ids[metabolite_id] = f"m_{uuid.uuid4().hex[:8]}"
                    metabolite_node_id = node_ids[metabolite_id]

                # Store metabolite data
                if metabolite_id not in metabolite_data:
                    metabolite_data[metabolite_id] = {
                        "original_id": metabolite_id,
                        "node_id": metabolite_node_id,
                        "type": "intermediate" if is_intermediate else "main",
                        "reactions": [],
                    }

                # Add reaction participation
                metabolite_data[metabolite_id]["reactions"].append(
                    {
                        "reaction_id": reaction_id,
                        "reaction_node_id": reaction_node_id,
                        "role": role,
                    }
                )
    return {
        "reaction_data": reaction_data,
        "gene_data": gene_data,
        "metabolite_data": metabolite_data,
        "node_ids": node_ids,
    }


def _create_raw_reaction_glyphs(
    extracted_data: Dict[str, Any],
    raw_glyphs: List[Dict[str, Any]],
) -> None:
    """Create raw reaction glyphs from extracted data.

    Args:
        extracted_data: Data extracted from _extract_reaction_data()
        raw_glyphs: List to append reaction glyph data to
    """
    reaction_data = extracted_data["reaction_data"]

    for reaction_id, data in reaction_data.items():
        raw_reaction_glyph = {
            "id": data["node_id"],
            "glyph_type": "reaction",
            "original_id": reaction_id,
            "label": reaction_id,
            "width": DefaultDimensions.REACTION_WIDTH,
            "height": DefaultDimensions.REACTION_HEIGHT,
            "x": 0.0,
            "y": 0.0,
        }
        raw_glyphs.append(raw_reaction_glyph)


def _create_raw_metabolite_glyphs(
    extracted_data: Dict[str, Any],
    raw_glyphs: List[Dict[str, Any]],
    raw_connections: List[Dict[str, Any]],
) -> None:
    """Create raw metabolite glyphs and metabolic connections from extracted data.

    Args:
        extracted_data: Data extracted from _extract_reaction_data()
        raw_glyphs: List to append metabolite glyph data to
        raw_connections: List to append metabolic connection data to
    """
    metabolite_data = extracted_data["metabolite_data"]

    # Create metabolite glyphs
    for metabolite_id, data in metabolite_data.items():
        metabolite_type = data["type"]

        if metabolite_type == "intermediate":
            # Create one intermediate metabolite per reaction
            for reaction_info in data["reactions"]:
                reaction_id = reaction_info["reaction_id"]
                metabolite_node_id = f"m_{reaction_id}_{metabolite_id}"

                # Check if we already created this intermediate metabolite
                if any(g["id"] == metabolite_node_id for g in raw_glyphs):
                    continue

                raw_metabolite_glyph = {
                    "id": metabolite_node_id,
                    "glyph_type": "metabolite",
                    "original_id": metabolite_id,
                    "label": metabolite_id,
                    "width": DefaultDimensions.METABOLITE_INTERMEDIATE_WIDTH,
                    "height": DefaultDimensions.METABOLITE_INTERMEDIATE_HEIGHT,
                    "subtype": "intermediate",
                    "associated_reaction": reaction_id,
                    "x": 0.0,
                    "y": 0.0,
                }
                raw_glyphs.append(raw_metabolite_glyph)
        else:
            # Main metabolite - create one shared node
            if not any(g["id"] == data["node_id"] for g in raw_glyphs):
                raw_metabolite_glyph = {
                    "id": data["node_id"],
                    "glyph_type": "metabolite",
                    "original_id": metabolite_id,
                    "label": metabolite_id,
                    "width": DefaultDimensions.METABOLITE_MAIN_WIDTH,
                    "height": DefaultDimensions.METABOLITE_MAIN_HEIGHT,
                    "subtype": "main",
                    "x": 0.0,
                    "y": 0.0,
                }
                raw_glyphs.append(raw_metabolite_glyph)

    # Create metabolic connections
    for metabolite_id, data in metabolite_data.items():
        for reaction_info in data["reactions"]:
            reaction_node_id = reaction_info["reaction_node_id"]
            role = reaction_info["role"]

            # Determine metabolite node ID
            if data["type"] == "intermediate":
                reaction_id = reaction_info["reaction_id"]
                metabolite_node_id = f"m_{reaction_id}_{metabolite_id}"
            else:
                metabolite_node_id = data["node_id"]

            # Create connection based on role
            if role == "reactant":
                # Metabolite → Reaction
                connection = {
                    "id": f"conn_{metabolite_node_id}_to_{reaction_node_id}",
                    "connection_type": "metabolic",
                    "source": metabolite_node_id,
                    "target": reaction_node_id,
                    "connection_class": ConnectionClass.METABOLIC,
                }
            else:  # role == "product"
                # Reaction → Metabolite
                connection = {
                    "id": f"conn_{reaction_node_id}_to_{metabolite_node_id}",
                    "connection_type": "metabolic",
                    "source": reaction_node_id,
                    "target": metabolite_node_id,
                    "connection_class": ConnectionClass.METABOLIC,
                }
            raw_connections.append(connection)


def _create_raw_enzyme_group_glyphs(
    extracted_data: Dict[str, Any],
    raw_glyphs: List[Dict[str, Any]],
    raw_connections: List[Dict[str, Any]],
) -> None:
    """Create raw enzyme group glyphs and regulatory connections from extracted data.

    Args:
        extracted_data: Data extracted from _extract_reaction_data()
        raw_glyphs: List to append enzyme group glyph data to
        raw_connections: List to append regulatory connection data to
    """
    gene_data = extracted_data["gene_data"]

    for reaction_node_id, genes in gene_data.items():
        enzyme_group_id = f"eg_{reaction_node_id}"
        num_genes = len(genes)

        # Determine layout of genes within the group (rows and columns)
        if num_genes < DefaultEnzymeGroupLayout.SINGLE_COLUMN_THRESHOLD:
            cols = 1
            rows = num_genes
        else:
            cols = int(num_genes**0.5)  # Square root for balanced grid
            rows = (num_genes + cols - 1) // cols  # Ceiling division

        # Size enzyme group to fit all genes with some padding
        gene_width = DefaultDimensions.GENE_WIDTH
        gene_height = DefaultDimensions.GENE_HEIGHT
        padding = DefaultEnzymeGroupLayout.GENE_PADDING
        border_padding = DefaultEnzymeGroupLayout.BORDER_PADDING

        # Calculate group size with both padding types
        group_width = 2 * border_padding + cols * gene_width + (cols - 1) * padding
        group_height = 2 * border_padding + rows * gene_height + (rows - 1) * padding

        # Create raw enzyme group glyph
        raw_enzyme_group = {
            "id": enzyme_group_id,
            "glyph_type": "enzyme_group",
            "label": f"Enzymes for {reaction_node_id.split('_', 1)[1]}",
            "genes": genes,
            "width": group_width,
            "height": group_height,
            "num_rows": rows,
            "num_cols": cols,
            "associated_reaction": reaction_node_id,
            "x": 0.0,
            "y": 0.0,
        }
        raw_glyphs.append(raw_enzyme_group)

        # Create regulatory connection from enzyme group to reaction
        regulatory_connection = {
            "id": f"regulatory_{enzyme_group_id}_{reaction_node_id}",
            "connection_type": "regulatory",
            "source": enzyme_group_id,
            "target": reaction_node_id,
            "connection_class": ConnectionClass.REGULATORY,
        }
        raw_connections.append(regulatory_connection)


def _create_raw_gene_glyphs(
    extracted_data: Dict[str, Any],
    raw_glyphs: List[Dict[str, Any]],
) -> None:
    """Create raw gene glyphs from extracted data.

    Args:
        extracted_data: Data extracted from _extract_reaction_data()
        raw_glyphs: List to append gene glyph data to
    """
    gene_data = extracted_data["gene_data"]

    for reaction_node_id, genes in gene_data.items():
        enzyme_group_id = f"eg_{reaction_node_id}"

        # Create raw gene glyphs (positions will be calculated later)
        for i, gene in enumerate(genes):
            gene_node_id = f"g_{reaction_node_id}_{gene}"
            raw_gene_glyph = {
                "id": gene_node_id,
                "glyph_type": "gene",
                "label": gene,
                "associated_reaction": reaction_node_id,
                "enzyme_group": enzyme_group_id,
                "gene_index": i,  # For positioning within enzyme group
                "width": DefaultDimensions.GENE_WIDTH,
                "height": DefaultDimensions.GENE_HEIGHT,
                "x": 0.0,
                "y": 0.0,
            }
            raw_glyphs.append(raw_gene_glyph)


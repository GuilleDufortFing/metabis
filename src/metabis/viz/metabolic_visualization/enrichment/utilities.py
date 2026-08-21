"""Utility functions for graph enrichment.

This module contains utility functions for geometric calculations, data preparation,
alpha/highlighting calculations, and other helper functions used throughout the
enrichment process.
"""

from math import atan2, cos, sin, sqrt

import pandas as pd

from ..generation.data_structures.constants import VisualizationConstants
from ..generation.data_structures.core_types import GlyphClass

# ============================================================================
# Geometric Calculations
# ============================================================================


def calculate_arrowhead_vertices(
    x0, y0, x1, y1, head_size=VisualizationConstants.ARROWHEAD_SIZE
):
    """Calculate arrowhead triangle vertices.

    Args:
        x0: Start x coordinate of the line
        y0: Start y coordinate of the line
        x1: End x coordinate of the line
        y1: End y coordinate of the line
        head_size: Size of the arrowhead

    Returns:
        Tuple of (x_coordinates, y_coordinates) for arrowhead triangle
    """
    # Calculate angle of the line
    angle = atan2(y1 - y0, x1 - x0)

    # Arrowhead angles (30 degrees on each side)
    head_angle = VisualizationConstants.ARROWHEAD_ANGLE_RADIANS

    # Calculate arrowhead vertices
    x_left = x1 - head_size * cos(angle - head_angle)
    y_left = y1 - head_size * sin(angle - head_angle)
    x_right = x1 - head_size * cos(angle + head_angle)
    y_right = y1 - head_size * sin(angle + head_angle)

    return [x1, x_left, x_right], [y1, y_left, y_right]


def get_connection_coordinates(
    source_node, target_node, offset=VisualizationConstants.CONNECTION_OFFSET
):
    """Calculate the start and end coordinates for a connection.

    Args:
        source_node: Source glyph dictionary with x, y coordinates
        target_node: Target glyph dictionary with x, y coordinates
        offset: Offset distance from node centers

    Returns:
        Tuple of ((x0, y0), (x1, y1)) for connection start and end points
    """
    x0, y0 = source_node["x"], source_node["y"]
    x1, y1 = target_node["x"], target_node["y"]

    dx = x1 - x0
    dy = y1 - y0
    dist = sqrt(dx * dx + dy * dy)

    if dist > 0:
        new_x0 = x0 + (dx / dist) * offset
        new_y0 = y0 + (dy / dist) * offset
        new_x1 = x1 - (dx / dist) * offset
        new_y1 = y1 - (dy / dist) * offset
    else:
        new_x0, new_y0 = x0, y0
        new_x1, new_y1 = x1, y1

    return (new_x0, new_y0), (new_x1, new_y1)


def calculate_influence_arrow_coords(reaction_glyph, gene_glyph):
    """Calculate the coordinates for an influence arrow.

    Args:
        reaction_glyph: Reaction glyph dictionary
        gene_glyph: Gene glyph dictionary

    Returns:
        Tuple of (x0, y0, x1, y1) for arrow coordinates
    """
    x0, y0 = gene_glyph["x"], gene_glyph["y"]
    x1, y1 = reaction_glyph["x"], reaction_glyph["y"]
    dx, dy = x1 - x0, y1 - y0
    dist = sqrt(dx * dx + dy * dy)

    # Calculate offsets
    offset_reaction = reaction_glyph.get("size", 40) / 3
    offset_gene = gene_glyph.get("height", 30) / 3

    if dist > offset_reaction + offset_gene:
        new_x0 = x0 + (dx / dist) * offset_gene
        new_y0 = y0 + (dy / dist) * offset_gene
        new_x1 = x1 - (dx / dist) * offset_reaction
        new_y1 = y1 - (dy / dist) * offset_reaction
    else:
        new_x0, new_y0 = x0, y0
        new_x1, new_y1 = x1, y1

    return (new_x0, new_y0), (new_x1, new_y1)


# ============================================================================
# Data Preparation
# ============================================================================


def prepare_translations(glyphs, data_manager):
    """Pre-compute identifier translations for batch operations.

    Args:
        glyphs: List of glyph dictionaries
        data_manager: AnalysisWorkspace instance with experimental data

    Returns:
        Dict containing metabolite, gene, and reaction translations
    """
    translations = {
        "metabolites": {},
        "genes": {},
        "reactions": {},
    }

    # Batch translate metabolites
    metabolite_ids = [
        g["label"] for g in glyphs if g["glyph_class"] == GlyphClass.METABOLITE
    ]
    for met_id in metabolite_ids:
        if data_manager and data_manager.metabolic_model:
            translated = data_manager.metabolic_model.translate_metabolite(
                met_id,
                output_format=data_manager.output_metabolite_format,
            )
            translations["metabolites"][met_id] = (
                translated
                if not pd.isna(translated) and translated != "Nan"
                else met_id
            )
        else:
            # Fallback: use original ID when no data manager
            translations["metabolites"][met_id] = met_id

    # Batch translate genes
    gene_ids = [g["label"] for g in glyphs if g["glyph_class"] == GlyphClass.GENE]
    for gene_id in gene_ids:
        if data_manager and data_manager.metabolic_model:
            gene_translation = data_manager.metabolic_model.translate_gene(
                gene_id,
                input_format="ensmbl_id",
                output_format=data_manager.output_gene_format,
            )
            translations["genes"][gene_id] = gene_translation
        else:
            # Fallback: use original ID when no data manager
            translations["genes"][gene_id] = gene_id

    # Batch translate reactions
    reaction_ids = [
        g["label"] for g in glyphs if g["glyph_class"] == GlyphClass.REACTION
    ]
    for reaction_id in reaction_ids:
        if data_manager and data_manager.metabolic_model:
            translations["reactions"][reaction_id] = (
                data_manager.metabolic_model.translate_reaction(
                    reaction_id,
                    input_format="mar_id",
                    output_format=data_manager.output_reaction_format,
                )
            )
        else:
            # Fallback: use original ID when no data manager
            translations["reactions"][reaction_id] = reaction_id

    return translations


# ============================================================================
# Alpha and Highlighting Calculations
# ============================================================================


def calculate_node_alpha(
    glyph_id,
    glyph_class,
    highlight_gene_influence,
    highlight_significant_subgraph,
    influential_genes_set,
    significant_nodes_set,
    high_alpha,
    low_alpha,
    is_missing_data=False,
):
    """Calculate node alpha value based on highlighting rules.

    Args:
        glyph_id: ID of the glyph
        glyph_class: Type of glyph (from GlyphClass enum)
        highlight_gene_influence: Whether gene influence highlighting is active
        highlight_significant_subgraph: Whether significance highlighting is active
        influential_genes_set: Set of influential gene IDs
        significant_nodes_set: Set of significant node IDs
        high_alpha: Alpha value for highlighted elements
        low_alpha: Alpha value for non-highlighted elements
        is_missing_data: Whether this element has missing data

    Returns:
        float: Alpha value to use for this glyph
    """
    if is_missing_data:
        return low_alpha

    # Compartments always get high alpha
    if glyph_class == GlyphClass.COMPARTMENT:
        return high_alpha

    if glyph_class == GlyphClass.SUBSYSTEM_CATEGORY:
        return VisualizationConstants.DEFAULT_SUBSYSTEM_CATEGORY_ALPHA

    # Gene influence highlighting takes precedence for genes
    if highlight_gene_influence and glyph_class == GlyphClass.GENE:
        return high_alpha if glyph_id in influential_genes_set else low_alpha

    # Significant subgraph highlighting
    if highlight_significant_subgraph:
        return high_alpha if glyph_id in significant_nodes_set else low_alpha

    return high_alpha


def get_edge_alpha(
    source_id,
    target_id,
    highlight_significant_subgraph,
    significant_nodes,
    high_alpha,
    low_alpha,
    is_missing_connection,
):
    """Determine the alpha value for an edge.

    Args:
        source_id: ID of source node
        target_id: ID of target node
        highlight_significant_subgraph: Whether significance highlighting is active
        significant_nodes: Set of significant node IDs
        high_alpha: Alpha value for highlighted elements
        low_alpha: Alpha value for non-highlighted elements
        is_missing_connection: Whether this is a missing data connection

    Returns:
        float: Alpha value to use for this edge
    """
    if is_missing_connection:
        return low_alpha

    if highlight_significant_subgraph and significant_nodes is not None:
        if source_id not in significant_nodes or target_id not in significant_nodes:
            return low_alpha

    return high_alpha


# ============================================================================
# Simple Helper Functions
# ============================================================================


def get_connection_color(
    source_node, target_node, data_manager, color_metric, is_missing_connection=False
):
    """Determine the color for a connection using AnalysisWorkspace API.

    Args:
        source_node: Source glyph dictionary
        target_node: Target glyph dictionary
        data_manager: AnalysisWorkspace instance for accessing experimental data
        color_metric: The metric to use for coloring (e.g., 'log2_fc')

    Returns:
        str: Hex color code for the connection
    """
    # Check if this is a connection between a metabolite and a reaction

    if is_missing_connection:
        # If this is a missing connection, return gray color
        return VisualizationConstants.MISSING_CONNECTION_COLOR

    if (
        source_node.get("glyph_class") == GlyphClass.METABOLITE
        and target_node.get("glyph_class") == GlyphClass.REACTION
    ):
        reaction_id = target_node["label"]
    elif (
        source_node.get("glyph_class") == GlyphClass.REACTION
        and target_node.get("glyph_class") == GlyphClass.METABOLITE
    ):
        reaction_id = source_node["label"]
    else:
        # Not a metabolic connection (metabolite ↔ reaction)
        return "gray"

    # Use AnalysisWorkspace API for consistent coloring
    if data_manager:
        color = data_manager.get_element_color(
            reaction_id, GlyphClass.REACTION, color_metric, use_main_direction=True
        )
        # Fallback to default color if get_element_color returns None or invalid color
        if not color or color == "white":
            return VisualizationConstants.DEFAULT_REACTION_COLOR
        return color
    else:
        return VisualizationConstants.DEFAULT_REACTION_COLOR


def should_reverse_direction(source_node, target_node, display_direction):
    """Whether a metabolite/reaction edge is drawn against model stoichiometry.

    A reversible reaction is drawn in one direction only, and this decides which:
    the backward direction runs the model's arrows the other way, so the
    stoichiometric alignment is inverted.

    ``display_direction`` is resolved once per reaction by
    :func:`~metabis.managers.element_presentation.resolve_display_direction`, the
    same answer the node colour and the hover are built from. It used to be
    re-derived here from the group's raw activity averages, and that private copy
    of the rule disagreed with the rest of the graph two ways (WP10/D-69):

    * It was keyed on the *forward* column, so a reaction Compass measured only in
      reverse -- its forward direction blocked, and dropped in preprocessing --
      fell through to "no reversal" and drew its arrows forward while its own
      hover read "Reverse (more active direction)". 210 base reactions across 25
      subsystems on the bundled Th17 data alone.
    * It ignored the Target/Baseline criterion, always reading the first compared
      group.

    Args:
        source_node: Source glyph dictionary
        target_node: Target glyph dictionary
        display_direction: "forward", "backward", or None when unresolved

    Returns:
        bool: True if direction should be reversed
    """
    if display_direction != "backward":
        return False

    # Only metabolite/reaction edges carry stoichiometric direction; enzyme and
    # subsystem edges are undirected and must never be flipped.
    is_metabolite_to_reaction = (
        source_node.get("glyph_class") == GlyphClass.METABOLITE
        and target_node.get("glyph_class") == GlyphClass.REACTION
    )
    is_reaction_to_metabolite = (
        source_node.get("glyph_class") == GlyphClass.REACTION
        and target_node.get("glyph_class") == GlyphClass.METABOLITE
    )
    return is_metabolite_to_reaction or is_reaction_to_metabolite

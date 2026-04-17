"""Auxiliary visualization functions for metabolic visualization.

This module contains functions for creating auxiliary visualization data
such as influence circles, enzyme-reaction connections, and other
specialized visualization elements.
"""

from ..generation.data_structures.constants import VisualizationConstants
from ..generation.data_structures.core_types import GlyphClass
from .connection_enrichment import process_influence_connections

# Public API
__all__ = [
    "create_auxiliary_visualization_data",
    "create_influence_circle_data",
    "create_enzyme_reaction_connections",
]


# ============================================================================
# Auxiliary Visualization Data Creation
# ============================================================================


def create_auxiliary_visualization_data(
    enriched_glyphs,
    influence_connections_data,
    highly_influenced_reaction_colors,
    highlight_significant_subgraph,
    significant_nodes_set,
    data_manager,
):
    """Create auxiliary data structures for specialized visualizations.

    Args:
        enriched_glyphs: List of enriched glyph dictionaries
        influence_connections_data: Dictionary mapping reaction IDs to gene influences
        highly_influenced_reaction_colors: Dictionary mapping reaction IDs to colors
        highlight_significant_subgraph: Whether significant subgraph highlighting is enabled
        significant_nodes_set: Set of significant node IDs
        data_manager: DataManager instance for accessing experimental data

    Returns:
        dict: Dictionary containing auxiliary visualization data
    """
    auxiliary_data = {}

    # Create influence line data
    auxiliary_data["influence_lines"] = process_influence_connections(
        influence_connections_data,
        glyph_dict={g["id"]: g for g in enriched_glyphs},
        data_manager=data_manager,
    )

    # Create influence circle data
    reaction_glyphs = [
        g for g in enriched_glyphs if g["glyph_class"] == GlyphClass.REACTION
    ]
    auxiliary_data["influence_circles"] = create_influence_circle_data(
        reaction_glyphs, highly_influenced_reaction_colors
    )

    # Create enzyme-reaction connection data
    auxiliary_data["enzyme_reaction_connections"] = create_enzyme_reaction_connections(
        enriched_glyphs, highlight_significant_subgraph, significant_nodes_set
    )

    return auxiliary_data


def create_influence_circle_data(reaction_glyphs, highly_influenced_reaction_colors):
    """Create influence circle data from enriched reaction glyphs.

    Args:
        reaction_glyphs: List of reaction glyph dictionaries
        highly_influenced_reaction_colors: Dictionary mapping reaction IDs to colors

    Returns:
        dict: Dictionary containing influence circle data for batch rendering
    """
    influence_circle_data = {"x": [], "y": [], "color": [], "alpha": [], "radius": []}

    for reaction_glyph in reaction_glyphs:
        reaction_id = reaction_glyph["id"]
        if reaction_id in highly_influenced_reaction_colors:
            influence_circle_data["x"].append(reaction_glyph["x"])
            influence_circle_data["y"].append(reaction_glyph["y"])
            influence_circle_data["color"].append(
                highly_influenced_reaction_colors[reaction_id]
            )
            influence_circle_data["alpha"].append(0.3)
            influence_circle_data["radius"].append(reaction_glyph.get("size", 40) * 1.5)

    return influence_circle_data


def create_enzyme_reaction_connections(
    enriched_glyphs, highlight_significant_subgraph, significant_nodes_set
):
    """Create enzyme-reaction connection data from enriched glyphs.

    Args:
        enriched_glyphs: List of enriched glyph dictionaries
        highlight_significant_subgraph: Whether significant subgraph highlighting is enabled
        significant_nodes_set: Set of significant node IDs

    Returns:
        dict: Dictionary containing enzyme-reaction connection data for batch rendering
    """
    connection_data = {"x0": [], "y0": [], "x1": [], "y1": [], "line_alpha": []}

    # Group glyphs by type for efficiency
    gene_glyphs = [g for g in enriched_glyphs if g["glyph_class"] == GlyphClass.GENE]
    reaction_dict = {
        g["label"]: g
        for g in enriched_glyphs
        if g["glyph_class"] == GlyphClass.REACTION
    }

    for gene_glyph in gene_glyphs:
        associated_reaction = gene_glyph.get("associated_reaction")
        if associated_reaction and associated_reaction in reaction_dict:
            reaction_glyph = reaction_dict[associated_reaction]

            # Calculate alpha for this connection
            alpha = VisualizationConstants.HIGH_ALPHA
            if highlight_significant_subgraph:
                if (
                    gene_glyph["id"] not in significant_nodes_set
                    and reaction_glyph["id"] not in significant_nodes_set
                ):
                    alpha = VisualizationConstants.LOW_ALPHA

            connection_data["x0"].append(gene_glyph["x"])
            connection_data["y0"].append(gene_glyph["y"])
            connection_data["x1"].append(reaction_glyph["x"])
            connection_data["y1"].append(reaction_glyph["y"])
            connection_data["line_alpha"].append(alpha)

    return connection_data

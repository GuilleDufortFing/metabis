"""Connection enrichment functions for metabolic visualization.

This module contains functions for enriching connections with visualization
properties and handling different types of connections (metabolic, regulatory,
gene influence).
"""

from ..generation.data_structures.constants import VisualizationConstants
from ..generation.data_structures.core_types import GlyphClass
from ..generation.data_structures.enriched_factories import enrich_metabolic_connection
from .utilities import (
    calculate_arrowhead_vertices,
    calculate_influence_arrow_coords,
    get_connection_color,
    get_connection_coordinates,
    get_edge_alpha,
    should_reverse_direction,
)

# ============================================================================
# Connection Enrichment Functions
# ============================================================================


def enrich_connections(
    connections,
    enriched_glyphs,
    highlight_significant_subgraph,
    significant_nodes_set,
    missing_data_elements_set,
    data_manager=None,
    color_metric="log2_fc",
):
    """Enrich connections with visualization fields.

    Args:
        connections: List of basic connection dictionaries
        enriched_glyphs: List of enriched glyph dictionaries
        highlight_significant_subgraph: Whether significant subgraph highlighting
                                         is enabled
        significant_nodes_set: Set of significant node IDs
        missing_data_elements_set: Set of missing data element IDs
        data_manager: AnalysisWorkspace instance for accessing experimental data
        color_metric: The metric to use for coloring (e.g., 'log2_fc')

    Returns:
        list: List of enriched connection dictionaries
    """
    enriched_connections = []
    glyph_dict = {g["id"]: g for g in enriched_glyphs}

    for connection in connections:
        source_glyph = glyph_dict.get(connection["source"])
        target_glyph = glyph_dict.get(connection["target"])

        if not source_glyph or not target_glyph:
            continue

        # Check if this connection involves missing data elements or missing reactions
        # Two cases:
        # 1. Standard missing data check (both endpoints marked as missing)
        # 2. Connection involves a missing reaction (even if other endpoint is active)
        is_missing_connection = (
            source_glyph["id"] in missing_data_elements_set
            or target_glyph["id"] in missing_data_elements_set
            or is_missing_reaction_connection(
                source_glyph, target_glyph, missing_data_elements_set, glyph_dict
            )
        )

        # Calculate alpha for highlighting
        edge_alpha = get_edge_alpha(
            connection["source"],
            connection["target"],
            highlight_significant_subgraph,
            significant_nodes_set,
            VisualizationConstants.HIGH_ALPHA,
            VisualizationConstants.LOW_ALPHA,
            is_missing_connection,
        )

        # Calculate color using AnalysisWorkspace API for consistency
        color = get_connection_color(
            source_glyph,
            target_glyph,
            data_manager,
            color_metric,
            is_missing_connection,
        )

        # Handle different connection types
        connection_class = connection.get("connection_class", "metabolic")

        if connection_class == "subsystem":
            # Subsystem connections: alpha proportional to shared metabolites
            # Calculate geometry
            (x0, y0), (x1, y1) = get_connection_coordinates(source_glyph, target_glyph)

            alpha = _calculate_subsystem_connection_alpha(connection, connections)

            enriched_connection = {
                **connection,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                # A shared metabolite is a link, not a measurement: one neutral
                # grey, stronger for more shared metabolites.
                "color": VisualizationConstants.GLOBAL_BORDER_COLOR,
                "line_alpha": alpha,
                "line_width": VisualizationConstants.SUBSYSTEM_CONNECTION_WIDTH,
                "line_dash": "solid",
                "connection_class": "subsystem",
                # Propagate missing flag for consistency (not used for subsystem rendering)
                "is_missing_connection": is_missing_connection,
            }
        elif connection_class == "regulatory":
            # Regulatory connections: dashed lines, enzyme group color, no arrowheads
            # Calculate geometry
            (x0, y0), (x1, y1) = get_connection_coordinates(source_glyph, target_glyph)

            if not is_missing_connection:
                color = VisualizationConstants.ENZYME_GROUP_COLOR

            enriched_connection = {
                **connection,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "color": color,
                "line_alpha": edge_alpha,
                "line_width": VisualizationConstants.ENZYME_CONNECTION_WIDTH,
                "line_dash": "dashed",
                "connection_class": "regulatory",
                # Propagate missing flag for downstream rendering logic if needed
                "is_missing_connection": is_missing_connection,
            }
        else:
            # Metabolic connections: with arrowheads and reaction-based coloring

            # Calculate geometry
            (x0, y0), (x1, y1) = get_connection_coordinates(source_glyph, target_glyph)

            # Align direction with metabolic model stoichiometry
            should_swap = align_connection_direction(
                source_glyph, target_glyph, data_manager
            )
            if should_swap:
                # Swap coordinates to reverse arrow direction
                x0, y0, x1, y1 = x1, y1, x0, y0

            # Calculate arrowhead coordinates
            arrowhead_xs, arrowhead_ys = calculate_arrowhead_vertices(x0, y0, x1, y1)

            # Determine line dash pattern for missing data
            line_dash = (
                VisualizationConstants.MISSING_DATA_LINE_DASH
                if is_missing_connection
                else "solid"
            )

            # Create enriched connection
            enriched_connection = enrich_metabolic_connection(
                connection,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                color=color,
                line_alpha=edge_alpha,
                line_dash=line_dash,
                arrowhead_xs=arrowhead_xs,
                arrowhead_ys=arrowhead_ys,
                fill_color=color,
                fill_alpha=edge_alpha,
            )
            # Mark missing to control glow rendering downstream
            enriched_connection["is_missing_connection"] = is_missing_connection

        enriched_connections.append(enriched_connection)

    return enriched_connections


def process_influence_connections(
    influence_connections_data,
    glyph_dict=None,
    data_manager=None,
):
    """Process gene influence connections using batch rendering.

    Args:
        influence_connections_data: Dictionary mapping reaction IDs to gene influences
        glyph_dict: Dictionary mapping glyph IDs to glyph data
        data_manager: AnalysisWorkspace instance for color calculations

    Returns:
        dict: Dictionary containing influence line data for batch rendering
    """
    edge_alpha = VisualizationConstants.INFLUENCE_EDGE_ALPHA
    max_pct_variance = 100.0

    # Collect all influence connection data
    influence_line_data = {
        "x0": [],
        "y0": [],
        "x1": [],
        "y1": [],
        "line_color": [],
        "line_alpha": [],
        "line_width": [],
    }

    for reaction_glyph_id, gene_influences in influence_connections_data.items():
        if reaction_glyph_id not in glyph_dict:
            continue
        reaction_glyph = glyph_dict[reaction_glyph_id]

        for gene_glyph_id, pct_variance, gene_react_corr in gene_influences:
            if gene_glyph_id not in glyph_dict:
                continue
            gene_glyph = glyph_dict[gene_glyph_id]

            # Get color using generic AnalysisWorkspace method
            if data_manager:
                color = data_manager.get_influence_color(
                    correlation=gene_react_corr,
                    influence=pct_variance,
                    max_influence=max_pct_variance,
                )
            else:
                color = "gray"

            # Calculate coordinates
            start_coords, end_coords = calculate_influence_arrow_coords(
                reaction_glyph, gene_glyph
            )

            # Add to batch data
            influence_line_data["x0"].append(start_coords[0])
            influence_line_data["y0"].append(start_coords[1])
            influence_line_data["x1"].append(end_coords[0])
            influence_line_data["y1"].append(end_coords[1])
            influence_line_data["line_color"].append(color)
            influence_line_data["line_alpha"].append(edge_alpha)
            influence_line_data["line_width"].append(7)

    return influence_line_data


# Public API
__all__ = [
    "enrich_connections",
    "process_influence_connections",
    "is_missing_reaction_connection",
    "align_connection_direction",
]


# ============================================================================
# Helper Functions
# ============================================================================


def _calculate_subsystem_connection_alpha(connection, all_connections):
    """Calculate alpha for subsystem connections based on shared metabolites.

    Args:
        connection: The subsystem connection to calculate alpha for
        all_connections: All connections to determine min/max weights

    Returns:
        float: Alpha value between min and max subsystem connection alpha
    """
    # Get weight (number of shared metabolites) from this connection
    weight = connection.get("weight", 1)

    # Find min/max weights among all subsystem connections
    subsystem_weights = []
    for conn in all_connections:
        if conn.get("connection_class") == "subsystem":
            subsystem_weights.append(conn.get("weight", 1))

    if not subsystem_weights:
        return VisualizationConstants.SUBSYSTEM_CONNECTION_MIN_ALPHA

    min_weight = min(subsystem_weights)
    max_weight = max(subsystem_weights)

    # Avoid division by zero
    if max_weight == min_weight:
        return VisualizationConstants.SUBSYSTEM_CONNECTION_MAX_ALPHA

    # Normalize weight to [0, 1] range
    normalized_weight = (weight - min_weight) / (max_weight - min_weight)

    # Scale to alpha range
    alpha_range = (
        VisualizationConstants.SUBSYSTEM_CONNECTION_MAX_ALPHA
        - VisualizationConstants.SUBSYSTEM_CONNECTION_MIN_ALPHA
    )
    alpha = VisualizationConstants.SUBSYSTEM_CONNECTION_MIN_ALPHA + (
        normalized_weight * alpha_range
    )

    return alpha


def is_missing_reaction_connection(
    source_glyph, target_glyph, missing_data_set, glyph_dict
):
    """Check if a connection involves a missing reaction.

    This is used to ensure connections to/from missing reactions get dashed lines,
    even if the other endpoint (e.g., metabolite) is not marked as missing.

    Args:
        source_glyph: Source glyph dictionary
        target_glyph: Target glyph dictionary
        missing_data_set: Set of glyph IDs marked as missing
        glyph_dict: Dictionary mapping glyph IDs to glyph data

    Returns:
        bool: True if either endpoint is a missing reaction
    """
    source_is_missing_reaction = (
        source_glyph["glyph_class"] == GlyphClass.REACTION
        and source_glyph["id"] in missing_data_set
    )
    target_is_missing_reaction = (
        target_glyph["glyph_class"] == GlyphClass.REACTION
        and target_glyph["id"] in missing_data_set
    )

    return source_is_missing_reaction or target_is_missing_reaction


def align_connection_direction(source_glyph, target_glyph, data_manager):
    """Determine if connection direction should be swapped based on metabolic model.

    This function aligns connection arrows with the underlying metabolic model's
    stoichiometry. Arrows should point:
    - FROM metabolite TO reaction (if metabolite is a reactant)
    - FROM reaction TO metabolite (if metabolite is a product)

    Args:
        source_glyph: Source glyph dictionary with 'glyph_class' and 'label'
        target_glyph: Target glyph dictionary with 'glyph_class' and 'label'
        data_manager: AnalysisWorkspace with metabolic_model access

    Returns:
        bool: True if coordinates should be swapped (reversed), False otherwise

    Logic:
        - If metabolite is a REACTANT: arrow should go metabolite → reaction
          - Current direction metabolite → reaction: no swap (correct)
          - Current direction reaction → metabolite: swap (incorrect)
        - If metabolite is a PRODUCT: arrow should go reaction → metabolite
          - Current direction metabolite → reaction: swap (incorrect)
          - Current direction reaction → metabolite: no swap (correct)
    """
    # Check if this is a metabolite-reaction connection
    source_is_metabolite = source_glyph.get("glyph_class") == GlyphClass.METABOLITE
    source_is_reaction = source_glyph.get("glyph_class") == GlyphClass.REACTION
    target_is_metabolite = target_glyph.get("glyph_class") == GlyphClass.METABOLITE
    target_is_reaction = target_glyph.get("glyph_class") == GlyphClass.REACTION

    # Only process metabolite-reaction connections
    if not (
        (source_is_metabolite and target_is_reaction)
        or (source_is_reaction and target_is_metabolite)
    ):
        return False  # Not a metabolite-reaction connection, no swap

    # Identify which is reaction and which is metabolite
    if source_is_reaction and target_is_metabolite:
        reaction_glyph = source_glyph
        metabolite_glyph = target_glyph
        current_direction_is_reaction_to_metabolite = True
    else:  # source_is_metabolite and target_is_reaction
        reaction_glyph = target_glyph
        metabolite_glyph = source_glyph
        current_direction_is_reaction_to_metabolite = False

    # Get reaction and metabolite IDs from labels
    reaction_id = reaction_glyph["label"]
    metabolite_id = metabolite_glyph["label"]

    # Check if data_manager has metabolic_model
    if not data_manager or not hasattr(data_manager, "metabolic_model"):
        return False  # Can't align without model, keep original direction

    # Query the metabolic model for reactants and products
    try:
        metabolites_info = data_manager.metabolic_model.get_metabolites_for_reaction(
            reaction_id
        )
        reactants = metabolites_info.get("reactants", [])
        products = metabolites_info.get("products", [])

        # Check if metabolite is a reactant or product
        is_reactant = metabolite_id in reactants
        is_product = metabolite_id in products

        should_swap = False

        # Determine if swap is needed based on stoichiometry
        if is_reactant:
            # Reactant → Reaction (arrow should point TO reaction)
            # If current direction is reaction → metabolite, we need to swap
            should_swap = current_direction_is_reaction_to_metabolite
        elif is_product:  # is_product
            # Reaction → Product (arrow should point FROM reaction)
            # If current direction is metabolite → reaction, we need to swap
            should_swap = not current_direction_is_reaction_to_metabolite

        # A reversible reaction is drawn in one direction only, and which one is a
        # single decision owned by the shared reaction-direction map -- the same
        # answer the node colour and the hover are resolved from. Asking for it
        # here, rather than re-deriving it from group averages, is what keeps the
        # arrowhead from contradicting the hover it sits under (WP10/D-69).
        if should_reverse_direction(
            source_glyph,
            target_glyph,
            data_manager.get_display_direction(reaction_id),
        ):
            should_swap = not should_swap

        return should_swap

    except Exception:
        # If anything goes wrong, keep original direction
        return False

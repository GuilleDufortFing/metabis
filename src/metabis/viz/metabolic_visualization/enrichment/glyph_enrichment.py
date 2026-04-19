"""Glyph enrichment functions for metabolic visualization.

This module contains functions for enriching different types of glyphs
(metabolites, genes, reactions, compartments, enzyme groups) with
visualization properties like colors, alphas, hover text, and styling.
"""

import math

from ...click_behavior import get_external_reference_data
from ..generation.data_structures.constants import VisualizationConstants
from ..generation.data_structures.core_types import GlyphClass
from ..generation.data_structures.enriched_factories import (
    enrich_compartment_glyph,
    enrich_enzyme_group_glyph,
    enrich_gene_glyph,
    enrich_metabolite_glyph,
    enrich_reaction_glyph,
    enrich_subsystem_category_glyph,
    enrich_subsystem_glyph,
)
from .utilities import calculate_node_alpha


def _wrap_subsystem_name(name: str) -> str:
    """Wrap long subsystem names into multiple lines.

    Args:
        name: The subsystem name to wrap

    Returns:
        str: The wrapped name with newlines inserted at appropriate points
    """
    max_line_length = VisualizationConstants.SUBSYSTEM_NAME_MAX_LINE_LENGTH
    max_lines = VisualizationConstants.SUBSYSTEM_NAME_MAX_LINES

    if len(name) <= max_line_length:
        return name

    # Split into words for proper wrapping
    words = name.split()
    lines = []
    current_line = ""

    for word in words:
        # Check if adding this word would exceed line length
        test_line = current_line + " " + word if current_line else word

        if len(test_line) <= max_line_length:
            current_line = test_line
        else:
            # Start a new line
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                # Single word is too long, just add it
                lines.append(word)

        # Stop if we've reached max lines
        if len(lines) >= max_lines - 1 and current_line:
            # Truncate with ellipsis if needed
            remaining_words = words[words.index(word) :]
            remaining_text = " ".join(remaining_words)
            if len(current_line + " " + remaining_text) > max_line_length:
                # Truncate current line and add ellipsis
                # Account for " ..." space
                space_for_ellipsis = max_line_length - len(current_line) - 4
                if space_for_ellipsis > 0:
                    current_line += " ..."
                else:
                    current_line = current_line[: max_line_length - 3] + "..."
            else:
                current_line += " " + remaining_text
            break

    # Add the last line
    if current_line:
        lines.append(current_line)

    return "\n".join(lines)


# Public API
__all__ = [
    "enrich_single_glyph",
    "enrich_metabolite_glyph_with_data",
    "enrich_gene_glyph_with_data",
    "enrich_reaction_glyph_with_data",
    "enrich_compartment_glyph_with_data",
    "enrich_enzyme_group_glyph_with_data",
    "enrich_subsystem_glyph_with_data",
    "enrich_subsystem_category_glyph_with_data",
]


# ============================================================================
# Main Glyph Enrichment Function
# ============================================================================


def enrich_single_glyph(
    glyph,
    data_manager,
    translations,
    color_metric,
    delta_options,
    highlight_significant_subgraph,
    highlight_gene_influence,
    significant_nodes_set,
    influential_genes_set,
    missing_data_elements_dict,
    show_hover_stats=True,
    show_hover_distribution_plots=True,
    show_hover_color_reference=True,
    focus_instance=None,
    graph_type=None,
):
    """Enrich a single glyph with visualization fields.

    Args:
        glyph: The glyph dictionary to enrich
        data_manager: DataManager instance for accessing experimental data
        translations: Dictionary containing translation mappings for
                     different element types
        color_metric: The metric to use for coloring (e.g., 'log2_fold_change')
        highlight_significant_subgraph: Whether to highlight significant
                                        subgraph elements
        highlight_gene_influence: Whether to highlight gene influence relationships
        significant_nodes_set: Set of node IDs that are significant
        influential_genes_set: Set of gene IDs that are influential
        missing_data_elements_dict: Dictionary of element IDs that have missing data, with the reason as the value
        show_hover_stats: Whether to include textual hover statistics
        show_hover_distribution_plots: Whether to include violin/distribution hover plots
        show_hover_color_reference: Whether to include color-reference hover blocks
        focus_instance: object that retrieves the IDs of the elements being visualized
        graph_type: Type of graph being generated ("subsystem", "compartment", etc.)

    Returns:
        dict: Enriched glyph dictionary with visualization properties
    """
    glyph_class = glyph["glyph_class"]
    glyph_id = glyph["id"]

    # Check if this glyph should have missing data styling and why
    missing_reason = missing_data_elements_dict.get(glyph_id)
    is_missing_data = missing_reason is not None

    # Calculate alpha for highlighting
    alpha = calculate_node_alpha(
        glyph_id,
        glyph_class,
        highlight_gene_influence,
        highlight_significant_subgraph,
        influential_genes_set,
        significant_nodes_set,
        VisualizationConstants.HIGH_ALPHA,
        VisualizationConstants.LOW_ALPHA,
        is_missing_data,
    )

    # Enrich based on glyph type
    if glyph_class == GlyphClass.METABOLITE:
        return enrich_metabolite_glyph_with_data(
            glyph,
            alpha,
            translations["metabolites"],
            data_manager,
            color_metric,
            missing_reason,
            focus_instance,
            graph_type,
        )
    elif glyph_class == GlyphClass.GENE:
        return enrich_gene_glyph_with_data(
            glyph,
            alpha,
            translations["genes"],
            data_manager,
            color_metric,
            missing_reason,
            show_hover_stats=show_hover_stats,
            show_hover_distribution_plots=show_hover_distribution_plots,
            show_hover_color_reference=show_hover_color_reference,
        )
    elif glyph_class == GlyphClass.REACTION:
        return enrich_reaction_glyph_with_data(
            glyph,
            alpha,
            translations["reactions"],
            data_manager,
            color_metric,
            missing_reason,
            show_hover_stats=show_hover_stats,
            show_hover_distribution_plots=show_hover_distribution_plots,
            show_hover_color_reference=show_hover_color_reference,
        )
    elif glyph_class == GlyphClass.COMPARTMENT:
        return enrich_compartment_glyph_with_data(
            glyph, alpha
        )  # Compartments don't need missing data styling
    elif glyph_class == GlyphClass.ENZYME_GROUP:
        return enrich_enzyme_group_glyph_with_data(glyph, alpha, missing_reason)
    elif glyph_class == GlyphClass.SUBSYSTEM:
        return enrich_subsystem_glyph_with_data(
            glyph,
            alpha,
            translations.get("subsystems", {}),
            data_manager,
            color_metric,
            delta_options,
            missing_reason,
            show_hover_stats=show_hover_stats,
            show_hover_distribution_plots=show_hover_distribution_plots,
            show_hover_color_reference=show_hover_color_reference,
            focus_instance=focus_instance,
            graph_type=graph_type,
        )
    elif glyph_class == GlyphClass.SUBSYSTEM_CATEGORY:
        return enrich_subsystem_category_glyph_with_data(
            glyph,
            alpha,
            translations.get("subsystem_categories", {}),
            missing_reason,
            focus_instance,
            graph_type,
        )
    else:
        # Fallback for unknown glyph types
        enriched = glyph.copy()
        enriched.update(
            {
                "alpha": alpha,
                "display_label": glyph["label"],
                "color": "gray",
            }
        )
        return enriched


# ============================================================================
# Individual Glyph Type Enrichment Functions
# ============================================================================


def enrich_metabolite_glyph_with_data(
    glyph,
    alpha,
    metabolite_translations,
    data_manager=None,
    color_metric="log2_fc",
    missing_reason=None,
    focus_instance=None,
    graph_type=None,
):
    """Enrich a metabolite glyph with visualization fields.

    Args:
        glyph: The metabolite glyph dictionary
        alpha: Alpha transparency value
        metabolite_translations: Dictionary of metabolite label translations
        data_manager: DataManager instance for accessing experimental data
        color_metric: The metric to use for coloring
        missing_reason: Reason for the missing data, if any
        focus_instance: object that retrieves the IDs of the elements being visualized
        graph_type: Type of graph being generated ("subsystem", "compartment", etc.)

    Returns:
        dict: Enriched metabolite glyph
    """
    metabolite_id = glyph["label"]
    display_label = metabolite_translations.get(metabolite_id, metabolite_id)

    # Default values
    color = None
    line_dash = "solid"
    hover = f"Metabolite: {metabolite_id}"

    # Build hover via DataManager (HoverManager will adapt based on missing_reason)
    if data_manager:
        hover = data_manager.get_element_hover(
            metabolite_id,
            "metabolite",
            color_metric=color_metric,
            missing_reason=missing_reason,
            current_subsystem=focus_instance.get_ancestor_ids().get("subsystem", None),
        )
        if missing_reason in {"no_experimental_data", "no_metadata"}:
            color = VisualizationConstants.MISSING_DATA_COLOR
            line_dash = VisualizationConstants.MISSING_DATA_LINE_DASH
        else:
            color = "black" if glyph.get("subtype") == "intermediate" else "#64BDF9"
    else:
        # No data_manager available
        if missing_reason in {"no_experimental_data", "no_metadata"}:
            color = VisualizationConstants.MISSING_DATA_COLOR
            line_dash = VisualizationConstants.MISSING_DATA_LINE_DASH
        else:
            color = "black" if glyph.get("subtype") == "intermediate" else "#64BDF9"

    return enrich_metabolite_glyph(
        glyph,
        alpha=alpha,
        display_label=display_label,
        color=color,
        hover=hover,
        line_dash=line_dash,
    )


def enrich_gene_glyph_with_data(
    glyph,
    alpha,
    gene_translations,
    data_manager,
    color_metric,
    missing_reason=None,
    show_hover_stats=True,
    show_hover_distribution_plots=True,
    show_hover_color_reference=True,
    focus_instance=None,
    graph_type=None,
):
    """Enrich a gene glyph with visualization fields.

    Args:
        glyph: The gene glyph dictionary
        alpha: Alpha transparency value
        gene_translations: Dictionary of gene label translations
        data_manager: DataManager instance for accessing experimental data
        color_metric: The metric to use for coloring
        missing_reason: Reason for the missing data, if any
        focus_instance: object that retrieves the IDs of the elements being visualized
        graph_type: Type of graph being generated ("subsystem", "compartment", etc.)

    Returns:
        dict: Enriched gene glyph
    """
    gene_id = glyph["label"]
    display_label = gene_translations.get(gene_id, gene_id)
    if display_label.startswith("ENSG"):
        display_label = "ENSG..."  # Shorten long Ensembl IDs for display
    reference_data = get_external_reference_data(data_manager, gene_id, "gene")

    # Default values
    color = VisualizationConstants.DEFAULT_GENE_COLOR
    hover = f"Gene: {gene_id}"
    line_dash = "solid"  # Override with missing data styling if needed

    # Build hover via DataManager (HoverManager will adapt based on missing_reason)
    if data_manager:
        hover = data_manager.get_element_hover(
            gene_id,
            "gene",
            color_metric=color_metric,
            show_stats=show_hover_stats,
            show_plot=show_hover_distribution_plots,
            show_color_reference=show_hover_color_reference,
            missing_reason=missing_reason,
        )
        if missing_reason in {"no_experimental_data", "no_metadata"}:
            color = VisualizationConstants.MISSING_DATA_COLOR
            line_dash = VisualizationConstants.MISSING_DATA_LINE_DASH
        else:
            color = data_manager.get_element_color(gene_id, "gene", color_metric)
    else:
        # Fallback when no data manager
        if missing_reason in {"no_experimental_data", "no_metadata"}:
            color = VisualizationConstants.MISSING_DATA_COLOR
            line_dash = VisualizationConstants.MISSING_DATA_LINE_DASH
        else:
            color = VisualizationConstants.DEFAULT_GENE_COLOR
            hover = f"Gene: {gene_id}"

    # Fallback to default color if get_element_color returns None or invalid color
    if not color:
        color = VisualizationConstants.DEFAULT_GENE_COLOR

    enriched = enrich_gene_glyph(
        glyph,
        alpha=alpha,
        display_label=display_label,
        color=color,
        hover=hover,
        line_dash=line_dash,
    )
    enriched["external_url"] = reference_data["external_url"]
    return enriched


def enrich_reaction_glyph_with_data(
    glyph,
    alpha,
    reaction_translations,
    data_manager,
    color_metric,
    missing_reason=None,
    show_hover_stats=True,
    show_hover_distribution_plots=True,
    show_hover_color_reference=True,
    focus_instance=None,
    graph_type=None,
):
    """Enrich a reaction glyph with visualization fields.

    Args:
        glyph: The reaction glyph dictionary
        alpha: Alpha transparency value
        reaction_translations: Dictionary of reaction label translations
        data_manager: DataManager instance for accessing experimental data
        color_metric: The metric to use for coloring
        missing_reason: Reason for the missing data, if any
        focus_instance: object that retrieves the IDs of the elements being visualized
        graph_type: Type of graph being generated ("subsystem", "compartment", etc.)

    Returns:
        dict: Enriched reaction glyph
    """
    reaction_id = glyph["label"]
    display_label = reaction_translations.get(reaction_id, reaction_id)
    reference_data = get_external_reference_data(data_manager, reaction_id, "reaction")

    # Default values
    color = VisualizationConstants.DEFAULT_REACTION_COLOR
    hover = f"Reaction: {reaction_id}"
    line_dash = "solid"  # Override with missing data styling if needed

    if data_manager:
        hover = data_manager.get_element_hover(
            reaction_id,
            "reaction",
            use_main_direction=True,
            color_metric=color_metric,
            show_stats=show_hover_stats,
            show_plot=show_hover_distribution_plots,
            show_color_reference=show_hover_color_reference,
            missing_reason=missing_reason,
        )
        if missing_reason in {"no_experimental_data", "no_metadata"}:
            color = VisualizationConstants.MISSING_DATA_COLOR
            line_dash = VisualizationConstants.MISSING_DATA_LINE_DASH
        else:
            color = data_manager.get_element_color(
                reaction_id, "reaction", color_metric, use_main_direction=True
            )
    else:
        # Fallback when no data manager
        if missing_reason in {"no_experimental_data", "no_metadata"}:
            color = VisualizationConstants.MISSING_DATA_COLOR
            line_dash = VisualizationConstants.MISSING_DATA_LINE_DASH
        else:
            color = VisualizationConstants.DEFAULT_REACTION_COLOR
            hover = f"Reaction: {reaction_id}"

    # Fallback to default color if get_element_color returns None or invalid color
    if not color:
        color = VisualizationConstants.DEFAULT_REACTION_COLOR

    enriched = enrich_reaction_glyph(
        glyph,
        alpha=alpha,
        display_label=display_label,
        color=color,
        hover=hover,
        line_dash=line_dash,
    )
    enriched["external_url"] = reference_data["external_url"]
    return enriched


def enrich_compartment_glyph_with_data(glyph, alpha):
    """Enrich a compartment glyph with visualization fields.

    Args:
        glyph: The compartment glyph dictionary
        alpha: Alpha transparency value

    Returns:
        dict: Enriched compartment glyph
    """
    return enrich_compartment_glyph(
        glyph,
        alpha=alpha,
        display_label=glyph["label"],
    )


def enrich_enzyme_group_glyph_with_data(glyph, alpha, missing_reason=None):
    """Enrich an enzyme group glyph with visualization fields.

    Args:
        glyph: The enzyme group glyph dictionary
        alpha: Alpha transparency value
        missing_reason: Reason for the missing data, if any

    Returns:
        dict: Enriched enzyme group glyph
    """
    # Determine color and line dash based on missing status
    if missing_reason in {"no_experimental_data", "no_metadata"}:
        color = VisualizationConstants.MISSING_DATA_COLOR
        line_dash = VisualizationConstants.MISSING_DATA_LINE_DASH
    else:
        color = VisualizationConstants.ENZYME_GROUP_COLOR
        line_dash = VisualizationConstants.ENZYME_GROUP_DASH
    return enrich_enzyme_group_glyph(
        glyph,
        alpha=alpha,
        color=color,
        display_label=glyph["label"],
        line_dash=line_dash,
    )


def enrich_subsystem_glyph_with_data(
    glyph,
    alpha,
    subsystem_translations,
    data_manager,
    color_metric,
    delta_options,
    missing_reason=None,
    show_hover_stats=True,
    show_hover_distribution_plots=True,
    show_hover_color_reference=True,
    focus_instance=None,
    graph_type=None,
):
    """Enrich a subsystem glyph with experimental data and visualization properties.

    Args:
        glyph: Basic subsystem glyph dictionary
        alpha: Alpha transparency value for the glyph
        subsystem_translations: Translation data for subsystems
        data_manager: DataManager instance for accessing experimental data
        color_metric: The metric to use for coloring
        missing_reason: Reason for the missing data, if any
        focus_instance: object that retrieves the IDs of the elements being visualized
        graph_type: Type of graph being generated ("subsystem", "compartment", etc.)

    Returns:
        dict: Enriched subsystem glyph with visualization properties
    """
    subsystem_id = glyph.get("subsystem_name", glyph.get("label", "Unknown"))

    is_missing_data = missing_reason is not None

    # Get the human-readable subsystem name from the model
    display_name = subsystem_id  # fallback
    if data_manager and data_manager.metabolic_model:
        subsystem_obj = data_manager.metabolic_model.get_subsystem(subsystem_id)
        if subsystem_obj:
            display_name = subsystem_obj.name

    # Apply text wrapping to long subsystem names
    wrapped_display_name = _wrap_subsystem_name(display_name)

    # Build hover via DataManager and determine color/line_dash from missing_reason
    if data_manager:
        hover_text = data_manager.get_element_hover(
            subsystem_id,
            "subsystem",
            color_metric=color_metric,
            delta_options=delta_options,
            missing_reason=missing_reason,
            show_stats=show_hover_stats,
            show_plot=show_hover_distribution_plots,
            show_color_reference=show_hover_color_reference,
        )
        if missing_reason in {"no_experimental_data", "no_metadata"}:
            subsystem_color = VisualizationConstants.MISSING_DATA_COLOR
            line_dash = VisualizationConstants.MISSING_DATA_LINE_DASH
        else:
            subsystem_color = data_manager.get_element_color(
                subsystem_id, "subsystem", color_metric, delta_options
            )
            line_dash = "solid"
    else:
        # Fallback when no data manager
        hover_text = f"Subsystem: {display_name}"
        subsystem_color = "lightblue"
        line_dash = "solid"

    # Fallback to default color if get_element_color returns None or invalid color
    if not subsystem_color or subsystem_color == "white":
        subsystem_color = "lightblue"

    # Calculate dynamic radius based on reaction and metabolite counts
    reaction_count = glyph.get("reaction_count", 0)
    metabolite_count = glyph.get("metabolite_count", 0)

    # Combine reaction and metabolite counts for sizing
    total_elements = reaction_count + metabolite_count

    # Calculate radius with configurable scaling
    if total_elements > 0:
        # Apply scaling based on the configured method
        scaling_method = VisualizationConstants.SUBSYSTEM_SCALING_METHOD

        if scaling_method == "linear":
            # Linear scaling: direct proportion (normalized to 100 elements)
            scale = min(total_elements / 100.0, 1.0)
        elif scaling_method == "sqrt":
            # Square root scaling: moderate compression
            scale = math.sqrt(total_elements) / math.sqrt(100)
            scale = min(scale, 1.0)
        elif scaling_method == "power":
            # Power scaling with configurable exponent
            exponent = VisualizationConstants.SUBSYSTEM_POWER_EXPONENT
            scale = (total_elements**exponent) / (100**exponent)
            scale = min(scale, 1.0)
        else:  # Default to log scaling
            # Logarithmic scaling: most compression
            scale = math.log(1 + total_elements) / math.log(1 + 100)

        radius = (
            VisualizationConstants.SUBSYSTEM_BASE_RADIUS
            + scale
            * VisualizationConstants.SUBSYSTEM_SIZE_SCALE_FACTOR
            * (
                VisualizationConstants.SUBSYSTEM_MAX_RADIUS
                - VisualizationConstants.SUBSYSTEM_BASE_RADIUS
            )
        )
        # Clamp to min/max bounds
        radius = max(
            VisualizationConstants.SUBSYSTEM_MIN_RADIUS,
            min(radius, VisualizationConstants.SUBSYSTEM_MAX_RADIUS),
        )
    else:
        radius = VisualizationConstants.SUBSYSTEM_MIN_RADIUS

    # Create enriched subsystem glyph
    border_color = (
        VisualizationConstants.GLOBAL_BORDER_COLOR
        if not missing_reason
        else VisualizationConstants.MISSING_DATA_BORDER_COLOR
    )

    if focus_instance.has_element_type("subsystem"):
        if subsystem_id != focus_instance.get_ancestor_ids().get("subsystem"):
            alpha = VisualizationConstants.SUBSYSTEM_UNSELECTED_ALPHA

    enriched = enrich_subsystem_glyph(
        base_glyph=glyph,
        hover=hover_text,
        color=subsystem_color,
        alpha=alpha,
        line_color=border_color,
        # line_dash=line_dash,
        label_alpha=VisualizationConstants.SUBSYSTEM_LABEL_ALPHA_MIN,
        # border_width=1,
    )

    # Add enrichment metadata
    enriched.update(
        {
            "display_label": wrapped_display_name,
            "is_missing_data": is_missing_data,
            "enrichment_applied": True,
            "is_subsystem": True,
            "radius": radius,  # Add calculated radius for dynamic sizing
        }
    )

    return enriched


def enrich_subsystem_category_glyph_with_data(
    glyph,
    alpha,
    category_translations,
    missing_reason=None,
    focus_instance=None,
    graph_type=None,
):
    """Enrich a subsystem category glyph with visualization properties.

    Args:
        glyph: Basic subsystem category glyph dictionary
        alpha: Alpha transparency value for the glyph
        category_translations: Translation data for category names
        missing_reason: Reason for the missing data, if any
        focus_instance: object that retrieves the IDs of the elements being visualized
        graph_type: Type of graph being generated ("subsystem", "compartment", etc.)

    Returns:
        dict: Enriched subsystem category glyph with visualization properties
    """
    category_name = glyph.get("label", "Unknown Category")
    display_label = category_translations.get(category_name, category_name)
    is_missing_data = missing_reason is not None

    # Categories are visual boundaries, so they don't need data-based coloring
    # but may need missing data styling
    if missing_reason is not None:
        line_color = VisualizationConstants.MISSING_DATA_OUTLINE_COLOR
        line_dash = VisualizationConstants.MISSING_DATA_LINE_DASH
        fill_alpha = VisualizationConstants.MISSING_DATA_ALPHA
    else:
        # Use CategoryManager for dynamic color lookup based on category file order
        line_color = focus_instance.model.get_category_color(category_name)
        line_dash = VisualizationConstants.DEFAULT_CATEGORY_LINE_DASH
        fill_alpha = VisualizationConstants.DEFAULT_CATEGORY_FILL_ALPHA

    label_alpha = alpha
    if focus_instance.has_element_type("category"):
        if category_name != focus_instance.get_ancestor_ids().get("category"):
            alpha = VisualizationConstants.CATEGORY_UNSELECTED_ALPHA
            label_alpha = VisualizationConstants.TRANSPARENT_ALPHA

    # Create enriched category glyph
    enriched = enrich_subsystem_category_glyph(
        base_glyph=glyph,
        alpha=alpha,
        display_label=display_label,
        line_color=line_color,
        line_dash=line_dash,
        fill_alpha=fill_alpha,
        font_size=VisualizationConstants.FONT_SIZE_CATEGORY,
        label_alpha=label_alpha,
    )

    # Add enrichment metadata
    enriched.update(
        {
            "is_missing_data": is_missing_data,
            "enrichment_applied": True,
        }
    )

    return enriched

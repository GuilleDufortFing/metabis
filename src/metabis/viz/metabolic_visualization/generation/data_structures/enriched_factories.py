"""Enriched factory functions for creating visualization-ready graph elements.

This module provides factory functions for enriching basic glyphs and connections
with visualization-specific fields such as colors, styling, positioning, and
interaction data. These functions take basic structures created by basic_factories
and add the visual properties needed for rendering.

Usage:
    >>> from .basic_factories import create_reaction_glyph
    >>> from .enriched_factories import enrich_reaction_glyph
    >>> base_glyph = create_reaction_glyph(
    ...     "r1", "Glucose phosphorylation", 100.0, 200.0
    ... )
    >>> enriched_glyph = enrich_reaction_glyph(
    ...     base_glyph, color="blue", hover="Reaction details"
    ... )
"""

from typing import Any, Dict, List, Optional

from .constants import VisualizationConstants

# ============================================================================
# Factory Functions for Enriched Glyph Creation
# ============================================================================


def enrich_reaction_glyph(
    base_glyph: Dict[str, Any],
    alpha: float = VisualizationConstants.HIGH_ALPHA,
    display_label: Optional[str] = None,
    color: str = VisualizationConstants.DEFAULT_REACTION_COLOR,
    size: float = VisualizationConstants.REACTION_SIZE,
    angle: float = VisualizationConstants.REACTION_ANGLE,
    hover: str = "",
    line_dash: str = "solid",
) -> Dict[str, Any]:
    """Enrich a basic reaction glyph with visualization fields.

    Args:
        base_glyph: Basic reaction glyph dictionary
        alpha: Transparency value
        display_label: Translated/display label (uses base label if None)
        color: Fill color
        size: Visual size
        angle: Rotation angle in radians
        hover: Hover tooltip text
        line_dash: Border dash pattern ("solid", dashed pattern, or list)

    Returns:
        Enriched reaction glyph dictionary
    """
    enriched = base_glyph.copy()
    enriched.update(
        {
            "alpha": alpha,
            "display_label": display_label or base_glyph["label"],
            "color": color,
            "size": size,
            "angle": angle,
            "hover": hover,
            "line_dash": line_dash,
        }
    )
    return enriched


def enrich_metabolite_glyph(
    base_glyph: Dict[str, Any],
    alpha: float = VisualizationConstants.HIGH_ALPHA,
    display_label: Optional[str] = None,
    color: str = "skyblue",  # Default metabolite color
    size: Optional[float] = None,
    hover: str = "",
    line_dash: str = "solid",
) -> Dict[str, Any]:
    """Enrich a basic metabolite glyph with visualization fields.

    Args:
        base_glyph: Basic metabolite glyph dictionary
        alpha: Transparency value
        display_label: Translated/display label (uses base label if None)
        color: Fill color
        size: Visual size (auto-determined from subtype if None)
        hover: Hover tooltip text
        line_dash: Border dash pattern ("solid", dashed pattern, or list)

    Returns:
        Enriched metabolite glyph dictionary
    """
    # Auto-determine size based on subtype if not provided
    if size is None:
        size = (
            VisualizationConstants.INTERMEDIATE_METABOLITE_SIZE
            if base_glyph.get("subtype") == "intermediate"
            else VisualizationConstants.MAIN_METABOLITE_SIZE
        )

    enriched = base_glyph.copy()
    enriched.update(
        {
            "alpha": alpha,
            "display_label": display_label or base_glyph["label"],
            "color": color,
            "size": size,
            "hover": hover,
            "line_dash": line_dash,
        }
    )
    return enriched


def enrich_gene_glyph(
    base_glyph: Dict[str, Any],
    alpha: float = VisualizationConstants.HIGH_ALPHA,
    display_label: Optional[str] = None,
    color: str = VisualizationConstants.DEFAULT_GENE_COLOR,
    hover: str = "",
    line_dash: str = "solid",
) -> Dict[str, Any]:
    """Enrich a basic gene glyph with visualization fields.

    Args:
        base_glyph: Basic gene glyph dictionary
        alpha: Transparency value
        display_label: Translated/display label (uses base label if None)
        color: Fill color
        hover: Hover tooltip text
        line_dash: Border dash pattern ("solid", dashed pattern, or list)

    Returns:
        Enriched gene glyph dictionary
    """
    enriched = base_glyph.copy()
    enriched.update(
        {
            "alpha": alpha,
            "display_label": display_label or base_glyph["label"],
            "color": color,
            "hover": hover,
            "line_dash": line_dash,
        }
    )
    return enriched


def enrich_compartment_glyph(
    base_glyph: Dict[str, Any],
    alpha: float = VisualizationConstants.HIGH_ALPHA,
    display_label: Optional[str] = None,
    color: str = "white",  # Default compartment fill color
    line_color: Optional[str] = None,
) -> Dict[str, Any]:
    """Enrich a basic compartment glyph with visualization fields.

    Args:
        base_glyph: Basic compartment glyph dictionary
        alpha: Transparency value
        display_label: Translated/display label (uses base label if None)
        color: Fill color
        line_color: Border color (auto-determined if None)

    Returns:
        Enriched compartment glyph dictionary
    """
    # Auto-determine line color if not provided
    if line_color is None:
        line_color = (
            VisualizationConstants.CYTOSOL_LINE_COLOR
            if base_glyph["label"] == "Cytosol"
            else VisualizationConstants.DEFAULT_LINE_COLOR
        )

    # Calculate label positions
    x_label = base_glyph["x"] - base_glyph["width"] / 2 + 100
    y_label = base_glyph["y"] - base_glyph["height"] / 2 + 100

    enriched = base_glyph.copy()
    enriched.update(
        {
            "alpha": alpha,
            "display_label": display_label or base_glyph["label"],
            "color": color,
            "line_color": line_color,
            "x_label": x_label,
            "y_label": y_label,
        }
    )
    return enriched


def enrich_subsystem_glyph(
    base_glyph: Dict[str, Any],
    alpha: float = VisualizationConstants.HIGH_ALPHA,
    display_label: Optional[str] = None,
    label_alpha: float = VisualizationConstants.SUBSYSTEM_LABEL_ALPHA_MIN,
    color: str = VisualizationConstants.DEFAULT_SUBSYSTEM_COLOR,
    line_color: str = VisualizationConstants.DEFAULT_LINE_COLOR,
    hover: str = "",
    size_scale: float = 1.0,
) -> Dict[str, Any]:
    """Enrich a basic subsystem glyph with visualization fields.

    Args:
        base_glyph: Basic subsystem glyph dictionary
        alpha: Transparency value
        display_label: Translated/display label (uses base label if None)
        color: Fill color
        line_color: Border color
        hover: Hover tooltip text
        size_scale: Scale factor for size adjustment

    Returns:
        Enriched subsystem glyph dictionary
    """
    enriched = base_glyph.copy()
    enriched.update(
        {
            "alpha": alpha,
            "display_label": display_label or base_glyph["label"],
            "label_alpha": label_alpha,
            "color": color,
            "line_color": line_color,
            "hover": hover,
            "size_scale": size_scale,
        }
    )
    return enriched


def enrich_enzyme_group_glyph(
    base_glyph: Dict[str, Any],
    alpha: float = VisualizationConstants.HIGH_ALPHA,
    display_label: Optional[str] = None,
    color: str = "transparent",  # Enzyme groups are typically transparent
    enzymes_display: Optional[str] = None,
    line_dash: str = "solid",
) -> Dict[str, Any]:
    """Enrich a basic enzyme group glyph with visualization fields.

    Args:
        base_glyph: Basic enzyme group glyph dictionary
        alpha: Transparency value
        display_label: Translated/display label (uses base label if None)
        color: Fill color
        enzymes_display: Comma-separated enzyme list (auto-generated if None)
        line_dash: Border dash pattern ("solid", dashed pattern, or list)

    Returns:
        Enriched enzyme group glyph dictionary
    """
    # Auto-generate enzymes display string if not provided
    if enzymes_display is None:
        enzymes_display = ", ".join(base_glyph.get("enzymes", []))

    enriched = base_glyph.copy()
    enriched.update(
        {
            "alpha": alpha,
            "display_label": display_label or base_glyph["label"],
            "color": color,
            "enzymes_display": enzymes_display,
            "line_dash": line_dash,
        }
    )
    return enriched


def enrich_subsystem_category_glyph(
    base_glyph: Dict[str, Any],
    alpha: float = VisualizationConstants.CATEGORY_LABEL_ALPHA,
    display_label: Optional[str] = None,
    color: str = VisualizationConstants.DEFAULT_CATEGORY_COLOR,
    line_color: Optional[str] = None,  # Will use category-specific color if None
    text_color: Optional[str] = None,  # Will use category-specific color if None
    line_width: float = VisualizationConstants.DEFAULT_CATEGORY_LINE_WIDTH,
    line_dash: str = VisualizationConstants.DEFAULT_CATEGORY_LINE_DASH,
    fill_alpha: float = VisualizationConstants.DEFAULT_CATEGORY_FILL_ALPHA,
    font_size: str = VisualizationConstants.FONT_SIZE_CATEGORY,
    label_alpha: float = VisualizationConstants.CATEGORY_LABEL_ALPHA,
) -> Dict[str, Any]:
    """Enrich a basic subsystem category glyph with visualization fields.

    Args:
        base_glyph: Basic subsystem category glyph dictionary
        alpha: Transparency value (for text/labels)
        display_label: Translated/display label (uses base label if None)
        color: Fill color (typically transparent for boundaries)
        line_color: Border color for the circular boundary (uses category color if None)
        text_color: Text color for category labels (uses category color if None)
        line_width: Border thickness
        line_dash: Border dash pattern ("solid", "dashed", etc.)
        fill_alpha: Fill transparency (usually 0 for transparent boundaries)
        font_size: Font size for category labels
        label_alpha: Alpha transparency for labels

    Returns:
        Enriched subsystem category glyph dictionary
    """
    # Use provided colors or fall back to default line color
    # (caller should provide the correct color from CategoryManager)
    final_line_color = line_color or VisualizationConstants.DEFAULT_CATEGORY_LINE_COLOR
    final_text_color = text_color or final_line_color

    enriched = base_glyph.copy()
    enriched.update(
        {
            "alpha": alpha,
            "display_label": display_label or base_glyph["label"],
            "color": color,
            "line_color": final_line_color,
            "text_color": final_text_color,
            "line_width": line_width,
            "line_dash": line_dash,
            "fill_alpha": fill_alpha,
            "font_size": font_size,
            "label_alpha": label_alpha,
        }
    )
    return enriched


# ============================================================================
# Factory Functions for Enriched Connection Creation
# ============================================================================


def enrich_metabolic_connection(
    base_connection: Dict[str, Any],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str = VisualizationConstants.DEFAULT_LINE_COLOR,
    line_alpha: float = VisualizationConstants.HIGH_ALPHA,
    line_width: float = VisualizationConstants.DEFAULT_LINE_WIDTH,
    line_dash: str = "solid",
    arrowhead_xs: Optional[List[float]] = None,
    arrowhead_ys: Optional[List[float]] = None,
    fill_color: Optional[str] = None,
    fill_alpha: float = VisualizationConstants.HIGH_ALPHA,
    arrowhead_line_color: str = "lightgray",
    arrowhead_line_alpha: float = VisualizationConstants.HIGH_ALPHA,
) -> Dict[str, Any]:
    """Enrich a basic metabolic connection with visualization fields.

    Args:
        base_connection: Basic connection dictionary
        x0: Start X coordinate
        y0: Start Y coordinate
        x1: End X coordinate
        y1: End Y coordinate
        color: Line color
        line_alpha: Line transparency
        line_width: Line thickness
        line_dash: Line dash pattern ("solid", dashed pattern, or list)
        arrowhead_xs: Arrowhead X coordinates
        arrowhead_ys: Arrowhead Y coordinates
        fill_color: Arrowhead fill color (defaults to line color)
        fill_alpha: Arrowhead fill transparency
        arrowhead_line_color: Arrowhead border color
        arrowhead_line_alpha: Arrowhead border transparency

    Returns:
        Enriched metabolic connection dictionary
    """
    enriched = base_connection.copy()
    enriched.update(
        {
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "color": color,
            "line_alpha": line_alpha,
            "line_width": line_width,
            "line_dash": line_dash,
            "arrowhead_xs": arrowhead_xs or [],
            "arrowhead_ys": arrowhead_ys or [],
            "fill_color": fill_color or color,
            "fill_alpha": fill_alpha,
            "arrowhead_line_color": arrowhead_line_color,
            "arrowhead_line_alpha": arrowhead_line_alpha,
        }
    )
    return enriched


def enrich_influence_connection(
    base_connection: Dict[str, Any],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str = VisualizationConstants.DEFAULT_LINE_COLOR,
    line_alpha: float = VisualizationConstants.HIGH_ALPHA,
    line_width: float = VisualizationConstants.INFLUENCE_LINE_WIDTH,
    line_dash: str = "solid",
) -> Dict[str, Any]:
    """Enrich a basic connection as an influence connection.

    Args:
        base_connection: Basic connection dictionary
        x0: Start X coordinate
        y0: Start Y coordinate
        x1: End X coordinate
        y1: End Y coordinate
        color: Line color
        line_alpha: Line transparency
        line_width: Line thickness
        line_dash: Line dash pattern ("solid", dashed pattern, or list)

    Returns:
        Enriched influence connection dictionary
    """
    enriched = base_connection.copy()
    enriched.update(
        {
            "connection_class": "influence",
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "color": color,
            "line_alpha": line_alpha,
            "line_width": line_width,
            "line_dash": line_dash,
        }
    )
    return enriched


def enrich_enzyme_reaction_connection(
    base_connection: Dict[str, Any],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str = VisualizationConstants.ENZYME_GROUP_COLOR,
    line_alpha: float = VisualizationConstants.HIGH_ALPHA,
    line_width: float = VisualizationConstants.ENZYME_CONNECTION_WIDTH,
    line_dash: str = "solid",
) -> Dict[str, Any]:
    """Enrich a basic connection as an enzyme-reaction connection.

    Args:
        base_connection: Basic connection dictionary
        x0: Start X coordinate
        y0: Start Y coordinate
        x1: End X coordinate
        y1: End Y coordinate
        color: Line color
        line_alpha: Line transparency
        line_width: Line thickness
        line_dash: Line dash pattern ("solid", dashed pattern, or list)

    Returns:
        Enriched enzyme-reaction connection dictionary
    """
    enriched = base_connection.copy()
    enriched.update(
        {
            "connection_class": "enzyme_reaction",
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "color": color,
            "line_alpha": line_alpha,
            "line_width": line_width,
            "line_dash": line_dash,
        }
    )
    return enriched

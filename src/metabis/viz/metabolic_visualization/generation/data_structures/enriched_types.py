# """Enriched data structures with visualization fields.

# This module contains enriched versions of the base data structures that include
# visualization-specific fields such as colors, styling, positioning, and
# interaction data. These structures are used by the visualization pipeline
# to render the final graph.

# Usage:
#     >>> from .enriched_types import EnrichedReactionGlyph, EnrichedMetabolicConnection
#     >>> # These types are typically created using enriched factory functions
#     >>> from .enriched_factories import enrich_reaction_glyph
# """

# from typing import List, Union

# from .core_types import (
#     BaseConnection,
#     BaseGlyph,
#     CompartmentGlyph,
#     EnzymeGroupGlyph,
#     GeneGlyph,
#     MetabolicConnection,
#     MetaboliteGlyph,
#     ReactionGlyph,
#     SubsystemCategoryGlyph,
#     SubsystemGlyph,
# )

# # ============================================================================
# # Enriched Glyph Structures with Visualization Fields
# # ============================================================================


# class EnrichedBaseGlyph(BaseGlyph):
#     """Base structure for enriched glyphs with visualization fields."""

#     alpha: float  # Transparency for highlighting/dimming
#     display_label: str  # Translated/display name for visualization
#     color: str  # Fill color (hex format or named color)


# class EnrichedReactionGlyph(EnrichedBaseGlyph, ReactionGlyph):
#     """Enriched reaction glyph with visualization fields."""

#     size: float  # Visual size (typically 40)
#     angle: float  # Rotation angle in radians (typically 0.785398 for 45°)
#     hover: str  # Hover tooltip text


# class EnrichedMetaboliteGlyph(EnrichedBaseGlyph, MetaboliteGlyph):
#     """Enriched metabolite glyph with visualization fields."""

#     size: float  # Visual size (15 for intermediate, 20 for main)


# class EnrichedGeneGlyph(EnrichedBaseGlyph, GeneGlyph):
#     """Enriched gene glyph with visualization fields."""

#     hover: str  # Hover tooltip text


# class EnrichedCompartmentGlyph(EnrichedBaseGlyph, CompartmentGlyph):
#     """Enriched compartment glyph with visualization fields."""

#     line_color: str  # Border color (white for Cytosol, black otherwise)
#     x_label: float  # Label X position
#     y_label: float  # Label Y position


# class EnrichedEnzymeGroupGlyph(EnrichedBaseGlyph, EnzymeGroupGlyph):
#     """Enriched enzyme group glyph with visualization fields."""

#     enzymes_display: str  # Comma-separated enzyme list for display


# class EnrichedSubsystemGlyph(EnrichedBaseGlyph, SubsystemGlyph):
#     """Enriched subsystem glyph with visualization fields."""

#     hover: str  # Hover tooltip text
#     line_color: str  # Border color
#     size_scale: float  # Scaling factor for sizing based on content
#     label_alpha: float  # Label transparency (usually very low or 0 for boundaries)


# class EnrichedSubsystemCategoryGlyph(EnrichedBaseGlyph, SubsystemCategoryGlyph):
#     """Enriched subsystem category glyph with visualization fields."""

#     line_color: str  # Border color for the circular boundary
#     line_width: float  # Border thickness
#     line_dash: str  # Border dash pattern ("solid", "dashed", etc.)
#     fill_alpha: float  # Fill transparency (usually very low or 0 for boundaries)


# # ============================================================================
# # Enriched Connection Structures with Visualization Fields
# # ============================================================================


# class EnrichedBaseConnection(BaseConnection):
#     """Base structure for enriched connections with visualization fields."""

#     # Line geometry
#     x0: float  # Start X coordinate
#     y0: float  # Start Y coordinate
#     x1: float  # End X coordinate
#     y1: float  # End Y coordinate

#     # Styling
#     color: str  # Line color
#     line_alpha: float  # Line transparency
#     line_width: float  # Line thickness


# class EnrichedMetabolicConnection(EnrichedBaseConnection, MetabolicConnection):
#     """Enriched metabolic connection with arrowhead visualization fields."""

#     # Arrowhead geometry (for triangular arrowheads)
#     arrowhead_xs: List[float]  # Arrowhead X coordinates
#     arrowhead_ys: List[float]  # Arrowhead Y coordinates

#     # Arrowhead styling
#     fill_color: str  # Arrowhead fill color
#     fill_alpha: float  # Arrowhead fill transparency
#     arrowhead_line_color: str  # Arrowhead border color
#     arrowhead_line_alpha: float  # Arrowhead border transparency


# class InfluenceConnection(EnrichedBaseConnection):
#     """Connection representing gene influence relationships."""

#     # Influence connections may have different styling
#     line_width: float  # Typically thinner than metabolic connections


# class EnzymeReactionConnection(EnrichedBaseConnection):
#     """Connection between enzymes and reactions."""

#     # These typically have specific styling (e.g., #F8C471 color)
#     pass


# # ============================================================================
# # Union Types for Enriched Structures
# # ============================================================================


# # Union type for all enriched glyph types
# EnrichedGlyph = Union[
#     EnrichedReactionGlyph,
#     EnrichedMetaboliteGlyph,
#     EnrichedGeneGlyph,
#     EnrichedEnzymeGroupGlyph,
#     EnrichedCompartmentGlyph,
#     EnrichedSubsystemGlyph,
#     EnrichedSubsystemCategoryGlyph,
# ]

# # Union type for all enriched connection types
# EnrichedConnection = Union[
#     EnrichedMetabolicConnection,
#     InfluenceConnection,
#     EnzymeReactionConnection,
#     EnrichedBaseConnection,
# ]

"""Core type definitions for graph visualization.

This module contains the fundamental type system for metabolic graph visualization,
including constants, base types, and specialized glyph/connection structures.

Classes:
    - GlyphClass: Standardized glyph type constants
    - ConnectionClass: Standardized connection type constants
    - MetaboliteSubtype: Metabolite classification constants
    - BaseGlyph: Foundation for all visual elements
    - BaseConnection: Foundation for all connections
    - Specialized types: ReactionGlyph, MetaboliteGlyph, etc.

Type Aliases:
    - Glyph: Union of all glyph types
    - Connection: Union of all connection types
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

# ============================================================================
# Type Literals and Constants
# ============================================================================


class GlyphClass:
    """Standardized glyph class constants."""

    REACTION = "reaction"
    METABOLITE = "metabolite"
    GENE = "gene"
    ENZYME_GROUP = "enzyme_group"
    COMPARTMENT = "compartment"
    SUBSYSTEM = "subsystem"
    SUBSYSTEM_CATEGORY = "subsystem_category"


class ConnectionClass:
    """Standardized connection class constants."""

    METABOLIC = "metabolic"
    REGULATORY = "regulatory"
    GENERIC = "generic"
    SUBSYSTEM = "subsystem"


class MetaboliteSubtype:
    """Standardized metabolite subtype constants."""

    MAIN = "main"
    INTERMEDIATE = "intermediate"


# Type literals for better type checking
GlyphClassType = Literal[
    "reaction",
    "metabolite",
    "gene",
    "enzyme_group",
    "compartment",
    "subsystem",
    "subsystem_category",
]
ConnectionClassType = Literal["metabolic", "regulatory", "generic", "subsystem"]
MetaboliteSubtypeType = Literal["main", "intermediate"]


# ============================================================================
# Base Data Structure Definitions
# ============================================================================


class BaseGlyph(TypedDict):
    """Base structure for all glyph types.

    Contains the minimum required fields that every glyph must have.
    """

    id: str  # Unique identifier
    glyph_class: GlyphClassType  # Visual/functional type
    label: str  # Human-readable name
    x: float  # X position coordinate
    y: float  # Y position coordinate
    width: float  # Width dimension
    height: float  # Height dimension


class BaseConnection(TypedDict):
    """Base structure for all connection types.

    Contains the minimum required fields that every connection must have.
    """

    id: str  # Unique identifier
    connection_class: ConnectionClassType  # Connection type/style
    source: str  # Source glyph ID
    target: str  # Target glyph ID


# ============================================================================
# Specialized Glyph Structures
# ============================================================================


class ReactionGlyph(BaseGlyph):
    """Glyph representing a biochemical reaction."""

    original_id: str  # Original reaction ID from source data


class MetaboliteGlyph(BaseGlyph):
    """Glyph representing a metabolite/compound."""

    original_id: str  # Original metabolite ID from source data
    subtype: MetaboliteSubtypeType  # Classification as main or intermediate


class GeneGlyph(BaseGlyph):
    """Glyph representing a gene/enzyme."""

    associated_reaction: str  # Related reaction ID
    enzyme_group: Optional[str]  # Parent enzyme group ID (if grouped)


class EnzymeGroupGlyph(BaseGlyph):
    """Glyph representing a group of related enzymes."""

    reaction_id: str  # Associated reaction ID
    enzymes: List[str]  # List of enzyme names/labels
    glyph_ids: List[str]  # List of contained gene glyph IDs
    outline_color: str  # Border color (hex format)
    outline_width: int  # Border thickness in pixels
    outline_dash: Union[str, List[int]]  # Dash pattern (CSS or list format)


class CompartmentGlyph(BaseGlyph):
    """Glyph representing a cellular compartment boundary."""

    # Inherits all fields from BaseGlyph, no additional fields needed
    pass


class SubsystemGlyph(BaseGlyph):
    """Glyph representing a metabolic subsystem."""

    reaction_count: int  # Number of reactions in this subsystem
    metabolite_count: int  # Number of unique metabolites in this subsystem
    subsystem_name: str  # Original subsystem name from model


class SubsystemCategoryGlyph(BaseGlyph):
    """Glyph representing a circular boundary around subsystem categories."""

    radius: float  # Radius of the circular boundary
    subsystem_count: int  # Number of subsystems within this category


# ============================================================================
# Specialized Connection Structures
# ============================================================================


class MetabolicConnection(BaseConnection):
    """Connection representing metabolic relationships."""

    stoichiometry: Optional[float]  # Reaction coefficient (optional)


class SubsystemConnection(BaseConnection):
    """Connection representing subsystem relationships through shared metabolites."""

    shared_metabolites: List[str]  # List of metabolites shared between subsystems
    reaction_count: int  # Number of reactions contributing to this connection
    weight: float  # Connection strength (e.g., number of shared metabolites)
    metabolite_flow_direction: str  # Direction of metabolite flow


# ============================================================================
# Type Aliases for Convenience
# ============================================================================


# Union type for all glyph types
Glyph = Union[
    ReactionGlyph,
    MetaboliteGlyph,
    GeneGlyph,
    EnzymeGroupGlyph,
    CompartmentGlyph,
    SubsystemGlyph,
    SubsystemCategoryGlyph,
]

# Union type for all connection types
Connection = Union[MetabolicConnection, SubsystemConnection, BaseConnection]


# ============================================================================
# Export List
# ============================================================================

__all__ = [
    # Constants
    "GlyphClass",
    "ConnectionClass",
    "MetaboliteSubtype",
    # Type literals
    "GlyphClassType",
    "ConnectionClassType",
    "MetaboliteSubtypeType",
    # Base types
    "BaseGlyph",
    "BaseConnection",
    # Specialized glyph types
    "ReactionGlyph",
    "MetaboliteGlyph",
    "GeneGlyph",
    "EnzymeGroupGlyph",
    "CompartmentGlyph",
    "SubsystemGlyph",
    "SubsystemCategoryGlyph",
    # Specialized connection types
    "MetabolicConnection",
    "SubsystemConnection",
    # Union types
    "Glyph",
    "Connection",
]

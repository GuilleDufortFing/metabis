"""Basic factory functions for creating graph glyphs and connections.

This module provides factory functions for creating basic glyphs and connections
without visualization-specific fields. These functions create the core data
structures that can later be enriched with styling and positioning information.

Usage:
    >>> from .basic_factories import create_reaction_glyph, create_metabolite_glyph
    >>> reaction = create_reaction_glyph("r1", "Glucose phosphorylation", 100.0, 200.0)
    >>> metabolite = create_metabolite_glyph("m1", "Glucose", 50.0, 100.0)
"""

from typing import Any, Dict, List, Optional, Union

from .constants import DefaultStyling
from .core_types import (
    ConnectionClass,
    GlyphClass,
    MetaboliteSubtype,
)
from .utilities import get_default_dimensions

# ============================================================================
# Factory Functions for Basic Glyph Creation
# ============================================================================


def create_reaction_glyph(
    id: str,
    label: str,
    x: float,
    y: float,
    original_id: Optional[str] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a reaction glyph with standardized structure.

    Args:
        id: Unique identifier for the glyph
        label: Human-readable name/label
        x: X position coordinate
        y: Y position coordinate
        original_id: Original reaction ID from source data
        width: Width dimension (uses default if None)
        height: Height dimension (uses default if None)

    Returns:
        Dictionary representing a reaction glyph

    Example:
        >>> glyph = create_reaction_glyph("r1", "Reaction 1", 100.0, 200.0)
        >>> assert glyph["glyph_class"] == GlyphClass.REACTION
    """
    if width is None or height is None:
        default_width, default_height = get_default_dimensions(GlyphClass.REACTION)
        width = width or default_width
        height = height or default_height

    return {
        "id": id,
        "glyph_class": GlyphClass.REACTION,
        "label": label,
        "x": float(x),
        "y": float(y),
        "width": float(width),
        "height": float(height),
        "original_id": original_id or id,
    }


def create_metabolite_glyph(
    id: str,
    label: str,
    x: float,
    y: float,
    original_id: Optional[str] = None,
    subtype: str = MetaboliteSubtype.MAIN,
    width: Optional[float] = None,
    height: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a metabolite glyph with standardized structure.

    Args:
        id: Unique identifier for the glyph
        label: Human-readable name/label
        x: X position coordinate
        y: Y position coordinate
        original_id: Original metabolite ID from source data
        subtype: Metabolite subtype ("main" or "intermediate")
        width: Width dimension (uses default if None)
        height: Height dimension (uses default if None)

    Returns:
        Dictionary representing a metabolite glyph

    Example:
        >>> glyph = create_metabolite_glyph("m1", "Glucose", 50.0, 100.0,
        ...                                 subtype=MetaboliteSubtype.MAIN)
        >>> assert glyph["subtype"] == "main"
    """
    if width is None or height is None:
        default_width, default_height = get_default_dimensions(
            GlyphClass.METABOLITE, subtype
        )
        width = width or default_width
        height = height or default_height

    return {
        "id": id,
        "glyph_class": GlyphClass.METABOLITE,
        "label": label,
        "x": float(x),
        "y": float(y),
        "width": float(width),
        "height": float(height),
        "original_id": original_id or id,
        "subtype": subtype,
    }


def create_gene_glyph(
    id: str,
    label: str,
    x: float,
    y: float,
    associated_reaction: str,
    enzyme_group: Optional[str] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a gene glyph with standardized structure.

    Args:
        id: Unique identifier for the glyph
        label: Human-readable gene name/label
        x: X position coordinate
        y: Y position coordinate
        associated_reaction: ID of the associated reaction
        enzyme_group: ID of parent enzyme group (if grouped)
        width: Width dimension (uses default if None)
        height: Height dimension (uses default if None)

    Returns:
        Dictionary representing a gene glyph

    Example:
        >>> glyph = create_gene_glyph("g1", "ENSG123", 75.0, 150.0, "r1")
        >>> assert glyph["associated_reaction"] == "r1"
    """
    if width is None or height is None:
        default_width, default_height = get_default_dimensions(GlyphClass.GENE)
        width = width or default_width
        height = height or default_height

    return {
        "id": id,
        "glyph_class": GlyphClass.GENE,
        "label": label,
        "x": float(x),
        "y": float(y),
        "width": float(width),
        "height": float(height),
        "associated_reaction": associated_reaction,
        "enzyme_group": enzyme_group,
    }


def create_enzyme_group_glyph(
    id: str,
    label: str,
    x: float,
    y: float,
    reaction_id: str,
    enzymes: List[str],
    glyph_ids: List[str],
    width: float,
    height: float,
    outline_color: Optional[str] = None,
    outline_width: Optional[int] = None,
    outline_dash: Optional[Union[str, List[int]]] = None,
) -> Dict[str, Any]:
    """Create an enzyme group glyph with standardized structure.

    Args:
        id: Unique identifier for the glyph
        label: Human-readable group label
        x: X position coordinate
        y: Y position coordinate
        reaction_id: ID of the associated reaction
        enzymes: List of enzyme names in this group
        glyph_ids: List of gene glyph IDs contained in this group
        width: Width dimension
        height: Height dimension
        outline_color: Border color (uses default if None)
        outline_width: Border thickness (uses default if None)
        outline_dash: Dash pattern (uses default if None)

    Returns:
        Dictionary representing an enzyme group glyph

    Example:
        >>> glyph = create_enzyme_group_glyph("eg1", "Enzymes for R1", 100.0, 200.0,
        ...                                   "r1", ["ENZ1", "ENZ2"], ["g1", "g2"],
        ...                                   120.0, 80.0)
        >>> assert len(glyph["enzymes"]) == 2
    """
    return {
        "id": id,
        "glyph_class": GlyphClass.ENZYME_GROUP,
        "label": label,
        "x": float(x),
        "y": float(y),
        "width": float(width),
        "height": float(height),
        "reaction_id": reaction_id,
        "enzymes": enzymes,
        "glyph_ids": glyph_ids,
        "outline_color": outline_color or DefaultStyling.ENZYME_GROUP_COLOR,
        "outline_width": outline_width or DefaultStyling.ENZYME_GROUP_WIDTH,
        "outline_dash": outline_dash or DefaultStyling.ENZYME_GROUP_DASH,
    }


def create_compartment_glyph(
    id: str, label: str, x: float, y: float, width: float, height: float
) -> Dict[str, Any]:
    """Create a compartment glyph with standardized structure.

    Args:
        id: Unique identifier for the glyph
        label: Compartment name (e.g., "Cytosol", "Mitochondria")
        x: X position coordinate (center)
        y: Y position coordinate (center)
        width: Width dimension
        height: Height dimension

    Returns:
        Dictionary representing a compartment glyph

    Example:
        >>> glyph = create_compartment_glyph("comp1", "Cytosol", 500.0, 300.0,
        ...                                  800.0, 600.0)
        >>> assert glyph["label"] == "Cytosol"
    """
    return {
        "id": id,
        "glyph_class": GlyphClass.COMPARTMENT,
        "label": label,
        "x": float(x),
        "y": float(y),
        "width": float(width),
        "height": float(height),
    }


def create_subsystem_category_glyph(
    id: str,
    label: str,
    x: float,
    y: float,
    radius: float,
    subsystem_count: int = 0,
) -> Dict[str, Any]:
    """Create a subsystem category glyph with circular boundary structure.

    Args:
        id: Unique identifier for the glyph
        label: Category name (e.g., "Carbohydrate Metabolism", "Lipid Metabolism")
        x: X position coordinate (center)
        y: Y position coordinate (center)
        radius: Radius of the circular boundary
        subsystem_count: Number of subsystems in this category

    Returns:
        Dictionary representing a subsystem category glyph

    Example:
        >>> glyph = create_subsystem_category_glyph(
        ...     "cat1", "Carbohydrate Metabolism", 500.0, 400.0, 200.0, 6
        ... )
        >>> assert glyph["label"] == "Carbohydrate Metabolism"
        >>> assert glyph["glyph_class"] == GlyphClass.SUBSYSTEM_CATEGORY
    """
    return {
        "id": id,
        "glyph_class": GlyphClass.SUBSYSTEM_CATEGORY,
        "label": label,
        "x": float(x),
        "y": float(y),
        "radius": float(radius),
        "width": float(radius * 2),  # For compatibility with rectangular calculations
        "height": float(radius * 2),  # For compatibility with rectangular calculations
        "subsystem_count": subsystem_count,
    }


def create_subsystem_glyph(
    id: str,
    label: str,
    x: float,
    y: float,
    subsystem_name: str,
    reaction_count: int = 0,
    metabolite_count: int = 0,
    width: Optional[float] = None,
    height: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a subsystem glyph with standardized structure.

    Args:
        id: Unique identifier for the glyph
        label: Human-readable subsystem label
        x: X position coordinate (center)
        y: Y position coordinate (center)
        subsystem_name: Original subsystem name from model
        reaction_count: Number of reactions in this subsystem
        metabolite_count: Number of unique metabolites in this subsystem
        width: Width dimension (uses default if None)
        height: Height dimension (uses default if None)

    Returns:
        Dictionary representing a subsystem glyph

    Example:
        >>> glyph = create_subsystem_glyph("sub1", "Glycolysis", 200.0, 300.0,
        ...                                "Glycolysis/Gluconeogenesis", 15, 25)
        >>> assert glyph["subsystem_name"] == "Glycolysis/Gluconeogenesis"
    """
    if width is None or height is None:
        default_width, default_height = get_default_dimensions(GlyphClass.SUBSYSTEM)
        width = width or default_width
        height = height or default_height

    return {
        "id": id,
        "glyph_class": GlyphClass.SUBSYSTEM,
        "label": label,
        "x": float(x),
        "y": float(y),
        "width": float(width),
        "height": float(height),
        "subsystem_name": subsystem_name,
        "reaction_count": reaction_count,
        "metabolite_count": metabolite_count,
    }


# ============================================================================
# Factory Functions for Connection Creation
# ============================================================================


def create_metabolic_connection(
    id: str,
    source: str,
    target: str,
    connection_class: str = ConnectionClass.METABOLIC,
    stoichiometry: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a metabolic connection with standardized structure.

    Args:
        id: Unique identifier for the connection
        source: Source glyph ID
        target: Target glyph ID
        connection_class: Type of connection (metabolic, regulatory, generic)
        stoichiometry: Reaction coefficient (optional)

    Returns:
        Dictionary representing a metabolic connection

    Example:
        >>> conn = create_metabolic_connection("c1", "m1", "r1",
        ...                                    ConnectionClass.METABOLIC)
        >>> assert conn["source"] == "m1"
    """
    return {
        "id": id,
        "connection_class": connection_class,
        "source": source,
        "target": target,
        "stoichiometry": stoichiometry,
    }


def create_subsystem_connection(
    id: str,
    source: str,
    target: str,
    shared_metabolites: List[str],
    weight: float = 1.0,
    reaction_count: int = 1,
    metabolite_flow_direction: str = "forward",
    connection_class: str = ConnectionClass.SUBSYSTEM,
) -> Dict[str, Any]:
    """Create a subsystem connection with standardized structure.

    Args:
        id: Unique identifier for the connection
        source: Source subsystem glyph ID
        target: Target subsystem glyph ID
        shared_metabolites: List of metabolites shared between subsystems
        weight: Connection strength (e.g., number of shared metabolites)
        reaction_count: Number of reactions contributing to this connection
        metabolite_flow_direction: Direction of metabolite flow
        connection_class: Type of connection (default: subsystem)

    Returns:
        Dictionary representing a subsystem connection

    Example:
        >>> conn = create_subsystem_connection(
        ...     "sc1", "sub1", "sub2", ["glucose", "atp"],
        ...     weight=2.0, reaction_count=3
        ... )
        >>> assert conn["weight"] == 2.0
    """
    return {
        "id": id,
        "connection_class": connection_class,
        "source": source,
        "target": target,
        "shared_metabolites": shared_metabolites,
        "weight": weight,
        "reaction_count": reaction_count,
        "metabolite_flow_direction": metabolite_flow_direction,
    }

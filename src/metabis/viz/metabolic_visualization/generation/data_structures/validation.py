"""Validation utilities for graph data structures.

This module provides validation functions for ensuring the integrity and
consistency of graph glyphs and connections. It validates structure,
required fields, data types, and referential integrity.

Usage:
    >>> from .validation import validate_glyph_structure, validate_graph_data
    >>> from .basic_factories import create_reaction_glyph
    >>> glyph = create_reaction_glyph("r1", "Reaction 1", 100.0, 200.0)
    >>> assert validate_glyph_structure(glyph) is True
"""

from typing import Any, Dict, List

from .core_types import ConnectionClass, GlyphClass


def validate_glyph_structure(glyph: Dict[str, Any]) -> bool:
    """Validate that a glyph has the required structure.

    Args:
        glyph: Glyph dictionary to validate

    Returns:
        True if glyph structure is valid, False otherwise

    Example:
        >>> from .basic_factories import create_reaction_glyph
        >>> glyph = create_reaction_glyph("r1", "Reaction 1", 100.0, 200.0)
        >>> assert validate_glyph_structure(glyph) is True
    """
    required_fields = ["id", "glyph_class", "label", "x", "y", "width", "height"]

    # Check required fields exist
    for field in required_fields:
        if field not in glyph:
            return False

    # Check glyph_class is valid
    valid_classes = [
        GlyphClass.REACTION,
        GlyphClass.METABOLITE,
        GlyphClass.GENE,
        GlyphClass.ENZYME_GROUP,
        GlyphClass.COMPARTMENT,
    ]
    if glyph["glyph_class"] not in valid_classes:
        return False

    # Check coordinate and dimension types
    try:
        float(glyph["x"])
        float(glyph["y"])
        float(glyph["width"])
        float(glyph["height"])
    except (ValueError, TypeError):
        return False

    return True


def validate_connection_structure(connection: Dict[str, Any]) -> bool:
    """Validate that a connection has the required structure.

    Args:
        connection: Connection dictionary to validate

    Returns:
        True if connection structure is valid, False otherwise

    Example:
        >>> from .basic_factories import create_metabolic_connection
        >>> conn = create_metabolic_connection("c1", "m1", "r1")
        >>> assert validate_connection_structure(conn) is True
    """
    required_fields = ["id", "connection_class", "source", "target"]

    # Check required fields exist
    for field in required_fields:
        if field not in connection:
            return False

    # Check connection_class is valid
    valid_classes = [
        ConnectionClass.METABOLIC,
        ConnectionClass.REGULATORY,
        ConnectionClass.GENERIC,
    ]
    if connection["connection_class"] not in valid_classes:
        return False

    return True


def validate_graph_data(
    glyphs: List[Dict[str, Any]], connections: List[Dict[str, Any]]
) -> tuple[bool, List[str]]:
    """Validate complete graph data structure.

    Args:
        glyphs: List of glyph dictionaries
        connections: List of connection dictionaries

    Returns:
        Tuple of (is_valid, list_of_error_messages)

    Example:
        >>> from .basic_factories import create_reaction_glyph
        >>> glyphs = [create_reaction_glyph("r1", "Reaction 1", 100.0, 200.0)]
        >>> connections = []
        >>> is_valid, errors = validate_graph_data(glyphs, connections)
        >>> assert is_valid is True
    """
    errors = []

    # Validate all glyphs
    for i, glyph in enumerate(glyphs):
        if not validate_glyph_structure(glyph):
            errors.append(
                f"Invalid glyph structure at index {i}: {glyph.get('id', 'unknown')}"
            )

    # Validate all connections
    for i, connection in enumerate(connections):
        if not validate_connection_structure(connection):
            errors.append(
                f"Invalid connection structure at index {i}: "
                f"{connection.get('id', 'unknown')}"
            )

    # Check that connection endpoints reference existing glyphs
    glyph_ids = {glyph["id"] for glyph in glyphs if "id" in glyph}
    for connection in connections:
        if "source" in connection and connection["source"] not in glyph_ids:
            errors.append(
                f"Connection {connection.get('id', 'unknown')} references "
                f"non-existent source: {connection['source']}"
            )
        if "target" in connection and connection["target"] not in glyph_ids:
            errors.append(
                f"Connection {connection.get('id', 'unknown')} references "
                f"non-existent target: {connection['target']}"
            )

    return len(errors) == 0, errors


def validate_enriched_glyph_structure(glyph: Dict[str, Any]) -> bool:
    """Validate that an enriched glyph has the required visualization fields.

    Args:
        glyph: Enriched glyph dictionary to validate

    Returns:
        True if enriched glyph structure is valid, False otherwise

    Example:
        >>> from .basic_factories import create_reaction_glyph
        >>> from .enriched_factories import enrich_reaction_glyph
        >>> base_glyph = create_reaction_glyph("r1", "Reaction 1", 100.0, 200.0)
        >>> enriched_glyph = enrich_reaction_glyph(base_glyph)
        >>> assert validate_enriched_glyph_structure(enriched_glyph) is True
    """
    # First validate basic structure
    if not validate_glyph_structure(glyph):
        return False

    # Check for enriched fields
    enriched_fields = ["alpha", "display_label", "color"]
    for field in enriched_fields:
        if field not in glyph:
            return False

    # Check field types
    try:
        float(glyph["alpha"])
        str(glyph["display_label"])
        str(glyph["color"])
    except (ValueError, TypeError):
        return False

    return True


def validate_enriched_connection_structure(connection: Dict[str, Any]) -> bool:
    """Validate that an enriched connection has the required visualization fields.

    Args:
        connection: Enriched connection dictionary to validate

    Returns:
        True if enriched connection structure is valid, False otherwise

    Example:
        >>> from .basic_factories import create_metabolic_connection
        >>> from .enriched_factories import enrich_metabolic_connection
        >>> base_conn = create_metabolic_connection("c1", "m1", "r1")
        >>> enriched_conn = enrich_metabolic_connection(base_conn, 0, 0, 100, 100)
        >>> assert validate_enriched_connection_structure(enriched_conn) is True
    """
    # First validate basic structure
    if not validate_connection_structure(connection):
        return False

    # Check for enriched fields
    enriched_fields = ["x0", "y0", "x1", "y1", "color", "line_alpha", "line_width"]
    for field in enriched_fields:
        if field not in connection:
            return False

    # Check field types
    try:
        float(connection["x0"])
        float(connection["y0"])
        float(connection["x1"])
        float(connection["y1"])
        str(connection["color"])
        float(connection["line_alpha"])
        float(connection["line_width"])
    except (ValueError, TypeError):
        return False

    return True

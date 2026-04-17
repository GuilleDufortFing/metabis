"""Utility functions for graph data structures.

This module provides utility functions that support graph creation and manipulation,
including ID generation, coordinate calculations, and other helper functions.

Usage:
    >>> from .utilities import generate_unique_id, calculate_distance
    >>> unique_id = generate_unique_id("reaction")
    >>> distance = calculate_distance((0, 0), (3, 4))
"""

import uuid
from typing import Optional, Tuple


def generate_unique_id(prefix: str = "") -> str:
    """Generate a unique identifier with optional prefix.

    Args:
        prefix: Optional prefix for the ID

    Returns:
        Unique string identifier

    Example:
        >>> generate_unique_id("reaction")  # doctest: +ELLIPSIS
        'reaction_...'
        >>> len(generate_unique_id()) == 8
        True
    """
    unique_part = uuid.uuid4().hex[:8]
    return f"{prefix}_{unique_part}" if prefix else unique_part


def get_default_dimensions(
    glyph_class: str, subtype: Optional[str] = None
) -> tuple[float, float]:
    """Get default width and height for a glyph type.

    Args:
        glyph_class: The type of glyph (from GlyphClass constants)
        subtype: Optional subtype (e.g., for metabolites)

    Returns:
        Tuple of (width, height)

    Example:
        >>> get_default_dimensions("reaction")
        (40, 40)
        >>> get_default_dimensions("metabolite", "main")
        (60, 30)
    """
    # Import here to avoid circular imports
    from .constants import DefaultDimensions
    from .core_types import GlyphClass, MetaboliteSubtype

    if glyph_class == GlyphClass.REACTION:
        return (DefaultDimensions.REACTION_WIDTH, DefaultDimensions.REACTION_HEIGHT)
    elif glyph_class == GlyphClass.METABOLITE:
        if subtype == MetaboliteSubtype.INTERMEDIATE:
            return (
                DefaultDimensions.METABOLITE_INTERMEDIATE_WIDTH,
                DefaultDimensions.METABOLITE_INTERMEDIATE_HEIGHT,
            )
        else:
            return (
                DefaultDimensions.METABOLITE_MAIN_WIDTH,
                DefaultDimensions.METABOLITE_MAIN_HEIGHT,
            )
    elif glyph_class == GlyphClass.GENE:
        return (DefaultDimensions.GENE_WIDTH, DefaultDimensions.GENE_HEIGHT)
    elif glyph_class == GlyphClass.SUBSYSTEM:
        return (DefaultDimensions.SUBSYSTEM_WIDTH, DefaultDimensions.SUBSYSTEM_HEIGHT)
    elif glyph_class == GlyphClass.ENZYME_GROUP:
        return (
            DefaultDimensions.ENZYME_GROUP_WIDTH,
            DefaultDimensions.ENZYME_GROUP_HEIGHT,
        )
    elif glyph_class == GlyphClass.COMPARTMENT:
        return (
            DefaultDimensions.COMPARTMENT_WIDTH,
            DefaultDimensions.COMPARTMENT_HEIGHT,
        )
    elif glyph_class == GlyphClass.SUBSYSTEM_CATEGORY:
        # For category glyphs, return diameter as both width and height
        radius = DefaultDimensions.SUBSYSTEM_CATEGORY_RADIUS
        return (radius * 2, radius * 2)
    else:
        return (DefaultDimensions.DEFAULT_WIDTH, DefaultDimensions.DEFAULT_HEIGHT)


def calculate_distance(
    point1: Tuple[float, float], point2: Tuple[float, float]
) -> float:
    """Calculate Euclidean distance between two points.

    Args:
        point1: First point as (x, y) tuple
        point2: Second point as (x, y) tuple

    Returns:
        Euclidean distance between the points

    Example:
        >>> calculate_distance((0, 0), (3, 4))
        5.0
        >>> calculate_distance((1, 1), (4, 5))
        5.0
    """
    x1, y1 = point1
    x2, y2 = point2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def calculate_midpoint(
    point1: Tuple[float, float], point2: Tuple[float, float]
) -> Tuple[float, float]:
    """Calculate the midpoint between two points.

    Args:
        point1: First point as (x, y) tuple
        point2: Second point as (x, y) tuple

    Returns:
        Midpoint as (x, y) tuple

    Example:
        >>> calculate_midpoint((0, 0), (4, 4))
        (2.0, 2.0)
        >>> calculate_midpoint((1, 3), (5, 7))
        (3.0, 5.0)
    """
    x1, y1 = point1
    x2, y2 = point2
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def normalize_coordinates(
    coordinates: list[Tuple[float, float]],
    target_width: float = 1000.0,
    target_height: float = 1000.0,
) -> list[Tuple[float, float]]:
    """Normalize a list of coordinates to fit within target dimensions.

    Args:
        coordinates: List of (x, y) coordinate tuples
        target_width: Target width for normalization
        target_height: Target height for normalization

    Returns:
        List of normalized (x, y) coordinate tuples

    Example:
        >>> coords = [(0, 0), (100, 200), (50, 100)]
        >>> normalized = normalize_coordinates(coords, 500, 500)
        >>> len(normalized) == 3
        True
    """
    if not coordinates:
        return []

    # Find min/max values
    x_coords = [x for x, y in coordinates]
    y_coords = [y for x, y in coordinates]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Calculate scaling factors
    x_range = max_x - min_x
    y_range = max_y - min_y

    if x_range == 0:
        x_scale = 1
    else:
        x_scale = target_width / x_range

    if y_range == 0:
        y_scale = 1
    else:
        y_scale = target_height / y_range

    # Use the smaller scale to maintain aspect ratio
    scale = min(x_scale, y_scale)

    # Normalize coordinates
    normalized = []
    for x, y in coordinates:
        new_x = (x - min_x) * scale
        new_y = (y - min_y) * scale
        normalized.append((new_x, new_y))

    return normalized


def clamp_value(value: float, min_value: float, max_value: float) -> float:
    """Clamp a value to be within specified bounds.

    Args:
        value: Value to clamp
        min_value: Minimum allowed value
        max_value: Maximum allowed value

    Returns:
        Clamped value

    Example:
        >>> clamp_value(5, 0, 10)
        5
        >>> clamp_value(-5, 0, 10)
        0
        >>> clamp_value(15, 0, 10)
        10
    """
    return max(min_value, min(value, max_value))


def round_to_precision(value: float, precision: int = 2) -> float:
    """Round a float value to specified decimal precision.

    Args:
        value: Value to round
        precision: Number of decimal places

    Returns:
        Rounded value

    Example:
        >>> round_to_precision(3.14159, 2)
        3.14
        >>> round_to_precision(2.71828, 3)
        2.718
    """
    return round(value, precision)

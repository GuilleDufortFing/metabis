"""Shared utilities and validation functions for graph generation.

This module provides common utilities, validation functions, and type definitions
used by both Human1 map generation and model-based graph creation.
"""

import os
from typing import Any, Dict, List

from ..data_structures.core_types import Glyph  # Importing Glyph from core_types module

# Type definitions for better code clarity and type safety

# Configuration for graph generation
# Contains: sbgn_file, svg_file, hmr_mar_file
GraphConfig = Dict[str, str]


def validate_config(config: GraphConfig) -> None:
    """Validate configuration parameters for graph generation.

    Args:
        config: Configuration dictionary containing file paths

    Raises:
        ValueError: If required keys are missing or files don't exist
    """
    required_keys = ["sbgn_file"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Required config key '{key}' is missing")

    # Check if required files exist
    if not os.path.exists(config["sbgn_file"]):
        raise ValueError(f"SBGN file not found: {config['sbgn_file']}")

    # SVG file is optional, but if provided, should exist
    if "svg_file" in config and config["svg_file"]:
        if not os.path.exists(config["svg_file"]):
            raise ValueError(f"SVG file not found: {config['svg_file']}")


def validate_glyphs(glyphs: List[Glyph]) -> None:
    """Validate glyph data structure.

    Args:
        glyphs: List of glyph dictionaries

    Raises:
        ValueError: If glyph data is invalid
    """
    required_fields = ["id", "glyph_class", "x", "y"]
    for i, glyph in enumerate(glyphs):
        for field in required_fields:
            if field not in glyph:
                raise ValueError(f"Glyph {i} missing required field: {field}")


def is_null_value(value: Any) -> bool:
    """Check if a value represents null/NaN in the context of this module.

    Args:
        value: Value to check

    Returns:
        True if value is null/NaN, False otherwise
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value == "Nan"
    # Handle numpy NaN if pandas were used
    try:
        import math

        return math.isnan(float(value))
    except (ValueError, TypeError):
        return False

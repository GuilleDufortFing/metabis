"""Utilities for packaged graph resource path resolution and JSON loading."""

import hashlib
import json
import os
from copy import deepcopy
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from ..data_structures.core_types import Connection, Glyph
from .compartment_resources import (
    COMPARTMENT_GRAPH_VARIANT,
    normalize_compartment_resource_id,
)

SUBSYSTEM_HUMAN1_VARIANT = "human1"
SUBSYSTEM_STANDARD_VARIANT = "standard"
SUBSYSTEM_GRAPH_VARIANTS = (
    SUBSYSTEM_HUMAN1_VARIANT,
    SUBSYSTEM_STANDARD_VARIANT,
)
DEFAULT_SUBSYSTEM_LAYOUT_ENGINE = "dot"
SUPPORTED_LAYOUT_ENGINES = (
    "dot",
    "fdp",
    "neato",
    "sfdp",
    "circo",
    "twopi",
)


def _normalize_element_name(graph_type: str, element_name: str) -> str:
    """Normalize a graph element name into a stable resource filename stem."""
    if graph_type == "pathway":
        return f"pathway_{element_name}"

    return (
        element_name.replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace(" / ", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
        .lower()
    )


def _build_metabolic_network_resource_stem(
    layout_engine: str = "dot",
    subsystem_filter: Optional[List[str]] = None,
) -> str:
    """Build a deterministic official resource stem for metabolic networks."""
    resource_parts = [f"layout_{layout_engine}"]

    if subsystem_filter:
        sorted_subsystems = sorted(subsystem_filter)
        subsystem_str = "_".join(sorted_subsystems)
        if len(subsystem_str) > 50:
            subsystem_hash = hashlib.md5(subsystem_str.encode()).hexdigest()[:8]
            resource_parts.append(f"subsystems_{subsystem_hash}")
        else:
            clean_subsystems = subsystem_str.replace(" ", "_").replace("/", "_")
            resource_parts.append(f"subsystems_{clean_subsystems}")
    else:
        resource_parts.append("all_subsystems")

    resource_stem = "_".join(resource_parts)
    return (
        resource_stem.replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
        .lower()
    )


def get_graph_resource_path(
    graph_type: str,
    element_name: str,
    base_paths: Dict[str, str],
    layout_engine: str = "dot",
    use_human1_map: bool = True,
    subsystem_filter: Optional[List[str]] = None,
) -> str:
    """Generate the packaged graph-resource path for a graph request."""
    if graph_type == "metabolic_network":
        resource_stem = _build_metabolic_network_resource_stem(
            layout_engine=layout_engine,
            subsystem_filter=subsystem_filter,
        )
        return os.path.join(base_paths["metabolic_network"], f"{resource_stem}.json")

    if graph_type == "subsystem":
        variant = SUBSYSTEM_HUMAN1_VARIANT if use_human1_map else SUBSYSTEM_STANDARD_VARIANT
        return get_subsystem_graph_resource_path(
            element_name,
            base_paths,
            variant=variant,
        )

    if graph_type == "compartment":
        resource_id = normalize_compartment_resource_id(element_name)
        return os.path.join(
            base_paths["compartment"],
            COMPARTMENT_GRAPH_VARIANT,
            f"{resource_id}.json",
        )

    element_name_normalized = _normalize_element_name(graph_type, element_name)
    return os.path.join(base_paths[graph_type], f"{element_name_normalized}.json")


def get_cache_file_path(
    graph_type: str,
    element_name: str,
    base_paths: Dict[str, str],
    layout_engine: str = "dot",
    use_human1_map: bool = True,
    subsystem_filter: Optional[List[str]] = None,
) -> str:
    """Generate the packaged graph-resource path for any graph type.

    Args:
        graph_type: Type of graph ('subsystem', 'pathway', 'compartment',
                   'metabolic_network')
        element_name: Name of the element to visualize
        base_paths: Dictionary mapping graph types to their base paths
        layout_engine: GraphViz layout algorithm (optional, defaults to "dot")
        use_human1_map: Selects the packaged Human1 or standard subsystem
            resource variant when `graph_type` is `subsystem`.
        subsystem_filter: List of subsystems for metabolic_network (optional)

    Returns:
        str: Full path to the packaged graph resource
    """
    return get_graph_resource_path(
        graph_type=graph_type,
        element_name=element_name,
        base_paths=base_paths,
        layout_engine=layout_engine,
        use_human1_map=use_human1_map,
        subsystem_filter=subsystem_filter,
    )


def get_subsystem_graph_resource_path(
    element_name: str,
    base_paths: Dict[str, str],
    *,
    variant: str = SUBSYSTEM_HUMAN1_VARIANT,
) -> str:
    """Return the packaged subsystem graph-resource path for a specific variant."""
    if variant not in SUBSYSTEM_GRAPH_VARIANTS:
        raise ValueError(
            f"Unsupported subsystem graph variant '{variant}'. "
            f"Expected one of: {SUBSYSTEM_GRAPH_VARIANTS}"
        )

    element_name_normalized = _normalize_element_name("subsystem", element_name)
    return os.path.join(
        base_paths["subsystem"],
        variant,
        f"{element_name_normalized}.json",
    )


def get_available_subsystem_graph_resource_paths(
    element_name: str,
    base_paths: Dict[str, str],
) -> Dict[str, str]:
    """Return the packaged subsystem graph variants available for an element."""
    available_paths = {}
    for variant in SUBSYSTEM_GRAPH_VARIANTS:
        resource_path = get_subsystem_graph_resource_path(
            element_name,
            base_paths,
            variant=variant,
        )
        if os.path.exists(resource_path):
            available_paths[variant] = resource_path

    return available_paths


def load_graph_resource(
    resource_file: str,
) -> Tuple[Optional[List[Glyph]], Optional[List[Connection]]]:
    """Load packaged graph data with consistent logging.

    Args:
        resource_file: Full path to the packaged graph-resource file

    Returns:
        Tuple of (glyphs, connections) or (None, None) if not found/error
    """
    if not os.path.exists(resource_file):
        return None, None

    try:
        glyphs, connections = load_visualization_cache(resource_file)
        if glyphs is not None and connections is not None:
            return glyphs, connections
    except Exception as e:
        print(f"⚠ Error loading packaged graph resource {resource_file}: {e}")

    return None, None


def save_to_cache(
    cache_file: str, glyphs: List[Glyph], connections: List[Connection]
) -> None:
    """Save visualization data to cache with consistent logging.

    Args:
        cache_file: Full path to cache file
        glyphs: List of glyph data to save
        connections: List of connection data to save
    """
    try:
        # Ensure cache directory exists
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        # Save to cache
        save_visualization_cache(cache_file, glyphs, connections)

    except Exception as e:
        print(f"⚠ Error saving cache file {cache_file}: {e}")


def save_visualization_cache(cache_file, glyphs, connections):
    """Save graph visualization data to a JSON file.

    Args:
        cache_file: Path where the cache file should be saved
        glyphs: List of glyph data dictionaries
        connections: List of connection data dictionaries
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    # Prepare data for serialization
    cache_data = {"glyphs": glyphs, "connections": connections}

    # Save data to file
    with open(cache_file, "w") as f:
        json.dump(cache_data, f, indent=2)

@lru_cache(maxsize=256)
def _load_visualization_cache_cached(cache_file: str, mtime_ns: int, size: int):
    """Parse one immutable packaged resource version exactly once."""
    with open(cache_file, "r", encoding="utf-8") as handle:
        cache_data = json.load(handle)
    return cache_data.get("glyphs", []), cache_data.get("connections", [])


def clear_graph_resource_cache() -> None:
    _load_visualization_cache_cached.cache_clear()


def get_graph_resource_cache_info():
    return _load_visualization_cache_cached.cache_info()


def load_visualization_cache(cache_file):
    """Load graph visualization data from a JSON file.

    Args:
        cache_file: Path to the cache file

    Returns:
        Tuple of (glyphs, connections)
    """
    if not os.path.exists(cache_file):
        return None, None

    try:
        stat = os.stat(cache_file)
        glyphs, connections = _load_visualization_cache_cached(
            os.path.abspath(cache_file), stat.st_mtime_ns, stat.st_size
        )
        # Enrichment mutates nested dictionaries. Never expose cached objects.
        return deepcopy(glyphs), deepcopy(connections)
    except Exception as e:
        print(f"Error loading graph visualization resource: {e}")
        return None, None

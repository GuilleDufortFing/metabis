"""Metabolic graph generation utilities."""

import os
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional

from metabis.resource_registry import ResourceRegistry

from .graph_building.compartment_graph import create_compartmentalized_graph_from_model
from .utilities.cache_utils import (
    DEFAULT_SUBSYSTEM_LAYOUT_ENGINE,
    SUBSYSTEM_HUMAN1_VARIANT,
    SUBSYSTEM_STANDARD_VARIANT,
    get_available_subsystem_graph_resource_paths,
    get_graph_resource_path,
    load_graph_resource,
)


@dataclass(frozen=True)
class GraphGenerationConfig:
    """Configuration for graph generation options."""

    use_human1_map: bool = True
    layout_engine: str = "dot"
    base_paths: Optional[Mapping[str, str]] = None
    resources_path: Optional[str] = None
    resource_registry: Optional[ResourceRegistry] = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self):
        """Set default paths if not provided."""
        registry = self.resource_registry
        if registry is None:
            registry = ResourceRegistry(resources_root=self.resources_path)
            object.__setattr__(self, "resource_registry", registry)

        if self.base_paths is None:
            object.__setattr__(
                self,
                "base_paths",
                MappingProxyType(registry.get_graph_base_paths(required=False)),
            )
        else:
            object.__setattr__(
                self,
                "base_paths",
                MappingProxyType(dict(self.base_paths)),
            )

        if self.resources_path is None:
            object.__setattr__(self, "resources_path", registry.resources_root)


class MetabolicGraphGenerator:
    """Class to handle the generation of metabolic graphs."""

    def __init__(self, metabolic_model, config: Optional[GraphGenerationConfig] = None):
        """Initialize the graph generator.

        Args:
            metabolic_model: The metabolic model to use for graph generation.
            config: Configuration for graph generation options.
                   If None, defaults will be used.
        """
        self.model = metabolic_model
        self.config = config or GraphGenerationConfig(
            resource_registry=getattr(metabolic_model, "resource_registry", None)
        )
        self.last_resource_path: str | None = None
        self.last_resource_variant = "runtime"
        self.used_runtime_layout = False

    def create_graph(self, element_id, graph_type):
        """Create a metabolic graph with full configuration support.

        This is the unified interface for creating all types of metabolic graphs
        with support for packaged Human1 layouts, packaged standard dot layouts,
        and engine-based subsystem generation.

        Args:
            element_id (str): ID of the subsystem, compartment,
                             or filter for metabolic_network (already validated)
            graph_type (str): Type of graph to generate
                             ('subsystem', 'compartment', 'metabolic_network')

        Returns:
            tuple: (glyphs, connections, display_title) for visualizing the graph

        Raises:
            ValueError: If graph_type is invalid or required parameters are missing
        """
        # Step 1: Validate input parameters
        self.last_resource_path = None
        self.last_resource_variant = "runtime"
        self.used_runtime_layout = False
        valid_graph_types = ["subsystem", "compartment", "metabolic_network"]
        if graph_type not in valid_graph_types:
            raise ValueError(
                f"Invalid graph_type '{graph_type}'. "
                f"Must be one of: {valid_graph_types}"
            )

        # Step 2: Build unified configuration
        config = self._build_config(element_id, graph_type)
        layout_engine = self.config.layout_engine

        if graph_type == "compartment" and not config["use_human1_map"]:
            raise ValueError(
                "Compartment graphs require packaged Human1 layouts. "
                "Automatic compartment layout is not supported."
            )

        if graph_type == "subsystem":
            glyphs, connections = self._load_packaged_graph(
                config,
                graph_type,
                layout_engine,
            )
            if glyphs is None or connections is None:
                print(
                    "Falling back to automatic subsystem layout for "
                    f"{config['element_name_normalized']} using {layout_engine}."
                )
                glyphs, connections = create_compartmentalized_graph_from_model(
                    config["element_name_normalized"],
                    self.model,
                    layout_engine,
                )
                self.used_runtime_layout = True
        else:
            glyphs, connections, graph_resource_file = self._load_packaged_graph(
                config,
                graph_type,
                layout_engine,
                return_resource_file=True,
            )
            if glyphs is None or connections is None:
                raise FileNotFoundError(
                    "Missing packaged graph resource for "
                    f"{graph_type} '{config['element_name_normalized']}': "
                    f"{graph_resource_file}"
                )

        return glyphs, connections, config["display_title"]

    def _load_packaged_graph(
        self,
        config,
        graph_type,
        layout_engine,
        *,
        return_resource_file=False,
    ):
        """Load a packaged graph resource when one is part of the runtime flow."""
        if graph_type == "subsystem":
            resource_candidates = self._get_subsystem_packaged_candidates(
                config,
                layout_engine,
            )
            last_resource_file = None

            for graph_resource_file in resource_candidates:
                last_resource_file = graph_resource_file
                glyphs, connections = load_graph_resource(graph_resource_file)
                if glyphs is not None and connections is not None:
                    self.last_resource_path = graph_resource_file
                    self.last_resource_variant = os.path.basename(
                        os.path.dirname(graph_resource_file)
                    )
                    if return_resource_file:
                        return glyphs, connections, graph_resource_file
                    return glyphs, connections

            if return_resource_file:
                return None, None, last_resource_file
            return None, None

        graph_resource_file = get_graph_resource_path(
            graph_type=graph_type,
            element_name=config["element_name_normalized"],
            base_paths=self.config.base_paths,
            layout_engine=layout_engine,
            use_human1_map=config["use_human1_map"],
            subsystem_filter=config.get("subsystem_filter"),
        )
        glyphs, connections = load_graph_resource(graph_resource_file)
        if glyphs is not None and connections is not None:
            self.last_resource_path = graph_resource_file
            self.last_resource_variant = "packaged"
        if return_resource_file:
            return glyphs, connections, graph_resource_file
        return glyphs, connections

    def _get_subsystem_packaged_candidates(self, config, layout_engine):
        """Return the packaged subsystem resource files eligible for loading."""
        available_paths = get_available_subsystem_graph_resource_paths(
            config["element_name_normalized"],
            self.config.base_paths,
        )
        resource_candidates = []

        if config["use_human1_map"] and SUBSYSTEM_HUMAN1_VARIANT in available_paths:
            resource_candidates.append(available_paths[SUBSYSTEM_HUMAN1_VARIANT])

        if (
            layout_engine == DEFAULT_SUBSYSTEM_LAYOUT_ENGINE
            and SUBSYSTEM_STANDARD_VARIANT in available_paths
            and available_paths[SUBSYSTEM_STANDARD_VARIANT] not in resource_candidates
        ):
            resource_candidates.append(available_paths[SUBSYSTEM_STANDARD_VARIANT])

        return resource_candidates

    def _build_config(self, element_id, graph_type, use_human1_map=None):
        """Build configuration based on graph type and element ID.

        Args:
            element_id (str): ID of the element (subsystem, compartment,
                             or metabolic_network) - already validated and normalized
            graph_type (str): Type of graph ('subsystem', 'compartment',
                             'metabolic_network')
            use_human1_map (bool, optional): Whether to use Human1 maps.
                                           If None, uses config default.

        Returns:
            dict: Configuration dictionary with paths and settings
        """
        # Use config default if not specified
        if use_human1_map is None:
            use_human1_map = self.config.use_human1_map

        # Since element_id is already normalized/validated, use it directly
        element_id_normalized = element_id

        # Generate display titles and parse any special cases
        if graph_type == "compartment":
            display_title = f"Compartment: {element_id}"
        elif graph_type == "metabolic_network":
            # For metabolic network, element_id might be a filter or empty
            display_title = "Metabolic Network"
            # Parse subsystem filter from element_id if provided
            subsystem_filter = None
            if isinstance(element_id, list) and element_id:
                subsystem_filter = element_id
            elif isinstance(element_id, str) and element_id.strip():
                subsystem_filter = [element_id.strip()]
        else:  # subsystem
            display_title = element_id

        base_path = self.config.base_paths[graph_type]

        config_dict = {
            "element_name_normalized": element_id_normalized,
            "display_title": display_title,
            "base_path": base_path,
            "exclude_intermediate_metabolites": True,
            "use_human1_map": use_human1_map,
        }

        # Add subsystem filter for metabolic_network graphs
        if graph_type == "metabolic_network":
            config_dict["subsystem_filter"] = subsystem_filter

        return config_dict

"""Graph enrichment class for metabolic visualization.

This module provides a class-based interface for enriching metabolic graphs
with experimental data, styling, and highlighting features.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .analysis import (
    identify_gene_influence_connections,
    identify_missing_data_subgraph,
    identify_significant_subgraph,
)
from .auxiliary_visualization import create_auxiliary_visualization_data
from .connection_enrichment import enrich_connections
from .glyph_enrichment import enrich_single_glyph
from .utilities import prepare_translations


@dataclass
class EnrichmentConfig:
    """Configuration for graph enrichment operations."""

    color_metric: str = "log2_fc"
    delta_options: Dict[str, Any] = None
    highlight_significant_reactions: bool = False
    highlight_significant_genes: bool = False
    highlight_gene_influence: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for function calls."""
        return {
            "color_metric": self.color_metric,
            "delta_options": self.delta_options,
            "highlight_significant_reactions": self.highlight_significant_reactions,
            "highlight_significant_genes": self.highlight_significant_genes,
            "highlight_gene_influence": self.highlight_gene_influence,
        }


class GraphEnricher:
    """Class for enriching metabolic graphs with experimental data and styling.

    This class provides a clean interface for the enrichment phase of the
    metabolic visualization pipeline, wrapping the existing enrichment
    functions in a structured, configurable way.
    """

    def __init__(self, config: Optional[EnrichmentConfig] = None):
        """Initialize the graph enricher.

        Args:
            config: Configuration for enrichment operations.
                   If None, defaults will be used.
        """
        self.config = config or EnrichmentConfig()

    def enrich(
        self,
        glyphs: Any,
        connections: Any,
        focus_instance,
        data_manager: Any = None,
        config_override: Optional[EnrichmentConfig] = None,
        style_missing_data: bool = True,
        graph_type: str = None,
    ) -> Tuple[Any, Any, Any]:
        """Enrich graph data with experimental data and styling.

        This method transforms basic glyphs and connections into enriched versions
        that contain all the visualization metadata needed for Bokeh rendering.

        Args:
            glyphs: Basic graph glyphs from generation phase
            connections: Basic graph connections from generation phase
            focus_instance: object that retrieves the IDs of the elements being visualized
            data_manager: DataManager object with experimental data
            config_override: Optional config to override instance config
            style_missing_data: Whether to apply special styling to missing
                               data elements
            graph_type: Type of graph being generated ("subsystem", "compartment", etc.)

        Returns:
            Tuple of (enriched_glyphs, enriched_connections, auxiliary_data)
        """
        # Use override config if provided, otherwise use instance config
        active_config = config_override or self.config
        config_dict = active_config.to_dict()

        # Prepare highlighting sets
        significant_nodes_set = set()
        if (
            config_dict["highlight_significant_reactions"]
            or config_dict["highlight_significant_genes"]
        ):
            significant_nodes = identify_significant_subgraph(
                glyphs,
                connections,
                data_manager,
                config_dict["highlight_significant_reactions"],
                config_dict["highlight_significant_genes"],
            )
            significant_nodes_set = set(significant_nodes)

        # Prepare gene influence data
        (
            influential_gene_glyph_ids,
            influence_connections_data,
            highly_influenced_reaction_colors,
        ) = identify_gene_influence_connections(
            glyphs, focus_instance, data_manager, config_dict["highlight_gene_influence"]
        )
        influential_genes_set = (
            set(influential_gene_glyph_ids) if influential_gene_glyph_ids else set()
        )

        # Prepare missing data highlighting dict
        missing_data_elements = {}
        if style_missing_data:
            missing_data_elements = identify_missing_data_subgraph(
                glyphs, connections, data_manager
            )
        # Pre-compute translations and reaction colors for batch operations
        translations = prepare_translations(
            glyphs, data_manager, config_dict["color_metric"]
        )

        # Enrich all glyphs
        enriched_glyphs = []
        for glyph in glyphs:
            enriched_glyph = enrich_single_glyph(
                glyph,
                data_manager,
                translations,
                config_dict["color_metric"],
                config_dict["delta_options"],
                (
                    config_dict["highlight_significant_reactions"]
                    or config_dict["highlight_significant_genes"]
                ),
                config_dict["highlight_gene_influence"],
                significant_nodes_set,
                influential_genes_set,
                missing_data_elements,
                focus_instance=focus_instance,
                graph_type=graph_type,
            )
            enriched_glyphs.append(enriched_glyph)

        # Enrich connections
        enriched_connections = enrich_connections(
            connections,
            enriched_glyphs,
            translations["reaction_colors"],
            (
                config_dict["highlight_significant_reactions"]
                or config_dict["highlight_significant_genes"]
            ),
            significant_nodes_set,
            set(missing_data_elements.keys()),
            data_manager,
            config_dict["color_metric"],
        )

        # Create auxiliary data structures
        auxiliary_data = create_auxiliary_visualization_data(
            enriched_glyphs,
            influence_connections_data,
            highly_influenced_reaction_colors,
            (
                config_dict["highlight_significant_reactions"]
                or config_dict["highlight_significant_genes"]
            ),
            significant_nodes_set,
            data_manager,
        )

        return enriched_glyphs, enriched_connections, auxiliary_data

    def update_config(self, **kwargs) -> None:
        """Update enrichment configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

    def get_config(self) -> EnrichmentConfig:
        """Get current enrichment configuration.

        Returns:
            Current EnrichmentConfig instance
        """
        return self.config

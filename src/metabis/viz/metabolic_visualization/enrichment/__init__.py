"""Enrichment module for metabolic graph visualization.

This module provides functionality for enriching basic metabolic graphs
with experimental data, styling, and highlighting features.

The main interface is through the `enrich_glyph_data` function and the
`GraphEnricher` class. The modular components (glyph_enrichment,
connection_enrichment, auxiliary_visualization, analysis, utilities)
are also available for advanced use cases.
"""

# Make modular components available for advanced users
from . import (
    analysis,
    auxiliary_visualization,
    connection_enrichment,
    glyph_enrichment,
    utilities,
)
from .enricher import EnrichmentConfig, GraphEnricher

__all__ = [
    "EnrichmentConfig",
    "GraphEnricher",
    # Modular components for advanced use
    "analysis",
    "auxiliary_visualization",
    "connection_enrichment",
    "glyph_enrichment",
    "utilities",
]

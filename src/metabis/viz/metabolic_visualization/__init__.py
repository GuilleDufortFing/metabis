"""Graph plotting utilities for metabolic network visualization.

Graph analysis produces a :class:`GraphArtifact`. Rendering and export are two
adapters over that artifact; neither re-implements the analysis. See ADR-0006.

This is a supported public namespace (ADR-0008), and it deliberately exposes
both halves: the calculation that builds an artifact and the Bokeh adapter that
paints one. Names are therefore bound **lazily** (PEP 562), so importing the
calculation half never drags the rendering half -- and its Bokeh import -- in
behind it. Reaching a rendering name imports Bokeh exactly as before.
"""

import importlib
from typing import TYPE_CHECKING

_MODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    ".artifact": (
        "GraphArtifact",
        "GraphColorMeaning",
        "GraphRunMetadata",
        "PipelinePhaseMetric",
    ),
    ".calculation_input": ("GraphCalculationInput",),
    ".enrichment": ("EnrichmentConfig", "GraphEnricher"),
    ".errors": ("GraphPipelineError",),
    ".export": (
        "graph_audit_frame",
        "graph_audit_rows",
        "graph_auxiliary_frame",
        "graph_edge_frame",
    ),
    ".generation.generator": ("GraphGenerationConfig", "MetabolicGraphGenerator"),
    ".generation.utilities.cache_utils": (
        "DEFAULT_SUBSYSTEM_LAYOUT_ENGINE",
        "SUBSYSTEM_HUMAN1_VARIANT",
        "SUBSYSTEM_STANDARD_VARIANT",
        "get_available_subsystem_graph_resource_paths",
    ),
    ".pipeline": (
        "MetabolicGraphPipeline",
        "PipelineConfig",
        "create_metabolic_graph_pipeline",
    ),
    # The Bokeh adapter. Importing any of these imports Bokeh, which is exactly
    # why they are not bound at package import time.
    ".rendering": (
        "GraphRenderer",
        "RenderedGraph",
        "RenderingConfig",
        "create_bokeh_plot",
        "draw_graph_elements",
        "render_graph_artifact",
    ),
}

_NAME_TO_MODULE: dict[str, str] = {
    name: module for module, names in _MODULE_EXPORTS.items() for name in names
}


def __getattr__(name: str):
    """Import and cache a supported graph name on first access (PEP 562)."""
    module_path = _NAME_TO_MODULE.get(name)
    if module_path is None:
        raise AttributeError(
            f"module 'metabis.viz.metabolic_visualization' has no attribute '{name}'"
        )
    value = getattr(importlib.import_module(module_path, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    # Lazy binding is invisible to type checkers; declare the same names here.
    from .artifact import (
        GraphArtifact,
        GraphColorMeaning,
        GraphRunMetadata,
        PipelinePhaseMetric,
    )
    from .calculation_input import GraphCalculationInput
    from .enrichment import EnrichmentConfig, GraphEnricher
    from .errors import GraphPipelineError
    from .export import (
        graph_audit_frame,
        graph_audit_rows,
        graph_auxiliary_frame,
        graph_edge_frame,
    )
    from .generation.generator import GraphGenerationConfig, MetabolicGraphGenerator
    from .generation.utilities.cache_utils import (
        DEFAULT_SUBSYSTEM_LAYOUT_ENGINE,
        SUBSYSTEM_HUMAN1_VARIANT,
        SUBSYSTEM_STANDARD_VARIANT,
        get_available_subsystem_graph_resource_paths,
    )
    from .pipeline import (
        MetabolicGraphPipeline,
        PipelineConfig,
        create_metabolic_graph_pipeline,
    )
    from .rendering import (
        GraphRenderer,
        RenderedGraph,
        RenderingConfig,
        create_bokeh_plot,
        draw_graph_elements,
        render_graph_artifact,
    )

__all__ = [
    # The artifact seam every adapter consumes
    "GraphArtifact",
    "GraphColorMeaning",
    "GraphRunMetadata",
    "PipelinePhaseMetric",
    "GraphPipelineError",
    "GraphCalculationInput",
    # Graph execution
    "MetabolicGraphPipeline",
    "PipelineConfig",
    "create_metabolic_graph_pipeline",
    # Graph generation
    "MetabolicGraphGenerator",
    "GraphGenerationConfig",
    # Graph enrichment
    "GraphEnricher",
    "EnrichmentConfig",
    # Bokeh rendering adapter
    "GraphRenderer",
    "RenderedGraph",
    "RenderingConfig",
    "render_graph_artifact",
    "create_bokeh_plot",
    "draw_graph_elements",
    # Export adapter
    "graph_audit_frame",
    "graph_audit_rows",
    "graph_auxiliary_frame",
    "graph_edge_frame",
    # Packaged-subsystem resource selection (promoted from generation.utilities
    # so callers never reach into the private generation package)
    "DEFAULT_SUBSYSTEM_LAYOUT_ENGINE",
    "SUBSYSTEM_HUMAN1_VARIANT",
    "SUBSYSTEM_STANDARD_VARIANT",
    "get_available_subsystem_graph_resource_paths",
]

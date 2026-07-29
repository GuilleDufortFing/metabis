"""Main plotting interface for metabolic graph visualization.

This module provides the high-level interface for creating and rendering
metabolic network visualizations using Bokeh. It integrates graph generation,
enrichment, and visualization in a unified pipeline.
"""

from .click_behavior import add_click_behavior
from .metabolic_visualization import (
    EnrichmentConfig,
    GraphGenerationConfig,
    MetabolicGraphPipeline,
    PipelineConfig,
    RenderingConfig,
)


def plot_metabolic_graph(
    element_name,  # Renamed from subsystem_name
    data_manager=None,
    *,
    generation_config: GraphGenerationConfig | None = None,
    enrichment_config: EnrichmentConfig | None = None,
    rendering_config: RenderingConfig | None = None,
    graph_type="subsystem",
    output_html=None,
    show_plot=True,
    enable_timing=False,
    return_metadata=False,
    add_click_interactions=True,
    enabled_toggle=None,
    subsystem_select=None,
    click_filename_prefix: str = "metabolic_element",
    provenance=None,
    source_fingerprints=None,
):
    """Create a Bokeh visualization of a metabolic graph.

    Structure, styling, and layout are configured through the three nested
    config objects the pipeline consumes. Any that is omitted takes its
    dataclass defaults, so ``plot_metabolic_graph(name, dm)`` renders a default
    subsystem plot; a caller that wants to tune colouring passes
    ``enrichment_config=EnrichmentConfig(color_metric=...)`` and so on. The flat
    per-field keyword surface (``use_human1_map``, ``color_metric``, ``width``,
    the ``show_hover_plots`` alias, …) was removed in AM7 (see the compatibility
    register); it duplicated the nested configs and translated between them in a
    second place.

    Args:
        element_name (str): Name of the subsystem or pathway module ID to visualize.
        data_manager: AnalysisWorkspace object with experimental data.
        generation_config (GraphGenerationConfig): Structure/layout options
            (``use_human1_map``, ``layout_engine``, resource paths).
        enrichment_config (EnrichmentConfig): Colouring, hover, delta, and
            highlighting options (``color_metric``, ``delta_options``,
            ``show_hover_*``, ``highlight_*``).
        rendering_config (RenderingConfig): Geometry options (``width``,
            ``height``, ``sizing_mode``, ``aspect_ratio``, ``zoom``,
            ``match_aspect``, ``fit_view_to_bounds``).
        graph_type (str): Type of graph to generate ("subsystem",
            "compartment", or "metabolic_network"). Metabolic-network plots
            keep their existing packaged-resource behavior.
        output_html (str): Path for HTML output file.
        show_plot (bool): Whether to call Bokeh `show`; server callers must use False.
        enable_timing (bool): Whether to collect per-phase pipeline timings.
        return_metadata (bool): If True, return a `(plot, metadata)` tuple.
        provenance: Live workspace freshness token, captured on the document
            thread from the loaded ``data_manager``. Supplied together with
            ``source_fingerprints`` to give the graph portable provenance; a
            structure-only plot with neither carries none. Identity is read from
            the workspace, never recomputed here (ADR-0007).
        source_fingerprints: Portable input identity the workspace already paid
            for at load, passed through to the artifact metadata unchanged.

    Returns:
        Bokeh plot object, or `(plot, metadata)` when `return_metadata=True`
    """
    pipeline_config = PipelineConfig(
        generation_config=generation_config,
        enrichment_config=enrichment_config,
        rendering_config=rendering_config,
        output_html=output_html,
        show_plot=show_plot,
        enable_timing=enable_timing,
    )

    # Create and run pipeline
    pipeline = MetabolicGraphPipeline(data_manager.metabolic_model, pipeline_config)

    plot = pipeline.run_full_pipeline(
        element_name,
        graph_type,
        data_manager=data_manager,
        provenance=provenance,
        source_fingerprints=source_fingerprints,
    )

    if add_click_interactions:
        add_click_behavior(
            plot,
            filename_prefix=click_filename_prefix,
            enabled_toggle=enabled_toggle,
            subsystem_select=subsystem_select,
        )

    if return_metadata:
        return plot, pipeline.get_run_metadata()

    return plot

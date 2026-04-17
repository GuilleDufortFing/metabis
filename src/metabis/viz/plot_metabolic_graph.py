"""Main plotting interface for metabolic graph visualization.

This module provides the high-level interface for creating and rendering
metabolic network visualizations using Bokeh. It integrates graph generation,
enrichment, and visualization in a unified pipeline.
"""

from .click_behavior import add_click_behavior
from .metabolic_visualization.pipeline import (
    MetabolicGraphPipeline,
    build_pipeline_config,
)


def plot_metabolic_graph(
    element_name,  # Renamed from subsystem_name
    data_manager=None,
    output_html=None,
    width=1000,
    height=1200,
    use_human1_map=True,
    layout_engine="dot",
    sizing_mode="fixed",
    aspect_ratio=2,
    zoom=1,
    match_aspect=False,
    fit_view_to_bounds=False,
    color_metric="log2_fc",
    delta_options=None,
    highlight_significant_reactions=False,
    highlight_significant_genes=False,
    highlight_gene_influence=False,
    graph_type="subsystem",  # New parameter
    enabled_toggle=None,
    subsystem_select=None,
    click_filename_prefix: str = "metabolic_element",
):
    """Create a Bokeh visualization of a metabolic graph.

    This function now uses the new MetabolicGraphPipeline architecture for improved
    modularity and maintainability, while maintaining the same interface.

    Args:
        element_name (str): Name of the subsystem or pathway module ID to visualize
        data_manager: DataManager object with experimental data
        output_html (str): Path for HTML output file
        width (int): Plot width
        height (int): Plot height
        use_human1_map (bool): For subsystem plots, prefer packaged curated
            Human1 layouts before falling back to packaged standard dot
            layouts or engine-based generation. For compartment plots,
            packaged Human1 layouts are required.
        layout_engine (str): GraphViz layout algorithm. For subsystem plots,
            dot matches the packaged standard layout variant; other engines
            trigger runtime generation when Human1 is not used. The Bokeh
            server keeps this at the default and does not expose it in the UI.
        sizing_mode (str): Bokeh sizing mode
        aspect_ratio (float): Plot aspect ratio
        zoom (float): Zoom level
        match_aspect (bool): Whether to maintain data aspect ratio when resizing
        fit_view_to_bounds (bool): Whether to derive viewport bounds from graph geometry
        color_metric (str): Metric for coloring nodes
        delta_options (dict): Options for subsystem delta calculation
            - 'metric': Metric to use for delta calculation ['cohens_d', 'log2_fc']
            - 'transform': Transformation to apply to delta values ['raw', 'abs', 'sign']
            - 'numerator_filter': Filter for numerator values ['significant', 'all']
            - 'denominator_filter': Filter for denominator values ['significant', 'all']
        highlight_significant_reactions (bool): Highlight significant reactions
        highlight_significant_genes (bool): Highlight significant genes
        highlight_gene_influence (bool): Highlight gene influence
        graph_type (str): Type of graph to generate ("subsystem",
            "compartment", or "metabolic_network"). Metabolic-network plots
            keep their existing packaged-resource behavior.

    Returns:
        Bokeh plot object
    """
    pipeline_config = build_pipeline_config(
        use_human1_map=use_human1_map,
        layout_engine=layout_engine,
        color_metric=color_metric,
        delta_options=delta_options,
        highlight_significant_reactions=highlight_significant_reactions,
        highlight_significant_genes=highlight_significant_genes,
        highlight_gene_influence=highlight_gene_influence,
        width=width,
        height=height,
        sizing_mode=sizing_mode,
        aspect_ratio=aspect_ratio,
        zoom=zoom,
        match_aspect=match_aspect,
        fit_view_to_bounds=fit_view_to_bounds,
        output_html=output_html,
        show_plot=True,
        enable_timing=False,
    )

    # Create and run pipeline
    pipeline = MetabolicGraphPipeline(data_manager.metabolic_model, pipeline_config)

    plot = pipeline.run_full_pipeline(
        element_name, graph_type, data_manager=data_manager
    )

    add_click_behavior(
        plot,
        filename_prefix=click_filename_prefix,
        enabled_toggle=enabled_toggle,
        subsystem_select=subsystem_select,
    )

    return plot

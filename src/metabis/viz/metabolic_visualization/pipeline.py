"""Metabolic Graph Visualization Pipeline.

This module provides a unified pipeline for metabolic graph visualization that
coordinates graph generation, enrichment, and rendering phases.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from bokeh.plotting import output_file, show

from .enrichment.enricher import EnrichmentConfig, GraphEnricher
from .enrichment.focus_instance import FocusInstance
from .generation.generator import GraphGenerationConfig, MetabolicGraphGenerator
from .generation.utilities.compartment_resources import (
    get_compartment_display_title,
    normalize_compartment_resource_id,
)
from .rendering.renderer import GraphRenderer, RenderingConfig


@dataclass(frozen=True)
class ResolvedGraphRequest:
    """Resolved graph request used by generation and enrichment."""

    generator_element_id: Optional[str]
    display_title: str
    focus_element_id: Optional[str] = None
    focus_element_type: Optional[str] = None


@dataclass
class PipelineRunState:
    """Mutable state captured for the most recent pipeline execution."""

    original_element_name: Optional[str] = None
    resolved_request: Optional[ResolvedGraphRequest] = None
    timings: Dict[str, float] = field(default_factory=dict)


def resolve_graph_request(
    metabolic_model,
    element_name: Optional[str],
    graph_type: str,
) -> ResolvedGraphRequest:
    """Resolve a user graph request into generation and focus identifiers."""
    if graph_type == "compartment":
        resource_id = normalize_compartment_resource_id(element_name)
        return ResolvedGraphRequest(
            generator_element_id=resource_id,
            display_title=get_compartment_display_title(resource_id),
            focus_element_id=resource_id,
            focus_element_type="compartment",
        )

    element_id, display_title = metabolic_model.retrieve_element_id(element_name, graph_type)

    if graph_type == "subsystem":
        return ResolvedGraphRequest(
            generator_element_id=element_id,
            display_title=display_title,
            focus_element_id=element_id,
            focus_element_type="subsystem",
        )

    return ResolvedGraphRequest(
        generator_element_id=element_id,
        display_title=display_title,
    )


@dataclass
class PipelineConfig:
    """Configuration for the metabolic graph visualization pipeline."""

    generation_config: Optional[GraphGenerationConfig] = None
    enrichment_config: Optional[EnrichmentConfig] = None
    rendering_config: Optional[RenderingConfig] = None
    output_html: Optional[str] = None
    show_plot: bool = True
    enable_timing: bool = False

    def __post_init__(self):
        """Initialize default nested configurations if not provided."""
        if self.generation_config is None:
            self.generation_config = GraphGenerationConfig()

        if self.enrichment_config is None:
            self.enrichment_config = EnrichmentConfig()

        if self.rendering_config is None:
            self.rendering_config = RenderingConfig()


def build_pipeline_config(
    *,
    generation_config: Optional[GraphGenerationConfig] = None,
    enrichment_config: Optional[EnrichmentConfig] = None,
    rendering_config: Optional[RenderingConfig] = None,
    use_human1_map: bool = True,
    layout_engine: str = "dot",
    color_metric: str = "log2_fc",
    delta_options: Optional[Dict[str, Any]] = None,
    show_hover_plots: Optional[bool] = None,
    show_hover_stats: bool = True,
    show_hover_distribution_plots: Optional[bool] = None,
    show_hover_color_reference: bool = True,
    highlight_significant_reactions: bool = False,
    highlight_significant_genes: bool = False,
    highlight_gene_influence: bool = False,
    width: int = 1000,
    height: int = 800,
    sizing_mode: str = "fixed",
    aspect_ratio: Optional[float] = 2.0,
    zoom: float = 1.0,
    match_aspect: bool = False,
    fit_view_to_bounds: bool = False,
    output_html: Optional[str] = None,
    show_plot: bool = True,
    enable_timing: bool = False,
) -> PipelineConfig:
    """Build a pipeline config from flat plotting parameters.

    Explicit nested config objects take precedence over flat keyword values.
    """
    if show_hover_distribution_plots is None:
        if show_hover_plots is None:
            show_hover_distribution_plots = True
        else:
            show_hover_distribution_plots = show_hover_plots

    if generation_config is None:
        generation_config = GraphGenerationConfig(
            use_human1_map=use_human1_map,
            layout_engine=layout_engine,
        )

    if enrichment_config is None:
        enrichment_config = EnrichmentConfig(
            color_metric=color_metric,
            delta_options=delta_options,
            show_hover_stats=show_hover_stats,
            show_hover_distribution_plots=show_hover_distribution_plots,
            show_hover_color_reference=show_hover_color_reference,
            highlight_significant_reactions=highlight_significant_reactions,
            highlight_significant_genes=highlight_significant_genes,
            highlight_gene_influence=highlight_gene_influence,
        )

    if rendering_config is None:
        rendering_config = RenderingConfig(
            width=width,
            height=height,
            sizing_mode=sizing_mode,
            aspect_ratio=aspect_ratio,
            zoom=zoom,
            match_aspect=match_aspect,
            fit_view_to_bounds=fit_view_to_bounds,
        )

    return PipelineConfig(
        generation_config=generation_config,
        enrichment_config=enrichment_config,
        rendering_config=rendering_config,
        output_html=output_html,
        show_plot=show_plot,
        enable_timing=enable_timing,
    )


class MetabolicGraphPipeline:
    """Pipeline for coordinating metabolic graph visualization phases.

    This class provides a unified interface for the three main phases:
    1. Graph Generation - Creating the basic graph structure
    2. Graph Enrichment - Adding data-driven styling and highlighting
    3. Graph Rendering - Creating the final Bokeh visualization
    """

    def __init__(self, metabolic_model, config: Optional[PipelineConfig] = None):
        """Initialize the pipeline.

        Args:
            metabolic_model: The metabolic model to use for graph generation
            config: Pipeline configuration. If None, defaults will be used.
        """
        self.metabolic_model = metabolic_model
        self.config = config or PipelineConfig()

        # Initialize the three main components
        self.generator = MetabolicGraphGenerator(
            metabolic_model, self.config.generation_config
        )
        self.enricher = GraphEnricher(self.config.enrichment_config)
        self.renderer = GraphRenderer(self.config.rendering_config)

        # Storage for intermediate results
        self._raw_graph_data = None
        self._enriched_graph_data = None
        self._plot = None
        self._last_run_state = PipelineRunState()

    def run_full_pipeline(self, element_name: str, graph_type: str, data_manager=None):
        """Run the complete visualization pipeline.

        Args:
            element_name: Name of the subsystem, compartment, or filter to visualize
            graph_type: Type of graph to generate
                       ("subsystem", "compartment", "metabolic_network")
            data_manager: DataManager object with experimental data

        Returns:
            Bokeh plot object
        """
        config = self.config
        self._last_run_state = PipelineRunState(original_element_name=element_name)

        t_start = time.time()

        try:
            resolved_request = resolve_graph_request(
                self.metabolic_model,
                element_name,
                graph_type,
            )
            focus_instance = FocusInstance(
                self.metabolic_model,
                resolved_request.focus_element_id,
                element_type=resolved_request.focus_element_type,
            )
            self._last_run_state.resolved_request = resolved_request

        except ValueError as e:
            # Handle element not found with helpful error message
            raise ValueError(f"Element '{element_name}' not found: {e}")

        # Phase 1: Generate graph
        glyphs, connections, display_title = self.generate_graph(
            resolved_request.generator_element_id,
            graph_type,
        )

        # Phase 2: Enrich graph
        enriched_glyphs, enriched_connections, auxiliary_data = self.enrich_graph(
            glyphs,
            connections,
            focus_instance,
            data_manager=data_manager,
            graph_type=graph_type,
        )

        # Phase 3: Render graph
        final_display_title = resolved_request.display_title or display_title
        plot = self.render_graph(
            enriched_glyphs,
            enriched_connections,
            auxiliary_data,
            final_display_title,
        )

        self._handle_output(plot, config)

        # Total timing
        if config.enable_timing:
            total_time = time.time() - t_start
            self._record_timing("total_pipeline", total_time)

        return plot

    def _handle_output(self, plot: Any, config: PipelineConfig) -> None:
        """Write and/or display the rendered plot for a pipeline run."""
        if config.output_html:
            output_file(config.output_html)

        if config.show_plot:
            show(plot)

    def generate_graph(
        self,
        element_id: str,
        graph_type: str,
    ) -> Tuple[Any, Any, str]:
        """Phase 1: Generate the basic graph structure.

        Args:
            element_id: Resolved identifier or filter used for generation
            graph_type: Type of graph to generate ("subsystem" or "pathway")

        Returns:
            Tuple of (glyphs, connections, display_title)
        """
        config = self.config

        if config.enable_timing:
            t_start = time.time()

        glyphs, connections, display_title = self.generator.create_graph(element_id, graph_type)

        # Store intermediate results
        self._raw_graph_data = (glyphs, connections, display_title)

        if config.enable_timing:
            elapsed = time.time() - t_start
            self._record_timing("generate_graph", elapsed)

        return glyphs, connections, display_title

    def enrich_graph(
        self,
        glyphs,
        connections,
        focus_instance: FocusInstance,
        data_manager=None,
        graph_type: str = None,
    ) -> Tuple[Any, Any, Any]:
        """Phase 2: Enrich graph with data-driven styling and highlighting.

        Args:
            glyphs: Raw glyph data from generation phase
            connections: Raw connection data from generation phase
            focus_instance: object that retrieves the IDs of the elements being visualized
            data_manager: DataManager object with experimental data
            graph_type: Type of graph being generated ("subsystem", "compartment", etc.)

        Returns:
            Tuple of (enriched_glyphs, enriched_connections, auxiliary_data)
        """
        config = self.config

        if config.enable_timing:
            t_start = time.time()

        enriched_glyphs, enriched_connections, auxiliary_data = self.enricher.enrich(
            glyphs=glyphs,
            connections=connections,
            focus_instance=focus_instance,
            data_manager=data_manager,
            config_override=config.enrichment_config,
            graph_type=graph_type,
        )

        # Store intermediate results
        self._enriched_graph_data = (
            enriched_glyphs,
            enriched_connections,
            auxiliary_data,
        )

        if config.enable_timing:
            elapsed = time.time() - t_start
            self._record_timing("enrich_graph", elapsed)

        return enriched_glyphs, enriched_connections, auxiliary_data

    def render_graph(
        self,
        enriched_glyphs,
        enriched_connections,
        auxiliary_data,
        display_title: str,
    ):
        """Phase 3: Create the final Bokeh visualization.

        Args:
            enriched_glyphs: Enriched glyph data from enrichment phase
            enriched_connections: Enriched connection data from enrichment phase
            auxiliary_data: Auxiliary data from enrichment phase
            display_title: Title for the plot

        Returns:
            Bokeh plot object
        """
        config = self.config

        if config.enable_timing:
            t_start = time.time()

        plot = self.renderer.render(
            enriched_glyphs=enriched_glyphs,
            enriched_connections=enriched_connections,
            auxiliary_data=auxiliary_data,
            display_title=display_title,
            config_override=config.rendering_config,
        )

        # Store final result
        self._plot = plot

        if config.enable_timing:
            elapsed = time.time() - t_start
            self._record_timing("render_graph", elapsed)

        return plot

    def get_intermediate_results(self) -> Dict[str, Any]:
        """Get intermediate results from pipeline execution.

        Returns:
            Dictionary containing intermediate results from each phase
        """
        resolved_request = None
        timings = {}
        if self._last_run_state is not None:
            resolved_request = self._last_run_state.resolved_request
            timings = dict(self._last_run_state.timings)

        return {
            "raw_graph_data": self._raw_graph_data,
            "enriched_graph_data": self._enriched_graph_data,
            "final_plot": self._plot,
            "resolved_request": resolved_request,
            "timings": timings,
        }

    def get_timings(self) -> Dict[str, float]:
        """Get timing information for pipeline phases.

        Returns:
            Dictionary of phase names to execution times in seconds
        """
        if self._last_run_state is None:
            return {}

        return dict(self._last_run_state.timings)

    def _record_timing(self, phase_name: str, elapsed: float) -> None:
        """Persist a phase timing on the current run state."""
        if self._last_run_state is None:
            self._last_run_state = PipelineRunState()

        self._last_run_state.timings[phase_name] = elapsed


def create_metabolic_graph_pipeline(
    metabolic_model,
    config: Optional[PipelineConfig] = None,
    **config_params,
) -> MetabolicGraphPipeline:
    """Create a MetabolicGraphPipeline with the specified configuration.

    Args:
        metabolic_model: The metabolic model to use
        config: Explicit pipeline configuration object.
        **config_params: Flat configuration parameters translated into nested configs.

    Returns:
        Configured MetabolicGraphPipeline instance
    """
    if config is not None and config_params:
        raise TypeError("Pass either 'config' or flat configuration parameters, not both")

    if config is None:
        config = build_pipeline_config(**config_params)

    return MetabolicGraphPipeline(metabolic_model, config)

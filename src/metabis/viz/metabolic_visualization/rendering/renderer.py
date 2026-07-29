"""Bokeh rendering adapter for metabolic graphs.

Rendering translates an artifact into Bokeh models. It performs no scientific
calculation: colour, alpha, label, and hover content were all decided during
enrichment and recorded on the artifact. This module is the only place in a graph
run that may construct a Bokeh model, and it may only run on the document thread.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Optional

from ..artifact import GraphArtifact, GraphRunMetadata, PipelinePhaseMetric
from ..errors import GraphPipelineError
from .bokeh_plotting import create_bokeh_plot, draw_graph_elements


@dataclass(frozen=True)
class RenderingConfig:
    """Configuration for graph rendering operations."""

    width: int = 1000
    height: int = 800
    sizing_mode: str = "fixed"
    aspect_ratio: Optional[float] = 2.0
    zoom: float = 1.0
    match_aspect: bool = False
    fit_view_to_bounds: bool = False

    def to_plot_params(self) -> dict:
        """Convert config to parameters for plot creation."""
        return {
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "sizing_mode": self.sizing_mode,
            "zoom": self.zoom,
            "match_aspect": self.match_aspect,
            "fit_view_to_bounds": self.fit_view_to_bounds,
        }


class GraphRenderer:
    """Class for rendering metabolic graphs using Bokeh.

    This class provides a clean interface for the rendering phase of the
    metabolic visualization pipeline, wrapping the existing Bokeh plotting
    functions in a structured, configurable way.
    """

    def __init__(self, config: Optional[RenderingConfig] = None):
        """Initialize the graph renderer.

        Args:
            config: Configuration for rendering operations.
                   If None, defaults will be used.
        """
        self.config = config or RenderingConfig()

    def render(
        self,
        enriched_glyphs: Any,
        enriched_connections: Any,
        auxiliary_data: Any,
        display_title: str,
        config_override: Optional[RenderingConfig] = None,
    ) -> Any:
        """Render enriched graph data as a Bokeh plot.

        Args:
            enriched_glyphs: Enriched graph glyphs
            enriched_connections: Enriched graph connections
            auxiliary_data: Additional data from enrichment phase
            display_title: Title for the plot
            config_override: Optional config to override instance config

        Returns:
            Bokeh plot object
        """
        # Use override config if provided, otherwise use instance config
        active_config = config_override or self.config

        # Create the Bokeh plot
        plot = self._create_plot(
            enriched_glyphs,
            enriched_connections,
            auxiliary_data,
            display_title,
            active_config,
        )

        # Draw graph elements on the plot
        self._draw_elements(plot, enriched_glyphs, enriched_connections, auxiliary_data)

        return plot

    def _create_plot(
        self,
        glyphs: Any,
        connections: Any,
        auxiliary_data: Any,
        display_title: str,
        config: RenderingConfig,
    ) -> Any:
        """Create the base Bokeh plot.

        Args:
            glyphs: Graph glyphs for plot sizing
            connections: Graph connections for viewport sizing
            auxiliary_data: Auxiliary geometry used in the rendered plot
            display_title: Title for the plot
            config: Rendering configuration

        Returns:
            Bokeh plot object
        """
        return create_bokeh_plot(
            display_title=display_title,
            glyphs=glyphs,
            connections=connections,
            auxiliary_data=auxiliary_data,
            **config.to_plot_params(),
        )

    def _draw_elements(
        self,
        plot: Any,
        enriched_glyphs: Any,
        enriched_connections: Any,
        auxiliary_data: Any,
    ) -> None:
        """Draw graph elements on the plot.

        Args:
            plot: Bokeh plot object
            enriched_glyphs: Enriched graph glyphs
            enriched_connections: Enriched graph connections
            auxiliary_data: Additional data from enrichment phase
        """
        draw_graph_elements(
            plot=plot,
            enriched_glyphs=enriched_glyphs,
            enriched_connections=enriched_connections,
            auxiliary_data=auxiliary_data,
        )

    def update_config(self, **kwargs) -> None:
        """Update rendering configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

    def get_config(self) -> RenderingConfig:
        """Get current rendering configuration.

        Returns:
            Current RenderingConfig instance
        """
        return self.config


@dataclass(frozen=True)
class RenderedGraph:
    """A Bokeh plot and the telemetry of the run that produced it."""

    plot: Any
    metadata: GraphRunMetadata


def render_graph_artifact(
    artifact: GraphArtifact,
    config: Optional[RenderingConfig] = None,
) -> RenderedGraph:
    """Build Bokeh models from a graph artifact, on the document thread.

    Bokeh will not accept the artifact's read-only payload, so it is copied into
    the mutable containers Bokeh requires here, at the adapter boundary, rather
    than by weakening the artifact everybody else shares.
    """
    started = perf_counter()
    try:
        plot = GraphRenderer(config or RenderingConfig()).render(
            enriched_glyphs=list(artifact.nodes),
            enriched_connections=list(artifact.edges),
            auxiliary_data={
                key: dict(value) for key, value in artifact.auxiliary.items()
            },
            display_title=artifact.display_title,
        )
    except Exception as exc:
        raise GraphPipelineError(
            f"Graph rendering failed: {exc}",
            phase="render",
            resource_path=artifact.resource_path,
            element_id=artifact.element_id,
        ) from exc

    render_phase = PipelinePhaseMetric(
        "render_graph",
        perf_counter() - started,
        item_count=artifact.node_count + artifact.edge_count,
    )
    total_phase = PipelinePhaseMetric(
        "total_pipeline",
        sum(phase.elapsed_seconds for phase in artifact.phases)
        + render_phase.elapsed_seconds,
    )
    try:
        model_count = len(plot.references())
    except Exception:
        model_count = None

    return RenderedGraph(
        plot=plot,
        metadata=GraphRunMetadata.from_artifact(
            artifact,
            bokeh_model_count=model_count,
            extra_phases=(render_phase, total_phase),
        ),
    )

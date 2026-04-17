"""Graph rendering class for metabolic visualization.

This module provides a class-based interface for rendering metabolic graphs
using Bokeh, handling plot creation, styling, and element drawing.
"""

from dataclasses import dataclass
from typing import Any, Optional

from .bokeh_plotting import create_bokeh_plot, draw_graph_elements


@dataclass
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

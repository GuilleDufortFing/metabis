"""Shared gradient color references for graph and ranking hovers.

Every gradient plot shows the distribution that supplies its colours and the
inspected element's position within that distribution. Graph hovers render a
standalone panel. Dot plots keep one static histogram SVG on their ``HoverTool``
and store only a compact marker coordinate and value per point, so a large
ranking does not duplicate the SVG in every data-source row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .plot_histograms import create_histogram_background, render_histogram_with_marker

DEFAULT_REFERENCE_WIDTH = 250
DEFAULT_REFERENCE_HEIGHT = 150


def color_reference_panel(body_html: str) -> str:
    """Wrap a rendered reference in the labelled chrome the hovers use.

    The panel is deliberately identical to the one
    ``ColorExplanationSection.build`` emits, so a user who has learned to read the
    reference on a graph node reads the same thing on a dot plot.
    """
    if not body_html:
        return ""
    return (
        "<div style='margin-top:8px; border-top:1px solid #ddd; padding-top:8px;'>"
        "<div><strong>Color Reference:</strong></div>"
        "<div style='display:flex; justify-content:center; margin-top:8px;'>"
        f"{body_html}</div></div>"
    )


@dataclass(frozen=True)
class GradientColorReference:
    """One gradient's distribution, rendered once and marked per element.

    ``background`` is the static SVG produced from the registered distribution's
    histogram: the same bins, in the same colors, that painted the glyphs. Each
    call to :meth:`panel_for_value` adds only the marker for one element.
    """

    metric_label: str
    background: Mapping[str, Any]

    def html_for_value(self, value: float | None) -> str:
        """The reference SVG with ``value`` marked, without the outer chrome."""
        return render_histogram_with_marker(dict(self.background), value)

    def panel_for_value(self, value: float | None) -> str:
        """The reference for one element, ready to append to a hover."""
        return color_reference_panel(self.html_for_value(value))

    def marker_x(self, value: float | None) -> float:
        """Return the SVG x coordinate for a compact per-point marker."""
        params = self.background["scaling_params"]
        numeric_value = 0.0
        if value is not None:
            try:
                candidate = float(value)
            except (TypeError, ValueError):
                candidate = 0.0
            if np.isfinite(candidate):
                numeric_value = candidate
        return float(
            params["margin_left"]
            + (
                (numeric_value - params["plot_x_min"])
                / params["plot_x_range"]
                * params["plot_area_width"]
            )
        )

    def hover_template(self, *, marker_x_field: str, value_field: str) -> str:
        """Build one shared Bokeh tooltip template for all points in a plot.

        The static histogram SVG lives on the ``HoverTool`` once. Bokeh
        substitutes only the selected row's compact marker coordinate and value,
        avoiding a complete SVG copy in every ``ColumnDataSource`` row.
        """
        params = self.background["scaling_params"]
        marker_x = f"@{marker_x_field}"
        marker_line = f"""
    <line x1="{marker_x}" y1="{params["margin_top"]}" x2="{marker_x}"
          y2="{params["margin_top"] + params["plot_area_height"]}"
          stroke="#56565E" stroke-width="2" opacity="0.9"/>
    """
        current_value_label = f"""
    <text x="{marker_x}" y="{params["margin_top"] - 5}" text-anchor="middle"
          font-size="9" fill="#56565E" font-weight="bold">
          Current: @{value_field}{{0.000}}</text>
    """
        svg = self.background["svg_content"].replace(
            "<!-- MARKER_PLACEHOLDER -->", marker_line
        )
        svg = svg.replace(
            "<!-- CURRENT_VALUE_LABEL_PLACEHOLDER -->", current_value_label
        )
        body = (
            f'<div style="width:{self.background["width"]}px;'
            f'height:{self.background["height"]}px;font-family:Arial,sans-serif;'
            'font-size:11px;background:white;border:1px solid #ddd;'
            f'border-radius:4px;">{svg}</div>'
        )
        return color_reference_panel(body)


def build_gradient_color_reference(
    histogram: Sequence[Mapping[str, Any]] | None,
    *,
    metric_label: str,
    threshold: float | None = None,
    symmetric_threshold: bool = True,
    width: int = DEFAULT_REFERENCE_WIDTH,
    height: int = DEFAULT_REFERENCE_HEIGHT,
) -> GradientColorReference | None:
    """Build the reference for one registered distribution, or ``None``.

    Args:
        histogram: The ``"histogram"`` entry of
            ``ColorManager.get_distribution_details``: one dict per bin carrying
            its count, its value bounds, and the color that bin paints.
        metric_label: What the encoded value is, in the user's words. It titles
            the reference and labels its x-axis.
        threshold: Optional significance threshold to draw on the distribution.
        symmetric_threshold: Whether the threshold applies in both directions.
            A one-sided metric such as a p-value passes ``False``.

    Returns:
        The reference, or ``None`` when the distribution is empty -- a caller with
        nothing registered gets no panel rather than an empty frame.
    """
    if not histogram:
        return None
    background = create_histogram_background(
        histogram_data=list(histogram),
        metric_name=metric_label,
        width=width,
        height=height,
        threshold=threshold,
        symmetric_threshold=symmetric_threshold,
    )
    if not background:
        return None
    return GradientColorReference(metric_label=metric_label, background=background)


def build_reference_from_distribution(
    color_manager: Any,
    *,
    mode: Any,
    element_type: str,
    element_id: Any,
    metric_name: str,
    metric_label: str,
    threshold: float | None = None,
    symmetric_threshold: bool = True,
) -> GradientColorReference | None:
    """Build the reference from a colour registry, given any member element.

    The histogram belongs to the distribution, not to the element, so any
    registered ``element_id`` yields the same bars; the caller marks each element
    itself through :meth:`GradientColorReference.panel_for_value`. Returns
    ``None`` when no distribution is registered under the key.
    """
    if color_manager is None:
        return None
    try:
        details = color_manager.get_distribution_details(
            mode=mode,
            element_type=element_type,
            element_id=element_id,
            metric_name=metric_name,
        )
    except Exception:
        # A missing distribution must cost the plot its reference, never its
        # render: the glyphs are still correct without the panel.
        return None
    return build_gradient_color_reference(
        details.get("histogram"),
        metric_label=metric_label,
        threshold=threshold,
        symmetric_threshold=symmetric_threshold,
    )

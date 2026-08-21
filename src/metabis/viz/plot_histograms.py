"""Histogram plotting module for hover tooltips.

This module provides simplified histogram generation logic that works directly
with pre-calculated histogram data from ColorManager. It generates lightweight
SVG visualizations suitable for hover tooltips.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from bokeh.models import Label
from bokeh.plotting import figure

from ..visual_config import theme


def create_color_reference_histogram(
    histogram_data: List[Dict[str, Any]],
    current_value: float,
    metric_name: str = "Metric",
    width: int = 250,
    height: int = 150,
    threshold: Optional[float] = None,
    not_significant_color: Optional[str] = None,
    symmetric_threshold: bool = True,
    log_scale_y: bool = False,
) -> str:
    """Create a fast HTML/SVG histogram showing metric distribution.

    This function generates an SVG histogram using pre-calculated bin data.
    It is designed to be paired with ColorManager.get_distribution_details().

    Args:
        histogram_data: List of dicts containing 'bin', 'count', 'color'.
                       Typically provided by ColorManager.get_distribution_details().
        current_value: Value of the current element to highlight.
        metric_name: Human-readable name for the metric.
        width: Plot width in pixels.
        height: Plot height in pixels.
        threshold: Optional threshold value for significance. Bars within
                  [-threshold, threshold] will be colored as not significant.
        not_significant_color: Color for non-significant bars.
                             Defaults to theme.NOT_SIGNIFICANT.
        symmetric_threshold: Whether to draw symmetric thresholds (-t and +t).
                           Defaults to True. If False, only draws +t.
        log_scale_y: Whether to use log scale for y-axis (counts).

    Returns:
        str: HTML div containing the SVG histogram plot.
    """
    try:
        background = create_histogram_background(
            histogram_data=histogram_data,
            current_value=current_value,
            metric_name=metric_name,
            width=width,
            height=height,
            threshold=threshold,
            not_significant_color=not_significant_color,
            symmetric_threshold=symmetric_threshold,
            log_scale_y=log_scale_y,
        )

        if not background:
            return _create_error_div(width, height, "No data available")

        return render_histogram_with_marker(background, current_value)

    except Exception as e:
        return _create_error_div(width, height, f"Histogram error: {str(e)}")


def create_histogram_background(
    histogram_data: List[Dict[str, Any]],
    current_value: Optional[float] = None,
    metric_name: str = "Metric",
    width: int = 250,
    height: int = 150,
    threshold: Optional[float] = None,
    not_significant_color: Optional[str] = None,
    symmetric_threshold: bool = True,
    log_scale_y: bool = False,
) -> Optional[Dict[str, Any]]:
    """Create the static background SVG for a histogram (bars, axes, grid).

    Args:
        histogram_data: List of dicts containing 'bin', 'count', 'color'.
        current_value: Optional value to ensure is included in the x-range.
                      This ensures the current value marker will be visible.
    """
    if not histogram_data:
        return None

    if not_significant_color is None:
        not_significant_color = theme.NOT_SIGNIFICANT

    # Sort by bin index to ensure correct order
    sorted_data = sorted(histogram_data, key=lambda x: x['bin'])

    # Check if we have value-based data
    use_values = 'value_center' in sorted_data[0] if sorted_data else False

    if use_values:
        bin_centers = [d['value_center'] for d in sorted_data]
        x_min = min(d['value_start'] for d in sorted_data)
        x_max = max(d['value_end'] for d in sorted_data)

        if not symmetric_threshold:
            x_min = max(x_min, 0)

        # If range is 0 (single point), add padding
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5

        bin_width = None
    else:
        bin_centers = [d['bin'] for d in sorted_data]
        x_min = min(bin_centers)
        x_max = max(bin_centers)

        if not symmetric_threshold:
            x_min = max(x_min, 0)

        bin_width = 1.0

    # Extend range to include current_value if provided (ensures marker is visible)
    if current_value is not None and not np.isnan(current_value):
        if current_value < x_min:
            x_min = current_value
        if current_value > x_max:
            x_max = current_value

    hist = [d['count'] for d in sorted_data]
    bin_colors = [d['color'] for d in sorted_data]

    # Apply log scale to counts if requested
    if log_scale_y:
        hist = np.log1p(hist)

    # Calculate plot range with padding
    x_range = x_max - x_min

    if use_values:
        x_padding = x_range * 0.05 if x_range > 0 else 0.5
    else:
        x_padding = 0.5  # Half bin width padding

    plot_x_min = x_min - x_padding
    plot_x_max = x_max + x_padding
    plot_x_range = plot_x_max - plot_x_min

    # Avoid division by zero
    if plot_x_range == 0:
        plot_x_range = 1.0

    y_max = np.max(hist)
    if y_max == 0:
        y_max = 1

    # SVG dimensions and margins
    margin_left, margin_right = 40, 20
    margin_top, margin_bottom = 40, 40
    plot_area_width = width - margin_left - margin_right
    plot_area_height = height - margin_top - margin_bottom

    # Scaling functions
    def scale_x(val):
        return margin_left + ((val - plot_x_min) / plot_x_range * plot_area_width)

    def scale_y(val):
        return margin_top + plot_area_height - (val / y_max * plot_area_height)

    def scale_width(w):
        return w / plot_x_range * plot_area_width

    # Convert colors to RGB with alpha
    def hex_to_rgba(hex_color, alpha=1.0):
        if not hex_color:
            return "rgba(0,0,0,1)"
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 6:
            try:
                r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
                return f"rgba({r},{g},{b},{alpha})"
            except ValueError:
                return hex_color
        return hex_color

    # Generate histogram bars SVG
    histogram_bars = ""
    for i, (center, bar_count, color) in enumerate(zip(bin_centers, hist, bin_colors)):
        if bar_count > 0:
            if use_values:
                d = sorted_data[i]
                bar_start = d['value_start']
                bar_end = d['value_end']
                bar_w_val = bar_end - bar_start

                if bar_w_val <= 0:
                    # Handle zero width bin (e.g. bin 0)
                    bar_x = scale_x(bar_start)
                    bar_width = 1 # 1 pixel minimum
                    bar_x -= 0.5
                else:
                    bar_x = scale_x(bar_start)
                    bar_width = scale_width(bar_w_val)
            else:
                bar_x = scale_x(center - bin_width / 2)
                bar_width = scale_width(bin_width)

            bar_y = scale_y(bar_count)
            bar_height = plot_area_height - (bar_y - margin_top)

            # Determine fill and stroke based on significance
            fill_color = color
            stroke_color = "#999595"  # Visible border for light colors

            # Apply threshold logic if provided
            if threshold is not None:
                # Check if bin center is within threshold (not significant)
                # Note: This assumes threshold is positive
                abs_threshold = abs(threshold)
                if abs(center) < abs_threshold:
                    fill_color = not_significant_color

            histogram_bars += f'''
            <rect x="{bar_x}" y="{bar_y}" width="{bar_width}" height="{bar_height}"
                  fill="{hex_to_rgba(fill_color)}" stroke="{stroke_color}" stroke-width="0.5"/>
            '''

    # Threshold lines
    threshold_lines = ""
    if threshold is not None and threshold != 0:
        t_vals = [-abs(threshold), abs(threshold)] if symmetric_threshold else [threshold]
        for t_val in t_vals:
            if plot_x_min <= t_val <= plot_x_max:
                t_x = scale_x(t_val)
                threshold_lines += f'''
                <line x1="{t_x}" y1="{margin_top}" x2="{t_x}"
                      y2="{margin_top + plot_area_height}"
                      stroke="{theme.THRESHOLD_LINE_COLOR}" stroke-width="{theme.THRESHOLD_LINE_WIDTH}" stroke-dasharray="{theme.THRESHOLD_SVG_DASH}"/>
                '''

    # Calculate axis labels
    n_x_ticks = min(5, len(bin_centers))
    if n_x_ticks > 1:
        x_tick_indices = np.linspace(0, len(bin_centers) - 1, n_x_ticks, dtype=int)
    else:
        x_tick_indices = [0] if len(bin_centers) > 0 else []

    x_ticks = ""
    for idx in x_tick_indices:
        if idx < len(bin_centers):
            val = bin_centers[idx]
            tick_x = scale_x(val)
            tick_y = margin_top + plot_area_height
            x_ticks += f'''
            <line x1="{tick_x}" y1="{tick_y}" x2="{tick_x}" y2="{tick_y + 5}"
                  stroke="#333" stroke-width="1"/>
            <text x="{tick_x}" y="{tick_y + 18}" text-anchor="middle"
                  font-size="9" fill="#666">{val:.2f}</text>
            '''

    # Y-axis ticks
    n_y_ticks = 3
    y_tick_values = np.linspace(0, y_max, n_y_ticks)
    y_ticks = ""
    for tick_val in y_tick_values:
        tick_y = scale_y(tick_val)
        y_ticks += f'''
        <line x1="{margin_left - 5}" y1="{tick_y}" x2="{margin_left}" y2="{tick_y}"
              stroke="#333" stroke-width="1"/>
        <text x="{margin_left - 8}" y="{tick_y + 3}" text-anchor="end"
              font-size="9" fill="#666">{int(tick_val)}</text>
        '''

    y_label = "Log(Count)" if log_scale_y else "Count"

    # Axis labels
    display_metric_name = metric_name
    if "neg_log_pval" in metric_name:
        display_metric_name = "-Log(p-value)"
    elif "neg_log_adj_pval" in metric_name:
        display_metric_name = "-Log(adj. p-value)"

    # Create the SVG content
    svg_content = f'''
        <svg width="{width}" height="{height}" style="background:white;">
            <!-- Grid lines -->
            <defs>
                <pattern id="grid" width="20" height="20"
                         patternUnits="userSpaceOnUse">
                    <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#f0f0f0"
                          stroke-width="1" opacity="0.3"/>
                </pattern>
            </defs>
            <rect width="{plot_area_width}" height="{plot_area_height}"
                  x="{margin_left}" y="{margin_top}" fill="url(#grid)"/>

            <!-- Axes -->
            <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}"
                  y2="{margin_top + plot_area_height}" stroke="#333"
                  stroke-width="1"/>
            <line x1="{margin_left}" y1="{margin_top + plot_area_height}"
                  x2="{margin_left + plot_area_width}"
                  y2="{margin_top + plot_area_height}" stroke="#333"
                  stroke-width="1"/>

            <!-- Histogram bars -->
            {histogram_bars}

            <!-- Threshold lines -->
            {threshold_lines}

            <!-- MARKER_PLACEHOLDER -->

            <!-- Axis ticks and labels -->
            {x_ticks}
            {y_ticks}

            <!-- Axis labels -->
            <text x="{margin_left + plot_area_width / 2}" y="{height - 8}"
                  text-anchor="middle" font-size="11"
                  fill="#333">{display_metric_name}</text>
            <text x="15" y="{margin_top + plot_area_height / 2}"
                  text-anchor="middle" font-size="11" fill="#333"
                  transform="rotate(-90 15 {margin_top + plot_area_height / 2})">
                  {y_label}</text>

            <!-- Title -->
            <text x="{width / 2}" y="20" text-anchor="middle"
                  font-size="12" font-weight="bold"
                  fill="#333">{display_metric_name} Distribution</text>

            <!-- CURRENT_VALUE_LABEL_PLACEHOLDER -->
        </svg>
    '''

    return {
        "svg_content": svg_content,
        "scaling_params": {
            "margin_left": margin_left,
            "margin_top": margin_top,
            "plot_x_min": plot_x_min,
            "plot_x_range": plot_x_range,
            "plot_area_width": plot_area_width,
            "plot_area_height": plot_area_height
        },
        "width": width,
        "height": height
    }


def render_histogram_with_marker(
    background_data: Dict[str, Any],
    current_value: float,
    marker_color: str = "#56565E"
) -> str:
    """Render the final histogram HTML by adding the marker to the background."""
    if not background_data:
        return '<div style="font-size:10px;color:red;">Histogram error</div>'

    svg_content = background_data["svg_content"]
    params = background_data["scaling_params"]

    # Calculate marker position
    def scale_x(val):
        return params["margin_left"] + ((val - params["plot_x_min"]) / params["plot_x_range"] * params["plot_area_width"])

    # Handle case where current_value might be None or NaN
    if current_value is None or np.isnan(current_value):
        current_value = 0.0

    current_x = scale_x(current_value)

    marker_line = f'''
    <line x1="{current_x}" y1="{params["margin_top"]}" x2="{current_x}"
          y2="{params["margin_top"] + params["plot_area_height"]}"
          stroke="{marker_color}" stroke-width="2" opacity="0.9"/>
    <polygon points="{current_x - 4},{params["margin_top"]} {current_x + 4},{params["margin_top"]}
                     {current_x},{params["margin_top"] + 8}"
             fill="{marker_color}" opacity="0.9"/>
    '''

    current_value_label = f'''
    <text x="{current_x}" y="{params["margin_top"] - 5}" text-anchor="middle"
          font-size="9" fill="{marker_color}" font-weight="bold">
          Current: {current_value:.3f}</text>
    '''

    final_svg = svg_content.replace("<!-- MARKER_PLACEHOLDER -->", marker_line)
    final_svg = final_svg.replace("<!-- CURRENT_VALUE_LABEL_PLACEHOLDER -->", current_value_label)

    return f'''
    <div style="width:{background_data["width"]}px;height:{background_data["height"]}px;font-family:Arial,sans-serif;
                font-size:11px;background:white;border:1px solid #ddd;
                border-radius:4px;">
        {final_svg}
    </div>
    '''


def create_bokeh_histogram(
    histogram_data: List[Dict[str, Any]],
    title: str,
    x_label: str,
    y_label: str = "Count",
    width: int = 500,
    height: int = 400,
    threshold: Optional[float] = None,
    not_significant_color: Optional[str] = None,
    background_fill_color: str = "#ffffff",
    show_threshold_annotation: bool = True,
    threshold_counts: Optional[tuple] = None,
) -> figure:
    """Create a Bokeh histogram figure using pre-calculated histogram data.

    Args:
        histogram_data: List of dicts containing 'bin', 'count', 'color'.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        width: Figure width in pixels.
        height: Figure height in pixels.
        threshold: Optional threshold value.
        not_significant_color: Color for non-significant bars.
        background_fill_color: Background color.
        show_threshold_annotation: Whether to show threshold annotation.
        threshold_counts: Optional tuple of (count, percentage) for annotation.

    Returns:
        Bokeh figure with histogram.
    """
    if not_significant_color is None:
        not_significant_color = theme.NOT_SIGNIFICANT

    hist_fig = figure(
        aspect_ratio=1,
        sizing_mode="stretch_width",
        title=title,
        x_axis_label=x_label,
        y_axis_label=y_label,
        background_fill_color=background_fill_color,
        width=width,
        height=height,
    )

    if not histogram_data:
        return hist_fig

    # Sort by bin index
    sorted_data = sorted(histogram_data, key=lambda x: x['bin'])

    # Check if we have value-based data
    use_values = 'value_center' in sorted_data[0] if sorted_data else False

    # Infer symmetry
    is_symmetric = True
    if "p-value" in x_label.lower() or "pval" in x_label.lower():
        is_symmetric = False

    if not is_symmetric:
        if use_values:
             sorted_data = [d for d in sorted_data if d['value_end'] > 0]
        else:
             sorted_data = [d for d in sorted_data if d['bin'] >= 0]

    if not sorted_data:
        return hist_fig

    if use_values:
        bin_centers = [d['value_center'] for d in sorted_data]
        left_edges = [d['value_start'] for d in sorted_data]
        right_edges = [d['value_end'] for d in sorted_data]
    else:
        bin_centers = [d['bin'] for d in sorted_data]
        # Bin width is 1.0 for integer bins (standard in ColorManager distributions)
        bin_width = 1.0
        # Calculate edges for quad plot
        left_edges = [c - bin_width / 2 for c in bin_centers]
        right_edges = [c + bin_width / 2 for c in bin_centers]

    hist_counts = [d['count'] for d in sorted_data]
    bin_colors = [d['color'] for d in sorted_data]

    # Apply threshold logic to colors
    final_colors = []
    for center, color in zip(bin_centers, bin_colors):
        fill_color = color
        if threshold is not None:
             # Check if bin center is within threshold (not significant)
             if abs(center) < abs(threshold):
                 fill_color = not_significant_color
        final_colors.append(fill_color)

    # Plot histogram bars
    hist_fig.quad(
        top=hist_counts,
        bottom=0,
        left=left_edges,
        right=right_edges,
        fill_color=final_colors,
        line_color="#999595",
        line_width=0.5,
        alpha=0.8,
    )

    if hist_counts:
        hist_fig.y_range.end = max(hist_counts) * 1.1

    # Add threshold lines and annotations
    if threshold is not None and threshold != 0:
        t_vals = [threshold]
        if is_symmetric:
             t_vals = [threshold, -threshold]

        for t_val in t_vals:
             hist_fig.line(
                [t_val, t_val],
                [0, hist_fig.y_range.end],
                line_color=theme.THRESHOLD_LINE_COLOR,
                line_dash=theme.THRESHOLD_LINE_DASH,
                line_width=theme.THRESHOLD_LINE_WIDTH,
            )

        if show_threshold_annotation and threshold_counts:
            count, percent = threshold_counts

            if is_symmetric:
                 annotation_text = f"|{x_label}| > {threshold:.2f}: {count} ({percent:.1f}%)"
                 x_pos = 0
            else:
                 annotation_text = f"> {threshold:.2f}: {count} ({percent:.1f}%)"
                 x_pos = threshold / 2

            annotation = Label(
                x=x_pos,
                y=hist_fig.y_range.end * 0.9,
                text=annotation_text,
                text_font_size="9pt",
                text_color="red",
                background_fill_color="white",
                background_fill_alpha=0.7,
            )
            hist_fig.add_layout(annotation)

    return hist_fig


def _create_error_div(width: int, height: int, message: str) -> str:
    """Create a simple error message div."""
    return (
        f'<div style="width:{width}px;height:{height}px;background:#f0f0f0;'
        f"display:flex;align-items:center;justify-content:center;"
        f'font-size:10px;color:red;">{message}</div>'
    )

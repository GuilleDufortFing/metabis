import base64
import io
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, show
from bokeh.transform import jitter
from scipy.stats import gaussian_kde

from .click_behavior import add_click_behavior, get_external_reference_data


def create_violin(
    dist1: list,
    dist2: list,
    color1: str,
    color2: str,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    significant: bool = False,
    width: int = 400,
    height: int = 400,
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
    border_color: str = "#1f77b4",  # Default blue border
    is_reversed: bool = False,  # Flag for reversed reactions
    hover_text: str = None,  # Added parameter for hover text
    element_id: str = "",
    display_label: str = "",
    external_url: str = "",
) -> figure:
    """Create an interactive violin plot using Bokeh."""
    # Calculate means for flow direction analysis
    mean1 = np.mean(dist1) if len(dist1) > 0 else 0
    mean2 = np.mean(dist2) if len(dist2) > 0 else 0

    # Check if reaction has reversed flow (means have opposite signs)
    reversed_flow = (mean1 * mean2) < 0

    # Prepare data for scatter points - ADD NUMERIC X VALUES
    data_dict = {
        "value": dist1 + dist2,
        "group": [group1_name] * len(dist1) + [group2_name] * len(dist2),
        "color": [color1] * len(dist1) + [color2] * len(dist2),
        "x_pos": [1] * len(dist1)
        + [2] * len(dist2),  # Add numeric positions for jittering
    }

    source = ColumnDataSource(data_dict)

    # Calculate statistics for violin curves
    curves = {}
    group_positions = {group1_name: 1, group2_name: 2}

    # Calculate reasonable y-range for the plot
    all_values = np.array(dist1 + dist2)
    if len(all_values) > 0:
        # Use percentiles instead of min/max to avoid extreme outliers affecting the plot
        y_min = min(all_values)
        y_max = max(all_values)

        # Ensure zero is included in the range if there's reversed flow
        if reversed_flow:
            y_min = min(y_min, 0)
            y_max = max(y_max, 0)

        # Add padding (10% on each side)
        y_range_size = y_max - y_min
        y_min -= 0.5 * y_range_size
        y_max += 0.5 * y_range_size
    else:
        y_min, y_max = -1, 1  # Default range if no data

    for group, dist in [(group1_name, dist1), (group2_name, dist2)]:
        if len(dist) > 0:
            # Use the calculated y-range for KDE instead of the full data range
            y_values = np.linspace(y_min, y_max, 100)

            kernel = gaussian_kde(dist)
            density = kernel(y_values)
            density = density / density.max() * 0.4
            x_pos = group_positions[group]

            curves[group] = {
                "x": np.concatenate([x_pos + density, x_pos - density[::-1]]),
                "y": np.concatenate([y_values, y_values[::-1]]),
            }

    y_min = min(curves[group]["y"])
    y_max = max(curves[group]["y"])

    # Invert y-axis if mean is negative
    if mean1 < 0:
        y_range = (y_max, y_min)
    else:
        y_range = (y_min, y_max)

    # Create figure with explicit y_range
    tools = "pan,box_zoom,wheel_zoom,reset,save"

    # Add HoverTool if hover text is provided
    if hover_text:
        tools += ",hover"

    p = figure(
        aspect_ratio=0.8,
        sizing_mode="scale_width",
        title=title,
        x_axis_label=x_label,
        y_axis_label=y_label,
        x_range=(0.5, 2.5),
        y_range=y_range,
        y_axis_location="left",
        tools=tools,
    )

    # Add violin curves with improved styling
    for group, color in [(group1_name, color1), (group2_name, color2)]:
        if group in curves:
            p.patch(
                curves[group]["x"],
                curves[group]["y"],
                fill_color=color,
                fill_alpha=0.6,
                line_color="gray",
                line_alpha=0.2,
                line_width=0.5,
            )

    # Add jittered points
    p.circle(
        x=jitter("x_pos", width=0.2, range=p.x_range),
        y="value",
        source=source,
        radius=0.01,
        alpha=0.5,
        color="black",
        line_alpha=0,
    )

    # Add box plots
    for group, dist, x, color in [
        (group1_name, dist1, 1, color1),
        (group2_name, dist2, 2, color2),
    ]:
        if len(dist) > 0:
            qmin, q1, q2, q3, qmax = np.percentile(dist, [0, 25, 50, 75, 100])
            # Box
            p.vbar(x=x, bottom=q1, top=q3, width=0.1, fill_color="black", alpha=0.5)
            # Median line
            p.line([x - 0.05, x + 0.05], [q2, q2], line_color="white", line_width=2)
            # Whiskers
            p.line([x, x], [qmin, qmax], line_color="black", line_width=1)
            p.line([x - 0.05, x + 0.05], [qmin, qmin], line_color="black", line_width=1)
            p.line([x - 0.05, x + 0.05], [qmax, qmax], line_color="black", line_width=1)

    # Add zero line if reaction has reversed flow
    if reversed_flow:
        p.line(
            [0.5, 2.5],
            [0, 0],
            line_color="red",
            line_dash="dashed",
            alpha=0.7,
        )

    # Customize appearance
    p.grid.grid_line_color = "white"
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.minor_tick_line_color = None
    p.outline_line_color = None

    # Set x-axis ticks with group names and rotate labels by 30 degrees
    p.xaxis.ticker = [1, 2]
    p.xaxis.major_label_overrides = {1: group1_name, 2: group2_name}
    p.xaxis.major_label_orientation = np.pi / 6  # 30 degrees in radians

    # Add borders based on significance and reversed status
    if significant:
        if is_reversed:
            # For reversed significant reactions: yellow outer border
            p.outline_line_color = "#FFCC00"  # Bright yellow
            p.outline_line_width = 3
            p.outline_line_alpha = 0.8
        else:
            p.outline_line_color = border_color  # Use the metric-based color
            p.outline_line_width = 5
            p.outline_line_alpha = 1

    # Create a transparent rectangle that covers the entire plot area for hover
    if hover_text:
        # Create a ColumnDataSource with hover text for the rectangle
        rect_source = ColumnDataSource(
            {
                "x": [0.5],  # x_min of the plot
                "y": [y_range[0]],  # y_min of the plot
                "width": [2 * 2.0],  # width of the plot (from 0.5 to 2.5)
                "height": [2 * (y_range[1] - y_range[0])],  # height of the plot
                "hover_text": [hover_text],  # hover text for the plot
                "element_id": [element_id or title],
                "display_label": [display_label or title],
                "external_url": [external_url],
            }
        )

        # Add a transparent rectangle that spans the entire plot area
        p.rect(
            x="x",
            y="y",
            width="width",
            height="height",
            source=rect_source,
            fill_color=None,  # Transparent fill
            fill_alpha=0,  # Fully transparent
            line_color=None,  # No border
            line_alpha=0,  # Fully transparent border
            hover_line_color=None,  # No line on hover
            hover_fill_color=None,  # No fill on hover
        )

        # Configure the HoverTool with appropriate tooltip
        hover = p.select_one(HoverTool)
        if hover:
            hover.tooltips = """
            <div>
                @hover_text{safe}
            </div>
            """
            hover.attachment = "vertical"
            hover.point_policy = "follow_mouse"
            # Set the hover tool to only work with the rectangle
            hover.renderers = [p.renderers[-1]]  # Use the last renderer (the rectangle)

    return p


def create_html_violin_for_hover(
    dist1: list,
    dist2: list,
    color1: str,
    color2: str,
    x_label: str = "",
    y_label: str = "",
    plot_title: str = "",
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
    size: Optional[Tuple[int, int]] = None,
    min_y: Optional[float] = None,
    max_y: Optional[float] = None,
) -> str:
    """Create a fast HTML/CSS violin plot that imitates the matplotlib version.

    This function creates an HTML/SVG-based violin plot that visually matches
    the create_static_violin_for_hover output but renders much faster.

    Args:
        dist1: Distribution data for first group
        dist2: Distribution data for second group
        color1: Color for first group's violin
        color2: Color for second group's violin
        x_label: Label for the x-axis
        y_label: Label for the y-axis
        plot_title: Title for the plot
        group1_name: Name for first group
        group2_name: Name for second group
        size: Optional (width, height) of the SVG container in pixels
        min_y: Optional lower bound for the y-axis
        max_y: Optional upper bound for the y-axis

    Returns:
        str: HTML div containing the SVG violin plot
    """
    default_width, default_height = 350, 300
    svg_width, svg_height = size if size else (default_width, default_height)
    svg_width, svg_height = int(svg_width), int(svg_height)
    empty_state_label = (
        "No expression" if y_label.strip().lower() == "expression" else "No flow"
    )

    # Calculate sample sizes
    n1 = len(dist1)
    n2 = len(dist2)

    # Check for empty distributions (n=0)
    if n1 == 0 and n2 == 0:
        return (
            f'<div style="width:{svg_width}px;height:{svg_height}px;background:#f5f5f5;display:flex;align-items:center;justify-content:center;border:1px solid #ddd;border-radius:4px;">'
            f'<span style="color:#888;font-size:12px;text-align:center;">{empty_state_label}<br/>(n = 0)</span></div>'
        )
    if n1 == 0:
        return (
            f'<div style="width:{svg_width}px;height:{svg_height}px;background:#f5f5f5;display:flex;align-items:center;justify-content:center;border:1px solid #ddd;border-radius:4px;">'
            f'<span style="color:#888;font-size:12px;text-align:center;">{empty_state_label}<br/>(n₁ = 0)</span></div>'
        )
    if n2 == 0:
        return (
            f'<div style="width:{svg_width}px;height:{svg_height}px;background:#f5f5f5;display:flex;align-items:center;justify-content:center;border:1px solid #ddd;border-radius:4px;">'
            f'<span style="color:#888;font-size:12px;text-align:center;">{empty_state_label}<br/>(n₂ = 0)</span></div>'
        )

    # Check for all-zeros distributions
    all_zeros_1 = all(v == 0 for v in dist1)
    all_zeros_2 = all(v == 0 for v in dist2)

    if all_zeros_1 and all_zeros_2:
        return (
            f'<div style="width:{svg_width}px;height:{svg_height}px;background:#f5f5f5;display:flex;align-items:center;justify-content:center;border:1px solid #ddd;border-radius:4px;">'
            f'<span style="color:#888;font-size:12px;text-align:center;">{empty_state_label}<br/>(all values are 0)</span></div>'
        )

    mean1, mean2 = np.mean(dist1), np.mean(dist2)
    std1, std2 = np.std(dist1), np.std(dist2)
    q1_1, med1, q3_1 = np.percentile(dist1, [25, 50, 75])
    q1_2, med2, q3_2 = np.percentile(dist2, [25, 50, 75])

    # Determine if flow is reversed
    reversed_flow = (mean1 * mean2) < 0

    # Calculate plot range (allow overrides)
    all_values = np.concatenate([dist1, dist2])
    data_min, data_max = np.min(all_values), np.max(all_values)

    effective_min = data_min if min_y is None else min_y
    effective_max = data_max if max_y is None else max_y

    if effective_max <= effective_min:
        # Fallback to data-driven bounds with padding if provided values are invalid
        effective_min, effective_max = data_min, data_max

    if min_y is None or max_y is None or effective_max == data_max == data_min:
        y_range = data_max - data_min
        y_padding = y_range * 0.1 if y_range > 0 else 1.0
        plot_min = effective_min - (0 if min_y is not None else y_padding)
        plot_max = effective_max + (0 if max_y is not None else y_padding)
    else:
        plot_min = effective_min
        plot_max = effective_max

    plot_height = plot_max - plot_min if plot_max > plot_min else 1.0

    # SVG dimensions - increased size for better visibility
    # Maintain proportions based on default dimensions
    width_scale = svg_width / default_width
    height_scale = svg_height / default_height

    margin_left = int(50 * width_scale)  # Increased for y-label
    margin_right = int(30 * width_scale)
    margin_top = int(40 * height_scale)
    margin_bottom = int(55 * height_scale)  # Increased for rotated x-axis labels

    plot_area_width = max(svg_width - margin_left - margin_right, 50)
    plot_area_height = max(svg_height - margin_top - margin_bottom, 50)

    # Scaling functions - y-axis direction depends on first group mean sign
    def scale_y(val):
        if mean1 >= 0:
            # Normal direction: larger values at top
            return (
                margin_top
                + plot_area_height
                - ((val - plot_min) / plot_height * plot_area_height)
            )
        else:
            # Inverted direction: larger (more negative) values at top
            return margin_top + ((val - plot_min) / plot_height * plot_area_height)

    def scale_x(pos):
        return margin_left + (pos - 0.5) / 2.0 * plot_area_width

    # Create violin shapes using Gaussian KDE for smooth curves
    def create_violin_path(data, x_center, width=60):
        if len(data) == 0:
            return ""

        try:
            # Use Gaussian KDE for smooth density estimation
            kde = gaussian_kde(data)

            # Create y values covering the data range with padding
            data_min, data_max = np.min(data), np.max(data)
            data_range = data_max - data_min
            y_padding = data_range * 0.1 if data_range > 0 else 1
            y_values = np.linspace(data_min - y_padding, data_max + y_padding, 50)

            # Calculate density
            density = kde(y_values)

            # Normalize to violin width
            max_density = np.max(density) if np.max(density) > 0 else 1
            normalized_widths = (density / max_density) * (width / 2)

            # Create smooth path points
            left_points = []
            right_points = []

            for y_val, half_width in zip(y_values, normalized_widths):
                y_scaled = scale_y(y_val)
                left_points.append(f"{scale_x(x_center) - half_width},{y_scaled}")
                right_points.append(f"{scale_x(x_center) + half_width},{y_scaled}")

            # Create smooth closed path
            if left_points and right_points:
                path = f"M {left_points[0]} "

                # Add smooth curves using quadratic Bezier curves for left side
                for i in range(1, len(left_points)):
                    if i == 1:
                        path += f"L {left_points[i]} "
                    else:
                        # Use previous point as control point for smoothing
                        path += f"Q {left_points[i - 1]} {left_points[i]} "

                # Connect to right side
                path += f"L {right_points[-1]} "

                # Add smooth curves for right side (reversed)
                for i in range(len(right_points) - 2, -1, -1):
                    if i == len(right_points) - 2:
                        path += f"L {right_points[i]} "
                    else:
                        path += f"Q {right_points[i + 1]} {right_points[i]} "

                path += " Z"
                return path

        except (np.linalg.LinAlgError, ValueError):
            # Fallback to histogram method if KDE fails (e.g., all values identical)
            pass

        # Fallback: Simple density estimation using histogram
        n_bins = min(20, len(data) // 2 + 1)
        hist, edges = np.histogram(data, bins=n_bins, density=True)
        bin_centers = (edges[:-1] + edges[1:]) / 2

        # Normalize to violin width
        max_density = np.max(hist) if np.max(hist) > 0 else 1
        normalized_widths = (hist / max_density) * (width / 2)

        # Create path points
        left_points = []
        right_points = []

        for y_val, half_width in zip(bin_centers, normalized_widths):
            y_scaled = scale_y(y_val)
            left_points.append(f"{scale_x(x_center) - half_width},{y_scaled}")
            right_points.append(f"{scale_x(x_center) + half_width},{y_scaled}")

        # Create closed path
        if left_points and right_points:
            path = f"M {left_points[0]} L " + " L ".join(left_points)
            path += " L " + " L ".join(reversed(right_points)) + " Z"
            return path
        return ""

    # Convert colors to RGB with alpha
    def hex_to_rgba(hex_color, alpha=0.6):
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 6:
            r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
            return f"rgba({r},{g},{b},{alpha})"
        return hex_color

    # Generate violin paths
    violin1_path = create_violin_path(dist1, 1.0)
    violin2_path = create_violin_path(dist2, 2.0)

    # Create the HTML/SVG plot
    html = f'''
    <div style="width:{svg_width}px;height:{
        svg_height
    }px;font-family:Arial,sans-serif;font-size:12px;background:white;border:1px solid #ddd;border-radius:4px;">
        <svg width="{svg_width}" height="{svg_height}" style="background:white;">
            <!-- Grid lines -->
            <defs>
                <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                    <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#f0f0f0" stroke-width="1" opacity="0.3"/>
                </pattern>
            </defs>
            <rect width="{plot_area_width}" height="{plot_area_height}" x="{
        margin_left
    }" y="{margin_top}" fill="url(#grid)"/>
            
            <!-- Y-axis -->
            <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{
        margin_top + plot_area_height
    }" stroke="#333" stroke-width="1"/>
            
            <!-- X-axis -->
            <line x1="{margin_left}" y1="{margin_top + plot_area_height}" x2="{
        margin_left + plot_area_width
    }" y2="{margin_top + plot_area_height}" stroke="#333" stroke-width="1"/>
            
            <!-- Zero line (if reversed flow) -->
            {
        f'<line x1="{margin_left}" y1="{scale_y(0)}" x2="{margin_left + plot_area_width}" y2="{scale_y(0)}" stroke="red" stroke-width="1" stroke-dasharray="5,5" opacity="0.7"/>'
        if reversed_flow
        else ""
    }
            
            <!-- Sample size display (above each violin) -->
            <text x="{scale_x(1.0)}" y="{margin_top - 8}" 
                  text-anchor="middle" font-size="10" fill="#888">n₁={n1}</text>
            <text x="{scale_x(2.0)}" y="{margin_top - 8}" 
                  text-anchor="middle" font-size="10" fill="#888">n₂={n2}</text>
            
            <!-- Violin 1 -->
            {
        f'<text x="{scale_x(1.0)}" y="{margin_top + plot_area_height / 2}" text-anchor="middle" font-size="10" fill="#888">{empty_state_label}</text>'
        if all_zeros_1
        else (
            f'<path d="{violin1_path}" fill="{hex_to_rgba(color1)}" stroke="gray" stroke-width="0.5"/>'
            if violin1_path
            else ""
        )
    }
            
            <!-- Violin 2 -->
            {
        f'<text x="{scale_x(2.0)}" y="{margin_top + plot_area_height / 2}" text-anchor="middle" font-size="10" fill="#888">{empty_state_label}</text>'
        if all_zeros_2
        else (
            f'<path d="{violin2_path}" fill="{hex_to_rgba(color2)}" stroke="gray" stroke-width="0.5"/>'
            if violin2_path
            else ""
        )
    }
            
            <!-- Box plots (scaled for larger size) -->
            <!-- Box 1 (only if not all zeros) -->
            {
        ""
        if all_zeros_1
        else f"""<rect x="{scale_x(1.0) - 12}" y="{scale_y(q3_1)}" width="24" height="{
            scale_y(q1_1) - scale_y(q3_1)
        }" 
                  fill="black" opacity="0.5" stroke="none"/>
            <line x1="{scale_x(1.0) - 12}" y1="{scale_y(med1)}" x2="{
            scale_x(1.0) + 12
        }" y2="{scale_y(med1)}" 
                  stroke="white" stroke-width="2"/>
            <line x1="{scale_x(1.0)}" y1="{scale_y(np.min(dist1))}" x2="{
            scale_x(1.0)
        }" y2="{scale_y(np.max(dist1))}" 
                  stroke="black" stroke-width="1"/>"""
    }
            
            <!-- Box 2 (only if not all zeros) -->
            {
        ""
        if all_zeros_2
        else f"""<rect x="{scale_x(2.0) - 12}" y="{scale_y(q3_2)}" width="24" height="{
            scale_y(q1_2) - scale_y(q3_2)
        }" 
                  fill="black" opacity="0.5" stroke="none"/>
            <line x1="{scale_x(2.0) - 12}" y1="{scale_y(med2)}" x2="{
            scale_x(2.0) + 12
        }" y2="{scale_y(med2)}" 
                  stroke="white" stroke-width="2"/>
            <line x1="{scale_x(2.0)}" y1="{scale_y(np.min(dist2))}" x2="{
            scale_x(2.0)
        }" y2="{scale_y(np.max(dist2))}" 
                  stroke="black" stroke-width="1"/>"""
    }
            
            <!-- Scatter points (simplified, only for non-zero distributions) -->
            {
        "" if all_zeros_1 else _generate_scatter_points(dist1, 1.0, scale_x, scale_y)
    }
            {
        "" if all_zeros_2 else _generate_scatter_points(dist2, 2.0, scale_x, scale_y)
    }
            
            <!-- Median reference lines (only for non-zero distributions) -->
            <!-- Horizontal dashed lines at median heights with group colors and low opacity -->
            {
        ""
        if all_zeros_1
        else f"""<line x1="{margin_left}" y1="{scale_y(med1)}" 
                  x2="{margin_left + plot_area_width}" y2="{scale_y(med1)}" 
                  stroke="{color1}" stroke-width="1" 
              stroke-dasharray="3,3" opacity="0.3"/>"""
    }
            {
        ""
        if all_zeros_2
        else f"""<line x1="{margin_left}" y1="{scale_y(med2)}" 
                  x2="{margin_left + plot_area_width}" y2="{scale_y(med2)}" 
                  stroke="{color2}" stroke-width="1" 
              stroke-dasharray="3,3" opacity="0.3"/>"""
    }
            
            <!-- X-axis labels -->
            <text x="{scale_x(1.0)}" y="{
        margin_top + plot_area_height + 25
    }" text-anchor="middle" 
                  font-size="12" fill="#333" transform="rotate(-30 {scale_x(1.0)} {
        margin_top + plot_area_height + 25
    })">{group1_name}</text>
            <text x="{scale_x(2.0)}" y="{
        margin_top + plot_area_height + 25
    }" text-anchor="middle" 
                  font-size="12" fill="#333" transform="rotate(-30 {scale_x(2.0)} {
        margin_top + plot_area_height + 25
    })">{group2_name}</text>
            
            <!-- Y-axis labels (simplified) -->
            <text x="{margin_left - 5}" y="{
        scale_y(plot_max)
    }" text-anchor="end" font-size="10" fill="#666">{plot_max:.2f}</text>
            <text x="{margin_left - 5}" y="{
        scale_y((plot_max + plot_min) / 2)
    }" text-anchor="end" font-size="10" fill="#666">{
        (plot_max + plot_min) / 2:.2f}</text>
            <text x="{margin_left - 5}" y="{
        scale_y(plot_min)
    }" text-anchor="end" font-size="10" fill="#666">{plot_min:.2f}</text>
            
            <!-- Axis labels -->
            {
        f'<text x="{margin_left + plot_area_width / 2}" y="{svg_height - 5}" text-anchor="middle" font-size="11" fill="#333">{x_label}</text>'
        if x_label
        else ""
    }
            {
        f'<text x="10" y="{margin_top + plot_area_height / 2}" text-anchor="middle" font-size="11" fill="#333" transform="rotate(-90 10 {margin_top + plot_area_height / 2})">{y_label}</text>'
        if y_label
        else ""
    }
            
            <!-- Title -->
            {
        f'<text x="{svg_width / 2}" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">{plot_title}</text>'
        if plot_title
        else ""
    }
        </svg>
    </div>
    '''

    return html


def _generate_scatter_points(data, x_center, scale_x, scale_y, max_points=50):
    """Generate SVG scatter points for the violin plot."""
    if len(data) == 0:
        return ""

    # Subsample if too many points
    if len(data) > max_points:
        indices = np.random.choice(len(data), max_points, replace=False)
        data_sample = np.array(data)[indices]
    else:
        data_sample = data

    points = []
    for i, val in enumerate(data_sample):
        # Add small jitter - increased for larger plot
        jitter = np.random.normal(0, 0.03)
        x = scale_x(x_center + jitter)
        y = scale_y(val)
        # Increased point size for larger plot
        points.append(f'<circle cx="{x}" cy="{y}" r="1.2" fill="black" opacity="0.5"/>')

    return "\n".join(points)


def create_static_violin_for_hover(
    dist1: list,
    dist2: list,
    color1: str,
    color2: str,
    x_label: str = "",
    y_label: str = "",
    plot_title: str = "",
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
    use_html: bool = True,
    size: Optional[Tuple[int, int]] = None,
    min_y: Optional[float] = None,
    max_y: Optional[float] = None,
) -> str:
    """Create a static violin plot for hover text.

    Args:
        dist1: Distribution data for first group
        dist2: Distribution data for second group
        color1: Color for first group's violin
        color2: Color for second group's violin
        x_label: Label for the x-axis
        y_label: Label for the y-axis
        plot_title: Title for the plot
        group1_name: Name for first group
        group2_name: Name for second group
    use_html: If True, use fast HTML/SVG version; if False, use matplotlib
    size: Optional (width, height) of the SVG container in pixels
    min_y: Optional lower bound for the y-axis
    max_y: Optional upper bound for the y-axis

    Returns:
        str: HTML content containing the violin plot
    """
    if use_html:
        return create_html_violin_for_hover(
            dist1,
            dist2,
            color1,
            color2,
            x_label,
            y_label,
            plot_title,
            group1_name,
            group2_name,
            size=size,
            min_y=min_y,
            max_y=max_y,
        )
    else:
        return create_matplotlib_violin_for_hover(
            dist1,
            dist2,
            color1,
            color2,
            x_label,
            y_label,
            plot_title,
            group1_name,
            group2_name,
        )


def create_matplotlib_violin_for_hover(
    dist1: list,
    dist2: list,
    color1: str,
    color2: str,
    x_label: str = "",
    y_label: str = "",
    plot_title: str = "",
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
) -> str:
    """Create a static violin plot image for hover text based on create_violin styling.

    This function creates a static image that follows the same visual styling as
    the create_violin function for consistency across the application.

    Args:
        dist1: Distribution data for first group
        dist2: Distribution data for second group
        color1: Color for first group's violin
        color2: Color for second group's violin
        x_label: Label for the x-axis
        y_label: Label for the y-axis
        plot_title: Title for the plot
        group1_name: Name for first group
        group2_name: Name for second group

    Returns:
        str: HTML img tag containing the plot as a base64-encoded SVG
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(3, 2.5), dpi=800)

    # Calculate means for flow direction analysis
    mean1 = np.mean(dist1) if len(dist1) > 0 else 0
    mean2 = np.mean(dist2) if len(dist2) > 0 else 0
    reversed_flow = (mean1 * mean2) < 0

    # Prepare data and positions
    data = [dist1, dist2]
    positions = [1, 2]

    # Create violin plot
    parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.5,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    if mean1 < 0:
        # Flip y axis if the mean is negative
        ax.invert_yaxis()

    # Customize violin bodies - match create_violin styling
    for i, (pc, color) in enumerate(zip(parts["bodies"], [color1, color2])):
        if color.startswith("#"):
            rgb = tuple(
                int(color.lstrip("#")[i : i + 2], 16) / 255.0 for i in (0, 2, 4)
            )
            rgba = (*rgb, 0.6)  # Match the 0.6 alpha from create_violin
        else:
            rgba = color
        pc.set_facecolor(rgba)
        pc.set_edgecolor("gray")
        pc.set_alpha(0.6)
        pc.set_linewidth(0.5)

    # Add box plot inside violin - match create_violin styling
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.1,  # Match the 0.1 width from create_violin
        patch_artist=True,
        showcaps=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 2},
        boxprops={"facecolor": "black", "alpha": 0.5, "linewidth": 0},
        whiskerprops={"color": "black", "linewidth": 1},
        capprops={"color": "black", "linewidth": 1},
    )

    # Add jittered points - match create_violin styling
    for idx, (dist, pos) in enumerate(zip(data, positions)):
        # Apply jitter similar to create_violin
        points_x = np.random.normal(pos, 0.03, size=len(dist))
        ax.scatter(points_x, dist, c="black", alpha=0.5, s=1)

    # Add zero line if reaction has reversed flow - match create_violin styling
    if reversed_flow:
        ax.axhline(
            y=0,
            color="red",
            linestyle="dashed",
            alpha=0.7,
        )

    # Set the grid styling to match create_violin
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.grid(False, axis="x")

    # Set up x-tick labels with group names - match create_violin styling
    ax.set_xticks(positions)
    ax.set_xticklabels([group1_name, group2_name], fontsize=7, rotation=30, ha="right")

    # Set labels
    if x_label:
        ax.set_xlabel(x_label, fontsize=8, labelpad=5)
    if y_label:
        ax.set_ylabel(y_label, fontsize=8, labelpad=5)
    if plot_title:
        ax.set_title(plot_title, fontsize=9, pad=5)

    # Customize appearance to match create_violin
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", labelsize=7, width=0.8, length=3)

    # Make sure the plot is tight
    plt.tight_layout()

    # Convert to SVG and encode as base64
    buf = io.BytesIO()
    plt.savefig(
        buf, format="svg", bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close(fig)
    buf.seek(0)

    img_data = base64.b64encode(buf.read()).decode("utf-8")
    return f'<img src="data:image/svg+xml;base64,{img_data}" alt="Violin Plot" style="max-width:100%; height:auto;"/>'


def plot_reaction_violins(
    data_manager,
    subsystem: str,
    significant_only: bool = False,
    plots_per_row: int = 4,
    figsize: tuple = (800, None),
    group1_color: Optional[str] = None,
    group2_color: Optional[str] = None,
    metric: str = "log2_fc",
    return_plot: bool = False,
    enabled_toggle=None,
    click_filename_prefix: str = "reaction_violin",
) -> Optional[figure]:
    """Create a grid of interactive violin plots for reactions in a subsystem."""
    # Extract experiment and group_comparison from data_manager
    group_comparison = data_manager.group_comparison

    if not group_comparison:
        raise ValueError("Group comparison not set in the data_manager")

    # Get test results from group_comparison
    test_results = group_comparison.get_test_results(for_genes=False)

    # Filter for the specific subsystem
    if subsystem:
        mask = test_results["subsystems"].apply(
            lambda subsystems_list: (
                subsystem in subsystems_list
                if isinstance(subsystems_list, list)
                else subsystem == subsystems_list
            )
        )
        test_results = test_results[mask].copy()

    if test_results.empty:
        raise ValueError(f"No reactions found for subsystem '{subsystem}'")

    # Get colors for groups using ColorManager delegation
    color1 = (
        group1_color
        if group1_color
        else data_manager.get_group_color(group_comparison.group_id_1)
    )
    color2 = (
        group2_color
        if group2_color
        else data_manager.get_group_color(group_comparison.group_id_2)
    )

    # Get significant elements from group_comparison
    significant_reactions = group_comparison.get_significant_elements(for_genes=False)

    # Create a column for sorting (1 for significant, 0 for non-significant)
    test_results["is_significant"] = test_results.index.isin(
        significant_reactions
    ).astype(int)

    # Check for duplicate reaction IDs before sorting
    if test_results.index.duplicated().any():
        print(
            f"Warning: Found {test_results.index.duplicated().sum()} duplicate reaction IDs"
        )
        test_results = test_results[~test_results.index.duplicated()]

    # Add the index as a column for sorting
    test_results["reaction_id"] = test_results.index

    # Sort by significance first (1=significant), then alphabetically by reaction ID
    test_results = test_results.sort_values(
        ["is_significant", "reaction_id"], ascending=[False, True]
    )

    # Filter for significant reactions if requested
    if significant_only:
        test_results = test_results[test_results["is_significant"] == 1]
        if test_results.empty:
            raise ValueError(
                f"No significant reactions found for subsystem '{subsystem}'"
            )

    # Calculate dimensions
    plot_width = figsize[0] // plots_per_row
    plot_height = plot_width

    plots = []
    for idx, (reaction_id, row) in enumerate(test_results.iterrows()):
        # Get reaction data using data_manager's method instead of experiment directly
        reaction_info = data_manager.get_sample_reaction(reaction_id)

        # Get activity values for both groups
        if "activity_values" in reaction_info:
            dist1 = reaction_info["activity_values"]["group1"]
            dist2 = reaction_info["activity_values"]["group2"]
        else:
            # Skip reactions with no activity values
            continue

        # Check if reaction is significant
        is_significant = row["is_significant"] == 1

        # Check if reaction is reversed
        mean1 = np.mean(dist1) if len(dist1) > 0 else 0
        mean2 = np.mean(dist2) if len(dist2) > 0 else 0
        is_reversed = (mean1 * mean2) < 0

        # Get border color based on metric if significant using data_manager's method
        if is_significant:
            # Get hover text and color from unified get_element_hover interface
            hover_text = data_manager.get_element_hover(
                element_id=reaction_id,
                element_type="reaction",
                reaction_format="mar_id",
                color_metric=metric,
                show_stats=True,
                show_color_reference=True,
            )

            # Get color separately
            color_rgb = data_manager.get_element_color(reaction_id, "reaction", metric)

            if isinstance(color_rgb, tuple):
                r, g, b = color_rgb
                border_color = f"rgb({r}, {g}, {b})"
            else:
                border_color = color_rgb
        else:
            # For non-significant reactions, still get hover text but use gray border
            hover_text = data_manager.get_element_hover(
                element_id=reaction_id,
                element_type="reaction",
                reaction_format="mar_id",
                color_metric=metric,
                show_stats=True,
                show_color_reference=True,
            )
            border_color = "#888888"  # Gray for non-significant

        reference_data = get_external_reference_data(
            data_manager,
            reaction_id,
            "reaction",
        )

        # Create violin plot with hover text
        plot = create_violin(
            dist1=dist1,
            dist2=dist2,
            color1=color1,
            color2=color2,
            title=reaction_id,
            x_label="Group",
            y_label="Activity",
            significant=is_significant,
            group1_name=group_comparison.group_id_1,
            group2_name=group_comparison.group_id_2,
            border_color=border_color,
            is_reversed=is_reversed,
            hover_text=hover_text,  # Pass the hover text to the create_violin function
            element_id=reference_data["element_id"],
            display_label=reference_data["display_label"],
            external_url=reference_data["external_url"],
        )
        plots.append(plot)

    # Create grid
    grid = gridplot(children=plots, ncols=plots_per_row, sizing_mode="stretch_width")

    add_click_behavior(
        grid,
        filename_prefix=click_filename_prefix,
        enabled_toggle=enabled_toggle,
    )

    if return_plot:
        return grid
    else:
        show(grid)

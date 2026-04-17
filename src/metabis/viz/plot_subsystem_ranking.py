from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from bokeh.models import (
    BoxAnnotation,
    ColumnDataSource,
    FactorRange,
    HoverTool,
    LabelSet,
    Range1d,
    Span,
)
from bokeh.plotting import figure, show

from ..managers.data_manager_mode import DataManagerMode
from ..visual_config import theme
from .click_behavior import add_click_behavior, get_external_reference_data
from .metabolic_visualization.generation.data_structures.constants import (
    VisualizationConstants,
)
from .metabolic_visualization.rendering.js_callbacks import (
    add_factor_range_zoom_alpha_callback,
)
from .plot_utils import apply_academic_style


def plot_subsystem_comparison(
    data_manager,
    subsystem_stats: pd.DataFrame,
    metric: str = "cohens_d",
    max_points_per_subsystem: int = 50,
    width: int = None,
    height: int = None,
    for_genes: bool = False,
    return_plot: bool = False,
    title: str = None,
    background_fill_color: str = "#ffffff",
    sizing_mode: str = "stretch_width",
    enabled_toggle=None,
    click_filename_prefix: str = "subsystem_element",
):
    """Create a category-grouped dot plot comparison of subsystems.

    Subsystems are displayed on the y-axis, grouped by metabolic category.
    Categories are ordered by average absolute delta score, and subsystems
    within each category are ordered by their delta score.
    Metric values are displayed on the x-axis.

    Args:
        data_manager: The DataManager object.
        subsystem_stats: DataFrame containing ranked subsystem statistics.
        metric: The metric to plot (e.g., "cohens_d", "log2_fc").
        max_points_per_subsystem: Maximum number of points to plot per subsystem.
        width: Width of the plot.
        height: Height of the plot. If None, calculated based on number of subsystems.
        for_genes: Whether to plot gene data (True) or reaction data (False).
        return_plot: Whether to return the figure or show it.
        title: Optional title for the plot.
        background_fill_color: Background color for the plot.
        sizing_mode: Optional sizing mode (e.g., "stretch_width").

    Returns:
        Bokeh figure if return_plot is True, else None.
    """
    if subsystem_stats is None or subsystem_stats.empty:
        print("No subsystem statistics available.")
        return None

    # Get global test results
    test_results = data_manager.group_comparison.get_test_results(for_genes=for_genes)
    if test_results is None or metric not in test_results.columns:
        print(f"Metric '{metric}' not found in test results.")
        return None

    # Prepare category-ordered data
    ordered_data = _prepare_category_ordered_subsystems(
        data_manager=data_manager,
        subsystem_stats=subsystem_stats,
    )

    if not ordered_data["factors"]:
        print("No subsystems found after category ordering.")
        return None

    # Reverse factors for y-axis (top to bottom display: highest score at top)
    y_factors = ordered_data["factors"][::-1]
    # Expand the range bounds slightly to ensure top category label is visible
    y_range = FactorRange(factors=y_factors, bounds="auto", range_padding=0.02)

    # Also store reversed factors in ordered_data for decoration positioning
    ordered_data["y_factors"] = y_factors

    # Collect all scatter data
    all_data = {
        "y": [],  # Will be (category, subsystem) tuples
        "x": [],  # Metric values
        "element_id": [],
        "display_label": [],
        "external_url": [],
        "subsystem": [],
        "category": [],
        "size": [],
        "alpha": [],
        "color": [],
        "desc": [],
    }

    # Get significant elements
    significant_set = data_manager.group_comparison.get_significant_elements(
        for_genes=for_genes
    )
    element_type = "gene" if for_genes else "reaction"

    # Collect all metric values for x-range calculation
    all_metric_values = []

    for category, subsystem in ordered_data["factors"]:
        # Get elements for this subsystem
        if for_genes:
            elements = data_manager.metabolic_model.get_genes_for_subsystem(subsystem)
        else:
            elements = data_manager.metabolic_model.get_reactions_for_subsystem(
                subsystem
            )
            # Add _back suffix for reverse reactions
            elements += [f"{e}_back" for e in elements]

        # Filter elements present in test results
        valid_elements = [e for e in elements if e in test_results.index]
        if not valid_elements:
            continue

        subsystem_results = test_results.loc[valid_elements]

        # Prepare data for this subsystem (with subsampling)
        prepared_data = _prepare_subsystem_scatter_data_horizontal(
            data_manager=data_manager,
            subsystem_results=subsystem_results,
            subsystem_name=subsystem,
            category_name=category,
            metric=metric,
            max_points=max_points_per_subsystem,
            for_genes=for_genes,
            significant_set=significant_set,
            element_type=element_type,
        )

        if prepared_data:
            for key in all_data:
                all_data[key].extend(prepared_data[key])
            all_metric_values.extend(prepared_data["x"])

    if not all_metric_values:
        print("No data points found for the selected subsystems.")
        return None

    # Calculate x-range with padding
    x_min, x_max = min(all_metric_values), max(all_metric_values)
    padding = (x_max - x_min) * 0.1 if x_max != x_min else 1.0
    x_range = Range1d(x_min - padding, x_max + padding)

    # Calculate height if not provided
    if height is None:
        height = max(600, len(ordered_data["factors"]) * 20)

    # Create the figure
    if title is None:
        title = f"{metric.replace('_', ' ').title()} Distribution by Subsystem Category"

    figure_kwargs = {
        "width": width,
        "height": height,
        "x_range": x_range,
        "y_range": y_range,
        "title": title,
        "tools": "hover,tap,pan,wheel_zoom,reset,save",
        "toolbar_location": "above",
        "x_axis_location": "above",
        "background_fill_color": background_fill_color,
        "border_fill_color": "rgba(255,255,255,0)",
    }

    if sizing_mode:
        figure_kwargs["sizing_mode"] = sizing_mode

    p = figure(**figure_kwargs)
    apply_academic_style(p)

    # Add category visual elements (backgrounds, separators, headers)
    _add_category_decorations_horizontal(
        p=p,
        ordered_data=ordered_data,
        x_range=x_range,
    )

    # Add vertical line at 0
    p.add_layout(
        Span(
            location=0,
            dimension="height",
            line_color=theme.THRESHOLD_LINE_COLOR,
            line_dash=theme.THRESHOLD_LINE_DASH,
            line_width=theme.THRESHOLD_LINE_WIDTH,
            line_alpha=0.7,
        )
    )

    # Grid styling
    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = "#404040"
    p.xgrid.grid_line_alpha = 0.5

    # Create the data source
    source = ColumnDataSource(all_data)

    # Plot scatter points
    p.scatter(
        x="x",
        y="y",
        source=source,
        marker="circle",
        size="size",
        color="color",
        alpha="alpha",
        line_color="white",
        line_width=0.5,
    )

    # Set axis labels
    p.xaxis.axis_label = metric.replace("_", " ").title()
    p.yaxis.axis_label = None  # Categories shown via nested factor range

    # Hide default y-axis completely (we use custom LabelSet instead)
    p.yaxis.visible = False

    # =========================================================================
    # Add custom zoom-dependent labels for subsystems and categories
    # =========================================================================
    subsystem_label_source, category_label_source = _create_zoom_dependent_labels(
        p=p,
        ordered_data=ordered_data,
        x_range=x_range,
        data_manager=data_manager,
    )

    # Configure Hover
    hover = p.select(dict(type=HoverTool))
    hover.tooltips = "@desc{safe}"

    add_click_behavior(
        p,
        filename_prefix=click_filename_prefix,
        enabled_toggle=enabled_toggle,
    )

    if return_plot:
        return p
    else:
        show(p)
        return None


def _prepare_category_ordered_subsystems(
    data_manager,
    subsystem_stats: pd.DataFrame,
) -> Dict[str, Any]:
    """Prepare subsystems ordered by category and delta score.

    Returns a dict with:
        - factors: List of (category, subsystem) tuples for FactorRange
        - category_info: Dict with category metadata (color, subsystem count, boundaries)
        - subsystem_full_names: Dict mapping truncated names to full names
    """
    metabolic_model = data_manager.metabolic_model

    # Build category -> subsystems mapping with delta scores
    category_subsystems: Dict[str, List[Tuple[str, float]]] = {}

    for _, row in subsystem_stats.iterrows():
        subsystem = row["subsystem"]
        delta_score = row.get("delta_score", 0)

        # Get category for this subsystem
        category = metabolic_model.get_subsystem_category(subsystem)
        if not category:
            category = "Unknown"

        if category not in category_subsystems:
            category_subsystems[category] = []
        category_subsystems[category].append((subsystem, delta_score))

    # Calculate average absolute delta score per category
    category_avg_scores: Dict[str, float] = {}
    for category, subsystems in category_subsystems.items():
        if subsystems:
            avg_abs_score = np.mean([abs(score) for _, score in subsystems])
            category_avg_scores[category] = avg_abs_score
        else:
            category_avg_scores[category] = 0.0

    # Sort categories by average absolute delta score (descending)
    sorted_categories = sorted(
        category_subsystems.keys(),
        key=lambda c: category_avg_scores.get(c, 0),
        reverse=True,
    )

    # Build ordered factors and category info
    factors = []
    category_info = {}
    current_idx = 0

    for category in sorted_categories:
        subsystems = category_subsystems[category]

        # Sort subsystems within category by absolute delta score (descending)
        sorted_subsystems = sorted(subsystems, key=lambda x: abs(x[1]), reverse=True)

        # Get category color using CategoryManager
        category_color = metabolic_model.get_category_color(category)

        start_idx = current_idx

        for subsystem, _ in sorted_subsystems:
            factors.append((category, subsystem))
            current_idx += 1

        end_idx = current_idx

        category_info[category] = {
            "color": category_color,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "count": len(sorted_subsystems),
            "avg_delta_score": category_avg_scores[category],
        }

    return {
        "factors": factors,
        "category_info": category_info,
        "sorted_categories": sorted_categories,
    }


def _prepare_subsystem_scatter_data_horizontal(
    data_manager,
    subsystem_results: pd.DataFrame,
    subsystem_name: str,
    category_name: str,
    metric: str,
    max_points: int,
    for_genes: bool,
    significant_set: set,
    element_type: str,
) -> Optional[Dict[str, list]]:
    """Prepare scatter data for a single subsystem's elements (horizontal orientation)."""
    if subsystem_results.empty or metric not in subsystem_results.columns:
        return None

    # Separate significant and non-significant elements
    interesting_indices = [
        idx for idx in subsystem_results.index if idx in significant_set
    ]
    non_interesting_indices = [
        idx for idx in subsystem_results.index if idx not in significant_set
    ]

    if len(interesting_indices) > max_points:
        interesting_indices = _select_prioritized_subsystem_indices(
            subsystem_results.loc[interesting_indices],
            metric=metric,
            max_points=max_points,
        )

    # Subsample non-significant if needed
    n_interesting = len(interesting_indices)
    n_remaining_slots = max(0, max_points - n_interesting)

    if len(non_interesting_indices) > n_remaining_slots:
        if n_remaining_slots > 0:
            sampled_non_interesting = np.random.choice(
                non_interesting_indices, size=n_remaining_slots, replace=False
            ).tolist()
        else:
            sampled_non_interesting = []
    else:
        sampled_non_interesting = non_interesting_indices

    indices_to_keep = interesting_indices + sampled_non_interesting

    if not indices_to_keep:
        return None

    filtered_results = subsystem_results.loc[indices_to_keep].copy()

    # Prepare data dictionary
    data = {
        "y": [(category_name, subsystem_name)] * len(filtered_results),
        "x": list(filtered_results[metric]),
        "element_id": list(filtered_results.index),
        "display_label": [],
        "external_url": [],
        "subsystem": [subsystem_name] * len(filtered_results),
        "category": [category_name] * len(filtered_results),
        "size": [8 if idx in significant_set else 5 for idx in filtered_results.index],
        "alpha": [
            1 if idx in significant_set else 0.4 for idx in filtered_results.index
        ],
    }

    # Assign colors
    colors = data_manager.color_manager.get_colors_from_distribution(
        mode=DataManagerMode.GROUP_COMPARISON,
        element_type=element_type,
        metric_name=metric,
        element_ids=filtered_results.index,
    )

    color_values = list(colors.values)
    for i, idx in enumerate(filtered_results.index):
        if idx not in significant_set:
            color_values[i] = theme.NOT_SIGNIFICANT

    data["color"] = color_values

    # Add hover text
    hover_texts = []
    for idx in filtered_results.index:
        reference_data = get_external_reference_data(
            data_manager,
            idx,
            element_type,
        )
        data["display_label"].append(reference_data["display_label"])
        data["external_url"].append(reference_data["external_url"])

        hover_text = data_manager.get_element_hover(
            element_id=idx, element_type=element_type, show_stats=True
        )
        hover_texts.append(hover_text)
    data["desc"] = hover_texts

    return data


def _select_prioritized_subsystem_indices(
    subsystem_results: pd.DataFrame,
    metric: str,
    max_points: int,
) -> List[str]:
    """Return a deterministic hard-capped subset of subsystem element IDs.

    When a subsystem has more significant elements than the configured point cap,
    keep the elements with the strongest absolute metric values first and use
    adjusted p-value as a stable tie-breaker when available.
    """
    if max_points <= 0 or subsystem_results.empty:
        return []

    ranked_results = subsystem_results.assign(
        _abs_metric=subsystem_results[metric].abs().fillna(float("-inf"))
    )
    sort_columns = ["_abs_metric"]
    ascending = [False]

    if "adjusted_pval" in ranked_results.columns:
        ranked_results = ranked_results.assign(
            _adjusted_pval=ranked_results["adjusted_pval"].fillna(float("inf"))
        )
        sort_columns.append("_adjusted_pval")
        ascending.append(True)

    return (
        ranked_results.sort_values(
            sort_columns,
            ascending=ascending,
            kind="mergesort",
        )
        .index[:max_points]
        .tolist()
    )


def _add_category_decorations_horizontal(
    p: figure,
    ordered_data: Dict[str, Any],
    x_range: Range1d,
) -> None:
    """Add category visual decorations for horizontal layout: backgrounds, separators, and header bars.

    The key challenge here is that Bokeh's nested FactorRange adds visual spacing (group_padding)
    between category groups. This spacing causes the BoxAnnotation numeric coordinates to become
    misaligned with the actual visual positions of factors.

    To fix this, we calculate a cumulative offset based on the number of category boundaries
    crossed as we move up from the bottom of the plot.
    """
    sorted_categories = ordered_data["sorted_categories"]
    y_factors = ordered_data.get("y_factors", [])

    if not y_factors:
        return

    # Build position mapping: find where each category starts and ends in the REVERSED factor list
    # y_factors is already reversed (bottom to top), so we iterate through it directly
    category_positions: Dict[str, Dict[str, int]] = {}
    category_order_in_plot: List[str] = []  # Order as they appear bottom-to-top

    for idx, (category, subsystem) in enumerate(y_factors):
        if category not in category_positions:
            category_positions[category] = {"start": idx, "end": idx + 1}
            category_order_in_plot.append(category)
        else:
            category_positions[category]["end"] = idx + 1

    # The group_padding in FactorRange adds visual space between groups.
    # Default group_padding is typically ~0.2, and it affects visual positioning.
    # We need to add a cumulative offset as we move up through categories.
    #
    # The offset per category boundary is approximately equal to the group_padding value.
    # We'll use the figure's y_range.group_padding if available, otherwise use default.
    group_padding = getattr(p.y_range, "group_padding", 0.1)

    # Calculate cumulative offsets for each category based on order in plot (bottom to top)
    category_separator_counts = {}
    for plot_idx, category in enumerate(category_order_in_plot):
        # Number of separators below this category = its position in the bottom-to-top order
        category_separator_counts[category] = plot_idx

    # Get category_info which contains pre-computed colors
    category_info = ordered_data.get("category_info", {})

    # Get colors for categories and draw decorations
    for i, category in enumerate(sorted_categories):
        if category not in category_positions:
            continue

        pos = category_positions[category]
        # Use pre-computed color from category_info (already using CategoryManager)
        info = category_info.get(category, {})
        color = info.get(
            "color", VisualizationConstants.DEFAULT_CATEGORY_COLOR_UNMAPPED
        )

        # Get the number of separators below this category
        num_separators_below = category_separator_counts.get(category, 0)

        # Calculate offset: each separator adds group_padding worth of visual space
        offset = num_separators_below * group_padding

        # Apply offset to positions
        bottom = pos["start"] + offset
        top = pos["end"] + offset + group_padding  # Add padding for the separator above

        # Colored header bar on left side
        header_box = BoxAnnotation(
            top=top,
            bottom=bottom,
            left=x_range.start,
            right=x_range.start + (x_range.end - x_range.start) * 0.015,
            fill_color=color,
            fill_alpha=0.9,
            level="underlay",
        )
        p.add_layout(header_box)

        # Horizontal separator line at bottom of category (between this and next category)
        # Draw for all categories except the last one (nothing below the last category)
        if i < len(category_order_in_plot) - 1:
            separator = Span(
                location=bottom,
                dimension="width",
                line_color=color,
                line_width=2,
                line_alpha=0.7,
            )
            p.add_layout(separator)


def _create_zoom_dependent_labels(
    p,
    ordered_data: Dict[str, Any],
    x_range: Range1d,
    data_manager=None,
) -> Tuple[ColumnDataSource, ColumnDataSource]:
    """Create custom LabelSet glyphs with zoom-dependent visibility.

    Creates two sets of labels:
    1. Subsystem labels - always visible, positioned on left side
    2. Category labels - positioned at top of each category section, centered on x-axis

    Args:
        p: Bokeh figure
        ordered_data: Dict with factors, category_info, sorted_categories, y_factors
        x_range: Range1d for x-axis positioning
        data_manager: DataManager for getting subsystem names

    Returns:
        Tuple of (subsystem_label_source, category_label_source) ColumnDataSources
    """
    y_factors = ordered_data.get("y_factors", [])
    category_info = ordered_data.get("category_info", {})
    sorted_categories = ordered_data.get("sorted_categories", [])

    if not y_factors:
        # Return empty sources if no data
        empty_source = ColumnDataSource(data=dict(x=[], y=[], text=[], text_alpha=[]))
        return empty_source, empty_source

    # Calculate x position for subsystem labels (left side of plot)
    label_x = x_range.start + (x_range.end - x_range.start) * 0.02
    # Calculate x position for category labels (center of plot)
    category_label_x = (x_range.start + x_range.end) / 2

    # Get metabolic model for subsystem name lookup
    metabolic_model = data_manager.metabolic_model if data_manager else None

    # =========================================================================
    # Build subsystem labels data
    # =========================================================================
    subsystem_data = {
        "x": [],
        "y": [],  # Will be (category, subsystem) tuples
        "text": [],
        "text_alpha": [],
    }

    # Subsystem labels are always visible
    initial_subsystem_alpha = VisualizationConstants.CATEGORY_LABEL_ALPHA

    for category, subsystem_id in y_factors:
        subsystem_data["x"].append(label_x)
        subsystem_data["y"].append((category, subsystem_id))

        # Get subsystem name from metabolic model, fallback to ID if not found
        if metabolic_model:
            subsystem_name = metabolic_model.get_subsystem_name(subsystem_id)
            if not subsystem_name:
                subsystem_name = subsystem_id
        else:
            subsystem_name = subsystem_id

        # Truncate long subsystem names for display
        display_name = (
            subsystem_name if len(subsystem_name) <= 35 else subsystem_name[:32] + "..."
        )
        subsystem_data["text"].append(display_name)
        subsystem_data["text_alpha"].append(initial_subsystem_alpha)

    subsystem_label_source = ColumnDataSource(subsystem_data)

    # Add subsystem LabelSet
    subsystem_labels = LabelSet(
        x="x",
        y="y",
        text="text",
        text_font_size=VisualizationConstants.COMPARISON_SUBSYSTEM_FONT_SIZE,
        text_color="black",
        text_alpha="text_alpha",
        text_align="left",
        text_baseline="middle",
        x_offset=VisualizationConstants.COMPARISON_LABEL_X_OFFSET,
        level="annotation",
        source=subsystem_label_source,
    )
    p.add_layout(subsystem_labels)

    # =========================================================================
    # Build category labels data
    # =========================================================================
    category_data = {
        "x": [],
        "y": [],  # Will be (category, subsystem) tuples pointing to top of category section
        "text": [],
        "text_alpha": [],
        "text_color": [],
    }

    # Initial alpha for categories (visible when zoomed out)
    initial_category_alpha = VisualizationConstants.COMPARISON_CATEGORY_LABEL_ALPHA_MAX

    # Get group_padding for position calculation
    group_padding = getattr(p.y_range, "group_padding", 0.1)

    # Build position mapping for categories (same logic as _add_category_decorations_horizontal)
    category_positions: Dict[str, Dict[str, int]] = {}
    category_order_in_plot: List[str] = []

    for idx, (category, subsystem) in enumerate(y_factors):
        if category not in category_positions:
            category_positions[category] = {
                "start": idx,
                "end": idx + 1,
                "first_subsystem": subsystem,
                "last_subsystem": subsystem,
            }
            category_order_in_plot.append(category)
        else:
            category_positions[category]["end"] = idx + 1
            category_positions[category]["last_subsystem"] = subsystem

    # Position category labels at the TOP of each category section (last subsystem = highest y position)
    # Get category_info which contains pre-computed colors
    category_info = ordered_data.get("category_info", {})

    for category in sorted_categories:
        if category not in category_positions:
            continue

        pos = category_positions[category]
        # Use the LAST subsystem in this category (which is at the top of the section in the reversed list)
        top_factor = (category, pos["last_subsystem"])

        # Use pre-computed color from category_info (already using CategoryManager)
        info = category_info.get(category, {})
        color = info.get(
            "color", VisualizationConstants.DEFAULT_CATEGORY_COLOR_UNMAPPED
        )

        # Position at center of x-axis, at the top subsystem of the category
        category_data["x"].append(category_label_x)
        category_data["y"].append(top_factor)
        category_data["text"].append(category)
        category_data["text_alpha"].append(initial_category_alpha)
        category_data["text_color"].append(color)

    category_label_source = ColumnDataSource(category_data)

    # Add category LabelSet - positioned above the top subsystem
    category_labels = LabelSet(
        x="x",
        y="y",
        text="text",
        text_font_size=VisualizationConstants.COMPARISON_CATEGORY_FONT_SIZE,
        text_color="text_color",
        text_alpha="text_alpha",
        text_align="center",  # Center horizontally
        text_baseline="bottom",  # Position above the y coordinate
        text_font_style="bold",
        y_offset=8,  # Small offset above the subsystem row
        level="annotation",
        source=category_label_source,
    )
    p.add_layout(category_labels)

    # =========================================================================
    # Add zoom-based alpha callback (only for category labels now)
    # =========================================================================
    add_factor_range_zoom_alpha_callback(
        plot=p,
        subsystem_label_source=None,  # Subsystem labels always visible, no callback needed
        category_label_source=category_label_source,
    )

    return subsystem_label_source, category_label_source

"""Utilities for creating Bokeh visualizations."""

from bokeh.models import CustomJS

# Import visualization constants
from ..generation.data_structures.constants import VisualizationConstants


def _resolve_zoom_alpha_parameters(alpha_min, alpha_max, zoom_min, zoom_max):
    """Resolve optional zoom-alpha parameters against subsystem defaults."""
    return (
        (
            alpha_min
            if alpha_min is not None
            else VisualizationConstants.SUBSYSTEM_LABEL_ALPHA_MIN
        ),
        (
            alpha_max
            if alpha_max is not None
            else VisualizationConstants.SUBSYSTEM_LABEL_ALPHA_MAX
        ),
        (
            zoom_min
            if zoom_min is not None
            else VisualizationConstants.SUBSYSTEM_LABEL_ZOOM_THRESHOLD_MIN
        ),
        (
            zoom_max
            if zoom_max is not None
            else VisualizationConstants.SUBSYSTEM_LABEL_ZOOM_THRESHOLD_MAX
        ),
    )


def _calculate_zoom_based_alpha(plot, alpha_min, alpha_max, zoom_min, zoom_max):
    """Compute the current label alpha from the plot viewport."""
    x_span = plot.x_range.end - plot.x_range.start
    y_span = plot.y_range.end - plot.y_range.start
    zoom_level = min(1.0, max(0.1, 1000.0 / max(x_span, y_span, 1e-9)))

    if zoom_level <= zoom_min:
        return alpha_min

    if zoom_level >= zoom_max:
        return alpha_max

    interpolation = (zoom_level - zoom_min) / (zoom_max - zoom_min)
    return alpha_min + interpolation * (alpha_max - alpha_min)


def _apply_initial_zoom_alpha(
    plot,
    label_source,
    alpha_min,
    alpha_max,
    zoom_min,
    zoom_max,
):
    """Seed label alpha from the initial plot ranges before JS callbacks run."""
    data = dict(label_source.data)
    point_count = len(data.get("x", []))
    if point_count == 0:
        return

    base_alpha = data.get("base_alpha") or [1.0] * point_count
    alpha = _calculate_zoom_based_alpha(
        plot,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        zoom_min=zoom_min,
        zoom_max=zoom_max,
    )
    data["text_alpha"] = [
        alpha * (base_alpha[index] if base_alpha[index] is not None else 1.0)
        for index in range(point_count)
    ]
    label_source.data = data


def add_zoom_based_alpha_callback(
    plot,
    label_source,
    glyph_type="labels",
    alpha_min=None,
    alpha_max=None,
    zoom_min=None,
    zoom_max=None,
):
    """Add JavaScript callback to dynamically adjust label alpha based on zoom level.

    Args:
        plot: Bokeh plot figure
        label_source: ColumnDataSource for the labels
        glyph_type: String describing the type of glyph (for debugging)
        alpha_min: Minimum alpha value (defaults to subsystem settings)
        alpha_max: Maximum alpha value (defaults to subsystem settings)
        zoom_min: Zoom level for minimum alpha (defaults to subsystem settings)
        zoom_max: Zoom level for maximum alpha (defaults to subsystem settings)
    """
    alpha_min, alpha_max, zoom_min, zoom_max = _resolve_zoom_alpha_parameters(
        alpha_min,
        alpha_max,
        zoom_min,
        zoom_max,
    )

    _apply_initial_zoom_alpha(
        plot,
        label_source,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        zoom_min=zoom_min,
        zoom_max=zoom_max,
    )

    callback = CustomJS(
        args=dict(
            source=label_source,
            plot=plot,
            glyph_type=glyph_type,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            zoom_min=zoom_min,
            zoom_max=zoom_max,
        ),
        code="""
        
        // Get current zoom level from the plot range
        const x_range = plot.x_range;
        const y_range = plot.y_range;
        
        // Calculate zoom level based on the range span
        // Smaller range = more zoomed in, larger range = more zoomed out
        const x_span = x_range.end - x_range.start;
        const y_span = y_range.end - y_range.start;
        const zoom_level = Math.min(1.0, Math.max(0.1,
            1000.0 / Math.max(x_span, y_span, 1e-9)));
    
        // Calculate alpha based on zoom level
        let alpha;
        if (zoom_level <= zoom_min) {
            alpha = alpha_min;
        } else if (zoom_level >= zoom_max) {
            alpha = alpha_max;
        } else {
            // Linear interpolation between min and max
            const t = (zoom_level - zoom_min) / (zoom_max - zoom_min);
            alpha = alpha_min + t * (alpha_max - alpha_min);
        }
        
        // Update all label alphas, factoring in base_alpha if provided
        const data = source.data;
        const n = data.x.length;
        const base = data.base_alpha || Array(n).fill(1.0);
        
        for (let i = 0; i < n; i++) {
            const base_alpha = base[i] != null ? base[i] : 1.0;
            data.text_alpha[i] = alpha * base_alpha;
        }

        // Trigger data source change to update the plot
        source.change.emit();
        """,
    )

    # Attach the callback to the plot's range changes
    plot.x_range.js_on_change("start", callback)
    plot.x_range.js_on_change("end", callback)
    plot.y_range.js_on_change("start", callback)
    plot.y_range.js_on_change("end", callback)


def add_factor_range_zoom_alpha_callback(
    plot,
    subsystem_label_source,
    category_label_source=None,
    subsystem_alpha_min=None,
    subsystem_alpha_max=None,
    subsystem_zoom_min=None,
    subsystem_zoom_max=None,
    category_alpha_min=None,
    category_alpha_max=None,
):
    """Add JavaScript callback to adjust label alpha based on FactorRange zoom level.

    For subsystem comparison plots with nested FactorRange, this callback adjusts
    label visibility based on the number of visible factors (y-range span).

    When zoomed out (many factors visible): category labels bright, subsystem labels dim
    When zoomed in (few factors visible): subsystem labels bright, category labels dimmer

    Args:
        plot: Bokeh plot figure with FactorRange y_range
        subsystem_label_source: ColumnDataSource for subsystem labels
        category_label_source: ColumnDataSource for category labels (optional)
        subsystem_alpha_min: Min alpha for subsystem labels when zoomed out
        subsystem_alpha_max: Max alpha for subsystem labels when zoomed in
        subsystem_zoom_min: Span threshold for max alpha (few factors = zoomed in)
        subsystem_zoom_max: Span threshold for min alpha (many factors = zoomed out)
        category_alpha_min: Min alpha for category labels when zoomed in
        category_alpha_max: Max alpha for category labels when zoomed out
    """
    # Use constants as defaults
    subsystem_alpha_min = (
        subsystem_alpha_min
        if subsystem_alpha_min is not None
        else VisualizationConstants.COMPARISON_SUBSYSTEM_LABEL_ALPHA_MIN
    )
    subsystem_alpha_max = (
        subsystem_alpha_max
        if subsystem_alpha_max is not None
        else VisualizationConstants.COMPARISON_SUBSYSTEM_LABEL_ALPHA_MAX
    )
    subsystem_zoom_min = (
        subsystem_zoom_min
        if subsystem_zoom_min is not None
        else VisualizationConstants.COMPARISON_SUBSYSTEM_ZOOM_THRESHOLD_MIN
    )
    subsystem_zoom_max = (
        subsystem_zoom_max
        if subsystem_zoom_max is not None
        else VisualizationConstants.COMPARISON_SUBSYSTEM_ZOOM_THRESHOLD_MAX
    )
    category_alpha_min = (
        category_alpha_min
        if category_alpha_min is not None
        else VisualizationConstants.COMPARISON_CATEGORY_LABEL_ALPHA_MIN
    )
    category_alpha_max = (
        category_alpha_max
        if category_alpha_max is not None
        else VisualizationConstants.COMPARISON_CATEGORY_LABEL_ALPHA_MAX
    )

    # Build args dictionary
    callback_args = dict(
        plot=plot,
        subsystem_alpha_min=subsystem_alpha_min,
        subsystem_alpha_max=subsystem_alpha_max,
        zoom_min=subsystem_zoom_min,
        zoom_max=subsystem_zoom_max,
        category_alpha_min=category_alpha_min,
        category_alpha_max=category_alpha_max,
    )

    if subsystem_label_source is not None:
        callback_args["subsystem_source"] = subsystem_label_source

    if category_label_source is not None:
        callback_args["category_source"] = category_label_source

    callback = CustomJS(
        args=callback_args,
        code="""
        // Get current y-range span from FactorRange
        // For FactorRange, start/end are in synthetic coordinates
        const y_range = plot.y_range;
        const y_span = Math.abs(y_range.end - y_range.start);

        // Update subsystem label alphas (if source provided)
        if (typeof subsystem_source !== 'undefined') {
            // Calculate subsystem label alpha based on visible span
            // Smaller span = more zoomed in = higher alpha
            // Larger span = more zoomed out = lower alpha
            let subsystem_alpha;
            if (y_span <= zoom_min) {
                // Very zoomed in - full visibility
                subsystem_alpha = subsystem_alpha_max;
            } else if (y_span >= zoom_max) {
                // Very zoomed out - minimum visibility
                subsystem_alpha = subsystem_alpha_min;
            } else {
                // Linear interpolation between thresholds
                // Note: inverted because larger span = lower alpha
                const t = (y_span - zoom_min) / (zoom_max - zoom_min);
                subsystem_alpha = subsystem_alpha_max - t * (subsystem_alpha_max - subsystem_alpha_min);
            }

            const sub_data = subsystem_source.data;
            const n_sub = sub_data.x.length;
            for (let i = 0; i < n_sub; i++) {
                sub_data.text_alpha[i] = subsystem_alpha;
            }
            subsystem_source.change.emit();
        }

        // Update category label alphas (inverse behavior - brighter when zoomed out)
        if (typeof category_source !== 'undefined') {
            let category_alpha;
            if (y_span <= zoom_min) {
                // Very zoomed in - dimmer categories
                category_alpha = category_alpha_min;
            } else if (y_span >= zoom_max) {
                // Very zoomed out - full visibility
                category_alpha = category_alpha_max;
            } else {
                // Linear interpolation (normal direction for categories)
                const t = (y_span - zoom_min) / (zoom_max - zoom_min);
                category_alpha = category_alpha_min + t * (category_alpha_max - category_alpha_min);
            }

            const cat_data = category_source.data;
            const n_cat = cat_data.x.length;
            for (let i = 0; i < n_cat; i++) {
                cat_data.text_alpha[i] = category_alpha;
            }
            category_source.change.emit();
        }
        """,
    )

    # Attach callback to y_range changes (start/end for FactorRange)
    plot.y_range.js_on_change("start", callback)
    plot.y_range.js_on_change("end", callback)

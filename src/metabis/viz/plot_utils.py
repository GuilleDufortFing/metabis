import numpy as np

# Create new ColumnDataSource objects
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.themes import Theme

# --- Academic Theme Definition and Helper ---

ACADEMIC_THEME_JSON = {
    "attrs": {
        "Figure": {
            "background_fill_color": "#FFFFFF",
            "border_fill_color": "#FFFFFF",
            "outline_line_color": "#444444",
            "outline_line_width": 1,
            "min_border_left": 20,
            "min_border_right": 20,
            "min_border_top": 20,
            "min_border_bottom": 20,
            "toolbar_location": "right",
        },
        "Axis": {
            "axis_line_color": "#333333",
            "axis_line_width": 1.2,
            "major_tick_in": 5,
            "major_tick_out": 0,
            "major_tick_line_color": "#666666",
            "major_tick_line_width": 1,
            "minor_tick_line_color": None,
            "axis_label_text_font": "Helvetica",
            "axis_label_text_font_size": "11pt",
            "axis_label_text_font_style": "normal",
            "axis_label_standoff": 12,
            "major_label_text_font": "Helvetica",
            "major_label_text_font_size": "9pt",
            "major_label_standoff": 8,
            "major_label_orientation": "horizontal",
        },
        "Grid": {
            "grid_line_color": "#E0E0E0",
            "grid_line_dash": [3, 3],
            "grid_line_width": 0.8,
            "minor_grid_line_color": None,
        },
        "Legend": {
            "background_fill_color": "#FFFFFF",
            "background_fill_alpha": 0.80,
            "border_line_color": "#DDDDDD",
            "border_line_width": 0.5,
            "label_text_font": "Helvetica",
            "label_text_font_size": "9pt",
            "title_text_font": "Helvetica",
            "title_text_font_style": "bold",
            "title_text_font_size": "10pt",
            "spacing": 4,
            "padding": 7,
            "margin": 10,
            "glyph_width": 15,
            "label_standoff": 5,
            "click_policy": "hide",
            "orientation": "vertical",
        },
        "Title": {
            "text_font": "Helvetica",
            "text_font_size": "13pt",
            "text_font_style": "bold",
            "align": "center",
            "text_color": "#333333",
            "standoff": 10,
        },
        "Line": {
            "line_width": 1.75,
            "line_alpha": 0.8,
        },
        "Scatter": {
            "size": 15,
            "alpha": 0.8,
        },
        "Text": {
            "text_font": "Helvetica",
            "text_font_size": "8pt",
            "text_color": "#555555",
        },
    }
}

def apply_academic_style(
    p: figure, suppress_grid: bool = False, minimal_axes: bool = False
) -> figure:
    """Applies the predefined academic theme and common styling adjustments."""
    Theme(json=ACADEMIC_THEME_JSON).apply_to_model(p)
    p.toolbar.autohide = True

    if suppress_grid:
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

    if minimal_axes:
        # Hide top and right axis lines and ticks for a cleaner look
        if p.right:
            for axis_obj in p.right:
                axis_obj.visible = False
        if p.above:
            for axis_obj in p.above:
                axis_obj.visible = False

    return p

def format_scientific(value, threshold=0.01):
    """Format number in scientific notation if small enough."""
    if abs(value) < threshold and value != 0:
        return f"{value:.2e}"
    return f"{value:.3f}"


# Function to subsample data while preserving outliers
def subsample_with_outliers(source_data, x_col, y_col, max_points, outlier_threshold):
    # Extract x and y values from the source
    x_values = source_data.data[x_col]
    y_values = source_data.data[y_col]

    if len(x_values) <= max_points:
        # No subsampling needed
        return source_data, source_data

    # Calculate z-scores to identify outliers
    from scipy import stats

    x_zscore = np.abs(stats.zscore(x_values, nan_policy="omit"))
    y_zscore = np.abs(stats.zscore(y_values, nan_policy="omit"))

    # Identify outliers as points where either x or y exceeds threshold
    outlier_indices = np.where(
        (x_zscore > outlier_threshold) | (y_zscore > outlier_threshold)
    )[0]
    non_outlier_indices = np.where(
        (x_zscore <= outlier_threshold) & (y_zscore <= outlier_threshold)
    )[0]

    # If we have too many non-outliers, randomly subsample them
    if len(non_outlier_indices) > (max_points - len(outlier_indices)):
        # Calculate how many non-outliers we can include
        n_to_sample = max(0, max_points - len(outlier_indices))
        # Random subsample of non-outliers
        if n_to_sample > 0:
            np.random.seed(42)  # For reproducibility
            sampled_indices = np.random.choice(
                non_outlier_indices, size=n_to_sample, replace=False
            )
        else:
            sampled_indices = np.array([])
    else:
        # We can include all non-outliers
        sampled_indices = non_outlier_indices

    # Combine outliers and sampled non-outliers
    final_indices = np.concatenate([outlier_indices, sampled_indices])

    # Create new data dictionaries
    outlier_data = {
        k: [v[i] for i in outlier_indices] for k, v in source_data.data.items()
    }
    sampled_data = {
        k: [v[i] for i in final_indices] for k, v in source_data.data.items()
    }

    outlier_source = ColumnDataSource(data=outlier_data)
    sampled_source = ColumnDataSource(data=sampled_data)

    return sampled_source, outlier_source

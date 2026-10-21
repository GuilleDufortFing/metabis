import numpy as np

# Create new ColumnDataSource objects
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.themes import Theme

from ..visual_config import theme as visual_theme

# --- The plot theme ---
#
# Bokeh matches theme entries to model classes by name, through each class's
# ancestors, and only when the theme is set on the Document that holds the models:
# applied to one figure it reaches that figure alone, never its axes, legends or
# title. So the theme is set where figures are serialized (the API's json_item, the
# Bokeh app's curdoc) and keyed on "Plot", since the figure class itself is the
# lowercase ``figure``. Every value comes from the React client's tokens (see
# metabis.visual_config); explicit settings on a model still win over it.

_FONT = visual_theme.FONT

PLOT_THEME_JSON = {
    "attrs": {
        "Plot": {
            "background_fill_color": "#ffffff",
            "border_fill_color": "#ffffff",
            "outline_line_color": None,
        },
        "Toolbar": {
            "autohide": True,
            "logo": None,
        },
        "Title": {
            "text_font": _FONT,
            "text_font_size": "14px",
            "text_font_style": "bold",
            "text_color": visual_theme.INK,
        },
        "Axis": {
            "axis_label_text_font": _FONT,
            "axis_label_text_font_size": "13px",
            "axis_label_text_font_style": "normal",
            "axis_label_text_color": visual_theme.INK,
            "major_label_text_font": _FONT,
            "major_label_text_font_size": "12px",
            "major_label_text_color": visual_theme.INK_MUTED,
            "axis_line_color": visual_theme.FRAME_LINE,
            "major_tick_line_color": visual_theme.FRAME_LINE,
            "major_tick_in": 0,
            "major_tick_out": 4,
            "minor_tick_line_color": None,
        },
        "Grid": {
            "grid_line_color": visual_theme.GRID_LINE,
            "minor_grid_line_color": None,
        },
        "Legend": {
            "label_text_font": _FONT,
            "label_text_font_size": "12px",
            "label_text_color": visual_theme.INK,
            "title_text_font": _FONT,
            "title_text_font_size": "12px",
            "title_text_font_style": "bold",
            "title_text_color": visual_theme.INK,
            "border_line_color": None,
            "background_fill_alpha": 0.9,
            "click_policy": "hide",
        },
        "ColorBar": {
            "title_text_font": _FONT,
            "title_text_font_size": "12px",
            "title_text_font_style": "normal",
            "title_text_color": visual_theme.INK,
            "major_label_text_font": _FONT,
            "major_label_text_font_size": "11px",
            "major_label_text_color": visual_theme.INK_MUTED,
            "border_line_color": None,
            "bar_line_color": None,
        },
        "Label": {"text_font": _FONT, "text_color": visual_theme.INK},
        "LabelSet": {"text_font": _FONT, "text_color": visual_theme.INK},
    }
}

PLOT_THEME = Theme(json=PLOT_THEME_JSON)


def apply_academic_style(
    p: figure, suppress_grid: bool = False, minimal_axes: bool = False
) -> figure:
    """Figure-level settings the document theme cannot reach.

    The theme itself is set on the document (see ``PLOT_THEME``); this keeps the
    toolbar out of sight until the pointer is over the plot, and drops the Bokeh
    logo, for figures rendered without that theme too.
    """
    p.toolbar.autohide = True
    p.toolbar.logo = None

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

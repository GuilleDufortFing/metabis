from typing import Dict, List

import numpy as np
from bokeh.models import (  # Add more Bokeh models
    Label,
)
from bokeh.palettes import Category10  # Add palettes
from bokeh.plotting import figure
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

# Consistent palettes (can be expanded and managed centrally)
DEFAULT_CATEGORICAL_PALETTE = Category10[10]
DEFAULT_SINGLE_COLOR = DEFAULT_CATEGORICAL_PALETTE[0]  # Fallback color


def calculate_cluster_metrics(
    coordinates: np.ndarray, group_labels: List[str]
) -> Dict[str, float]:
    """Calculate metrics to quantify cluster separation.

    Args:
        coordinates: 2D array of shape (n_samples, n_features) with reduced dimension coordinates
        group_labels: List of group labels for each sample

    Returns:
        Dict[str, float]: Dictionary with different separation metrics
    """
    # Need at least 2 clusters for these metrics
    unique_labels = np.unique(group_labels)
    if len(unique_labels) < 2:
        return {}

    # Need at least 1 sample per cluster
    for label in unique_labels:
        if np.sum(np.array(group_labels) == label) < 1:
            return {}

    metrics = {}

    # Silhouette score (-1 to 1, higher is better)
    # Measures how similar points are to their own cluster vs other clusters
    try:
        metrics["silhouette"] = silhouette_score(coordinates, group_labels)
    except Exception:
        pass

    # Davies-Bouldin index (lower is better)
    # Ratio of within-cluster to between-cluster distances
    try:
        metrics["davies_bouldin"] = davies_bouldin_score(coordinates, group_labels)
    except Exception:
        pass

    # Calinski-Harabasz index (higher is better)
    # Ratio of between-cluster to within-cluster dispersion
    try:
        metrics["calinski_harabasz"] = calinski_harabasz_score(
            coordinates, group_labels
        )
    except Exception:
        pass

    return metrics


def get_metric_interpretation(metric_name: str, value: float) -> str:
    """Get human-readable interpretation of metric value.

    Args:
        metric_name: Name of the metric
        value: Numeric value of the metric

    Returns:
        str: Qualitative interpretation (Excellent/Good/Fair/Poor)
    """
    if metric_name == "silhouette":
        if value > 0.7:
            return "Excellent"
        elif value > 0.5:
            return "Good"
        elif value > 0.25:
            return "Fair"
        else:
            return "Poor"
    elif metric_name == "davies_bouldin":
        if value < 0.5:
            return "Excellent"
        elif value < 1.0:
            return "Good"
        elif value < 1.5:
            return "Fair"
        else:
            return "Poor"
    elif metric_name == "calinski_harabasz":
        # This is more context-dependent, so use a simpler scale
        if value > 100:
            return "Excellent"
        elif value > 50:
            return "Good"
        elif value > 10:
            return "Fair"
        else:
            return "Poor"
    return ""


# Beyond this many entries a legend drawn inside the frame runs past the bottom of the
# plot and the last groups become unreadable.
MAX_LEGEND_ITEMS_INSIDE_PLOT = 6


def move_legend_outside_if_crowded(p: figure, num_legend_items: int) -> None:
    """Place a crowded legend beside the plot instead of clipping it inside the frame."""
    if num_legend_items <= MAX_LEGEND_ITEMS_INSIDE_PLOT or not p.legend:
        return

    legend = p.legend[0]
    if legend in p.right:
        return

    p.add_layout(legend, "right")
    legend.location = "top_left"
    legend.border_line_color = None
    legend.background_fill_alpha = 0.0


def add_cluster_metrics_to_plot(
    p: figure, metrics: Dict[str, float], base_title: str
) -> None:
    """Updates plot title and adds metrics annotation to the figure.

    Args:
        p: The Bokeh figure to update
        metrics: Dictionary of calculated metrics
        base_title: The original title of the plot
    """
    if not metrics:
        return

    # Update title with Silhouette score
    if "silhouette" in metrics:
        silhouette = metrics["silhouette"]
        quality = get_metric_interpretation("silhouette", silhouette)
        p.title.text = f"{base_title} - Cluster Sep: {quality} (Sil: {silhouette:.2f})"

    # Add annotation for other metrics
    metrics_text_lines = []
    if "davies_bouldin" in metrics:
        db = metrics["davies_bouldin"]
        db_q = get_metric_interpretation("davies_bouldin", db)
        metrics_text_lines.append(f"Davies-Bouldin: {db:.2f} ({db_q})")
    if "calinski_harabasz" in metrics:
        ch = metrics["calinski_harabasz"]
        ch_q = get_metric_interpretation("calinski_harabasz", ch)
        metrics_text_lines.append(f"Calinski-Harabasz: {ch:.1f} ({ch_q})")

    if metrics_text_lines:
        metrics_label = Label(
            x=10,
            y=p.height - 25 if p.height else 500,
            x_units="screen",
            y_units="screen",
            text="\n".join(metrics_text_lines),
            text_font_size="7pt",
            text_font_style="italic",
            text_align="left",
            background_fill_color="white",
            background_fill_alpha=0.6,
            border_line_color="#CCCCCC",
            border_line_alpha=0.5,
            border_line_width=0.5,
            padding=4,
        )
        p.add_layout(metrics_label)

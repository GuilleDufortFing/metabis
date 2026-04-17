from typing import Dict, List, Optional

import numpy as np
import pandas as pd  # Add pandas
from bokeh.models import (  # Add more Bokeh models
    Arrow,
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    FactorRange,
    HoverTool,
    Label,
    LinearColorMapper,
    Range1d,
    VeeHead,
)
from bokeh.palettes import Category10  # Add palettes
from bokeh.plotting import figure, show
from bokeh.transform import transform  # Add transform
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from ..visual_config import theme
from .click_behavior import add_click_behavior, get_external_reference_data
from .plot_utils import apply_academic_style

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


def plot_reduced_dimension(
    data_manager,
    method: str = "PCA",
    data_type: str = "reactions",
    highlight_group: Optional[str] = None,
    figsize: tuple = (800, 600),
    n_variable_features: int = 500,
    return_plot: bool = False,
    show_separation_metrics: bool = True,
    p=None,
) -> Optional[figure]:
    """Plot an interactive 2D dimensionality reduction of the experiment data.
    (Docstring remains the same)
    """
    # Validate method
    method = method.upper()
    if method not in ["PCA", "T-SNE", "UMAP"]:
        raise ValueError("Method must be one of: 'PCA', 'T-SNE', or 'UMAP'")

    experiment_analysis = getattr(data_manager, "experiment_analysis", None)
    if experiment_analysis is None:
        raise ValueError("experiment_analysis not available in data_manager")

    # Ensure results are available or calculate them
    results_dict = None

    if method == "PCA":
        if data_type not in experiment_analysis.pca_results:
            print(f"Calculating PCA for {data_type}...")
            res = experiment_analysis.calculate_pca(
                data_type=data_type, n_variable_features=n_variable_features
            )
            if res["status"] != "success":
                raise ValueError(f"PCA calculation failed: {res['messages']}")

        if data_type in experiment_analysis.pca_results:
            results_dict = experiment_analysis.pca_results[data_type]

    elif method == "T-SNE":
        if data_type not in experiment_analysis.tsne_results:
            print(f"Calculating t-SNE for {data_type}...")
            res = experiment_analysis.calculate_tsne(
                data_type=data_type, n_variable_features=n_variable_features
            )
            if res["status"] != "success":
                raise ValueError(f"t-SNE calculation failed: {res['messages']}")

        if data_type in experiment_analysis.tsne_results:
            results_dict = experiment_analysis.tsne_results[data_type]

    elif method == "UMAP":
        if data_type not in experiment_analysis.umap_results:
            print(f"Calculating UMAP for {data_type}...")
            res = experiment_analysis.calculate_umap(
                data_type=data_type, n_variable_features=n_variable_features
            )
            if res["status"] != "success":
                raise ValueError(f"UMAP calculation failed: {res['messages']}")

        if data_type in experiment_analysis.umap_results:
            results_dict = experiment_analysis.umap_results[data_type]

    if results_dict is None:
        raise ValueError(f"Could not obtain {method} results for {data_type}")

    coordinates = results_dict["transformed_data"]
    sample_ids = results_dict["sample_ids"]

    plot_groups = {}
    # Populate plot_groups if DataManager/Experiment has groups
    try:
        group_manager = getattr(data_manager, "group_manager", None)
        if group_manager and hasattr(group_manager, "groups") and group_manager.groups:
            # groups expected as dict[group_id] -> list[sample_ids]
            plot_groups = group_manager.groups.copy()
        # Fallback: attribute named groups directly
        elif hasattr(data_manager, "groups") and isinstance(data_manager.groups, dict):
            plot_groups = data_manager.groups.copy()
    except Exception:
        plot_groups = {}

    # Prefer color manager colors if available
    group_colors_map = {}
    try:
        if hasattr(data_manager, "color_manager"):
            group_colors_map = data_manager.color_manager.get_group_colors()
        elif hasattr(data_manager, "get_group_colors"):
            group_colors_map = data_manager.get_group_colors()
    except Exception:
        group_colors_map = {}

    # Calculate cluster separation metrics
    separation_metrics = {}
    if show_separation_metrics and plot_groups and len(plot_groups) >= 2:
        sample_to_group = {
            sample_id: group_id
            for group_id, samples_in_group in plot_groups.items()
            for sample_id in samples_in_group
        }
        valid_coords, valid_labels = [], []
        for i, sample_id_val in enumerate(sample_ids):
            if sample_id_val in sample_to_group:
                valid_coords.append(coordinates[i])
                valid_labels.append(sample_to_group[sample_id_val])

        if len(valid_coords) > 0 and len(set(valid_labels)) >= 2:
            valid_coords_np = np.array(valid_coords)
            if valid_coords_np.shape[0] >= 2 * len(
                set(valid_labels)
            ):  # Ensure enough samples per cluster for metrics
                separation_metrics = calculate_cluster_metrics(
                    valid_coords_np, valid_labels
                )

    # Create Bokeh figure title
    title = f"{method} of {data_type.capitalize()} Data"

    if separation_metrics and "silhouette" in separation_metrics:
        silhouette = separation_metrics["silhouette"]
        quality = get_metric_interpretation("silhouette", silhouette)
        title += f" - Cluster Sep: {quality} (Sil: {silhouette:.2f})"

    if not p:
        p = figure(
            width=figsize[0],
            height=figsize[1],
            title=title,
            tools="pan,box_zoom,wheel_zoom,reset,save",  # Hover will be added later
        )
        p = apply_academic_style(p)  # Apply the theme

    # Define colors for groups
    unique_plot_group_ids = list(plot_groups.keys())

    if hasattr(data_manager, "group_colors") and data_manager.group_colors:
        # Merge existing colors if needed, or just use them
        for gid, color in data_manager.group_colors.items():
            group_colors_map[gid] = color

    # Fill in missing colors from palette
    palette_to_use = DEFAULT_CATEGORICAL_PALETTE
    for i, group_id in enumerate(unique_plot_group_ids):
        if group_id not in group_colors_map:
            group_colors_map[group_id] = palette_to_use[i % len(palette_to_use)]

    # Plot points
    if plot_groups:
        for group_id, group_samples_list in plot_groups.items():
            sample_indices = [
                i for i, s_id in enumerate(sample_ids) if s_id in group_samples_list
            ]
            if sample_indices:
                group_coords = coordinates[sample_indices]
                group_samples_found = [sample_ids[i] for i in sample_indices]
                source = ColumnDataSource(
                    data=dict(
                        x=group_coords[:, 0],
                        y=group_coords[:, 1],
                        sample=group_samples_found,
                        group=[group_id] * len(group_samples_found),
                    )
                )
                color = group_colors_map.get(group_id, DEFAULT_SINGLE_COLOR)
                p.scatter(
                    "x",
                    "y",
                    source=source,
                    color=color,
                    legend_label=group_id,
                    # size and alpha are handled by theme's Scatter defaults
                )
                if (
                    highlight_group
                    and group_id == highlight_group
                    and group_coords.shape[0] > 0
                ):
                    centroid = group_coords.mean(axis=0)
                    p.text(  # Style from theme's Text, can be overridden
                        x=[centroid[0]],
                        y=[centroid[1]],
                        text=[group_id],
                        text_font_size="10pt",
                        text_baseline="middle",
                        text_align="center",
                        text_color=color,
                        text_font_style="bold",  # Make highlighted label stand out
                    )
    else:  # No groups, plot all points
        source = ColumnDataSource(
            data=dict(
                x=coordinates[:, 0],
                y=coordinates[:, 1],
                sample=list(sample_ids),
                group=["All samples"] * len(sample_ids),
            )
        )
        p.scatter(
            "x",
            "y",
            source=source,
            color=DEFAULT_SINGLE_COLOR,
            legend_label="All samples",
        )

    hover = HoverTool(
        tooltips=[
            ("Sample", "@sample"),
            ("Group", "@group"),
            ("X", "@x{0.00}"),
            ("Y", "@y{0.00}"),
        ]
    )
    p.add_tools(hover)

    p.xaxis.axis_label = f"{method} Component 1"
    p.yaxis.axis_label = f"{method} Component 2"

    # Legend styling (theme provides base, specific adjustments here)
    num_legend_items = len(list(plot_groups.keys()) if plot_groups else ["No groups"])
    max_rows_per_column = 25  # Max items in one legend column before adding another
    num_legend_columns = max(
        1, (num_legend_items + max_rows_per_column - 1) // max_rows_per_column
    )

    p.legend.location = "right"
    p.legend.orientation = "vertical"
    # p.legend.label_text_font_size = "8pt" # Theme sets this
    # p.legend.background_fill_alpha = 0.7 # Theme sets this
    # p.legend.padding = 5 # Theme sets this
    # p.legend.margin = 5 # Theme sets this
    # p.legend.spacing = 1 # Theme sets this

    if num_legend_columns > 1:
        # Bokeh's default legend doesn't directly support multi-column layout in the same legend object.
        # The previous approach of adjusting p.width and using p.legend.ncols is not standard.
        # For true multi-column, one might need multiple legend objects or an external HTML legend.
        # Keeping it simple: if many items, legend might be long.
        p.legend.items = sorted(
            p.legend.items, key=lambda item: item.label["value"]
        )  # Sort legend items
        pass  # Theme handles single column legend styling. Multi-column is complex.

    # Add additional metrics as text annotations
    add_cluster_metrics_to_plot(p, separation_metrics, title)

    # Remove x/y grid lines for cleaner scatter plot if desired
    # p.xgrid.grid_line_color = None
    # p.ygrid.grid_line_color = None
    # Or use apply_academic_style(p, suppress_grid=True)

    if return_plot:
        return p
    else:
        show(p)


def plot_pca_explained_variance(
    data_manager,
    data_type: str = "reactions",
    return_plot: bool = False,
    figsize: tuple = (700, 450),  # Adjusted default size
) -> Optional[figure]:
    """Plots the explained variance and cumulative explained variance for PCA components.

    Args:
        data_manager: DataManager object containing experiment_analysis
        data_type: Type of data ('reactions' or 'expression')
        return_plot: If True, returns the plot object; otherwise displays it
        figsize: Tuple of (width, height) for the plot

    Returns:
        Optional[figure]: Bokeh figure if return_plot=True, otherwise None
    """
    # Extract experiment_analysis from data_manager
    experiment_analysis = getattr(data_manager, "experiment_analysis", None)
    if experiment_analysis is None:
        raise ValueError("experiment_analysis not available in data_manager")

    if data_type not in experiment_analysis.pca_results:
        try:
            experiment_analysis.calculate_pca(
                data_type=data_type, n_variable_features=10
            )
        except:
            raise ValueError(
                f"PCA results for '{data_type}' not found. Run experiment_analysis.calculate_pca() first."
            )

    pca_data = experiment_analysis.pca_results[data_type]
    explained_variance_ratio = pca_data["explained_variance_ratio"]
    n_components = len(explained_variance_ratio)
    components_indices = np.arange(1, n_components + 1)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    source = ColumnDataSource(
        data={
            "components": components_indices,
            "explained_variance": explained_variance_ratio,
            "cumulative_variance": cumulative_variance,
            "explained_variance_str": [
                f"{x * 100:.1f}%" for x in explained_variance_ratio
            ],
            "cumulative_variance_str": [f"{x * 100:.1f}%" for x in cumulative_variance],
        }
    )

    title = f"PCA Explained Variance: {pca_data['data_source_name']}"
    title += f" ({pca_data['n_components_retained']} PCs from {pca_data['n_variable_features_used']} features)"

    p = figure(
        width=figsize[0],
        height=figsize[1],
        x_range=Range1d(0.5, n_components + 0.5),  # Ensure all bars are visible
        title=title,
        tools="pan,wheel_zoom,box_zoom,reset,save",  # Hover added below
    )
    p = apply_academic_style(p)  # Apply theme
    p.xgrid.grid_line_color = None  # Clean x-axis for bar/line plot

    bar_color = DEFAULT_CATEGORICAL_PALETTE[0]
    line_color = DEFAULT_CATEGORICAL_PALETTE[1]

    p.vbar(
        x="components",
        top="explained_variance",
        width=0.7,
        source=source,
        legend_label="Explained Variance per PC",
        color=bar_color,
        alpha=0.7,
        line_color="#333333",
        line_width=0.5,  # Subtle border for bars
    )
    p.line(
        x="components",
        y="cumulative_variance",
        source=source,
        legend_label="Cumulative Explained Variance",
        line_width=2.5,
        color=line_color,
    )
    p.scatter(  # Markers on the cumulative line
        x="components",
        y="cumulative_variance",
        source=source,
        color=line_color,
        size=6,
        marker="circle",
    )

    hover = HoverTool(
        tooltips=[
            ("PC", "@components"),
            ("Explained Var.", "@explained_variance_str"),
            ("Cumulative Var.", "@cumulative_variance_str"),
        ]
    )
    p.add_tools(hover)

    p.xaxis.axis_label = "Principal Component"
    p.yaxis.axis_label = "Proportion of Variance Explained"
    p.y_range.start = 0
    p.y_range.end = max(1.0, np.max(cumulative_variance) * 1.05)

    p.legend.location = "center_right"
    # p.legend.click_policy = "hide" # Set by theme

    if return_plot:
        return p
    else:
        show(p)


def plot_pca_loadings_heatmap(
    data_manager,
    data_type: str = "reactions",
    n_components_to_show: int = 10,
    n_features_to_show: int = 30,
    sort_features_by_abs_loading_on_pc: Optional[int] = 1,
    return_plot: bool = False,
    figsize: tuple = (800, 600),  # Adjusted default size
    enabled_toggle=None,
    click_filename_prefix: str = "pca_feature",
) -> Optional[figure]:
    """Plots a heatmap of PCA loadings with hover information.

    Args:
        data_manager: DataManager object containing experiment_analysis
        data_type: Type of data ('reactions' or 'expression')
        n_components_to_show: Number of principal components to display
        n_features_to_show: Number of features to display
        sort_features_by_abs_loading_on_pc: PC to use for sorting (1-indexed) or None
        return_plot: If True, returns the plot object; otherwise displays it
        figsize: Tuple of (width, height) for the plot

    Returns:
        Optional[figure]: Bokeh figure if return_plot=True, otherwise None
    """
    # Extract experiment_analysis from data_manager
    experiment_analysis = getattr(data_manager, "experiment_analysis", None)
    if experiment_analysis is None:
        raise ValueError("experiment_analysis not available in data_manager")

    if data_type not in experiment_analysis.pca_results:
        raise ValueError(
            f"PCA results for '{data_type}' not found. Run experiment_analysis.calculate_pca() first."
        )

    pca_data = experiment_analysis.pca_results[data_type]
    loadings = pca_data["components"]
    all_feature_names = pca_data["feature_names"]

    if loadings.shape[0] == 0 or loadings.shape[1] == 0:
        raise ValueError("Loadings matrix is empty.")

    num_actual_components, num_actual_features = loadings.shape
    n_components_to_show = min(n_components_to_show, num_actual_components)

    if sort_features_by_abs_loading_on_pc is not None:
        if not (1 <= sort_features_by_abs_loading_on_pc <= num_actual_components):
            raise ValueError(
                f"PC for sorting must be between 1 and {num_actual_components}"
            )
        pc_idx_sort = sort_features_by_abs_loading_on_pc - 1
        abs_loadings_on_pc = np.abs(loadings[pc_idx_sort, :])
        sorted_feature_indices = np.argsort(abs_loadings_on_pc)[::-1]
    else:
        # Default sort: sum of absolute loadings across displayed components
        sum_abs_loadings = np.sum(np.abs(loadings[:n_components_to_show, :]), axis=0)
        sorted_feature_indices = np.argsort(sum_abs_loadings)[::-1]

    n_features_to_show = min(n_features_to_show, num_actual_features)
    final_feature_indices = sorted_feature_indices[:n_features_to_show]

    selected_loadings = loadings[:n_components_to_show, final_feature_indices]
    selected_feature_names = [all_feature_names[i] for i in final_feature_indices]
    pc_labels = [f"PC{i + 1}" for i in range(n_components_to_show)]

    df_heatmap = pd.DataFrame(
        selected_loadings, index=pc_labels, columns=selected_feature_names
    )
    df_heatmap = df_heatmap.stack().rename("loading").reset_index()
    df_heatmap.columns = ["PC", "Feature", "loading"]

    # Add hover information for features
    element_type = "reaction" if data_type.lower() == "reactions" else "gene"
    hover_texts = []
    element_ids = []
    display_labels = []
    external_urls = []
    for feature_name in df_heatmap["Feature"]:
        reference_data = get_external_reference_data(
            data_manager,
            feature_name,
            element_type,
        )
        element_ids.append(reference_data["element_id"])
        display_labels.append(reference_data["display_label"])
        external_urls.append(reference_data["external_url"])

        try:
            hover_text = data_manager.get_element_hover(
                element_id=feature_name,
                element_type=element_type,
            )
            hover_texts.append(hover_text)
        except Exception:
            # Fallback if hover generation fails
            hover_texts.append(f"{element_type.capitalize()}: {feature_name}")

    df_heatmap["hover"] = hover_texts
    df_heatmap["element_id"] = element_ids
    df_heatmap["display_label"] = display_labels
    df_heatmap["external_url"] = external_urls

    max_abs_loading = (
        np.max(np.abs(selected_loadings)) if selected_loadings.size > 0 else 1.0
    )
    # Using a standard Bokeh palette for diverging data
    color_mapper = LinearColorMapper(
        palette="RdBu11", low=-max_abs_loading, high=max_abs_loading
    )

    title = f"PCA Loadings: {pca_data['data_source_name']} (Top {n_components_to_show} PCs, {n_features_to_show} Features)"
    if sort_features_by_abs_loading_on_pc:
        title += f", sorted by PC{sort_features_by_abs_loading_on_pc}"
    else:
        title += ", sorted by sum of abs. loadings"

    p = figure(
        width=figsize[0],
        height=figsize[1],
        x_range=FactorRange(factors=selected_feature_names),
        y_range=FactorRange(factors=list(reversed(pc_labels))),  # PC1 at top
        title=title,
        tools="save,pan,box_zoom,wheel_zoom,reset",  # Hover added below
        x_axis_location="above",
    )
    p = apply_academic_style(p, suppress_grid=True)  # Suppress grid for heatmap
    p.axis.axis_line_color = None  # Cleaner look for heatmap axes
    p.axis.major_tick_line_color = None

    p.rect(
        x="Feature",
        y="PC",
        width=1,
        height=1,
        source=df_heatmap,
        fill_color=transform("loading", color_mapper),
        line_color=None,  # No lines between cells
    )

    color_bar = ColorBar(
        color_mapper=color_mapper,
        ticker=BasicTicker(desired_num_ticks=7),
        label_standoff=8,
        border_line_color=None,
        location=(0, 0),
        major_label_text_font_size="8pt",  # Match theme's smaller fonts
        title="Loading",
        title_text_font_style="normal",
        title_text_font_size="9pt",
        title_standoff=8,
    )
    p.add_layout(color_bar, "right")

    hover = HoverTool(
        tooltips="""
            <div>
                <div><strong>PC:</strong> @PC</div>
                <div><strong>Loading:</strong> @loading{0.000}</div>
                @hover{safe}
            </div>
        """
    )
    p.add_tools(hover)

    add_click_behavior(
        p,
        filename_prefix=click_filename_prefix,
        enabled_toggle=enabled_toggle,
    )

    p.xaxis.major_label_orientation = np.pi / 4  # Angle labels for readability
    p.xaxis.axis_label = "Features"
    p.yaxis.axis_label = "Principal Components"
    # Font sizes for axis labels and ticks are handled by theme

    if return_plot:
        return p
    else:
        show(p)


def plot_pca_biplot(
    data_manager,
    data_type: str = "reactions",
    pc_x: int = 1,
    pc_y: int = 2,
    highlight_group: Optional[str] = None,
    return_plot: bool = False,
    figsize: tuple = (800, 700),  # Adjusted default size
    arrow_scale_factor: float = 0.7,
    top_n_arrows: Optional[int] = 15,
    show_separation_metrics: bool = True,
    enabled_toggle=None,
    click_filename_prefix: str = "pca_biplot_feature",
) -> Optional[figure]:
    """Creates a PCA biplot showing samples (scores) and variable loadings (arrows) with hover information.

    Args:
        data_manager: DataManager object containing experiment_analysis
        data_type: Type of data ('reactions' or 'expression')
        pc_x: X-axis principal component (1-indexed)
        pc_y: Y-axis principal component (1-indexed)
        highlight_group: Optional group ID to highlight with a label
        return_plot: If True, returns the plot object; otherwise displays it
        figsize: Tuple of (width, height) for the plot
        arrow_scale_factor: Scaling factor for loading arrows
        top_n_arrows: Number of top loading arrows to show (by magnitude)

    Returns:
        Optional[figure]: Bokeh figure if return_plot=True, otherwise None
    """
    # Extract experiment_analysis from data_manager
    experiment_analysis = getattr(data_manager, "experiment_analysis", None)
    if experiment_analysis is None:
        raise ValueError("experiment_analysis not available in data_manager")

    if data_type not in experiment_analysis.pca_results:
        raise ValueError(
            f"PCA results for '{data_type}' not found. Run experiment_analysis.calculate_pca() first."
        )

    pca_data = experiment_analysis.pca_results[data_type]
    scores = pca_data["transformed_data"]
    loadings = pca_data["components"]
    feature_names = pca_data["feature_names"]
    pca_sample_ids = pca_data["sample_ids"]

    num_components = scores.shape[1]
    if not (
        1 <= pc_x <= num_components and 1 <= pc_y <= num_components and pc_x != pc_y
    ):
        raise ValueError(
            f"Invalid PC selection ({pc_x}, {pc_y}) for {num_components} components."
        )
    pc_x_idx, pc_y_idx = pc_x - 1, pc_y - 1

    df_scores = pd.DataFrame(
        {
            "x": scores[:, pc_x_idx],
            "y": scores[:, pc_y_idx],
            "sample_id": pca_sample_ids,
        }
    )

    # Group assignment and coloring (mirrors plot_reduced_dimension logic)
    # data_manager is now passed as parameter
    group_assignments: Dict[str, str] = {}
    candidate_group_ids: List[str] = []

    if data_manager is not None:
        try:
            candidate_group_ids = list(data_manager.get_group_ids())
        except Exception:
            candidate_group_ids = []

        for group_id in candidate_group_ids:
            try:
                samples_in_group = data_manager.get_group_samples(group_id)
            except Exception:
                continue
            for sample_id in samples_in_group:
                group_assignments.setdefault(sample_id, group_id)

    if not group_assignments:
        fallback_groups = getattr(experiment_analysis, "groups", {}) or {}
        for group_id, sample_list in fallback_groups.items():
            for sample_id in sample_list:
                group_assignments.setdefault(sample_id, group_id)
        if not candidate_group_ids:
            candidate_group_ids = list(fallback_groups.keys())

    unassigned_label = "Unassigned" if candidate_group_ids else "N/A"
    df_scores["group"] = [
        group_assignments.get(sample_id, unassigned_label)
        for sample_id in df_scores["sample_id"]
    ]

    # Build color map using DataManager colors when available, fallback to palette
    color_map: Dict[str, str] = {}
    if data_manager is not None:
        try:
            color_map = data_manager.get_group_colors() or {}
        except Exception:
            color_map = {}

    groups_in_plot_order = list(dict.fromkeys(df_scores["group"].tolist()))
    palette_source = DEFAULT_CATEGORICAL_PALETTE

    if not color_map:
        ordered_groups = [
            g
            for g in candidate_group_ids
            if g in groups_in_plot_order and g != unassigned_label
        ]
        if not ordered_groups:
            ordered_groups = [g for g in groups_in_plot_order if g != unassigned_label]
        for idx, group_id in enumerate(ordered_groups):
            color_map[group_id] = palette_source[idx % len(palette_source)]

    if unassigned_label in groups_in_plot_order and unassigned_label not in color_map:
        color_map[unassigned_label] = DEFAULT_SINGLE_COLOR

    df_scores["color"] = [
        color_map.get(group_id, DEFAULT_SINGLE_COLOR) for group_id in df_scores["group"]
    ]

    # Calculate cluster separation metrics
    separation_metrics = {}
    if show_separation_metrics:
        # Filter for valid groups (exclude Unassigned/N/A)
        valid_mask = ~df_scores["group"].isin(["Unassigned", "N/A", "N/A (Outlier)"])
        valid_df = df_scores[valid_mask]

        if not valid_df.empty:
            valid_coords_np = valid_df[["x", "y"]].values
            valid_labels = valid_df["group"].tolist()

            if len(set(valid_labels)) >= 2:
                separation_metrics = calculate_cluster_metrics(
                    valid_coords_np, valid_labels
                )

    title = f"PCA Biplot (PC{pc_x} vs PC{pc_y}) for {pca_data['data_source_name']}"

    p = figure(
        width=figsize[0],
        height=figsize[1],
        title=title,
        tools="pan,wheel_zoom,box_zoom,reset,save",  # Hover added later
    )
    p = apply_academic_style(p)  # Apply theme

    p.xaxis.axis_label = (
        f"PC{pc_x} ({pca_data['explained_variance_ratio'][pc_x_idx] * 100:.1f}%)"
    )
    p.yaxis.axis_label = (
        f"PC{pc_y} ({pca_data['explained_variance_ratio'][pc_y_idx] * 100:.1f}%)"
    )

    # Plot samples by group for legend, ensuring consistent color use
    # Iterate over unique groups present in the *filtered* df_scores for plotting
    # but use master_color_map for legend consistency if some groups have no data points

    plotted_groups_in_order = [
        g
        for g in groups_in_plot_order
        if g != unassigned_label and g != "N/A (Outlier)"
    ]
    if unassigned_label in groups_in_plot_order:
        plotted_groups_in_order.append(unassigned_label)

    # Ensure at least one entry for legend
    if not plotted_groups_in_order and groups_in_plot_order:
        plotted_groups_in_order = groups_in_plot_order

    # Track sample renderers for hover tool
    sample_renderers = []

    for group_name in plotted_groups_in_order:
        group_df = df_scores[df_scores["group"] == group_name]
        if not group_df.empty:
            color_for_legend = color_map.get(group_name, group_df["color"].iloc[0])
            renderer = p.scatter(
                x="x",
                y="y",
                source=ColumnDataSource(group_df),
                legend_label=str(group_name),
                color=color_for_legend,
                name="sample_points",  # Explicit name for identification
                # size and alpha from theme
            )
            sample_renderers.append(renderer)

    if highlight_group and highlight_group in df_scores["group"].unique():
        highlight_df = df_scores[df_scores["group"] == highlight_group]
        if not highlight_df.empty:
            centroid_x, centroid_y = highlight_df["x"].mean(), highlight_df["y"].mean()
            p.text(
                x=[centroid_x],
                y=[centroid_y],
                text=[highlight_group],
                text_font_size="10pt",
                text_baseline="middle",
                text_align="center",
                text_color=color_map.get(highlight_group, "black"),  # Color from map
                text_font_style="bold",
            )

    # Variable Loadings Arrows
    loadings_pcx = loadings[pc_x_idx, :]
    loadings_pcy = loadings[pc_y_idx, :]

    score_range_x = (
        (df_scores["x"].max() - df_scores["x"].min()) if not df_scores.empty else 1.0
    )
    score_range_y = (
        (df_scores["y"].max() - df_scores["y"].min()) if not df_scores.empty else 1.0
    )
    max_score_range = max(score_range_x, score_range_y, 1.0)  # Avoid zero

    arrow_magnitudes = np.sqrt(loadings_pcx**2 + loadings_pcy**2)
    max_abs_loading_val = np.max(arrow_magnitudes) if arrow_magnitudes.size > 0 else 1.0
    if max_abs_loading_val == 0:
        max_abs_loading_val = 1.0

    # Scale arrows to be a fraction of the plot's data range
    effective_arrow_scale = (
        (max_score_range / max_abs_loading_val) * 0.20 * arrow_scale_factor
    )

    df_arrows = pd.DataFrame(
        {
            "x_end": loadings_pcx * effective_arrow_scale,
            "y_end": loadings_pcy * effective_arrow_scale,
            "feature_name": feature_names,
            "magnitude": arrow_magnitudes,
            "x_start": 0,
            "y_start": 0,
        }
    )

    if top_n_arrows is not None and top_n_arrows < len(df_arrows):
        df_arrows = df_arrows.nlargest(top_n_arrows, "magnitude")

    # Add hover information for features (reactions/genes)
    element_type = "reaction" if data_type.lower() == "reactions" else "gene"
    hover_texts = []
    element_ids = []
    display_labels = []
    external_urls = []
    for feature_name in df_arrows["feature_name"]:
        reference_data = get_external_reference_data(
            data_manager,
            feature_name,
            element_type,
        )
        element_ids.append(reference_data["element_id"])
        display_labels.append(reference_data["display_label"])
        external_urls.append(reference_data["external_url"])

        try:
            hover_text = data_manager.get_element_hover(
                element_id=feature_name,
                element_type=element_type,
            )
            hover_texts.append(hover_text)
        except Exception:
            # Fallback if hover generation fails
            hover_texts.append(f"{element_type.capitalize()}: {feature_name}")

    df_arrows["hover"] = hover_texts
    df_arrows["element_id"] = element_ids
    df_arrows["display_label"] = display_labels
    df_arrows["external_url"] = external_urls

    arrow_source = ColumnDataSource(df_arrows)
    arrow_color = theme.ACTIVITY_UP

    p.add_layout(
        Arrow(
            end=VeeHead(
                size=7, fill_color=arrow_color, line_color=arrow_color, line_width=1
            ),
            x_start="x_start",
            y_start="y_start",
            x_end="x_end",
            y_end="y_end",
            source=arrow_source,
            line_color=arrow_color,
            line_width=1.5,
            line_alpha=0.7,
        )
    )

    # Add invisible circles at arrow endpoints for hover interaction
    arrow_hover_renderer = p.circle(
        x="x_end",
        y="y_end",
        source=arrow_source,
        size=10,
        alpha=0,  # Invisible but still interactive
        name="arrow_endpoints",
    )

    p.text(
        x="x_end",
        y="y_end",
        text="feature_name",
        source=arrow_source,
        text_font_size="7pt",
        text_color="#333333",
        text_alpha=0.85,
        text_baseline="middle",
        text_align="left",
        x_offset=4,
        y_offset=4,
    )

    # Hover tool for arrow/feature information
    hover_arrows = HoverTool(
        tooltips="""
            <div>
                <div><strong>Feature:</strong> @feature_name</div>
                <div><strong>Loading Magnitude:</strong> @magnitude{0.000}</div>
                @hover{safe}
            </div>
        """,
        renderers=[arrow_hover_renderer],
        name="arrow_hover",
    )
    p.add_tools(hover_arrows)

    # Hover tool for samples - use the tracked renderers from sample plotting
    hover_biplot = HoverTool(
        tooltips=[
            ("Sample", "@sample_id"),
            ("Group", "@group"),
            (f"PC{pc_x}", "@x{{0.00}}"),
            (f"PC{pc_y}", "@y{{0.00}}"),
        ],
        renderers=sample_renderers,  # Use the list we built during plotting
        name="sample_hover",
    )  # Attach hover only to sample points
    p.add_tools(hover_biplot)

    add_click_behavior(
        p,
        filename_prefix=click_filename_prefix,
        enabled_toggle=enabled_toggle,
    )

    # Add additional metrics as text annotations
    add_cluster_metrics_to_plot(p, separation_metrics, title)

    p.legend.location = "top_right"
    # p.legend.click_policy = "mute" # Set by theme
    # p.legend.label_text_font_size = "8pt" # Set by theme
    # p.legend.background_fill_alpha = 0.6 # Set by theme

    if return_plot:
        return p
    else:
        show(p)

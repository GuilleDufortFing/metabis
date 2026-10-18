from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    DataTable,
    Div,
    HoverTool,
    Span,
    TableColumn,
)
from bokeh.plotting import figure, show

from ..managers.workspace_mode import AnalysisWorkspaceMode
from ..visual_config import theme
from .click_behavior import add_click_behavior, get_external_reference_data
from .plot_histograms import create_bokeh_histogram
from .table_formatting import (
    build_external_link_formatter,
    build_statistical_number_formatter,
    format_statistical_value,
)

_STATISTICAL_METRIC_SPECS = [
    {"field": "adjusted_pval", "title": "adjusted_pval", "width": 110},
    {"field": "pval", "title": "pval", "width": 95},
    {"field": "cohens_d", "title": "Cohen's d (T − B)", "width": 125},
    {"field": "log2_fc", "title": "log2 FC (T − B sign)", "width": 145},
    {"field": "mean_difference", "title": "mean Δ (T − B)", "width": 130},
]


def _get_statistical_results_column_specs(
    element_type: str,
) -> list[dict[str, Any]]:
    """Return the fixed display schema for the on-screen statistical results table.

    Exports are not built from this schema. A downloaded table is derived from
    `GroupComparisonArtifact.export_frame`, which carries full-precision values
    and the calculation's own provenance rather than the display formatting a
    widget happened to apply.
    """
    element_title = element_type.capitalize()
    return [
        {"field": "display_id", "title": element_title, "width": 180, "kind": "link"},
        *[
            {**metric_spec, "kind": "numeric"}
            for metric_spec in _STATISTICAL_METRIC_SPECS
        ],
        {"field": "subsystems", "title": "subsystems", "width": 220, "kind": "text"},
    ]


def _normalize_subsystems(subsystems: Any) -> str:
    """Normalize subsystem values to a pipe-separated string."""
    if subsystems is None:
        return ""

    if isinstance(subsystems, str):
        items = [item.strip() for item in subsystems.split(",") if item.strip()]
    elif isinstance(subsystems, set):
        items = sorted(str(item).strip() for item in subsystems if str(item).strip())
    elif isinstance(subsystems, (list, tuple)):
        items = [str(item).strip() for item in subsystems if str(item).strip()]
    else:
        item = str(subsystems).strip()
        items = [item] if item else []

    deduped_items = list(dict.fromkeys(items))
    return " | ".join(deduped_items)


def _get_statistical_reference_data(
    data_manager,
    element_id: str,
    for_genes: bool = False,
) -> Dict[str, str]:
    """Get display metadata for a reaction or gene table row."""
    metabolic_model = getattr(data_manager, "metabolic_model", None)
    if metabolic_model is None:
        return {
            "raw_id": element_id,
            "display_label": element_id,
            "external_url": "",
        }

    if for_genes:
        reference_data = metabolic_model.get_gene_reference_data(element_id)
    else:
        reference_data = metabolic_model.get_reaction_reference_data(element_id)

    return {
        "raw_id": reference_data.get("raw_id", element_id),
        "display_label": reference_data.get("display_label", element_id),
        "external_url": reference_data.get("external_url", ""),
    }


def _get_statistical_subsystems(
    data_manager,
    element_id: str,
    result_row: pd.Series,
    for_genes: bool = False,
) -> str:
    """Resolve subsystem display values from model metadata with a safe fallback."""
    metabolic_model = getattr(data_manager, "metabolic_model", None)
    if metabolic_model is not None:
        metadata = (
            metabolic_model.get_gene_metadata(element_id)
            if for_genes
            else metabolic_model.get_reaction_metadata(element_id)
        )
        normalized_subsystems = _normalize_subsystems(metadata.get("subsystems"))
        if normalized_subsystems:
            return normalized_subsystems

    return _normalize_subsystems(result_row.get("subsystems"))


def _build_statistical_table_rows(
    data_manager,
    results_df: pd.DataFrame,
    for_genes: bool = False,
) -> list[dict[str, Any]]:
    """Build curated row dictionaries shared by the webpage table and CSV export."""
    rows: list[dict[str, Any]] = []

    for element_id, result_row in results_df.iterrows():
        raw_id = str(element_id)
        reference_data = _get_statistical_reference_data(
            data_manager,
            raw_id,
            for_genes=for_genes,
        )
        rows.append(
            {
                "display_id": reference_data["display_label"],
                "ensembl_id": reference_data["raw_id"] if for_genes else "",
                "external_url": reference_data["external_url"],
                "adjusted_pval": result_row.get("adjusted_pval"),
                "pval": result_row.get("pval"),
                "cohens_d": result_row.get("cohens_d"),
                "log2_fc": result_row.get("log2_fc"),
                "mean_difference": result_row.get("mean_difference"),
                "subsystems": _get_statistical_subsystems(
                    data_manager,
                    raw_id,
                    result_row,
                    for_genes=for_genes,
                ),
            }
        )

    return rows


def _rows_to_column_data(
    rows: list[dict[str, Any]],
    fields: list[str],
) -> dict[str, list[Any]]:
    """Convert row dictionaries into ColumnDataSource-compatible column lists."""
    column_data = {field: [] for field in fields}
    for row_data in rows:
        for field in fields:
            column_data[field].append(row_data.get(field))
    return column_data


def _sort_statistical_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return a consistently sorted statistical results frame."""
    if "adjusted_pval" in results_df.columns:
        return results_df.sort_values(
            by="adjusted_pval",
            ascending=True,
            na_position="last",
            kind="mergesort",
        )
    return results_df


def statistical_results_columns(element_type: str = "reaction") -> list[dict[str, Any]]:
    """Return the on-screen results table's columns, as the Bokeh table shows them.

    Each column has a ``field``, ``title``, ``width`` and ``kind``: ``"link"`` for
    the element column (linked through the row's ``external_url``), ``"numeric"``
    for a metric, and ``"text"`` otherwise. A caller drawing its own table reads
    the schema :func:`create_statistical_results_table` is built from.
    """
    return [dict(spec) for spec in _get_statistical_results_column_specs(element_type)]


def statistical_results_rows(
    data_manager,
    for_genes: bool = False,
    max_rows: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Return the installed comparison's on-screen result rows, smallest adjusted p first.

    Each row carries the fields of :func:`statistical_results_columns` plus
    ``external_url`` and ``ensembl_id`` (empty for reactions). These are display
    values, for reading: a download is built from
    ``GroupComparisonArtifact.export_frame``, at full precision.
    """
    artifact = data_manager.get_group_comparison_artifact()
    if artifact is None:
        raise ValueError("No group comparison set in the data_manager")
    results_df = artifact.get_test_results(for_genes=for_genes)
    if results_df is None or len(results_df) == 0:
        return []
    sorted_results_df = _sort_statistical_results(results_df)
    if max_rows is not None:
        sorted_results_df = sorted_results_df.head(max_rows)
    return _build_statistical_table_rows(
        data_manager, sorted_results_df, for_genes=for_genes
    )


def _create_scatter_figure(
    dm,
    source: ColumnDataSource,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    color_col: str = "color_cohens_d",
    background_fill_color="#ffffff",
    thresholds: dict = None,
    threshold_counts: Optional[tuple] = None,
    width: int = 500,
    height: int = 400,
    element_type: str = "reaction",
    enabled_toggle=None,
    filename_prefix: str = "statistical_element",
    download_flag: Optional[str] = None,
):
    """Create a scatter figure comparing two parameters.

    Args:
        dm: The dm object
        source: Single ColumnDataSource containing all points
        x_col: Column name for x-axis data
        y_col: Column name for y-axis data
        x_label: X-axis label
        y_label: Y-axis label
        color_col: Column name for color data
        thresholds: Dict with threshold values for x and y axes
        threshold_counts: Optional tuple of (count, percentage) for threshold annotation
        width: Figure width in pixels
        height: Figure height in pixels
        element_type: The type of element ("reaction" or "gene")

    Returns:
        Bokeh figure with scatter plot
    """
    # Create figure
    scatter_fig = figure(
        sizing_mode="stretch_width",
        title=f"{x_label} vs {y_label}",
        x_axis_label=x_label,
        y_axis_label=y_label,
        background_fill_color=background_fill_color,
        height=height,
        tools="pan,box_zoom,wheel_zoom,reset,save,hover",
        active_scroll="wheel_zoom",
    )

    # Add scatter points
    scatter_fig.scatter(
        x=x_col,
        y=y_col,
        source=source,
        marker="circle",
        size="size",
        color=color_col,
        alpha="alpha",
    )

    # Add thresholds if provided
    if thresholds:
        x_thresh = thresholds.get("x_thresh")
        y_thresh = thresholds.get("y_thresh")
        x_symmetric = thresholds.get("x_symmetric", False)
        y_symmetric = thresholds.get("y_symmetric", False)

        if x_thresh is not None and x_thresh != 0:
            # Add positive threshold
            scatter_fig.add_layout(
                Span(
                    location=x_thresh,
                    dimension="height",
                    line_color=theme.THRESHOLD_LINE_COLOR,
                    line_dash=theme.THRESHOLD_LINE_DASH,
                    line_width=theme.THRESHOLD_LINE_WIDTH,
                )
            )
            # Add negative threshold if symmetric
            if x_symmetric:
                scatter_fig.add_layout(
                    Span(
                        location=-x_thresh,
                        dimension="height",
                        line_color=theme.THRESHOLD_LINE_COLOR,
                        line_dash=theme.THRESHOLD_LINE_DASH,
                        line_width=theme.THRESHOLD_LINE_WIDTH,
                    )
                )

        if y_thresh is not None and y_thresh != 0:
            # Add positive threshold
            scatter_fig.add_layout(
                Span(
                    location=y_thresh,
                    dimension="width",
                    line_color=theme.THRESHOLD_LINE_COLOR,
                    line_dash=theme.THRESHOLD_LINE_DASH,
                    line_width=theme.THRESHOLD_LINE_WIDTH,
                )
            )
            # Add negative threshold if symmetric
            if y_symmetric:
                scatter_fig.add_layout(
                    Span(
                        location=-y_thresh,
                        dimension="width",
                        line_color=theme.THRESHOLD_LINE_COLOR,
                        line_dash=theme.THRESHOLD_LINE_DASH,
                        line_width=theme.THRESHOLD_LINE_WIDTH,
                    )
                )

    # Configure hover tool
    hover = scatter_fig.select(dict(type=HoverTool))
    hover.tooltips = "@desc{safe}"

    add_click_behavior(
        scatter_fig,
        filename_prefix=filename_prefix,
        enabled_toggle=enabled_toggle,
        download_flag=download_flag,
    )

    return scatter_fig


def _wrap_statistical_plot(plot):
    """Wrap a plot in a flex-friendly card container."""
    return column(
        plot,
        sizing_mode="stretch_width",
        styles={
            "min-width": "0",
        },
        css_classes=["stats-plot-card"],
    )


def _build_responsive_plot_layout(plots: List[Any], plots_per_row: int = 4):
    """Arrange plots into wrapping rows that can collapse on narrow screens."""
    rows = []
    for start_idx in range(0, len(plots), plots_per_row):
        row_plots = [
            _wrap_statistical_plot(plot)
            for plot in plots[start_idx : start_idx + plots_per_row]
        ]
        rows.append(
            row(
                *row_plots,
                sizing_mode="stretch_width",
                styles={
                    "min-width": "0",
                    "gap": "1rem",
                    "align-items": "stretch",
                    "flex-wrap": "wrap",
                },
                css_classes=["stats-plot-row"],
            )
        )

    return column(
        *rows,
        sizing_mode="stretch_width",
        styles={
            "min-width": "0",
            "gap": "1rem",
        },
        css_classes=["stats-plot-grid"],
    )


def plot_statistical_histograms(
    data_manager,
    parameters_to_plot: list = None,
    return_plot: bool = False,
    show_thresholds: bool = True,
    for_genes: bool = False,
    plot_width: int = 500,
    plot_height: int = 400,
    background_fill_color="#ffffff",
    **kwargs,
) -> Optional[Any]:
    """Create a row of histogram plots for statistical parameters.

    Args:
        data_manager: The AnalysisWorkspace object containing dm and group_comparison
        parameters_to_plot: List of parameters to plot ('pval', 'cohens_d', 'log2_fc', 'mean_difference')
        return_plot: Whether to return the plot (True) or show it (False)
        show_thresholds: Whether to show threshold lines on the plots
        for_genes: Whether to plot gene data (True) or reaction data (False)

    Returns:
        Optional[gridplot]: The row of histogram plots if return_plot is True, otherwise None
    """
    # Get necessary components from the data_manager
    group_comparison = data_manager.get_group_comparison_artifact()
    color_manager = data_manager.color_manager

    if not group_comparison:
        raise ValueError("No group comparison set in the data_manager")

    test_results = group_comparison.get_test_results(for_genes=for_genes)

    if show_thresholds:
        # Get thresholds from group comparison
        thresholds = group_comparison.get_thresholds(for_genes=for_genes)

        pval_threshold = thresholds["adjusted_pval"]
        cohens_d_threshold = thresholds["cohens_d"]
        log2_fc_threshold = thresholds["log2_fc"]
        mean_difference_threshold = thresholds["mean_difference"]

    # If no parameters specified, use all
    if parameters_to_plot is None:
        parameters_to_plot = ["pval", "cohens_d", "log2_fc", "mean_difference"]

    # Ensure parameters are valid
    valid_params = ["pval", "cohens_d", "log2_fc", "mean_difference"]
    parameters_to_plot = [p for p in parameters_to_plot if p in valid_params]

    if not parameters_to_plot:
        raise ValueError("No valid parameters to plot specified")

    # Get total number of elements for percentage calculations
    total_elements = len(test_results)

    if show_thresholds:
        # Calculate elements meeting each threshold
        threshold_counts = {
            "pval": sum(test_results["adjusted_pval"] < pval_threshold),
            "cohens_d": sum(abs(test_results["cohens_d"]) > cohens_d_threshold),
            "log2_fc": sum(abs(test_results["log2_fc"]) > log2_fc_threshold),
            "mean_difference": sum(
                abs(test_results["mean_difference"]) > mean_difference_threshold
            ),
        }

    # Create histogram figures for selected parameters
    histograms = []
    element_type = "gene" if for_genes else "reaction"

    # Helper to get histogram data from registered distributions
    def get_hist_data(metric_name):
        # Use the first element ID to get the distribution details
        # We only care about the 'histogram' part
        first_id = test_results.index[0] if not test_results.empty else "dummy"
        details = color_manager.get_distribution_details(
            mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
            element_type=element_type,
            element_id=first_id,
            metric_name=metric_name,
        )
        return details["histogram"]

    if "pval" in parameters_to_plot:
        log_threshold = None
        if show_thresholds:
            count = threshold_counts["pval"]
            percent = (count / total_elements) * 100
            # Calculate threshold in -log10 space
            log_threshold = -np.log10(pval_threshold) if pval_threshold > 0 else 0

        # Use registered neg_log_adjusted_pval distribution
        hist_data = get_hist_data("neg_log_adjusted_pval")

        hist_p = create_bokeh_histogram(
            histogram_data=hist_data,
            title="Distribution of -log10(Adjusted P-values)",
            x_label="-log10(Adjusted P-value)",
            width=plot_width,
            height=plot_height,
            threshold=log_threshold,
            threshold_counts=(count, percent) if show_thresholds else None,
            show_threshold_annotation=show_thresholds,
            background_fill_color=background_fill_color,
        )
        histograms.append(hist_p)

    if "cohens_d" in parameters_to_plot:
        if show_thresholds:
            count = threshold_counts["cohens_d"]
            percent = (count / total_elements) * 100

        hist_data = get_hist_data("cohens_d")

        hist_d = create_bokeh_histogram(
            histogram_data=hist_data,
            title="Distribution of Cohen's d",
            x_label="Cohen's d",
            width=plot_width,
            height=plot_height,
            threshold=cohens_d_threshold if show_thresholds else None,
            threshold_counts=(count, percent) if show_thresholds else None,
            show_threshold_annotation=show_thresholds,
            background_fill_color=background_fill_color,
        )
        histograms.append(hist_d)

    if "log2_fc" in parameters_to_plot:
        if show_thresholds:
            count = threshold_counts["log2_fc"]
            percent = (count / total_elements) * 100

        hist_data = get_hist_data("log2_fc")

        hist_fc = create_bokeh_histogram(
            histogram_data=hist_data,
            title="Distribution of Log2 Fold Change",
            x_label="Log2 FC",
            width=plot_width,
            height=plot_height,
            threshold=log2_fc_threshold if show_thresholds else None,
            threshold_counts=(count, percent) if show_thresholds else None,
            show_threshold_annotation=show_thresholds,
            background_fill_color=background_fill_color,
        )
        histograms.append(hist_fc)

    if "mean_difference" in parameters_to_plot:
        if show_thresholds:
            count = threshold_counts["mean_difference"]
            percent = (count / total_elements) * 100

        hist_data = get_hist_data("mean_difference")

        hist_mean_diff = create_bokeh_histogram(
            histogram_data=hist_data,
            title="Distribution of Mean Differences",
            x_label="Mean Difference",
            width=plot_width,
            height=plot_height,
            threshold=mean_difference_threshold if show_thresholds else None,
            threshold_counts=(count, percent) if show_thresholds else None,
            show_threshold_annotation=show_thresholds,
            background_fill_color=background_fill_color,
        )
        histograms.append(hist_mean_diff)

    histogram_layout = _build_responsive_plot_layout(histograms)

    if return_plot:
        return histogram_layout
    else:
        show(histogram_layout)
        return None


def _prepare_scatter_source(
    data_manager,
    test_results: pd.DataFrame,
    significant_elements: List[str],
    parameters_to_plot: List[str] = None,
    for_genes: bool = False,
) -> ColumnDataSource:
    """Prepare the ColumnDataSource for scatter plots."""
    element_type = "gene" if for_genes else "reaction"
    color_manager = data_manager.color_manager

    # Identify interesting elements for full tooltips
    total_elements = len(test_results)
    interesting_indices = set()

    if total_elements < 100:
        interesting_indices = set(test_results.index)
    else:
        # Calculate top 1% count (at least 5 elements)
        top_n = max(5, int(total_elements * 0.01))

        if parameters_to_plot:
            if "pval" in parameters_to_plot and "adjusted_pval" in test_results.columns:
                interesting_indices.update(
                    test_results.nsmallest(top_n, "adjusted_pval").index
                )

            if "cohens_d" in parameters_to_plot and "cohens_d" in test_results.columns:
                interesting_indices.update(
                    test_results["cohens_d"].abs().nlargest(top_n).index
                )

            if "log2_fc" in parameters_to_plot and "log2_fc" in test_results.columns:
                interesting_indices.update(
                    test_results["log2_fc"].abs().nlargest(top_n).index
                )

            if (
                "mean_difference" in parameters_to_plot
                and "mean_difference" in test_results.columns
            ):
                interesting_indices.update(
                    test_results["mean_difference"].abs().nlargest(top_n).index
                )

    # Subsample non-interesting indices if there are too many while preserving
    # the original result ordering for deterministic rendering.
    all_indices = list(test_results.index)
    non_interesting_indices = [
        idx for idx in all_indices if idx not in interesting_indices
    ]

    if len(non_interesting_indices) > 200:
        # Randomly sample 200 indices
        sampled_indices = np.random.choice(
            non_interesting_indices, size=200, replace=False
        )
        indices_to_keep = [idx for idx in all_indices if idx in interesting_indices]
        indices_to_keep.extend(sampled_indices)
    else:
        indices_to_keep = all_indices

    # Filter test_results to only include kept indices
    test_results = test_results.loc[indices_to_keep]

    # Prepare data dict
    data = {
        "index": test_results.index,
        "element_id": list(test_results.index),
        "display_label": [],
        "external_url": [],
        "pval": test_results["adjusted_pval"],
        "neg_log_pval": test_results["neg_log_pval"],
        "neg_log_adjusted_pval": test_results["neg_log_adjusted_pval"],
        "cohens_d": test_results["cohens_d"],
        "log2_fc": test_results["log2_fc"],
        "mean_difference": test_results["mean_difference"],
        "alpha": [
            0.8 if idx in significant_elements else 0.3 for idx in test_results.index
        ],
        "size": [5 if idx in significant_elements else 3 for idx in test_results.index],
    }

    for idx in test_results.index:
        reference_data = get_external_reference_data(
            data_manager,
            idx,
            element_type,
        )
        data["display_label"].append(reference_data["display_label"])
        data["external_url"].append(reference_data["external_url"])

    # Generate colors for each metric
    metrics_to_color = ["cohens_d", "log2_fc", "mean_difference"]
    significant_set = set(significant_elements)

    for metric in metrics_to_color:
        if metric in test_results.columns:
            colors = color_manager.get_colors_from_distribution(
                mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
                element_type=element_type,
                metric_name=metric,
                element_ids=test_results.index,
            )
            # Convert to list to allow modification
            color_values = list(colors.values)

            # Override color for non-significant elements
            for i, idx in enumerate(test_results.index):
                if idx not in significant_set:
                    color_values[i] = theme.NOT_SIGNIFICANT

            data[f"color_{metric}"] = color_values

    # Generate hover text for all elements
    hover_texts = []
    for idx in test_results.index:
        if idx in interesting_indices:
            hover_text = data_manager.get_element_hover(
                element_id=idx,
                element_type=element_type,
                show_stats=True,
                show_plot=False,
                show_color_reference=False,
                show_influence=False,
            )
        else:
            # Simple lightweight tooltip
            row = test_results.loc[idx]
            pval_val = format_statistical_value(row.get("adjusted_pval", float("nan")))
            fc_val = format_statistical_value(row.get("log2_fc", float("nan")))
            d_val = format_statistical_value(row.get("cohens_d", float("nan")))
            mean_diff_val = format_statistical_value(
                row.get("mean_difference", float("nan"))
            )

            hover_text = f"""
            <div style="font-size: 12px; padding: 5px;">
                <b>{idx}</b><br>
                <span style="color: #666;">Adj. P-val:</span> {pval_val}<br>
                <span style="color: #666;">Log2 FC:</span> {fc_val}<br>
                <span style="color: #666;">Cohen's d:</span> {d_val}<br>
                <span style="color: #666;">Mean Diff:</span> {mean_diff_val}
            </div>
            """

        hover_texts.append(hover_text)
    data["desc"] = hover_texts

    # Add extra columns if available
    if "subsystems" in test_results.columns:
        data["subsystems"] = test_results["subsystems"]

    return ColumnDataSource(data)


def plot_statistical_scatterplots(
    data_manager,
    parameters_to_plot: list = None,
    return_plot: bool = False,
    show_thresholds: bool = True,
    for_genes: bool = False,
    plot_width: int = 500,
    plot_height: int = 400,
    background_fill_color="#ffffff",
    enabled_toggle=None,
    click_filename_prefix: str = "statistical_element",
    *,
    download_flag: Optional[str] = None,
    **kwargs,
) -> Optional[Any]:
    """Create a row of scatter plots for statistical parameters.

    Args:
        data_manager: The AnalysisWorkspace object containing dm and group_comparison
        parameters_to_plot: List of parameters to plot ('pval', 'cohens_d', 'log2_fc', 'mean_difference')
        return_plot: Whether to return the plot (True) or show it (False)
        show_thresholds: Whether to show threshold lines on the plots
        for_genes: Whether to plot gene data (True) or reaction data (False)
        enabled_toggle: A Bokeh ``Toggle`` enabling hover-panel downloads on tap,
            for a figure shown in a Bokeh server document.
        download_flag: The ``window`` property enabling hover-panel downloads on
            tap, for a figure embedded without a Bokeh server; see
            :func:`metabis.viz.click_behavior.add_click_behavior`.

    Returns:
        Optional[gridplot]: The row of scatter plots if return_plot is True, otherwise None
    """
    # Get necessary components from the data_manager
    group_comparison = data_manager.get_group_comparison_artifact()

    if not group_comparison:
        raise ValueError("No group comparison set in the data_manager")

    test_results = group_comparison.get_test_results(for_genes=for_genes)

    # Get thresholds from group comparison
    thresholds = group_comparison.get_thresholds(for_genes=for_genes)

    # Get significant elements (reactions or genes)
    significant_elements = group_comparison.get_significant_elements(
        for_genes=for_genes
    )

    pval_threshold = thresholds["adjusted_pval"]
    cohens_d_threshold = thresholds["cohens_d"]
    log2_fc_threshold = thresholds["log2_fc"]
    mean_difference_threshold = thresholds["mean_difference"]
    neg_log_pval_threshold = thresholds.get("neg_log_adjusted_pval")

    # If no parameters specified, use all
    if parameters_to_plot is None:
        parameters_to_plot = ["pval", "cohens_d", "log2_fc", "mean_difference"]

    # Ensure parameters are valid
    valid_params = ["pval", "cohens_d", "log2_fc", "mean_difference"]
    parameters_to_plot = [p for p in parameters_to_plot if p in valid_params]

    if not parameters_to_plot:
        raise ValueError("No valid parameters to plot specified")

    # Get total number of elements for percentage calculations
    total_elements = len(test_results)

    # Prepare data sources for scatter plots
    source = _prepare_scatter_source(
        data_manager,
        test_results,
        significant_elements,
        parameters_to_plot=parameters_to_plot,
        for_genes=for_genes,
    )

    # Create scatter plots for parameter combinations
    scatter_plots = []

    # Define plot configurations
    plot_configs = [
        {
            "params": ["pval", "cohens_d"],
            "x_col": "cohens_d",
            "y_col": "neg_log_adjusted_pval",
            "x_label": "Cohen's d",
            "y_label": "-log10(P-value)",
            "color_col": "color_cohens_d",
            "x_thresh": cohens_d_threshold,
            "y_thresh": neg_log_pval_threshold,
            "x_symmetric": True,
            "y_symmetric": False,
            "mask_func": lambda df: (
                (df["adjusted_pval"] < pval_threshold)
                & (abs(df["cohens_d"]) > cohens_d_threshold)
            ),
        },
        {
            "params": ["pval", "log2_fc"],
            "x_col": "log2_fc",
            "y_col": "neg_log_adjusted_pval",
            "x_label": "Log2 FC",
            "y_label": "-log10(P-value)",
            "color_col": "color_log2_fc",
            "x_thresh": log2_fc_threshold,
            "y_thresh": neg_log_pval_threshold,
            "x_symmetric": True,
            "y_symmetric": False,
            "mask_func": lambda df: (
                (df["adjusted_pval"] < pval_threshold)
                & (abs(df["log2_fc"]) > log2_fc_threshold)
            ),
        },
        {
            "params": ["cohens_d", "log2_fc"],
            "x_col": "cohens_d",
            "y_col": "log2_fc",
            "x_label": "Cohen's d",
            "y_label": "Log2 FC",
            "color_col": "color_cohens_d",
            "x_thresh": cohens_d_threshold,
            "y_thresh": log2_fc_threshold,
            "x_symmetric": True,
            "y_symmetric": True,
            "mask_func": lambda df: (
                (abs(df["cohens_d"]) > cohens_d_threshold)
                & (abs(df["log2_fc"]) > log2_fc_threshold)
            ),
        },
        {
            "params": ["mean_difference", "log2_fc"],
            "x_col": "mean_difference",
            "y_col": "log2_fc",
            "x_label": "Mean Difference",
            "y_label": "Log2 FC",
            "color_col": "color_mean_difference",
            "x_thresh": mean_difference_threshold,
            "y_thresh": log2_fc_threshold,
            "x_symmetric": True,
            "y_symmetric": True,
            "mask_func": lambda df: (
                (abs(df["mean_difference"]) > mean_difference_threshold)
                & (abs(df["log2_fc"]) > log2_fc_threshold)
            ),
        },
    ]

    for config in plot_configs:
        if all(p in parameters_to_plot for p in config["params"]):
            # Calculate stats
            mask = config["mask_func"](test_results)
            count = sum(mask)
            percent = (count / total_elements) * 100

            plot = _create_scatter_figure(
                dm=data_manager,
                source=source,
                x_col=config["x_col"],
                y_col=config["y_col"],
                x_label=config["x_label"],
                y_label=config["y_label"],
                color_col=config["color_col"],
                background_fill_color=background_fill_color,
                thresholds={
                    "x_thresh": config["x_thresh"] if show_thresholds else None,
                    "y_thresh": config["y_thresh"] if show_thresholds else None,
                    "x_symmetric": config["x_symmetric"],
                    "y_symmetric": config["y_symmetric"],
                },
                threshold_counts=(count, percent) if show_thresholds else None,
                width=plot_width,
                height=plot_height,
                element_type="gene" if for_genes else "reaction",
                enabled_toggle=enabled_toggle,
                filename_prefix=click_filename_prefix,
                download_flag=download_flag,
            )
            scatter_plots.append(plot)

    scatter_layout = _build_responsive_plot_layout(scatter_plots)

    if return_plot:
        return scatter_layout
    else:
        show(scatter_layout)
        return None


def create_statistical_results_table(
    data_manager,
    element_type="reaction",
    for_genes=False,
    table_height=300,
    max_rows=None,
    css_classes=None,
    stylesheets=None,
):
    """Create the on-screen data table for statistical results.

    This builds what a user *reads*: display labels, links, and rounded values.
    It does not build what a user *downloads* -- that is derived from
    `GroupComparisonArtifact.export_frame`, at full precision and with the
    calculation's own provenance.

    Args:
        data_manager: The AnalysisWorkspace object containing group_comparison
        element_type: Either "reaction" or "gene" (default: "reaction")
        for_genes: Whether to use gene data (True) or reaction data (False)
        table_height: Height of the table in pixels (default: 300)
        max_rows: Maximum number of rows to display (default: None, shows all)

    Returns:
        dict: Dictionary containing:
            - 'table': The DataTable component
            - 'source': The ColumnDataSource for the table
            - 'title': The title Div element
    """
    css_classes = list(css_classes or [])
    stylesheets = list(stylesheets or [])

    display_specs = _get_statistical_results_column_specs(element_type)

    # Get the appropriate test results
    artifact = data_manager.get_group_comparison_artifact()
    results_df = artifact.get_test_results(for_genes=for_genes)

    if results_df is None or len(results_df) == 0:
        # Return empty components if no data
        empty_fields = [spec["field"] for spec in display_specs] + ["external_url"]
        empty_source = ColumnDataSource({field: [] for field in empty_fields})
        empty_table = DataTable(
            source=empty_source,
            columns=[],
            sizing_mode="stretch_width",
            height=table_height,
            visible=False,
            css_classes=css_classes,
            stylesheets=stylesheets,
        )
        title = Div(
            text=f"<h3>{element_type.capitalize()} Test Results</h3>",
            sizing_mode="stretch_width",
            visible=False,
        )
        return {
            "table": empty_table,
            "source": empty_source,
            "title": title,
        }

    sorted_results_df = _sort_statistical_results(results_df)

    # Make a copy to avoid modifying the original data
    if max_rows is not None and len(sorted_results_df) > max_rows:
        display_df = sorted_results_df.head(max_rows).copy()
        title_text = (
            f"<h3>{element_type.capitalize()} Test Results "
            f"(Showing {max_rows} of {len(results_df)} rows)</h3>"
        )
    else:
        display_df = sorted_results_df.copy()
        title_text = f"<h3>{element_type.capitalize()} Test Results</h3>"

    display_rows = _build_statistical_table_rows(
        data_manager,
        display_df,
        for_genes=for_genes,
    )

    display_fields = [spec["field"] for spec in display_specs] + ["external_url"]
    table_source = ColumnDataSource(_rows_to_column_data(display_rows, display_fields))

    table_columns = []
    for spec in display_specs:
        table_column_kwargs = {
            "field": spec["field"],
            "title": spec["title"],
            "width": spec["width"],
        }
        if spec["kind"] == "link":
            table_column_kwargs["formatter"] = build_external_link_formatter(
                "external_url"
            )
        elif spec["kind"] == "numeric":
            table_column_kwargs["formatter"] = build_statistical_number_formatter()

        table_columns.append(TableColumn(**table_column_kwargs))

    # Create the data table
    results_table = DataTable(
        source=table_source,
        columns=table_columns,
        sizing_mode="stretch_width",
        height=table_height,
        index_position=None,
        selectable=True,
        sortable=True,
        editable=False,
        css_classes=css_classes,
        stylesheets=stylesheets,
    )

    # Create title
    title = Div(text=title_text, sizing_mode="stretch_width")

    # Return components dictionary
    return {
        "table": results_table,
        "source": table_source,
        "title": title,
    }


def _calculate_summary_stats(test_results, thresholds, parameters_to_plot):
    """Calculate summary statistics for elements meeting criteria."""
    total_elements = len(test_results)
    all_criteria_mask = np.ones(total_elements, dtype=bool)
    criteria_used = []

    pval_threshold = thresholds["adjusted_pval"]
    cohens_d_threshold = thresholds["cohens_d"]
    log2_fc_threshold = thresholds["log2_fc"]
    mean_difference_threshold = thresholds["mean_difference"]

    if "pval" in parameters_to_plot:
        all_criteria_mask &= test_results["adjusted_pval"] < pval_threshold
        criteria_used.append("p-value")

    if "cohens_d" in parameters_to_plot:
        all_criteria_mask &= abs(test_results["cohens_d"]) > cohens_d_threshold
        criteria_used.append("Cohen's d")

    if "log2_fc" in parameters_to_plot:
        all_criteria_mask &= abs(test_results["log2_fc"]) > log2_fc_threshold
        criteria_used.append("log2 fold change")

    if "mean_difference" in parameters_to_plot:
        all_criteria_mask &= (
            abs(test_results["mean_difference"]) > mean_difference_threshold
        )
        criteria_used.append("mean difference")

    all_criteria_count = sum(all_criteria_mask)
    all_criteria_percent = (
        (all_criteria_count / total_elements) * 100 if total_elements > 0 else 0
    )

    return all_criteria_count, all_criteria_percent, criteria_used

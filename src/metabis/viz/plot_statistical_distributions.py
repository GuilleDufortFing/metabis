from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import (
    BooleanFilter,
    CDSView,
    ColumnDataSource,
    DataTable,
    Div,
    HoverTool,
    LabelSet,
    Legend,
    LegendItem,
    Span,
    TableColumn,
    Title,
)
from bokeh.plotting import figure, show

from ..managers.workspace_mode import AnalysisWorkspaceMode
from ..visual_config import theme
from .click_behavior import add_click_behavior, get_external_reference_data
from .color_bars import registered_color_bar
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
    # A reaction's identifier says little to a reader: its short name sits beside it.
    name = (
        [{"field": "name", "title": "Name", "width": 240, "kind": "text"}]
        if element_type == "reaction"
        else []
    )
    return [
        {"field": "display_id", "title": element_title, "width": 180, "kind": "link"},
        *name,
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
            "short_name": element_id,
            "external_url": "",
        }

    if for_genes:
        reference_data = metabolic_model.get_gene_reference_data(element_id)
    else:
        reference_data = metabolic_model.get_reaction_reference_data(element_id)

    return {
        "raw_id": reference_data.get("raw_id", element_id),
        "display_label": reference_data.get("display_label", element_id),
        "short_name": reference_data.get("short_name", reference_data.get("display_label", element_id)),
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
                "name": "" if for_genes else reference_data["short_name"],
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


# The p-value histogram either side of its threshold: it has no direction, so it
# takes no part of the red-blue gradient. Purple past the threshold, light grey
# before it, so a significant bar never reads as a not-significant grey.
P_SIGNIFICANT = theme.SIGNIFICANT_PVAL
P_OTHER_GREY = "#dddddd"  # the client's --card-grey
STRONGEST_NAMED = 6
STAT_PLOT_HEIGHT = 300
SCATTER_PLOT_HEIGHT = 340


def _metric_color(data_manager, element_type: str, metric: str):
    """A function from a value to the colour the registered distribution paints it."""
    lookup = data_manager.color_manager.color_lookup()
    info = lookup.distributions.get(
        (AnalysisWorkspaceMode.GROUP_COMPARISON, element_type, metric)
    )
    if not info:
        return lambda _value: theme.NEUTRAL
    binner, colors = info["binner"], info["bin_colors"]
    return lambda value: colors.get(int(binner.bin_for_value(float(value))), colors[0])


def _threshold_spans(plot, value, *, dimension: str, symmetric: bool) -> None:
    if value is None or value == 0 or not np.isfinite(value):
        return
    for location in ((value, -value) if symmetric else (value,)):
        plot.add_layout(
            Span(
                location=location,
                dimension=dimension,
                line_color=theme.THRESHOLD_LINE_COLOR,
                line_dash=theme.THRESHOLD_LINE_DASH,
                line_width=theme.THRESHOLD_LINE_WIDTH,
            )
        )


def _strongest(data: Dict[str, list], y_col: str, significant: List[bool], count: int) -> List[int]:
    """The rows of the most significant elements, one per base reaction."""
    ys = np.asarray(data[y_col], dtype=float)
    order = sorted(
        (i for i, flag in enumerate(significant) if flag and np.isfinite(ys[i])),
        key=lambda i: -ys[i],
    )
    chosen, seen = [], set()
    ids = list(data["element_id"])
    for index in order:
        base = str(ids[index]).removesuffix("_back")
        if base in seen:
            continue
        seen.add(base)
        chosen.append(index)
        if len(chosen) == count:
            break
    return chosen


def _name_points(plot, data, rows: List[int], x_col: str, y_col: str) -> None:
    """Name a few points, nudged apart where their names would meet.

    Names right of the plot's middle read leftwards, so they stay inside it; a
    name near one already placed moves down a line.
    """
    if not rows:
        return
    x_values = np.asarray(data[x_col], dtype=float)
    y_values = np.asarray(data[y_col], dtype=float)
    # A reaction by its short name, a gene by its symbol.
    names = list(data.get("short_name") or data["display_label"])
    xs = x_values[np.isfinite(x_values)]
    ys = y_values[np.isfinite(y_values)]
    x_low, x_span = float(xs.min()), float(xs.max() - xs.min()) or 1.0
    y_low, y_span = float(ys.min()), float(ys.max() - ys.min()) or 1.0
    placed: List[tuple] = []
    labels = {"x": [], "y": [], "text": [], "align": [], "x_offset": [], "y_offset": []}
    for index in rows:
        x, y = float(x_values[index]), float(y_values[index])
        nx, ny = (x - x_low) / x_span, (y - y_low) / y_span
        right = nx > 0.6
        clashes = sum(1 for px, py, pright in placed if pright == right and abs(px - nx) < 0.35 and abs(py - ny) < 0.06)
        placed.append((nx, ny, right))
        labels["x"].append(x)
        labels["y"].append(y)
        labels["text"].append(str(names[index]))
        labels["align"].append("right" if right else "left")
        labels["x_offset"].append(-6 if right else 6)
        labels["y_offset"].append(-13 * clashes)
    plot.add_layout(
        LabelSet(
            x="x",
            y="y",
            text="text",
            text_align="align",
            x_offset="x_offset",
            y_offset="y_offset",
            text_baseline="middle",
            text_font_size="11px",
            text_color=theme.INK,
            background_fill_color="#ffffff",
            background_fill_alpha=0.75,
            source=ColumnDataSource(labels),
        )
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
    height: int = SCATTER_PLOT_HEIGHT,
    element_type: str = "reaction",
    enabled_toggle=None,
    filename_prefix: str = "statistical_element",
    download_flag: Optional[str] = None,
    color_metric: Optional[str] = None,
    color_label: Optional[str] = None,
):
    """Every element on two statistics: significant ones on the gradient, the rest grey.

    The non-significant points are drawn first and the significant ones over them,
    each set listed in the legend below with its count; the strongest significant
    elements are named on a volcano plot (``y_col`` an adjusted p), and a colour bar
    keys the gradient.
    """
    scatter_fig = figure(
        sizing_mode="stretch_width",
        title=f"{x_label} and {y_label}",
        x_axis_label=x_label,
        y_axis_label=y_label,
        background_fill_color=background_fill_color,
        height=height,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    significant = list(source.data["significant"])
    views = {
        flag: CDSView(filter=BooleanFilter([value is flag for value in significant]))
        for flag in (False, True)
    }
    others = scatter_fig.scatter(
        x=x_col,
        y=y_col,
        source=source,
        view=views[False],
        marker="circle",
        size=4,
        fill_color=theme.NOT_SIGNIFICANT,
        fill_alpha=0.55,
        line_color=None,
    )
    marked = scatter_fig.scatter(
        x=x_col,
        y=y_col,
        source=source,
        view=views[True],
        marker="circle",
        size=6,
        fill_color=color_col,
        fill_alpha=0.9,
        line_color=theme.MARK_EDGE,
        line_width=theme.MARK_EDGE_WIDTH,
    )
    scatter_fig.add_tools(HoverTool(renderers=[others, marked], tooltips="@desc{safe}"))

    if thresholds:
        _threshold_spans(
            scatter_fig,
            thresholds.get("x_thresh"),
            dimension="height",
            symmetric=thresholds.get("x_symmetric", False),
        )
        _threshold_spans(
            scatter_fig,
            thresholds.get("y_thresh"),
            dimension="width",
            symmetric=thresholds.get("y_symmetric", False),
        )

    n_marked = sum(1 for value in significant if value)
    coloured = f", coloured by {color_label}" if color_label else ""
    scatter_fig.add_layout(
        Legend(
            items=[
                LegendItem(label=f"Significant ({n_marked:,}){coloured}", renderers=[marked]),
                LegendItem(label=f"Not significant ({len(significant) - n_marked:,})", renderers=[others]),
            ],
            orientation="horizontal",
            location="center",
        ),
        "below",
    )
    if "adjusted_pval" in y_col:
        _name_points(
            scatter_fig,
            source.data,
            _strongest(source.data, y_col, significant, STRONGEST_NAMED),
            x_col,
            y_col,
        )
    if color_metric is not None:
        bar = registered_color_bar(
            dm.color_manager,
            mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
            element_type=element_type,
            metric_name=color_metric,
            title=color_label or color_metric,
        )
        if bar is not None:
            bar.height = 120
            scatter_fig.add_layout(bar, "right")

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


def _build_responsive_plot_layout(plots: List[Any], plots_per_row: int = 2):
    """Two by two, in rows that wrap on narrow screens; every plot keeps its toolbar."""
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


def _significance_histogram(
    values: pd.Series,
    significant: np.ndarray,
    *,
    title: str,
    x_label: str,
    color_for,
    threshold: Optional[float],
    counted: Optional[int],
    total: int,
    background_fill_color: str = "#ffffff",
    element_label: str = "reactions",
    bins: int = 40,
):
    """One metric's distribution, its significant elements on the gradient beneath the rest.

    Bins cover the 0.5th to the 99.5th percentile, so one extreme value cannot
    flatten every bar; the few values beyond fall in the edge bins. With
    ``color_for`` None (a p-value, which has no direction) the bars are light
    grey, and purple past the threshold.
    """
    numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(numbers)
    plot = figure(
        sizing_mode="stretch_width",
        height=STAT_PLOT_HEIGHT,
        title=title,
        x_axis_label=x_label,
        y_axis_label=f"Number of {element_label}",
        background_fill_color=background_fill_color,
        tools="xpan,xwheel_zoom,reset,save",
    )
    if not finite.any():
        return plot
    low, high = np.nanpercentile(numbers[finite], [0.5, 99.5])
    if color_for is None:
        low = min(low, 0.0)
    if high <= low:
        low, high = low - 0.5, high + 0.5
    edges = np.linspace(low, high, bins + 1)
    clipped = np.clip(numbers[finite], low, high)
    flags = np.asarray(significant, dtype=bool)[finite]
    marked, _ = np.histogram(clipped[flags], edges)
    others, _ = np.histogram(clipped[~flags], edges)
    centres = (edges[:-1] + edges[1:]) / 2
    left, right = edges[:-1], edges[1:]
    data = {
        "left": left,
        "right": right,
        "marked": marked,
        "total": marked + others,
        "others": others,
    }
    if color_for is None:
        data["fill"] = [
            P_SIGNIFICANT if threshold is not None and centre >= threshold else P_OTHER_GREY
            for centre in centres
        ]
        source = ColumnDataSource(data)
        bars = [
            plot.quad(left="left", right="right", bottom=0, top="total", fill_color="fill", line_color=theme.MARK_EDGE, line_width=theme.MARK_EDGE_WIDTH, source=source)
        ]
    else:
        data["fill"] = [color_for(centre) for centre in centres]
        source = ColumnDataSource(data)
        bars = [
            plot.quad(left="left", right="right", bottom=0, top="marked", fill_color="fill", line_color=theme.MARK_EDGE, line_width=theme.MARK_EDGE_WIDTH, source=source),
            plot.quad(
                left="left",
                right="right",
                bottom="marked",
                top="total",
                fill_color=theme.NOT_SIGNIFICANT,
                fill_alpha=0.6,
                line_color=theme.MARK_EDGE,
                line_width=theme.MARK_EDGE_WIDTH,
                source=source,
            ),
        ]
    plot.add_tools(
        HoverTool(
            renderers=bars,
            tooltips=[
                (x_label, "@left{0.000} to @right{0.000}"),
                ("Significant", "@marked"),
                ("Not significant", "@others"),
            ],
        )
    )
    plot.y_range.start = 0
    _threshold_spans(plot, threshold, dimension="height", symmetric=color_for is not None)
    if threshold and counted is not None and total:
        # Under the title, clear of the bars: how many pass this threshold alone.
        past = f"{x_label} above {threshold:.2f}" if color_for is None else f"|{x_label}| above {threshold:.2f}"
        plot.add_layout(
            Title(
                text=f"{counted:,} of {total:,} {element_label} ({counted / total:.0%}) with {past}",
                text_font_size="12px",
                text_font_style="normal",
                text_color=theme.INK_MUTED,
            ),
            "above",
        )
    return plot


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

    element_type = "gene" if for_genes else "reaction"
    significant = test_results.index.isin(
        list(group_comparison.get_significant_elements(for_genes=for_genes))
    )
    specs = {
        "pval": ("neg_log_adjusted_pval", "-log10 adjusted p", "pval"),
        "cohens_d": ("cohens_d", "Cohen's d", "cohens_d"),
        "log2_fc": ("log2_fc", "log2 fold change", "log2_fc"),
        "mean_difference": ("mean_difference", "Mean difference", "mean_difference"),
    }
    thresholds_by_metric = {}
    if show_thresholds:
        thresholds_by_metric = {
            "pval": -np.log10(pval_threshold) if pval_threshold > 0 else None,
            "cohens_d": cohens_d_threshold,
            "log2_fc": log2_fc_threshold,
            "mean_difference": mean_difference_threshold,
        }
    histograms = []
    for parameter in parameters_to_plot:
        column_name, label, metric = specs[parameter]
        histograms.append(
            _significance_histogram(
                test_results[column_name],
                significant,
                title=label,
                x_label=label,
                color_for=None
                if parameter == "pval"
                else _metric_color(data_manager, element_type, metric),
                threshold=thresholds_by_metric.get(parameter),
                counted=threshold_counts.get(parameter) if show_thresholds else None,
                total=total_elements,
                background_fill_color=background_fill_color,
                element_label="genes" if for_genes else "reactions",
            )
        )

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

    # Every element is drawn, the same on every render: the non-significant first,
    # so the significant ones sit on top of them.
    significant_set = set(significant_elements)
    all_indices = list(test_results.index)
    ordered = [idx for idx in all_indices if idx not in significant_set] + [
        idx for idx in all_indices if idx in significant_set
    ]
    test_results = test_results.loc[ordered]

    # Prepare data dict
    data = {
        "index": test_results.index,
        "element_id": list(test_results.index),
        "display_label": [],
        "short_name": [],
        "external_url": [],
        "pval": test_results["adjusted_pval"],
        "neg_log_pval": test_results["neg_log_pval"],
        "neg_log_adjusted_pval": test_results["neg_log_adjusted_pval"],
        "cohens_d": test_results["cohens_d"],
        "log2_fc": test_results["log2_fc"],
        "mean_difference": test_results["mean_difference"],
        "significant": [idx in significant_set for idx in test_results.index],
    }

    for idx in test_results.index:
        reference_data = get_external_reference_data(
            data_manager,
            idx,
            element_type,
        )
        data["display_label"].append(reference_data["display_label"])
        data["short_name"].append(reference_data.get("short_name", reference_data["display_label"]))
        data["external_url"].append(reference_data["external_url"])

    # Generate colors for each metric
    metrics_to_color = ["cohens_d", "log2_fc", "mean_difference"]

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
            "y_label": "-log10 adjusted p",
            "color_col": "color_cohens_d",
            "color_metric": "cohens_d",
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
            "x_label": "log2 fold change",
            "y_label": "-log10 adjusted p",
            "color_col": "color_log2_fc",
            "color_metric": "log2_fc",
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
            "y_label": "log2 fold change",
            "color_col": "color_cohens_d",
            "color_metric": "cohens_d",
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
            "x_label": "Mean difference",
            "y_label": "log2 fold change",
            "color_col": "color_mean_difference",
            "color_metric": "mean_difference",
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
                color_metric=config["color_metric"],
                color_label=config["x_label"],
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

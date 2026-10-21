from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    Range1d,
)
from bokeh.plotting import show

from ..analysis.activity_analysis import (
    PLOT_SATURATION_QUANTILE,
    ActivityOverlay,
    displayed_feature_id,
)
from ..managers.workspace_mode import AnalysisWorkspaceMode
from ..visual_config import theme
from .click_behavior import add_click_behavior, get_external_reference_data
from .color_reference import (
    GradientColorReference,
    build_reference_from_distribution,
)
from .ranking_layout import (
    RANKING_DOTS,
    RankedSubsystem,
    RankingRows,
    add_focus_link,
    order_rows,
    ranking_figure,
    ranking_layout,
    spread_offset,
    summary_panel,
)


def _attach_color_reference(
    all_data: Dict[str, List[Any]],
    reference: Optional[GradientColorReference],
) -> str:
    """Attach compact point markers and return one shared hover template.

    ``all_data["x"]`` still holds each element's true value at call time; the
    marker must show where the element sits in the distribution, not where the
    axis clamp later drew it.
    """
    if reference is None:
        return "@desc{safe}"
    values = [
        float(value) if value is not None and np.isfinite(value) else 0.0
        for value in all_data["x"]
    ]
    all_data["color_reference_value"] = values
    all_data["color_reference_marker_x"] = [
        reference.marker_x(value) for value in values
    ]
    return "@desc{safe}" + reference.hover_template(
        marker_x_field="color_reference_marker_x",
        value_field="color_reference_value",
    )


def _ranked_subsystems(
    data_manager, subsystem_stats: pd.DataFrame, score_column: str
) -> List[RankedSubsystem]:
    """One entry per subsystem with a score, named and placed in its category."""
    model = data_manager.metabolic_model
    entries = []
    for _, row in subsystem_stats.iterrows():
        score = row.get(score_column, None)
        if score is None or pd.isna(score):
            continue
        subsystem = row["subsystem"]
        entries.append(
            RankedSubsystem(
                subsystem_id=str(subsystem),
                name=str(model.get_subsystem_name(subsystem) or subsystem),
                category=str(model.get_subsystem_category(subsystem) or "Unknown"),
                score=float(score),
            )
        )
    return entries


def _metric_label(metric: str) -> str:
    return {"cohens_d": "Cohen's d", "log2_fc": "log2 fold change"}.get(
        metric, metric.replace("_", " ")
    )


def _padded_range(values: List[float], *, include_zero: bool = True) -> Range1d:
    low, high = min(values), max(values)
    if include_zero:
        low, high = min(low, 0.0), max(high, 0.0)
    padding = (high - low) * 0.06 if high != low else 1.0
    return Range1d(low - padding, high + padding)


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
    *,
    download_flag: str | None = None,
    navigation_event: str | None = None,
):
    """Each subsystem's reactions (or genes) on the metric's axis, beside its delta score.

    One row per subsystem, grouped by category in the network's order (see
    :mod:`.ranking_layout`). The left panel names the row and marks the
    subsystem's delta score on its own scale; the right panel spreads the row's
    elements within it, significant ones on the gradient and the rest grey.

    Args:
        data_manager: The AnalysisWorkspace object.
        subsystem_stats: DataFrame containing ranked subsystem statistics.
        metric: The metric to plot (e.g., "cohens_d", "log2_fc").
        max_points_per_subsystem: Maximum number of points to plot per subsystem.
        for_genes: Whether to plot gene data (True) or reaction data (False).
        return_plot: Whether to return the layout or show it.
        title: Optional title for the plot.
        download_flag: The ``window`` property enabling hover-panel downloads on
            tap, for a figure embedded without a Bokeh server; see
            :func:`metabis.viz.click_behavior.add_click_behavior`.
        navigation_event: A ``window`` event a tapped row dispatches with its
            subsystem, for a page embedding the figure without a server.

    Returns:
        The two panels as one grid if return_plot is True, else None.
    """
    if subsystem_stats is None or subsystem_stats.empty:
        print("No subsystem statistics available.")
        return None

    artifact = data_manager.get_group_comparison_artifact()
    test_results = artifact.get_test_results(for_genes=for_genes)
    if test_results is None or metric not in test_results.columns:
        print(f"Metric '{metric}' not found in test results.")
        return None

    entries = _ranked_subsystems(data_manager, subsystem_stats, "delta_score")
    if not entries:
        print("No subsystems found after category ordering.")
        return None
    rows = order_rows(entries)

    significant_set = artifact.get_significant_elements(for_genes=for_genes)
    element_type = "gene" if for_genes else "reaction"
    direction_map = (
        data_manager.subsystem_analysis.get_reaction_direction_map()
        if getattr(data_manager, "subsystem_analysis", None) is not None
        else None
    )

    all_data = _empty_points()
    for entry in entries:
        subsystem = entry.subsystem_id
        if for_genes:
            elements = data_manager.metabolic_model.get_genes_for_subsystem(subsystem)
        else:
            base_elements = (
                data_manager.metabolic_model.get_reactions_for_subsystem(subsystem) or []
            )
            # Use the exact same one-column-per-base map as Delta aggregation.
            # A compatibility caller without an installed Delta Artifact receives
            # base columns only; it never regains the old forward+_back duplication.
            elements = (
                [
                    source
                    for base_id in base_elements
                    if (source := direction_map.source_column_for(base_id)) is not None
                ]
                if direction_map is not None
                else list(base_elements)
            )
        valid_elements = [e for e in elements if e in test_results.index]
        if not valid_elements:
            continue
        prepared = _prepare_subsystem_scatter_data_horizontal(
            data_manager=data_manager,
            subsystem_results=test_results.loc[valid_elements],
            subsystem_name=subsystem,
            category_name=entry.category,
            factor=rows.factor_of[subsystem],
            metric=metric,
            max_points=max_points_per_subsystem,
            for_genes=for_genes,
            significant_set=significant_set,
            element_type=element_type,
        )
        if prepared:
            for key in all_data:
                all_data[key].extend(prepared[key])

    if not all_data["x"]:
        print("No data points found for the selected subsystems.")
        return None

    label = _metric_label(metric)
    plot = ranking_figure(
        rows,
        title=title or f"{label} per {element_type}",
        x_range=_padded_range(all_data["x"]),
        x_label=label,
        names=False,
        width=width,
        sizing_mode=sizing_mode,
    )
    plot.background_fill_color = background_fill_color
    add_focus_link(plot, rows, navigation_event=navigation_event)
    dots = _draw_points(plot, all_data)
    plot.add_tools(HoverTool(renderers=[dots], tooltips="@desc{safe}"))

    scores = {entry.subsystem_id: entry.score for entry in entries}
    reach = max([1.0, *(abs(value) for value in scores.values())])
    summary = summary_panel(
        rows,
        plot,
        scores,
        title="Delta score",
        x_range=Range1d(-reach * 1.08, reach * 1.08),
        x_label="Delta score",
    )
    layout = ranking_layout(summary, plot)
    add_click_behavior(
        layout,
        filename_prefix=click_filename_prefix,
        enabled_toggle=enabled_toggle,
        download_flag=download_flag,
    )

    if return_plot:
        return layout
    show(layout)
    return None


def _empty_points() -> Dict[str, list]:
    return {
        key: []
        for key in (
            "y", "x", "element_id", "display_label", "external_url", "subsystem",
            "category", "size", "alpha", "color", "marker", "desc",
        )
    }


def _draw_points(plot, data: Dict[str, list]):
    """The elements of every row, in front of the shading."""
    return plot.scatter(
        x="x",
        y="y",
        source=ColumnDataSource(data),
        marker="marker",
        size="size",
        fill_color="color",
        fill_alpha="alpha",
        line_color="#ffffff",
        line_width=0.5,
        name=RANKING_DOTS,
    )


def _saturating_axis_bounds(
    values: List[float], quantile: float = PLOT_SATURATION_QUANTILE
) -> Tuple[float, float]:
    """Axis bounds covering the bulk of the data, always including zero.

    Zero is the reference activity, so it must stay on the axis even when every
    plotted value sits to one side of it.
    """
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return -1.0, 1.0

    tail = (1.0 - quantile) / 2.0
    lower = float(np.nanquantile(array, tail))
    upper = float(np.nanquantile(array, 1.0 - tail))

    lower = min(lower, 0.0)
    upper = max(upper, 0.0)

    if upper <= lower:
        # Degenerate (every value identical): fall back to a unit window so the
        # points are still drawn instead of collapsing to a zero-width axis.
        span = abs(upper) or 1.0
        return lower - span, upper + span
    return lower, upper


def plot_subsystem_activity_ranking(
    data_manager,
    activity_artifact,
    overlay: ActivityOverlay | str | None = None,
    max_points_per_subsystem: int = 50,
    width: int = None,
    height: int = None,
    return_plot: bool = False,
    title: str = None,
    background_fill_color: str = "#ffffff",
    sizing_mode: str = "stretch_width",
    enabled_toggle=None,
    click_filename_prefix: str = "subsystem_element",
    *,
    download_flag: str | None = None,
    navigation_event: str | None = None,
):
    """Rank subsystems by single-group activity, grouped by metabolic category.

    The comparison-mode ranking (:func:`plot_subsystem_comparison`) is built on a
    delta score over *significant* elements, which does not exist here: nothing is
    being compared and nothing is significant. This plot ranks subsystems by their
    activity score (the median of measured member scores) and plots every measured
    member's relative activity score, with coverage reported in the hover. The
    subsystem's score is on the same scale as its members, so its diamond sits
    among them.

    Args:
        data_manager: The AnalysisWorkspace object.
        activity_artifact: The installed ActivityArtifact.
        max_points_per_subsystem: Cap on plotted member points per subsystem.
        return_plot: Whether to return the figure or show it.
        download_flag: The ``window`` property enabling hover-panel downloads on
            tap, for a figure embedded without a Bokeh server.
        navigation_event: A ``window`` event a tapped row dispatches with its
            subsystem, for a page embedding the figure without a server.

    Returns:
        Bokeh figure if return_plot is True, else None.
    """
    family = ActivityOverlay(
        overlay
        if overlay is not None
        else getattr(data_manager, "activity_overlay", ActivityOverlay.REACTION)
    )
    subsystem_stats = activity_artifact.subsystems_for(family)
    if subsystem_stats is None or subsystem_stats.empty:
        print("No subsystem activity available.")
        return None

    feature_result = activity_artifact.feature_result_for(family)
    scores = feature_result.activity_score
    overlay_name = family.value
    is_reaction_overlay = family is ActivityOverlay.REACTION

    entries = _ranked_subsystems(data_manager, subsystem_stats, "activity_score")
    if not entries:
        print("No subsystems contained measured features.")
        return None
    rows = order_rows(entries)

    all_data = _empty_points()
    all_metric_values: List[float] = []
    activity_metric = activity_artifact.metric_key_for(family)

    for entry in entries:
        subsystem = entry.subsystem_id
        if is_reaction_overlay:
            members = data_manager.metabolic_model.get_reactions_for_subsystem(subsystem)
            feature_ids = [
                displayed_feature_id(feature_result.activity_score, member)
                for member in members
            ]
        else:
            feature_ids = data_manager.metabolic_model.get_genes_for_subsystem(subsystem)

        measured = [
            feature_id
            for feature_id in dict.fromkeys(feature_ids)
            if feature_id in scores.index and not pd.isna(scores.loc[feature_id])
        ]
        if not measured:
            continue

        if len(measured) > max_points_per_subsystem:
            # Keep the strongest signals deterministically rather than sampling.
            measured = (
                scores.loc[measured]
                .abs()
                .sort_values(ascending=False, kind="mergesort")
                .index[:max_points_per_subsystem]
                .tolist()
            )

        colors = data_manager.color_manager.get_colors_from_distribution(
            mode=AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS,
            element_type=overlay_name,
            metric_name=activity_metric,
            element_ids=pd.Index(measured),
        )

        factor = rows.factor_of[subsystem]
        for feature_id in measured:
            value = float(scores.loc[feature_id])
            reference = get_external_reference_data(data_manager, feature_id, overlay_name)
            all_data["y"].append((factor, spread_offset(feature_id)))
            all_data["x"].append(value)
            all_data["element_id"].append(feature_id)
            all_data["display_label"].append(reference["display_label"])
            all_data["external_url"].append(reference["external_url"])
            all_data["subsystem"].append(subsystem)
            all_data["category"].append(entry.category)
            all_data["size"].append(7)
            all_data["alpha"].append(0.85)
            all_data["color"].append(colors.loc[feature_id])
            all_data["marker"].append("circle")
            all_data["desc"].append(
                data_manager.get_element_hover(
                    element_id=feature_id,
                    element_type=overlay_name,
                    show_stats=True,
                )
            )
            all_metric_values.append(value)

    if not all_metric_values:
        print("No measured elements found for the ranked subsystems.")
        return None

    # The plot colors its dots from the activity distribution, so it carries the
    # same color reference as graph hovers. The distribution is shared by every
    # point and only the marker moves.
    # This runs before the axis clamp so a clamped point still marks its true score.
    hover_tooltip = _attach_color_reference(
        all_data,
        build_reference_from_distribution(
            getattr(data_manager, "color_manager", None),
            mode=AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS,
            element_type=overlay_name,
            element_id=all_data["element_id"][0],
            metric_name=activity_metric,
            metric_label="Activity score",
        ),
    )

    # Bound the axis by the bulk of the data, not by its most extreme point. A
    # single reaction far from the rest otherwise stretches the axis until every
    # other point collapses onto the zero line and the ranking cannot be read.
    lower, upper = _saturating_axis_bounds(all_metric_values)
    clamped = 0
    for index, value in enumerate(all_data["x"]):
        if value < lower or value > upper:
            all_data["x"][index] = min(max(value, lower), upper)
            all_data["marker"][index] = "triangle"
            clamped += 1

    padding = (upper - lower) * 0.05
    x_range = Range1d(lower - padding, upper + padding)

    if title is None:
        element_label = "reaction" if is_reaction_overlay else "gene"
        title = (
            f"Subsystem activity ranking - {activity_artifact.group_id} "
            f"({element_label} overlay)"
        )

    axis_label = "Normalized group summary"
    if clamped:
        # Say that the axis is bounded rather than quietly moving points: a clamped
        # point is drawn as a triangle at the edge and its hover still reports the
        # true score.
        axis_label += (
            f" - {clamped} point(s) beyond [{lower:.2f}, {upper:.2f}] "
            "shown as triangles at the edge"
        )

    plot = ranking_figure(
        rows,
        title=title,
        x_range=x_range,
        x_label=axis_label,
        names=True,
        width=width,
        sizing_mode=sizing_mode,
    )
    plot.background_fill_color = background_fill_color
    add_focus_link(plot, rows, navigation_event=navigation_event)
    dots = _draw_points(plot, all_data)
    plot.add_tools(HoverTool(renderers=[dots], tooltips=hover_tooltip))
    # The subsystem's own score, the median of its members, on their scale.
    plot.scatter(
        x=[min(max(entry.score, lower), upper) for entry in entries if entry.subsystem_id in rows.factor_of],
        y=[rows.factor_of[entry.subsystem_id] for entry in entries if entry.subsystem_id in rows.factor_of],
        marker="diamond",
        size=13,
        fill_color=theme.INK,
        line_color="#ffffff",
        line_width=1,
    )

    add_click_behavior(
        plot,
        filename_prefix=click_filename_prefix,
        enabled_toggle=enabled_toggle,
        download_flag=download_flag,
    )

    if return_plot:
        return plot
    show(plot)
    return None


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
    factor: Optional[str] = None,
) -> Optional[Dict[str, list]]:
    """Points for one subsystem's elements, each at a fixed place within its row.

    Past the cap, the strongest elements are kept, significant ones first: the
    same points on every render.
    """
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

    # The strongest non-significant elements fill the remaining slots.
    n_interesting = len(interesting_indices)
    n_remaining_slots = max(0, max_points - n_interesting)

    if len(non_interesting_indices) > n_remaining_slots:
        sampled_non_interesting = (
            _select_prioritized_subsystem_indices(
                subsystem_results.loc[non_interesting_indices],
                metric=metric,
                max_points=n_remaining_slots,
            )
            if n_remaining_slots > 0
            else []
        )
    else:
        sampled_non_interesting = non_interesting_indices

    indices_to_keep = interesting_indices + sampled_non_interesting

    if not indices_to_keep:
        return None

    filtered_results = subsystem_results.loc[indices_to_keep].copy()

    # Prepare data dictionary
    row = factor if factor is not None else subsystem_name
    data = {
        "y": [(row, spread_offset(idx)) for idx in filtered_results.index],
        "x": list(filtered_results[metric]),
        "element_id": list(filtered_results.index),
        "display_label": [],
        "external_url": [],
        "subsystem": [subsystem_name] * len(filtered_results),
        "category": [category_name] * len(filtered_results),
        "size": [8 if idx in significant_set else 5 for idx in filtered_results.index],
        "alpha": [
            0.95 if idx in significant_set else 0.6 for idx in filtered_results.index
        ],
        "marker": ["circle"] * len(filtered_results),
    }

    # Assign colors
    colors = data_manager.color_manager.get_colors_from_distribution(
        mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
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

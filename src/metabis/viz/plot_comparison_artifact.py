"""Bokeh plot factories over immutable group-comparison artifacts."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from bokeh.models import ColumnDataSource, HoverTool, Span
from bokeh.plotting import figure

from ..analysis.comparison_artifact import GroupComparisonArtifact
from ..visual_config import theme

ComparisonFeature = Literal["reaction", "gene"]
ComparisonMetric = Literal["pval", "cohens_d", "log2_fc", "mean_difference"]

_METRICS = {
    "pval": ("neg_log_adjusted_pval", "−log10(Adjusted p-value)"),
    "cohens_d": ("cohens_d", "Cohen's d"),
    "log2_fc": ("log2_fc", "Log2 fold change"),
    "mean_difference": ("mean_difference", "Mean difference"),
}


def _validate(
    artifact: GroupComparisonArtifact,
    feature_type: ComparisonFeature,
    metric: ComparisonMetric,
):
    if not isinstance(artifact, GroupComparisonArtifact):
        raise TypeError("artifact must be a GroupComparisonArtifact")
    if feature_type not in {"reaction", "gene"}:
        raise ValueError("feature_type must be 'reaction' or 'gene'")
    if metric not in _METRICS:
        raise ValueError(f"Unsupported comparison metric: {metric}")


def _plot_values(frame, metric: ComparisonMetric) -> np.ndarray:
    field, _label = _METRICS[metric]
    if field in frame.columns:
        return np.asarray(frame[field], dtype=float)
    if metric == "pval" and "adjusted_pval" in frame.columns:
        values = np.asarray(frame["adjusted_pval"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return -np.log10(np.maximum(values, np.finfo(float).tiny))
    raise ValueError(f"Comparison results do not contain '{field}'")


def _threshold(artifact, feature_type, metric):
    thresholds = (
        artifact.reaction_thresholds
        if feature_type == "reaction"
        else artifact.gene_thresholds
    )
    if thresholds is None:
        raise ValueError(f"This comparison has no {feature_type} thresholds")
    if metric == "pval":
        if thresholds.pval is None or not 0 < thresholds.pval < 1:
            return None
        return -math.log10(thresholds.pval)
    return getattr(thresholds, metric)


def _labels(artifact, feature_type):
    pairs = (
        artifact.reaction_display_labels
        if feature_type == "reaction"
        else artifact.gene_display_labels
    )
    return dict(pairs)


def _tag(plot, artifact, **values):
    plot.tags.append(
        {
            "artifact_id": artifact.metadata.artifact_id,
            "artifact_kind": artifact.metadata.kind.value,
            **values,
        }
    )
    return plot


def plot_comparison_histogram(
    artifact: GroupComparisonArtifact,
    feature_type: ComparisonFeature,
    metric: ComparisonMetric,
    *,
    show_threshold: bool = True,
    bins: int = 30,
    width: int = 500,
    height: int = 400,
):
    """Render one deterministic distribution from stored comparison values."""
    _validate(artifact, feature_type, metric)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    frame = artifact.results_frame(feature_type)
    values = _plot_values(frame, metric)
    finite = values[np.isfinite(values)]
    counts, edges = (
        np.histogram(finite, bins=bins)
        if len(finite)
        else (np.asarray([], dtype=int), np.asarray([], dtype=float))
    )
    source = ColumnDataSource(
        {
            "left": edges[:-1],
            "right": edges[1:],
            "top": counts,
        }
    )
    _field, label = _METRICS[metric]
    plot = figure(
        title=f"Distribution of {label}",
        x_axis_label=label,
        y_axis_label=f"{feature_type.capitalize()} count",
        width=width,
        height=height,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        active_scroll="wheel_zoom",
    )
    renderer = plot.quad(
        left="left",
        right="right",
        top="top",
        bottom=0,
        source=source,
        fill_color="#4c78a8",
        fill_alpha=0.8,
        line_color="white",
    )
    plot.select_one(HoverTool).renderers = [renderer]
    plot.select_one(HoverTool).tooltips = [
        ("Range", "@left{0.000} – @right{0.000}"),
        ("Count", "@top"),
    ]
    threshold = _threshold(artifact, feature_type, metric)
    if show_threshold and threshold is not None and threshold != 0:
        plot.add_layout(
            Span(
                location=threshold,
                dimension="height",
                line_color=theme.THRESHOLD_LINE_COLOR,
                line_dash=theme.THRESHOLD_LINE_DASH,
                line_width=theme.THRESHOLD_LINE_WIDTH,
            )
        )
        if metric != "pval":
            plot.add_layout(
                Span(
                    location=-threshold,
                    dimension="height",
                    line_color=theme.THRESHOLD_LINE_COLOR,
                    line_dash=theme.THRESHOLD_LINE_DASH,
                    line_width=theme.THRESHOLD_LINE_WIDTH,
                )
            )
    return _tag(
        plot,
        artifact,
        plot_kind="comparison_histogram",
        feature_type=feature_type,
        metric=metric,
        show_threshold=show_threshold,
    )


def plot_comparison_scatter(
    artifact: GroupComparisonArtifact,
    feature_type: ComparisonFeature,
    x_metric: ComparisonMetric,
    y_metric: ComparisonMetric,
    *,
    show_thresholds: bool = True,
    width: int = 500,
    height: int = 400,
):
    """Render one comparison scatter directly from stored artifact rows."""
    _validate(artifact, feature_type, x_metric)
    _validate(artifact, feature_type, y_metric)
    if x_metric == y_metric:
        raise ValueError("Scatter metrics must differ")
    frame = artifact.results_frame(feature_type)
    x_values = _plot_values(frame, x_metric)
    y_values = _plot_values(frame, y_metric)
    significant = set(
        artifact.significant_reactions
        if feature_type == "reaction"
        else artifact.significant_genes
    )
    labels = _labels(artifact, feature_type)
    feature_ids = tuple(map(str, frame.index))
    source = ColumnDataSource(
        {
            "feature_id": feature_ids,
            "display_label": [labels.get(item, item) for item in feature_ids],
            "x": x_values,
            "y": y_values,
            "adjusted_pval": frame.get("adjusted_pval", np.nan),
            "cohens_d": frame.get("cohens_d", np.nan),
            "log2_fc": frame.get("log2_fc", np.nan),
            "mean_difference": frame.get("mean_difference", np.nan),
            "is_significant": [item in significant for item in feature_ids],
            "color": [
                ("#d73027" if float(x) >= 0 else "#4575b4")
                if item in significant and np.isfinite(x)
                else theme.NOT_SIGNIFICANT
                for item, x in zip(feature_ids, x_values)
            ],
            "size": [8 if item in significant else 5 for item in feature_ids],
            "alpha": [0.85 if item in significant else 0.35 for item in feature_ids],
        }
    )
    _x_field, x_label = _METRICS[x_metric]
    _y_field, y_label = _METRICS[y_metric]
    plot = figure(
        title=f"{x_label} vs {y_label}",
        x_axis_label=x_label,
        y_axis_label=y_label,
        width=width,
        height=height,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        active_scroll="wheel_zoom",
    )
    renderer = plot.scatter(
        x="x",
        y="y",
        source=source,
        size="size",
        color="color",
        alpha="alpha",
        marker="circle",
        name="comparison_scatter_points",
    )
    plot.select_one(HoverTool).renderers = [renderer]
    plot.select_one(HoverTool).tooltips = [
        (feature_type.capitalize(), "@display_label"),
        ("Canonical ID", "@feature_id"),
        ("Adjusted p", "@adjusted_pval{0.000e}"),
        ("Cohen's d", "@cohens_d{0.000}"),
        ("Log2 fold change", "@log2_fc{0.000}"),
        ("Mean difference", "@mean_difference{0.000}"),
        ("Significant", "@is_significant"),
    ]
    if show_thresholds:
        for metric, dimension in ((x_metric, "height"), (y_metric, "width")):
            threshold = _threshold(artifact, feature_type, metric)
            if threshold is None or threshold == 0:
                continue
            plot.add_layout(
                Span(
                    location=threshold,
                    dimension=dimension,
                    line_color=theme.THRESHOLD_LINE_COLOR,
                    line_dash=theme.THRESHOLD_LINE_DASH,
                    line_width=theme.THRESHOLD_LINE_WIDTH,
                )
            )
            if metric != "pval":
                plot.add_layout(
                    Span(
                        location=-threshold,
                        dimension=dimension,
                        line_color=theme.THRESHOLD_LINE_COLOR,
                        line_dash=theme.THRESHOLD_LINE_DASH,
                        line_width=theme.THRESHOLD_LINE_WIDTH,
                    )
                )
    return _tag(
        plot,
        artifact,
        plot_kind="comparison_scatter",
        feature_type=feature_type,
        x_metric=x_metric,
        y_metric=y_metric,
        show_thresholds=show_thresholds,
    )


__all__ = ["plot_comparison_histogram", "plot_comparison_scatter"]

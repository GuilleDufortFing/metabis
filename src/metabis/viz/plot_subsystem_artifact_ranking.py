"""Artifact-first subsystem ranking plot.

Scientific values come only from typed artifacts.  Labels, links, colours, and
hover HTML are captured separately as an immutable presentation snapshot so a
later mutation of the live workspace cannot change an exported plot.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from bokeh.models import ColumnDataSource, FactorRange, HoverTool, Range1d, Span
from bokeh.plotting import figure

from metabis.analysis.activity_analysis import ActivityArtifact, ActivityOverlay
from metabis.analysis.artifacts import PlainTable
from metabis.analysis.comparison_artifact import GroupComparisonArtifact
from metabis.visual_config import theme
from metabis.viz.plot_utils import apply_academic_style


@dataclass(frozen=True)
class RankingFeaturePresentation:
    feature_id: str
    subsystem_id: str
    category: str
    display_label: str
    external_url: str
    color: str
    hover_html: str


@dataclass(frozen=True)
class SubsystemRankingPresentation:
    """Frozen display-only values for one artifact-backed ranking."""

    features: tuple[RankingFeaturePresentation, ...]


def plot_subsystem_ranking_artifact(
    *,
    subsystem_table: PlainTable,
    presentation: SubsystemRankingPresentation,
    metric: str,
    activity_artifact: ActivityArtifact | None = None,
    activity_overlay: ActivityOverlay | str = ActivityOverlay.REACTION,
    comparison_artifact: GroupComparisonArtifact | None = None,
    width: int | None = None,
    height: int | None = None,
    sizing_mode: str | None = "stretch_width",
    title: str | None = None,
):
    """Build a fresh ranking plot without consulting live analysis state."""
    if (activity_artifact is None) == (comparison_artifact is None):
        raise ValueError("Provide exactly one activity or comparison artifact")
    stats = subsystem_table.to_frame()
    if stats.empty or not presentation.features:
        return None

    if activity_artifact is not None:
        family = ActivityOverlay(activity_overlay)
        values = (
            activity_artifact.feature_result_for(family).activity_score
        )
        score_column = "activity_score"
        default_title = (
            f"Subsystem activity ranking - {activity_artifact.config.group_id} "
            f"({family.value} overlay)"
        )
        x_label = "Normalized group summary"
    else:
        values = comparison_artifact.results_frame("reaction")[metric]
        score_column = "delta_subsystem_score"
        default_title = f"Subsystem comparison ranking ({metric})"
        x_label = metric

    if score_column not in stats.columns:
        raise ValueError(f"Subsystem table is missing '{score_column}'")
    subsystem_scores = {
        str(row.get("subsystem", index)): float(row[score_column])
        for index, row in stats.iterrows()
        if not pd.isna(row[score_column])
    }
    ordered = sorted(
        {
            (item.category, item.subsystem_id)
            for item in presentation.features
            if item.subsystem_id in subsystem_scores
        },
        key=lambda pair: (-abs(subsystem_scores[pair[1]]), pair[0], pair[1]),
    )
    if not ordered:
        return None
    factor_set = set(ordered)
    rows = []
    for item in presentation.features:
        factor = (item.category, item.subsystem_id)
        if factor not in factor_set or item.feature_id not in values.index:
            continue
        value = values.loc[item.feature_id]
        if pd.isna(value):
            continue
        rows.append(
            {
                "x": float(value),
                "y": factor,
                "feature_id": item.feature_id,
                "display_label": item.display_label,
                "external_url": item.external_url,
                "subsystem": item.subsystem_id,
                "category": item.category,
                "color": item.color,
                "desc": item.hover_html,
            }
        )
    if not rows:
        return None
    x_values = [row["x"] for row in rows]
    lower = min(min(x_values), 0.0)
    upper = max(max(x_values), 0.0)
    if lower == upper:
        lower, upper = lower - 1.0, upper + 1.0
    padding = (upper - lower) * 0.05
    if height is None:
        height = max(500, len(ordered) * 24)
    kwargs = {
        "width": width,
        "height": height,
        "x_range": Range1d(lower - padding, upper + padding),
        "y_range": FactorRange(factors=list(reversed(ordered))),
        "title": title or default_title,
        "tools": "hover,pan,wheel_zoom,reset,save",
        "toolbar_location": "above",
        "x_axis_location": "above",
    }
    if sizing_mode is not None:
        kwargs["sizing_mode"] = sizing_mode
    plot = figure(**kwargs)
    apply_academic_style(plot)
    plot.add_layout(
        Span(
            location=0,
            dimension="height",
            line_color=theme.THRESHOLD_LINE_COLOR,
            line_dash=theme.THRESHOLD_LINE_DASH,
            line_width=theme.THRESHOLD_LINE_WIDTH,
        )
    )
    source = ColumnDataSource(pd.DataFrame(rows))
    plot.scatter(
        x="x",
        y="y",
        source=source,
        size=7,
        color="color",
        alpha=0.85,
        line_color="white",
        line_width=0.5,
    )
    hover = plot.select_one(HoverTool)
    hover.tooltips = "@desc{safe}"
    plot.xaxis.axis_label = x_label
    plot.yaxis.visible = False
    plot.ygrid.grid_line_color = None
    return plot


__all__ = [
    "RankingFeaturePresentation",
    "SubsystemRankingPresentation",
    "plot_subsystem_ranking_artifact",
]

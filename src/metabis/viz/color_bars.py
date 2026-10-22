"""Colour bars read from the registered distribution that coloured a figure.

Every figure coloured by a value carries a colour key. The key is sampled from the
same registered distribution, bins and palette that painted the glyphs, so it cannot
drift from them: the standardized ramp of Supplementary Section 1.5, the value over
its global standard deviation, saturating at the configured number of deviations.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np
from bokeh.models import ColorBar, Legend, LegendItem, LinearColorMapper

from ..managers.workspace_mode import AnalysisWorkspaceMode

COLOR_BAR_SAMPLES = 121
COLOR_BAR_WIDTH = 10
COLOR_BAR_LENGTH = 160


def _lookup(registry: Any) -> Any:
    """The read-only lookup behind a colour manager or a detached registry."""
    for name in ("color_lookup", "lookup"):
        method = getattr(registry, name, None)
        if callable(method):
            return method()
    return registry


def registered_color_bar(
    registry: Any,
    *,
    mode: AnalysisWorkspaceMode,
    element_type: str,
    metric_name: str,
    title: str,
    bounds: Optional[Tuple[float, float]] = None,
    samples: int = COLOR_BAR_SAMPLES,
) -> Optional[ColorBar]:
    """A vertical colour bar for one registered distribution, or ``None``.

    Args:
        registry: The colour manager or registry the distribution lives in.
        bounds: Values the bar never runs past, such as -1 and +1 for a delta
            score: the ramp's saturation can lie beyond what the score can reach.
    """
    if registry is None:
        return None
    distributions = getattr(_lookup(registry), "distributions", None) or {}
    info = distributions.get((mode, element_type, metric_name))
    if not info:
        return None
    binner = info["binner"]
    colors = info["bin_colors"]
    low = float(binner.center - binner.lower_color_limit)
    high = float(binner.center + binner.upper_color_limit)
    if bounds is not None:
        low, high = max(low, bounds[0]), min(high, bounds[1])
    if not (np.isfinite(low) and np.isfinite(high)) or high <= low:
        return None
    values = np.linspace(low, high, samples)
    palette = [colors.get(int(binner.bin_for_value(value)), colors[0]) for value in values]
    return color_bar(palette, low, high, title)


def color_bar(palette: Sequence[str], low: float, high: float, title: str) -> ColorBar:
    """A vertical colour bar over ``palette``, from ``low`` to ``high``."""
    return ColorBar(
        color_mapper=LinearColorMapper(palette=list(palette), low=low, high=high),
        title=title,
        width=COLOR_BAR_WIDTH,
        height=COLOR_BAR_LENGTH,
        label_standoff=6,
        margin=12,
        padding=4,
    )


def overview_color_bar(
    data_manager: Any,
    *,
    visualization_mode: str,
    for_genes: bool = False,
    delta_options: Optional[dict] = None,
    activity_artifact: Any = None,
    activity_overlay: Any = None,
    set_enrichment_artifact: Any = None,
) -> Optional[ColorBar]:
    """The colour key of the metabolic network, for its colouring summary.

    Delta scores read the subsystem distribution the comparison registered, kept
    within -1 and +1 under the sign transform, which cannot pass them.
    Over-representation reads the signed log2 enrichment ratio's scale; single-group
    activity reads the group's subsystem activity.
    """
    registry = getattr(data_manager, "color_manager", None)
    if visualization_mode == "activity":
        if activity_artifact is None or activity_overlay is None:
            return None
        return registered_color_bar(
            registry,
            mode=AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS,
            element_type="subsystem",
            metric_name=activity_artifact.metric_key_for(activity_overlay),
            title="Activity",
        )
    if set_enrichment_artifact is not None:
        # Imported on use: the worker imports set_enrichment_overview without Bokeh.
        from .set_enrichment_overview import (
            SET_ENRICHMENT_SUBSYSTEM_METRIC,
            build_set_enrichment_overview_colors,
        )
        from ..managers.color_distributions import ColorDistributionRegistry
        from ..visual_config import theme

        scale = ColorDistributionRegistry(
            configurations={
                AnalysisWorkspaceMode.GROUP_COMPARISON: {
                    "up_color": theme.UP_REGULATED,
                    "down_color": theme.DOWN_REGULATED,
                    "neutral_color": theme.NEUTRAL,
                    "significant_pval_color": theme.SIGNIFICANT_PVAL,
                }
            },
            current_mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
        )
        build_set_enrichment_overview_colors(set_enrichment_artifact, scale)
        return registered_color_bar(
            scale,
            mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
            element_type="subsystem",
            metric_name=SET_ENRICHMENT_SUBSYSTEM_METRIC,
            title="Signed log2 ER",
        )
    options = dict(delta_options or {})
    transform = str(options.get("transform", "sign")).lower()
    return registered_color_bar(
        registry,
        mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
        element_type="subsystem_genes" if for_genes else "subsystem",
        metric_name=options.get("metric", "default"),
        title="Delta score",
        bounds=(-1.0, 1.0) if transform == "sign" else None,
    )


METRIC_LABELS = {
    "cohens_d": "Cohen's d",
    "log2_fc": "log2 fold change",
    "mean_difference": "mean difference",
}


def add_pathway_key(
    plot,
    data_manager: Any,
    *,
    visualization_mode: str,
    color_metric: str = "cohens_d",
    activity_artifact: Any = None,
) -> None:
    """A key row under a subsystem graph: what each shape is, and what each colour means.

    A diamond is a reaction, a rectangle a gene, an oval a metabolite; grey is a
    feature that is not significant or has no data. Each feature family that
    colours the graph adds its colour bar, read from its own distribution.
    """
    from ..visual_config import theme

    outline = dict(fill_color="#ffffff", line_color=theme.INK_MUTED, line_width=1)
    # A legend draws a glyph from its renderer's first row, so each key glyph has
    # one, at a place the plot never shows.
    nowhere = dict(x=[float("nan")], y=[float("nan")])
    shapes = [
        ("Reaction", plot.scatter(**nowhere, marker="diamond", size=13, **outline)),
        (
            "Gene",
            plot.rect(**nowhere, width=18, height=10, width_units="screen", height_units="screen", **outline),
        ),
        (
            "Metabolite",
            plot.ellipse(**nowhere, width=18, height=11, width_units="screen", height_units="screen", **outline),
        ),
        (
            "Not significant, or no data",
            plot.scatter(**nowhere, marker="square", size=11, fill_color=theme.NOT_SIGNIFICANT, line_color=None),
        ),
    ]
    legend = Legend(
        items=[LegendItem(label=label, renderers=[renderer]) for label, renderer in shapes],
        orientation="horizontal",
        location="center",
        click_policy="none",
        # A rectangle's key fills its glyph box: a low box keeps it a gene's shape.
        glyph_width=18,
        glyph_height=11,
    )
    plot.add_layout(legend, "below")

    registry = getattr(data_manager, "color_manager", None)
    if visualization_mode == "activity":
        if activity_artifact is None:
            return
        from ..analysis.activity_analysis import ActivityOverlay

        families = [
            (
                AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS,
                overlay.value,
                activity_artifact.metric_key_for(overlay),
                f"{label} activity",
            )
            for overlay, label in ((ActivityOverlay.REACTION, "Reaction"), (ActivityOverlay.GENE, "Gene"))
        ]
    else:
        metric = METRIC_LABELS.get(color_metric, color_metric)
        families = [
            (AnalysisWorkspaceMode.GROUP_COMPARISON, "reaction", color_metric, f"Reactions, {metric}"),
            (AnalysisWorkspaceMode.GROUP_COMPARISON, "gene", color_metric, f"Genes, {metric}"),
        ]
    for mode, element_type, metric_name, title in families:
        try:
            bar = registered_color_bar(
                registry, mode=mode, element_type=element_type, metric_name=metric_name, title=title
            )
        except Exception:
            # A family with nothing registered draws no bar; the graph is unaffected.
            bar = None
        if bar is None:
            continue
        # Beside the graph, which is wider than tall: below, the bars would take
        # its height.
        bar.height = 140
        plot.add_layout(bar, "right")

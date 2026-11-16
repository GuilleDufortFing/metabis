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
# The tag of the colour bars a page sets side by side in one row under a figure.
# BokehJS gives each item under a plot a row of its own; the React client's
# patchKeyRow groups the tagged ones that follow each other into one.
KEY_ROW = "metabis-key-row"
# Pixels a pathway key's rows of shapes and fills, and its row of colour bars,
# take under a figure, as BokehJS 3.8 lays them out.
KEY_SHAPES_HEIGHT = 45
KEY_BARS_HEIGHT = 65


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
        from ..managers.color_distributions import ColorDistributionRegistry
        from ..visual_config import theme
        from .set_enrichment_overview import (
            SET_ENRICHMENT_SUBSYSTEM_METRIC,
            build_set_enrichment_overview_colors,
        )

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
    """The key under a subsystem graph or a map: what each shape is, and what each colour means.

    A diamond is a reaction, a rectangle a gene, an oval a metabolite. A feature
    with no data (not measured) is white with a dashed edge; grey is a measured
    feature that is not significant in a comparison, and the typical level in
    single-group activity, whose colour bar shows it. Each feature family that colours the graph adds its colour bar,
    read from its own distribution: a short horizontal bar under the shapes, the
    bars side by side where the page sets them so (:data:`KEY_ROW`), and in a row
    each otherwise. A figure scaled to its width grows by its key, which takes
    none of its drawing's height.
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
    ]
    fills = []
    if visualization_mode != "activity":
        fills.append(
            (
                "Not significant",
                plot.scatter(**nowhere, marker="square", size=11, fill_color=theme.NOT_SIGNIFICANT, line_color=None),
            )
        )
    fills.append(("No data", plot.scatter(**nowhere, marker="square", size=11, **outline, line_dash="dashed")))
    # The key reads from the left, as the figure's title and caption do: the shapes
    # in one row and the fills that mean no colour in the next, so a graph narrowed
    # by the details beside it still shows the whole key.
    for row, margin in ((shapes, 10), (fills, 0)):
        legend = Legend(
            items=[LegendItem(label=label, renderers=[renderer]) for label, renderer in row],
            orientation="horizontal",
            location="left",
            click_policy="none",
            # A rectangle's key fills its glyph box: a low box keeps it a gene's shape.
            glyph_width=18,
            glyph_height=11,
            margin=margin,
            padding=6,
        )
        plot.add_layout(legend, "below")

    registry = getattr(data_manager, "color_manager", None)
    if visualization_mode == "activity" and activity_artifact is None:
        families = []
    elif visualization_mode == "activity":
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
    bars = 0
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
        # Under the graph: beside it, a column as tall as the graph held two short
        # bars, and took a map's width. In a row under a plot BokehJS spaces a bar
        # by its padding, not its margin.
        bar.update(
            orientation="horizontal",
            width=COLOR_BAR_LENGTH,
            height=COLOR_BAR_WIDTH,
            location="left",
            padding=10,
            tags=[*bar.tags, KEY_ROW],
        )
        plot.add_layout(bar, "below")
        bars += 1
    _grow_by_key(plot, KEY_SHAPES_HEIGHT + (KEY_BARS_HEIGHT if bars else 0))
    _credit_last(plot)


# Marks the credit lines, which stay under everything else under a graph.
ATLAS_CREDIT_TAG = "metabis-atlas-credit"
CREDIT_LINE_HEIGHT = 13  # pixels a credit line takes


def add_atlas_credit(plot) -> None:
    """Credit Metabolic Atlas's maps under a figure drawn from them, in small grey type.

    Drawn in the figure, the credit goes with an exported PNG or SVG, as the
    notice in a Human-maps SVG's corner goes with it; :func:`add_pathway_key`
    keeps it under the key.
    """
    from bokeh.models import Title

    from .atlas_paths import ATLAS_CREDIT
    from .metabolic_visualization.generation.data_structures.constants import (
        VisualizationConstants,
    )

    for line in ATLAS_CREDIT:
        plot.add_layout(
            Title(
                text=line,
                align="left",
                text_font=VisualizationConstants.FONT_FAMILY,
                text_font_size="9px",
                text_font_style="normal",
                text_color="#606060",
                standoff=1,
                tags=[ATLAS_CREDIT_TAG],
            ),
            "below",
        )
    _grow_by_key(plot, CREDIT_LINE_HEIGHT * len(ATLAS_CREDIT))


def _credit_last(plot) -> None:
    """Keep the credit under whatever else lies under the figure."""
    credit = [item for item in plot.below if ATLAS_CREDIT_TAG in (getattr(item, "tags", None) or [])]
    if credit:
        plot.below = [item for item in plot.below if item not in credit] + credit


def _grow_by_key(plot, height: int) -> None:
    """A figure scaled to its width, ``height`` pixels taller at its own width, for its key.

    Its aspect ratio is the whole figure's, so a key under it would otherwise take
    its drawing's height.
    """
    ratio = plot.aspect_ratio
    scaled = plot.sizing_mode in ("scale_width", "scale_both")
    if not scaled or not isinstance(ratio, (int, float)) or not ratio or not plot.width:
        return
    plot.aspect_ratio = plot.width / (plot.width / ratio + height)


# The network's category key, which the page shows on demand (an export's key).
CATEGORY_KEY = "metabis-category-key"
KEY_CHARACTER_WIDTH = 5.1  # pixels, about, of a character of 10 px Arial
KEY_LINE_HEIGHT = 14  # pixels a line of the key takes
KEY_LINE_LENGTH = 28  # characters a line of a category's name in the key
KEY_INDENT = " " * 7  # as wide, in Arial, as two digits and the three spaces after


def add_category_key(plot, *, visible: bool = False) -> Optional[Legend]:
    """A numbered key of the network's categories, under it, in two columns.

    The network marks each category with a number on its rim, and the rankings name
    it beside the same number; an exported network has no ranking beside it, so its
    key names the numbers. Hidden unless ``visible``: the page shows it for export.
    Its ``tags`` give the height it takes, in pixels.
    """
    import math
    import textwrap

    categories = None
    for renderer in plot.renderers:
        data = getattr(getattr(renderer, "data_source", None), "data", {}) or {}
        if "badge" in data and "display_label" in data:
            categories = data
            break
    if not categories or not categories["badge"]:
        return None
    numbered = sorted(
        (int(number), str(label))
        for number, label in zip(categories["badge"], categories["display_label"])
        if str(number).isdigit()
    )
    # A legend draws a glyph from its renderer's first row: a point the plot never
    # shows, so each row reads as its number and name alone.
    nowhere = plot.scatter(x=[float("nan")], y=[float("nan")], size=0, line_color=None)
    # In Arial two spaces are as wide as a digit, so one digit padded by two lines
    # its name up with the names after two digits. A long name wraps under itself,
    # its second line a row of its own: BokehJS measures a legend row as one line.
    half = math.ceil(len(numbered) / 2)
    columns = []
    for part in (numbered[:half], numbered[half:]):
        lines = []
        for number, label in part:
            wrapped = textwrap.wrap(label, KEY_LINE_LENGTH, break_long_words=False) or [label]
            lines.append(f"{'  ' if number < 10 else ''}{number}   {wrapped[0]}")
            lines.extend(KEY_INDENT + line for line in wrapped[1:])
        columns.append(lines)
    rows = max(len(column) for column in columns)
    labels = [line for column in columns for line in column + [" "] * (rows - len(column))]
    legend = Legend(
        items=[LegendItem(label=label, renderers=[nowhere]) for label in labels],
        orientation="vertical",
        nrows=rows,
        location="top_left",
        click_policy="none",
        glyph_width=1,
        glyph_height=KEY_LINE_HEIGHT,
        label_height=KEY_LINE_HEIGHT,
        label_standoff=0,
        label_text_font_size="10px",
        # The first column's longest line, and a gap, set the columns' width.
        label_width=int(max(len(line) for line in columns[0]) * KEY_CHARACTER_WIDTH) + 16,
        spacing=0,
        visible=visible,
        name=CATEGORY_KEY,
    )
    # The pixels the key adds under the plot, for a page that grows the plot by as much.
    legend.tags = [{"height": rows * KEY_LINE_HEIGHT + 2 * legend.padding + 2 * legend.margin}]
    plot.add_layout(legend, "below")
    return legend

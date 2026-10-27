"""Heatmaps across groups: Delta by subsystem and group, and scores by sample.

Both use the app's blue through white to red scale, as Delta and the PCA
loadings are drawn, with rows grouped and labelled on the left.
"""

from __future__ import annotations

from html import escape
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    CustomJSHover,
    FactorRange,
    FixedTicker,
    HoverTool,
    LinearColorMapper,
    Range1d,
)
from bokeh.plotting import figure
from bokeh.transform import transform

from ..analysis.landscape import Landscape
from ..managers.color_distributions import gradient_palette
from ..managers.group_palettes import group_palette
from ..managers.hover.panel import note, panel, row, section
from ..visual_config import theme
from .plot_utils import apply_academic_style

__all__ = [
    "plot_delta_landscape",
    "plot_sample_heatmap",
    "sample_heatmap_rows",
]

ROW_HEIGHT = 20
SAMPLE_ROW_HEIGHT = 14
BAND_ROW = "Group"


def _diverging_palette() -> list[str]:
    half = 128
    return gradient_palette(theme.DOWN_REGULATED, theme.NEUTRAL, half)[:-1] + gradient_palette(
        theme.NEUTRAL, theme.UP_REGULATED, half
    )


def _color_bar(mapper, title: str) -> ColorBar:
    return ColorBar(
        color_mapper=mapper,
        ticker=BasicTicker(desired_num_ticks=5),
        label_standoff=8,
        border_line_color=None,
        location=(0, 0),
        major_label_text_font_size="8pt",
        title=title,
        title_text_font_style="normal",
        title_text_font_size="9pt",
        title_standoff=8,
        width=12,
    )


def _style_categorical(plot) -> None:
    plot.axis.axis_line_color = None
    plot.axis.major_tick_line_color = None
    plot.yaxis.major_label_text_font_size = "11px"
    plot.yaxis.major_label_text_color = theme.INK
    plot.xaxis.major_label_text_color = theme.INK


def plot_delta_landscape(
    landscape: Landscape,
    subsystems: Sequence[str],
    *,
    title: str,
    width: int = 900,
):
    """Delta of the chosen subsystems (rows) in every group (columns).

    Rows are grouped by category, as ``select_landscape_subsystems`` orders
    them. A cell is +1 when every reaction of the subsystem is significantly
    higher in the group than in the rest, and -1 when every one is lower.
    """
    if not subsystems:
        raise ValueError("No subsystem meets the rule: lower the minimum size or pick another rule")
    groups = [str(group) for group in landscape.delta.columns]
    factors = [
        (landscape.categories[item] or "Uncategorized", landscape.names[item]) for item in subsystems
    ]
    cells: dict[str, list] = {key: [] for key in ("y", "group", "delta", "hover")}
    for item, factor in zip(subsystems, factors):
        for group in groups:
            cells["y"].append(factor)
            cells["group"].append(group)
            cells["delta"].append(float(landscape.delta.at[item, group]))
            cells["hover"].append(
                panel(
                    row("Subsystem", escape(landscape.names[item])),
                    row("Category", escape(factor[0])),
                    row("Group", escape(group)),
                    section(
                        row("Delta", f"{landscape.delta.at[item, group]:+.2f}"),
                        row(
                            "Significant",
                            f"{int(landscape.significant.at[item, group])} of "
                            f"{int(landscape.reactions[item])} reactions",
                        ),
                    ),
                    note(f"{escape(group)} against all other samples"),
                )
            )
    mapper = LinearColorMapper(palette=_diverging_palette(), low=-1.0, high=1.0)
    plot = apply_academic_style(
        figure(
            width=width,
            height=ROW_HEIGHT * len(factors) + 150,
            x_range=FactorRange(factors=groups),
            y_range=FactorRange(factors=list(reversed(factors)), range_padding=0.0),
            title=title,
            tools="save,reset",
            x_axis_location="above",
            sizing_mode="stretch_width",
        ),
        suppress_grid=True,
    )
    _style_categorical(plot)
    plot.yaxis.group_text_color = theme.INK_MUTED
    plot.yaxis.group_text_font_style = "normal"
    plot.xaxis.major_label_orientation = 0.6 if len(groups) > 6 else 0.0
    renderer = plot.rect(
        x="group",
        y="y",
        width=1,
        height=1,
        source=ColumnDataSource(cells),
        fill_color=transform("delta", mapper),
        line_color="#ffffff",
        line_width=1,
    )
    plot.add_layout(_color_bar(mapper, "Delta"), "right")
    plot.add_tools(HoverTool(renderers=[renderer], tooltips="@hover{safe}"))
    return plot


def sample_heatmap_rows(
    scores: pd.DataFrame, count: int
) -> list[list[str]]:
    """The ``count`` most variable distinct reaction profiles across samples.

    Reactions are ranked by their variance over samples. Reactions with the same
    profile (several reactions a tool scores alike, or one flux through a chain)
    share a row, so the rows show distinct profiles. Each item lists a row's
    reactions, the first naming it.
    """
    if count < 1:
        raise ValueError("Show at least one reaction")
    variance = scores.var(axis=0, ddof=1).fillna(0.0)
    order = variance.sort_values(ascending=False, kind="mergesort").index
    rows: dict[bytes, list[str]] = {}
    for column in order:
        if variance[column] <= 0:
            break
        key = np.round(scores[column].to_numpy(dtype=float), 9).tobytes()
        if key in rows:
            rows[key].append(str(column))
        elif len(rows) < count:
            rows[key] = [str(column)]
    return list(rows.values())


def plot_sample_heatmap(
    scores: pd.DataFrame,
    sample_groups: Mapping[str, str],
    *,
    count: int = 40,
    labels: Mapping[str, str] | None = None,
    group_colors: Mapping[str, str] | None = None,
    title: str,
    width: int = 900,
):
    """The most variable reactions (rows) across samples (columns), z-scored.

    Each row is scaled to mean zero and unit standard deviation over the shown
    samples, so the colour reads high or low for that reaction. Samples are
    ordered by group, in the groups' own order, under a band in each group's
    colour. Samples outside every group are left out. The matrix is one image,
    so hundreds of samples stay light to send and to draw.
    """
    labels = dict(labels or {})
    order = list(dict.fromkeys(sample_groups.values()))
    samples = [
        sample for group in order for sample, own in sample_groups.items() if own == group and sample in scores.index
    ]
    if len(samples) < 2:
        raise ValueError("The heatmap needs at least two grouped samples")
    block = scores.loc[samples]
    rows = sample_heatmap_rows(block, count)
    if not rows:
        raise ValueError("No reaction varies across these samples")
    colors = dict(group_colors or zip(order, group_palette(len(order))))

    names = []
    for members in rows:
        name = labels.get(members[0], members[0])
        names.append(name + (f" and {len(members) - 1} alike" if len(members) > 1 else ""))

    values = block[[members[0] for members in rows]].to_numpy(dtype=float)
    spread = values.std(axis=0, ddof=1)
    spread[spread == 0] = 1.0
    z = (values - values.mean(axis=0)) / spread
    reach = float(min(3.0, max(1.0, np.nanmax(np.abs(z)))))
    n_samples, n_rows = len(samples), len(rows)
    # Image rows run bottom to top: the most variable reaction is drawn last, on top.
    image = np.clip(z.T[::-1], -reach, reach).astype(np.float32)

    mapper = LinearColorMapper(palette=_diverging_palette(), low=-reach, high=reach)
    plot = apply_academic_style(
        figure(
            width=width,
            height=SAMPLE_ROW_HEIGHT * (n_rows + 1) + 170,
            x_range=Range1d(0, n_samples, bounds=(0, n_samples)),
            y_range=Range1d(0, n_rows + 1, bounds=(0, n_rows + 1)),
            title=title,
            tools="save,xpan,xwheel_zoom,reset",
            sizing_mode="stretch_width",
        ),
        suppress_grid=True,
    )
    _style_categorical(plot)
    plot.yaxis.major_label_text_font_size = "10px"
    plot.yaxis.ticker = FixedTicker(ticks=[index + 0.5 for index in range(n_rows + 1)])
    plot.yaxis.major_label_overrides = {
        **{n_rows - 1 - index + 0.5: name for index, name in enumerate(names)},
        n_rows + 0.5: BAND_ROW,
    }
    # Hundreds of sample names cannot be read; the band and the hover name them.
    plot.xaxis.ticker = FixedTicker(ticks=[])
    plot.xaxis.axis_label = f"{n_samples} samples, ordered by group"
    plot.xaxis.axis_label_text_color = theme.INK_MUTED
    plot.xaxis.axis_label_text_font_style = "normal"

    renderer = plot.image(
        image=[image], x=0, y=0, dw=n_samples, dh=n_rows, color_mapper=mapper
    )
    segments: dict[str, list] = {key: [] for key in ("left", "right", "color", "group", "samples")}
    start = 0
    for group in order:
        size = sum(1 for sample in samples if sample_groups[sample] == group)
        if not size:
            continue
        segments["left"].append(start)
        segments["right"].append(start + size)
        segments["color"].append(colors.get(group, theme.NOT_SIGNIFICANT))
        segments["group"].append(group)
        segments["samples"].append(size)
        start += size
    band = plot.quad(
        left="left", right="right", bottom=n_rows, top=n_rows + 1,
        fill_color="color", line_color="#ffffff", line_width=1,
        source=ColumnDataSource(segments),
    )
    lookup = CustomJSHover(
        args={
            "samples": samples,
            "groups": [sample_groups[sample] for sample in samples],
            "rows": list(reversed(names)),
            "ids": [members[0] for members in reversed(rows)],
            "scores": [float(f"{value:.4g}") for value in values.T[::-1].ravel()],
            "n": n_samples,
        },
        code="""
        const column = Math.min(n - 1, Math.max(0, Math.floor(special_vars.x)))
        const line = Math.min(rows.length - 1, Math.max(0, Math.floor(special_vars.y)))
        if (format === 'sample') return samples[column]
        if (format === 'group') return groups[column]
        if (format === 'reaction') return rows[line]
        if (format === 'id') return ids[line]
        return scores[line * n + column].toPrecision(4)
        """,
    )
    plot.add_tools(
        HoverTool(
            renderers=[renderer],
            tooltips=panel(
                row("Reaction", "$x{reaction}"),
                row("ID", "$x{id}"),
                row("Sample", "$x{sample}"),
                row("Group", "$x{group}"),
                section(row("Score", "$x{score}"), row("Standardized", "@image{+0.00}")),
            ),
            formatters={"$x": lookup},
        ),
        HoverTool(
            renderers=[band],
            tooltips=panel(row("Group", "@group"), row("Samples", "@samples")),
        ),
    )
    plot.add_layout(_color_bar(mapper, "Standardized score"), "right")
    return plot

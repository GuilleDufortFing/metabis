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
    CustomJS,
    CustomJSHover,
    FactorRange,
    FixedTicker,
    HoverTool,
    LinearColorMapper,
    Range1d,
    Span,
)
from bokeh.plotting import figure
from bokeh.transform import transform

from ..analysis.landscape import EnrichmentLandscape, Landscape
from ..managers.color_distributions import gradient_palette
from ..managers.group_palettes import group_palette
from ..managers.hover.panel import note, panel, row, section
from ..visual_config import theme
from .plot_reduction_artifact import (
    BAND_PREVIEW_CODE,
    FEATURE_REACH,
    FEATURE_TAP_CODE,
    feature_level_palette,
    feature_rows_source,
    gradient_choice,
    group_blocks,
)
from .plot_utils import apply_academic_style, short_row_label

__all__ = [
    "plot_delta_landscape",
    "plot_enrichment_landscape",
    "plot_marker_heatmap",
    "plot_sample_heatmap",
    "sample_heatmap_block",
    "sample_heatmap_rows",
]

# Genes ranked over every sample before the grouped samples are read: enough
# candidates that the most variable over the grouped samples are among them.
HEATMAP_CANDIDATES = 2000

ROW_HEIGHT = 20
# A table's columns keep at least this width, so labels never squeeze them out:
# a narrow page scrolls the figure instead. 11px Arial advances about 6.2px a character.
MIN_CELL_WIDTH = 72
LABEL_CHAR_WIDTH = 6.2
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


def _min_width(label_columns: Sequence[Sequence[str]], groups: int) -> int:
    """The width that leaves every column ``MIN_CELL_WIDTH`` beside its row labels and colour bar."""
    labels = sum(max((len(label) for label in column), default=0) * LABEL_CHAR_WIDTH + 24 for column in label_columns)
    return int(labels + groups * MIN_CELL_WIDTH + 110)


def _landscape_figure(
    factors: Sequence[tuple[str, str]],
    groups: Sequence[str],
    *,
    title: str,
    width: int,
    group_colors: Mapping[str, str] | None,
):
    """The shared frame of a subsystem by group table: rows grouped by category,
    one column per group, and above them a band in each group's colour."""
    band = ("", BAND_ROW)
    rows = list(reversed(factors)) + ([band] if group_colors else [])
    plot = apply_academic_style(
        figure(
            width=width,
            height=ROW_HEIGHT * len(rows) + 150,
            x_range=FactorRange(factors=list(groups)),
            y_range=FactorRange(factors=rows, range_padding=0.0),
            title=title,
            tools="save,reset",
            x_axis_location="above",
            sizing_mode="stretch_width",
            min_width=_min_width([[category for category, _ in factors], [name for _, name in factors]], len(groups)),
        ),
        suppress_grid=True,
    )
    _style_categorical(plot)
    plot.yaxis.group_text_color = theme.INK_MUTED
    plot.yaxis.group_text_font_style = "normal"
    plot.yaxis.group_text_font_size = "11px"
    plot.yaxis.group_label_orientation = "horizontal"
    plot.yaxis.separator_line_color = theme.GRID_LINE
    plot.xaxis.major_label_orientation = 0.6 if len(groups) > 6 else 0.0
    if group_colors:
        plot.rect(
            x=list(groups),
            y=[band] * len(groups),
            width=1,
            height=0.8,
            fill_color=[group_colors.get(group, theme.NOT_SIGNIFICANT) for group in groups],
            line_color="#ffffff",
            line_width=1,
        )
    return plot


def _against(group: str, against: Mapping[str, str] | None) -> str:
    return f"{group} against {(against or {}).get(group, 'all other samples')}"


def plot_delta_landscape(
    landscape: Landscape,
    subsystems: Sequence[str],
    *,
    title: str,
    width: int = 900,
    noun: str = "reactions",
    against: Mapping[str, str] | None = None,
    group_colors: Mapping[str, str] | None = None,
):
    """Delta of the chosen subsystems (rows) in every group (columns).

    Rows are grouped by category, as ``select_landscape_subsystems`` orders
    them. A cell is +1 when every reaction of the subsystem is significantly
    higher in the group than in the rest, and -1 when every one is lower.
    ``noun`` names the features counted (``genes`` for a gene landscape),
    ``against`` each group's other side, and ``group_colors`` a band over the
    columns in the groups' colours.
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
                            f"{int(landscape.reactions[item])} {noun}",
                        ),
                    ),
                    note(escape(_against(group, against))),
                )
            )
    mapper = LinearColorMapper(palette=_diverging_palette(), low=-1.0, high=1.0)
    plot = _landscape_figure(factors, groups, title=title, width=width, group_colors=group_colors)
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


def plot_enrichment_landscape(
    landscape: EnrichmentLandscape,
    subsystems: Sequence[str],
    *,
    title: str,
    width: int = 900,
    noun: str = "reactions",
    against: Mapping[str, str] | None = None,
    group_colors: Mapping[str, str] | None = None,
):
    """Over-representation of the chosen subsystems (rows) in every group (columns).

    A cell is the signed log2 enrichment of the significant direction: red when
    the subsystem is over-represented among the features higher in the group,
    blue among the lower. Grey is not significant, as everywhere a comparison is
    drawn; white with a dashed edge was not tested, too few of its features
    being measured. Laid out as :func:`plot_delta_landscape`.
    """
    if not subsystems:
        raise ValueError("No subsystem is over-represented in any group")
    groups = [str(group) for group in landscape.enrichment.columns]
    factors = [
        (landscape.categories[item] or "Uncategorized", landscape.names[item]) for item in subsystems
    ]
    tested: dict[str, list] = {key: [] for key in ("y", "group", "value", "hover")}
    untested: dict[str, list] = {key: [] for key in ("y", "group", "hover")}
    for item, factor in zip(subsystems, factors):
        for group in groups:
            head = (row("Subsystem", escape(landscape.names[item])), row("Group", escape(group)))
            if not bool(landscape.tested.at[item, group]):
                untested["y"].append(factor)
                untested["group"].append(group)
                untested["hover"].append(
                    panel(*head, section(row("Not tested", f"too few measured {noun}")), note(escape(_against(group, against))))
                )
                continue
            value = float(landscape.enrichment.at[item, group])
            significant = np.isfinite(value)
            verdict = (
                f"over-represented among the {noun} {'higher' if value > 0 else 'lower'} in {group}"
                if significant
                else "not significant"
            )
            tested["y"].append(factor)
            tested["group"].append(group)
            tested["value"].append(value if significant else float("nan"))
            tested["hover"].append(
                panel(
                    *head,
                    section(
                        row("Result", escape(verdict)),
                        row("Signed log2 enrichment", f"{value:+.2f}" if significant else "-"),
                        row("Adjusted p", f"{float(landscape.adjusted_pval.at[item, group]):.3g}"),
                    ),
                    note(escape(_against(group, against))),
                )
            )
    finite = [abs(v) for v in tested["value"] if np.isfinite(v)]
    reach = max(finite + [1.0])
    mapper = LinearColorMapper(palette=_diverging_palette(), low=-reach, high=reach, nan_color=theme.NOT_SIGNIFICANT)
    plot = _landscape_figure(factors, groups, title=title, width=width, group_colors=group_colors)
    renderers = [
        plot.rect(
            x="group", y="y", width=1, height=1, source=ColumnDataSource(tested),
            fill_color=transform("value", mapper), line_color="#ffffff", line_width=1,
        )
    ]
    if untested["y"]:
        renderers.append(
            plot.rect(
                x="group", y="y", width=0.92, height=0.84, source=ColumnDataSource(untested),
                fill_color="#ffffff", line_color=theme.INK_MUTED, line_dash="dashed", line_width=1,
            )
        )
    plot.add_layout(_color_bar(mapper, "Signed log2 enrichment"), "right")
    plot.add_tools(HoverTool(renderers=renderers, tooltips="@hover{safe}"))
    return plot


def plot_marker_heatmap(
    markers: pd.DataFrame,
    means: pd.DataFrame,
    *,
    title: str,
    labels: Mapping[str, str] | None = None,
    group_colors: Mapping[str, str] | None = None,
    width: int = 640,
):
    """Each group's markers (rows, grouped by group) by group (columns).

    ``markers`` are the rows to draw, as :func:`metabis.analysis.markers.top_markers`
    gives them; ``means`` the features' group means, features by groups. A cell
    is the group's mean standardized across the groups, on a level's scale, as
    the OSCA book's heatmap of markers centres each row. A band above the
    columns carries the groups' colours, and a line closes each group's block.
    """
    if markers.empty:
        raise ValueError("No feature is higher in one group than in every other")
    labels = dict(labels or {})
    groups = [str(group) for group in means.columns]
    names: list[str] = []
    for feature in markers["feature_id"]:
        name = str(labels.get(feature, feature))
        name = short_row_label(name, 0, 40)
        while name in names:
            name += "\u200b"
        names.append(name)
    block = means.loc[list(markers["feature_id"]), groups].astype(float)
    spread = block.std(axis=1, ddof=1).replace(0, 1.0)
    standardized = block.sub(block.mean(axis=1), axis=0).div(spread, axis=0).clip(-FEATURE_REACH, FEATURE_REACH)
    cells: dict[str, list] = {key: [] for key in ("x", "y", "z", "hover")}
    for name, (_, marker) in zip(names, markers.iterrows()):
        feature = marker["feature_id"]
        for group in groups:
            cells["x"].append(group)
            cells["y"].append(name)
            cells["z"].append(float(standardized.at[feature, group]))
            cells["hover"].append(
                panel(
                    row("Feature", escape(str(labels.get(feature, feature)))),
                    row("ID", escape(str(feature))),
                    row("Marker of", escape(str(marker["group"]))),
                    section(
                        row(f"Mean in {escape(group)}", f"{float(block.at[feature, group]):.4g}"),
                        row("Standardized", f"{float(standardized.at[feature, group]):+.2f}"),
                    ),
                    section(
                        row("Smallest Cohen's d", f"{float(marker['min_cohens_d']):.2f}"),
                        row("Closest group", escape(str(marker["closest_group"]))),
                    ),
                )
            )
    mapper = LinearColorMapper(palette=feature_level_palette(), low=-FEATURE_REACH, high=FEATURE_REACH)
    rows = list(reversed(names)) + ([BAND_ROW] if group_colors else [])
    plot = apply_academic_style(
        figure(
            width=width,
            height=ROW_HEIGHT * len(rows) + 150,
            x_range=FactorRange(factors=groups),
            y_range=FactorRange(factors=rows, range_padding=0.0),
            title=title,
            tools="save,reset",
            x_axis_location="above",
            sizing_mode="stretch_width",
            min_width=_min_width([names], len(groups)),
        ),
        suppress_grid=True,
    )
    _style_categorical(plot)
    plot.xaxis.major_label_orientation = 0.6 if len(groups) > 4 else 0.0
    if group_colors:
        plot.rect(
            x=groups, y=[BAND_ROW] * len(groups), width=1, height=0.8,
            fill_color=[group_colors.get(group, theme.NOT_SIGNIFICANT) for group in groups],
            line_color="#ffffff", line_width=1,
        )
    renderer = plot.rect(
        x="x", y="y", width=1, height=1, source=ColumnDataSource(cells),
        fill_color=transform("z", mapper), line_color="#ffffff", line_width=1,
    )
    # A line under each group's block of markers, counted from the top.
    edge = 0
    sizes = markers["group"].value_counts(sort=False)
    for group in groups:
        count = int(sizes.get(group, 0))
        edge += count
        if count and 0 < edge < len(names):
            plot.add_layout(Span(location=len(names) - edge, dimension="width", line_color=theme.INK, line_width=1))
    plot.add_layout(_color_bar(mapper, "Group mean, standardized"), "right")
    plot.add_tools(HoverTool(renderers=[renderer], tooltips="@hover{safe}"))
    return plot


def sample_heatmap_block(matrix, samples: Sequence[str], count: int) -> pd.DataFrame:
    """The grouped samples' values for the features a heatmap of ``count`` rows can show.

    An expression matrix holds tens of thousands of genes, often sparse, so it
    is never made dense whole: genes are ranked by their variance over every
    sample, which the matrix computes without densifying, and the most variable
    ``max(HEATMAP_CANDIDATES, 20 * count)`` are read for the grouped samples.
    :func:`sample_heatmap_rows` then ranks them over those samples alone.
    """
    variances = matrix.feature_variances().fillna(0.0)
    pool = variances.sort_values(ascending=False, kind="mergesort").index[: max(HEATMAP_CANDIDATES, 20 * count)]
    block = matrix.select_features(list(pool))
    block.index = block.index.map(str)
    return block.loc[[sample for sample in samples if sample in block.index]]


def sample_heatmap_rows(
    scores: pd.DataFrame, count: int
) -> list[list[str]]:
    """The ``count`` most variable distinct feature profiles across samples.

    Reactions or genes are ranked by their variance over samples. Features with
    the same profile (several reactions a tool scores alike, or one flux through
    a chain) share a row, so the rows show distinct profiles. Each item lists a
    row's features, the first naming it.
    """
    if count < 1:
        raise ValueError("Show at least one row")
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
    data_type: str = "reactions",
    label_limit: int | None = None,
    feature_event: str | None = None,
    highlight_groups: Sequence[str] = (),
    hidden_groups: Sequence[str] = (),
    fill: bool = False,
    group_event: str | None = None,
    group_preview_event: str | None = None,
):
    """The most variable reactions or genes (rows) across samples (columns), z-scored.

    Each row is scaled to mean zero and unit standard deviation over the shown
    samples, so the colour reads high or low for that feature. Samples are
    ordered by group, in the groups' own order, under a band in each group's
    colour. Samples outside every group are left out. The matrix is one image,
    so hundreds of samples stay light to send and to draw.

    With ``label_limit`` the row names are cut to that many characters, for a
    heatmap set beside another figure; the hover keeps the full name and ID. A
    click on a row outlines it in the logo's gradient and dispatches
    ``feature_event`` with the feature, so the page can colour a reduction by it.

    Rows are chosen and scaled over every grouped sample; ``hidden_groups``
    then leave the figure, so hiding a group changes no other cell's colour.
    With ``highlight_groups`` the other groups' columns fade and the highlighted
    bands are outlined in the logo's gradient; the page moves the focus through
    the source named ``GROUP_FOCUS``, as on the map. A click on a group's segment
    of the band dispatches ``group_event`` with the group, and the pointer on
    one ``group_preview_event`` (with no group as it leaves), so the band works
    as the page's group chips do.
    With ``fill`` the figure takes its container's height, its rows dividing
    it, rather than a height of its own.
    """
    noun = "Reaction" if data_type == "reactions" else "Gene"
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
        raise ValueError(f"No {noun.lower()} varies across these samples")
    colors = dict(group_colors or zip(order, group_palette(len(order))))

    # Two reactions can share a name (one transport into several compartments):
    # a repeated name carries its ID, so every row reads as its own.
    first_names = [labels.get(members[0], members[0]) for members in rows]
    repeated = {name for name in first_names if first_names.count(name) > 1}
    names = []
    for members, name in zip(rows, first_names):
        if name in repeated:
            name += f" [{members[0]}]"
        names.append(name + (f" and {len(members) - 1} alike" if len(members) > 1 else ""))

    values = block[[members[0] for members in rows]].to_numpy(dtype=float)
    spread = values.std(axis=0, ddof=1)
    spread[spread == 0] = 1.0
    z = (values - values.mean(axis=0)) / spread
    # Three standard deviations either way, always: a reduction coloured by a
    # feature uses the same scale, so a dot has the colour of its cell.
    reach = FEATURE_REACH
    hidden = set(hidden_groups)
    keep = [index for index, sample in enumerate(samples) if sample_groups[sample] not in hidden]
    if not keep:
        raise ValueError("Every group is hidden")
    samples = [samples[index] for index in keep]
    values, z = values[keep], z[keep]
    n_samples, n_rows = len(samples), len(rows)
    # Image rows run bottom to top: the most variable reaction is drawn last, on top.
    image = np.clip(z.T[::-1], -reach, reach).astype(np.float32)

    # A level, purple for low through light grey to gold for high, as the
    # reduction coloured by one feature is drawn.
    mapper = LinearColorMapper(palette=feature_level_palette(), low=-reach, high=reach)
    plot = apply_academic_style(
        figure(
            width=width,
            height=SAMPLE_ROW_HEIGHT * (n_rows + 1) + 170,
            x_range=Range1d(0, n_samples, bounds=(0, n_samples)),
            y_range=Range1d(0, n_rows + 1, bounds=(0, n_rows + 1)),
            title=title,
            tools="save,xpan,xwheel_zoom,reset",
            sizing_mode="stretch_both" if fill else "stretch_width",
        ),
        suppress_grid=True,
    )
    _style_categorical(plot)
    plot.yaxis.major_label_text_font_size = "10px"
    plot.yaxis.ticker = FixedTicker(ticks=[index + 0.5 for index in range(n_rows + 1)])
    ticks = (
        [short_row_label(name, len(members) - 1, label_limit) for members, name in zip(rows, first_names)]
        if label_limit is not None
        else names
    )
    plot.yaxis.major_label_overrides = {
        **{n_rows - 1 - index + 0.5: tick for index, tick in enumerate(ticks)},
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
    # A group is named on the band where its segment is wide enough to hold the
    # name; the hover names the rest.
    named = [
        index
        for index, (group, left, right) in enumerate(zip(segments["group"], segments["left"], segments["right"]))
        if (right - left) / n_samples * width >= 7 * len(group) + 8
    ]
    if named:
        plot.text(
            x=[(segments["left"][i] + segments["right"][i]) / 2 for i in named],
            y=[n_rows + 0.5] * len(named),
            text=[segments["group"][i] for i in named],
            text_align="center", text_baseline="middle", text_font_size="10px",
            text_color=theme.INK, background_fill_color="#ffffff", background_fill_alpha=0.85,
            padding=2,
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
        if (format === 'feature') return rows[line]
        if (format === 'id') return ids[line]
        return scores[line * n + column].toPrecision(4)
        """,
    )
    plot.add_tools(
        HoverTool(
            renderers=[renderer],
            tooltips=panel(
                row(noun, "$x{feature}"),
                row("ID", "$x{id}"),
                row("Sample", "$x{sample}"),
                row("Group", "$x{group}"),
                section(row("Score" if data_type == "reactions" else "Expression", "$x{score}"), row("Standardized", "@image{+0.00}")),
            ),
            formatters={"$x": lookup},
        ),
        HoverTool(
            renderers=[band],
            tooltips=panel(row("Group", "@group"), row("Samples", "@samples")),
        ),
    )
    plot.add_layout(_color_bar(mapper, "Standardized score"), "right")
    # Under the chosen row's outline, so a faded block never hides it.
    blocks = group_blocks(
        plot,
        list(zip(segments["group"], segments["left"], segments["right"])),
        highlight_groups,
        bottom=0,
        top=n_rows + 1,
        band=(n_rows + 0.06, n_rows + 0.94),
        piece=max(n_samples / 120, 0.05),
    )
    choice = gradient_choice(plot, 0, n_samples)
    plot.tags = [*plot.tags, feature_rows_source(rows, [n_rows - 1 - index + 0.5 for index in range(n_rows)])]
    if feature_event or group_event:
        plot.js_on_event(
            "tap",
            CustomJS(
                args=dict(
                    choice=choice,
                    rows=list(reversed(names)),
                    ids=[members[0] for members in reversed(rows)],
                    labels=list(reversed(first_names)),
                    data_type=data_type,
                    feature_event=feature_event,
                    component_event=None,
                    base=n_rows,
                    count=0,
                    numbers=[],
                    group_event=group_event,
                    blocks=blocks,
                ),
                code=FEATURE_TAP_CODE,
            ),
        )
    if group_preview_event:
        preview = CustomJS(args=dict(blocks=blocks, base=n_rows, event=group_preview_event), code=BAND_PREVIEW_CODE)
        plot.js_on_event("mousemove", preview)
        plot.js_on_event("mouseleave", preview)
    return plot

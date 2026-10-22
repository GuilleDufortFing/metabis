"""The subsystem rankings' shared layout, and the link they share with the network.

A ranking lists every subsystem in its own row, grouped by category, as a forest
plot lists its studies:

* **Names in a column.** Each row is named on the y-axis, beside the plot rather
  than inside it, so a name never sits under a dot and never pans with the data.
* **Rows a reader can follow.** Every other row is shaded, as DataTables stripes a
  table (its ``stripe`` rows are about #f9f9f9, here the client's #f8f8f8), and each
  category opens with a header row on the client's card tint, its name in capitals.
* **The network's order.** Categories follow the metabolic network's fixed order
  (:mod:`..metabolic_visualization.generation.graph_building.spaced_network`), so a
  category is found in the same place in both; within one, the strongest score
  comes first.
* **Axes at both ends.** The value axis is drawn above and below, so it is in view
  wherever the column is scrolled.

The network and the ranking also share one selection. Each figure holds a data
source named :data:`FOCUS_SOURCE` that the page writes (the subsystem in focus,
and the matches of a search), and redraws its own highlight from it. Pointing at
a subsystem in either figure dispatches :data:`FOCUS_EVENT` on ``window`` for the
page to relay to both.
"""

from __future__ import annotations

import math
import textwrap
import zlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from bokeh.models import (
    ColumnDataSource,
    CustomJS,
    FactorRange,
    FixedTicker,
    HoverTool,
    LinearAxis,
    Range1d,
    Span,
    TapTool,
)
from bokeh.plotting import figure

from ..visual_config import theme
from .subsystem_link import (
    FOCUS_EVENT,
    FOCUS_SOURCE,
    attach,
    focus_dispatch,
    focus_source,
    navigation_dispatch,
)
from .metabolic_visualization.generation.graph_building.spaced_network import (
    category_sort_key,
)
from .plot_utils import apply_academic_style

# The page reads these to find a subsystem's row (web/src/bokeh/subsystemLink.ts).
RANKING_ROWS_SOURCE = "metabis-ranking-rows"
MATCHED_ROWS = "metabis-ranking-matched"
FOCUSED_ROWS = "metabis-ranking-focused"
RANKING_PLOT = "metabis-ranking"
RANKING_DOTS = "metabis-ranking-dots"

# Beside the network the ranking has 450 to 550 px: names wrap onto a second line
# past this many characters, so the name column stays near 180 px, and every row
# is tall enough for two lines of 11 px text.
NAME_LENGTH = 28
HEADER_LENGTH = 22  # capitals run wider
ROW_HEIGHT = 30
HEADER_SHADE = "#eeeeee"  # the client's --card-tint
ROW_SHADE = theme.ROW_SHADE  # the client's --sheet, about DataTables' stripe
FOCUS_SHADE = "#d6ecff"  # the client's link blue, #0162b7, at 16% on white
MATCH_SHADE = "#fff7bf"  # GOV.UK's yellow tag, the client's --warning-bg
SUMMARY_COLOR = theme.INK
SPREAD = 0.56  # the share of a row the dots spread over


@dataclass(frozen=True)
class RankedSubsystem:
    subsystem_id: str
    name: str
    category: str
    score: float


@dataclass
class RankingRows:
    """Rows top to bottom: a header factor per category, then its subsystems."""

    factors: List[str] = field(default_factory=list)
    headers: List[str] = field(default_factory=list)
    subsystems: List[str] = field(default_factory=list)  # their factors, in order
    factor_of: Dict[str, str] = field(default_factory=dict)  # subsystem id -> factor
    category_of: Dict[str, str] = field(default_factory=dict)  # subsystem id -> category
    name_of: Dict[str, str] = field(default_factory=dict)  # subsystem id -> full name

    @property
    def y_range_factors(self) -> List[str]:
        """Bokeh lists categorical factors bottom up."""
        return list(reversed(self.factors))


def wrapped_name(name: str, length: int = NAME_LENGTH) -> str:
    """``name`` on lines of at most ``length`` characters, broken at words, two at most.

    A name too long for two lines keeps its words on the second, which runs longer;
    nothing is cut, so no two subsystems ever read the same.
    """
    lines = textwrap.wrap(name, length, break_long_words=False, break_on_hyphens=False)
    if len(lines) <= 2:
        return "\n".join(lines)
    # Too long for two lines of ``length``: two balanced lines, a little wider.
    balanced = textwrap.wrap(
        name, math.ceil(len(name) / 2) + 2, break_long_words=False, break_on_hyphens=False
    )
    return "\n".join([balanced[0], " ".join(balanced[1:])])


def order_rows(entries: Iterable[RankedSubsystem]) -> RankingRows:
    """Categories in the network's order; within one, the strongest score first."""
    by_category: Dict[str, List[RankedSubsystem]] = {}
    for entry in entries:
        by_category.setdefault(entry.category or "Uncategorized", []).append(entry)

    rows = RankingRows()
    used = set()

    def unique(text: str) -> str:
        candidate = text
        suffix = 2
        while candidate in used:
            candidate = f"{text} ({suffix})"
            suffix += 1
        used.add(candidate)
        return candidate

    for category in sorted(by_category, key=category_sort_key):
        header = unique(wrapped_name(category.upper(), HEADER_LENGTH))
        rows.factors.append(header)
        rows.headers.append(header)
        members = sorted(
            by_category[category],
            key=lambda entry: (-abs(entry.score), entry.name.lower()),
        )
        for entry in members:
            factor = unique(wrapped_name(entry.name))
            rows.factors.append(factor)
            rows.subsystems.append(factor)
            rows.factor_of[entry.subsystem_id] = factor
            rows.category_of[entry.subsystem_id] = category
            rows.name_of[entry.subsystem_id] = entry.name
    return rows


def spread_offset(element_id: object) -> float:
    """A fixed place within the row for one dot, the same on every render."""
    fraction = (zlib.crc32(str(element_id).encode()) % 1000) / 999.0
    return (fraction - 0.5) * SPREAD


def ranking_figure(
    rows: RankingRows,
    *,
    title: str,
    x_range: Range1d,
    x_label: str,
    names: bool = True,
    width: Optional[int] = None,
    sizing_mode: Optional[str] = "stretch_width",
    tools: str = "xpan,xwheel_zoom,reset,save",
    name: Optional[str] = RANKING_PLOT,
):
    """A figure with the ranking's rows, shading, zero line and both axes."""
    y_range = FactorRange(factors=rows.y_range_factors, range_padding=0.0)
    kwargs = dict(
        title=title,
        x_range=x_range,
        y_range=y_range,
        height=ROW_HEIGHT * len(rows.factors) + 110,
        tools=tools,
        toolbar_location="above",
        x_axis_location="above",
        min_border_left=8,
        min_border_right=16,
        name=name,
    )
    if width is not None:
        kwargs["width"] = width
    if sizing_mode:
        kwargs["sizing_mode"] = sizing_mode
    plot = figure(**kwargs)
    apply_academic_style(plot)
    shade_rows(plot, rows, x_range)
    # The title names the top axis; the bottom one, out of the title's sight
    # once the column is scrolled, carries the label.
    plot.add_layout(LinearAxis(axis_label=x_label), "below")
    plot.add_layout(
        Span(
            location=0,
            dimension="height",
            line_color=theme.THRESHOLD_LINE_COLOR,
            line_dash=theme.THRESHOLD_LINE_DASH,
            line_width=theme.THRESHOLD_LINE_WIDTH,
        )
    )
    plot.ygrid.grid_line_color = None
    yaxis = plot.yaxis[0]
    yaxis.axis_line_color = None
    yaxis.major_tick_line_color = None
    yaxis.major_label_text_font_size = "11px"
    for axis in plot.xaxis:
        axis.ticker.desired_num_ticks = 4
    yaxis.major_label_text_color = theme.INK
    yaxis.visible = names
    return plot


def shade_rows(plot, rows: RankingRows, x_range: Range1d) -> None:
    """Header rows on the card tint, every other subsystem row on the sheet grey."""
    span = (x_range.start - 1e6, x_range.end + 1e6)
    striped = []
    index = 0
    for factor in rows.factors:
        if factor in rows.headers:
            index = 0
            continue
        if index % 2 == 1:
            striped.append(factor)
        index += 1
    for factors, color in ((rows.headers, HEADER_SHADE), (striped, ROW_SHADE)):
        if factors:
            plot.hbar(
                y=factors,
                left=span[0],
                right=span[1],
                height=1.0,
                fill_color=color,
                line_color=None,
                level="underlay",
            )


# Pixels per character of the 11 px names, lower case and capitals (Arial), and the
# room the summary's own axis takes beside them.
NAME_CHARACTER_WIDTH = 5.4
HEADER_CHARACTER_WIDTH = 7.3
SUMMARY_FRAME_WIDTH = 84


def names_width(rows: RankingRows) -> int:
    """About how wide the name column is, in pixels, from its longest line."""
    widest = 0.0
    for factor in rows.factors:
        per_character = HEADER_CHARACTER_WIDTH if factor in rows.headers else NAME_CHARACTER_WIDTH
        for line in factor.split("\n"):
            widest = max(widest, len(line) * per_character)
    return int(widest) + 20


def summary_panel(
    rows: RankingRows,
    plot,
    scores: Mapping[str, float],
    *,
    title: str,
    x_range: Range1d,
    x_label: str,
    width: Optional[int] = None,
):
    """The row names and each subsystem's summary score, beside the dots.

    The summary is on its own scale (a delta score runs from -1 to +1 under the sign
    transform), so it takes its own panel rather than a second axis over the dots.
    """
    panel = ranking_figure(
        rows,
        title=title,
        x_range=x_range,
        x_label=x_label,
        names=True,
        width=width or names_width(rows) + SUMMARY_FRAME_WIDTH,
        sizing_mode=None,
        tools="save",
        name=None,
    )
    panel.min_border_right = 4
    panel.y_range = plot.y_range
    matched = plot.select_one({"name": MATCHED_ROWS})
    focused = plot.select_one({"name": FOCUSED_ROWS})
    if matched is not None and focused is not None:
        highlight_rows(panel, matched, focused)
    # A narrow axis: its ends and the middle.
    reach = max(abs(x_range.start), abs(x_range.end)) / 1.08
    for axis in panel.xaxis:
        axis.ticker = FixedTicker(ticks=[-round(reach, 2), 0.0, round(reach, 2)])
    panel.toolbar_location = None
    ids = [sid for sid in rows.factor_of if sid in scores]
    diamonds = panel.scatter(
        x="score",
        y="y",
        source=ColumnDataSource(
            {
                "score": [float(scores[sid]) for sid in ids],
                "y": [rows.factor_of[sid] for sid in ids],
                "name": [rows.name_of.get(sid, rows.factor_of[sid]) for sid in ids],
            }
        ),
        marker="diamond",
        size=13,
        fill_color=SUMMARY_COLOR,
        line_color="#ffffff",
        line_width=1,
    )
    panel.add_tools(HoverTool(renderers=[diamonds], tooltips=[("", "@name"), (title, "@score{0.00}")]))
    return panel


def add_focus_link(
    plot, rows: RankingRows, *, navigation_event: Optional[str] = None
) -> ColumnDataSource:
    """Highlight the subsystem in focus, and the matches of a search, in this ranking.

    Pointing at a row dispatches :data:`FOCUS_EVENT`, and tapping it dispatches
    ``navigation_event`` when given; the page writes the shared selection into the
    returned source, and the bands follow it.
    """
    span = (plot.x_range.start - 1e6, plot.x_range.end + 1e6)
    ids = list(rows.factor_of)
    factors = [rows.factor_of[sid] for sid in ids]
    lookup = ColumnDataSource({"id": ids, "factor": factors}, name=RANKING_ROWS_SOURCE)
    focus = focus_source()
    matched = ColumnDataSource({"y": []}, name=MATCHED_ROWS)
    focused = ColumnDataSource({"y": []}, name=FOCUSED_ROWS)
    highlight_rows(plot, matched, focused)
    focus.js_on_change(
        "data",
        CustomJS(
            args=dict(focus=focus, lookup=lookup, matched=matched, focused=focused),
            code="""
            const factor = new Map(lookup.data.id.map((id, i) => [id, lookup.data.factor[i]]))
            const pick = (kind) => focus.data.id
              .filter((id, i) => focus.data.kind[i] === kind && factor.has(id))
              .map((id) => factor.get(id))
            matched.data = {y: pick('match')}
            focused.data = {y: pick('focus')}
            """,
        ),
    )
    # An invisible band per row, for pointing at and tapping anywhere along it.
    hits = ColumnDataSource({"y": factors, "id": ids, "label": factors})
    band = plot.hbar(
        y="y",
        left=span[0],
        right=span[1],
        height=1.0,
        fill_alpha=0,
        line_alpha=0,
        hover_fill_alpha=0,
        selection_fill_alpha=0,
        nonselection_fill_alpha=0,
        nonselection_line_alpha=0,
        level="underlay",
        source=hits,
    )
    plot.add_tools(HoverTool(renderers=[band], tooltips=None, callback=focus_dispatch(hits)))
    if navigation_event is not None:
        plot.add_tools(TapTool(renderers=[band]))
        hits.selected.js_on_change("indices", navigation_dispatch(hits, navigation_event))
    attach(plot, focus)
    return focus


def highlight_rows(plot, matched: ColumnDataSource, focused: ColumnDataSource) -> None:
    """Bands behind a search's matches and the subsystem in focus, across ``plot``."""
    span = (plot.x_range.start - 1e6, plot.x_range.end + 1e6)
    for source, color in ((matched, MATCH_SHADE), (focused, FOCUS_SHADE)):
        plot.hbar(
            y="y",
            left=span[0],
            right=span[1],
            height=1.0,
            fill_color=color,
            line_color=None,
            level="underlay",
            source=source,
        )


def ranking_layout(summary, plot):
    """The summary panel and the dots side by side, sharing their rows.

    One toolbar serves both, and its save button exports the whole grid, names
    included: BokehJS's SaveTool exports the view that owns its toolbar.
    """
    from bokeh.layouts import gridplot

    grid = gridplot(
        [[summary, plot]],
        merge_tools=True,
        toolbar_location="above",
        sizing_mode="stretch_width",
    )
    # The names and the summary keep their width; the dots take the rest.
    grid.cols = [f"{summary.width}px", "1fr"]
    return grid


__all__ = [
    "FOCUS_EVENT",
    "FOCUS_SOURCE",
    "RANKING_DOTS",
    "RANKING_PLOT",
    "RANKING_ROWS_SOURCE",
    "RankedSubsystem",
    "RankingRows",
    "add_focus_link",
    "order_rows",
    "ranking_figure",
    "ranking_layout",
    "shade_rows",
    "spread_offset",
    "summary_panel",
]

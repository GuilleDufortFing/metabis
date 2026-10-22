"""The subsystem rankings' shared layout, and the link they share with the network.

A ranking lists every subsystem in its own row, grouped by category, as a forest
plot lists its studies:

* **Names in a panel of their own.** Each row's name is drawn right-aligned in a
  panel beside the dots, which never pans with the data and, in the comparison
  ranking, also holds each subsystem's Delta score.
* **Categories as the network draws them.** Each category opens with its number
  from the network and its name, flush left and bold, and its rows sit in a frame
  with the border of the network's category circles, with a space before the next.
  The heading is a kind of text of its own, never a row among the subsystems.
* **Rows a reader can follow.** Every other row is shaded, as DataTables stripes a
  table (its ``stripe`` rows are about #f9f9f9, here the client's #f8f8f8).
* **The network's order.** Categories follow the metabolic network's fixed order
  (:mod:`..metabolic_visualization.generation.graph_building.spaced_network`), so a
  category is found in the same place, under the same number, in both; within one,
  the strongest score comes first.
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
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from bokeh.core.properties import value
from bokeh.models import (
    BoxAnnotation,
    ColumnDataSource,
    CustomJS,
    FactorRange,
    FixedTicker,
    HoverTool,
    LinearAxis,
    Range1d,
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
    category_numbers,
    category_sort_key,
)
from .plot_utils import apply_academic_style

# The page reads these to find a subsystem's row (web/src/bokeh/subsystemLink.ts).
RANKING_ROWS_SOURCE = "metabis-ranking-rows"
MATCHED_ROWS = "metabis-ranking-matched"
FOCUSED_ROWS = "metabis-ranking-focused"
RANKING_PLOT = "metabis-ranking"
RANKING_DOTS = "metabis-ranking-dots"
RANKING_NAMES = "metabis-ranking-names"

# Beside the network the ranking has 450 to 700 px: names wrap onto a second line
# past this many characters, so the name column stays near 170 px, and every row
# is tall enough for two lines of 11 px text.
NAME_LENGTH = 28
ROW_HEIGHT = 30
ROW_SHADE = theme.ROW_SHADE  # the client's --sheet, about DataTables' stripe
FOCUS_SHADE = "#d6ecff"  # the client's link blue, #0162b7, at 16% on white
MATCH_SHADE = "#fff7bf"  # GOV.UK's yellow tag, the client's --warning-bg
SUMMARY_COLOR = theme.INK
FRAME_LINE = "#b1b4b6"  # the network's category circles (DEFAULT_CATEGORY_LINE_COLOR)
FRAME_LINE_WIDTH = 1.2
SPREAD = 0.56  # the share of a row the dots spread over

# Pixels, about, per character of the 11 px names and the 13 px bold headings
# (Arial, measured on the Human1 names), and the panel's fixed parts.
NAME_CHARACTER_WIDTH = 5.4
HEADING_CHARACTER_WIDTH = 6.9
HEADING_FONT_SIZE = "13px"
NAME_FONT_SIZE = "11px"
BADGE_WIDTH = 26  # from the panel's edge to a heading's first letter
SUMMARY_FRAME_WIDTH = 84  # the Delta score's own axis, right of the names
NAMES_MIN_WIDTH = 240  # room for a heading of about 30 characters on one line


@dataclass(frozen=True)
class RankedSubsystem:
    subsystem_id: str
    name: str
    category: str
    score: float


@dataclass(frozen=True)
class Heading:
    """A category's heading: its number, its name in lines, and the rows it takes."""

    number: int
    category: str
    text: str
    factors: Tuple[str, ...]


@dataclass
class RankingRows:
    """Rows top to bottom: per category, its heading, its subsystems, then a gap."""

    factors: List[str] = field(default_factory=list)
    headings: List[Heading] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    subsystems: List[str] = field(default_factory=list)  # their factors, in order
    members: Dict[str, List[str]] = field(default_factory=dict)  # category -> factors
    factor_of: Dict[str, str] = field(default_factory=dict)  # subsystem id -> factor
    category_of: Dict[str, str] = field(default_factory=dict)  # subsystem id -> category
    name_of: Dict[str, str] = field(default_factory=dict)  # subsystem id -> full name
    text_of: Dict[str, str] = field(default_factory=dict)  # factor -> its name in lines
    names_width: int = 0  # pixels the longest name line takes

    @property
    def y_range_factors(self) -> List[str]:
        """Bokeh lists categorical factors bottom up."""
        return list(reversed(self.factors))

    def edges(self, category: str) -> Tuple[float, float]:
        """The bottom and top of a category's rows, in the y-range's own units."""
        bottom_up = self.y_range_factors
        members = self.members[category]
        return float(bottom_up.index(members[-1])), float(bottom_up.index(members[0]) + 1)


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


def panel_width(names_width: int, scores: bool) -> int:
    """The names panel's width: the names, and the Delta score's axis when it has one."""
    return max(names_width + (SUMMARY_FRAME_WIDTH if scores else 0), NAMES_MIN_WIDTH)


def order_rows(
    entries: Iterable[RankedSubsystem],
    *,
    numbers: Optional[Mapping[str, int]] = None,
    scores: bool = True,
) -> RankingRows:
    """Categories in the network's order; within one, the strongest score first.

    ``numbers`` are the network's category numbers (:func:`category_numbers` over
    every category in use), so a ranking that lists only some categories keeps
    their numbers; without them the categories listed are numbered. ``scores``
    says whether the names panel also holds a score, which sets its width and so
    where a heading wraps.
    """
    by_category: Dict[str, List[RankedSubsystem]] = {}
    for entry in entries:
        by_category.setdefault(entry.category or "Uncategorized", []).append(entry)
    if numbers is None or not set(by_category) <= set(numbers):
        numbers = category_numbers([*(numbers or {}), *by_category])

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

    ordered = {
        category: sorted(
            by_category[category], key=lambda entry: (-abs(entry.score), entry.name.lower())
        )
        for category in sorted(by_category, key=category_sort_key)
    }
    texts = {
        entry.subsystem_id: wrapped_name(entry.name)
        for members in ordered.values()
        for entry in members
    }
    rows.names_width = int(
        max(
            (len(line) for text in texts.values() for line in text.split("\n")),
            default=0,
        )
        * NAME_CHARACTER_WIDTH
    ) + 20
    heading_room = panel_width(rows.names_width, scores) - BADGE_WIDTH - 8
    heading_length = max(12, int(heading_room / HEADING_CHARACTER_WIDTH))

    for index, (category, members) in enumerate(ordered.items()):
        if index:
            gap = unique(f"gap:{numbers[category]}")
            rows.factors.append(gap)
            rows.gaps.append(gap)
        text = wrapped_name(category, heading_length)
        factors = tuple(
            unique(f"heading:{numbers[category]}:{line}")
            for line in range(len(text.split("\n")))
        )
        rows.factors.extend(factors)
        rows.headings.append(Heading(numbers[category], category, text, factors))
        rows.members[category] = []
        for entry in members:
            factor = unique(texts[entry.subsystem_id])
            rows.factors.append(factor)
            rows.subsystems.append(factor)
            rows.members[category].append(factor)
            rows.factor_of[entry.subsystem_id] = factor
            rows.category_of[entry.subsystem_id] = category
            rows.name_of[entry.subsystem_id] = entry.name
            rows.text_of[factor] = texts[entry.subsystem_id]
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
    width: Optional[int] = None,
    sizing_mode: Optional[str] = "stretch_width",
    tools: str = "xpan,xwheel_zoom,reset,save",
    name: Optional[str] = RANKING_PLOT,
    zero_line: bool = True,
):
    """A figure with the ranking's rows, shading, frames, zero lines and both axes.

    It has no y-axis: the names panel (:func:`names_panel`) names the rows.
    """
    y_range = FactorRange(factors=rows.y_range_factors, range_padding=0.0)
    kwargs = dict(
        title=title,
        x_range=x_range,
        y_range=y_range,
        height=ROW_HEIGHT * len(rows.factors) + 110,
        tools=tools,
        toolbar_location="above",
        x_axis_location="above",
        y_axis_location=None,
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
    frame_categories(plot, rows, zero_line=zero_line)
    # The title names the top axis; the bottom one, out of the title's sight
    # once the column is scrolled, carries the label.
    plot.add_layout(LinearAxis(axis_label=x_label), "below")
    plot.ygrid.grid_line_color = None
    for axis in plot.xaxis:
        axis.ticker.desired_num_ticks = 4
    return plot


def shade_rows(plot, rows: RankingRows, x_range: Range1d) -> None:
    """Every other row of a category on the sheet grey, from its second."""
    span = (x_range.start - 1e6, x_range.end + 1e6)
    striped = [
        factor
        for members in rows.members.values()
        for index, factor in enumerate(members)
        if index % 2 == 1
    ]
    if striped:
        plot.hbar(
            y=striped,
            left=span[0],
            right=span[1],
            height=1.0,
            fill_color=ROW_SHADE,
            line_color=None,
            level="underlay",
        )


def frame_categories(plot, rows: RankingRows, *, zero_line: bool = True) -> None:
    """A frame around each category's rows, as its circle in the network, and its zero line.

    The zero line runs inside the frames only, so it stops at the headings.
    """
    zero = {"x": [], "bottom": [], "top": []}
    for category in rows.members:
        bottom, top = rows.edges(category)
        plot.add_layout(
            BoxAnnotation(
                bottom=bottom,
                top=top,
                fill_alpha=0,
                line_color=FRAME_LINE,
                line_width=FRAME_LINE_WIDTH,
                level="underlay",
                editable=False,
            )
        )
        zero["x"].append(0.0)
        zero["bottom"].append(bottom)
        zero["top"].append(top)
    if not zero_line:
        return
    plot.segment(
        x0="x",
        x1="x",
        y0="bottom",
        y1="top",
        source=ColumnDataSource(zero),
        line_color=theme.THRESHOLD_LINE_COLOR,
        line_dash=theme.THRESHOLD_LINE_DASH,
        line_width=theme.THRESHOLD_LINE_WIDTH,
        level="underlay",
    )


def names_panel(
    rows: RankingRows,
    plot,
    scores: Optional[Mapping[str, float]] = None,
    *,
    title: str,
    x_label: str,
    navigation_event: Optional[str] = None,
):
    """The row names and category headings, beside the dots, and each Delta score.

    The names never pan with the dots: they take a panel of their own, sharing the
    dots' rows. A summary score is on its own scale (a Delta score runs from -1 to
    +1 under the sign transform), so given ``scores`` the panel also marks it, on
    its own axis right of the names, rather than as a second axis over the dots.
    Without them, the axes stay for the space they take, so the two panels' rows
    line up, but draw nothing.
    """
    width = panel_width(rows.names_width, scores is not None)
    if scores is not None:
        reach = max([1.0, *(abs(float(v)) for v in scores.values())]) * 1.08
        names = width - SUMMARY_FRAME_WIDTH
        start = -reach - 2 * reach * names / SUMMARY_FRAME_WIDTH
        x_range = Range1d(start, reach)
        names_right = -reach
    else:
        x_range = Range1d(0, width)
        start = 0.0
        names_right = float(width)

    panel = ranking_figure(
        rows,
        title=title,
        x_range=x_range,
        x_label=x_label,
        width=width,
        sizing_mode=None,
        tools="save",
        name=RANKING_NAMES,
        zero_line=scores is not None,
    )
    panel.min_border_left = 0
    panel.min_border_right = 0
    panel.toolbar_location = None
    panel.y_range = plot.y_range
    if scores is None:
        for axis in panel.xaxis:
            axis.ticker = FixedTicker(ticks=[width / 2])
            axis.major_label_text_alpha = 0
            axis.major_tick_line_color = None
            axis.axis_line_color = None
            axis.axis_label_text_alpha = 0
        panel.xgrid.grid_line_color = None
    else:
        reach_tick = round(reach / 1.08, 2)
        for axis in panel.xaxis:
            axis.ticker = FixedTicker(ticks=[-reach_tick, 0.0, reach_tick])
        # Zero is each frame's dashed line; a grid line there would cover it.
        panel.xgrid.ticker = FixedTicker(ticks=[-reach_tick, reach_tick])

    matched = plot.select_one({"name": MATCHED_ROWS})
    focused = plot.select_one({"name": FOCUSED_ROWS})
    if matched is not None and focused is not None:
        highlight_rows(panel, matched, focused)
    if rows.factor_of:
        _row_bands(panel, rows, navigation_event)

    font = value(theme.FONT)
    panel.text(
        x=names_right,
        y="y",
        text="text",
        source=ColumnDataSource(
            {"y": rows.subsystems, "text": [rows.text_of[f] for f in rows.subsystems]}
        ),
        x_offset=-8,
        text_align="right",
        text_baseline="middle",
        text_font=font,
        text_font_size=NAME_FONT_SIZE,
        text_color=theme.INK,
        text_line_height=1.1,
    )
    # A two-line heading is centred on the line between its two rows.
    anchors = [
        heading.factors[0] if len(heading.factors) == 1 else (heading.factors[0], -0.5)
        for heading in rows.headings
    ]
    headings = ColumnDataSource(
        {
            "y": anchors,
            "text": [heading.text for heading in rows.headings],
            "number": [str(heading.number) for heading in rows.headings],
        }
    )
    panel.text(
        x=start,
        y="y",
        text="text",
        source=headings,
        x_offset=BADGE_WIDTH,
        text_align="left",
        text_baseline="middle",
        text_font=font,
        text_font_size=HEADING_FONT_SIZE,
        text_font_style="bold",
        text_color=theme.INK,
        text_line_height=1.1,
    )
    # The category's number, as the network draws it on the category's rim.
    panel.text(
        x=start,
        y="y",
        text="number",
        source=headings,
        x_offset=BADGE_WIDTH / 2,
        text_align="center",
        text_baseline="middle",
        text_font=font,
        text_font_size="10px",
        text_font_style="bold",
        text_color="#ffffff",
        outline_shape="circle",
        background_fill_color=theme.INK,
        padding=2,
    )

    if scores is not None:
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
        panel.add_tools(
            HoverTool(renderers=[diamonds], tooltips=[("", "@name"), (title, "@score{0.00}")])
        )
    return panel


def add_focus_link(
    plot, rows: RankingRows, *, navigation_event: Optional[str] = None
) -> ColumnDataSource:
    """Highlight the subsystem in focus, and the matches of a search, in this ranking.

    Pointing at a row dispatches :data:`FOCUS_EVENT`, and tapping it dispatches
    ``navigation_event`` when given; the page writes the shared selection into the
    returned source, and the bands follow it.
    """
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
    _row_bands(plot, rows, navigation_event)
    attach(plot, focus)
    return focus


def _row_bands(plot, rows: RankingRows, navigation_event: Optional[str]) -> None:
    """An invisible band per row, for pointing at and tapping anywhere along it."""
    span = (plot.x_range.start - 1e6, plot.x_range.end + 1e6)
    ids = list(rows.factor_of)
    factors = [rows.factor_of[sid] for sid in ids]
    hits = ColumnDataSource({"y": factors, "id": ids, "label": [rows.name_of[sid] for sid in ids]})
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


def ranking_layout(names, plot):
    """The names panel and the dots side by side, sharing their rows.

    One toolbar serves both, and its save button exports the whole grid, names
    included: BokehJS's SaveTool exports the view that owns its toolbar.
    """
    from bokeh.layouts import gridplot

    grid = gridplot(
        [[names, plot]],
        merge_tools=True,
        toolbar_location="above",
        sizing_mode="stretch_width",
    )
    # The names keep their width; the dots take the rest.
    grid.cols = [f"{names.width}px", "1fr"]
    return grid


__all__ = [
    "FOCUS_EVENT",
    "FOCUS_SOURCE",
    "Heading",
    "RANKING_DOTS",
    "RANKING_NAMES",
    "RANKING_PLOT",
    "RANKING_ROWS_SOURCE",
    "RankedSubsystem",
    "RankingRows",
    "add_focus_link",
    "frame_categories",
    "names_panel",
    "order_rows",
    "ranking_figure",
    "ranking_layout",
    "shade_rows",
    "spread_offset",
]

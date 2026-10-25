"""The subsystem rankings' shared layout, and the link they share with the network.

A ranking lists every subsystem in its own row, grouped by category, as a forest
plot lists its studies:

* **Names in a panel of their own.** Each row's name is drawn right-aligned in a
  panel of its own (:func:`names_panel`), which never pans with the data.
* **Two parts beside them.** A summary score panel (:func:`score_panel`) marks each
  subsystem's own score, such as its Delta score, as a diamond; an elements panel
  spreads the row's reactions or genes as dots. The page shows either or both
  (:func:`ranking_layout`), and the panels shown always fill the ranking's width.
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
  wherever the column is scrolled; each panel's title stands over its zero.

The network and the ranking also share one selection. Each figure holds a data
source named :data:`FOCUS_SOURCE` that the page writes (the subsystem in focus,
the matches of a search, and a selected category), and redraws its own highlight
from it. Pointing at a subsystem in either figure dispatches :data:`FOCUS_EVENT`
on ``window`` for the page to relay to both. The row in focus, and a selected
category's rows, are outlined in the logo's gradient across the panels shown,
never filled, and a selected category's number takes the logo's hexagon, as in
the network.
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
from .brand import GRADIENT_JS, OUTLINE_WIDTH, SELECTED_HEXAGON, SELECTED_HEXAGON_SIZE
from .metabolic_visualization.generation.graph_building.spaced_network import (
    category_numbers,
    category_sort_key,
)
from .plot_utils import apply_academic_style
from .subsystem_link import (
    FOCUS_EVENT,
    FOCUS_SOURCE,
    attach,
    focus_dispatch,
    focus_source,
    navigation_dispatch,
)

# The page reads these to find a subsystem's row (web/src/bokeh/subsystemLink.ts).
RANKING_ROWS_SOURCE = "metabis-ranking-rows"
MATCHED_ROWS = "metabis-ranking-matched"
FOCUSED_ROWS = "metabis-ranking-focused"
RANKING_PLOT = "metabis-ranking"
RANKING_DOTS = "metabis-ranking-dots"
RANKING_NAMES = "metabis-ranking-names"
# Each category's first heading row, for the page to scroll a selected category to.
RANKING_HEADINGS_SOURCE = "metabis-ranking-headings"
RANKING_SCORE = "metabis-ranking-score"
# Which of the score and elements panels the page shows: {"score": [bool], "elements": [bool]}.
RANKING_PARTS = "metabis-ranking-parts"
# The category the shared selection names, which the outline and headings follow.
CHOSEN_CATEGORY = "metabis-ranking-category"
FRAME_PREFIX = "metabis-frame:"

# Beside the network the ranking has 450 to 700 px: names wrap onto a second line
# past this many characters, so the name column stays near 170 px, and every row
# is tall enough for two lines of 11 px text.
NAME_LENGTH = 28
ROW_HEIGHT = 30
ROW_SHADE = theme.ROW_SHADE  # the client's --sheet, about DataTables' stripe
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
HEXAGON_SIZE = 19  # pixels across a heading's hexagon, the network's category mark
SCORE_WIDTH = 120  # the score panel beside the elements; alone it takes the width
# The score axis reaches past the largest score by this share, so a diamond at
# either end is drawn whole inside the frame.
SCORE_REACH = 1.25
DIAMOND_EDGE = "#000000"
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


def panel_width(names_width: int) -> int:
    """The names panel's width: the longest name line, or room for a heading."""
    return max(names_width, NAMES_MIN_WIDTH)


def order_rows(
    entries: Iterable[RankedSubsystem],
    *,
    numbers: Optional[Mapping[str, int]] = None,
) -> RankingRows:
    """Categories in the network's order; within one, the strongest score first.

    ``numbers`` are the network's category numbers (:func:`category_numbers` over
    every category in use), so a ranking that lists only some categories keeps
    their numbers; without them the categories listed are numbered.
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
    heading_room = panel_width(rows.names_width) - BADGE_WIDTH - 8
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
                name=FRAME_PREFIX + category,
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
    *,
    title: str = "",
    x_label: str = "",
    navigation_event: Optional[str] = None,
):
    """The row names and category headings, sharing ``plot``'s rows.

    The names never pan with the data: they take a panel of their own. Its axes
    stay for the space they take, so the panels' rows line up, but draw nothing.
    """
    width = panel_width(rows.names_width)
    x_range = Range1d(0, width)
    panel = ranking_figure(
        rows,
        title=title,
        x_range=x_range,
        x_label=x_label,
        width=width,
        sizing_mode=None,
        tools="save",
        name=RANKING_NAMES,
        zero_line=False,
    )
    panel.min_border_left = 0
    panel.min_border_right = 0
    panel.toolbar_location = None
    panel.y_range = plot.y_range
    for axis in panel.xaxis:
        axis.ticker = FixedTicker(ticks=[width / 2])
        axis.major_label_text_alpha = 0
        axis.major_tick_line_color = None
        axis.axis_line_color = None
        axis.axis_label_text_alpha = 0
    panel.xgrid.grid_line_color = None
    _share_highlights(panel, plot, rows, navigation_event)

    font = value(theme.FONT)
    panel.text(
        x=float(width),
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
            "category": [heading.category for heading in rows.headings],
            "x": [BADGE_WIDTH / 2] * len(anchors),
            "alpha": [1.0] * len(anchors),
        }
    )
    panel.text(
        x=0.0,
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
    # The category's number in the hexagon the network draws on the category's
    # rim, as the side menu numbers the steps; once selected, in the logo's.
    panel.scatter(
        x="x",
        y="y",
        source=headings,
        marker="hex",
        angle=math.pi / 6,
        size=HEXAGON_SIZE,
        fill_color=theme.INK_NODE,
        fill_alpha="alpha",
        line_color=theme.SELECTED,
        line_alpha="alpha",
        line_width=1.5,
    )
    badge = ColumnDataSource({"url": [], "x": [], "y": []})
    panel.image_url(
        url="url",
        x="x",
        y="y",
        w=SELECTED_HEXAGON_SIZE,
        h=SELECTED_HEXAGON_SIZE,
        w_units="screen",
        h_units="screen",
        anchor="center",
        source=badge,
    )
    panel.text(
        x="x",
        y="y",
        text="number",
        source=headings,
        text_align="center",
        text_baseline="middle",
        text_font=font,
        text_font_size="10px",
        text_font_style="bold",
        text_color="#ffffff",
    )
    chosen = plot.select_one({"name": CHOSEN_CATEGORY})
    if chosen is not None:
        follow_category(chosen, headings, badge)
    return panel


def score_panel(
    rows: RankingRows,
    plot,
    scores: Mapping[str, float],
    *,
    title: str,
    x_label: str,
    colors: Optional[Mapping[str, str]] = None,
    navigation_event: Optional[str] = None,
    x_range: Optional[Range1d] = None,
):
    """Each subsystem's own score as a diamond, on an axis of its own, sharing ``plot``'s rows.

    A summary score is on its own scale (a Delta score runs from -1 to +1 under
    the sign transform), so it takes a panel of its own rather than a second axis
    over the elements. Each diamond has a black edge and is filled with
    ``colors`` (its node's colour in the network) where given. By default the axis
    is symmetric and reaches past the largest score, so a diamond at either end
    is drawn whole; ``x_range`` sets another, such as the members' own.
    """
    largest = max([1.0, *(abs(float(v)) for v in scores.values())])
    reach = largest * SCORE_REACH
    panel = ranking_figure(
        rows,
        title=title,
        x_range=x_range if x_range is not None else Range1d(-reach, reach),
        x_label=x_label,
        width=SCORE_WIDTH,
        sizing_mode="stretch_width",
        tools="save",
        name=RANKING_SCORE,
    )
    panel.toolbar_location = None
    panel.y_range = plot.y_range
    if x_range is None:
        tick = round(largest, 2)
        for axis in panel.xaxis:
            axis.ticker = FixedTicker(ticks=[-tick, 0.0, tick])
        # Zero is each frame's dashed line; a grid line there would cover it.
        panel.xgrid.ticker = FixedTicker(ticks=[-tick, tick])
    _share_highlights(panel, plot, rows, navigation_event)
    ids = [sid for sid in rows.factor_of if sid in scores]
    diamonds = panel.scatter(
        x="score",
        y="y",
        source=ColumnDataSource(
            {
                "score": [float(scores[sid]) for sid in ids],
                "y": [rows.factor_of[sid] for sid in ids],
                "name": [rows.name_of.get(sid, rows.factor_of[sid]) for sid in ids],
                "color": [(colors or {}).get(sid) or SUMMARY_COLOR for sid in ids],
            }
        ),
        marker="diamond",
        size=13,
        fill_color="color",
        line_color=DIAMOND_EDGE,
        line_width=1,
    )
    panel.add_tools(
        HoverTool(renderers=[diamonds], tooltips=[("", "@name"), (title, "@score{0.00}")])
    )
    center_title_on_zero(panel)
    return panel


def _share_highlights(panel, plot, rows: RankingRows, navigation_event: Optional[str]) -> None:
    """A panel beside ``plot`` bands a search's matches as ``plot`` does, and takes its pointing."""
    matched = plot.select_one({"name": MATCHED_ROWS})
    if matched is not None:
        highlight_rows(panel, matched)
    if rows.factor_of:
        _row_bands(panel, rows, navigation_event)


def center_title_on_zero(plot) -> None:
    """Stand the title over the axis' zero, as near as the frame allows.

    Bokeh ignores a centred title's offset, so the title stays left-aligned and
    moves by its own measured width.
    """
    plot.title.align = "left"
    callback = CustomJS(
        args=dict(plot=plot),
        code="""
        let width
        try { width = plot.inner_width } catch { return }
        const {start, end} = plot.x_range
        if (!(width > 0) || end === start) return
        const title = plot.title
        const context = document.createElement('canvas').getContext('2d')
        context.font = `${title.text_font_style} ${title.text_font_size} ${title.text_font}`
        const text = context.measureText(title.text).width
        const zero = (0 - start) / (end - start) * width
        title.offset = Math.round(Math.min(Math.max(zero - text / 2, 0), Math.max(width - text, 0)))
        """,
    )
    for attribute in ("start", "end"):
        plot.x_range.js_on_change(attribute, callback)
    plot.js_on_change("inner_width", callback)


def add_focus_link(
    plot, rows: RankingRows, *, navigation_event: Optional[str] = None
) -> ColumnDataSource:
    """Highlight the subsystem in focus, and the matches of a search, in this ranking.

    Pointing at a row dispatches :data:`FOCUS_EVENT`, and tapping it dispatches
    ``navigation_event`` when given; the page writes the shared selection into the
    returned source, and the bands and outlines follow it.
    """
    ids = list(rows.factor_of)
    factors = [rows.factor_of[sid] for sid in ids]
    lookup = ColumnDataSource({"id": ids, "factor": factors}, name=RANKING_ROWS_SOURCE)
    headings = ColumnDataSource(
        {
            "category": [heading.category for heading in rows.headings],
            "factor": [heading.factors[0] for heading in rows.headings],
        },
        name=RANKING_HEADINGS_SOURCE,
    )
    focus = focus_source()
    matched = ColumnDataSource({"y": []}, name=MATCHED_ROWS)
    focused = ColumnDataSource({"y": []}, name=FOCUSED_ROWS)
    chosen = ColumnDataSource({"category": []}, name=CHOSEN_CATEGORY)
    highlight_rows(plot, matched)
    focus.js_on_change(
        "data",
        CustomJS(
            args=dict(focus=focus, lookup=lookup, matched=matched, focused=focused, chosen=chosen),
            code="""
            const factor = new Map(lookup.data.id.map((id, i) => [id, lookup.data.factor[i]]))
            const pick = (kind) => focus.data.id
              .filter((id, i) => focus.data.kind[i] === kind && factor.has(id))
              .map((id) => factor.get(id))
            matched.data = {y: pick('match')}
            focused.data = {y: pick('focus')}
            const category = focus.data.id.filter((id, i) => focus.data.kind[i] === 'category')
            if (String(category) !== String(chosen.data.category)) chosen.data = {category}
            """,
        ),
    )
    _row_bands(plot, rows, navigation_event)
    attach(plot, focus)
    attach(plot, lookup)
    attach(plot, headings)
    attach(plot, chosen)
    return focus


def follow_category(chosen: ColumnDataSource, headings: ColumnDataSource, badge: ColumnDataSource) -> None:
    """Give the chosen category's heading the logo's hexagon, as the network gives its number."""
    chosen.js_on_change(
        "data",
        CustomJS(
            args=dict(chosen=chosen, headings=headings, badge=badge, hexagon=SELECTED_HEXAGON),
            code="""
            const category = chosen.data.category[0] ?? null
            const h = headings.data
            h.alpha = h.category.map((name) => name === category ? 0 : 1)
            headings.change.emit()
            const i = h.category.indexOf(category)
            badge.data = i < 0 ? {url: [], x: [], y: []} : {url: [hexagon], x: [h.x[i]], y: [h.y[i]]}
            """,
        ),
    )


def outline_selection(panels: List, rows: RankingRows, chosen: ColumnDataSource, focused: ColumnDataSource) -> None:
    """Outline the row in focus, and a selected category's rows, in the logo's gradient.

    Each shown panel draws its part of the outline, a box round the rows; the
    gradient runs from the first shown panel's left to the last one's right, as if
    they were one.
    """
    sources = []
    for panel in panels:
        source = ColumnDataSource({key: [] for key in ("x0", "y0", "x1", "y1", "color")})
        panel.segment(
            x0="x0",
            y0="y0",
            x1="x1",
            y1="y1",
            line_color="color",
            line_width=OUTLINE_WIDTH,
            line_cap="round",
            level="annotation",
            source=source,
        )
        sources.append(source)
    edges = {category: list(rows.edges(category)) for category in rows.members}
    callback = CustomJS(
        args=dict(panels=panels, sources=sources, chosen=chosen, focused=focused, edges=edges, edge=OUTLINE_WIDTH),
        code=GRADIENT_JS
        + """
        const shown = panels.map((panel) => panel.visible !== false)
        let widths
        try { widths = panels.map((panel, k) => shown[k] ? panel.inner_width : 0) } catch { return }
        const total = widths.reduce((a, b) => a + b, 0)
        if (!(total > 0)) return
        const boxes = []
        const category = chosen.data.category[0]
        if (category != null && edges[category]) boxes.push(edges[category])
        const factors = panels[0].y_range.factors
        for (const factor of focused.data.y) {
          const i = factors.indexOf(factor)
          if (i >= 0) boxes.push([i, i + 1])
        }
        let offset = 0
        panels.forEach((panel, k) => {
          const out = {x0: [], y0: [], x1: [], y1: [], color: []}
          const width = widths[k]
          if (shown[k] && width > 0 && boxes.length) {
            const start = Math.min(panel.x_range.start, panel.x_range.end)
            const end = Math.max(panel.x_range.start, panel.x_range.end)
            // Inset by half the outline, so the frame does not cut its sides.
            const inset = edge / 2 * (end - start) / width
            const left = start + inset
            const right = end - inset
            const at = (f) => metabisGradient((offset + f * width) / total)
            const pieces = 16
            for (const [bottom, top] of boxes) {
              for (const y of [bottom, top]) {
                for (let i = 0; i < pieces; i++) {
                  out.x0.push(left + (right - left) * i / pieces)
                  out.x1.push(left + (right - left) * (i + 1) / pieces)
                  out.y0.push(y)
                  out.y1.push(y)
                  out.color.push(at((i + 0.5) / pieces))
                }
              }
              out.x0.push(left, right)
              out.x1.push(left, right)
              out.y0.push(bottom, bottom)
              out.y1.push(top, top)
              out.color.push(at(0), at(1))
            }
          }
          if (shown[k]) offset += width
          sources[k].data = out
        })
        """,
    )
    chosen.js_on_change("data", callback)
    focused.js_on_change("data", callback)
    for panel in panels:
        panel.js_on_change("inner_width", callback)
        panel.js_on_change("visible", callback)
        for attribute in ("start", "end"):
            panel.x_range.js_on_change(attribute, callback)


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


def highlight_rows(plot, matched: ColumnDataSource) -> None:
    """Bands behind a search's matches, across ``plot``, in GOV.UK's yellow."""
    span = (plot.x_range.start - 1e6, plot.x_range.end + 1e6)
    plot.hbar(
        y="y",
        left=span[0],
        right=span[1],
        height=1.0,
        fill_color=MATCH_SHADE,
        line_color=None,
        level="underlay",
        source=matched,
    )


def ranking_layout(rows: RankingRows, names, plot, score=None):
    """The names panel, the score panel when there is one, and the elements, sharing their rows.

    One toolbar serves them all, and its save button exports the whole grid, names
    included: BokehJS's SaveTool exports the view that owns its toolbar. With a
    score panel, the page shows the score, the elements, or both, by writing
    :data:`RANKING_PARTS`; the panels shown always fill the width, the elements
    taking what the names and a narrow score leave, or the score all of it.
    """
    from bokeh.layouts import gridplot

    panels = [names, *([score] if score is not None else []), plot]
    grid = gridplot(
        [panels],
        merge_tools=True,
        toolbar_location="above",
        sizing_mode="stretch_width",
    )
    # The names keep their width; the elements take the rest.
    grid.cols = [f"{names.width}px", *([f"{SCORE_WIDTH}px"] if score is not None else []), "1fr"]
    center_title_on_zero(plot)
    chosen = plot.select_one({"name": CHOSEN_CATEGORY})
    focused = plot.select_one({"name": FOCUSED_ROWS})
    if chosen is not None and focused is not None:
        outline_selection(panels, rows, chosen, focused)
    if score is not None:
        parts = ColumnDataSource({"score": [True], "elements": [True]}, name=RANKING_PARTS)
        parts.js_on_change(
            "data",
            CustomJS(
                args=dict(parts=parts, grid=grid, names=names, score=score, elements=plot, width=SCORE_WIDTH),
                code="""
                const showScore = parts.data.score[0] !== false
                // One part is always shown.
                const showElements = parts.data.elements[0] !== false || !showScore
                score.visible = showScore
                elements.visible = showElements
                grid.cols = [
                  `${names.width}px`,
                  showScore ? (showElements ? `${width}px` : '1fr') : '0px',
                  showElements ? '1fr' : '0px',
                ]
                """,
            ),
        )
        attach(plot, parts)
    return grid


__all__ = [
    "FOCUS_EVENT",
    "FOCUS_SOURCE",
    "Heading",
    "CHOSEN_CATEGORY",
    "RANKING_DOTS",
    "RANKING_HEADINGS_SOURCE",
    "RANKING_NAMES",
    "RANKING_PARTS",
    "RANKING_PLOT",
    "RANKING_ROWS_SOURCE",
    "RANKING_SCORE",
    "RankedSubsystem",
    "RankingRows",
    "add_focus_link",
    "center_title_on_zero",
    "follow_category",
    "frame_categories",
    "names_panel",
    "order_rows",
    "outline_selection",
    "ranking_figure",
    "ranking_layout",
    "score_panel",
    "shade_rows",
    "spread_offset",
]

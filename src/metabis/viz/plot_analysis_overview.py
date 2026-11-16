"""An analysis of three groups or more at a glance: each subsystem in each comparison.

Rows are subsystems, grouped by category as step 4's ranking groups them
(:mod:`.ranking_layout`), without the network's numbers, or features, grouped by
the group each is highest in; columns are the
comparisons, each group against the rest of the chosen, in one family. Each
column's name reads upward above it, as ComplexHeatmap prints a heatmap's column
names, and a band carries the group's colour. The page reads the figure as a way
into one comparison: a tap anywhere in a column puts its comparison in view, and
the page names the column in view through a source the figure redraws its
outline from, in the logo's gradient. The rows are read, never selected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape
from typing import Literal, Mapping, Optional, Sequence

from bokeh.models import (
    ColumnDataSource,
    CustomJS,
    FixedTicker,
    HoverTool,
    Range1d,
    TapTool,
)

from ..managers.hover import panel as hover_panel
from ..visual_config import theme
from .brand import GRADIENT_JS, OUTLINE_WIDTH
from .plot_comparison_ranking import FAMILY_NAMES, _diverging
from .ranking_layout import (
    NAME_CHARACTER_WIDTH,
    RankedSubsystem,
    names_panel,
    order_rows,
    ranking_figure,
    ranking_layout,
)
from .subsystem_link import attach

__all__ = [
    "OVERVIEW_COLUMNS",
    "OverviewCell",
    "overview_events",
    "plot_analysis_overview",
]

# The column in view, which the page writes: {"group": [...]}.
OVERVIEW_COLUMNS = "metabis-overview-columns"
BAND = "band:"
BAND_HEIGHT = 0.55
CELL_WIDTH = 0.9
CELL_HEIGHT = 0.82
# The outline stays this share of a row inside the frame, so the frame never cuts it.
OUTLINE_INSET = 0.08


@dataclass(frozen=True)
class OverviewCell:
    """One subsystem, or one feature, in one comparison.

    ``value`` is its Delta score, its signed log2 enrichment, or a feature's
    Cohen's d; ``None`` with ``measured`` True is a subsystem not
    over-represented or a feature not significant (grey), and with ``measured``
    False one with no features of the family (white, dashed).
    ``hover`` is what its hover says of it, as the network's says it, in place
    of its reading and ``detail``.
    """

    value: Optional[float]
    detail: str = ""
    measured: bool = True
    hover: str = ""


# What a row is, by kind: a subsystem, or for "effect" a reaction or a gene.
ROW_WORDS = {"delta": "Subsystem", "ora": "Subsystem"}


def overview_events(page: str) -> str:
    """The window event a tap in a column dispatches, with the column's group."""
    return f"metabis:{page}-column"


def plot_analysis_overview(
    entries: Sequence[RankedSubsystem],
    groups: Sequence[str],
    cells: Mapping[str, Mapping[str, OverviewCell]],
    *,
    family: Literal["reaction", "gene"] = "reaction",
    kind: Literal["delta", "ora", "effect"] = "delta",
    group_colors: Optional[Mapping[str, str]] = None,
    against: Optional[Mapping[str, str]] = None,
    notes: Optional[Mapping[str, str]] = None,
    category_numbers: Optional[Mapping[str, int]] = None,
    page: str = "overview",
    title: Optional[str] = None,
    category_order: Optional[Sequence[str]] = None,
):
    """Rows of ``entries``, a column per group of ``groups``, from ``cells``.

    ``cells`` maps each group to its cells, by subsystem, in ``family``.
    ``against`` names what each group is set against, and ``notes`` add a line
    of each group's own, such as how much is significant, to its hovers.
    ``kind`` "effect" draws features' Cohen's d, its scale reaching the largest;
    ``category_order`` then orders their headings, such as the groups'.
    """
    if not groups:
        raise ValueError("The overview needs a comparison")
    column_event = overview_events(page)
    rows = order_rows(entries, numbers=category_numbers, numbered=False, order=category_order)
    if not rows.factor_of:
        raise ValueError("No subsystem to show")
    rows.factors.insert(0, BAND)
    values = [abs(cell.value) for column in cells.values() for cell in column.values() if cell.value is not None]
    reach = 1.0 if kind == "delta" else max([1.0, *values])
    label = {"delta": "Delta score", "ora": "Signed log2 enrichment", "effect": "Cohen's d"}[kind]
    heading = {"delta": "Delta score", "ora": "over-represented subsystems", "effect": "Cohen's d, varying most"}[kind]
    plot = ranking_figure(
        rows,
        title=title or f"{FAMILY_NAMES[family]}: {heading}",
        x_range=Range1d(0, len(groups)),
        x_label="",
        tools="save",
        zero_line=False,
    )
    plot.xgrid.grid_line_color = None
    centres = {group: k + 0.5 for k, group in enumerate(groups)}
    top, bottom = plot.xaxis
    top.ticker = FixedTicker(ticks=list(centres.values()))
    top.major_label_overrides = {centre: str(group) for group, centre in centres.items()}
    top.major_label_orientation = math.pi / 2
    top.major_tick_line_color = None
    top.major_label_text_font_size = "11px"
    bottom.visible = False

    against = dict(against or {})
    notes = dict(notes or {})

    def about(group: str) -> str:
        return (hover_panel.note(escape(f"{group} against {against[group]}")) if group in against else "") + (
            hover_panel.note(escape(notes[group])) if group in notes else ""
        )

    data = {key: [] for key in ("x", "y", "color", "line", "dash", "hover")}
    for group in groups:
        column = cells.get(group, {})
        for sid in rows.factor_of:
            cell = column.get(sid)
            if cell is None:
                continue
            missing = cell.value is None and not cell.measured
            data["x"].append(centres[group])
            data["y"].append(rows.factor_of[sid])
            data["color"].append(
                "#ffffff" if missing else theme.NOT_SIGNIFICANT if cell.value is None else _diverging(cell.value, reach)
            )
            data["line"].append(theme.INK_MUTED if missing else None)
            data["dash"].append("dashed" if missing else "solid")
            reading = (
                f"no {FAMILY_NAMES[family].lower()}"
                if missing
                else ("not significant" if kind == "effect" else "not over-represented")
                if cell.value is None
                else f"{cell.value:+.2f}"
            )
            data["hover"].append(
                hover_panel.row("Group", escape(group)) + hover_panel.inner(cell.hover) + about(group)
                if cell.hover
                else hover_panel.row(ROW_WORDS.get(kind, FAMILY_NAMES[family][:-1]), escape(rows.name_of[sid]))
                + (hover_panel.row("Category", escape(rows.category_of[sid])) if kind != "effect" else "")
                + hover_panel.row("Group", escape(group))
                + hover_panel.row(label, escape(reading))
                + (hover_panel.note(escape(cell.detail)) if cell.detail else "")
                + about(group)
            )
    squares = plot.rect(
        x="x",
        y="y",
        width=CELL_WIDTH,
        height=CELL_HEIGHT,
        source=ColumnDataSource(data),
        fill_color="color",
        line_color="line",
        line_dash="dash",
        line_width=1,
    )
    plot.add_tools(HoverTool(renderers=[squares], tooltips=hover_panel.panel("@hover{safe}")))

    colors = dict(group_colors or {})
    band = ColumnDataSource(
        {
            "x": list(centres.values()),
            "y": [BAND] * len(groups),
            "color": [colors.get(group, theme.INK_MUTED) for group in groups],
            "hover": [hover_panel.row("Group", escape(group)) + about(group) for group in groups],
        }
    )
    stripes = plot.rect(x="x", y="y", width=CELL_WIDTH, height=BAND_HEIGHT, source=band, fill_color="color", line_color=None)
    plot.add_tools(HoverTool(renderers=[stripes], tooltips=hover_panel.panel("@hover{safe}")))

    # Each column, band and cells, is one target to tap, drawn invisible under them.
    height = float(len(rows.factors))
    hits = ColumnDataSource({"left": [c - 0.5 for c in centres.values()], "right": [c + 0.5 for c in centres.values()], "group": list(groups)})
    columns_hit = plot.quad(
        left="left",
        right="right",
        bottom=0.0,
        top=height,
        source=hits,
        fill_alpha=0,
        line_alpha=0,
        selection_fill_alpha=0,
        nonselection_fill_alpha=0,
        nonselection_line_alpha=0,
        level="underlay",
    )
    plot.add_tools(TapTool(renderers=[columns_hit]))
    hits.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(source=hits, event=column_event),
            code="""
            const index = source.selected.indices[0]
            if (index === undefined) return
            source.selected.indices = []
            window.dispatchEvent(new CustomEvent(event, {detail: {group: source.data.group[index]}}))
            """,
        ),
    )

    # The column in view, outlined in the logo's gradient from its top to its bottom.
    outline = ColumnDataSource({key: [] for key in ("x0", "y0", "x1", "y1", "color")})
    plot.segment(
        x0="x0", y0="y0", x1="x1", y1="y1", line_color="color", line_width=OUTLINE_WIDTH, line_cap="round", level="annotation", source=outline
    )
    columns = ColumnDataSource({"group": []}, name=OVERVIEW_COLUMNS)
    columns.js_on_change(
        "data",
        CustomJS(
            args=dict(columns=columns, outline=outline, centres=centres, height=height, inset=OUTLINE_INSET),
            code=GRADIENT_JS
            + """
            const out = {x0: [], y0: [], x1: [], y1: [], color: []}
            const centre = centres[columns.data.group[0]]
            if (centre !== undefined) {
              const [left, right, bottom, top] = [centre - 0.5 + inset / 2, centre + 0.5 - inset / 2, inset, height - inset]
              const sides = [[left, top, right, top], [right, top, right, bottom], [right, bottom, left, bottom], [left, bottom, left, top]]
              const pieces = 64
              for (const [ax, ay, bx, by] of sides) {
                for (let i = 0; i < pieces; i++) {
                  const [x0, y0] = [ax + (bx - ax) * i / pieces, ay + (by - ay) * i / pieces]
                  const [x1, y1] = [ax + (bx - ax) * (i + 1) / pieces, ay + (by - ay) * (i + 1) / pieces]
                  out.x0.push(x0); out.y0.push(y0); out.x1.push(x1); out.y1.push(y1)
                  out.color.push(metabisGradient(((x0 + x1) / 2 - left + top - (y0 + y1) / 2) / (right - left + top - bottom)))
                }
              }
            }
            outline.data = out
            """,
        ),
    )
    attach(plot, columns)

    names = names_panel(rows, plot)
    # The names read upward take their length above the rows, which keep their height.
    plot.height += math.ceil(max(len(str(group)) for group in groups) * NAME_CHARACTER_WIDTH)
    names.height = plot.height
    return ranking_layout(rows, names, plot)

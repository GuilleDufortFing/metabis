"""One comparison's subsystems, one family's score in each row, for step 3.

Step 3 reads one comparison beside its volcano plot, one family at a time. Each
subsystem takes a row, grouped by category in the network's order as step 4's
ranking groups them (:mod:`.ranking_layout`), without the network's numbers,
which only mean something beside the network. The score is a diamond, as step
4's score panel draws a subsystem's own score. Rows and headings dispatch the
page's own events (:func:`.subsystem_link.link_events`), so the volcano plot can
follow the subsystem pointed at or kept. A row can carry the hover and the colour
its subsystem has in the network, and the colour reference its colours come from.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Literal, Mapping, Optional, Sequence

from bokeh.models import ColumnDataSource, FixedTicker, HoverTool, Range1d

from ..managers.color_distributions import gradient_palette
from ..managers.hover import panel as hover_panel
from ..visual_config import theme
from .color_reference import GradientColorReference
from .ranking_layout import (
    SCORE_REACH,
    RankedSubsystem,
    add_focus_link,
    names_panel,
    order_rows,
    ranking_figure,
    ranking_layout,
)
from .subsystem_link import LinkEvents

__all__ = [
    "RankingScore",
    "plot_comparison_ranking",
]

FAMILY_NAMES = {"reaction": "Reactions", "gene": "Genes"}
EDGE = "#000000"
# A subsystem not over-represented sits at zero, grey with a grey edge.
QUIET_EDGE = "#949494"
PALETTE_STEPS = 128


@dataclass(frozen=True)
class RankingScore:
    """A subsystem's score, and what its hover adds.

    ``value`` is a Delta score or a signed log2 enrichment; ``None`` for a
    subsystem not over-represented, drawn grey at zero. ``hover`` is the row's
    whole hover, in place of the facts and ``detail``; ``color`` its colour, in
    place of the score's on the diverging scale.
    """

    value: Optional[float]
    detail: str = ""
    hover: str = ""
    color: str = ""


def _diverging(value: float, reach: float) -> str:
    palette = gradient_palette(theme.DOWN_REGULATED, theme.NEUTRAL, PALETTE_STEPS)[:-1] + gradient_palette(
        theme.NEUTRAL, theme.UP_REGULATED, PALETTE_STEPS
    )
    share = (max(-reach, min(reach, value)) + reach) / (2 * reach)
    return palette[round(share * (len(palette) - 1))]


def plot_comparison_ranking(
    entries: Sequence[RankedSubsystem],
    scores: Mapping[str, RankingScore],
    *,
    kind: Literal["delta", "ora"] = "delta",
    category_numbers: Optional[Mapping[str, int]] = None,
    events: Optional[LinkEvents] = None,
    title: Optional[str] = None,
    width: Optional[int] = None,
    reference: Optional[GradientColorReference] = None,
):
    """The subsystems of ``entries`` in rows, each with its score from ``scores``.

    ``entries`` carry each row's strength in ``score``, which orders a category's
    rows; ``category_numbers`` order the categories as the network numbers them.
    ``scores`` holds each subsystem's score in the family shown; one without a
    score has no row. ``kind`` "delta" draws Delta scores, from -1 to +1; "ora" the signed
    log2 enrichment of the over-represented direction. ``reference`` is the
    distribution the rows' colours come from, marked with each row's score in
    its hover, as step 4's ORA ranking marks its own.
    """
    entries = [entry for entry in entries if entry.subsystem_id in scores]
    if not entries:
        raise ValueError("No subsystem has a score to rank")
    events = events or LinkEvents()
    rows = order_rows(entries, numbers=category_numbers, numbered=False)
    largest = max([1.0, *(abs(score.value) for score in scores.values() if score.value is not None)])
    reach = largest * SCORE_REACH
    label = "Delta score" if kind == "delta" else "Signed log2 enrichment"
    plot = ranking_figure(
        rows,
        # The family is the page's switch and the volcano plot's title.
        title=title or label,
        x_range=Range1d(-reach, reach),
        x_label=label,
        width=width,
        tools="save",
    )
    if kind == "delta":
        for axis in plot.xaxis:
            axis.ticker = FixedTicker(ticks=[-1.0, 0.0, 1.0])
        plot.xgrid.ticker = FixedTicker(ticks=[-1.0, 1.0])
    add_focus_link(plot, rows, navigation_event=events.navigate, events=events)

    ids = list(rows.factor_of)
    values = [scores[sid].value for sid in ids]
    source = ColumnDataSource(
        {
            "x": [0.0 if value is None else float(value) for value in values],
            "y": [rows.factor_of[sid] for sid in ids],
            "color": [
                scores[sid].color
                or (theme.NOT_SIGNIFICANT if value is None else _diverging(float(value), 1.0 if kind == "delta" else largest))
                for sid, value in zip(ids, values)
            ],
            "edge": [QUIET_EDGE if value is None else EDGE for value in values],
            "hover": [
                hover_panel.inner(scores[sid].hover)
                or hover_panel.row("Subsystem", escape(rows.name_of[sid]))
                + hover_panel.row("Category", escape(rows.category_of[sid]))
                + hover_panel.row(label, "not over-represented" if value is None else f"{float(value):+.2f}")
                + (hover_panel.note(escape(scores[sid].detail)) if scores[sid].detail else "")
                for sid, value in zip(ids, values)
            ],
        }
    )
    tooltip = hover_panel.panel("@hover{safe}")
    if reference is not None:
        source.data["color_reference_value"] = [0.0 if value is None else float(value) for value in values]
        source.data["color_reference_marker_x"] = [reference.marker_x(value) for value in source.data["color_reference_value"]]
        tooltip = hover_panel.panel(
            "@hover{safe}",
            reference.hover_template(marker_x_field="color_reference_marker_x", value_field="color_reference_value"),
        )
    diamonds = plot.scatter(
        x="x",
        y="y",
        source=source,
        marker="diamond",
        size=13,
        fill_color="color",
        line_color="edge",
        line_width=1,
    )
    plot.add_tools(HoverTool(renderers=[diamonds], tooltips=tooltip))

    names = names_panel(rows, plot, navigation_event=events.navigate, events=events)
    return ranking_layout(rows, names, plot)

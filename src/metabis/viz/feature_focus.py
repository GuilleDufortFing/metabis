"""The features of the subsystem in focus, drawn over the rest of a scatter.

Step 3's volcano plot follows the subsystem, or the category, pointed at or
kept in the comparison's ranking. It holds a source named :data:`.subsystem_link.FOCUS_SOURCE` that the page writes,
as step 4's figures hold one (:mod:`.subsystem_link`): ``kind`` "focus" names a
subsystem and "category" a category. Their features are drawn again over a
white veil that washes the others out in their own colours, as the groups
highlighted on step 2's map are drawn over the rest, and the names the plot writes
are theirs, the most significant first. The page can zoom to them by writing
:data:`FEATURE_ZOOM`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from bokeh.models import BoxAnnotation, ColumnDataSource, CustomJS

from ..visual_config import theme
from .subsystem_link import FOCUS_SOURCE, attach, focus_source

__all__ = ["FEATURE_ZOOM", "FocusLayer", "FocusNames", "add_feature_focus"]

FEATURE_ZOOM = "metabis-feature-zoom"
# How much of the rest the veil hides: enough that the features in focus stand
# out, little enough that the rest still shows where they sit.
VEIL_ALPHA = 0.78


@dataclass(frozen=True)
class FocusLayer:
    """One source of a scatter's points: its coordinates, its colour and its IDs.

    ``color`` names a column of colours, or is one colour for every point.
    """

    source: ColumnDataSource
    x: str
    y: str
    color: str
    id_field: str = "element_id"


@dataclass(frozen=True)
class FocusNames:
    """The names a scatter writes by its points, and how many it writes.

    ``labels`` is the source of its ``LabelSet``, which holds the names written
    with nothing in focus; ``name`` and ``significant`` name the layers' columns
    of names and of significance.
    """

    labels: ColumnDataSource
    count: int = 6
    name: str = "short_name"
    significant: str = "significant"


def add_feature_focus(
    plot,
    layers: Sequence[FocusLayer],
    members: Mapping[str, Sequence[str]],
    categories: Mapping[str, str],
    *,
    floor: Optional[float] = None,
    names: Optional[FocusNames] = None,
) -> ColumnDataSource:
    """Draw the features of the subsystems in focus over a veil, on the page's word.

    ``members`` maps each feature ID to the subsystems it counts in;
    ``categories`` each subsystem to its category. A zoom never takes the y axis
    below ``floor``, as -log10 p never is. With ``names``, the plot's names are
    those of the features in focus, the significant first, then the highest; one
    per reaction, whichever direction is drawn. Returns the focus source.
    """
    for layer in layers:
        layer.source.data["focus_subsystems"] = [
            list(members.get(str(feature), ())) for feature in layer.source.data[layer.id_field]
        ]
    focus = next((tag for tag in plot.tags if getattr(tag, "name", None) == FOCUS_SOURCE), None)
    if focus is None:
        focus = focus_source()
        attach(plot, focus)
    zoom = ColumnDataSource({"zoom": [False]}, name=FEATURE_ZOOM)
    attach(plot, zoom)

    veil = BoxAnnotation(fill_color="#ffffff", fill_alpha=VEIL_ALPHA, line_alpha=0, visible=False, editable=False)
    plot.add_layout(veil)
    chosen = ColumnDataSource({"x": [], "y": [], "color": []})
    plot.scatter(
        x="x",
        y="y",
        source=chosen,
        size=7,
        fill_color="color",
        line_color=theme.INK,
        line_width=0.8,
        level="overlay",
    )
    place = None
    if names is not None:
        # Each name beside its point, on whichever side and at whichever of a few
        # heights keeps it inside the frame, off the names placed before it and off
        # every point named or in focus, so no name hides a point and passes for
        # another's; a name with no such place is left out. Features at one point
        # share a name, the first's with how many more; a name lifted off its
        # point is joined to it by a line. Placed again as the view moves.
        leaders = ColumnDataSource({"x0": [], "y0": [], "x1": [], "y1": []})
        plot.segment(
            x0="x0", y0="y0", x1="x1", y1="y1", source=leaders, line_color=theme.INK, line_alpha=0.6, line_width=1, level="overlay"
        )
        place = CustomJS(
            args=dict(plot=plot, names=names.labels, unfocused=dict(names.labels.data), dots=chosen, leaders=leaders),
            code="""
            // The names' own font (the LabelSet's 11px Helvetica), measured.
            const context = (window.__metabisNameCanvas ??= document.createElement('canvas').getContext('2d'))
            context.font = '11px helvetica'
            const measure = (text) => context.measureText(text).width
            const run = (tries) => {
              const chosen = (window.__metabisFocusNames ?? {})[names.id]
              const candidates = chosen ?? unfocused.text.map((text, i) => ({x: unfocused.x[i], y: unfocused.y[i], text}))
              const xr = plot.x_range
              const yr = plot.y_range
              // Unset, and so throwing, until the plot is first laid out.
              const size = (name) => { try { return plot[name] } catch { return 0 } }
              const width = size('inner_width')
              const height = size('inner_height')
              if (!(width > 0 && height > 0 && xr.end > xr.start && yr.end > yr.start)) {
                // The ranges come before the plot's size on its first render: wait for it.
                if (!chosen) names.data = unfocused
                if (tries < 300) requestAnimationFrame(() => run(tries + 1))
                return
              }
              const px = (x) => ((x - xr.start) / (xr.end - xr.start)) * width
              const py = (y) => ((yr.end - y) / (yr.end - yr.start)) * height
              const dataX = (x) => xr.start + (x / width) * (xr.end - xr.start)
              const dataY = (y) => yr.end - (y / height) * (yr.end - yr.start)
              // The points no name may cover: those named, and those in focus.
              const points = candidates.map((name) => [px(name.x), py(name.y)])
              if (chosen) dots.data.x.forEach((x, i) => points.push([px(x), py(dots.data.y[i])]))
              const covers = (box) => points.some(([x, y]) => x > box.l - 4 && x < box.r + 4 && y > box.t - 4 && y < box.b + 4)
              // Features at one point, as two reactions of the same statistics, share one name.
              const groups = []
              for (const name of candidates) {
                const x = px(name.x)
                const y = py(name.y)
                const same = groups.find((group) => Math.abs(group.px - x) < 3 && Math.abs(group.py - y) < 3)
                if (same) same.more += 1
                else groups.push({x: name.x, y: name.y, text: name.text, px: x, py: y, more: 0})
              }
              const boxes = []
              const labels = {x: [], y: [], text: [], align: [], x_offset: [], y_offset: []}
              const lines = {x0: [], y0: [], x1: [], y1: []}
              for (const group of groups) {
                const x = group.px
                const y = group.py
                if (x < 0 || x > width || y < 0 || y > height) continue
                // A name longer than the room on its wider side ends in an ellipsis.
                const more = group.more ? ` +${group.more}` : ''
                const room = Math.max(width - x, x) - 18
                let text = group.text
                while (text.length > 10 && measure(text + more) > room) text = text.slice(0, -2) + '\u2026'
                text += more
                if (measure(text) > room) continue
                const w = measure(text) + 6
                const sides = x > width * 0.6 ? [true, false] : [false, true]
                let spot = null
                for (const lift of [0, 14, -14, 28, -28, 42, -42]) {
                  for (const right of sides) {
                    const left = right ? x - 6 - w : x + 6
                    const top = y - lift - 8
                    const box = {l: left, r: left + w, t: top, b: top + 16}
                    if (box.l < 0 || box.r > width || box.t < 0 || box.b > height) continue
                    if (boxes.some((o) => box.l < o.r && box.r > o.l && box.t < o.b && box.b > o.t)) continue
                    if (covers(box)) continue
                    spot = {box, right, lift}
                    break
                  }
                  if (spot) break
                }
                if (!spot) continue
                boxes.push(spot.box)
                labels.x.push(group.x)
                labels.y.push(group.y)
                labels.text.push(text)
                labels.align.push(spot.right ? 'right' : 'left')
                labels.x_offset.push(spot.right ? -6 : 6)
                labels.y_offset.push(spot.lift)
                if (spot.lift !== 0) {
                  // From the point's edge to the name's near end.
                  const endX = spot.right ? x - 6 : x + 6
                  const endY = y - spot.lift
                  const length = Math.hypot(endX - x, endY - y)
                  lines.x0.push(dataX(x + ((endX - x) / length) * 4))
                  lines.y0.push(dataY(y + ((endY - y) / length) * 4))
                  lines.x1.push(dataX(endX))
                  lines.y1.push(dataY(endY))
                }
              }
              names.data = labels
              leaders.data = lines
            }
            // Once a frame, however many of the view's ends and sizes change in it.
            if (names.__metabis_placing) return
            names.__metabis_placing = true
            requestAnimationFrame(() => { names.__metabis_placing = false; run(0) })
            """,
        )
        for model, attribute in ((plot.x_range, "start"), (plot.x_range, "end"), (plot.y_range, "start"), (plot.y_range, "end"), (plot, "inner_width"), (plot, "inner_height")):
            model.js_on_change(attribute, place)
    update = CustomJS(
        args=dict(
            plot=plot,
            focus=focus,
            zoom=zoom,
            veil=veil,
            chosen=chosen,
            sources=[layer.source for layer in layers],
            xs=[layer.x for layer in layers],
            ys=[layer.y for layer in layers],
            colors=[layer.color for layer in layers],
            ids=[layer.id_field for layer in layers],
            categories=dict(categories),
            floor=floor,
            names=names.labels if names is not None else None,
            place=place,
            naming=dict(count=names.count, name=names.name, significant=names.significant) if names is not None else None,
        ),
        code="""
        const kinds = focus.data.kind
        const subsystems = new Set(focus.data.id.filter((id, i) => kinds[i] === 'focus'))
        const wanted = new Set(focus.data.id.filter((id, i) => kinds[i] === 'category'))
        const active = subsystems.size > 0 || wanted.size > 0
        const out = {x: [], y: [], color: []}
        const picks = []
        sources.forEach((source, k) => {
          const d = source.data
          const fixed = d[colors[k]] === undefined
          for (let i = 0; i < d[xs[k]].length; i++) {
            const x = d[xs[k]][i]
            const y = d[ys[k]][i]
            if (x == null || y == null || !isFinite(x) || !isFinite(y)) continue
            if (active && d.focus_subsystems[i].some((s) => subsystems.has(s) || wanted.has(categories[s]))) {
              out.x.push(x)
              out.y.push(y)
              out.color.push(fixed ? colors[k] : d[colors[k]][i])
              if (naming !== null) picks.push({
                x, y,
                text: String(d[naming.name][i]),
                significant: d[naming.significant][i] ? 1 : 0,
                base: String(d[ids[k]][i]).replace(/_back$/, ''),
              })
            }
          }
        })
        chosen.data = out
        veil.visible = active
        // Zoomed to the features in focus while asked to and there are some;
        // otherwise back to the view the figure had before.
        const xr = plot.x_range
        const yr = plot.y_range
        const saved = (window.__metabisFeatureView ??= {})
        const zoomed = zoom.data.zoom[0] === true && out.x.length > 0
        if (zoomed) {
          if (!saved[plot.id]) saved[plot.id] = [xr.start, xr.end, yr.start, yr.end]
          const pad = (lo, hi) => { const m = Math.max((hi - lo) * 0.12, 0.05); return [lo - m, hi + m] }
          const [x0, x1] = pad(Math.min(...out.x), Math.max(...out.x))
          const [y0, y1] = pad(Math.min(...out.y), Math.max(...out.y))
          xr.have_updated_interactively = true
          yr.have_updated_interactively = true
          xr.setv({start: x0, end: x1})
          yr.setv({start: floor === null ? y0 : Math.max(floor, y0), end: y1})
        } else if (saved[plot.id]) {
          const [x0, x1, y0, y1] = saved[plot.id]
          delete saved[plot.id]
          xr.setv({start: x0, end: x1})
          yr.setv({start: y0, end: y1})
        }
        if (names !== null) {
          // The names of the features in focus, the significant first, then the
          // highest, one per reaction; with nothing in focus, the figure's own.
          let candidates = null
          if (active) {
            picks.sort((a, b) => b.significant - a.significant || b.y - a.y)
            const seen = new Set()
            candidates = []
            for (const pick of picks) {
              if (seen.has(pick.base)) continue
              seen.add(pick.base)
              candidates.push(pick)
              if (candidates.length === naming.count) break
            }
          }
          ;(window.__metabisFocusNames ??= {})[names.id] = candidates
          place.execute(names)
        }
        """,
    )
    focus.js_on_change("data", update)
    zoom.js_on_change("data", update)
    return focus

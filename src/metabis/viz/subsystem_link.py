"""One subsystem selection shared by the metabolic network and its ranking.

The two figures are embedded separately, so they cannot share a data source.
Instead each holds a source named :data:`FOCUS_SOURCE`, with one row per
highlighted subsystem: ``kind`` is ``"focus"`` for the subsystem pointed at or
pinned, ``"match"`` for each match of a search, ``"search"`` (with no ``id``)
while a search is active, and ``"category"`` (its ``id`` the category's name) for
a selected category. The page writes it, and each figure redraws its own
highlight from it. Pointing at a subsystem in either figure dispatches
:data:`FOCUS_EVENT` on ``window`` with ``detail = {id, kind}``, ``kind`` being
``"hover"`` or ``"leave"``; a tap dispatches the page's navigation event, which
also pins it. A tap on a category's circle or number in the network selects the
category, or clears it when it is already selected, and dispatches
:data:`CATEGORY_EVENT` with ``detail = {category}``. The page's side is
web/src/bokeh/subsystemLink.ts.
"""

from __future__ import annotations

from typing import Optional

from bokeh.models import ColumnDataSource, CustomJS, HoverTool

from .brand import GRADIENT_JS, OUTLINE_WIDTH, SELECTED_HEXAGON

FOCUS_EVENT = "metabis:subsystem-focus"
FOCUS_SOURCE = "metabis-subsystem-focus"
CATEGORY_EVENT = "metabis:category-focus"
# The network's category circles, their numbered hexagons, and the logo's hexagon
# a selected one takes (bokeh_plotting._draw_enriched_subsystem_categories).
CATEGORY_CIRCLES = "metabis-category-circles"
CATEGORY_HEXES = "metabis-category-hexes"
CATEGORY_BADGE = "metabis-category-badge"
# The network's outlines of what is selected, in the logo's gradient.
NETWORK_OUTLINE = "metabis-network-outline"
# Pixels round a hexagon's centre that a tap counts as on it.
CATEGORY_HEX_REACH = 12


def focus_source() -> ColumnDataSource:
    return ColumnDataSource({"id": [], "kind": []}, name=FOCUS_SOURCE)


def focus_dispatch(source: ColumnDataSource, id_field: str = "id") -> CustomJS:
    """A hover callback that names the subsystem under the pointer, once per change."""
    return CustomJS(
        args=dict(source=source, field=id_field, event=FOCUS_EVENT),
        code="""
        const index = cb_data.index.indices[0]
        const id = index === undefined ? null : source.data[field][index]
        const key = '__metabisFocus'
        if (window[key] === id) return
        window[key] = id
        window.dispatchEvent(new CustomEvent(event, {detail: {id, kind: id === null ? 'leave' : 'hover'}}))
        """,
    )


def navigation_dispatch(
    source: ColumnDataSource, event: str, *, id_field: str = "id", label_field: str = "label"
) -> CustomJS:
    """A tap on a row names its subsystem to the page, as a network tap does."""
    return CustomJS(
        args=dict(source=source, event=event, id_field=id_field, label_field=label_field),
        code="""
        const index = source.selected.indices[0]
        if (index === undefined) return
        const element_id = source.data[id_field][index]
        const label = source.data[label_field][index]
        source.selected.indices = []
        window.dispatchEvent(new CustomEvent(event, {detail: {element_id, label}}))
        """,
    )


def _named_source(plot, name: str) -> Optional[ColumnDataSource]:
    for renderer in plot.renderers:
        source = getattr(renderer, "data_source", None)
        if source is not None and source.name == name:
            return source
    return None


def add_network_focus_link(plot, nodes: Optional[ColumnDataSource] = None) -> Optional[ColumnDataSource]:
    """Ring the subsystem in focus, fade what a search does not match, and select categories.

    ``nodes`` is the subsystem glyphs' source, found in ``plot`` when omitted: the
    renderer whose source marks ``is_subsystem``. The ring round the subsystem in
    focus, and round a selected category, is outlined in the logo's gradient and
    never filled, so it cannot be read as a score. A selected category's number
    takes the logo's hexagon, the rest of the network fades and the view zooms to
    the category; clearing it returns to the view it was selected from. Its
    subsystems are named by js_callbacks.add_fitting_label_callback.
    """
    if nodes is None:
        for renderer in plot.renderers:
            source = getattr(renderer, "data_source", None)
            if source is not None and "is_subsystem" in source.data:
                nodes = source
                break
    if nodes is None:
        return None
    nodes.data["base_alpha"] = list(nodes.data.get("alpha", [1.0] * len(nodes.data["x"])))
    # The network's names already follow a focus source of their own; the ring
    # and the fading follow the same one.
    focus = next((tag for tag in plot.tags if getattr(tag, "name", None) == FOCUS_SOURCE), None)
    if focus is None:
        focus = focus_source()
        attach(plot, focus)
    # The ring round the subsystem in focus and round a selected category, outlined
    # in the logo's gradient as short segments (brand.GRADIENT_JS). It is drawn over
    # the subsystems but under the categories' numbers, which sit on their rims.
    ring = ColumnDataSource({"x0": [], "y0": [], "x1": [], "y1": [], "color": []}, name=NETWORK_OUTLINE)
    plot.segment(
        x0="x0",
        y0="y0",
        x1="x1",
        y1="y1",
        line_color="color",
        line_width=OUTLINE_WIDTH,
        line_cap="round",
        level="glyph",
        source=ring,
    )
    # The shared-metabolite edges fade with the nodes while a search is active.
    edges = None
    for renderer in plot.renderers:
        source = getattr(renderer, "data_source", None)
        if source is not None and {"x0", "line_alpha"} <= set(source.data):
            edges = source
            break
    if edges is not None:
        edges.data["base_line_alpha"] = list(edges.data["line_alpha"])
    circles = _named_source(plot, CATEGORY_CIRCLES)
    hexes = _named_source(plot, CATEGORY_HEXES)
    badge = _named_source(plot, CATEGORY_BADGE)
    if circles is not None:
        circles.data["base_alpha"] = list(circles.data["alpha"])
    focus.js_on_change(
        "data",
        CustomJS(
            args=dict(
                plot=plot,
                focus=focus,
                nodes=nodes,
                ring=ring,
                edges=edges,
                circles=circles,
                hexes=hexes,
                badge=badge,
                hexagon=SELECTED_HEXAGON,
            ),
            code=GRADIENT_JS
            + """
            const ids = nodes.data.element_id
            const kinds = focus.data.kind
            const focused = focus.data.id.filter((id, i) => kinds[i] === 'focus')
            const searching = kinds.includes('search')
            const matches = new Set(focus.data.id.filter((id, i) => kinds[i] === 'match'))
            const category = focus.data.id.find((id, i) => kinds[i] === 'category') ?? null
            const index = focused.length ? ids.indexOf(focused[0]) : -1
            const down = plot.y_range.start > plot.y_range.end ? 1 : -1
            const outline = {x0: [], y0: [], x1: [], y1: [], color: []}
            if (index >= 0) {
              metabisRing(outline, nodes.data.x[index], nodes.data.y[index], nodes.data.radius[index] * 1.25 + 40, down)
            }
            const base = nodes.data.base_alpha
            const members = nodes.data.category ?? []
            nodes.data.alpha = ids.map((id, i) => {
              const faded = (searching && !matches.has(id)) || (category !== null && members[i] !== category)
              return faded && i !== index ? base[i] * 0.2 : base[i]
            })
            nodes.change.emit()
            const chosen = circles && category !== null ? circles.data.category.indexOf(category) : -1
            if (circles) {
              // Outlined, never filled, so the selection is not read as a score.
              const c = circles.data
              c.alpha = c.base_alpha.map((alpha, i) => category !== null && i !== chosen ? alpha * 0.3 : alpha)
              circles.change.emit()
              if (chosen >= 0) metabisRing(outline, c.x[chosen], c.y[chosen], c.radius[chosen], down, 192)
            }
            ring.data = outline
            if (hexes) {
              const h = hexes.data
              // The selected number's own hexagon gives way to the logo's.
              h.alpha = h.category.map((name) => name === category ? 0 : category !== null ? 0.3 : 1)
              h.text_alpha = h.category.map((name) => category !== null && name !== category ? 0.3 : 1)
              hexes.change.emit()
              if (badge) {
                const i = h.category.indexOf(category)
                badge.data = i < 0 ? {url: [], x: [], y: []} : {url: [hexagon], x: [h.x[i]], y: [h.y[i]]}
              }
            }
            if (edges) {
              const e = edges.data
              const inside = (i) => {
                if (chosen < 0) return true
                const x = (e.x0[i] + e.x1[i]) / 2 - circles.data.x[chosen]
                const y = (e.y0[i] + e.y1[i]) / 2 - circles.data.y[chosen]
                return Math.hypot(x, y) <= circles.data.radius[chosen]
              }
              e.line_alpha = e.base_line_alpha.map((alpha, i) => searching || !inside(i) ? alpha * 0.2 : alpha)
              edges.change.emit()
            }
            // Zoom to a newly selected category; clearing it returns to the view
            // it was selected from.
            const views = (window.__metabisCategoryView ??= {})
            const view = (views[plot.id] ??= {category: null, ranges: null})
            if (category === view.category) return
            const xr = plot.x_range
            const yr = plot.y_range
            if (view.category === null) view.ranges = [xr.start, xr.end, yr.start, yr.end]
            view.category = category
            // As a zoom tool does, so a data range keeps the view rather than
            // computing it again from the data; cleared, it fits the data again.
            xr.have_updated_interactively = category !== null
            yr.have_updated_interactively = category !== null
            if (chosen >= 0) {
              let width = 0
              let height = 0
              try { width = plot.inner_width; height = plot.inner_height } catch { return }
              if (!(width > 0 && height > 0)) return
              const radius = circles.data.radius[chosen] * 1.08
              const unit = Math.max(2 * radius / width, 2 * radius / height)
              const cx = circles.data.x[chosen]
              const cy = circles.data.y[chosen]
              xr.setv({start: cx - unit * width / 2, end: cx + unit * width / 2})
              const flipped = yr.start > yr.end
              const dy = unit * height / 2
              yr.setv(flipped ? {start: cy + dy, end: cy - dy} : {start: cy - dy, end: cy + dy})
            } else if (category === null && view.ranges) {
              const [x0, x1, y0, y1] = view.ranges
              xr.setv({start: x0, end: x1})
              yr.setv({start: y0, end: y1})
            }
            """,
        ),
    )
    # A zoomed view keeps both axes at one scale when the plot changes size, as
    # when the key under it is shown or its column is widened; an unzoomed one is
    # fitted to the data again by Bokeh itself.
    same_scale = CustomJS(
        args=dict(plot=plot),
        code="""
        const xr = plot.x_range
        const yr = plot.y_range
        if (!xr.have_updated_interactively && !yr.have_updated_interactively) return
        let width, height
        try { width = plot.inner_width; height = plot.inner_height } catch { return }
        if (!(width > 0 && height > 0)) return
        const half = Math.abs(xr.end - xr.start) / width * height / 2
        const middle = (yr.start + yr.end) / 2
        yr.have_updated_interactively = true
        yr.setv(yr.start > yr.end ? {start: middle + half, end: middle - half} : {start: middle - half, end: middle + half})
        """,
    )
    plot.js_on_change("inner_width", same_scale)
    plot.js_on_change("inner_height", same_scale)
    if circles is not None:
        plot.js_on_event(
            "tap",
            CustomJS(
                args=dict(
                    plot=plot,
                    focus=focus,
                    nodes=nodes,
                    circles=circles,
                    hexes=hexes,
                    reach=CATEGORY_HEX_REACH,
                    event=CATEGORY_EVENT,
                ),
                code="""
                const x = cb_obj.x
                const y = cb_obj.y
                let scale
                try { scale = plot.inner_width / Math.abs(plot.x_range.end - plot.x_range.start) } catch { return }
                // A tap on a subsystem's node opens the subsystem instead.
                const n = nodes.data
                for (let i = 0; i < n.x.length; i++) {
                  if (Math.hypot(n.x[i] - x, n.y[i] - y) <= n.radius[i]) return
                }
                let hit = null
                if (hexes) {
                  const h = hexes.data
                  for (let i = 0; i < h.x.length; i++) {
                    if (Math.hypot(h.x[i] - x, h.y[i] - y) * scale <= reach) hit = h.category[i]
                  }
                }
                const c = circles.data
                for (let i = 0; hit === null && i < c.x.length; i++) {
                  if (Math.hypot(c.x[i] - x, c.y[i] - y) <= c.radius[i]) hit = c.category[i]
                }
                if (hit === null) return
                const kinds = focus.data.kind
                const current = focus.data.id.find((id, i) => kinds[i] === 'category') ?? null
                const category = current === hit ? null : hit
                // The figure keeps its own selection, and the page relays it.
                const id = []
                const kind = []
                focus.data.id.forEach((value, i) => {
                  if (kinds[i] !== 'category') { id.push(value); kind.push(kinds[i]) }
                })
                if (category !== null) { id.push(category); kind.push('category') }
                focus.data = {id, kind}
                window.dispatchEvent(new CustomEvent(event, {detail: {category}}))
                """,
            ),
        )
    for tool in plot.select(type=HoverTool):
        if nodes in [getattr(r, "data_source", None) for r in (tool.renderers or [])]:
            tool.callback = focus_dispatch(nodes, "element_id")
    return focus


def attach(plot, source: ColumnDataSource) -> None:
    """Keep a source no renderer draws in the serialized document, through the plot's tags."""
    plot.tags = [*plot.tags, source]


__all__ = [
    "CATEGORY_BADGE",
    "CATEGORY_CIRCLES",
    "CATEGORY_EVENT",
    "CATEGORY_HEXES",
    "FOCUS_EVENT",
    "attach",
    "FOCUS_SOURCE",
    "NETWORK_OUTLINE",
    "add_network_focus_link",
    "focus_dispatch",
    "focus_source",
    "navigation_dispatch",
]

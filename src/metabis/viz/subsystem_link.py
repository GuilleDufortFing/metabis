"""One subsystem selection shared by the metabolic network and its ranking.

The two figures are embedded separately, so they cannot share a data source.
Instead each holds a source named :data:`FOCUS_SOURCE`, with one row per
highlighted subsystem: ``kind`` is ``"focus"`` for the subsystem pointed at or
pinned, ``"match"`` for each match of a search, and ``"search"`` (with no ``id``)
while a search is active. The page writes it, and each figure redraws its own
highlight from it. Pointing at a subsystem in either figure dispatches
:data:`FOCUS_EVENT` on ``window`` with ``detail = {id, kind}``, ``kind`` being
``"hover"`` or ``"leave"``; a tap dispatches the page's navigation event, which
also pins it. The page's side is web/src/bokeh/subsystemLink.ts.
"""

from __future__ import annotations

from typing import Optional

from bokeh.models import ColumnDataSource, CustomJS, HoverTool

FOCUS_EVENT = "metabis:subsystem-focus"
FOCUS_SOURCE = "metabis-subsystem-focus"
FOCUS_RING_COLOR = "#0162b7"  # the client's link blue


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


def add_network_focus_link(plot, nodes: Optional[ColumnDataSource] = None) -> Optional[ColumnDataSource]:
    """Ring the subsystem in focus and fade what a search does not match, in the network.

    ``nodes`` is the subsystem glyphs' source, found in ``plot`` when omitted: the
    renderer whose source marks ``is_subsystem``.
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
    focus = focus_source()
    ring = ColumnDataSource({"x": [], "y": [], "radius": []})
    plot.circle(
        x="x",
        y="y",
        radius="radius",
        fill_alpha=0,
        line_color=FOCUS_RING_COLOR,
        line_width=3,
        level="overlay",
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
    focus.js_on_change(
        "data",
        CustomJS(
            args=dict(focus=focus, nodes=nodes, ring=ring, edges=edges),
            code="""
            const ids = nodes.data.element_id
            const kinds = focus.data.kind
            const focused = focus.data.id.filter((id, i) => kinds[i] === 'focus')
            const searching = kinds.includes('search')
            const matches = new Set(focus.data.id.filter((id, i) => kinds[i] === 'match'))
            const index = focused.length ? ids.indexOf(focused[0]) : -1
            ring.data = index < 0
              ? {x: [], y: [], radius: []}
              : {x: [nodes.data.x[index]], y: [nodes.data.y[index]], radius: [nodes.data.radius[index] * 1.25 + 40]}
            const base = nodes.data.base_alpha
            nodes.data.alpha = ids.map((id, i) => searching && !matches.has(id) && i !== index ? base[i] * 0.2 : base[i])
            nodes.change.emit()
            if (edges) {
              edges.data.line_alpha = edges.data.base_line_alpha.map((alpha) => searching ? alpha * 0.2 : alpha)
              edges.change.emit()
            }
            """,
        ),
    )
    for tool in plot.select(type=HoverTool):
        if nodes in [getattr(r, "data_source", None) for r in (tool.renderers or [])]:
            tool.callback = focus_dispatch(nodes, "element_id")
    attach(plot, focus)
    return focus


def attach(plot, source: ColumnDataSource) -> None:
    """Keep a source no renderer draws in the serialized document, through the plot's tags."""
    plot.tags = [*plot.tags, source]


__all__ = [
    "FOCUS_EVENT",
    "attach",
    "FOCUS_SOURCE",
    "add_network_focus_link",
    "focus_dispatch",
    "focus_source",
    "navigation_dispatch",
]

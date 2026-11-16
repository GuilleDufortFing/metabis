"""Rings round chosen elements of a drawn metabolic graph or network.

A group's markers are marked where the graphs draw them: each marker reaction's
diamond, or gene's box, in a subsystem graph, and each subsystem holding one in
the network, gets a ring in the group's colour round its own shape. The ring is
an outline, never a fill, so the node keeps the colour its statistics give it.
"""

from __future__ import annotations

from typing import Iterable

from bokeh.models import ColumnDataSource, GlyphRenderer

RING_WIDTH = 3  # pixels, as the selected subsystem's ring
# How far a ring stands off its node, as a share of the node's size.
RING_SCALE = 1.35


def _key(element) -> str:
    return str(element).removesuffix("_back")


def _feature(element) -> str:
    """The reaction or gene a subsystem graph's node stands for.

    The graph names its nodes by type: ``r_<reaction>`` for a reaction and
    ``g_r_<reaction>_<gene>`` for a gene drawn beside one; any other node, a
    subsystem of the network or a map's reaction, is named by its own ID.
    """
    element = str(element)
    if element.startswith("g_r_"):
        return element[len("g_r_"):].split("_", 1)[-1]
    if element.startswith("r_"):
        return element[len("r_"):]
    return element


# The glyphs a node is drawn with; a label drawn over a node is left alone.
NODE_GLYPHS = {"Patches", "Rect", "Circle", "Ellipse", "Scatter"}


def _sources(plot):
    seen = set()
    for renderer in plot.renderers:
        if not isinstance(renderer, GlyphRenderer) or type(renderer.glyph).__name__ not in NODE_GLYPHS:
            continue
        source = renderer.data_source
        if id(source) in seen or not isinstance(source, ColumnDataSource):
            continue
        seen.add(id(source))
        if "element_id" in source.data and "x" in source.data and "y" in source.data:
            yield source


def outline_elements(plot, element_ids: Iterable[str], *, color: str) -> int:
    """Ring every drawn element whose ID is in ``element_ids``; return how many.

    A subsystem graph's node matches the reaction or gene it stands for, and a
    reaction with or without its ``_back`` direction. A reaction's
    diamond is ringed by a larger diamond, a gene's box by a larger box, and a
    subsystem's circle by a larger circle; an element of another shape gets a
    circle of fixed size on screen.
    """
    wanted = {_key(element) for element in element_ids}
    ringed = 0
    for source in _sources(plot):
        data = source.data
        rows = [
            index
            for index, element in enumerate(data["element_id"])
            if _key(element) in wanted or _key(_feature(element)) in wanted
        ]
        if not rows:
            continue
        ringed += len(rows)
        x = [data["x"][index] for index in rows]
        y = [data["y"][index] for index in rows]
        style = dict(fill_alpha=0, line_color=color, line_width=RING_WIDTH, level="overlay")
        if "xs" in data and "ys" in data:
            xs = [[x0 + (px - x0) * RING_SCALE for px in data["xs"][index]] for x0, index in zip(x, rows)]
            ys = [[y0 + (py - y0) * RING_SCALE for py in data["ys"][index]] for y0, index in zip(y, rows)]
            plot.patches(xs="xs", ys="ys", source=ColumnDataSource({"xs": xs, "ys": ys}), **style)
        elif "radius" in data:
            radius = [float(data["radius"][index]) * 1.15 for index in rows]
            plot.circle(x="x", y="y", radius="radius", source=ColumnDataSource({"x": x, "y": y, "radius": radius}), **style)
        elif "width" in data and "height" in data:
            width = [float(data["width"][index]) * 1.25 for index in rows]
            height = [float(data["height"][index]) * 1.6 for index in rows]
            plot.rect(
                x="x", y="y", width="width", height="height",
                source=ColumnDataSource({"x": x, "y": y, "width": width, "height": height}), **style,
            )
        else:
            plot.scatter(x="x", y="y", size=22, source=ColumnDataSource({"x": x, "y": y}), **style)
    return ringed


__all__ = ["outline_elements"]

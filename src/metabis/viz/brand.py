"""The logo's gradient, which marks what a figure has selected.

The logo strokes its hexagon in a gradient from #4facfe at the top left to
#00f2fe at the bottom right. A selection is outlined in it, never filled, so it
cannot be read as a down-regulated blue. Bokeh strokes a line in one colour, so a
figure draws a gradient outline as short segments, each in the gradient's colour
at its place: :data:`GRADIENT_JS` holds the functions its callbacks use, and
:func:`gradient_color` the same in Python. A category's number, once selected,
takes the logo's own hexagon, :data:`SELECTED_HEXAGON`, an image the figure
draws at a fixed size on screen.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.parse import quote

from ..visual_config import theme

GRADIENT = (theme.SELECTED, theme.SELECTED_END)
OUTLINE_WIDTH = 3  # pixels, as the ring round the subsystem in focus has been


def _rgb(color: str) -> Tuple[int, int, int]:
    return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


def gradient_color(t: float) -> str:
    """The gradient's colour a share ``t`` of the way from its start to its end."""
    t = min(1.0, max(0.0, float(t)))
    start, end = _rgb(GRADIENT[0]), _rgb(GRADIENT[1])
    return "#" + "".join(f"{round(a + (b - a) * t):02x}" for a, b in zip(start, end))


# The same in a callback: metabisGradient(t), and metabisRing, which appends a
# circle's outline as segments to {x0, y0, x1, y1, color} with its top left in
# the gradient's start colour; ``down`` is +1 where y grows down the screen.
GRADIENT_JS = """
const metabisGradient = (t) => {
  const start = [%s], end = [%s]
  const k = Math.min(1, Math.max(0, t))
  return '#' + start.map((v, i) => Math.round(v + (end[i] - v) * k).toString(16).padStart(2, '0')).join('')
}
const metabisRing = (out, cx, cy, r, down, pieces = 96) => {
  for (let i = 0; i < pieces; i++) {
    const a0 = 2 * Math.PI * i / pieces
    const a1 = 2 * Math.PI * (i + 1) / pieces
    const am = (a0 + a1) / 2
    out.x0.push(cx + r * Math.cos(a0))
    out.y0.push(cy + r * Math.sin(a0))
    out.x1.push(cx + r * Math.cos(a1))
    out.y1.push(cy + r * Math.sin(a1))
    out.color.push(metabisGradient((Math.SQRT2 + Math.cos(am) + down * Math.sin(am)) / (2 * Math.SQRT2)))
  }
}
""" % (
    ", ".join(str(v) for v in _rgb(GRADIENT[0])),
    ", ".join(str(v) for v in _rgb(GRADIENT[1])),
)


def gradient_segments(
    lines: Iterable[Sequence[float]], *, piece: float, down: int = 1
) -> Dict[str, List[float]]:
    """Polylines as short segments coloured along the gradient across their extent.

    ``lines`` are flat ``[x0, y0, x1, y1, ...]`` lists; a segment is at most
    ``piece`` long, and takes the colour of its midpoint, from the top left of
    everything drawn to its bottom right (``down`` is +1 where y grows down the
    screen).
    """
    lines = [list(line) for line in lines]
    xs = [v for line in lines for v in line[0::2]]
    ys = [v * down for line in lines for v in line[1::2]]
    out: Dict[str, List[float]] = {"x0": [], "y0": [], "x1": [], "y1": [], "color": []}
    if not xs:
        return out
    left, top = min(xs), min(ys)
    span = max(max(xs) - left, 1e-9) + max(max(ys) - top, 1e-9)
    for line in lines:
        points = list(zip(line[0::2], line[1::2]))
        for (ax, ay), (bx, by) in zip(points, points[1:]):
            count = max(1, math.ceil(math.hypot(bx - ax, by - ay) / piece))
            for i in range(count):
                x0, y0 = ax + (bx - ax) * i / count, ay + (by - ay) * i / count
                x1, y1 = ax + (bx - ax) * (i + 1) / count, ay + (by - ay) * (i + 1) / count
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2 * down
                out["x0"].append(x0)
                out["y0"].append(y0)
                out["x1"].append(x1)
                out["y1"].append(y1)
                out["color"].append(gradient_color(((mx - left) + (my - top)) / span))
    return out


# The logo's hexagon frame: the side menu's dark node, stroked in the gradient.
_HEXAGON_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='96' height='96' viewBox='0 0 100 100'>"
    "<defs><linearGradient id='g' x1='0%' y1='0%' x2='100%' y2='100%'>"
    f"<stop offset='0%' stop-color='{GRADIENT[0]}'/><stop offset='100%' stop-color='{GRADIENT[1]}'/>"
    "</linearGradient></defs>"
    "<path d='M50 6L88 28V72L50 94L12 72V28Z' "
    f"fill='{theme.INK_NODE}' stroke='url(#g)' stroke-width='11' stroke-linejoin='round'/></svg>"
)
SELECTED_HEXAGON = "data:image/svg+xml;utf8," + quote(_HEXAGON_SVG)
# Pixels across the selected hexagon, a little larger than the plain one it covers.
SELECTED_HEXAGON_SIZE = 25


__all__ = [
    "GRADIENT",
    "GRADIENT_JS",
    "OUTLINE_WIDTH",
    "SELECTED_HEXAGON",
    "SELECTED_HEXAGON_SIZE",
    "gradient_color",
    "gradient_segments",
]

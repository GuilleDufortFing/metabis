"""The one look every hover panel has: the subsystem graph's reaction and gene panel.

A panel is a white box in 12 px Arial with a light border and shadow; each fact
is a row led by its name in bold, and a group of facts is a section set off by a
rule above it, led by its title. The hover builders (:mod:`.builder`) draw every
graph element's panel in it, and a figure's own hover is built from the same
pieces, as finished HTML a column holds or as a Bokeh tooltip template whose
``@field`` placeholders Bokeh fills.

Plain strings only, so the graph worker imports it without Bokeh.
"""

from __future__ import annotations

PANEL_STYLE = (
    "font-family: Arial, sans-serif; font-size:12px; padding:10px; "
    "border:1px solid #ddd; border-radius:4px; background-color:rgba(255, 255, 255, 0.95); "
    "box-shadow: 0 2px 5px rgba(0,0,0,0.1); max-width: 700px;"
)
SECTION_STYLE = "margin-top:8px; border-top:1px solid #ddd; padding-top:8px;"
# A note under the facts, such as how a value was computed.
NOTE_STYLE = "font-size:11px; color:#555;"

_OPEN = f"<div style='{PANEL_STYLE}'>"
_CLOSE = "</div>"


def row(label: str, value: object) -> str:
    """One fact: its name in bold, then its value."""
    return f"<div><strong>{label}:</strong> {value}</div>"


def note(text: str) -> str:
    """A line in smaller grey type under the facts."""
    return f"<div style='{NOTE_STYLE}'>{text}</div>"


def section(*parts: str, title: str | None = None) -> str:
    """Facts set off from those above by a rule, led by ``title`` when given."""
    head = f"<div><strong>{title}:</strong></div>" if title else ""
    return f"<div style='{SECTION_STYLE}'>{head}{''.join(parts)}</div>"


def columns(*parts: str) -> str:
    """Side-by-side blocks, as the statistics section sets its two directions."""
    return (
        "<div style='display:flex; justify-content:center; gap:12px; "
        f"align-items:flex-start; margin-top:8px;'>{''.join(parts)}</div>"
    )


def column(title: str, *parts: str, accent: str = "transparent") -> str:
    """One of :func:`columns`' blocks, outlined in ``accent`` when it is the one that counts."""
    return (
        f"<div style='flex:1; min-width:160px; border:1.5px solid {accent}; border-radius:4px; padding:6px;'>"
        f"<div style='font-weight:bold; margin-bottom:4px;'>{title}</div>{''.join(parts)}</div>"
    )


def significance(is_significant: bool) -> str:
    """Whether a test passed, in red, or not, in grey."""
    if is_significant:
        return "<div style='color:red; font-weight:bold;'>Significant</div>"
    return "<div style='color:#aaaaaa; font-weight:bold;'>Not Significant</div>"


def panel(*parts: str) -> str:
    """The panel round ``parts``."""
    return f"{_OPEN}{''.join(parts)}{_CLOSE}"


def inner(html: str) -> str:
    """A panel's content without the panel, to set it in another; other HTML as it is."""
    if html.startswith(_OPEN) and html.endswith(_CLOSE):
        return html[len(_OPEN) : -len(_CLOSE)]
    return html


def extend(html: str, *parts: str) -> str:
    """A panel with ``parts`` added at its end; a new one when ``html`` is empty."""
    return panel(inner(html), *parts) if html else panel(*parts)

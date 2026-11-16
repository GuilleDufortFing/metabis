"""A compartment map's names, set as Metabolic Atlas sets them.

Metabolic Atlas lays its maps out for names of a size in map units: a
metabolite's in 18-unit bold Arial, a reaction's in 16, a gene's in 12, a long name
broken over lines, a small metabolite's name beside its dot. Drawn at a fixed
screen size instead, names grow against the map as it is zoomed out and cover
each other. A map with Metabolic Atlas's labels (``scripts/import_atlas_paths.py``)
draws its names in map units, at Metabolic Atlas's places and breaks. All its
names appear together, once a metabolite's can be read; a gene's name is as large
as its box takes, from Metabolic Atlas's 12 units up to a metabolite's 18, so it
can be read when the others can.

The text stays the model's name, which the hover, the details and the find use:
where Metabolic Atlas's lines spell it, broken where Metabolic Atlas breaks it;
otherwise wrapped as Metabolic Atlas wraps names. A metabolite the model does not
name (its name is its identifier) takes Metabolic Atlas's name.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Mapping, Optional

# Metabolic Atlas's sizes, in map units, of each kind's names.
MAP_LABEL_UNITS: Dict[str, float] = {"metabolite": 18.0, "gene": 12.0, "reaction": 16.0}
# Its lines are 20 units apart in 18-unit text.
MAP_LABEL_LINE_HEIGHT = 20.0 / 18.0
# Pixels a name must be to be drawn: smaller, it cannot be read.
MAP_LABEL_MIN_PIXELS = 6.0
# Pixels a map unit takes once a map's names are drawn, every kind at once: when a
# metabolite's name, the largest, can be read. The find's ring goes then too
# (compartment_maps.add_map_find): the elements are large enough to see.
MAP_NAMES_SHOWN_FROM = MAP_LABEL_MIN_PIXELS / MAP_LABEL_UNITS["metabolite"]
# A gene's name fills its box but its margins, in map units.
GENE_BOX_MARGIN = 3.0
# The widths of Helvetica Bold's characters, in thousandths of its size (its Adobe
# font metrics, which Arial Bold shares): a name's width in its box.
_BOLD_WIDTHS = dict(
    zip(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        (722, 722, 722, 722, 667, 611, 778, 722, 278, 556, 722, 611, 833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611),
    )
)
_BOLD_WIDTHS.update(
    zip(
        "abcdefghijklmnopqrstuvwxyz",
        (556, 611, 556, 611, 556, 333, 611, 611, 278, 278, 556, 278, 889, 611, 611, 611, 611, 389, 556, 333, 611, 556, 778, 556, 556, 500),
    )
)
_BOLD_WIDTHS.update({digit: 556 for digit in "0123456789"})
_BOLD_WIDTHS.update({" ": 278, "-": 333, ".": 278, ",": 278, "_": 556, "(": 333, ")": 333, "/": 278, ":": 333})
# Metabolic Atlas keeps a name of up to 21 characters on one line, and breaks a
# longer one into lines of 15 characters or so. Lines of 20 follow its breaks most
# often, where it gives none.
ONE_LINE_LENGTH = 21
LINE_LENGTH = 20
# Metabolic Atlas writes Greek letters where the model spells them out.
GREEK = str.maketrans({"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ω": "omega"})

_PIECE = re.compile(r"[^\s-]*-+|[^\s-]+")


def _letters(text: str) -> str:
    return "".join(
        character for character in text.translate(GREEK).casefold() if not character.isspace() and character != "-"
    )


def follow_breaks(name: str, lines: Iterable[str]) -> Optional[List[str]]:
    """``name`` broken where ``lines`` break, or ``None`` when they do not spell it.

    The lines spell the name when they hold its characters, spaces, hyphens and
    the spelling of Greek letters aside: Metabolic Atlas breaks a word with a hyphen
    of its own. A line that ends inside one of the name's words ends with a hyphen.
    """
    lines = list(lines)
    if _letters("".join(lines)) != _letters(name):
        return None
    broken: List[str] = []
    start = 0
    for line in lines[:-1]:
        wanted = len(_letters(line))
        end, counted = start, 0
        while counted < wanted:
            if not (name[end].isspace() or name[end] == "-"):
                counted += 1
            end += 1
        while end < len(name) and name[end] == "-":
            end += 1
        piece = name[start:end].strip()
        if piece and end < len(name) and not name[end].isspace() and not piece.endswith("-"):
            piece += "-"
        if piece:
            broken.append(piece)
        start = end
    rest = name[start:].strip()
    if rest:
        broken.append(rest)
    return broken


def wrap_name(name: str, length: int = LINE_LENGTH) -> List[str]:
    """``name`` in lines of about ``length`` characters, broken at spaces and after hyphens.

    A word longer than a line runs on from the line before and is broken, with a
    hyphen.
    """
    lines: List[str] = []
    line = ""
    for found in _PIECE.finditer(name):
        piece = found.group()
        spaced = found.start() > 0 and name[found.start() - 1].isspace()
        joined = f"{line} {piece}" if line and spaced else line + piece
        if len(joined) <= length or not line or len(piece) > length:
            line = joined
        else:
            lines.append(line)
            line = piece
        while len(line) > length + 1:
            lines.append(line[: length - 1] + "-")
            line = line[length - 1 :]
    if line:
        lines.append(line)
    return lines


def label_lines(name: str, placed: Optional[Mapping] = None, identifier: Optional[str] = None) -> List[str]:
    """The lines a metabolite's ``name`` takes, as Metabolic Atlas sets it where ``placed`` says.

    A name that is the metabolite's ``identifier`` gives way to Metabolic Atlas's.
    """
    atlas_lines = list((placed or {}).get("lines") or ())
    if atlas_lines and identifier is not None and name == identifier:
        return atlas_lines
    broken = follow_breaks(name, atlas_lines) if atlas_lines else None
    if broken:
        return broken
    return [name] if len(name) <= ONE_LINE_LENGTH else wrap_name(name)


def gene_label_units(name: str, width: float = 60.0, height: float = 30.0) -> float:
    """The size, in whole map units, a gene's ``name`` takes in its ``width`` by ``height`` box.

    As large as fits across the box, within its margins, and no taller than a
    metabolite's name: from Metabolic Atlas's 12 units, which a long name keeps
    as Metabolic Atlas does, up to 18.
    """
    em = sum(_BOLD_WIDTHS.get(character, 611) for character in name) / 1000.0
    across = (float(width) - 2 * GENE_BOX_MARGIN) / em if em else MAP_LABEL_UNITS["metabolite"]
    fitted = min(across, 0.6 * float(height), MAP_LABEL_UNITS["metabolite"])
    return float(max(MAP_LABEL_UNITS["gene"], math.floor(fitted)))


def map_label_columns(glyphs: List[Mapping], placed: Mapping[str, Mapping], *, metabolites: bool) -> Dict[str, list]:
    """Where each glyph's name is drawn, in map units, and its text, for a map's label set.

    A metabolite's name takes Metabolic Atlas's place and breaks (``placed``, under
    its glyph's identifier), except a copy's (``is_local_copy``), which stands
    elsewhere and keeps its name on it. Other names are centred on their glyph.
    """
    xs, ys, texts = [], [], []
    for glyph in glyphs:
        name = str(glyph["display_label"])
        entry = placed.get(str(glyph["id"])) if metabolites else None
        at = entry if entry and not glyph.get("is_local_copy") else {}
        xs.append(glyph["x"] + float(at.get("dx", 0.0)))
        ys.append(glyph["y"] + float(at.get("dy", 0.0)))
        identifier = str(glyph.get("label") or glyph["id"])
        texts.append("\n".join(label_lines(name, entry, identifier)) if metabolites else name)
    return {"label_x": xs, "label_y": ys, "label_text": texts}


__all__ = [
    "LINE_LENGTH",
    "MAP_LABEL_LINE_HEIGHT",
    "MAP_LABEL_MIN_PIXELS",
    "MAP_LABEL_UNITS",
    "MAP_NAMES_SHOWN_FROM",
    "ONE_LINE_LENGTH",
    "follow_breaks",
    "gene_label_units",
    "label_lines",
    "map_label_columns",
    "wrap_name",
]

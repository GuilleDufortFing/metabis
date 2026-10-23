"""Import Metabolic Atlas's own link paths and subsystem bands beside the packaged maps.

The packaged Human1 maps (``resources/graphs/compartment/human1`` and
``resources/graphs/subsystem/human1``) place every reaction and metabolite where
Metabolic Atlas's Human-maps SVG files place them, shifted by one offset per map.
Their links, though, were drawn straight from reaction to metabolite, so a
reaction Metabolic Atlas draws twice, or routes along a bent path to many
partners, crossed the map. The SVG files hold those paths, and a band per
subsystem that Metabolic Atlas lays under its reactions' links. This script reads
them and writes, beside each layout ``<map>.json``, a ``<map>.atlas.json``::

    {
      "source": "<the SVG file's address>",
      "license": "CC BY-SA 4.0",
      "offset": [dx, dy],
      "links": {"MAR03577": [[x0, y0, x1, y1, ...], ...]},
      "heads": {"MAR03577": [[x0, y0, ...], ...]},
      "subsystems": {"Tricarboxylic acid cycle and ...": [[x0, y0, ...], ...]}
    }

with every coordinate in the layout's own frame, to a tenth of a unit. A
reaction's paths are kept only where the layout draws it exactly where Metabolic
Atlas does; any other reaction keeps its straight links. Metabolic Atlas also
draws some metabolites the layout leaves out (those outside Human1, such as
``Rtotal_c``); a path whose end is nearest one of those is left out with it.
Usage::

    python -I scripts/import_atlas_paths.py SVG_DIRECTORY [--check]

``SVG_DIRECTORY`` holds Human-maps' ``compartment/svg`` and ``subsystem/svg``
files as ``compartment/<map>.svg`` and ``subsystem/<map>.svg``. They are Git LFS
files, fetched from
``https://media.githubusercontent.com/media/SysBioChalmers/Human-maps/main/<kind>/svg/<map>.svg``.
The files are parsed as text and nothing in them runs. ``--check`` reports what it
would write, changing nothing.

The SVG files declare the CC BY-SA 4.0 licence in their metadata, so the files
written here are shared under it too (THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

GRAPHS = Path(__file__).resolve().parent.parent / "src/metabis/resources/graphs"
KINDS = ("compartment", "subsystem")
SOURCE = "https://github.com/SysBioChalmers/Human-maps/blob/main/{kind}/svg/{name}.svg"
LICENSE = "CC BY-SA 4.0"
# How far, in map units, a reaction or metabolite may sit from Metabolic Atlas's
# place for it, after the map's offset, and still count as the same place.
TOLERANCE = 1.0

TOKEN = re.compile(r"[MmLlHhVvZz]|-?(?:\d+\.?\d*|\.\d+)(?:e-?\d+)?")
REACTION = re.compile(
    r'<g id="([^"]+)" class="rea">\s*<path class="shape" d="[^"]*" [^>]*'
    r'transform="matrix\(1,0,0,1,([-\d.]+),([-\d.]+)\)"'
)
METABOLITE = re.compile(r'<g class="met ([^"]+)" transform="matrix\(1,0,0,1,([-\d.]+),([-\d.]+)\)"')
PATH = re.compile(r'<path class="(fe|feh) ([^"]+)" d="([^"]*)"')
BAND = re.compile(
    r'<g id="([^"]*)" class="subsystem"[^>]*>\s*<path class="shape" d="([^"]*)"'
)

Point = Tuple[float, float]


def polylines(d: str) -> List[List[Point]]:
    """The subpaths of an SVG path of straight segments, as lists of points."""
    out: List[List[Point]] = []
    current: List[Point] = []
    x = y = 0.0
    start = (0.0, 0.0)
    command = None
    tokens = TOKEN.findall(d)
    i = 0

    def number() -> float:
        nonlocal i
        value = float(tokens[i])
        i += 1
        return value

    while i < len(tokens):
        token = tokens[i]
        if token.isalpha():
            command = token
            i += 1
            if command in "Zz":
                if current:
                    current.append(start)
                    out.append(current)
                    current = []
                x, y = start
                continue
        if command in ("M", "m"):
            nx, ny = number(), number()
            if command == "m":
                nx, ny = nx + x, ny + y
            if current:
                out.append(current)
            current = [(nx, ny)]
            x, y = start = (nx, ny)
            # Further pairs after a move are lines.
            command = "l" if command == "m" else "L"
        elif command in ("L", "l"):
            nx, ny = number(), number()
            if command == "l":
                nx, ny = nx + x, ny + y
            x, y = nx, ny
            current.append((x, y))
        elif command in ("H", "h"):
            nx = number()
            x = nx + x if command == "h" else nx
            current.append((x, y))
        elif command in ("V", "v"):
            ny = number()
            y = ny + y if command == "v" else ny
            current.append((x, y))
        else:
            raise ValueError(f"Unexpected path command {command!r}")
    if current:
        out.append(current)
    return [line for line in out if len(line) > 1]


def _flat(line: List[Point], offset: Point) -> List[float]:
    return [round(value, 1) for x, y in line for value in (x + offset[0], y + offset[1])]


def _near(place: Point, places: List[Point]) -> bool:
    return any(abs(place[0] - x) <= TOLERANCE and abs(place[1] - y) <= TOLERANCE for x, y in places)


def import_map(svg_path: Path, layout_path: Path, kind: str) -> Tuple[dict, dict]:
    """The map's Atlas paths in its layout's frame, and a report of what was kept."""
    svg = svg_path.read_text()
    glyphs = json.loads(layout_path.read_text())["glyphs"]
    ours: Dict[str, List[Point]] = defaultdict(list)
    metabolites: List[Point] = []
    for glyph in glyphs:
        if glyph["glyph_class"] == "reaction":
            ours[str(glyph["id"])].append((glyph["x"], glyph["y"]))
        elif glyph["glyph_class"] == "metabolite":
            metabolites.append((glyph["x"], glyph["y"]))

    atlas: Dict[str, List[Point]] = defaultdict(list)
    for reaction, x, y in REACTION.findall(svg):
        atlas[reaction].append((float(x), float(y)))
    # The offset, from the reactions each of the two draws once.
    shifts = [
        (ours[reaction][0][0] - places[0][0], ours[reaction][0][1] - places[0][1])
        for reaction, places in atlas.items()
        if len(places) == 1 and len(ours.get(reaction, ())) == 1
    ]
    if not shifts:
        raise ValueError(f"{svg_path.name} shares no reaction with {layout_path.name}")
    offset = (
        round(statistics.median(shift[0] for shift in shifts), 3),
        round(statistics.median(shift[1] for shift in shifts), 3),
    )
    shifted = lambda place: (place[0] + offset[0], place[1] + offset[1])  # noqa: E731

    # A reaction keeps its paths where it sits exactly where Metabolic Atlas draws it.
    misplaced = {
        reaction
        for reaction, places in atlas.items()
        if reaction not in ours or not all(_near(shifted(place), ours[reaction]) for place in places)
    }
    # Each reaction's metabolites in the SVG, and whether the layout draws each.
    partners: Dict[str, List[Tuple[Point, bool]]] = defaultdict(list)
    for classes, x, y in METABOLITE.findall(svg):
        place = (float(x), float(y))
        drawn = _near(shifted(place), metabolites)
        for reaction in classes.split()[1:]:
            partners[reaction].append((place, drawn))

    def reaches_a_drawn_metabolite(reaction: str, line: List[Point]) -> bool:
        """Whether the metabolite nearest either end of ``line`` is one the layout draws."""
        candidates = partners.get(reaction)
        if not candidates:
            return True
        ends = (line[0], line[-1])
        _distance, drawn = min(
            (min((end[0] - place[0]) ** 2 + (end[1] - place[1]) ** 2 for end in ends), drawn)
            for place, drawn in candidates
        )
        return drawn

    links: Dict[str, List[List[float]]] = defaultdict(list)
    heads: Dict[str, List[List[float]]] = defaultdict(list)
    dropped = 0
    for kind_of_path, reaction, d in PATH.findall(svg):
        if reaction in misplaced:
            continue
        target = links if kind_of_path == "fe" else heads
        for line in polylines(d):
            if reaches_a_drawn_metabolite(reaction, line):
                target[reaction].append(_flat(line, offset))
            else:
                dropped += 1

    subsystems: Dict[str, List[List[float]]] = {}
    if kind == "compartment":
        for name, d in BAND.findall(svg):
            subsystems[name] = [_flat(line, offset) for line in polylines(d)]

    name = layout_path.stem
    data = {
        "source": SOURCE.format(kind=kind, name=name),
        "license": LICENSE,
        "offset": list(offset),
        "links": dict(sorted(links.items())),
        "heads": dict(sorted(heads.items())),
        "subsystems": subsystems,
    }
    report = {
        "map": f"{kind}/{name}",
        "reactions": len(ours),
        "routed": len(links),
        "kept_straight": len(set(ours) - set(links)),
        "dropped": dropped,
        "subsystems": len(subsystems),
    }
    return data, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("svg_directory", type=Path)
    parser.add_argument("--check", action="store_true", help="report, writing nothing")
    args = parser.parse_args(argv)

    for kind in KINDS:
        for layout_path in sorted((GRAPHS / kind / "human1").glob("*.json")):
            if layout_path.name.endswith(".atlas.json"):
                continue
            svg_path = args.svg_directory / kind / f"{layout_path.stem}.svg"
            if not svg_path.exists():
                print(f"{kind}/{layout_path.stem}: no SVG", file=sys.stderr)
                continue
            data, report = import_map(svg_path, layout_path, kind)
            target = layout_path.with_name(f"{layout_path.stem}.atlas.json")
            text = json.dumps(data, separators=(",", ":"))
            print(
                f"{report['map']}: {report['routed']} of {report['reactions']} reactions routed, "
                f"{report['dropped']} paths to undrawn metabolites left out, "
                f"{report['subsystems']} subsystem bands, {len(text) / 1e6:.2f} MB"
            )
            if not args.check:
                target.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

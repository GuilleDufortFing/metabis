"""Import Metabolic Atlas's own link paths, subsystem bands and names beside the packaged maps.

The packaged Human1 maps (``resources/graphs/compartment/human1`` and
``resources/graphs/subsystem/human1``) place every reaction and metabolite where
Metabolic Atlas's Human-maps files place them (``import_human_maps.py``, which
runs this script after it writes a layout). A layout links a reaction straight to
each of its metabolites, but Metabolic Atlas routes a link along a bent path, to
many partners. The SVG files hold those paths, and a band per subsystem that
Metabolic Atlas lays under its reactions' links. This script reads them and
writes, beside each layout ``<map>.json``, a ``<map>.atlas.json``::

    {
      "source": "<the SVG file's address>",
      "license": "CC BY-SA 4.0",
      "offset": [dx, dy],
      "links": {"MAR03577": [[x0, y0, x1, y1, ...], ...]},
      "heads": {"MAR03577": [[x0, y0, ...], ...]},
      "subsystems": {"Tricarboxylic acid cycle and ...": [[x0, y0, ...], ...]},
      "metabolites": ["dolp_U_r", ...],
      "labels": {"met881": {"lines": ["chondroitin sulfate", "C (GalNAc6S-GlcA)"]},
                 "met882": {"lines": ["acetyl-CoA"], "dx": -61.5, "dot": true}}
    }

with every coordinate in the layout's own frame, to a tenth of a unit: the
``offset`` that takes the SVG's frame to the layout's, nothing for a layout
``import_human_maps.py`` wrote. A reaction's paths are kept only where the layout
draws it exactly where Metabolic Atlas does; any other reaction keeps its straight
links. Where the layout names a reaction otherwise (a Recon3D identifier, say) but
draws it exactly where Metabolic Atlas draws one reaction, it takes that
reaction's paths. Metabolic Atlas may also draw metabolites the layout leaves
out; a path whose end is nearest one of those is left out with it.
``metabolites`` lists the layout's glyphs classed as reactions that sit exactly
where Metabolic Atlas draws a metabolite and no reaction. Both arose in layouts
drawn from older Human-maps, whose metabolites outside Human1 (``Rtotal_c``) or
with Recon3D identifiers they drew otherwise.

Metabolic Atlas's transport maps, one per membrane, are packaged and read as
compartment maps, from Human-maps' ``subsystem/svg`` files (:func:`svg_kind`).

A compartment map also takes Metabolic Atlas's labels for its metabolites:
``labels`` gives, under the layout glyph drawn where Metabolic Atlas draws the
metabolite, its name's ``lines``, where their centre sits from the metabolite
(``dx`` and ``dy``, given when not on it: a small metabolite's name stands beside
its dot), and ``dot`` where Metabolic Atlas draws it as a small dot rather than an
oval.
Usage::

    python -I scripts/import_atlas_paths.py HUMAN_MAPS [--check]

``HUMAN_MAPS`` holds the Human-maps repository at the commit
:data:`metabis.viz.atlas_paths.ATLAS_COMMIT`, whose ``compartment/svg`` and
``subsystem/svg`` files the layouts were built from (``import_human_maps.py``).
The files are parsed as text and nothing in them runs. ``--check`` reports what it
would write, changing nothing.

The SVG files declare the CC BY-SA 4.0 licence in their metadata, so the files
written here are shared under it too (THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from metabis.viz.atlas_paths import ATLAS_COMMIT, ATLAS_SUFFIX

GRAPHS = Path(__file__).resolve().parent.parent / "src/metabis/resources/graphs"
KINDS = ("compartment", "subsystem")
SOURCE = f"https://github.com/SysBioChalmers/Human-maps/blob/{ATLAS_COMMIT}/{{kind}}/svg/{{name}}.svg"
# Metabolic Atlas files its transport maps, one per membrane, with its subsystem
# maps; the library draws them as maps, beside the compartments'.
TRANSPORT_PREFIX = "transport_"
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
GROUP_TAG = re.compile(r"<g\b|</g>")
# A small metabolite's shape: a dot, where a main one's is an oval.
DOT = re.compile(r"<circle\b")
TEXT = re.compile(r"<text([^>]*)>\s*(.*?)\s*</text>", re.S)
TEXT_X = re.compile(r'\bx="([-\d.]+)"')
TEXT_Y = re.compile(r'\by="([-\d.]+)"')
# Where Metabolic Atlas sets a name's baseline below its centre: its 18-unit text
# on one line, centred on the metabolite, has its baseline 6 units down.
LABEL_BASELINE = 6.0

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


def _group_body(svg: str, start: int) -> str:
    """The text of the ``<g>`` element whose opening tag begins before ``start``, to its end."""
    depth = 1
    for tag in GROUP_TAG.finditer(svg, start):
        depth += 1 if tag.group() == "<g" else -1
        if depth == 0:
            return svg[start : tag.start()]
    return svg[start:]


def metabolite_labels(
    svg: str, offset: Point, glyphs: List[dict], drawn_as_metabolites: Iterable[str] = ()
) -> Dict[str, dict]:
    """Where Metabolic Atlas sets each metabolite's name, under the layout glyph drawn there.

    The layout's metabolites, and the glyphs it classes as reactions that the map
    draws as metabolites, take the name Metabolic Atlas sets where they are: its
    ``lines``, and ``dx`` and ``dy`` placing their centre from the metabolite,
    given when it is not on it. ``dot`` marks a metabolite Metabolic Atlas draws as
    a small dot.
    """
    also = set(drawn_as_metabolites)
    by_place: Dict[Tuple[int, int], List[Tuple[str, Point]]] = defaultdict(list)
    for glyph in glyphs:
        if glyph["glyph_class"] == "metabolite" or str(glyph["id"]) in also:
            by_place[(round(glyph["x"]), round(glyph["y"]))].append((str(glyph["id"]), (glyph["x"], glyph["y"])))
    labels: Dict[str, dict] = {}
    for found in METABOLITE.finditer(svg):
        x, y = float(found.group(2)) + offset[0], float(found.group(3)) + offset[1]
        drawn = [
            glyph_id
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for glyph_id, place in by_place.get((round(x) + dx, round(y) + dy), ())
            if _near(place, [(x, y)])
        ]
        body = _group_body(svg, found.end())
        texts = TEXT.findall(body)
        if len(drawn) != 1 or not texts:
            continue
        lines = [" ".join(html.unescape(text).split()) for _attributes, text in texts]
        across = TEXT_X.search(texts[0][0])
        downs = [float(match.group(1)) for attributes, _text in texts if (match := TEXT_Y.search(attributes))]
        label = {"lines": lines}
        if across and float(across.group(1)):
            label["dx"] = round(float(across.group(1)), 1)
        if downs and abs(statistics.mean(downs) - LABEL_BASELINE) >= 0.05:
            label["dy"] = round(statistics.mean(downs) - LABEL_BASELINE, 1)
        if DOT.search(body):
            label["dot"] = True
        labels[drawn[0]] = label
    return labels


def svg_kind(kind: str, name: str) -> str:
    """The Human-maps folder that holds the SVG file of the packaged ``kind`` map ``name``."""
    return "subsystem" if kind == "compartment" and name.startswith(TRANSPORT_PREFIX) else kind


def import_map(svg_path: Path, layout_path: Path, kind: str) -> Tuple[dict, dict]:
    """The map's Atlas paths in its layout's frame, and a report of what was kept."""
    name = layout_path.stem
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
    atlas_metabolites = [shifted((float(x), float(y))) for _classes, x, y in METABOLITE.findall(svg)]
    atlas_reactions = [shifted(place) for places in atlas.values() for place in places]

    # Glyphs the layout classes as reactions that Metabolic Atlas draws as metabolites.
    drawn_as_metabolites = sorted(
        reaction
        for reaction, places in ours.items()
        if all(_near(place, atlas_metabolites) and not _near(place, atlas_reactions) for place in places)
    )
    for reaction in drawn_as_metabolites:
        metabolites.extend(ours.pop(reaction))

    # A reaction named otherwise in the layout takes the paths of the one reaction
    # Metabolic Atlas draws exactly where the layout draws it.
    single = {reaction: shifted(places[0]) for reaction, places in atlas.items() if len(places) == 1}
    alias: Dict[str, str] = {}
    for reaction, places in ours.items():
        if reaction in atlas:
            continue
        found = {
            other
            for place in places
            for other, there in single.items()
            if other not in ours and _near(place, [there])
        }
        if len(found) == 1 and len(places) == 1:
            alias[found.pop()] = reaction

    # A reaction keeps its paths where it sits exactly where Metabolic Atlas draws it.
    misplaced = {
        reaction
        for reaction, places in atlas.items()
        if reaction not in alias
        and (reaction not in ours or not all(_near(shifted(place), ours[reaction]) for place in places))
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
                target[alias.get(reaction, reaction)].append(_flat(line, offset))
            else:
                dropped += 1

    subsystems: Dict[str, List[List[float]]] = {}
    labels: Dict[str, dict] = {}
    if kind == "compartment":
        # A transport map's one band, Human-GEM's "Transport reactions", lies under
        # all of it, and the model divides that subsystem by compartment
        # (loaders.transport_part): it names no subsystem a map can highlight.
        if not name.startswith(TRANSPORT_PREFIX):
            for band, d in BAND.findall(svg):
                subsystems[band] = [_flat(line, offset) for line in polylines(d)]
        labels = metabolite_labels(svg, offset, glyphs, drawn_as_metabolites)

    data = {
        "source": SOURCE.format(kind=svg_kind(kind, name), name=name),
        "license": LICENSE,
        "offset": list(offset),
        "links": dict(sorted(links.items())),
        "heads": dict(sorted(heads.items())),
        "subsystems": subsystems,
        "metabolites": drawn_as_metabolites,
    }
    if kind == "compartment":
        data["labels"] = labels
    report = {
        "map": f"{kind}/{name}",
        "reactions": len(ours),
        "routed": len(links),
        "routed_by_place": len(set(alias.values()) & set(links)),
        "kept_straight": len(set(ours) - set(links)),
        "dropped": dropped,
        "subsystems": len(subsystems),
        "metabolites": len(drawn_as_metabolites),
        "labels": len(labels),
    }
    return data, report


def describe(report: dict) -> str:
    """What :func:`import_map` kept of a map, on one line."""
    return (
        f"{report['map']}: {report['routed']} of {report['reactions']} reactions routed "
        f"({report['routed_by_place']} by place), "
        f"{report['dropped']} paths to undrawn metabolites left out, "
        f"{report['metabolites']} metabolites drawn as reactions, "
        f"{report['subsystems']} subsystem bands, {report['labels']} names placed"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("human_maps", type=Path)
    parser.add_argument("--check", action="store_true", help="report, writing nothing")
    args = parser.parse_args(argv)

    for kind in KINDS:
        for layout_path in sorted((GRAPHS / kind / "human1").glob("*.json")):
            if layout_path.name.endswith(ATLAS_SUFFIX):
                continue
            svg_path = args.human_maps / svg_kind(kind, layout_path.stem) / "svg" / f"{layout_path.stem}.svg"
            if not svg_path.exists():
                print(f"{kind}/{layout_path.stem}: no SVG", file=sys.stderr)
                continue
            data, report = import_map(svg_path, layout_path, kind)
            target = layout_path.with_name(f"{layout_path.stem}{ATLAS_SUFFIX}")
            text = json.dumps(data, separators=(",", ":"))
            print(f"{describe(report)}, {len(text) / 1e6:.2f} MB")
            if not args.check:
                target.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

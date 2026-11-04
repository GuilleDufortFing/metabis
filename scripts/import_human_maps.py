"""Build the packaged Human1 maps from Metabolic Atlas's Human-maps.

The compartment and subsystem maps under ``resources/graphs/compartment/human1``
and ``resources/graphs/subsystem/human1`` are Metabolic Atlas's Human-maps
(github.com/SysBioChalmers/Human-maps), read from its SBGN and SVG files at the
commit :data:`metabis.viz.atlas_paths.ATLAS_COMMIT`. For each map this script
writes the layout ``<map>.json``, then, with ``import_atlas_paths.py``, the links,
arrowheads, subsystem bands and names beside it, ``<map>.atlas.json``.

A layout places everything where Metabolic Atlas draws it, in its units:

* **Reactions** are the SBGN file's processes, named by their Human-GEM
  identifiers, a glyph for each place the SVG draws one: Metabolic Atlas draws
  some twice, and links each drawing's metabolites and genes to the one beside
  them.
* **Metabolites** are its simple chemicals, a glyph per place. Each takes the
  Human-GEM identifier of the SVG metabolite drawn at that place, and its shape:
  a small dot is ``intermediate``, an oval ``main``.
* **Genes** are its macromolecules. Each takes the Ensembl identifier of the SVG
  gene box drawn at that place, which also names its reaction; a reaction's boxes
  that touch form one enzyme group, linked to the reaction.
* **Compartments** are the SVG's outlines; the SBGN files leave some out, such as
  the mitochondrial inner membrane.
* **Links** run from a reactant to its reaction and from the reaction to a
  product: the side of the process's port the SBGN arc meets, its first port for
  reactants. The class of the arc does not say it, as the SBGN files draw a
  reversible reaction's reactants as products too.

Which maps: every compartment map the library lists (``COMPARTMENT_GRAPH_MANIFEST``),
Metabolic Atlas's transport maps of each membrane among them, and every subsystem
map whose file is named as the graph card looks a subsystem's map up: by the name
of a subsystem of the packaged model (``reaction_catalog.csv``).

Usage::

    python -I scripts/import_human_maps.py HUMAN_MAPS [--check]

``HUMAN_MAPS`` holds the repository at that commit, as ``git clone`` and
``git checkout`` leave it, or as its archive unpacks
(``https://codeload.github.com/SysBioChalmers/Human-maps/tar.gz/<commit>``). The
files are parsed as data and nothing in them runs. ``--check`` reports what it
would write, changing nothing.

The SVG files declare the CC BY-SA 4.0 licence in their metadata, so the files
written here are shared under it too (THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import re
import sys
import xml.etree.ElementTree as ElementTree
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from metabis.viz.metabolic_visualization.generation.utilities.cache_utils import (
    _normalize_element_name,
    save_visualization_cache,
)
from metabis.viz.metabolic_visualization.generation.utilities.compartment_resources import (
    COMPARTMENT_GRAPH_MANIFEST,
)

HERE = Path(__file__).resolve().parent
GRAPHS = HERE.parent / "src/metabis/resources/graphs"
CATALOG = HERE.parent / "src/metabis/resources/reference_model/reactions/reaction_catalog.csv"
SBGN = {"s": "http://sbgn.org/libsbgn/0.3"}

METABOLITE = re.compile(r'<g class="met ([^"]+)" transform="matrix\(1,0,0,1,([-\d.]+),([-\d.]+)\)"')
GENE = re.compile(
    r'<g class="enz (\S+) (\S+)">\s*<path class="shape"[^>]*'
    r'transform="matrix\(1,0,0,1,([-\d.]+),([-\d.]+)\)"'
)
COMPARTMENT = re.compile(r'<g id="([^"]+)" class="compartment">')
RECT = re.compile(r'<rect class="shape" x="([-\d.]+)" y="([-\d.]+)" width="([\d.]+)" height="([\d.]+)"')
OUTLINE = re.compile(r'<path [^>]*\bd="([^"]*)"')
NAME = re.compile(r"<text([^>]*)>\s*(.*?)\s*</text>", re.S)
MOVED = re.compile(r'transform="matrix\(1,0,0,1,([-\d.]+),([-\d.]+)\)"')
TEXT_X = re.compile(r'\bx="([-\d.]+)"')
TEXT_Y = re.compile(r'\by="([-\d.]+)"')
FONT_SIZE = re.compile(r'font-size="([\d.]+)"')
MIDDLE = 'text-anchor="middle"'
# A bold sans-serif letter's width, on average, in its font size.
CHARACTER_WIDTH = 0.6
DOT = re.compile(r"<circle\b")
# The name Metabolic Atlas's fitting script gives a compartment it adds for
# reactions it adds to a map; its box is labelled without it.
ADDED = " (added)"
# A reaction as the layouts have always sized it; the renderer draws every reaction
# at one size, and these only bound the plot.
REACTION_SIZE = 50.0
ENZYME_PADDING = 5.0
# How far apart, in map units, two gene boxes may be and still touch.
TOUCHING = 2.0

Place = Tuple[float, float]


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paths = _sibling("import_atlas_paths")


def _place(x: float, y: float) -> Place:
    return round(x, 1), round(y, 1)


def _bbox(glyph) -> Tuple[float, float, float, float]:
    box = glyph.find("s:bbox", SBGN)
    return tuple(float(box.get(key)) for key in ("x", "y", "w", "h"))


def _name_corner(body: str) -> Optional[Place]:
    """About where the top-left corner of a compartment's name falls, as Metabolic Atlas sets it."""
    texts = list(NAME.finditer(body))
    if not texts:
        return None
    size = float(FONT_SIZE.search(body).group(1)) if FONT_SIZE.search(body) else 72.0
    lefts, tops = [], []
    for text in texts:
        attributes = text.group(1)
        moved = MOVED.search(attributes)
        x = float(moved.group(1)) if moved else 0.0
        y = float(moved.group(2)) if moved else 0.0
        x += float(TEXT_X.search(attributes).group(1)) if TEXT_X.search(attributes) else 0.0
        y += float(TEXT_Y.search(attributes).group(1)) if TEXT_Y.search(attributes) else 0.0
        width = CHARACTER_WIDTH * size * len(" ".join(text.group(2).split()))
        lefts.append(x - width / 2 if MIDDLE in attributes or MIDDLE in body else x)
        tops.append(y - size)
    return min(lefts), min(tops)


def compartments(svg: str) -> List[dict]:
    """The SVG's compartments, as boxes named by their labels.

    A compartment is a box (a rounded rectangle), the line or lines along its
    membrane, or only its name. Each is the box that holds its outline: a line's
    box reaches out to the name beside it, and a name alone is a box of no size
    where the name starts, as the renderer hangs a name from its box's corner.
    """
    found = []
    for index, match in enumerate(COMPARTMENT.finditer(svg), start=1):
        body = paths._group_body(svg, match.end())
        rect = RECT.search(body)
        outline = OUTLINE.search(body)
        # An outline of a single point (one map has one) draws nothing.
        points = [point for line in paths.polylines(outline.group(1)) for point in line] if outline else []
        corner = _name_corner(body)
        if rect:
            x, y, width, height = (float(value) for value in rect.groups())
        elif points:
            across = max(p[0] for p in points) - min(p[0] for p in points)
            down = max(p[1] for p in points) - min(p[1] for p in points)
            if corner and min(across, down) < 1:
                points.append(corner)
            x, y = min(p[0] for p in points), min(p[1] for p in points)
            width, height = max(p[0] for p in points) - x, max(p[1] for p in points) - y
        elif corner:
            (x, y), width, height = corner, 0.0, 0.0
        else:
            continue
        words = [" ".join(html.unescape(text.group(2)).split()) for text in NAME.finditer(body)]
        name = " ".join(words) or match.group(1).removesuffix(ADDED)
        found.append(
            {
                "id": f"compartment{index}",
                "glyph_class": "compartment",
                "label": name,
                "x": round(x + width / 2, 1),
                "y": round(y + height / 2, 1),
                "width": round(width, 1),
                "height": round(height, 1),
            }
        )
    return found


def _enzyme_groups(genes: List[dict], reaction_of: Dict[str, str]) -> Tuple[List[dict], List[dict]]:
    """A reaction's gene boxes that touch, as one enzyme group each, linked to the reaction."""
    by_reaction: Dict[str, List[dict]] = defaultdict(list)
    for gene in genes:
        by_reaction[reaction_of[gene["id"]]].append(gene)

    def touch(a: dict, b: dict) -> bool:
        return abs(a["x"] - b["x"]) <= (a["width"] + b["width"]) / 2 + TOUCHING and abs(a["y"] - b["y"]) <= (
            a["height"] + b["height"]
        ) / 2 + TOUCHING

    groups: List[dict] = []
    links: List[dict] = []
    for reaction, boxes in by_reaction.items():
        unseen = list(boxes)
        while unseen:
            members = [unseen.pop(0)]
            for member in members:
                near = [box for box in unseen if touch(member, box)]
                members.extend(near)
                unseen = [box for box in unseen if box not in near]
            members.sort(key=lambda box: (box["y"], box["x"]))
            left = min(box["x"] - box["width"] / 2 for box in members) - ENZYME_PADDING
            right = max(box["x"] + box["width"] / 2 for box in members) + ENZYME_PADDING
            top = min(box["y"] - box["height"] / 2 for box in members) - ENZYME_PADDING
            bottom = max(box["y"] + box["height"] / 2 for box in members) + ENZYME_PADDING
            group_id = f"enzyme_group_{reaction}_{len(groups)}"
            groups.append(
                {
                    "id": group_id,
                    "glyph_class": "enzyme_group",
                    "label": f"Enzymes for {reaction}",
                    "x": round((left + right) / 2, 1),
                    "y": round((top + bottom) / 2, 1),
                    "width": round(right - left, 1),
                    "height": round(bottom - top, 1),
                    "reaction_id": reaction,
                    "enzymes": [box["label"] for box in members],
                    "glyph_ids": [box["id"] for box in members],
                    "outline_color": "#7B68EE",
                    "outline_width": 2,
                    "outline_dash": [3, 3],
                }
            )
            links.append(
                {
                    "id": f"regulatory_{group_id}_{reaction}",
                    "connection_class": "regulatory",
                    "source": group_id,
                    "target": reaction,
                    "stoichiometry": None,
                }
            )
    return groups, links


def build_layout(sbgn_path: Path, svg_path: Path) -> Tuple[List[dict], List[dict], dict]:
    """A map's glyphs and connections, and a report of what could not be placed."""
    svg = svg_path.read_text()
    root = ElementTree.parse(sbgn_path).getroot()

    metabolite_at: Dict[Place, Tuple[str, bool]] = {}
    for match in METABOLITE.finditer(svg):
        dot = bool(DOT.search(paths._group_body(svg, match.end())))
        metabolite_at[_place(float(match.group(2)), float(match.group(3)))] = (match.group(1).split()[0], dot)
    # Metabolic Atlas draws some reactions twice; the SBGN file has one process
    # for both drawings, linked to the metabolites of each.
    reaction_places: Dict[str, List[Place]] = defaultdict(list)
    for reaction, x, y in paths.REACTION.findall(svg):
        reaction_places[reaction].append(_place(float(x), float(y)))
    gene_at: Dict[Place, Tuple[str, str]] = {
        _place(float(x), float(y)): (gene, reaction) for gene, reaction, x, y in GENE.findall(svg)
    }

    glyphs = compartments(svg)
    metabolites: List[dict] = []
    reactions: List[dict] = []
    genes: List[dict] = []
    reaction_of: Dict[str, str] = {}
    process_of_port: Dict[str, Tuple[str, str]] = {}
    unplaced: List[str] = []
    for glyph in root.iter(f"{{{SBGN['s']}}}glyph"):
        kind = glyph.get("class")
        if kind == "compartment":
            continue
        x, y, width, height = _bbox(glyph)
        centre = _place(x + width / 2, y + height / 2)
        if kind == "process":
            reaction = glyph.find("s:label", SBGN).get("text")
            for place in reaction_places.get(reaction) or [centre]:
                reactions.append(
                    {
                        "id": reaction,
                        "glyph_class": "reaction",
                        "label": reaction,
                        "x": place[0],
                        "y": place[1],
                        "width": REACTION_SIZE,
                        "height": REACTION_SIZE,
                        "original_id": glyph.get("id"),
                    }
                )
            for port in glyph.findall("s:port", SBGN):
                process_of_port[port.get("id")] = (reaction, port.get("id").rsplit(".", 1)[1])
        elif kind == "simple chemical":
            if centre not in metabolite_at:
                unplaced.append(glyph.get("id"))
                continue
            metabolite, dot = metabolite_at[centre]
            metabolites.append(
                {
                    "id": glyph.get("id"),
                    "glyph_class": "metabolite",
                    "label": metabolite,
                    "x": centre[0],
                    "y": centre[1],
                    "width": width,
                    "height": height,
                    "original_id": glyph.get("id"),
                    "subtype": "intermediate" if dot else "main",
                }
            )
        elif kind == "macromolecule":
            corner = _place(x, y)
            if corner not in gene_at:
                unplaced.append(glyph.get("id"))
                continue
            gene, reaction = gene_at[corner]
            reaction_of[glyph.get("id")] = reaction
            genes.append(
                {
                    "id": glyph.get("id"),
                    "glyph_class": "gene",
                    "label": gene,
                    "x": centre[0],
                    "y": centre[1],
                    "width": width,
                    "height": height,
                    "associated_reaction": None,
                    "enzyme_group": None,
                }
            )

    drawn = {glyph["id"] for glyph in metabolites}
    connections: List[dict] = []
    for arc in root.iter(f"{{{SBGN['s']}}}arc"):
        if arc.get("class") not in ("consumption", "production"):
            continue
        source, target = arc.get("source"), arc.get("target")
        port, metabolite = (source, target) if source in process_of_port else (target, source)
        if port not in process_of_port or metabolite not in drawn:
            continue
        reaction, side = process_of_port[port]
        connections.append(
            {
                "id": arc.get("id"),
                "connection_class": "metabolic",
                "source": metabolite if side == "1" else reaction,
                "target": reaction if side == "1" else metabolite,
                "stoichiometry": None,
            }
        )

    groups, regulatory = _enzyme_groups(genes, reaction_of)
    glyphs = glyphs + metabolites + reactions + genes + groups
    report = {
        "reactions": len({glyph["id"] for glyph in reactions}),
        "drawn_twice": len(reactions) - len({glyph["id"] for glyph in reactions}),
        "metabolites": len(metabolites),
        "genes": len(genes),
        "enzyme_groups": len(groups),
        "compartments": len(glyphs) - len(metabolites) - len(reactions) - len(genes) - len(groups),
        "unplaced": unplaced,
    }
    return glyphs, connections + regulatory, report


def subsystem_maps() -> Dict[str, str]:
    """The packaged model's subsystems, by the file name a map of each would have."""
    with CATALOG.open(newline="") as handle:
        names = {row["SUBSYSTEM"] for row in csv.DictReader(handle) if row["SUBSYSTEM"]}
    return {_normalize_element_name("subsystem", name): name for name in names}


def maps(human_maps: Path) -> List[Tuple[str, str]]:
    """Each map to build, as (kind, name), of those the checkout holds.

    ``kind`` is the packaged folder it is written to; a transport map's files
    are among Human-maps' subsystem maps (import_atlas_paths.svg_kind).
    """
    wanted = {"compartment": set(COMPARTMENT_GRAPH_MANIFEST), "subsystem": set(subsystem_maps())}
    return [
        (kind, name)
        for kind in ("compartment", "subsystem")
        for name in sorted(wanted[kind])
        if (human_maps / paths.svg_kind(kind, name) / "svg" / f"{name}.svg").exists()
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("human_maps", type=Path)
    parser.add_argument("--check", action="store_true", help="report, writing nothing")
    args = parser.parse_args(argv)

    for kind, name in maps(args.human_maps):
        source = args.human_maps / paths.svg_kind(kind, name)
        sbgn_path = source / "sbgn" / f"{name}.sbgn"
        svg_path = source / "svg" / f"{name}.svg"
        glyphs, connections, report = build_layout(sbgn_path, svg_path)
        layout_path = GRAPHS / kind / "human1" / f"{name}.json"
        print(
            f"{kind}/{name}: {report['reactions']} reactions, {report['metabolites']} metabolites, "
            f"{report['genes']} genes in {report['enzyme_groups']} enzyme groups, "
            f"{report['compartments']} compartments"
            + (f", {report['drawn_twice']} reactions drawn again" if report["drawn_twice"] else "")
            + (f", {len(report['unplaced'])} glyphs the SVG does not draw" if report["unplaced"] else "")
        )
        if args.check:
            continue
        save_visualization_cache(str(layout_path), glyphs, connections)
        data, atlas_report = paths.import_map(svg_path, layout_path, kind)
        layout_path.with_name(f"{name}{paths.ATLAS_SUFFIX}").write_text(
            json.dumps(data, separators=(",", ":")) + "\n"
        )
        print(f"  {paths.describe(atlas_report)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

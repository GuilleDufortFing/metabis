"""The metabolic network's overview layout: categories in a fixed order, no overlaps.

The packaged force layout places each subsystem near the subsystems it shares
metabolites with, but its nodes were placed as dots: drawn at a size that shows
their reactions, they overlap, and its category positions carry no meaning. This
module keeps the first and replaces the second.

* **Size.** A node's area is proportional to the reactions its subsystem contains
  in the model, as the GigaScience note describes the network ("a size proportional
  to the number of reactions it contains"). It does not depend on the dataset, so
  the layout never moves between datasets.
* **No overlaps.** Each category starts from the packaged positions of its
  subsystems, then its nodes are pushed apart until every pair clears ``NODE_GAP``,
  while linked nodes are held near each other and the whole category is drawn
  towards its centre. The gap keeps every shared-metabolite edge visible.
* **A fixed order.** Categories follow the top level of KEGG's metabolism classes,
  as a biochemistry textbook reads them (carbohydrate and energy; amino acids and
  nucleotides; lipids; glycans; cofactors, redox and xenobiotics; proteins), with
  transport and the model's bookkeeping last, since they are not pathways. An
  uploaded category the order does not know goes after the known ones, by name.
* **Current categories.** Membership comes from the model, so uploaded categories
  get a layout of their own; edges are the packaged ones whose two subsystems now
  share a category.

Coordinates stay in the packaged layout's units, about 10,000 across, which the
label fading of :mod:`..rendering.js_callbacks` is tuned to. ``y`` grows downwards,
as the overview's flipped y-range draws it.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

NODE_RADIUS_PER_ROOT_REACTION = 15.0
NODE_MIN_RADIUS = 25.0
NODE_GAP = 90.0
CATEGORY_PADDING = 80.0
CATEGORY_GAP = 220.0
# A category's name keeps its size on screen while the layout scales with the
# overview's column. At its usual width, about 600 px for these 10,000 units, a cell
# of 2,000 units is 120 px: room for 16 characters of 11 px text a line, and the
# band below it for four such lines.
CATEGORY_MIN_CELL = 2000.0
CATEGORY_LABEL_BAND = 1000.0
ROW_WIDTH = 10000.0
RELAXATION_ROUNDS = 600

# Textbook order, after the top level of KEGG's metabolism classes.
CATEGORY_ORDER: Tuple[str, ...] = (
    # Carbohydrate and energy
    "Central Carbon & Bioenergetics",
    "Carbohydrate Interconversions",
    # Amino acids and nucleotides
    "Amino Acid Metabolism",
    "Sulfur Amino Acids & One-Carbon Metabolism",
    "Nucleotide Metabolism",
    # Lipids
    "Fatty Acid Biosynthesis & Activation",
    "Fatty Acid Oxidation",
    "Short-Chain Fatty Acids",
    "Polyunsaturated Fatty Acids & Eicosanoids",
    "Membrane Lipids & Phospholipid Signaling",
    "Sterols, Steroid Hormones & Bile Acids",
    "Fat-Soluble Vitamins & Isoprenoids",
    # Glycans
    "Glycans & Glycosylation",
    # Cofactors, redox and xenobiotics
    "Vitamins, Cofactors & Heme",
    "Redox & Antioxidant Defense",
    "Xenobiotics & Detoxification",
    # Proteins
    "Protein Processing & Turnover",
)
# Not pathways: set apart, last.
CATEGORIES_SET_APART: Tuple[str, ...] = (
    "Membrane Transport & Exchange",
    "Model Utilities & Non-human Pathways",
)


def category_sort_key(name: str) -> Tuple[int, int, str]:
    """Known categories in textbook order, then unknown ones by name, then the set-apart."""
    if name in CATEGORY_ORDER:
        return (0, CATEGORY_ORDER.index(name), "")
    if name in CATEGORIES_SET_APART:
        return (2, CATEGORIES_SET_APART.index(name), "")
    return (1, 0, name.lower())


def node_radius(reaction_count: int) -> float:
    """A radius whose area is proportional to the reactions, never under the minimum."""
    return max(
        NODE_MIN_RADIUS,
        NODE_RADIUS_PER_ROOT_REACTION * math.sqrt(max(int(reaction_count or 0), 0)),
    )


def category_glyph_id(name: str) -> str:
    """The packaged layout's identifier for a category boundary."""
    return f"category_{name.lower().replace(' ', '_')}"


def relax_category(
    seeds: np.ndarray,
    radii: np.ndarray,
    links: Sequence[Tuple[int, int]],
    gap: float = NODE_GAP,
    rounds: int = RELAXATION_ROUNDS,
) -> Tuple[np.ndarray, float]:
    """Positions for one category's nodes, centred on its enclosing circle, and its radius.

    No two nodes end closer than ``gap`` between their edges. The seeds give the
    arrangement; they are rescaled so the category's spread matches its nodes' area.
    """
    count = len(radii)
    if count == 0:
        return np.zeros((0, 2)), 0.0
    r = np.asarray(radii, dtype=float)
    if count == 1:
        return np.zeros((1, 2)), float(r[0])

    pos = np.asarray(seeds, dtype=float).copy()
    pos -= pos.mean(axis=0)
    spread = math.sqrt(float((pos**2).sum(axis=1).mean())) or 1.0
    pos *= 1.2 * math.sqrt(float((r**2).sum())) / spread
    # A fixed nudge separates coincident seeds; seeded, so the layout is reproducible.
    pos += np.random.default_rng(7).normal(0.0, 1e-3, pos.shape)

    area = r**2
    links = [(a, b) for a, b in links if a != b]
    sources = np.array([a for a, _ in links], dtype=int)
    targets = np.array([b for _, b in links], dtype=int)
    off_diagonal = ~np.eye(count, dtype=bool)
    # Each node of a clashing pair moves by the other's share of their areas.
    share = area[None, :] / (area[:, None] + area[None, :])

    def separate() -> float:
        difference = pos[None, :, :] - pos[:, None, :]  # from row node to column node
        distance = np.hypot(difference[..., 0], difference[..., 1])
        # Half a unit over the gap, so what the relaxation leaves still clears it.
        need = r[:, None] + r[None, :] + gap + 0.5
        clash = (distance < need) & off_diagonal
        if not clash.any():
            return 0.0
        safe = np.where(distance > 1e-9, distance, 1.0)
        unit = difference / safe[..., None]
        overlap = np.where(clash, need - distance, 0.0)
        move = -(unit * (overlap * share)[..., None]).sum(axis=1)
        pos[:] += 0.5 * move
        return float(overlap.max())

    for step in range(rounds):
        if step < rounds * 0.75:
            centre = (pos * area[:, None]).sum(axis=0) / area.sum()
            pos[:] = centre + (pos - centre) * 0.99
            if len(links):
                difference = pos[targets] - pos[sources]
                distance = np.hypot(difference[:, 0], difference[:, 1])
                want = r[sources] + r[targets] + 2 * gap
                stretched = distance > want
                pull = np.zeros_like(difference)
                pull[stretched] = (
                    0.03
                    * ((distance - want) / np.maximum(distance, 1e-9))[stretched, None]
                    * difference[stretched]
                )
                np.add.at(pos, sources, pull)
                np.add.at(pos, targets, -pull)
        separate()
    # Settle: collisions only, until none is left.
    for _ in range(rounds):
        if separate() < 1e-6:
            break

    centre = _enclosing_centre(pos, r)
    pos -= centre
    radius = float((np.hypot(pos[:, 0], pos[:, 1]) + r).max())
    return pos, radius


def _enclosing_centre(pos: np.ndarray, r: np.ndarray, rounds: int = 200) -> np.ndarray:
    """An approximate centre of the smallest circle enclosing every node (Badoiu-Clarkson)."""
    lo = (pos - r[:, None]).min(axis=0)
    hi = (pos + r[:, None]).max(axis=0)
    centre = (lo + hi) / 2
    for step in range(1, rounds + 1):
        reach = np.hypot(*(pos - centre).T) + r
        far = int(np.argmax(reach))
        direction = pos[far] - centre
        length = math.hypot(*direction) or 1.0
        target = pos[far] + direction / length * r[far]
        centre = centre + (target - centre) / (step + 1)
    return centre


@lru_cache(maxsize=8)
def _spaced_layout(
    members: Tuple[Tuple[str, str, float, float, int], ...],
    links: Tuple[Tuple[str, str], ...],
) -> Tuple[Dict[str, Tuple[float, float, float]], Tuple[Tuple[str, float, float, float, int], ...]]:
    """Node positions and radii, and category circles, for one category assignment.

    ``members`` holds (subsystem, category, seed x, seed y, reactions) and ``links``
    the (source, target) pairs inside a category. Cached: the default categories
    are laid out once per process.
    """
    by_category: Dict[str, List[Tuple[str, float, float, int]]] = {}
    for subsystem, category, x, y, reactions in members:
        by_category.setdefault(category, []).append((subsystem, x, y, reactions))

    cells = []
    for category in sorted(by_category, key=category_sort_key):
        rows = by_category[category]
        ids = [row[0] for row in rows]
        index = {subsystem: i for i, subsystem in enumerate(ids)}
        seeds = np.array([[row[1], row[2]] for row in rows], dtype=float)
        radii = np.array([node_radius(row[3]) for row in rows], dtype=float)
        pairs = [
            (index[a], index[b]) for a, b in links if a in index and b in index
        ]
        positions, radius = relax_category(seeds, radii, pairs)
        cells.append((category, ids, positions, radii, radius + CATEGORY_PADDING))

    # Categories left to right in rows, each row centred, wrapping at ROW_WIDTH.
    rows_of_cells: List[List[Tuple[Any, float]]] = [[]]
    width = 0.0
    for cell in cells:
        size = max(2 * cell[4], CATEGORY_MIN_CELL) + CATEGORY_GAP
        if rows_of_cells[-1] and width + size > ROW_WIDTH:
            rows_of_cells.append([])
            width = 0.0
        rows_of_cells[-1].append((cell, size))
        width += size

    nodes: Dict[str, Tuple[float, float, float]] = {}
    circles = []
    top = 0.0
    for row in rows_of_cells:
        height = max(2 * cell[4] for cell, _ in row)
        x = (ROW_WIDTH - sum(size for _, size in row)) / 2
        for (category, ids, positions, radii, radius), size in row:
            cx, cy = x + size / 2, top + height / 2
            for subsystem, (dx, dy), node in zip(ids, positions, radii):
                nodes[subsystem] = (float(cx + dx), float(cy + dy), float(node))
            circles.append((category, float(cx), float(cy), float(radius), len(ids)))
            x += size
        top += height + CATEGORY_LABEL_BAND + CATEGORY_GAP
    return nodes, tuple(circles)


def space_network(
    glyphs: Iterable[Mapping[str, Any]],
    connections: Iterable[Mapping[str, Any]],
    category_of: Mapping[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """The packaged network, laid out for the current categories.

    Args:
        glyphs: The packaged glyphs; subsystem glyphs carry their seed position
            and ``reaction_count``. Its category boundaries are replaced.
        connections: The packaged shared-metabolite edges.
        category_of: Each subsystem's current category.

    Returns:
        Each new category boundary followed by its subsystem glyphs, at their new
        positions and with a ``radius``, as the packaged layout orders them; and
        the edges inside a category.
    """
    subsystems = [
        dict(glyph)
        for glyph in glyphs
        if glyph.get("glyph_class") == "subsystem" and glyph.get("id") in category_of
    ]
    same_category = [
        dict(connection)
        for connection in connections
        if category_of.get(connection.get("source")) is not None
        and category_of.get(connection.get("source"))
        == category_of.get(connection.get("target"))
    ]
    members = tuple(
        sorted(
            (
                glyph["id"],
                category_of[glyph["id"]],
                round(float(glyph.get("x", 0.0)), 3),
                round(float(glyph.get("y", 0.0)), 3),
                int(glyph.get("reaction_count", 0) or 0),
            )
            for glyph in subsystems
        )
    )
    links = tuple(sorted((c["source"], c["target"]) for c in same_category))
    nodes, circles = _spaced_layout(members, links)

    for glyph in subsystems:
        x, y, radius = nodes[glyph["id"]]
        glyph.update(x=x, y=y, radius=radius, width=2 * radius, height=2 * radius)
    ordered: List[Dict[str, Any]] = []
    for category, x, y, radius, count in circles:
        ordered.append(
            {
                "id": category_glyph_id(category),
                "glyph_class": "subsystem_category",
                "label": category,
                "x": x,
                "y": y,
                "radius": radius,
                "width": 2 * radius,
                "height": 2 * radius,
                "subsystem_count": count,
            }
        )
        ordered.extend(glyph for glyph in subsystems if category_of[glyph["id"]] == category)
    return ordered, same_category

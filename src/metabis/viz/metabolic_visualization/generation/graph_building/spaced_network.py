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
* **A fixed order, by number.** Categories follow the top level of KEGG's
  metabolism classes, as a biochemistry textbook reads them (carbohydrate and
  energy; amino acids and nucleotides; lipids; glycans; cofactors, redox and
  xenobiotics; proteins), with transport and the model's bookkeeping last, since
  they are not pathways. An uploaded category the order does not know goes after
  the known ones, by name. Each category carries its place in this order as a
  number on its rim, and the subsystem rankings head each category with the same
  number and its name, so the network needs no band of names under its circles.
* **Packed.** The pathways' circles pack on the left, each in turn rising to the
  highest spot free above it, so small categories fill the gaps beside large ones;
  the set-apart categories stack in a column on the right. The packing's width is
  the one whose drawing is closest to ``TARGET_ASPECT``.
* **Current categories.** Membership comes from the model, so uploaded categories
  get a layout of their own; edges are the packaged ones whose two subsystems now
  share a category.

Coordinates stay in the packaged layout's units, a node of one reaction being 25
across. ``y`` grows downwards, as the overview's flipped y-range draws it.
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
RELAXATION_ROUNDS = 600
# The drawing's width over its height: a little wider than tall, so the network
# beside its ranking fits under the find box on a laptop's screen.
TARGET_ASPECT = 1.2
PACKING_STEP = 20.0  # layout units between the places a circle is tried at
PACKING_WIDTHS = 48  # packing widths tried, from the widest circle to one row

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


def category_numbers(categories: Iterable[str]) -> Dict[str, int]:
    """Each category's number, from 1, in the network's order.

    The network and the rankings both number the categories in use from this, so a
    category keeps its number in a ranking that lists only some of them.
    """
    return {
        name: number
        for number, name in enumerate(sorted(set(categories), key=category_sort_key), start=1)
    }


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


def _rise(radius: float, placed: np.ndarray, width: float, gap: float) -> Tuple[float, float]:
    """Where a circle comes to rest rising from below into ``placed``, as high as it can.

    ``placed`` holds rows of (x, y, radius); the top edge is ``y = 0`` and the
    circle stays within ``0 <= x <= width``. Among the places tried, the highest
    wins, then the leftmost.
    """
    if width <= 2 * radius:
        xs = np.array([radius])
    else:
        xs = np.append(np.arange(radius, width - radius, PACKING_STEP), width - radius)
    ys = np.full(len(xs), radius)
    if len(placed):
        reach = radius + placed[:, 2] + gap
        dx = np.abs(xs[:, None] - placed[None, :, 0])
        touching = dx < reach[None, :]
        below = placed[None, :, 1] + np.sqrt(np.where(touching, reach**2 - dx**2, 0.0))
        ys = np.maximum(ys, np.where(touching, below, -np.inf).max(axis=1))
    best = int(np.lexsort((xs, np.round(ys, 6)))[0])
    return float(xs[best]), float(ys[best])


def pack_circles(
    radii: Sequence[float], width: float, gap: float = CATEGORY_GAP
) -> np.ndarray:
    """Centres for circles placed in turn, each rising as high as it can within ``width``."""
    placed = np.zeros((0, 3))
    for radius in radii:
        x, y = _rise(float(radius), placed, width, gap)
        placed = np.vstack([placed, [x, y, float(radius)]])
    return placed[:, :2]


def _extent(centres: np.ndarray, radii: np.ndarray) -> Tuple[float, float]:
    if not len(radii):
        return 0.0, 0.0
    return (
        float((centres[:, 0] + radii).max()),
        float((centres[:, 1] + radii).max()),
    )


def pack_categories(
    pathways: Sequence[float], set_apart: Sequence[float], gap: float = CATEGORY_GAP
) -> Tuple[np.ndarray, np.ndarray]:
    """Centres for the pathway categories, packed, and the set-apart ones, in a column.

    The pathways pack from the top left; the set-apart categories stack in order in
    a column to their right. Of the packing widths tried, the one whose whole
    drawing is closest to ``TARGET_ASPECT`` wins; between equally close ones, the
    smaller drawing.
    """
    pathway_radii = np.asarray(pathways, dtype=float)
    apart_radii = np.asarray(set_apart, dtype=float)
    column = np.zeros((len(apart_radii), 2))
    column_width = 2 * float(apart_radii.max()) if len(apart_radii) else 0.0
    top = 0.0
    for index, radius in enumerate(apart_radii):
        column[index] = (column_width / 2, top + radius)
        top += 2 * radius + gap
    column_height = top - gap if len(apart_radii) else 0.0

    if not len(pathway_radii):
        return np.zeros((0, 2)), column
    narrowest = 2 * float(pathway_radii.max())
    widest = float((2 * pathway_radii).sum() + gap * (len(pathway_radii) - 1))
    best = None
    for width in np.linspace(narrowest, max(widest, narrowest), PACKING_WIDTHS):
        centres = pack_circles(pathway_radii, float(width), gap)
        used_width, used_height = _extent(centres, pathway_radii)
        total_width = used_width + (gap + column_width if len(apart_radii) else 0.0)
        total_height = max(used_height, column_height)
        miss = abs(math.log(total_width / total_height / TARGET_ASPECT))
        key = (round(miss, 2), total_width * total_height)
        if best is None or key < best[0]:
            best = (key, centres, used_width)
    _, centres, used_width = best
    column[:, 0] += used_width + gap
    return centres, column


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

    pathways = [cell for cell in cells if cell[0] not in CATEGORIES_SET_APART]
    apart = [cell for cell in cells if cell[0] in CATEGORIES_SET_APART]
    pathway_centres, apart_centres = pack_categories(
        [cell[4] for cell in pathways], [cell[4] for cell in apart]
    )

    nodes: Dict[str, Tuple[float, float, float]] = {}
    circles = []
    for (category, ids, positions, radii, radius), (cx, cy) in zip(
        [*pathways, *apart], [*pathway_centres, *apart_centres]
    ):
        for subsystem, (dx, dy), node in zip(ids, positions, radii):
            nodes[subsystem] = (float(cx + dx), float(cy + dy), float(node))
        circles.append((category, float(cx), float(cy), float(radius), len(ids)))
    return nodes, tuple(circles)


def add_missing_subsystems(
    glyphs: Iterable[Mapping[str, Any]],
    connections: Iterable[Mapping[str, Any]],
    category_of: Mapping[str, str],
    reaction_counts: Mapping[str, int],
    main_metabolites: Mapping[str, Iterable[str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """The packaged network with a node for each model subsystem it lacks.

    A subsystem the model has and the packaged layout does not, such as a part of
    the divided transport, is seeded at its category's centre (the mean of the
    category's packaged seeds), spread on a small circle in name order; the
    relaxation then spaces it. It is joined to each subsystem of its category it
    shares a main metabolite with, as the packaged edges join subsystems.
    """
    glyphs = [dict(glyph) for glyph in glyphs]
    connections = [dict(connection) for connection in connections]
    present = {glyph.get("id") for glyph in glyphs if glyph.get("glyph_class") == "subsystem"}
    missing = sorted(subsystem for subsystem in category_of if subsystem not in present)
    if not missing:
        return glyphs, connections

    seeds: Dict[str, List[Tuple[float, float]]] = {}
    for glyph in glyphs:
        category = category_of.get(glyph.get("id"))
        if glyph.get("glyph_class") == "subsystem" and category is not None:
            seeds.setdefault(category, []).append((float(glyph["x"]), float(glyph["y"])))
    for index, subsystem in enumerate(missing):
        category = category_of[subsystem]
        points = seeds.get(category) or [(0.0, 0.0)]
        cx = sum(x for x, _ in points) / len(points)
        cy = sum(y for _, y in points) / len(points)
        angle = 2 * math.pi * index / len(missing)
        main = set(main_metabolites.get(subsystem, ()))
        glyphs.append(
            {
                "id": subsystem,
                "glyph_class": "subsystem",
                "label": subsystem,
                "x": cx + 300.0 * math.cos(angle),
                "y": cy + 300.0 * math.sin(angle),
                "width": 120.0,
                "height": 80.0,
                "subsystem_name": subsystem,
                "reaction_count": int(reaction_counts.get(subsystem, 0) or 0),
                "metabolite_count": len(main),
            }
        )

    added = set(missing)
    members: Dict[str, List[str]] = {}
    for subsystem, category in category_of.items():
        members.setdefault(category, []).append(subsystem)
    for subsystem in missing:
        main = set(main_metabolites.get(subsystem, ()))
        for other in sorted(members[category_of[subsystem]]):
            # Each pair once: two new subsystems are joined from the first by name.
            if other == subsystem or (other in added and other < subsystem):
                continue
            shared = sorted(main & set(main_metabolites.get(other, ())))
            if not shared:
                continue
            connections.append(
                {
                    "id": f"conn_{subsystem}_to_{other}",
                    "connection_class": "subsystem",
                    "source": subsystem,
                    "target": other,
                    "shared_metabolites": shared,
                    "weight": len(shared),
                    "reaction_count": len(shared),
                    "metabolite_flow_direction": "forward",
                }
            )
    return glyphs, connections


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
        Each new category boundary, with its ``number``, followed by its subsystem
        glyphs, at their new positions and with a ``radius``, as the packaged
        layout orders them; and the edges inside a category.
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
    numbers = category_numbers(category_of.values())

    for glyph in subsystems:
        x, y, radius = nodes[glyph["id"]]
        glyph.update(
            x=x, y=y, radius=radius, width=2 * radius, height=2 * radius,
            category=category_of[glyph["id"]],
        )
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
                "number": numbers[category],
            }
        )
        ordered.extend(glyph for glyph in subsystems if category_of[glyph["id"]] == category)
    return ordered, same_category

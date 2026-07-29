"""Custom serpentine layout for compartment metabolic graphs.

This module implements a deterministic serpentine layout that keeps reactions
aligned in alternating rows while enforcing biochemical constraints such as
uniform reactant/product spacing, enzyme proximity, and constrained edge
angles. The implementation is purposely heuristic but produces coordinates that
are immediately usable by the existing glyph pipeline.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

from ..data_structures.constants import MetabolicSerpentineLayoutDefaults

# Public engine name that can be used by callers.
METABOLIC_SERPENTINE_ENGINE = "metabolic_serpentine"


@dataclass
class MetabolicSerpentineLayoutConfig:
    """Runtime configuration for the serpentine layout engine."""

    reactions_per_row: int = MetabolicSerpentineLayoutDefaults.REACTIONS_PER_ROW
    reaction_horizontal_spacing: float = (
        MetabolicSerpentineLayoutDefaults.REACTION_HORIZONTAL_SPACING
    )
    reaction_vertical_spacing: float = (
        MetabolicSerpentineLayoutDefaults.REACTION_VERTICAL_SPACING
    )
    serpentine_row_indent: float = (
        MetabolicSerpentineLayoutDefaults.SERPENTINE_ROW_INDENT
    )

    metabolite_horizontal_offset: float = (
        MetabolicSerpentineLayoutDefaults.METABOLITE_HORIZONTAL_OFFSET
    )
    metabolite_vertical_spacing: float = (
        MetabolicSerpentineLayoutDefaults.METABOLITE_VERTICAL_SPACING
    )
    intermediate_vertical_offset: float = (
        MetabolicSerpentineLayoutDefaults.INTERMEDIATE_VERTICAL_OFFSET
    )
    shared_metabolite_jitter: float = (
        MetabolicSerpentineLayoutDefaults.SHARED_METABOLITE_JITTER
    )

    enzyme_vertical_offset: float = (
        MetabolicSerpentineLayoutDefaults.ENZYME_VERTICAL_OFFSET
    )

    min_edge_length: float = MetabolicSerpentineLayoutDefaults.MIN_EDGE_LENGTH
    angle_snap_degrees: List[float] = field(
        default_factory=lambda: list(MetabolicSerpentineLayoutDefaults.ANGLE_SNAP_DEGREES)
    )
    fixed_metabolite_angles: Dict[str, float] = field(
        default_factory=lambda: dict(
            MetabolicSerpentineLayoutDefaults.FIXED_METABOLITE_ANGLES
        )
    )

    base_anchor_x: float = MetabolicSerpentineLayoutDefaults.BASE_ANCHOR_X
    base_anchor_y: float = MetabolicSerpentineLayoutDefaults.BASE_ANCHOR_Y


class MetabolicSerpentineLayout:
    """Compute serpentine coordinates for metabolic reaction graphs."""

    def __init__(self, config: Optional[MetabolicSerpentineLayoutConfig] = None):
        self.config = config or MetabolicSerpentineLayoutConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def layout_graph(self, graph: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """Return layout positions for the provided metabolic graph."""
        reaction_nodes = [
            node
            for node, data in graph.nodes(data=True)
            if data.get("type") == "reaction"
        ]
        if not reaction_nodes:
            raise ValueError("Serpentine layout requires reaction nodes")

        ordered_reactions = self._order_reactions(graph, reaction_nodes)
        placement = self._assign_reaction_positions(ordered_reactions)

        positions: Dict[str, Tuple[float, float]] = {
            node: info["coords"] for node, info in placement.items()
        }

        self._position_enzyme_groups(graph, placement, positions)
        self._position_metabolites(graph, placement, positions)

        self._position_remaining_nodes(graph, positions)
        return positions

    # ------------------------------------------------------------------
    # Reaction ordering and placement
    # ------------------------------------------------------------------
    def _order_reactions(
        self, graph: nx.DiGraph, reactions: List[str]
    ) -> List[str]:
        reaction_graph = nx.DiGraph()
        reaction_graph.add_nodes_from(reactions)

        for metabolite, attrs in graph.nodes(data=True):
            if attrs.get("type") != "metabolite":
                continue

            producers = {
                pred
                for pred in graph.predecessors(metabolite)
                if graph.nodes[pred].get("type") == "reaction"
            }
            consumers = {
                succ
                for succ in graph.successors(metabolite)
                if graph.nodes[succ].get("type") == "reaction"
            }

            for producer in producers:
                for consumer in consumers:
                    if producer != consumer:
                        reaction_graph.add_edge(producer, consumer)

        try:
            ordered = list(nx.topological_sort(reaction_graph))
            if len(ordered) != len(reactions):
                remaining = [node for node in reactions if node not in ordered]
                ordered.extend(sorted(remaining))
        except nx.NetworkXUnfeasible:
            ordered = self._bfs_order(reaction_graph, reactions)

        if not ordered:
            ordered = sorted(reactions)

        return ordered

    def _bfs_order(self, graph: nx.DiGraph, seeds: Iterable[str]) -> List[str]:
        visited = set()
        order: List[str] = []
        for start in sorted(seeds):
            if start in visited:
                continue
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                order.append(node)

                neighbors = set(graph.successors(node)) | set(graph.predecessors(node))
                for neighbor in sorted(neighbors):
                    if neighbor not in visited:
                        queue.append(neighbor)
        return order

    def _assign_reaction_positions(
        self, ordered_reactions: List[str]
    ) -> Dict[str, Dict[str, object]]:
        if not ordered_reactions:
            return {}

        cfg = self.config
        columns = max(1, min(cfg.reactions_per_row, len(ordered_reactions)))
        placement: Dict[str, Dict[str, object]] = {}

        for index, reaction_id in enumerate(ordered_reactions):
            row = index // columns
            position_in_row = index % columns
            direction = 1 if row % 2 == 0 else -1
            if direction > 0:
                column_index = position_in_row
            else:
                column_index = columns - 1 - position_in_row

            x = (
                cfg.base_anchor_x
                + column_index * cfg.reaction_horizontal_spacing
                + (row % 2) * cfg.serpentine_row_indent
            )
            y = cfg.base_anchor_y + row * cfg.reaction_vertical_spacing

            placement[reaction_id] = {
                "coords": (float(x), float(y)),
                "row": row,
                "column": column_index,
                "direction": direction,
            }

        return placement

    # ------------------------------------------------------------------
    # Enzyme positioning
    # ------------------------------------------------------------------
    def _position_enzyme_groups(
        self,
        graph: nx.DiGraph,
        placement: Dict[str, Dict[str, object]],
        positions: Dict[str, Tuple[float, float]],
    ) -> None:
        cfg = self.config

        for reaction_id, info in placement.items():
            reaction_pos = info["coords"]
            enzyme_groups = [
                node
                for node in graph.predecessors(reaction_id)
                if graph.nodes[node].get("type") == "enzyme_group"
                and graph.nodes[node].get("genes")
            ]
            if not enzyme_groups:
                continue

            offsets = self._distribute_offsets(len(enzyme_groups), cfg.metabolite_vertical_spacing)
            for group_id, offset in zip(enzyme_groups, offsets):
                x = reaction_pos[0] + offset
                y = reaction_pos[1] - cfg.enzyme_vertical_offset
                positions[group_id] = (x, y)

    # ------------------------------------------------------------------
    # Metabolite positioning
    # ------------------------------------------------------------------
    def _position_metabolites(
        self,
        graph: nx.DiGraph,
        placement: Dict[str, Dict[str, object]],
        positions: Dict[str, Tuple[float, float]],
    ) -> None:
        metabolite_positions: Dict[str, Tuple[float, float]] = {}

        for reaction_id, info in placement.items():
            reaction_pos = info["coords"]
            direction = info["direction"]

            reactants = self._collect_metabolites(graph, reaction_id, inbound=True)
            products = self._collect_metabolites(graph, reaction_id, inbound=False)

            self._place_metabolite_side(
                graph,
                reaction_id,
                reaction_pos,
                direction,
                reactants,
                metabolite_positions,
                side_multiplier=-1 if direction > 0 else 1,
            )
            self._place_metabolite_side(
                graph,
                reaction_id,
                reaction_pos,
                direction,
                products,
                metabolite_positions,
                side_multiplier=1 if direction > 0 else -1,
            )

        positions.update(metabolite_positions)

    def _collect_metabolites(
        self, graph: nx.DiGraph, reaction_id: str, inbound: bool
    ) -> List[str]:
        if inbound:
            neighbors = [
                node
                for node in graph.predecessors(reaction_id)
                if graph.nodes[node].get("type") == "metabolite"
                and graph[node][reaction_id].get("connection_type") == "metabolic"
            ]
        else:
            neighbors = [
                node
                for node in graph.successors(reaction_id)
                if graph.nodes[node].get("type") == "metabolite"
                and graph[reaction_id][node].get("connection_type") == "metabolic"
            ]

        def sort_key(node: str) -> Tuple[int, str]:
            subtype = graph.nodes[node].get("subtype", "main")
            subtype_rank = 0 if subtype == "main" else 1
            label = graph.nodes[node].get("label") or graph.nodes[node].get("original_id") or node
            return (subtype_rank, label)

        neighbors.sort(key=sort_key)
        return neighbors

    def _place_metabolite_side(
        self,
        graph: nx.DiGraph,
        reaction_id: str,
        reaction_pos: Tuple[float, float],
        direction: int,
        metabolites: List[str],
        metabolite_positions: Dict[str, Tuple[float, float]],
        side_multiplier: int,
    ) -> None:
        if not metabolites:
            return

        cfg = self.config
        main_metabolites = [
            node for node in metabolites if graph.nodes[node].get("subtype") != "intermediate"
        ]
        intermediate_metabolites = [
            node for node in metabolites if graph.nodes[node].get("subtype") == "intermediate"
        ]

        offsets = self._distribute_offsets(len(main_metabolites), cfg.metabolite_vertical_spacing)
        for node_id, offset in zip(main_metabolites, offsets):
            target = self._build_candidate_position(
                graph,
                node_id,
                reaction_pos,
                side_multiplier,
                offset,
                preferred_angles=self._horizontal_angles(side_multiplier),
                base_distance=cfg.metabolite_horizontal_offset,
            )
            self._store_metabolite_position(node_id, target, metabolite_positions)

        if intermediate_metabolites:
            vertical_step = cfg.metabolite_vertical_spacing / 2 or 1.0
            for index, node_id in enumerate(intermediate_metabolites):
                sign = -1 if index % 2 == 0 else 1
                offset_y = sign * (cfg.intermediate_vertical_offset + (index // 2) * vertical_step)
                target = self._build_candidate_position(
                    graph,
                    node_id,
                    reaction_pos,
                    side_multiplier,
                    offset_y,
                    preferred_angles=[90, -90],
                    base_distance=cfg.metabolite_horizontal_offset * 0.6,
                )
                self._store_metabolite_position(node_id, target, metabolite_positions)

    def _build_candidate_position(
        self,
        graph: nx.DiGraph,
        metabolite_id: str,
        origin: Tuple[float, float],
        side_multiplier: int,
        vertical_offset: float,
        preferred_angles: List[float],
        base_distance: float,
    ) -> Tuple[float, float]:
        cfg = self.config
        x = origin[0] + side_multiplier * base_distance
        y = origin[1] + vertical_offset
        candidate = (x, y)

        original_id = graph.nodes[metabolite_id].get("original_id")
        if original_id and original_id in cfg.fixed_metabolite_angles:
            angle = cfg.fixed_metabolite_angles[original_id]
            return self._position_at_angle(origin, angle, max(base_distance, cfg.min_edge_length))

        return self._snap_to_angles(
            origin,
            candidate,
            preferred_angles,
            base_distance=max(base_distance, cfg.min_edge_length),
        )

    def _store_metabolite_position(
        self,
        metabolite_id: str,
        candidate: Tuple[float, float],
        metabolite_positions: Dict[str, Tuple[float, float]],
    ) -> None:
        if metabolite_id in metabolite_positions:
            existing = metabolite_positions[metabolite_id]
            averaged = (
                (existing[0] + candidate[0]) / 2.0,
                (existing[1] + candidate[1]) / 2.0,
            )
            jitter = self.config.shared_metabolite_jitter
            if jitter:
                checksum = sum(ord(ch) for ch in metabolite_id)
                direction = -1 if checksum % 2 == 0 else 1
                averaged = (
                    averaged[0],
                    averaged[1] + direction * jitter,
                )
            metabolite_positions[metabolite_id] = averaged
        else:
            metabolite_positions[metabolite_id] = candidate

    # ------------------------------------------------------------------
    # Fallback positioning for nodes without explicit rules
    # ------------------------------------------------------------------
    def _position_remaining_nodes(
        self, graph: nx.DiGraph, positions: Dict[str, Tuple[float, float]]
    ) -> None:
        cfg = self.config
        for node in graph.nodes:
            if node in positions:
                continue
            neighbors = list(graph.predecessors(node)) + list(graph.successors(node))
            anchored = next((positions[n] for n in neighbors if n in positions), None)
            if anchored:
                positions[node] = (anchored[0], anchored[1] + cfg.metabolite_vertical_spacing)
            else:
                positions[node] = (cfg.base_anchor_x, cfg.base_anchor_y)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _distribute_offsets(self, count: int, spacing: float) -> List[float]:
        if count <= 1:
            return [0.0] if count == 1 else []
        midpoint = (count - 1) / 2.0
        return [((index - midpoint) * spacing) for index in range(count)]

    def _horizontal_angles(self, side_multiplier: int) -> List[float]:
        forward = [angle for angle in self.config.angle_snap_degrees if abs(angle) <= 45]
        backward = [
            angle
            for angle in self.config.angle_snap_degrees
            if abs(angle) >= 135 or abs(angle) == 180
        ]
        return forward if side_multiplier > 0 else backward

    def _snap_to_angles(
        self,
        origin: Tuple[float, float],
        candidate: Tuple[float, float],
        allowed_angles: List[float],
        base_distance: float,
    ) -> Tuple[float, float]:
        if not allowed_angles:
            return candidate

        dx = candidate[0] - origin[0]
        dy = candidate[1] - origin[1]
        distance = math.hypot(dx, dy) or base_distance
        angle = math.degrees(math.atan2(dy, dx))
        snapped_angle = min(allowed_angles, key=lambda a: self._angular_distance(angle, a))
        radians = math.radians(snapped_angle)
        length = max(distance, base_distance)
        return (
            origin[0] + math.cos(radians) * length,
            origin[1] + math.sin(radians) * length,
        )

    def _position_at_angle(
        self, origin: Tuple[float, float], angle_degrees: float, radius: float
    ) -> Tuple[float, float]:
        radians = math.radians(angle_degrees)
        return (
            origin[0] + math.cos(radians) * radius,
            origin[1] + math.sin(radians) * radius,
        )

    def _angular_distance(self, angle_a: float, angle_b: float) -> float:
        diff = abs(angle_a - angle_b) % 360
        return min(diff, 360 - diff)


__all__ = [
    "MetabolicSerpentineLayout",
    "MetabolicSerpentineLayoutConfig",
    "METABOLIC_SERPENTINE_ENGINE",
]

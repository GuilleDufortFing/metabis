"""The overview's spaced layout: no overlaps, sizes by reactions, a fixed order."""

import json
import math
from pathlib import Path

import pytest

import metabis
from metabis import MetabolicModel
from metabis.viz.metabolic_visualization.generation.graph_building.spaced_network import (
    CATEGORY_ORDER,
    NODE_GAP,
    category_sort_key,
    node_radius,
    space_network,
)

LAYOUT = (
    Path(metabis.__file__).parent
    / "resources/graphs/metabolic_network/layout_fdp_all_subsystems.json"
)


@pytest.fixture(scope="module")
def packaged():
    return json.loads(LAYOUT.read_text())


@pytest.fixture(scope="module")
def category_of():
    model = MetabolicModel(use_cache=True)
    return {
        subsystem: category
        for category, subsystems in model.get_subsystems_by_category().items()
        for subsystem in subsystems
    }


def _split(glyphs):
    nodes = [g for g in glyphs if g["glyph_class"] == "subsystem"]
    circles = {g["label"]: g for g in glyphs if g["glyph_class"] == "subsystem_category"}
    return nodes, circles


def test_no_two_nodes_of_a_category_come_closer_than_the_gap(packaged, category_of):
    glyphs, _ = space_network(packaged["glyphs"], packaged["connections"], category_of)
    nodes, _ = _split(glyphs)
    for index, a in enumerate(nodes):
        for b in nodes[index + 1:]:
            if category_of[a["id"]] != category_of[b["id"]]:
                continue
            clearance = math.dist((a["x"], a["y"]), (b["x"], b["y"])) - a["radius"] - b["radius"]
            assert clearance >= NODE_GAP - 1e-6, (a["id"], b["id"], clearance)


def test_every_node_sits_inside_its_category_and_categories_never_meet(
    packaged, category_of
):
    glyphs, _ = space_network(packaged["glyphs"], packaged["connections"], category_of)
    nodes, circles = _split(glyphs)
    for node in nodes:
        circle = circles[category_of[node["id"]]]
        reach = math.dist((node["x"], node["y"]), (circle["x"], circle["y"])) + node["radius"]
        assert reach <= circle["radius"] + 1e-6
    ordered = list(circles.values())
    for index, a in enumerate(ordered):
        for b in ordered[index + 1:]:
            assert math.dist((a["x"], a["y"]), (b["x"], b["y"])) > a["radius"] + b["radius"]


def test_a_node_area_is_proportional_to_its_reactions(packaged, category_of):
    glyphs, _ = space_network(packaged["glyphs"], packaged["connections"], category_of)
    nodes, _ = _split(glyphs)
    counts = {g["id"]: g["reaction_count"] for g in packaged["glyphs"] if g["glyph_class"] == "subsystem"}
    for node in nodes:
        assert node["radius"] == pytest.approx(node_radius(counts[node["id"]]))
    large = [n for n in nodes if node_radius(counts[n["id"]]) > node_radius(0)]
    a, b = large[0], large[1]
    assert (a["radius"] / b["radius"]) ** 2 == pytest.approx(
        counts[a["id"]] / counts[b["id"]]
    )


def test_categories_read_in_textbook_order(packaged, category_of):
    glyphs, _ = space_network(packaged["glyphs"], packaged["connections"], category_of)
    _, circles = _split(glyphs)
    # A row's circles share their centre line; rows read top to bottom, then left to right.
    reading = [c["label"] for c in sorted(circles.values(), key=lambda c: (round(c["y"]), c["x"]))]

    assert reading == sorted(circles, key=category_sort_key)
    assert reading[: len(CATEGORY_ORDER)] == list(CATEGORY_ORDER)
    assert reading[-2:] == ["Membrane Transport & Exchange", "Model Utilities & Non-human Pathways"]


def test_edges_stay_within_a_category_and_the_layout_is_reproducible(
    packaged, category_of
):
    first, edges = space_network(packaged["glyphs"], packaged["connections"], category_of)
    second, _ = space_network(packaged["glyphs"], packaged["connections"], category_of)
    assert first == second
    assert edges and all(category_of[e["source"]] == category_of[e["target"]] for e in edges)


def test_an_uploaded_category_gets_its_own_place_and_keeps_only_its_edges(
    packaged, category_of
):
    moved = dict(category_of)
    moved["glycolysis_gluconeogenesis"] = "My pathways"
    moved["pentose_phosphate_pathway"] = "My pathways"
    glyphs, edges = space_network(packaged["glyphs"], packaged["connections"], moved)
    nodes, circles = _split(glyphs)

    assert "My pathways" in circles
    assert circles["My pathways"]["subsystem_count"] == 2
    # An unknown category reads after the known ones and before the set-apart two.
    order = sorted(circles, key=category_sort_key)
    assert order.index("My pathways") == len(order) - 3
    pair = {"glycolysis_gluconeogenesis", "pentose_phosphate_pathway"}
    assert any({e["source"], e["target"]} == pair for e in edges)
    assert not any(
        ("glycolysis_gluconeogenesis" in (e["source"], e["target"]))
        and moved[e["source"]] != moved[e["target"]]
        for e in edges
    )

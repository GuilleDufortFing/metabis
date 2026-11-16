"""The overview's spaced layout: no overlaps, sizes by reactions, a fixed order."""

import json
import math
from pathlib import Path

import pytest

import metabis
from metabis import MetabolicModel
from metabis.viz.metabolic_visualization.generation.graph_building.spaced_network import (
    CATEGORIES_SET_APART,
    CATEGORY_ORDER,
    NODE_GAP,
    TARGET_ASPECT,
    add_missing_subsystems,
    category_numbers,
    category_sort_key,
    node_radius,
    pack_circles,
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


def test_categories_are_numbered_in_textbook_order(packaged, category_of):
    glyphs, _ = space_network(packaged["glyphs"], packaged["connections"], category_of)
    _, circles = _split(glyphs)
    numbered = [c["label"] for c in sorted(circles.values(), key=lambda c: c["number"])]

    assert [c["number"] for c in circles.values()] == list(range(1, len(circles) + 1))
    assert numbered == sorted(circles, key=category_sort_key)
    assert numbered[: len(CATEGORY_ORDER)] == list(CATEGORY_ORDER)
    assert numbered[-2:] == ["Membrane Transport & Exchange", "Model Utilities & Non-human Pathways"]
    assert category_numbers(category_of.values()) == {
        label: circle["number"] for label, circle in circles.items()
    }


def test_the_pathways_pack_left_and_the_set_apart_stack_in_a_column_right(
    packaged, category_of
):
    glyphs, _ = space_network(packaged["glyphs"], packaged["connections"], category_of)
    _, circles = _split(glyphs)
    apart = [circles[name] for name in CATEGORIES_SET_APART]
    pathways = [c for name, c in circles.items() if name not in CATEGORIES_SET_APART]

    assert max(c["x"] + c["radius"] for c in pathways) < min(c["x"] - c["radius"] for c in apart)
    assert apart[0]["y"] < apart[1]["y"]
    # Packed, the pathways start at the top: the first row sits against it.
    assert min(c["y"] - c["radius"] for c in circles.values()) == pytest.approx(0.0)
    width = max(c["x"] + c["radius"] for c in circles.values())
    height = max(c["y"] + c["radius"] for c in circles.values())
    assert width / height == pytest.approx(TARGET_ASPECT, rel=0.15)


def test_a_circle_rises_into_the_highest_gap_that_fits():
    # Two large circles side by side leave room above-right for a small one, not below.
    centres = pack_circles([500.0, 500.0, 100.0], width=2400.0, gap=50.0)
    small = centres[2]
    assert small[1] == pytest.approx(100.0)
    assert small[0] > centres[1][0]


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


def test_a_subsystem_the_packaged_layout_lacks_gets_a_node_and_its_edges():
    glyphs = [
        {"id": "a", "glyph_class": "subsystem", "x": 0.0, "y": 0.0, "reaction_count": 4},
        {"id": "b", "glyph_class": "subsystem", "x": 600.0, "y": 0.0, "reaction_count": 9},
    ]
    category_of = {"a": "One", "b": "One", "new": "One", "newer": "One"}
    main = {"a": ["m1"], "b": ["m2"], "new": ["m1", "m3"], "newer": ["m3"]}
    completed, edges = add_missing_subsystems(
        glyphs, [], category_of, {"new": 16, "newer": 1}, main
    )

    new = {glyph["id"]: glyph for glyph in completed}
    assert new["new"]["reaction_count"] == 16 and new["newer"]["reaction_count"] == 1
    # Seeded around the category's centre, then spaced like any other node.
    assert abs(new["new"]["x"] - 300.0) <= 300.0
    pairs = {frozenset((edge["source"], edge["target"])) for edge in edges}
    assert pairs == {frozenset(("new", "a")), frozenset(("new", "newer"))}
    laid_out, _ = space_network(completed, edges, category_of)
    assert {g["id"] for g in laid_out if g["glyph_class"] == "subsystem"} == set(category_of)

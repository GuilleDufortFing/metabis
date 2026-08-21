"""Measured readability of the packaged graph layouts, and the determinism it needs.

`MetabisReview/graph-layout-analysis.md` diagnoses the automatic subsystem layout
against the curated Human1 maps and proposes a redesign. Its argument rests on
four metrics computed from the packaged resources. This module recomputes them so
the redesign has a baseline to move, and so a later change cannot silently undo
the improvement.

Two of the metrics here were added after the redesign shipped, because their
absence is what let it ship. Counting only crossings and edge lengths, the
2026-07-25 layouts were the best this suite had ever measured; on screen they
were a field of detached motifs with enzyme groups lying across other
components' reactions. A crossing count cannot see either fault — duplicating a
shared metabolite until nothing connects to anything *improves* it, and a glyph
buried under another glyph does not cross a single edge. So connectivity and
overlap are now asserted in their own right:

* no packaged automatic layout draws one glyph on top of another, at all;
* a packaged layout has the connected components its model has, not more;
* the curated Human1 maps really are the better target on crossings;
* the automatic layouts stay within measured crossing and edge-length ceilings;
* the generator is deterministic, which is what makes regeneration auditable.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from metabis import MetabolicModel
from metabis.resource_registry import ResourceRegistry
from metabis.viz.metabolic_visualization.generation.data_structures.constants import (
    DefaultOptimization,
)
from metabis.viz.metabolic_visualization.generation.graph_building.compartment_graph import (  # noqa: E501
    create_compartmentalized_graph_from_model,
)
from metabis.viz.metabolic_visualization.generation.utilities.cache_utils import (
    DEFAULT_SUBSYSTEM_LAYOUT_ENGINE,
    SUBSYSTEM_HUMAN1_VARIANT,
    SUBSYSTEM_STANDARD_VARIANT,
)

GRAPH_ROOT = Path(ResourceRegistry().get_graph_base_paths(required=False)["subsystem"])
STANDARD_DIRECTORY = GRAPH_ROOT / SUBSYSTEM_STANDARD_VARIANT
HUMAN1_DIRECTORY = GRAPH_ROOT / SUBSYSTEM_HUMAN1_VARIANT

# Measured 2026-07-25 over all 145 automatic layouts, repackaged against the
# connectivity-preserving split: median 0.056 crossings per edge, 14,428 in total.
#
# These ceilings went **up**, from 0.021 / 5,142, and the rise is the point rather
# than a regression to be smoothed over. The previous numbers were bought by
# duplicating every arm of every shared metabolite, which drove crossings toward
# zero by taking the drawing apart: a compartment became a field of detached
# motifs, and a metric suite that counted only crossings scored that as the best
# result it had ever recorded. Connectivity is now asserted separately and
# absolutely (`test_packaged_layouts_stay_as_connected_as_the_model`), so this
# metric is free to say what it honestly says.
#
# Six of the 145 still exceed 0.5 crossings per edge — `pool_reactions` (2.05) and
# `formation_and_hydrolysis_of_cholesterol_esters` (1.50) worst — and they are
# dense graphs rather than stars, so no duplication rule reaches them. They are
# planar and `dot` is what draws them badly; the remaining work is per-component
# layout.
BASELINE_STANDARD_CROSSINGS_PER_EDGE_MEDIAN = 0.10
BASELINE_STANDARD_TOTAL_CROSSINGS = 20_000

# The curated maps are the target the redesign aims at (measured median 0.000).
# Asserted as a floor on how much better they are, so a resource swap that
# quietly degraded them would fail here rather than silently lower the bar.
HUMAN1_CROSSINGS_PER_EDGE_MEDIAN_CEILING = 0.05

# Subsystems the exhaustive glyph-ID check runs over. The full 146-subsystem
# sweep takes minutes and belongs to `scripts/regenerate_subsystem_graphs.py`,
# which runs it on every regeneration; these are the largest and most
# duplication-heavy cases, which is where an ID collision would appear first.
ID_COLLISION_SUBSYSTEMS = (
    "xenobiotics_metabolism",
    "sphingolipid_metabolism",
    "retinol_metabolism",
    "beta_alanine_metabolism",
)

# Subsystems the analysis calls out by name, so a change to any of them is
# visible per-subsystem rather than only in an aggregate.
WITNESS_SUBSYSTEMS = (
    "retinol_metabolism",
    "androgen_metabolism",
    "beta_alanine_metabolism",
)

# Subsystems the fragmentation check re-derives from the model. It regenerates
# each one, so it runs over a representative handful rather than all 145: the
# three witnesses plus the two pathways whose curated maps the complaint compared
# against directly.
FRAGMENTATION_SUBSYSTEMS = (
    "androgen_metabolism",
    "beta_alanine_metabolism",
    "glycolysis_gluconeogenesis",
    "pyrimidine_metabolism",
    "retinol_metabolism",
)

# How much of a drawing must hang together as one network. Measured over the
# repackaged files, these five run 0.570 (androgen) to 0.644 (beta_alanine), and
# the median over all 145 automatic layouts is 0.647 against the curated maps'
# 0.608 — the automatic maps are now at least as connected as the curated ones.
# Under the previous rule the same five ran 0.06 to 0.33.
#
# The floor is deliberately not applied to all 145: `exchange_demand_reactions` is
# thousands of genuinely independent two-node reactions and its share is 0.0006,
# which is correct. Connectivity is checked against each subsystem's own model
# connectivity instead, and this floor guards the five that have a curated map to
# be compared against.
LARGEST_COMPONENT_SHARE_FLOOR = 0.50

# The named witnesses, measured from the repackaged files 2026-07-25: retinol
# 0.230 crossings/edge, 9.7x longest/median edge and a degree-10 metabolite;
# androgen 0.071 / 5.0x / degree 4; beta_alanine 0.000 / 2.6x / degree 3. Retinol
# is the case to watch — it draws six components instead of the 61 the previous
# rule produced, and holding that together is what its crossings pay for. The
# ceilings keep headroom for a different Graphviz build re-placing these graphs.
WITNESS_CROSSINGS_PER_EDGE_CEILING = 0.35
WITNESS_EDGE_RATIO_CEILING = 15.0

# A metabolite every one of whose arms is the only path to its reaction is drawn
# whole up to the pool cap, so the bound is the cap itself rather than the split
# threshold. Retinol's highest is 10 and the highest anywhere in the 145 is the
# cap, against 14 in the curated maps.
WITNESS_MAX_DEGREE_CEILING = DefaultOptimization.HUB_METABOLITE_DEGREE_CAP


def _load(path):
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("glyphs", []), payload.get("connections", [])


def _segments(glyphs, connections):
    """Return each connection as the straight segment the renderer draws for it.

    Every edge is drawn with ``plot.segment`` between node centres, so the
    segment between centres is what a reader actually sees — which is why
    crossings are counted on it rather than on any routing Graphviz computed.
    """
    centres = {glyph["id"]: (glyph["x"], glyph["y"]) for glyph in glyphs}
    segments = []
    for connection in connections:
        source = centres.get(connection.get("source"))
        target = centres.get(connection.get("target"))
        if source is not None and target is not None and source != target:
            segments.append((source, target))
    return segments


def _orientation(first, second, third):
    return (third[1] - first[1]) * (second[0] - first[0]) - (
        second[1] - first[1]
    ) * (third[0] - first[0])


def _crosses(one, other):
    a, b = one
    c, d = other
    if a in (c, d) or b in (c, d):
        return False
    return (
        (_orientation(a, b, c) > 0) != (_orientation(a, b, d) > 0)
        and (_orientation(c, d, a) > 0) != (_orientation(c, d, b) > 0)
    )


def count_crossings(segments):
    """Count pairs of drawn edges that cross away from a shared endpoint."""
    crossings = 0
    for index, segment in enumerate(segments):
        for other in segments[index + 1 :]:
            if _crosses(segment, other):
                crossings += 1
    return crossings


def crossings_per_edge(glyphs, connections):
    segments = _segments(glyphs, connections)
    if not segments:
        return 0.0
    return count_crossings(segments) / len(segments)


def edge_length_ratio(glyphs, connections):
    """Return longest edge / median edge — how far one line dominates the drawing."""
    segments = _segments(glyphs, connections)
    if not segments:
        return 0.0
    lengths = sorted(
        ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for a, b in segments
    )
    median = lengths[len(lengths) // 2]
    if median <= 0:
        return float("inf")
    return lengths[-1] / median


def duplication_factor(glyphs):
    """Return metabolite glyphs per distinct metabolite.

    Curators split shared metabolites into several glyphs; this is how much of
    that the layout does.
    """
    originals = [
        glyph.get("original_id")
        for glyph in glyphs
        if glyph.get("glyph_class") == "metabolite" and glyph.get("original_id")
    ]
    distinct = len(set(originals))
    return len(originals) / distinct if distinct else 0.0


def _core_graph(glyphs, connections):
    """Return the drawn metabolite↔reaction graph as an adjacency mapping."""
    core = {
        glyph["id"]
        for glyph in glyphs
        if glyph.get("glyph_class") in ("reaction", "metabolite")
    }
    adjacency = {node: set() for node in core}
    for connection in connections:
        if connection.get("connection_class") != "metabolic":
            continue
        source, target = connection.get("source"), connection.get("target")
        if source in adjacency and target in adjacency:
            adjacency[source].add(target)
            adjacency[target].add(source)
    return adjacency


def component_sizes(glyphs, connections):
    """Return the drawn connected components, largest first.

    A metabolic map is read as a network, so how much of it hangs together is a
    readability metric in its own right — and the one the crossing count is blind
    to. Duplicating every arm of a shared metabolite drives crossings to zero by
    making the drawing a field of detached motifs, which scores perfectly and
    reads as nothing.
    """
    adjacency = _core_graph(glyphs, connections)
    seen = set()
    sizes = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def largest_component_share(glyphs, connections):
    """Return the fraction of drawn nodes in the largest connected component."""
    sizes = component_sizes(glyphs, connections)
    total = sum(sizes)
    return sizes[0] / total if total else 0.0


def count_glyph_overlaps(glyphs):
    """Count pairs of drawn glyph boxes that intersect.

    Genes are excluded: they are drawn *inside* their enzyme group by design.
    Compartment boundaries are excluded for the same reason — they contain their
    contents. Everything else is a solid box the reader is meant to be able to
    tell apart from its neighbours, and a curated map has essentially none of
    these.
    """
    boxes = [
        (
            glyph["x"] - (glyph.get("width") or 0.0) / 2,
            glyph["y"] - (glyph.get("height") or 0.0) / 2,
            glyph["x"] + (glyph.get("width") or 0.0) / 2,
            glyph["y"] + (glyph.get("height") or 0.0) / 2,
        )
        for glyph in glyphs
        if glyph.get("glyph_class") in ("reaction", "metabolite", "enzyme_group")
    ]
    overlaps = 0
    for index, (left, bottom, right, top) in enumerate(boxes):
        for other_left, other_bottom, other_right, other_top in boxes[index + 1 :]:
            if (
                left < other_right
                and other_left < right
                and bottom < other_top
                and other_bottom < top
            ):
                overlaps += 1
    return overlaps


def max_metabolite_degree(glyphs, connections):
    """Return the highest metabolite degree — every hub is a star that crosses."""
    degree = Counter()
    for connection in connections:
        degree[connection.get("source")] += 1
        degree[connection.get("target")] += 1
    metabolite_degrees = [
        degree[glyph["id"]]
        for glyph in glyphs
        if glyph.get("glyph_class") == "metabolite"
    ]
    return max(metabolite_degrees) if metabolite_degrees else 0


def _median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else 0.0


@pytest.fixture(scope="module")
def standard_resources():
    return sorted(STANDARD_DIRECTORY.glob("*.json"))


@pytest.fixture(scope="module")
def human1_resources():
    return sorted(HUMAN1_DIRECTORY.glob("*.json"))


@pytest.mark.resource
def test_packaged_automatic_layouts_have_not_degraded(standard_resources):
    """Pin the current automatic-layout readability so it cannot drift further."""
    assert standard_resources, f"No automatic layouts below {STANDARD_DIRECTORY}"

    per_edge = []
    total = 0
    for path in standard_resources:
        glyphs, connections = _load(path)
        segments = _segments(glyphs, connections)
        crossings = count_crossings(segments)
        total += crossings
        if segments:
            per_edge.append(crossings / len(segments))

    assert _median(per_edge) <= BASELINE_STANDARD_CROSSINGS_PER_EDGE_MEDIAN
    assert total <= BASELINE_STANDARD_TOTAL_CROSSINGS


@pytest.mark.resource
def test_no_packaged_layout_draws_one_glyph_on_top_of_another(standard_resources):
    """Nothing overlaps anything, anywhere in the packaged automatic layouts.

    This is an absolute claim rather than a budget, because a glyph hidden under
    another is not a readability trade-off — it is a reaction, a metabolite or a
    gene set the reader cannot see. It has to be asserted separately from the
    crossing metrics, which are blind to it: the enzyme groups that shipped on
    2026-07-25 sat on other components' reactions and metabolites in every
    subsystem measured, at a crossing count of 0.019 per edge.
    """
    assert standard_resources, f"No automatic layouts below {STANDARD_DIRECTORY}"

    offenders = {}
    for path in standard_resources:
        glyphs, _ = _load(path)
        overlaps = count_glyph_overlaps(glyphs)
        if overlaps:
            offenders[path.stem] = overlaps

    assert not offenders, f"Overlapping glyphs in {offenders}"


@pytest.mark.resource
def test_packaged_layouts_stay_as_connected_as_the_model(standard_resources):
    """The drawing keeps the connectivity the model has, not less.

    A layout may duplicate a shared metabolite to shorten its arms, but it may not
    duplicate away the pathway: the drawn components must match the components the
    reactions and metabolites actually form. Splitting every arm of a hub instead
    of only the redundant ones took the largest component of a compartment to
    roughly a third of the curated map's, which is what made the maps stop reading
    as metabolic networks — while *improving* every metric this module had.

    The regenerated graph is compared against its own model-level connectivity, so
    this holds for any subsystem rather than encoding a per-subsystem number.
    """
    model = MetabolicModel()
    checked = 0
    for path in standard_resources:
        if path.stem not in FRAGMENTATION_SUBSYSTEMS:
            continue
        glyphs, connections = _load(path)
        drawn = component_sizes(glyphs, connections)

        model_glyphs, model_connections = create_compartmentalized_graph_from_model(
            path.stem, model, DEFAULT_SUBSYSTEM_LAYOUT_ENGINE
        )
        expected = component_sizes(model_glyphs, model_connections)

        assert len(drawn) == len(expected), path.stem
        assert drawn[0] == expected[0], path.stem
        # And the map is dominated by one network rather than scattered.
        assert largest_component_share(glyphs, connections) >= (
            LARGEST_COMPONENT_SHARE_FLOOR
        ), f"{path.stem}: {largest_component_share(glyphs, connections):.2f}"
        checked += 1

    assert checked == len(FRAGMENTATION_SUBSYSTEMS)


@pytest.mark.resource
def test_curated_layouts_are_the_better_target(human1_resources):
    """The curated maps must stay far cleaner than the automatic ones."""
    assert human1_resources, f"No curated layouts below {HUMAN1_DIRECTORY}"

    per_edge = []
    for path in human1_resources:
        glyphs, connections = _load(path)
        if _segments(glyphs, connections):
            per_edge.append(crossings_per_edge(glyphs, connections))

    assert _median(per_edge) <= HUMAN1_CROSSINGS_PER_EDGE_MEDIAN_CEILING


@pytest.mark.resource
@pytest.mark.parametrize("subsystem_id", WITNESS_SUBSYSTEMS)
def test_witness_subsystem_metrics_are_reportable(subsystem_id):
    """Every metric the redesign is measured on resolves for the named cases."""
    path = STANDARD_DIRECTORY / f"{subsystem_id}.json"
    if not path.exists():
        pytest.skip(f"{subsystem_id} has no automatic layout")

    glyphs, connections = _load(path)

    assert crossings_per_edge(glyphs, connections) >= 0
    assert edge_length_ratio(glyphs, connections) > 0
    assert duplication_factor(glyphs) >= 1.0
    assert max_metabolite_degree(glyphs, connections) > 0


@pytest.mark.resource
@pytest.mark.parametrize("subsystem_id", WITNESS_SUBSYSTEMS)
def test_a_packaged_witness_layout_is_near_planar(subsystem_id):
    """The named cases the redesign was measured on are near-planar, absolutely.

    Before repackaging this compared regenerated output against the file it would
    replace, to show the pipeline had improved without an absolute bar to hold it
    to. The packaged files *are* the redesign now, so the bar is absolute: retinol
    went from a 66x-letterbox star of 983 crossings to this. The witnesses carry
    the redesign per-subsystem, so a resource swap that quietly degraded one fails
    here rather than only moving an aggregate.
    """
    path = STANDARD_DIRECTORY / f"{subsystem_id}.json"
    if not path.exists():
        pytest.skip(f"{subsystem_id} has no automatic layout")

    glyphs, connections = _load(path)

    assert crossings_per_edge(glyphs, connections) <= WITNESS_CROSSINGS_PER_EDGE_CEILING
    assert edge_length_ratio(glyphs, connections) <= WITNESS_EDGE_RATIO_CEILING
    # Hub splitting shortens a star to the arms that hold the drawing together, so
    # a metabolite may still exceed the degree threshold — a metabolite whose every
    # arm is the only path to its reaction is drawn whole, because duplicating it
    # would detach that many reactions. What is bounded is the star that splitting
    # can legitimately remove.
    assert max_metabolite_degree(glyphs, connections) <= WITNESS_MAX_DEGREE_CEILING


@pytest.mark.resource
def test_automatic_layout_regenerates_identically():
    """Two generations of one subsystem must agree, IDs included.

    Without this the packaged resources cannot be regenerated and compared at
    all: set-ordered reaction lists (D-42) and ``uuid4`` glyph IDs made every run
    differ from every other, so no diff could distinguish a layout change from
    noise.
    """
    model = MetabolicModel(use_cache=True)

    first_glyphs, first_connections = create_compartmentalized_graph_from_model(
        "beta_alanine_metabolism", model, DEFAULT_SUBSYSTEM_LAYOUT_ENGINE
    )
    second_glyphs, second_connections = create_compartmentalized_graph_from_model(
        "beta_alanine_metabolism", model, DEFAULT_SUBSYSTEM_LAYOUT_ENGINE
    )

    assert first_glyphs == second_glyphs
    assert first_connections == second_connections


def test_subsystem_membership_is_ordered():
    """Subsystem reaction and gene lists must not depend on set iteration order.

    They are stored as sets, so an unsorted list of them reorders in every
    process. The reaction order reaches the layout; the gene order reaches the
    ORA universe WP6 reports as auditable.
    """
    model = MetabolicModel(use_cache=True)

    for subsystem_id in ("beta_alanine_metabolism", "retinol_metabolism"):
        reactions = model.get_reactions_for_subsystem(subsystem_id)
        genes = model.get_genes_for_subsystem(subsystem_id)

        assert reactions == sorted(reactions)
        assert genes == sorted(genes)


@pytest.mark.resource
def test_generated_glyph_ids_are_unique_across_namespaces():
    """Shared and reaction-local metabolite IDs must not collide.

    A shared metabolite is now ``m_{metabolite}`` and a reaction-local one
    ``m_{reaction}_{metabolite}``. Nothing structurally forbids a metabolite ID
    that makes those equal, so it is checked over every subsystem rather than
    assumed from the Human1 naming convention.
    """
    model = MetabolicModel(use_cache=True)

    collisions = defaultdict(list)
    for subsystem_id in ID_COLLISION_SUBSYSTEMS:
        glyphs, _ = create_compartmentalized_graph_from_model(
            subsystem_id, model, DEFAULT_SUBSYSTEM_LAYOUT_ENGINE
        )
        seen = set()
        for glyph in glyphs:
            if glyph["id"] in seen:
                collisions[subsystem_id].append(glyph["id"])
            seen.add(glyph["id"])

    assert not collisions, f"Duplicate glyph IDs: {dict(collisions)}"

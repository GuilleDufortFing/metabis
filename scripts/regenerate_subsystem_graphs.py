"""Regenerate the packaged automatic (``standard``) subsystem graph resources.

The 137 files under ``resources/graphs/subsystem/standard/`` are the layouts the
product ships for every subsystem Human1 has no curated map for. Until now they
had no committed producer: ``save_visualization_cache`` existed, but nothing in
the generator or the pipeline called it, so the shipped files could be read and
validated but never reproduced or audited. D-60 asks for them to be regenerated
and re-packaged, which is not an auditable act without this script.

Usage::

    python scripts/regenerate_subsystem_graphs.py --check
    python scripts/regenerate_subsystem_graphs.py --write

``--check`` regenerates into a temporary directory and reports how each shipped
file compares, changing nothing. ``--write`` replaces the packaged files.

Reproducibility has three requirements, all of which must hold for a regenerated
file to be comparable to the one it replaces:

1. Stable input order. ``Subsystem.reaction_ids`` is a set, so an unsorted list
   of it orders differently in every process and reaches the layout as the order
   reactions are placed in (D-42). ``get_reactions_for_subsystem`` sorts.
2. Stable glyph IDs. Compartment boundaries and shared metabolite nodes were
   named from ``uuid.uuid4()``, so no two runs agreed even on node identity.
   Both are now derived from what they represent.
3. A fixed Graphviz build. Coordinates come from ``dot``; a different Graphviz
   version may place the same graph differently. The script records the version
   it ran with so a later diff can tell a layout change from a toolchain change.

Requirements 1 and 2 are properties of the library and are asserted by
``tests/viz/test_graph_layout_quality.py``. Requirement 3 is environmental and is
reported, not enforced.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from metabis import MetabolicModel
from metabis.resource_registry import ResourceRegistry
from metabis.viz.metabolic_visualization.generation.graph_building.compartment_graph import (  # noqa: E501
    create_compartmentalized_graph_from_model,
)
from metabis.viz.metabolic_visualization.generation.utilities.cache_utils import (
    DEFAULT_SUBSYSTEM_LAYOUT_ENGINE,
    SUBSYSTEM_STANDARD_VARIANT,
    save_visualization_cache,
)

# What a regenerated file is compared on. Glyph and connection IDs are excluded
# deliberately: the shipped files were written with random IDs, so comparing them
# would report a difference that carries no information.
STRUCTURAL_GLYPH_FIELDS = ("glyph_class", "label", "original_id", "subtype")


def graphviz_version() -> str:
    """Return the Graphviz build coordinates were computed with."""
    try:
        completed = subprocess.run(
            ["dot", "-V"], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return "unavailable"
    return (completed.stderr or completed.stdout).strip() or "unknown"


def standard_variant_directory() -> Path:
    """Return the packaged ``standard`` subsystem-resource directory."""
    base_paths = ResourceRegistry().get_graph_base_paths(required=False)
    return Path(base_paths["subsystem"]) / SUBSYSTEM_STANDARD_VARIANT


def shipped_subsystem_ids(directory: Path) -> List[str]:
    """Return the subsystem IDs that currently ship an automatic layout."""
    return sorted(path.stem for path in directory.glob("*.json"))


def generate(model, subsystem_id: str) -> Tuple[List[Dict], List[Dict]]:
    """Generate one subsystem's automatic layout from the model."""
    return create_compartmentalized_graph_from_model(
        subsystem_id, model, DEFAULT_SUBSYSTEM_LAYOUT_ENGINE
    )


def structural_signature(glyphs: List[Dict], connections: List[Dict]) -> Dict[str, Any]:
    """Reduce a graph to what should survive an ID change.

    Two layouts with the same signature contain the same elements connected the
    same way; they may still differ in coordinates.
    """
    glyph_rows = sorted(
        tuple(str(glyph.get(field, "")) for field in STRUCTURAL_GLYPH_FIELDS)
        for glyph in glyphs
    )
    by_id = {glyph["id"]: glyph for glyph in glyphs}
    connection_rows = sorted(
        (
            str(by_id.get(connection.get("source"), {}).get("original_id", "")),
            str(by_id.get(connection.get("target"), {}).get("original_id", "")),
            str(connection.get("connection_class", "")),
        )
        for connection in connections
    )
    return {
        "glyph_count": len(glyphs),
        "connection_count": len(connections),
        "glyphs": glyph_rows,
        "connections": connection_rows,
    }


def _duplicates(values) -> List[str]:
    """Return the values that appear more than once, in first-seen order."""
    seen: set = set()
    repeated: List[str] = []
    for value in values:
        if value in seen and value not in repeated:
            repeated.append(value)
        seen.add(value)
    return repeated


def compare(shipped_path: Path, glyphs: List[Dict], connections: List[Dict]) -> str:
    """Classify how a regenerated graph relates to the shipped one.

    The classes are the ones the first full run actually produced, which is the
    evidence D-60 rests on:

    ``identical``
        The regenerator reproduces the shipped elements exactly.
    ``direction-only``
        Same glyphs, but some edges run the other way. The shipped file drew a
        reversible reaction's products as its reactants.
    ``incomplete``
        The shipped file is a strict subset: the model contains elements the
        shipped layout omits. No shipped file was ever found to contain an
        element the model does not, so these are stale, not divergent.
    ``duplication-only``
        The same entities connected the same way, drawn with a different number
        of glyphs — the shipped file duplicated a shared metabolite more or less
        than the current pipeline does. ``structural_signature`` compares *sets*
        of rows, so every copy of one metabolite reduces to the same row and a
        change in how many copies exist is invisible to the set comparisons
        below. Without this class such a change was reported as
        ``direction-only (0 edges drawn the other way)``, which named a
        difference that had not occurred.
    """
    with shipped_path.open(encoding="utf-8") as handle:
        shipped = json.load(handle)
    shipped_signature = structural_signature(
        shipped.get("glyphs", []), shipped.get("connections", [])
    )
    regenerated_signature = structural_signature(glyphs, connections)
    if shipped_signature == regenerated_signature:
        return "identical"

    shipped_glyph_rows = set(shipped_signature["glyphs"])
    regenerated_glyph_rows = set(regenerated_signature["glyphs"])
    if shipped_glyph_rows < regenerated_glyph_rows:
        missing = len(regenerated_glyph_rows - shipped_glyph_rows)
        return f"incomplete (shipped omits {missing} glyphs the model contains)"
    if shipped_glyph_rows != regenerated_glyph_rows:
        extra = len(shipped_glyph_rows - regenerated_glyph_rows)
        return f"diverged (shipped carries {extra} glyphs the model does not)"

    shipped_edges = set(shipped_signature["connections"])
    regenerated_edges = set(regenerated_signature["connections"])
    reversed_shipped = {(target, source, kind) for source, target, kind in shipped_edges}
    flipped = len(regenerated_edges - shipped_edges)
    if flipped:
        if regenerated_edges - shipped_edges <= reversed_shipped:
            return f"direction-only ({flipped} edges drawn the other way)"
        return "differs"

    shipped_glyphs = shipped_signature["glyph_count"]
    regenerated_glyphs = regenerated_signature["glyph_count"]
    if shipped_glyphs != regenerated_glyphs:
        delta = regenerated_glyphs - shipped_glyphs
        return f"duplication-only ({delta:+d} glyphs, same entities and edges)"
    return "differs"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="regenerate without writing and report how each shipped file compares",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="replace the packaged standard/ resources",
    )
    parser.add_argument(
        "--subsystem",
        action="append",
        default=None,
        help="regenerate only this subsystem ID (repeatable)",
    )
    args = parser.parse_args(argv)

    directory = standard_variant_directory()
    subsystem_ids = args.subsystem or shipped_subsystem_ids(directory)
    if not subsystem_ids:
        print(f"No automatic subsystem resources found below {directory}")
        return 1

    print(f"Graphviz: {graphviz_version()}")
    print(f"Target:   {directory}")
    print(f"Subsystems: {len(subsystem_ids)}")

    model = MetabolicModel(use_cache=True)
    output_directory = (
        directory if args.write else Path(tempfile.mkdtemp(prefix="metabis-graphs-"))
    )

    summary: Dict[str, int] = {}
    failures: List[str] = []
    for index, subsystem_id in enumerate(subsystem_ids, start=1):
        try:
            glyphs, connections = generate(model, subsystem_id)
        except Exception as exc:  # pragma: no cover - reported, not swallowed
            failures.append(f"{subsystem_id}: {type(exc).__name__}: {exc}")
            print(f"[{index}/{len(subsystem_ids)}] {subsystem_id}: FAILED ({exc})")
            continue

        if not glyphs:
            failures.append(f"{subsystem_id}: produced no glyphs")
            print(f"[{index}/{len(subsystem_ids)}] {subsystem_id}: FAILED (empty)")
            continue

        # Glyph IDs are now derived from what they represent rather than drawn
        # from uuid4, so a naming scheme that collides would silently merge two
        # nodes. Checked here on every regeneration, over every subsystem.
        duplicate_ids = _duplicates(glyph["id"] for glyph in glyphs)
        if duplicate_ids:
            failures.append(f"{subsystem_id}: duplicate glyph IDs {duplicate_ids}")
            print(
                f"[{index}/{len(subsystem_ids)}] {subsystem_id}: "
                f"FAILED (duplicate glyph IDs: {duplicate_ids[:5]})"
            )
            continue

        shipped_path = directory / f"{subsystem_id}.json"
        verdict = (
            compare(shipped_path, glyphs, connections)
            if shipped_path.exists()
            else "new"
        )
        verdict_class = verdict.split()[0]
        summary[verdict_class] = summary.get(verdict_class, 0) + 1

        save_visualization_cache(
            str(output_directory / f"{subsystem_id}.json"), glyphs, connections
        )
        print(
            f"[{index}/{len(subsystem_ids)}] {subsystem_id}: "
            f"{len(glyphs)} glyphs, {len(connections)} connections — {verdict}"
        )

    print()
    for verdict, count in sorted(summary.items()):
        print(f"{verdict}: {count}")
    if failures:
        print(f"\n{len(failures)} failed:")
        for failure in failures:
            print(f"  {failure}")

    if args.write:
        print(f"\nWrote {directory}")
    else:
        print(f"\nWrote nothing; regenerated copies are in {output_directory}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

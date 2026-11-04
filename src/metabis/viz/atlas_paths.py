"""Metabolic Atlas's own link paths, and subsystem bands, for the packaged Human1 maps.

Beside each Human1 layout ``<map>.json``, a compartment map or a subsystem's
curated map, ``scripts/import_human_maps.py`` writes ``<map>.atlas.json``: each
reaction's link paths and arrowheads as Metabolic Atlas draws them, and, for a
compartment map, the band Metabolic Atlas lays under each subsystem's links and
how it sets each metabolite's name (rendering.map_labels). A map
draws a reaction's links along these paths, so a reaction drawn twice, or routed
to many partners along one bent line, no longer crosses the map; a reaction
without them keeps its straight links. They come from the Human-maps SVG files,
which are shared under CC BY-SA 4.0, and so are these files.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

ATLAS_SUFFIX = ".atlas.json"
ATLAS_LICENSE = "CC BY-SA 4.0"
# The Human-maps commit the packaged maps were read from (scripts/import_human_maps.py):
# its maps fitted to Human-GEM 2.1.1, of 2026-10-28.
ATLAS_COMMIT = "8ab97f21dd23cf3afe71f2acf385b6a0c08e4634"
# What a figure drawn from them credits, as CC BY-SA 4.0 asks (Section 3): the
# copyright notice the SVG files hold, where they come from, their licence, that
# they were changed, and that the figure is shared under the same licence. Lines
# short enough to fit under a graph.
ATLAS_CREDIT = (
    f"Map layout adapted from Metabolic Atlas Human-maps (github.com/SysBioChalmers/Human-maps, commit {ATLAS_COMMIT[:7]}),",
    "© Department of Biology and Biological Engineering, Chalmers University of Technology, licensed under CC BY-SA 4.0.",
    "Redrawn by Metabis with its own colours, glyphs and data; this adapted map is shared under CC BY-SA 4.0, without warranty.",
)


@lru_cache(maxsize=8)
def _read(path: str) -> Mapping[str, Any]:
    return json.loads(Path(path).read_text())


def atlas_paths(resource_path: Optional[str]) -> Optional[Mapping[str, Any]]:
    """The Atlas paths beside a layout file, or ``None`` when it has none.

    The result has ``links`` and ``heads`` (each reaction's paths, as flat
    ``[x0, y0, x1, y1, ...]`` lists in the layout's frame), ``subsystems``
    (each subsystem's band, the same way) and, for a compartment map, ``labels``
    (each metabolite's name's lines and place, under its glyph), and must not be
    changed.
    """
    if not resource_path:
        return None
    path = Path(resource_path)
    companion = path.with_name(path.name.removesuffix(".json") + ATLAS_SUFFIX)
    if not companion.is_file():
        return None
    return _read(str(companion))


__all__ = ["ATLAS_COMMIT", "ATLAS_CREDIT", "ATLAS_LICENSE", "ATLAS_SUFFIX", "atlas_paths"]

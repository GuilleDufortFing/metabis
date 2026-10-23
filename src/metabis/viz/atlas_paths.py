"""Metabolic Atlas's own link paths, and subsystem bands, for the packaged Human1 maps.

Beside each Human1 layout ``<map>.json``, a compartment map or a subsystem's
curated map, ``scripts/import_atlas_paths.py`` writes ``<map>.atlas.json``: each
reaction's link paths and arrowheads as Metabolic Atlas draws them, and, for a
compartment map, the band Metabolic Atlas lays under each subsystem's links. A map
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


@lru_cache(maxsize=8)
def _read(path: str) -> Mapping[str, Any]:
    return json.loads(Path(path).read_text())


def atlas_paths(resource_path: Optional[str]) -> Optional[Mapping[str, Any]]:
    """The Atlas paths beside a layout file, or ``None`` when it has none.

    The result has ``links`` and ``heads`` (each reaction's paths, as flat
    ``[x0, y0, x1, y1, ...]`` lists in the layout's frame) and ``subsystems``
    (each subsystem's band, the same way), and must not be changed.
    """
    if not resource_path:
        return None
    path = Path(resource_path)
    companion = path.with_name(path.name.removesuffix(".json") + ATLAS_SUFFIX)
    if not companion.is_file():
        return None
    return _read(str(companion))


__all__ = ["ATLAS_LICENSE", "ATLAS_SUFFIX", "atlas_paths"]

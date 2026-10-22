"""The Metabolic Atlas compartment maps: their names, their reactions, and links into them.

The library packages eleven compartment maps from Metabolic Atlas's Human-maps
(``resources/graphs/compartment/human1``), drawn by the code that draws a
subsystem's graph, so they take the same colours, hovers and key. This module
names them, says which model reactions each draws, and leads between a subsystem's
graph and the maps of its compartments:

* **Names.** Metabolic Atlas splits the cytosol into five maps by content, so each
  part is named by its largest pathways.
* **Membership.** Most of a map's reactions carry Human1's own identifiers; the
  rest carry Recon3D's or BiGG's, which the model translates back.
* **Into a map.** :func:`highlight_reactions` keeps a subsystem's reactions and
  fades the rest of the map, then frames them.
* **Out of a map.** :func:`add_map_links` names each reaction's subsystem, and a
  tap on a reaction dispatches :data:`MAP_REACTION_EVENT` on ``window`` with it,
  for the page to offer that subsystem's graph.
* **Find.** :func:`add_map_find` rings the reactions, genes and metabolites whose
  name holds the query the page writes into its source, and centres the view on
  the first, dispatching :data:`MAP_FIND_EVENT` with what it found.
"""

from __future__ import annotations

import json
import weakref
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from bokeh.models import ColumnDataSource, CustomJS, Range1d, TapTool

from .subsystem_link import attach

MAPS_DIRECTORY = Path(__file__).resolve().parent.parent / "resources/graphs/compartment/human1"
MAP_REACTION_EVENT = "metabis:map-reaction"
MAP_FIND_EVENT = "metabis:map-find"
MAP_FIND_SOURCE = "metabis-map-find"
FIND_RING_COLOR = "#0162b7"  # the client's link blue, as the overview's ring
FADED = 0.08  # what is left of a faded glyph's alpha
# The formats, besides Human1's own, that the maps' reaction identifiers use.
MAP_REACTION_FORMATS = ("rxnRecon3DID", "rxnBiGGID", "rxnHMR2ID")


@dataclass(frozen=True)
class CompartmentMap:
    """One packaged map: its file's name, its name, and its Human1 compartment."""

    map_id: str
    name: str
    compartment: str


COMPARTMENT_MAPS: Tuple[CompartmentMap, ...] = (
    CompartmentMap("cytosol_1", "Cytosol 1 (bile acids, drugs, steroids)", "c"),
    CompartmentMap(
        "cytosol_2", "Cytosol 2 (arachidonic acid, fatty acid oxidation, glycerophospholipids)", "c"
    ),
    CompartmentMap("cytosol_3", "Cytosol 3 (peptides, inositol phosphates, glycosphingolipids)", "c"),
    CompartmentMap("cytosol_4", "Cytosol 4 (xenobiotics, purines)", "c"),
    CompartmentMap("cytosol_5", "Cytosol 5 (aromatic amino acids, arginine and proline)", "c"),
    CompartmentMap("endoplasmic_reticulum", "Endoplasmic reticulum", "r"),
    CompartmentMap("golgi_apparatus", "Golgi apparatus", "g"),
    CompartmentMap("lysosome", "Lysosome", "l"),
    CompartmentMap("mitochondria", "Mitochondria", "m"),
    CompartmentMap("nucleus", "Nucleus", "n"),
    CompartmentMap("peroxisome", "Peroxisome", "x"),
)
MAPS_BY_ID: Dict[str, CompartmentMap] = {entry.map_id: entry for entry in COMPARTMENT_MAPS}


@lru_cache(maxsize=None)
def map_reaction_ids(map_id: str) -> Tuple[str, ...]:
    """The identifiers of the reactions a packaged map draws, once each, in its order."""
    data = json.loads((MAPS_DIRECTORY / f"{map_id}.json").read_text())
    glyphs = data["glyphs"] if isinstance(data, dict) else data
    return tuple(
        dict.fromkeys(str(glyph["id"]) for glyph in glyphs if glyph.get("glyph_class") == "reaction")
    )


_MODEL_REACTIONS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _model_reactions(model) -> Tuple[frozenset, Dict[str, str]]:
    """The model's reaction identifiers, and its other formats' identifiers back to them."""
    cached = _MODEL_REACTIONS.get(model)
    if cached is not None:
        return cached
    own = frozenset(str(reaction) for reaction in model.get_reactions())
    formats = set(getattr(model, "get_available_reaction_formats", lambda: [])())
    back: Dict[str, str] = {}
    for reaction in own:
        for name in MAP_REACTION_FORMATS:
            if name not in formats:
                continue
            other = model.translate_reaction(reaction, input_format="mar_id", output_format=name)
            if isinstance(other, str) and other and other != reaction:
                for part in other.split(";"):
                    back.setdefault(part.strip(), reaction)
    _MODEL_REACTIONS[model] = (own, back)
    return own, back


def resolve_map_reactions(model, map_id: str) -> Dict[str, str]:
    """Each of a map's reaction identifiers that the model knows, to the model's own."""
    own, back = _model_reactions(model)
    resolved = {}
    for reaction in map_reaction_ids(map_id):
        if reaction in own:
            resolved[reaction] = reaction
        elif reaction in back:
            resolved[reaction] = back[reaction]
    return resolved


_REACTION_SUBSYSTEMS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def reaction_subsystems(model) -> Dict[str, str]:
    """Each model reaction's subsystem, the first that lists it."""
    cached = _REACTION_SUBSYSTEMS.get(model)
    if cached is None:
        cached = {}
        for subsystem in model.get_subsystems():
            for reaction in model.get_reactions_for_subsystem(subsystem) or []:
                cached.setdefault(str(reaction), str(subsystem))
        _REACTION_SUBSYSTEMS[model] = cached
    return cached


def map_subsystems(model, map_id: str) -> Dict[str, Tuple[str, str]]:
    """Each reaction a map draws, to its subsystem's identifier and name."""
    owners = reaction_subsystems(model)
    named: Dict[str, Tuple[str, str]] = {}
    for drawn, own in resolve_map_reactions(model, map_id).items():
        subsystem = owners.get(own)
        if subsystem is not None:
            named[drawn] = (subsystem, str(model.get_subsystem_name(subsystem) or subsystem))
    return named


def maps_for_reactions(model, reactions: Iterable[str]) -> List[Tuple[CompartmentMap, int]]:
    """The maps that draw any of ``reactions``, with how many, the most first."""
    wanted = {str(reaction).removesuffix("_back") for reaction in reactions}
    found = []
    for entry in COMPARTMENT_MAPS:
        count = len(set(resolve_map_reactions(model, entry.map_id).values()) & wanted)
        if count:
            found.append((entry, count))
    return sorted(found, key=lambda pair: (-pair[1], COMPARTMENT_MAPS.index(pair[0])))


def map_ids_for_reactions(model, map_id: str, reactions: Iterable[str]) -> List[str]:
    """The identifiers a map draws ``reactions`` under."""
    wanted = {str(reaction).removesuffix("_back") for reaction in reactions}
    return [drawn for drawn, own in resolve_map_reactions(model, map_id).items() if own in wanted]


def _sources(plot):
    for renderer in plot.renderers:
        source = getattr(renderer, "data_source", None)
        if isinstance(source, ColumnDataSource) and source.data:
            yield renderer, source


def _is_reactions(data: Mapping) -> bool:
    return "angle" in data and "element_id" in data


def highlight_reactions(plot, drawn_ids: Iterable[str], *, padding: float = 600.0) -> int:
    """Keep ``drawn_ids``' reactions in a map, fade everything else, and frame them.

    Returns how many reactions it kept. With none, the map is left as drawn.
    """
    keep = {str(drawn).removesuffix("_back") for drawn in drawn_ids}
    xs: List[float] = []
    ys: List[float] = []
    for _renderer, source in _sources(plot):
        data = source.data
        if _is_reactions(data):
            for x, y, element in zip(data["x"], data["y"], data["element_id"]):
                if str(element).removesuffix("_back") in keep:
                    xs.append(float(x))
                    ys.append(float(y))
    if not xs:
        return 0
    for _renderer, source in _sources(plot):
        data = dict(source.data)
        count = len(next(iter(data.values())))
        if _is_reactions(data):
            data["alpha"] = [
                alpha if str(element).removesuffix("_back") in keep else alpha * FADED
                for alpha, element in zip(data["alpha"], data["element_id"])
            ]
        elif "alpha" in data and count:
            data["alpha"] = [alpha * 0.25 for alpha in data["alpha"]]
        if "line_alpha" in data and count and not _is_reactions(data):
            data["line_alpha"] = [alpha * 0.25 for alpha in data["line_alpha"]]
        if "text_alpha" in data and count:
            data["text_alpha"] = [alpha * 0.3 for alpha in data["text_alpha"]]
        source.data = data
    # Frame them at the map's own proportions, so nothing is stretched.
    left, right = min(xs) - padding, max(xs) + padding
    top, bottom = min(ys) - padding, max(ys) + padding
    width, height = right - left, bottom - top
    current_width = abs(float(plot.x_range.end) - float(plot.x_range.start)) or width
    current_height = abs(float(plot.y_range.end) - float(plot.y_range.start)) or height
    aspect = current_width / current_height
    if width / height < aspect:
        width = height * aspect
    else:
        height = width / aspect
    cx, cy = (left + right) / 2, (top + bottom) / 2
    plot.x_range = Range1d(cx - width / 2, cx + width / 2)
    # The map's y-range runs downwards, as its layout does.
    plot.y_range = Range1d(cy + height / 2, cy - height / 2)
    return len(xs)


def add_map_links(plot, subsystem_of: Mapping[str, Tuple[str, str]]) -> None:
    """Name each reaction's subsystem in a map, and offer it on a tap.

    ``subsystem_of`` gives, for each drawn reaction identifier, its subsystem's
    identifier and name. A tap on a reaction dispatches :data:`MAP_REACTION_EVENT`
    with ``{element_id, label, subsystem_id, subsystem}``.
    """
    for renderer, source in _sources(plot):
        data = source.data
        if not _is_reactions(data):
            continue
        pairs = [
            subsystem_of.get(str(element).removesuffix("_back"), ("", ""))
            for element in data["element_id"]
        ]
        data["subsystem_id"] = [pair[0] for pair in pairs]
        data["subsystem"] = [pair[1] for pair in pairs]
        plot.add_tools(TapTool(renderers=[renderer]))
        source.selected.js_on_change(
            "indices",
            CustomJS(
                args=dict(source=source, event=MAP_REACTION_EVENT),
                code="""
                const index = source.selected.indices[0]
                if (index === undefined) return
                const data = source.data
                window.dispatchEvent(new CustomEvent(event, {detail: {
                  element_id: data.element_id[index],
                  label: data.display_label[index],
                  subsystem_id: data.subsystem_id[index],
                  subsystem: data.subsystem[index],
                }}))
                """,
            ),
        )


def add_map_find(plot) -> Optional[ColumnDataSource]:
    """A find for a map: the page writes a query; the map rings what matches and centres on it.

    Reactions, genes and metabolites are found by their name or identifier,
    ignoring case. The view moves to the first match, zoomed in at least to a
    fifth of the map's width, and :data:`MAP_FIND_EVENT` reports how many matched.
    """
    searched = [source for _renderer, source in _sources(plot) if "display_label" in source.data and "x" in source.data]
    if not searched:
        return None
    find = ColumnDataSource({"query": []}, name=MAP_FIND_SOURCE)
    ring = ColumnDataSource({"x": [], "y": []})
    plot.scatter(
        x="x",
        y="y",
        source=ring,
        marker="circle",
        size=34,
        fill_alpha=0,
        line_color=FIND_RING_COLOR,
        line_width=3,
        level="overlay",
    )
    xs = [x for source in searched for x in source.data["x"]]
    full_width = (max(xs) - min(xs)) if xs else 1.0
    find.js_on_change(
        "data",
        CustomJS(
            args=dict(find=find, sources=searched, ring=ring, plot=plot, full_width=full_width, event=MAP_FIND_EVENT),
            code="""
            const query = String(find.data.query[0] ?? '').trim().toLowerCase()
            const found = {x: [], y: []}
            let first = null
            if (query) {
              for (const source of sources) {
                const data = source.data
                const names = data.display_label
                const ids = data.element_id ?? data.id ?? names
                for (let i = 0; i < names.length; i++) {
                  const name = String(names[i] ?? '').toLowerCase()
                  const id = String(ids[i] ?? '').toLowerCase()
                  if (!name.includes(query) && !id.includes(query)) continue
                  found.x.push(data.x[i])
                  found.y.push(data.y[i])
                  if (first === null) first = {x: data.x[i], y: data.y[i], label: String(names[i])}
                }
              }
            }
            ring.data = found
            if (first !== null) {
              const xr = plot.x_range
              const yr = plot.y_range
              const width = Math.min(Math.abs(xr.end - xr.start), full_width / 5)
              const height = width * Math.abs(yr.end - yr.start) / Math.abs(xr.end - xr.start)
              xr.setv({start: first.x - width / 2, end: first.x + width / 2})
              // The map's y-range runs downwards.
              yr.setv({start: first.y + height / 2, end: first.y - height / 2})
            }
            window.dispatchEvent(new CustomEvent(event, {detail: {
              query, count: found.x.length, label: first === null ? null : first.label,
            }}))
            """,
        ),
    )
    attach(plot, find)
    return find


__all__ = [
    "COMPARTMENT_MAPS",
    "CompartmentMap",
    "MAP_FIND_EVENT",
    "MAP_FIND_SOURCE",
    "MAP_REACTION_EVENT",
    "MAPS_BY_ID",
    "add_map_find",
    "add_map_links",
    "highlight_reactions",
    "map_ids_for_reactions",
    "map_reaction_ids",
    "map_subsystems",
    "maps_for_reactions",
    "reaction_subsystems",
    "resolve_map_reactions",
]

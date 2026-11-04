"""The Metabolic Atlas compartment maps: their names, their reactions, and links into them.

The library packages twelve compartment maps from Metabolic Atlas's Human-maps
(``resources/graphs/compartment/human1``), drawn by the code that draws a
subsystem's graph, so they take the same colours, hovers and key. This module
names them, says which model reactions each draws, and leads between a subsystem's
graph and the maps of its compartments:

* **Names.** Metabolic Atlas splits the cytosol into five maps by content, so each
  part is named by its largest pathways.
* **Membership.** A map's reactions carry Human-GEM's identifiers, of the
  release the maps were fitted to (2.1.1): the few that release added are not the
  model's, and take no data. A layout may also name a reaction by Recon3D's or
  BiGG's identifier, as older Human-maps did, which the model translates back.
* **Into a map.** :func:`highlight_reactions` frames a subsystem's reactions,
  and :func:`add_map_bands` lays its band under them.
* **Out of a map.** :func:`add_map_links` names each reaction's subsystem, and a
  tap opens no page: on a reaction it dispatches :data:`MAP_REACTION_EVENT` on
  ``window``, for the page to outline that subsystem's band, offer its graph and
  show the reaction's details; on a gene or a metabolite, :data:`MAP_ELEMENT_EVENT`,
  for its details.
* **Find.** :func:`add_map_find` marks the reactions, genes and metabolites whose
  name holds the query the page writes into its source, or is it, and centres the
  view on one, dispatching :data:`MAP_FIND_EVENT` with what it found; or, for an
  element clicked, marks every place it is drawn where the view is. A subsystem's
  graph takes the same find.
* **Scale.** :func:`keep_map_scale` fits a map, or a subsystem's graph, to its plot
  at one scale across and down, and keeps that scale whatever its plot's size: a
  narrower map shows less of itself, never squeezed.
* **Subsystems.** :func:`add_map_bands` lays Metabolic Atlas's band under one
  subsystem's links at a time, as a soft yellow glow: the subsystem the map was
  opened from, or one the page names: one chosen, or a tapped reaction's.

A map draws its links along Metabolic Atlas's own paths (:mod:`.atlas_paths`),
and its names as Metabolic Atlas sets them, in map units (rendering.map_labels).
"""

from __future__ import annotations

import json
import weakref
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from bokeh.models import ColumnDataSource, CustomJS, TapTool

from .atlas_paths import atlas_paths
from .brand import GRADIENT_JS
from .click_behavior import OWN_TAP_TAG
from .metabolic_visualization.rendering.map_labels import MAP_NAMES_SHOWN_FROM
from .subsystem_link import attach

MAPS_DIRECTORY = Path(__file__).resolve().parent.parent / "resources/graphs/compartment/human1"
MAP_REACTION_EVENT = "metabis:map-reaction"
# A tapped gene or metabolite, whose details the page shows.
MAP_ELEMENT_EVENT = "metabis:map-element"
MAP_FIND_EVENT = "metabis:map-find"
MAP_FIND_SOURCE = "metabis-map-find"
# The subsystem whose band a map shows, which the page writes.
MAP_BAND_SOURCE = "metabis-map-band"
# The find's ring round each match, a fixed size on screen, drawn in the logo's
# gradient as short arcs; and the border it gives a match's own shape.
FIND_RING_RADIUS = 17  # pixels: the ring the find drew before
FIND_RING_PIECES = 48
FIND_RING_WIDTH = 3
FIND_BORDER_WIDTH = 4
# Metabolic Atlas's bands are 80 map units wide, under the links. A highlighted
# subsystem's glows there in GitHub Primer's attention yellows: along it, halfway
# from its yellow-0 (#fff8c5) to its yellow-1 (#fae17d); then yellow-0; then
# halfway from yellow-0 to white at its edge, wider than the band.
BAND_WIDTH = 80.0
BAND_GLOW = ((BAND_WIDTH + 40.0, "#fffbe2"), (BAND_WIDTH + 20.0, "#fff8c5"), (BAND_WIDTH, "#fceca1"))
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
    CompartmentMap("extracellular", "Extracellular", "e"),
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


@lru_cache(maxsize=None)
def map_metabolite_ids(map_id: str) -> frozenset:
    """The identifiers of the metabolites a packaged map draws, those its layout classes as reactions included."""
    data = json.loads((MAPS_DIRECTORY / f"{map_id}.json").read_text())
    glyphs = data["glyphs"] if isinstance(data, dict) else data
    atlas = atlas_paths(str(MAPS_DIRECTORY / f"{map_id}.json")) or {}
    reclassed = set(atlas.get("metabolites") or ())
    return frozenset(
        str(glyph.get("label") or glyph["id"])
        for glyph in glyphs
        if glyph.get("glyph_class") == "metabolite" or str(glyph.get("id")) in reclassed
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


_MODEL_METABOLITES: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def model_metabolite_id(model, metabolite: str) -> str:
    """The model's own identifier of a metabolite a layout names in another format (a map's
    BiGG-style ``12dgr120_m``), or the name as given when the model knows it by no format."""
    known = _MODEL_METABOLITES.setdefault(model, {})
    metabolite = str(metabolite)
    if metabolite not in known:
        registry = model.metabolites
        entity = registry.get(metabolite, "primary")
        for name in registry.get_available_formats():
            if entity is not None:
                break
            if name != "primary":
                entity = registry.get(metabolite, name)
        known[metabolite] = str(entity.id) if entity is not None else metabolite
    return known[metabolite]


def map_metabolites(model, map_id: str) -> frozenset:
    """The model's identifiers of the metabolites a packaged map draws."""
    return frozenset(model_metabolite_id(model, metabolite) for metabolite in map_metabolite_ids(map_id))


def use_model_identifiers(plot, model) -> None:
    """Name what a graph draws by the model's own identifiers, in its ``model_id`` column.

    A Human1 layout names a few reactions by Recon3D's or BiGG's identifiers, a
    map its metabolites by BiGG's, and a reaction's other direction with
    ``_back``: the details a tap opens, the places they count and the find they
    start all ask by the model's identifier, which the column then holds.
    """
    own, back = _model_reactions(model)
    for _renderer, source in _sources(plot):
        data = source.data
        kind = _element_kind(data)
        if "model_id" not in data or kind not in ("reaction", "metabolite"):
            continue
        if kind == "reaction":
            bare = [str(identifier).removesuffix("_back") for identifier in data["model_id"]]
            data["model_id"] = [identifier if identifier in own else back.get(identifier, identifier) for identifier in bare]
        else:
            data["model_id"] = [model_metabolite_id(model, identifier) for identifier in data["model_id"]]


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


def map_reaction_names(model, map_id: str) -> Dict[str, str]:
    """Each reaction a map draws, to its short name (reaction_names), as figures label it."""
    return {
        drawn: model.get_reaction_reference_data(own).get("short_name") or drawn
        for drawn, own in resolve_map_reactions(model, map_id).items()
    }


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


def _element_kind(data: Mapping) -> Optional[str]:
    """What a map source draws, read from its columns: reactions, genes or metabolites."""
    if _is_reactions(data):
        return "reaction"
    if "model_id" not in data or "hover" not in data:
        return None
    if "width" in data and "height" in data:
        return "gene"
    if "size" in data:
        return "metabolite"
    return None


def tap_dispatches(plot, renderer, source, kind: str) -> None:
    """A tap on one of ``source``'s elements dispatches :data:`MAP_ELEMENT_EVENT`, and opens no page."""
    if OWN_TAP_TAG not in source.tags:
        source.tags = [*source.tags, OWN_TAP_TAG]
    # One tap tool a renderer: the click behaviour's may already take its taps.
    if not any(tool.renderers == "auto" or renderer in tool.renderers for tool in plot.select(type=TapTool)):
        plot.add_tools(TapTool(renderers=[renderer]))
    source.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(source=source, event=MAP_ELEMENT_EVENT, kind=kind),
            code="""
            const index = source.selected.indices[0]
            if (index === undefined) return
            const data = source.data
            window.dispatchEvent(new CustomEvent(event, {detail: {
              kind,
              element_id: data.element_id[index],
              model_id: data.model_id[index],
              label: data.display_label[index],
            }}))
            // Nothing stays selected, so no other element is drawn as unselected.
            setTimeout(() => { source.selected.indices = [] }, 0)
            """,
        ),
    )


def highlight_reactions(plot, drawn_ids: Iterable[str], *, padding: float = 600.0) -> int:
    """Frame ``drawn_ids``' reactions in a map, leaving every glyph as drawn.

    The view moves to them at the map's own proportions; the plot keeps its range
    objects, which its names and its band follow as it zooms. Returns how many
    reactions it framed. With none, the map is left as drawn.
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
    plot.x_range.start, plot.x_range.end = cx - width / 2, cx + width / 2
    # The map's y-range runs downwards, as its layout does.
    plot.y_range.start, plot.y_range.end = cy + height / 2, cy - height / 2
    return len(xs)


def add_map_links(
    plot, subsystem_of: Mapping[str, Tuple[str, str]], names: Optional[Mapping[str, str]] = None
) -> None:
    """Name each reaction's subsystem in a map, and offer it on a tap.

    ``subsystem_of`` gives, for each drawn reaction identifier, its subsystem's
    identifier and name, and ``names`` its short name (:func:`map_reaction_names`).
    A tap on a reaction dispatches :data:`MAP_REACTION_EVENT` with
    ``{element_id, model_id, label, name, subsystem_id, subsystem}``, for the page to
    outline the subsystem, offer its graph and show its details; ``name`` is the
    identifier without ``names``. A tap on a gene or a metabolite dispatches
    :data:`MAP_ELEMENT_EVENT` with ``{kind, element_id, model_id, label}``, for the
    page to show its details. No tap opens a page: the sources carry
    :data:`~metabis.viz.click_behavior.OWN_TAP_TAG`, which the click behaviour
    leaves to these callbacks.
    """
    for renderer, source in _sources(plot):
        data = source.data
        kind = _element_kind(data)
        if kind in ("gene", "metabolite"):
            tap_dispatches(plot, renderer, source, kind)
        if not _is_reactions(data):
            continue
        pairs = [
            subsystem_of.get(str(element).removesuffix("_back"), ("", ""))
            for element in data["element_id"]
        ]
        data["subsystem_id"] = [pair[0] for pair in pairs]
        data["subsystem"] = [pair[1] for pair in pairs]
        if names is not None:
            data["short_name"] = [
                names.get(str(element).removesuffix("_back")) or str(element) for element in data["element_id"]
            ]
        if OWN_TAP_TAG not in source.tags:
            source.tags = [*source.tags, OWN_TAP_TAG]
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
                  model_id: (data.model_id ?? data.element_id)[index],
                  label: data.display_label[index],
                  name: (data.short_name ?? data.display_label)[index],
                  subsystem_id: data.subsystem_id[index],
                  subsystem: data.subsystem[index],
                }}))
                // Nothing stays selected, so no other reaction is drawn as unselected.
                setTimeout(() => { source.selected.indices = [] }, 0)
                """,
            ),
        )


def keep_map_scale(plot) -> None:
    """A map shows its view whole at one scale, then less of itself as its frame narrows, never squeezed.

    The view it was drawn with, the whole map or a subsystem's part of it, is
    fitted to its plot whole, centred, at one scale across and down: the plot
    takes the page's shape, not the map's. It is fitted again whenever its frame
    changes height until the reader moves it, so a plot laid out before its box
    took the screen's height is not left small at its top. When only its width
    changes, as when the page sets a sidebar beside it at the same height, or
    once the reader has panned or zoomed it, the map keeps its scale, a map unit
    as many pixels as before across and down, and its top right corner stays
    where it was, so nothing on screen moves: what the sidebar takes is cut from
    the map's left, and a lower frame cuts its bottom. A subsystem's graph keeps
    its scale the same way.
    """
    keep = CustomJS(
        args=dict(plot=plot),
        code="""
        let width, height
        try { width = plot.inner_width; height = plot.inner_height } catch { return }
        const xr = plot.x_range
        const yr = plot.y_range
        const along = xr.end - xr.start
        const upwards = yr.end - yr.start
        const down = Math.abs(upwards)
        // Before the first layout, or before a data range has its span, nothing to keep.
        if (!(width > 0 && height > 0 && down > 0 && Number.isFinite(along) && along !== 0)) return
        const before = plot.metabis_frame
        plot.metabis_frame = [width, height]
        // As a zoom tool does, so a data range keeps the view rather than fitting it again.
        xr.have_updated_interactively = true
        yr.have_updated_interactively = true
        const now = [xr.start, xr.end, yr.start, yr.end]
        // The view drawn, to fit again; and the view last set here, which a pan,
        // a zoom or a find changes.
        plot.metabis_view ??= now
        const set = plot.metabis_set
        const untouched = set !== undefined && set.every((end, i) => Math.abs(end - now[i]) <= 1e-9 * Math.abs(set[i % 2 === 0 ? i + 1 : i - 1] - set[i]))
        if (!before || (untouched && height !== before[1])) {
          const [x0, x1, y0, y1] = plot.metabis_view
          const scale = Math.min(width / Math.abs(x1 - x0), height / Math.abs(y1 - y0))
          const x = (x0 + x1) / 2
          const y = (y0 + y1) / 2
          const across = Math.sign(x1 - x0) * width / scale / 2
          const up = Math.sign(y1 - y0) * height / scale / 2
          xr.setv({start: x - across, end: x + across})
          yr.setv({start: y - up, end: y + up})
          plot.metabis_set = [xr.start, xr.end, yr.start, yr.end]
          return
        }
        // Pixels a map unit took in the frame before; the ranges' ends at the top and right stay.
        const scale = before[1] / down
        const span = width / scale
        if (Math.abs(Math.abs(along) - span) > 1e-9 * span) xr.start = xr.end - Math.sign(along) * span
        const rise = height / scale
        if (Math.abs(down - rise) > 1e-9 * rise) yr.start = yr.end - Math.sign(upwards) * rise
        if (untouched) plot.metabis_set = [xr.start, xr.end, yr.start, yr.end]
        """,
    )
    plot.js_on_change("inner_width", keep)
    plot.js_on_change("inner_height", keep)


def _shape(renderer) -> Optional[dict]:
    """The shape a renderer draws its elements in, in map units, for the find's border round one."""
    glyph = renderer.glyph
    kind = type(glyph).__name__
    if kind == "Rect":
        return {"kind": "rect"}
    if kind == "Ellipse" and isinstance(glyph.width, (int, float)) and isinstance(glyph.height, (int, float)):
        return {"kind": "ellipse", "width": float(glyph.width), "height": float(glyph.height)}
    if kind == "Circle" and isinstance(glyph.radius, (int, float)):
        return {"kind": "circle", "width": 2.0 * glyph.radius, "height": 2.0 * glyph.radius}
    if kind == "Patches":
        return {"kind": "patch"}
    return None


def add_map_find(plot, *, zoom: bool = True) -> Optional[ColumnDataSource]:
    """A find for a map: the page writes a query; the map marks what matches and centres on one.

    Reactions, genes and metabolites are found by their name or identifiers,
    ignoring case: any that holds the query, or, when the page writes ``exact``,
    only those named exactly so (a gene "HADH" without "HADHA"). Each match takes
    a thick border round its own shape, and a ring a fixed size on screen round it
    while the map is too far for its elements to be seen, until its names are
    drawn (map_labels.MAP_NAMES_SHOWN_FROM); both in the logo's gradient, from
    its top left. The view moves to the match the page's ``index`` names, the
    first by default and counting round, zoomed in at least to a fifth of the
    map's width, as Metabolic Atlas's map search steps through its matches.
    Without ``zoom``, as a subsystem's graph is found, the view keeps its zoom and
    moves only for a match outside its middle. When the page writes ``still``, as
    a click marks every place the element it selects is drawn, the view stays
    where it is. :data:`MAP_FIND_EVENT` reports how many matched, which is shown
    and its label.
    """
    searched = [
        (source, _shape(renderer))
        for renderer, source in _sources(plot)
        if "display_label" in source.data and "x" in source.data
    ]
    if not searched:
        return None
    find = ColumnDataSource({"query": []}, name=MAP_FIND_SOURCE)
    ring = ColumnDataSource({"x": [], "y": [], "start": [], "end": [], "color": []})
    border = ColumnDataSource({"x0": [], "y0": [], "x1": [], "y1": [], "color": []})
    # Over the map's shapes, on the glyph level, which the frame clips: on the
    # overlay level a ring beyond the frame was drawn over the key.
    plot.segment(
        x0="x0", y0="y0", x1="x1", y1="y1", line_color="color", line_width=FIND_BORDER_WIDTH,
        line_cap="round", source=border, level="glyph",
    )
    rings = plot.arc(
        x="x", y="y", radius=FIND_RING_RADIUS, radius_units="screen", start_angle="start", end_angle="end",
        line_color="color", line_width=FIND_RING_WIDTH, source=ring, level="glyph",
    )
    xs = [x for source, _shape_of in searched for x in source.data["x"]]
    full_width = (max(xs) - min(xs)) if xs else 1.0
    find.js_on_change(
        "data",
        CustomJS(
            args=dict(
                find=find,
                sources=[source for source, _shape_of in searched],
                shapes=[shape or {"kind": "none"} for _source, shape in searched],
                ring=ring,
                border=border,
                plot=plot,
                closest=full_width / 5 if zoom else 0.0,
                event=MAP_FIND_EVENT,
                pieces=FIND_RING_PIECES,
            ),
            code=GRADIENT_JS
            + """
            const query = String(find.data.query[0] ?? '').trim().toLowerCase()
            const exact = Boolean(find.data.exact?.[0])
            const still = Boolean(find.data.still?.[0])
            const matches = (text) => exact ? text === query : text.includes(query)
            // Where y grows down the screen, as a map's and a graph's do.
            const down = Math.sign(plot.y_range.start - plot.y_range.end) || 1
            const outline = (shape, data, i) => {
              const x = data.x[i], y = data.y[i]
              if (shape.kind === 'patch') return Array.from(data.xs[i], (px, k) => [px, data.ys[i][k]])
              if (shape.kind === 'rect') {
                const w = data.width[i] / 2, h = data.height[i] / 2
                return [[x - w, y - h], [x + w, y - h], [x + w, y + h], [x - w, y + h]]
              }
              if (shape.kind !== 'ellipse' && shape.kind !== 'circle') return []
              return Array.from({length: 48}, (_, k) => [
                x + shape.width / 2 * Math.cos(2 * Math.PI * k / 48),
                y + shape.height / 2 * Math.sin(2 * Math.PI * k / 48),
              ])
            }
            const edges = {x0: [], y0: [], x1: [], y1: [], color: []}
            const surround = (points) => {
              if (points.length < 2) return
              const xs = points.map((point) => point[0]), ys = points.map((point) => point[1] * down)
              const left = Math.min(...xs), top = Math.min(...ys)
              const span = Math.max(Math.max(...xs) - left, 1e-9) + Math.max(Math.max(...ys) - top, 1e-9)
              // A side of a box or a diamond in short pieces, so its colour runs along it.
              const pieces = points.length > 8 ? 1 : 12
              points.forEach(([ax, ay], k) => {
                const [bx, by] = points[(k + 1) % points.length]
                for (let j = 0; j < pieces; j++) {
                  const x0 = ax + (bx - ax) * j / pieces, y0 = ay + (by - ay) * j / pieces
                  const x1 = ax + (bx - ax) * (j + 1) / pieces, y1 = ay + (by - ay) * (j + 1) / pieces
                  edges.x0.push(x0); edges.y0.push(y0); edges.x1.push(x1); edges.y1.push(y1)
                  edges.color.push(metabisGradient(((x0 + x1) / 2 - left + (y0 + y1) / 2 * down - top) / span))
                }
              })
            }
            const arcs = {x: [], y: [], start: [], end: [], color: []}
            const found = {x: [], y: [], label: []}
            if (query) {
              sources.forEach((source, s) => {
                const data = source.data
                const names = data.display_label
                const ids = data.element_id ?? data.id ?? names
                const models = data.model_id ?? ids
                for (let i = 0; i < names.length; i++) {
                  const texts = [names[i], ids[i], models[i]].map((text) => String(text ?? '').toLowerCase())
                  if (!texts.some(matches)) continue
                  found.x.push(data.x[i])
                  found.y.push(data.y[i])
                  found.label.push(String(names[i]))
                  surround(outline(shapes[s], data, i))
                  // A ring's arcs: angles turn anticlockwise on screen, whichever
                  // way the axes run, so its top left takes the gradient's start.
                  for (let k = 0; k < pieces; k++) {
                    const a0 = 2 * Math.PI * k / pieces, a1 = 2 * Math.PI * (k + 1) / pieces, am = (a0 + a1) / 2
                    arcs.x.push(data.x[i]); arcs.y.push(data.y[i]); arcs.start.push(a0); arcs.end.push(a1)
                    arcs.color.push(metabisGradient((Math.SQRT2 + Math.cos(am) - Math.sin(am)) / (2 * Math.SQRT2)))
                  }
                }
              })
            }
            const count = found.x.length
            const index = count ? ((Number(find.data.index?.[0] ?? 0) % count) + count) % count : 0
            const first = count ? {x: found.x[index], y: found.y[index], label: found.label[index]} : null
            ring.data = arcs
            border.data = edges
            // A graph that keeps its zoom moves only to bring a match in from its edges.
            const within = (value, range) => {
              const low = Math.min(range.start, range.end), high = Math.max(range.start, range.end)
              return value >= low + (high - low) * 0.1 && value <= high - (high - low) * 0.1
            }
            const seen = first !== null && closest === 0 && within(first.x, plot.x_range) && within(first.y, plot.y_range)
            if (first !== null && !seen && !still) {
              const xr = plot.x_range
              const yr = plot.y_range
              // As a zoom tool does, so a data range keeps the view rather than
              // computing it again from the data.
              xr.have_updated_interactively = true
              yr.have_updated_interactively = true
              const along = xr.end - xr.start, upwards = yr.end - yr.start
              const width = closest > 0 ? Math.min(Math.abs(along), closest) : Math.abs(along)
              const height = width * Math.abs(upwards) / Math.abs(along)
              xr.setv({start: first.x - Math.sign(along) * width / 2, end: first.x + Math.sign(along) * width / 2})
              yr.setv({start: first.y - Math.sign(upwards) * height / 2, end: first.y + Math.sign(upwards) * height / 2})
            }
            window.dispatchEvent(new CustomEvent(event, {detail: {
              query, count, index, label: first === null ? null : first.label, still,
            }}))
            """,
        ),
    )
    # The ring goes once the elements are near enough to be seen, their borders stay.
    near = CustomJS(
        args=dict(plot=plot, rings=rings, shown_from=MAP_NAMES_SHOWN_FROM),
        code="""
        let height
        try { height = plot.inner_height } catch { return }
        const span = Math.abs(plot.y_range.end - plot.y_range.start)
        if (!(height > 0 && span > 0)) return
        const far = height / span < shown_from
        if (rings.visible !== far) rings.visible = far
        """,
    )
    plot.y_range.js_on_change("start", near)
    plot.y_range.js_on_change("end", near)
    plot.js_on_change("inner_height", near)
    attach(plot, find)
    return find


def map_bands(map_id: str) -> Dict[str, list]:
    """Each subsystem's band in a packaged map, by Metabolic Atlas's name for it."""
    atlas = atlas_paths(str(MAPS_DIRECTORY / f"{map_id}.json"))
    return dict(atlas.get("subsystems") or {}) if atlas else {}


def band_name(bands: Iterable[str], subsystem_name: Optional[str]) -> Optional[str]:
    """The band a subsystem has in a map, matched to its name ignoring case."""
    if not subsystem_name:
        return None
    wanted = str(subsystem_name).strip().lower()
    return next((name for name in bands if name.strip().lower() == wanted), None)


def add_map_bands(plot, map_id: str, shown: Optional[str] = None) -> Optional[ColumnDataSource]:
    """Lay Metabolic Atlas's band under one subsystem's links, as a soft yellow glow.

    ``shown`` is the band shown first, by its name in :func:`map_bands`. The page
    writes another name, or none, into the returned source (:data:`MAP_BAND_SOURCE`,
    ``{"name": [...]}``), for a band chosen or a reaction clicked; pointing changes
    nothing. The glow is a background, not a line: the map has lines enough. It is
    three strokes along the band, each narrower and yellower than the one under it
    (:data:`BAND_GLOW`), so its edge fades into the page; all opaque, so where a
    band crosses itself it merges rather than darkens. A stroke keeps its width in
    map units at every zoom. All lie under everything but the compartments.
    Returns ``None`` for a map without bands.
    """
    bands = map_bands(map_id)
    if not bands:
        return None
    every: dict = {"name": [], "xs": [], "ys": []}
    for name, lines in bands.items():
        for line in lines:
            every["name"].append(name)
            every["xs"].append(list(line[0::2]))
            every["ys"].append(list(line[1::2]))
    every = ColumnDataSource(every)
    drawn = ColumnDataSource({"xs": [], "ys": []})
    # On the links' own level, before them: under the links and the nodes, over
    # the compartments only.
    glow = [
        plot.multi_line(
            xs="xs", ys="ys", line_color=color, line_width=1, line_cap="round", line_join="round",
            level="image", source=drawn,
        )
        for _units, color in BAND_GLOW
    ]
    renderers = [renderer for renderer in plot.renderers if renderer not in glow]
    after = max(
        (
            index + 1
            for index, renderer in enumerate(renderers)
            if "x_label" in (getattr(getattr(renderer, "data_source", None), "data", None) or {})
        ),
        default=0,
    )
    plot.renderers = renderers[:after] + glow + renderers[after:]

    chosen = ColumnDataSource({"name": [shown] if shown in bands else []}, name=MAP_BAND_SOURCE)
    # Only the page changes the band: one that followed the pointer would repaint
    # the whole map at every reaction the pointer crossed.
    show = CustomJS(
        args=dict(every=every, drawn=drawn, chosen=chosen),
        code="""
        const name = (chosen.data.name[0] ?? '').toLowerCase()
        const rows = every.data.name.map((band, i) => band.toLowerCase() === name ? i : -1).filter((i) => i >= 0)
        drawn.data = {xs: rows.map((i) => every.data.xs[i]), ys: rows.map((i) => every.data.ys[i])}
        """,
    )
    chosen.js_on_change("data", show)
    # Each stroke's width in map units, whatever the zoom.
    width = CustomJS(
        args=dict(plot=plot, glow=glow, units=[units for units, _color in BAND_GLOW]),
        code="""
        let pixels
        try { pixels = plot.inner_width } catch { return }
        const span = Math.abs(plot.x_range.end - plot.x_range.start)
        if (!(pixels > 0 && span > 0)) return
        glow.forEach((stroke, i) => { stroke.glyph.line_width = Math.max(1, units[i] * pixels / span) })
        """,
    )
    for attribute in ("start", "end"):
        plot.x_range.js_on_change(attribute, width)
    plot.js_on_change("inner_width", width)
    # The band shown first.
    rows = [i for i, name in enumerate(every.data["name"]) if name == shown]
    drawn.data = {key: [every.data[key][i] for i in rows] for key in ("xs", "ys")}
    attach(plot, every)
    attach(plot, chosen)
    return chosen


__all__ = [
    "BAND_GLOW",
    "BAND_WIDTH",
    "COMPARTMENT_MAPS",
    "CompartmentMap",
    "MAP_BAND_SOURCE",
    "MAP_ELEMENT_EVENT",
    "MAP_FIND_EVENT",
    "MAP_FIND_SOURCE",
    "MAP_REACTION_EVENT",
    "MAPS_BY_ID",
    "add_map_bands",
    "add_map_find",
    "band_name",
    "map_bands",
    "add_map_links",
    "highlight_reactions",
    "keep_map_scale",
    "map_ids_for_reactions",
    "map_metabolite_ids",
    "map_metabolites",
    "map_reaction_ids",
    "map_reaction_names",
    "map_subsystems",
    "maps_for_reactions",
    "model_metabolite_id",
    "reaction_subsystems",
    "resolve_map_reactions",
    "tap_dispatches",
    "use_model_identifiers",
]

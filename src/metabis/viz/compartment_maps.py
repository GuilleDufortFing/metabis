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
* **Into a map.** :func:`highlight_reactions` frames a subsystem's reactions,
  and :func:`add_map_bands` lays its band under them.
* **Out of a map.** :func:`add_map_links` names each reaction's subsystem, and a
  tap opens no page: on a reaction it dispatches :data:`MAP_REACTION_EVENT` on
  ``window``, for the page to outline that subsystem's band, offer its graph and
  show the reaction's details; on a gene or a metabolite, :data:`MAP_ELEMENT_EVENT`,
  for its details.
* **Find.** :func:`add_map_find` rings the reactions, genes and metabolites whose
  name holds the query the page writes into its source, or is it, and centres the
  view on one, dispatching :data:`MAP_FIND_EVENT` with what it found.
* **Scale.** :func:`keep_map_scale` keeps a map at one scale across and down
  whatever its plot's size: a narrower map shows less of itself, never squeezed.
* **Subsystems.** :func:`add_map_bands` outlines Metabolic Atlas's band under one
  subsystem's links at a time, in the logo's gradient: the subsystem the map was
  opened from, or one the page names: one chosen, or a tapped reaction's.

A map draws its links along Metabolic Atlas's own paths (:mod:`.atlas_paths`).
"""

from __future__ import annotations

import json
import weakref
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from bokeh.models import ColumnDataSource, CustomJS, TapTool

from ..visual_config import theme
from .atlas_paths import atlas_paths
from .brand import OUTLINE_WIDTH, gradient_segments
from .click_behavior import OWN_TAP_TAG
from .subsystem_link import attach

MAPS_DIRECTORY = Path(__file__).resolve().parent.parent / "resources/graphs/compartment/human1"
MAP_REACTION_EVENT = "metabis:map-reaction"
# A tapped gene or metabolite, whose details the page shows.
MAP_ELEMENT_EVENT = "metabis:map-element"
MAP_FIND_EVENT = "metabis:map-find"
MAP_FIND_SOURCE = "metabis-map-find"
# The subsystem whose band a map shows, which the page writes.
MAP_BAND_SOURCE = "metabis-map-band"
FIND_RING_COLOR = theme.SELECTED  # the logo's blue, as the overview's ring
# Metabolic Atlas's bands are 80 map units wide, under the links.
BAND_WIDTH = 80.0
# Map units per segment of a band's outline, short enough for its gradient.
BAND_PIECE = 120.0
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


def _tap_dispatches(plot, renderer, source, kind: str) -> None:
    """A tap on one of ``source``'s elements dispatches :data:`MAP_ELEMENT_EVENT`, and opens no page."""
    if OWN_TAP_TAG not in source.tags:
        source.tags = [*source.tags, OWN_TAP_TAG]
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
            _tap_dispatches(plot, renderer, source, kind)
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
    """A map shows less of itself as it narrows, never squeezed: across as down, one scale.

    A map is drawn at its own proportions. When its frame changes size, as when the
    page sets a sidebar beside it at the same height, its horizontal span follows its
    vertical one, so a map unit takes as many pixels across as down. The right edge
    stays where it was, so nothing on screen moves: what the sidebar takes is cut from
    the map's left.
    """
    keep = CustomJS(
        args=dict(plot=plot),
        code="""
        let width, height
        try { width = plot.inner_width; height = plot.inner_height } catch { return }
        const xr = plot.x_range
        const yr = plot.y_range
        const along = xr.end - xr.start
        const down = Math.abs(yr.end - yr.start)
        // Before the first layout, or before a data range has its span, nothing to keep.
        if (!(width > 0 && height > 0 && down > 0 && Number.isFinite(along) && along !== 0)) return
        const span = down * width / height
        if (Math.abs(Math.abs(along) - span) <= 1e-9 * span) return
        // As a zoom tool does, so a data range keeps the view rather than fitting it again.
        xr.have_updated_interactively = true
        xr.start = xr.end - Math.sign(along) * span
        """,
    )
    plot.js_on_change("inner_width", keep)
    plot.js_on_change("inner_height", keep)


def add_map_find(plot) -> Optional[ColumnDataSource]:
    """A find for a map: the page writes a query; the map rings what matches and centres on one.

    Reactions, genes and metabolites are found by their name or identifiers,
    ignoring case: any that holds the query, or, when the page writes ``exact``,
    only those named exactly so (a gene "HADH" without "HADHA"). The view moves to
    the match the page's ``index`` names, the first by default and counting round,
    zoomed in at least to a fifth of the map's width, as Metabolic Atlas's map
    search steps through its matches. :data:`MAP_FIND_EVENT` reports how many
    matched, which is shown and its label.
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
            const exact = Boolean(find.data.exact?.[0])
            const matches = (text) => exact ? text === query : text.includes(query)
            const found = {x: [], y: [], label: []}
            if (query) {
              for (const source of sources) {
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
                }
              }
            }
            const count = found.x.length
            const index = count ? ((Number(find.data.index?.[0] ?? 0) % count) + count) % count : 0
            const first = count ? {x: found.x[index], y: found.y[index], label: found.label[index]} : null
            ring.data = {x: found.x, y: found.y}
            if (first !== null) {
              const xr = plot.x_range
              const yr = plot.y_range
              // As a zoom tool does, so a data range keeps the view rather than
              // computing it again from the data.
              xr.have_updated_interactively = true
              yr.have_updated_interactively = true
              const width = Math.min(Math.abs(xr.end - xr.start), full_width / 5)
              const height = width * Math.abs(yr.end - yr.start) / Math.abs(xr.end - xr.start)
              xr.setv({start: first.x - width / 2, end: first.x + width / 2})
              // The map's y-range runs downwards.
              yr.setv({start: first.y + height / 2, end: first.y - height / 2})
            }
            window.dispatchEvent(new CustomEvent(event, {detail: {
              query, count, index, label: first === null ? null : first.label,
            }}))
            """,
        ),
    )
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
    """Outline Metabolic Atlas's band under one subsystem's links, in the logo's gradient.

    ``shown`` is the band shown first, by its name in :func:`map_bands`. The page
    writes another name, or none, into the returned source (:data:`MAP_BAND_SOURCE`,
    ``{"name": [...]}``), for a band chosen or a reaction clicked; pointing changes
    nothing. A band keeps Metabolic Atlas's width in map units at
    every zoom. It is outlined, never filled, so it is not read as a score: a
    stroke in the gradient, and over it a white stroke narrower by the outline on
    each side, both opaque, so where a band crosses itself the outlines merge
    rather than darken. Both lie under everything but the compartments. Returns
    ``None`` for a map without bands.
    """
    bands = map_bands(map_id)
    if not bands:
        return None
    every: dict = {"name": [], "x0": [], "y0": [], "x1": [], "y1": [], "color": []}
    for name, lines in bands.items():
        # Each band runs the whole gradient across its own extent.
        pieces = gradient_segments(lines, piece=BAND_PIECE)
        every["name"].extend([name] * len(pieces["x0"]))
        for key, values in pieces.items():
            every[key].extend(values)
    every = ColumnDataSource(every)
    drawn = ColumnDataSource({key: [] for key in ("x0", "y0", "x1", "y1", "color")})
    # On the links' own level, before them: under the links and the nodes, over
    # the compartments only.
    common = dict(x0="x0", y0="y0", x1="x1", y1="y1", line_cap="round", level="image", source=drawn)
    outline = plot.segment(line_color="color", line_width=1, **common)
    core = plot.segment(line_color="#ffffff", line_width=0, **common)
    renderers = [renderer for renderer in plot.renderers if renderer not in (outline, core)]
    after = max(
        (
            index + 1
            for index, renderer in enumerate(renderers)
            if "x_label" in (getattr(getattr(renderer, "data_source", None), "data", None) or {})
        ),
        default=0,
    )
    plot.renderers = renderers[:after] + [outline, core] + renderers[after:]

    chosen = ColumnDataSource({"name": [shown] if shown in bands else []}, name=MAP_BAND_SOURCE)
    # Only the page changes the band: one that followed the pointer would repaint
    # the whole map at every reaction the pointer crossed.
    show = CustomJS(
        args=dict(every=every, drawn=drawn, chosen=chosen),
        code="""
        const name = (chosen.data.name[0] ?? '').toLowerCase()
        const rows = every.data.name.map((band, i) => band.toLowerCase() === name ? i : -1).filter((i) => i >= 0)
        const pick = (key) => rows.map((i) => every.data[key][i])
        drawn.data = {x0: pick('x0'), y0: pick('y0'), x1: pick('x1'), y1: pick('y1'), color: pick('color')}
        """,
    )
    chosen.js_on_change("data", show)
    # Metabolic Atlas's width in map units, whatever the zoom; the outline stays
    # a fixed width on screen.
    width = CustomJS(
        args=dict(plot=plot, outline=outline, core=core, units=BAND_WIDTH, edge=OUTLINE_WIDTH),
        code="""
        let pixels
        try { pixels = plot.inner_width } catch { return }
        const span = Math.abs(plot.x_range.end - plot.x_range.start)
        if (!(pixels > 0 && span > 0)) return
        const across = Math.max(2 * edge, units * pixels / span)
        outline.glyph.line_width = across
        core.glyph.line_width = across - 2 * edge
        """,
    )
    for attribute in ("start", "end"):
        plot.x_range.js_on_change(attribute, width)
    plot.js_on_change("inner_width", width)
    # The band shown first.
    rows = [i for i, name in enumerate(every.data["name"]) if name == shown]
    drawn.data = {key: [every.data[key][i] for i in rows] for key in ("x0", "y0", "x1", "y1", "color")}
    attach(plot, every)
    attach(plot, chosen)
    return chosen


__all__ = [
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
    "map_reaction_ids",
    "map_reaction_names",
    "map_subsystems",
    "maps_for_reactions",
    "reaction_subsystems",
    "resolve_map_reactions",
]

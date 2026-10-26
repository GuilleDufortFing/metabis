"""Transport subsystems drawn by transporter family, in the subsystem rankings' layout.

Even divided by compartment (``domain.metabolic_model.loaders.transport_part``), a
transport subsystem is mostly one metabolite crossing one membrane per reaction:
drawn as a graph it is parallel lines, and the largest has thousands. So the
graph card draws one as rows instead, in the rankings' layout
(:mod:`.ranking_layout`): one row per transporter family, its reactions as dots,
with the family's own score.

* **Families.** HGNC's transporter gene groups, read from the gene symbols each
  reaction's rule names: the solute carriers by family (SLC7, SLCO1), the ABC
  transporters by subfamily (ABCA) and the aquaporins (AQP). A reaction whose
  genes name several families, only genes of no transporter family, or no gene
  at all, takes a row for those.
* **Exchange and demand.** Their reactions each exchange one metabolite, so they
  take a row per metabolite.
* **Size.** The families with the most measured reactions take rows of their
  own; the rest share one.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from html import escape
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from ..managers.hover import panel as hover_panel
from ..managers.hover.sections.subsystem_info import delta_equation

TRANSPORT_SUBSYSTEMS = frozenset(
    {
        "transport_extracellular",
        "transport_mitochondrial",
        "transport_endoplasmic_reticular",
        "transport_lysosomal",
        "transport_peroxisomal",
        "transport_golgi_apparatus",
        "transport_nuclear",
        "transport_between_organelles",
        "transport_within_one_compartment",
    }
)
EXCHANGE_SUBSYSTEM = "exchange_demand_reactions"
# The most rows a view draws before the rest share one.
MOST_ROWS = 20

# HGNC's names for the transporter gene groups Human1's transport reactions name
# (genenames.org, gene groups), shortened to what the family carries.
FAMILY_NAMES: Dict[str, str] = {
    "SLC1": "glutamate and neutral amino acid transporters",
    "SLC2": "glucose transporters (GLUT)",
    "SLC3": "amino acid transporter heavy chains",
    "SLC4": "bicarbonate transporters",
    "SLC5": "sodium glucose cotransporters",
    "SLC6": "sodium and chloride dependent transporters",
    "SLC7": "cationic amino acid transporters",
    "SLC9": "sodium hydrogen exchangers",
    "SLC10": "sodium bile acid cotransporters",
    "SLC12": "cation chloride cotransporters",
    "SLC13": "sodium sulfate and carboxylate cotransporters",
    "SLC15": "proton oligopeptide cotransporters",
    "SLC16": "monocarboxylate transporters",
    "SLC17": "vesicular glutamate and phosphate transporters",
    "SLC19": "folate and thiamine transporters",
    "SLC22": "organic cation and anion transporters",
    "SLC25": "mitochondrial carriers",
    "SLC26": "multifunctional anion exchangers",
    "SLC27": "fatty acid transporters",
    "SLC28": "sodium coupled nucleoside transporters",
    "SLC29": "facilitative nucleoside transporters",
    "SLC30": "zinc efflux transporters",
    "SLC35": "nucleotide sugar transporters",
    "SLC36": "proton coupled amino acid transporters",
    "SLC38": "sodium coupled neutral amino acid transporters",
    "SLC39": "zinc importers",
    "SLC43": "system L like amino acid transporters",
    "SLC44": "choline like transporters",
    "SLCO1": "organic anion transporting polypeptides",
    "SLCO2": "organic anion transporting polypeptides",
    "ABCA": "ATP binding cassette, subfamily A",
    "ABCB": "ATP binding cassette, subfamily B",
    "ABCC": "ATP binding cassette, subfamily C",
    "ABCD": "ATP binding cassette, subfamily D",
    "ABCG": "ATP binding cassette, subfamily G",
    "AQP": "aquaporins",
}
SEVERAL_FAMILIES = "Several transporter families"
OTHER_GENES = "Genes of no transporter family"
NO_GENE = "No gene"
OTHER_FAMILIES = "Other transporter families"
_FAMILY = re.compile(r"^(SLCO\d+|SLC\d+|ABC[A-G]|AQP)")


def transporter_family(symbol: str) -> Optional[str]:
    """The HGNC transporter group of a gene symbol, or None."""
    match = _FAMILY.match(symbol.strip().upper())
    return match.group(1) if match else None


def is_transport_subsystem(subsystem_id: str) -> bool:
    """Whether the graph card draws this subsystem by family, not as a graph."""
    return subsystem_id in TRANSPORT_SUBSYSTEMS or subsystem_id == EXCHANGE_SUBSYSTEM


@dataclass
class TransportRow:
    """One row of a transport view: its key, its name, and its reactions and genes."""

    key: str
    name: str
    reactions: List[str] = field(default_factory=list)
    genes: List[str] = field(default_factory=list)


def _symbols(rule: object) -> List[str]:
    if not isinstance(rule, str):
        return []
    return [token for token in re.findall(r"[A-Za-z][A-Za-z0-9-]*", rule) if token.lower() not in ("and", "or")]


def _exchanged(equation: str) -> str:
    """The metabolite an exchange or demand reaction exchanges, without its compartment."""
    sides = [side.strip() for side in re.split(r"<=>|=>|<=", equation or "")]
    side = next((side for side in sides if side), "")
    first = side.split(" + ")[0]
    first = re.sub(r"^\d+(\.\d+)?\s+", "", first)
    return re.sub(r"\[\w\]$", "", first).strip() or "Unnamed metabolite"


def _family_label(key: str) -> str:
    name = FAMILY_NAMES.get(key)
    return f"{key}, {name}" if name else key


def transport_rows(
    model,
    subsystem_id: str,
    measured: Optional[Iterable[str]] = None,
    significant: Iterable[str] = (),
) -> List[TransportRow]:
    """The rows of a transport subsystem's view, the most measured first.

    ``measured`` holds the reactions (or genes) with data, and ``significant``
    those found significant, each counting twice: together they order the rows
    and decide which take a row of their own. Without ``measured``, every
    reaction counts.
    """
    rows: Dict[str, TransportRow] = {}
    for reaction_id in model.get_reactions_for_subsystem(subsystem_id) or []:
        reaction = model.reactions.get(reaction_id)
        if reaction is None:
            continue
        genes = [str(gene) for gene in getattr(reaction, "associated_genes", None) or []]
        if subsystem_id == EXCHANGE_SUBSYSTEM:
            key = name = _exchanged(reaction.equation)
        else:
            families = {
                family
                for symbol in _symbols(getattr(reaction, "gene_rule_symbol", ""))
                if (family := transporter_family(symbol)) is not None
            }
            if len(families) == 1:
                key = next(iter(families))
                name = _family_label(key)
            elif families:
                key = name = SEVERAL_FAMILIES
            elif genes:
                key = name = OTHER_GENES
            else:
                key = name = NO_GENE
        row = rows.setdefault(key, TransportRow(key, name))
        row.reactions.append(str(reaction_id))
        row.genes.extend(gene for gene in genes if gene not in row.genes)

    wanted = None if measured is None else {str(element) for element in measured}
    flagged = {str(element) for element in significant}

    def weight(row: TransportRow) -> int:
        if wanted is None:
            return len(row.reactions)
        return sum(
            (element in wanted or f"{element}_back" in wanted)
            + (element in flagged or f"{element}_back" in flagged)
            for element in [*row.reactions, *row.genes]
        )

    special = (SEVERAL_FAMILIES, OTHER_GENES, NO_GENE)
    ordered = sorted(
        (row for row in rows.values() if row.key not in special and weight(row)),
        key=lambda row: (-weight(row), row.key),
    )
    kept, rest = ordered[:MOST_ROWS], ordered[MOST_ROWS:]
    if rest:
        other = TransportRow(
            OTHER_FAMILIES,
            f"{OTHER_FAMILIES if subsystem_id != EXCHANGE_SUBSYSTEM else 'Other metabolites'} ({len(rest)})",
        )
        for row in rest:
            other.reactions.extend(row.reactions)
            other.genes.extend(row.genes)
        kept.append(other)
    kept.extend(rows[key] for key in special if key in rows and weight(rows[key]))
    return kept


@dataclass(frozen=True)
class GroupDelta:
    """A group's Delta score, and how many of its elements were measured and significant."""

    score: float
    measured: int
    significant: int


def group_delta(
    test_results,
    elements: Sequence[str],
    significant: Iterable[str],
    delta_options: Mapping[str, str],
) -> Optional[GroupDelta]:
    """A group's Delta score, as a subsystem's is computed (SubsystemAnalysis).

    The transformed metric, summed over the numerator's elements, over the count
    of the denominator's; None when no element has data.
    """
    from ..analysis.subsystem_analysis import SubsystemAnalysis

    metric = delta_options.get("metric", "cohens_d")
    if test_results is None or metric not in test_results.columns:
        return None
    present = [element for element in elements if element in test_results.index]
    values = test_results.loc[present, metric].dropna().astype(float)
    if values.empty:
        return None
    significant = set(significant)
    flagged = [index for index in values.index if index in significant]
    denominator = flagged if delta_options.get("denominator_filter") == "significant" else list(values.index)
    if not denominator:
        return GroupDelta(0.0, len(values), len(flagged))
    numerator = flagged if delta_options.get("numerator_filter", "significant") == "significant" else list(values.index)
    transform = delta_options.get("transform", "sign")
    total = sum(SubsystemAnalysis._apply_delta_transform(float(values[index]), transform) for index in numerator)
    return GroupDelta(total / len(denominator), len(values), len(flagged))


def group_delta_score(
    test_results,
    elements: Sequence[str],
    significant: Iterable[str],
    delta_options: Mapping[str, str],
) -> Optional[float]:
    """A group's Delta score alone (:func:`group_delta`)."""
    delta = group_delta(test_results, elements, significant, delta_options)
    return None if delta is None else delta.score


def _row_label(subsystem_id: str, row: TransportRow) -> str:
    """What a row is, as its hover names it."""
    if subsystem_id == EXCHANGE_SUBSYSTEM:
        return "Metabolite"
    if row.key in (SEVERAL_FAMILIES, OTHER_GENES, NO_GENE, OTHER_FAMILIES):
        return "Reactions"
    return "Transporter family"


def _delta_hover(subsystem_id: str, subsystem: str, row: TransportRow, delta: GroupDelta, options, noun: str) -> str:
    """A row's hover, as a subsystem node's: its name, its significant elements and its Delta score."""
    share = delta.significant / delta.measured * 100 if delta.measured else 0.0
    return hover_panel.panel(
        hover_panel.row(_row_label(subsystem_id, row), escape(row.name)),
        hover_panel.row("Subsystem", escape(subsystem)),
        hover_panel.row(
            f"Significant {noun.capitalize()}",
            f"{delta.significant} out of {delta.measured} ({share:.1f}%)",
        ),
        "<div style='margin-top:4px;'><strong>Delta Score (ΔS):</strong></div>"
        "<div style='font-size:11px; color:#333; margin-left:10px;'>"
        f"ΔS = {delta_equation(dict(options), options.get('metric', 'cohens_d'))} = "
        f"<strong>{delta.score:.3f}</strong></div>",
    )


def _activity_hover(subsystem_id: str, subsystem: str, row: TransportRow, score: float, measured: int, noun: str) -> str:
    """A row's hover in single-group activity: its name, its score and how many members it is of."""
    return hover_panel.panel(
        hover_panel.row(_row_label(subsystem_id, row), escape(row.name)),
        hover_panel.row("Subsystem", escape(subsystem)),
        hover_panel.row("Activity score", f"{score:.3f}"),
        hover_panel.row(f"Measured {noun}", measured),
        hover_panel.note(f"The median of its measured {noun}' activity scores."),
    )


def plot_transport_view(
    data_manager,
    subsystem_id: str,
    *,
    visualization_mode: str = "group_comparison",
    metric: str = "cohens_d",
    delta_options: Optional[Mapping[str, str]] = None,
    activity_artifact=None,
    overlay=None,
    for_genes: bool = False,
    download_flag: Optional[str] = None,
    hover_detail: Optional[str] = None,
):
    """A transport subsystem's families as rows, each with its score, in one grid.

    In a comparison, each row's dots are its reactions (or genes) on ``metric``
    and its score is a Delta score with ``delta_options``; in single-group
    activity, its dots are activity scores and its score their median. A row's
    score is coloured on its subsystems' scale, as a subsystem's with that score
    would be, and hovers as a subsystem node does; its elements hover at the
    graph card's ``hover_detail``.
    """
    from ..managers.workspace_mode import AnalysisWorkspaceMode
    from .metabolic_visualization.generation.graph_building.spaced_network import (
        category_numbers,
    )
    from .plot_subsystem_ranking import (
        model_category_numbers,
        plot_subsystem_activity_ranking,
        plot_subsystem_comparison,
    )
    from .ranking_layout import RankedSubsystem

    model = data_manager.metabolic_model
    name = str(model.get_subsystem_name(subsystem_id) or subsystem_id)
    number = model_category_numbers(data_manager).get(
        str(model.get_subsystem_category(subsystem_id) or ""), None
    )
    numbers = {name: number} if number is not None else category_numbers([name])

    if visualization_mode == "activity":
        from ..analysis.activity_analysis import ActivityOverlay, displayed_feature_id

        family = ActivityOverlay(overlay or ActivityOverlay.REACTION)
        scores = activity_artifact.feature_result_for(family).activity_score
        reaction_overlay = family is ActivityOverlay.REACTION
        rows = transport_rows(model, subsystem_id, measured=scores.dropna().index)
        groups: Dict[str, List[str]] = {}
        entries = []
        hovers: Dict[str, str] = {}
        noun = "reactions" if reaction_overlay else "genes"
        for row in rows:
            members = (
                [displayed_feature_id(scores, reaction) for reaction in row.reactions]
                if reaction_overlay
                else row.genes
            )
            measured = [m for m in dict.fromkeys(members) if m in scores.index and scores.loc[m] == scores.loc[m]]
            if not measured:
                continue
            groups[row.key] = row.reactions if reaction_overlay else row.genes
            score = float(statistics.median(float(scores.loc[m]) for m in measured))
            entries.append(RankedSubsystem(row.key, row.name, name, score))
            hovers[row.key] = _activity_hover(subsystem_id, name, row, score, len(measured), noun)
        if not entries:
            return None
        metric_key = activity_artifact.metric_key_for(family)
        color_manager = data_manager.color_manager
        return plot_subsystem_activity_ranking(
            data_manager,
            activity_artifact,
            overlay=family,
            return_plot=True,
            download_flag=download_flag,
            entries=entries,
            groups=groups,
            category_numbers=numbers,
            score_colors={
                entry.subsystem_id: color_manager.get_color_for_value(
                    AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS, "subsystem", metric_key, entry.score
                )
                for entry in entries
            },
            score_hovers=hovers,
            hover_detail=hover_detail,
        )

    artifact = data_manager.get_group_comparison_artifact()
    test_results = artifact.get_test_results(for_genes=for_genes)
    significant = artifact.get_significant_elements(for_genes=for_genes)
    direction_map = (
        data_manager.subsystem_analysis.get_reaction_direction_map()
        if getattr(data_manager, "subsystem_analysis", None) is not None
        else None
    )

    def columns(reactions: Sequence[str]) -> List[str]:
        if direction_map is None:
            return list(reactions)
        return [source for base in reactions if (source := direction_map.source_column_for(base)) is not None]

    rows = transport_rows(
        model,
        subsystem_id,
        measured=test_results.index if test_results is not None else None,
        significant=significant,
    )
    options = dict(delta_options or {"metric": metric})
    groups = {}
    entries = []
    hovers: Dict[str, str] = {}
    for row in rows:
        elements = row.genes if for_genes else columns(row.reactions)
        delta = group_delta(test_results, elements, significant, options)
        if delta is None:
            continue
        groups[row.key] = row.genes if for_genes else row.reactions
        entries.append(RankedSubsystem(row.key, row.name, name, float(delta.score)))
        hovers[row.key] = _delta_hover(subsystem_id, name, row, delta, options, "genes" if for_genes else "reactions")
    if not entries:
        return None
    # The subsystems' Delta scores, registered as the network colours them (the
    # subsystem's own colour asks for them), are the scale a family is put on.
    element_type = "subsystem_genes" if for_genes else "subsystem"
    data_manager.get_element_color(subsystem_id, element_type, options.get("metric", metric), delta_options=options)
    return plot_subsystem_comparison(
        data_manager,
        None,
        metric=metric,
        for_genes=for_genes,
        return_plot=True,
        download_flag=download_flag,
        entries=entries,
        groups=groups,
        category_numbers=numbers,
        score_colors={
            entry.subsystem_id: data_manager.color_manager.get_color_for_value(
                AnalysisWorkspaceMode.GROUP_COMPARISON, element_type, options.get("metric", metric), entry.score
            )
            for entry in entries
        },
        score_hovers=hovers,
        hover_detail=hover_detail,
    )


__all__ = [
    "EXCHANGE_SUBSYSTEM",
    "TRANSPORT_SUBSYSTEMS",
    "GroupDelta",
    "TransportRow",
    "group_delta",
    "group_delta_score",
    "is_transport_subsystem",
    "plot_transport_view",
    "transport_rows",
    "transporter_family",
]

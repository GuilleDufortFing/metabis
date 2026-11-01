"""What a click on a graph's element shows: its facts, its statistics and the elements linked to it.

A compartment map's details sidebar asks for one reaction, gene or metabolite at a
time. Its facts come from the metabolic model; its statistics from the same data
provider the hover panels read, in the workspace's mode (a comparison's two groups,
or a single group's activity); its violins and colour histogram are the hover
sections' own SVG (:mod:`.hover.sections`), so the sidebar and the hover cannot
disagree. Each element names the ones linked to it, with their statistics: a
reaction its genes, a gene its reactions, a metabolite the reactions that consume
or produce it, those on the map when ``within`` names its reactions.

Everything returned is plain data, numbers and strings, except the SVG figures.
"""

from __future__ import annotations

import math
import re
from typing import Any, Collection, Optional

from .hover.sections.activity_distribution import ActivityDistributionSection
from .hover.sections.color_explanation import ColorExplanationSection
from .hover.sections.distribution_plot import DistributionPlotSection
from .workspace_mode import AnalysisWorkspaceMode

KINDS = ("reaction", "gene", "metabolite")
METABOLIC_ATLAS = "https://metabolicatlas.org/explore/Human-GEM/gem-browser"
_ENSEMBL = re.compile(r"ENSG\d+")
# The statistics each linked row carries, in a comparison.
_ROW_STATS = ("cohens_d", "adjusted_pval")


def element_details(
    workspace,
    kind: str,
    element_id: str,
    *,
    color_metric: Optional[str] = None,
    within: Optional[Collection[str]] = None,
) -> Optional[dict[str, Any]]:
    """One element's details, or None when the model has no such element.

    ``element_id`` is a reaction's identifier (a ``_back`` suffix is dropped), a
    gene's Ensembl identifier or symbol, or a metabolite's identifier.
    ``color_metric`` is the metric the graph is coloured by; ``within``, the model
    reactions a map draws, to which linked reactions are limited.
    """
    if kind not in KINDS:
        raise ValueError(f"Unknown element kind {kind!r}")
    builder = _Details(workspace, color_metric, within)
    return getattr(builder, kind)(str(element_id))


class _Details:
    def __init__(self, workspace, color_metric, within):
        self.hovers = workspace.hover_manager
        self.provider = workspace.data_provider
        self.model = workspace.metabolic_model
        self.mode = self.hovers.mode
        self.color_metric = color_metric or "cohens_d"
        self.within = set(within) if within is not None else None

    # ------------------------------------------------------------------ elements

    def reaction(self, element_id: str) -> Optional[dict[str, Any]]:
        reaction_id = element_id.removesuffix("_back")
        data = self.provider.get_reaction_data(
            reaction_id, include_stats=True, include_influence=False, include_distributions=True
        )
        if not data.get("is_in_model"):
            return None
        metadata = data.get("metadata", {})
        main = self.hovers.main_direction(reaction_id)
        context = self._context("reaction", reaction_id, main)
        genes = list(dict.fromkeys(_ENSEMBL.findall(_text(metadata.get("gpr_ensmbl")))))
        facts = [
            ("Equation", _text(data.get("equation"))),
            ("Compartment", _text(metadata.get("compartment"))),
            ("Reversible", "Yes" if data.get("is_reversible") else "No"),
        ]
        if data.get("blocked_directions"):
            facts.append(("Blocked", ", ".join(data["blocked_directions"])))
        subsystem = _text(data.get("subsystem"))
        return {
            "kind": "reaction",
            "id": reaction_id,
            "label": reaction_id,
            "name": _text(data.get("name")),
            "facts": _facts(facts),
            "subsystem": subsystem if subsystem and subsystem != "N/A" else None,
            "category": _text(data.get("subsystem_category")) or None,
            "external_url": _text(data.get("external_url")) or f"{METABOLIC_ATLAS}/reaction/{reaction_id}",
            "statistics": self._statistics("reaction", data, context, main),
            "colour": self._colour(data, context),
            "linked": {"kind": "gene", "rows": [self._gene_row(gene) for gene in genes], "total": len(genes)},
        }

    def gene(self, element_id: str) -> Optional[dict[str, Any]]:
        ensembl = self._ensembl(element_id)
        model_data = self.provider.get_gene_metabolic_model_data(ensembl)
        if not model_data.get("is_in_model"):
            return None
        data = self._gene_data(ensembl, _text(model_data.get("symbol")), include_distributions=True)
        context = self._context("gene", data.get("gene_id", ensembl), None)
        reactions = self._on_map(self.model.get_reactions_for_gene(ensembl))
        return {
            "kind": "gene",
            "id": ensembl,
            "label": _text(model_data.get("symbol")) or ensembl,
            "name": _text(model_data.get("description")),
            "facts": _facts([("Ensembl", ensembl)]),
            "subsystem": None,
            "category": None,
            "external_url": _text(model_data.get("external_url")),
            "statistics": self._statistics("gene", data, context, None),
            "colour": self._colour(data, context),
            "linked": self._reaction_rows(reactions),
        }

    def metabolite(self, element_id: str) -> Optional[dict[str, Any]]:
        data = self.provider.get_metabolite_data(element_id)
        if not data.get("is_in_model"):
            return None
        roles = self.model.get_reactions_for_metabolite(element_id)
        role_of: dict[str, str] = {}
        for reaction in roles.get("as_reactant", []):
            role_of[reaction] = "consumes"
        for reaction in roles.get("as_product", []):
            role_of[reaction] = "consumes and produces" if role_of.get(reaction) == "consumes" else "produces"
        linked = self._reaction_rows(self._on_map(role_of))
        for row in linked["rows"]:
            row["role"] = role_of.get(row["id"], "")
        return {
            "kind": "metabolite",
            "id": element_id,
            "label": _text(data.get("name")) or element_id,
            "name": _text(data.get("compartment")),
            "facts": _facts([("Identifier", element_id), ("Formula", _text(data.get("formula"))), ("Charge", _text(data.get("charge")))]),
            "subsystem": None,
            "category": None,
            "external_url": f"{METABOLIC_ATLAS}/metabolite/{element_id}",
            "statistics": None,
            "colour": None,
            "linked": linked,
        }

    # ------------------------------------------------------------ their parts

    def _context(self, kind: str, element_id: str, main: Optional[str]):
        return self.hovers.context(
            element_type=kind,
            element_id=element_id,
            main_direction=main or "forward",
            show_stats=True,
            show_plot=True,
            show_color_reference=True,
            color_metric=self.color_metric,
        )

    def _statistics(self, kind: str, data: dict, context, main: Optional[str]) -> Optional[dict[str, Any]]:
        comparison = data.get("comparison_stats") or {}
        if self.mode == AnalysisWorkspaceMode.GROUP_COMPARISON and comparison.get("has_comparison_data"):
            directions = []
            for direction in ("forward", "backward"):
                stats = comparison.get(direction)
                if not stats:
                    continue
                directions.append(
                    {
                        "direction": direction if kind == "reaction" else None,
                        "main": kind != "reaction" or direction == (main or "forward"),
                        "cohens_d": _number(stats.get("cohens_d")),
                        "log2_fc": _number(stats.get("log2_fc")),
                        "mean_difference": _number(stats.get("mean_difference")),
                        "pval": _number(stats.get("pval")),
                        "adjusted_pval": _number(stats.get("adjusted_pval")),
                        "significant": bool(stats.get("is_significant")),
                        "test": stats.get("test"),
                    }
                )
            section = DistributionPlotSection(kind)
            violins = section.violins(data, context) if section.should_include(data, context) else []
            return {
                "kind": "comparison",
                "target": _text(comparison.get("group1")),
                "baseline": _text(comparison.get("group2")),
                "directions": directions,
                "violins": [{"direction": direction if kind == "reaction" else None, "svg": svg, "caption": ""} for direction, svg in violins],
            }
        experiment = data.get("experiment_stats") or {}
        if self.mode == AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS and experiment.get("has_experiment_data"):
            directions = []
            if kind == "reaction" and experiment.get("is_reversible"):
                for direction, key in (("forward", "forward_activity_score"), ("backward", "backward_activity_score")):
                    directions.append(
                        {"direction": direction, "main": direction == experiment.get("main_direction", "forward"), "activity_score": _number(experiment.get(key))}
                    )
            else:
                directions.append({"direction": None, "main": True, "activity_score": _number(experiment.get("activity_score"))})
            section = ActivityDistributionSection(kind)
            violins = []
            if section.should_include(data, context):
                svg, caption = section.violin(data, context)
                violins.append({"direction": None, "svg": svg, "caption": caption})
            return {
                "kind": "activity",
                "group": _text(experiment.get("group_id")),
                "summary": _text(experiment.get("summary_description")),
                "directions": directions,
                "violins": violins,
            }
        return None

    def _colour(self, data: dict, context) -> Optional[dict[str, Any]]:
        section = ColorExplanationSection()
        if not section.should_include(data, context):
            return None
        colouring = next((histogram for histogram in section.histograms(data, context) if histogram["colours"]), None)
        if colouring is None:
            return None
        return {"metric": colouring["metric"], "value": _number(colouring["value"]), "histogram": colouring["html"]}

    def _ensembl(self, gene: str) -> str:
        if self.model.get_gene_metadata(gene):
            return gene
        return self.model.translate_gene(gene, input_format="geneSymbols", output_format="ensmbl_id") or gene

    def _gene_data(self, ensembl: str, symbol: str, *, include_distributions: bool) -> dict:
        """The gene's data under whichever identifier the loaded expression uses."""
        first = None
        for gene in dict.fromkeys(name for name in (ensembl, symbol) if name):
            data = self.provider.get_gene_data(gene, include_stats=True, include_distributions=include_distributions)
            first = first or data
            if (data.get("comparison_stats") or {}).get("has_comparison_data") or (
                data.get("experiment_stats") or {}
            ).get("has_experiment_data"):
                return data
        return first or {}

    def _on_map(self, reactions) -> list[str]:
        reactions = list(dict.fromkeys(reactions))
        return reactions if self.within is None else [reaction for reaction in reactions if reaction in self.within]

    def _gene_row(self, ensembl: str) -> dict[str, Any]:
        model_data = self.provider.get_gene_metabolic_model_data(ensembl)
        symbol = _text(model_data.get("symbol")) if model_data.get("is_in_model") else ""
        data = self._gene_data(ensembl, symbol, include_distributions=False)
        row = {"id": ensembl, "label": symbol or ensembl, "name": _text(model_data.get("description")), "role": ""}
        row.update(self._row_statistics(data, None))
        return row

    def _reaction_rows(self, reactions: list[str]) -> dict[str, Any]:
        rows = []
        for reaction in reactions:
            data = self.provider.get_reaction_data(
                reaction, include_stats=True, include_influence=False, include_distributions=False
            )
            if not data.get("is_in_model"):
                continue
            row = {"id": reaction, "label": reaction, "name": _text(data.get("name")), "role": ""}
            row.update(self._row_statistics(data, self.hovers.main_direction(reaction)))
            rows.append(row)
        # Significant first, then the most significant.
        rows.sort(key=lambda row: (not row["significant"], row["adjusted_pval"] if row["adjusted_pval"] is not None else 2.0, row["id"]))
        return {"kind": "reaction", "rows": rows, "total": len(rows)}

    def _row_statistics(self, data: dict, main: Optional[str]) -> dict[str, Any]:
        """A linked element's statistics in one row: its main direction's, or its most significant one's."""
        row = {"cohens_d": None, "adjusted_pval": None, "significant": False, "activity_score": None, "scored": False}
        comparison = data.get("comparison_stats") or {}
        if self.mode == AnalysisWorkspaceMode.GROUP_COMPARISON and comparison.get("has_comparison_data"):
            directions = {direction: comparison.get(direction) for direction in ("forward", "backward") if comparison.get(direction)}
            if not directions:
                return row
            stats = directions.get(main or "") or min(directions.values(), key=lambda stats: stats.get("adjusted_pval", 1.0))
            row.update({key: _number(stats.get(key)) for key in _ROW_STATS})
            row["significant"] = bool(stats.get("is_significant"))
            row["scored"] = True
            return row
        experiment = data.get("experiment_stats") or {}
        if self.mode == AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS and experiment.get("has_experiment_data"):
            row["activity_score"] = _number(experiment.get("activity_score"))
            row["scored"] = row["activity_score"] is not None
        return row


def _number(value) -> Optional[float]:
    """A statistic as JSON can carry it: a finite float, or None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value) -> str:
    """A model's text field as a string: empty for a missing one, which pandas reads as NaN."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    return str(value).strip()


def _facts(pairs) -> list[dict[str, str]]:
    return [{"label": label, "value": value} for label, value in pairs if value]

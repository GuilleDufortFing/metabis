"""What a click on a graph's element shows: its facts, its statistics and the elements linked to it.

A graph's details sidebar asks for one reaction, gene or metabolite at a time. Its
facts come from the metabolic model; its statistics from the same data provider
the hover panels read, in the workspace's mode (a comparison's two groups, or a
single group's activity). A comparison's violins come as the groups the hover's
violins draw (:mod:`.hover.sections`), their values, quartiles and density, for
the page to draw at its own size; a single group's, as its feature against all of
its kind in the group. Each statistic comes with where it sits among every
element's of its kind, the distribution the graph's colours are registered from:
its bins, its percentile, its thresholds, and, for the metric the graph is
coloured by, the colour of each bin. The colour the graph paints the element in
comes with it, as does each linked element's. Each element names the ones linked to it, with their
statistics: a reaction its genes, a gene its reactions, a metabolite the reactions
that consume or produce it, those the graph draws when ``within`` names its
reactions. A graph drawn from Metabolic Atlas's maps may set beside a reaction
genes its gene rule leaves out; given what it draws (``drawn_with``), those are
linked too, marked as the map's.

Everything returned is plain data, numbers and strings.
"""

from __future__ import annotations

import math
import re
from typing import Any, Collection, Mapping, Optional

import numpy as np

from ..analysis.activity_analysis import displayed_feature_id
from ..analysis.reaction_direction import (
    DEFAULT_REACTION_DIRECTION_CRITERION,
    ReactionDirectionCriterion,
)
from ..visual_config import theme
from .element_presentation import comparison_direction_criterion
from .hover.sections.distribution_plot import DistributionPlotSection
from .workspace_mode import AnalysisWorkspaceMode

KINDS = ("reaction", "gene", "metabolite")
METABOLIC_ATLAS = "https://metabolicatlas.org/explore/Human-GEM/gem-browser"
_ENSEMBL = re.compile(r"ENSG\d+")
# The statistics each linked row carries, in a comparison.
_ROW_STATS = ("cohens_d", "adjusted_pval")
# Points along a violin's outline, between its group's least and greatest values.
_DENSITY_POINTS = 48
# A violin of more values than this is drawn without its points.
_VIOLIN_POINTS = 400
# The statistics an element of a comparison is set among all of its kind's, in
# the order the details show them: its metric, its name, and the registered
# distribution its axis reads, the adjusted p on -log10's as a volcano plot's y
# axis is, and named as the hover's colour reference names it
# (plot_histograms.create_histogram_background). A graph coloured by the p-value
# before correction, or by the mean difference, shows that one instead of, or
# after, these.
_SPREADS = (
    ("adjusted_pval", "-Log(adj. p-value)", "neg_log_adjusted_pval"),
    ("cohens_d", "Cohen's d", "cohens_d"),
    ("log2_fc", "Log2 fold change", "log2_fc"),
)
_RAW_P = ("pval", "-Log(p-value)", "neg_log_pval")
_MEAN_DIFFERENCE = ("mean_difference", "Mean difference", "mean_difference")
# The bins a spread is counted in, between the values' 0.5th and 99.5th
# percentiles (the -log10 p's from 0), the rest counted in the end bins.
_SPREAD_BINS = 30
_SPREAD_TAIL = 0.005
# Painted as the hover's colour reference paints them, a bin short of a
# threshold is grey (plot_histograms.create_color_reference_histogram).
_SHORT_OF_THRESHOLD = theme.NOT_SIGNIFICANT
# The model's formula of a reaction: its two sides, and a side's terms.
_FORMULA_ARROW = re.compile(r"\s*(?:<=>|<->|=>|->)\s*")
_FORMULA_TERM = re.compile(r"(?:(\d+(?:\.\d+)?)\s+)?(\S+)")


def element_details(
    workspace,
    kind: str,
    element_id: str,
    *,
    color_metric: Optional[str] = None,
    within: Optional[Collection[str]] = None,
    dual_evidence: bool = False,
    drawn_with: Optional[Mapping[str, Collection[str]]] = None,
) -> Optional[dict[str, Any]]:
    """One element's details, or None when the model has no such element.

    ``element_id`` is a reaction's identifier (a ``_back`` suffix is dropped), a
    gene's Ensembl identifier or symbol, or a metabolite's identifier.
    ``color_metric`` is the metric the graph is coloured by; ``within``, the model
    reactions a map draws, to which linked reactions are limited. With
    ``dual_evidence``, a single group's activity is read as a subsystem's graph
    draws it, its genes by the genes' scores and its reactions by the reactions'
    (the graph pipeline's ``activity_dual_evidence``), not by the overlay alone.
    ``drawn_with`` gives, for each reaction a graph draws, the genes it draws
    beside it; those its gene rule leaves out are linked after its own, and a
    gene drawn beside reactions it does not catalyse in the model lists them,
    each row marked ``map_only``.
    """
    if kind not in KINDS:
        raise ValueError(f"Unknown element kind {kind!r}")
    builder = _Details(workspace, color_metric, within, drawn_with)
    provider = workspace.data_provider
    before = getattr(provider, "activity_dual_evidence", False)
    provider.activity_dual_evidence = dual_evidence
    try:
        return getattr(builder, kind)(str(element_id))
    finally:
        provider.activity_dual_evidence = before


class _Details:
    def __init__(self, workspace, color_metric, within, drawn_with=None):
        self.workspace = workspace
        self.drawn_with = {str(reaction): list(dict.fromkeys(map(str, genes))) for reaction, genes in (drawn_with or {}).items()}
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
        drawn = [gene for gene in self.drawn_with.get(reaction_id, []) if gene not in genes]
        rows = [self._gene_row(gene) for gene in genes] + [{**self._gene_row(gene), "map_only": True} for gene in drawn]
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
            "equation": self._equation(reaction_id),
            "subsystem": subsystem if subsystem and subsystem != "N/A" else None,
            "category": _text(data.get("subsystem_category")) or None,
            "external_url": _text(data.get("external_url")) or f"{METABOLIC_ATLAS}/reaction/{reaction_id}",
            "statistics": self._statistics("reaction", data, context, main, reaction_id),
            "colour": self._colour("reaction", reaction_id, main),
            "linked": {"kind": "gene", "rows": rows, "total": len(rows)},
        }

    def gene(self, element_id: str) -> Optional[dict[str, Any]]:
        ensembl = self._ensembl(element_id)
        model_data = self.provider.get_gene_metabolic_model_data(ensembl)
        if not model_data.get("is_in_model"):
            return None
        data = self._gene_data(ensembl, _text(model_data.get("symbol")), include_distributions=True)
        context = self._context("gene", data.get("gene_id", ensembl), None)
        every = list(dict.fromkeys(self.model.get_reactions_for_gene(ensembl)))
        # Reactions a map draws it beside, which its gene rules do not give it.
        beside = [reaction for reaction, genes in self.drawn_with.items() if ensembl in genes and reaction not in every]
        linked = self._reaction_rows(self._on_map(every))
        extra = self._reaction_rows(self._on_map(beside))["rows"]
        linked["rows"] += [{**row, "map_only": True} for row in extra]
        linked["total"] = len(linked["rows"])
        return {
            "kind": "gene",
            "id": ensembl,
            "label": _text(model_data.get("symbol")) or ensembl,
            "name": _text(model_data.get("description")),
            "facts": _facts([("Ensembl", ensembl)]),
            "subsystem": None,
            "category": None,
            "subsystems": self._subsystems(every),
            "external_url": _text(model_data.get("external_url")),
            # Its Metabolic Atlas page too, as a reaction's and a metabolite's are.
            "atlas_url": f"{METABOLIC_ATLAS}/gene/{ensembl}",
            "statistics": self._statistics("gene", data, context, None, data.get("gene_id", ensembl)),
            "colour": self._colour("gene", data.get("gene_id", ensembl), None),
            "linked": linked,
        }

    def metabolite(self, element_id: str) -> Optional[dict[str, Any]]:
        data = self.provider.get_metabolite_data(element_id)
        if not data.get("is_in_model"):
            return None
        # A map names some metabolites in another format: the model's own identifier.
        element_id = _text((data.get("metadata") or {}).get("primary_id")) or element_id
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

    def _equation(self, reaction_id: str) -> Optional[dict[str, Any]]:
        """A reaction's metabolites, each side in the model's order, with their names and coefficients.

        The coefficients are read from the model's formula ("MAM01249c + 2 MAM02039c
        -> ..."); without one, each metabolite counts once.
        """
        reaction = self.model.get_reaction(reaction_id)
        if reaction is None:
            return None
        sides = _FORMULA_ARROW.split(_text(reaction.formula))
        if len(sides) == 2:
            terms = [[_FORMULA_TERM.fullmatch(term.strip()) for term in side.split(" + ") if term.strip()] for side in sides]
            parsed = [[(float(term.group(1) or 1), term.group(2)) for term in side if term] for side in terms]
        else:
            parsed = [[(1.0, metabolite) for metabolite in reaction.reactants or []], [(1.0, metabolite) for metabolite in reaction.products or []]]

        def side(metabolites) -> list[dict[str, Any]]:
            parts = []
            for coefficient, metabolite in metabolites:
                entity = self.model.get_metabolite(metabolite)
                parts.append(
                    {
                        "id": str(metabolite),
                        "name": _text(getattr(entity, "name", "")) or str(metabolite),
                        "compartment": str(metabolite)[-1:] if str(metabolite)[-1:].isalpha() else "",
                        "coefficient": coefficient,
                    }
                )
            return parts

        return {"reactants": side(parsed[0]), "products": side(parsed[1]), "reversible": bool(reaction.is_reversible)}

    def _subsystems(self, reactions: list[str]) -> list[dict[str, str]]:
        """The subsystems of ``reactions``, once each, with their categories, in the reactions' order."""
        owners = {}
        for reaction in reactions:
            entity = self.model.get_reaction(reaction)
            subsystem = _text(getattr(entity, "subsystem_id", "")) if entity is not None else ""
            if subsystem and subsystem not in owners:
                owners[subsystem] = {
                    "name": _text(self.model.get_subsystem_name(subsystem)) or subsystem,
                    "category": _text(self.model.get_subsystem_category(subsystem)),
                }
        return list(owners.values())

    def _paint(self, element_id: str, kind: str) -> str:
        """The colour the graph paints an element in, by the metric it is coloured by."""
        try:
            return _text(self.workspace.get_element_color(element_id, kind, self.color_metric))
        except Exception:
            return ""

    def _statistics(self, kind: str, data: dict, context, main: Optional[str], key: str) -> Optional[dict[str, Any]]:
        comparison = data.get("comparison_stats") or {}
        if self.mode == AnalysisWorkspaceMode.GROUP_COMPARISON and comparison.get("has_comparison_data"):
            directions = []
            for direction in ("forward", "backward"):
                stats = comparison.get(direction)
                if not stats:
                    continue
                shown = {
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
                shown["spreads"] = self._comparison_spreads(kind, _direction_key(key, direction), shown)
                directions.append(shown)
            section = DistributionPlotSection(kind)
            drawn = section.groups(data, context) if section.should_include(data, context) else []
            target = _text(comparison.get("group1"))
            baseline = _text(comparison.get("group2"))
            # A reversible reaction is read in the direction more active in the
            # population its comparison's criterion names (reaction_direction).
            criterion = comparison_direction_criterion(getattr(self.provider, "subsystem_analysis", None))
            criterion = criterion or DEFAULT_REACTION_DIRECTION_CRITERION
            in_target = criterion == ReactionDirectionCriterion.DOMINANT_IN_TARGET
            more_active_in = target if in_target else baseline
            axis = section.shared_axis(drawn)
            return {
                "kind": "comparison",
                "target": target,
                "baseline": baseline,
                "more_active_in": more_active_in,
                "more_active_side": "Target" if in_target else "Baseline",
                "more_active_color": self._group_colour(more_active_in),
                "directions": directions,
                # Two directions' violins on one axis, as the hover's are.
                "axis": [float(end) for end in axis] if axis else None,
                "distributions": [
                    {
                        "direction": direction["direction"] if kind == "reaction" else None,
                        "significant": direction["significant"],
                        "groups": [_violin(group) for group in direction["groups"]],
                    }
                    for direction in drawn
                ],
            }
        experiment = data.get("experiment_stats") or {}
        if self.mode == AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS and experiment.get("has_experiment_data"):
            directions = []
            if kind == "reaction" and experiment.get("is_reversible"):
                for direction, field in (("forward", "forward_activity_score"), ("backward", "backward_activity_score")):
                    directions.append(
                        {
                            "direction": direction,
                            "main": direction == experiment.get("main_direction", "forward"),
                            "activity_score": _number(experiment.get(field)),
                            "spreads": self._activity_spreads(kind, _direction_key(key, direction)),
                        }
                    )
            else:
                directions.append(
                    {
                        "direction": None,
                        "main": True,
                        "activity_score": _number(experiment.get("activity_score")),
                        "spreads": self._activity_spreads(kind, key),
                    }
                )
            return {
                "kind": "activity",
                "group": _text(experiment.get("group_id")),
                "more_active_in": _text(experiment.get("group_id")),
                "summary": _text(experiment.get("summary_description")),
                "directions": directions,
                "distributions": self._activity_violins(kind, data),
            }
        return None

    def _group_colour(self, group: str) -> str:
        """A group's colour, as the hover names the group a direction is more active in."""
        try:
            return _text(self.workspace.color_manager.get_group_color(group))
        except Exception:
            return ""

    def _comparison_spreads(self, kind: str, key: str, stats: dict) -> list[dict[str, Any]]:
        """Where a direction's statistics sit among every element's of its kind in the comparison."""
        metrics = list(_SPREADS)
        if self.color_metric == _RAW_P[2]:
            metrics[0] = _RAW_P
        elif self.color_metric == _MEAN_DIFFERENCE[0]:
            metrics.append(_MEAN_DIFFERENCE)
        artifact = getattr(self.provider, "group_comparison_artifact", None)
        thresholds = artifact.get_thresholds(for_genes=kind == "gene") if artifact is not None else {}
        spreads = []
        for metric, label, axis in metrics:
            spread = self._spread(
                AnalysisWorkspaceMode.GROUP_COMPARISON,
                kind,
                key,
                axis,
                metric=metric,
                label=label,
                value=stats.get(metric),
                colours=axis == self.color_metric or metric == self.color_metric,
                # The p-value's bins wear the hover's colours whatever colours the graph.
                painted=axis.startswith("neg_log"),
                threshold=thresholds.get(axis),
            )
            if spread is not None:
                spreads.append(spread)
        return spreads

    def _activity_spreads(self, kind: str, key: str) -> list[dict[str, Any]]:
        """Where a feature's activity score sits among every one of its kind in the group."""
        artifact = getattr(self.provider, "activity_artifact", None)
        if artifact is None:
            return []
        overlay = self.provider.activity_overlay_for_element(kind)
        if not artifact.colors_element_type(overlay, kind):
            return []
        spread = self._spread(
            AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS,
            kind,
            key,
            artifact.metric_key_for(overlay),
            metric="activity_score",
            label="Activity score",
            value=None,
            colours=True,
            threshold=None,
        )
        return [spread] if spread is not None else []

    def _spread(self, mode, kind: str, key: str, axis: str, *, metric, label, value, colours, threshold, painted=False) -> Optional[dict[str, Any]]:
        """One statistic among every element's of its kind: the bins of the registered distribution ``axis``.

        ``at`` is the element's place on the axis, ``percentile`` the share of
        elements below it (ties counted half), and ``thresholds`` the axis's
        significance lines, either side of 0 for an effect. A metric that colours
        the graph, or one ``painted``, colours each bin as the hover's colour
        reference does: as the graph's key colours its middle, grey short of the
        threshold.
        """
        manager = self.workspace.color_manager
        series = manager.get_distribution_values(mode, kind, axis)
        if series is None:
            return None
        values = np.asarray(series, dtype=float)
        values = values[np.isfinite(values)]
        at = _number(series.get(key)) if key in series.index else None
        if not values.size or at is None:
            return None
        log = axis.startswith("neg_log")
        low = 0.0 if log else float(np.quantile(values, _SPREAD_TAIL))
        high = float(np.quantile(values, 1 - _SPREAD_TAIL))
        lines = [] if threshold is None or not math.isfinite(threshold) else ([threshold] if log else [-abs(threshold), abs(threshold)])
        low = min(low, at, *lines)
        high = max(high, at, *lines)
        if high <= low:
            high = low + 1.0
        edges = np.linspace(low, high, _SPREAD_BINS + 1)
        counts, _ = np.histogram(np.clip(values, low, high), edges)
        bins = []
        for start, end, count in zip(edges[:-1], edges[1:], counts):
            middle = (start + end) / 2
            colour = ""
            if colours or painted:
                short = lines and abs(middle) < abs(lines[-1])
                colour = _SHORT_OF_THRESHOLD if short else manager.get_color_for_value(mode, kind, axis, middle)
            bins.append({"start": float(start), "end": float(end), "count": int(count), "color": colour})
        below = float(np.sum(values < at)) + 0.5 * float(np.sum(values == at))
        return {
            "metric": metric,
            "label": label,
            "value": _number(value) if value is not None else at,
            "at": at,
            "percentile": round(100.0 * below / values.size, 1),
            "count": int(values.size),
            "log": log,
            "colours": bool(colours),
            "thresholds": [float(line) for line in lines],
            "bins": bins,
        }

    def _activity_violins(self, kind: str, data: dict) -> list[dict[str, Any]]:
        """A feature's activity across its group's samples, beside every one of its kind's in the group."""
        distributions = data.get("distributions") or {}
        if not (distributions.get("has_distributions") and distributions.get("is_activity")):
            return []
        noun = "reactions" if kind == "reaction" else "genes"
        everything = _violin(
            {"name": distributions.get("reference_name") or f"All {noun}", "color": theme.NOT_SIGNIFICANT, "values": distributions["reference_values"]},
            unit=noun,
        )
        # Its violin is drawn from a summary of their values; it stands for all of them.
        everything["n"] = int(distributions.get("reference_count") or everything["n"])
        feature = distributions.get("feature_id") or ""
        this = _violin(
            {"name": f"This {kind}", "color": self._paint(feature, kind) or theme.ACTIVITY_MID, "values": distributions["element_values"]}
        )
        return [
            {
                "direction": None,
                "significant": True,
                "groups": [everything, this],
                "caption": (
                    f"Left, all {everything['n']:,} {noun} measured in "
                    f"{distributions.get('group_id', '')}, each by its group summary; right, this {kind} "
                    f"across the group's {this['n']:,} samples."
                ),
            }
        ]

    def _colour(self, kind: str, key: str, main: Optional[str]) -> Optional[dict[str, Any]]:
        """The metric the graph is coloured by, the element's value of it, and the colour it takes."""
        manager = self.workspace.color_manager
        if self.mode == AnalysisWorkspaceMode.GROUP_COMPARISON:
            metric = self.color_metric
            key = _direction_key(key, main or "forward")
        elif self.mode == AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS:
            artifact = getattr(self.provider, "activity_artifact", None)
            if artifact is None:
                return None
            overlay = self.provider.activity_overlay_for_element(kind)
            if not artifact.colors_element_type(overlay, kind):
                return None
            metric = artifact.metric_key_for(overlay)
            if kind == "reaction":
                key = displayed_feature_id(artifact.feature_result_for(overlay).activity_score, key)
        else:
            return None
        series = manager.get_distribution_values(self.mode, kind, metric)
        if series is None or key not in series.index:
            return None
        return {
            "metric": "activity_score" if self.mode == AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS else metric,
            "value": _number(series.get(key)),
            # The colour its value takes on the graph's key: a map may also draw it
            # beside a reaction without data, as no data.
            "color": self._paint(key, kind),
        }

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
        row = {"id": ensembl, "label": symbol or ensembl, "name": _text(model_data.get("description")), "role": "", "map_only": False}
        row.update(self._row_statistics(data, None))
        row["color"] = self._paint(data.get("gene_id", ensembl), "gene") if row["scored"] else ""
        return row

    def _reaction_rows(self, reactions: list[str]) -> dict[str, Any]:
        rows = []
        for reaction in reactions:
            data = self.provider.get_reaction_data(
                reaction, include_stats=True, include_influence=False, include_distributions=False
            )
            if not data.get("is_in_model"):
                continue
            row = {"id": reaction, "label": reaction, "name": _text(data.get("name")), "role": "", "map_only": False}
            row.update(self._row_statistics(data, self.hovers.main_direction(reaction)))
            row["color"] = self._paint(reaction, "reaction") if row["scored"] else ""
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


def _violin(group: dict, unit: str = "samples") -> dict[str, Any]:
    """One group's violin as the page draws it: its values, quartiles and density.

    The density is a Gaussian kernel estimate, as the hover's violin's, between
    the group's least and greatest values (trimmed, as ggplot2's violins are), and
    scaled to 1 at its widest; none for fewer than two distinct values. ``unit``
    names what its values are of; past a few hundred, they are not sent, and the
    violin is drawn without its points.
    """
    values = np.asarray([value for value in group["values"] if _number(value) is not None], dtype=float)
    shape: dict[str, Any] = {"name": _text(group["name"]), "color": group["color"], "n": int(values.size), "unit": unit}
    if not values.size:
        return {**shape, "values": [], "density": []}
    low, q1, median, q3, high = np.percentile(values, [0, 25, 50, 75, 100])
    shape.update(
        values=[float(f"{value:.5g}") for value in values] if values.size <= _VIOLIN_POINTS else [],
        mean=float(values.mean()),
        median=float(median),
        q1=float(q1),
        q3=float(q3),
        min=float(low),
        max=float(high),
        density=[],
    )
    if high > low and np.unique(values).size > 1:
        from scipy.stats import gaussian_kde

        try:
            along = np.linspace(low, high, _DENSITY_POINTS)
            density = gaussian_kde(values)(along)
            widest = float(density.max()) or 1.0
            shape["density"] = [[float(f"{y:.5g}"), round(float(d) / widest, 4)] for y, d in zip(along, density)]
        except (np.linalg.LinAlgError, ValueError):
            pass
    return shape


def _direction_key(key: str, direction: Optional[str]) -> str:
    """The key a reaction's direction is registered under: its backward direction with ``_back``."""
    base = key.removesuffix("_back")
    return f"{base}_back" if direction == "backward" else base


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

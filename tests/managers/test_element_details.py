"""A clicked element's details: its facts, its statistics, and the elements linked to it."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from metabis import element_details
from metabis.managers.element_details import _violin
from metabis.managers.workspace_mode import AnalysisWorkspaceMode

GENES = {
    "ENSG1": {"symbol": "HADH", "description": "hydroxyacyl-CoA dehydrogenase"},
    "ENSG2": {"symbol": "ACADM", "description": "acyl-CoA dehydrogenase medium chain"},
}
REACTIONS = {
    "MAR1": {"name": "beta oxidation C12", "gpr_ensmbl": "ENSG2 and ENSG1", "subsystem": "Fatty acid oxidation"},
    "MAR2": {"name": "beta oxidation C10", "gpr_ensmbl": "ENSG1", "subsystem": "Fatty acid oxidation"},
    "MAR3": {"name": "elsewhere", "gpr_ensmbl": "ENSG1", "subsystem": "Other"},
    # No gene catalyses it: the model reads its empty gene rule as NaN.
    "MAR4": {"name": float("nan"), "gpr_ensmbl": float("nan"), "subsystem": "Other"},
}


def _stats(d, q, significant):
    return {"cohens_d": d, "log2_fc": d / 10, "mean_difference": d / 20, "pval": q / 10, "adjusted_pval": q, "is_significant": significant, "test": "wilcoxon"}


# Per reaction direction, and per gene, as a comparison of Inflamed against Healthy reads them.
COMPARISON = {
    "MAR1": {"forward": _stats(-1.47, 1e-6, True)},
    "MAR2": {"forward": _stats(-0.2, 0.4, False), "backward": _stats(0.9, 1e-3, True)},
    "MAR3": {"forward": _stats(-1.1, 1e-4, True)},
    "ENSG1": {"forward": _stats(-0.87, 3.6e-5, True)},
}


class _Provider:
    def get_reaction_data(self, reaction_id, **_):
        if reaction_id not in REACTIONS:
            return {"is_in_model": False}
        reaction = REACTIONS[reaction_id]
        stats = COMPARISON.get(reaction_id)
        return {
            "is_in_model": True,
            "reaction_id": reaction_id,
            "name": reaction["name"],
            "equation": "a[m] => b[m]",
            "subsystem": reaction["subsystem"],
            "subsystem_category": "Lipid",
            "is_reversible": "backward" in (stats or {}),
            "external_url": "",
            "metadata": {"gpr_ensmbl": reaction["gpr_ensmbl"], "compartment": "Mitochondria"},
            "comparison_stats": {"has_comparison_data": True, "group1": "Inflamed", "group2": "Healthy", **stats}
            if stats
            else {"has_comparison_data": False},
        }

    def get_gene_metabolic_model_data(self, gene_id):
        gene = GENES.get(gene_id)
        if gene is None:
            return {"is_in_model": False}
        return {"is_in_model": True, "gene_id": gene_id, "external_url": "", **gene}

    def get_gene_data(self, gene_id, **_):
        data = {"gene_id": gene_id, **self.get_gene_metabolic_model_data(gene_id)}
        stats = COMPARISON.get(gene_id)
        data["comparison_stats"] = (
            {"has_comparison_data": True, "group1": "Inflamed", "group2": "Healthy", **stats} if stats else {"has_comparison_data": False}
        )
        return data

    def get_metabolite_data(self, metabolite_id):
        if metabolite_id != "MAM1m":
            return {"is_in_model": False}
        return {"is_in_model": True, "metabolite_id": "MAM1m", "name": "H2O", "formula": "H2O", "charge": 0, "compartment": "Mitochondria"}


class _Model:
    def get_gene_metadata(self, gene_id):
        return {"ensmble_id": gene_id} if gene_id in GENES else {}

    def translate_gene(self, gene, input_format, output_format):
        by_symbol = {gene_data["symbol"]: ensembl for ensembl, gene_data in GENES.items()}
        return by_symbol.get(gene, gene)

    def get_reactions_for_gene(self, gene_id):
        return [reaction for reaction, data in REACTIONS.items() if gene_id in str(data["gpr_ensmbl"])]

    def get_reactions_for_metabolite(self, metabolite_id):
        return {"as_reactant": ["MAR1", "MAR2"], "as_product": ["MAR2", "MAR3"]}

    def get_reaction(self, reaction_id):
        if reaction_id not in REACTIONS:
            return None
        # The model's formula gives the coefficients its equation's names leave out.
        return SimpleNamespace(
            formula="MAM1m + 2 MAM2m -> MAM3m",
            reactants=["MAM1m", "MAM2m"],
            products=["MAM3m"],
            is_reversible=reaction_id == "MAR2",
            subsystem_id=REACTIONS[reaction_id]["subsystem"],
        )

    def get_metabolite(self, metabolite_id):
        return SimpleNamespace(name={"MAM1m": "H2O", "MAM2m": "NAD+", "MAM3m": "NADH"}[metabolite_id])

    def get_subsystem_name(self, subsystem):
        return subsystem

    def get_subsystem_category(self, subsystem):
        return "Lipid" if subsystem == "Fatty acid oxidation" else "Other category"


class _Hovers:
    mode = AnalysisWorkspaceMode.GROUP_COMPARISON

    def main_direction(self, reaction_id):
        # The comparison draws MAR2 by its backward direction.
        return "backward" if reaction_id == "MAR2" else "forward"

    def context(self, **options):
        return SimpleNamespace(
            mode=self.mode, color_manager=None, data_provider=None, show_plot=False, show_color_reference=False,
            hover_type=options["element_type"], **{
                key: value for key, value in options.items() if key not in ("show_plot", "show_color_reference")
            }
        )


# Each metric's value for every reaction direction a comparison scored: the four
# above, and 96 others spread evenly, so MAR1's Cohen's d of -1.47 is the lowest.
_OTHERS = {f"R{i}": value for i, value in enumerate(np.linspace(-1.0, 1.0, 96))}


def _registered(metric: str) -> pd.Series:
    named = {
        "MAR1": COMPARISON["MAR1"]["forward"],
        "MAR2": COMPARISON["MAR2"]["forward"],
        "MAR2_back": COMPARISON["MAR2"]["backward"],
        "MAR3": COMPARISON["MAR3"]["forward"],
    }
    if metric.startswith("neg_log_"):
        source = "adjusted_pval" if metric == "neg_log_adjusted_pval" else "pval"
        values = {key: -np.log10(stats[source]) for key, stats in named.items()}
        values.update({key: abs(value) * 3 for key, value in _OTHERS.items()})
    else:
        values = {key: stats[metric] for key, stats in named.items()}
        values.update({key: value / (10 if metric == "log2_fc" else 1) for key, value in _OTHERS.items()})
    return pd.Series(values)


class _Colours:
    """The distributions a comparison registers for its reactions, and its key's colours."""

    def get_distribution_values(self, mode, kind, metric):
        if mode != AnalysisWorkspaceMode.GROUP_COMPARISON or kind != "reaction":
            return None
        return _registered(metric)

    def get_color_for_value(self, mode, kind, metric, value):
        return "#ff0000" if value > 0 else "#0000ff"

    def get_group_color(self, group):
        return {"Healthy": "#e08214", "Inflamed": "#4393c3"}[group]


def _workspace(mode=AnalysisWorkspaceMode.GROUP_COMPARISON):
    hovers = _Hovers()
    hovers.mode = mode
    # The colours the graph paints its elements in, by their values on its key.
    colours = {"MAR1": "#8080ff", "MAR2": "#ffbbbb", "MAR3": "#9999ff", "ENSG1": "#bbbbff"}
    return SimpleNamespace(
        hover_manager=hovers,
        data_provider=_Provider(),
        metabolic_model=_Model(),
        color_manager=_Colours(),
        get_element_color=lambda element, kind, metric: colours.get(element, "#c7c7c7"),
    )


def test_a_reaction_has_its_facts_its_statistics_and_its_genes_with_theirs():
    details = element_details(_workspace(), "reaction", "MAR1_back")

    assert (details["id"], details["name"], details["subsystem"]) == ("MAR1", "beta oxidation C12", "Fatty acid oxidation")
    assert {"label": "Equation", "value": "a[m] => b[m]"} in details["facts"]
    statistics = details["statistics"]
    assert (statistics["kind"], statistics["target"], statistics["baseline"]) == ("comparison", "Inflamed", "Healthy")
    # A reversible reaction is read in the direction more active in the Baseline, by default.
    assert statistics["more_active_in"] == "Healthy"
    # Named as the hover names it: the Baseline's, in its group's colour.
    assert (statistics["more_active_side"], statistics["more_active_color"]) == ("Baseline", "#e08214")
    # One direction's violins have their own axis.
    assert statistics["axis"] is None
    (forward,) = statistics["directions"]
    assert {key: value for key, value in forward.items() if key != "spreads"} == {
        "direction": "forward", "main": True, "cohens_d": -1.47, "log2_fc": -0.147, "mean_difference": -0.0735,
        "pval": 1e-7, "adjusted_pval": 1e-6, "significant": True, "test": "wilcoxon",
    }
    assert details["colour"] == {"metric": "cohens_d", "value": -1.47, "color": "#8080ff"}
    # Its genes in the order of its gene rule, each with its own comparison.
    genes = details["linked"]
    assert (genes["kind"], [row["label"] for row in genes["rows"]]) == ("gene", ["ACADM", "HADH"])
    assert genes["rows"][1] == {
        "id": "ENSG1", "label": "HADH", "name": "hydroxyacyl-CoA dehydrogenase", "role": "",
        "cohens_d": -0.87, "adjusted_pval": 3.6e-5, "significant": True, "activity_score": None, "scored": True,
        "color": "#bbbbff", "map_only": False,
    }
    # A gene without data takes no colour: the page draws it white and dashed, as the map does.
    assert (genes["rows"][0]["scored"], genes["rows"][0]["color"]) == (False, "")
    # Its equation's metabolites, each a link, with the formula's coefficients.
    assert details["equation"] == {
        "reactants": [
            {"id": "MAM1m", "name": "H2O", "compartment": "m", "coefficient": 1.0},
            {"id": "MAM2m", "name": "NAD+", "compartment": "m", "coefficient": 2.0},
        ],
        "products": [{"id": "MAM3m", "name": "NADH", "compartment": "m", "coefficient": 1.0}],
        "reversible": False,
    }


def test_a_gene_found_by_its_symbol_lists_its_reactions_on_the_map_most_significant_first():
    details = element_details(_workspace(), "gene", "HADH", within={"MAR1", "MAR2"})

    assert (details["id"], details["label"], details["name"]) == ("ENSG1", "HADH", "hydroxyacyl-CoA dehydrogenase")
    # Its Metabolic Atlas page, by its Ensembl identifier, as Metabolic Atlas names its genes.
    assert details["atlas_url"] == "https://metabolicatlas.org/explore/Human-GEM/gem-browser/gene/ENSG1"
    rows = details["linked"]["rows"]
    # MAR3 is not on the map. MAR2 reads its backward direction, as the map draws it.
    assert [(row["id"], row["cohens_d"], row["color"]) for row in rows] == [("MAR1", -1.47, "#8080ff"), ("MAR2", 0.9, "#ffbbbb")]
    assert details["linked"]["total"] == 2
    # Its reactions' subsystems, those off the map too, each once.
    assert details["subsystems"] == [
        {"name": "Fatty acid oxidation", "category": "Lipid"},
        {"name": "Other", "category": "Other category"},
    ]


def test_a_metabolite_names_what_each_of_its_reactions_does_with_it():
    details = element_details(_workspace(), "metabolite", "MAM1m")

    assert (details["label"], details["name"], details["statistics"]) == ("H2O", "Mitochondria", None)
    assert {row["id"]: row["role"] for row in details["linked"]["rows"]} == {
        "MAR1": "consumes",
        "MAR2": "consumes and produces",
        "MAR3": "produces",
    }
    assert details["external_url"].endswith("/metabolite/MAM1m")


def test_without_a_comparison_an_element_has_its_facts_only():
    details = element_details(_workspace(AnalysisWorkspaceMode.NO_EXPERIMENT), "reaction", "MAR1")

    assert details["statistics"] is None and details["colour"] is None
    assert all(row["scored"] is False for row in details["linked"]["rows"])


def test_an_element_the_model_lacks_has_no_details():
    assert element_details(_workspace(), "reaction", "MAR9") is None
    assert element_details(_workspace(), "gene", "NOPE") is None
    with pytest.raises(ValueError):
        element_details(_workspace(), "subsystem", "x")


def test_a_reaction_no_gene_catalyses_has_no_genes_and_no_name():
    details = element_details(_workspace(), "reaction", "MAR4")

    assert (details["name"], details["linked"]["rows"], details["linked"]["total"]) == ("", [], 0)


def test_a_groups_violin_is_its_values_quartiles_and_density_trimmed_to_its_range():
    violin = _violin({"name": "Th17p", "color": "#0072B2", "values": [1.0, 2.0, 3.0, 4.0, 5.0, float("nan")]})

    assert (violin["n"], violin["min"], violin["q1"], violin["median"], violin["q3"], violin["max"]) == (5, 1, 2, 3, 4, 5)
    density = violin["density"]
    # From the least value to the greatest, scaled to 1 at its widest.
    assert (density[0][0], density[-1][0], max(share for _, share in density)) == (1, 5, 1)
    # One value has no density to draw, only its point.
    assert _violin({"name": "a", "color": "#000000", "values": [2.0, 2.0]})["density"] == []


def test_each_statistic_sits_among_every_reactions_with_its_percentile_and_the_colouring_one_in_the_keys_colours():
    details = element_details(_workspace(), "reaction", "MAR1")

    spreads = details["statistics"]["directions"][0]["spreads"]
    assert [spread["label"] for spread in spreads] == ["-Log(adj. p-value)", "Cohen's d", "Log2 fold change"]
    p_value, effect, fold = spreads
    # The adjusted p on -log10's axis, the most significant of the 100.
    assert (p_value["value"], p_value["at"], p_value["log"]) == (1e-6, pytest.approx(6.0), True)
    assert p_value["percentile"] == 99.5 and p_value["count"] == 100
    # Cohen's d, the lowest; the graph is coloured by it, so its bins take the key's colours.
    assert (effect["value"], effect["percentile"], effect["colours"], p_value["colours"]) == (-1.47, 0.5, True, False)
    assert len(effect["bins"]) == 30 and sum(bin["count"] for bin in effect["bins"]) == 100
    assert {bin["color"] for bin in effect["bins"]} == {"#0000ff", "#ff0000"}
    # The adjusted p's bins wear the hover's colours, whatever colours the graph; the fold change's none.
    assert {bin["color"] for bin in p_value["bins"]} == {"#ff0000"}
    assert all(bin["color"] == "" for bin in fold["bins"])
    # The axis reaches the value even when it lies past the 99.5th percentile.
    assert effect["bins"][0]["start"] <= -1.47 and fold["bins"][0]["start"] <= -0.147
    assert fold["value"] == -0.147


def test_a_reversible_reactions_backward_direction_is_set_among_them_by_its_own_values():
    details = element_details(_workspace(), "reaction", "MAR2")

    forward, backward = details["statistics"]["directions"]
    assert (forward["spreads"][1]["at"], backward["spreads"][1]["at"]) == (-0.2, 0.9)
    # The graph draws MAR2 backward: that is the value its colour reads.
    assert details["colour"]["value"] == 0.9


def test_a_graph_coloured_by_the_p_value_before_correction_sets_that_one_first():
    workspace = _workspace()
    from metabis.managers.element_details import _Details

    builder = _Details(workspace, "neg_log_pval", None)
    spreads = builder._comparison_spreads("reaction", "MAR1", {"pval": 1e-7, "cohens_d": -1.47, "log2_fc": -0.147})
    assert [(spread["label"], spread["colours"]) for spread in spreads] == [
        ("-Log(p-value)", True), ("Cohen's d", False), ("Log2 fold change", False)
    ]
    assert spreads[0]["value"] == 1e-7 and spreads[0]["at"] == pytest.approx(7.0)


class _Activity:
    """A single group's activity, scoring the reactions of its group, MAR2 shown backward."""

    metric = "activity|family=reaction"

    def __init__(self):
        self.scores = pd.Series({"MAR1": -1.0, "MAR2": 0.2, "MAR2_back": 1.4, **{f"R{i}": v for i, v in enumerate(np.linspace(-0.5, 0.5, 97))}})

    def metric_key_for(self, overlay):
        return self.metric

    def colors_element_type(self, overlay, kind):
        return kind == "reaction"

    def feature_result_for(self, overlay):
        return SimpleNamespace(activity_score=self.scores)


def test_a_single_groups_activity_sits_among_its_kinds_and_is_drawn_against_them():
    workspace = _workspace(AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS)
    activity = _Activity()
    workspace.data_provider.activity_artifact = activity
    workspace.data_provider.activity_overlay_for_element = lambda kind: "reaction"
    workspace.color_manager = SimpleNamespace(
        get_distribution_values=lambda mode, kind, metric: activity.scores if metric == activity.metric else None,
        get_color_for_value=lambda mode, kind, metric, value: "#b8860b" if value > 0 else "#7b68ee",
    )
    reference = list(np.linspace(-1, 1, 600))
    workspace.data_provider.get_reaction_data = lambda reaction_id, **_: {
        "is_in_model": True, "reaction_id": reaction_id, "name": "beta oxidation C12", "equation": "a[m] => b[m]",
        "subsystem": "Fatty acid oxidation", "is_reversible": True, "metadata": {},
        "experiment_stats": {
            "has_experiment_data": True, "group_id": "Th17p", "is_reversible": True, "main_direction": "backward",
            "forward_activity_score": 0.2, "backward_activity_score": 1.4, "summary_description": "normalized mean",
        },
        "distributions": {
            "has_distributions": True, "is_activity": True, "group_id": "Th17p", "feature_id": "MAR2_back",
            "reference_values": reference, "reference_name": "All reactions", "reference_count": 6065,
            "element_values": [1.2, 1.4, 1.5, 1.6],
        },
    }

    details = element_details(workspace, "reaction", "MAR2")

    statistics = details["statistics"]
    forward, backward = statistics["directions"]
    assert [spread["label"] for spread in backward["spreads"]] == ["Activity score"]
    assert (backward["spreads"][0]["at"], backward["spreads"][0]["percentile"], backward["spreads"][0]["colours"]) == (1.4, 99.5, True)
    assert forward["spreads"][0]["at"] == 0.2
    # Its samples against every reaction of the group: the reaction's own points
    # drawn, the reactions' only as their violin, from a 600-value summary of all 6,065.
    (violins,) = statistics["distributions"]
    everything, this = violins["groups"]
    assert (everything["name"], everything["n"], everything["unit"], everything["values"]) == ("All reactions", 6065, "reactions", [])
    assert (this["name"], this["n"], this["unit"], this["values"]) == ("This reaction", 4, "samples", [1.2, 1.4, 1.5, 1.6])
    assert "6,065 reactions measured in Th17p" in violins["caption"]
    # Its colour reads the direction the graph draws.
    assert details["colour"]["metric"] == "activity_score" and details["colour"]["value"] == 1.4


def test_a_subsystems_graph_has_its_genes_activity_read_as_it_draws_them_and_leaves_the_provider_as_it_was():
    workspace = _workspace(AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS)
    provider = workspace.data_provider
    provider.activity_dual_evidence = False
    seen = []
    reading = provider.get_gene_data

    def get_gene_data(gene_id, **options):
        seen.append(provider.activity_dual_evidence)
        return reading(gene_id, **options)

    provider.get_gene_data = get_gene_data
    element_details(workspace, "gene", "HADH", dual_evidence=True)
    element_details(workspace, "gene", "HADH")

    assert seen[0] is True and seen[-1] is False
    assert provider.activity_dual_evidence is False


def test_bins_short_of_a_threshold_are_grey_as_the_hovers_colour_reference_draws_them():
    workspace = _workspace()
    workspace.data_provider.group_comparison_artifact = SimpleNamespace(
        get_thresholds=lambda for_genes=False: {"neg_log_adjusted_pval": 1.3, "cohens_d": 0.5, "log2_fc": None}
    )

    p_value, effect, fold = element_details(workspace, "reaction", "MAR1")["statistics"]["directions"][0]["spreads"]

    assert p_value["thresholds"] == [1.3] and effect["thresholds"] == [-0.5, 0.5] and fold["thresholds"] == []
    for spread, line in ((p_value, 1.3), (effect, 0.5)):
        for bin in spread["bins"]:
            middle = (bin["start"] + bin["end"]) / 2
            assert (bin["color"] == "#c7c7c7") == (abs(middle) < line)


def test_a_graph_drawn_from_the_maps_links_the_genes_it_draws_beside_a_reaction_both_ways():
    # The map draws ACADM beside MAR2 too, whose gene rule names HADH only.
    drawn = {"MAR2": ["ENSG1", "ENSG2"], "MAR1": ["ENSG2", "ENSG1"]}

    reaction = element_details(_workspace(), "reaction", "MAR2", drawn_with=drawn)
    gene = element_details(_workspace(), "gene", "ACADM", drawn_with=drawn)

    assert [(row["label"], row["map_only"]) for row in reaction["linked"]["rows"]] == [("HADH", False), ("ACADM", True)]
    assert reaction["linked"]["total"] == 2
    assert [(row["id"], row["map_only"]) for row in gene["linked"]["rows"]] == [("MAR1", False), ("MAR2", True)]
    # Without a drawing, the model's own.
    assert [row["label"] for row in element_details(_workspace(), "reaction", "MAR2")["linked"]["rows"]] == ["HADH"]


def test_two_directions_violins_share_one_axis_as_the_hovers_do_past_the_activity_floor():
    from metabis.managers.hover.sections.distribution_plot import (
        DistributionPlotSection,
    )

    def direction(*groups):
        return {"groups": [{"values": list(values)} for values in groups]}

    shared = DistributionPlotSection.shared_axis
    assert shared([direction([3.0, 3.2], [3.1]), direction([2.9, 0.0], [3.3])]) == (2.9, 3.3)
    # One direction keeps its own axis; directions all at the floor leave each its own.
    assert shared([direction([3.0, 3.2], [3.1])]) is None
    assert shared([direction([0.0], [0.0]), direction([0.0], [0.0])]) is None

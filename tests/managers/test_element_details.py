"""A clicked element's details: its facts, its statistics, and the elements linked to it."""

from types import SimpleNamespace

import pytest

from metabis import element_details
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


def _workspace(mode=AnalysisWorkspaceMode.GROUP_COMPARISON):
    hovers = _Hovers()
    hovers.mode = mode
    return SimpleNamespace(hover_manager=hovers, data_provider=_Provider(), metabolic_model=_Model())


def test_a_reaction_has_its_facts_its_statistics_and_its_genes_with_theirs():
    details = element_details(_workspace(), "reaction", "MAR1_back")

    assert (details["id"], details["name"], details["subsystem"]) == ("MAR1", "beta oxidation C12", "Fatty acid oxidation")
    assert {"label": "Equation", "value": "a[m] => b[m]"} in details["facts"]
    statistics = details["statistics"]
    assert (statistics["kind"], statistics["target"], statistics["baseline"]) == ("comparison", "Inflamed", "Healthy")
    assert statistics["directions"] == [
        {"direction": "forward", "main": True, "cohens_d": -1.47, "log2_fc": -0.147, "mean_difference": -0.0735,
         "pval": 1e-7, "adjusted_pval": 1e-6, "significant": True, "test": "wilcoxon"}
    ]
    # Its genes in the order of its gene rule, each with its own comparison.
    genes = details["linked"]
    assert (genes["kind"], [row["label"] for row in genes["rows"]]) == ("gene", ["ACADM", "HADH"])
    assert genes["rows"][1] == {
        "id": "ENSG1", "label": "HADH", "name": "hydroxyacyl-CoA dehydrogenase", "role": "",
        "cohens_d": -0.87, "adjusted_pval": 3.6e-5, "significant": True, "activity_score": None, "scored": True,
    }
    assert genes["rows"][0]["scored"] is False


def test_a_gene_found_by_its_symbol_lists_its_reactions_on_the_map_most_significant_first():
    details = element_details(_workspace(), "gene", "HADH", within={"MAR1", "MAR2"})

    assert (details["id"], details["label"], details["name"]) == ("ENSG1", "HADH", "hydroxyacyl-CoA dehydrogenase")
    rows = details["linked"]["rows"]
    # MAR3 is not on the map. MAR2 reads its backward direction, as the map draws it.
    assert [(row["id"], row["cohens_d"]) for row in rows] == [("MAR1", -1.47), ("MAR2", 0.9)]
    assert details["linked"]["total"] == 2


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

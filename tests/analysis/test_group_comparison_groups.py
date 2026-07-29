import pandas as pd
import pytest

from metabis import GroupComparison, GroupComparisonSpec
from metabis.analysis.statistical_analysis import StatisticalAnalysis
from metabis.managers.group_manager import GroupDefinition, GroupManager


class _ReactionData:
    def __init__(self):
        self.frame = pd.DataFrame({"R1": [1.0, 2.0, 4.0, 5.0]}, index=["A1", "A2", "B1", "B2"])
    def get_sample_ids(self): return list(self.frame.index)
    def get_reaction_df(self): return self.frame


class _Model:
    def get_reaction_metadata(self, _): return {"subsystems": []}


class _Manager:
    def __init__(self):
        self.reaction_data = _ReactionData()
        self.expression_data = None
        self.metabolic_model = _Model()
        self.group_manager = GroupManager()
        self.group_manager.define_group("A", ["A1", "A2"], definition=GroupDefinition("custom"))
        self.group_manager.define_group("B", ["B1", "B2"], definition=GroupDefinition("custom"))
    def get_group_samples(self, group): return self.group_manager.get_group_samples(group)


def test_explicit_one_vs_rest_resolves_all_other_reaction_samples():
    manager = _Manager()
    comparison = GroupComparison(
        spec=GroupComparisonSpec.one_vs_rest(("A",)),
        test="ttest", correction="holm",
    )
    comparison.run_test(manager, test="ttest", correction="holm")

    assert comparison.membership.target_samples == ("A1", "A2")
    assert comparison.membership.baseline_samples == ("B1", "B2")
    assert comparison.membership.excluded_base_group_ids == ("A",)
    assert comparison.group_id_2 == "All other samples"
    assert comparison.get_test_results().iloc[0]["group_2"] == "All other samples"


def test_a_comparison_runs_on_membership_captured_before_analysis():
    manager = _Manager()
    membership = manager.group_manager.resolve_comparison_membership(
        GroupComparisonSpec.pairwise(("A",), ("B",)),
        manager.reaction_data.get_sample_ids(),
    )
    comparison = GroupComparison(test="ttest", correction="holm", membership=membership)

    # Redefining the group after capture must not move the samples under a running test.
    manager.group_manager.define_group(
        "B", ["A1", "B2"], definition=GroupDefinition("custom"), replace=True
    )
    comparison.run_test(manager, test="ttest", correction="holm", register_distributions=False)

    assert comparison.membership is membership
    assert comparison.membership.baseline_samples == ("B1", "B2")


def test_computed_results_carry_derived_metrics_without_registering_distributions():
    manager = _Manager()
    comparison = GroupComparison(
        spec=GroupComparisonSpec.pairwise(("A",), ("B",)),
        test="ttest", correction="holm",
    )

    comparison.run_test(manager, test="ttest", correction="holm", register_distributions=False)

    results = comparison.get_test_results()
    assert "neg_log_pval" in results.columns
    assert "neg_log_adjusted_pval" in results.columns
    assert results["neg_log_pval"].notna().all()


def test_comparison_signed_effects_are_target_minus_baseline():
    manager = _Manager()
    comparison = GroupComparison(
        spec=GroupComparisonSpec.pairwise(("B",), ("A",)),
        test="ttest",
        correction="holm",
    )

    comparison.run_test(
        manager,
        test="ttest",
        correction="holm",
        register_distributions=False,
    )

    result = comparison.get_test_results().loc["R1"]
    assert result["group_1"] == "B"
    assert result["group_2"] == "A"
    assert result["group_1_mean"] == 4.5
    assert result["group_2_mean"] == 1.5
    assert result["mean_difference"] == 3.0
    assert result["cohens_d"] > 0
    assert result["log2_fc"] == pytest.approx(1.584962500721156)


def test_pairwise_overlap_is_rejected_at_analysis_boundary():
    manager = _Manager()
    manager.group_manager.define_group("overlap", ["A2", "B1"], definition=GroupDefinition("custom"))
    comparison = GroupComparison(
        spec=GroupComparisonSpec.pairwise(("A",), ("overlap",)),
        test="ttest", correction="holm",
    )
    with pytest.raises(ValueError, match="overlap"):
        comparison.run_test(manager)


def test_computed_results_are_installed_through_the_public_comparison_seam():
    reaction_results = pd.DataFrame(
        {
            "pval": [0.01, 0.5],
            "adjusted_pval": [0.02, 0.5],
            "cohens_d": [1.5, 0.1],
            "log2_fc": [2.0, 0.0],
            "mean_difference": [3.0, 0.0],
        },
        index=["R1", "R2"],
    )
    gene_results = pd.DataFrame(
        {
            "pval": [0.03],
            "adjusted_pval": [0.04],
            "cohens_d": [-1.2],
            "log2_fc": [-1.0],
            "mean_difference": [-2.0],
        },
        index=["G1"],
    )
    reaction_metrics = {"cohens_d": {"mean": 0.8, "std": 0.7}}
    gene_metrics = {"cohens_d": {"mean": -1.2, "std": 0.0}}
    reaction_thresholds = {
        "pval": 0.05,
        "cohens_d": 0.5,
        "log2_fc": 1.0,
        "mean_difference": 2.0,
    }
    gene_thresholds = {
        "pval": 0.1,
        "cohens_d": 0.25,
        "log2_fc": 0.5,
        "mean_difference": 1.0,
    }

    comparison = GroupComparison.from_computed_results(
        spec=GroupComparisonSpec.pairwise(("A",), ("B",)),
        test="ttest",
        correction="holm",
        reaction_results=reaction_results,
        reaction_metrics=reaction_metrics,
        significant_reactions={"R1"},
        reaction_thresholds=reaction_thresholds,
        gene_results=gene_results,
        gene_metrics=gene_metrics,
        significant_genes={"G1"},
        gene_thresholds=gene_thresholds,
    )

    pd.testing.assert_frame_equal(comparison.get_test_results(), reaction_results)
    pd.testing.assert_frame_equal(
        comparison.get_test_results(for_genes=True), gene_results
    )
    assert comparison.get_metrics_stats() == reaction_metrics
    assert comparison.get_metrics_stats(for_genes=True) == gene_metrics
    assert comparison.get_significant_elements() == {"R1"}
    assert comparison.get_significant_elements(for_genes=True) == {"G1"}
    assert comparison.get_thresholds()["pval"] == 0.05
    assert comparison.get_thresholds(for_genes=True)["log2_fc"] == 0.5

    reaction_results.loc["R1", "pval"] = 1.0
    reaction_metrics["cohens_d"]["mean"] = 99.0
    assert comparison.get_test_results().loc["R1", "pval"] == 0.01
    assert comparison.get_metrics_stats()["cohens_d"]["mean"] == 0.8


def test_public_distribution_registration_does_not_mutate_scientific_tables():
    results = StatisticalAnalysis.attach_derived_metrics(
        pd.DataFrame(
            {
                "pval": [0.01],
                "adjusted_pval": [0.02],
                "cohens_d": [1.5],
                "log2_fc": [2.0],
                "mean_difference": [3.0],
            },
            index=["R1"],
        )
    )
    comparison = GroupComparison.from_computed_results(
        spec=GroupComparisonSpec.pairwise(("A",), ("B",)),
        test="ttest",
        correction="holm",
        reaction_results=results,
        reaction_metrics={"cohens_d": {"mean": 1.5, "std": 0.0}},
        significant_reactions={"R1"},
        reaction_thresholds={
            "pval": 0.05,
            "cohens_d": 0.5,
            "log2_fc": 1.0,
            "mean_difference": 2.0,
        },
    )
    calls = []

    class _ColorManager:
        def register_distribution(self, **kwargs):
            calls.append(kwargs)

    manager = type("PresentationAdapter", (), {"color_manager": _ColorManager()})()
    columns_before = comparison.get_test_results().columns.tolist()

    comparison.register_distributions(manager)

    assert {call["metric_name"] for call in calls} == {
        "pval",
        "adjusted_pval",
        "cohens_d",
        "log2_fc",
        "mean_difference",
        "neg_log_pval",
        "neg_log_adjusted_pval",
    }
    assert comparison.get_test_results().columns.tolist() == columns_before

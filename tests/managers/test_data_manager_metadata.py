import pandas as pd

from metabis import AnalysisWorkspace, GroupDefinition, SampleMetadata


class _ReactionData:
    def get_sample_ids(self):
        return ["S1", "S2"]


def test_metadata_update_advances_revision_and_invalidates_groups_and_results():
    manager = AnalysisWorkspace(reaction_data=_ReactionData())
    manager.group_manager.define_group(
        "old",
        ["S1"],
        definition=GroupDefinition("custom"),
        all_sample_ids=["S1", "S2"],
    )
    manager.group_manager.set_active_group("old")
    experiment = object()
    comparison = object()
    activity = object()
    manager.experiment_analysis = experiment
    manager.group_comparison_artifact = comparison
    manager.activity_artifact = activity
    manager.data_provider.experiment_analysis = experiment
    manager.data_provider.group_comparison_artifact = comparison
    manager.data_provider.activity_artifact = activity
    manager.cache_manager.cache_hover("reaction", "MAR1", "cached")
    manager.color_manager.generate_group_colors(["old"])

    report = manager.update_sample_metadata(
        SampleMetadata(
            pd.DataFrame({"label": ["one", "two"]}, index=["S1", "S2"]),
            display_label_column="label",
        )
    )

    assert report.is_strict_match
    assert manager.metadata_revision == 1
    assert manager.group_revision == 1
    assert manager.get_group_ids() == []
    assert manager.group_manager.group_records == {}
    assert manager.group_manager.get_active_group() is None
    assert manager.experiment_analysis is None
    assert manager.group_comparison_artifact is None
    assert manager.activity_artifact is None
    assert manager.data_provider.experiment_analysis is None
    assert manager.data_provider.group_comparison_artifact is None
    assert manager.data_provider.activity_artifact is None
    assert manager.cache_manager._hover_cache == {}
    assert manager.color_manager.get_group_colors() == {}
    assert manager.get_sample_display_label("S1") == "one"


def test_csv_expression_internal_anndata_is_not_exposed_as_sample_metadata():
    class Expression:
        format = "csv"
        adata = type("Adata", (), {"obs": pd.DataFrame({"internal": ["x"]}, index=["S1"])})()

    manager = AnalysisWorkspace(reaction_data=_ReactionData(), expression_data=Expression())
    assert manager.get_grouping_metadata() is None

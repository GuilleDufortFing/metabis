"""Single-group activity mode wiring: validation, colors, caches and terminology."""

import pandas as pd
import pytest

from metabis import (
    ActivityConfig,
    ActivityOverlay,
    AnalysisWorkspace,
    AnalysisWorkspaceMode,
    ExperimentAnalysis,
    GroupComparison,
    GroupComparisonSpec,
    GroupSummary,
)
from metabis.domain.expression_matrix import DenseExpressionMatrix
from metabis.managers.color_manager import ColorManager
from metabis.visual_config import theme


class FakeReactionData:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def get_reaction_df(self) -> pd.DataFrame:
        return self.frame

    def get_sample_ids(self):
        return list(self.frame.index)

    def get_reaction_row(self, reaction_id, reaction_format="mar_id"):
        if reaction_id not in self.frame.columns:
            return None
        return self.frame[reaction_id]


class FakeExpressionData:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def get_expression_df(self) -> pd.DataFrame:
        return self.frame

    def matrix(self) -> DenseExpressionMatrix:
        return DenseExpressionMatrix(self.frame)

    def get_feature_ids(self):
        return [str(column) for column in self.frame.columns]

    def get_sample_ids(self):
        return list(self.frame.index)

    def get_gene_row(self, gene_id):
        if gene_id not in self.frame.columns:
            return None
        return self.frame[gene_id]


class FakeSubsystem:
    def __init__(self, subsystem_id: str):
        self.id = subsystem_id
        self.name = subsystem_id.title()


class FakeModel:
    """The model surface the activity code and its hovers read."""

    def __init__(self, members):
        self.members = members

    def get_subsystems(self):
        return list(self.members)

    def get_reactions_for_subsystem(self, subsystem):
        return list(self.members[subsystem]["reactions"])

    def get_genes_for_subsystem(self, subsystem):
        return list(self.members[subsystem]["genes"])

    def get_subsystem(self, subsystem):
        return FakeSubsystem(subsystem) if subsystem in self.members else None

    def get_subsystem_category(self, subsystem):
        return "Test category"

    def get_reaction_metadata(self, reaction_id):
        return {
            "name": reaction_id,
            "display_label": reaction_id,
            "subsystems": ["glycolysis"],
            "is_reversible": True,
            "equation": "",
            "formula": "",
        }

    def get_gene_metadata(self, gene_id):
        return {
            "geneSymbols": gene_id,
            "geneNames": "",
            "display_label": gene_id,
        }


SAMPLES = ["A_1", "A_2", "B_1", "B_2"]

REACTIONS = pd.DataFrame(
    {
        "MAR1": [8.0, 10.0, 2.0, 0.0],
        "MAR1_back": [0.0, 0.0, 0.0, 0.0],
        "MAR2": [1.0, 3.0, 9.0, 7.0],
        "MAR3": [4.0, 4.0, 4.0, 4.0],
    },
    index=SAMPLES,
)

EXPRESSION = pd.DataFrame(
    {"GENE1": [10.0, 12.0, 1.0, 3.0], "GENE2": [5.0, 5.0, 5.0, 5.0]},
    index=SAMPLES,
)

MODEL = FakeModel(
    {
        "glycolysis": {"reactions": ["MAR1", "MAR2"], "genes": ["GENE1", "GENE2"]},
        "sparse pathway": {
            "reactions": ["MAR3"] + [f"UNMEASURED{index}" for index in range(9)],
            "genes": ["GENE1"] + [f"UNKNOWN{index}" for index in range(9)],
        },
    }
)


def build_manager(*, with_expression: bool = True) -> AnalysisWorkspace:
    manager = AnalysisWorkspace(
        metabolic_model=MODEL,
        reaction_data=FakeReactionData(REACTIONS),
        expression_data=FakeExpressionData(EXPRESSION) if with_expression else None,
    )
    manager.load_groups(criteria="sample_prefix")
    analysis = ExperimentAnalysis(manager, auto_run=False)
    analysis.run_analysis(include_pca=False)
    provenance = manager.capture_analysis_snapshot().provenance
    manager.install_experiment_analysis(provenance, analysis)
    manager.refresh_experiment_analysis_presentation()
    return manager


def config(**overrides) -> ActivityConfig:
    base = dict(
        group_id="A",
        summary=GroupSummary.MEAN,
        minimum_subsystem_coverage=0.5,
    )
    base.update(overrides)
    return ActivityConfig(**base)


def test_activating_activity_mode_selects_the_group_and_the_mode():
    manager = build_manager()

    context = manager.activate_activity_mode(config())

    assert manager.get_mode() is AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS
    assert manager.get_active_group() == "A"
    assert context.group_id == "A"
    assert context.features.sample_count == 2


def test_one_activity_artifact_carries_both_families_on_independent_scales():
    manager = build_manager()

    context = manager.activate_activity_mode(config())
    artifact = manager.get_activity_artifact()

    assert context.reaction_features.score("MAR1") == pytest.approx(9.0)
    assert context.gene_features.score("GENE1") == pytest.approx(11.0)
    assert artifact.available_families == (
        ActivityOverlay.REACTION,
        ActivityOverlay.GENE,
    )
    assert artifact.feature_result_for("reaction").score("MAR1") == pytest.approx(9.0)
    assert artifact.feature_result_for("gene").score("GENE1") == pytest.approx(11.0)
    assert set(artifact.feature_frame()["overlay"]) == {"reaction", "gene"}
    assert set(artifact.subsystem_frame()["overlay"]) == {"reaction", "gene"}

    reaction_key = artifact.metric_key_for(ActivityOverlay.REACTION)
    gene_key = artifact.metric_key_for(ActivityOverlay.GENE)
    assert reaction_key != gene_key
    reaction_limit = manager.color_manager.get_color_limit(
        AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS, "reaction", reaction_key
    )
    gene_limit = manager.color_manager.get_color_limit(
        AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS, "gene", gene_key
    )
    assert reaction_limit is not None
    assert gene_limit is not None
    assert reaction_limit != gene_limit


def test_an_unknown_active_group_is_rejected_with_the_available_groups():
    manager = build_manager()

    with pytest.raises(ValueError, match="does not exist"):
        manager.activate_activity_mode(config(group_id="does-not-exist"))


def test_a_comparison_local_population_is_scored_without_being_a_base_group():
    """A Step-3 Target/Baseline union is not a saved group, so it brings its samples."""
    manager = build_manager()

    context = manager.activate_activity_mode(
        config(group_id="Target: A + B", sample_ids=("A_1", "B_1"))
    )
    artifact = manager.get_activity_artifact()

    # MAR1 over [A_1, B_1] is mean(8, 2); the group-A-only view scores 9.
    assert context.reaction_features.score("MAR1") == pytest.approx(5.0)
    assert context.features.sample_count == 2
    # The artifact records the exact membership it summarized, so an export names
    # the samples rather than a label the workspace cannot resolve.
    assert artifact.selected_sample_ids == ("A_1", "B_1")
    assert artifact.group_id == "Target: A + B"
    # No base group was activated, because none of them is this population.
    assert manager.get_active_group() != "Target: A + B"


def test_a_population_and_a_base_group_with_the_same_label_do_not_share_a_result():
    manager = build_manager()

    manager.activate_activity_mode(config(group_id="A"))
    base_only = manager.get_activity_artifact()
    manager.activate_activity_mode(config(group_id="A", sample_ids=("A_1", "B_1")))
    population = manager.get_activity_artifact()

    assert base_only.metric_key != population.metric_key
    assert base_only.selected_sample_ids == ("A_1", "A_2")
    assert population.selected_sample_ids == ("A_1", "B_1")
    assert population.feature_result_for("reaction").score("MAR1") == pytest.approx(5.0)


def test_a_population_with_no_aligned_samples_is_rejected():
    manager = build_manager()

    with pytest.raises(ValueError, match="no samples in the aligned matrix"):
        manager.activate_activity_mode(
            config(group_id="Target: ghosts", sample_ids=("nope_1", "nope_2"))
        )


def test_the_gene_overlay_is_rejected_without_expression_data():
    manager = build_manager(with_expression=False)

    with pytest.raises(ValueError, match="needs an expression matrix"):
        manager.activate_activity_mode(config(), overlay=ActivityOverlay.GENE)


def test_comparison_mode_is_rejected_rather_than_silently_downgraded():
    manager = build_manager()

    with pytest.raises(ValueError, match="Run a group comparison"):
        manager.apply_visualization_context(AnalysisWorkspaceMode.GROUP_COMPARISON)


def test_reactions_are_colored_from_the_activity_result():
    manager = build_manager()
    manager.activate_activity_mode(config())

    # MAR1 is far above its dataset reference in group A; MAR2 far below.
    high = manager.get_element_color("MAR1", "reaction")
    low = manager.get_element_color("MAR2", "reaction")

    assert high != low
    assert high != theme.ACTIVITY_UNAVAILABLE
    assert low != theme.ACTIVITY_UNAVAILABLE


def test_activity_runs_purple_grey_gold_not_the_regulation_heatmap():
    """Activity is a level against the typical one: purple below, grey at it, gold above."""
    manager = build_manager()
    manager.activate_activity_mode(config())

    high = ColorManager.hex_to_rgb(manager.get_element_color("MAR1", "reaction"))
    low = ColorManager.hex_to_rgb(manager.get_element_color("MAR2", "reaction"))
    grey = ColorManager.hex_to_rgb(theme.ACTIVITY_MID)

    # Away from the grey centre, high activity warms toward gold and low activity
    # cools toward purple: red orders the two one way and blue the other.
    assert high[0] > low[0]
    assert high[2] < grey[2] and low[2] < grey[2] and high[2] < low[2]
    assert high != grey and low != grey

    # And never the red/blue of the comparison heatmap.
    assert manager.get_element_color("MAR1", "reaction") != theme.UP_REGULATED
    assert manager.get_element_color("MAR2", "reaction") != theme.DOWN_REGULATED


def test_positive_activity_uses_white_at_the_lowest_measured_value():
    """An impossible negative range must not consume the low end of the ramp."""
    manager = build_manager()
    context = manager.activate_activity_mode(config())

    scores = context.features.activity_score
    lowest_id = str(scores.idxmin())
    highest_id = str(scores.idxmax())
    metric_key = context.config.metric_key_for("reaction")

    def registered_color(element_id: str) -> str:
        return manager.color_manager.get_element_color_from_distribution(
            AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS,
            "reaction",
            element_id,
            metric_key,
        )

    assert float(scores.min()) >= 0.0
    assert (
        registered_color(lowest_id).upper() == theme.ACTIVITY_LOW.upper()
    )
    assert (
        registered_color(highest_id).upper() == theme.ACTIVITY_HIGH.upper()
    )

    details = manager.color_manager.get_distribution_details(
        AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS,
        "reaction",
        lowest_id,
        metric_key,
    )
    plotted_lower_edge = details["histogram"][0]["value_start"]
    measured_span = float(scores.max() - scores.min())
    assert plotted_lower_edge >= float(scores.min()) - measured_span / 100
    widths = [
        bin_data["value_end"] - bin_data["value_start"]
        for bin_data in details["histogram"]
    ]
    assert widths == pytest.approx([widths[0]] * len(widths))
    assert sum(bin_data["count"] for bin_data in details["histogram"]) == len(scores)


def test_a_gene_overlay_leaves_reaction_glyphs_uncolored_and_vice_versa():
    manager = build_manager()

    manager.activate_activity_mode(config(), overlay=ActivityOverlay.GENE)
    assert manager.get_element_color("MAR1", "reaction") == theme.ACTIVITY_UNAVAILABLE
    assert manager.get_element_color("GENE1", "gene") != theme.ACTIVITY_UNAVAILABLE

    manager.activate_activity_mode(config(), overlay=ActivityOverlay.REACTION)
    assert manager.get_element_color("GENE1", "gene") == theme.ACTIVITY_UNAVAILABLE
    assert manager.get_element_color("MAR1", "reaction") != theme.ACTIVITY_UNAVAILABLE


def test_detailed_subsystem_evidence_colors_both_activity_families():
    manager = build_manager()
    manager.activate_activity_mode(config(), overlay=ActivityOverlay.GENE)

    manager.data_provider.activity_dual_evidence = True

    reaction_color = manager.get_element_color("MAR1", "reaction")
    gene_color = manager.get_element_color("GENE1", "gene")
    reaction_stats = manager.data_provider.get_activity_statistics(
        "MAR1", "reaction"
    )
    gene_stats = manager.data_provider.get_activity_statistics("GENE1", "gene")

    assert reaction_color != theme.ACTIVITY_UNAVAILABLE
    assert gene_color != theme.ACTIVITY_UNAVAILABLE
    assert reaction_stats["overlay"] == "reaction"
    assert gene_stats["overlay"] == "gene"
    assert reaction_stats["activity_score"] == pytest.approx(9.0)
    assert gene_stats["activity_score"] == pytest.approx(11.0)


def test_a_subsystem_below_the_coverage_threshold_is_gray_not_neutral():
    manager = build_manager()
    manager.activate_activity_mode(config())

    # "sparse pathway" measures 1 of 10 model reactions, below the 50% threshold.
    assert (
        manager.get_element_color("sparse pathway", "subsystem")
        == theme.ACTIVITY_UNAVAILABLE
    )
    assert manager.get_element_color("glycolysis", "subsystem") != theme.ACTIVITY_UNAVAILABLE


def test_changing_the_group_overlay_or_statistic_changes_the_hover_cache_identity():
    manager = build_manager()

    keys = set()
    for activity_config, overlay in (
        (config(), ActivityOverlay.REACTION),
        (config(group_id="B"), ActivityOverlay.REACTION),
        (config(), ActivityOverlay.GENE),
        (config(summary=GroupSummary.MEDIAN), ActivityOverlay.REACTION),
        (config(minimum_subsystem_coverage=0.25), ActivityOverlay.REACTION),
    ):
        manager.activate_activity_mode(activity_config, overlay=overlay)
        keys.add(
            manager.cache_manager.get_cache_key(
                manager.group_comparison_artifact,
                mode=manager.get_mode().value,
                active_group=manager.get_active_group(),
                activity_key=manager.activity_artifact.metric_key_for(overlay),
            )
        )

    assert len(keys) == 5


def test_activity_hover_states_the_definition_and_omits_comparison_statistics():
    manager = build_manager()
    manager.activate_activity_mode(config())

    stats = manager.data_provider.get_activity_statistics("MAR1", "reaction")

    assert stats["has_experiment_data"] is True
    assert stats["group_id"] == "A"
    assert stats["sample_count"] == 2
    assert stats["activity_score"] == pytest.approx(9.0)
    assert "p_value" not in stats
    assert "cohens_d" not in stats
    assert "log2_fc" not in stats
    assert stats["summary_description"] == "mean across the group's samples"
    assert "scaling" not in stats


def test_subsystem_activity_hover_reports_measured_coverage():
    manager = build_manager()
    manager.activate_activity_mode(config())

    stats = manager.data_provider.get_activity_statistics("sparse pathway", "subsystem")

    assert stats["activity_score"] is None
    assert stats["measured_features"] == 1
    assert stats["total_model_features"] == 10
    assert stats["coverage"] == pytest.approx(0.1)


def test_the_activity_legend_never_claims_regulation():
    manager = build_manager()
    manager.activate_activity_mode(config())

    legend = manager.get_color_legend_text()

    assert "regulated" not in legend.lower()
    assert "normalized mean" in legend.lower()
    assert "coverage threshold" not in legend.lower()
    # The legend names levels, not directions.
    assert "gold = high" in legend.lower()
    assert "grey = typical" in legend.lower()
    assert "purple = low" in legend.lower()


def test_the_activity_hover_violin_compares_the_feature_against_all_features():
    manager = build_manager()
    context = manager.activate_activity_mode(config())

    distributions = manager.data_provider.get_activity_distributions("MAR1", "reaction")

    assert distributions["has_distributions"] is True
    assert distributions["is_activity"] is True
    # Left violin: every measured reaction's group summary. Right violin: this
    # reaction across the group's own samples.
    assert distributions["reference_values"] == list(context.reference_summary)
    assert distributions["element_values"] == [8.0, 10.0]
    assert distributions["sample_count"] == 2
    assert distributions["group_id"] == "A"


def test_the_hover_violin_follows_the_direction_the_group_displays():
    manager = build_manager()
    manager.activate_activity_mode(config())

    # MAR1_back is all zeros while MAR1 is strongly positive, so forward is shown.
    distributions = manager.data_provider.get_activity_distributions(
        "MAR1_back", "reaction"
    )

    assert distributions["feature_id"] == "MAR1"
    assert distributions["element_values"] == [8.0, 10.0]


def test_uncolored_glyphs_get_no_activity_violin():
    """A reaction under a gene overlay has no activity score to plot."""
    manager = build_manager()
    manager.activate_activity_mode(config(), overlay=ActivityOverlay.GENE)

    assert (
        manager.data_provider.get_activity_distributions("MAR1", "reaction")[
            "has_distributions"
        ]
        is False
    )
    assert (
        manager.data_provider.get_activity_distributions("GENE1", "gene")[
            "has_distributions"
        ]
        is True
    )


def test_the_violin_is_only_built_under_the_full_hover_detail_preset():
    manager = build_manager()
    manager.activate_activity_mode(config())

    full = manager.get_element_hover(
        "MAR1", "reaction", show_stats=True, show_plot=True
    )
    light = manager.get_element_hover(
        "MAR1", "reaction", show_stats=True, show_plot=False
    )

    assert "Activity in context" in full
    assert "<svg" in full
    assert "Activity in context" not in light
    assert "<svg" not in light


def test_uncolored_glyphs_still_report_their_own_measurement():
    """A reaction under a gene overlay is uncolored, but it is not missing data."""
    manager = build_manager()
    manager.activate_activity_mode(config(), overlay=ActivityOverlay.GENE)

    stats = manager.data_provider.get_activity_statistics("MAR1", "reaction")

    assert stats["has_experiment_data"] is True
    assert stats["is_colored"] is False
    assert stats["activity_score"] is None


def test_a_reaction_absent_from_the_matrix_reports_no_data():
    manager = build_manager()
    manager.activate_activity_mode(config())

    stats = manager.data_provider.get_activity_statistics("UNMEASURED0", "reaction")

    assert stats["has_experiment_data"] is False


def test_reloading_groups_discards_the_previous_activity_result():
    manager = build_manager()
    manager.activate_activity_mode(config())
    assert manager.activity_artifact is not None

    manager.load_groups(criteria="sample_prefix")

    assert manager.activity_artifact is None
    assert manager.data_provider.activity_artifact is None
    assert manager.get_mode() is AnalysisWorkspaceMode.NO_EXPERIMENT


def test_a_comparison_hover_survives_a_leftover_activity_artifact(install_comparison):
    """Installing a comparison flips the mode but does not clear the artifact.

    The hover then chooses its violin section from the *mode* and its
    distributions from the *artifact*, and the single-group shape has no
    ``group2_values`` for the comparison violin -- which reached users as a red
    "Plot error: 'group2_values'" card on every reaction of a rendered graph.
    Nothing in this state can draw a comparison violin, so the requirement is
    that the hover stays readable rather than that a plot appears.
    """
    manager = build_manager()
    manager.activate_activity_mode(config())

    comparison = GroupComparison(
        membership=manager.group_manager.resolve_comparison_membership(
            GroupComparisonSpec.pairwise(("A",), ("B",)), SAMPLES
        )
    )
    comparison.run_test(manager, test="ttest", correction="holm")
    install_comparison(manager, comparison)

    assert manager.get_mode() is AnalysisWorkspaceMode.GROUP_COMPARISON
    assert manager.data_provider.activity_artifact is not None

    hover = manager.get_element_hover(
        "MAR1", "reaction", show_plot=True, show_stats=True
    )

    assert "Plot error:" not in hover
    assert "(Error:" not in hover

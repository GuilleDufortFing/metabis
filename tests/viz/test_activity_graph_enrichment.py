"""End-to-end single-group activity rendering against the packaged Human1 model."""

import numpy as np
import pandas as pd
import pytest
from bokeh.models import HoverTool

from metabis import (
    ActivityConfig,
    AnalysisWorkspace,
    AnalysisWorkspaceMode,
    ExperimentAnalysis,
    GroupSummary,
    MetabolicModel,
    ReactionData,
    ReactionPreprocessingConfig,
    plot_metabolic_graph,
    plot_subsystem_activity_ranking,
)
from metabis.viz.metabolic_visualization import EnrichmentConfig
from metabis.viz.plot_subsystem_ranking import _saturating_axis_bounds

SUBSYSTEM_ID = "galactose_metabolism"


@pytest.fixture(scope="module")
def model() -> MetabolicModel:
    return MetabolicModel()


@pytest.fixture(scope="module")
def workspace(model, tmp_path_factory):
    """Two prefix groups over real Human1 reaction IDs, with one planted signal.

    Returns the manager alongside the planted reaction ID: subsystem membership is
    not returned in a stable order, so the ID has to be carried, not re-derived.
    """
    reactions = model.get_reactions_for_subsystem(SUBSYSTEM_ID)
    planted = reactions[0]
    samples = [f"A_{index}" for index in range(3)] + [f"B_{index}" for index in range(3)]

    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        rng.normal(5.0, 2.0, size=(len(samples), len(reactions))),
        index=samples,
        columns=reactions,
    )
    # Make one reaction unambiguously more active in group A.
    frame.loc[frame.index[:3], planted] += 20.0

    path = tmp_path_factory.mktemp("activity") / "reactions.tsv"
    frame.to_csv(path, sep="\t")

    reaction_data = ReactionData(
        str(path),
        "single",
        model,
        csv_separator="\t",
        samples_in="rows",
        preprocessing=ReactionPreprocessingConfig(),
    )

    manager = AnalysisWorkspace(metabolic_model=model, reaction_data=reaction_data)
    manager.load_groups(criteria="sample_prefix")
    analysis = ExperimentAnalysis(manager, auto_run=False)
    analysis.run_analysis(include_pca=False)
    provenance = manager.capture_analysis_snapshot().provenance
    manager.install_experiment_analysis(provenance, analysis)
    manager.refresh_experiment_analysis_presentation()
    return {"manager": manager, "planted": planted}


@pytest.fixture(scope="module")
def data_manager(workspace) -> AnalysisWorkspace:
    return workspace["manager"]


def activity_config(**overrides) -> ActivityConfig:
    base = dict(
        group_id="A",
        summary=GroupSummary.MEAN,
        minimum_subsystem_coverage=0.1,
    )
    base.update(overrides)
    return ActivityConfig(**base)


@pytest.mark.integration
def test_a_subsystem_renders_in_activity_mode_without_any_comparison(model, data_manager):
    context = data_manager.activate_activity_mode(activity_config())
    subsystem_name = model.get_subsystem(SUBSYSTEM_ID).name

    plot, metadata = plot_metabolic_graph(
        element_name=subsystem_name,
        data_manager=data_manager,
        graph_type="subsystem",
        show_plot=False,
        enable_timing=True,
        return_metadata=True,
        add_click_interactions=False,
        enrichment_config=EnrichmentConfig(color_metric=context.metric_key),
    )

    assert data_manager.group_comparison_artifact is None
    assert plot is not None
    assert metadata.glyph_count > 0
    assert data_manager.data_provider.activity_dual_evidence is False


@pytest.mark.integration
def test_the_planted_signal_is_colored_and_hovers_carry_no_comparison_statistics(
    workspace, data_manager
):
    context = data_manager.activate_activity_mode(activity_config())
    planted = workspace["planted"]

    assert context.features.activity_score.idxmax() == planted
    assert context.features.score(planted) > 20.0

    hover = data_manager.get_element_hover(
        planted, "reaction", show_stats=True, use_main_direction=True
    )

    assert "Single-group activity" in hover
    assert "Normalized group summary" in hover
    for comparison_term in ("p-value", "Cohen", "Fold Change", "up-regulated"):
        assert comparison_term.lower() not in hover.lower()


@pytest.mark.integration
def test_the_subsystem_hover_reports_coverage(model, data_manager):
    data_manager.activate_activity_mode(activity_config())

    hover = data_manager.get_element_hover(
        SUBSYSTEM_ID, "subsystem", show_stats=True
    )

    assert "Coverage:" in hover
    assert "Significant" not in hover


@pytest.mark.integration
def test_activity_ranking_plots_measured_members_without_a_significance_filter(
    data_manager,
):
    data_manager.activate_activity_mode(activity_config())

    plot = plot_subsystem_activity_ranking(
        data_manager=data_manager,
        activity_artifact=data_manager.get_activity_artifact(),
        return_plot=True,
    )

    assert plot is not None
    assert plot.xaxis[0].axis_label.startswith("Normalized group summary")
    assert "Subsystem activity ranking - A" in plot.title.text
    # Points pinned to the bounded axis are drawn as diamonds, not silently moved.
    markers = set(plot.renderers[0].data_source.data["marker"])
    assert markers <= {"circle", "diamond"}


@pytest.mark.integration
def test_the_activity_ranking_carries_its_gradient_color_reference(data_manager):
    """R-09. Supp. 1.5 claims this panel for every gradient plot.

    The ranking colours its dots from the activity distribution, so every hover
    shows that distribution with the dot's own score marked. A point clamped to
    the bounded axis still marks its true score, not the edge it was drawn at.
    """
    data_manager.activate_activity_mode(activity_config())

    plot = plot_subsystem_activity_ranking(
        data_manager=data_manager,
        activity_artifact=data_manager.get_activity_artifact(),
        return_plot=True,
    )
    source = plot.renderers[0].data_source.data
    hover = plot.select_one(HoverTool)

    assert source["desc"]
    assert "<strong>Color Reference:</strong>" in hover.tooltips
    assert "Activity score Distribution" in hover.tooltips
    assert hover.tooltips.count("<svg") == 1
    assert all("<svg" not in description for description in source["desc"])

    scores = data_manager.get_activity_artifact().reaction_feature_result.activity_score
    assert list(source["color_reference_value"]) == [
        float(scores.loc[element_id]) for element_id in source["element_id"]
    ]


def test_the_ranking_axis_is_bounded_by_the_bulk_not_by_one_outlier():
    # 200 reactions inside a narrow band, one an order of magnitude further out.
    values = list(np.linspace(-0.3, 0.3, 200)) + [8.0]

    lower, upper = _saturating_axis_bounds(values)

    # The axis frames the band the data actually occupies, so the ranking is
    # readable, instead of stretching to 8 and collapsing every point onto zero.
    assert upper < 1.0
    assert lower < 0.0 < upper
    assert lower == pytest.approx(-0.3, abs=0.02)


def test_the_ranking_axis_always_contains_the_reference():
    """Zero is the typical-activity reference, so it stays visible."""
    lower, upper = _saturating_axis_bounds([1.0, 1.2, 1.4, 1.6])

    assert lower <= 0.0 <= upper


@pytest.mark.integration
def test_activity_colors_use_the_full_palette_on_real_data(workspace, data_manager):
    """The bug this guards: a 4-SD palette painted every real graph amber."""
    context = data_manager.activate_activity_mode(activity_config())
    scores = context.features.activity_score

    colors = {
        data_manager.get_element_color(feature_id, "reaction")
        for feature_id in scores.index
    }

    # A real, concentrated activity distribution must produce visibly different
    # colours, not one flat neutral.
    assert len(colors) > 5

    limit = data_manager.color_manager.get_color_limit(
        AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS,
        "reaction",
        context.config.metric_key_for("reaction"),
    )
    # The palette is stretched over the data, so its limit tracks the data's range
    # rather than sitting at four standard deviations out.
    assert 0 < limit <= float(scores.abs().max())


@pytest.mark.integration
def test_changing_the_group_recolors_the_same_reaction(workspace, data_manager):
    planted = workspace["planted"]

    data_manager.activate_activity_mode(activity_config(group_id="A"))
    color_a = data_manager.get_element_color(planted, "reaction")

    data_manager.activate_activity_mode(activity_config(group_id="B"))
    color_b = data_manager.get_element_color(planted, "reaction")

    assert color_a != color_b

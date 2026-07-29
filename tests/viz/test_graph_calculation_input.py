"""Graph calculation must run from plain values, and paint what a workspace paints.

``GraphCalculationInput`` replaces the ``AnalysisWorkspace`` that graph
enrichment used to take. The server's version of that workspace was
*reconstructed* per job in eight steps, including six revision counters assigned
by server code and a colour-manager deepcopy with an explicit hover rewiring
(AM10 inventory, part 2). The point of the type is that none of that is
necessary: everything a graph reads is a plain artifact, frame, membership list,
or already-resolved colour.

Two properties carry that claim, and the second is the one with teeth:

* a graph built from plain values -- no workspace constructed anywhere -- is
  byte-identical to one built from the live workspace holding the same data. Not
  "looks right": the same golden colour digest the AM4 goldens assert;
* the whole path imports and computes with Bokeh unavailable, asserted against a
  blocked import in a fresh interpreter. Slices 2 and 3b each found a Bokeh edge
  that every static scan called clean, so only a runtime probe is evidence.

See ``MetabisReview/architecture-migration/am10-inventory.md`` (M-12 A').
"""

import hashlib
import re

import numpy as np
import pandas as pd
import pytest

from metabis import (
    AnalysisWorkspace,
    ExperimentAnalysis,
    GroupComparison,
    GroupComparisonSpec,
    MetabolicModel,
    ReactionData,
    ReactionPreprocessingConfig,
)
from metabis.managers.color_distributions import ColorDistributionRegistry
from metabis.managers.workspace_mode import AnalysisWorkspaceMode
from metabis.viz.metabolic_visualization.calculation_input import GraphCalculationInput
from metabis.viz.metabolic_visualization.enrichment.enricher import EnrichmentConfig
from metabis.viz.metabolic_visualization.generation.generator import (
    GraphGenerationConfig,
)
from metabis.viz.metabolic_visualization.pipeline import (
    MetabolicGraphPipeline,
    PipelineConfig,
    as_calculation_input,
)
from tests.managers.test_color_lookup import run_without_bokeh

SUBSYSTEM_NAME = "Glycolysis / Gluconeogenesis"
SUBSYSTEM_ID = "glycolysis_gluconeogenesis"


def build(model, calculation_input, *, color_metric="cohens_d"):
    pipeline = MetabolicGraphPipeline(
        model,
        PipelineConfig(
            generation_config=GraphGenerationConfig(
                use_human1_map=True, layout_engine="dot"
            ),
            enrichment_config=EnrichmentConfig(
                color_metric=color_metric,
                show_hover_stats=True,
            ),
            show_plot=False,
        ),
    )
    return pipeline.build_artifact(
        SUBSYSTEM_NAME, "subsystem", data_manager=calculation_input
    )


def comparable_hover(hover) -> str:
    """Hover text with the violin SVG removed.

    ``create_static_violin_for_hover`` jitters its sample points with
    ``np.random``, so the same distribution draws a different SVG every call --
    hover HTML is not reproducible and cannot be compared byte-for-byte. That is
    pre-existing and orthogonal to this seam; everything *around* the plot is
    deterministic and is what these tests compare.
    """
    return re.sub(r"<svg.*?</svg>", "<svg/>", str(hover or ""), flags=re.DOTALL)


def colour_digest(artifact) -> str:
    """What each node is painted, keyed by canonical ID (the AM4 golden shape)."""
    payload = tuple(
        (str(node["id"]), str(node.get("color")), round(float(node.get("alpha", 1.0)), 3))
        for node in artifact.nodes
    )
    return hashlib.sha256(repr(payload).encode()).hexdigest()


@pytest.fixture(scope="module")
def model():
    return MetabolicModel(use_cache=True)


@pytest.fixture(scope="module")
def loaded(model, tmp_path_factory):
    """A real two-group comparison over packaged glycolysis reactions."""
    reactions = model.get_reactions_for_subsystem(SUBSYSTEM_ID)
    samples = [f"A_{index}" for index in range(8)] + [f"B_{index}" for index in range(8)]
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        rng.normal(5.0, 1.0, size=(len(samples), len(reactions))),
        index=samples,
        columns=reactions,
    )
    frame.loc[frame.index[:8], reactions[0]] += 25.0

    path = tmp_path_factory.mktemp("calculation-input") / "reactions.tsv"
    frame.to_csv(path, sep="\t")
    reaction_data = ReactionData(
        str(path),
        "single",
        model,
        csv_separator="\t",
        samples_in="rows",
        preprocessing=ReactionPreprocessingConfig(),
    )
    workspace = AnalysisWorkspace(metabolic_model=model, reaction_data=reaction_data)
    workspace.load_groups(criteria="sample_prefix")
    analysis = ExperimentAnalysis(workspace, auto_run=False)
    analysis.run_analysis(include_pca=False)
    analysis_provenance = workspace.capture_analysis_snapshot().provenance
    workspace.install_experiment_analysis(analysis_provenance, analysis)
    workspace.refresh_experiment_analysis_presentation()

    comparison = GroupComparison(
        spec=GroupComparisonSpec.pairwise(("A",), ("B",)),
        test="ttest",
        correction="fdr_bh",
    )
    comparison.run_test(workspace, for_genes=False)
    comparison.define_significant_elements(
        pval_threshold=0.05,
        cohens_d_threshold=0.0,
        log2_fc_threshold=0.0,
        mean_difference_threshold=0.0,
        for_genes=False,
    )
    snapshot = workspace.capture_analysis_snapshot(set())
    artifact = comparison.build_artifact(
        provenance=snapshot.provenance,
        source_fingerprints=workspace.source_fingerprints,
    )
    workspace.install_group_comparison_artifact(snapshot.provenance, artifact)
    workspace.refresh_group_comparison_presentation()
    return {"workspace": workspace, "artifact": artifact, "samples": samples}


def plain_input_for(loaded, model):
    """Build the input the way a worker would: plain values, no workspace."""
    workspace = loaded["workspace"]
    calculation_input = GraphCalculationInput.from_plain_values(
        metabolic_model=model,
        mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
        reaction_data=workspace.reaction_data,
        group_comparison_artifact=loaded["artifact"],
        color_registry=workspace.color_manager.detached_color_registry(),
        group_memberships=[
            (group_id, workspace.get_group_samples(group_id))
            for group_id in workspace.get_group_ids()
        ],
        all_sample_ids=loaded["samples"],
    )
    return calculation_input.with_experiment_analysis(
        ExperimentAnalysis.from_summary_snapshot(
            calculation_input,
            workspace.experiment_analysis.snapshot_summaries(),
        )
    )


# --- The property the type exists for ------------------------------------


@pytest.mark.integration
def test_a_plain_input_paints_the_same_graph_as_the_workspace(model, loaded):
    """No workspace on the input path, and not one pixel of difference."""
    from_workspace = build(model, loaded["workspace"])
    from_plain = build(model, plain_input_for(loaded, model))

    assert colour_digest(from_plain) == colour_digest(from_workspace)
    assert [node["id"] for node in from_plain.nodes] == [
        node["id"] for node in from_workspace.nodes
    ]
    assert [comparable_hover(node.get("hover")) for node in from_plain.nodes] == [
        comparable_hover(node.get("hover")) for node in from_workspace.nodes
    ]
    assert any(node.get("hover") for node in from_plain.nodes)
    # And the run was not vacuous: real comparison colours, not all-neutral.
    assert len({node.get("color") for node in from_plain.nodes}) > 2


@pytest.mark.integration
def test_a_plain_input_holds_no_workspace(model, loaded):
    """The whole point of M-12: nothing reachable from the input is a workspace."""
    calculation_input = plain_input_for(loaded, model)

    assert not isinstance(calculation_input, AnalysisWorkspace)
    assert not isinstance(calculation_input.subsystem_analysis.dm, AnalysisWorkspace)
    assert calculation_input.subsystem_analysis.dm is calculation_input
    assert not isinstance(
        calculation_input.experiment_analysis.data_manager, AnalysisWorkspace
    )
    # No revision counter a caller could advance, and no installation seam.
    for forbidden in (
        "apply_model_transition",
        "install_activity_artifact",
        "update_group_comparison",
        "capture_analysis_snapshot",
        "source_fingerprints",
    ):
        assert not hasattr(calculation_input, forbidden)


@pytest.mark.integration
def test_a_worker_registration_does_not_reach_the_document_thread(model, loaded):
    """A distribution the render computes stays in the render (AM8 detachment)."""
    workspace = loaded["workspace"]
    calculation_input = plain_input_for(loaded, model)
    before = set(workspace.color_manager._distributions)

    calculation_input.color_manager.register_distribution(
        mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
        element_type="subsystem",
        data=pd.Series({"glycolysis_gluconeogenesis": 1.0, "other": -1.0}),
        metric_name="worker_only",
    )

    assert set(workspace.color_manager._distributions) == before


# --- The capture producer ------------------------------------------------


def test_a_workspace_is_captured_into_the_typed_input(loaded):
    workspace = loaded["workspace"]

    captured = as_calculation_input(workspace)

    assert isinstance(captured, GraphCalculationInput)
    assert captured.metabolic_model is workspace.metabolic_model
    assert captured.get_mode() == workspace.get_mode()
    assert captured.get_group_comparison_artifact() is (
        workspace.group_comparison_artifact
    )
    assert captured.output_reaction_format == workspace.output_reaction_format


def test_an_input_is_passed_through_unchanged(loaded, model):
    calculation_input = plain_input_for(loaded, model)

    assert as_calculation_input(calculation_input) is calculation_input


def test_subsystem_capture_scopes_dual_activity_evidence_to_the_graph_input(loaded):
    workspace = loaded["workspace"]
    captured = as_calculation_input(workspace, activity_dual_evidence=True)

    assert captured.activity_dual_evidence is True
    assert captured.data_provider.activity_dual_evidence is False
    assert workspace.data_provider.activity_dual_evidence is False
    assert as_calculation_input(None) is None


def test_the_input_is_frozen(loaded):
    """Enrichment cannot rebind a collaborator mid-run."""
    import dataclasses

    captured = as_calculation_input(loaded["workspace"])

    with pytest.raises(dataclasses.FrozenInstanceError):
        captured.metabolic_model = None


def test_a_detached_registry_switches_to_the_requested_graph_mode():
    """The prior live mode cannot leak into the next worker's palette."""
    comparison = AnalysisWorkspaceMode.GROUP_COMPARISON
    activity = AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS
    registry = ColorDistributionRegistry(
        configurations={
            comparison: {
                "up_color": "#AA0000",
                "down_color": "#0000AA",
                "neutral_color": "#AAAAAA",
            },
            activity: {
                "up_color": "#00AA00",
                "down_color": "#AA00AA",
                "neutral_color": "#00AAAA",
            },
        },
        current_mode=comparison,
    )

    calculation_input = GraphCalculationInput.from_plain_values(
        metabolic_model=None,
        mode=activity,
        color_registry=registry,
    )

    assert calculation_input.color_manager.current_mode is activity
    assert calculation_input.color_manager.up_color == "#00AA00"


# --- Import purity -------------------------------------------------------


def test_the_calculation_input_imports_and_builds_without_bokeh():
    """Every collaborator the input assembles must be reachable without Bokeh."""
    result = run_without_bokeh(
        """
        import pandas as pd

        from metabis.viz.metabolic_visualization.calculation_input import (
            GraphCalculationInput,
        )
        from metabis.managers.workspace_mode import AnalysisWorkspaceMode
        from metabis.managers.color_distributions import ColorDistributionRegistry

        mode = AnalysisWorkspaceMode.GROUP_COMPARISON
        calculation_input = GraphCalculationInput.from_plain_values(
            metabolic_model=None,
            mode=mode,
            color_registry=ColorDistributionRegistry(
                configurations={
                    mode: {
                        "max_z_score": 6,
                        "up_color": "#D62728",
                        "down_color": "#1F77B4",
                        "neutral_color": "#F0F0F0",
                    }
                },
                current_mode=mode,
            ),
            group_memberships=[("A", ["s1", "s2"])],
            all_sample_ids=["s1", "s2"],
            active_group="A",
        )
        calculation_input.color_manager.register_distribution(
            mode=mode,
            element_type="reaction",
            data=pd.Series({"MAR00001": -3.0, "MAR00002": 3.0}),
            metric_name="log2_fc",
        )
        print(
            calculation_input.get_element_color("MAR00002", "reaction", "log2_fc"),
            calculation_input.get_active_group(),
            calculation_input.get_mode().value,
        )
        """
    )

    assert result.returncode == 0, result.stderr
    color, active_group, mode = result.stdout.strip().splitlines()[-1].split()
    assert color.startswith("#")
    assert active_group == "A"
    assert mode == "group_comparison"

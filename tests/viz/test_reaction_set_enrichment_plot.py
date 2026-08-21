"""The enrichment plot is a rendering adapter over one completed artifact."""

import math
from dataclasses import replace

import pandas as pd
from bokeh.models import HoverTool

from metabis.analysis.artifacts import (
    AnalysisArtifactMetadata,
    ArtifactDimension,
    ArtifactKind,
    ArtifactProvenance,
    PlainTable,
)
from metabis.analysis.comparison_artifact import (
    ComparisonThresholds,
    GroupComparisonArtifact,
    MetricSummary,
)
from metabis.analysis.comparison_membership import (
    ComparisonMembership,
    GroupComparisonSpec,
)
from metabis.analysis.reaction_set_enrichment import (
    ReactionSetEnrichmentConfig,
    SubsystemAnnotation,
    run_reaction_set_enrichment,
)
from metabis.analysis_workspace import RevisionKind, WorkspaceRevisions
from metabis.viz.plot_reaction_set_enrichment import plot_reaction_set_enrichment


def _artifact(significant=("R1", "R2"), *, minimum_size=1):
    membership = ComparisonMembership.resolve(
        GroupComparisonSpec.pairwise(("case",), ("control",)),
        group_samples={"case": ("c1", "c2"), "control": ("k1", "k2")},
        aligned_sample_ids=("c1", "c2", "k1", "k2"),
    )
    frame = pd.DataFrame(
        {
            "cohens_d": [2.0, 1.0, -1.0, -2.0],
            "log2_fc": [1.0, 0.5, -0.5, -1.0],
            "mean_difference": [3.0, 2.0, -2.0, -3.0],
            "group_1_mean": [5.0, 4.0, 2.0, 1.0],
            "group_2_mean": [2.0, 2.0, 4.0, 5.0],
            "is_significant": [value in significant for value in ("R1", "R2", "R3", "R4")],
        },
        index=["R1", "R2", "R3", "R4"],
    )
    metadata = AnalysisArtifactMetadata(
        kind=ArtifactKind.GROUP_COMPARISON,
        source=ArtifactProvenance(
            workspace_id="workspace",
            revisions=WorkspaceRevisions(dataset=1, groups=1, thresholds=1),
            dependencies=frozenset(
                {RevisionKind.DATASET, RevisionKind.GROUPS, RevisionKind.THRESHOLDS}
            ),
            source_fingerprints={"dataset": "sha256:dataset"},
        ),
        method="ttest",
        configuration={"correction": "fdr_bh", "keep_reversed": False},
        scientific_identity={
            "membership_fingerprint": membership.fingerprint,
            "mode": membership.mode.value,
            "target_base_group_ids": membership.target_base_group_ids,
            "baseline_base_group_ids": membership.baseline_base_group_ids,
            "target_samples": membership.target_samples,
            "baseline_samples": membership.baseline_samples,
            "universe": membership.universe,
            "excluded_base_group_ids": membership.excluded_base_group_ids,
        },
        dimensions=(
            ArtifactDimension("samples", len(membership.universe)),
            ArtifactDimension("target_samples", len(membership.target_samples)),
            ArtifactDimension("baseline_samples", len(membership.baseline_samples)),
            ArtifactDimension("reactions", 4),
            ArtifactDimension("genes", 0),
        ),
    )
    comparison = GroupComparisonArtifact(
        metadata=metadata,
        membership=membership,
        test="ttest",
        correction="fdr_bh",
        reaction_results=PlainTable.capture(frame),
        gene_results=None,
        reaction_metrics=(MetricSummary("cohens_d", 1.0, 0.0),),
        gene_metrics=(),
        significant_reactions=tuple(significant),
        significant_genes=(),
        reaction_thresholds=ComparisonThresholds(0.05, None, None, None),
        gene_thresholds=None,
    )
    return run_reaction_set_enrichment(
        comparison,
        (
            SubsystemAnnotation("s1", "First set", ("R1", "R2", "R3"), "core"),
            SubsystemAnnotation("s2", "Second set", ("R2", "R3", "R4"), "other"),
        ),
        ReactionSetEnrichmentConfig(minimum_measured_set_size=minimum_size),
    )


def test_plot_consumes_exact_artifact_rows_without_running_analysis(monkeypatch):
    artifact = _artifact()
    monkeypatch.setattr(
        "metabis.analysis.reaction_set_enrichment.run_reaction_set_enrichment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("plot recomputed enrichment")
        ),
    )

    plot = plot_reaction_set_enrichment(artifact)
    renderer = plot.select_one({"name": "reaction_set_enrichment_dots"})
    rendered = renderer.data_source.data
    expected = artifact.result_frame().set_index("subsystem_id")
    actual = pd.DataFrame(rendered).set_index("subsystem_id")

    for column_name in artifact.result_frame().columns.drop("subsystem_id"):
        assert actual.loc[expected.index, column_name].tolist() == expected[column_name].tolist()
    assert plot.tags[-1]["artifact_id"] == artifact.metadata.artifact_id
    assert renderer.glyph.size == "point_size"


def test_empty_query_artifact_remains_renderable_and_auditable():
    artifact = _artifact(significant=())

    plot = plot_reaction_set_enrichment(artifact)
    renderer = plot.select_one({"name": "reaction_set_enrichment_dots"})

    assert len(renderer.data_source.data["subsystem_id"]) == 2
    assert set(renderer.data_source.data["query_size"]) == {0}
    assert plot.tags[-1]["artifact_id"] == artifact.metadata.artifact_id


def test_plot_uses_readable_labels_and_a_bounded_scientific_hover():
    plot = plot_reaction_set_enrichment(_artifact())
    renderer = plot.select_one({"name": "reaction_set_enrichment_dots"})
    hover = plot.select_one(HoverTool)
    hover_html = renderer.data_source.data["hover_html"][0]

    assert set(plot.y_range.factors) == {"First set", "Second set"}
    assert not any(label.startswith("set-") for label in plot.y_range.factors)
    assert hover.tooltips == "@hover_html{safe}"
    assert hover.attachment == "vertical"
    assert "width:340px;max-width:340px" in hover_html
    assert "Adjusted p-value" in hover_html
    assert "Enrichment ratio" in hover_html
    assert "Subsystem overlap" in hover_html
    assert "Analysis query" in hover_html
    assert "Reaction IDs" not in hover_html
    assert "overlap_reaction_ids" not in hover_html
    assert max(renderer.data_source.data["point_size"]) <= 25.0


def test_plot_rejects_non_artifact_inputs():
    try:
        plot_reaction_set_enrichment(pd.DataFrame())
    except TypeError as exc:
        assert "ReactionSetEnrichmentArtifact" in str(exc)
    else:
        raise AssertionError("non-artifact input was accepted")


def test_no_eligible_rows_and_missing_adjusted_pvalues_are_structural():
    empty_artifact = _artifact(minimum_size=99)
    empty_plot = plot_reaction_set_enrichment(empty_artifact)
    empty_renderer = empty_plot.select_one(
        {"name": "reaction_set_enrichment_dots"}
    )
    assert len(empty_renderer.data_source.data["subsystem_id"]) == 0

    artifact = _artifact()
    frame = artifact.result_frame()
    frame.loc[0, "adjusted_pval"] = math.nan
    missing_artifact = replace(artifact, results=PlainTable.capture(frame))
    plot = plot_reaction_set_enrichment(missing_artifact)
    renderer = plot.select_one({"name": "reaction_set_enrichment_dots"})
    assert any(math.isnan(value) for value in renderer.data_source.data["neg_log10_adjusted_pval"])

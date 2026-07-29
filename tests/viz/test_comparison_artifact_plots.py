from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
from bokeh.models import Span

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
from metabis.analysis_workspace import RevisionKind, WorkspaceRevisions
from metabis.viz.plot_comparison_artifact import (
    plot_comparison_histogram,
    plot_comparison_scatter,
)


def _artifact():
    membership = ComparisonMembership.resolve(
        GroupComparisonSpec.pairwise(("case",), ("control",)),
        group_samples={"case": ("c1", "c2"), "control": ("k1", "k2")},
        aligned_sample_ids=("c1", "c2", "k1", "k2"),
    )
    frame = pd.DataFrame(
        {
            "pval": [0.001, 0.02, 0.4],
            "adjusted_pval": [0.003, 0.03, 0.4],
            "neg_log_pval": [3.0, 1.7, 0.4],
            "neg_log_adjusted_pval": [2.522878745, 1.522878745, 0.397940009],
            "cohens_d": [2.0, -1.0, 0.1],
            "log2_fc": [1.5, -0.75, 0.05],
            "mean_difference": [4.0, -2.0, 0.2],
            "is_significant": [True, True, False],
        },
        index=["R1", "R2", "R3"],
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
            ArtifactDimension("samples", 4),
            ArtifactDimension("target_samples", 2),
            ArtifactDimension("baseline_samples", 2),
            ArtifactDimension("reactions", 3),
            ArtifactDimension("genes", 0),
        ),
    )
    return GroupComparisonArtifact(
        metadata=metadata,
        membership=membership,
        test="ttest",
        correction="fdr_bh",
        reaction_results=PlainTable.capture(frame),
        gene_results=None,
        reaction_metrics=(MetricSummary("cohens_d", 1.0, 0.0),),
        gene_metrics=(),
        significant_reactions=("R1", "R2"),
        significant_genes=(),
        reaction_thresholds=ComparisonThresholds(0.05, 0.5, 0.5, 1.0),
        gene_thresholds=None,
        reaction_display_labels=(("R1", "First"), ("R2", "Second"), ("R3", "Third")),
    )


def test_histogram_uses_stored_values_and_artifact_threshold_only():
    artifact = _artifact()
    plot = plot_comparison_histogram(
        artifact, "reaction", "cohens_d", show_threshold=True, bins=3
    )
    source = plot.renderers[0].data_source.data

    assert sum(source["top"]) == 3
    assert plot.tags[-1]["artifact_id"] == artifact.metadata.artifact_id
    assert plot.tags[-1]["metric"] == "cohens_d"
    assert sorted(span.location for span in plot.select({"type": Span})) == [-0.5, 0.5]


def test_scatter_preserves_canonical_rows_labels_and_significance():
    artifact = _artifact()
    plot = plot_comparison_scatter(
        artifact, "reaction", "cohens_d", "pval", show_thresholds=False
    )
    source = plot.select_one({"name": "comparison_scatter_points"}).data_source.data

    assert list(source["feature_id"]) == ["R1", "R2", "R3"]
    assert list(source["display_label"]) == ["First", "Second", "Third"]
    assert list(source["x"]) == [2.0, -1.0, 0.1]
    assert list(source["y"]) == [2.522878745, 1.522878745, 0.397940009]
    assert list(source["is_significant"]) == [True, True, False]
    assert plot.tags[-1]["artifact_id"] == artifact.metadata.artifact_id


def test_factories_have_no_manager_or_server_dependency():
    path = Path(__file__).parents[2] / "src/metabis/viz/plot_comparison_artifact.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    assert not any(name.startswith("bokeh_server") for name in imports)
    assert "AnalysisWorkspace" not in names


def test_empty_artifact_remains_renderable():
    artifact = _artifact()
    empty = artifact.with_threshold_results(
        revisions=WorkspaceRevisions(dataset=1, groups=1, thresholds=2),
        reaction_results=artifact.results_frame("reaction").iloc[0:0],
        reaction_thresholds=artifact.reaction_thresholds,
        significant_reactions=(),
        gene_results=None,
        gene_thresholds=None,
        significant_genes=(),
        keep_reversed=False,
    )

    histogram = plot_comparison_histogram(empty, "reaction", "pval")
    scatter = plot_comparison_scatter(empty, "reaction", "cohens_d", "pval")

    assert len(histogram.renderers[0].data_source.data["top"]) == 0
    assert len(scatter.renderers[0].data_source.data["feature_id"]) == 0

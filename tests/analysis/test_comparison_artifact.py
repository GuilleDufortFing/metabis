from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

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
from metabis.analysis.group_exports import (
    COMPARISON_MEMBERSHIP_EXPORT_COLUMNS,
    comparison_membership_frame,
)
from metabis.analysis_workspace import RevisionKind, WorkspaceRevisions

RESULT_COLUMNS = (
    "test_stat",
    "pval",
    "group_1_mean",
    "group_2_mean",
    "n_group_1",
    "n_group_2",
    "cohens_d",
    "log2_fc",
    "mean_difference",
    "reversed",
    "calculation_status",
    "reason",
    "adjusted_pval",
    "neg_log_pval",
    "neg_log_adjusted_pval",
    "group_1",
    "group_2",
    "reaction_id",
    "subsystems",
    "is_significant",
)


def _membership() -> ComparisonMembership:
    return ComparisonMembership.resolve(
        GroupComparisonSpec.pairwise(("canonical-a",), ("canonical-b",)),
        group_samples={"canonical-a": ("s1", "s2"), "canonical-b": ("s3", "s4")},
        aligned_sample_ids=("s1", "s2", "s3", "s4", "unused"),
        labels={"canonical-a": "Cases", "canonical-b": "Controls"},
    )


def _metadata(membership: ComparisonMembership, reactions: int) -> AnalysisArtifactMetadata:
    return AnalysisArtifactMetadata(
        kind=ArtifactKind.GROUP_COMPARISON,
        source=ArtifactProvenance(
            workspace_id="workspace",
            revisions=WorkspaceRevisions(dataset=1, groups=2),
            dependencies=frozenset({RevisionKind.DATASET, RevisionKind.GROUPS}),
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
            ArtifactDimension("reactions", reactions),
            ArtifactDimension("genes", 0),
        ),
    )


def _artifact(frame: pd.DataFrame) -> GroupComparisonArtifact:
    membership = _membership()
    return GroupComparisonArtifact(
        metadata=_metadata(membership, len(frame)),
        membership=membership,
        test="ttest",
        correction="fdr_bh",
        reaction_results=PlainTable.capture(frame),
        gene_results=None,
        reaction_metrics=(MetricSummary("cohens_d", 0.5, 0.25),),
        gene_metrics=(),
        significant_reactions=tuple(
            sorted(map(str, frame.index[frame.get("is_significant", False)]))
        ),
        significant_genes=(),
        reaction_thresholds=ComparisonThresholds(0.05, 0.1, 0.2, 0.3),
        gene_thresholds=None,
        reaction_display_labels=tuple(
            sorted((str(feature_id), f"Label for {feature_id}") for feature_id in frame.index)
        ),
    )


def test_export_owns_canonical_ids_full_statistics_and_unformatted_precision():
    precise = 0.12345678901234566
    frame = pd.DataFrame(
        [[4.5, precise, 8.0, 3.0, 2.0, 2.0, 1.25, -0.75, -5.0, False,
          "ok", "", precise * 2, 0.9, 0.8, "Cases", "Controls", "wrong-id",
          "Glycolysis", True]],
        index=pd.Index(["R_CANONICAL"], name="reaction"),
        columns=RESULT_COLUMNS,
    )
    artifact = _artifact(frame)

    exported = artifact.export_frame("reaction")

    assert exported.loc[0, "feature_id"] == "R_CANONICAL"
    assert exported.loc[0, "display_label"] == "Label for R_CANONICAL"
    assert "reaction_id" not in exported
    assert exported.loc[0, "pval"] == precise
    assert exported.loc[0, "target_mean"] == 8.0
    assert exported.loc[0, "n_target"] == 2.0
    assert exported.loc[0, "baseline_mean"] == 3.0
    assert exported.loc[0, "n_baseline"] == 2.0
    assert exported.loc[0, "target_label"] == "Cases"
    assert exported.loc[0, "baseline_label"] == "Controls"
    assert exported.loc[0, "effect_direction"] == "target_minus_baseline"
    assert json.loads(exported.loc[0, "target_base_group_ids"]) == ["canonical-a"]
    assert json.loads(exported.loc[0, "baseline_base_group_ids"]) == ["canonical-b"]
    assert "group_1" not in exported
    assert "group_2" not in exported
    assert exported.loc[0, "calculation_status"] == "ok"
    assert exported.loc[0, "reason"] == ""
    assert exported.loc[0, "test"] == "ttest"
    assert exported.loc[0, "correction"] == "fdr_bh"
    assert exported.loc[0, "threshold_pval"] == 0.05
    assert exported.loc[0, "is_significant"] is True or bool(exported.loc[0, "is_significant"])
    assert exported.loc[0, "membership_fingerprint"] == artifact.membership.fingerprint


def test_empty_result_is_valid_and_keeps_canonical_export_headers():
    artifact = _artifact(pd.DataFrame(columns=RESULT_COLUMNS))

    exported = artifact.export_frame("reaction")

    assert exported.empty
    assert tuple(exported.columns[:3]) == ("feature_id", "display_label", "feature_type")
    assert {"target_mean", "baseline_mean", "calculation_status", "reason"} <= set(exported)
    assert {"target_label", "baseline_label", "effect_direction"} <= set(exported)
    assert {"test", "correction", "membership_fingerprint"} <= set(exported)


def test_membership_adapter_reproduces_exact_sides_and_aligned_universe():
    artifact = _artifact(pd.DataFrame(columns=RESULT_COLUMNS))

    membership = comparison_membership_frame(artifact.membership)

    assert tuple(membership.columns) == COMPARISON_MEMBERSHIP_EXPORT_COLUMNS
    assert membership.query("side == 'target'")["sample_id"].tolist() == ["s1", "s2"]
    assert membership.query("side == 'baseline'")["sample_id"].tolist() == ["s3", "s4"]
    assert json.loads(membership.query("side == 'target'")["base_group_ids"].iloc[0]) == [
        "canonical-a"
    ]
    assert json.loads(membership.query("side == 'baseline'")["base_group_ids"].iloc[0]) == [
        "canonical-b"
    ]
    assert membership["comparison_fingerprint"].unique().tolist() == [
        artifact.membership.fingerprint
    ]


def test_keep_reversed_is_explicit_and_changes_artifact_and_export_identity():
    frame = pd.DataFrame(
        [[1.0, 0.1, 1.0, -1.0, 2.0, 2.0, 1.0, 0.0, 2.0, True,
          "ok", "", 0.2, 1.0, 1.0, "Cases", "Controls", "R1", "", False]],
        index=["R1"],
        columns=RESULT_COLUMNS,
    )
    baseline = _artifact(frame)
    kept_frame = frame.copy()
    kept_frame["is_significant"] = True

    kept = baseline.with_threshold_results(
        revisions=baseline.metadata.source.revisions,
        reaction_results=kept_frame,
        reaction_thresholds=baseline.reaction_thresholds,
        significant_reactions=("R1",),
        gene_results=None,
        gene_thresholds=None,
        significant_genes=(),
        keep_reversed=True,
    )

    assert kept.keep_reversed is True
    assert kept.metadata.artifact_id != baseline.metadata.artifact_id
    assert kept.export_frame("reaction")["keep_reversed"].tolist() == [True]
    assert kept.export_frame("reaction")["is_significant"].tolist() == [True]


def test_comparison_rejects_tampered_membership_identity_and_dimensions():
    artifact = _artifact(pd.DataFrame(columns=RESULT_COLUMNS))
    tampered_identity = dict(artifact.metadata.scientific_identity)
    tampered_identity["universe"] = ["different"]
    tampered_metadata = replace(
        artifact.metadata, scientific_identity=tampered_identity
    )

    with pytest.raises(ValueError, match="universe"):
        replace(artifact, metadata=tampered_metadata)

    dimensions = tuple(
        ArtifactDimension(item.name, item.size + (1 if item.name == "reactions" else 0))
        for item in artifact.metadata.dimensions
    )
    with pytest.raises(ValueError, match="dimensions"):
        replace(artifact, metadata=replace(artifact.metadata, dimensions=dimensions))


def test_artifact_and_export_modules_do_not_import_bokeh_or_live_managers():
    package = Path(__file__).parents[2] / "src" / "metabis" / "analysis"
    for filename in ("artifacts.py", "comparison_artifact.py", "dimensionality_reduction.py"):
        tree = ast.parse((package / filename).read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(name == "bokeh" or name.startswith("bokeh.") for name in imports)
        assert not any("managers" in name for name in imports)

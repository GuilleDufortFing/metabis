from dataclasses import replace

import pandas as pd
import pytest

from metabis.analysis.activity_analysis import (
    ACTIVITY_FEATURE_EXPORT_COLUMNS,
    ACTIVITY_SUBSYSTEM_EXPORT_COLUMNS,
    SUBSYSTEM_ACTIVITY_COLUMNS,
    ActivityArtifact,
    ActivityConfig,
    ActivityContext,
    FeatureActivityResult,
)
from metabis.analysis.artifacts import (
    AnalysisArtifactMetadata,
    ArtifactDimension,
    ArtifactKind,
    ArtifactProvenance,
)
from metabis.analysis_workspace import RevisionKind, WorkspaceRevisions


def _metadata(*, warnings=("R1 has a non-finite group summary",)):
    return AnalysisArtifactMetadata(
        kind=ArtifactKind.ACTIVITY,
        source=ArtifactProvenance(
            workspace_id="workspace-1",
            revisions=WorkspaceRevisions(dataset=2, groups=3),
            dependencies=frozenset(
                {
                    RevisionKind.DATASET,
                    RevisionKind.GROUPS,
                    RevisionKind.PREPROCESSING,
                }
            ),
            source_fingerprints={"dataset": "sha256:activity-input"},
        ),
        method="single_group_normalized_activity",
        configuration={
            "summary": "mean",
            "minimum_subsystem_coverage": 0.1,
        },
        scientific_identity={
            "group_id": "group-a",
            "selected_sample_ids": ["sample-2", "sample-1"],
            "reference_sample_ids": ["sample-1", "outside", "sample-2"],
            "available_families": ["reaction"],
        },
        dimensions=(
            ArtifactDimension("reaction_features", 2),
            ArtifactDimension("reaction_subsystems", 1),
            ArtifactDimension("gene_features", 0),
            ArtifactDimension("gene_subsystems", 0),
            ArtifactDimension("selected_samples", 2),
            ArtifactDimension("reference_samples", 3),
        ),
        warnings=warnings,
    )


def _context(*, empty_subsystems=False):
    config = ActivityConfig("group-a")
    features = FeatureActivityResult(
        group_id="group-a",
        overlay="reaction",
        activity_score=pd.Series([5.0, float("nan")], index=["R2", "R1"]),
        sample_count=2,
        reference_sample_count=3,
        config=config,
        non_finite_feature_ids=("R1",),
        warnings=("R1 has a non-finite group summary",),
    )
    subsystems = pd.DataFrame(columns=list(SUBSYSTEM_ACTIVITY_COLUMNS))
    if not empty_subsystems:
        subsystems.loc[0] = ["glycolysis", 0.75, 4, 3, 0.75, 2, 1]
        subsystems = subsystems.set_index("subsystem", drop=False)
    return ActivityContext(
        config=config,
        reaction_features=features,
        reaction_subsystems=subsystems,
    )


def _artifact(*, empty_subsystems=False):
    return ActivityArtifact.from_context(
        _context(empty_subsystems=empty_subsystems),
        metadata=_metadata(warnings=("R1 has a non-finite group summary",)),
        selected_sample_ids=("sample-2", "sample-1"),
        reference_sample_ids=("sample-1", "outside", "sample-2"),
    )


def test_activity_artifact_carries_exact_samples_config_results_and_warnings():
    artifact = _artifact()

    assert artifact.metadata.kind is ArtifactKind.ACTIVITY
    assert artifact.selected_sample_ids == ("sample-2", "sample-1")
    assert artifact.reference_sample_ids == ("sample-1", "outside", "sample-2")
    assert artifact.config == artifact.features.config
    assert artifact.warnings == ("R1 has a non-finite group summary",)
    assert (
        artifact.subsystem_table.to_frame().loc["glycolysis", "activity_score"]
        == 0.75
    )


def test_activity_artifact_captures_the_subsystem_table_immutably():
    context = _context()
    artifact = ActivityArtifact.from_context(
        context,
        metadata=_metadata(),
        selected_sample_ids=("sample-2", "sample-1"),
        reference_sample_ids=("sample-1", "outside", "sample-2"),
    )

    context.reaction_subsystems.loc["glycolysis", "activity_score"] = 99.0
    context.reaction_features.activity_score.loc["R2"] = 99.0

    assert artifact.subsystem_frame().loc[0, "activity_score"] == 0.75
    assert artifact.feature_frame().loc[0, "activity_score"] == 5.0


def test_activity_feature_export_is_canonical_and_audits_non_finite_summaries():
    artifact = _artifact()

    frame = artifact.feature_frame()

    assert tuple(frame.columns) == ACTIVITY_FEATURE_EXPORT_COLUMNS
    assert frame["artifact_id"].tolist() == [
        artifact.metadata.artifact_id,
        artifact.metadata.artifact_id,
    ]
    assert frame["feature_id"].tolist() == ["R2", "R1"]
    assert frame.loc[0, "activity_score"] == 5.0
    assert pd.isna(frame.loc[1, "activity_score"])
    assert frame["is_non_finite"].tolist() == [False, True]


def test_activity_subsystem_export_is_canonical():
    artifact = _artifact()

    frame = artifact.subsystem_frame()

    assert tuple(frame.columns) == ACTIVITY_SUBSYSTEM_EXPORT_COLUMNS
    assert frame.loc[0, "artifact_id"] == artifact.metadata.artifact_id
    assert frame.loc[0, "group_id"] == "group-a"
    assert frame.loc[0, "subsystem"] == "glycolysis"
    assert frame.loc[0, "measured_features"] == 3


def test_empty_activity_exports_keep_stable_headers():
    context = _context(empty_subsystems=True)
    empty_features = replace(
        context.features,
        activity_score=pd.Series(dtype=float),
        non_finite_feature_ids=(),
    )
    context = replace(context, reaction_features=empty_features)
    metadata = AnalysisArtifactMetadata(
        kind=ArtifactKind.ACTIVITY,
        source=_metadata().source,
        method="single_group_normalized_activity",
        configuration={
            "summary": "mean",
            "minimum_subsystem_coverage": 0.1,
        },
        scientific_identity={
            "group_id": "group-a",
            "selected_sample_ids": ["sample-2", "sample-1"],
            "reference_sample_ids": ["sample-1", "outside", "sample-2"],
            "available_families": ["reaction"],
        },
        dimensions=(
            ArtifactDimension("reaction_features", 0),
            ArtifactDimension("reaction_subsystems", 0),
            ArtifactDimension("gene_features", 0),
            ArtifactDimension("gene_subsystems", 0),
            ArtifactDimension("selected_samples", 2),
            ArtifactDimension("reference_samples", 3),
        ),
        warnings=("R1 has a non-finite group summary",),
    )
    artifact = ActivityArtifact.from_context(
        context,
        metadata=metadata,
        selected_sample_ids=("sample-2", "sample-1"),
        reference_sample_ids=("sample-1", "outside", "sample-2"),
    )

    assert artifact.feature_frame().empty
    assert tuple(artifact.feature_frame().columns) == ACTIVITY_FEATURE_EXPORT_COLUMNS
    assert artifact.subsystem_frame().empty
    assert (
        tuple(artifact.subsystem_frame().columns)
        == ACTIVITY_SUBSYSTEM_EXPORT_COLUMNS
    )


def test_activity_artifact_rejects_membership_that_does_not_match_calculation():
    with pytest.raises(ValueError, match="selected samples"):
        ActivityArtifact.from_context(
            _context(),
            metadata=_metadata(),
            selected_sample_ids=("sample-1",),
            reference_sample_ids=("sample-1", "outside", "sample-2"),
        )


def test_activity_artifact_rejects_metadata_dimensions_that_diverge_from_payload():
    metadata = replace(
        _metadata(),
        dimensions=(
            ArtifactDimension("reaction_features", 99),
            ArtifactDimension("reaction_subsystems", 1),
            ArtifactDimension("gene_features", 0),
            ArtifactDimension("gene_subsystems", 0),
            ArtifactDimension("selected_samples", 2),
            ArtifactDimension("reference_samples", 3),
        ),
    )

    with pytest.raises(ValueError, match="dimensions"):
        ActivityArtifact.from_context(
            _context(),
            metadata=metadata,
            selected_sample_ids=("sample-2", "sample-1"),
            reference_sample_ids=("sample-1", "outside", "sample-2"),
        )

"""Portable common metadata for typed analysis artifacts."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pandas as pd
import pytest

import metabis
from metabis.analysis.artifacts import (
    AnalysisArtifactMetadata,
    ArtifactDimension,
    ArtifactKind,
    ArtifactProvenance,
    PlainTable,
    fingerprint_frame,
    fingerprint_json,
)
from metabis.analysis_workspace import (
    RevisionKind,
    StableFrameReference,
    WorkspaceProvenance,
    WorkspaceRevisions,
)


def _workspace_provenance(
    *,
    dependencies: frozenset[RevisionKind] = frozenset(RevisionKind),
    revisions: WorkspaceRevisions | None = None,
    workspace_id: str = "workspace-17",
) -> WorkspaceProvenance:
    reaction_frame = pd.DataFrame({"R1": [1.0]}, index=["sample-1"])
    return WorkspaceProvenance(
        workspace_id=workspace_id,
        revisions=revisions
        or WorkspaceRevisions(
            dataset=1,
            metadata=2,
            groups=3,
            preprocessing=4,
            thresholds=5,
            visualization=6,
        ),
        dependencies=dependencies,
        reaction_frame=StableFrameReference.capture(reaction_frame),
        expression_matrix=None,
        group_memberships=(("group-a", ("sample-1",)),),
    )


def _metadata(**overrides) -> AnalysisArtifactMetadata:
    values = {
        "kind": ArtifactKind.GROUP_COMPARISON,
        "provenance": _workspace_provenance(),
        "source_fingerprints": {
            "dataset": "sha256:dataset-17",
            "metadata": "sha256:metadata-4",
            "model": "human1:1.19.0",
            "preprocessing": "compass-negative-log:v1",
        },
        "method": "welch_t_test",
        "configuration": {
            "alternative": "two-sided",
            "correction": {"method": "fdr_bh", "alpha": 0.05},
            "features": ["R1", "R2"],
        },
        "scientific_identity": {
            "comparison_membership": "sha256:membership-a-vs-b",
            "sample_universe": "sha256:aligned-samples-17",
        },
        "dimensions": (
            ArtifactDimension("samples", 42, label="Samples"),
            ArtifactDimension("reactions", 2, label="Reactions"),
        ),
        "warnings": ("R9 was not mapped",),
        "software_version": "7.8.9",
    }
    values.update(overrides)
    return AnalysisArtifactMetadata.from_workspace(**values)


def test_canonical_fingerprints_cover_json_and_exact_frames_without_mutation():
    assert fingerprint_json({"b": [2, 3], "a": 1}) == fingerprint_json(
        {"a": 1, "b": [2, 3]}
    )

    frame = pd.DataFrame(
        {
            "dense": [1.0, 2.0, 3.0],
            "sparse": pd.arrays.SparseArray([0.0, 4.0, 0.0]),
        },
        index=pd.Index(["s1", "s2", "s3"], name="sample_id"),
    )
    before = frame.copy(deep=True)

    baseline = fingerprint_frame(frame)
    changed_value = frame.copy()
    changed_value.loc["s2", "dense"] = 2.000000000000001
    changed_index = frame.rename(index={"s3": "different"})

    assert baseline.startswith("sha256:")
    assert fingerprint_frame(frame) == baseline
    assert fingerprint_frame(changed_value) != baseline
    assert fingerprint_frame(changed_index) != baseline
    # Fingerprinting reads but never mutates the input (a sparse column is
    # densified on a private copy, one column at a time, not in place).
    pd.testing.assert_frame_equal(frame, before)

    # Storage invariance (M-08): the sparse column fingerprints identically to its
    # dense twin -- the invariant that matters, stronger than "never densify".
    dense_twin = frame.copy()
    dense_twin["sparse"] = frame["sparse"].sparse.to_dense()
    assert fingerprint_frame(dense_twin) == baseline


def test_plain_table_preserves_valid_empty_headers_and_full_precision():
    frame = pd.DataFrame(
        columns=["feature_id", "p_value"],
        index=pd.Index([], name="row"),
    )
    table = PlainTable.capture(frame)

    assert table.columns == ("feature_id", "p_value")
    pd.testing.assert_frame_equal(table.to_frame(), frame, check_dtype=False)

    precise = pd.DataFrame({"value": [0.12345678901234566]}, index=["R1"])
    assert PlainTable.capture(precise).to_frame().loc["R1", "value"] == precise.loc[
        "R1", "value"
    ]


def test_plain_table_rejects_mutable_object_cells_instead_of_aliasing_them():
    mutable = ["subsystem-a"]

    with pytest.raises(TypeError, match="immutable scalar or tuple cells"):
        PlainTable.capture(pd.DataFrame({"value": [mutable]}))


def test_workspace_projection_keeps_all_revisions_and_dependencies_but_no_live_state():
    workspace = _workspace_provenance()

    source = ArtifactProvenance.from_workspace(
        workspace,
        source_fingerprints={
            "dataset": "sha256:dataset-17",
            "metadata": "sha256:metadata-4",
        },
    )

    assert source.workspace_id == "workspace-17"
    assert source.revisions == WorkspaceRevisions(
        dataset=1,
        metadata=2,
        groups=3,
        preprocessing=4,
        thresholds=5,
        visualization=6,
    )
    assert source.dependencies == frozenset(RevisionKind)
    assert source.source_fingerprints == {
        "dataset": "sha256:dataset-17",
        "metadata": "sha256:metadata-4",
    }
    assert not hasattr(source, "reaction_frame")
    assert not hasattr(source, "expression_matrix")
    assert not hasattr(source, "group_memberships")
    with pytest.raises(TypeError):
        source.source_fingerprints["dataset"] = "replacement"


def test_portable_provenance_requires_an_explicit_canonical_dataset_fingerprint():
    workspace = _workspace_provenance()

    with pytest.raises(TypeError, match="source_fingerprints"):
        ArtifactProvenance.from_workspace(workspace)
    with pytest.raises(ValueError, match="dataset fingerprint"):
        ArtifactProvenance.from_workspace(
            workspace,
            source_fingerprints={"metadata": "sha256:metadata-4"},
        )
    with pytest.raises(ValueError, match="canonical strings"):
        ArtifactProvenance.from_workspace(
            workspace,
            source_fingerprints={"dataset": "  "},
        )


def test_source_fingerprints_not_workspace_uuid_supply_portable_input_identity():
    first = _metadata()
    same_inputs_new_workspace = _metadata(
        provenance=_workspace_provenance(workspace_id="workspace-99")
    )
    changed_dataset = _metadata(
        source_fingerprints={
            "dataset": "sha256:different-dataset",
            "metadata": "sha256:metadata-4",
            "model": "human1:1.19.0",
            "preprocessing": "compass-negative-log:v1",
        }
    )

    assert same_inputs_new_workspace.artifact_id == first.artifact_id
    assert same_inputs_new_workspace.source.workspace_id != first.source.workspace_id
    assert changed_dataset.artifact_id != first.artifact_id


@pytest.mark.parametrize("dependency", list(RevisionKind))
def test_each_workspace_revision_dependency_participates_in_scientific_identity(
    dependency,
):
    revisions = WorkspaceRevisions()
    changed = replace(revisions, **{dependency.value: 1})
    baseline = _metadata(
        provenance=_workspace_provenance(
            dependencies=frozenset({dependency}), revisions=revisions
        )
    )
    revised = _metadata(
        provenance=_workspace_provenance(
            dependencies=frozenset({dependency}), revisions=changed
        )
    )

    assert revised.artifact_id != baseline.artifact_id


def test_undeclared_workspace_revisions_do_not_change_scientific_identity():
    dependencies = frozenset({RevisionKind.DATASET})
    baseline = _metadata(
        provenance=_workspace_provenance(
            dependencies=dependencies,
            revisions=WorkspaceRevisions(dataset=3, visualization=1),
        )
    )
    presentation_changed = _metadata(
        provenance=_workspace_provenance(
            dependencies=dependencies,
            revisions=WorkspaceRevisions(dataset=3, visualization=99),
        )
    )

    assert presentation_changed.artifact_id == baseline.artifact_id
    assert presentation_changed.source != baseline.source


def test_configuration_is_deeply_immutable_and_mapping_order_is_canonical():
    input_configuration = {
        "z": [{"beta": 2, "alpha": [1, 2]}],
        "a": {"enabled": True},
    }
    metadata = _metadata(configuration=input_configuration)
    reordered = _metadata(
        configuration={
            "a": {"enabled": True},
            "z": [{"alpha": [1, 2], "beta": 2}],
        }
    )

    input_configuration["z"][0]["alpha"].append(3)
    input_configuration["a"]["enabled"] = False

    assert tuple(metadata.configuration) == ("a", "z")
    assert metadata.configuration["z"][0]["alpha"] == (1, 2)
    assert metadata.artifact_id == reordered.artifact_id
    with pytest.raises(TypeError):
        metadata.configuration["new"] = "value"
    with pytest.raises(TypeError):
        metadata.configuration["z"][0]["alpha"][0] = 99


def test_result_specific_identity_is_immutable_and_prevents_collision():
    first = _metadata(
        scientific_identity={
            "membership": {"fingerprint": "sha256:a-vs-b", "universe": ["A", "B"]}
        }
    )
    same_dimensions_different_membership = _metadata(
        scientific_identity={
            "membership": {"fingerprint": "sha256:c-vs-d", "universe": ["C", "D"]}
        }
    )

    assert first.dimensions == same_dimensions_different_membership.dimensions
    assert first.artifact_id != same_dimensions_different_membership.artifact_id
    with pytest.raises(TypeError):
        first.scientific_identity["membership"]["universe"][0] = "replacement"


def test_display_annotations_survive_manifest_without_changing_artifact_identity():
    first = _metadata(annotations={"display_labels": {"R1": "Reaction one"}})
    renamed = _metadata(annotations={"display_labels": {"R1": "Renamed reaction"}})

    restored = AnalysisArtifactMetadata.from_manifest(first.to_manifest())

    assert first.artifact_id == renamed.artifact_id
    assert restored.annotations == {"display_labels": {"R1": "Reaction one"}}
    with pytest.raises(TypeError):
        first.annotations["display_labels"]["R1"] = "mutable"


def test_metadata_and_dimensions_are_frozen_and_root_interface_does_not_expand():
    metadata = _metadata()

    with pytest.raises(FrozenInstanceError):
        metadata.method = "replacement"
    with pytest.raises(FrozenInstanceError):
        metadata.dimensions[0].size = 99
    assert not hasattr(metabis, "AnalysisArtifactMetadata")
    assert not hasattr(metabis, "ArtifactProvenance")
    assert not hasattr(metabis, "ArtifactKind")


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("kind", ArtifactKind.REDUCTION),
        ("method", "mann_whitney_u"),
        ("configuration", {"alternative": "greater"}),
        ("scientific_identity", {"comparison_membership": "different"}),
        ("dimensions", (ArtifactDimension("samples", 41),)),
    ],
)
def test_stable_id_changes_with_each_scientific_identity_component(override, value):
    assert _metadata(**{override: value}).artifact_id != _metadata().artifact_id


def test_warnings_and_display_labels_do_not_change_scientific_identity():
    baseline = _metadata()
    annotation_changed = _metadata(
        warnings=("different warning",),
        dimensions=(
            ArtifactDimension("samples", 42, label="Cells"),
            ArtifactDimension("reactions", 2, label="Metabolic reactions"),
        ),
    )

    assert annotation_changed.artifact_id == baseline.artifact_id
    assert annotation_changed.warnings != baseline.warnings
    assert annotation_changed.dimensions != baseline.dimensions


def test_software_version_changes_scientific_identity():
    assert _metadata(software_version="99.0.0").artifact_id != _metadata().artifact_id


def test_dimension_order_is_canonical_for_metadata_manifest_and_identity():
    forward = _metadata(
        dimensions=(
            ArtifactDimension("samples", 42),
            ArtifactDimension("reactions", 2),
        )
    )
    reverse = _metadata(
        dimensions=(
            ArtifactDimension("reactions", 2),
            ArtifactDimension("samples", 42),
        )
    )

    assert forward.dimensions == reverse.dimensions
    assert forward.artifact_id == reverse.artifact_id
    assert forward.to_manifest()["dimensions"] == reverse.to_manifest()["dimensions"]


def test_plain_manifest_survives_json_round_trip_without_aliasing_metadata():
    metadata = _metadata()

    manifest = metadata.to_manifest()
    encoded = json.dumps(manifest, allow_nan=False, sort_keys=True)
    decoded = json.loads(encoded)
    restored = AnalysisArtifactMetadata.from_manifest(decoded)

    assert restored == metadata
    assert restored.artifact_id == metadata.artifact_id
    manifest["configuration"]["features"].append("R3")
    manifest["warnings"].append("later warning")
    assert metadata.configuration["features"] == ("R1", "R2")
    assert metadata.warnings == ("R9 was not mapped",)


def test_manifest_rejects_identity_field_tampering():
    manifest = _metadata().to_manifest()
    manifest["method"] = "different_method"

    with pytest.raises(ValueError, match="identity does not match"):
        AnalysisArtifactMetadata.from_manifest(manifest)


@pytest.mark.parametrize("revision", [-1, True, "1", 1.5])
def test_manifest_rejects_invalid_revision_types_and_values(revision):
    manifest = _metadata().to_manifest()
    manifest["source"]["revisions"]["dataset"] = revision

    with pytest.raises(ValueError, match="non-negative integer"):
        AnalysisArtifactMetadata.from_manifest(manifest)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("kind",), 7, "kind must be a string"),
        (("method",), 7, "method must be a string"),
        (("warnings",), [7], "warnings must be a JSON list"),
        (("software_version",), 7, "software_version must be a string"),
        (("source", "workspace_id"), 7, "workspace_id must be a string"),
        (("source", "source_fingerprints", "dataset"), 7, "canonical strings"),
        (("dimensions", 0, "size"), "42", "non-negative integers"),
    ],
)
def test_manifest_rejects_type_coercion(path, value, message):
    manifest = _metadata().to_manifest()
    target = manifest
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises((TypeError, ValueError), match=message):
        AnalysisArtifactMetadata.from_manifest(manifest)


def test_direct_construction_rejects_coercion_and_negative_revisions():
    workspace = _workspace_provenance()
    with pytest.raises(TypeError, match="kind must be an ArtifactKind"):
        _metadata(kind="group_comparison")
    with pytest.raises(ValueError, match="non-negative integer"):
        ArtifactProvenance(
            workspace_id=workspace.workspace_id,
            revisions=WorkspaceRevisions(dataset=-1),
            dependencies=frozenset({RevisionKind.DATASET}),
            source_fingerprints={"dataset": "sha256:dataset-17"},
        )


def test_configuration_rejects_non_json_and_non_finite_values():
    with pytest.raises(TypeError, match="non-JSON"):
        _metadata(configuration={"manager": object()})
    with pytest.raises(ValueError, match="finite JSON"):
        _metadata(configuration={"threshold": float("nan")})

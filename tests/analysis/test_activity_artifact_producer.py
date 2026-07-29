"""A computed activity context must be able to become its artifact (AM5, MR-12).

`ActivityArtifact.from_context` existed and was never called: it takes an
already-built `AnalysisArtifactMetadata`, and nothing in production built one.
`ActivityArtifact.build` is the seam that does, reading input identity from the
workspace rather than recomputing it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from metabis import AnalysisWorkspace
from metabis.analysis import artifacts, source_identity
from metabis.analysis.activity_analysis import (
    ACTIVITY_METHOD,
    ActivityArtifact,
    ActivityConfig,
    ActivityContext,
    calculate_feature_activity,
    calculate_subsystem_activity,
)
from metabis.analysis.artifacts import ArtifactKind
from metabis.analysis.experiment_analysis import ExperimentAnalysis
from metabis.analysis_workspace import RevisionKind
from metabis.managers.group_manager import GroupDefinition

SAMPLES = ["s1", "s2", "s3", "s4"]
SELECTED = ["s1", "s2"]

ACTIVITY_DEPENDENCIES = [
    RevisionKind.DATASET,
    RevisionKind.GROUPS,
    RevisionKind.PREPROCESSING,
]


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "R1": [5.0, 5.4, 1.0, 1.2],
            "R2": [2.0, 2.1, 2.0, 2.2],
        },
        index=SAMPLES,
    )


def _context() -> ActivityContext:
    config = ActivityConfig(group_id="group-a")
    features = calculate_feature_activity(_frame(), SELECTED, config)
    subsystems = calculate_subsystem_activity(features, {"Glycolysis": ["R1", "R2"]})
    return ActivityContext(
        config=config,
        reaction_features=features,
        reaction_subsystems=subsystems,
        reaction_reference_summary=tuple(
            float(value) for value in _frame().mean(axis=0)
        ),
    )


class _ReactionData:
    def __init__(self) -> None:
        self.frame = _frame()

    def get_reaction_df(self) -> pd.DataFrame:
        return self.frame

    def get_sample_ids(self):
        return list(self.frame.index)


class _Collection:
    def __init__(self, values) -> None:
        self._values = list(values)

    def get_all(self):
        return list(self._values)


class _Model:
    def __init__(self) -> None:
        self.reactions = _Collection(["R1", "R2"])
        self.subsystems = _Collection(["Glycolysis"])

    def get_reaction_metadata(self, reaction_id):
        return {"subsystems": ["Glycolysis"]}

    def get_subsystems(self):
        return ["Glycolysis"]

    def get_reactions_for_subsystem(self, subsystem):
        return ["R1", "R2"]

    def get_genes_for_subsystem(self, subsystem):
        return []


def _analysable_workspace() -> AnalysisWorkspace:
    """A workspace that can accept an installation: loaded, with an analysis."""
    manager = AnalysisWorkspace(metabolic_model=_Model(), reaction_data=_ReactionData())
    manager.group_manager.define_group(
        "group-a", SELECTED, definition=GroupDefinition("custom")
    )
    analysis = ExperimentAnalysis(manager, auto_run=False)
    provenance = manager.capture_analysis_snapshot().provenance
    manager.install_experiment_analysis(provenance, analysis)
    manager.refresh_experiment_analysis_presentation()
    return manager


def _provenance(manager: AnalysisWorkspace):
    return manager.capture_analysis_snapshot(ACTIVITY_DEPENDENCIES).provenance


def test_a_computed_activity_context_builds_its_artifact():
    manager = AnalysisWorkspace()
    context = _context()

    artifact = ActivityArtifact.build(
        context,
        provenance=_provenance(manager),
        source_fingerprints=manager.source_fingerprints,
        selected_sample_ids=SELECTED,
        reference_sample_ids=SAMPLES,
    )

    assert isinstance(artifact, ActivityArtifact)
    assert artifact.metadata.kind is ArtifactKind.ACTIVITY
    assert artifact.metadata.method == ACTIVITY_METHOD
    assert artifact.config == context.config
    assert artifact.selected_sample_ids == ("s1", "s2")
    assert artifact.reference_sample_ids == ("s1", "s2", "s3", "s4")


def test_the_artifact_carries_the_workspace_identity_of_its_real_inputs():
    manager = AnalysisWorkspace()

    artifact = ActivityArtifact.build(
        _context(),
        provenance=_provenance(manager),
        source_fingerprints=manager.source_fingerprints,
        selected_sample_ids=SELECTED,
        reference_sample_ids=SAMPLES,
    )

    source = artifact.metadata.source
    assert source.source_fingerprints["dataset"] == manager.source_fingerprints["dataset"]
    assert source.source_fingerprints["dataset"].startswith("sha256:")
    assert source.workspace_id == manager.workspace_id


def test_building_an_artifact_never_fingerprints_a_full_matrix(monkeypatch):
    manager = AnalysisWorkspace()
    context = _context()
    provenance = _provenance(manager)
    fingerprints = manager.source_fingerprints

    def forbidden(_frame):
        raise AssertionError(
            "The activity producer fingerprinted a full matrix; read identity "
            "from the workspace instead (ADR-0007)"
        )

    monkeypatch.setattr(source_identity, "fingerprint_frame", forbidden)
    monkeypatch.setattr(artifacts, "fingerprint_frame", forbidden)

    artifact = ActivityArtifact.build(
        context,
        provenance=provenance,
        source_fingerprints=fingerprints,
        selected_sample_ids=SELECTED,
        reference_sample_ids=SAMPLES,
    )

    assert artifact.metadata.source.source_fingerprints["dataset"].startswith("sha256:")


def test_the_artifact_installs_into_the_workspace():
    """The install seam had no production caller; the producer is what reaches it."""
    manager = _analysable_workspace()
    context = _context()
    provenance = _provenance(manager)

    artifact = ActivityArtifact.build(
        context,
        provenance=provenance,
        source_fingerprints=manager.source_fingerprints,
        selected_sample_ids=SELECTED,
        reference_sample_ids=SAMPLES,
    )
    installed = manager.install_activity_artifact(provenance, artifact)

    assert installed is artifact
    assert manager.get_activity_artifact() is artifact


def test_computation_warnings_travel_onto_the_artifact():
    """A non-finite summary warning must reach the export, not just the log."""
    manager = AnalysisWorkspace()
    config = ActivityConfig(group_id="group-a")
    missing = pd.DataFrame({"R1": [float("nan"), float("nan"), 1.0, 1.0]}, index=SAMPLES)
    features = calculate_feature_activity(missing, SELECTED, config)
    context = ActivityContext(
        config=config,
        reaction_features=features,
        reaction_subsystems=calculate_subsystem_activity(
            features, {"Glycolysis": ["R1"]}
        ),
    )

    artifact = ActivityArtifact.build(
        context,
        provenance=_provenance(manager),
        source_fingerprints=manager.source_fingerprints,
        selected_sample_ids=SELECTED,
        reference_sample_ids=SAMPLES,
    )

    assert artifact.warnings
    assert artifact.warnings == artifact.metadata.warnings


def test_selected_samples_must_belong_to_the_reference_universe():
    manager = AnalysisWorkspace()

    with pytest.raises(ValueError, match="reference universe"):
        ActivityArtifact.build(
            _context(),
            provenance=_provenance(manager),
            source_fingerprints=manager.source_fingerprints,
            selected_sample_ids=["s1", "not-a-sample"],
            reference_sample_ids=SAMPLES,
        )


def test_the_workspace_resolves_the_samples_the_scoring_actually_used():
    """The caller must not be asked to restate which samples the scores summarize.

    A group can name a sample the aligned matrix does not have; the scoring drops
    it, and an artifact that took the caller's word for it would claim a sample
    count its own scores contradict.
    """
    manager = _analysable_workspace()
    manager.group_manager.define_group(
        "group-a",
        ["s1", "s2", "absent-sample"],
        definition=GroupDefinition("custom"),
        replace=True,
    )
    analysis = manager.experiment_analysis
    context = analysis.compute_activity(ActivityConfig(group_id="group-a"))

    artifact = analysis.build_activity_artifact(
        context,
        provenance=_provenance(manager),
        source_fingerprints=manager.source_fingerprints,
    )

    assert artifact.selected_sample_ids == ("s1", "s2")
    assert artifact.reference_sample_ids == tuple(SAMPLES)
    assert artifact.features.sample_count == len(artifact.selected_sample_ids)
    assert artifact.features.reference_sample_count == len(
        artifact.reference_sample_ids
    )


def test_the_resolved_artifact_installs_through_the_workspace_seam():
    manager = _analysable_workspace()
    analysis = manager.experiment_analysis
    context = analysis.compute_activity(ActivityConfig(group_id="group-a"))
    provenance = _provenance(manager)

    artifact = analysis.build_activity_artifact(
        context,
        provenance=provenance,
        source_fingerprints=manager.source_fingerprints,
    )
    installed = manager.install_activity_artifact(provenance, artifact)

    assert installed is artifact
    assert manager.get_activity_artifact() is artifact
    assert (
        artifact.metadata.source.source_fingerprints["dataset"]
        == manager.source_fingerprints["dataset"]
    )

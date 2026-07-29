"""A computed comparison must be able to become its artifact (AM5).

The AM5 library half landed ``GroupComparisonArtifact``, its install seam, its
threshold-update path, its plot adapters, and an enrichment analysis that refuses
to install without it -- but nothing in production ever *built* one.  Every test
hand-assembled the artifact from literals, ``source_fingerprints`` included, so
``AnalysisArtifactMetadata.from_workspace`` had no production caller and
``install_group_comparison_artifact`` was never reached.

That gap is what makes the MR-10 regression tempting: a caller that needs
provenance for an artifact the library gives it no way to build will compute the
identity itself, at the only moment it has the data -- which is per render.  So
the producer belongs here, beside the analysis that owns the science, and it must
*read* input identity from the workspace rather than compute it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from metabis import (
    AnalysisWorkspace,
    GroupComparison,
    GroupComparisonSpec,
)
from metabis.analysis import artifacts, source_identity
from metabis.analysis.artifacts import ArtifactKind
from metabis.analysis.comparison_artifact import GroupComparisonArtifact
from metabis.analysis_workspace import RevisionKind
from metabis.domain.expression_matrix import DenseExpressionMatrix
from metabis.managers.group_manager import GroupDefinition

SAMPLES = ["A1", "A2", "A3", "B1", "B2", "B3"]

COMPARISON_DEPENDENCIES = frozenset(
    {
        RevisionKind.DATASET,
        RevisionKind.METADATA,
        RevisionKind.GROUPS,
        RevisionKind.PREPROCESSING,
    }
)


class _ReactionData:
    def __init__(self) -> None:
        self.frame = pd.DataFrame(
            {
                "R1": [1.0, 1.2, 0.9, 5.0, 5.4, 4.8],
                "R2": [2.0, 2.1, 1.9, 2.2, 2.0, 2.1],
            },
            index=SAMPLES,
        )

    def get_reaction_df(self) -> pd.DataFrame:
        return self.frame

    def get_sample_ids(self):
        return list(self.frame.index)


class _ExpressionData:
    def __init__(self) -> None:
        self.frame = pd.DataFrame(
            {
                "G1": [10.0, 11.0, 9.5, 30.0, 31.0, 29.0],
                "G2": [4.0, 4.2, 3.9, 4.1, 4.0, 4.2],
            },
            index=SAMPLES,
        )

    def get_expression_df(self) -> pd.DataFrame:
        return self.frame

    def get_feature_ids(self):
        return [str(column) for column in self.frame.columns]

    def matrix(self):
        return DenseExpressionMatrix(self.frame)

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

    def get_gene_metadata(self, gene_id):
        return {"subsystems": ["Glycolysis"]}


def _loaded_workspace() -> AnalysisWorkspace:
    """A session workspace as the app builds one: constructed empty, then loaded."""
    manager = AnalysisWorkspace(metabolic_model=_Model())
    manager.replace_reaction_data(_ReactionData(), metabolic_model=_Model())
    manager.replace_expression_data(_ExpressionData())
    manager.group_manager.define_group(
        "A", ["A1", "A2", "A3"], definition=GroupDefinition("custom")
    )
    manager.group_manager.define_group(
        "B", ["B1", "B2", "B3"], definition=GroupDefinition("custom")
    )
    return manager


def _computed_comparison(manager: AnalysisWorkspace) -> GroupComparison:
    membership = manager.group_manager.resolve_comparison_membership(
        GroupComparisonSpec.pairwise(("A",), ("B",)), SAMPLES
    )
    comparison = GroupComparison(membership=membership, test="ttest")
    for for_genes in (False, True):
        comparison.run_test(
            manager,
            for_genes=for_genes,
            test="ttest",
            correction="fdr_bh",
            register_distributions=False,
        )
    return comparison


def test_a_computed_comparison_builds_its_artifact():
    manager = _loaded_workspace()
    comparison = _computed_comparison(manager)
    snapshot = manager.capture_analysis_snapshot(COMPARISON_DEPENDENCIES)

    artifact = comparison.build_artifact(
        provenance=snapshot.provenance,
        source_fingerprints=manager.source_fingerprints,
    )

    assert isinstance(artifact, GroupComparisonArtifact)
    assert artifact.metadata.kind is ArtifactKind.GROUP_COMPARISON
    assert artifact.test == "ttest"
    assert artifact.correction == "fdr_bh"
    assert artifact.membership == comparison.membership
    assert tuple(artifact.reaction_results.index) == ("R1", "R2")
    assert artifact.gene_results is not None
    assert tuple(artifact.gene_results.index) == ("G1", "G2")


def test_composite_target_and_labels_round_trip_through_the_artifact():
    manager = _loaded_workspace()
    manager.group_manager.define_group(
        "A-left", ["A1", "A2"], definition=GroupDefinition("metadata")
    )
    manager.group_manager.define_group(
        "A-right", ["A3"], definition=GroupDefinition("metadata")
    )
    spec = GroupComparisonSpec.pairwise(
        ("A-left", "A-right"),
        ("B",),
        target_label="Combined A",
        baseline_label="Reference B",
    )
    membership = manager.group_manager.resolve_comparison_membership(spec, SAMPLES)
    comparison = GroupComparison(membership=membership, test="ttest")
    comparison.run_test(
        manager,
        test="ttest",
        correction="fdr_bh",
        register_distributions=False,
    )
    snapshot = manager.capture_analysis_snapshot(COMPARISON_DEPENDENCIES)

    artifact = comparison.build_artifact(
        provenance=snapshot.provenance,
        source_fingerprints=manager.source_fingerprints,
    )
    reresolved = manager.group_manager.resolve_comparison_membership(
        artifact.membership.spec, SAMPLES
    )

    assert artifact.membership.target_base_group_ids == ("A-left", "A-right")
    assert artifact.membership.target_samples == ("A1", "A2", "A3")
    assert artifact.membership.target_label == "Combined A"
    assert artifact.membership.baseline_label == "Reference B"
    assert reresolved == artifact.membership
    assert artifact.metadata.scientific_identity["target_base_group_ids"] == (
        "A-left",
        "A-right",
    )


def test_the_artifact_carries_the_workspace_identity_of_its_real_inputs():
    """The fingerprints must be the loaded data's, not a literal a test invented."""
    manager = _loaded_workspace()
    comparison = _computed_comparison(manager)
    snapshot = manager.capture_analysis_snapshot(COMPARISON_DEPENDENCIES)

    artifact = comparison.build_artifact(
        provenance=snapshot.provenance,
        source_fingerprints=manager.source_fingerprints,
    )

    fingerprints = artifact.metadata.source.source_fingerprints
    assert fingerprints["dataset"] == manager.source_fingerprints["dataset"]
    assert fingerprints["dataset"].startswith("sha256:")
    assert artifact.metadata.source.workspace_id == manager.workspace_id
    assert artifact.metadata.source.dependencies == COMPARISON_DEPENDENCIES


def test_building_an_artifact_never_fingerprints_a_full_matrix(monkeypatch):
    """The producer reads identity; it does not recompute it (ADR-0007, MR-10).

    A producer that hashed its inputs would put a full-matrix hash on every
    comparison -- the regression that got the first AM5 server slice reverted,
    re-entered through the seam built to prevent it.
    """
    manager = _loaded_workspace()
    comparison = _computed_comparison(manager)
    snapshot = manager.capture_analysis_snapshot(COMPARISON_DEPENDENCIES)

    def forbidden(frame):
        raise AssertionError(
            "The comparison producer fingerprinted a full matrix; read identity "
            "from the workspace instead (ADR-0007)"
        )

    # Both namespaces: rebinding only one leaves the other reachable by import.
    monkeypatch.setattr(source_identity, "fingerprint_frame", forbidden)
    monkeypatch.setattr(artifacts, "fingerprint_frame", forbidden)
    monkeypatch.setattr(
        source_identity,
        "model_identity",
        lambda model: pytest.fail("The producer walked the whole model"),
    )

    artifact = comparison.build_artifact(
        provenance=snapshot.provenance,
        source_fingerprints=manager.source_fingerprints,
    )

    assert artifact.metadata.source.source_fingerprints["dataset"].startswith("sha256:")


def test_the_built_artifact_installs_into_the_workspace():
    """The install seam had no production caller; the producer is what reaches it."""
    manager = _loaded_workspace()
    comparison = _computed_comparison(manager)
    snapshot = manager.capture_analysis_snapshot(COMPARISON_DEPENDENCIES)
    artifact = comparison.build_artifact(
        provenance=snapshot.provenance,
        source_fingerprints=manager.source_fingerprints,
    )

    installed = manager.install_group_comparison_artifact(
        snapshot.provenance, artifact
    )

    assert installed is artifact
    assert manager.get_group_comparison_artifact() is artifact


def test_display_labels_reach_the_export_without_a_manager_lookup():
    """The export table is the artifact's, so labels must travel on the artifact."""
    manager = _loaded_workspace()
    comparison = _computed_comparison(manager)
    snapshot = manager.capture_analysis_snapshot(COMPARISON_DEPENDENCIES)

    artifact = comparison.build_artifact(
        provenance=snapshot.provenance,
        source_fingerprints=manager.source_fingerprints,
        reaction_display_labels={"R1": "Hexokinase", "R2": "Enolase"},
        gene_display_labels={"G1": "HK1", "G2": "ENO1"},
    )

    export = artifact.export_frame("reaction")
    assert list(export["feature_id"]) == ["R1", "R2"]
    assert list(export["display_label"]) == ["Hexokinase", "Enolase"]
    assert set(export["membership_fingerprint"]) == {comparison.membership.fingerprint}
    assert list(artifact.export_frame("gene")["display_label"]) == ["HK1", "ENO1"]


def test_an_unresolved_comparison_cannot_become_an_artifact():
    manager = _loaded_workspace()
    comparison = GroupComparison(spec=GroupComparisonSpec.pairwise(("A",), ("B",)))

    snapshot = manager.capture_analysis_snapshot(COMPARISON_DEPENDENCIES)

    with pytest.raises(ValueError, match="membership"):
        comparison.build_artifact(
            provenance=snapshot.provenance,
            source_fingerprints=manager.source_fingerprints,
        )


def test_an_uncomputed_comparison_cannot_become_an_artifact():
    manager = _loaded_workspace()
    membership = manager.group_manager.resolve_comparison_membership(
        GroupComparisonSpec.pairwise(("A",), ("B",)), SAMPLES
    )
    comparison = GroupComparison(membership=membership)
    snapshot = manager.capture_analysis_snapshot(COMPARISON_DEPENDENCIES)

    with pytest.raises(ValueError, match="no reaction results"):
        comparison.build_artifact(
            provenance=snapshot.provenance,
            source_fingerprints=manager.source_fingerprints,
        )

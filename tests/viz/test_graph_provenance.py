"""A production graph artifact must carry provenance, like every other family (AM5).

`build_artifact` had built its `GraphArtifact` with no `metadata=`, so every
production graph run carried none: the node audit emitted `artifact_id`,
`workspace_id`, `dataset_fingerprint`, and `source_fingerprints` as empty strings,
while the tests passed because they hand-built an artifact *with* metadata. That is
the MR-12 shape -- a seam that looks like it works and exports blanks in production.

The graph is the last family to close it, and it closes it under the MR-10 rule:
`build_artifact` reads input identity from an explicit parameter the caller
captured on the document thread, and never off `data_manager`. The server hands the
graph worker a *detached* facade over frames the session already owns, so hashing
that facade's inputs would SHA-256 the whole reaction matrix, the whole expression
matrix, and the entire model on every render. See ADR-0007, MR-10, and MR-12.
"""

import hashlib

import pandas as pd
import pytest

from metabis import AnalysisWorkspace, MetabolicModel
from metabis.analysis import artifacts, source_identity
from metabis.analysis.artifacts import ArtifactKind
from metabis.analysis_workspace import RevisionKind
from metabis.viz import plot_metabolic_graph as plot_module
from metabis.viz.metabolic_visualization import (
    GraphArtifact,
    MetabolicGraphPipeline,
    PipelineConfig,
    graph_audit_frame,
)
from metabis.viz.metabolic_visualization.generation.generator import (
    GraphGenerationConfig,
)

SUBSYSTEM_NAME = "Glycolysis / Gluconeogenesis"

# The graph depends on the same workspace axes the server captures for it.
GRAPH_DEPENDENCIES = [
    RevisionKind.DATASET,
    RevisionKind.METADATA,
    RevisionKind.GROUPS,
    RevisionKind.PREPROCESSING,
]


class _ReactionData:
    """A frame-backed reaction dataset, the shape a dataset load writer accepts."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def get_reaction_df(self) -> pd.DataFrame:
        return self.frame

    def get_sample_ids(self):
        return list(self.frame.index)

    def get_reaction_row(self, reaction_id):
        return self.frame[reaction_id] if reaction_id in self.frame.columns else None

    def get_gene_row(self, gene_id):
        return None


def _reaction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"MAR04358": [1.0, 2.0, 3.0], "MAR04379": [4.0, 5.0, 6.0]},
        index=["s1", "s2", "s3"],
    )


def _pipeline(model) -> MetabolicGraphPipeline:
    return MetabolicGraphPipeline(
        model,
        PipelineConfig(
            generation_config=GraphGenerationConfig(layout_engine="dot"),
            show_plot=False,
            enable_timing=True,
        ),
    )


def _structure_digest(artifact: GraphArtifact) -> str:
    """Identity, class, geometry, and drawing order of every node and edge."""
    payload = (
        tuple(
            (
                str(node["id"]),
                str(node["glyph_class"]),
                round(float(node["x"]), 3),
                round(float(node["y"]), 3),
            )
            for node in artifact.nodes
        ),
        tuple(
            (str(edge["source"]), str(edge["target"])) for edge in artifact.edges
        ),
    )
    return hashlib.sha256(repr(payload).encode()).hexdigest()


def _colour_digest(artifact: GraphArtifact) -> str:
    payload = tuple(
        (str(node["id"]), str(node.get("color")), round(float(node.get("alpha", 1.0)), 3))
        for node in artifact.nodes
    )
    return hashlib.sha256(repr(payload).encode()).hexdigest()


@pytest.fixture(scope="module")
def model() -> MetabolicModel:
    return MetabolicModel(use_cache=True)


@pytest.fixture
def loaded_workspace(model) -> AnalysisWorkspace:
    """A session workspace as the app builds one: constructed empty, then loaded.

    Loading through the dataset load writer is what warms the SourceIdentityCache,
    so reading ``source_fingerprints`` is free and returns the identity of the
    data actually loaded -- the value the graph artifact must carry.
    """
    manager = AnalysisWorkspace(metabolic_model=model)
    manager.replace_reaction_data(_ReactionData(_reaction_frame()), metabolic_model=model)
    return manager


@pytest.mark.integration
def test_building_an_artifact_never_reads_workspace_identity(model, monkeypatch):
    """The MR-10 guard: reading identity per render is the reverted regression.

    The graph worker is handed a detached ``AnalysisWorkspace`` that recorded its inputs
    but never went through a load writer, so its ``source_fingerprints`` are unpaid
    and reading them hashes the full matrices. This drives a real ``build_artifact``
    with ``fingerprint_frame`` and ``model_identity`` rigged to raise -- in both
    namespaces they are reachable through, since rebinding only one leaves the
    other importable -- and it must complete. It fails the instant someone makes
    ``build_artifact`` read ``data_manager.source_fingerprints``.
    """
    # Constructed, not loaded through a writer: recorded, unpaid -- the exact shape
    # of the server's per-job worker facade.
    manager = AnalysisWorkspace(metabolic_model=model, reaction_data=_ReactionData(_reaction_frame()))

    def forbidden(_input):
        raise AssertionError(
            "build_artifact recomputed input identity; read it from the workspace "
            "the caller captured, never off data_manager (ADR-0007)"
        )

    # Both namespaces the hashers are reachable through: rebinding only one leaves
    # the other importable, so a producer that imported it directly would slip by.
    monkeypatch.setattr(source_identity, "fingerprint_frame", forbidden)
    monkeypatch.setattr(artifacts, "fingerprint_frame", forbidden)
    monkeypatch.setattr(source_identity, "model_identity", forbidden)

    artifact = _pipeline(model).build_artifact(
        SUBSYSTEM_NAME, "subsystem", data_manager=manager
    )

    assert artifact.node_count > 0
    # A structure-only graph built with no captured identity carries none, and
    # does not invent one from the manager it was handed.
    assert artifact.metadata is None

    # Prove the trap was armed: identity on this manager is genuinely unpaid, so
    # any read of it during the build would have hashed and raised above.
    with pytest.raises(AssertionError):
        _ = manager.source_fingerprints


@pytest.mark.integration
def test_provenance_from_a_real_loaded_workspace_reaches_the_artifact(
    model, loaded_workspace
):
    """The artifact carries the loaded data's identity, not a literal a test invents."""
    snapshot = loaded_workspace.capture_analysis_snapshot(GRAPH_DEPENDENCIES)
    pipeline = _pipeline(model)

    # Rendering it too, so the identity travels through the whole document-thread
    # path a notebook caller uses, not only the producer in isolation.
    pipeline.run_full_pipeline(
        SUBSYSTEM_NAME,
        "subsystem",
        data_manager=loaded_workspace,
        provenance=snapshot.provenance,
        source_fingerprints=loaded_workspace.source_fingerprints,
    )
    artifact = pipeline.get_artifact()

    assert artifact.metadata is not None
    assert artifact.metadata.kind is ArtifactKind.GRAPH
    source = artifact.metadata.source
    assert source.workspace_id == loaded_workspace.workspace_id
    # Read from the workspace that paid for it at load, not fingerprinted here.
    assert (
        source.source_fingerprints["dataset"]
        == loaded_workspace.source_fingerprints["dataset"]
    )
    assert source.source_fingerprints["dataset"].startswith("sha256:")
    assert source.dependencies == frozenset(GRAPH_DEPENDENCIES)
    # Dimensions are facts of the payload, and GraphArtifact validation demands it.
    assert {d.name: d.size for d in artifact.metadata.dimensions} == {
        "nodes": artifact.node_count,
        "edges": artifact.edge_count,
    }


@pytest.mark.integration
def test_the_audit_export_carries_real_provenance_matching_the_nodes(
    model, loaded_workspace
):
    """The blanks-in-production defect: audit columns must not come back empty."""
    snapshot = loaded_workspace.capture_analysis_snapshot(GRAPH_DEPENDENCIES)
    artifact = _pipeline(model).build_artifact(
        SUBSYSTEM_NAME,
        "subsystem",
        data_manager=loaded_workspace,
        provenance=snapshot.provenance,
        source_fingerprints=loaded_workspace.source_fingerprints,
    )

    frame = graph_audit_frame(artifact)

    # None of these may be blank in production -- the exact MR-12 defect.
    for column in ("artifact_id", "workspace_id", "dataset_fingerprint", "source_fingerprints"):
        assert (frame[column] != "").all(), f"{column} exported blank"
    assert set(frame["artifact_id"]) == {artifact.metadata.artifact_id}
    assert set(frame["workspace_id"]) == {loaded_workspace.workspace_id}
    assert set(frame["dataset_fingerprint"]) == {
        loaded_workspace.source_fingerprints["dataset"]
    }

    # A stage exit criterion: audit rows match the rendered nodes exactly.
    assert list(frame["node_id"]) == list(artifact.node_ids())
    assert list(frame["color"]) == [str(node.get("color", "")) for node in artifact.nodes]


@pytest.mark.integration
def test_provenance_does_not_change_graph_structure_or_colour(model, loaded_workspace):
    """Attaching provenance is metadata only: it may not move a node or repaint one."""
    snapshot = loaded_workspace.capture_analysis_snapshot(GRAPH_DEPENDENCIES)
    pipeline = _pipeline(model)

    without = pipeline.build_artifact(
        SUBSYSTEM_NAME, "subsystem", data_manager=loaded_workspace
    )
    with_provenance = _pipeline(model).build_artifact(
        SUBSYSTEM_NAME,
        "subsystem",
        data_manager=loaded_workspace,
        provenance=snapshot.provenance,
        source_fingerprints=loaded_workspace.source_fingerprints,
    )

    assert without.metadata is None
    assert with_provenance.metadata is not None
    assert _structure_digest(with_provenance) == _structure_digest(without)
    assert _colour_digest(with_provenance) == _colour_digest(without)


def test_plot_metabolic_graph_forwards_provenance_to_the_pipeline(monkeypatch):
    """The public entry point threads identity through rather than dropping it."""
    recorded = {}

    class _FakePipeline:
        def __init__(self, *_args, **_kwargs):
            pass

        def run_full_pipeline(self, *_args, **kwargs):
            recorded.update(kwargs)
            return object()

    monkeypatch.setattr(plot_module, "MetabolicGraphPipeline", _FakePipeline)

    class _Manager:
        metabolic_model = object()

    provenance = object()
    fingerprints = {"dataset": "sha256:example"}
    plot_module.plot_metabolic_graph(
        SUBSYSTEM_NAME,
        data_manager=_Manager(),
        show_plot=False,
        add_click_interactions=False,
        provenance=provenance,
        source_fingerprints=fingerprints,
    )

    assert recorded["provenance"] is provenance
    assert recorded["source_fingerprints"] is fingerprints

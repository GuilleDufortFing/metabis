"""The graph artifact is the seam, and Bokeh and export are two adapters over it.

What these tests defend is the direction of dependency. Analysis decides what a node
is and what colour it is; the adapters translate that and decide nothing. If a
scientific decision ever leaks back into rendering, or the export starts disagreeing
with the picture, it fails here.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from metabis import AnalysisWorkspace, MetabolicModel
from metabis.analysis.artifacts import (
    AnalysisArtifactMetadata,
    ArtifactDimension,
    ArtifactKind,
    ArtifactProvenance,
)
from metabis.analysis_workspace import RevisionKind, WorkspaceRevisions
from metabis.viz.metabolic_visualization import (
    GraphArtifact,
    GraphColorMeaning,
    MetabolicGraphPipeline,
    PipelineConfig,
    RenderingConfig,
    graph_audit_frame,
    graph_audit_rows,
    graph_auxiliary_frame,
    graph_edge_frame,
    render_graph_artifact,
)
from metabis.viz.metabolic_visualization.artifact import STRUCTURAL_MODE
from metabis.viz.metabolic_visualization.export import (
    AUXILIARY_AUDIT_COLUMNS,
    EDGE_AUDIT_COLUMNS,
)
from metabis.viz.metabolic_visualization.generation.generator import (
    GraphGenerationConfig,
)

SUBSYSTEM_NAME = "Glycolysis / Gluconeogenesis"


@pytest.fixture(scope="module")
def model():
    return MetabolicModel(use_cache=True)


@pytest.fixture(scope="module")
def manager(model):
    return AnalysisWorkspace(metabolic_model=model)


def build(model, manager, element_name, graph_type, layout_engine="dot") -> GraphArtifact:
    pipeline = MetabolicGraphPipeline(
        model,
        PipelineConfig(
            generation_config=GraphGenerationConfig(layout_engine=layout_engine),
            show_plot=False,
            enable_timing=True,
        ),
    )
    return pipeline.build_artifact(element_name, graph_type, data_manager=manager)


@pytest.fixture(scope="module")
def subsystem_artifact(model, manager):
    return build(model, manager, SUBSYSTEM_NAME, "subsystem")


@pytest.fixture(scope="module")
def overview_artifact(model, manager):
    return build(model, manager, None, "metabolic_network", layout_engine="fdp")


@pytest.mark.integration
def test_building_an_artifact_constructs_no_bokeh_model(model, manager, monkeypatch):
    """The graph worker runs off the document thread, so this may never be violated."""
    from bokeh.model import Model as BokehModel

    monkeypatch.setattr(
        BokehModel,
        "__init__",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("graph analysis constructed a Bokeh model")
        ),
    )

    artifact = build(model, manager, SUBSYSTEM_NAME, "subsystem")

    assert artifact.node_count > 0
    assert artifact.edge_count > 0


@pytest.mark.integration
def test_an_artifact_is_read_only(subsystem_artifact):
    """Two adapters share one artifact, so neither may edit what the other reads."""
    node = subsystem_artifact.nodes[0]

    with pytest.raises(TypeError):
        node["color"] = "#000000"
    with pytest.raises(TypeError):
        subsystem_artifact.auxiliary["influence_circles"]["x"] = []
    with pytest.raises(AttributeError):
        subsystem_artifact.display_title = "renamed"


@pytest.mark.integration
@pytest.mark.parametrize("fixture_name", ["subsystem_artifact", "overview_artifact"])
def test_the_bokeh_adapter_renders_a_representative_artifact(fixture_name, request):
    artifact = request.getfixturevalue(fixture_name)

    rendered = render_graph_artifact(artifact, RenderingConfig(width=800, height=600))

    assert rendered.plot is not None
    assert rendered.plot.title.text == artifact.display_title
    # The model count is a rendering fact, so it exists only once models exist.
    assert rendered.metadata.bokeh_model_count > 0
    assert rendered.metadata.glyph_count == artifact.node_count
    assert rendered.metadata.connection_count == artifact.edge_count
    assert [phase.name for phase in rendered.metadata.phases][-2:] == [
        "render_graph",
        "total_pipeline",
    ]


@pytest.mark.integration
def test_the_export_adapter_reports_exactly_what_was_drawn(subsystem_artifact):
    """A row cannot disagree with the pixel: both are read from the same artifact."""
    rendered = render_graph_artifact(subsystem_artifact)
    rows = graph_audit_rows(subsystem_artifact)

    assert len(rows) == subsystem_artifact.node_count
    assert tuple(row["node_id"] for row in rows) == subsystem_artifact.node_ids()
    assert [row["color"] for row in rows] == [
        node["color"] for node in subsystem_artifact.nodes
    ]
    assert rendered.metadata.glyph_count == len(rows)


@pytest.mark.integration
def test_the_audit_table_has_a_stable_shape(subsystem_artifact):
    frame = graph_audit_frame(subsystem_artifact)

    assert list(frame.columns) == [
        "node_id",
        "glyph_class",
        "label",
        "display_label",
        "original_id",
        "x",
        "y",
        "color",
        "alpha",
        "is_missing_data",
        "graph_type",
        "element_id",
        "resource_path",
        "resource_variant",
        "color_mode",
        "color_metric",
        "color_saturation_limit",
        "display_title",
        "layout_engine",
        "used_runtime_layout",
        "artifact_warnings",
        "artifact_id",
        "workspace_id",
        "dataset_fingerprint",
        "source_fingerprints",
        "raw_metric",
        "score",
        "significance",
        "missing_reason",
        "direction",
    ]
    assert len(frame) == subsystem_artifact.node_count
    assert frame["node_id"].is_unique
    assert set(frame["glyph_class"]) == {
        node["glyph_class"] for node in subsystem_artifact.nodes
    }


@pytest.mark.integration
def test_an_uncoloured_graph_says_so_rather_than_claiming_a_metric(subsystem_artifact):
    """With no experimental data loaded, the graph is structure and admits it."""
    assert subsystem_artifact.color.is_structural
    assert subsystem_artifact.color.mode == STRUCTURAL_MODE
    assert subsystem_artifact.color.saturation_limit is None

    rows = graph_audit_rows(subsystem_artifact)

    assert {row["color_mode"] for row in rows} == {STRUCTURAL_MODE}


def test_audit_copies_optional_scientific_node_values_without_parsing_hover():
    artifact = GraphArtifact(
        graph_type="subsystem",
        element_id="S1",
        display_title="Subsystem one",
        nodes=(
            {
                "id": "R1",
                "glyph_class": "reaction",
                "label": "R1",
                "x": 1.0,
                "y": 2.0,
                "color": "#123456",
                "hover": "<b>wrong values: 999</b>",
                "raw_metric": 2.5,
                "score": -0.75,
                "significance": True,
                "missing_reason": "below_threshold",
                "direction": "backward",
            },
        ),
        edges=(),
        auxiliary={},
        color=GraphColorMeaning(mode="experiment_analysis", metric="activity"),
        resource_path="fixture.json",
        resource_variant="standard",
        layout_engine="dot",
        used_runtime_layout=False,
        hover_detail="full",
    )

    row = graph_audit_rows(artifact)[0]

    assert row["raw_metric"] == 2.5
    assert row["score"] == -0.75
    assert row["significance"] is True
    assert row["missing_reason"] == "below_threshold"
    assert row["direction"] == "backward"
    assert 999 not in row.values()


def _graph_metadata(*, nodes=1, edges=0):
    return AnalysisArtifactMetadata(
        kind=ArtifactKind.GRAPH,
        source=ArtifactProvenance(
            workspace_id="workspace-1",
            revisions=WorkspaceRevisions(dataset=1),
            dependencies=frozenset({RevisionKind.DATASET}),
            source_fingerprints={"dataset": "sha256:fixture"},
        ),
        method="metabolic_graph_pipeline",
        configuration={},
        scientific_identity={},
        dimensions=(
            ArtifactDimension("nodes", nodes),
            ArtifactDimension("edges", edges),
        ),
    )


def test_graph_artifact_rejects_metadata_dimensions_that_diverge_from_geometry():
    with pytest.raises(ValueError, match="dimensions"):
        GraphArtifact(
            graph_type="subsystem",
            element_id="S1",
            display_title="S1",
            nodes=({"id": "R1"},),
            edges=(),
            auxiliary={},
            color=GraphColorMeaning(mode="structural", metric="default"),
            resource_path=None,
            resource_variant="standard",
            layout_engine="dot",
            used_runtime_layout=False,
            hover_detail="light",
            metadata=_graph_metadata(nodes=2),
        )


def test_graph_artifact_recursively_freezes_nested_and_numpy_geometry():
    artifact = GraphArtifact(
        graph_type="subsystem",
        element_id="S1",
        display_title="S1",
        nodes=({"id": "R1", "nested": {"points": np.array([1.0, 2.0])}},),
        edges=(),
        auxiliary={"overlay": {"labels": ["a", "b"]}},
        color=GraphColorMeaning(mode="structural", metric="default"),
        resource_path=None,
        resource_variant="standard",
        layout_engine="dot",
        used_runtime_layout=False,
        hover_detail="light",
        metadata=_graph_metadata(),
    )

    with pytest.raises(TypeError):
        artifact.nodes[0]["nested"]["points"][0] = 99.0
    with pytest.raises(TypeError):
        artifact.auxiliary["overlay"]["labels"] += ("c",)

    with pytest.raises(TypeError, match="pandas"):
        replace(artifact, nodes=({"id": "R1", "frame": pd.DataFrame({"x": [1]})},))


def test_edge_and_auxiliary_exports_preserve_complete_payloads_and_empty_headers():
    artifact = GraphArtifact(
        graph_type="subsystem",
        element_id="S1",
        display_title="S1",
        nodes=({"id": "R1"}, {"id": "R2"}),
        edges=({"source": "R1", "target": "R2", "nested": {"points": [1, 2]}},),
        auxiliary={"influence": {"x": [1.0], "label": ["R1"]}},
        color=GraphColorMeaning(mode="structural", metric="default"),
        resource_path=None,
        resource_variant="standard",
        layout_engine="dot",
        used_runtime_layout=False,
        hover_detail="light",
        metadata=_graph_metadata(nodes=2, edges=1),
    )

    edges = graph_edge_frame(artifact)
    auxiliary = graph_auxiliary_frame(artifact)

    assert edges.loc[0, "source_id"] == "R1"
    assert '"nested":{"points":[1,2]}' in edges.loc[0, "payload_json"]
    assert auxiliary.loc[0, "layer"] == "influence"
    assert '"label":["R1"]' in auxiliary.loc[0, "payload_json"]
    empty_edges = replace(
        artifact,
        edges=(),
        metadata=_graph_metadata(nodes=2, edges=0),
    )
    assert tuple(graph_edge_frame(empty_edges).columns) == EDGE_AUDIT_COLUMNS
    assert tuple(graph_auxiliary_frame(replace(artifact, auxiliary={})).columns) == (
        AUXILIARY_AUDIT_COLUMNS
    )

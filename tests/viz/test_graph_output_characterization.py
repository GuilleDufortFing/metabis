"""Golden characterization of what the graph pipeline produces, pinned before AM4.

AM4 pulls generation, enrichment, and Bokeh rendering apart and routes every caller
through one plain artifact. The risk that carries is silence: a node class quietly
dropped, drawing order reordered, a reaction recoloured, an auxiliary overlay lost.
None of that fails a type check and none of it fails a mounted smoke reliably, so
it is recorded here first, against the packaged resources the product actually
ships, and asserted for the rest of the migration.

The digests were recorded from the pre-AM4 implementation and are now asserted
against the artifact the single execution returns. They are the evidence that the
new seam preserved the old output rather than merely type-checking.

The digests cover packaged layouts only. A runtime-generated layout depends on the
GraphViz build, so it is not a golden value.
"""

import hashlib

import numpy as np
import pandas as pd
import pytest

from metabis import (
    ActivityConfig,
    AnalysisWorkspace,
    ExperimentAnalysis,
    GroupComparison,
    GroupComparisonSpec,
    MetabolicModel,
    ReactionData,
    ReactionPreprocessingConfig,
)
from metabis.viz.metabolic_visualization import GraphArtifact
from metabis.viz.metabolic_visualization.enrichment.enricher import EnrichmentConfig
from metabis.viz.metabolic_visualization.generation.generator import (
    GraphGenerationConfig,
)
from metabis.viz.metabolic_visualization.generation.utilities.cache_utils import (
    clear_graph_resource_cache,
)
from metabis.viz.metabolic_visualization.pipeline import (
    MetabolicGraphPipeline,
    PipelineConfig,
)

SUBSYSTEM_NAME = "Glycolysis / Gluconeogenesis"
SUBSYSTEM_ID = "glycolysis_gluconeogenesis"

# Golden values recorded on 2026-07-14 from the pre-AM4 implementation.
HUMAN1_STRUCTURE = "97aa9426b748b16100ebfdf9e7ea2bcdc7738d3628b16c271b7fad57ba54f42f"
# STANDARD_STRUCTURE is a coordinate digest of the packaged automatic layout, so
# it is re-recorded whenever that resource is regenerated — 2026-07-25 for the
# connectivity-preserving split, which redraws glycolysis as 10 components with
# 124 nodes in the largest instead of 23 components with 34.
STANDARD_STRUCTURE = "4233e408abb355939b0372d16f2a7f32657abedf48860333edb0158054894725"
# OVERVIEW_STRUCTURE is re-recorded on 2026-10-21 for the spaced overview: the
# packaged positions now seed a layout per category (spaced_network), so every
# node moved, while the nodes, edges and their order stayed.
OVERVIEW_STRUCTURE = "7d375b006e428e2d88f81841d2a3d4fa17c822bed558546004195449981281bf"


def prepare_graph(
    model,
    element_name,
    graph_type,
    *,
    data_manager=None,
    use_human1_map=True,
    layout_engine="dot",
    color_metric="log2_fc",
    hover_distribution_plots=False,
    hover_color_reference=False,
) -> GraphArtifact:
    """Run the one graph execution, which stops before any Bokeh model."""
    pipeline = MetabolicGraphPipeline(
        model,
        PipelineConfig(
            generation_config=GraphGenerationConfig(
                use_human1_map=use_human1_map,
                layout_engine=layout_engine,
            ),
            enrichment_config=EnrichmentConfig(
                color_metric=color_metric,
                show_hover_stats=True,
                show_hover_distribution_plots=hover_distribution_plots,
                show_hover_color_reference=hover_color_reference,
            ),
            show_plot=False,
            enable_timing=True,
        ),
    )
    return pipeline.build_artifact(
        element_name,
        graph_type,
        data_manager=(
            data_manager
            if data_manager is not None
            else AnalysisWorkspace(metabolic_model=model)
        ),
    )


def structure_digest(artifact: GraphArtifact) -> str:
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
            (
                str(edge["id"]),
                str(edge["source"]),
                str(edge["target"]),
                str(edge["connection_class"]),
            )
            for edge in artifact.edges
        ),
    )
    return hashlib.sha256(repr(payload).encode()).hexdigest()


def colour_digest(artifact: GraphArtifact) -> str:
    """What each node is painted, keyed by canonical ID."""
    payload = tuple(
        (str(node["id"]), str(node.get("color")), round(float(node.get("alpha", 1.0)), 3))
        for node in artifact.nodes
    )
    return hashlib.sha256(repr(payload).encode()).hexdigest()


@pytest.fixture(scope="module")
def model():
    return MetabolicModel(use_cache=True)


@pytest.fixture(scope="module")
def structural_manager(model):
    """No experimental data: the graph is browsable before anything is loaded."""
    return AnalysisWorkspace(metabolic_model=model)


@pytest.fixture(scope="module")
def coloured_workspace(model, tmp_path_factory):
    """Two prefix groups over real Human1 glycolysis reactions, with one planted signal.

    Eight samples a side, because the comparison filters on the FDR-adjusted p-value:
    with four a side the smallest attainable p is 0.029, which no correction over
    ~55 reactions can carry below 0.05, and the graph would have nothing to paint.
    """
    reactions = model.get_reactions_for_subsystem(SUBSYSTEM_ID)
    samples = [f"A_{index}" for index in range(8)] + [f"B_{index}" for index in range(8)]
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        rng.normal(5.0, 1.0, size=(len(samples), len(reactions))),
        index=samples,
        columns=reactions,
    )
    frame.loc[frame.index[:8], reactions[0]] += 25.0

    path = tmp_path_factory.mktemp("graph-characterization") / "reactions.tsv"
    frame.to_csv(path, sep="\t")
    reaction_data = ReactionData(
        str(path),
        "single",
        model,
        csv_separator="\t",
        samples_in="rows",
        preprocessing=ReactionPreprocessingConfig(),
    )
    manager = AnalysisWorkspace(metabolic_model=model, reaction_data=reaction_data)
    manager.load_groups(criteria="sample_prefix")
    analysis = ExperimentAnalysis(manager, auto_run=False)
    analysis.run_analysis(include_pca=False)
    provenance = manager.capture_analysis_snapshot().provenance
    manager.install_experiment_analysis(provenance, analysis)
    manager.refresh_experiment_analysis_presentation()
    return {"manager": manager, "planted": reactions[0]}


@pytest.mark.integration
def test_the_packaged_human1_subsystem_keeps_its_structure(model, structural_manager):
    artifact = prepare_graph(
        model, SUBSYSTEM_NAME, "subsystem", data_manager=structural_manager
    )

    assert artifact.resource_variant == "human1"
    assert artifact.used_runtime_layout is False
    assert artifact.warnings == ()
    assert artifact.display_title == SUBSYSTEM_NAME
    assert artifact.node_count == 348
    assert artifact.edge_count == 230
    assert {node["glyph_class"] for node in artifact.nodes} == {
        "compartment",
        "enzyme_group",
        "gene",
        "metabolite",
        "reaction",
    }
    assert structure_digest(artifact) == HUMAN1_STRUCTURE


@pytest.mark.integration
def test_the_packaged_standard_subsystem_layout_is_a_different_graph(
    model, structural_manager
):
    """Disabling Human1 must fall back to the packaged standard layout, not regenerate one."""
    artifact = prepare_graph(
        model,
        SUBSYSTEM_NAME,
        "subsystem",
        data_manager=structural_manager,
        use_human1_map=False,
    )

    assert artifact.resource_variant == "standard"
    assert artifact.used_runtime_layout is False
    assert structure_digest(artifact) == STANDARD_STRUCTURE
    assert STANDARD_STRUCTURE != HUMAN1_STRUCTURE


# Subsystems with no curated Human1 map — the case D-60 is about. `retinol` never
# had one; `hippurate` and `drug_metabolism` shipped no automatic layout at all
# until this slice repackaged them, so they exercise the "was runtime, now a
# packaged standard resource" path as well.
NO_HUMAN1_MAP_SUBSYSTEMS = (
    "retinol_metabolism",
    "hippurate_metabolism",
    "drug_metabolism",
)


@pytest.mark.integration
@pytest.mark.parametrize("subsystem_id", NO_HUMAN1_MAP_SUBSYSTEMS)
def test_a_no_human1_map_subsystem_renders_from_its_packaged_layout(
    model, structural_manager, subsystem_id
):
    """D-60: a subsystem with no curated map renders its packaged layout, not a failure.

    Human1 is *requested* (the default), but these subsystems have no curated map,
    so the pipeline serves the repackaged automatic resource — a real graph with
    nodes and edges, laid out ahead of time rather than at request. This is the
    coverage the amendment asks for: no-Human1-map subsystem resources render
    rather than fail.
    """
    artifact = prepare_graph(
        model,
        subsystem_id,
        "subsystem",
        data_manager=structural_manager,
        use_human1_map=True,
    )

    assert artifact.resource_variant == "standard"
    assert artifact.used_runtime_layout is False
    assert artifact.node_count > 0
    assert artifact.edge_count > 0


@pytest.mark.integration
def test_the_overview_network_keeps_its_structure(model, structural_manager):
    artifact = prepare_graph(
        model,
        None,
        "metabolic_network",
        data_manager=structural_manager,
        layout_engine="fdp",
    )

    assert artifact.display_title == "Complete Metabolic Network"
    assert artifact.node_count == 165
    assert artifact.edge_count == 374
    # The overview draws subsystems and their categories, never single reactions.
    assert {node["glyph_class"] for node in artifact.nodes} == {
        "subsystem",
        "subsystem_category",
    }
    assert structure_digest(artifact) == OVERVIEW_STRUCTURE


@pytest.mark.integration
def test_node_and_edge_ids_are_unique_and_edges_land_on_real_nodes(
    model, structural_manager
):
    """The export adapter keys on these IDs, so uniqueness and closure are contracts."""
    for element_name, graph_type, engine in (
        (SUBSYSTEM_NAME, "subsystem", "dot"),
        (None, "metabolic_network", "fdp"),
    ):
        artifact = prepare_graph(
            model,
            element_name,
            graph_type,
            data_manager=structural_manager,
            layout_engine=engine,
        )
        node_ids = artifact.node_ids()
        edge_ids = [edge["id"] for edge in artifact.edges]

        assert len(set(node_ids)) == len(node_ids)
        assert len(set(edge_ids)) == len(edge_ids)
        for edge in artifact.edges:
            assert edge["source"] in set(node_ids)
            assert edge["target"] in set(node_ids)


@pytest.mark.integration
def test_a_cached_resource_produces_the_same_graph_as_a_cold_one(
    model, structural_manager
):
    """The resource cache hands out copies, so a second render cannot inherit a mutation."""
    clear_graph_resource_cache()
    cold = prepare_graph(model, SUBSYSTEM_NAME, "subsystem", data_manager=structural_manager)
    warm = prepare_graph(model, SUBSYSTEM_NAME, "subsystem", data_manager=structural_manager)

    assert structure_digest(cold) == structure_digest(warm)
    assert colour_digest(cold) == colour_digest(warm)


@pytest.mark.integration
def test_activity_and_comparison_paint_the_same_structure_differently(
    model, coloured_workspace, install_comparison
):
    """The two coloring contexts answer different questions and must not be interchangeable."""
    manager = coloured_workspace["manager"]

    activity_context = manager.activate_activity_mode(ActivityConfig(group_id="A"))
    activity = prepare_graph(
        model,
        SUBSYSTEM_NAME,
        "subsystem",
        data_manager=manager,
        color_metric=activity_context.metric_key,
    )

    comparison = GroupComparison(
        spec=GroupComparisonSpec.pairwise(("A",), ("B",)),
        test="ttest", correction="fdr_bh",
    )
    comparison.run_test(manager, for_genes=False)
    comparison.define_significant_elements(
        pval_threshold=0.05,
        cohens_d_threshold=0.0,
        log2_fc_threshold=0.0,
        mean_difference_threshold=0.0,
        for_genes=False,
    )
    install_comparison(manager, comparison)
    compared = prepare_graph(
        model,
        SUBSYSTEM_NAME,
        "subsystem",
        data_manager=manager,
        color_metric="cohens_d",
    )

    # Same resource, so the same nodes in the same places...
    assert structure_digest(activity) == structure_digest(compared)
    # ...painted from different evidence, and each artifact says which.
    assert colour_digest(activity) != colour_digest(compared)
    assert activity.color.mode == "experiment_analysis"
    assert compared.color.mode == "group_comparison"
    assert compared.color.metric == "cohens_d"
    assert comparison.significant_reactions


@pytest.mark.integration
def test_hover_detail_changes_hover_content_and_nothing_structural(
    model, coloured_workspace
):
    """Hover detail is a presentation choice: it may not move a node or change a colour."""
    manager = coloured_workspace["manager"]
    manager.activate_activity_mode(ActivityConfig(group_id="A"))

    light = prepare_graph(
        model,
        SUBSYSTEM_NAME,
        "subsystem",
        data_manager=manager,
        hover_distribution_plots=False,
        hover_color_reference=False,
    )
    full = prepare_graph(
        model,
        SUBSYSTEM_NAME,
        "subsystem",
        data_manager=manager,
        hover_distribution_plots=True,
        hover_color_reference=True,
    )

    assert structure_digest(light) == structure_digest(full)
    assert colour_digest(light) == colour_digest(full)
    assert light.hover_detail == "light"
    assert full.hover_detail == "full"

    def hovers(artifact):
        return [node["hover"] for node in artifact.nodes_of_class("reaction")]

    assert hovers(light) != hovers(full)
    assert all(hover for hover in hovers(light))

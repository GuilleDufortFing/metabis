"""Freeze the supported public interface described by ADR-0008.

This test is the contract: the root ``__all__`` is the supported flat surface,
the seams a coordinator installs and reads are documented public modules with
their own ``__all__``, and everything else is internal. A change to the public
surface has to change this file deliberately.
"""

import importlib
import inspect

import metabis

# The supported flat root surface. `from metabis import *` yields exactly this.
# Adding or removing a root export is a public-interface change and must edit
# this set deliberately.
EXPECTED_ROOT_ALL = frozenset(
    {
        # Domain inputs
        "MetabolicModel",
        "ReactionData",
        "ExpressionData",
        "SampleMetadata",
        "MetadataAlignmentReport",
        "check_subsystem_categories_file",
        "detect_expression_csv_params",
        "detect_reaction_csv_params",
        # Preprocessing
        "COMPASS_MINIMUM_RANGE",
        "ExpressionInputState",
        "ExpressionPreprocessingConfig",
        "PreprocessingReport",
        "ReactionPreprocessingConfig",
        "ReactionTransform",
        "blocked_compass_reactions",
        "compass_negative_log",
        "estimate_dense_bytes",
        "filter_features_by_range",
        "partially_blocked_compass_reactions",
        "DetectedEmbedding",
        "EmbeddingAlignmentError",
        "build_h5ad_reduction_artifact",
        "discover_h5ad_embeddings",
        "preprocess_reactions",
        "quantile_normalize",
        "rank_percentile_by_feature",
        "signed_cuberoot",
        # Analysis workspace facade and entry points
        "AnalysisWorkspace",
        "AnalysisWorkspaceMode",
        "ExperimentAnalysis",
        # Group and comparison identity
        "SampleGroup",
        "GroupDefinition",
        "GroupLoadResult",
        "MetadataPredicate",
        "GroupComparison",
        "ComparisonMembership",
        "ComparisonMode",
        "ComparisonSideSpec",
        "GroupComparisonSpec",
        "each_against_the_rest",
        "AdjustmentColumn",
        "ComparisonAdjustment",
        "AGREEMENT_CLASSES",
        "Agreement",
        "ResultEffects",
        "compare_results",
        "CONCORDANCE_CLASSES",
        "reaction_gene_concordance",
        "DEFAULT_LANDSCAPE_DELTA",
        "EnrichmentLandscape",
        "Landscape",
        "build_landscape",
        "enrichment_landscape",
        "marker_reactions",
        "select_enriched_subsystems",
        "select_landscape_subsystems",
        "select_shared_subsystems",
        "subsystem_delta",
        "with_thresholds",
        "group_means",
        "pairwise_markers",
        "top_markers",
        "plot_delta_landscape",
        "plot_enrichment_landscape",
        "plot_marker_heatmap",
        "plot_sample_heatmap",
        "sample_heatmap_block",
        "plot_reaction_gene_concordance",
        "plot_result_agreement",
        # Typed analysis results
        "ReductionKey",
        "ReductionResult",
        "ActivityConfig",
        "ActivityContext",
        "ActivityOverlay",
        "FeatureActivityResult",
        "GroupSummary",
        "SubsystemActivityRow",
        "calculate_feature_activity",
        "calculate_subsystem_activity",
        "displayed_feature_id",
        "resolve_display_direction",
        # Graph metadata
        "GraphPipelineError",
        "GraphRunMetadata",
        "PipelinePhaseMetric",
        "plot_metabolic_graph",
        # Plotting
        "COMPARTMENT_MAPS",
        "PLOT_THEME",
        "add_category_key",
        "add_map_bands",
        "add_map_find",
        "add_map_links",
        "add_network_focus_link",
        "add_pathway_key",
        "apply_academic_style",
        "band_name",
        "category_numbers",
        "highlight_reactions",
        "is_transport_subsystem",
        "map_bands",
        "map_ids_for_reactions",
        "map_reaction_ids",
        "map_subsystems",
        "maps_for_reactions",
        "plot_transport_view",
        "overview_color_bar",
        "registered_color_bar",
        "theme",
        "build_external_link_formatter",
        "build_statistical_number_formatter",
        "build_reduction_presentation",
        "create_statistical_results_table",
        "plot_comparison_histogram",
        "plot_comparison_scatter",
        "plot_gene_reaction_influence_heatmap",
        "plot_pca_biplot_artifact",
        "plot_pca_components_artifact",
        "plot_pca_loadings_artifact",
        "plot_pca_variance_artifact",
        "plot_reaction_set_enrichment",
        "plot_set_enrichment_category",
        "plot_reaction_violins",
        "plot_reduction_artifact",
        "reduction_colouring",
        "add_arrow_colouring",
        "biplot_arrow_features",
        "FEATURE_CHOICE",
        "FEATURE_ROWS",
        "ReductionColouring",
        "plot_statistical_histograms",
        "plot_statistical_scatterplots",
        "statistical_results_columns",
        "statistical_results_rows",
        "plot_subsystem_activity_ranking",
        "plot_subsystem_comparison",
        "plot_subsystem_influence_summary",
        "plot_subsystem_ranking_artifact",
        "plot_subsystem_ranking_by_influence",
        "ReductionFeaturePresentation",
        "ReductionGroupPresentation",
        "ReductionPlotPresentation",
        "ReductionSamplePresentation",
        "RankingFeaturePresentation",
        "SubsystemRankingPresentation",
    }
)


# Each documented public seam module -> the names a coordinator/server is
# entitled to import from it. The module's own __all__ must be a superset; a
# name here that leaves the module's __all__ is a broken promise.
PUBLIC_SEAM_REQUIRED = {
    "metabis.analysis_workspace": {
        "AnalysisSnapshot",
        "WorkspaceProvenance",
        "StableFrameReference",
        "RevisionKind",
        "StaleAnalysisResultError",
    },
    "metabis.analysis.activity_analysis": {"ActivityArtifact"},
    "metabis.analysis.artifacts": {
        "AnalysisArtifactMetadata",
        "ArtifactDimension",
        "ArtifactKind",
        "ArtifactProvenance",
        "PlainTable",
        "fingerprint_frame",
        "fingerprint_json",
    },
    "metabis.analysis.comparison_artifact": {
        "ComparisonThresholds",
        "GroupComparisonArtifact",
        "MetricSummary",
        "comparison_export_frame",
    },
    "metabis.analysis.dimensionality_reduction": {
        "DimensionalityReduction",
        "ReductionArtifact",
    },
    "metabis.analysis.h5ad_embeddings": {
        "EMBEDDING_METHODS",
        "DetectedEmbedding",
        "EmbeddingAlignmentError",
        "UnknownEmbeddingError",
        "build_h5ad_reduction_artifact",
        "discover_h5ad_embeddings",
    },
    "metabis.analysis.group_exports": {
        "COMPARISON_MEMBERSHIP_EXPORT_COLUMNS",
        "SAMPLE_GROUP_EXPORT_COLUMNS",
        "comparison_membership_frame",
        "sample_group_fingerprint",
        "sample_groups_frame",
    },
    "metabis.analysis.subsystem_analysis": {"SubsystemDeltaArtifact"},
    "metabis.analysis.experiment_analysis": {"ExperimentSummarySnapshot"},
    "metabis.analysis.statistical_analysis": {"CORRECTION_METHODS"},
    "metabis.analysis.reaction_set_enrichment": {
        "ENRICHMENT_RESULT_COLUMNS",
        "EnrichmentDirection",
        "ReactionSetEnrichmentArtifact",
        "ReactionSetEnrichmentConfig",
        "ReactionSetUniverse",
        "SubsystemAnnotation",
        "build_reaction_set_universe",
        "build_subsystem_annotations",
        "run_reaction_set_enrichment",
        "to_base_reaction_id",
    },
    "metabis.analysis.reaction_direction": {
        "ReactionDirectionCriterion",
        "ReactionDirectionMap",
        "ReactionDirectionSelection",
        "ReactionSourceDirection",
        "build_reaction_direction_map",
        "to_base_reaction_id",
    },
    "metabis.analysis.set_enrichment": {
        "SET_ENRICHMENT_ARTIFACT_RESULT_COLUMNS",
        "SET_ENRICHMENT_RESULT_COLUMNS",
        "SetEnrichmentAnnotation",
        "SetEnrichmentArtifact",
        "SetEnrichmentConfig",
        "SetEnrichmentDirection",
        "SetEnrichmentFeatureType",
        "SetEnrichmentResult",
        "SetEnrichmentUniverse",
        "build_comparison_set_enrichment_universe",
        "build_gene_set_annotations",
        "build_set_enrichment_annotations",
        "run_comparison_set_enrichment",
        "run_gene_set_enrichment",
        "run_paired_reaction_set_enrichment",
        "run_set_enrichment",
    },
    "metabis.domain.expression_matrix": {"DenseExpressionMatrix"},
    "metabis.viz.click_behavior": {"add_click_behavior"},
    "metabis.viz.metabolic_visualization": {
        "GraphArtifact",
        "GraphCalculationInput",
        "MetabolicGraphPipeline",
        "PipelineConfig",
        "EnrichmentConfig",
        "GraphGenerationConfig",
        "RenderingConfig",
        "render_graph_artifact",
        # Promoted from generation.utilities so no caller reaches into the
        # private generation package.
        "DEFAULT_SUBSYSTEM_LAYOUT_ENGINE",
        "SUBSYSTEM_HUMAN1_VARIANT",
        "SUBSYSTEM_STANDARD_VARIANT",
        "get_available_subsystem_graph_resource_paths",
    },
}


def test_root_all_is_frozen_to_the_supported_surface():
    assert set(metabis.__all__) == EXPECTED_ROOT_ALL


def test_root_all_has_no_duplicates():
    assert len(metabis.__all__) == len(set(metabis.__all__))


def test_star_import_yields_exactly_root_all():
    namespace: dict = {}
    exec("from metabis import *", namespace)
    exported = {name for name in namespace if not name.startswith("__")}
    assert exported == set(metabis.__all__)


def test_every_root_export_is_importable():
    missing = [name for name in metabis.__all__ if not hasattr(metabis, name)]
    assert not missing, f"root __all__ names not importable: {missing}"


def test_public_seams_declare_all_and_expose_required_names():
    problems = []
    for module_name, required in PUBLIC_SEAM_REQUIRED.items():
        module = importlib.import_module(module_name)
        exported = getattr(module, "__all__", None)
        if exported is None:
            problems.append(f"{module_name} declares no __all__")
            continue
        exported_set = set(exported)
        missing_attr = [n for n in exported if not hasattr(module, n)]
        if missing_attr:
            problems.append(
                f"{module_name} __all__ names not importable: {missing_attr}"
            )
        not_promised = required - exported_set
        if not_promised:
            problems.append(
                f"{module_name} __all__ omits required public names: {sorted(not_promised)}"
            )
    assert not problems, "\n".join(problems)


def test_promoted_artifact_seams_expose_exactly_the_documented_names():
    promoted_modules = {
        "metabis.analysis.artifacts",
        "metabis.analysis.comparison_artifact",
        "metabis.analysis.group_exports",
        "metabis.analysis.h5ad_embeddings",
        "metabis.analysis.reaction_set_enrichment",
        "metabis.analysis.set_enrichment",
    }

    for module_name in promoted_modules:
        module = importlib.import_module(module_name)
        assert set(module.__all__) == PUBLIC_SEAM_REQUIRED[module_name]


def test_workspace_compatibility_update_shortcuts_are_removed():
    workspace = metabis.AnalysisWorkspace()

    assert not hasattr(workspace, "update_experiment_analysis")
    assert not hasattr(workspace, "update_group_comparison")


def test_analysis_workspace_constructor_has_no_legacy_experiment_slot():
    parameters = inspect.signature(metabis.AnalysisWorkspace).parameters

    assert "experiment" not in parameters
    assert tuple(parameters)[:4] == (
        "metabolic_model",
        "reaction_data",
        "expression_data",
        "experiment_analysis",
    )


def test_subsystem_resource_helpers_are_public_without_the_private_path():
    from metabis.viz.metabolic_visualization import (
        get_available_subsystem_graph_resource_paths,
    )

    assert callable(get_available_subsystem_graph_resource_paths)

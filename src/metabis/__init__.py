"""Public interface of the metabis library.

Every supported root name is bound **lazily** (PEP 562). Importing ``metabis``,
or any module inside it, therefore costs nothing beyond this file until a caller
actually reaches for a name.

Lazy binding preserves the analysis/rendering boundary described by ADR-0006.
Analysis and domain modules remain importable without Bokeh, while reaching a
plotting name imports its rendering dependencies on demand. This keeps Graph
Artifact calculation usable in workers that do not own presentation objects.

Reaching a name still imports its module, so ``metabis.plot_reduction_artifact``
imports Bokeh exactly as before. ``__all__`` remains the exact supported surface,
and ``from metabis import *`` yields exactly those names.
"""

import importlib
import os
from typing import TYPE_CHECKING

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Supported name -> the module that defines it. Grouped by module so a new
# export is added beside its siblings; `_NAME_TO_MODULE` inverts it below.
_MODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    ".analysis.activity_analysis": (
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
    ),
    ".analysis.adjustment": ("AdjustmentColumn", "ComparisonAdjustment"),
    ".analysis.comparison_membership": (
        "ComparisonMembership",
        "ComparisonMode",
        "ComparisonSideSpec",
        "GroupComparisonSpec",
    ),
    ".analysis.dimensionality_reduction": ("ReductionKey", "ReductionResult"),
    ".analysis.h5ad_embeddings": (
        "DetectedEmbedding",
        "EmbeddingAlignmentError",
        "build_h5ad_reduction_artifact",
        "discover_h5ad_embeddings",
    ),
    ".analysis.experiment_analysis": ("ExperimentAnalysis",),
    ".analysis.group_comparison": ("GroupComparison",),
    ".domain.expression_data": ("ExpressionData",),
    ".domain.format_detection": (
        "detect_expression_csv_params",
        "detect_reaction_csv_params",
    ),
    ".domain.metabolic_model": ("MetabolicModel",),
    ".domain.metabolic_model.loaders": ("check_subsystem_categories_file",),
    ".domain.preprocessing": (
        "COMPASS_MINIMUM_RANGE",
        "ExpressionInputState",
        "ExpressionPreprocessingConfig",
        "PreprocessingReport",
        "ReactionPreprocessingConfig",
        "ReactionTransform",
        "blocked_compass_reactions",
        "compass_negative_log",
        "partially_blocked_compass_reactions",
        "estimate_dense_bytes",
        "filter_features_by_range",
        "preprocess_reactions",
        "quantile_normalize",
        "rank_percentile_by_feature",
        "signed_cuberoot",
    ),
    ".domain.reaction_data": ("ReactionData",),
    ".domain.sample_metadata": ("MetadataAlignmentReport", "SampleMetadata"),
    ".managers.group_manager": (
        "GroupDefinition",
        "GroupLoadResult",
        "MetadataPredicate",
        "SampleGroup",
    ),
    ".managers.workspace": ("AnalysisWorkspace", "AnalysisWorkspaceMode"),
    # Graph metadata only. The artifact seam itself is the public module
    # metabis.viz.metabolic_visualization (ADR-0008).
    ".viz.metabolic_visualization.pipeline": (
        "GraphPipelineError",
        "GraphRunMetadata",
        "PipelinePhaseMetric",
    ),
    ".viz.plot_comparison_artifact": (
        "plot_comparison_histogram",
        "plot_comparison_scatter",
    ),
    ".viz.plot_gene_reaction_influence": (
        "plot_gene_reaction_influence_heatmap",
        "plot_subsystem_influence_summary",
        "plot_subsystem_ranking_by_influence",
    ),
    ".viz.plot_metabolic_graph": ("plot_metabolic_graph",),
    ".viz.plot_reaction_set_enrichment": ("plot_reaction_set_enrichment",),
    ".viz.plot_set_enrichment": ("plot_set_enrichment_category",),
    ".viz.plot_reduction_artifact": (
        "ReductionFeaturePresentation",
        "ReductionGroupPresentation",
        "ReductionPlotPresentation",
        "ReductionSamplePresentation",
        "build_reduction_presentation",
        "plot_pca_biplot_artifact",
        "plot_pca_loadings_artifact",
        "plot_pca_variance_artifact",
        "plot_reduction_artifact",
    ),
    ".viz.plot_statistical_distributions": (
        "create_statistical_results_table",
        "plot_statistical_histograms",
        "plot_statistical_scatterplots",
        "statistical_results_columns",
        "statistical_results_rows",
    ),
    ".viz.plot_subsystem_artifact_ranking": (
        "RankingFeaturePresentation",
        "SubsystemRankingPresentation",
        "plot_subsystem_ranking_artifact",
    ),
    ".viz.plot_subsystem_ranking": (
        "plot_subsystem_activity_ranking",
        "plot_subsystem_comparison",
    ),
    ".viz.color_bars": (
        "add_category_key",
        "add_pathway_key",
        "overview_color_bar",
        "registered_color_bar",
    ),
    ".viz.plot_utils": ("PLOT_THEME", "apply_academic_style"),
    ".viz.subsystem_link": ("add_network_focus_link",),
    ".viz.transport_view": ("is_transport_subsystem", "plot_transport_view"),
    ".viz.compartment_maps": (
        "COMPARTMENT_MAPS",
        "add_map_bands",
        "add_map_find",
        "add_map_links",
        "band_name",
        "highlight_reactions",
        "map_bands",
        "map_ids_for_reactions",
        "map_reaction_ids",
        "map_subsystems",
        "maps_for_reactions",
    ),
    ".viz.metabolic_visualization.generation.graph_building.spaced_network": (
        "category_numbers",
    ),
    ".visual_config": ("theme",),
    ".viz.plot_violins": ("plot_reaction_violins",),
    ".viz.table_formatting": (
        "build_external_link_formatter",
        "build_statistical_number_formatter",
    ),
}

_NAME_TO_MODULE: dict[str, str] = {
    name: module for module, names in _MODULE_EXPORTS.items() for name in names
}


def __getattr__(name: str):
    """Import and cache a supported root name on first access (PEP 562)."""
    module_path = _NAME_TO_MODULE.get(name)
    if module_path is None:
        raise AttributeError(f"module 'metabis' has no attribute '{name}'")
    value = getattr(importlib.import_module(module_path, __name__), name)
    # Cache in the module namespace so later lookups skip __getattr__ entirely.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*__all__, "PROJECT_ROOT"])


if TYPE_CHECKING:
    # Lazy binding is invisible to type checkers and IDEs, so the same names are
    # declared statically here. This block never runs.
    from .analysis.activity_analysis import (
        ActivityConfig,
        ActivityContext,
        ActivityOverlay,
        FeatureActivityResult,
        GroupSummary,
        SubsystemActivityRow,
        calculate_feature_activity,
        calculate_subsystem_activity,
        displayed_feature_id,
        resolve_display_direction,
    )
    from .analysis.adjustment import AdjustmentColumn, ComparisonAdjustment
    from .analysis.comparison_membership import (
        ComparisonMembership,
        ComparisonMode,
        ComparisonSideSpec,
        GroupComparisonSpec,
    )
    from .analysis.dimensionality_reduction import ReductionKey, ReductionResult
    from .analysis.experiment_analysis import ExperimentAnalysis
    from .analysis.group_comparison import GroupComparison
    from .analysis.h5ad_embeddings import (
        DetectedEmbedding,
        EmbeddingAlignmentError,
        build_h5ad_reduction_artifact,
        discover_h5ad_embeddings,
    )
    from .domain.expression_data import ExpressionData
    from .domain.format_detection import (
        detect_expression_csv_params,
        detect_reaction_csv_params,
    )
    from .domain.metabolic_model import MetabolicModel
    from .domain.metabolic_model.loaders import check_subsystem_categories_file
    from .domain.preprocessing import (
        COMPASS_MINIMUM_RANGE,
        ExpressionInputState,
        ExpressionPreprocessingConfig,
        PreprocessingReport,
        ReactionPreprocessingConfig,
        ReactionTransform,
        blocked_compass_reactions,
        compass_negative_log,
        estimate_dense_bytes,
        filter_features_by_range,
        partially_blocked_compass_reactions,
        preprocess_reactions,
        quantile_normalize,
        rank_percentile_by_feature,
        signed_cuberoot,
    )
    from .domain.reaction_data import ReactionData
    from .domain.sample_metadata import MetadataAlignmentReport, SampleMetadata
    from .managers.group_manager import (
        GroupDefinition,
        GroupLoadResult,
        MetadataPredicate,
        SampleGroup,
    )
    from .managers.workspace import AnalysisWorkspace, AnalysisWorkspaceMode
    from .viz.metabolic_visualization.pipeline import (
        GraphPipelineError,
        GraphRunMetadata,
        PipelinePhaseMetric,
    )
    from .viz.plot_comparison_artifact import (
        plot_comparison_histogram,
        plot_comparison_scatter,
    )
    from .viz.plot_gene_reaction_influence import (
        plot_gene_reaction_influence_heatmap,
        plot_subsystem_influence_summary,
        plot_subsystem_ranking_by_influence,
    )
    from .viz.plot_metabolic_graph import plot_metabolic_graph
    from .viz.plot_reaction_set_enrichment import plot_reaction_set_enrichment
    from .viz.plot_reduction_artifact import (
        ReductionFeaturePresentation,
        ReductionGroupPresentation,
        ReductionPlotPresentation,
        ReductionSamplePresentation,
        build_reduction_presentation,
        plot_pca_biplot_artifact,
        plot_pca_loadings_artifact,
        plot_pca_variance_artifact,
        plot_reduction_artifact,
    )
    from .viz.plot_set_enrichment import plot_set_enrichment_category
    from .viz.plot_statistical_distributions import (
        create_statistical_results_table,
        plot_statistical_histograms,
        plot_statistical_scatterplots,
        statistical_results_columns,
        statistical_results_rows,
    )
    from .viz.plot_subsystem_artifact_ranking import (
        RankingFeaturePresentation,
        SubsystemRankingPresentation,
        plot_subsystem_ranking_artifact,
    )
    from .viz.plot_subsystem_ranking import (
        plot_subsystem_activity_ranking,
        plot_subsystem_comparison,
    )
    from .visual_config import theme
    from .viz.color_bars import (
        add_category_key,
        add_pathway_key,
        overview_color_bar,
        registered_color_bar,
    )
    from .viz.plot_utils import PLOT_THEME, apply_academic_style
    from .viz.plot_violins import plot_reaction_violins
    from .viz.subsystem_link import add_network_focus_link
    from .viz.transport_view import is_transport_subsystem, plot_transport_view
    from .viz.compartment_maps import (
        COMPARTMENT_MAPS,
        add_map_bands,
        add_map_find,
        add_map_links,
        band_name,
        highlight_reactions,
        map_bands,
        map_ids_for_reactions,
        map_reaction_ids,
        map_subsystems,
        maps_for_reactions,
    )
    from .viz.metabolic_visualization.generation.graph_building.spaced_network import (
        category_numbers,
    )
    from .viz.table_formatting import (
        build_external_link_formatter,
        build_statistical_number_formatter,
    )

# The supported flat public surface. `from metabis import *` yields exactly
# these names. Seams a coordinator installs and reads (the Analysis Workspace,
# the graph artifact, the analysis/matrix artifacts) live in their own
# documented public modules, each with its own ``__all__``; see ADR-0008.
__all__ = [
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
    "AdjustmentColumn",
    "ComparisonAdjustment",
    # Typed analysis results
    "ReductionKey",
    "ReductionResult",
    # Precomputed h5ad embeddings
    "DetectedEmbedding",
    "EmbeddingAlignmentError",
    "build_h5ad_reduction_artifact",
    "discover_h5ad_embeddings",
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
    # Graph metadata (the artifact seam itself is
    # metabis.viz.metabolic_visualization)
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
    "band_name",
    "add_network_focus_link",
    "add_pathway_key",
    "apply_academic_style",
    "category_numbers",
    "highlight_reactions",
    "is_transport_subsystem",
    "map_bands",
    "map_ids_for_reactions",
    "map_reaction_ids",
    "map_subsystems",
    "maps_for_reactions",
    "overview_color_bar",
    "plot_transport_view",
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
    "plot_pca_loadings_artifact",
    "plot_pca_variance_artifact",
    "plot_reaction_set_enrichment",
    "plot_set_enrichment_category",
    "plot_reaction_violins",
    "plot_reduction_artifact",
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
]

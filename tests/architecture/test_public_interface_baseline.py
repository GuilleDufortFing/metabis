"""Pin the public and compatibility interface baseline."""

from dataclasses import fields

import metabis
from metabis import (
    ActivityConfig,
    ActivityContext,
    GraphRunMetadata,
    MetadataAlignmentReport,
    PreprocessingReport,
)

EXPECTED_ROOT_EXPORTS = {
    "ActivityConfig",
    "ActivityContext",
    "AnalysisWorkspace",
    "ExpressionData",
    "GraphRunMetadata",
    "GroupComparison",
    "GroupComparisonSpec",
    "MetadataAlignmentReport",
    "MetabolicModel",
    "PreprocessingReport",
    "ReactionData",
    "ReductionKey",
    "ReductionResult",
    "SampleMetadata",
    "plot_metabolic_graph",
    "preprocess_reactions",
}


def test_current_root_exports_remain_importable():
    assert EXPECTED_ROOT_EXPORTS <= set(dir(metabis))


def test_typed_result_field_shapes_are_characterized():
    assert tuple(field.name for field in fields(PreprocessingReport)) == (
        "method",
        "parameters",
        "input_shape",
        "output_shape",
        "zeros_before",
        "zeros_after",
        "non_finite_before",
        "non_finite_after",
        "filtered_feature_ids",
        "warnings",
        # Storage-mode provenance records how the input was stored and any dense
        # conversion the load performed.
        "input_storage",
        "conversions",
        # Removed features that were blocked by Compass are distinct from merely
        # invariant features, so presentation can explain them instead of
        # rendering them as missing data. The complement is
        # `invariant_feature_ids`.
        "blocked_feature_ids",
    )
    assert tuple(field.name for field in fields(MetadataAlignmentReport)) == (
        "metadata_samples",
        "reaction_samples",
        "expression_samples",
        "aligned_samples",
        "missing_from_metadata",
        "metadata_only",
        "missing_from_expression",
        "duplicate_ids",
    )
    assert tuple(field.name for field in fields(ActivityContext)) == (
        "config",
        "reaction_features",
        "reaction_subsystems",
        "gene_features",
        "gene_subsystems",
        "dataset_revision",
        "group_revision",
        "reaction_reference_summary",
        "gene_reference_summary",
    )
    assert tuple(field.name for field in fields(GraphRunMetadata)) == (
        "graph_type",
        "element_id",
        "display_title",
        "resource_path",
        "resource_variant",
        "layout_engine",
        "used_runtime_layout",
        "glyph_count",
        "connection_count",
        "bokeh_model_count",
        "hover_detail",
        "phases",
        "warnings",
    )


def test_activity_config_identity_contains_every_current_scoring_input():
    config = ActivityConfig(group_id="group-a")

    assert config.metric_key == "activity|group-a|mean|coverage=0.1"
    assert config.metric_key_for("reaction") == (
        "activity|group-a|mean|coverage=0.1|family=reaction"
    )

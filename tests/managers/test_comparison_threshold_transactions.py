"""Workspace transaction guarantees for comparison threshold changes.

AM7 deleted the mutable comparison view, so the installed artifact is the single
source of truth. A threshold change recomputes significance against the
artifact's own result frame and installs a replacement artifact; these tests
assert the same transactional guarantees against that seam.
"""

import pandas as pd
import pytest

from metabis import (
    AnalysisWorkspace,
    AnalysisWorkspaceMode,
    GroupComparison,
    GroupComparisonSpec,
)
from metabis.managers.group_manager import GroupDefinition

SAMPLES = ["A1", "A2", "B1", "B2"]


def _results(prefix):
    return pd.DataFrame(
        {
            "adjusted_pval": [0.01, 0.2],
            "pval": [0.005, 0.1],
            "cohens_d": [1.5, 0.2],
            "log2_fc": [2.0, 0.1],
            "mean_difference": [3.0, 0.2],
            "reversed": [False, False],
            "group_1": ["A", "A"],
            "group_2": ["B", "B"],
        },
        index=[f"{prefix}1", f"{prefix}2"],
    )


def _manager():
    comparison = GroupComparison(spec=GroupComparisonSpec.pairwise(("A",), ("B",)))
    comparison._reaction_test_results = _results("R")
    comparison._gene_test_results = _results("G")
    comparison.define_significant_elements(
        pval_threshold=1.0,
        cohens_d_threshold=0.0,
        log2_fc_threshold=0.0,
        mean_difference_threshold=0.0,
        for_genes=False,
    )
    comparison.define_significant_elements(
        pval_threshold=1.0,
        cohens_d_threshold=0.0,
        log2_fc_threshold=0.0,
        mean_difference_threshold=0.0,
        for_genes=True,
    )
    manager = AnalysisWorkspace()
    manager.group_manager.define_group(
        "A", ["A1", "A2"], definition=GroupDefinition("custom"), all_sample_ids=SAMPLES
    )
    manager.group_manager.define_group(
        "B", ["B1", "B2"], definition=GroupDefinition("custom"), all_sample_ids=SAMPLES
    )
    comparison.membership = manager.group_manager.resolve_comparison_membership(
        comparison.spec, SAMPLES
    )

    snapshot = manager.capture_analysis_snapshot()
    artifact = comparison.build_artifact(
        provenance=snapshot.provenance,
        source_fingerprints=manager.source_fingerprints,
    )
    manager.install_group_comparison_artifact(snapshot.provenance, artifact)
    manager.refresh_group_comparison_presentation()

    summaries = object()
    manager.experiment_analysis = summaries
    manager.data_provider.experiment_analysis = summaries
    # Simulate stale threshold-dependent presentation state the apply must clear.
    manager.subsystem_analysis._base_stats_cache = {
        ("old", False): {"significant_elements": 99}
    }
    manager.subsystem_analysis._delta_stats_cache = {
        ("old", False): {"delta_subsystem_score": 99.0}
    }
    manager.cache_manager.cache_hover("old-thresholds", "R1", "stale")
    return manager, manager.get_group_comparison_artifact(), summaries


@pytest.mark.parametrize(
    ("for_genes", "expected"),
    ((False, {"R1"}), (True, {"G1"})),
)
def test_successful_threshold_apply_increments_revision_exactly_once(
    for_genes,
    expected,
):
    manager, _artifact, _summaries = _manager()
    revision_before = manager.threshold_revision

    significant, info = manager.apply_comparison_thresholds(
        pval_threshold=0.05,
        cohens_d_threshold=1.0,
        log2_fc_threshold=1.0,
        mean_difference_threshold=1.0,
        for_genes=for_genes,
    )

    assert significant == expected
    assert info["significant_count"] == 1
    assert manager.threshold_revision == revision_before + 1
    assert (
        manager.get_group_comparison_artifact().get_significant_elements(
            for_genes=for_genes
        )
        == expected
    )


def test_threshold_apply_preserves_base_results_and_group_summaries():
    manager, artifact, summaries = _manager()
    reaction_table = artifact.get_test_results(for_genes=False)
    base_columns = reaction_table.drop(columns="is_significant").copy(deep=True)

    manager.apply_comparison_thresholds(
        pval_threshold=0.05,
        cohens_d_threshold=1.0,
        log2_fc_threshold=1.0,
        mean_difference_threshold=1.0,
    )

    updated = manager.get_group_comparison_artifact().get_test_results(for_genes=False)
    pd.testing.assert_frame_equal(
        updated.drop(columns="is_significant"),
        base_columns,
    )
    assert manager.experiment_analysis is summaries
    assert manager.data_provider.experiment_analysis is summaries


def test_successful_threshold_apply_invalidates_all_dependent_presentation_state():
    manager, _artifact, _summaries = _manager()

    manager.apply_comparison_thresholds(
        pval_threshold=0.05,
        cohens_d_threshold=1.0,
        log2_fc_threshold=1.0,
        mean_difference_threshold=1.0,
    )

    installed = manager.get_group_comparison_artifact()
    assert manager.subsystem_analysis.gc is installed
    assert manager.subsystem_analysis._base_stats_cache == {}
    assert manager.subsystem_analysis._delta_stats_cache == {}
    assert manager.cache_manager._hover_cache == {}
    assert manager.hover_manager.group_comparison_artifact is installed
    assert manager.data_provider.group_comparison_artifact is installed
    assert manager.get_mode() is AnalysisWorkspaceMode.GROUP_COMPARISON
    assert manager.color_manager.get_mode() is AnalysisWorkspaceMode.GROUP_COMPARISON


def test_invalid_threshold_fails_without_revision_or_state_change():
    manager, artifact, summaries = _manager()
    table_before = artifact.get_test_results().copy(deep=True)
    significant_before = set(artifact.significant_reactions)
    thresholds_before = artifact.get_thresholds().copy()
    base_cache_before = manager.subsystem_analysis._base_stats_cache.copy()
    delta_cache_before = manager.subsystem_analysis._delta_stats_cache.copy()
    hover_cache_before = manager.cache_manager._hover_cache.copy()
    revision_before = manager.threshold_revision

    with pytest.raises(ValueError, match="at most 1"):
        manager.apply_comparison_thresholds(pval_threshold=1.5)

    assert manager.threshold_revision == revision_before
    # The failed apply installs no replacement, so the artifact is untouched.
    assert manager.get_group_comparison_artifact() is artifact
    pd.testing.assert_frame_equal(artifact.get_test_results(), table_before)
    assert set(artifact.significant_reactions) == significant_before
    assert artifact.get_thresholds() == thresholds_before
    assert manager.subsystem_analysis._base_stats_cache == base_cache_before
    assert manager.subsystem_analysis._delta_stats_cache == delta_cache_before
    assert manager.cache_manager._hover_cache == hover_cache_before
    assert manager.experiment_analysis is summaries


def test_threshold_apply_requires_the_installed_comparison_mirror():
    manager, artifact, _summaries = _manager()
    # Corrupt the provider's artifact mirror so it no longer matches the installed
    # authoritative artifact.
    manager.data_provider.group_comparison_artifact = object()
    revision_before = manager.threshold_revision
    table_before = artifact.get_test_results().copy(deep=True)

    with pytest.raises(ValueError, match="provider mirror is not current"):
        manager.apply_comparison_thresholds(pval_threshold=0.05)

    assert manager.threshold_revision == revision_before
    assert manager.get_group_comparison_artifact() is artifact
    pd.testing.assert_frame_equal(artifact.get_test_results(), table_before)

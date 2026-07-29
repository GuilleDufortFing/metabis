"""Transaction guarantees for scientific result installation."""


import pandas as pd
import pytest

from metabis import (
    AnalysisWorkspace,
    AnalysisWorkspaceMode,
    GroupComparison,
    GroupComparisonSpec,
)
from metabis.analysis_workspace import RevisionKind, StaleAnalysisResultError
from metabis.managers.group_manager import GroupDefinition

_SAMPLES = ["A1", "A2", "B1", "B2"]


def _hand_crafted_comparison_artifact(manager):
    """Build an installable comparison artifact without a real analysis run."""
    for group_id, members in (("A", ["A1", "A2"]), ("B", ["B1", "B2"])):
        manager.group_manager.define_group(
            group_id,
            members,
            definition=GroupDefinition("custom"),
            all_sample_ids=_SAMPLES,
        )
    comparison = GroupComparison(
        spec=GroupComparisonSpec.pairwise(("A",), ("B",))
    )
    comparison._reaction_test_results = pd.DataFrame(
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
        index=["R1", "R2"],
    )
    comparison.define_significant_elements(
        pval_threshold=1.0,
        cohens_d_threshold=0.0,
        log2_fc_threshold=0.0,
        mean_difference_threshold=0.0,
        for_genes=False,
    )
    comparison.membership = manager.group_manager.resolve_comparison_membership(
        comparison.spec, _SAMPLES
    )
    snapshot = manager.capture_analysis_snapshot(
        {RevisionKind.GROUPS, RevisionKind.THRESHOLDS}
    )
    return comparison.build_artifact(
        provenance=snapshot.provenance,
        source_fingerprints=manager.source_fingerprints,
    )


class _Analysis:
    def __init__(self, *, presentation_error: Exception | None = None):
        self.presentation_error = presentation_error
        self.presentation_calls = 0

    def register_presentation_distributions(self):
        self.presentation_calls += 1
        if self.presentation_error is not None:
            raise self.presentation_error


def _scientific_and_presentation_state(manager):
    """Capture every result mirror and presentation adapter touched by installs."""
    return (
        manager.experiment_analysis,
        manager.group_comparison_artifact,
        manager.activity_artifact,
        manager.data_provider.experiment_analysis,
        manager.data_provider.group_comparison_artifact,
        manager.data_provider.activity_artifact,
        manager.subsystem_analysis._gc,
        manager.hover_manager.group_comparison_artifact,
        manager.mode,
        manager.color_manager.get_mode(),
        dict(manager.cache_manager._hover_cache),
        dict(manager.color_manager._distributions),
    )


def test_stale_install_rejects_before_any_scientific_or_presentation_write():
    manager = AnalysisWorkspace()
    old_analysis = _Analysis()
    old_comparison = object()
    old_activity = object()
    manager.experiment_analysis = old_analysis
    manager.group_comparison_artifact = old_comparison
    manager.activity_artifact = old_activity
    manager.data_provider.experiment_analysis = old_analysis
    manager.data_provider.group_comparison_artifact = old_comparison
    manager.data_provider.activity_artifact = old_activity
    manager.subsystem_analysis._gc = old_comparison
    manager.hover_manager.group_comparison_artifact = old_comparison
    manager.mode = AnalysisWorkspaceMode.GROUP_COMPARISON
    manager.color_manager.set_mode(AnalysisWorkspaceMode.GROUP_COMPARISON)
    manager.cache_manager.cache_hover("old-context", "R1", "old-hover")

    provenance = manager.capture_analysis_snapshot(
        {RevisionKind.GROUPS}
    ).provenance
    manager.mark_groups_changed()
    before = _scientific_and_presentation_state(manager)

    with pytest.raises(StaleAnalysisResultError, match="groups revision"):
        manager.install_experiment_analysis(provenance, _Analysis())

    assert _scientific_and_presentation_state(manager) == before


def test_matching_installs_publish_authoritative_state_and_provider_mirrors():
    manager = AnalysisWorkspace()
    old_activity = object()
    manager.activity_artifact = old_activity
    manager.data_provider.activity_artifact = old_activity
    analysis = _Analysis()
    analysis_provenance = manager.capture_analysis_snapshot(
        {RevisionKind.GROUPS}
    ).provenance

    returned_analysis = manager.install_experiment_analysis(
        analysis_provenance,
        analysis,
    )

    assert returned_analysis is analysis
    assert manager.experiment_analysis is analysis
    assert manager.data_provider.experiment_analysis is analysis
    assert manager.activity_artifact is None
    assert manager.data_provider.activity_artifact is None

    comparison_artifact = _hand_crafted_comparison_artifact(manager)
    comparison_provenance = manager.capture_analysis_snapshot(
        {RevisionKind.GROUPS, RevisionKind.THRESHOLDS}
    ).provenance
    returned_comparison = manager.install_group_comparison_artifact(
        comparison_provenance,
        comparison_artifact,
    )

    assert returned_comparison is comparison_artifact
    assert manager.group_comparison_artifact is comparison_artifact
    assert manager.data_provider.group_comparison_artifact is comparison_artifact
    # Scientific install does not cross into presentation adapters.
    assert manager.subsystem_analysis._gc is None
    assert manager.hover_manager.group_comparison_artifact is None


def test_presentation_failure_leaves_scientific_state_coherently_installed():
    manager = AnalysisWorkspace()
    analysis = _Analysis(presentation_error=RuntimeError("color adapter failed"))
    provenance = manager.capture_analysis_snapshot(
        {RevisionKind.GROUPS}
    ).provenance

    manager.install_experiment_analysis(provenance, analysis)

    with pytest.raises(RuntimeError, match="color adapter failed"):
        manager.refresh_experiment_analysis_presentation()

    assert manager.experiment_analysis is analysis
    assert manager.data_provider.experiment_analysis is analysis
    assert manager.activity_artifact is None
    assert manager.data_provider.activity_artifact is None
    assert analysis.presentation_calls == 1

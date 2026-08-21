import pandas as pd

from metabis import (
    AnalysisWorkspace,
    GroupComparison,
    GroupComparisonSpec,
)
from metabis.analysis_workspace import RevisionKind
from metabis.managers.cache_manager import CacheManager
from metabis.managers.group_manager import GroupDefinition

_SAMPLES = ["a1", "a2", "b1", "b2"]


def _comparison_artifact():
    """Build the installed comparison artifact the hover cache key reads."""
    manager = AnalysisWorkspace()
    for group_id, members in (("a", ["a1", "a2"]), ("b", ["b1", "b2"])):
        manager.group_manager.define_group(
            group_id,
            members,
            definition=GroupDefinition("custom"),
            all_sample_ids=_SAMPLES,
        )
    comparison = GroupComparison(
        spec=GroupComparisonSpec.pairwise(("a",), ("b",)),
        test="ttest",
        correction="holm",
    )
    comparison._reaction_test_results = pd.DataFrame(
        {
            "adjusted_pval": [0.01, 0.2],
            "pval": [0.005, 0.1],
            "cohens_d": [1.5, 0.2],
            "log2_fc": [2.0, 0.1],
            "mean_difference": [3.0, 0.2],
            "reversed": [False, False],
        },
        index=["R1", "R2"],
    )
    comparison.define_significant_elements(
        pval_threshold=0.05,
        cohens_d_threshold=0.5,
        log2_fc_threshold=1.0,
        mean_difference_threshold=2.0,
        for_genes=False,
    )
    comparison.membership = manager.group_manager.resolve_comparison_membership(
        comparison.spec, _SAMPLES
    )
    snapshot = manager.capture_analysis_snapshot({RevisionKind.GROUPS})
    return comparison.build_artifact(
        provenance=snapshot.provenance,
        source_fingerprints=manager.source_fingerprints,
    )


def test_hover_cache_key_includes_correction_thresholds_mode_and_active_group():
    cache = CacheManager()
    artifact = _comparison_artifact()

    key = cache.get_cache_key(artifact, mode="group_comparison")

    assert "correction=holm" in key
    assert "adjusted_pval" in key
    assert "mode=group_comparison" in key
    assert (
        cache.get_cache_key(None, mode="experiment_analysis", active_group="a")
        == "mode=experiment_analysis|active_group=a"
    )

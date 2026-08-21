from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from metabis import AnalysisWorkspace
from metabis.analysis.artifacts import (
    AnalysisArtifactMetadata,
    ArtifactDimension,
    ArtifactKind,
    ArtifactProvenance,
    PlainTable,
)
from metabis.analysis.comparison_artifact import (
    ComparisonThresholds,
    GroupComparisonArtifact,
)
from metabis.analysis.comparison_membership import (
    ComparisonMembership,
    GroupComparisonSpec,
)
from metabis.analysis.reaction_direction import (
    ReactionDirectionCriterion,
    ReactionSourceDirection,
    build_reaction_direction_map,
)
from metabis.analysis.subsystem_analysis import SubsystemAnalysis
from metabis.analysis_workspace import RevisionKind, WorkspaceRevisions


def _comparison(rows, *, significant=()):
    membership = ComparisonMembership.resolve(
        GroupComparisonSpec.pairwise(("target",), ("baseline",)),
        group_samples={"target": ("t1", "t2"), "baseline": ("b1", "b2")},
        aligned_sample_ids=("t1", "t2", "b1", "b2"),
    )
    frame = pd.DataFrame.from_dict(rows, orient="index")
    metadata = AnalysisArtifactMetadata(
        kind=ArtifactKind.GROUP_COMPARISON,
        source=ArtifactProvenance(
            workspace_id="workspace",
            revisions=WorkspaceRevisions(dataset=1, groups=1),
            dependencies=frozenset({RevisionKind.DATASET, RevisionKind.GROUPS}),
            source_fingerprints={"dataset": "sha256:test"},
        ),
        method="ttest",
        configuration={"correction": "fdr_bh", "keep_reversed": False},
        scientific_identity={
            "membership_fingerprint": membership.fingerprint,
            "mode": membership.mode.value,
            "target_base_group_ids": membership.target_base_group_ids,
            "baseline_base_group_ids": membership.baseline_base_group_ids,
            "target_samples": membership.target_samples,
            "baseline_samples": membership.baseline_samples,
            "universe": membership.universe,
            "excluded_base_group_ids": membership.excluded_base_group_ids,
        },
        dimensions=(
            ArtifactDimension("samples", 4),
            ArtifactDimension("target_samples", 2),
            ArtifactDimension("baseline_samples", 2),
            ArtifactDimension("reactions", len(frame)),
            ArtifactDimension("genes", 0),
        ),
    )
    return GroupComparisonArtifact(
        metadata=metadata,
        membership=membership,
        test="ttest",
        correction="fdr_bh",
        reaction_results=PlainTable.capture(frame),
        gene_results=None,
        reaction_metrics=(),
        gene_metrics=(),
        significant_reactions=tuple(sorted(significant)),
        significant_genes=(),
        reaction_thresholds=ComparisonThresholds(None, None, None, None),
        gene_thresholds=None,
    )


def test_target_and_baseline_criteria_select_different_exact_columns():
    comparison = _comparison(
        {
            "R1": {"group_1_mean": 8.0, "group_2_mean": 1.0},
            "R1_back": {"group_1_mean": 3.0, "group_2_mean": 9.0},
        }
    )

    target = build_reaction_direction_map(
        comparison, ReactionDirectionCriterion.DOMINANT_IN_TARGET
    ).entries[0]
    baseline = build_reaction_direction_map(
        comparison, ReactionDirectionCriterion.DOMINANT_IN_BASELINE
    ).entries[0]

    assert target.source_column == "R1"
    assert target.selected_direction is ReactionSourceDirection.FORWARD
    assert target.target_mean_activity == 8.0
    assert target.baseline_mean_activity == 1.0
    assert target.reason == "forward_selected_higher_target_mean"
    assert baseline.source_column == "R1_back"
    assert baseline.selected_direction is ReactionSourceDirection.BACKWARD
    assert baseline.target_mean_activity == 3.0
    assert baseline.baseline_mean_activity == 9.0
    assert baseline.reason == "backward_selected_higher_baseline_mean"


def test_ties_choose_forward_and_lone_directions_are_retained_once():
    # The tie under test is between the two *target* means, so this names the
    # criterion rather than relying on the default (now Baseline dominance).
    direction_map = build_reaction_direction_map(
        _comparison(
            {
                "R1": {"group_1_mean": 4.0, "group_2_mean": 2.0},
                "R1_back": {"group_1_mean": 4.0, "group_2_mean": 7.0},
                "R2": {"group_1_mean": 1.0, "group_2_mean": 1.0},
                "R3_back": {"group_1_mean": 5.0, "group_2_mean": 6.0},
            }
        ),
        ReactionDirectionCriterion.DOMINANT_IN_TARGET,
    )

    assert [entry.base_reaction_id for entry in direction_map.entries] == [
        "R1",
        "R2",
        "R3",
    ]
    assert direction_map.source_column_for("R1_back") == "R1"
    assert direction_map.get("R1").reason == "forward_selected_target_mean_tie"
    assert direction_map.get("R2").reason == "only_forward_available"
    assert direction_map.get("R3").reason == "only_backward_available"


def test_map_is_frozen_deterministic_and_criterion_changes_its_fingerprint():
    comparison = _comparison(
        {
            "R1": {"group_1_mean": 2.0, "group_2_mean": 3.0},
            "R1_back": {"group_1_mean": 1.0, "group_2_mean": 4.0},
        }
    )
    first = build_reaction_direction_map(
        comparison, ReactionDirectionCriterion.DOMINANT_IN_TARGET
    )
    second = build_reaction_direction_map(
        comparison, ReactionDirectionCriterion.DOMINANT_IN_TARGET
    )
    baseline = build_reaction_direction_map(
        comparison, ReactionDirectionCriterion.DOMINANT_IN_BASELINE
    )

    assert first == second
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != baseline.fingerprint
    with pytest.raises(FrozenInstanceError):
        first.criterion = ReactionDirectionCriterion.DOMINANT_IN_BASELINE


def test_non_finite_population_means_are_rejected_at_the_shared_boundary():
    comparison = _comparison(
        {"R1": {"group_1_mean": float("nan"), "group_2_mean": 1.0}}
    )

    with pytest.raises(ValueError, match="non-finite Target mean"):
        build_reaction_direction_map(comparison)


class _Model:
    def get_subsystems(self):
        return ["Toy"]

    def get_reactions_for_subsystem(self, _subsystem):
        return ["R1"]

    def get_genes_for_subsystem(self, _subsystem):
        return []


class _ReactionData:
    def get_reaction_df(self):
        return pd.DataFrame(columns=["R1", "R1_back"])


class _Workspace:
    metabolic_model = _Model()
    reaction_data = _ReactionData()
    expression_data = None
    color_manager = None

    def __init__(self, comparison):
        self.comparison = comparison

    def get_group_comparison_artifact(self):
        return self.comparison


def test_delta_uses_one_exact_selected_column_and_audits_the_shared_map():
    comparison = _comparison(
        {
            "R1": {
                "group_1_mean": 10.0,
                "group_2_mean": 1.0,
                "cohens_d": 2.0,
                "log2_fc": 1.0,
                "mean_difference": 9.0,
            },
            "R1_back": {
                "group_1_mean": 2.0,
                "group_2_mean": 9.0,
                "cohens_d": -4.0,
                "log2_fc": -2.0,
                "mean_difference": -7.0,
            },
        },
        significant=("R1", "R1_back"),
    )
    analysis = SubsystemAnalysis(
        _Workspace(comparison),
        default_delta_config={
            "metric": "cohens_d",
            "transform": "sign",
            "numerator_filter": "significant",
            "denominator_filter": "all",
            "reaction_direction_criterion": "dominant_in_target",
        },
    )

    analysis.set_group_comparison(comparison)
    target = analysis.get_subsystem_statistics("Toy")
    target_artifact = analysis.get_delta_artifact()
    analysis.update_delta_config(
        {
            "metric": "cohens_d",
            "transform": "sign",
            "numerator_filter": "significant",
            "denominator_filter": "all",
            "reaction_direction_criterion": "dominant_in_baseline",
        }
    )
    baseline = analysis.get_subsystem_statistics("Toy")

    assert target["delta_subsystem_score"] == 1.0
    assert target_artifact.reaction_direction_map is not None
    assert target_artifact.reaction_direction_map.source_column_for("R1") == "R1"
    assert target["total_elements"] == 1
    assert target["delta_subsystem_meta"]["denominator_count"] == 1
    assert target["delta_subsystem_meta"]["selected_reaction_directions"][0][
        "source_column"
    ] == "R1"
    assert baseline["delta_subsystem_score"] == -1.0
    assert baseline["total_elements"] == 1
    assert baseline["delta_subsystem_meta"]["denominator_count"] == 1
    assert baseline["delta_subsystem_meta"]["selected_reaction_directions"][0][
        "source_column"
    ] == "R1_back"
    assert (
        target["delta_subsystem_meta"]["reaction_direction_fingerprint"]
        != baseline["delta_subsystem_meta"]["reaction_direction_fingerprint"]
    )


def test_workspace_criterion_change_invalidates_reaction_but_not_gene_delta_state():
    workspace = AnalysisWorkspace()
    workspace.reaction_set_enrichment_artifact = object()
    workspace.subsystem_analysis._base_stats_cache = {
        ("Toy", False): {"family": "reaction"},
        ("Toy", True): {"family": "gene"},
    }
    workspace.subsystem_analysis._delta_stats_cache = {
        ("Toy", False): {"delta_subsystem_score": 1.0},
        ("Toy", True): {"delta_subsystem_score": -1.0},
    }

    # Away from the default (Baseline dominance), so the setter actually
    # transitions rather than short-circuiting on an unchanged value.
    workspace.set_reaction_direction_criterion("dominant_in_target")

    assert workspace.reaction_set_enrichment_artifact is None
    assert workspace.subsystem_analysis._base_stats_cache == {
        ("Toy", True): {"family": "gene"}
    }
    assert workspace.subsystem_analysis._delta_stats_cache == {
        ("Toy", True): {"delta_subsystem_score": -1.0}
    }

import math

import pandas as pd
import pytest

from metabis.analysis.agreement import ResultEffects, compare_results
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
from metabis.analysis.concordance import reaction_gene_concordance
from metabis.analysis.landscape import with_thresholds
from metabis.analysis_workspace import RevisionKind, WorkspaceRevisions

P05 = {"pval": 0.05, "cohens_d": None, "log2_fc": None, "mean_difference": None}


def _frame(effects, pvalues):
    return pd.DataFrame(
        {
            "cohens_d": list(effects.values()),
            "adjusted_pval": pvalues,
            "log2_fc": list(effects.values()),
            "mean_difference": list(effects.values()),
            "group_1_mean": [1.0] * len(effects),
            "group_2_mean": [0.5] * len(effects),
        },
        index=list(effects),
    )


def _artifact(reactions: pd.DataFrame, genes: pd.DataFrame | None = None):
    membership = ComparisonMembership.resolve(
        GroupComparisonSpec.pairwise(("target",), ("baseline",)),
        group_samples={"target": ("t1", "t2"), "baseline": ("b1", "b2")},
        aligned_sample_ids=("t1", "t2", "b1", "b2"),
    )
    metadata = AnalysisArtifactMetadata(
        kind=ArtifactKind.GROUP_COMPARISON,
        source=ArtifactProvenance(
            workspace_id="workspace",
            revisions=WorkspaceRevisions(dataset=1, groups=1),
            dependencies=frozenset({RevisionKind.DATASET, RevisionKind.GROUPS}),
            source_fingerprints={"dataset": "sha256:test"},
        ),
        method="wilcoxon",
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
            ArtifactDimension("reactions", len(reactions)),
            ArtifactDimension("genes", len(genes) if genes is not None else 0),
        ),
    )
    unset = ComparisonThresholds(None, None, None, None)
    return GroupComparisonArtifact(
        metadata=metadata,
        membership=membership,
        test="wilcoxon",
        correction="fdr_bh",
        reaction_results=PlainTable.capture(reactions),
        gene_results=PlainTable.capture(genes) if genes is not None else None,
        reaction_metrics=(),
        gene_metrics=(),
        significant_reactions=(),
        significant_genes=(),
        reaction_thresholds=unset,
        gene_thresholds=unset if genes is not None else None,
    )


def test_agreement_classifies_shared_features_and_ranks_effects():
    first = with_thresholds(
        _artifact(_frame({"R1": 1.0, "R2": -0.8, "R3": 0.6, "R4": 0.1, "R5": 2.0}, [0.01, 0.01, 0.01, 0.9, 0.01])),
        P05,
    )
    second = with_thresholds(
        _artifact(_frame({"R1": 0.7, "R2": 0.9, "R3": 0.2, "R4": 1.2, "R6": 1.0}, [0.01, 0.01, 0.5, 0.01, 0.01])),
        P05,
    )

    agreement = compare_results(
        ResultEffects.from_artifact("first", first), ResultEffects.from_artifact("second", second)
    )

    status = dict(zip(agreement.table.feature_id, agreement.table.status))
    assert status == {"R1": "both", "R2": "opposite", "R3": "first", "R4": "second"}
    assert agreement.shared == 4
    assert agreement.counts == {"both": 1, "opposite": 1, "first": 1, "second": 1, "neither": 0}
    assert agreement.spearman == pytest.approx(-0.6)


def test_an_exported_table_is_judged_by_the_same_thresholds():
    table = pd.DataFrame(
        {
            "feature_id": ["R1", "R2", "R3"],
            "cohens_d": [0.9, -0.3, 1.5],
            "adjusted_pval": [0.01, 0.2, 0.04],
        }
    )

    effects = ResultEffects.from_table("file", table, {**P05, "cohens_d": 1.0})

    assert effects.significant == frozenset({"R3"})
    assert effects.effects.to_dict() == {"R1": 0.9, "R2": -0.3, "R3": 1.5}
    with pytest.raises(ValueError, match="no column for the threshold on log2_fc"):
        ResultEffects.from_table("file", table, {**P05, "log2_fc": 1.0})
    with pytest.raises(ValueError, match="no column adjusted_pval"):
        ResultEffects.from_table("file", table.drop(columns="adjusted_pval"), P05)


def test_few_shared_features_have_no_correlation():
    first = ResultEffects("a", pd.Series({"R1": 1.0, "R2": 2.0}), frozenset())
    second = ResultEffects("b", pd.Series({"R1": 1.0, "R2": 2.0}), frozenset())

    assert math.isnan(compare_results(first, second).spearman)


def test_reactions_are_classified_against_their_genes():
    reactions = _frame(
        {"R1": 1.0, "R1_back": -1.0, "R2": 0.9, "R3": 0.8, "R4": 0.1, "R5": 0.2, "R6": 1.0},
        [0.01, 0.01, 0.01, 0.01, 0.6, 0.6, 0.01],
    )
    genes = _frame(
        {"G1": 0.7, "G2": -0.9, "G3": 0.1, "G4": 1.4, "G5": 0.05},
        [0.01, 0.01, 0.7, 0.01, 0.9],
    )
    gene_map = {"G1": ["R1"], "G2": ["R2", "R1"], "G3": ["R3"], "G4": ["R4"], "G5": ["R5"]}
    artifact = with_thresholds(_artifact(reactions, genes), P05, P05)

    table = reaction_gene_concordance(artifact, lambda gene: gene_map.get(gene, []))

    status = dict(zip(table.feature_id, table.status))
    # R1 moves up with G1; its back direction moves down with G2.
    assert status == {
        "R1": "follows",
        "R1_back": "follows",
        "R2": "against",
        "R3": "network",
        "R4": "genes",
        "R5": "neither",
    }
    r1 = table.set_index("feature_id").loc["R1"]
    assert (r1.strongest_gene, r1.gene_effect, r1.genes, r1.significant_genes) == ("G2", -0.9, 2, 2)


def test_concordance_needs_gene_results():
    artifact = _artifact(_frame({"R1": 1.0}, [0.01]))

    with pytest.raises(ValueError, match="Load expression data"):
        reaction_gene_concordance(artifact, lambda gene: [])

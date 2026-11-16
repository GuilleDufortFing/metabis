"""A comparison read the other way is the comparison run with its sides exchanged."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metabis import AnalysisWorkspace, GroupComparison, GroupComparisonSpec
from metabis.analysis.comparison_artifact import swapped_comparison
from metabis.analysis_workspace import RevisionKind
from metabis.domain.expression_matrix import DenseExpressionMatrix
from metabis.managers.group_manager import GroupDefinition

SAMPLES = ["A1", "A2", "A3", "A4", "B1", "B2", "B3"]
DEPENDENCIES = frozenset({RevisionKind.DATASET, RevisionKind.METADATA, RevisionKind.GROUPS, RevisionKind.PREPROCESSING})


class _ReactionData:
    def __init__(self) -> None:
        self.frame = pd.DataFrame(
            {
                "R1": [1.0, 1.2, 0.9, 1.4, 5.0, 5.4, 4.8],
                "R2": [2.0, 2.1, 1.9, 2.3, 2.2, 2.0, 2.1],
                "R3": [-1.0, -0.8, -1.3, -0.9, 0.5, 0.7, 0.4],
                # The same constant in both groups: left untested.
                "R4": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            },
            index=SAMPLES,
        )

    def get_reaction_df(self) -> pd.DataFrame:
        return self.frame

    def get_sample_ids(self):
        return list(self.frame.index)


class _ExpressionData:
    def __init__(self) -> None:
        self.frame = pd.DataFrame(
            {
                "G1": [10.0, 11.0, 9.5, 10.5, 30.0, 31.0, 29.0],
                "G2": [4.0, 4.2, 3.9, 4.4, 4.1, 4.0, 4.2],
            },
            index=SAMPLES,
        )

    def get_expression_df(self) -> pd.DataFrame:
        return self.frame

    def get_feature_ids(self):
        return [str(column) for column in self.frame.columns]

    def matrix(self):
        return DenseExpressionMatrix(self.frame)

    def get_sample_ids(self):
        return list(self.frame.index)


class _Collection:
    def __init__(self, values) -> None:
        self._values = list(values)

    def get_all(self):
        return list(self._values)


class _Model:
    def __init__(self) -> None:
        self.reactions = _Collection(["R1", "R2", "R3", "R4"])
        self.subsystems = _Collection(["Glycolysis"])

    def get_reaction_metadata(self, reaction_id):
        return {"subsystems": ["Glycolysis"]}

    def get_gene_metadata(self, gene_id):
        return {"subsystems": ["Glycolysis"]}


def _workspace() -> AnalysisWorkspace:
    manager = AnalysisWorkspace(metabolic_model=_Model())
    manager.replace_reaction_data(_ReactionData(), metabolic_model=_Model())
    manager.replace_expression_data(_ExpressionData())
    manager.group_manager.define_group("A", SAMPLES[:4], definition=GroupDefinition("custom"))
    manager.group_manager.define_group("B", SAMPLES[4:], definition=GroupDefinition("custom"))
    return manager


def _artifact(manager: AnalysisWorkspace, target: str, baseline: str, test: str):
    membership = manager.group_manager.resolve_comparison_membership(
        GroupComparisonSpec.pairwise((target,), (baseline,)), SAMPLES
    )
    comparison = GroupComparison(membership=membership, test=test)
    for for_genes in (False, True):
        comparison.run_test(manager, for_genes=for_genes, test=test, correction="fdr_bh", register_distributions=False)
    snapshot = manager.capture_analysis_snapshot(DEPENDENCIES)
    return comparison.build_artifact(provenance=snapshot.provenance, source_fingerprints=manager.source_fingerprints)


def _assert_same_frame(left: pd.DataFrame, right: pd.DataFrame) -> None:
    assert list(left.columns) == list(right.columns)
    assert list(left.index) == list(right.index)
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]) and not pd.api.types.is_bool_dtype(left[column]):
            np.testing.assert_allclose(left[column].to_numpy(float), right[column].to_numpy(float), rtol=1e-9, atol=1e-12, err_msg=column)
        else:
            assert list(left[column]) == list(right[column]), column


@pytest.mark.parametrize("test", ["wilcoxon", "ttest", "moderated_t"])
def test_reading_a_comparison_the_other_way_equals_running_it_so(test):
    manager = _workspace()
    forward = _artifact(manager, "A", "B", test)
    backward = _artifact(manager, "B", "A", test)

    swapped = swapped_comparison(forward)

    assert swapped.membership == backward.membership
    assert dict(swapped.metadata.scientific_identity) == dict(backward.metadata.scientific_identity)
    assert swapped.metadata.dimensions == backward.metadata.dimensions
    for for_genes in (False, True):
        _assert_same_frame(swapped.get_test_results(for_genes=for_genes), backward.get_test_results(for_genes=for_genes))
    assert swapped.significant_reactions == backward.significant_reactions
    assert swapped.significant_genes == backward.significant_genes
    for mine, theirs in zip(swapped.reaction_metrics + swapped.gene_metrics, backward.reaction_metrics + backward.gene_metrics):
        assert mine.name == theirs.name
        assert mine.mean == pytest.approx(theirs.mean)
        assert mine.std == pytest.approx(theirs.std)


def test_reading_it_back_returns_the_comparison():
    manager = _workspace()
    forward = _artifact(manager, "A", "B", "wilcoxon")

    twice = swapped_comparison(swapped_comparison(forward))

    assert twice.membership == forward.membership
    _assert_same_frame(twice.get_test_results(), forward.get_test_results())


def test_a_group_against_the_rest_has_no_other_way():
    manager = _workspace()
    membership = manager.group_manager.resolve_comparison_membership(GroupComparisonSpec.one_vs_rest(("A",)), SAMPLES)
    comparison = GroupComparison(membership=membership, test="wilcoxon")
    comparison.run_test(manager, test="wilcoxon", correction="fdr_bh", register_distributions=False)
    snapshot = manager.capture_analysis_snapshot(DEPENDENCIES)
    artifact = comparison.build_artifact(provenance=snapshot.provenance, source_fingerprints=manager.source_fingerprints)

    with pytest.raises(ValueError, match="two sides"):
        swapped_comparison(artifact)

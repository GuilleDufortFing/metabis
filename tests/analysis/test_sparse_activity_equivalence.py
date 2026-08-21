"""Normalized activity summaries are storage-invariant and bounded.

The activity scorer reads only the selected group's rows through the matrix seam.
It never materializes the whole matrix dense, and the result equals the dense
public function's mean/median over the same values.

These drive the *real* bounded producer over a real, sparse-``X`` h5ad load and
assert the numbers equal the identical data scored densely -- the numeric face of
MR-04 for activity, with the sparse side driven by a genuine loaded workspace.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from metabis import (
    ActivityConfig,
    ActivityOverlay,
    AnalysisWorkspace,
    ExperimentAnalysis,
    ExpressionData,
    GroupSummary,
)
from metabis.analysis.activity_analysis import (
    calculate_feature_activity,
    calculate_feature_activity_from_matrix,
)
from metabis.domain.expression_matrix import (
    DenseExpressionMatrix,
    ExpressionMatrix,
    SparseExpressionMatrix,
)
from metabis.managers.group_manager import GroupDefinition

SAMPLES = [f"s{i}" for i in range(10)]
GENES = [f"g{j}" for j in range(16)]
GROUP_A = SAMPLES[:5]
GROUP_B = SAMPLES[5:]

SUMMARIES = [GroupSummary.MEAN, GroupSummary.MEDIAN]


def _sparse_expression(tmp_path, density=0.5, seed=5) -> ExpressionData:
    """A real h5ad-backed ``ExpressionData`` whose ``X`` stays sparse."""
    rng = np.random.default_rng(seed)
    matrix = sparse.random(
        len(SAMPLES), len(GENES), density=density, format="csr",
        random_state=rng, dtype=np.float32,
    )
    path = tmp_path / "expr.h5ad"
    AnnData(
        X=matrix,
        obs=pd.DataFrame(index=SAMPLES),
        var=pd.DataFrame(index=GENES),
    ).write_h5ad(path)
    expression = ExpressionData(
        path=str(path), gene_list=GENES, data_format="h5ad"
    )
    assert isinstance(expression.matrix(), SparseExpressionMatrix)
    return expression


def _config(summary) -> ActivityConfig:
    return ActivityConfig(group_id="A", summary=summary)


def _assert_results_equal(a, b) -> None:
    pd.testing.assert_series_equal(a.activity_score, b.activity_score)
    assert a.non_finite_feature_ids == b.non_finite_feature_ids
    assert a.sample_count == b.sample_count
    assert a.reference_sample_count == b.reference_sample_count


@pytest.mark.parametrize("summary", SUMMARIES)
def test_bounded_activity_matches_the_dense_function(tmp_path, summary):
    """The bounded seam producer equals the dense public function, sparse or dense.

    The refactor is behavior-preserving (bounded == dense over the same frame) and
    storage-invariant (sparse adapter == dense frame): all three assertions hold.
    """
    expression = _sparse_expression(tmp_path)
    dense_frame = expression.get_expression_df()
    config = _config(summary)

    dense = calculate_feature_activity(
        dense_frame, GROUP_A, config, overlay=ActivityOverlay.GENE
    )
    bounded_sparse = calculate_feature_activity_from_matrix(
        expression.matrix(), GROUP_A, config, overlay=ActivityOverlay.GENE
    )
    bounded_dense = calculate_feature_activity_from_matrix(
        DenseExpressionMatrix(dense_frame),
        GROUP_A,
        config,
        overlay=ActivityOverlay.GENE,
    )

    _assert_results_equal(bounded_sparse, dense)
    _assert_results_equal(bounded_dense, dense)
    # Scoring did not densify the whole matrix in place.
    assert sparse.issparse(expression.adata.X)


class _NoFullDensify(ExpressionMatrix):
    """Delegates every bounded operation but refuses a full densification.

    The guard that the activity scorer stays bounded: if it ever called
    ``to_dense_frame`` it would materialize the whole matrix, which this forbids.
    """

    def __init__(self, inner: ExpressionMatrix) -> None:
        super().__init__()
        self._inner = inner

    @property
    def sample_ids(self):
        return self._inner.sample_ids

    @property
    def feature_ids(self):
        return self._inner.feature_ids

    def feature_column(self, feature_id):
        return self._inner.feature_column(feature_id)

    def select_samples(self, sample_ids):
        return self._inner.select_samples(sample_ids)

    def feature_variances(self):
        return self._inner.feature_variances()

    def select_features(self, feature_ids):
        return self._inner.select_features(feature_ids)

    def estimate_dense_bytes(self):
        return self._inner.estimate_dense_bytes()

    def _materialize_dense(self):
        raise AssertionError("the activity scorer must not densify the whole matrix")


@pytest.mark.parametrize("summary", SUMMARIES)
def test_activity_never_densifies_the_whole_matrix(tmp_path, summary):
    expression = _sparse_expression(tmp_path)
    config = _config(summary)
    guarded = _NoFullDensify(expression.matrix())

    # Runs to completion without ever calling ``to_dense_frame``.
    result = calculate_feature_activity_from_matrix(guarded, GROUP_A, config)
    assert not result.activity_score.empty


class _ReactionData:
    def __init__(self) -> None:
        rng = np.random.default_rng(1)
        self.frame = pd.DataFrame(
            rng.normal(size=(len(SAMPLES), 3)), index=SAMPLES, columns=["R1", "R2", "R3"]
        )

    def get_reaction_df(self):
        return self.frame

    def get_sample_ids(self):
        return list(self.frame.index)


class _Model:
    def get_subsystems(self):
        return ["Glycolysis"]

    def get_genes_for_subsystem(self, _subsystem):
        return list(GENES)

    def get_reactions_for_subsystem(self, _subsystem):
        return ["R1", "R2", "R3"]

    def get_gene_metadata(self, _gene_id):
        return {"subsystems": ["Glycolysis"]}

    def get_reaction_metadata(self, _reaction_id):
        return {"subsystems": ["Glycolysis"]}


class _DenseRoutedExpression:
    """The same numbers as a real load, forced through the dense adapter."""

    def __init__(self, expression: ExpressionData) -> None:
        self._frame = expression.get_expression_df()

    def get_expression_df(self):
        return self._frame

    def get_feature_ids(self):
        return [str(column) for column in self._frame.columns]

    def get_sample_ids(self):
        return list(self._frame.index)

    def matrix(self):
        return DenseExpressionMatrix(self._frame)

    def get_gene_row(self, gene_id):
        if gene_id not in self._frame.columns:
            return None
        return self._frame[gene_id]


def _manager(expression) -> AnalysisWorkspace:
    model = _Model()
    manager = AnalysisWorkspace(metabolic_model=model)
    manager.replace_reaction_data(_ReactionData(), metabolic_model=model)
    manager.replace_expression_data(expression)
    manager.group_manager.define_group("A", GROUP_A, definition=GroupDefinition("custom"))
    manager.group_manager.define_group("B", GROUP_B, definition=GroupDefinition("custom"))
    return manager


def test_compute_activity_end_to_end_matches_between_sparse_and_dense(tmp_path):
    """MR-12: the production caller (``compute_activity``) is driven, not a fixture.

    A real ``ExperimentAnalysis`` over a sparse-``X`` workspace scores the gene
    overlay through the seam and equals the identical data routed dense.
    """
    sparse_expression = _sparse_expression(tmp_path)
    sparse_manager = _manager(sparse_expression)
    dense_manager = _manager(_DenseRoutedExpression(sparse_expression))
    config = _config(GroupSummary.MEDIAN)

    sparse_context = ExperimentAnalysis(
        sparse_manager, auto_run=False, register_distributions=False
    ).compute_activity(config)
    dense_context = ExperimentAnalysis(
        dense_manager, auto_run=False, register_distributions=False
    ).compute_activity(config)

    _assert_results_equal(sparse_context.features, dense_context.features)
    pd.testing.assert_frame_equal(sparse_context.subsystems, dense_context.subsystems)
    # The production gene overlay scored without densifying the whole matrix.
    assert sparse.issparse(sparse_expression.adata.X)


def test_gene_overlay_reads_the_sparse_adapter(tmp_path):
    """The production path is genuinely the sparse one (MR-12)."""
    manager = _manager(_sparse_expression(tmp_path))
    assert isinstance(manager.expression_data.matrix(), SparseExpressionMatrix)

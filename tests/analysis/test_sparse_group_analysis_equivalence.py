"""AM6 slice 4: group summaries and the gene comparison are storage-invariant.

The migrated ``_calculate_gene_avgs`` and gene ``GroupComparison.run_test`` read
expression through the matrix seam and materialize only a group's samples dense.
These tests drive the *real* methods over a real, sparse-``X`` h5ad load and
assert the numbers equal the identical data routed through the dense adapter --
the numeric face of MR-04, and MR-12-clean because the sparse side is a genuine
loaded workspace, not a fixture. The dense twin is the very frame the sparse load
built, forced through ``DenseExpressionMatrix``, so only the storage path differs.
"""

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from metabis import (
    ExperimentAnalysis,
    ExpressionData,
    GroupComparison,
    GroupComparisonSpec,
)
from metabis.domain.expression_matrix import (
    DenseExpressionMatrix,
    SparseExpressionMatrix,
)
from metabis.managers.group_manager import GroupDefinition
from metabis.managers.workspace import AnalysisWorkspace

SAMPLES = [f"s{i}" for i in range(8)]
GENES = [f"g{j}" for j in range(12)]
GROUP_A = SAMPLES[:4]
GROUP_B = SAMPLES[4:]


def _sparse_expression(tmp_path) -> ExpressionData:
    """A real h5ad-backed ``ExpressionData`` whose ``X`` stays sparse."""
    rng = np.random.default_rng(3)
    matrix = sparse.random(
        len(SAMPLES), len(GENES), density=0.4, format="csr",
        random_state=rng, dtype=np.float32,
    )
    path = tmp_path / "expr.h5ad"
    AnnData(
        X=matrix,
        obs=pd.DataFrame(index=SAMPLES),
        var=pd.DataFrame(index=GENES),
    ).write_h5ad(path)
    expression = ExpressionData(
        path=str(path),
        gene_list=GENES,
        data_format="h5ad",
    )
    # Precondition: this is genuinely the sparse path, not a densified load.
    assert sparse.issparse(expression.adata.X)
    assert isinstance(expression.matrix(), SparseExpressionMatrix)
    return expression


class _DenseRoutedExpression:
    """The same numbers as a real load, forced through the dense adapter.

    It wraps the frame the sparse load built, so a manager holding it computes
    over identical values but never reaches the sparse path -- isolating exactly
    the storage difference the equivalence tests measure.
    """

    def __init__(self, expression: ExpressionData) -> None:
        self._frame = expression.get_expression_df()

    def get_expression_df(self) -> pd.DataFrame:
        return self._frame

    def get_feature_ids(self):
        return [str(column) for column in self._frame.columns]

    def get_sample_ids(self):
        return list(self._frame.index)

    def matrix(self) -> DenseExpressionMatrix:
        return DenseExpressionMatrix(self._frame)

    def get_gene_row(self, gene_id):
        if gene_id not in self._frame.columns:
            return None
        return self._frame[gene_id]


class _MatrixOnlyExpression:
    """Serves the seam only; ``get_expression_df`` raises.

    The guard against silently keeping the full-frame accessor: the migrated
    summaries and comparison must materialize through the bounded seam, so they
    must run without ever calling ``get_expression_df``. This wrapper fails the
    pre-slice-4 analysis code, which read the whole dense frame here. It is swapped
    in *after* the load, so the load-time source-identity fingerprint (a separate
    reader, ADR-0007) has already run against the real frame.
    """

    def __init__(self, expression: ExpressionData) -> None:
        self._matrix = expression.matrix()

    def matrix(self):
        return self._matrix

    def get_expression_df(self):
        raise AssertionError("the full-frame accessor must not be read")

    def get_sample_ids(self):
        return self._matrix.sample_ids

    def get_feature_ids(self):
        return self._matrix.feature_ids

    def get_gene_row(self, gene_id):
        return self._matrix.feature_column(gene_id)


class _ReactionData:
    """Reaction values over the same samples; the comparison universe comes from
    here, and the reaction path is untouched by AM6.
    """

    def __init__(self) -> None:
        rng = np.random.default_rng(7)
        self.frame = pd.DataFrame(
            rng.normal(size=(len(SAMPLES), 3)),
            index=SAMPLES,
            columns=["R1", "R2", "R3"],
        )

    def get_reaction_df(self) -> pd.DataFrame:
        return self.frame

    def get_sample_ids(self):
        return list(self.frame.index)


class _Model:
    def get_reaction_metadata(self, _reaction_id):
        return {"subsystems": ["Glycolysis"]}

    def get_gene_metadata(self, _gene_id):
        return {"subsystems": ["Glycolysis"]}


def _manager(expression) -> AnalysisWorkspace:
    """A session workspace as the app builds one: constructed empty, then loaded."""
    model = _Model()
    manager = AnalysisWorkspace(metabolic_model=model)
    manager.replace_reaction_data(_ReactionData(), metabolic_model=model)
    manager.replace_expression_data(expression)
    manager.group_manager.define_group(
        "A", GROUP_A, definition=GroupDefinition("custom")
    )
    manager.group_manager.define_group(
        "B", GROUP_B, definition=GroupDefinition("custom")
    )
    return manager


def test_gene_group_averages_match_between_sparse_and_dense_routing(tmp_path):
    sparse_expression = _sparse_expression(tmp_path)
    sparse_manager = _manager(sparse_expression)
    dense_manager = _manager(_DenseRoutedExpression(sparse_expression))

    sparse_avgs = ExperimentAnalysis(
        sparse_manager, auto_run=False, register_distributions=False
    )._calculate_gene_avgs()
    dense_avgs = ExperimentAnalysis(
        dense_manager, auto_run=False, register_distributions=False
    )._calculate_gene_avgs()

    pd.testing.assert_frame_equal(sparse_avgs, dense_avgs)
    # Computing the group medians did not densify the whole matrix in place.
    assert sparse.issparse(sparse_expression.adata.X)


def test_gene_comparison_matches_between_sparse_and_dense_routing(tmp_path):
    sparse_expression = _sparse_expression(tmp_path)
    sparse_manager = _manager(sparse_expression)
    dense_manager = _manager(_DenseRoutedExpression(sparse_expression))

    def _run(manager) -> pd.DataFrame:
        membership = manager.group_manager.resolve_comparison_membership(
            GroupComparisonSpec.pairwise(("A",), ("B",)),
            manager.reaction_data.get_sample_ids(),
        )
        comparison = GroupComparison(membership=membership, test="ttest")
        comparison.run_test(
            manager,
            for_genes=True,
            test="ttest",
            correction="fdr_bh",
            register_distributions=False,
        )
        return comparison.get_test_results(for_genes=True)

    pd.testing.assert_frame_equal(_run(sparse_manager), _run(dense_manager))
    # The gene test ran without densifying the whole expression matrix in place.
    assert sparse.issparse(sparse_expression.adata.X)


def test_the_gene_comparison_reads_the_sparse_adapter(tmp_path):
    """MR-12: the production path is genuinely the sparse one, not a fixture."""
    manager = _manager(_sparse_expression(tmp_path))
    assert isinstance(manager.expression_data.matrix(), SparseExpressionMatrix)


def test_summaries_and_comparison_never_read_the_full_frame_accessor(tmp_path):
    """Both migrated readers run through the bounded seam alone.

    ``_MatrixOnlyExpression.get_expression_df`` raises, so this passes only
    because the summaries and the gene comparison materialize through
    ``matrix().select_samples`` -- it fails the pre-slice-4 code that read the
    whole dense frame.
    """
    expression = _sparse_expression(tmp_path)
    manager = _manager(expression)
    # Load-time source identity is warmed; forbid the accessor for the analysis.
    manager.expression_data = _MatrixOnlyExpression(expression)

    # Group gene averages: no full-frame read.
    ExperimentAnalysis(
        manager, auto_run=False, register_distributions=False
    )._calculate_gene_avgs()

    # Gene comparison: no full-frame read.
    membership = manager.group_manager.resolve_comparison_membership(
        GroupComparisonSpec.pairwise(("A",), ("B",)),
        manager.reaction_data.get_sample_ids(),
    )
    comparison = GroupComparison(membership=membership, test="ttest")
    comparison.run_test(
        manager,
        for_genes=True,
        test="ttest",
        correction="fdr_bh",
        register_distributions=False,
    )
    assert comparison.get_test_results(for_genes=True) is not None

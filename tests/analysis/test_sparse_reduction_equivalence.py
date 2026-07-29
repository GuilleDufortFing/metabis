"""AM6 slice 5: the supported reductions are storage-invariant.

The migrated reductions read expression through the matrix seam: the variance
ranking is sparse-native, and only the selected top-N variable features are
materialized dense (the block the sklearn estimators need). These tests drive the
*real* ``ExperimentAnalysis.calculate_{pca,tsne,umap}`` over a real, sparse-``X``
h5ad load and assert the selected features and the coordinates equal the same
data routed through the dense adapter -- the numeric face of MR-04, and
MR-12-clean because the sparse side is a genuine loaded workspace, not a fixture.

The dense twin is the very frame the sparse load built, forced through
``DenseExpressionMatrix``, so only the storage path differs. Feature variances
are deliberately well separated (each column scaled by its index), so the top-N
selection is unambiguous and identical either way; once selection matches, the
bounded block is byte-identical and the estimators receive identical input.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from metabis import ExperimentAnalysis, ExpressionData
from metabis.domain.expression_matrix import (
    DenseExpressionMatrix,
    SparseExpressionMatrix,
)
from metabis.managers.group_manager import GroupDefinition
from metabis.managers.workspace import AnalysisWorkspace

SAMPLES = [f"s{i}" for i in range(40)]
GENES = [f"g{j}" for j in range(60)]


def _sparse_expression(tmp_path) -> ExpressionData:
    """A real h5ad-backed ``ExpressionData`` whose ``X`` stays sparse.

    Columns are scaled by their index so per-feature variances are well
    separated, making the top-N variance ranking unambiguous.
    """
    rng = np.random.default_rng(19)
    matrix = sparse.random(
        len(SAMPLES), len(GENES), density=0.3, format="csr",
        random_state=rng, dtype=np.float32,
    )
    scale = np.linspace(1.0, 8.0, len(GENES)).astype(np.float32)
    matrix = sparse.csr_matrix(matrix.multiply(scale)).astype(np.float32)
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
    """The same numbers as a real load, forced through the dense adapter."""

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
    reduction must rank and materialize through the bounded seam, so it must run
    without ever calling ``get_expression_df``. Swapped in *after* the load, so
    the load-time source-identity fingerprint has already run against the real
    frame.
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
    """Reaction values over the same samples; untouched by the expression reduction."""

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
        "A", SAMPLES[:20], definition=GroupDefinition("custom")
    )
    manager.group_manager.define_group(
        "B", SAMPLES[20:], definition=GroupDefinition("custom")
    )
    return manager


def _reduce(manager, method, n_variable_features):
    analysis = ExperimentAnalysis(
        manager, auto_run=False, register_distributions=False
    )
    if method == "PCA":
        outcome = analysis.calculate_pca(
            data_type="expression", n_variable_features=n_variable_features
        )
    elif method == "T-SNE":
        outcome = analysis.calculate_tsne(
            data_type="expression", n_variable_features=n_variable_features
        )
    else:
        outcome = analysis.calculate_umap(
            data_type="expression", n_variable_features=n_variable_features
        )
    assert outcome["status"] == "success", outcome["messages"]
    return outcome["result"]


@pytest.mark.parametrize("method", ["PCA", "T-SNE", "UMAP"])
def test_reduction_matches_between_sparse_and_dense_routing(tmp_path, method):
    """A sparse load and its dense twin select the same features and produce the
    same coordinates -- the reduction never densified the whole matrix to do it.
    """
    sparse_expression = _sparse_expression(tmp_path)
    sparse_manager = _manager(sparse_expression)
    dense_manager = _manager(_DenseRoutedExpression(sparse_expression))

    sparse_result = _reduce(sparse_manager, method, n_variable_features=25)
    dense_result = _reduce(dense_manager, method, n_variable_features=25)

    # The variance ranking picked the same top-25 features, in the same order.
    assert sparse_result.feature_names == dense_result.feature_names
    assert len(sparse_result.feature_names) == 25
    # Identical selection -> identical bounded block -> identical coordinates.
    np.testing.assert_allclose(
        sparse_result.transformed_data,
        dense_result.transformed_data,
        rtol=1e-5,
        atol=1e-5,
    )
    # The reduction ran without densifying the whole expression matrix in place.
    assert sparse.issparse(sparse_expression.adata.X)


def test_reduction_reads_the_sparse_adapter(tmp_path):
    """MR-12: the production reduction path is genuinely the sparse one."""
    manager = _manager(_sparse_expression(tmp_path))
    assert isinstance(manager.expression_data.matrix(), SparseExpressionMatrix)


def test_reduction_never_reads_the_full_frame_accessor(tmp_path):
    """The reduction runs through the bounded seam alone.

    ``_MatrixOnlyExpression.get_expression_df`` raises, so this passes only
    because the reduction ranks features and materializes the top-N block through
    ``matrix()`` -- it fails the pre-slice-5 code that read the whole dense frame.
    """
    expression = _sparse_expression(tmp_path)
    manager = _manager(expression)
    # Load-time source identity is warmed; forbid the accessor for the analysis.
    manager.expression_data = _MatrixOnlyExpression(expression)

    result = _reduce(manager, "PCA", n_variable_features=25)
    assert len(result.feature_names) == 25

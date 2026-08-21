"""Characterization of the sparse expression path across AM6.

Slice 1 pinned what the load did *before* AM6 changed it; slice 7 makes the load
sparse-first and flips the two assertions the inventory said it consciously would:

- a sparse h5ad load now **stays sparse** -- ``build`` no longer densifies, so
  ``df_expression_data`` is ``None`` and ``matrix()`` is the sparse adapter, while
  ``get_expression_df`` remains an explicit escape hatch that densifies on demand;
- ``estimate_dense_bytes`` still predicts the realized dense size, so it is a
  trustworthy budget for the escape-hatch rejection;
- ``fingerprint_frame`` gives a sparse column and its dense twin the same identity
  (float64 *and* float32), so a sparse-first load does not move the dataset
  fingerprint.
"""

import inspect

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

from metabis import (
    ExpressionData,
    ExpressionPreprocessingConfig,
    ReactionData,
)
from metabis.analysis.artifacts import fingerprint_frame
from metabis.domain import reaction_data as reaction_data_module
from metabis.domain.expression_matrix import SparseExpressionMatrix
from metabis.domain.preprocessing import estimate_dense_bytes


def _sparse_h5ad(tmp_path, n_obs=64, n_vars=200, density=0.05, seed=0):
    """A sparse-CSR AnnData written to disk, the shape a sparse h5ad load takes."""
    rng = np.random.default_rng(seed)
    matrix = sparse.random(
        n_obs, n_vars, density=density, format="csr", random_state=rng, dtype=np.float32
    )
    genes = [f"g{j}" for j in range(n_vars)]
    path = tmp_path / "sparse.h5ad"
    AnnData(
        X=matrix,
        obs=pd.DataFrame(index=[f"s{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=genes),
    ).write_h5ad(path)
    return path, genes, matrix


def test_a_sparse_h5ad_load_stays_sparse(tmp_path):
    """Slice 7: sparse in, sparse resident -- the load no longer densifies."""
    path, genes, _ = _sparse_h5ad(tmp_path)

    expression = ExpressionData(
        path=str(path),
        gene_list=genes,
        data_format="h5ad",
    )

    # The load built no dense frame; the seam is the sparse adapter and ``X`` is
    # still sparse. This is the AM6 goal: supported workflows read the sparse seam.
    assert expression.df_expression_data is None
    assert isinstance(expression.matrix(), SparseExpressionMatrix)
    assert sparse.issparse(expression.adata.X)

    # ``get_expression_df`` remains the explicit escape hatch: it densifies on
    # demand (and its result is a plain dense frame), for a caller that genuinely
    # wants the whole matrix. The supported analyses no longer reach it.
    frame = expression.get_expression_df()
    assert isinstance(frame, pd.DataFrame)
    assert not any(isinstance(dtype, pd.SparseDtype) for dtype in frame.dtypes)
    assert frame.shape[1] == len(genes)


def test_get_feature_ids_matches_the_dense_columns_without_a_dense_frame(tmp_path):
    """The cheap-reader decoupling: labels come from var_names, not a densify.

    ``get_feature_ids`` must equal ``get_expression_df().columns`` exactly (order
    included), and must still return them when the dense frame has not been built
    -- proving the label readers no longer depend on densification, the AM6
    groundwork for a sparse-first ``build``.
    """
    path, genes, _ = _sparse_h5ad(tmp_path)
    expression = ExpressionData(
        path=str(path),
        gene_list=genes,
        data_format="h5ad",
    )

    assert expression.get_feature_ids() == list(expression.get_expression_df().columns)

    # With no dense frame but a live AnnData, the labels are still available and
    # reading them does not build one.
    expression.df_expression_data = None
    assert expression.get_feature_ids() == [str(name) for name in expression.adata.var_names]
    assert expression.df_expression_data is None


def test_estimate_dense_bytes_predicts_the_realized_dense_frame(tmp_path):
    """The preflight budget is trustworthy: the estimate matches what is allocated."""
    path, genes, matrix = _sparse_h5ad(tmp_path)
    expression = ExpressionData(
        path=str(path),
        gene_list=genes,
        data_format="h5ad",
    )
    frame = expression.get_expression_df()

    estimated = estimate_dense_bytes(expression.adata)
    realized_values = frame.to_numpy(dtype=expression.adata.X.dtype).nbytes

    # The estimate is n_obs * n_vars * itemsize; the realized value buffer is the
    # same product, so the preflight can reject before allocation, not after.
    assert estimated == realized_values

    # And the dense materialization is materially larger than the sparse storage
    # it replaced -- the whole reason AM6 exists. At 5% density the stored CSR
    # buffers are a small fraction of the dense buffer.
    sparse_stored = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    assert sparse_stored < estimated / 2


def test_fingerprint_frame_is_invariant_to_storage_mode():
    """Identity is over logical values, not storage (decision M-08).

    ``fingerprint_frame`` gives a sparse column and its numerically identical
    dense twin the **same** identity, so making a workflow sparse-first does not
    move the dataset fingerprint across the AM5 provenance seam. A different fill
    value that encodes the same logical values must not change identity either,
    and columns whose values actually differ still diverge.
    """
    dense = pd.DataFrame(
        {"g0": [0.0, 1.5, 0.0, 2.0], "g1": [0.0, 0.0, 3.0, 0.0]},
        index=["s0", "s1", "s2", "s3"],
    )
    sparse_zero = dense.astype(pd.SparseDtype("float64", 0.0))
    # A fill value that appears nowhere in the data: same logical values, other
    # storage. Identity must still match the dense twin.
    sparse_other_fill = dense.astype(pd.SparseDtype("float64", -99.0))

    assert any(isinstance(dtype, pd.SparseDtype) for dtype in sparse_zero.dtypes)
    assert fingerprint_frame(sparse_zero) == fingerprint_frame(dense)
    assert fingerprint_frame(sparse_other_fill) == fingerprint_frame(dense)

    # Different logical values still produce different identities.
    changed = dense.copy()
    changed.iloc[0, 0] = 5.0
    assert fingerprint_frame(changed.astype(pd.SparseDtype("float64", 0.0))) != (
        fingerprint_frame(dense)
    )


def test_fingerprint_frame_is_invariant_to_storage_mode_float32():
    """Storage invariance holds for float32 too (the h5ad ``X`` dtype).

    ``hash_pandas_object`` alone is *not* storage-invariant for float32 -- a
    float32 sparse column hashes differently from its dense twin -- so
    ``fingerprint_frame`` densifies each sparse column before hashing. This is the
    guard on that: a ``from_spmatrix`` float32 frame (the shape ``identity_frame``
    builds for a sparse load) must fingerprint identically to its dense twin.
    """
    dense = pd.DataFrame(
        {"g0": [0.0, 1.5, 0.0, 2.0], "g1": [0.0, 0.0, 3.0, 0.0]},
        index=["s0", "s1", "s2", "s3"],
    ).astype(np.float32)
    spmatrix = sparse.csr_matrix(dense.to_numpy())
    from_spmatrix = pd.DataFrame.sparse.from_spmatrix(
        spmatrix, index=dense.index, columns=dense.columns
    )

    assert any(isinstance(dtype, pd.SparseDtype) for dtype in from_spmatrix.dtypes)
    assert fingerprint_frame(from_spmatrix) == fingerprint_frame(dense)


def test_deprecated_normalization_paths_are_deleted(tmp_path):
    """Only typed preprocessing crosses the public Reaction/Expression Data seam."""
    assert not hasattr(ExpressionData, "_apply_normalizations")
    assert not hasattr(ReactionData, "_apply_normalizations")
    assert not hasattr(reaction_data_module, "apply_neg_log_normalization")
    for callable_ in (
        ExpressionData,
        ExpressionData.create_with_validation,
        ExpressionData.build,
        ReactionData,
        ReactionData.create_with_validation,
    ):
        assert "normalization_options" not in inspect.signature(callable_).parameters

    path, genes, _ = _sparse_h5ad(tmp_path)
    expression = ExpressionData(
        path=str(path),
        gene_list=genes,
        preprocessing=ExpressionPreprocessingConfig(normalize_total=True, log1p=True),
        data_format="h5ad",
    )
    assert expression.preprocessing_report.method == "normalize_total+log1p"

"""The expression-matrix seam and its adapters (AM6 slices 2-3).

These pin: the ``DenseExpressionMatrix`` contract; the ``SparseExpressionMatrix``
contract and its byte-for-byte equivalence to the dense frame; and that routing
``ExpressionData``'s value readers through the seam returns exactly what reading
the dense frame directly returns, whichever adapter serves them. A sparse load
is served by the sparse adapter without densifying the whole matrix; the
equivalence tests are the guardrail that this changed only the path, not the
values.
"""

import tracemalloc

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from metabis import ExpressionData
from metabis.domain.expression_matrix import (
    DenseExpressionMatrix,
    DenseMaterializationError,
    ExpressionMatrix,
    SparseExpressionMatrix,
    freeze_matrix_buffers,
    is_sparse,
)
from metabis.domain.preprocessing import estimate_dense_bytes


def _sparse_h5ad(tmp_path, n_obs=32, n_vars=120, density=0.05, seed=1):
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
    return path, genes


@pytest.fixture
def expression(tmp_path):
    path, genes = _sparse_h5ad(tmp_path)
    return ExpressionData(
        path=str(path),
        gene_list=genes,
        data_format="h5ad",
    )


# --- freeze_matrix_buffers (M-11, AM9) --------------------------------------


def test_freeze_matrix_buffers_prevents_in_place_edits_of_a_sparse_matrix():
    """A frozen sparse ``X`` refuses same-shape in-place edits to every buffer."""
    matrix = sparse.random(6, 8, density=0.4, format="csr", dtype=np.float32)
    original = matrix.toarray()

    freeze_matrix_buffers(matrix)

    with pytest.raises(ValueError):
        matrix.data[0] = 0.0
    with pytest.raises(ValueError):
        matrix.indices[0] = 0
    with pytest.raises(ValueError):
        matrix.indptr[0] = 1
    # The values are unchanged and still readable.
    assert np.array_equal(matrix.toarray(), original)


def test_freeze_matrix_buffers_prevents_in_place_edits_of_a_dense_matrix():
    """A frozen dense ``X`` (a CSV load's storage) refuses in-place edits."""
    matrix = np.ones((4, 5), dtype=np.float32)

    freeze_matrix_buffers(matrix)

    with pytest.raises(ValueError):
        matrix[0, 0] = 9.0
    assert matrix[0, 0] == 1.0


def test_freeze_matrix_buffers_tolerates_none():
    """No matrix loaded is a no-op, not an error."""
    freeze_matrix_buffers(None)  # does not raise


def test_expression_freeze_seals_the_loaded_sparse_store(expression):
    """``ExpressionData.freeze`` freezes ``adata.X`` while reads stay intact."""
    expression.freeze()

    with pytest.raises(ValueError):
        expression.adata.X.data[0] = 0.0
    # A single-column read through the seam is unaffected.
    assert expression.matrix().feature_column(expression.adata.var_names[0]) is not None


# --- DenseExpressionMatrix unit contract -----------------------------------


def test_dense_adapter_reflects_the_frame():
    frame = pd.DataFrame(
        {"g0": [1.0, 2.0], "g1": [3.0, 4.0]}, index=["s0", "s1"]
    )
    matrix = DenseExpressionMatrix(frame)

    assert isinstance(matrix, ExpressionMatrix)
    assert matrix.sample_ids == ["s0", "s1"]
    assert matrix.feature_ids == ["g0", "g1"]
    pd.testing.assert_series_equal(matrix.feature_column("g1"), frame["g1"])
    # The escape hatch hands back the very same object, no copy.
    assert matrix.to_dense_frame() is frame


def test_dense_adapter_absent_feature_is_none():
    frame = pd.DataFrame({"g0": [1.0]}, index=["s0"])
    assert DenseExpressionMatrix(frame).feature_column("missing") is None


def test_dense_adapter_select_samples_slices_rows_in_order():
    frame = pd.DataFrame(
        {"g0": [1.0, 2.0, 3.0], "g1": [4.0, 5.0, 6.0]},
        index=["s0", "s1", "s2"],
    )
    matrix = DenseExpressionMatrix(frame)

    block = matrix.select_samples(["s2", "s0"])
    pd.testing.assert_frame_equal(block, frame.loc[["s2", "s0"]])
    assert block.index.tolist() == ["s2", "s0"]


def test_dense_adapter_select_samples_absent_sample_raises():
    frame = pd.DataFrame({"g0": [1.0, 2.0]}, index=["s0", "s1"])
    with pytest.raises(KeyError):
        DenseExpressionMatrix(frame).select_samples(["s0", "missing"])


def test_dense_adapter_feature_variances_are_pandas_variance():
    frame = pd.DataFrame(
        {"g0": [1.0, 2.0, 3.0], "g1": [4.0, 4.0, 10.0]}, index=["s0", "s1", "s2"]
    )
    pd.testing.assert_series_equal(
        DenseExpressionMatrix(frame).feature_variances(),
        frame.var(numeric_only=True),
    )


def test_dense_adapter_select_features_slices_columns_in_order():
    frame = pd.DataFrame(
        {"g0": [1.0, 2.0], "g1": [3.0, 4.0], "g2": [5.0, 6.0]},
        index=["s0", "s1"],
    )
    matrix = DenseExpressionMatrix(frame)

    block = matrix.select_features(["g2", "g0"])
    pd.testing.assert_frame_equal(block, frame[["g2", "g0"]])
    assert block.columns.tolist() == ["g2", "g0"]


def test_dense_adapter_select_features_absent_feature_raises():
    frame = pd.DataFrame({"g0": [1.0, 2.0]}, index=["s0", "s1"])
    with pytest.raises(KeyError):
        DenseExpressionMatrix(frame).select_features(["g0", "missing"])


# --- ExpressionData.matrix() adapter selection ------------------------------


def test_matrix_is_a_sparse_adapter_for_a_sparse_load(expression):
    """A sparse ``X`` routes to the sparse adapter, and serving it leaves X sparse."""
    matrix = expression.matrix()

    assert isinstance(matrix, SparseExpressionMatrix)
    # Reading identity and a column did not densify X in place -- the whole
    # point of the sparse path; the escape hatch still yields the dense values.
    matrix.sample_ids
    matrix.feature_ids
    matrix.feature_column(expression.get_feature_ids()[0])
    assert sparse.issparse(expression.adata.X)
    pd.testing.assert_frame_equal(
        matrix.to_dense_frame(), expression.get_expression_df()
    )


def test_matrix_is_a_dense_adapter_for_a_dense_load(tmp_path):
    """A dense ``X`` (a CSV load) keeps the dense adapter over the built frame."""
    frame = pd.DataFrame(
        {"g0": [1.0, 2.0, 3.0], "g1": [4.0, 5.0, 6.0]},
        index=["s0", "s1", "s2"],
    )
    path = tmp_path / "dense.csv"
    frame.to_csv(path)
    expression = ExpressionData(
        path=str(path),
        gene_list=["g0", "g1"],
        data_format="csv",
    )

    matrix = expression.matrix()
    assert isinstance(matrix, DenseExpressionMatrix)
    assert not is_sparse(expression.adata.X)
    # The dense adapter wraps the built frame by reference, no re-materialization.
    assert matrix.to_dense_frame() is expression.get_expression_df()


def test_matrix_operations_equal_the_direct_accessors(expression):
    matrix = expression.matrix()
    frame = expression.get_expression_df()

    assert matrix.sample_ids == frame.index.tolist()
    assert matrix.feature_ids == frame.columns.tolist()

    a_gene = frame.columns[0]
    pd.testing.assert_series_equal(matrix.feature_column(a_gene), frame[a_gene])


def test_routed_accessors_are_unchanged(expression):
    """The accessors that now go through the seam return what they always did."""
    frame = expression.get_expression_df()

    assert expression.get_sample_ids() == frame.index.tolist()
    assert expression.get_feature_ids() == [str(c) for c in frame.columns]

    a_gene = frame.columns[0]
    pd.testing.assert_series_equal(expression.get_gene_row(a_gene), frame[a_gene])
    assert expression.get_gene_row("not-a-gene") is None


def test_matrix_is_none_when_no_data_is_loaded():
    """An error/unloaded instance has no matrix, and the seam says so."""
    instance = ExpressionData.__new__(ExpressionData)
    instance.adata = None
    instance.df_expression_data = None

    assert instance.matrix() is None
    assert instance.get_sample_ids() == []
    assert instance.get_gene_row("g0") is None


def test_get_feature_ids_fallback_uses_the_seam_without_an_adata():
    """With no AnnData, feature ids come off the seam's dense adapter."""
    frame = pd.DataFrame({"g0": [1.0], "g1": [2.0]}, index=["s0"])
    instance = ExpressionData.__new__(ExpressionData)
    instance.adata = None
    instance.df_expression_data = frame

    assert instance.get_feature_ids() == ["g0", "g1"]


# --- SparseExpressionMatrix contract and dense/sparse equivalence -----------


def _sparse_anndata(rows, cols, values, fill=0.0, dtype=np.float32):
    """A sparse-CSR ``AnnData`` from an explicit dense specification.

    ``values`` is a dense list-of-rows; it is stored sparsely so the adapter is
    exercised over real sparse storage, while the dense twin for comparison is
    the same numbers.
    """
    dense = np.asarray(values, dtype=dtype)
    return AnnData(
        X=sparse.csr_matrix(dense),
        obs=pd.DataFrame(index=list(rows)),
        var=pd.DataFrame(index=list(cols)),
    )


def test_sparse_adapter_is_an_expression_matrix_and_reads_identity():
    adata = _sparse_anndata(
        ["s0", "s1", "s2"],
        ["g0", "g1"],
        [[0.0, 1.5], [2.0, 0.0], [0.0, 0.0]],
    )
    matrix = SparseExpressionMatrix(adata)

    assert isinstance(matrix, ExpressionMatrix)
    assert matrix.sample_ids == ["s0", "s1", "s2"]
    assert matrix.feature_ids == ["g0", "g1"]
    # Reading identity leaves the storage sparse -- no densification.
    assert sparse.issparse(adata.X)


def test_sparse_adapter_feature_column_matches_the_dense_twin():
    adata = _sparse_anndata(
        ["s0", "s1", "s2"],
        ["g0", "g1"],
        [[0.0, 1.5], [2.0, 0.0], [0.0, 3.0]],
    )
    twin = DenseExpressionMatrix(adata.to_df())
    sparse_matrix = SparseExpressionMatrix(adata)

    column = sparse_matrix.feature_column("g1")
    assert list(column) == [1.5, 0.0, 3.0]
    assert column.index.tolist() == ["s0", "s1", "s2"]
    pd.testing.assert_series_equal(column, twin.feature_column("g1"))
    # A single-column read does not densify the whole matrix.
    assert sparse.issparse(adata.X)


def test_sparse_adapter_absent_feature_is_none():
    adata = _sparse_anndata(["s0"], ["g0"], [[1.0]])
    assert SparseExpressionMatrix(adata).feature_column("missing") is None


def test_sparse_adapter_select_samples_matches_the_dense_twin():
    adata = _sparse_anndata(
        ["s0", "s1", "s2"],
        ["g0", "g1"],
        [[0.0, 1.5], [2.0, 0.0], [0.0, 3.0]],
    )
    twin = DenseExpressionMatrix(adata.to_df())
    sparse_matrix = SparseExpressionMatrix(adata)

    # A bounded block of two of the three samples, requested out of matrix order.
    block = sparse_matrix.select_samples(["s2", "s0"])
    assert block.index.tolist() == ["s2", "s0"]
    assert block.columns.tolist() == ["g0", "g1"]
    pd.testing.assert_frame_equal(block, twin.select_samples(["s2", "s0"]))
    # Materializing a group's rows did not densify the whole matrix in place.
    assert sparse.issparse(adata.X)


def test_sparse_adapter_select_samples_absent_sample_raises():
    adata = _sparse_anndata(["s0", "s1"], ["g0"], [[1.0], [2.0]])
    with pytest.raises(KeyError):
        SparseExpressionMatrix(adata).select_samples(["s0", "missing"])


def test_sparse_adapter_preserves_feature_and_sample_order():
    """Ordering is matrix order, not sorted -- both axes, and the columns."""
    adata = _sparse_anndata(
        ["s2", "s0", "s1"],
        ["g1", "g0", "g2"],
        [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0]],
    )
    matrix = SparseExpressionMatrix(adata)

    assert matrix.sample_ids == ["s2", "s0", "s1"]
    assert matrix.feature_ids == ["g1", "g0", "g2"]
    assert matrix.to_dense_frame().columns.tolist() == ["g1", "g0", "g2"]
    assert list(matrix.feature_column("g0")) == [1.0, 0.0, 0.0]


def test_duplicate_gene_names_are_deduplicated_before_the_sparse_adapter(tmp_path):
    """A sparse load with duplicate genes keeps the first occurrence, so the
    adapter serves unique features -- the dense frame's contract, unchanged.
    """
    dense = np.array(
        [[1.0, 9.0, 2.0], [3.0, 8.0, 4.0]], dtype=np.float32
    )
    path = tmp_path / "dup.h5ad"
    AnnData(
        X=sparse.csr_matrix(dense),
        obs=pd.DataFrame(index=["s0", "s1"]),
        var=pd.DataFrame(index=["g0", "g0", "g1"]),  # g0 duplicated
    ).write_h5ad(path)

    expression = ExpressionData(
        path=str(path),
        gene_list=["g0", "g1"],
        data_format="h5ad",
    )
    matrix = expression.matrix()

    assert isinstance(matrix, SparseExpressionMatrix)
    assert matrix.feature_ids == ["g0", "g1"]
    # First occurrence kept: g0 is the first column, not the second.
    assert list(matrix.feature_column("g0")) == [1.0, 3.0]
    pd.testing.assert_series_equal(
        matrix.feature_column("g0"), expression.get_expression_df()["g0"]
    )


def test_sparse_adapter_feature_variances_match_the_dense_twin():
    """Sparse-native variance equals pandas' per-column variance, and ranks the
    same features -- the input the reducers select on (AM6 slice 5).
    """
    adata = _sparse_anndata(
        ["s0", "s1", "s2", "s3"],
        ["g0", "g1", "g2"],
        [[0.0, 1.0, 5.0], [2.0, 0.0, 5.0], [0.0, 3.0, 5.0], [4.0, 0.0, 5.0]],
    )
    frame = adata.to_df()
    sparse_variances = SparseExpressionMatrix(adata).feature_variances()
    dense_variances = DenseExpressionMatrix(frame).feature_variances()

    assert sparse_variances.index.tolist() == ["g0", "g1", "g2"]
    np.testing.assert_allclose(
        sparse_variances.values, dense_variances.values, rtol=1e-6, atol=1e-6
    )
    # g2 is constant -> zero variance either way; g0/g1 vary.
    assert (
        sparse_variances.nlargest(2).index.tolist()
        == dense_variances.nlargest(2).index.tolist()
    )
    # The read did not densify X in place.
    assert sparse.issparse(adata.X)


def test_sparse_adapter_feature_variances_are_nan_below_two_samples():
    """ddof=1 variance is undefined for one sample; both adapters return NaN."""
    adata = _sparse_anndata(["s0"], ["g0", "g1"], [[1.0, 2.0]])
    variances = SparseExpressionMatrix(adata).feature_variances()
    assert variances.index.tolist() == ["g0", "g1"]
    assert variances.isna().all()


def test_sparse_adapter_select_features_matches_the_dense_twin():
    adata = _sparse_anndata(
        ["s0", "s1", "s2"],
        ["g0", "g1", "g2"],
        [[0.0, 1.0, 2.0], [3.0, 0.0, 4.0], [0.0, 5.0, 0.0]],
    )
    frame = adata.to_df()
    sparse_matrix = SparseExpressionMatrix(adata)

    # An out-of-order subset -- a top-N variable-feature block.
    block = sparse_matrix.select_features(["g2", "g0"])
    pd.testing.assert_frame_equal(block, frame[["g2", "g0"]])
    assert block.columns.tolist() == ["g2", "g0"]
    assert sparse.issparse(adata.X)


def test_sparse_adapter_select_features_absent_feature_raises():
    adata = _sparse_anndata(["s0", "s1"], ["g0"], [[1.0], [2.0]])
    with pytest.raises(KeyError):
        SparseExpressionMatrix(adata).select_features(["g0", "missing"])


def test_feature_variances_and_selection_agree_over_a_real_sparse_load(expression):
    """The ranking the reducer selects on is identical between the two adapters
    over a real sparse load, so both pick the same top-N variable features.
    """
    frame = expression.get_expression_df()
    sparse_variances = SparseExpressionMatrix(expression.adata).feature_variances()
    dense_variances = DenseExpressionMatrix(frame).feature_variances()

    np.testing.assert_allclose(
        sparse_variances.values, dense_variances.values, rtol=1e-5, atol=1e-8
    )
    for n_top in (10, 30, len(frame.columns)):
        assert (
            sparse_variances.nlargest(n_top).index.tolist()
            == dense_variances.nlargest(n_top).index.tolist()
        )


def test_sparse_and_dense_adapters_agree_over_a_real_sparse_load(expression):
    """Every seam operation agrees between the two adapters over the same data.

    The dense frame the load built and the sparse ``X`` it kept describe the same
    numbers, so the sparse adapter must be indistinguishable from a dense adapter
    over that frame -- for every column, not a sampled one.
    """
    frame = expression.get_expression_df()
    dense = DenseExpressionMatrix(frame)
    sparse_matrix = SparseExpressionMatrix(expression.adata)

    assert sparse_matrix.sample_ids == dense.sample_ids
    assert sparse_matrix.feature_ids == dense.feature_ids
    pd.testing.assert_frame_equal(sparse_matrix.to_dense_frame(), dense.to_dense_frame())
    for gene in frame.columns:
        pd.testing.assert_series_equal(
            sparse_matrix.feature_column(gene), dense.feature_column(gene)
        )


def test_select_samples_agrees_with_the_dense_frame_over_a_real_sparse_load(expression):
    """A bounded sample block from the sparse adapter equals the dense frame's
    rows -- every column -- so the group summaries and comparison that read it get
    identical numbers whichever adapter serves them (AM6 slice 4, MR-04).
    """
    frame = expression.get_expression_df()
    sparse_matrix = SparseExpressionMatrix(expression.adata)

    # An arbitrary, out-of-order subset -- a "group" of samples.
    group = [frame.index[5], frame.index[0], frame.index[9], frame.index[2]]
    pd.testing.assert_frame_equal(
        sparse_matrix.select_samples(group), frame.loc[group]
    )
    # A median over the block -- what the gene averages compute -- also agrees.
    pd.testing.assert_series_equal(
        sparse_matrix.select_samples(group).median(axis=0),
        frame.loc[group].median(axis=0),
    )


def test_a_sparse_load_serves_gene_rows_through_the_sparse_adapter(expression):
    """End to end: the routed accessors on a sparse load hit the sparse path and
    return exactly the dense values.
    """
    frame = expression.get_expression_df()
    assert isinstance(expression.matrix(), SparseExpressionMatrix)

    a_gene = frame.columns[0]
    pd.testing.assert_series_equal(expression.get_gene_row(a_gene), frame[a_gene])
    assert expression.get_sample_ids() == frame.index.tolist()


def test_single_column_read_does_not_densify_the_whole_matrix(tmp_path):
    """A feature column allocates one column's worth, not the whole dense matrix.

    The memory face of the sparse path: ``feature_column`` peaks far below the
    full dense frame ``to_dense_frame`` builds. This is the property AM6 exists
    for, measured rather than asserted by design.
    """
    from metabis.domain.preprocessing import estimate_dense_bytes

    path, genes = _sparse_h5ad(tmp_path, n_obs=200, n_vars=4000, density=0.05, seed=7)
    expression = ExpressionData(
        path=str(path),
        gene_list=genes,
        data_format="h5ad",
    )
    matrix = SparseExpressionMatrix(expression.adata)
    dense_bytes = estimate_dense_bytes(expression.adata)
    a_gene = expression.get_feature_ids()[len(genes) // 2]

    tracemalloc.start()
    tracemalloc.reset_peak()
    column = matrix.feature_column(a_gene)
    _, column_peak = tracemalloc.get_traced_memory()

    tracemalloc.reset_peak()
    _ = matrix.to_dense_frame()
    _, dense_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert column is not None
    # One column of 200 floats is a tiny fraction of a 200 x 4000 dense frame.
    assert column_peak < dense_bytes / 20
    assert column_peak < dense_peak / 10


def test_select_samples_block_is_bounded_below_the_full_dense_frame(tmp_path):
    """Materializing a group's samples peaks far below the whole dense matrix.

    The memory face of the bounded operation the summaries/comparison use: a
    handful of rows costs a small fraction of ``to_dense_frame``.
    """
    from metabis.domain.preprocessing import estimate_dense_bytes

    path, genes = _sparse_h5ad(tmp_path, n_obs=400, n_vars=4000, density=0.05, seed=11)
    expression = ExpressionData(
        path=str(path),
        gene_list=genes,
        data_format="h5ad",
    )
    matrix = SparseExpressionMatrix(expression.adata)
    dense_bytes = estimate_dense_bytes(expression.adata)
    # A group of 20 of the 400 samples -- a bounded block, 5% of the rows.
    group = [f"s{i}" for i in range(0, 40, 2)]

    tracemalloc.start()
    tracemalloc.reset_peak()
    block = matrix.select_samples(group)
    _, block_peak = tracemalloc.get_traced_memory()

    tracemalloc.reset_peak()
    _ = matrix.to_dense_frame()
    _, dense_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert block.shape == (20, 4000)
    # 20 of 400 rows: an order of magnitude below the full dense allocation.
    assert block_peak < dense_bytes / 10
    assert block_peak < dense_peak / 5


def test_select_features_block_is_bounded_below_the_full_dense_frame(tmp_path):
    """Materializing the top-N variable features peaks far below the whole matrix.

    The memory face of the reduction's bounded operation: the sklearn estimators
    fit on a top-N feature block, so ``select_features`` of a few hundred columns
    costs a small fraction of ``to_dense_frame`` (AM6 slice 5).
    """
    from metabis.domain.preprocessing import estimate_dense_bytes

    path, genes = _sparse_h5ad(tmp_path, n_obs=400, n_vars=4000, density=0.05, seed=13)
    expression = ExpressionData(
        path=str(path),
        gene_list=genes,
        data_format="h5ad",
    )
    matrix = SparseExpressionMatrix(expression.adata)
    dense_bytes = estimate_dense_bytes(expression.adata)
    # The top-N block the reducer would fit on: 200 of the 4000 features.
    top_features = matrix.feature_variances().nlargest(200).index.tolist()

    tracemalloc.start()
    tracemalloc.reset_peak()
    block = matrix.select_features(top_features)
    _, block_peak = tracemalloc.get_traced_memory()

    tracemalloc.reset_peak()
    _ = matrix.to_dense_frame()
    _, dense_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert block.shape == (400, 200)
    # 200 of 4000 columns: an order of magnitude below the full dense allocation.
    assert block_peak < dense_bytes / 10
    assert block_peak < dense_peak / 5


# --- Budgeted dense materialization (AM6 slice 6) --------------------------


def test_dense_adapter_estimate_matches_shape_and_dtype():
    frame = pd.DataFrame(
        np.zeros((3, 4), dtype=np.float32),
        index=["s0", "s1", "s2"],
        columns=["g0", "g1", "g2", "g3"],
    )
    # n_samples * n_features * itemsize, computed without allocating anything.
    assert DenseExpressionMatrix(frame).estimate_dense_bytes() == 3 * 4 * 4


def test_sparse_adapter_estimate_equals_the_module_estimator(expression):
    """The seam's estimate is the same trustworthy preflight the load path uses."""
    matrix = SparseExpressionMatrix(expression.adata)
    assert matrix.estimate_dense_bytes() == estimate_dense_bytes(expression.adata)


def test_to_dense_frame_without_a_budget_is_unchanged(expression):
    """No budget means densify unconditionally, exactly as earlier slices expect."""
    dense = SparseExpressionMatrix(expression.adata).to_dense_frame()
    pd.testing.assert_frame_equal(dense, expression.get_expression_df())


def test_to_dense_frame_under_budget_returns_the_frame():
    frame = pd.DataFrame(
        np.arange(6, dtype=np.float32).reshape(2, 3),
        index=["s0", "s1"],
        columns=["g0", "g1", "g2"],
    )
    matrix = DenseExpressionMatrix(frame)
    estimate = matrix.estimate_dense_bytes()
    # A budget exactly at the estimate is allowed (strictly-greater rejects).
    pd.testing.assert_frame_equal(matrix.to_dense_frame(max_bytes=estimate), frame)


def test_to_dense_frame_over_budget_is_rejected_before_allocation():
    """The preflight raises from the estimate, carrying the two byte counts.

    The point is rejection *before* allocation: the estimate reads only shape and
    dtype, so a matrix far larger than the budget is refused without building it.
    """
    frame = pd.DataFrame(
        np.zeros((100, 100), dtype=np.float64),
        index=[f"s{i}" for i in range(100)],
        columns=[f"g{j}" for j in range(100)],
    )
    matrix = DenseExpressionMatrix(frame)
    estimate = matrix.estimate_dense_bytes()

    with pytest.raises(DenseMaterializationError) as excinfo:
        matrix.to_dense_frame(max_bytes=estimate - 1)

    assert excinfo.value.estimated_bytes == estimate
    assert excinfo.value.max_bytes == estimate - 1


def test_sparse_over_budget_is_rejected_without_densifying(expression):
    """A sparse matrix over budget is refused, and its ``X`` stays sparse."""
    matrix = SparseExpressionMatrix(expression.adata)
    with pytest.raises(DenseMaterializationError):
        matrix.to_dense_frame(max_bytes=1)
    # The rejection did not build the dense frame: storage is still sparse.
    assert sparse.issparse(expression.adata.X)


# --- Budgeted selected dense blocks (M-11, AM9) ----------------------------


def test_dense_block_within_budget_is_returned():
    """A bounded block whose estimate is under the adapter budget materializes."""
    frame = pd.DataFrame(
        np.zeros((4, 3), dtype=np.float64),
        index=["s0", "s1", "s2", "s3"],
        columns=["g0", "g1", "g2"],
    )
    # select_samples of 2 rows x 3 features x 8 bytes = 48 bytes.
    matrix = DenseExpressionMatrix(frame, max_bytes=48)
    block = matrix.select_samples(["s0", "s2"])
    pd.testing.assert_frame_equal(block, frame.loc[["s0", "s2"]])


def test_dense_select_samples_over_budget_is_rejected_before_allocation():
    """The block preflight raises from its own (rows x all features) estimate."""
    frame = pd.DataFrame(
        np.zeros((10, 4), dtype=np.float64),
        index=[f"s{i}" for i in range(10)],
        columns=["g0", "g1", "g2", "g3"],
    )
    # A budget one byte under a 2-row block: 2 x 4 x 8 = 64 bytes.
    matrix = DenseExpressionMatrix(frame, max_bytes=63)
    with pytest.raises(DenseMaterializationError) as excinfo:
        matrix.select_samples(["s0", "s1"])
    assert excinfo.value.estimated_bytes == 64
    assert excinfo.value.max_bytes == 63


def test_dense_select_features_over_budget_is_rejected_before_allocation():
    """A feature block is estimated as (all samples x these features)."""
    frame = pd.DataFrame(
        np.zeros((5, 6), dtype=np.float64),
        index=[f"s{i}" for i in range(5)],
        columns=[f"g{j}" for j in range(6)],
    )
    # 5 samples x 2 features x 8 = 80 bytes; budget under that rejects.
    matrix = DenseExpressionMatrix(frame, max_bytes=79)
    with pytest.raises(DenseMaterializationError) as excinfo:
        matrix.select_features(["g0", "g3"])
    assert excinfo.value.estimated_bytes == 80


def test_no_budget_never_rejects_a_block():
    """The direct-library default (no budget) preflights nothing, as AM6 expects."""
    frame = pd.DataFrame(
        np.zeros((3, 3), dtype=np.float64), index=["s0", "s1", "s2"],
        columns=["g0", "g1", "g2"],
    )
    matrix = DenseExpressionMatrix(frame)  # max_bytes defaults to None
    # Every block returns, whatever its size.
    assert matrix.select_samples(["s0"]).shape == (1, 3)
    assert matrix.select_features(["g0"]).shape == (3, 1)


def test_sparse_block_over_budget_is_rejected_without_densifying(expression):
    """A sparse selected block over budget is refused, and ``X`` stays sparse."""
    matrix = SparseExpressionMatrix(expression.adata, max_bytes=1)
    with pytest.raises(DenseMaterializationError):
        matrix.select_samples(expression.get_sample_ids()[:2])
    with pytest.raises(DenseMaterializationError):
        matrix.select_features(expression.get_feature_ids()[:2])
    # No densification happened at either rejection.
    assert sparse.issparse(expression.adata.X)


def test_matrix_threads_the_load_budget_onto_the_seam(tmp_path):
    """``ExpressionData.matrix()`` carries ``dense_memory_budget_bytes`` (M-11).

    A worker reads expression through ``StableMatrixReference`` -> this
    ``matrix()``, so threading the budget here means every worker's bounded block
    is preflight-gated by the server's budget without per-call plumbing.
    """
    path, genes = _sparse_h5ad(tmp_path, n_obs=20, n_vars=60)
    expression = ExpressionData(
        path=str(path),
        gene_list=genes,
        data_format="h5ad",
        dense_memory_budget_bytes=1,  # every dense block is over this
    )
    matrix = expression.matrix()
    assert isinstance(matrix, SparseExpressionMatrix)
    # A bounded block is rejected because the load budget rode onto the adapter.
    with pytest.raises(DenseMaterializationError):
        matrix.select_samples(expression.get_sample_ids()[:1])
    # And the whole-matrix escape hatch falls back to the same adapter budget.
    with pytest.raises(DenseMaterializationError):
        matrix.to_dense_frame()
    # The load itself never densified: a sparse load keeps ``X`` sparse.
    assert expression.df_expression_data is None
    assert sparse.issparse(expression.adata.X)

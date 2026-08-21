"""AM6 slices 6-7: the dense-memory budget and storage-mode provenance.

Slice 6 added an injected ``dense_memory_budget_bytes`` that rejected an
over-budget densification before allocation, and a ``PreprocessingReport`` that
records storage mode. Slice 7 makes the sparse load sparse-first, which moves
where the budget bites:

- a **dense** input still densifies at load, so the budget rejects it there
  before the ``to_df`` allocation;
- a **sparse** input no longer densifies at load, so a large sparse dataset now
  *loads* under any budget; the budget instead guards the explicit escape hatch
  (``get_expression_df``), rejecting a forced full densification before allocation;
- the ``PreprocessingReport`` records storage mode, and now records **no**
  conversion at load (the dense path arrived dense; the sparse path stayed sparse).
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from metabis import ExpressionData
from metabis.domain.expression_matrix import DenseMaterializationError
from metabis.domain.preprocessing import estimate_dense_bytes


def _sparse_h5ad(tmp_path, n_obs=64, n_vars=200, density=0.05, seed=0):
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


def _dense_csv(tmp_path):
    frame = pd.DataFrame(
        {"g0": [1.0, 2.0, 3.0], "g1": [4.0, 5.0, 6.0]}, index=["s0", "s1", "s2"]
    )
    path = tmp_path / "dense.csv"
    frame.to_csv(path)
    return path, ["g0", "g1"]


def test_no_budget_loads_as_before(tmp_path):
    """The default (no budget) is unchanged: a sparse h5ad still loads and densifies."""
    path, genes = _sparse_h5ad(tmp_path)
    expression = ExpressionData(
        path=str(path), gene_list=genes, data_format="h5ad"
    )
    assert expression.get_expression_df().shape[1] == len(genes)


def test_generous_budget_loads(tmp_path):
    path, genes = _sparse_h5ad(tmp_path)
    expression = ExpressionData(
        path=str(path),
        gene_list=genes,
        data_format="h5ad",
        dense_memory_budget_bytes=1_000_000_000,
    )
    assert expression.get_expression_df().shape[1] == len(genes)


def test_a_large_sparse_load_is_not_rejected_at_load(tmp_path):
    """Slice 7: a sparse dataset loads under any budget -- it never densifies at load.

    The whole point of AM6: a large sparse dataset must be loadable and analysable
    through its bounded reads. The dense-memory budget no longer rejects it at load.
    """
    path, genes = _sparse_h5ad(tmp_path)
    expression = ExpressionData(
        path=str(path),
        gene_list=genes,
        data_format="h5ad",
        dense_memory_budget_bytes=1,
    )
    # It loaded, stayed sparse, and carries no dense frame.
    assert expression.df_expression_data is None


def test_over_budget_escape_hatch_is_rejected_before_allocation(tmp_path):
    """The budget now guards the escape hatch: a forced full densify is rejected."""
    path, genes = _sparse_h5ad(tmp_path)
    expression = ExpressionData(
        path=str(path),
        gene_list=genes,
        data_format="h5ad",
        dense_memory_budget_bytes=1,
    )
    with pytest.raises(DenseMaterializationError) as excinfo:
        expression.get_expression_df()
    assert excinfo.value.max_bytes == 1
    assert excinfo.value.estimated_bytes > 1
    # The rejection did not leave a partial dense frame behind.
    assert expression.df_expression_data is None


def test_escape_hatch_budget_boundary_is_the_dense_estimate(tmp_path):
    """A budget exactly at the estimate densifies; one byte under it rejects."""
    path, genes = _sparse_h5ad(tmp_path)
    reference = ExpressionData(
        path=str(path), gene_list=genes, data_format="h5ad"
    )
    estimate = estimate_dense_bytes(reference.adata)

    at_budget = ExpressionData(
        path=str(path),
        gene_list=genes,
        data_format="h5ad",
        dense_memory_budget_bytes=estimate,
    )
    assert at_budget.get_expression_df().shape[1] == len(genes)

    under_budget = ExpressionData(
        path=str(path),
        gene_list=genes,
        data_format="h5ad",
        dense_memory_budget_bytes=estimate - 1,
    )
    with pytest.raises(DenseMaterializationError):
        under_budget.get_expression_df()


def test_over_budget_dense_load_is_rejected_before_allocation(tmp_path):
    """A dense input densifies at load, so the budget still rejects it there."""
    path, genes = _dense_csv(tmp_path)
    with pytest.raises(DenseMaterializationError) as excinfo:
        ExpressionData(
            path=str(path),
            gene_list=genes,
            data_format="csv",
            dense_memory_budget_bytes=1,
        )
    assert excinfo.value.max_bytes == 1
    assert excinfo.value.estimated_bytes > 1


def test_sparse_load_records_storage_and_no_conversion(tmp_path):
    """Provenance names the sparse input and records that it converted nothing."""
    path, genes = _sparse_h5ad(tmp_path)
    expression = ExpressionData(
        path=str(path), gene_list=genes, data_format="h5ad"
    )
    report = expression.preprocessing_report
    assert report.input_storage == "sparse"
    # Slice 7: the load no longer densifies, so no conversion is recorded.
    assert report.conversions == ()


def test_dense_load_records_no_conversion(tmp_path):
    """A dense CSV arrived dense and converted nothing."""
    path, genes = _dense_csv(tmp_path)
    expression = ExpressionData(
        path=str(path), gene_list=genes, data_format="csv"
    )
    report = expression.preprocessing_report
    assert report.input_storage == "dense"
    assert report.conversions == ()

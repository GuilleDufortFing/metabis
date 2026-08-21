import pandas as pd
from anndata import AnnData
from scipy import sparse

from metabis import ExpressionData


def test_expression_tsv_loads_samples_in_rows(fixture_dir):
    path = fixture_dir / "expression.tsv"

    expression = ExpressionData(
        path=str(path),
        gene_list=["ENSG000001", "ENSG000002", "ENSG000003"],
        data_format="csv",
        csv_separator="\t",
        samples_in="rows",
    )

    result = expression.get_expression_df()
    assert list(result.index) == [
        "T_naive_ctrl_1",
        "T_naive_stim_1",
        "B_cell_ctrl_1",
        "B_cell_stim_1",
        "expression_only_1",
    ]
    assert list(result.columns) == ["ENSG000001", "ENSG000002", "ENSG000003"]
    assert result.loc["T_naive_stim_1", "ENSG000001"] == 1
    assert isinstance(result, pd.DataFrame)
    assert expression.preprocessing_report.method == "none"
    assert expression.preprocessing_report.input_shape == result.shape
    assert expression.preprocessing_report.output_shape == result.shape


def test_dense_table_and_sparse_h5ad_preserve_the_same_identity(fixture_dir, tmp_path):
    table = pd.read_csv(fixture_dir / "expression.tsv", sep="\t", index_col=0)
    genes = ["ENSG000001", "ENSG000002", "ENSG000003"]
    h5ad_path = tmp_path / "expression.h5ad"
    AnnData(
        X=sparse.csr_matrix(table.to_numpy()),
        obs=pd.DataFrame(index=table.index.copy()),
        var=pd.DataFrame(index=table.columns.copy()),
    ).write_h5ad(h5ad_path)

    dense = ExpressionData(
        path=str(fixture_dir / "expression.tsv"),
        gene_list=genes,
        data_format="csv",
        csv_separator="\t",
        samples_in="rows",
    ).get_expression_df()
    from_sparse = ExpressionData(
        path=str(h5ad_path),
        gene_list=genes,
        data_format="h5ad",
    ).get_expression_df()

    assert list(from_sparse.index) == list(dense.index)
    assert list(from_sparse.columns) == list(dense.columns)

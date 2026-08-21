import numpy as np
import pandas as pd


def test_tiny_fixtures_cover_revision_edge_cases(fixture_dir):
    single = pd.read_csv(fixture_dir / "reaction_single.tsv", sep="\t", index_col=0)
    dual = pd.read_csv(fixture_dir / "reaction_dual.tsv", sep="\t", index_col=0)
    expression = pd.read_csv(fixture_dir / "expression.tsv", sep="\t", index_col=0)
    metadata = pd.read_csv(fixture_dir / "metadata.tsv", sep="\t", index_col=0)
    embedding = pd.read_csv(fixture_dir / "embedding.tsv", sep="\t", index_col=0)

    shared = {
        "T_naive_ctrl_1",
        "T_naive_stim_1",
        "B_cell_ctrl_1",
        "B_cell_stim_1",
    }
    assert shared <= set(single.index) & set(expression.index) & set(metadata.index)
    assert "reaction_only_1" in set(single.index) - set(expression.index)
    assert "expression_only_1" in set(expression.index) - set(single.index)

    assert {"MAR00001_pos", "MAR00001_neg"} <= set(dual.columns)
    assert (single["MAR00004"] == 0).all()
    assert single["MAR00003"].nunique() == 1
    assert (single["MAR00001"] < 0).any()
    assert not np.isfinite(single["MAR00005"]).all()

    assert {"cell_type", "condition", "label", "custom_union"} <= set(metadata.columns)
    assert metadata["custom_union"].nunique() >= 2
    assert list(embedding.index) != sorted(embedding.index)
    assert {"PCA_1", "PCA_2", "UMAP_1", "UMAP_2"} <= set(embedding.columns)

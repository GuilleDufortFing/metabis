import numpy as np
import pandas as pd

from metabis.analysis.dimensionality_reduction import DimensionalityReduction
from metabis.domain.expression_matrix import DenseExpressionMatrix


def frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.normal(size=(12, 8)),
        index=[f"sample-{i}" for i in range(12)],
        columns=[f"feature-{i}" for i in range(8)],
    )


def test_feature_count_and_preprocessing_revision_create_distinct_pca_results():
    reduction = DimensionalityReduction()
    data = frame()

    first = reduction.calculate_pca(DenseExpressionMatrix(data), "reactions", "test", n_variable_features=3)
    second = reduction.calculate_pca(DenseExpressionMatrix(data), "reactions", "test", n_variable_features=5)
    revised = reduction.calculate_pca(
        DenseExpressionMatrix(data),
        "reactions",
        "test",
        n_variable_features=5,
        preprocessing_revision="2",
    )

    assert first["key"] != second["key"] != revised["key"]
    assert len(reduction.results) == 3
    assert second["result"].feature_names != first["result"].feature_names


def test_identical_complete_request_reuses_cached_result():
    reduction = DimensionalityReduction()
    data = frame()
    first = reduction.calculate_pca(DenseExpressionMatrix(data), "reactions", "test", n_variable_features=4)
    repeated = reduction.calculate_pca(
        DenseExpressionMatrix(data), "reactions", "test", n_variable_features=4
    )

    assert repeated["reused"] is True
    assert repeated["result"] is first["result"]


def test_umap_parameters_are_part_of_cache_identity():
    reduction = DimensionalityReduction()
    data = frame()
    first = reduction.calculate_umap(
        DenseExpressionMatrix(data), "reactions", "test", n_neighbors=3, min_dist=0.1
    )
    second = reduction.calculate_umap(
        DenseExpressionMatrix(data), "reactions", "test", n_neighbors=4, min_dist=0.2
    )

    assert first["key"] != second["key"]
    assert len(reduction.results) == 2

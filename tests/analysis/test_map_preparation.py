"""How features are prepared before a reduction draws its map."""

import numpy as np
import pandas as pd
import pytest

from metabis.analysis.dimensionality_reduction import DimensionalityReduction, _reduction_parameters
from metabis.analysis.map_preparation import (
    prepare_features,
    scale_features,
    signed_cube_root,
    standardized_variances,
)
from metabis.domain.expression_matrix import DenseExpressionMatrix


def fluxes(n_samples: int = 60) -> pd.DataFrame:
    """Features whose spread follows their mean, as fluxes' does, and three that do not."""
    rng = np.random.default_rng(7)
    means = np.geomspace(1e-3, 10, 40)
    data = {f"typical-{i}": rng.normal(mean, 0.2 * mean, n_samples) for i, mean in enumerate(means)}
    # Twice the typical spread for its mean: the most variable for its level.
    data["unusual"] = rng.normal(0.05, 0.05, n_samples)
    # Mostly negative, as an uptake flux: no logarithm of its mean.
    data["uptake"] = rng.normal(-5.0, 3.0, n_samples)
    data["constant"] = np.full(n_samples, 1.0)
    return pd.DataFrame(data, index=[f"sample-{i}" for i in range(n_samples)])


def test_the_signed_cube_root_keeps_each_values_sign():
    frame = pd.DataFrame({"flux": [-8.0, 0.0, 1e-9, 27.0]})

    assert signed_cube_root(frame)["flux"].tolist() == pytest.approx([-2.0, 0.0, 1e-3, 3.0])


def test_standardized_variance_ranks_a_feature_by_its_spread_for_its_mean():
    frame = fluxes()
    standardized = standardized_variances(frame)

    # By raw variance the largest means come first; by standardized variance the
    # feature spread most for its level does.
    assert frame.var().drop(["uptake"]).idxmax() == "typical-39"
    assert standardized.idxmax() == "unusual"
    # A mean that is not positive cannot be placed on the trend, nor can no spread.
    assert "uptake" not in standardized.index
    assert "constant" not in standardized.index


def test_scaling_gives_each_feature_mean_zero_and_unit_spread_clipped_at_ten():
    values = np.column_stack([np.r_[np.zeros(199), 1000.0], np.arange(200.0)])
    scaled = scale_features(values)

    assert scaled.mean(axis=0)[1] == pytest.approx(0.0)
    assert scaled.std(axis=0, ddof=1)[1] == pytest.approx(1.0)
    # The lone outlier sits about 14 standard deviations out, so it is clipped.
    assert scaled[:, 0].max() == 10.0


def test_read_as_loaded_the_features_are_the_most_variable():
    frame = fluxes()
    values, kept, how = prepare_features(frame, 5)

    assert how == "variance"
    assert kept == frame.var().nlargest(5).index.tolist()
    assert np.array_equal(values, frame[kept].to_numpy())


def test_a_prepared_reduction_says_how_and_is_kept_apart_from_the_plain_one():
    reduction = DimensionalityReduction()
    frame = fluxes()
    plain = reduction.calculate_pca(DenseExpressionMatrix(frame), "reactions", "test", n_variable_features=10)
    prepared = reduction.calculate_pca(
        DenseExpressionMatrix(frame),
        "reactions",
        "test",
        n_variable_features=10,
        feature_transform="signed_cuberoot",
        feature_selection="standardized_variance",
        scale_features=True,
    )

    assert plain["key"] != prepared["key"]
    assert "Selected top 10 features by standardized variance." in prepared["messages"]
    assert "uptake" not in prepared["result"].feature_names
    # The plain fit records nothing new; the prepared one records all three settings.
    assert "feature_transform" not in dict(_reduction_parameters(plain["key"]))
    assert dict(_reduction_parameters(prepared["key"]))["feature_selection"] == "standardized_variance"


def test_an_unknown_preparation_is_refused():
    with pytest.raises(ValueError, match="Unknown feature transform"):
        prepare_features(fluxes(), 5, transform="log")

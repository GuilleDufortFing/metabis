import numpy as np
import pandas as pd

from metabis.domain.reaction_data import apply_cbrt_normalization


def test_cubic_root_normalization_preserves_axes_and_signs(reaction_frame):
    finite = reaction_frame[["MAR00001", "MAR00002", "MAR00003", "MAR00004"]]

    result = apply_cbrt_normalization(finite)

    pd.testing.assert_index_equal(result.index, finite.index)
    pd.testing.assert_index_equal(result.columns, finite.columns)
    np.testing.assert_allclose(result.to_numpy() ** 3, finite.to_numpy())

"""A value off a distribution is coloured on its scale as an element of it would be."""

import pandas as pd
import pytest

from metabis.analysis.distribution_binner import DistributionBinner


@pytest.mark.parametrize("spread_method", ["std", "mad", "asymmetric_saturating"])
def test_a_value_bins_as_a_registered_element_with_that_value_does(spread_method):
    # A transport family's Delta score is coloured on its subsystems' scale. A
    # second, older bin_for_value once shadowed this one and ignored the spread
    # method, so a value off a std or mad scale could land a bin away.
    scores = pd.Series({"a": -0.9, "b": -0.2, "c": 0.0, "d": 0.35, "e": 0.8, "f": 1.0})
    binner = DistributionBinner(scores, center_method="zero", spread_method=spread_method)

    assert [binner.bin_for_value(value) for value in scores] == list(binner.get_all_bins())


def test_a_value_that_is_not_a_finite_number_is_neutral():
    binner = DistributionBinner(pd.Series({"a": -1.0, "b": 1.0}), spread_method="std")

    assert binner.bin_for_value(float("nan")) == 0
    assert binner.bin_for_value("not a number") == 0

import numpy as np
import pandas as pd
import pytest

from metabis import AdjustmentColumn, ComparisonAdjustment, GroupComparison
from metabis.analysis.comparison_membership import GroupComparisonSpec

SHEET = pd.DataFrame(
    {
        "reads": [100.0, 1000.0, 10.0, 10000.0],
        "batch": ["b", "a", "a", "c"],
        "flat": ["x", "x", "x", "x"],
    },
    index=["s1", "s2", "s3", "s4"],
)


def test_numeric_columns_enter_as_slopes_on_their_scale():
    adjustment = ComparisonAdjustment.from_metadata(
        SHEET, [AdjustmentColumn("reads", log10=True)]
    )

    design = adjustment.design(["s4", "s1", "s2"])

    assert list(design.columns) == ["log10 reads"]
    np.testing.assert_allclose(design["log10 reads"], [4.0, 2.0, 3.0])
    assert adjustment.labels == ("log10 reads",)


def test_text_columns_enter_as_one_indicator_per_extra_level():
    adjustment = ComparisonAdjustment.from_metadata(SHEET, [AdjustmentColumn("batch")])

    design = adjustment.design(["s1", "s2", "s3", "s4"])

    assert list(design.columns) == ["batch: b", "batch: c"]
    assert design["batch: b"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert design["batch: c"].tolist() == [0.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize(
    ("column", "samples", "message"),
    [
        (AdjustmentColumn("flat"), ["s1", "s2"], "one value across"),
        (AdjustmentColumn("batch", log10=True), ["s1", "s2"], "not numeric"),
        (AdjustmentColumn("reads"), ["s1", "s9"], "no row for 1"),
    ],
)
def test_unusable_columns_are_explained(column, samples, message):
    adjustment = ComparisonAdjustment.from_metadata(SHEET, [column])

    with pytest.raises(ValueError, match=message):
        adjustment.design(samples)


def test_log10_needs_positive_values():
    sheet = SHEET.assign(reads=[0.0, 1.0, 2.0, 3.0])
    adjustment = ComparisonAdjustment.from_metadata(
        sheet, [AdjustmentColumn("reads", log10=True)]
    )

    with pytest.raises(ValueError, match="above zero"):
        adjustment.design(["s1", "s2"])


def test_unknown_columns_are_rejected():
    with pytest.raises(ValueError, match="no column depth"):
        ComparisonAdjustment.from_metadata(SHEET, [AdjustmentColumn("depth")])


def test_only_the_moderated_t_test_adjusts():
    adjustment = ComparisonAdjustment.from_metadata(SHEET, [AdjustmentColumn("reads")])
    spec = GroupComparisonSpec.pairwise(("A",), ("B",))

    with pytest.raises(ValueError, match="Only the moderated t-test"):
        GroupComparison(spec=spec, test="wilcoxon", adjustment=adjustment)

    comparison = GroupComparison(spec=spec, test="moderated_t", adjustment=adjustment)
    assert comparison.adjustment is adjustment


def test_numbers_read_as_text_are_numbers():
    sheet = SHEET.assign(reads=["100", "1000", "10", "10000"])
    adjustment = ComparisonAdjustment.from_metadata(
        sheet, [AdjustmentColumn("reads", log10=True)]
    )

    np.testing.assert_allclose(adjustment.design(["s1", "s2"])["log10 reads"], [2.0, 3.0])

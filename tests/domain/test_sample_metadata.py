import pandas as pd
import pytest

from metabis import SampleMetadata


def test_metadata_delimiter_id_selection_alignment_and_labels(fixture_dir):
    metadata = SampleMetadata.from_table(
        fixture_dir / "metadata.tsv",
        sample_id_column="sample_id",
        display_label_column="label",
    )

    report = metadata.align(
        ["T_naive_ctrl_1", "T_naive_stim_1", "B_cell_ctrl_1", "B_cell_stim_1"],
        policy="reaction_required",
    )

    assert report.metadata_only == ("metadata_only_1",)
    assert len(report.aligned_samples) == 4
    assert metadata.get_display_label("B_cell_ctrl_1") == "B-cell control"
    assert metadata.get_display_label("unknown") == "unknown"


def test_strict_alignment_reports_mismatches(fixture_dir):
    metadata = SampleMetadata.from_table(fixture_dir / "metadata.tsv")

    with pytest.raises(ValueError, match="4 metadata-only samples"):
        metadata.align(["T_naive_ctrl_1"], policy="strict")


def test_alignment_accepts_pandas_index_sequences():
    metadata = SampleMetadata(pd.DataFrame({"kind": ["A", "B"]}, index=["S1", "S2"]))
    report = metadata.align(pd.Index(["S1", "S2"]), pd.Index(["S1", "S2"]))
    assert report.is_strict_match


@pytest.mark.parametrize("ids", [["", "S2"], ["S1", "S1"]])
def test_blank_and_duplicate_ids_are_rejected(ids):
    with pytest.raises(ValueError, match="Sample IDs must be"):
        SampleMetadata(pd.DataFrame({"kind": ["A", "B"]}, index=ids))


def test_the_sample_id_column_may_be_named_as_its_own_display_label(fixture_dir):
    """It is the index by the time labels resolve, so it means the canonical ID.

    Rejecting it made a column the sheet plainly contains look like a typo.
    """
    metadata = SampleMetadata.from_table(
        fixture_dir / "metadata.tsv",
        sample_id_column="sample_id",
        display_label_column="sample_id",
    )

    assert metadata.display_label_column is None
    assert metadata.get_display_label("B_cell_ctrl_1") == "B_cell_ctrl_1"


def test_a_display_label_column_that_is_absent_is_still_rejected(fixture_dir):
    with pytest.raises(ValueError, match="Display label column 'absent' was not found"):
        SampleMetadata.from_table(
            fixture_dir / "metadata.tsv",
            sample_id_column="sample_id",
            display_label_column="absent",
        )

import pandas as pd
import pytest

from metabis import SampleMetadata
from metabis.domain.sample_metadata import read_metadata_table


def test_metadata_delimiter_id_selection_alignment_and_labels(fixture_dir):
    metadata = SampleMetadata.from_table(
        fixture_dir / "metadata.tsv",
        sample_id_column="sample_id",
        display_label_column="label",
    )

    report = metadata.align(
        ["T_naive_ctrl_1", "T_naive_stim_1", "B_cell_ctrl_1", "B_cell_stim_1"],
        policy="partial",
    )

    assert report.metadata_only == ("metadata_only_1",)
    assert len(report.aligned_samples) == 4
    assert metadata.get_display_label("B_cell_ctrl_1") == "B-cell control"
    assert metadata.get_display_label("unknown") == "unknown"


def test_strict_alignment_reports_mismatches(fixture_dir):
    metadata = SampleMetadata.from_table(fixture_dir / "metadata.tsv")

    with pytest.raises(ValueError, match="4 of its rows name no sample of the reaction data"):
        metadata.align(["T_naive_ctrl_1"], policy="strict")


def test_a_partial_sheet_leaves_the_samples_it_lacks_out():
    metadata = SampleMetadata(pd.DataFrame({"kind": ["A", "B"]}, index=["S1", "extra"]))

    report = metadata.align(["S1", "S2", "S3"], policy="partial")

    assert report.aligned_samples == ("S1",)
    assert report.missing_from_metadata == ("S2", "S3")
    with pytest.raises(ValueError, match="No sample of the reaction data has a row in the sample sheet"):
        metadata.align(["S7", "S8"], policy="partial")


def test_a_sheet_missing_samples_names_some_of_them():
    metadata = SampleMetadata(pd.DataFrame({"kind": ["A"]}, index=["S1"]))

    with pytest.raises(ValueError) as raised:
        metadata.align(["S1", "S2", "S3"])

    assert str(raised.value) == (
        "The sample sheet does not match the data: 2 samples of the reaction data's 3 have no row in it (S2, S3)."
    )


@pytest.mark.parametrize(
    ("name", "text"),
    [
        # Excel writes a byte-order mark before the first column's name.
        ("sheet.csv", "\ufeffsample_id,condition\nS1,a\nS2,b\n"),
        ("sheet.tsv", "\ufeffsample_id\tcondition\nS1\ta\nS2\tb\n"),
        ("sheet.csv", "sample_id;condition\nS1;a\nS2;b\n"),
        ("sheet.csv", " sample_id , condition\nS1,a\nS2,b\n"),
    ],
)
def test_a_sheet_reads_its_own_column_names(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")

    assert list(read_metadata_table(path).columns) == ["sample_id", "condition"]
    assert SampleMetadata.from_table(path, sample_id_column="sample_id").sample_ids == ("S1", "S2")


def test_a_one_column_sheet_is_not_split_on_a_letter(tmp_path):
    path = tmp_path / "sheet.tsv"
    path.write_text("sample_id\nS1\nS2\n")

    assert list(read_metadata_table(path).columns) == ["sample_id"]


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

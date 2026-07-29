"""The published example files must say what they are (WP8/D-14).

Every generated example carried a `.csv` suffix over tab-delimited content until
2026-07-24. Nothing broke -- the loaders detect the delimiter -- but the file name
is the one part of the contract a user reads before uploading anything, and it was
false. It forced a caveat into the README, the demo notebooks, the third-party
notices, and the reviewer-facing format table.

These tests run the generator's file-writing stages against fixtures. Downloading
is deliberately out of scope: it needs the network, and the defect was never in
the download.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from metabis.domain.format_detection import _detect_csv_separator

GENERATOR = Path(__file__).resolve().parents[2] / "data" / "download_and_process_compass.py"

# The contract: a suffix names a delimiter, and the file must use that one.
DELIMITER_FOR_SUFFIX = {".csv": ",", ".tsv": "\t"}


def _load_generator():
    spec = importlib.util.spec_from_file_location("compass_example_generator", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _load_generator()


def assert_extension_matches_delimiter(path: Path) -> None:
    """The suffix is a claim about the bytes; hold the file to it."""
    expected = DELIMITER_FOR_SUFFIX[path.suffix.lower()]
    detected = _detect_csv_separator(str(path))
    assert detected == expected, (
        f"{path.name} is delimited with {detected!r}, but its suffix promises {expected!r}"
    )


def test_the_declared_outputs_are_tsv(generator):
    for source_name, output_name in generator.EXAMPLE_OUTPUTS.items():
        assert source_name.endswith(".tsv")
        assert output_name.endswith(".tsv"), output_name


def test_written_matrices_match_the_delimiter_their_name_promises(generator, tmp_path):
    for source_name in generator.EXAMPLE_OUTPUTS:
        frame = pd.DataFrame(
            {"Th17p_1": [1.0, 2.0], "Th17n_1": [3.0, 4.0]},
            index=["MAR00001", "MAR00002"],
        )
        frame.to_csv(tmp_path / source_name, sep="\t")

    generator.write_metabis_examples(tmp_path)

    for output_name in generator.EXAMPLE_OUTPUTS.values():
        written = tmp_path / output_name
        assert written.is_file(), output_name
        assert_extension_matches_delimiter(written)


def test_the_matrices_keep_their_sample_and_feature_ids(generator, tmp_path):
    """A rename must not become a silent re-serialisation of the data."""
    frame = pd.DataFrame(
        {"Th17p_1": [1.5, 2.5], "Th17n_1": [3.5, 4.5]},
        index=["MAR00001", "MAR00002"],
    )
    frame.index.name = "reaction"
    for source_name in generator.EXAMPLE_OUTPUTS:
        frame.to_csv(tmp_path / source_name, sep="\t")

    generator.write_metabis_examples(tmp_path)

    for output_name in generator.EXAMPLE_OUTPUTS.values():
        written = pd.read_csv(tmp_path / output_name, sep="\t", index_col=0)
        pd.testing.assert_frame_equal(written, frame)


def test_the_sample_sheet_is_tab_delimited_and_keyed_on_the_shipped_sample_id(
    generator, tmp_path
):
    pd.DataFrame(
        {
            "cell_id": ["1", "2"],
            "cell_type": ["Th17p", "Th17n"],
            "SRX_accession": ["SRX1", "SRX2"],
        }
    ).to_csv(tmp_path / "cell_metadata.csv", index=False)

    generator.write_sample_metadata(tmp_path)

    sheet = tmp_path / "Th17_Compass_metadata.tsv"
    assert_extension_matches_delimiter(sheet)
    frame = pd.read_csv(sheet, sep="\t")
    # The matrices name samples `{cell_type}_{cell_id}`; the sheet aligns by exact
    # sample ID, so its first column has to carry that joined form.
    assert list(frame.columns[:2]) == ["sample", "cell_type"]
    assert list(frame["sample"]) == ["Th17p_1", "Th17n_2"]
    assert "SRX_accession" in frame.columns


def test_the_generator_cleans_up_its_intermediates_but_not_its_outputs(
    generator, tmp_path
):
    for name in (*generator.EXAMPLE_OUTPUTS, "reactions.tsv", "cell_metadata.csv"):
        (tmp_path / name).write_text("x")
    for name in generator.EXAMPLE_OUTPUTS.values():
        (tmp_path / name).write_text("x")

    generator.cleanup_files(tmp_path)

    for name in generator.EXAMPLE_OUTPUTS.values():
        assert (tmp_path / name).exists(), f"{name} is a published output, not an intermediate"
    for name in (*generator.EXAMPLE_OUTPUTS, "reactions.tsv", "cell_metadata.csv"):
        assert not (tmp_path / name).exists(), name

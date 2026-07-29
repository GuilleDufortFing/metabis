"""Gzipped text uploads (A-15 Tier 1).

Reaction and expression inputs accept ``.csv``/``.tsv`` and their ``.gz``
variants so a large sparse matrix uploads compressed. ``pandas.read_csv`` infers
gzip from the ``.gz`` suffix, so the domain loaders read a compressed file with
no change; the one hand-rolled reader — separator detection — is covered here.
"""

import gzip
import shutil
from pathlib import Path

import pandas as pd

from metabis import ExpressionData
from metabis.domain.format_detection import _detect_csv_separator


def _gzip_fixture(source: Path, destination: Path) -> Path:
    """Write a gzip-compressed copy of ``source`` at ``destination``."""
    with open(source, "rb") as raw, gzip.open(destination, "wb") as compressed:
        shutil.copyfileobj(raw, compressed)
    return destination


def test_separator_detection_reads_a_gzipped_reaction_file(fixture_dir, tmp_path):
    plain = fixture_dir / "reaction_single.tsv"
    gzipped = _gzip_fixture(plain, tmp_path / "reaction_single.tsv.gz")

    # The gzip magic bytes, not the extension, drive decompression.
    assert gzipped.read_bytes()[:2] == b"\x1f\x8b"
    assert _detect_csv_separator(str(gzipped)) == "\t"
    assert _detect_csv_separator(str(plain)) == "\t"


def test_separator_detection_reads_a_gzipped_expression_file(fixture_dir, tmp_path):
    gzipped = _gzip_fixture(
        fixture_dir / "expression.tsv", tmp_path / "expression.tsv.gz"
    )
    assert _detect_csv_separator(str(gzipped)) == "\t"


def test_expression_loads_from_a_gzipped_tsv_identically(fixture_dir, tmp_path):
    genes = ["ENSG000001", "ENSG000002", "ENSG000003"]
    gzipped = _gzip_fixture(
        fixture_dir / "expression.tsv", tmp_path / "expression.tsv.gz"
    )

    plain = ExpressionData(
        path=str(fixture_dir / "expression.tsv"),
        gene_list=genes,
        data_format="csv",
        csv_separator="\t",
        samples_in="rows",
    ).get_expression_df()
    compressed = ExpressionData(
        path=str(gzipped),
        gene_list=genes,
        data_format="csv",
        csv_separator="\t",
        samples_in="rows",
    ).get_expression_df()

    assert list(compressed.index) == list(plain.index)
    assert list(compressed.columns) == list(plain.columns)
    pd.testing.assert_frame_equal(compressed, plain)

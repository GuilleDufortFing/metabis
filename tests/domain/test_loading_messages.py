"""What a reader is told when a reaction or expression file cannot be loaded.

A broken file used to surface as a pandas or NumPy error wrapped twice, with the
server's paths, the loader's parameter names and a traceback. Each message now
says what is wrong in the file's own terms and, for a bad cell, where it is.
"""

from pathlib import Path

import pytest

from metabis.domain.expression_data import ExpressionData
from metabis.domain.metabolic_model import MetabolicModel
from metabis.domain.preprocessing import ExpressionPreprocessingConfig
from metabis.domain.reaction_data import ReactionData


@pytest.fixture(scope="module")
def model() -> MetabolicModel:
    return MetabolicModel()


def _write(tmp_path: Path, name: str, text: str) -> str:
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def _reaction_message(model, path: str, samples_in: str = "columns") -> str:
    _, results = ReactionData.create_with_validation(
        path, "single", metabolic_model=model, reaction_name_format="mar_id", csv_separator="\t", samples_in=samples_in
    )
    assert results["status"] == "error"
    (message,) = results["messages"]
    return message


def _expression_message(model, path: str, *, log1p: bool = False) -> str:
    _, results = ExpressionData.create_with_validation(
        path,
        gene_list=model.get_genes("ensmbl_id"),
        gene_format="geneSymbols",
        metabolic_model=model,
        data_format="csv",
        csv_separator="\t",
        samples_in="columns",
        preprocessing=ExpressionPreprocessingConfig(log1p=log1p),
    )
    assert results["status"] == "error"
    (message,) = results["messages"]
    return message


def _plain(message: str, tmp_path: Path) -> None:
    for internal in ("Traceback", "Error loading", str(tmp_path), "gene_fmt", "translated to", "__init__"):
        assert internal not in message


def test_a_reaction_cell_that_is_not_a_number_is_named(model, tmp_path):
    text = "reaction\tS1\tS2\nMAR03905\t1\tabc\nMAR04358\t1\t2\n"
    message = _reaction_message(model, _write(tmp_path, "r.tsv", text))

    assert message == "The reaction file holds 'abc' for reaction MAR03905 in sample S2, which is not a number."
    _plain(message, tmp_path)


def test_a_decimal_comma_is_pointed_out(model, tmp_path):
    message = _reaction_message(model, _write(tmp_path, "r.tsv", "reaction\tS1\tS2\nMAR03905\t1\t1,5\n"))

    assert message.endswith("Write decimals with a point, not a comma.")


def test_blank_and_infinite_reaction_cells_are_counted_and_the_first_named(model, tmp_path):
    text = "reaction\tS1\tS2\tS3\nMAR03905\t1\t\t2\nMAR04358\t1\t2\tinf\n"
    message = _reaction_message(model, _write(tmp_path, "r.tsv", text))

    assert message.startswith("The reaction file is incomplete: 2 cells are blank, NaN or infinite.")
    assert "The first is reaction MAR03905 in sample S2." in message


def test_a_column_that_is_no_reaction_may_hold_text(model, tmp_path):
    text = "sample\tMAR03905\tMAR04358\tnote\nS1\t1\t2\tfirst\nS2\t2\t3\tsecond\n"
    _, results = ReactionData.create_with_validation(
        _write(tmp_path, "r.tsv", text), "single", metabolic_model=model, reaction_name_format="mar_id", csv_separator="\t", samples_in="rows"
    )

    assert results["status"] == "success"


def test_a_binary_reaction_file_is_said_not_to_be_text(model, tmp_path):
    path = tmp_path / "r.tsv"
    path.write_bytes(bytes(range(256)) * 8)

    message = _reaction_message(model, str(path))

    assert message.startswith("The reaction file is not text.")


def test_no_matching_reaction_says_how_the_file_was_read(model, tmp_path):
    path = _write(tmp_path, "r.tsv", "reaction\tS1\nFOO\t1\n")
    message = _reaction_message(model, path)

    assert message.startswith(f"No reaction in '{path}' matches a Human1 reaction")
    assert "read as 'mar_id', in single direction mode, separated by tabs, with the samples in columns" in message
    assert "translated to" not in message


def test_blank_expression_cells_name_the_gene_as_the_file_does(model, tmp_path):
    text = "gene\tS1\tS2\tS3\nGCK\t1\t\t2\nHK1\t1\t2\t3\n"
    message = _expression_message(model, _write(tmp_path, "e.tsv", text))

    assert message == (
        "The expression file is incomplete: 1 cell is blank, NaN or infinite. It is gene GCK in sample S2. "
        "Every cell must hold a number."
    )
    _plain(message, tmp_path)


def test_an_expression_cell_that_is_not_a_number_is_named(model, tmp_path):
    message = _expression_message(model, _write(tmp_path, "e.tsv", "gene\tS1\tS2\nGCK\t1\tabc\nHK1\t1\t2\n"))

    assert message == "The expression file holds 'abc' for gene GCK in sample S2, which is not a number."


def test_values_log1p_cannot_take_are_blamed_on_the_normalization(model, tmp_path):
    message = _expression_message(model, _write(tmp_path, "e.tsv", "gene\tS1\tS2\nGCK\t1\t-5\nHK1\t1\t2\n"), log1p=True)

    assert message.startswith("The normalization left 1 cell without a finite value (gene GCK in sample S2)")
    assert "set the normalization to None" in message


def test_a_file_that_is_no_h5ad_says_so(model, tmp_path):
    _, results = ExpressionData.create_with_validation(
        _write(tmp_path, "e.h5ad", "not an h5ad"),
        gene_list=model.get_genes("ensmbl_id"),
        gene_format="geneSymbols",
        metabolic_model=model,
        data_format="h5ad",
    )

    (message,) = results["messages"]
    assert message.startswith("The expression file is not a readable h5ad file")
    _plain(message, tmp_path)


def test_a_reaction_named_twice_keeps_its_first_column_and_says_so(model, tmp_path):
    text = "sample\tMAR03905\tMAR04358\tMAR03905\nS1\t1\t2\t9\nS2\t2\t3\t9\n"
    instance, results = ReactionData.create_with_validation(
        _write(tmp_path, "r.tsv", text), "single", metabolic_model=model, reaction_name_format="mar_id", csv_separator="\t", samples_in="rows"
    )

    assert results["status"] == "success"
    assert results["warnings"] == ["1 reaction appears more than once in the file (MAR03905): the first of each was kept."]
    # The repeat is neither kept nor counted as an identifier Human1 lacks.
    assert results["alignment_report"]["unmapped_count"] == 0
    # The first column's 1 and 2, signed-cube-rooted as single-direction data is by default; not the repeat's 9s.
    assert instance.get_reaction_df()["MAR03905"].tolist() == pytest.approx([1.0, 2 ** (1 / 3)])


def test_reaction_data_of_zeros_loads_with_a_warning(model, tmp_path):
    text = "sample\tMAR03905\tMAR04358\nS1\t0\t0\nS2\t0\t0\n"
    _, results = ReactionData.create_with_validation(
        _write(tmp_path, "r.tsv", text), "single", metabolic_model=model, reaction_name_format="mar_id", csv_separator="\t", samples_in="rows"
    )

    assert "Every value of the reactions read is zero" in results["warnings"][0]


def test_a_gene_named_twice_keeps_its_first_row_and_says_so(model, tmp_path):
    text = "gene\tS1\tS2\nGCK\t1\t2\nHK1\t1\t2\nGCK\t5\t5\n"
    _, results = ExpressionData.create_with_validation(
        _write(tmp_path, "e.tsv", text),
        gene_list=model.get_genes("ensmbl_id"),
        gene_format="geneSymbols",
        metabolic_model=model,
        data_format="csv",
        csv_separator="\t",
        samples_in="columns",
    )

    assert results["status"] == "success"
    assert results["warnings"] == ["1 gene appears more than once in the file (GCK): the first of each was kept."]

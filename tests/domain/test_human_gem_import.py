"""The packaged Human1 reference model's import from a Human-GEM release."""

import csv
import importlib.util
import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "import_human_gem.py"
REFERENCE = Path(__file__).resolve().parents[2] / "src" / "metabis" / "resources" / "reference_model"


@pytest.fixture(scope="module")
def importer():
    spec = importlib.util.spec_from_file_location("import_human_gem", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _table(relpath: str, delimiter: str = "\t") -> list:
    with open(REFERENCE / relpath, newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


# Human-GEM 2's YAML lists a reaction's metabolites by identifier; the model's
# metabolite order decides their order on each side.
ORDER = {"MAM01796c": 0, "MAM02552c": 1, "MAM01249c": 2, "MAM02039c": 3, "MAM02553c": 4}
NAMES = {
    "MAM01796c": "ethanol[c]",
    "MAM02552c": "NAD+[c]",
    "MAM01249c": "acetaldehyde[c]",
    "MAM02039c": "H+[c]",
    "MAM02553c": "NADH[c]",
}
ALCOHOL_DEHYDROGENASE = {
    "id": "MAR03905",
    "metabolites": [
        ("MAM01249c", 1.0),
        ("MAM01796c", -1.0),
        ("MAM02039c", 2.0),
        ("MAM02552c", -1.0),
        ("MAM02553c", 1.0),
    ],
    "lower_bound": 0.0,
    "upper_bound": 1000.0,
}


def test_a_reaction_is_written_in_identifiers_as_cobra_writes_it(importer):
    assert (
        importer.formula(ALCOHOL_DEHYDROGENASE, ORDER)
        == "MAM01796c + MAM02552c  -> MAM01249c + 2 MAM02039c + MAM02553c "
    )
    reversible = dict(ALCOHOL_DEHYDROGENASE, lower_bound=-1000.0)
    assert " <=> " in importer.formula(reversible, ORDER)


def test_a_reaction_is_written_in_names_as_raven_writes_it(importer):
    assert (
        importer.equation(ALCOHOL_DEHYDROGENASE, ORDER, NAMES)
        == "ethanol[c] + NAD+[c] => acetaldehyde[c] + 2 H+[c] + NADH[c]"
    )
    exchange = {"metabolites": [("MAM01796c", -1.0)], "lower_bound": -1000.0}
    assert importer.equation(exchange, ORDER, NAMES) == "ethanol[c] <=> "


def test_a_gene_rule_takes_a_space_inside_each_innermost_parenthesis(importer):
    assert importer.cobra_rule("((A and B) or (C and D)) or E") == "(( A and B ) or ( C and D )) or E"


def test_the_packaged_tables_describe_one_model():
    catalog = _table("reactions/reaction_catalog.csv", delimiter=",")
    formulas = _table("reactions/reaction_formulas.txt")
    crosswalk = _table("reactions/identifier_crosswalk.tsv")
    metabolites = {row["mets"] for row in _table("metabolites/identifier_crosswalk.tsv")}
    genes = {row["genes"] for row in _table("genes/identifier_crosswalk.tsv")}

    # Human-GEM 2.1.1's reactions, in one order across the catalog and formulas.
    assert len(catalog) == 12748
    assert [row["ID"] for row in catalog] == [row["Rxn name"] for row in formulas]
    assert {row["ID"] for row in catalog} == {row["rxns"] for row in crosswalk}
    assert set(re.findall(r"MAM\d+\w", " ".join(row["Formula"] for row in formulas))) <= metabolites
    assert set(re.findall(r"ENSG\d+", " ".join(row["GENE ASSOCIATION"] for row in catalog))) <= genes
    # Each rule is also written in symbols, gene for gene.
    for row in catalog:
        assert len(re.findall(r"ENSG\d+", row["GENE ASSOCIATION"])) == len(
            [token for token in re.split(r"[\s()]+", row["GENE_ASSOCIATION (SYMBOL)"]) if token not in ("", "and", "or")]
        )


def test_a_metabolite_name_keeps_its_own_punctuation():
    # Up to 1.19.0 the packaged names split at a ";" or an "&apos;" into columns.
    names = {row["mets"]: row["metName"] for row in _table("metabolites/identifier_crosswalk.tsv")}
    assert all("\t" not in name and "&apos" not in name for name in names.values())
    assert names["MAM01902c"] == "1'-OH-midazolam-glucuronide"

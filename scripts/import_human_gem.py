"""Build the packaged Human1 reference model from a Human-GEM release.

The reference model under ``resources/reference_model`` is Human-GEM
(github.com/SysBioChalmers/Human-GEM) at the release :data:`HUMAN_GEM_VERSION`,
read from the files of its ``model`` folder: ``Human-GEM.yml`` and the three
annotation tables beside it. This script writes, under ``resources``:

``reference_model/reactions/reaction_catalog.csv``
    A row per reaction, laid out as RAVEN's spreadsheet export lays its
    reactions sheet out: the equation in metabolite names, each gene rule also
    in the gene symbols of ``genes.tsv``, and the MIRIAM cross-references of
    ``reactions.tsv``. The rows keep the packaged catalog's order, and the
    reactions a release adds follow, in its order: the catalog's order is the
    model's, in which an identifier several reactions list resolves to the last
    (``ReactionData``), and Human-GEM reorders its reactions between releases.
``reference_model/reactions/reaction_formulas.txt``
    Each reaction in metabolite identifiers, as COBRA's text export writes it,
    in the catalog's order.
    Human-GEM's own ``Human-GEM.txt`` wrote them so up to 1.19.0; since 2.0 it
    writes metabolite names, so the file is written here.
``reference_model/reactions/identifier_crosswalk.tsv``
    ``reactions.tsv``, with the identifiers Metabis adds (:data:`ADDED`).
``reference_model/metabolites/identifier_crosswalk.tsv``
    ``metabolites.tsv`` in identifier order, with each metabolite's name and
    without the structures (SMILES, InChI), which nothing reads.
``reference_model/genes/identifier_crosswalk.tsv``
    ``genes.tsv``.
``model_interoperability/reaction_mappings/bigg_legacy_to_mar.csv``
    The identifiers BiGG once gave each reaction's BiGG reactions: their
    ``old_bigg_ids`` in BiGG's reaction namespace. The table follows the
    release's BiGG annotation, so an identifier it no longer gives a reaction,
    or a reaction it removed, claims nothing. Rows and identifiers keep the
    packaged file's order, which lookups depend on (an identifier several
    reactions claim goes to the last), and new ones follow theirs.

The tables keep the quoting of Human-GEM's tables up to 1.19.0, so that a new
release shows, file against file, only what it changes.

Usage::

    python -I scripts/import_human_gem.py HUMAN_GEM BIGG_REACTIONS [--check]

``HUMAN_GEM`` holds the repository at the release tag, as ``git clone`` and
``git checkout`` leave it, or as its archive unpacks
(``https://codeload.github.com/SysBioChalmers/Human-GEM/tar.gz/<tag>``); only its
``model`` folder is read. ``BIGG_REACTIONS`` is BiGG 1.6.0's reaction namespace,
``http://bigg.ucsd.edu/static/namespace/bigg_models_reactions.txt``, checked
against :data:`BIGG_SHA256`. Both are parsed as data. ``--check`` reports how
each file would change, writing nothing.

Human-GEM is shared under the CC BY 4.0 licence (THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import sys
from pathlib import Path
from typing import Dict, List

import yaml

HUMAN_GEM_VERSION = "2.1.1"
BIGG_SHA256 = "e1c11add0cb2a1ed3403348eacefd96f1fe487e95a2a03a06eae0edd9a20947d"

HERE = Path(__file__).resolve().parent
RESOURCES = HERE.parent / "src/metabis/resources"
CATALOG = "reference_model/reactions/reaction_catalog.csv"
LEGACY_BIGG = "model_interoperability/reaction_mappings/bigg_legacy_to_mar.csv"

CATALOG_COLUMNS = (
    "ID",
    "NAME",
    "EQUATION",
    "EC-NUMBER",
    "GENE ASSOCIATION",
    "GENE_ASSOCIATION (SYMBOL)",
    "LOWER BOUND",
    "UPPER BOUND",
    "OBJECTIVE",
    "COMPARTMENT",
    "MIRIAM",
    "SUBSYSTEM",
    "REPLACEMENT ID",
    "NOTE",
    "REFERENCE",
    "CONFIDENCE SCORE",
)
# The cross-references a reaction's MIRIAM column lists, in this order, from these
# columns of ``reactions.tsv``.
MIRIAM = (
    ("rxnKEGGID", "kegg.reaction"),
    ("rxnBiGGID", "bigg.reaction"),
    ("rxnRecon3DID", "vmhreaction"),
    ("rxnMetaNetXID", "metanetx.reaction"),
    ("rxnSeedID", "seed.reaction"),
    ("rxnREACTOMEID", "reactome"),
    ("rxnTCDBID", "tcdb"),
    ("rxnRheaID", "rhea"),
    ("rxnRheaMasterID", "rhea"),
)
SBO_REACTION = "sbo/SBO:0000176"
# Identifiers Metabis adds to ``reactions.tsv``: Recon3D's acetyl-CoA carboxylase
# and biomass reactions, which its data files name and Human-GEM does not.
ADDED = {
    ("MAR07673", "rxnRecon3DID"): "ACCOAC",
    ("MAR04413", "rxnRecon3DID"): "biomass_reaction",
}
STRUCTURES = ("metSmiles", "metInChI")
# Unquoted numbers in Human-GEM's tables.
NUMERIC = {"spontaneous"}
GENE = re.compile(r"ENSG\d+")


def read_model(human_gem: Path) -> dict:
    with open(human_gem / "model" / "Human-GEM.yml") as handle:
        model = dict(yaml.load(handle, Loader=yaml.CSafeLoader))
    for key in ("metabolites", "reactions", "genes"):
        model[key] = [dict(entry) for entry in model[key]]
    model["metaData"] = dict(model["metaData"])
    return model


def read_table(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_table(rows: List[Dict[str, str]], columns: List[str]) -> str:
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append(
            "\t".join(row[column] if column in NUMERIC else f'"{row[column]}"' for column in columns)
        )
    return "\n".join(lines) + "\n"


def _number(value: float) -> str:
    return f"{value:g}"


def _sides(reaction: dict, order: Dict[str, int]):
    metabolites = sorted(dict(reaction["metabolites"]).items(), key=lambda item: order[item[0]])
    reactants = [(met, -coefficient) for met, coefficient in metabolites if coefficient < 0]
    products = [(met, coefficient) for met, coefficient in metabolites if coefficient > 0]
    return reactants, products


def _reversible(reaction: dict) -> bool:
    return reaction["lower_bound"] < 0


def formula(reaction: dict, order: Dict[str, int]) -> str:
    """The reaction in metabolite identifiers, as COBRA's ``printRxnFormula``."""
    reactants, products = _sides(reaction, order)

    def side(terms):
        return "+ ".join(
            (f"{_number(coefficient)} " if coefficient != 1 else "") + f"{met} "
            for met, coefficient in terms
        )

    arrow = " <=> " if _reversible(reaction) else " -> "
    return side(reactants) + arrow + side(products)


def equation(reaction: dict, order: Dict[str, int], names: Dict[str, str]) -> str:
    """The reaction in metabolite names, as RAVEN's ``constructEquations``."""
    reactants, products = _sides(reaction, order)

    def side(terms):
        return " + ".join(
            (f"{_number(coefficient)} " if coefficient != 1 else "") + names[met]
            for met, coefficient in terms
        )

    arrow = " <=> " if _reversible(reaction) else " => "
    return side(reactants) + arrow + side(products)


def cobra_rule(rule: str) -> str:
    """The rule as COBRA writes it, a space inside each innermost parenthesis."""
    return re.sub(r"(?<!\))\)", " )", re.sub(r"\((?!\()", "( ", rule))


def legacy_bigg(reactions: List[Dict[str, str]], bigg_reactions: Path) -> str:
    """The legacy BiGG table for the release's BiGG annotation."""
    if hashlib.sha256(bigg_reactions.read_bytes()).hexdigest() != BIGG_SHA256:
        raise SystemExit(f"{bigg_reactions} is not the BiGG reaction namespace this script reads")
    csv.field_size_limit(sys.maxsize)
    with open(bigg_reactions, newline="") as handle:
        old_ids = {
            row["bigg_id"]: [old.strip() for old in row["old_bigg_ids"].split(";") if old.strip()]
            for row in csv.DictReader(handle, delimiter="\t")
        }
    with open(RESOURCES / LEGACY_BIGG, newline="") as handle:
        packaged = {row[0]: row[1].split(", ") for row in list(csv.reader(handle))[1:]}

    claims = {}
    for row in reactions:
        given = []
        for bigg_id in filter(None, row["rxnBiGGID"].split(";")):
            given += [old for old in old_ids.get(bigg_id, ()) if old not in given]
        if given:
            before = [old for old in packaged.get(row["rxns"], ()) if old in given]
            claims[row["rxns"]] = before + [old for old in given if old not in before]
    order = [rxn for rxn in packaged if rxn in claims] + [rxn for rxn in claims if rxn not in packaged]
    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(("rxn_id", "old_identifiers"))
    writer.writerows((rxn, ", ".join(claims[rxn])) for rxn in order)
    return text.getvalue()


def build(human_gem: Path, bigg_reactions: Path) -> Dict[str, str]:
    model = read_model(human_gem)
    version = model["metaData"]["version"]
    if version != HUMAN_GEM_VERSION:
        raise SystemExit(f"{human_gem} holds Human-GEM {version}, not {HUMAN_GEM_VERSION}")
    compartments = dict(model["compartments"])
    order = {met["id"]: index for index, met in enumerate(model["metabolites"])}
    names = {met["id"]: f"{met['name']}[{met['compartment']}]" for met in model["metabolites"]}
    assert set(compartments) >= {met["compartment"] for met in model["metabolites"]}

    genes = read_table(human_gem / "model" / "genes.tsv")
    symbols = {row["genes"]: row["geneSymbols"] or row["genes"] for row in genes}
    reactions = read_table(human_gem / "model" / "reactions.tsv")
    reaction_columns = list(reactions[0])
    for (reaction_id, column), value in ADDED.items():
        row = next(row for row in reactions if row["rxns"] == reaction_id)
        assert row[column] in ("", value), (reaction_id, column, row[column])
        row[column] = value
    annotations = {row["rxns"]: row for row in reactions}

    catalog = io.StringIO()
    writer = csv.writer(catalog, lineterminator="\n")
    writer.writerow(CATALOG_COLUMNS)
    formulas = ["Rxn name\tFormula\tGene-reaction association\tLB\tUB\tObjective"]
    with open(RESOURCES / CATALOG, newline="") as handle:
        packaged = {row["ID"]: index for index, row in enumerate(csv.DictReader(handle))}
    ordered = sorted(
        model["reactions"],
        key=lambda reaction: packaged.get(reaction["id"], len(packaged)),
    )
    for reaction in ordered:
        rule = reaction.get("gene_reaction_rule", "") or ""
        objective = float(reaction.get("objective_coefficient", 0) or 0)
        subsystem = reaction.get("subsystem") or [""]
        # Up to 1.19.0 a reaction's EC numbers and notes were fields of their own;
        # since 2.0 the EC numbers are an annotation and the notes ``notes``.
        ec_codes = reaction.get("eccodes") or ";".join(
            dict(reaction.get("annotation") or {}).get("ec-code") or ()
        )
        annotation = annotations.get(reaction["id"], {})
        miriam = [
            f"{prefix}/{value.replace('RHEA:', '')}"
            for column, prefix in MIRIAM
            for value in filter(None, annotation.get(column, "").split(";"))
        ]
        writer.writerow(
            (
                reaction["id"],
                reaction.get("name", "") or "",
                equation(reaction, order, names),
                ec_codes,
                rule,
                GENE.sub(lambda match: symbols.get(match.group(0), match.group(0)), rule),
                "",
                str(float(reaction["upper_bound"])) if reaction["upper_bound"] == 0 else "",
                str(objective) if objective else "",
                "",
                ";".join(miriam + [SBO_REACTION]),
                (subsystem[0] if isinstance(subsystem, list) else subsystem).strip(),
                "",
                reaction.get("rxnNotes") or reaction.get("notes") or "",
                reaction.get("references", "") or "",
                _number(float(reaction.get("confidence_score", 0) or 0)),
            )
        )
        formulas.append(
            "\t".join(
                (
                    reaction["id"],
                    formula(reaction, order),
                    cobra_rule(rule),
                    f"{float(reaction['lower_bound']):6.2f}",
                    f"{float(reaction['upper_bound']):6.2f}",
                    f"{objective:6.2f}",
                )
            )
        )

    metabolites = read_table(human_gem / "model" / "metabolites.tsv")
    met_names = {met["id"]: met["name"] for met in model["metabolites"]}
    for row in metabolites:
        row["metRetiredID"] = row.pop("metRetired", row.get("metRetiredID", ""))
        row["metName"] = met_names[row["mets"]]
    met_columns = [
        column
        for column in list(metabolites[0])
        if column not in STRUCTURES
    ]
    metabolites.sort(key=lambda row: row["mets"])

    return {
        CATALOG: catalog.getvalue(),
        "reference_model/reactions/reaction_formulas.txt": "\n".join(formulas) + "\n",
        "reference_model/reactions/identifier_crosswalk.tsv": write_table(reactions, reaction_columns),
        "reference_model/metabolites/identifier_crosswalk.tsv": write_table(metabolites, met_columns),
        "reference_model/genes/identifier_crosswalk.tsv": write_table(genes, list(genes[0])),
        LEGACY_BIGG: legacy_bigg(reactions, bigg_reactions),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("human_gem", type=Path)
    parser.add_argument("bigg_reactions", type=Path)
    parser.add_argument("--check", action="store_true", help="report, writing nothing")
    args = parser.parse_args(argv)

    for relpath, text in build(args.human_gem, args.bigg_reactions).items():
        path = RESOURCES / relpath
        old = path.read_text() if path.exists() else ""
        old_lines, new_lines = old.splitlines(), text.splitlines()
        changed = len(set(old_lines) ^ set(new_lines))
        print(f"{relpath}: {len(new_lines) - 1} rows, " + ("unchanged" if old == text else f"{changed} lines differ"))
        if not args.check and old != text:
            path.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

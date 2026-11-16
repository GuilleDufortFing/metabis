"""Regenerate the packaged source-model reaction tables and orientation files.

For each source model, ``resources/model_interoperability/model_alignments/``
holds two files written here:

``reactions_<MODEL>.tsv``
    Every reaction of the model, under the ID its data files carry, with the
    Human1 reaction it maps to and how the two are written: ``1`` alike, ``-1``
    reversed, ``0`` not read. Loading a file declared to come from that model
    maps the IDs it lists through it first, and reverses, in the session's copy
    of Human1, each reaction whose kept source is ``-1``, so that a source's
    forward score is drawn and tested as the forward direction.
``alignment_<MODEL>.txt``
    One value per Human1 reaction: that of the source reaction loading keeps when
    a file holds the whole model. ``MetabolicModel.apply_alignment_corrections``
    reads it.

The orientation files had no committed producer, and they did not agree with the
models they name: against Compass's own Recon2, the RECON2 file called 723 of the
881 reversed reactions aligned. Both files are now read from the two models'
stoichiometry, so they can be reproduced and audited.

How a row is read:

1. Pairing. A source reaction pairs with the Human1 reaction the loader's
   identifier lookup gives it (``ReactionData._map_base_reactions_to_model``)
   when that reaction holds the same metabolites in the same compartments. When
   instead exactly one other Human1 reaction does, the chemistry wins: that is
   how a Recon1 ID that Recon2 reuses for another reaction, or an identifier
   Human1 files under the wrong exchange, still lands on its reaction, and how a
   reaction no identifier names finds one. Failing both, the identifier pairing
   stands.
2. Orientation. The two reactions' metabolites are compared side by side, in
   their compartments. Protons and water are left out: models balance them on
   either side. A reaction is aligned when every shared metabolite is on the
   same side, reversed when every one is on the other side. A transported
   metabolite counts only between the same two compartments. Failing a match in
   the same compartment, a reaction that is not a transport is compared by
   metabolite alone; a transport its two models name differently, by its
   compartments; a reaction with a single metabolite in each model - an
   exchange, demand or sink - by that metabolite's side.
3. What that leaves open is read by hand (``READ_BY_HAND``) or written ``0``,
   and reported.

Usage::

    python scripts/regenerate_model_alignments.py --check
    python scripts/regenerate_model_alignments.py --write

``--check`` reports how each regenerated file compares with the packaged one and
changes nothing; ``--write`` replaces the packaged files. The source models are
downloaded once into ``--cache`` and verified against the SHA-256 recorded here.
No public copy of HMR 1.0 was found, so it has no table and
``alignment_HMR1.0.txt`` is not regenerated; it holds no reversal. ``--write``
keeps its values for the Human1 reactions the packaged model still has and
gives the reactions it adds 0: HMR 1.0 has no counterpart to them that this
script could read.

Each file lists the packaged model's reactions, in its order, so after a new
Human1 release (``scripts/import_human_gem.py``) the reactions it removed leave
every file and those it adds come in, which ``--check`` reports.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import scipy.io

from metabis.domain.metabolic_model.model import MetabolicModel
from metabis.domain.reaction_data import ReactionData

ROOT = Path(__file__).resolve().parents[1]
ALIGNMENTS = ROOT / "src" / "metabis" / "resources" / "model_interoperability" / "model_alignments"

COMPASS = (
    "https://raw.githubusercontent.com/wagnerlab-berkeley/Compass/"
    "29d114372d8efffc0886e15a12fbc3d853f12307/compass/Resources/Metabolic%20Models/"
)
HUMAN_GEM = "https://github.com/SysBioChalmers/Human-GEM/raw/v1.0.0/ComplementaryData/"


@dataclass(frozen=True)
class Source:
    """A source model: where it is published, how it is read, and its ID format."""

    files: dict[str, tuple[str, str]]  # local name -> (URL, SHA-256)
    reader: str
    reaction_format: str
    origin: str
    # Human1's compartment letters as the source writes them, where they differ.
    compartments: tuple[tuple[str, str], ...] = ()


SOURCES = {
    "RECON1": Source(
        {
            "rxns": (COMPASS + "RECON1_mat/model/model.rxns.json", "edfc37ec975d0f064315b49ce5814a7a59d07e570592e18c7d0d8834a0719b09"),
            "mets": (COMPASS + "RECON1_mat/model/model.mets.json", "61ac2fb3d981b4673543a969533baf2daf31d011d540cdedac56a35822eada78"),
            "S": (COMPASS + "RECON1_mat/model/model.S.json", "692c5d70e29e620b2c4eeb10e9e491e34bf3ae4dd58b7ae632b61af265cfa257"),
        },
        "compass",
        "rxnRecon3DID",
        "Compass RECON1_mat, the Recon1 Compass scores",
    ),
    "RECON2": Source(
        {
            "rxns": (COMPASS + "RECON2_mat/model/model.rxns.json", "3ae23ca869c6d0a811bcb821c525a6c05acc5f0cf399a98cd4ba259045122c48"),
            "mets": (COMPASS + "RECON2_mat/model/model.mets.json", "9b5952f0ddadabe22cd5212fbf424af58931a6bb45e72f5d330dbefb653c1b92"),
            "S": (COMPASS + "RECON2_mat/model/model.S.json", "1de9ac584bc2f2946e5947410b0636a5fd73d51c8c77951e25e49d06ab981150"),
        },
        "compass",
        "rxnRecon3DID",
        "Compass RECON2_mat, the Recon2 Compass scores",
    ),
    "RECON2.2": Source(
        {
            "sbml": (COMPASS + "RECON2.2/MODEL1603150001.xml.gz", "c2241fac15e519ab39a68e9aa2abb5fa375f93d934f079a8ed94b2d9d6054693"),
        },
        "sbml",
        "rxnRecon3DID",
        "Recon 2.2 (BioModels MODEL1603150001), as Compass ships it",
    ),
    "RECON3D": Source(
        {
            "mat": (HUMAN_GEM + "Recon3D/Recon3DModel_301.mat", "69319d0eadd1c3eb96699b831aea9fbc0f1b3a96d8a0745f1f9d53c15bdca9f1"),
        },
        "cobra",
        "rxnRecon3DID",
        "VMH Recon3D 3.01, as Human-GEM v1.0.0 integrated it",
    ),
    "HMR2.0": Source(
        {
            "mat": (HUMAN_GEM + "HMR2/HMRdatabase2_00.mat", "0ed9c68a50a7766b0eb0e099d3515949e66a6d479d644b9b95204044630c09ec"),
        },
        "raven",
        "rxnHMR2ID",
        "HMR 2.0 database, as Human-GEM v1.0.0 integrated it",
        # HMR names the extracellular space s and the peroxisome p.
        (("e", "s"), ("x", "p")),
    ),
}

# Rows the stoichiometry leaves open, read from both equations by hand: each source
# reaction keeps the Human1 reaction its identifier names, in the orientation given.
# They share their main metabolites but not their cofactors or names: the models
# give r0400's cofactors opposite sides; write r1450's acyl-CoAs as HC01409
# (hexanoyl-CoA) and HC01410 ((2E)-hexenoyl-CoA); lump OIVD2m's
# 2-oxoisovalerate dehydrogenase where Human1 has its first step, which also runs
# forward; hydrolyse SCP21x's CoA in Human1 only, as phytanoyl-CoA enters the
# peroxisome; move HMR_1630's THCA-CoA and HMR_1666's cholate through the
# mitochondrion where Human1 has the peroxisome; and reduce r1135's, HMR_1500's and
# HMR_1748's sterols with one cofactor and Human1's with another. Human-GEM 2
# corrected four reactions the sources still write as Human1 1.x did: the
# 26-hydroxylation of P45027A12m's and HMR_1610's cholestane tetrol is an oxidation
# by NADP+ in the sources and CYP27A1's monooxygenation in Human1; SFCYSc makes
# S-sulfocysteine from cysteine, Human1 from cystine; r0927's acetone leaves the
# mitochondrion for the extracellular space, Human1's for the cytosol; and
# HMR_6401 octanoylates apoA1 in the cytosol, Human1 the glycine cleavage system's
# H protein in the mitochondrion. biomass_reaction makes biomass from its parts,
# as Human1's does.
_ALIGNED = {
    "OIVD2m": "MAR06416",
    "SCP21x": "MAR03481",
    "r0400": "MAR04531",
    "r1450": "MAR03156",
    "HMR_1500": "MAR01500",
    "HMR_1610": "MAR01610",
    "HMR_1630": "MAR01629",
    "HMR_1666": "MAR01665",
    "HMR_1748": "MAR01748",
    "HMR_3748": "MAR06416",
    "HMR_3481": "MAR03481",
    "HMR_6401": "MAR06402",
    "P45027A12m": "MAR01610",
    "SFCYSc": "MAR08993",
    "biomass_reaction": "MAR04413",
    "r0927": "MAR02222",
    "r1135": "MAR02301",
}


def _aligned(*sources: str, sbml: bool = False) -> dict[str, tuple[str, int]]:
    return {("R_" if sbml else "") + s: (_ALIGNED[s], 1) for s in sources}


READ_BY_HAND: dict[str, dict[str, tuple[str, int]]] = {
    "RECON1": _aligned("OIVD2m", "SCP21x", "P45027A12m"),
    "RECON2": _aligned("OIVD2m", "SCP21x", "r0400", "r1450", "P45027A12m", "SFCYSc", "r0927", "r1135"),
    "RECON2.2": _aligned(
        "OIVD2m", "SCP21x", "r0400", "biomass_reaction", "P45027A12m", "SFCYSc", "r0927", sbml=True
    ),
    "RECON3D": _aligned(
        "OIVD2m", "SCP21x", "HMR_1500", "HMR_1630", "HMR_1666", "HMR_1748", "P45027A12m", "SFCYSc", "r0927",
        "r1135",
    ),
    "HMR2.0": _aligned("HMR_1500", "HMR_1610", "HMR_1630", "HMR_1666", "HMR_3481", "HMR_3748", "HMR_6401"),
}

# Source reactions whose identifier names a Human1 reaction that shares none of
# their chemistry, and no Human1 reaction has it: the table pairs them with none,
# so loading leaves them unmapped instead of putting their scores on another
# reaction. r1446 (butyryl-CoA dehydrogenation) is filed under dodecenoyl-CoA
# hydration. Recon1 and Recon 2.2 also use the dolichol pathway's IDs for other
# steps than Recon2 and Recon3D, whose meaning Human1 follows.
_DOLICHOL = (
    "DOLPGT3_Ler", "DOLPGT3_Uer", "DOLPH_Uer", "DOLPMT3_Uer", "DOLPMT4_Uer", "DOLPMT_U", "DOLP_Uter",
)
NO_COUNTERPART: dict[str, frozenset[str]] = {
    "RECON1": frozenset(("DOLPGT1_Ler", "DOLPMT4_Ler", *_DOLICHOL[1:])),
    "RECON2": frozenset(("r1446",)),
    "RECON2.2": frozenset("R_" + s for s in (*_DOLICHOL, "r1446")),
    "RECON3D": frozenset(("r1446",)),
}

# SBML and BiGG identifiers escape what an ID may not hold.
ESCAPES = {
    "_LPAREN_": "(", "_RPAREN_": ")", "_LSQBKT_": "[", "_RSQBKT_": "]", "_DASH_": "-",
    "_COMMA_": ",", "_SLASH_": "/", "_PERIOD_": ".", "_COLON_": ":", "_APOS_": "'",
}


def unescape(identifier: str) -> str:
    for escaped, character in ESCAPES.items():
        identifier = identifier.replace(escaped, character)
    return identifier


def key(name: str) -> str:
    """A metabolite name compared across models: BiGG's glc__D is Recon's glc_D."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


# -- Reading the source models --------------------------------------------------------
# Each reader returns {reaction ID as a data file carries it: {(metabolite, compartment): side}},
# side -1 for a substrate and 1 for a product, without boundary species.


def _bracketed(metabolite: str) -> tuple[str, str]:
    # Compass's Recon1 escapes its names as SBML does: ala_DASH_L[c].
    match = re.match(r"^(.*)\[(\w+)\]$", unescape(metabolite))
    return key(match.group(1)), match.group(2)


def read_compass(paths: dict[str, Path]) -> dict[str, dict]:
    reactions = json.loads(paths["rxns"].read_text())
    metabolites = json.loads(paths["mets"].read_text())
    sides: dict[str, dict] = {reaction: {} for reaction in reactions}
    for metabolite, reaction, coefficient in json.loads(paths["S"].read_text()):
        if coefficient:
            species = _bracketed(metabolites[int(metabolite) - 1])
            sides[reactions[int(reaction) - 1]][species] = 1 if coefficient > 0 else -1
    return sides


def read_cobra(paths: dict[str, Path]) -> dict[str, dict]:
    data = scipy.io.loadmat(paths["mat"], squeeze_me=True, struct_as_record=False)
    model = next(value for name, value in data.items() if not name.startswith("__"))
    stoichiometry = model.S.tocsc()
    sides = {}
    for column, reaction in enumerate(model.rxns):
        part = stoichiometry[:, [column]].tocoo()
        sides[str(reaction)] = {
            _bracketed(str(model.mets[row])): 1 if coefficient > 0 else -1
            for row, coefficient in zip(part.row, part.data)
            if coefficient
        }
    return sides


def read_raven(paths: dict[str, Path]) -> dict[str, dict]:
    data = scipy.io.loadmat(paths["mat"], squeeze_me=True, struct_as_record=False)
    model = next(value for name, value in data.items() if not name.startswith("__"))
    boundary = {
        str(compartment)
        for compartment, name in zip(model.comps, model.compNames)
        if "boundary" in str(name).lower()
    }
    stoichiometry = model.S.tocsc()
    sides = {}
    for column, reaction in enumerate(model.rxns):
        part = stoichiometry[:, [column]].tocoo()
        side = {}
        for row, coefficient in zip(part.row, part.data):
            metabolite = str(model.mets[row])  # m01690c: HMR's number and compartment
            if coefficient and metabolite[-1] not in boundary:
                side[(key(metabolite[:-1]), metabolite[-1])] = 1 if coefficient > 0 else -1
        sides[str(reaction)] = side
    return sides


def read_sbml(paths: dict[str, Path]) -> dict[str, dict]:
    with gzip.open(paths["sbml"], "rb") as handle:
        root = ET.parse(handle).getroot()
    namespace = root.tag[: root.tag.index("}") + 1]
    boundary = {
        compartment.get("id")
        for compartment in root.iter(namespace + "compartment")
        if "boundary" in (compartment.get("name") or "").lower()
    }
    species = {}
    for item in root.iter(namespace + "species"):
        compartment = item.get("compartment")
        if compartment in boundary:
            continue
        name = unescape(item.get("id"))
        name = name.removeprefix("M_").removesuffix(f"_{compartment}")
        species[item.get("id")] = (key(name), compartment)
    sides = {}
    for reaction in root.iter(namespace + "reaction"):
        side = {}
        for part, sign in (("listOfReactants", -1), ("listOfProducts", 1)):
            for listed in reaction.iter(namespace + part):
                for reference in listed.iter(namespace + "speciesReference"):
                    if reference.get("species") in species and float(reference.get("stoichiometry", "1")):
                        side[species[reference.get("species")]] = sign
        # Keyed as Compass writes it, R_EX_glc_LPAREN_e_RPAREN_; the table also
        # lists the conventional EX_glc(e) (see ``conventional``).
        sides[reaction.get("id")] = side
    return sides


def conventional(identifier: str) -> str:
    """An SBML reaction ID as the model's own literature writes it."""
    return unescape(identifier).removeprefix("R_")


READERS = {"compass": read_compass, "cobra": read_cobra, "raven": read_raven, "sbml": read_sbml}


# -- Human1's side ---------------------------------------------------------------------


class Human1:
    """Human1's reactions as sides of keyed metabolites, in the same terms as a source's."""

    def __init__(self, model: MetabolicModel) -> None:
        self.model = model
        self._keys: dict[str, set] = {}
        balancing = [
            metabolite.id
            for metabolite in model.metabolites.get_all()
            if metabolite.name in ("H+", "H2O")
        ]
        self.ignored = {name for metabolite in balancing for name, _ in self.keys(metabolite)}

    def keys(self, metabolite_id: str) -> set:
        """A metabolite under every source's name: Recon3D's, BiGG's and HMR's."""
        if metabolite_id not in self._keys:
            metabolite = self.model.metabolites.get(metabolite_id)
            mappings = metabolite.id_mappings if metabolite is not None else {}
            compartment = metabolite_id[-1]
            found = set()
            for name in ("metRecon3DID", "metBiGGID"):
                for token in str(mappings.get(name) or "").split(";"):
                    if token.strip():
                        found.add((key(token.strip()), compartment))
            for token in str(mappings.get("metRetiredID") or "").split(";"):
                token = token.strip()
                if re.fullmatch(r"m\d{5}[a-z]", token):  # HMR's m01690c, in HMR's letters
                    found.add((key(token[:-1]), token[-1]))
            self._keys[metabolite_id] = found
        return self._keys[metabolite_id]

    def side(self, reaction_id: str) -> dict:
        reaction = self.model.reactions.get(reaction_id)
        side = {}
        for metabolites, sign in ((reaction.reactants, -1), (reaction.products, 1)):
            for metabolite in metabolites:
                for found in self.keys(metabolite):
                    side[found] = sign
        return side

    def metabolite_count(self, reaction_id: str) -> int:
        reaction = self.model.reactions.get(reaction_id)
        return len(set(reaction.reactants) | set(reaction.products))

    def compartments(self, reaction_id: str, letters: dict[str, str]) -> tuple[set, set]:
        """The compartments of the reaction's substrates and of its products, in the source's letters."""
        reaction = self.model.reactions.get(reaction_id)
        return (
            {letters.get(m[-1], m[-1]) for m in reaction.reactants},
            {letters.get(m[-1], m[-1]) for m in reaction.products},
        )

    def where(self, reaction_id: str, letters: dict[str, str]) -> dict[str, set]:
        """The compartments each metabolite name of the reaction is in, by species.

        A species goes by several names, filed in different letters (HMR's s is
        Human1's e), so where a name sits is read from the species, in one system.
        """
        reaction = self.model.reactions.get(reaction_id)
        found = collections.defaultdict(set)
        for metabolite in (*reaction.reactants, *reaction.products):
            for name, _ in self.keys(metabolite):
                found[name].add(letters.get(metabolite[-1], metabolite[-1]))
        return found


def _sides_agree(source: dict, human1: dict) -> int | None:
    shared = set(source) & set(human1)
    agree = sum(1 for k in shared if source[k] == human1[k])
    if shared and agree == len(shared):
        return 1
    if shared and agree == 0:
        return -1
    return None


def orientation(
    source: dict,
    human1: dict,
    ignored: set,
    *,
    human1_count: int,
    human1_compartments: tuple[set, set],
    human1_where: dict[str, set],
) -> tuple[int | None, str]:
    """How the two reactions compare, and what showed it.

    The first value is 1 when they are written alike, -1 when reversed, None when
    the stoichiometry leaves it open. ``source`` holds one entry per species,
    ``human1`` one per name a species goes by, so ``human1_count`` says how many
    species Human1's reaction has and ``human1_where`` where each name's species are.
    """
    if len(source) == 1 and human1_count == 1:
        # An exchange, demand or sink: one metabolite, on one side, in each model.
        # Its names may not match, and it may be water or a proton.
        sign = 1 if next(iter(source.values())) == next(iter(human1.values())) else -1
        return sign, "one metabolite"
    balanced = {k: v for k, v in source.items() if k[0] not in ignored}
    balanced_human1 = {k: v for k, v in human1.items() if k[0] not in ignored}
    # A metabolite the reaction moves is evidence only between the same two
    # compartments: e -> c and c -> m share c and nothing else. One moved between
    # others, or named differently in its two compartments, is left out.
    where = collections.defaultdict(set)
    for name, compartment in balanced:
        where[name].add(compartment)
    unmatched_moves = {
        name
        for name in where
        if name in human1_where
        and (len(where[name]) > 1 or len(human1_where[name]) > 1)
        and where[name] != human1_where[name]
    }
    comparable = {k: v for k, v in balanced.items() if k[0] not in unmatched_moves}
    comparable_human1 = {k: v for k, v in balanced_human1.items() if k[0] not in unmatched_moves}
    if set(comparable) & set(comparable_human1):
        return _sides_agree(comparable, comparable_human1), "metabolites"
    balanced, balanced_human1 = comparable, comparable_human1
    # The same chemistry filed under another compartment: compared by metabolite
    # alone, unless a metabolite sits on both sides, as in a transport, whose
    # orientation is its compartments.
    by_name, by_name_human1 = collections.defaultdict(set), collections.defaultdict(set)
    for (name, _), sign in balanced.items():
        by_name[name].add(sign)
    for (name, _), sign in balanced_human1.items():
        by_name_human1[name].add(sign)
    if all(len(signs) == 1 for signs in (*by_name.values(), *by_name_human1.values())):
        anywhere = {name: next(iter(signs)) for name, signs in by_name.items()}
        anywhere_human1 = {name: next(iter(signs)) for name, signs in by_name_human1.items()}
        if set(anywhere) & set(anywhere_human1):
            return _sides_agree(anywhere, anywhere_human1), "metabolites, any compartment"
    # A transport whose metabolites the two models name differently: every substrate
    # in one compartment and every product in another fixes its direction.
    substrates = {c for (_, c), sign in source.items() if sign < 0}
    products = {c for (_, c), sign in source.items() if sign > 0}
    substrates_human1, products_human1 = human1_compartments
    if (
        len(substrates) == len(products) == len(substrates_human1) == len(products_human1) == 1
        and substrates != products
        and substrates | products == substrates_human1 | products_human1
    ):
        return (1 if substrates == substrates_human1 else -1), "transport compartments"
    # Nothing else in common: the protons and water, as the last evidence.
    balancing = {k: v for k, v in source.items() if k[0] in ignored}
    balancing_human1 = {k: v for k, v in human1.items() if k[0] in ignored}
    if set(balancing) & set(balancing_human1):
        return _sides_agree(balancing, balancing_human1), "protons or water only"
    return None, "nothing in common"


# -- Building a file -------------------------------------------------------------------


def fetch(name: str, source: Source, cache: Path) -> dict[str, Path]:
    paths = {}
    for local, (url, digest) in source.files.items():
        path = cache / name / Path(url.replace("%20", " ")).name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {url}", flush=True)
            with urllib.request.urlopen(url) as response:
                path.write_bytes(response.read())
        found = hashlib.sha256(path.read_bytes()).hexdigest()
        if found != digest:
            raise SystemExit(f"{path} has SHA-256 {found}, not the {digest} this script was checked against")
        paths[local] = path
    return paths


class Chemistry:
    """Human1's reactions as sets of species, to find the one a source reaction is."""

    def __init__(self, human1: Human1) -> None:
        self.human1 = human1
        self.species: dict[tuple, set] = collections.defaultdict(set)
        for metabolite in human1.model.metabolites.get_all():
            for found in human1.keys(metabolite.id):
                self.species[found].add(metabolite.id)
        self.reactions: dict[str, dict] = {}
        self.containing: dict[str, set] = collections.defaultdict(set)
        for reaction in human1.model.reactions.get_all():
            side = {}
            for metabolites, sign in ((reaction.reactants, -1), (reaction.products, 1)):
                for metabolite in metabolites:
                    if not any(name in human1.ignored for name, _ in human1.keys(metabolite)):
                        side[metabolite] = sign
            self.reactions[reaction.id] = side
            for metabolite in side:
                self.containing[metabolite].add(reaction.id)

    def same(self, source: dict) -> dict[str, int]:
        """The Human1 reactions holding exactly the source's metabolites, with their orientation.

        Protons and water aside, every source species must name one Human1
        species, and the Human1 reaction must hold those species and no other, all
        on the source's sides (1) or all on the other sides (-1).
        """
        mapped = {}
        for found, sign in source.items():
            if found[0] in self.human1.ignored:
                continue
            candidates = self.species.get(found, set())
            if len(candidates) != 1:
                return {}
            metabolite = next(iter(candidates))
            if metabolite in mapped:
                return {}
            mapped[metabolite] = sign
        if not mapped:
            return {}
        matches = {}
        for reaction_id in set.intersection(*(self.containing[m] for m in mapped)):
            side = self.reactions[reaction_id]
            if set(side) != set(mapped):
                continue
            agree = sum(side[m] == sign for m, sign in mapped.items())
            if agree == len(mapped):
                matches[reaction_id] = 1
            elif agree == 0:
                matches[reaction_id] = -1
        return matches


def build(
    name: str, source: Source, cache: Path, human1: Human1, chemistry: Chemistry
) -> tuple[dict[str, tuple[str, int, str]], dict[str, int], dict]:
    """The model's table rows, its per-Human1-reaction values, and what to report."""
    sides = READERS[source.reader](fetch(name, source, cache))
    reader = ReactionData.__new__(ReactionData)
    reader.metabolic_model = human1.model
    mapping, _unmapped, _high, collisions = reader._map_base_reactions_to_model(
        sorted(sides), source.reaction_format, "single", check_old_identifiers=True
    )
    # Each source's identifier pairing, whether or not it wins its Human1 reaction.
    claims = dict(mapping)
    for record in collisions:
        for discarded, _ in record["discarded"]:
            claims[discarded] = record["mar_id"]
    letters = dict(source.compartments)
    by_hand = READ_BY_HAND.get(name, {})
    unrelated = NO_COUNTERPART.get(name, frozenset())
    used_by_hand, used_unrelated = set(), set()

    rows: dict[str, tuple[str, int, str]] = {}
    report, bases = collections.defaultdict(list), collections.Counter()
    for source_id in sorted(sides):
        side = sides[source_id]
        claimed = claims.get(source_id)
        same = chemistry.same(side)
        if claimed is not None and claimed in same:
            mar_id, value, basis = claimed, same[claimed], "identifier and chemistry"
        elif len(same) == 1:
            ((mar_id, value),) = same.items()
            basis = "chemistry"
            heading = "paired by chemistry, not identifier" if claimed else "paired by chemistry alone"
            report[heading].append((source_id, claimed, mar_id))
        elif claimed is not None:
            mar_id = claimed
            value, basis = orientation(
                side,
                human1.side(mar_id),
                human1.ignored,
                human1_count=human1.metabolite_count(mar_id),
                human1_compartments=human1.compartments(mar_id, letters),
                human1_where=human1.where(mar_id, letters),
            )
            if value is None and source_id in by_hand:
                read_against, value = by_hand[source_id]
                assert read_against == mar_id, (name, source_id, read_against, mar_id)
                basis = "read by hand"
                used_by_hand.add(source_id)
            elif value is None and source_id in unrelated:
                mar_id, value, basis = "", 0, "unrelated chemistry"
                used_unrelated.add(source_id)
            elif value is None:
                report["left open"].append((source_id, mar_id, basis))
                value = 0
        else:
            continue
        rows[source_id] = (mar_id, value, basis)
        bases[basis] += 1
    report["read from"] = sorted(bases.items(), key=lambda item: -item[1])
    report["read by hand, no longer needed"] = sorted(set(by_hand) - used_by_hand)
    report["unrelated, no longer needed"] = sorted(unrelated - used_unrelated)

    # One value per Human1 reaction, for apply_alignment_corrections: that of the
    # source loading keeps from the whole model through the table.
    table = {source_id: (mar_id, value) for source_id, (mar_id, value, _) in rows.items()}
    kept, *_ = reader._map_base_reactions_to_model(
        sorted(sides),
        source.reaction_format,
        "single",
        check_old_identifiers=True,
        source_reactions=table,
    )
    values = {mar_id: table[source_id][1] for source_id, mar_id in kept.items() if mar_id}

    if source.reader == "sbml":
        # Compass writes Recon 2.2's SBML IDs; other tools write them conventionally.
        for source_id, row in list(rows.items()):
            rows.setdefault(conventional(source_id), row)
    return rows, values, report


# Source models with no public copy, whose file is carried over (see above).
UNSOURCED = ("HMR1.0",)


def packaged(name: str) -> tuple[list[str], dict[str, int]]:
    order, values = [], {}
    for line in (ALIGNMENTS / f"alignment_{name}.txt").read_text().splitlines():
        if ":" in line:
            mar_id, value = (part.strip() for part in line.split(":"))
            order.append(mar_id)
            values[mar_id] = int(value)
    return order, values


def table_text(name: str, source: Source, rows: dict[str, tuple[str, int, str]]) -> str:
    header = [
        f"# The reactions of {name} ({source.origin}), each under the ID its data files",
        "# carry, with the Human1 reaction it maps to and how the two are written: 1 alike,",
        "# -1 reversed, 0 not read. Written by scripts/regenerate_model_alignments.py.",
        "source_id\thuman1_id\torientation\tbasis",
    ]
    lines = [f"{s}\t{m}\t{v}\t{b}" for s, (m, v, b) in sorted(rows.items())]
    return "\n".join(header + lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="compare with the packaged files; change nothing")
    mode.add_argument("--write", action="store_true", help="replace the packaged files")
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "metabis" / "alignment-sources")
    parser.add_argument("--model", action="append", choices=sorted(SOURCES), help="only these source models")
    arguments = parser.parse_args()

    human1 = Human1(MetabolicModel(use_cache=False))
    chemistry = Chemistry(human1)
    order = list(human1.model.get_reactions(format="mar_id"))
    for name in arguments.model or sorted(SOURCES):
        source = SOURCES[name]
        rows, values, report = build(name, source, arguments.cache, human1, chemistry)
        listed, current = packaged(name)
        regenerated = {mar_id: values.get(mar_id, 0) for mar_id in order}
        counts = collections.Counter(regenerated.values())
        changes = collections.Counter(
            (current[m], regenerated[m]) for m in order if m in current and current[m] != regenerated[m]
        )
        table_path = ALIGNMENTS / f"reactions_{name}.tsv"
        text = table_text(name, source, rows)
        print(f"\n{name}: {source.origin}")
        print(f"  table: {len(rows):,} source reactions" + (
            "" if not table_path.exists() else
            (", unchanged" if table_path.read_text() == text else ", changed")
        ))
        print(f"  {counts[1]:,} aligned, {counts[-1]:,} reversed, {counts[0]:,} without a source reaction")
        print("  changes from the packaged file: " + (
            ", ".join(f"{before:+d} -> {after:+d}: {n:,}" for (before, after), n in sorted(changes.items())) or "none"
        ))
        print(_release_change(listed, order))
        for heading, items in sorted(report.items()):
            print(f"  {heading}: {len(items)}")
            for item in items[:8]:
                print(f"    {item}")
        if arguments.write:
            table_path.write_text(text)
            (ALIGNMENTS / f"alignment_{name}.txt").write_text(
                "".join(f"{mar_id} : {regenerated[mar_id]}\n" for mar_id in order)
            )
    for name in UNSOURCED:
        if arguments.model and name not in arguments.model:
            continue
        listed, current = packaged(name)
        print(f"\n{name}: no public copy, carried over")
        print(_release_change(listed, order))
        if arguments.write:
            (ALIGNMENTS / f"alignment_{name}.txt").write_text(
                "".join(f"{mar_id} : {current.get(mar_id, 0)}\n" for mar_id in order)
            )
    sys.stdout.flush()


def _release_change(listed: list[str], order: list[str]) -> str:
    removed, added = set(listed) - set(order), set(order) - set(listed)
    if not removed and not added:
        return "  Human1 reactions: as the packaged file lists them"
    return f"  Human1 reactions: {len(removed):,} the model no longer has, {len(added):,} it adds"


if __name__ == "__main__":
    main()

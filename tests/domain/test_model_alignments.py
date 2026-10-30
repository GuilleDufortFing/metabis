"""The packaged reaction orientations, and the rule that reads them from two models.

Each alignment file says, per Human1 reaction, whether a source model writes the
reaction mapped to it the same way (1), reversed (-1), or has none (0); loading
reverses each -1 reaction so a source's forward score stays forward.
``scripts/regenerate_model_alignments.py`` reads every value from both models'
stoichiometry. The source models are downloads, so these tests pin the reading
rule on small reactions and the files on reactions whose two equations are
quoted here.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from metabis import MetabolicModel

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "regenerate_model_alignments.py"
BALANCING = {"h", "h2o"}


@pytest.fixture(scope="module")
def regen():
    spec = importlib.util.spec_from_file_location("regenerate_model_alignments", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def model() -> MetabolicModel:
    return MetabolicModel()


def alignment(model, name):
    return model._load_alignment_file(model.get_alignment_file_path(name))


def read(regen, source, human1, *, count=None, compartments=(set(), set())):
    # Each species here has one name, so where a name sits is read from its keys.
    where = {}
    for name, compartment in human1:
        where.setdefault(name, set()).add(compartment)
    return regen.orientation(
        source,
        human1,
        BALANCING,
        human1_count=count if count is not None else len(set(human1)),
        human1_compartments=compartments,
        human1_where=where,
    )


# --- The packaged files -------------------------------------------------------------


@pytest.mark.resource
@pytest.mark.parametrize(
    "name", [name for name, file in MetabolicModel.ALIGNMENT_MODEL_FILES.items() if file]
)
def test_each_alignment_file_orients_every_human1_reaction_once(model, name):
    path = model.get_alignment_file_path(name)
    listed = [line.split(":")[0].strip() for line in Path(path).read_text().splitlines() if line]

    assert len(listed) == len(set(listed))
    assert set(listed) == set(model.get_reactions(format="mar_id"))
    assert set(alignment(model, name).values()) <= {-1, 0, 1}


def test_recon2s_glycerol_3_phosphate_dehydrogenase_is_reversed(model):
    # Compass's Recon2, G3PD1: glyc3p[c] + nad[c] -> dhap[c] + h[c] + nadh[c].
    # Human1, MAR00479: DHAP[c] + H+[c] + NADH[c] => NAD+[c] + sn-glycerol-3-phosphate[c].
    # The file once called it aligned, as it did 723 of Recon2's 881 reversals.
    assert alignment(model, "RECON2")["MAR00479"] == -1


def test_a_reaction_both_models_write_alike_is_aligned(model):
    # Compass's Recon2, r1148: akg[c] + icit[x] -> akg[x] + icit[c].
    # Human1, MAR00713: AKG[c] + isocitrate[x] => AKG[x] + isocitrate[c].
    # The file once called it reversed.
    assert alignment(model, "RECON2")["MAR00713"] == 1


def test_an_hmr_exchange_is_reversed_against_human1s(model):
    # HMR 2.0 writes an exchange as uptake from its boundary, HMR_9098:
    # D-alanine[x] -> D-alanine[s]; Human1 writes MAR09098 as D-alanine[e] <=>.
    assert alignment(model, "HMR2.0")["MAR09098"] == -1


# --- The reading rule ---------------------------------------------------------------

G3P = {("glyc3p", "c"): -1, ("nad", "c"): -1, ("dhap", "c"): 1, ("h", "c"): 1, ("nadh", "c"): 1}


def test_metabolites_on_the_same_sides_read_as_aligned(regen):
    assert read(regen, G3P, dict(G3P)) == (1, "metabolites")


def test_metabolites_on_the_other_sides_read_as_reversed(regen):
    human1 = {species: -side for species, side in G3P.items()}

    assert read(regen, G3P, human1) == (-1, "metabolites")


def test_protons_and_water_do_not_decide_an_orientation(regen):
    # Models balance them on either side: here Human1 puts the proton opposite.
    human1 = dict(G3P) | {("h", "c"): -1}

    assert read(regen, G3P, human1) == (1, "metabolites")


def test_shared_metabolites_on_both_sides_leave_it_open(regen):
    human1 = dict(G3P) | {("nad", "c"): 1, ("nadh", "c"): -1}

    assert read(regen, G3P, human1) == (None, "metabolites")


def test_one_metabolite_in_each_model_reads_from_its_side(regen):
    # An exchange: the two models need not name the metabolite alike.
    assert read(regen, {("glcd", "e"): -1}, {("glucose", "e"): -1}) == (1, "one metabolite")
    assert read(regen, {("m01965", "s"): 1}, {("m01965", "s"): -1}) == (-1, "one metabolite")


def test_the_same_chemistry_in_another_compartment_reads_by_metabolite(regen):
    source = {("a", "m"): -1, ("b", "m"): 1}
    human1 = {("a", "c"): 1, ("b", "c"): -1}

    assert read(regen, source, human1) == (-1, "metabolites, any compartment")


def test_a_transport_named_differently_reads_from_its_compartments(regen):
    # Recon2's MAGt: maghs[e] -> maghs[c]; Human1 names the pool differently.
    source = {("maghs", "e"): -1, ("maghs", "c"): 1}
    human1 = {("pool", "c"): -1, ("pool", "e"): 1}

    assert read(regen, source, human1, compartments=({"c"}, {"e"})) == (-1, "transport compartments")


def test_a_transport_is_never_compared_by_metabolite_alone(regen):
    # The metabolite sits on both sides, so only its compartments orient it.
    source = {("glc", "e"): -1, ("glc", "c"): 1}
    human1 = {("glc", "c"): -1, ("glc", "m"): 1}

    assert read(regen, source, human1, compartments=({"c"}, {"m"}))[0] is None

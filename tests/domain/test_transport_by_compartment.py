"""Human1's transport, divided by compartment as Recon3D divides it."""

import pytest

from metabis import MetabolicModel
from metabis.domain.metabolic_model.loaders import (
    TRANSPORT_BETWEEN,
    TRANSPORT_WITHIN,
    transport_part,
)


@pytest.mark.parametrize(
    "equation, part",
    [
        ("glucose[e] <=> glucose[c]", "Transport, extracellular"),
        ("H2O[e] + TAG-VLDL pool[e] => fatty acid-VLDL pool[e]", "Transport, extracellular"),
        ("citrate[c] + malate[m] <=> citrate[m] + malate[c]", "Transport, mitochondrial"),
        # The inner mitochondrion belongs with the mitochondrion.
        ("H+[i] => H+[m]", "Transport, mitochondrial"),
        ("UDP-glucose[c] => UDP-glucose[g]", "Transport, Golgi apparatus"),
        ("cholesterol[l] => cholesterol[c]", "Transport, lysosomal"),
        ("CoA[x] <=> CoA[c]", "Transport, peroxisomal"),
        ("ATP[n] <=> ATP[c]", "Transport, nuclear"),
        ("glucose-6-phosphate[c] <=> glucose-6-phosphate[r]", "Transport, endoplasmic reticular"),
        # Touching the extracellular space wins over an organelle.
        ("taurocholate[x] => taurocholate[e]", "Transport, extracellular"),
        ("acetyl-CoA[m] => acetyl-CoA[x]", TRANSPORT_BETWEEN),
        ("glutamate[c] => glutamate[c]", TRANSPORT_WITHIN),
    ],
)
def test_a_reaction_joins_the_transport_of_its_compartments(equation, part):
    assert transport_part(equation) == part


@pytest.fixture(scope="module")
def model():
    return MetabolicModel(use_cache=True)


def test_human1_transport_divides_into_nine_subsystems_of_its_category(model):
    category = model.get_subsystems_by_category()["Membrane Transport & Exchange"]
    sizes = {
        model.get_subsystem_name(subsystem): len(model.get_reactions_for_subsystem(subsystem))
        for subsystem in category
    }

    assert "transport_reactions" not in model.get_subsystems()
    assert sizes == {
        "Transport, extracellular": 2791,
        "Transport, mitochondrial": 485,
        "Transport, endoplasmic reticular": 250,
        "Transport, lysosomal": 183,
        "Transport, peroxisomal": 179,
        "Transport, Golgi apparatus": 116,
        "Transport, nuclear": 94,
        "Transport, between organelles": 47,
        "Transport, within one compartment": 12,
        "Exchange/demand reactions": 1658,
    }


def test_a_transport_part_links_to_metabolic_atlas_transport_page(model):
    url = model.get_subsystem_reference_data("transport_golgi_apparatus")["external_url"]
    assert url.endswith("/subsystem/transport_reactions")

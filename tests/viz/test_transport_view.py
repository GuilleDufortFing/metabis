"""Transport subsystems drawn by transporter family."""

import pandas as pd
import pytest

from metabis import MetabolicModel
from metabis.viz.transport_view import (
    MOST_ROWS,
    NO_GENE,
    group_delta_score,
    is_transport_subsystem,
    transport_rows,
    transporter_family,
)


@pytest.mark.parametrize(
    "symbol, family",
    [("SLC7A5", "SLC7"), ("SLC25A1", "SLC25"), ("SLCO1B1", "SLCO1"), ("ABCA1", "ABCA"), ("AQP9", "AQP"), ("LPL", None)],
)
def test_a_gene_names_its_hgnc_transporter_group(symbol, family):
    assert transporter_family(symbol) == family


def test_only_transport_and_exchange_are_drawn_by_family():
    assert is_transport_subsystem("transport_extracellular")
    assert is_transport_subsystem("exchange_demand_reactions")
    assert not is_transport_subsystem("glycolysis_gluconeogenesis")


@pytest.fixture(scope="module")
def model():
    return MetabolicModel(use_cache=True)


def test_a_transport_subsystems_rows_are_its_families(model):
    rows = transport_rows(model, "transport_extracellular")
    keys = [row.key for row in rows]

    # At most MOST_ROWS families, then the rest together, then the special rows.
    assert len([key for key in keys if key.startswith(("SLC", "ABC", "AQP"))]) == MOST_ROWS
    assert NO_GENE in keys
    every = {reaction for row in rows for reaction in row.reactions}
    assert every == set(model.get_reactions_for_subsystem("transport_extracellular"))


def test_exchange_rows_are_metabolites(model):
    rows = transport_rows(model, "exchange_demand_reactions")
    assert all("[" not in row.name for row in rows)


def test_a_familys_delta_score_is_computed_as_a_subsystems():
    results = pd.DataFrame({"cohens_d": [1.2, -0.4, 0.3, None]}, index=["r1", "r2", "r3", "r4"])
    options = {"metric": "cohens_d", "transform": "sign", "numerator_filter": "significant", "denominator_filter": "all"}

    # Two significant reactions of opposite sign, over three measured.
    assert group_delta_score(results, ["r1", "r2", "r3", "r4"], {"r1", "r2"}, options) == 0.0
    assert group_delta_score(results, ["r1", "r3"], {"r1"}, options) == pytest.approx(0.5)
    assert group_delta_score(results, ["r9"], {"r1"}, options) is None

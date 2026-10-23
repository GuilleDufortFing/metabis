"""A short, human-readable name for every reaction."""

import pytest

from metabis.domain.metabolic_model.reaction_names import SHORT_NAME_LENGTH, short_reaction_name


@pytest.mark.parametrize(
    "name, equation, short",
    [
        ("NAD+ glycohydrolase", "H2O[e] + NAD+[e] => ADP-ribose[e] + H+[e] + nicotinamide[e]", "NAD+ glycohydrolase"),
        ("", "methylglyoxal[e] <=> ", "Exchange of methylglyoxal"),
        ("", "ADP-ribose[c] <=> ADP-ribose[e]", "ADP-ribose transport (c→e)"),
        ("Transport Reaction", "linoleoyl-CoA[c] <=> linoleoyl-CoA[r]", "linoleoyl-CoA transport (c→r)"),
        ("", "ATP[c] + CoA[c] + linoleate[c] => AMP[c] + linoleoyl-CoA[c] + PPi[c]", "linoleate → linoleoyl-CoA"),
        # Outside the cytosol a conversion names its compartment, as Human1 writes it.
        ("", "ATP[r] + CoA[r] + linoleate[r] => AMP[r] + linoleoyl-CoA[r] + PPi[r]", "linoleate → linoleoyl-CoA [r]"),
        (None, "2 H+[c] + citrate[c] => isocitrate[c]", "citrate → isocitrate"),
    ],
)
def test_a_reaction_is_named_by_its_name_or_its_equation(name, equation, short):
    assert short_reaction_name(name, equation) == short


def test_a_long_name_is_cut():
    name = short_reaction_name("NADP+:nicotinate ADP-ribosyltransferase of the extracellular space", "")
    assert len(name) == SHORT_NAME_LENGTH and name.endswith("…")

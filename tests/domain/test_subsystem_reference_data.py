"""WP9 primitive #3 + A-17: subsystem -> Metabolic Atlas reference links.

Completes the reaction/gene/subsystem external-reference trio. The atlas
subsystem slug is the model's canonical subsystem ``id`` (the display name run
through ``_clean_subsystem_name`` at load time). A-17 asked whether that slug is
deterministically derivable or needs a maintained name-to-slug map; these tests
prove it is deterministic and collision-free across the full Human-GEM model.
"""

import re

import pytest

from metabis import MetabolicModel
from metabis.domain.metabolic_model.loaders import _clean_subsystem_name
from metabis.domain.metabolic_model.model import TRANSPORT_PART_IDS
from metabis.viz.click_behavior import get_external_reference_data

_ATLAS_SUBSYSTEM_BASE = (
    "https://metabolicatlas.org/explore/Human-GEM/gem-browser/subsystem/"
)


@pytest.mark.integration
def test_subsystem_reference_data_resolves_by_id_and_by_name():
    model = MetabolicModel(use_cache=True)

    by_id = model.get_subsystem_reference_data("glycolysis_gluconeogenesis")
    by_name = model.get_subsystem_reference_data("Glycolysis / Gluconeogenesis")

    assert by_id == by_name
    assert by_id["display_label"] == "Glycolysis / Gluconeogenesis"
    assert by_id["external_id"] == "glycolysis_gluconeogenesis"
    assert by_id["external_source"] == "Metabolic Atlas"
    assert (
        by_id["external_url"]
        == _ATLAS_SUBSYSTEM_BASE + "glycolysis_gluconeogenesis"
    )


@pytest.mark.integration
def test_subsystem_reference_data_unknown_returns_no_link():
    model = MetabolicModel(use_cache=True)

    data = model.get_subsystem_reference_data("not_a_real_subsystem")

    assert data["external_url"] == ""
    assert data["external_source"] == ""
    assert data["display_label"] == "not_a_real_subsystem"


@pytest.mark.integration
def test_subsystem_atlas_slug_is_deterministically_derived():
    """A-17: every subsystem slug is the deterministic cleaned name, is
    URL-safe, and is unique across the full Human-GEM model."""
    model = MetabolicModel(use_cache=True)
    subsystems = list(model.subsystems.get_all())
    assert subsystems, "packaged model should load subsystems"

    seen_slugs: dict[str, str] = {}
    for subsystem in subsystems:
        slug = subsystem.id

        # The canonical id is exactly the deterministic cleaned display name.
        assert slug == _clean_subsystem_name(subsystem.name)

        # The slug is URL-safe (matches how the reaction slug is used verbatim).
        assert re.fullmatch(r"[a-z0-9_]+", slug), (subsystem.name, slug)

        # No two distinct display names collapse to the same slug.
        assert slug not in seen_slugs or seen_slugs[slug] == subsystem.name, (
            f"slug collision {slug!r}: "
            f"{seen_slugs.get(slug)!r} vs {subsystem.name!r}"
        )
        seen_slugs[slug] = subsystem.name

        # The reference URL is that slug appended to the gem-browser base; the
        # parts of the divided transport, which Metabolic Atlas does not divide,
        # link to its Transport reactions page.
        reference = model.get_subsystem_reference_data(slug)
        page = "transport_reactions" if slug in TRANSPORT_PART_IDS else slug
        assert reference["external_url"] == _ATLAS_SUBSYSTEM_BASE + page

    # Documented A-17 example maps exactly to the Metabolic Atlas slug.
    assert (
        _clean_subsystem_name("Fatty acid biosynthesis (even-chain)")
        == "fatty_acid_biosynthesis_even_chain"
    )


def test_get_external_reference_data_routes_subsystem_to_model():
    """The shared table/hover seam routes subsystems to the model method."""

    class _StubModel:
        def get_subsystem_reference_data(self, subsystem_id):
            return {
                "raw_id": subsystem_id,
                "display_label": "Glycolysis / Gluconeogenesis",
                "external_url": _ATLAS_SUBSYSTEM_BASE + subsystem_id,
            }

        def get_reaction_reference_data(self, reaction_id):  # pragma: no cover
            raise AssertionError("subsystem must not route to the reaction method")

    class _StubManager:
        metabolic_model = _StubModel()

    result = get_external_reference_data(
        _StubManager(), "glycolysis_gluconeogenesis", "subsystem"
    )

    assert result["element_id"] == "glycolysis_gluconeogenesis"
    assert result["display_label"] == "Glycolysis / Gluconeogenesis"
    assert result["external_url"] == (
        _ATLAS_SUBSYSTEM_BASE + "glycolysis_gluconeogenesis"
    )

"""Which elements of a graph are drawn as having no data."""

from types import SimpleNamespace

from metabis.viz.metabolic_visualization.enrichment.analysis import (
    identify_missing_data_subgraph,
)


def _workspace(scored):
    """A comparison whose data covers the elements in ``scored``, every element in the model."""

    def entry(element_id, **_):
        return {"is_in_model": True, "comparison_stats": {"has_comparison_data": element_id in scored}}

    provider = SimpleNamespace(get_reaction_data=entry, get_gene_data=entry, get_metabolite_data=entry)
    return SimpleNamespace(data_provider=provider)


def test_a_gene_with_its_own_test_keeps_its_colour_beside_a_reaction_without_data():
    # As SAT1 beside MAR08603 in the Th17 data: Compass found the reaction blocked
    # in every sample, so it has no data, but SAT1 has its own significant test.
    glyphs = [
        {"id": "R", "glyph_class": "reaction", "label": "MAR08603"},
        {"id": "g1", "glyph_class": "gene", "label": "SAT1"},
        {"id": "g2", "glyph_class": "gene", "label": "SAT2"},
        {"id": "e", "glyph_class": "enzyme_group", "label": "Enzymes for MAR08603", "reaction_id": "MAR08603", "glyph_ids": ["g1", "g2"]},
        {"id": "e2", "glyph_class": "enzyme_group", "label": "Enzymes for MAR08603", "reaction_id": "MAR08603", "glyph_ids": ["g2"]},
    ]
    connections = [
        {"source": "g1", "target": "R"},
        {"source": "g2", "target": "R"},
        {"source": "e", "target": "R"},
    ]

    missing = identify_missing_data_subgraph(glyphs, connections, _workspace(scored={"SAT1"}))

    # The reaction and the gene without data are drawn as no data; SAT1 keeps its
    # colour, and so does the group it is in.
    assert missing == {"R": "no_experimental_data", "g2": "no_experimental_data", "e2": "no_experimental_data"}

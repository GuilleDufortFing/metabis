"""No data is white with a dashed edge, and says so: grey only means measured and not significant."""

from types import SimpleNamespace

from metabis.viz.metabolic_visualization.enrichment.analysis import (
    identify_missing_data_subgraph,
)
from metabis.viz.metabolic_visualization.enrichment.glyph_enrichment import (
    _unmeasured_hover,
)
from metabis.viz.metabolic_visualization.generation.data_structures.core_types import (
    GlyphClass,
)


class _Provider:
    """Reactions all measured; one gene tested in the comparison, one not measured."""

    def get_reaction_data(self, reaction_id):
        return {"is_in_model": True, "comparison_stats": {"has_comparison_data": True}}

    def get_gene_data(self, gene_id):
        return {"is_in_model": True, "comparison_stats": {"has_comparison_data": gene_id == "GCK"}}

    def get_metabolite_data(self, metabolite_id):
        return {"is_in_model": True}


def test_a_gene_without_data_is_drawn_as_no_data_even_beside_measured_reactions():
    glyphs = [
        {"id": "r", "label": "MAR04394", "glyph_class": GlyphClass.REACTION},
        {"id": "measured", "label": "GCK", "glyph_class": GlyphClass.GENE},
        {"id": "unmeasured", "label": "HKDC1", "glyph_class": GlyphClass.GENE},
    ]
    data_manager = SimpleNamespace(data_provider=_Provider())

    missing = identify_missing_data_subgraph(glyphs, [], data_manager)

    assert missing == {"unmeasured": "no_experimental_data"}


def test_a_subsystem_with_nothing_measured_says_so_in_its_hover():
    hover = _unmeasured_hover("Pool reactions", "reaction", 12)

    assert "Pool reactions" in hover
    assert "No reaction of it was measured, so it has no score" in hover
    assert "12 reactions in Human1" in hover

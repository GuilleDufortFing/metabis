"""The subsystem hover states direction as Target minus Baseline (D-51).

The hover used to read "Regulation Direction: 63.2% of significant reactions are
Up-regulated". Two things were wrong: the reaction inputs are flux/activity
scores, so regulation is not established, and the number is a net share of
summed effect magnitude rather than a percentage of reactions.
"""

from metabis.managers.hover.models import HoverContext
from metabis.managers.hover.sections.subsystem_info import SubsystemSection
from metabis.managers.workspace_mode import AnalysisWorkspaceMode


def _context(**overrides) -> HoverContext:
    base = dict(
        mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
        hover_type="subsystem",
        element_type="subsystem",
        subsystem_metric="cohens_d",
    )
    base.update(overrides)
    return HoverContext(**base)


def _data(net_direction_score: float) -> dict:
    return {
        "subsystem_display_name": "Glycolysis / Gluconeogenesis",
        "n_significant": 6,
        "n_total": 10,
        "net_direction_score": net_direction_score,
        "total_metric_value": 1.5,
    }


def test_a_positive_net_direction_reads_as_higher_in_target():
    html = SubsystemSection().build(_data(0.632), _context())

    assert "<strong>Net direction:</strong>" in html
    assert "significant reactions are 63.2% net higher in Target" in html
    assert "regulated" not in html


def test_a_negative_net_direction_names_both_readings():
    html = SubsystemSection().build(_data(-0.25), _context())

    assert "25.0% net lower in Target (higher in Baseline)" in html
    assert "regulated" not in html


def test_a_balanced_subsystem_claims_no_direction():
    html = SubsystemSection().build(_data(0.0), _context())

    assert "balanced between both directions" in html
    assert "0.0% net" not in html


def test_gene_pathways_name_genes_rather_than_reactions():
    html = SubsystemSection().build(_data(0.4), _context(for_genes=True))

    assert "significant genes are 40.0% net higher in Target" in html

"""Every hover has one look: the subsystem graph's reaction and gene panel."""

from types import SimpleNamespace

import pytest

from metabis.managers.hover import panel as hover_panel
from metabis.managers.hover_manager import HoverManager
from metabis.managers.workspace_mode import AnalysisWorkspaceMode


def test_a_panels_content_can_be_set_in_another_and_added_to():
    content = hover_panel.row("ID", "R1")
    assert hover_panel.inner(hover_panel.panel(content)) == content
    # Other HTML is left as it is.
    assert hover_panel.inner("<div>R1</div>") == "<div>R1</div>"
    added = hover_panel.section(hover_panel.row("Loading on PC1", "0.900"))
    assert hover_panel.extend(hover_panel.panel(content), added) == hover_panel.panel(content, added)
    assert hover_panel.extend("", added) == hover_panel.panel(added)


@pytest.mark.parametrize("mode", list(AnalysisWorkspaceMode))
def test_every_mode_draws_its_panels_alike(mode):
    manager = HoverManager(SimpleNamespace(group_comparison_artifact=None), None, None, mode=mode)
    for builder in (
        manager.gene_hover_builder,
        manager.reaction_hover_builder,
        manager.metabolite_hover_builder,
        manager.subsystem_hover_builder,
    ):
        assert builder.container_style == hover_panel.PANEL_STYLE


def test_a_subsystem_read_through_its_genes_hovers_as_a_subsystem(monkeypatch):
    """The overview's gene overlay asks for ``subsystem_genes``: once a bare gene card."""
    manager = HoverManager(SimpleNamespace(group_comparison_artifact=None), None, None)
    asked = {}

    def subsystem_hover(**kwargs):
        asked.update(kwargs)
        return "subsystem panel"

    monkeypatch.setattr(manager, "get_subsystem_hover", subsystem_hover)
    assert manager.get_element_hover("Glycolysis", "subsystem_genes", show_stats=True) == "subsystem panel"
    assert asked["subsystem_name"] == "Glycolysis" and asked["for_genes"] is True
    assert manager.get_element_hover("Glycolysis", "subsystem") == "subsystem panel"
    assert asked["for_genes"] is False

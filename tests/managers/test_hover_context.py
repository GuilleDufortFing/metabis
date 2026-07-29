import pytest

from metabis.managers.hover.builder import HoverTextBuilder
from metabis.managers.hover.models import HoverContext
from metabis.managers.workspace_mode import AnalysisWorkspaceMode


def _context(*, show_stats: bool = False) -> HoverContext:
    return HoverContext(
        mode=AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS,
        hover_type="reaction",
        element_type="reaction",
        show_stats=show_stats,
    )


def test_hover_context_exposes_only_typed_attribute_access():
    context = _context()

    assert context.mode is AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS
    assert not hasattr(context, "get")
    with pytest.raises(TypeError):
        context["mode"]  # type: ignore[index]


def test_missing_model_banner_reads_the_group_from_hover_data():
    hover = HoverTextBuilder().build(
        {
            "reaction_id": "R1",
            "is_in_model": False,
            "experiment_stats": {"group_id": "treated"},
        },
        _context(),
    )

    assert "No Metadata" in hover
    assert "for group <span style='font-weight:bold;'>treated</span>" in hover


def test_missing_statistics_banner_reads_the_group_from_hover_data():
    hover = HoverTextBuilder().build(
        {
            "reaction_id": "R1",
            "experiment_stats": {
                "has_experiment_data": False,
                "group_id": "treated",
            },
        },
        _context(show_stats=True),
    )

    assert "No Experimental Data" in hover
    assert "for group <span style='font-weight:bold;'>treated</span>" in hover

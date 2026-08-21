from metabis.managers.hover.models import HoverContext
from metabis.managers.hover.sections.distribution_plot import DistributionPlotSection
from metabis.managers.hover.sections.statistics import StatisticsSection
from metabis.managers.workspace_mode import AnalysisWorkspaceMode


def test_group_comparison_violin_places_baseline_left_and_target_right(monkeypatch):
    calls = []

    def record_violin(*args, **kwargs):
        calls.append((args, kwargs))
        return "violin"

    monkeypatch.setattr(
        "metabis.viz.plot_violins.create_static_violin_for_hover",
        record_violin,
    )
    data = {
        "distributions": {
            "has_distributions": True,
            "group1_name": "Target",
            "group1_values": [10.0, 12.0],
            "group2_name": "Baseline",
            "group2_values": [1.0, 3.0],
        },
        "comparison_stats": {"forward": {"is_significant": True}},
    }
    context = HoverContext(
        mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
        hover_type="reaction",
        element_type="reaction",
        show_plot=True,
    )

    html = DistributionPlotSection("reaction").build(data, context)

    assert "violin" in html
    args, kwargs = calls[0]
    assert args[0] == [1.0, 3.0]
    assert args[1] == [10.0, 12.0]
    assert kwargs["group1_name"] == "Baseline"
    assert kwargs["group2_name"] == "Target"


def test_group_comparison_statistics_names_the_baseline_and_target():
    data = {
        "comparison_stats": {
            "has_comparison_data": True,
            "group1": "Th17p",
            "group2": "Th17n",
            "forward": {},
        }
    }
    context = HoverContext(
        mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
        hover_type="reaction",
        element_type="reaction",
        show_stats=True,
    )

    html = StatisticsSection("reaction").build(data, context)

    assert "Baseline: Th17n" in html
    assert "Target: Th17p" in html
    assert "Effect direction: Target &minus; Baseline" in html

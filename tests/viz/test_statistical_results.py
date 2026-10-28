"""The statistical results a caller can draw itself, and server-free scatter taps.

A page that renders its own results table reads the same columns and rows the
Bokeh ``DataTable`` shows. Scatter figures embedded without a Bokeh server read
the hover-download setting from a page flag, as ``add_click_behavior`` allows.
"""

import pandas as pd
import pytest
from bokeh.models import CustomJS, Plot

from metabis import (
    AnalysisWorkspace,
    GroupComparison,
    GroupComparisonSpec,
    create_statistical_results_table,
    plot_statistical_scatterplots,
    statistical_results_columns,
    statistical_results_rows,
)
from metabis.managers.group_manager import GroupDefinition

SAMPLES = ["A1", "A2", "B1", "B2"]


def _workspace(effect_threshold=0.0):
    results = pd.DataFrame(
        {
            "adjusted_pval": [0.2, 0.01, None],
            "pval": [0.1, 0.005, None],
            "neg_log_pval": [1.0, 2.3, None],
            "neg_log_adjusted_pval": [0.7, 2.0, None],
            "cohens_d": [0.2, 1.5, 0.0],
            "log2_fc": [0.1, 2.0, 0.0],
            "mean_difference": [0.2, 3.0, 0.0],
            "reversed": [False, False, False],
            "group_1": ["A", "A", "A"],
            "group_2": ["B", "B", "B"],
        },
        index=["R2", "R1", "R3"],
    )
    comparison = GroupComparison(spec=GroupComparisonSpec.pairwise(("A",), ("B",)))
    comparison._reaction_test_results = results
    comparison.define_significant_elements(
        pval_threshold=1.0,
        cohens_d_threshold=effect_threshold,
        log2_fc_threshold=effect_threshold,
        mean_difference_threshold=effect_threshold,
        for_genes=False,
    )
    workspace = AnalysisWorkspace()
    for group, samples in (("A", ["A1", "A2"]), ("B", ["B1", "B2"])):
        workspace.group_manager.define_group(
            group, samples, definition=GroupDefinition("custom"), all_sample_ids=SAMPLES
        )
    comparison.membership = workspace.group_manager.resolve_comparison_membership(
        comparison.spec, SAMPLES
    )
    snapshot = workspace.capture_analysis_snapshot()
    workspace.install_group_comparison_artifact(
        snapshot.provenance,
        comparison.build_artifact(
            provenance=snapshot.provenance,
            source_fingerprints=workspace.source_fingerprints,
        ),
    )
    workspace.refresh_group_comparison_presentation()
    return workspace


def test_rows_follow_the_bokeh_table_schema_smallest_adjusted_p_first():
    workspace = _workspace()

    columns = statistical_results_columns("reaction")
    rows = statistical_results_rows(workspace)

    # A reaction's short name sits beside its identifier; a gene has its symbol.
    assert [column["field"] for column in columns] == [
        "display_id",
        "name",
        "adjusted_pval",
        "pval",
        "cohens_d",
        "log2_fc",
        "mean_difference",
        "subsystems",
    ]
    assert "name" not in [column["field"] for column in statistical_results_columns("gene")]
    assert columns[0]["kind"] == "link" and columns[2]["kind"] == "numeric"
    # A missing p-value sorts last, as in the Bokeh table.
    assert [row["display_id"] for row in rows] == ["R1", "R2", "R3"]
    assert rows[0]["cohens_d"] == 1.5
    assert {"external_url", "ensembl_id", "name"} <= set(rows[0])
    assert [
        row["display_id"] for row in statistical_results_rows(workspace, max_rows=1)
    ] == ["R1"]

    table = create_statistical_results_table(workspace)
    assert list(table["source"].data["display_id"]) == [
        row["display_id"] for row in rows
    ]
    assert [column.field for column in table["table"].columns] == [
        column["field"] for column in columns
    ]


def test_rows_require_an_installed_comparison():
    try:
        statistical_results_rows(AnalysisWorkspace())
    except ValueError as exc:
        assert "No group comparison" in str(exc)
    else:
        raise AssertionError("rows were built without a comparison")


def test_server_free_scatter_taps_read_the_download_flag():
    layout = plot_statistical_scatterplots(
        _workspace(), return_plot=True, download_flag="metabisDownloadHoverOnClick"
    )

    callbacks = [
        callback
        for plot in layout.references()
        if isinstance(plot, Plot)
        for renderer in plot.renderers
        for callback in renderer.data_source.selected.js_property_callbacks.get(
            "change:indices", []
        )
        if isinstance(callback, CustomJS)
    ]
    # One source feeds all four plots, each drawing it twice (significant or not).
    assert len({id(callback) for callback in callbacks}) == 1
    assert all(
        callback.args["download_flag"] == "metabisDownloadHoverOnClick"
        for callback in callbacks
    )
    assert not any("enabled_toggle" in callback.args for callback in callbacks)


def test_every_element_is_drawn_and_the_significant_ones_last():
    """No sampling: the scatters are the same on every render."""
    layout = plot_statistical_scatterplots(_workspace(), return_plot=True)
    plots = [plot for plot in layout.references() if isinstance(plot, Plot)]
    source = plots[0].renderers[0].data_source

    flags = list(source.data["significant"])
    assert len(flags) == len(_workspace().get_group_comparison_artifact().get_test_results())
    assert flags == sorted(flags)
    assert len(plots) == 4
    assert all(len(plot.below) >= 2 for plot in plots)  # the axis and the legend


@pytest.mark.parametrize(
    ("effect", "label"),
    [("cohens_d", "Cohen's d"), ("log2_fc", "log2 fold change"), ("mean_difference", "Mean difference")],
)
@pytest.mark.parametrize("effect_threshold", [0.0, None])
def test_a_volcano_plot_sets_one_effect_against_the_adjusted_p(effect, label, effect_threshold):
    layout = plot_statistical_scatterplots(
        _workspace(effect_threshold), return_plot=True, volcano=effect
    )

    plots = [plot for plot in layout.references() if isinstance(plot, Plot)]
    assert len(plots) == 1
    assert plots[0].xaxis[0].axis_label == label
    assert plots[0].yaxis[0].axis_label == "-log10 adjusted p"


def test_the_mean_difference_volcano_is_drawn_only_when_asked_for():
    layout = plot_statistical_scatterplots(_workspace(), return_plot=True)

    labels = [
        (plot.xaxis[0].axis_label, plot.yaxis[0].axis_label)
        for plot in layout.references()
        if isinstance(plot, Plot)
    ]
    assert ("Mean difference", "-log10 adjusted p") not in labels


def test_a_volcano_plot_needs_an_effect():
    with pytest.raises(ValueError, match="No volcano plot for pval"):
        plot_statistical_scatterplots(_workspace(), return_plot=True, volcano="pval")

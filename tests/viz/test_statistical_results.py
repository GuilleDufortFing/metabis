"""The statistical results a caller can draw itself, and server-free scatter taps.

A page that renders its own results table reads the same columns and rows the
Bokeh ``DataTable`` shows. Scatter figures embedded without a Bokeh server read
the hover-download setting from a page flag, as ``add_click_behavior`` allows.
"""

import json
import shutil
import subprocess

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

    # The plot itself, not wrapped in the grid that lays several out.
    assert isinstance(layout, Plot)
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


def test_a_volcano_plot_can_set_the_effect_against_the_p_before_correction():
    plot = plot_statistical_scatterplots(_workspace(), return_plot=True, volcano="cohens_d", volcano_y="raw")

    assert plot.yaxis[0].axis_label == "-log10 p"
    with pytest.raises(ValueError, match="against median p-values"):
        plot_statistical_scatterplots(_workspace(), return_plot=True, volcano="cohens_d", volcano_y="median")


def test_a_volcano_plots_axis_ends_name_where_each_sides_higher_features_lie():
    plot = plot_statistical_scatterplots(_workspace(), return_plot=True, volcano="cohens_d", sides=("Th17n", "Th17p"))

    from bokeh.models import Title

    assert plot.xaxis[0].axis_label == "Cohen's d"
    # Each side's name at its own end, on lines of their own under the axis; no
    # legend: the colours and the grey say what it would.
    ends = [(model.text, model.align) for model in plot.below if isinstance(model, Title)]
    assert ends == [("← higher in Th17n", "left"), ("higher in Th17p →", "right")]
    assert plot.below.index(plot.xaxis[0]) < plot.below.index(next(m for m in plot.below if isinstance(m, Title)))
    assert not plot.legend


def test_a_volcano_plot_draws_the_subsystem_in_focus_over_the_rest():
    from metabis.viz.feature_focus import FEATURE_ZOOM
    from metabis.viz.subsystem_link import FOCUS_SOURCE

    plot = plot_statistical_scatterplots(
        _workspace(),
        return_plot=True,
        volcano="cohens_d",
        members={"R1": ["glyc"], "R2": ["glyc", "tca"]},
        categories={"glyc": "Carbohydrate", "tca": "Energy"},
    )

    names = {getattr(tag, "name", None) for tag in plot.tags}
    assert {FOCUS_SOURCE, FEATURE_ZOOM} <= names
    source = next(r.data_source for r in plot.renderers if "focus_subsystems" in r.data_source.data)
    members = dict(zip(source.data["element_id"], source.data["focus_subsystems"]))
    assert members == {"R2": ["glyc", "tca"], "R1": ["glyc"], "R3": []}
    focus = next(tag for tag in plot.tags if getattr(tag, "name", None) == FOCUS_SOURCE)
    callback = focus.js_property_callbacks["change:data"][0]
    assert callback.args["categories"] == {"glyc": "Carbohydrate", "tca": "Energy"}
    assert callback.args["floor"] == 0.0


def test_every_point_of_a_volcano_plot_hovers_as_its_element_does_in_the_graphs():
    workspace = _workspace()
    plot = plot_statistical_scatterplots(workspace, return_plot=True, volcano="cohens_d")

    source = next(r.data_source for r in plot.renderers if "desc" in r.data_source.data)
    for element, desc in zip(source.data["element_id"], source.data["desc"]):
        assert desc == workspace.get_element_hover(element_id=element, element_type="reaction", show_stats=True)


def test_the_names_of_a_volcano_plot_follow_the_subsystem_in_focus():
    from metabis.viz.plot_statistical_distributions import (
        STRONGEST_NAMED,
        VOLCANO_NAMES,
    )
    from metabis.viz.subsystem_link import FOCUS_SOURCE

    plot = plot_statistical_scatterplots(
        _workspace(),
        return_plot=True,
        volcano="cohens_d",
        members={"R1": ["glyc"], "R2": ["tca"]},
        categories={"glyc": "Carbohydrate", "tca": "Energy"},
    )

    names = plot.select_one({"name": VOLCANO_NAMES})
    # With nothing in focus, the most significant, as before.
    assert names.data["text"][0] == "R1"
    focus = next(tag for tag in plot.tags if getattr(tag, "name", None) == FOCUS_SOURCE)
    callback = focus.js_property_callbacks["change:data"][0]
    # The focus chooses the names; one placement, run again as the view moves,
    # writes them, and the figure's own come back when the focus lets go.
    assert callback.args["names"] is names
    assert callback.args["naming"] == {"count": STRONGEST_NAMED, "name": "short_name", "significant": "significant"}
    place = callback.args["place"]
    assert place.args["names"] is names and place.args["unfocused"]["text"] == names.data["text"]
    assert place in plot.x_range.js_property_callbacks["change:end"]
    assert place in plot.y_range.js_property_callbacks["change:end"]
    assert "names.data = labels" in place.code


def _place_names(place, *, size, x_range, y_range, unfocused=None) -> dict:
    """Run a volcano plot's name placement in Node, on a plot of ``size`` (None: not laid out yet)."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js runs the placement")
    script = """
    const [code, size, xRange, yRange, unfocused] = JSON.parse(process.argv[1]);
    globalThis.window = {};
    globalThis.requestAnimationFrame = (frame) => { globalThis.waiting = frame };
    globalThis.document = {createElement: () => ({getContext: () => ({font: '', measureText: (text) => ({width: text.length * 6})})})};
    const plot = {x_range: {start: xRange[0], end: xRange[1]}, y_range: {start: yRange[0], end: yRange[1]}};
    for (const [name, index] of [['inner_width', 0], ['inner_height', 1]]) {
      Object.defineProperty(plot, name, {get() { if (size === null) throw new Error(`Figure.${name} is unset`); return size[index] }});
    }
    const names = {id: 'names', data: null};
    new Function('plot', 'names', 'unfocused', code)(plot, names, unfocused);
    console.log(JSON.stringify({data: names.data, waiting: typeof globalThis.waiting === 'function'}));
    """
    arguments = json.dumps([place.code, size, x_range, y_range, unfocused or place.args["unfocused"]])
    result = subprocess.run([node, "-e", script, arguments], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_volcano_names_wait_for_the_plot_and_never_overlap():
    from metabis.viz.plot_statistical_distributions import VOLCANO_NAMES

    plot = plot_statistical_scatterplots(
        _workspace(), return_plot=True, volcano="cohens_d", members={"R1": ["glyc"]}, categories={"glyc": "Carbohydrate"}
    )
    names = plot.select_one({"name": VOLCANO_NAMES})
    place = plot.x_range.js_property_callbacks["change:end"][0]

    # Before the plot is laid out its size is unset: the figure's own names stay,
    # and the placement waits a frame rather than failing.
    early = _place_names(place, size=None, x_range=[-1, 2], y_range=[0, 3])
    assert early["data"]["text"] == names.data["text"] and early["waiting"]

    # Two names on almost the same point go to either side, or one above the other.
    crowded = {"x": [1.0, 1.01], "y": [2.0, 2.0], "text": ["CD36", "SLC14A1"]}
    placed = _place_names(place, size=[400, 300], x_range=[-1, 2], y_range=[0, 3], unfocused=crowded)["data"]
    assert placed["text"] == ["CD36", "SLC14A1"]
    assert placed["align"][0] != placed["align"][1] or placed["y_offset"][0] != placed["y_offset"][1]


def test_the_four_distributions_share_one_row_that_wraps_when_narrow():
    from bokeh.models import Column, Row

    from metabis.viz.plot_statistical_distributions import plot_statistical_histograms

    layout = plot_statistical_histograms(_workspace(), return_plot=True)

    rows = [child for child in layout.children if isinstance(child, Row)]
    assert len(rows) == 1 and len(rows[0].children) == 4
    assert rows[0].styles["flex-wrap"] == "wrap"
    assert all(isinstance(card, Column) and card.styles["min-width"] == "220px" for card in rows[0].children)

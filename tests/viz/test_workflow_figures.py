import json

import numpy as np
import pandas as pd
import pytest
from bokeh.embed import json_item
from bokeh.models import ColumnDataSource, HoverTool, Legend, LinearColorMapper

from metabis.analysis.agreement import ResultEffects, compare_results
from metabis.domain.expression_matrix import DenseExpressionMatrix
from metabis.viz.plot_landscape import (
    LABEL_CHAR_WIDTH,
    MIN_CELL_WIDTH,
    TILT,
    plot_feature_landscape,
    plot_sample_heatmap,
    sample_heatmap_block,
    sample_heatmap_rows,
)
from metabis.viz.plot_result_scatter import (
    plot_reaction_gene_concordance,
    plot_result_agreement,
)


def test_many_groups_tilt_their_names_over_narrow_columns():
    from metabis.analysis.landscape import FeatureLandscape

    groups = [f"Cell type {number} | Healthy" for number in range(7)]
    landscape = FeatureLandscape(
        effect=pd.DataFrame({group: [0.5] for group in groups}, index=["R1"]),
        adjusted_pval=pd.DataFrame({group: [0.01] for group in groups}, index=["R1"]),
        significant=pd.DataFrame({group: [True] for group in groups}, index=["R1"]),
    )

    plot = plot_feature_landscape(landscape, ["R1"], title="Varying")

    assert list(plot.x_range.factors) == groups
    assert plot.xaxis[0].major_label_orientation == TILT
    assert plot.min_width <= 2 * len("Cell type 0 | Healthy") * LABEL_CHAR_WIDTH + 7 * MIN_CELL_WIDTH + 200


def test_identical_profiles_share_a_heatmap_row():
    scores = pd.DataFrame(
        {"R1": [1.0, 2.0, 3.0], "R2": [1.0, 2.0, 3.0], "R3": [0.0, 5.0, 0.0], "flat": [1.0, 1.0, 1.0]},
        index=["s1", "s2", "s3"],
    )

    assert sample_heatmap_rows(scores, 5) == [["R3"], ["R1", "R2"]]
    assert sample_heatmap_rows(scores, 1) == [["R3"]]


def test_sample_heatmap_stays_light_with_hundreds_of_samples():
    rng = np.random.default_rng(1)
    samples = [f"s{i}" for i in range(722)]
    scores = pd.DataFrame(rng.normal(size=(722, 300)), index=samples, columns=[f"R{i}" for i in range(300)])
    groups = {sample: f"G{i % 21}" for i, sample in enumerate(samples)}

    plot = plot_sample_heatmap(scores, groups, count=40, title="Samples")

    payload = json.dumps(json_item(plot))
    assert len(payload) < 1_500_000
    assert plot.x_range.end == 722


def test_a_narrow_heatmap_names_rows_short_and_a_click_names_the_feature():
    scores = pd.DataFrame(
        {
            "MAR1": [1.0, 2.0, 3.0, 4.0],
            "MAR2": [1.0, 2.0, 3.0, 4.0],
            "MAR3": [0.0, 5.0, 0.0, 1.0],
        },
        index=["s1", "s2", "s3", "s4"],
    )
    labels = {"MAR1": "A transport reaction with a long descriptive name", "MAR3": "Short"}
    groups = {"s1": "A", "s2": "A", "s3": "B", "s4": "B"}

    plot = plot_sample_heatmap(
        scores, groups, count=5, labels=labels, title="Rows", label_limit=20, feature_event="pick"
    )

    # Twenty characters, the last an ellipsis, and the reaction scored alike counted.
    assert set(plot.yaxis[0].major_label_overrides.values()) == {"Short", "A transport reactio\u2026 (+1)", "Group"}
    rows = next(tag for tag in plot.tags if getattr(tag, "name", None) == "feature_rows")
    assert dict(zip(rows.data["feature"], rows.data["y"])) == {"MAR3": 1.5, "MAR1": 0.5, "MAR2": 0.5}
    # Three standard deviations either way, as a reduction coloured by a row is,
    # though no value here reaches two.
    (mapper,) = plot.select({"type": LinearColorMapper})
    assert (mapper.low, mapper.high) == (-3.0, 3.0)
    (callback,) = plot.js_event_callbacks["tap"]
    # Bottom to top, as the image's rows count.
    assert callback.args["ids"] == ["MAR1", "MAR3"]
    assert callback.args["labels"] == [labels["MAR1"], "Short"]
    assert callback.args["data_type"] == "reactions"


def test_a_gene_heatmap_reads_the_most_variable_genes_of_the_grouped_samples():
    frame = pd.DataFrame(
        {"FLAT": [1.0, 1.0, 1.0, 9.0], "UP": [0.0, 1.0, 5.0, 5.0], "SMALL": [0.0, 0.1, 0.2, 0.2]},
        index=["s1", "s2", "s3", "s4"],
    )

    block = sample_heatmap_block(DenseExpressionMatrix(frame), ["s3", "s1", "s2"], 2)
    plot = plot_sample_heatmap(block, {"s1": "A", "s2": "A", "s3": "B"}, count=2, title="Genes", data_type="expression")

    assert list(block.index) == ["s3", "s1", "s2"]
    assert set(block.columns) == {"FLAT", "UP", "SMALL"}
    # FLAT varies only through s4, which is in no group, so it shows no row.
    assert set(plot.yaxis[0].major_label_overrides.values()) == {"UP", "SMALL", "Group"}
    assert any("<strong>Gene:</strong>" in tool.tooltips for tool in plot.select({"type": HoverTool}))


def test_scatters_draw_one_legend_entry_per_present_class():
    table = pd.DataFrame(
        {
            "feature_id": ["R1", "R2", "R3"],
            "reaction_effect": [1.0, 0.5, -0.2],
            "gene_effect": [0.8, 0.0, 0.9],
            "strongest_gene": ["G1", "G2", "G3"],
            "genes": [1, 1, 1],
            "significant_genes": [1, 0, 1],
            "status": ["follows", "network", "genes"],
        }
    )
    plot = plot_reaction_gene_concordance(table, reaction_labels={}, gene_labels={"G1": "HK1"}, title="R")
    legend = plot.select_one({"type": Legend})
    assert [item.label.value for item in legend.items] == [
        "Reaction changes with its genes (1)",
        "Reaction changes, its genes do not (1)",
        "Genes change, reaction does not (1)",
    ]
    # Grey is kept for "neither changes", the not significant of a comparison.
    fills = {item.label.value.split(" (")[0]: item.renderers[0].glyph.fill_color for item in legend.items}
    assert fills["Genes change, reaction does not"] == "#009E73"
    json_item(plot)

    agreement = compare_results(
        ResultEffects("Myeloid", pd.Series({"R1": 1.0, "R2": 0.2, "R3": -0.5}), frozenset({"R1"})),
        ResultEffects("T cells", pd.Series({"R1": 0.8, "R2": 0.4, "R3": -0.1}), frozenset({"R1", "R3"})),
    )
    plot = plot_result_agreement(agreement, labels={})
    assert plot.title.text.startswith("Spearman's rho of Cohen's d over 3 shared features")
    json_item(plot)


def test_a_hidden_group_leaves_the_heatmap_and_a_highlighted_one_is_outlined_the_rest_faded():
    rng = np.random.default_rng(3)
    samples = [f"s{i}" for i in range(12)]
    scores = pd.DataFrame(rng.normal(size=(12, 6)), index=samples, columns=[f"R{i}" for i in range(6)])
    groups = {sample: ("A", "B", "C")[i % 3] for i, sample in enumerate(samples)}
    ordered = {sample: group for group in ("A", "B", "C") for sample, own in groups.items() if own == group}

    whole = plot_sample_heatmap(scores, ordered, count=4, title="All")
    part = plot_sample_heatmap(scores, ordered, count=4, title="Part", hidden_groups=("B",), highlight_groups=("C",), fill=True)

    image_whole = next(r for r in whole.renderers if r.glyph.__class__.__name__ == "Image").data_source.data["image"][0]
    image_part = next(r for r in part.renderers if r.glyph.__class__.__name__ == "Image").data_source.data["image"][0]
    # B's four columns gone; A's and C's cells keep the colours scaled over all three.
    assert image_part.shape == (4, 8)
    np.testing.assert_allclose(image_part, np.delete(image_whole, [4, 5, 6, 7], axis=1))
    sources = list(part.select({"type": ColumnDataSource}))
    # Every group's band has an outline; only the highlighted one's is drawn.
    (outline,) = [s for s in sources if "x0" in s.data and "group_id" in s.data]
    lit = [x for x, alpha in zip(outline.data["x0"], outline.data["alpha"]) if alpha]
    assert lit and min(lit) >= 4.0
    assert {group for group, alpha in zip(outline.data["group_id"], outline.data["alpha"]) if alpha} == {"C"}
    # The other group's columns are covered, to show at a faded sample's strength.
    (covers,) = [s for s in sources if "left" in s.data and "alpha" in s.data]
    assert dict(zip(covers.data["group_id"], covers.data["alpha"])) == {"A": pytest.approx(0.8), "C": 0.0}
    (focus,) = [s for s in sources if s.name == "group_focus"]
    assert list(focus.data["group_id"]) == ["C"] and focus.js_property_callbacks["change:data"]
    assert part.sizing_mode == "stretch_both" and whole.sizing_mode == "stretch_width"
    mapper = part.select_one({"type": LinearColorMapper})
    assert (mapper.palette[0].upper(), mapper.palette[-1].upper()) == ("#542788", "#E6AB02") and "#E0E0E0" in [c.upper() for c in mapper.palette]


def test_the_heatmaps_band_previews_and_highlights_a_group_as_its_chip_does():
    rng = np.random.default_rng(3)
    samples = [f"s{i}" for i in range(12)]
    scores = pd.DataFrame(rng.normal(size=(12, 6)), index=samples, columns=[f"R{i}" for i in range(6)])
    ordered = {sample: ("A", "B", "C")[i // 4] for i, sample in enumerate(samples)}

    plot = plot_sample_heatmap(
        scores, ordered, count=4, title="Band", group_event="pick", group_preview_event="preview"
    )

    (tap,) = plot.js_event_callbacks["tap"]
    blocks = tap.args["blocks"]
    assert tap.args["group_event"] == "pick" and tap.args["base"] == 4
    assert (list(blocks.data["group_id"]), list(blocks.data["left"]), list(blocks.data["right"])) == (
        ["A", "B", "C"], [0, 4, 8], [4, 8, 12]
    )
    # Moving over the band previews its group; leaving the figure ends it.
    (moved,) = plot.js_event_callbacks["mousemove"]
    (left,) = plot.js_event_callbacks["mouseleave"]
    assert moved is left and moved.args["event"] == "preview" and moved.args["blocks"] is blocks
    # Without the events, the band only names its groups.
    plain = plot_sample_heatmap(scores, ordered, count=4, title="Plain")
    assert "mousemove" not in plain.js_event_callbacks and "tap" not in plain.js_event_callbacks


def test_the_marker_heatmap_draws_each_groups_block_on_a_levels_scale():
    from metabis.viz.plot_landscape import plot_marker_heatmap
    from metabis.viz.plot_reduction_artifact import feature_level_palette

    markers = pd.DataFrame(
        {
            "group": ["A", "A", "C"],
            "feature_id": ["f1", "f5", "f3"],
            "min_cohens_d": [4.0, 2.0, 6.0],
            "closest_group": ["B", "C", "A"],
            "mean": [6.0, 3.0, 8.3],
            "closest_mean": [1.3, 1.0, 1.3],
        }
    )
    means = pd.DataFrame(
        {"A": {"f1": 6.0, "f5": 3.0, "f3": 1.3}, "B": {"f1": 1.3, "f5": 1.0, "f3": 1.3}, "C": {"f1": 1.3, "f5": 1.0, "f3": 8.3}}
    )

    plot = plot_marker_heatmap(
        markers, means, title="Markers", labels={"f1": "One", "f5": "One"}, group_colors={"A": "#0072B2", "B": "#D55E00", "C": "#009E73"}
    )

    # A name two markers share carries each one's ID.
    assert list(plot.y_range.factors) == ["f3", "One [f5]", "One [f1]", "Group"]
    (mapper,) = [m for m in plot.select({"type": LinearColorMapper})]
    assert mapper.palette == feature_level_palette()
    # One line between A's two markers and C's one.
    spans = [item for item in plot.center if item.__class__.__name__ == "Span"]
    assert [span.location for span in spans] == [1]
    json_item(plot)


def test_the_feature_table_colours_each_significant_d_and_greys_the_rest():
    from metabis.analysis.landscape import FeatureLandscape

    index = ["R1", "R2", "R7"]
    landscape = FeatureLandscape(
        effect=pd.DataFrame({"A": [1.5, 1.2, 2.5], "B": [-1.0, -1.1, -0.5], "C": [0.1, -0.1, 3.0]}, index=index),
        adjusted_pval=pd.DataFrame({"A": [0.01] * 3, "B": [0.01, 0.01, 0.5], "C": [0.5, 0.5, 0.01]}, index=index),
        significant=pd.DataFrame({"A": [True] * 3, "B": [True, True, False], "C": [False, False, True]}, index=index),
    )

    plot = plot_feature_landscape(
        landscape, ["R1", "R2", "R7"], title="Cohen's d",
        labels={"R1": "chloride transport (e→c)", "R2": "chloride transport (e→c)", "R7": "Exchange of putrescine"},
        against={"C": "A + B"}, group_colors={"A": "#0072B2", "B": "#D55E00", "C": "#009E73"},
    )

    # Two transports of one name are told apart by their IDs.
    assert list(plot.y_range.factors) == [
        "Exchange of putrescine", "chloride transport (e→c) [R2]", "chloride transport (e→c) [R1]", "Group",
    ]
    (mapper,) = plot.select({"type": LinearColorMapper})
    assert mapper.nan_color == "#c7c7c7" and mapper.low == -mapper.high
    cells = next(s for s in plot.select({"type": ColumnDataSource}) if "value" in s.data).data
    greyed = [np.isnan(value) for value in cells["value"]]
    assert sum(greyed) == 3
    assert any("higher in C" in hover and "C against A + B" in hover for hover in cells["hover"])
    # One line between A's block of two and C's of one.
    spans = [item for item in plot.center if item.__class__.__name__ == "Span"]
    assert [span.location for span in spans] == [1]
    json_item(plot)


def test_the_feature_table_needs_a_feature():
    from metabis.analysis.landscape import FeatureLandscape

    empty = FeatureLandscape(effect=pd.DataFrame({"A": []}), adjusted_pval=pd.DataFrame({"A": []}), significant=pd.DataFrame({"A": []}))
    with pytest.raises(ValueError, match="No genes are significant in any group"):
        plot_feature_landscape(empty, [], title="Cohen's d", noun="genes")


def test_reactions_against_their_genes_follow_the_subsystem_in_focus():
    from metabis.viz.subsystem_link import FOCUS_SOURCE

    table = pd.DataFrame(
        {
            "feature_id": ["R1", "R2"],
            "reaction_effect": [1.0, 0.5],
            "gene_effect": [0.8, 0.0],
            "strongest_gene": ["G1", "G2"],
            "genes": [1, 1],
            "significant_genes": [1, 0],
            "status": ["follows", "network"],
        }
    )
    plot = plot_reaction_gene_concordance(
        table, reaction_labels={}, gene_labels={}, title="R", members={"R1": ["glyc"]}, categories={"glyc": "Carbohydrate"}
    )

    focus = next(tag for tag in plot.tags if getattr(tag, "name", None) == FOCUS_SOURCE)
    callback = focus.js_property_callbacks["change:data"][0]
    # Each class keeps its colour when drawn over the veil.
    assert sorted(callback.args["colors"]) == sorted(["#0072B2", "#E69F00"])
    assert callback.args["floor"] is None
    json_item(plot)


def test_a_marker_row_tells_the_page_its_feature_and_its_group():
    from bokeh.models import CustomJS

    from metabis.analysis.landscape import FeatureLandscape
    from metabis.viz.plot_landscape import (
        marker_events,
        plot_feature_landscape,
        plot_marker_heatmap,
    )

    markers = pd.DataFrame(
        {"group": ["A", "C"], "feature_id": ["f1", "f3"], "min_cohens_d": [4.0, 6.0], "closest_group": ["B", "A"], "mean": [6.0, 8.3], "closest_mean": [1.3, 1.3]}
    )
    means = pd.DataFrame({"A": {"f1": 6.0, "f3": 1.3}, "B": {"f1": 1.3, "f3": 1.3}, "C": {"f1": 1.3, "f3": 8.3}})
    index = ["R1", "R7"]
    landscape = FeatureLandscape(
        effect=pd.DataFrame({"A": [1.5, 0.2], "B": [-1.0, -0.5], "C": [0.1, 3.0]}, index=index),
        adjusted_pval=pd.DataFrame({"A": [0.01, 0.5], "B": [0.01, 0.5], "C": [0.5, 0.01]}, index=index),
        significant=pd.DataFrame({"A": [True, False], "B": [True, False], "C": [False, True]}, index=index),
    )

    for plot, owners in (
        (plot_marker_heatmap(markers, means, title="Markers", page="overview"), {"f1": "A", "f3": "C"}),
        (plot_feature_landscape(landscape, index, title="Varying", page="overview"), {"R1": "A", "R7": "C"}),
    ):
        events = {model.args.get("event") for model in plot.references() if isinstance(model, CustomJS)}
        assert set(marker_events("overview")) <= events
        source = next(r.data_source for r in plot.renderers if "owner" in r.data_source.data)
        assert dict(zip(source.data["feature"], source.data["owner"])) == owners
        json_item(plot)

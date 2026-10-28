import json

import numpy as np
import pandas as pd
import pytest
from bokeh.embed import json_item
from bokeh.models import ColumnDataSource, HoverTool, Legend, LinearColorMapper

from metabis.analysis.agreement import ResultEffects, compare_results
from metabis.analysis.landscape import Landscape
from metabis.domain.expression_matrix import DenseExpressionMatrix
from metabis.viz.plot_landscape import (
    plot_delta_landscape,
    plot_sample_heatmap,
    sample_heatmap_block,
    sample_heatmap_rows,
)
from metabis.viz.plot_result_scatter import (
    plot_reaction_gene_concordance,
    plot_result_agreement,
)


def _landscape():
    index = ["glyc", "tca", "fao"]
    return Landscape(
        delta=pd.DataFrame({"A": [1.0, 0.5, 0.25], "B": [-0.75, 0.0, 0.0]}, index=index),
        significant=pd.DataFrame({"A": [4, 1, 1], "B": [3, 0, 0]}, index=index),
        names=pd.Series(["Glycolysis", "TCA cycle", "Fatty acid oxidation"], index=index),
        categories=pd.Series(["Carbohydrate", "Energy", "Lipid"], index=index),
        reactions=pd.Series([4, 2, 4], index=index),
        significant_reactions=pd.Series({"A": 9, "B": 3}),
    )


def test_landscape_rows_are_grouped_by_category():
    plot = plot_delta_landscape(_landscape(), ["glyc", "fao"], title="Delta")

    assert list(plot.y_range.factors) == [("Lipid", "Fatty acid oxidation"), ("Carbohydrate", "Glycolysis")]
    assert list(plot.x_range.factors) == ["A", "B"]
    json_item(plot)


def test_landscape_needs_a_subsystem():
    with pytest.raises(ValueError, match="No subsystem meets the rule"):
        plot_delta_landscape(_landscape(), [], title="Delta")


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


def _two_group_landscape():
    delta = pd.DataFrame({"Inflamed": {"glyc": 0.5, "fao": -1.0}, "Healthy": {"glyc": -0.5, "fao": 1.0}})
    return Landscape(
        delta=delta,
        significant=pd.DataFrame({"Inflamed": {"glyc": 2, "fao": 4}, "Healthy": {"glyc": 2, "fao": 4}}),
        names=pd.Series({"glyc": "Glycolysis", "fao": "Fatty acid oxidation"}),
        categories=pd.Series({"glyc": "Carbohydrate", "fao": "Lipid"}),
        reactions=pd.Series({"glyc": 4, "fao": 4}),
        significant_reactions=pd.Series({"Inflamed": 6, "Healthy": 6}),
    )


def test_the_delta_table_carries_the_groups_colours_and_their_other_side():
    from metabis.viz.plot_landscape import plot_delta_landscape

    plot = plot_delta_landscape(
        _two_group_landscape(), ["glyc", "fao"], title="Genes", noun="genes",
        against={"Inflamed": "Healthy", "Healthy": "Inflamed"},
        group_colors={"Inflamed": "#0072B2", "Healthy": "#D55E00"},
    )

    band = next(s for s in plot.select({"type": ColumnDataSource}) if "fill_color" in s.data)
    assert list(band.data["fill_color"]) == ["#0072B2", "#D55E00"]
    hovers = next(s for s in plot.select({"type": ColumnDataSource}) if "hover" in s.data).data["hover"]
    assert "Inflamed against Healthy" in hovers[0] and "of 4 genes" in hovers[0]
    json_item(plot)


def test_the_enrichment_table_greys_what_is_not_significant_and_dashes_what_was_not_tested():
    from metabis.analysis.landscape import EnrichmentLandscape
    from metabis.viz.plot_landscape import plot_enrichment_landscape

    landscape = EnrichmentLandscape(
        enrichment=pd.DataFrame({"Inflamed": {"glyc": 1.5, "fao": np.nan}, "Healthy": {"glyc": np.nan, "fao": -2.0}}),
        adjusted_pval=pd.DataFrame({"Inflamed": {"glyc": 0.01, "fao": 0.3}, "Healthy": {"glyc": 0.5, "fao": 0.001}}),
        tested=pd.DataFrame({"Inflamed": {"glyc": True, "fao": True}, "Healthy": {"glyc": False, "fao": True}}),
        names=pd.Series({"glyc": "Glycolysis", "fao": "Fatty acid oxidation"}),
        categories=pd.Series({"glyc": "Carbohydrate", "fao": "Lipid"}),
    )

    plot = plot_enrichment_landscape(landscape, ["glyc", "fao"], title="ORA")

    (mapper,) = plot.select({"type": LinearColorMapper})
    assert mapper.nan_color == "#c7c7c7" and mapper.high == 2.0
    dashed = [r for r in plot.renderers if getattr(r.glyph, "line_dash", None) == "dashed"]
    assert len(dashed) == 1 and len(dashed[0].data_source.data["group"]) == 1
    hovers = [h for s in plot.select({"type": ColumnDataSource}) if "hover" in s.data for h in s.data["hover"]]
    assert any("over-represented among the reactions higher in Inflamed" in h for h in hovers)
    assert any("Not tested" in h for h in hovers)
    json_item(plot)


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

    plot = plot_marker_heatmap(markers, means, title="Markers", labels={"f1": "One"}, group_colors={"A": "#0072B2", "B": "#D55E00", "C": "#009E73"})

    assert list(plot.y_range.factors) == ["f3", "f5", "One", "Group"]
    (mapper,) = [m for m in plot.select({"type": LinearColorMapper})]
    assert mapper.palette == feature_level_palette()
    # One line between A's two markers and C's one.
    spans = [item for item in plot.center if item.__class__.__name__ == "Span"]
    assert [span.location for span in spans] == [1]
    json_item(plot)

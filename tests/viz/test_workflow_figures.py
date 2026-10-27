import json

import numpy as np
import pandas as pd
import pytest
from bokeh.embed import json_item
from bokeh.models import HoverTool, Legend, LinearColorMapper

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
    json_item(plot)

    agreement = compare_results(
        ResultEffects("Myeloid", pd.Series({"R1": 1.0, "R2": 0.2, "R3": -0.5}), frozenset({"R1"})),
        ResultEffects("T cells", pd.Series({"R1": 0.8, "R2": 0.4, "R3": -0.1}), frozenset({"R1", "R3"})),
    )
    plot = plot_result_agreement(agreement, labels={})
    assert plot.title.text.startswith("Spearman's rho of Cohen's d over 3 shared features")
    json_item(plot)

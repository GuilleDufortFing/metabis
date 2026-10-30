"""The rankings' rows: each category's heading, then its subsystems, nothing between."""

from __future__ import annotations

from metabis.viz.ranking_layout import HEADING_CHARACTER_WIDTH, NAMES_MIN_WIDTH, RankedSubsystem, order_rows

LONG = "Sulfur Amino Acids & One-Carbon Metabolism"
ENTRIES = [
    RankedSubsystem("glyc", "Glycolysis", "Carbohydrate", 0.9),
    RankedSubsystem("tca", "TCA cycle", "Carbohydrate", 0.5),
    RankedSubsystem("met", "Methionine metabolism", LONG, 0.3),
]


def test_a_heading_is_the_only_row_between_two_categories():
    rows = order_rows(ENTRIES, numbered=False)

    # The network's order puts the named category first.
    assert rows.factors == ["heading:1:0", "Methionine metabolism", "heading:2:0", "Glycolysis", "TCA cycle"]


def test_away_from_the_network_the_names_widen_so_a_heading_keeps_one_line():
    rows = order_rows(ENTRIES, numbered=False)

    assert [heading.text for heading in rows.headings] == [LONG, "Carbohydrate"]
    assert rows.width >= len(LONG) * HEADING_CHARACTER_WIDTH


def test_beside_the_network_a_long_heading_takes_two_lines_and_the_names_keep_their_width():
    rows = order_rows(ENTRIES)

    long = rows.headings[0]
    assert long.text.count("\n") == 1 and len(long.factors) == 2
    assert rows.width == NAMES_MIN_WIDTH


def test_the_score_diamonds_mark_their_scores_on_one_shared_colour_reference():
    """The network's colour reference is drawn once, on the hover tool, each diamond marking its own score."""
    import pandas as pd
    from bokeh.models import HoverTool, Range1d

    from metabis.analysis.distribution_binner import DistributionBinner
    from metabis.viz.color_reference import build_gradient_color_reference
    from metabis.viz.ranking_layout import ranking_figure, score_panel

    binner = DistributionBinner(pd.Series({f"s{index}": index / 4 - 1 for index in range(9)}), center_method="zero", spread_method="std")
    reference = build_gradient_color_reference([{**bar, "color": "#ff0000"} for bar in binner.get_histogram_data()], metric_label="ΔS")
    rows = order_rows(ENTRIES)
    plot = ranking_figure(rows, title="Cohen's d", x_range=Range1d(-1, 1), x_label="Cohen's d")
    scores = {"glyc": 0.5, "tca": -0.25, "met": 1.0}

    panel = score_panel(rows, plot, scores, title="Delta score", x_label="Delta score", hovers={"glyc": "<div>node</div>"}, reference=reference)

    (tool,) = [tool for tool in panel.tools if isinstance(tool, HoverTool)]
    data = tool.renderers[0].data_source.data
    assert data["color_reference_value"] == data["score"]
    assert data["color_reference_marker_x"] == [reference.marker_x(value) for value in data["score"]]
    assert "@color_reference_marker_x" in tool.tooltips and "Color Reference" in tool.tooltips
    # The rows' own hovers carry no picture.
    assert all("<svg" not in hover for hover in data["hover"])

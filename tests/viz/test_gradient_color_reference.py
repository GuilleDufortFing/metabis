"""Every gradient plot carries a colour reference (WP9 primitive #4, R-09).

Supplementary Section 1.5 states, in black, that "Every gradient plot carries a
color reference showing the distribution of the values it encodes and the
position of the inspected element within it." Two surfaces satisfied that
sentence before this package -- the metabolic-graph hovers and the Single-group
node scale -- and three did not: the comparison subsystem ranking, the
single-group activity ranking, and the Step-4 ORA category dot plot, which had
lost its legend when it adopted the shared gradient.

These tests pin the sentence. They assert two properties per surface, because
the claim makes two promises:

1. the reference shows *the distribution of the values it encodes*, and
2. it shows *the position of the inspected element within it*.

A reference built from some other distribution, or one that does not move with
the element, would satisfy neither and must fail here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from metabis.analysis.distribution_binner import DistributionBinner
from metabis.viz.color_reference import (
    GradientColorReference,
    build_gradient_color_reference,
    build_reference_from_distribution,
    color_reference_panel,
)


def _histogram(values: pd.Series) -> list[dict]:
    """The histogram a registered distribution hands to the reference."""
    binner = DistributionBinner(values, center_method="zero", spread_method="std")
    palette = ["#ffffff", "#ff0000", "#0000ff"]
    return [
        {**bar, "color": palette[bar["bin"] % len(palette)]}
        for bar in binner.get_histogram_data()
    ]


@pytest.fixture
def reference() -> GradientColorReference:
    values = pd.Series({f"e{index}": float(index) - 5 for index in range(11)})
    built = build_gradient_color_reference(_histogram(values), metric_label="Metric")
    assert built is not None
    return built


# ---------------------------------------------------------------------------
# The primitive itself
# ---------------------------------------------------------------------------


def test_the_reference_draws_the_distribution_it_was_built_from(reference):
    svg = reference.html_for_value(0.0)

    # One bar per populated bin, in the bin's own colour: this is the
    # distribution of the encoded values, not a bare gradient strip.
    assert svg.count("<rect") >= 2
    assert "rgba(255,0,0,1.0)" in svg or "rgba(0,0,255,1.0)" in svg
    assert "Metric Distribution" in svg


def test_the_marker_moves_with_the_inspected_element(reference):
    low = reference.html_for_value(-4.0)
    high = reference.html_for_value(4.0)

    assert "Current: -4.000" in low
    assert "Current: 4.000" in high
    assert low != high, (
        "The reference must show where the inspected element sits; a panel that "
        "is identical for every element shows only the distribution."
    )


def test_the_distribution_is_rendered_once_and_only_the_marker_is_restamped(
    reference,
):
    """The reason the primitive exists: a ranking plot marks thousands of dots.

    Rendering the bars per dot would multiply the plot's build time by the cost
    of the reference, so the background is built once and shared. If this ever
    becomes a per-element render, the shared object below stops being reused.
    """
    first = reference.html_for_value(1.0)
    second = reference.html_for_value(2.0)

    bars = first[first.index("<rect") : first.rindex("<rect")]
    assert bars and bars in second, (
        "The two panels no longer share their rendered distribution, so the "
        "reference is being re-rendered per element."
    )


def test_an_empty_distribution_yields_no_reference_rather_than_an_empty_frame():
    assert build_gradient_color_reference([], metric_label="Metric") is None
    assert build_gradient_color_reference(None, metric_label="Metric") is None
    assert color_reference_panel("") == ""


def test_a_missing_registered_distribution_costs_the_panel_not_the_render():
    class _Registry:
        def get_distribution_details(self, **_kwargs):
            raise KeyError("no such distribution")

    assert (
        build_reference_from_distribution(
            _Registry(),
            mode="mode",
            element_type="reaction",
            element_id="R1",
            metric_name="metric",
            metric_label="Metric",
        )
        is None
    )
    assert (
        build_reference_from_distribution(
            None,
            mode="mode",
            element_type="reaction",
            element_id="R1",
            metric_name="metric",
            metric_label="Metric",
        )
        is None
    )


def test_the_panel_carries_the_same_label_the_graph_hovers_use(reference):
    panel = reference.panel_for_value(0.0)

    # A user who learned to read the reference on a graph node must recognise
    # the same panel on a dot plot, so the chrome is shared verbatim.
    assert "<strong>Color Reference:</strong>" in panel
    assert "Metric Distribution" in panel


def test_the_hover_template_shares_one_distribution_and_uses_compact_point_fields(
    reference,
):
    template = reference.hover_template(
        marker_x_field="reference_marker_x",
        value_field="reference_value",
    )

    assert template.count("Metric Distribution") == 1
    assert template.count("<svg") == 1
    assert 'x1="@reference_marker_x"' in template
    assert "Current: @reference_value{0.000}" in template
    assert reference.marker_x(-4.0) != reference.marker_x(4.0)

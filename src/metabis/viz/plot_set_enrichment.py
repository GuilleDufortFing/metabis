"""Artifact-first subsystem category plot for reaction- and gene-set ORA.

Each subsystem contributes one dot, placed at its dominant direction's signed
``log2(enrichment ratio)`` (the larger-magnitude of Target-higher/Target-lower,
a tie resolving to Target-higher) and coloured on the shared diverging gradient.
The other direction is reported on hover, so the plot reads exactly like the
overview subsystem glyph.
"""

from __future__ import annotations

import math
from html import escape
from typing import Mapping

import pandas as pd
from bokeh.models import ColumnDataSource, FactorRange, HoverTool, Range1d, Span
from bokeh.plotting import figure

from ..analysis.reaction_direction import ReactionDirectionCriterion
from ..analysis.set_enrichment import (
    SetEnrichmentArtifact,
    SetEnrichmentDirection,
    SetEnrichmentFeatureType,
)
from ..visual_config import theme
from .metabolic_visualization.generation.data_structures.constants import (
    VisualizationConstants,
)
from .plot_subsystem_ranking import (
    _add_category_decorations_horizontal,
    _create_zoom_dependent_labels,
)
from .plot_utils import apply_academic_style
from .set_enrichment_overview import signed_log2_gradient_scale

# One dot per subsystem takes the shared signed-log2 gradient (red for
# over-representation among Target-higher features, blue for Target-lower); a
# subsystem significant in neither direction is gray, the same gray the overview
# glyphs use. This reads exactly like the overview subsystem glyph: the dot sits
# at the dominant direction's signed value and the other direction is on hover.
NOT_SIGNIFICANT_COLOR = theme.NOT_SIGNIFICANT


def _finite_number(value: object, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _format_number(value: object, *, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Not available"
    if not math.isfinite(number):
        return "Not available"
    return f"{number:.{digits}f}"


def _format_probability(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Not available"
    if not math.isfinite(number):
        return "Not available"
    if number <= 0:
        return "&lt; 2.23 × 10<sup>−308</sup>"
    if number < 0.001:
        coefficient, exponent = f"{number:.2e}".split("e")
        return f"{coefficient} × 10<sup>{int(exponent)}</sup>"
    return f"{number:.3g}"


def _direction_criterion_label(artifact: SetEnrichmentArtifact) -> str:
    criterion = artifact.config.reaction_direction_criterion
    if criterion is ReactionDirectionCriterion.DOMINANT_IN_TARGET:
        return "Dominant in Target"
    if criterion is ReactionDirectionCriterion.DOMINANT_IN_BASELINE:
        return "Dominant in Baseline"
    return "Not applicable to gene-set enrichment"


def _paired_frame(artifact: SetEnrichmentArtifact) -> pd.DataFrame:
    """Return one display row per subsystem without changing scientific values."""
    frame = artifact.result_frame()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "subsystem_id",
                "subsystem_name",
                "category",
                "feature_type",
                "y_factor",
                "maximum_visual_magnitude",
                "target_visual",
                "baseline_visual",
                "dominant_visual",
                "dominant_color",
                "significant",
                "hover_html",
                "hover",
                "display_label",
                "is_subsystem",
            ]
        )

    duplicated = frame.duplicated(["subsystem_id", "direction"], keep=False)
    if duplicated.any():
        raise ValueError(
            "Set-enrichment artifacts must contain one row per subsystem direction"
        )

    indexed = frame.set_index(["subsystem_id", "direction"], drop=False)
    expected = {direction.value for direction in SetEnrichmentDirection}
    rows: list[dict[str, object]] = []
    for subsystem_id, subsystem_rows in frame.groupby("subsystem_id", sort=False):
        directions = set(map(str, subsystem_rows["direction"]))
        if directions != expected:
            raise ValueError(
                "Set-enrichment artifacts must contain both Target directions for "
                f"subsystem {subsystem_id!r}"
            )
        target = indexed.loc[(subsystem_id, SetEnrichmentDirection.TARGET_HIGHER.value)]
        baseline = indexed.loc[
            (subsystem_id, SetEnrichmentDirection.TARGET_LOWER.value)
        ]
        category = str(target.get("category") or "Uncategorised")
        subsystem_name = str(target["subsystem_name"])
        target_visual = max(0.0, _finite_number(target["signed_log2_enrichment_ratio"]))
        baseline_visual = min(
            0.0, _finite_number(baseline["signed_log2_enrichment_ratio"])
        )
        # One signed value per subsystem: the larger-magnitude direction with its
        # sign preserved, an exact tie resolving to the positive (Target-higher)
        # side. This mirrors the overview subsystem glyph exactly.
        dominant_visual = (
            baseline_visual
            if abs(baseline_visual) > abs(target_visual)
            else target_visual
        )
        target_significant = bool(target["is_significant"])
        baseline_significant = bool(baseline["is_significant"])
        row = {
            "subsystem_id": str(subsystem_id),
            "subsystem_name": subsystem_name,
            "category": category,
            "feature_type": str(target["feature_type"]),
            "target_visual": target_visual,
            "baseline_visual": baseline_visual,
            "dominant_visual": dominant_visual,
            "maximum_visual_magnitude": max(abs(target_visual), abs(baseline_visual)),
            "target_significant": target_significant,
            "baseline_significant": baseline_significant,
            "significant": target_significant or baseline_significant,
            "target_enrichment_ratio": target["enrichment_ratio"],
            "baseline_enrichment_ratio": baseline["enrichment_ratio"],
            "target_adjusted_pval": target["adjusted_pval"],
            "baseline_adjusted_pval": baseline["adjusted_pval"],
            "target_overlap_size": int(target["overlap_size"]),
            "baseline_overlap_size": int(baseline["overlap_size"]),
            "target_query_size": int(target["query_size"]),
            "baseline_query_size": int(baseline["query_size"]),
            "subsystem_size": int(target["subsystem_size"]),
            "universe_size": int(target["universe_size"]),
            "correction": str(target["correction"]),
            "correction_family_size": int(target["correction_family_size"]),
            "display_label": subsystem_name,
            "is_subsystem": True,
        }
        rows.append(row)

    paired = pd.DataFrame(rows)
    category_scores = paired.groupby("category")["maximum_visual_magnitude"].mean()
    category_order = {
        category: index
        for index, category in enumerate(
            sorted(
                category_scores.index,
                key=lambda category: (-category_scores[category], str(category)),
            )
        )
    }
    paired["category_order"] = paired["category"].map(category_order)
    paired = paired.sort_values(
        ["category_order", "maximum_visual_magnitude", "subsystem_id"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    reference = _assign_gradient_colors(paired)
    paired["y_factor"] = list(zip(paired["category"], paired["subsystem_name"]))
    paired["hover_html"] = [
        _hover_html(row, artifact=artifact) for _, row in paired.iterrows()
    ]
    paired["hover"] = paired["hover_html"]
    if reference is not None:
        values = paired["dominant_visual"].astype(float)
        paired["color_reference_value"] = values
        paired["color_reference_marker_x"] = [
            reference.marker_x(value) for value in values
        ]
    paired.attrs["color_reference"] = reference
    return paired


def _assign_gradient_colors(paired: pd.DataFrame):
    """Colour each subsystem's single dot on the shared signed-log2 gradient.

    The dot sits at the subsystem's dominant directional visual, so the diverging
    red/white/blue scale is registered over one signed value per subsystem --
    identical to the overview glyph distribution. A subsystem significant in
    neither direction is gray.

    Returns:
        The color reference for that same distribution, or ``None`` when there is
        nothing to plot.
    """
    signed = pd.Series(
        paired["dominant_visual"].to_list(),
        index=list(range(len(paired))),
        dtype=float,
    )
    gradient, reference = signed_log2_gradient_scale(signed)
    colors = list(gradient)
    paired["dominant_color"] = [
        colors[index] if significant else NOT_SIGNIFICANT_COLOR
        for index, significant in enumerate(paired["significant"])
    ]
    return reference


def _hover_html(row: pd.Series, *, artifact: SetEnrichmentArtifact) -> str:
    family = (
        "Reaction-Set Enrichment - Over-Representation Analysis"
        if artifact.feature_type is SetEnrichmentFeatureType.REACTION
        else "Gene-Set Enrichment - Over-Representation Analysis"
    )
    target_status = "Significant" if row["target_significant"] else "Not significant"
    baseline_status = (
        "Significant" if row["baseline_significant"] else "Not significant"
    )
    correction = {
        "fdr_bh": "Benjamini–Hochberg FDR",
        "holm": "Holm FWER",
    }.get(row["correction"], row["correction"])
    return (
        "<div style='font-family:Inter,-apple-system,BlinkMacSystemFont,&quot;Segoe UI&quot;,sans-serif;"
        "width:390px;max-width:390px;color:#1E2A32;background:#FFFFFF;"
        "border:1px solid #D9E1E8;border-radius:8px;overflow:hidden;font-size:12px;'>"
        "<div style='padding:10px 12px;background:#F7F9FB;border-bottom:1px solid #D9E1E8;'>"
        f"<div style='font-size:14px;font-weight:700;color:#20435C;'>{escape(str(row['subsystem_name']))}</div>"
        f"<div style='color:#5C6B76;'>{escape(str(row['category']))} · {family}</div></div>"
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px 14px;padding:11px 12px;'>"
        "<div style='border-left:4px solid #C62828;padding-left:8px;'><strong>Target-higher</strong><br/>"
        f"{target_status}<br/>Ratio {_format_number(row['target_enrichment_ratio'])}<br/>"
        f"Adjusted p {_format_probability(row['target_adjusted_pval'])}<br/>"
        f"Overlap {row['target_overlap_size']} of {row['subsystem_size']} · query {row['target_query_size']}</div>"
        "<div style='border-left:4px solid #1565C0;padding-left:8px;'><strong>Target-lower</strong><br/>"
        f"{baseline_status}<br/>Ratio {_format_number(row['baseline_enrichment_ratio'])}<br/>"
        f"Adjusted p {_format_probability(row['baseline_adjusted_pval'])}<br/>"
        f"Overlap {row['baseline_overlap_size']} of {row['subsystem_size']} · query {row['baseline_query_size']}</div>"
        "</div><div style='padding:8px 12px;border-top:1px solid #E5EBF0;background:#FBFCFD;color:#52616D;'>"
        f"Background N={row['universe_size']} · {escape(str(correction))} over "
        f"{row['correction_family_size']} hypotheses · q ≤ "
        f"{_format_probability(artifact.config.adjusted_pvalue_threshold)}<br/>"
        f"Reaction direction criterion: {escape(_direction_criterion_label(artifact))}"
        "</div></div>"
    )


def plot_set_enrichment_category(
    artifact: SetEnrichmentArtifact,
    *,
    category_colors: Mapping[str, str] | None = None,
):
    """Plot one dot per subsystem using the shared category-dot structure."""
    if not isinstance(artifact, SetEnrichmentArtifact):
        raise TypeError("artifact must be a SetEnrichmentArtifact")

    frame = _paired_frame(artifact)
    factors = list(reversed(frame.get("y_factor", pd.Series(dtype=object)).tolist()))
    y_range = FactorRange(factors=factors, bounds="auto", range_padding=0.02)

    values = [
        value
        for value in frame.get("dominant_visual", pd.Series(dtype=float)).tolist()
        if math.isfinite(value)
    ]
    x_min = min([0.0, *values])
    x_max = max([0.0, *values])
    padding = (x_max - x_min) * 0.1 if x_max != x_min else 1.0
    x_range = Range1d(x_min - padding, x_max + padding)

    colors = category_colors or {}
    sorted_categories = list(dict.fromkeys(frame.get("category", [])))
    ordered_data = {
        "factors": list(reversed(factors)),
        "y_factors": factors,
        "sorted_categories": sorted_categories,
        "category_info": {
            category: {
                "color": colors.get(
                    category,
                    VisualizationConstants.DEFAULT_CATEGORY_COLOR_UNMAPPED,
                ),
            }
            for category in sorted_categories
        },
    }
    family = (
        "Reaction"
        if artifact.feature_type is SetEnrichmentFeatureType.REACTION
        else "Gene"
    )
    plot = figure(
        title=(
            f"{family}-Set Enrichment - Over-Representation Analysis "
            "by subsystem category"
        ),
        x_axis_label=(
            "Signed log2(enrichment ratio): Target-lower ← 0 → Target-higher"
        ),
        x_range=x_range,
        y_range=y_range,
        height=max(600, 20 * len(factors)),
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save,tap",
        toolbar_location="above",
        x_axis_location="above",
        border_fill_color="rgba(255,255,255,0)",
    )
    apply_academic_style(plot)
    _add_category_decorations_horizontal(
        p=plot,
        ordered_data=ordered_data,
        x_range=x_range,
    )
    source = ColumnDataSource(frame)
    points = plot.scatter(
        x="dominant_visual",
        y="y_factor",
        source=source,
        size=10,
        marker="circle",
        fill_color="dominant_color",
        line_color="dominant_color",
        fill_alpha=0.9,
        name="set_enrichment_subsystem_points",
    )
    reference = frame.attrs.get("color_reference")
    hover_tooltip = "@hover_html{safe}"
    if reference is not None:
        hover_tooltip += reference.hover_template(
            marker_x_field="color_reference_marker_x",
            value_field="color_reference_value",
        )
    plot.add_tools(
        HoverTool(
            renderers=[points],
            tooltips=hover_tooltip,
            attachment="vertical",
            show_arrow=False,
        )
    )
    plot.add_layout(
        Span(
            location=0.0,
            dimension="height",
            line_color=theme.THRESHOLD_LINE_COLOR,
            line_dash=theme.THRESHOLD_LINE_DASH,
            line_width=theme.THRESHOLD_LINE_WIDTH,
            line_alpha=0.7,
        )
    )
    plot.xgrid.grid_line_color = "#404040"
    plot.xgrid.grid_line_alpha = 0.5
    plot.ygrid.grid_line_color = None
    plot.yaxis.axis_label = None
    plot.yaxis.visible = False
    _create_zoom_dependent_labels(
        p=plot,
        ordered_data=ordered_data,
        x_range=x_range,
    )
    plot.tags.append(
        {
            "artifact_id": artifact.metadata.artifact_id,
            "artifact_kind": artifact.metadata.kind.value,
            "feature_type": artifact.feature_type.value,
            "presentation": "paired_set_enrichment_category",
        }
    )
    return plot


__all__ = ["plot_set_enrichment_category"]

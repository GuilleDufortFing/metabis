"""Bokeh rendering adapter for reaction-set enrichment artifacts."""

from __future__ import annotations

import math
from html import escape

from bokeh.models import (
    ColorBar,
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    NumeralTickFormatter,
)
from bokeh.palettes import Viridis256
from bokeh.plotting import figure

from ..analysis.reaction_set_enrichment import ReactionSetEnrichmentArtifact
from .plot_utils import apply_academic_style

MAX_PLOT_ROWS = 30
MIN_POINT_SIZE = 9.0
MAX_POINT_SIZE = 25.0


def _negative_log10(value: object) -> float:
    """Return a finite-safe visual coordinate, preserving missingness as NaN."""
    if value is None:
        return math.nan
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    if math.isnan(numeric):
        return math.nan
    return -math.log10(max(numeric, float.fromhex("0x1.0p-1022")))


def _format_probability(value: object) -> str:
    """Format an adjusted p-value for compact scientific reading."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "Not available"
    if not math.isfinite(numeric):
        return "Not available"
    if numeric <= 0:
        return "&lt; 2.23 × 10<sup>−308</sup>"
    if numeric < 0.001:
        coefficient, exponent = f"{numeric:.2e}".split("e")
        return f"{coefficient} × 10<sup>{int(exponent)}</sup>"
    return f"{numeric:.3g}"


def _format_metric(metric: object) -> str:
    return {
        "cohens_d": "Cohen's d",
        "log2_fc": "log2 fold change",
        "mean_difference": "mean difference",
    }.get(str(metric), str(metric).replace("_", " "))


def _format_correction(correction: object) -> str:
    return {
        "fdr_bh": "Benjamini–Hochberg FDR",
        "holm": "Holm FWER",
    }.get(str(correction), str(correction))


def _format_ratio(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "Not available"
    if not math.isfinite(numeric):
        return "Not available"
    return f"{numeric:.3f}"


def _hover_html(row, *, adjusted_pvalue_threshold: float) -> str:
    """Build a bounded scientific summary, without provenance or reaction IDs."""
    significant = bool(row["is_significant"])
    status_label = "Significant enrichment" if significant else "Not significant"
    status_color = "#166B4D" if significant else "#52616D"
    status_background = "#E7F5EF" if significant else "#EEF2F5"
    subsystem_name = escape(str(row["subsystem_name"]))
    category = escape(str(row.get("category") or "Uncategorised"))
    direction = escape(str(row["direction"]).capitalize())
    metric = escape(_format_metric(row["direction_metric"]))
    correction = escape(_format_correction(row["correction"]))
    overlap_size = int(row["overlap_size"])
    subsystem_size = int(row["subsystem_size"])
    query_size = int(row["query_size"])
    universe_size = int(row["universe_size"])
    enrichment_ratio = _format_ratio(row["enrichment_ratio"])
    adjusted_pvalue = _format_probability(row["adjusted_pval"])
    threshold = _format_probability(adjusted_pvalue_threshold)

    return (
        "<div style='font-family:Inter,-apple-system,BlinkMacSystemFont,&quot;Segoe UI&quot;,sans-serif;"
        "width:340px;max-width:340px;padding:0;color:#1E2A32;background:#FFFFFF;"
        "border:1px solid #D9E1E8;border-radius:8px;box-shadow:0 4px 18px rgba(0,0,0,0.14);"
        "font-size:12px;line-height:1.4;overflow:hidden;'>"
        "<div style='padding:11px 13px 9px;border-bottom:1px solid #D9E1E8;background:#F7F9FB;'>"
        f"<div style='font-size:14px;font-weight:700;line-height:1.3;color:#20435C;'>{subsystem_name}</div>"
        f"<div style='margin-top:3px;color:#5C6B76;'>{category}</div>"
        f"<span style='display:inline-block;margin-top:7px;padding:2px 7px;border-radius:10px;"
        f"background:{status_background};color:{status_color};font-size:11px;font-weight:650;'>"
        f"{status_label}</span></div>"
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:9px 16px;padding:11px 13px;'>"
        "<div><div style='color:#6E7D88;font-size:10px;text-transform:uppercase;letter-spacing:.04em;'>"
        "Adjusted p-value</div>"
        f"<div style='margin-top:2px;font-weight:650;font-variant-numeric:tabular-nums;'>{adjusted_pvalue}</div></div>"
        "<div><div style='color:#6E7D88;font-size:10px;text-transform:uppercase;letter-spacing:.04em;'>"
        "Enrichment ratio</div>"
        f"<div style='margin-top:2px;font-weight:650;font-variant-numeric:tabular-nums;'>{enrichment_ratio}</div></div>"
        "<div><div style='color:#6E7D88;font-size:10px;text-transform:uppercase;letter-spacing:.04em;'>"
        "Subsystem overlap</div>"
        f"<div style='margin-top:2px;font-weight:650;font-variant-numeric:tabular-nums;'>{overlap_size} of {subsystem_size}</div>"
        "<div style='color:#6E7D88;font-size:10px;'>significant / measured</div></div>"
        "<div><div style='color:#6E7D88;font-size:10px;text-transform:uppercase;letter-spacing:.04em;'>"
        "Analysis query</div>"
        f"<div style='margin-top:2px;font-weight:650;font-variant-numeric:tabular-nums;'>{query_size} of {universe_size}</div>"
        "<div style='color:#6E7D88;font-size:10px;'>query / background</div></div></div>"
        "<div style='padding:9px 13px;border-top:1px solid #E5EBF0;background:#FBFCFD;color:#52616D;'>"
        f"<div><strong>{direction}</strong> direction · {metric}</div>"
        f"<div style='margin-top:2px;'>{correction} · significance q ≤ {threshold}</div>"
        "</div></div>"
    )


def _display_labels(frame) -> list[str]:
    """Return readable, unique categorical labels without exposing IDs."""
    names = [str(value) for value in frame.get("subsystem_name", ())]
    totals = {name: names.count(name) for name in set(names)}
    occurrences: dict[str, int] = {}
    labels = []
    for name in names:
        occurrences[name] = occurrences.get(name, 0) + 1
        suffix = f" ({occurrences[name]})" if totals[name] > 1 else ""
        labels.append(f"{name}{suffix}")
    return labels


def plot_reaction_set_enrichment(
    artifact: ReactionSetEnrichmentArtifact,
):
    """Render an enrichment dot plot without running scientific analysis.

    The artifact is the sole scientific input.  The adapter only computes visual
    encodings (``-log10`` position, point size, and colour) from its stored rows.
    """
    if not isinstance(artifact, ReactionSetEnrichmentArtifact):
        raise TypeError("artifact must be a ReactionSetEnrichmentArtifact")

    frame = artifact.result_frame()
    if not frame.empty:
        frame = frame.sort_values(
            ["adjusted_pval", "enrichment_ratio", "subsystem_id"],
            ascending=[True, False, True],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
    total_rows = len(frame)
    frame = frame.head(MAX_PLOT_ROWS).copy()
    frame["display_label"] = _display_labels(frame)
    frame["neg_log10_adjusted_pval"] = [
        _negative_log10(value) for value in frame.get("adjusted_pval", ())
    ]
    maximum_overlap = max((int(value) for value in frame.get("overlap_size", ())), default=0)
    frame["point_size"] = [
        MIN_POINT_SIZE
        + (MAX_POINT_SIZE - MIN_POINT_SIZE)
        * math.sqrt(max(0, int(value)) / maximum_overlap)
        if maximum_overlap
        else MIN_POINT_SIZE
        for value in frame.get("overlap_size", ())
    ]
    frame["hover_html"] = [
        _hover_html(
            row,
            adjusted_pvalue_threshold=artifact.config.adjusted_pvalue_threshold,
        )
        for _, row in frame.iterrows()
    ]

    factors = list(reversed(frame.get("display_label", []).tolist()))
    height = max(360, 100 + 25 * len(factors))
    title = "Reaction-set over-representation analysis"
    if total_rows > len(frame):
        title += f" · top {len(frame)} of {total_rows} eligible subsystems"
    plot = figure(
        title=title,
        x_axis_label="−log10(adjusted p-value)",
        y_range=factors,
        height=height,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save,tap",
        toolbar_location="above",
    )
    apply_academic_style(plot)

    ratios = [
        float(value)
        for value in frame.get("enrichment_ratio", ())
        if value is not None and not math.isnan(float(value))
    ]
    low = min(ratios, default=0.0)
    high = max(ratios, default=1.0)
    if low == high:
        high = low + 1.0
    mapper = LinearColorMapper(palette=Viridis256, low=low, high=high)
    source = ColumnDataSource(frame)
    renderer = plot.scatter(
        x="neg_log10_adjusted_pval",
        y="display_label",
        source=source,
        size="point_size",
        fill_color={"field": "enrichment_ratio", "transform": mapper},
        line_color="#263238",
        line_alpha=0.7,
        fill_alpha=0.85,
        marker="circle",
        name="reaction_set_enrichment_dots",
    )
    plot.add_tools(
        HoverTool(
            renderers=[renderer],
            tooltips="@hover_html{safe}",
            attachment="vertical",
            show_arrow=False,
        )
    )
    plot.add_layout(
        ColorBar(
            color_mapper=mapper,
            title="Enrichment ratio",
            formatter=NumeralTickFormatter(format="0.00"),
        ),
        "right",
    )
    plot.xgrid.grid_line_alpha = 0.2
    plot.ygrid.grid_line_color = "#E8EDF2"
    plot.ygrid.grid_line_alpha = 0.8
    plot.yaxis.major_label_text_font_size = "8pt"
    plot.yaxis.major_label_text_color = "#344854"
    plot.outline_line_color = "#D9E1E8"
    plot.tags.append(
        {
            "artifact_id": artifact.metadata.artifact_id,
            "artifact_kind": artifact.metadata.kind.value,
        }
    )
    return plot


__all__ = ["plot_reaction_set_enrichment"]

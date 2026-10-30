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
from bokeh.models import ColumnDataSource, HoverTool, Range1d

from ..analysis.reaction_direction import ReactionDirectionCriterion
from ..analysis.set_enrichment import (
    SetEnrichmentArtifact,
    SetEnrichmentDirection,
    SetEnrichmentFeatureType,
)
from ..managers.hover import panel as hover_panel
from ..visual_config import theme
from .ranking_layout import (
    RankedSubsystem,
    add_focus_link,
    names_panel,
    order_rows,
    ranking_figure,
    ranking_layout,
)
from .set_enrichment_overview import ora_family_label, ora_hover_facts, signed_log2_gradient_scale

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
    """A subsystem's hover, without its panel, which the tooltip template draws.

    The same facts as its node in the network (set_enrichment_overview), and the
    reaction direction criterion the reactions were read under.
    """
    reactions = artifact.feature_type is SetEnrichmentFeatureType.REACTION
    return hover_panel.row(
        "Subsystem",
        f"{escape(str(row['subsystem_name']))} <span style='color:#888;'>({escape(str(row['category']))})</span>",
    ) + hover_panel.row("Analysis", ora_family_label(artifact.feature_type)) + ora_hover_facts(
        row,
        target_significant=bool(row["target_significant"]),
        baseline_significant=bool(row["baseline_significant"]),
        feature_type=artifact.feature_type,
        threshold=artifact.config.adjusted_pvalue_threshold,
        criterion=_direction_criterion_label(artifact) if reactions else None,
    )


def set_enrichment_rows(artifact: SetEnrichmentArtifact) -> pd.DataFrame:
    """One row per subsystem as the ORA ranking draws it, and its hover.

    ``dominant_visual`` is where its dot sits, ``dominant_color`` its colour,
    ``significant`` whether either direction is, and ``hover_html`` its hover,
    both directions side by side; ``attrs["color_reference"]`` is the
    distribution the colours come from. Step 3 reads one comparison's ORA from it,
    so its rows say what step 4's do.
    """
    if not isinstance(artifact, SetEnrichmentArtifact):
        raise TypeError("artifact must be a SetEnrichmentArtifact")
    return _paired_frame(artifact)


def plot_set_enrichment_category(
    artifact: SetEnrichmentArtifact,
    *,
    category_colors: Mapping[str, str] | None = None,
    category_numbers: Mapping[str, int] | None = None,
    navigation_event: str | None = None,
):
    """One dot per subsystem, in the subsystem rankings' layout (see ranking_layout).

    ``category_colors`` is accepted for compatibility and unused: categories carry
    no hue, so the colour is left to the enrichment score. ``category_numbers`` are
    the network's numbers for every category in use, so the headings match it.
    Returns the names panel and the dots as one grid.
    """
    if not isinstance(artifact, SetEnrichmentArtifact):
        raise TypeError("artifact must be a SetEnrichmentArtifact")

    frame = _paired_frame(artifact)
    entries = [
        RankedSubsystem(
            subsystem_id=str(row["subsystem_id"]),
            name=str(row["subsystem_name"]),
            category=str(row["category"]),
            score=float(row["maximum_visual_magnitude"])
            if math.isfinite(float(row["maximum_visual_magnitude"]))
            else 0.0,
        )
        for _, row in frame.iterrows()
    ]
    rows = order_rows(entries, numbers=category_numbers)
    frame["y_factor"] = [rows.factor_of[str(sid)] for sid in frame.get("subsystem_id", [])]

    values = [
        value
        for value in frame.get("dominant_visual", pd.Series(dtype=float)).tolist()
        if math.isfinite(value)
    ]
    x_min = min([0.0, *values])
    x_max = max([0.0, *values])
    padding = (x_max - x_min) * 0.1 if x_max != x_min else 1.0
    x_range = Range1d(x_min - padding, x_max + padding)

    family = (
        "Reaction"
        if artifact.feature_type is SetEnrichmentFeatureType.REACTION
        else "Gene"
    )
    plot = ranking_figure(
        rows,
        title=f"{family}-Set Enrichment (ORA)",
        x_range=x_range,
        x_label="Signed log2(enrichment ratio), Target-lower to Target-higher",
        tools="xpan,xwheel_zoom,reset,save,tap",
    )
    add_focus_link(plot, rows, navigation_event=navigation_event)
    source = ColumnDataSource(frame)
    points = plot.scatter(
        x="dominant_visual",
        y="y_factor",
        source=source,
        size=10,
        marker="circle",
        fill_color="dominant_color",
        line_color=theme.MARK_EDGE,
        line_width=theme.MARK_EDGE_WIDTH,
        fill_alpha=0.95,
        name="set_enrichment_subsystem_points",
    )
    reference = frame.attrs.get("color_reference")
    # The colour reference sits in the subsystem's panel, as on a graph node's.
    hover_tooltip = hover_panel.panel(
        "@hover_html{safe}",
        reference.hover_template(marker_x_field="color_reference_marker_x", value_field="color_reference_value")
        if reference is not None
        else "",
    )
    plot.add_tools(
        HoverTool(
            renderers=[points],
            tooltips=hover_tooltip,
            attachment="vertical",
            show_arrow=False,
        )
    )
    names = names_panel(
        rows,
        plot,
        x_label="Signed log2(enrichment ratio), Target-lower to Target-higher",
        navigation_event=navigation_event,
    )
    layout = ranking_layout(rows, names, plot)
    layout.tags.append(
        {
            "artifact_id": artifact.metadata.artifact_id,
            "artifact_kind": artifact.metadata.kind.value,
            "feature_type": artifact.feature_type.value,
            "presentation": "paired_set_enrichment_category",
        }
    )
    return layout


__all__ = ["plot_set_enrichment_category", "set_enrichment_rows"]

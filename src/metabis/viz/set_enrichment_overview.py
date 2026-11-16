"""Subsystem-level overview colouring from a paired set-enrichment artifact.

When Set Enrichment is the chosen Step-4 subsystem summary, the metabolic
overview paints each subsystem glyph from the installed
:class:`~metabis.analysis.set_enrichment.SetEnrichmentArtifact`, exactly the way
delta score paints it -- through the same diverging red/white/blue distribution
machinery -- but with the delta score replaced by the subsystem's signed
``log2(enrichment ratio)``. Red is over-representation among Target-higher
features, blue among Target-lower, and the intensity grows with the magnitude.
The one and only difference from delta behaviour: a subsystem whose enrichment is
not significant is gray.

Each subsystem contributes one signed value. The paired kernel produces a
Target-higher visual (``>= 0``) and a Target-lower visual (``<= 0``); the
subsystem's single value is the larger-magnitude of the two (an exact tie
resolves to Target-higher, the positive side). Significance is either direction
being significant.

Process (category) glyphs are deliberately left untouched, and detailed
subsystem graphs keep their feature-level comparison colours, so nothing here
ever recolours a category, reaction, or gene node. The module imports no Bokeh,
so it runs inside the graph worker's enrichment phase (ADR-0006).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape
from typing import Any, Mapping

import pandas as pd

from ..analysis.set_enrichment import (
    SetEnrichmentArtifact,
    SetEnrichmentDirection,
    SetEnrichmentFeatureType,
)
from ..managers.color_distributions import ColorDistributionRegistry
from ..managers.hover import panel as hover_panel
from ..managers.workspace_mode import AnalysisWorkspaceMode
from ..visual_config import theme

# The metric key the overview signed-log2 distribution is registered under. It is
# distinct from any delta metric so the two subsystem distributions coexist.
SET_ENRICHMENT_SUBSYSTEM_METRIC = "set_enrichment_log2"

NOT_SIGNIFICANT_COLOR = theme.NOT_SIGNIFICANT


@dataclass(frozen=True)
class SubsystemEnrichmentDecision:
    """One subsystem's paired-direction significance and single signed value."""

    subsystem_id: str
    display_name: str
    feature_type: SetEnrichmentFeatureType
    target_significant: bool
    baseline_significant: bool
    target_visual: float
    baseline_visual: float
    signed_log2: float
    detail: Mapping[str, object]

    @property
    def significant(self) -> bool:
        return self.target_significant or self.baseline_significant


@dataclass(frozen=True)
class OverviewEnrichmentColor:
    """A subsystem's resolved overview glyph colour plus its decision."""

    decision: SubsystemEnrichmentDecision
    fill_color: str

    @property
    def significant(self) -> bool:
        return self.decision.significant


@dataclass(frozen=True)
class SetEnrichmentOverviewColors:
    """Precomputed overview colours for one installed set-enrichment family."""

    feature_type: SetEnrichmentFeatureType
    metric_name: str
    adjusted_pvalue_threshold: float
    correction: str
    minimum_measured_set_size: int
    subsystems: Mapping[str, OverviewEnrichmentColor]
    color_manager: Any = None

    @property
    def not_significant_color(self) -> str:
        """The gray used for both non-significant and ORA-ineligible subsystems."""
        return NOT_SIGNIFICANT_COLOR

    def for_subsystem(self, subsystem_id: object) -> OverviewEnrichmentColor | None:
        return self.subsystems.get(str(subsystem_id))

    def ineligible_hover_for(
        self, display_name: object, *, show_stats: bool = True
    ) -> str:
        """Hover for a subsystem the ORA never tested (below the minimum count).

        Under the Set-Enrichment summary such a subsystem is not
        over-representation eligible, so it reads as not-significant gray with a
        hover that explains why -- never a delta-score reading (D-59).
        """
        return _ineligible_hover_html(
            str(display_name),
            feature_type=self.feature_type,
            minimum_measured_set_size=self.minimum_measured_set_size,
            show_stats=show_stats,
        )

    def hover_for(
        self,
        subsystem_id: object,
        *,
        show_stats: bool = True,
        show_color_reference: bool = True,
    ) -> str:
        """The subsystem overview hover, with the log2(ER) color reference."""
        color = self.for_subsystem(subsystem_id)
        if color is None:
            return ""
        return _subsystem_hover_html(
            color.decision,
            feature_type=self.feature_type,
            color_manager=self.color_manager,
            metric_name=self.metric_name,
            show_stats=show_stats,
            show_color_reference=show_color_reference,
        )


def _finite(value: object, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _paired_rows(frame: pd.DataFrame) -> Mapping[str, dict[str, object]]:
    """Collapse the two directional rows per subsystem into one paired record."""
    higher = SetEnrichmentDirection.TARGET_HIGHER.value
    lower = SetEnrichmentDirection.TARGET_LOWER.value
    paired: dict[str, dict[str, object]] = {}
    for subsystem_id, rows in frame.groupby("subsystem_id", sort=False):
        indexed = {
            str(direction): record
            for direction, record in zip(rows["direction"], rows.to_dict("records"))
        }
        target = indexed.get(higher)
        baseline = indexed.get(lower)
        if target is None or baseline is None:
            # The kernel always emits both directions per eligible subsystem, so a
            # missing one means an incomplete table. Skip it rather than paint a
            # half-defined glyph.
            continue
        paired[str(subsystem_id)] = {"target": target, "baseline": baseline}
    return paired


def build_subsystem_decisions(
    artifact: SetEnrichmentArtifact,
) -> dict[str, SubsystemEnrichmentDecision]:
    """One paired decision per subsystem, independent of any colour registry."""
    if not isinstance(artifact, SetEnrichmentArtifact):
        raise TypeError("artifact must be a SetEnrichmentArtifact")
    frame = artifact.results.to_frame()
    decisions: dict[str, SubsystemEnrichmentDecision] = {}
    for subsystem_id, record in _paired_rows(frame).items():
        target = record["target"]
        baseline = record["baseline"]
        target_visual = max(0.0, _finite(target["signed_log2_enrichment_ratio"]))
        baseline_visual = min(0.0, _finite(baseline["signed_log2_enrichment_ratio"]))
        # One signed value per subsystem: the larger-magnitude direction, its sign
        # preserved. An exact tie resolves to the positive (Target-higher) side.
        signed_log2 = (
            baseline_visual
            if abs(baseline_visual) > abs(target_visual)
            else target_visual
        )
        decisions[subsystem_id] = SubsystemEnrichmentDecision(
            subsystem_id=subsystem_id,
            display_name=str(target["subsystem_name"]),
            feature_type=artifact.feature_type,
            target_significant=bool(target["is_significant"]),
            baseline_significant=bool(baseline["is_significant"]),
            target_visual=target_visual,
            baseline_visual=baseline_visual,
            signed_log2=signed_log2,
            detail=_subsystem_detail(target, baseline, artifact),
        )
    return decisions


def _subsystem_detail(
    target: Mapping[str, object],
    baseline: Mapping[str, object],
    artifact: SetEnrichmentArtifact,
) -> dict[str, object]:
    return {
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
        "adjusted_pvalue_threshold": float(artifact.config.adjusted_pvalue_threshold),
    }


def _register_overview_distribution(
    color_manager: Any,
    decisions: Mapping[str, SubsystemEnrichmentDecision],
    metric_name: str,
) -> None:
    """Publish the signed-log2 subsystem distribution, exactly as delta does."""
    if color_manager is None or not decisions:
        return
    values = pd.Series(
        {sid: decision.signed_log2 for sid, decision in decisions.items()},
        dtype=float,
    )
    color_manager.register_distribution(
        mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
        element_type="subsystem",
        data=values,
        metric_name=metric_name,
        center_method="zero",
        spread_method="std",
    )


def build_set_enrichment_overview_colors(
    artifact: SetEnrichmentArtifact,
    color_manager: Any = None,
) -> SetEnrichmentOverviewColors:
    """Build subsystem overview colours for one ORA artifact.

    ``color_manager`` is the worker's colour registry: the signed-log2
    distribution is registered into it so the glyph colour and the hover
    color-reference histogram read the exact same scale. A significant subsystem
    takes its gradient colour from that distribution; a non-significant one is
    gray.
    """
    decisions = build_subsystem_decisions(artifact)
    metric_name = SET_ENRICHMENT_SUBSYSTEM_METRIC
    _register_overview_distribution(color_manager, decisions, metric_name)

    subsystems: dict[str, OverviewEnrichmentColor] = {}
    for subsystem_id, decision in decisions.items():
        if not decision.significant:
            fill_color = NOT_SIGNIFICANT_COLOR
        elif color_manager is not None:
            fill_color = color_manager.get_element_color_from_distribution(
                mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
                element_type="subsystem",
                element_id=subsystem_id,
                metric_name=metric_name,
            )
        else:
            fill_color = theme.NEUTRAL
        subsystems[subsystem_id] = OverviewEnrichmentColor(
            decision=decision, fill_color=fill_color
        )

    return SetEnrichmentOverviewColors(
        feature_type=artifact.feature_type,
        metric_name=metric_name,
        adjusted_pvalue_threshold=float(artifact.config.adjusted_pvalue_threshold),
        correction=str(artifact.config.correction),
        minimum_measured_set_size=int(artifact.config.minimum_measured_set_size),
        subsystems=subsystems,
        color_manager=color_manager,
    )


def signed_log2_gradient_scale(values: pd.Series):
    """Diverging colours for signed log2 values, with their color reference.

    A standalone mirror of the overview subsystem distribution for callers that
    have no workspace colour registry (the Step-4 category dot plot). Same
    endpoints, centre, and binning as delta, so the two views read the same way.

    Returning the reference alongside the colours is what lets the category plot
    satisfy Supp. 1.5 (R-09): the panel is built from the very distribution that
    painted the dots, so the two cannot drift apart.

    Returns:
        ``(colors, reference)``. ``reference`` is ``None`` for an empty input or
        when the distribution produced no histogram.
    """
    # Call-time import: this module is imported by the graph worker, which must
    # not pull in the Bokeh-backed plotting helpers at import time (ADR-0006).
    from .color_reference import build_reference_from_distribution

    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.empty:
        return pd.Series(dtype=object), None
    registry = _comparison_registry()
    registry.register_distribution(
        mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
        element_type="subsystem",
        data=numeric,
        metric_name=SET_ENRICHMENT_SUBSYSTEM_METRIC,
        center_method="zero",
        spread_method="std",
    )
    colors = registry.get_element_colors_from_distribution(
        mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
        element_type="subsystem",
        metric_name=SET_ENRICHMENT_SUBSYSTEM_METRIC,
        element_ids=list(numeric.index),
    )
    reference = build_reference_from_distribution(
        registry,
        mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
        element_type="subsystem",
        element_id=numeric.index[0],
        metric_name=SET_ENRICHMENT_SUBSYSTEM_METRIC,
        metric_label="Signed log2(ER)",
    )
    return colors, reference


def signed_log2_gradient_colors(values: pd.Series) -> pd.Series:
    """The colours of :func:`signed_log2_gradient_scale`, without its reference."""
    colors, _reference = signed_log2_gradient_scale(values)
    return colors


def _comparison_registry() -> ColorDistributionRegistry:
    return ColorDistributionRegistry(
        configurations={
            AnalysisWorkspaceMode.GROUP_COMPARISON: {
                "up_color": theme.UP_REGULATED,
                "down_color": theme.DOWN_REGULATED,
                "neutral_color": theme.NEUTRAL,
                "significant_pval_color": theme.SIGNIFICANT_PVAL,
            }
        },
        current_mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
    )


def ora_family_label(feature_type: SetEnrichmentFeatureType) -> str:
    """The analysis a family's ORA is, as its hovers name it."""
    if feature_type is SetEnrichmentFeatureType.REACTION:
        return "Reaction-set over-representation analysis"
    return "Gene-set over-representation analysis"


CORRECTION_LABELS = {"fdr_bh": "Benjamini–Hochberg FDR", "holm": "Holm FWER"}


def ora_hover_facts(
    detail: Mapping[str, Any],
    *,
    target_significant: bool,
    baseline_significant: bool,
    feature_type: SetEnrichmentFeatureType,
    threshold: object,
    criterion: str | None = None,
) -> str:
    """A subsystem's two over-representation tests, side by side, and how they ran.

    ``detail`` holds each direction's ``target_``/``baseline_`` ratio, adjusted p,
    overlap and query, the subsystem's and background's sizes and the correction;
    the overview's network and its ORA ranking both show it in a hover panel.
    """
    noun = _feature_noun(feature_type)

    def side(title: str, prefix: str, significant: bool, accent: str) -> str:
        return hover_panel.column(
            title,
            hover_panel.significance(significant),
            hover_panel.row("Enrichment ratio", _format_number(detail[f"{prefix}_enrichment_ratio"])),
            hover_panel.row("Adjusted p-value", _format_probability(detail[f"{prefix}_adjusted_pval"])),
            hover_panel.row("Overlap", f"{detail[f'{prefix}_overlap_size']} of {detail['subsystem_size']} {noun}"),
            hover_panel.row("Query", f"{detail[f'{prefix}_query_size']} {noun}"),
            accent=accent if significant else "transparent",
        )

    correction = CORRECTION_LABELS.get(str(detail["correction"]), str(detail["correction"]))
    notes = [
        hover_panel.note(
            f"Against a background of {detail['universe_size']} {noun}, with {escape(correction)} "
            f"over {detail['correction_family_size']} hypotheses; significant at q ≤ "
            f"{_format_probability(threshold)}."
        )
    ]
    if criterion:
        notes.append(hover_panel.note(f"Reaction direction criterion: {escape(criterion)}."))
    return hover_panel.section(
        hover_panel.columns(
            side("Target-higher", "target", target_significant, theme.UP_REGULATED),
            side("Target-lower", "baseline", baseline_significant, theme.DOWN_REGULATED),
        ),
        *notes,
        title="Over-representation",
    )


def _feature_noun(feature_type: SetEnrichmentFeatureType) -> str:
    return "reactions" if feature_type is SetEnrichmentFeatureType.REACTION else "genes"


def _ineligible_hover_html(
    display_name: str,
    *,
    feature_type: SetEnrichmentFeatureType,
    minimum_measured_set_size: int,
    show_stats: bool,
) -> str:
    """Hover for an ORA-ineligible subsystem under the Set-Enrichment summary."""
    parts = [
        hover_panel.row("Subsystem", escape(display_name)),
        hover_panel.row("Analysis", ora_family_label(feature_type)),
    ]
    if show_stats:
        parts.append(
            hover_panel.section(
                hover_panel.note(
                    f"Not tested, as fewer than {minimum_measured_set_size} of its "
                    f"{_feature_noun(feature_type)} were measured; shown as not significant."
                )
            )
        )
    return hover_panel.panel(*parts)


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
        return "&lt; 2.23e-308"
    if number < 0.001:
        coefficient, exponent = f"{number:.2e}".split("e")
        return f"{coefficient}e{int(exponent)}"
    return f"{number:.3g}"


def _color_reference_html(
    decision: SubsystemEnrichmentDecision,
    color_manager: Any,
    metric_name: str,
) -> str:
    """The log2(ER) color-reference histogram, matching the delta hover section."""
    if color_manager is None:
        return ""
    # Call-time import: the histogram is a matplotlib PNG that lives beside Bokeh
    # plot builders, and hover generation runs on the graph worker, which must
    # import without Bokeh (ADR-0006, AM10).
    from .plot_histograms import create_color_reference_histogram

    try:
        details = color_manager.get_distribution_details(
            mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
            element_type="subsystem",
            element_id=decision.subsystem_id,
            metric_name=metric_name,
        )
        if not details.get("histogram"):
            return ""
        hist_html = create_color_reference_histogram(
            histogram_data=details["histogram"],
            current_value=details["element_value"],
            metric_name="log2(ER)",
            width=250,
            height=150,
            threshold=None,
            symmetric_threshold=True,
        )
    except Exception as exc:  # pragma: no cover - defensive, matches delta section
        return (
            "<div style='color:red;font-size:10px;'>"
            f"Color reference error: {exc}</div>"
        )
    return (
        "<div style='margin-top:8px;border-top:1px solid #ddd;padding-top:8px;'>"
        "<div><strong>Color Reference: signed log2(enrichment ratio)</strong></div>"
        "<div style='text-align:center;font-size:9px;font-weight:bold;color:#555;'>"
        "Used for coloring</div>"
        f"{hist_html}</div>"
    )


def _subsystem_hover_html(
    decision: SubsystemEnrichmentDecision,
    *,
    feature_type: SetEnrichmentFeatureType,
    color_manager: Any,
    metric_name: str,
    show_stats: bool,
    show_color_reference: bool,
) -> str:
    detail = decision.detail
    parts = [
        hover_panel.row("Subsystem", escape(decision.display_name)),
        hover_panel.row("Analysis", ora_family_label(feature_type)),
    ]
    if show_stats:
        parts.append(
            ora_hover_facts(
                detail,
                target_significant=decision.target_significant,
                baseline_significant=decision.baseline_significant,
                feature_type=feature_type,
                threshold=detail["adjusted_pvalue_threshold"],
            )
        )
    if show_color_reference:
        parts.append(_color_reference_html(decision, color_manager, metric_name))
    return hover_panel.panel(*parts)


__all__ = [
    "NOT_SIGNIFICANT_COLOR",
    "SET_ENRICHMENT_SUBSYSTEM_METRIC",
    "OverviewEnrichmentColor",
    "SetEnrichmentOverviewColors",
    "SubsystemEnrichmentDecision",
    "build_set_enrichment_overview_colors",
    "build_subsystem_decisions",
    "ora_family_label",
    "ora_hover_facts",
    "signed_log2_gradient_colors",
    "signed_log2_gradient_scale",
]

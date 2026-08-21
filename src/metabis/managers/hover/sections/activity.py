"""Hover section for single-group activity mode.

Shows what the selected group's color actually means and never shows comparison
statistics: in activity mode there is no second group, so a p-value, a Cohen's d
or a fold change would be undefined.
"""

from ...workspace_mode import AnalysisWorkspaceMode
from ..interfaces import HoverSection
from ..models import HoverContext, HoverData


def _format_number(value) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


class ActivitySection(HoverSection):
    """Section for normalized single-group activity summaries."""

    def should_include(self, data: HoverData, context: HoverContext) -> bool:
        if context.mode != AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS:
            return False
        if not context.show_stats:
            return False
        stats = data.get("experiment_stats", {})
        return bool(stats.get("has_experiment_data")) and bool(stats.get("is_activity"))

    def build(self, data: HoverData, context: HoverContext) -> str:
        stats = data["experiment_stats"]
        element_type = stats.get("element_type", "element")

        parts = [
            "<div style='margin-top:8px; border-top:1px solid #ddd; padding-top:8px;'>",
            "<div><strong>Single-group activity</strong></div>",
            (
                f"<div>Group: <strong>{stats.get('group_id')}</strong> "
                f"({stats.get('sample_count', 0)} samples)</div>"
            ),
        ]

        overlay_label = (
            "reaction activity scores"
            if stats.get("overlay") == "reaction"
            else "gene expression"
        )
        parts.append(f"<div>Overlay: {overlay_label}</div>")

        if element_type in ("subsystem", "subsystem_genes"):
            parts.extend(self._subsystem_body(data, stats))
        else:
            parts.extend(self._feature_body(stats))

        parts.append(
            "<div style='font-style:italic; margin-top:4px; color:#444;'>"
            f"{stats.get('summary_description', '')}; values retain the normalized "
            "input scale.</div>"
        )

        if not stats.get("is_colored", True):
            parts.append(
                "<div style='font-style:italic; margin-top:4px; color:#777;'>"
                "This element is not colored by the selected overlay.</div>"
            )

        parts.append(
            "<div style='font-style:italic; margin-top:4px; color:#777;'>"
            "Descriptive normalized summary derived from the uploaded input; not "
            "a significance result or measured physical flux.</div>"
        )
        parts.append("</div>")
        return "".join(parts)

    def _feature_body(self, stats) -> list[str]:
        parts = [
            "<div>Normalized group summary: "
            f"<strong>{_format_number(stats.get('activity_score'))}</strong></div>"
        ]

        if stats.get("is_non_finite"):
            parts.append(
                "<div style='font-size:11px; color:#a00;'>The selected group's "
                "summary is non-finite, so this feature is unavailable.</div>"
            )

        if stats.get("is_reversible"):
            shown = stats.get("main_direction", "forward")
            parts.append(
                "<div style='margin-top:4px;'><strong>Reversible reaction</strong> "
                f"(showing {shown})</div>"
            )
            parts.append(
                "<div style='font-size:11px; color:#555;'>forward "
                f"{_format_number(stats.get('forward_activity_score'))} &middot; "
                f"backward {_format_number(stats.get('backward_activity_score'))}</div>"
            )

        return parts

    def _subsystem_body(self, data: HoverData, stats) -> list[str]:
        display_name = data.get("subsystem_display_name") or stats.get("subsystem_id")
        score = stats.get("activity_score")
        coverage = stats.get("coverage", 0.0) or 0.0
        measured = stats.get("measured_features", 0)
        total = stats.get("total_model_features", 0)

        parts = [f"<div><strong>Subsystem:</strong> {display_name}</div>"]

        if score is None:
            parts.append(
                "<div style='color:#a00;'>No activity score: measured coverage "
                f"{coverage * 100:.0f}% is below the "
                f"{float(stats.get('minimum_coverage', 0)) * 100:.0f}% threshold.</div>"
            )
        else:
            parts.append(
                f"<div>Subsystem activity score: <strong>{_format_number(score)}</strong> "
                f"({stats.get('aggregation', 'median of measured member scores')})</div>"
            )

        parts.append(
            f"<div>Coverage: {measured} of {total} model features measured "
            f"({coverage * 100:.0f}%)</div>"
        )
        return parts

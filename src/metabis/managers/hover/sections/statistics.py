from html import escape
from typing import Any, Dict, List

from ....visual_config import theme
from ...workspace_mode import AnalysisWorkspaceMode
from ..interfaces import HoverSection
from ..models import HoverContext, HoverData


class StatisticsSection(HoverSection):
    """Section for statistical comparison information (mode-aware)."""

    def __init__(self, element_type: str = "element"):
        """Initialize with element type for appropriate labeling.

        Args:
            element_type: Type of element ("gene", "reaction", "metabolite", etc.)
        """
        self.element_type = element_type.lower()

    def should_include(self, data: HoverData, context: HoverContext) -> bool:
        """Include if stats are available and requested for GroupComparison mode."""
        mode = context.mode
        show_stats = context.show_stats

        # Only show for GROUP_COMPARISON mode
        if mode != AnalysisWorkspaceMode.GROUP_COMPARISON:
            return False

        has_stats = data.get("comparison_stats", {}).get("has_comparison_data", False)
        return show_stats and has_stats

    def _build_column(
        self, label: str, values: Dict[str, Any], is_main_direction: bool = False
    ) -> str:
        # Call-time import: `format_scientific` is a pure formatter, but it
        # lives in a Bokeh-importing plotting module, and hover generation runs
        # on the graph worker, which must import without Bokeh (ADR-0006, AM10).
        from ....viz.plot_utils import format_scientific

        # Define container style based on whether this is the main direction
        border_color = theme.ACTIVITY_UP if is_main_direction else "transparent"

        container_style = (
            f"flex:1; min-width:280px; border:1.5px solid {border_color}; "
            "border-radius:4px; padding:6px;"
        )

        column_parts: List[str] = [
            f"<div style='{container_style}'>",
            f"<div style='font-weight:bold; margin-bottom:4px;'>{label}</div>",
        ]

        pvalue = values.get("pval")
        if pvalue is not None:
            column_parts.append(f"<div>p-value: {format_scientific(pvalue)}</div>")

        adjusted_pval = values.get("adjusted_pval")
        if adjusted_pval is not None:
            column_parts.append(
                f"<div>Adjusted p-value: {format_scientific(adjusted_pval)}</div>"
            )

        log2_fc = values.get("log2_fc")
        if log2_fc is not None:
            column_parts.append(f"<div>Log2 FC: {log2_fc:.3f}</div>")

        cohens_d = values.get("cohens_d")
        if cohens_d is not None:
            column_parts.append(f"<div>Cohen's d: {cohens_d:.3f}</div>")

        mean_diff = values.get("mean_difference")
        if mean_diff is not None:
            column_parts.append(f"<div>Mean Δ: {mean_diff:.3f}</div>")

        if values.get("is_significant"):
            column_parts.append(
                "<div style='color:red; font-weight:bold;'>Significant</div>"
            )
        else:
            column_parts.append(
                "<div style='color:#aaaaaa; font-weight:bold;'>Not Significant</div>"
            )

        column_parts.append("</div>")
        return "".join(column_parts)

    def _more_active_suffix(
        self, stats: Dict[str, Any], context: HoverContext
    ) -> str:
        """Name the population whose means actually selected this direction.

        The deciding population is the Step-4 reaction-direction criterion's, not
        always Target. Under the default (Dominant in Baseline) this said "more
        active direction in Target <target group>" about a column chosen from the
        Baseline means -- a claim that is merely mislabelled when both populations
        agree, and false for the reactions whose dominance differs between them
        (WP10/D-70).
        """
        from ....analysis.reaction_direction import ReactionDirectionCriterion

        criterion = context.direction_criterion
        if criterion is ReactionDirectionCriterion.DOMINANT_IN_BASELINE:
            population, group_name = "Baseline", stats.get("group2", "Baseline")
        else:
            population, group_name = "Target", stats.get("group1", "Target")

        try:
            group_color = context.color_manager.get_group_color(group_name)
        except Exception:
            group_color = "black"

        return (
            f" (<span style='color:{theme.ACTIVITY_UP}; font-weight:bold;'>"
            f"more active direction</span> in {population} "
            f"<span style='color:{group_color};'>{escape(str(group_name))}</span>)"
        )

    def build(self, data: HoverData, context: HoverContext) -> str:
        """Build statistics section for GroupComparison mode."""
        stats = data["comparison_stats"]
        element_type = context.element_type or "element"

        parts = [
            "<div style='margin-top:8px; border-top:1px solid #ddd; padding-top:8px;'>",
            "<div><strong>Statistical Comparison:</strong></div>",
        ]

        if "group1" in stats and "group2" in stats:
            parts.append(
                f"<div>Baseline: {escape(str(stats['group2']))}; "
                f"Target: {escape(str(stats['group1']))}</div>"
            )
            parts.append("<div>Effect direction: Target &minus; Baseline</div>")

        # Check reversibility based on data presence
        is_reversible = data.get("is_reversible", False)

        label_column_1 = "Forward"
        if element_type == "gene":
            label_column_1 = ""

        # Add "more active" text if reversible and main direction is forward
        if is_reversible and context.main_direction == "forward":
            label_column_1 += self._more_active_suffix(stats, context)

        # A reaction can be measured in one direction only -- Compass blocks the
        # other, or the input supplies a single column -- so the forward column is
        # rendered only when it exists (WP10/D-64).
        forward_column = (
            self._build_column(
                label_column_1,
                stats["forward"],
                is_main_direction=(
                    is_reversible and context.main_direction == "forward"
                ),
            )
            if stats.get("forward")
            else None
        )

        # Check that 'backward' exists and is non-empty before using it
        if "backward" in stats and stats["backward"]:
            backward_label = "Reverse"

            # Add "more active" text if reversible and main direction is backward
            if is_reversible and context.main_direction == "backward":
                backward_label += self._more_active_suffix(stats, context)

            backward_column = self._build_column(
                backward_label,
                stats["backward"],
                is_main_direction=(
                    is_reversible and context.main_direction == "backward"
                ),
            )
            parts.append(
                "<div style='display:flex; justify-content:center; gap:12px; align-items:flex-start; margin-top:8px;'>"
            )
            parts.extend(
                column
                for column in (forward_column, backward_column)
                if column is not None
            )
            parts.append("</div>")
        elif forward_column is not None:
            parts.append(forward_column)

        parts.append("</div>")
        return "".join(parts)

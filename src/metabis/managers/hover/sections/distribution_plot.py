from typing import Any, Dict

from ....visual_config import theme
from ...workspace_mode import AnalysisWorkspaceMode
from ..interfaces import HoverSection
from ..models import HoverContext, HoverData


class DistributionPlotSection(HoverSection):
    """Section for distribution plots (violin plots)."""

    def __init__(self, element_type: str = "element"):
        """Initialize with element type for appropriate labeling.

        Args:
            element_type: Type of element ("gene", "reaction", "metabolite", etc.)
        """
        self.element_type = element_type.lower()


    def should_include(self, data: HoverData, context: HoverContext) -> bool:
        """Include when either direction carries comparison distributions.

        Two things this must not assume. The distributions dict nests the backward
        direction *inside* the forward one, so reading only the outer
        ``has_distributions`` asks about the forward direction alone -- and a
        reaction whose forward direction Compass reports as blocked has no forward
        column at all, which silently took the measured reverse direction's violin
        with it (WP10/D-65). And an activity artifact answers with a single-group
        shape this section cannot draw, so a comparison-mode hover that reads one
        is skipped rather than failed with a red error card.
        """
        distributions = data.get("distributions") or {}
        if distributions.get("is_activity"):
            return False
        backward = distributions.get("backward") or {}
        return context.show_plot and bool(
            distributions.get("has_distributions")
            or backward.get("has_distributions")
        )

    def build(self, data: HoverData, context: HoverContext) -> str:
        """Draw one violin panel per direction that actually has values.

        A reaction can arrive here with either direction missing: Compass reports
        no flux for a blocked direction, so preprocessing drops its column and the
        provider answers ``has_distributions: False`` for it. Neither direction is
        therefore the one that "must" exist -- the forward direction only looked
        privileged because the provider nests the backward one inside it. So the
        available directions are collected first and everything after is driven by
        that list (WP10/D-65).
        """
        # Call-time import: the violin is a matplotlib PNG, but it lives beside
        # Bokeh plot builders, and hover generation runs on the graph worker,
        # which must import without Bokeh (ADR-0006, AM10).
        from ....viz.plot_violins import create_static_violin_for_hover

        dist_data = data["distributions"]
        element_type = context.element_type or "element"
        back_dist_data = (
            dist_data.get("backward") or {} if element_type == "reaction" else {}
        )

        # Get comparison stats to check significance
        stats = data.get("comparison_stats", {})
        is_gc_mode = context.mode == AnalysisWorkspaceMode.GROUP_COMPARISON

        try:
            available = [
                (direction, distributions)
                for direction, distributions in (
                    ("forward", dist_data),
                    ("backward", back_dist_data),
                )
                if distributions.get("has_distributions")
            ]
            if not available:
                return ""

            y_label, _ = self._get_plot_labels(element_type, data)
            # Colours are resolved from a direction that has group labels, which
            # is not necessarily the forward one.
            group1_color, group2_color = self._get_plot_colors(available[0][1], context)

            # A shared axis only means something across two panels. Blocked
            # directions sit at the activity floor, so zeros are excluded -- and
            # when that leaves nothing, the axis is left to each plot rather than
            # taking min() of an empty sequence.
            min_y = max_y = None
            if len(available) > 1:
                plotted = [
                    value
                    for _, distributions in available
                    for value in (
                        list(distributions["group1_values"])
                        + list(distributions["group2_values"])
                    )
                    if value > 0
                ]
                if plotted:
                    min_y, max_y = min(plotted), max(plotted)

            panels = []
            for direction, distributions in available:
                significant = (
                    stats.get(direction, {}).get("is_significant", False)
                    if is_gc_mode
                    else True
                )
                left_color, right_color = group1_color, group2_color
                if is_gc_mode and not significant:
                    left_color = right_color = "#aaaaaa"

                plot_groups = self._ordered_plot_groups(
                    distributions,
                    left_color,
                    right_color,
                    baseline_first=is_gc_mode,
                )
                panels.append(
                    (
                        direction,
                        create_static_violin_for_hover(
                            plot_groups[0],
                            plot_groups[1],
                            plot_groups[2],
                            plot_groups[3],
                            x_label="Groups",
                            y_label=y_label,
                            plot_title="",
                            group1_name=plot_groups[4],
                            group2_name=plot_groups[5],
                            size=(300, 220),
                            min_y=min_y,
                            max_y=max_y,
                        ),
                    )
                )

            if len(panels) == 1:
                # One direction is comparable, so there is no other panel to
                # distinguish it from and no border to carry that meaning.
                container_style = "margin-top:8px; display:flex; justify-content:center;"
                return f"<div style='{container_style}'>{panels[0][1]}</div>"

            # Use a transparent border for the non-active panel to keep alignment.
            base_style = "flex:1; min-width:280px; border-radius:4px; padding:6px;"
            cells = []
            for direction, plot_html in panels:
                border_color = (
                    theme.ACTIVITY_UP
                    if context.main_direction == direction
                    else "transparent"
                )
                style = f"{base_style} border:1.5px solid {border_color};"
                cells.append(f"<div style='{style}'>{plot_html}</div>")

            return (
                "<div style='display:flex; justify-content:center; gap:12px; "
                "align-items:flex-start; margin-top:8px;'>"
                + "".join(cells)
                + "</div>"
            )

        except Exception as e:
            return (
                f"<div style='color: red; font-size: 10px;'>Plot error: {str(e)}</div>"
            )

    def _get_plot_labels(self, element_type: str, data: HoverData) -> tuple[str, str]:
        """Determine plot labels based on element type."""
        if element_type == "gene":
            y_label = "Expression"
            plot_title = f"{data.get('symbol', 'Gene')} Expression"
        elif element_type == "reaction":
            y_label = "Activity"
            plot_title = f"{data.get('name', 'Reaction')} Activity"
        else:
            y_label = "Value"
            plot_title = f"{element_type.title()} Distribution"
        return y_label, plot_title

    @staticmethod
    def _ordered_plot_groups(
        dist_data: Dict[str, Any],
        group1_color: str,
        group2_color: str,
        *,
        baseline_first: bool,
    ) -> tuple[list[float], list[float], str, str, str, str]:
        """Return values, colors, and labels in their visual left/right order.

        Comparison membership retains the scientific Group 1 = Target and
        Group 2 = Baseline contract. The comparison UI presents Baseline on the
        left, so only the plot presentation is reordered here.
        """
        if baseline_first:
            return (
                dist_data["group2_values"],
                dist_data["group1_values"],
                group2_color,
                group1_color,
                dist_data["group2_name"],
                dist_data["group1_name"],
            )
        return (
            dist_data["group1_values"],
            dist_data["group2_values"],
            group1_color,
            group2_color,
            dist_data["group1_name"],
            dist_data["group2_name"],
        )

    def _get_plot_colors(self, dist_data: Dict[str, Any], context: HoverContext) -> tuple[str, str]:
        """Determine group colors for the plot."""
        color_manager = context.color_manager

        # Default fallback colors
        group1_color = "#FF0000"
        group2_color = "#0000FF"

        if color_manager:
            group1_color = color_manager.up_color
            group2_color = color_manager.down_color

            # Try to get actual group colors if available
            if dist_data.get("group1_name") and dist_data.get("group2_name"):
                try:
                    group1_color = color_manager.get_group_color(dist_data["group1_name"])
                    group2_color = color_manager.get_group_color(dist_data["group2_name"])
                except (KeyError, AttributeError):
                    pass

        return group1_color, group2_color

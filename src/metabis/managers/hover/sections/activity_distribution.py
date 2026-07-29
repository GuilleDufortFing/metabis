"""Violin comparison for single-group activity hovers.

Comparison mode plots one violin per compared group. In activity mode there is
only one group, so the informative contrast is against the *other features*: how
does this reaction's activity compare with every reaction measured in this group,
and how consistent is it across the group's own samples?
"""

from ....visual_config import theme
from ...workspace_mode import AnalysisWorkspaceMode
from ..interfaces import HoverSection
from ..models import HoverContext, HoverData


class ActivityDistributionSection(HoverSection):
    """Reference-vs-feature violins, only under the full hover-detail preset."""

    def __init__(self, element_type: str = "reaction"):
        self.element_type = element_type.lower()

    def should_include(self, data: HoverData, context: HoverContext) -> bool:
        if context.mode != AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS:
            return False
        if not context.show_plot:
            return False
        distributions = data.get("distributions", {})
        return bool(
            distributions.get("has_distributions") and distributions.get("is_activity")
        )

    def build(self, data: HoverData, context: HoverContext) -> str:
        # Imported at call time, not module scope. The violin itself is a
        # matplotlib PNG and needs no Bokeh, but it lives in a module that also
        # holds Bokeh plot builders -- and hover generation runs on the graph
        # worker, which must import without Bokeh (ADR-0006, AM10). This is the
        # same transitive edge slice 2 found in `viz/click_behavior.py`: every
        # statically scanned module was clean and the property was still false.
        from ....viz.plot_violins import create_static_violin_for_hover

        distributions = data["distributions"]

        try:
            reference_values = distributions["reference_values"]
            element_values = distributions["element_values"]

            y_label = "Expression" if self.element_type == "gene" else "Activity"

            # Paint the feature's violin with the colour its glyph carries, so the
            # hover and the graph cannot disagree about how active it is.
            element_color = theme.ACTIVITY_MID
            color_manager = context.color_manager
            activity_artifact = getattr(
                context.data_provider, "activity_artifact", None
            )
            if color_manager and activity_artifact is not None:
                overlay = context.data_provider.activity_overlay_for_element(
                    self.element_type
                )
                element_color = color_manager.get_element_color_from_distribution(
                    mode=AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS,
                    element_type=self.element_type,
                    element_id=distributions["feature_id"],
                    metric_name=activity_artifact.metric_key_for(overlay),
                )

            all_values = list(reference_values) + list(element_values)
            min_y, max_y = min(all_values), max(all_values)

            plot_html = create_static_violin_for_hover(
                list(reference_values),
                list(element_values),
                theme.ACTIVITY_UNAVAILABLE,
                element_color,
                x_label="",
                y_label=y_label,
                plot_title="",
                group1_name=distributions["reference_name"],
                group2_name="This " + self.element_type,
                size=(340, 240),
                min_y=min_y,
                max_y=max_y,
            )

            reference_count = distributions.get("reference_count", len(reference_values))
            sample_count = distributions.get("sample_count", len(element_values))
            caption = (
                f"Left: group summary of all {reference_count} measured "
                f"{self.element_type}s in '{distributions['group_id']}'. "
                f"Right: this {self.element_type} across the group's {sample_count} "
                "samples."
            )

            return (
                "<div style='margin-top:8px; border-top:1px solid #ddd; padding-top:8px;'>"
                "<div><strong>Activity in context:</strong></div>"
                "<div style='display:flex; justify-content:center; margin-top:4px;'>"
                f"{plot_html}</div>"
                f"<div style='font-size:10px; color:#555; text-align:center;'>{caption}</div>"
                "</div>"
            )

        except Exception as exc:  # noqa: BLE001 - a hover must never break a render
            return (
                "<div style='color: red; font-size: 10px;'>"
                f"Activity plot error: {exc}</div>"
            )

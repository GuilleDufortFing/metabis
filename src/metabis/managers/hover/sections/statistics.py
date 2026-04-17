from typing import Any, Dict, List
from ..interfaces import HoverSection
from ..models import HoverContext, HoverData
from ...data_manager_mode import DataManagerMode
from ....viz.plot_utils import format_scientific
from ....visual_config import theme

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
        if mode != DataManagerMode.GROUP_COMPARISON:
            return False

        has_stats = data.get("comparison_stats", {}).get("has_comparison_data", False)
        return show_stats and has_stats

    def _build_column(self, label: str, values: Dict[str, Any], is_main_direction: bool = False) -> str:
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
            column_parts.append(
                f"<div>p-value: {format_scientific(pvalue)}</div>"
            )

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
            column_parts.append(
                f"<div>Mean Δ: {mean_diff:.3f}</div>"
            )

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

    def build(self, data: HoverData, context: HoverContext) -> str:
        """Build statistics section for GroupComparison mode."""
        stats = data["comparison_stats"]
        element_type = context.element_type or "element"

        parts = [
            "<div style='margin-top:8px; border-top:1px solid #ddd; padding-top:8px;'>",
            "<div><strong>Statistical Comparison:</strong></div>",
        ]

        if "group1" in stats and "group2" in stats:
            parts.append(f"<div>Groups: {stats['group1']} vs {stats['group2']}</div>")
        
        # Check reversibility based on data presence
        is_reversible = data.get("is_reversible", False)

        label_column_1 = "Forward"
        if element_type == "gene":
            label_column_1 = ""
        
        # Add "more active" text if reversible and main direction is forward
        if is_reversible and context.main_direction == "forward":
            group1_name = stats.get('group1', 'Group 1')
            # Try to get group color, fallback to black if not found
            try:
                group1_color = context.color_manager.get_group_color(group1_name)
            except:
                group1_color = "black"
                
            label_column_1 += (
                f" (<span style='color:{theme.ACTIVITY_UP}; font-weight:bold;'>more active direction</span> "
                f"in group <span style='color:{group1_color};'>{group1_name}</span>)"
            )

        forward_column = self._build_column(
            label_column_1, 
            stats['forward'], 
            is_main_direction=(is_reversible and context.main_direction == "forward")
        )

        # Check that 'backward' exists and is non-empty before using it
        if 'backward' in stats and stats['backward']:
            backward_label = "Reverse"
            
            # Add "more active" text if reversible and main direction is backward
            if is_reversible and context.main_direction == "backward":
                group1_name = stats.get('group1', 'Group 1')
                try:
                    group1_color = context.color_manager.get_group_color(group1_name)
                except:
                    group1_color = "black"
                    
                backward_label += (
                    f" (<span style='color:{theme.ACTIVITY_UP}; font-weight:bold;'>more active direction</span> "
                    f"in group <span style='color:{group1_color};'>{group1_name}</span>)"
                )

            backward_column = self._build_column(
                backward_label, 
                stats['backward'], 
                is_main_direction=(is_reversible and context.main_direction == "backward")
            )
            parts.append(
                "<div style='display:flex; justify-content:center; gap:12px; align-items:flex-start; margin-top:8px;'>"
            )
            parts.extend([forward_column, backward_column])
            parts.append("</div>")
        else:
            parts.append(forward_column)

        parts.append("</div>")
        return "".join(parts)

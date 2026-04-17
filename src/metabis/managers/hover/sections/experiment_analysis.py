from typing import Any, Dict
from ..interfaces import HoverSection
from ..models import HoverContext, HoverData
from ...data_manager_mode import DataManagerMode

class ExperimentAnalysisSection(HoverSection):
    """Section for within-group experiment analysis statistics."""

    def should_include(self, data: HoverData, context: HoverContext) -> bool:
        """Include if this is experiment analysis mode and stats are available."""
        mode = context.mode
        show_stats = context.show_stats
        has_stats = data.get("experiment_stats", {}).get("has_experiment_data", False)

        return mode == DataManagerMode.EXPERIMENT_ANALYSIS and show_stats and has_stats

    def build(self, data: HoverData, context: HoverContext) -> str:
        """Build experiment analysis section."""

        stats = data["experiment_stats"]
        
        parts = []
        parts.append(
            "<div style='margin-top:8px; border-top:1px solid #ddd; padding-top:8px;'>"
        )
        parts.append("<div><strong>Within-Group Analysis:</strong></div>")

        # Add group information
        if "group_id" in stats:
            parts.append(f"<div>Group: {stats['group_id']}</div>")

        # Add z-score information
        if "z_score" in stats:
            z_score = stats["z_score"]
            if abs(z_score) > 2.0:
                significance = (
                    "High relative activity"
                    # "High relative activity" if z_score > 0 else "Low relative activity"
                )

                parts.append(
                    f"<div><strong>Z-score: {abs(z_score):.2f} "
                    f"({significance})</strong></div>"
                )
            else:
                parts.append(f"<div>Z-score: {abs(z_score):.2f} (Moderate activity)</div>")

        # Add average activity
        if "average_activity" in stats:
            avg_activity = stats["average_activity"]
            parts.append(f"<div>Average activity: {avg_activity:.3f}</div>")

        # Add interpretation
        z_score = stats.get("z_score", 0.0)
        element_type = stats.get("element_type", "element")
        group_id = stats.get("group_id", "group")

        z_score_abs = abs(z_score)

        if z_score_abs > 2.0:
            interpretation = (
                f"This {element_type} is highly active relative to other "
                f"{element_type}s in {group_id}"
            )
        else:
            interpretation = (
                f"This {element_type} has moderate activity relative to other "
                f"{element_type}s in {group_id}"
            )

        parts.append(
            f"<div style='font-style: italic; margin-top: 4px;'>{interpretation}</div>"
        )

        parts.append("</div>")
        return "".join(parts)

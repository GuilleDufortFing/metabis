from typing import Any, Dict
from ..interfaces import HoverSection
from ..models import HoverContext, HoverData
from ...data_manager_mode import DataManagerMode

class NoExperimentSection(HoverSection):
    """Section for basic metadata when no experimental data is available."""

    def should_include(self, data: HoverData, context: HoverContext) -> bool:
        """Include if this is no-experiment mode."""
        mode = context.mode

        has_stats = data.get("comparison_stats", {}).get("has_comparison_data", False)
        has_exp_stats = data.get("experiment_stats", {}).get("has_experiment_data", False)

        return mode == DataManagerMode.NO_EXPERIMENT and (has_stats or has_exp_stats)

    def build(self, data: HoverData, context: HoverContext) -> str:
        """Build no-experiment section with basic metadata only."""
        parts = []
        parts.append(
            "<div style='margin-top:8px; border-top:1px solid #ddd; padding-top:8px;'>"
        )
        parts.append("<div><em>Metadata-only view (no experimental data)</em></div>")
        parts.append("</div>")
        return "".join(parts)

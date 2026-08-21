from typing import Any, Dict, List

from ..interfaces import HoverSection
from ..models import HoverContext, HoverData


class InfluenceSection(HoverSection):
    """Section for gene influence information on reactions."""

    def should_include(self, data: HoverData, context: HoverContext) -> bool:
        """Include if influence data is available and requested."""
        show_influence = context.show_influence
        has_influence = data.get("influence", {}).get("has_influence", False)
        return show_influence and has_influence

    def build(self, data: HoverData, context: HoverContext) -> str:
        """Build gene influence section."""
        influence_data = data["influence"]
        if not influence_data.get("top_genes"):
            return ""

        parts = []
        parts.append(
            "<div style='margin-top:8px; border-top:1px solid #ddd; padding-top:8px;'>"
        )
        parts.append("<div><strong>Top Influencing Genes:</strong></div>")
        # Get color manager from context
        color_manager = context.color_manager
        if color_manager:
            bar_chart_html = self._build_influence_bar_chart(
                influence_data["top_genes"],
                influence_data.get("max_influence", 100.0),
                color_manager,
            )
            parts.append(bar_chart_html)
        else:
            # Fallback to simple text list
            for gene_info in influence_data["top_genes"][:5]:
                symbol = gene_info.get("gene_symbol", "Unknown")
                correlation = gene_info.get("correlation", 0)
                pct_variance = gene_info.get("pct_variance", 0)
                parts.append(
                    f"<div style='font-size: 10px;'>{symbol}: {correlation:+.2f} ({pct_variance:.1f}%)</div>"
                )

        parts.append("</div>")
        return "".join(parts)

    def _build_influence_bar_chart(
        self,
        genes_data: List[Dict[str, Any]],
        max_influence: float,
        color_manager,
        max_genes_to_show: int = 5,
        bar_height: int = 15,
    ) -> str:
        """Build HTML for influence bar chart."""
        if not genes_data:
            return ""

        html_parts = []
        html_parts.append("<div style='margin-top: 4px;'>")

        for gene_info in genes_data[:max_genes_to_show]:
            gene_symbol = gene_info.get("gene_symbol", "Unknown")
            correlation = gene_info.get("correlation", 0)
            pct_variance = gene_info.get("pct_variance", 0)

            # Calculate bar width (percentage of container)
            bar_width = (pct_variance / max_influence) * 100 if max_influence > 0 else 0
            bar_width = min(100, max(5, bar_width))  # Ensure reasonable width

            # Get color based on correlation and influence
            color = color_manager.get_influence_color(
                correlation, pct_variance, max_influence
            )

            # Format the display values
            correlation_text = f"{correlation:+.2f}" if correlation != 0 else "0.00"
            pct_text = f"{pct_variance:.1f}%"

            html_parts.append(
                f"<div style='margin: 2px 0; font-size: 10px;'>"
                f"<div style='display: inline-block; width: 60px; text-align: right; margin-right: 4px;'>"
                f"{gene_symbol}:</div>"
                f"<div style='display: inline-block; width: {bar_width}%; height: {bar_height}px; "
                f"background-color: {color}; margin-right: 4px; vertical-align: middle;'></div>"
                f"<span style='font-size: 9px; color: #666;'>{correlation_text} ({pct_text})</span>"
                f"</div>"
            )

        html_parts.append("</div>")
        return "".join(html_parts)

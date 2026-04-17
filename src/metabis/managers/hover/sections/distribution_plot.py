from typing import Any, Dict, Optional
from ..interfaces import HoverSection
from ..models import HoverContext, HoverData
from ....viz.plot_violins import create_static_violin_for_hover
from ...data_manager_mode import DataManagerMode
from ....visual_config import theme

class DistributionPlotSection(HoverSection):
    """Section for distribution plots (violin plots)."""

    def __init__(self, element_type: str = "element"):
        """Initialize with element type for appropriate labeling.

        Args:
            element_type: Type of element ("gene", "reaction", "metabolite", etc.)
        """
        self.element_type = element_type.lower()


    def should_include(self, data: HoverData, context: HoverContext) -> bool:
        """Include if plot is requested and distribution data is available."""
        show_plot = context.show_plot
        has_distributions = data.get("distributions", {}).get(
            "has_distributions", False
        )
        return show_plot and has_distributions

    def build(self, data: HoverData, context: HoverContext) -> str:
        """Build distribution plot section with optional color reference histograms."""
        dist_data = data["distributions"]
        element_type = context.element_type or "element"
        back_dist_data = data.get("distributions", {}).get("backward", {}) if element_type == "reaction" else {}
        
        # Get comparison stats to check significance
        stats = data.get("comparison_stats", {})
        is_gc_mode = context.mode == DataManagerMode.GROUP_COMPARISON

        try:
            y_label, plot_title = self._get_plot_labels(element_type, data)
            group1_color, group2_color = self._get_plot_colors(dist_data, context)

            if not back_dist_data.get("has_distributions", False):
                # Determine if significant (forward direction)
                is_significant = True
                if is_gc_mode:
                    # For genes, stats are directly in 'forward' or root depending on structure
                    # Based on statistics.py, it seems stats has 'forward' key for reactions/genes
                    # Let's check the structure. In statistics.py: stats['forward']
                    # But for genes, statistics.py handles it.
                    # Let's look at data_provider.py get_gene_group_comparison_data
                    # It returns dict with 'forward' key.
                    # So we can safely access stats.get('forward', {}).get('is_significant')
                    is_significant = stats.get("forward", {}).get("is_significant", False)
                
                # Override colors if not significant
                if is_gc_mode and not is_significant:
                    group1_color = "#aaaaaa"
                    group2_color = "#aaaaaa"

                # Create the violin plot
                plot_html = create_static_violin_for_hover(
                    dist_data["group1_values"],
                    dist_data["group2_values"],
                    group1_color,
                    group2_color,
                    x_label="Groups",
                    y_label=y_label,
                    plot_title="",
                    group1_name=dist_data["group1_name"],
                    group2_name=dist_data["group2_name"],
                    size=(300, 220)
                )
                
                # Add border if main direction
                container_style = "margin-top:8px; display:flex; justify-content:center;"
                # Removed border highlighting for unidirectional plots as requested

                return f"<div style='{container_style}'>{plot_html}</div>"
        
            else:
                # Create the violin plot
                # Calculate min_y and max_y from both forward and backward distributions
                
                all_values = (
                    dist_data["group1_values"] + 
                    dist_data["group2_values"] + back_dist_data["group1_values"] +
                    back_dist_data["group2_values"]
                )
                # Remove 0 values to avoid skewing
                all_values = [v for v in all_values if v > 0]

                min_y = min(all_values)
                max_y = max(all_values)

                # Forward Plot
                fwd_significant = True
                if is_gc_mode:
                    fwd_significant = stats.get("forward", {}).get("is_significant", False)
                
                fwd_g1_color = group1_color
                fwd_g2_color = group2_color
                if is_gc_mode and not fwd_significant:
                    fwd_g1_color = "#aaaaaa"
                    fwd_g2_color = "#aaaaaa"

                plot_html = create_static_violin_for_hover(
                    dist_data["group1_values"],
                    dist_data["group2_values"],
                    fwd_g1_color,
                    fwd_g2_color,
                    x_label="Groups",
                    y_label=y_label,
                    plot_title="",
                    group1_name=dist_data["group1_name"],
                    group2_name=dist_data["group2_name"],
                    size=(300, 220),
                    min_y=min_y,
                    max_y=max_y
                )
                
                # Backward Plot
                bwd_significant = True
                if is_gc_mode:
                    bwd_significant = stats.get("backward", {}).get("is_significant", False)
                
                bwd_g1_color = group1_color
                bwd_g2_color = group2_color
                if is_gc_mode and not bwd_significant:
                    bwd_g1_color = "#aaaaaa"
                    bwd_g2_color = "#aaaaaa"
    
                backward_plot_html = create_static_violin_for_hover(
                    back_dist_data["group1_values"],
                    back_dist_data["group2_values"],
                    bwd_g1_color,
                    bwd_g2_color,
                    x_label="Groups",
                    y_label=y_label,
                    plot_title="",
                    group1_name=back_dist_data["group1_name"],
                    group2_name=back_dist_data["group2_name"],
                    size=(300, 220),
                    min_y=min_y,
                    max_y=max_y
                )
                
                # Styles for containers
                # Use transparent border for non-active to maintain alignment
                base_style = "flex:1; min-width:280px; border-radius:4px; padding:6px;"
                
                fwd_border_color = theme.ACTIVITY_UP if context.main_direction == "forward" else "transparent"
                fwd_style = f"{base_style} border:1.5px solid {fwd_border_color};"
                
                bwd_border_color = theme.ACTIVITY_UP if context.main_direction == "backward" else "transparent"
                bwd_style = f"{base_style} border:1.5px solid {bwd_border_color};"

                combined_html = (
                    "<div style='display:flex; justify-content:center; gap:12px; align-items:flex-start; margin-top:8px;'>"
                    f"<div style='{fwd_style}'>{plot_html}</div>"
                    f"<div style='{bwd_style}'>{backward_plot_html}</div>"
                    "</div>"
                )
    
                return combined_html

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


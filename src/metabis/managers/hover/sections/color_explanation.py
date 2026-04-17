from typing import Any, Dict
import numpy as np
from ..interfaces import HoverSection
from ..models import HoverContext, HoverData
from ...data_manager_mode import DataManagerMode
from ....viz.plot_histograms import create_color_reference_histogram
from ....visual_config import theme

class ColorExplanationSection(HoverSection):
    """Unified color explanation across modes and element types with histogram."""

    def __init__(self):
        self._background_cache = {}

    def should_include(self, data: HoverData, context: HoverContext) -> bool:
        
        # Always try to include if toggled on; fallback gracefully when no data
        show_color_reference = context.show_color_reference
        # Require a color manager to render colors

        mode = context.mode

        if context.mode == DataManagerMode.EXPERIMENT_ANALYSIS:
            has_stats = data.get("experiment_stats", {}).get("has_experiment_data", False)
        elif context.mode == DataManagerMode.GROUP_COMPARISON:
            has_stats = data.get("comparison_stats", {}).get("has_comparison_data", False)
            if not has_stats and context.hover_type == "subsystem":
                has_stats = data.get("is_in_model", False)
        else:
            has_stats = False

        return bool(show_color_reference and context.color_manager and has_stats)

    def build(self, data: HoverData, context: HoverContext) -> str:
        try:
            data_provider = context.data_provider
            color_manager = context.color_manager
            if not data_provider or not color_manager:
                return ""

            # Determine element ID and Type
            element_type = context.element_type
            element_id = None
            
            if element_type == "reaction":
                element_id = data.get("reaction_id")
            elif element_type == "gene":
                element_id = data.get("gene_id")
            elif "subsystem" in element_type: # subsystem or subsystem_genes
                element_id = data.get("subsystem_name")
            
            if not element_id:
                return ""

            # Determine metrics to show
            metrics_to_show = []
            
            if context.mode == DataManagerMode.GROUP_COMPARISON:
                if element_type in ["reaction", "gene"]:
                    metrics_to_show.append("neg_log_pval")
                    metrics_to_show.append(context.color_metric or "log2_fc")
                elif "subsystem" in element_type:
                    # Try to get metric from delta meta
                    metric = data.get("delta_subsystem_meta", {}).get("metric")
                    if not metric:
                        metric = context.color_metric or "default"
                    metrics_to_show.append(metric)
            
            elif context.mode == DataManagerMode.EXPERIMENT_ANALYSIS:
                metric = context.color_metric
                if not metric or metric == "default":
                    # Try to get active group
                    try:
                        metric = data_provider.get_active_group()
                    except Exception:
                        pass
                if metric:
                    metrics_to_show.append(metric)

            if not metrics_to_show:
                return ""

            # Build histograms
            histograms = []
            
            # Get thresholds for p-val if needed
            thresholds = {}
            if context.mode == DataManagerMode.GROUP_COMPARISON and data_provider.group_comparison:
                for_genes = (element_type == "gene")
                thresholds = data_provider.group_comparison.get_thresholds(for_genes=for_genes)

            # Adjust element_type for ColorManager (e.g. subsystem_genes)
            cm_element_type = element_type
            if element_type == "subsystem" and context.for_genes:
                cm_element_type = "subsystem_genes"

            for metric in metrics_to_show:
                hist_html = ""
                extra_text = ""
                
                details = color_manager.get_distribution_details(
                    mode=context.mode,
                    element_type=cm_element_type,
                    element_id=element_id,
                    metric_name=metric
                )
                
                if not details["histogram"]:
                    continue
                
                symmetric_threshold = True
                if "pval" in metric.lower():
                    symmetric_threshold = False

                # Use "Delta S" as label for subsystem histograms
                display_metric_name = metric
                if "subsystem" in element_type:
                    display_metric_name = "ΔS"

                hist_html = create_color_reference_histogram(
                    histogram_data=details["histogram"],
                    current_value=details["element_value"],
                    metric_name=display_metric_name,
                    width=250,
                    height=150,
                    threshold=thresholds.get(metric),
                    symmetric_threshold=symmetric_threshold,
                )

                # Add "Used for coloring" text if applicable
                is_coloring_metric = (metric == context.color_metric) or (len(metrics_to_show) == 1)
                # For subsystems, the metric IS the color metric usually
                if "subsystem" in element_type:
                     is_coloring_metric = True

                if is_coloring_metric:
                    extra_text = (
                        "<div style='text-align:center; font-size:9px; "
                        "font-weight:bold; color:#555; margin-top:-5px; margin-bottom:4px;'>"
                        "Used for coloring</div>"
                    )
                else:
                    extra_text = "<div style='margin-bottom:4px;'></div>"
                
                histograms.append(f"<div>{hist_html}{extra_text}</div>")
            
            if not histograms:
                return ""

            # Container style
            border_color = "transparent"
            if context.mode == DataManagerMode.GROUP_COMPARISON:
                 stats = data.get("comparison_stats", {})
                 if stats.get("backward"): # is_reversible
                     border_color = theme.ACTIVITY_UP

            container_style = (
                f"flex:1; min-width:280px; border:1.5px solid {border_color}; "
                "border-radius:4px; padding:6px; display:flex; flex-direction:row; "
                "gap:4px; justify-content:center;"
            )
            
            panel_html = f"<div style='{container_style}'>{''.join(histograms)}</div>"

            return (
                "<div style='margin-top:8px; border-top:1px solid #ddd; padding-top:8px;'>"
                "<div><strong>Color Reference:</strong></div>"
                "<div style='display:flex; justify-content:center; gap:12px; "
                f"align-items:flex-start; margin-top:8px;'>{panel_html}</div>"
                "</div>"
            )

        except Exception as e:
            return (
                f"<div style='color: red; font-size: 10px;'>"
                f"Color reference error: {str(e)}</div>"
            )

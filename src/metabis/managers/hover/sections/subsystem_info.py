from typing import Any, Dict
from ..interfaces import HoverSection
from ..models import HoverContext, HoverData
from ...data_manager_mode import DataManagerMode

class SubsystemSection(HoverSection):
    """Section for subsystem/pathway information."""

    def should_include(self, data: HoverData, context: HoverContext) -> bool:
        """Include if this is a subsystem hover request."""
        # Hide this GC-oriented section in Experiment Analysis mode
        if context.mode == DataManagerMode.EXPERIMENT_ANALYSIS:
            return False
        return context.hover_type == "subsystem"

    def build(self, data: HoverData, context: HoverContext) -> str:
        """Build subsystem information section."""
        subsystem_name = data.get("subsystem_display_name", "Unknown")
        n_significant = data.get("n_significant", 0)
        n_total = data.get("n_total", 0)
        regulation_score = data.get("regulation_score", 0)
        total_metric_value = data.get("total_metric_value", 0)
        metric = context.subsystem_metric or "log2_fc"
        for_genes = context.for_genes

        # Set terminology based on element type
        category_term = "Pathway" if for_genes else "Subsystem"
        element_term = "Genes" if for_genes else "Reactions"

        # Calculate percentage of significant elements
        percentage = (n_significant / n_total) * 100 if n_total > 0 else 0

        # Get display name for the metric
        metric_display_names = {
            "cohens_d": "Cohen's d",
            "log2_fc": "Log2 Fold Change",
            "mean_difference": "Mean Difference",
        }
        display_name = metric_display_names.get(metric, metric)

        # Format regulation direction text
        if regulation_score > 0:
            regulation_direction = "Up-regulated"
        elif regulation_score < 0:
            regulation_direction = "Down-regulated"
        else:
            regulation_direction = "Mixed regulation"

        parts = []
        parts.append(f"<div><strong>{category_term}:</strong> {subsystem_name}</div>")
        parts.append(
            f"<div><strong>Significant {element_term}:</strong> "
            f"{n_significant} out of {n_total} ({percentage:.1f}%)</div>"
        )
        parts.append(
            f"<div><strong>Regulation Direction:</strong> "
            f"{abs(regulation_score)*100:.1f}% of significant reactions are {regulation_direction}</div>"
        )
        
        # Display delta subsystem score with metadata
        delta_score = data.get("delta_subsystem_score")
        delta_meta = data.get("delta_subsystem_meta", {})
        
        if delta_score is not None and delta_meta.get("computed", False):
            # Get delta configuration details
            delta_metric = delta_meta.get("metric", metric)
            transform = delta_meta.get("transform", "sign")
            numerator_filter = delta_meta.get("numerator_filter", "all")
            denominator_filter = delta_meta.get("denominator_filter", "all")
            
            # Short metric name for equation
            metric_short_names = {
                "cohens_d": "Cohen's d",
                "log2_fc": "Log Fold Change",
                "mean_difference": "Mean dif.",
                "pval": "p",
                "neg_log_pval": "-log(p)",
                "neg_log_adj_pval": "-log(padj)"
            }
            m_sym = metric_short_names.get(delta_metric, delta_metric)
            
            # Transform notation
            if transform == "sign":
                term = f"sgn({m_sym})"
            elif transform == "abs":
                term = f"|{m_sym}|"
            else:
                term = f"({m_sym})"
                
            # Summation notation
            if numerator_filter == "significant":
                numerator_str = f"Σ_{{sig}} {term}"
            else:
                numerator_str = f"Σ {term}"
                
            # Denominator notation
            if denominator_filter == "significant":
                denom_str = "N_sig"
            else:
                denom_str = "N_total"
                
            equation = f"{numerator_str} / {denom_str}"
            
            parts.append(
                f"<div style='margin-top:4px;'><strong>Delta Score (ΔS):</strong></div>"
            )
            parts.append(
                f"<div style='font-size:11px; color:#333; margin-left:10px;'>"
                f"ΔS = {equation} = <strong>{delta_score:.3f}</strong></div>"
            )
        else:
            # Fallback: show the total metric value if delta is not computed
            parts.append(
                f"<div><strong>Total {display_name}:</strong> "
                f"{total_metric_value:.3f}</div>"
            )

        return "".join(parts)

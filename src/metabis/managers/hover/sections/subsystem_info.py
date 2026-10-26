from ...workspace_mode import AnalysisWorkspaceMode
from ..interfaces import HoverSection
from ..models import HoverContext, HoverData


class SubsystemSection(HoverSection):
    """Section for subsystem/pathway information."""

    def should_include(self, data: HoverData, context: HoverContext) -> bool:
        """Include if this is a subsystem hover request."""
        # Hide this GC-oriented section in Experiment Analysis mode
        if context.mode == AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS:
            return False
        return context.hover_type == "subsystem"

    def build(self, data: HoverData, context: HoverContext) -> str:
        """Build subsystem information section."""
        subsystem_name = data.get("subsystem_display_name", "Unknown")
        n_significant = data.get("n_significant", 0)
        n_total = data.get("n_total", 0)
        net_direction_score = data.get("net_direction_score", 0)
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

        # State the net direction as Target minus Baseline. The score is the signed
        # sum of the metric over the significant elements divided by its absolute
        # sum, so it is a net share of effect magnitude, not a count of elements.
        share = f"{abs(net_direction_score) * 100:.1f}%"
        if net_direction_score > 0:
            net_direction = f"{share} net higher in Target"
        elif net_direction_score < 0:
            net_direction = f"{share} net lower in Target (higher in Baseline)"
        else:
            net_direction = "balanced between both directions"

        parts = []
        parts.append(f"<div><strong>{category_term}:</strong> {subsystem_name}</div>")
        parts.append(
            f"<div><strong>Significant {element_term}:</strong> "
            f"{n_significant} out of {n_total} ({percentage:.1f}%)</div>"
        )
        parts.append(
            f"<div><strong>Net direction:</strong> "
            f"significant {element_term.lower()} are {net_direction}</div>"
        )

        # Display delta subsystem score with metadata
        delta_score = data.get("delta_subsystem_score")
        delta_meta = data.get("delta_subsystem_meta", {})

        if delta_score is not None and delta_meta.get("computed", False):
            equation = delta_equation(delta_meta, metric)

            parts.append(
                "<div style='margin-top:4px;'><strong>Delta Score (ΔS):</strong></div>"
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


def delta_equation(delta_meta: dict, metric: str = "log2_fc") -> str:
    """How a Delta score was computed, as a hover writes it: ``Σ_{sig} sgn(Cohen's d) / N_total``."""
    delta_metric = delta_meta.get("metric", metric)
    transform = delta_meta.get("transform", "sign")
    metric_short_names = {
        "cohens_d": "Cohen's d",
        "log2_fc": "Log Fold Change",
        "mean_difference": "Mean dif.",
        "pval": "p",
        "neg_log_pval": "-log(p)",
        "neg_log_adj_pval": "-log(padj)",
    }
    m_sym = metric_short_names.get(delta_metric, delta_metric)
    if transform == "sign":
        term = f"sgn({m_sym})"
    elif transform == "abs":
        term = f"|{m_sym}|"
    else:
        term = f"({m_sym})"
    numerator = f"Σ_{{sig}} {term}" if delta_meta.get("numerator_filter", "all") == "significant" else f"Σ {term}"
    denominator = "N_sig" if delta_meta.get("denominator_filter", "all") == "significant" else "N_total"
    return f"{numerator} / {denominator}"

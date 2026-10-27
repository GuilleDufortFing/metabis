"""Effect against effect: a reaction against its genes, or one result against another.

Points take the Okabe-Ito colours (Wong, Nat Methods 2011) the group palette
uses, so each class stays distinct under the common colour-vision deficiencies;
features that change in neither are the light grey of non-significant marks.
"""

from __future__ import annotations

from html import escape
from typing import Mapping

import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource, HoverTool, Legend, LegendItem, Span
from bokeh.plotting import figure

from ..analysis.agreement import AGREEMENT_CLASSES, Agreement
from ..analysis.concordance import CONCORDANCE_CLASSES
from ..managers.hover.panel import panel, row, section
from ..visual_config import theme
from .plot_utils import apply_academic_style

__all__ = ["plot_reaction_gene_concordance", "plot_result_agreement"]

# Drawn in this order, so the classes that matter sit on top.
CONCORDANCE_STYLE = {
    "neither": theme.NOT_SIGNIFICANT,
    "genes": "#999999",
    "network": "#E69F00",
    "against": "#D55E00",
    "follows": "#0072B2",
}
AGREEMENT_STYLE = {
    "neither": theme.NOT_SIGNIFICANT,
    "second": "#E69F00",
    "first": "#0072B2",
    "opposite": "#D55E00",
    "both": "#000000",
}


def _axes(plot) -> None:
    for dimension in ("width", "height"):
        plot.add_layout(
            Span(location=0, dimension=dimension, line_color=theme.FRAME_LINE, line_width=1)
        )
    plot.xaxis.axis_label_text_font_style = "normal"
    plot.yaxis.axis_label_text_font_style = "normal"


def _scatter(plot, frame: pd.DataFrame, x: str, y: str, styles: Mapping[str, str], names: Mapping[str, str]):
    items, renderers = [], []
    for status, color in styles.items():
        part = frame[frame["status"] == status]
        if part.empty:
            continue
        renderer = plot.scatter(
            x=x,
            y=y,
            source=ColumnDataSource(part),
            size=6 if status != "neither" else 5,
            fill_color=color,
            line_color=theme.MARK_EDGE if status != "neither" else None,
            line_width=0.4,
            fill_alpha=0.85 if status != "neither" else 0.6,
        )
        renderers.append(renderer)
        items.append(LegendItem(label=f"{names[status]} ({len(part):,})", renderers=[renderer]))
    # Largest class first in the legend, as the reader looks for the rare ones.
    plot.add_layout(
        Legend(items=list(reversed(items)), location="top_left", label_text_font_size="9pt",
               border_line_color=None, background_fill_alpha=0.8),
        "center",
    )
    return renderers


def plot_reaction_gene_concordance(
    table: pd.DataFrame,
    *,
    reaction_labels: Mapping[str, str],
    gene_labels: Mapping[str, str],
    title: str,
    width: int = 640,
):
    """Each reaction's Cohen's d against its strongest gene's, coloured by class."""
    if table.empty:
        raise ValueError("No reaction has a tested gene")
    frame = table.copy()
    frame["reaction"] = [reaction_labels.get(item, item) for item in frame["feature_id"]]
    frame["gene"] = [gene_labels.get(item, item) for item in frame["strongest_gene"]]
    frame["class"] = frame["status"].map(CONCORDANCE_CLASSES)
    for column in ("reaction", "gene", "feature_id"):
        frame[column] = frame[column].map(escape)
    plot = apply_academic_style(
        figure(
            width=width,
            height=520,
            title=title,
            tools="save,pan,box_zoom,wheel_zoom,reset",
            sizing_mode="stretch_width",
        )
    )
    plot.xaxis.axis_label = "Cohen's d of the reaction's gene with the largest effect"
    plot.yaxis.axis_label = "Cohen's d of the reaction"
    _axes(plot)
    renderers = _scatter(plot, frame, "gene_effect", "reaction_effect", CONCORDANCE_STYLE, CONCORDANCE_CLASSES)
    plot.add_tools(
        HoverTool(
            renderers=renderers,
            tooltips=panel(
                row("Reaction", "@reaction"),
                row("ID", "@feature_id"),
                row("Gene", "@gene"),
                section(
                    row("Reaction d", "@reaction_effect{+0.00}"),
                    row("Gene d", "@gene_effect{+0.00}"),
                    row("Genes significant", "@significant_genes of @genes"),
                    row("Class", "@class"),
                ),
            ),
        )
    )
    return plot


def plot_result_agreement(
    agreement: Agreement,
    *,
    labels: Mapping[str, str],
    width: int = 640,
):
    """This result's Cohen's d (y) against the other's (x), by where each is significant."""
    frame = agreement.table.copy()
    if frame.empty:
        raise ValueError("The two results share no feature")
    names = {
        **AGREEMENT_CLASSES,
        "first": f"{agreement.first} only",
        "second": f"{agreement.second} only",
    }
    frame["name"] = [escape(labels.get(item, item)) for item in frame["feature_id"]]
    frame["class"] = frame["status"].map(names).map(escape)
    frame["feature_id"] = frame["feature_id"].map(escape)
    rho = "not enough shared features" if np.isnan(agreement.spearman) else f"{agreement.spearman:.2f}"
    plot = apply_academic_style(
        figure(
            width=width,
            height=540,
            title=f"Spearman's rho of Cohen's d over {agreement.shared:,} shared features: {rho}",
            tools="save,pan,box_zoom,wheel_zoom,reset",
            sizing_mode="stretch_width",
        )
    )
    plot.xaxis.axis_label = f"Cohen's d, {agreement.second}"
    plot.yaxis.axis_label = f"Cohen's d, {agreement.first}"
    _axes(plot)
    renderers = _scatter(plot, frame, "second_effect", "first_effect", AGREEMENT_STYLE, names)
    plot.add_tools(
        HoverTool(
            renderers=renderers,
            tooltips=panel(
                row("Feature", "@name"),
                row("ID", "@feature_id"),
                section(
                    row(escape(agreement.first), "@first_effect{+0.00}"),
                    row(escape(agreement.second), "@second_effect{+0.00}"),
                    row("Significant in", "@class"),
                ),
            ),
        )
    )
    return plot

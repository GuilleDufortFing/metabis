import math

import numpy as np
import pandas as pd
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    HoverTool,  # Import HoverTool
    LinearColorMapper,
    PrintfTickFormatter,
    Text,  # Import Text glyph
)

# Import palettes needed for diverging colors
from bokeh.palettes import (  # Using Viridis for the magnitude color bar
    Greens,
    Greys,
    Oranges,  # Import a diverging palette
)
from bokeh.plotting import figure, show

from ..managers.hover import panel as hover_panel
from .click_behavior import add_click_behavior, get_external_reference_data


def plot_gene_reaction_influence_heatmap(
    results,
    data_manager,
    title=None,
    height=None,
    use_gene_symbols=True,
    max_reactions=50,
    max_genes=30,
    color_palette=None,  # Green palette
    gray_palette=None,  # Gray palette
    influence_threshold=None,  # Keep for potential future use, but not for coloring logic now
    show_highly_influenced_only=False,  # Keep for filtering reactions if needed
    return_plot=False,
):
    """Create a heatmap visualization of gene influence on reactions.

    Uses a green gradient for highly influenced reactions and a gray gradient
    for non-highly influenced reactions. Only the green color bar is shown.

    Args:
        results: Results dictionary from analyze_subsystem_gene_deltas function.
        title: Title for the plot.
        height: Height of the plot in pixels.
        use_gene_symbols: Whether to use gene symbols or IDs on y-axis.
        max_reactions: Maximum number of reactions to display.
        max_genes: Maximum number of genes to display.
        color_palette: List of colors for the green heatmap gradient.
        gray_palette: List of colors for the gray heatmap gradient.
        influence_threshold: (Currently unused for coloring) Optional threshold.
        show_highly_influenced_only: Only show reactions that are highly influenced.
        return_plot: Whether to return the plot object or show it.

    Returns:
        Bokeh figure object if return_plot=True, otherwise None (shows the plot).
    """
    # Extract reaction results
    per_reaction = results.get("per_reaction", {})
    if not per_reaction:
        # Handle case with no reaction data gracefully
        p = figure(
            height=height or 300,
            title=title or "Gene Influence Heatmap",
            sizing_mode="stretch_width",
        )
        p.text(
            x=0,
            y=0,
            text="No per-reaction influence data available.",
            text_align="center",
            text_baseline="middle",
        )
        if return_plot:
            return p
        else:
            show(p)
            return

    # Get the set of highly influenced reaction IDs
    highly_influenced_ids = set(results.get("highly_influenced_reactions", {}).keys())

    # Set default title
    if title is None and "subsystem" in results:
        title = f"Gene Influence on {results['subsystem']} Reactions (Color: Corr Sign, Intensity: % Var)"
    elif title is None:
        title = "Gene Influence on Reactions (Color: Corr Sign, Intensity: % Var)"

    # Filter reactions based on show_highly_influenced_only flag
    reactions = list(per_reaction.keys())
    if show_highly_influenced_only:
        reactions = [r for r in reactions if r in highly_influenced_ids]
        if not reactions:
            # Handle case where filtering leaves no reactions
            p = figure(height=height or 300, title=title, sizing_mode="stretch_width")
            p.text(
                x=0,
                y=0,
                text="No highly influenced reactions to display.",
                text_align="center",
                text_baseline="middle",
            )
            if return_plot:
                return p
            else:
                show(p)
                return
    else:
        # Prioritize highly influenced reactions even when showing all
        highly_influenced_in_list = [r for r in reactions if r in highly_influenced_ids]
        other_reactions_in_list = [
            r for r in reactions if r not in highly_influenced_ids
        ]
        reactions = highly_influenced_in_list + other_reactions_in_list

    # Limit to max_reactions
    reactions = reactions[:max_reactions]

    # Get top genes
    if use_gene_symbols and "top_genes_with_symbols" in results:
        top_genes_data = results["top_genes_with_symbols"][:max_genes]
        gene_ids = [item["gene_id"] for item in top_genes_data]
        gene_labels = [item["gene_symbol"] for item in top_genes_data]
    else:
        top_genes_data = results.get("top_genes", [])[:max_genes]
        gene_ids = [gene_id for gene_id, _ in top_genes_data]
        gene_labels = gene_ids

    if not gene_labels:
        # Handle case with no genes
        p = figure(height=height or 300, title=title, sizing_mode="stretch_width")
        p.text(
            x=0,
            y=0,
            text="No gene influence data available.",
            text_align="center",
            text_baseline="middle",
        )
        if return_plot:
            return p
        else:
            show(p)
            return

    # Create a dataframe to store influence values AND correlations
    df_influence = pd.DataFrame(0.0, index=gene_labels, columns=reactions)
    df_correlation = pd.DataFrame(
        0.0, index=gene_labels, columns=reactions
    )  # New DataFrame for correlations

    # Fill the dataframes
    for reaction_id in reactions:
        reaction_data = per_reaction.get(reaction_id, {})
        genes_ranked = reaction_data.get("genes_ranked", [])
        for gene_data in genes_ranked:
            # Check tuple length to get correct indices
            if len(gene_data) == 5:  # (id, coef, imp, pct, corr)
                gene_id = gene_data[0]
                pct_variance = gene_data[3]
                gene_react_corr = gene_data[4]
            elif (
                len(gene_data) == 6
            ):  # (id, symbol, coef, imp, pct, corr) - Should not happen here, but check just in case
                gene_id = gene_data[0]
                pct_variance = gene_data[4]
                gene_react_corr = gene_data[5]
            else:
                continue  # Skip if format is wrong

            if gene_id in gene_ids:
                idx = gene_ids.index(gene_id)
                gene_label = gene_labels[idx]
                df_influence.loc[gene_label, reaction_id] = pct_variance
                df_correlation.loc[gene_label, reaction_id] = (
                    gene_react_corr  # Store correlation
                )

    # Create a dataframe to store the influence values
    df = pd.DataFrame(0.0, index=gene_labels, columns=reactions)

    # Fill the dataframe
    for reaction_id in reactions:
        reaction_data = per_reaction.get(reaction_id, {})
        genes_ranked = reaction_data.get("genes_ranked", [])
        for gene_data in genes_ranked:
            if len(gene_data) > 3:  # Ensure data structure is as expected
                gene_id = gene_data[0]
                pct_variance = gene_data[3]
                if gene_id in gene_ids:
                    idx = gene_ids.index(gene_id)
                    gene_label = gene_labels[idx]
                    df.loc[gene_label, reaction_id] = pct_variance

    # Reshape for Bokeh
    df_inf_flat = df_influence.reset_index().melt(
        id_vars="index", var_name="reaction", value_name="influence"
    )
    df_corr_flat = df_correlation.reset_index().melt(
        id_vars="index", var_name="reaction", value_name="correlation"
    )

    # Merge the two flat dataframes
    df_flat = pd.merge(df_inf_flat, df_corr_flat, on=["index", "reaction"])
    df_flat.columns = ["gene", "reaction", "influence", "correlation"]

    # --- Define Color Palettes and Mappers ---
    # Diverging palettes for highly influenced - Darker first
    pos_palette = list(reversed(Greens[9][1:]))
    # Grayscale palette for non-highly influenced - Darker first
    gray_palette = list(reversed(Greys[9][1:]))

    # Define the range for mapping influence magnitude (0 to 100)
    high_val = 100

    # Create a mapper for the COLOR BAR (representing correlation for highly influenced)
    correlation_palette = list(Oranges[9][1:]) + pos_palette
    correlation_mapper = LinearColorMapper(palette=correlation_palette, low=-1, high=1)
    # --- End Color Definitions ---

    # --- Pre-calculate cell colors ---
    final_colors = []

    for _, row in df_flat.iterrows():
        reaction_id = row["reaction"]
        influence_val = row["influence"]
        correlation_val = row["correlation"]

        # Determine color using utility function
        color = data_manager.color_manager.get_influence_color(
            correlation=correlation_val,
            influence=influence_val,
            gray_palette=gray_palette,
            max_influence=high_val,
            use_gray_scale=(
                reaction_id not in highly_influenced_ids
            ),  # Use gray if not highly influenced
        )
        final_colors.append(color)

    df_flat["final_color"] = final_colors
    # --- End Pre-calculation ---

    # Create Bokeh data source
    source = ColumnDataSource(df_flat)

    # Calculate appropriate height
    if height is None:
        height = max(300, len(gene_labels) * 25)

    # Create figure
    p = figure(
        sizing_mode="stretch_width",
        height=height,
        title=title,
        x_range=reactions,
        y_range=list(reversed(gene_labels)),
        toolbar_location="right",
        tools="hover,save,pan,box_zoom,reset,wheel_zoom",
        x_axis_location="above",
        # Update tooltips to include correlation
        tooltips=hover_panel.panel(
            hover_panel.row("Gene", "@gene"),
            hover_panel.row("Reaction", "@reaction"),
            hover_panel.section(
                hover_panel.row("Influence", "@influence{0.00}%"),
                hover_panel.row("Correlation", "@correlation{0.00}"),
            ),
        ),
    )

    # Add rectangles using the pre-calculated colors
    p.rect(
        x="reaction",
        y="gene",
        width=1,
        height=1,
        source=source,
        line_color="white",
        line_width=0.5,
        fill_color="final_color",  # Use the pre-calculated diverging color
    )

    # Add color bar representing CORRELATION (for highly influenced reactions)
    color_bar = ColorBar(
        color_mapper=correlation_mapper,
        # ... (color bar parameters) ...
        title="Gene-Reaction Correlation (Highly Influenced)",
    )
    p.add_layout(color_bar, "right")

    # Styling
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = math.pi / 3

    if return_plot:
        return p
    else:
        show(p)


def plot_subsystem_influence_summary(
    data_manager,  # Added data_manager argument
    results,
    title=None,
    height=450,
    max_reactions=100,
    min_point_size=5,
    max_point_size=30,
    label_font_size="8pt",
    return_plot=False,
    enabled_toggle=None,
    click_filename_prefix: str = "influence_reaction",
):
    """Create a scatter plot summarizing gene influence analysis results per reaction.

    Highlights highly influenced reactions. For those, color indicates correlation
    (green=positive, orange/red=negative), and size indicates the dominance of the top gene.
    Non-highly influenced reactions are shown in gray. Reaction IDs are labeled.
    Hover text is generated using data_manager.get_reaction_hover.

    Args:
        data_manager: The AnalysisWorkspace object.
        results: Results dictionary from analyze_subsystem_gene_deltas function.
        title: Title for the plot (defaults to subsystem name if available).
        height: Height of the plot in pixels.
        max_reactions: Maximum number of reactions to display (points on the plot).
        min_point_size: Minimum pixel size for scatter points.
        max_point_size: Maximum pixel size for scatter points.
        label_font_size: Font size for the reaction ID labels.
        return_plot: Whether to return the plot object or show it.

    Returns:
        Bokeh figure object if return_plot=True, otherwise None (shows the plot).
    """
    if not data_manager:
        raise ValueError("A AnalysisWorkspace object must be provided.")

    per_reaction = results.get("per_reaction", {})
    subsystem_name = results.get("subsystem", "Unknown Subsystem")
    key_regulators_count = len(results.get("key_regulator_genes", []))
    highly_influenced_dict = results.get("highly_influenced_reactions", {})
    highly_influenced_ids = set(highly_influenced_dict.keys())
    total_reactions_analyzed = results.get(
        "analyzed_reactions_count", len(per_reaction)
    )

    if title is None:
        title = (
            f"{subsystem_name}: Reaction Analysis Summary\n"
            f"({key_regulators_count} Key Regulators, "
            f"{len(highly_influenced_ids)}/{total_reactions_analyzed} Highly Influenced Reactions)"
        )

    # --- Define Custom Color Palette ---
    neg_palette = list(Oranges[9][2:])
    pos_palette = list(reversed(Greens[9][1:]))
    custom_palette = neg_palette + pos_palette
    neutral_gray = "#C8C8C8"
    correlation_mapper = LinearColorMapper(palette=custom_palette, low=-1, high=1)
    # --- End Palette Definition ---

    plot_data = []
    reaction_ids = list(per_reaction.keys())
    limited_reaction_ids = reaction_ids[:max_reactions]

    for reaction_id in limited_reaction_ids:
        reaction_data = per_reaction.get(reaction_id)
        # Handle cases where analysis might have failed for a reaction
        if not reaction_data or reaction_data.get("error"):
            print(f"Skipping reaction {reaction_id} due to missing data or error.")
            continue

        genes_ranked = reaction_data.get("genes_ranked", [])

        # Calculate point size based on max influence % (pct_variance is index 3)
        max_pct_variance = 0
        if genes_ranked and len(genes_ranked[0]) > 3:
            max_pct_variance = genes_ranked[0][3]  # Get from top gene

        point_size = min_point_size + (max_pct_variance / 100.0) * (
            max_point_size - min_point_size
        )
        point_size = max(point_size, min_point_size)

        # --- Determine Color based on Top Gene's Correlation ---
        is_highly_influenced = reaction_id in highly_influenced_ids
        point_color = neutral_gray  # Default to gray
        top_gene_corr = np.nan  # Default

        if is_highly_influenced and genes_ranked:
            # Get correlation of the top gene (index 4 is the new gene_reaction_corr)
            if len(genes_ranked[0]) > 4:  # Check tuple length
                top_gene_corr = genes_ranked[0][4]

            if not pd.isna(top_gene_corr):
                # Use top_gene_corr for color mapping
                clamped_corr = max(-1.0, min(1.0, top_gene_corr))
                norm_corr = (clamped_corr + 1.0) / 2.0  # Normalize to 0-1
                palette_index = int(norm_corr * (len(custom_palette) - 1))
                point_color = custom_palette[palette_index]
            else:
                # Keep gray if correlation is NaN even if highly influenced
                point_color = neutral_gray
        # --- End Color Determination ---

        # --- Get Hover Text using data_manager ---
        # Use unified get_element_hover interface
        hover_text = data_manager.get_element_hover(
            element_id=reaction_id,
            element_type="reaction",
            current_subsystem=subsystem_name,  # Pass current subsystem for context
        )
        # --- End Hover Text ---

        reference_data = get_external_reference_data(
            data_manager,
            reaction_id,
            "reaction",
        )

        plot_data.append(
            {
                "element_id": reference_data["element_id"],
                "display_label": reference_data["display_label"],
                "external_url": reference_data["external_url"],
                "reaction_id": reaction_id,
                "r2_score": reaction_data.get("r2_score", np.nan),
                "significant_genes_count": reaction_data.get(
                    "significant_genes_count", 0
                ),
                "model_correlation": reaction_data.get(
                    "correlation", np.nan
                ),  # Keep original model correlation if needed
                "top_gene_correlation": top_gene_corr,  # Store the correlation used for color
                "rmse": reaction_data.get("rmse", np.nan),
                "max_pct_variance": max_pct_variance,
                # Get the actual param name used (e.g., alpha_used or reg_alpha_used)
                "alpha_used": reaction_data.get(
                    "alpha_used", reaction_data.get("reg_alpha_used", "N/A")
                ),
                "point_size": point_size,
                "point_color": point_color,
                "is_highly_influenced": is_highly_influenced,
                "hover_text": hover_text,
            }
        )

    if not plot_data:
        p = figure(height=height, title=title, sizing_mode="stretch_width")
        p.text(
            x=0,
            y=0,
            text="No reaction data available for this subsystem.",
            text_align="center",
            text_baseline="middle",
        )
        if return_plot:
            return p
        else:
            show(p)
            return

    source = ColumnDataSource(pd.DataFrame(plot_data))
    # Define tooltips
    tooltips = "@hover_text{safe}"

    # Create figure
    p = figure(
        sizing_mode="stretch_width",
        height=height,
        title=title,
        x_axis_label="Number of Significant Genes",
        y_axis_label="R² Score (Model Fit)",
        tools=["pan,wheel_zoom,box_zoom,reset,save"],
    )

    # Scatter renderer
    scatter_renderer = p.scatter(
        x="significant_genes_count",
        y="r2_score",
        source=source,
        size="point_size",
        color="point_color",
        alpha=0.7,
        line_color="black",
        line_width=0.5,
        name="reaction_points",
    )

    # Text labels
    p.add_glyph(
        source,
        Text(
            x="significant_genes_count",
            y="r2_score",
            text="reaction_id",
            x_offset=5,
            y_offset=0,
            text_font_size=label_font_size,
            text_color="#555555",
            text_align="left",
            text_baseline="middle",
        ),
    )

    # HoverTool configuration
    hover = HoverTool(
        renderers=[scatter_renderer],
        tooltips=tooltips,  # Use the updated tooltips variable
        attachment="vertical",  # Optional: Adjust hover attachment if needed
        point_policy="follow_mouse",  # Optional: Adjust hover policy if needed
    )
    p.add_tools(hover)

    add_click_behavior(
        p,
        filename_prefix=click_filename_prefix,
        enabled_toggle=enabled_toggle,
    )

    # Color bar
    color_bar = ColorBar(
        color_mapper=correlation_mapper,
        ticker=BasicTicker(desired_num_ticks=5),
        formatter=PrintfTickFormatter(format="%0.1f"),
        label_standoff=12,
        border_line_color=None,
        location=(0, 0),
        title="Top Gene Correlation (Highly Influenced)",
    )
    p.add_layout(color_bar, "right")

    # Styling
    p.xaxis.ticker = BasicTicker(min_interval=1)
    p.grid.grid_line_alpha = 0.3

    if return_plot:
        return p
    else:
        show(p)


def plot_subsystem_ranking_by_influence(
    data_manager,
    ranking_metric: str = "highly_influenced_reactions_count",
    color_palette=None,
    title: str = None,
    return_plot: bool = False,
):
    """Create a horizontal bar chart showing subsystems ranked by gene influence metrics.

    Args:
        data_manager: The AnalysisWorkspace object with experiment and group comparison data
        ranking_metric: Metric to use for ranking subsystems. Options:
            - "highly_influenced_reactions_count": Number of reactions with strong gene influence
            - "key_regulator_genes_count": Number of key regulator genes in the subsystem
            - "max_reaction_influence": Maximum influence any gene has on a reaction
            - "max_gene_regulation": Maximum regulation strength of any gene
            - "total_reaction_influence": Sum of all gene influence on reactions
            - "total_gene_regulation": Sum of all gene regulation strengths
        width: Width of the plot in pixels
        height: Height of the plot in pixels (calculated automatically if None)
        color_palette: List of colors for the bars
        title: Custom title for the plot
        return_plot: Whether to return the plot object or show it

    Returns:
        Bokeh figure object if return_plot=True, otherwise None (shows the plot)
    """
    # Extract group comparison from data_manager
    group_comparison = data_manager.get_group_comparison_artifact()

    if not group_comparison:
        raise ValueError("Group comparison not set in the data_manager")

    # Verify that subsystems gene deltas analysis has been performed
    if not hasattr(group_comparison, "all_subsystems_analysis_summary"):
        raise ValueError(
            "No subsystem gene deltas analysis found. Run analyze_all_subsystems_gene_deltas first."
        )

    analysis_summary = group_comparison.all_subsystems_analysis_summary
    subsystem_results = analysis_summary.get("subsystem_results", {})

    if not subsystem_results:
        raise ValueError("No subsystem results found in the analysis.")

    # Calculate metrics for each subsystem
    subsystem_metrics = []

    for subsystem, result in subsystem_results.items():
        # Skip subsystems with no data
        if not result:
            continue

        metrics = {
            "subsystem": subsystem,
            "highly_influenced_reactions_count": len(
                result.get("highly_influenced_reactions", {})
            ),
            "key_regulator_genes_count": len(result.get("key_regulator_genes", [])),
            "reactions_analyzed": result.get("analyzed_reactions_count", 0),
            "total_reactions": result.get("total_reactions_in_subsystem", 0),
            "model_type": result.get("analysis_parameters", {}).get(
                "model_type", "unknown"
            ),
            "importance_type": result.get("analysis_parameters", {}).get(
                "importance_type", "unknown"
            ),
        }

        # Calculate maximum and total influence metrics
        max_reaction_influence = 0
        total_reaction_influence = 0
        max_gene_regulation = 0
        total_gene_regulation = 0

        # Process per-reaction data
        for reaction_id, reaction_data in result.get("per_reaction", {}).items():
            genes_ranked = reaction_data.get("genes_ranked", [])
            if genes_ranked:
                # Maximum influence for this reaction
                max_influence = (
                    genes_ranked[0][3] if genes_ranked else 0
                )  # Get percentage variance value
                max_reaction_influence = max(max_reaction_influence, max_influence)

                # Total influence for this reaction
                reaction_total = sum(gene_data[3] for gene_data in genes_ranked)
                total_reaction_influence += reaction_total

        # Process key regulator genes
        for gene_data in result.get("key_regulator_genes", []):
            gene_influence = len(gene_data.get("influenced_reactions", []))
            max_gene_regulation = max(max_gene_regulation, gene_influence)
            total_gene_regulation += gene_influence

        # Add these metrics
        metrics["max_reaction_influence"] = max_reaction_influence
        metrics["total_reaction_influence"] = total_reaction_influence
        metrics["max_gene_regulation"] = max_gene_regulation
        metrics["total_gene_regulation"] = total_gene_regulation

        subsystem_metrics.append(metrics)

    # Create DataFrame and sort by the selected metric
    metrics_df = pd.DataFrame(subsystem_metrics)

    # Check if the selected metric exists
    if ranking_metric not in metrics_df.columns:
        valid_metrics = [col for col in metrics_df.columns if col != "subsystem"]
        raise ValueError(
            f"Invalid ranking_metric: {ranking_metric}. Valid options: {', '.join(valid_metrics)}"
        )

    metrics_df = metrics_df.sort_values(
        by=ranking_metric,
        ascending=False,
    )

    # Set default color palette if not provided
    if color_palette is None:
        color_palette = ["#00441b"]  # Default to green

    # Generate hover tooltips
    metrics_df["hover"] = metrics_df.apply(
        lambda row: hover_panel.panel(
            hover_panel.row("Subsystem", row["subsystem"]),
            hover_panel.section(
                hover_panel.row(
                    "Highly influenced reactions",
                    f"{row['highly_influenced_reactions_count']} of {row['total_reactions']} "
                    f"({row['highly_influenced_reactions_count'] / max(1, row['total_reactions']):.0%})",
                ),
                hover_panel.row("Key regulator genes", row["key_regulator_genes_count"]),
                hover_panel.row("Max reaction influence", f"{row['max_reaction_influence']:.1f}%"),
                hover_panel.row("Max gene regulation", f"{row['max_gene_regulation']} reactions"),
            ),
            hover_panel.section(
                hover_panel.note(f"{row['model_type']} model with {row['importance_type']} importance.")
            ),
        ),
        axis=1,
    )

    # Get subsystem order (reversed for proper display)
    subsys_order = metrics_df["subsystem"].tolist()[::-1]

    # Set title if not provided
    if title is None:
        title = f"Subsystems Ranked by {ranking_metric.replace('_', ' ').title()}"

    # Create figure
    p = figure(
        sizing_mode="stretch_width",
        height=max(400, len(subsys_order) * 250),
        title=title,
        y_range=subsys_order,
        toolbar_location="right",
        tools="hover,save,pan,box_zoom,reset,wheel_zoom",
        tooltips="@hover{safe}",
    )

    # Create ColumnDataSource for Bokeh
    source = ColumnDataSource(metrics_df)

    # Draw horizontal bars
    p.hbar(
        y="subsystem",
        right=ranking_metric,
        height=0.8,
        source=source,
        fill_color=color_palette[0],
        line_width=1,
        line_color="#808080",
    )

    # Format axes
    p.xaxis.axis_label = ranking_metric.replace("_", " ").title()
    p.yaxis.axis_label = "Subsystem"

    # Style the plot
    p.axis.major_tick_line_color = None
    p.axis.minor_tick_line_color = None

    if return_plot:
        return p
    else:
        show(p)

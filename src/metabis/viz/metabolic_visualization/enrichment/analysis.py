"""Graph analysis functions for metabolic visualization enrichment.

This module contains functions for analyzing graph structures to identify
significant subgraphs, missing data patterns, gene influence relationships,
and metabolite-reaction connectivity patterns.
"""

from collections import defaultdict

import pandas as pd

from ..generation.data_structures.core_types import GlyphClass

# ============================================================================
# Significance Analysis
# ============================================================================


def identify_significant_subgraph(
    glyphs,
    connections,
    data_manager,
    highlight_significant_reactions=True,
    highlight_significant_genes=False,
):
    """Identify nodes that are part of the significant subgraph.

    This includes significant reaction/gene nodes and all nodes directly
    connected to them (both incoming and outgoing connections).

    Args:
        glyphs: List of glyph dictionaries
        connections: List of connection dictionaries
        data_manager: AnalysisWorkspace object with group_comparison
        highlight_significant_reactions: Whether to highlight significant reactions
        highlight_significant_genes: Whether to highlight significant genes

    Returns:
        Set of node IDs that are part of the significant subgraph
    """
    # Initialize empty set for significant nodes
    significant_nodes = set()
    reaction_ids = set()
    gene_ids = set()
    significant_reaction_labels = set()
    significant_gene_labels = set()

    # Get significant reactions from the experiment's group comparison
    comparison_artifact = data_manager.get_group_comparison_artifact()
    if not comparison_artifact:
        return set()

    # Get significant reactions if requested
    if highlight_significant_reactions:
        significant_reactions = data_manager.data_provider.get_main_direction_significant_reactions()
    else:
        significant_reactions = set()

    # Get significant genes if requested
    if highlight_significant_genes:
        significant_genes = comparison_artifact.get_significant_elements(
            for_genes=True
        )
    else:
        significant_genes = set()

    if not significant_reactions and not significant_genes:
        return set()

    # First identify significant reaction nodes
    if highlight_significant_reactions:
        for glyph in glyphs:
            if (
                glyph["glyph_class"] == GlyphClass.REACTION
                and glyph["label"] in significant_reactions
            ):
                significant_nodes.add(glyph["id"])
                reaction_ids.add(glyph["id"])
                significant_reaction_labels.add(glyph["label"])

    # Then identify significant gene nodes
    if highlight_significant_genes:
        for glyph in glyphs:
            if (
                glyph["glyph_class"] == GlyphClass.GENE
                and glyph["label"] in significant_genes
            ):
                significant_nodes.add(glyph["id"])
                gene_ids.add(glyph["id"])
                significant_gene_labels.add(glyph["label"])
            if glyph["glyph_class"] == GlyphClass.ENZYME_GROUP:
                # Check if any of the enzymes in this group are significant
                for enzyme in glyph["glyph_ids"]:
                    if enzyme in significant_genes:
                        significant_nodes.add(glyph["id"])

    # If highlight_significant_genes = False, add enzyme groups associated
    # with significant reactions
    if not highlight_significant_genes:
        for glyph in glyphs:
            if (
                glyph["glyph_class"] == GlyphClass.ENZYME_GROUP
                and "reaction_id" in glyph
            ):
                if glyph["reaction_id"] in significant_reaction_labels:
                    significant_nodes.add(glyph["id"])
                    # Also add all enzyme nodes in this group
                    if "glyph_ids" in glyph:
                        significant_nodes.update(glyph["glyph_ids"])

    # Create a graph representation to find connected nodes
    graph = {}
    for glyph in glyphs:
        graph[glyph["id"]] = {"in": set(), "out": set()}

    # Add connections to the graph
    for conn in connections:
        source_id = conn["source"]
        target_id = conn["target"]

        if source_id in graph:
            graph[source_id]["out"].add(target_id)
        if target_id in graph:
            graph[target_id]["in"].add(source_id)

    # Add immediate neighbors of significant reactions to the significant nodes set
    for reaction_id in reaction_ids:
        # Add incoming connections (predecessors)
        significant_nodes.update(graph[reaction_id]["in"])
        # Add outgoing connections (successors)
        significant_nodes.update(graph[reaction_id]["out"])

    # Add immediate neighbors of significant genes to the significant nodes set
    for gene_id in gene_ids:
        # Add incoming connections (predecessors)
        significant_nodes.update(graph[gene_id]["in"])
        # Add outgoing connections (successors)
        significant_nodes.update(graph[gene_id]["out"])

    return significant_nodes


# ============================================================================
# Metabolite-Reaction Connectivity Analysis
# ============================================================================


def build_metabolite_reaction_map(glyphs, connections):
    """Build a mapping of metabolites to their connected reactions.

    Args:
        glyphs: List of glyph dictionaries
        connections: List of connection dictionaries

    Returns:
        Dict mapping metabolite glyph IDs to sets of connected reaction glyph IDs
    """
    # Create glyph type lookup
    glyph_types = {glyph["id"]: glyph["glyph_class"] for glyph in glyphs}

    # Build metabolite -> reactions mapping
    metabolite_reactions = {}

    for conn in connections:
        source_id = conn["source"]
        target_id = conn["target"]

        source_type = glyph_types.get(source_id)
        target_type = glyph_types.get(target_id)

        # Check for metabolite-reaction connections
        if source_type == GlyphClass.METABOLITE and target_type == GlyphClass.REACTION:
            if source_id not in metabolite_reactions:
                metabolite_reactions[source_id] = set()
            metabolite_reactions[source_id].add(target_id)

        elif (
            source_type == GlyphClass.REACTION and target_type == GlyphClass.METABOLITE
        ):
            if target_id not in metabolite_reactions:
                metabolite_reactions[target_id] = set()
            metabolite_reactions[target_id].add(source_id)

    return metabolite_reactions


def get_active_reactions(glyphs, data_manager):
    """Get the set of reaction glyph IDs that have data in data_manager.

    Args:
        glyphs: List of glyph dictionaries
        data_manager: AnalysisWorkspace object with experimental data

    Returns:
        Set of reaction glyph IDs that have experimental data
    """
    if not data_manager:
        return set()

    active_reaction_ids = set()

    for glyph in glyphs:
        if glyph["glyph_class"] == GlyphClass.REACTION:
            reaction_label = glyph["label"]
            try:
                reaction_data = data_manager.data_provider.get_reaction_data(
                    reaction_label
                )
                if reaction_data["is_in_model"]:
                    active_reaction_ids.add(glyph["id"])
            except Exception:
                # If we can't get data, it's not active
                pass

    return active_reaction_ids


def should_metabolite_be_missing(metabolite_id, metabolite_reactions, active_reactions):
    """Determine if a metabolite should be styled as missing data.

    A metabolite should only be marked as missing if ALL of its connected
    reactions are missing (have no experimental data).

    Args:
        metabolite_id: Glyph ID of the metabolite
        metabolite_reactions: Dict mapping metabolite IDs to connected reaction IDs
        active_reactions: Set of reaction glyph IDs that have experimental data

    Returns:
        bool: True if metabolite should have missing data styling
    """
    connected_reactions = metabolite_reactions.get(metabolite_id, set())

    # If no connected reactions, don't mark as missing
    if not connected_reactions:
        return False

    # Check if ANY connected reaction is active
    for reaction_id in connected_reactions:
        if reaction_id in active_reactions:
            return False  # At least one active reaction, so metabolite is not missing

    # All connected reactions are missing
    return True


# ============================================================================
# Missing Data Analysis
# ============================================================================


def identify_missing_data_subgraph(glyphs, connections, data_manager):
    """Identify nodes that should have missing data styling.

    This function uses sophisticated logic to determine which elements should be
    styled as missing data:
    - Missing reactions: always marked as missing
    - Genes: marked as missing if connected to missing reactions
    - Enzyme groups: marked as missing if associated with missing reactions
    - Metabolites: marked as missing only if ALL connected reactions are missing

    Args:
        glyphs: List of glyph dictionaries
        connections: List of connection dictionaries
        data_manager: AnalysisWorkspace object with experimental data

    Returns:
        Dict mapping node IDs to missing reason strings ('no_metadata' or 'no_experimental_data')
    """
    if not data_manager:
        return {}

    missing_data_elements = {}
    missing_reaction_ids = set()
    missing_reaction_labels = set()

    # Phase 1: Identify missing reactions
    for glyph in glyphs:
        if glyph["glyph_class"] == GlyphClass.REACTION:
            reaction_id = glyph["label"]
            try:
                reaction_data = data_manager.data_provider.get_reaction_data(reaction_id)
                if not reaction_data["is_in_model"]:
                    missing_data_elements[glyph["id"]] = "no_metadata"
                    missing_reaction_ids.add(glyph["id"])
                    missing_reaction_labels.add(reaction_id)
                elif not (reaction_data.get("comparison_stats", {}).get("has_comparison_data") or reaction_data.get("experiment_stats", {}).get("has_experiment_data")):
                    missing_data_elements[glyph["id"]] = "no_experimental_data"
                    missing_reaction_ids.add(glyph["id"])
                    missing_reaction_labels.add(reaction_id)
            except Exception:
                # If we can't get data, assume missing metadata
                missing_data_elements[glyph["id"]] = "no_metadata"
                missing_reaction_ids.add(glyph["id"])
                missing_reaction_labels.add(reaction_id)
        if glyph["glyph_class"] == GlyphClass.METABOLITE:
            # If metabolite has no associated data at all, mark as no_metadata
            try:
                metabolite_data = data_manager.data_provider.get_metabolite_data(
                    glyph["label"]
                )
                if not metabolite_data["is_in_model"]:
                    missing_data_elements[glyph["id"]] = "no_metadata"
            except Exception:
                missing_data_elements[glyph["id"]] = "no_metadata"
        if glyph["glyph_class"] == GlyphClass.GENE:
            # If gene has no associated data at all, mark as no_metadata
            try:
                gene_data = data_manager.data_provider.get_gene_data(glyph["label"])
                if not gene_data["is_in_model"]:
                    missing_data_elements[glyph["id"]] = "no_metadata"
            except Exception:
                missing_data_elements[glyph["id"]] = "no_metadata"


    # If no missing reactions, return empty dict
    if not missing_reaction_ids:
        return {}

    # Phase 2: Get active reactions for metabolite analysis
    active_reactions = get_active_reactions(glyphs, data_manager)

    # Phase 3: Build metabolite-reaction connectivity map
    metabolite_reactions = build_metabolite_reaction_map(glyphs, connections)

    # Phase 4: Create a graph representation for gene connectivity
    graph = {}
    for glyph in glyphs:
        graph[glyph["id"]] = {"in": set(), "out": set()}

    # Add connections to the graph
    for conn in connections:
        source_id = conn["source"]
        target_id = conn["target"]

        if source_id in graph:
            graph[source_id]["out"].add(target_id)
        if target_id in graph:
            graph[target_id]["in"].add(source_id)

    # Phase 5: Add genes connected to missing reactions
    for reaction_id in missing_reaction_ids:
        # Add connected genes (both incoming and outgoing)
        for connected_id in graph[reaction_id]["in"] | graph[reaction_id]["out"]:
            # Find the glyph to check its type
            connected_glyph = next((g for g in glyphs if g["id"] == connected_id), None)
            if connected_glyph and connected_glyph["glyph_class"] == GlyphClass.GENE:
                # If the gene is not already marked, mark as no_experimental_data (since it's due to missing reaction data)
                if connected_id not in missing_data_elements:
                    missing_data_elements[connected_id] = "no_experimental_data"

    # Phase 6: Add enzyme groups associated with missing reactions
    for glyph in glyphs:
        if glyph["glyph_class"] == GlyphClass.ENZYME_GROUP and "reaction_id" in glyph:
            if glyph["reaction_id"] in missing_reaction_labels:
                if glyph["id"] not in missing_data_elements:
                    missing_data_elements[glyph["id"]] = "no_experimental_data"
                # Also add all enzyme nodes in this group
                if "glyph_ids" in glyph:
                    for enzyme_id in glyph["glyph_ids"]:
                        if enzyme_id not in missing_data_elements:
                            missing_data_elements[enzyme_id] = "no_experimental_data"

    # Phase 7: Analyze metabolites with smart connectivity logic
    for glyph in glyphs:
        if glyph["glyph_class"] == GlyphClass.METABOLITE:
            if should_metabolite_be_missing(
                glyph["id"], metabolite_reactions, active_reactions
            ):
                if glyph["id"] not in missing_data_elements:
                    missing_data_elements[glyph["id"]] = "no_experimental_data"

    return missing_data_elements


# ============================================================================
# Gene Influence Analysis
# ============================================================================


def identify_gene_influence_connections(
    glyphs, element_name, data_manager, highlight_gene_influence
):
    """Identify and prepare gene influence highlighting data.

    Args:
        glyphs: List of glyph dictionaries
        element_name: Name of the element being analyzed
        data_manager: AnalysisWorkspace object with group comparison data
        highlight_gene_influence: Whether gene influence highlighting is enabled

    Returns:
        Tuple of (influential_gene_glyph_ids, influence_connections_data,
                  highly_influenced_reaction_colors)
    """
    influential_gene_glyph_ids = set()
    influence_connections_data = defaultdict(list)
    highly_influenced_reaction_colors = {}

    influence_comparison = (
        data_manager.get_group_comparison_artifact() if data_manager else None
    )
    if not (
        highlight_gene_influence
        and influence_comparison
        and hasattr(influence_comparison, "all_subsystems_analysis_summary")
    ):
        return (
            influential_gene_glyph_ids,
            influence_connections_data,
            highly_influenced_reaction_colors,
        )

    print("Identifying *highly* influential genes and connections...")
    element_results_dict = (
        influence_comparison.all_subsystems_analysis_summary.get(
            "subsystem_results", {}
        )
    )
    element_analysis = element_results_dict.get(element_name)

    if not (
        element_analysis
        and "per_reaction" in element_analysis
        and "highly_influenced_reactions" in element_analysis
    ):
        print(
            f"Warning: Gene influence analysis results not found for element "
            f"'{element_name}'. Cannot highlight influence."
        )
        return (
            influential_gene_glyph_ids,
            influence_connections_data,
            highly_influenced_reaction_colors,
        )

    all_gene_glyphs = [g for g in glyphs if g["glyph_class"] == GlyphClass.GENE]
    reaction_glyph_dict = {
        g["label"]: g for g in glyphs if g["glyph_class"] == GlyphClass.REACTION
    }

    highly_influenced_reactions_data = element_analysis.get(
        "highly_influenced_reactions", {}
    )
    max_pct_variance = 100.0  # Still needed for scaling

    for reaction_mar_id, influence_details in highly_influenced_reactions_data.items():
        if reaction_mar_id not in reaction_glyph_dict:
            continue

        reaction_glyph = reaction_glyph_dict[reaction_mar_id]
        reaction_glyph_id = reaction_glyph["id"]

        influential_gene_ids_for_this_reaction = set(
            influence_details.get("influential_genes", [])
        )
        per_reaction_data = element_analysis["per_reaction"].get(reaction_mar_id, {})
        genes_ranked = per_reaction_data.get("genes_ranked", [])

        # Find top gene by pct_variance for background color
        top_gene_pct_variance = -1.0
        top_gene_corr = 0.0
        found_top_gene = False

        for gene_data in genes_ranked:
            if len(gene_data) == 5:
                gene_id, _, _, pct_variance, gene_react_corr = gene_data
            elif len(gene_data) == 6:
                gene_id, _, _, _, pct_variance, gene_react_corr = gene_data
            else:
                continue

            # Track top gene by pct_variance
            if not pd.isna(pct_variance) and pct_variance > top_gene_pct_variance:
                top_gene_pct_variance = pct_variance
                top_gene_corr = gene_react_corr if not pd.isna(gene_react_corr) else 0.0
                found_top_gene = True

            if gene_id in influential_gene_ids_for_this_reaction:
                gene_glyph_id = find_closest_gene_glyph(
                    gene_id, reaction_glyph, all_gene_glyphs
                )
                if gene_glyph_id:
                    influential_gene_glyph_ids.add(gene_glyph_id)
                    influence_connections_data[reaction_glyph_id].append(
                        (gene_glyph_id, pct_variance, gene_react_corr)
                    )

        # Determine background circle color using top gene's influence color
        if found_top_gene:
            bg_color = data_manager.color_manager.get_influence_color(
                correlation=top_gene_corr,
                influence=top_gene_pct_variance,
                max_influence=max_pct_variance,
            )
            highly_influenced_reaction_colors[reaction_glyph_id] = bg_color

    return (
        influential_gene_glyph_ids,
        influence_connections_data,
        highly_influenced_reaction_colors,
    )


def find_closest_gene_glyph(gene_label, reaction_glyph, all_gene_glyphs):
    """Find the gene glyph with the matching label closest to the reaction glyph.

    Args:
        gene_label: Label of the gene to find
        reaction_glyph: Reference reaction glyph for distance calculation
        all_gene_glyphs: List of all gene glyphs to search through

    Returns:
        str or None: ID of the closest matching gene glyph, or None if not found
    """
    matching_glyphs = [g for g in all_gene_glyphs if g.get("label") == gene_label]

    if not matching_glyphs:
        return None

    if len(matching_glyphs) == 1:
        return matching_glyphs[0]["id"]

    # Calculate distances
    reaction_pos = (reaction_glyph["x"], reaction_glyph["y"])
    closest_glyph_id = None
    min_dist_sq = float("inf")

    for gene_glyph in matching_glyphs:
        gene_pos = (gene_glyph["x"], gene_glyph["y"])
        dist_sq = (reaction_pos[0] - gene_pos[0]) ** 2 + (
            reaction_pos[1] - gene_pos[1]
        ) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_glyph_id = gene_glyph["id"]

    return closest_glyph_id

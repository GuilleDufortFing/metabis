"""Data loading utilities for the modern metabolic model.

This module contains functions responsible for loading metabolic data from
files and building the core entities with direct field mappings.

The loaders focus on populating entity fields that can be directly mapped
from source files, leveraging existing parsing utilities where appropriate.
"""

import os
import re
from typing import Dict, List, Optional

import pandas as pd

from .entities import (
    Compartment,
    Gene,
    Metabolite,
    Reaction,
    Subsystem,
    SubsystemNeighborRelationship,
)
from .reaction_parser import parse_reaction_equation


def load_subsystem_classification(file_path: str) -> Dict[str, str]:
    """Parse the subsystem category taxonomy and return id->category mapping.

    Args:
        file_path: Path to the subsystem category taxonomy file

    Returns:
        Dict mapping subsystem_id to category_name

    Example:
        {
            'glycolysis_gluconeogenesis': 'Carbohydrate Metabolism',
            'pentose_phosphate_pathway': 'Carbohydrate Metabolism',
            'sphingolipid_metabolism': 'Lipid Metabolism',
            ...
        }
    """
    if not os.path.exists(file_path):
        print(f"Warning: Classification file not found: {file_path}")
        return {}

    classification_map = {}
    current_category = None

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Category header (starts with ###)
                if line.startswith("###"):
                    current_category = line[3:].strip()

                # Subsystem entry (starts with -)
                elif line.startswith("-") and current_category:
                    subsystem_id = line[1:].strip()
                    classification_map[subsystem_id] = current_category

    except Exception as e:
        print(f"Error reading classification file {file_path}: {e}")
        return {}

    print(f"Loaded {len(classification_map)} subsystem classifications")
    return classification_map


def check_subsystem_categories_file(file_path: str) -> Dict[str, any]:
    """Validate subsystem categories file syntax and return statistics.

    This function checks if a file has valid subsystem categories format
    without building the full classification mapping.

    Args:
        file_path: Path to the subsystem category taxonomy file

    Returns:
        Dict with keys:
            - "valid": bool - whether file has valid syntax
            - "num_subsystems": int - count of subsystem entries
            - "num_categories": int - count of unique categories
            - "categories": List[str] - list of category names
            - "error": Optional[str] - error message if invalid
    """
    result = {
        "valid": False,
        "num_subsystems": 0,
        "num_categories": 0,
        "categories": [],
        "error": None,
    }

    if not os.path.exists(file_path):
        result["error"] = f"File not found: {file_path}"
        return result

    categories = set()
    num_subsystems = 0
    current_category = None

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Category header (starts with ###)
                if line.startswith("###"):
                    current_category = line[3:].strip()
                    if current_category:
                        categories.add(current_category)
                    else:
                        result["error"] = f"Empty category name at line {line_num}"
                        return result

                # Subsystem entry (starts with -)
                elif line.startswith("-"):
                    if current_category is None:
                        result["error"] = f"Subsystem entry without category at line {line_num}"
                        return result
                    subsystem_id = line[1:].strip()
                    if subsystem_id:
                        num_subsystems += 1
                    else:
                        result["error"] = f"Empty subsystem name at line {line_num}"
                        return result

                # Invalid line (not empty, not category, not subsystem)
                else:
                    result["error"] = f"Invalid line format at line {line_num}: '{line[:50]}...'"
                    return result

        if num_subsystems == 0:
            result["error"] = "No subsystems found in file"
            return result

        if len(categories) == 0:
            result["error"] = "No categories found in file"
            return result

        result["valid"] = True
        result["num_subsystems"] = num_subsystems
        result["num_categories"] = len(categories)
        result["categories"] = sorted(list(categories))

    except Exception as e:
        result["error"] = f"Error reading file: {e}"
        return result

    return result


def load_reactions_from_files(
    meta_file_path: str, gem_file_path: str, mapping_file_path: str
) -> List[Reaction]:
    """Load reaction entities from the reference-model reaction tables.

    Args:
        meta_file_path: Path to the reaction catalog file
        gem_file_path: Path to the reaction formulas file
        mapping_file_path: Path to the reaction identifier crosswalk file

    Returns:
        List of Reaction entities with populated fields
    """
    reactions = []

    # Load main reaction data
    meta_df = pd.read_csv(meta_file_path)

    # Load formula data
    gem_df = pd.read_csv(gem_file_path, sep="\t")
    formula_mapping = dict(zip(gem_df["Rxn name"], gem_df["Formula"]))

    # Load ID mappings
    mapping_df = pd.read_csv(mapping_file_path, sep="\t")
    mapping_dict = {}
    for _, row in mapping_df.iterrows():
        rxn_id = row["rxns"].strip('"')
        mappings = {}
        for col in mapping_df.columns[1:]:  # Skip first column (rxns)
            if pd.notna(row[col]) and row[col] != "" and row[col] != '""':
                value = str(row[col]).strip('"')
                if value:
                    mappings[col] = value
        mapping_dict[rxn_id] = mappings

    for _, row in meta_df.iterrows():
        reaction_id = row["ID"]

        # Get original subsystem name for preservation
        original_subsystem = row.get("SUBSYSTEM", "")
        cleaned_subsystem = _clean_subsystem_name(original_subsystem)

        # Extract basic information
        reaction = Reaction(
            id=reaction_id,
            name=row.get("NAME", ""),
            equation=row.get("EQUATION", ""),
            formula=formula_mapping.get(reaction_id, None),
            subsystem_id=cleaned_subsystem,
            compartment_id="",  # Will be inferred from metabolites
            gene_rule=row.get("GENE ASSOCIATION", ""),
            gene_rule_symbol=row.get("GENE_ASSOCIATION (SYMBOL)", ""),
            confidence_score=_safe_float(row.get("CONFIDENCE SCORE")),
            references=row.get("REFERENCE", ""),
            miriam=row.get("MIRIAM", ""),
            id_mappings=mapping_dict.get(reaction_id, {}),
        )

        # Store original subsystem name for later use
        if original_subsystem:
            reaction._original_subsystem_name = original_subsystem

        # Parse equation to extract metabolites using existing parser
        if reaction.formula:
            try:
                reactants, products, is_reversible = parse_reaction_equation(reaction.formula)
                reaction.reactants = reactants
                reaction.products = products
                reaction.is_reversible = is_reversible

                # Infer compartment from metabolite IDs in the equation
                compartments = set()
                all_metabolites = reactants + products
                for metabolite in all_metabolites:
                    comp = _extract_compartment_from_metabolite(metabolite)
                    if comp:
                        compartments.add(comp)

                # Set reaction compartment (use most common or first found)
                if compartments:
                    # If multiple compartments, prefer cytosol, otherwise use first
                    if "Cytosol" in compartments:
                        reaction.compartment_id = "Cytosol"
                    else:
                        reaction.compartment_id = list(compartments)[0]

                # TODO: Extract stoichiometry coefficients from equation
                # This requires additional parsing logic
            except Exception:
                # Fallback to empty lists if parsing fails
                reaction.reactants = []
                reaction.products = []

        if reaction.gene_rule:
            reaction.associated_genes = _parse_gene_association(reaction.gene_rule)

        reactions.append(reaction)

    return reactions


def load_metabolites_from_file(file_path: str) -> List[Metabolite]:
    """Load metabolite entities from the metabolite identifier crosswalk.

    Args:
        file_path: Path to the metabolite identifier crosswalk file

    Returns:
        List of Metabolite entities with populated fields
    """
    metabolites = []

    df = pd.read_csv(file_path, sep="\t")

    for _, row in df.iterrows():
        metabolite_id = row["mets"].strip('"')

        compartment_ids = {_extract_compartment_from_metabolite(metabolite_id)}

        # Build ID mappings from available columns
        id_mappings = {}
        mapping_columns = [
            "metBiGGID",
            "metKEGGID",
            "metHMDBID",
            "metChEBIID",
            "metPubChemID",
            "metLipidMapsID",
            "metEHMNID",
            "metHepatoNET1ID",
            "metRecon3DID",
            "metMetaNetXID",
            "metHMR2ID",
            "metRetiredID",
            "metName",
        ]

        for col in mapping_columns:
            if (
                col in row
                and pd.notna(row[col])
                and row[col] != ""
                and row[col] != '""'
            ):
                value = str(row[col]).strip('"')
                if value:
                    id_mappings[col] = value

        # Handle NaN values in name
        metabolite_name = row.get("metName", "")
        if pd.isna(metabolite_name):
            metabolite_name = ""
        else:
            metabolite_name = str(metabolite_name).strip('"')

        metabolite = Metabolite(
            id=metabolite_id,
            name=metabolite_name,
            compartment_ids=compartment_ids,
            id_mappings=id_mappings,
        )

        # Note: produced_by and consumed_by lists will be populated
        # by the build_metabolite_reaction_relationships() post-processing function

        metabolites.append(metabolite)

    return metabolites


def load_genes_from_file(file_path: str) -> List[Gene]:
    """Load gene entities from the gene identifier crosswalk.

    Args:
        file_path: Path to the gene identifier crosswalk file

    Returns:
        List of Gene entities with populated fields
    """
    genes = []

    df = pd.read_csv(file_path, sep="\t")

    for _, row in df.iterrows():
        gene_id = row["genes"].strip('"')

        # Parse aliases
        aliases = []
        if pd.notna(row.get("geneAliases")) and str(row["geneAliases"]) != '""':
            aliases_str = str(row["geneAliases"]).strip('"')
            aliases = [
                alias.strip() for alias in aliases_str.split(";") if alias.strip()
            ]

        # Parse compartments
        compartments = set()
        if pd.notna(row.get("compartments")) and str(row["compartments"]) != '""':
            comp_str = str(row["compartments"]).strip('"')
            # Split by semicolon and normalize compartment names
            for comp in comp_str.split(";"):
                comp_clean = comp.strip()
                if comp_clean:
                    # Map full names to short codes
                    comp_id = comp_clean
                    compartments.add(comp_id)

        # Build ID mappings
        id_mappings = {}
        mapping_columns = [
            "geneENSTID",
            "geneENSPID",
            "geneUniProtID",
            "geneEntrezID",
            "compDataSource",
        ]

        for col in mapping_columns:
            if (
                col in row
                and pd.notna(row[col])
                and row[col] != ""
                and row[col] != '""'
            ):
                value = str(row[col]).strip('"')
                if value:
                    id_mappings[col] = value

        # Handle NaN values in gene fields
        gene_symbol = row.get("geneSymbols", "")
        if pd.isna(gene_symbol):
            gene_symbol = ""
        else:
            gene_symbol = str(gene_symbol).strip('"')

        gene_name = row.get("geneNames", "")
        if pd.isna(gene_name):
            gene_name = ""
        else:
            gene_name = str(gene_name).strip('"')

        gene = Gene(
            id=gene_id,
            symbol=gene_symbol,
            name=gene_name,
            aliases=aliases,
            compartments=compartments,
            id_mappings=id_mappings,
        )

        # Note: associated_reactions list will be populated
        # by the build_gene_reaction_associations() post-processing function

        # TODO: Build subsystem_ids set
        # This requires cross-referencing with reactions and subsystems

        genes.append(gene)

    return genes


def create_subsystems_from_reactions(reactions: List[Reaction]) -> List[Subsystem]:
    """Create subsystem entities from reaction subsystem information.

    Note: Categories are NOT assigned during subsystem creation. Categories
    are loaded separately by the CategoryManager after the model is fully
    initialized. This allows fast category reloading without recreating
    the model.

    Args:
        reactions: List of loaded reactions

    Returns:
        List of Subsystem entities (all with category="Uncategorized")
    """
    subsystem_data = {}

    for reaction in reactions:
        if reaction.subsystem_id:
            if reaction.subsystem_id not in subsystem_data:
                # Get the original subsystem name from the first reaction
                original_name = getattr(
                    reaction, "_original_subsystem_name", reaction.subsystem_id
                )
                subsystem_data[reaction.subsystem_id] = {
                    "reaction_ids": set(),
                    "name": original_name,  # Use original name for compatibility
                }
            subsystem_data[reaction.subsystem_id]["reaction_ids"].add(reaction.id)

    subsystems = []
    for subsystem_id, data in subsystem_data.items():
        # Categories are assigned later by CategoryManager.load_categories()
        subsystem = Subsystem(
            id=subsystem_id,
            name=data["name"],
            category="Uncategorized",  # Will be updated by CategoryManager
            reaction_ids=data["reaction_ids"],
            reaction_count=len(data["reaction_ids"]),
        )

        # Note: gene_ids and metabolite_ids sets will be populated
        # by the build_subsystem_content_relationships() post-processing function

        subsystems.append(subsystem)

    return subsystems


def create_compartments_from_entities(
    reactions: List[Reaction], metabolites: List[Metabolite], genes: List[Gene]
) -> List[Compartment]:
    """Create compartment entities from entity compartment information.

    Args:
        reactions: List of loaded reactions
        metabolites: List of loaded metabolites
        genes: List of loaded genes

    Returns:
        List of Compartment entities
    """
    compartment_data = {}

    # Collect compartments from all entities
    for reaction in reactions:
        if reaction.compartment_id:
            comp_id = reaction.compartment_id
            if comp_id not in compartment_data:
                compartment_data[comp_id] = {
                    "reaction_ids": set(),
                    "metabolite_ids": set(),
                    "gene_ids": set(),
                    "name": (comp_id),
                }
            compartment_data[comp_id]["reaction_ids"].add(reaction.id)

    for metabolite in metabolites:
        for comp_id in metabolite.compartment_ids:
            if comp_id not in compartment_data:
                compartment_data[comp_id] = {
                    "reaction_ids": set(),
                    "metabolite_ids": set(),
                    "gene_ids": set(),
                    "name": (comp_id),
                }
            compartment_data[comp_id]["metabolite_ids"].add(metabolite.id)

    for gene in genes:
        for comp_id in gene.compartments:
            if comp_id not in compartment_data:
                compartment_data[comp_id] = {
                    "reaction_ids": set(),
                    "metabolite_ids": set(),
                    "gene_ids": set(),
                    "name": (comp_id),
                }
            compartment_data[comp_id]["gene_ids"].add(gene.id)

    compartments = []
    for comp_id, data in compartment_data.items():
        compartment = Compartment(
            id=comp_id,
            name=data["name"],
            reaction_ids=data["reaction_ids"],
            metabolite_ids=data["metabolite_ids"],
            gene_ids=data["gene_ids"],
            reaction_count=len(data["reaction_ids"]),
            metabolite_count=len(data["metabolite_ids"]),
            gene_count=len(data["gene_ids"]),
        )
        compartments.append(compartment)

    return compartments


def load_subsystem_metabolite_classification(
    subsystems: List[Subsystem],
    metabolites: List[Metabolite],
    general_intermediate_metabolites: List[str],
) -> Dict[str, Dict[str, List[str]]]:
    """Load per-subsystem metabolite classification.

    This function classifies metabolites assigned to each subsystem as either
    'main' or 'intermediate' using the configured general intermediate
    metabolite list.

    Args:
        subsystems: List of subsystem entities to classify metabolites for
        metabolites: List of all metabolite entities (for ID mapping)
        general_intermediate_metabolites: List of general intermediate metabolite names

    Returns:
        Dictionary mapping subsystem names to classification dictionaries
        containing "main" and "intermediate" metabolite lists (in primary format)
    """
    classification = {}

    # Create metabolite lookup maps for efficient access
    metabolite_by_id = {met.id: met for met in metabolites}

    print("3. Loading per-subsystem metabolite classification...")
    processed_count = 0

    for subsystem in subsystems:
        # Get all metabolites assigned to this subsystem
        subsystem_metabolites = [
            metabolite_by_id[met_id]
            for met_id in subsystem.metabolite_ids
            if met_id in metabolite_by_id
        ]

        main_metabolites = []
        intermediate_metabolites = []

        for metabolite in subsystem_metabolites:
            if metabolite.name in general_intermediate_metabolites:
                intermediate_metabolites.append(metabolite.id)
            else:
                main_metabolites.append(metabolite.id)

        classification[subsystem.id] = {
            "main": main_metabolites,
            "intermediate": intermediate_metabolites,
        }

        processed_count += 1

    print(f"   ✅ Processed {processed_count}/{len(subsystems)} subsystems")
    print(
        f"   ✅ Used intermediate-metabolite fallback classification for all "
        f"{processed_count} subsystems"
    )

    total_main = sum(len(c["main"]) for c in classification.values())
    total_intermediate = sum(len(c["intermediate"]) for c in classification.values())
    print(
        f"   ✅ Classified {total_main} main and {total_intermediate} "
        f"intermediate metabolites"
    )

    return classification


def _extract_compartment_from_metabolite(metabolite: str) -> str:
    """Extract compartment ID from metabolite name.

    Args:
        metabolite: Metabolite name like "glucose[c]" or "ATP[m]"

    Returns:
        Last letter of the metabolite string
    """
    return _get_compartment_full_name(metabolite[-1]) if metabolite else ""


# Helper functions


def _clean_subsystem_name(subsystem: str) -> str:
    """Clean and normalize subsystem names for use as IDs."""
    if not subsystem:
        return ""
    # Remove extra whitespace and normalize
    cleaned = (
        subsystem.replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace(" / ", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
        .lower()
    )
    # Replace spaces with underscores for ID
    return cleaned


def _safe_float(value) -> Optional[float]:
    """Safely convert value to float, returning None if conversion fails."""
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _map_compartment_name_to_id(compartment_name: str) -> str:
    """Map full compartment names to short IDs."""
    mapping = {
        "Cytosol": "c",
        "Mitochondria": "m",
        "Nucleus": "n",
        "Endoplasmic reticulum": "r",
        "Extracellular": "e",
        "Peroxisome": "x",
        "Golgi": "g",
        "Golgi apparatus": "g",
        "Lysosome": "l",
    }
    return mapping.get(compartment_name, compartment_name.lower()[:1])


def _get_compartment_full_name(comp_id: str) -> str:
    """Get full compartment name from short ID."""
    mapping = {
        "c": "Cytosol",
        "m": "Mitochondria",
        "n": "Nucleus",
        "r": "Endoplasmic reticulum",
        "e": "Extracellular",
        "x": "Peroxisome",
        "g": "Golgi apparatus",
        "l": "Lysosome",
    }
    return mapping.get(comp_id, comp_id)


def _parse_gene_association(gene_rule: str) -> List[str]:
    """Parse GPR (Gene-Protein-Reaction) expression to extract individual gene IDs.

    Args:
        gene_rule: GPR expression like "ENSG00000001 or ENSG00000002" or
                  "(ENSG00000001 and ENSG00000002) or ENSG00000003"

    Returns:
        List of unique gene IDs (ENSG identifiers)

    Examples:
        >>> _parse_gene_association("ENSG00000147576 or ENSG00000172955")
        ['ENSG00000147576', 'ENSG00000172955']

        >>> _parse_gene_association(
        ...     "(ENSG00000111058 or ENSG00000154930) and ENSG00000999999"
        ... )
        ['ENSG00000111058', 'ENSG00000154930', 'ENSG00000999999']
    """
    if not gene_rule or pd.isna(gene_rule):
        return []

    # Convert to string and clean
    gene_rule = str(gene_rule).strip()
    if not gene_rule:
        return []

    # Extract all ENSG gene IDs using regex
    # Pattern matches ENSG followed by 11 digits
    gene_pattern = r"ENSG\d{11}"
    gene_matches = re.findall(gene_pattern, gene_rule)

    # Return unique gene IDs
    return list(set(gene_matches))


def build_metabolite_reaction_relationships(
    metabolites: List[Metabolite], reactions: List[Reaction]
) -> None:
    """Build metabolite-reaction relationships using internal IDs.

    This function populates the produced_by and consumed_by lists in metabolite
    entities by cross-referencing with reaction reactants and products.

    Args:
        metabolites: List of metabolite entities to update
        reactions: List of reaction entities with reactants/products

    Note:
        This function modifies metabolites in-place and uses internal entity IDs
        for all relationships to avoid translation overhead.
    """
    # Create mapping from metabolite ID to metabolite object for efficient lookup
    metabolite_map = {met.id: met for met in metabolites}

    print(
        f"Building metabolite-reaction relationships for {len(reactions)} reactions..."
    )

    relationships_built = 0
    metabolites_updated = 0

    for reaction in reactions:
        # Process reactants (metabolites consumed by this reaction)
        for reactant_name in reaction.reactants:
            # Find matching metabolite by trying different ID matching strategies
            matching_metabolite = _find_matching_metabolite(
                reactant_name, metabolite_map
            )

            if matching_metabolite:
                # This reaction consumes this metabolite
                if reaction.id not in matching_metabolite.consumed_by:
                    matching_metabolite.consumed_by.append(reaction.id)
                    relationships_built += 1

                if reaction.is_reversible and reaction.id not in matching_metabolite.produced_by:
                    matching_metabolite.produced_by.append(reaction.id)
                    # relationships_built += 1

        # Process products (metabolites produced by this reaction)
        for product_name in reaction.products:
            # Find matching metabolite by trying different ID matching strategies
            matching_metabolite = _find_matching_metabolite(
                product_name, metabolite_map
            )

            if matching_metabolite:
                # This reaction produces this metabolite
                if reaction.id not in matching_metabolite.produced_by:
                    matching_metabolite.produced_by.append(reaction.id)
                    relationships_built += 1

                if reaction.is_reversible and reaction.id not in matching_metabolite.consumed_by:
                    matching_metabolite.consumed_by.append(reaction.id)
                    # relationships_built += 1

    # Count metabolites that got updated
    for metabolite in metabolites:
        if metabolite.produced_by or metabolite.consumed_by:
            metabolites_updated += 1

    print(f"  ✅ Built {relationships_built} metabolite-reaction relationships")
    print(f"  ✅ Updated {metabolites_updated}/{len(metabolites)} metabolites")


def _find_matching_metabolite(
    metabolite_name: str, metabolite_map: dict
) -> Optional[Metabolite]:
    """Find matching metabolite entity for a given metabolite name from reactions.

    This function handles the mapping between metabolite names in reactions
    (like "glucose[c]" or "ATP[m]") and the actual metabolite IDs in the
    metabolite identifier crosswalk file (like "MAM00001c").

    Args:
        metabolite_name: Metabolite name from reaction (e.g., "glucose[c]", "ATP[m]")
        metabolite_map: Dictionary mapping metabolite IDs to metabolite objects

    Returns:
        Matching Metabolite object or None if not found

    Note:
        The matching logic needs to handle the difference between:
        - Reaction metabolite names: "glucose[c]", "ATP[m]" (human-readable)
        - Metabolite file IDs: "MAM00001c", "MAM02552m" (internal IDs)

        This is a complex mapping that may require additional lookup tables
        or fuzzy matching in the future.
    """
    if not metabolite_name:
        return None

    # Strategy 1: Direct ID match (unlikely but worth trying)
    if metabolite_name in metabolite_map:
        return metabolite_map[metabolite_name]

    # Strategy 2: Extract compartment and try to match by name + compartment
    compartment = _extract_compartment_from_metabolite(metabolite_name)
    base_name = (
        metabolite_name.split("[")[0] if "[" in metabolite_name else metabolite_name
    )

    # Try to find metabolite by name and compartment
    for metabolite_id, metabolite in metabolite_map.items():
        # Check if compartments match
        if compartment and compartment in metabolite.compartment_ids:
            # Check if the base name matches the metabolite name (case-insensitive)
            if (
                metabolite.name.lower() == base_name.lower()
                or base_name.lower() in metabolite.name.lower()
            ):
                return metabolite

    # Strategy 3: Try matching without compartment consideration
    for metabolite_id, metabolite in metabolite_map.items():
        if (
            metabolite.name.lower() == base_name.lower()
            or base_name.lower() in metabolite.name.lower()
        ):
            return metabolite

    # TODO: Strategy 4: Use ID mapping tables for more sophisticated matching
    # This would require cross-referencing with external database IDs
    # like BiGG, KEGG, etc. that might appear in both datasets

    # No match found
    return None


def build_gene_reaction_associations(
    genes: List[Gene], reactions: List[Reaction]
) -> None:
    """Build gene-reaction associations using parsed gene data from reactions.

    This function cross-references the parsed gene associations from reactions
    with the gene entities to populate the associated_reactions field.

    Args:
        genes: List of Gene entities to update (modified in-place)
        reactions: List of Reaction entities with parsed gene associations

    Note:
        Uses internal entity IDs for all relationships to avoid translation overhead.
    """
    # Create mapping from gene ID to gene entity for efficient lookup
    gene_map = {gene.id: gene for gene in genes}

    # Clear existing associations
    for gene in genes:
        gene.associated_reactions = []

    # Process each reaction and build gene associations
    reaction_gene_pairs = 0
    for reaction in reactions:
        if reaction.associated_genes:
            for gene_id in reaction.associated_genes:
                if gene_id in gene_map:
                    # Add reaction ID to gene's associated_reactions list
                    gene_map[gene_id].associated_reactions.append(reaction.id)
                    reaction_gene_pairs += 1

    # Sort gene associations for consistency
    for gene in genes:
        gene.associated_reactions.sort()

    print(f"   Built {reaction_gene_pairs} gene-reaction associations")


def build_subsystem_content_relationships(
    subsystems: List[Subsystem],
    reactions: List[Reaction],
    genes: List[Gene],
    metabolites: List[Metabolite],
) -> None:
    """Build subsystem content relationships (genes and metabolites).

    This function populates the gene_ids and metabolite_ids sets in subsystem
    entities by cross-referencing with reactions in each subsystem.

    Args:
        subsystems: List of Subsystem entities to update (modified in-place)
        reactions: List of Reaction entities with subsystem assignments
        genes: List of Gene entities with reaction associations
        metabolites: List of Metabolite entities with reaction relationships

    Note:
        Uses internal entity IDs for all relationships to avoid translation overhead.
    """
    print(f"Building subsystem content for {len(subsystems)} subsystems...")

    # Create mapping from reaction ID to reaction entity
    reaction_map = {reaction.id: reaction for reaction in reactions}

    total_gene_assignments = 0
    total_metabolite_assignments = 0

    # Process each subsystem
    for subsystem in subsystems:
        # Clear existing content
        subsystem.gene_ids = set()
        subsystem.metabolite_ids = set()

        # Process each reaction in this subsystem
        for reaction_id in subsystem.reaction_ids:
            if reaction_id in reaction_map:
                reaction = reaction_map[reaction_id]

                # Add genes associated with this reaction
                for gene_id in reaction.associated_genes:
                    subsystem.gene_ids.add(gene_id)
                    total_gene_assignments += 1

                # Add metabolites involved in this reaction
                for metabolite_name in reaction.reactants + reaction.products:
                    # Find matching metabolite ID
                    matching_metabolite = _find_matching_metabolite(
                        metabolite_name, {met.id: met for met in metabolites}
                    )
                    if matching_metabolite:
                        subsystem.metabolite_ids.add(matching_metabolite.id)
                        total_metabolite_assignments += 1

        # Update counts
        subsystem.gene_count = len(subsystem.gene_ids)
        subsystem.metabolite_count = len(subsystem.metabolite_ids)

    # Calculate statistics
    subsystems_with_genes = sum(1 for s in subsystems if s.gene_ids)
    subsystems_with_metabolites = sum(1 for s in subsystems if s.metabolite_ids)

    print("   ✅ Built subsystem content relationships")
    print(f"   ✅ {subsystems_with_genes}/{len(subsystems)} subsystems have genes")
    print(
        f"   ✅ {subsystems_with_metabolites}/{len(subsystems)} "
        f"subsystems have metabolites"
    )

    unique_genes = len(set(g for s in subsystems for g in s.gene_ids))
    unique_metabolites = len(set(m for s in subsystems for m in s.metabolite_ids))
    print(f"   ✅ Total gene assignments: {unique_genes}")
    print(f"   ✅ Total metabolite assignments: {unique_metabolites}")


def build_subsystem_neighbor_relationships(
    reactions: List[Reaction],
    subsystems: List[Subsystem],
    metabolites: List[Metabolite],
    metabolite_classification: Dict[str, Dict[str, str]],
) -> List[SubsystemNeighborRelationship]:
    """Build directional neighbor relationships between subsystems.

    Creates relationships where subsystem A is a neighbor of subsystem B
    if A produces a main metabolite that B consumes. Uses the metabolite
    objects' existing produced_by/consumed_by relationships for efficiency.

    Args:
        reactions: List of all reactions
        subsystems: List of all subsystems
        metabolites: List of all metabolites
        metabolite_classification: Dict[subsystem_id][metabolite_id] = classification

    Returns:
        List of SubsystemNeighborRelationship objects
    """
    print("🔗 Building subsystem neighbor relationships...")

    # Create mappings for quick lookup
    reaction_by_id = {r.id: r for r in reactions}
    metabolite_by_id = {m.id: m for m in metabolites}
    subsystem_by_id = {s.id: s for s in subsystems}

    relationships = []

    # For each subsystem, check its metabolites for cross-subsystem relationships
    for current_subsystem in subsystems:
        current_classification = metabolite_classification.get(current_subsystem.id, {})

        # Process each metabolite in this subsystem
        for metabolite_id in current_subsystem.metabolite_ids:
            # Only process main metabolites
            if metabolite_id not in current_classification["main"]:
                continue

            metabolite = metabolite_by_id.get(metabolite_id)
            if not metabolite:
                continue

            # Check if current subsystem produces this main metabolite
            current_subsystem_produces = False
            producing_reactions = []

            for reaction_id in metabolite.produced_by:
                reaction = reaction_by_id.get(reaction_id)
                if reaction and reaction.subsystem_id == current_subsystem.id:
                    current_subsystem_produces = True
                    producing_reactions.append(reaction_id)

            # If current subsystem produces this main metabolite,
            # check for consuming subsystems
            if current_subsystem_produces:
                for reaction_id in metabolite.consumed_by:
                    reaction = reaction_by_id.get(reaction_id)
                    if not reaction or reaction.subsystem_id == current_subsystem.id:
                        continue  # Skip if same subsystem or invalid reaction

                    consuming_subsystem_id = reaction.subsystem_id
                    consuming_subsystem = subsystem_by_id.get(consuming_subsystem_id)

                    if not consuming_subsystem:
                        continue

                    # Check if this metabolite is main for the consuming subsystem
                    consuming_classification = metabolite_classification.get(
                        consuming_subsystem_id, {}
                    )
                    if metabolite_id in consuming_classification["main"]:
                        # Get all consuming reactions for this subsystem
                        consuming_reactions = [
                            r_id
                            for r_id in metabolite.consumed_by
                            if reaction_by_id.get(r_id)
                            and reaction_by_id.get(r_id).subsystem_id
                            == consuming_subsystem_id
                        ]

                        # Create the relationship
                        relationship = SubsystemNeighborRelationship(
                            producing_subsystem_id=current_subsystem.id,
                            consuming_subsystem_id=consuming_subsystem_id,
                            metabolite_id=metabolite_id,
                            producing_reactions=producing_reactions,
                            consuming_reactions=consuming_reactions,
                        )

                        # Avoid duplicates
                        if relationship not in relationships:
                            relationships.append(relationship)

    print(f"   ✅ Found {len(relationships)} neighbor relationships")

    # Print some statistics
    if relationships:
        producing_subsystems = len(set(r.producing_subsystem_id for r in relationships))
        consuming_subsystems = len(set(r.consuming_subsystem_id for r in relationships))
        unique_metabolites = len(set(r.metabolite_id for r in relationships))

        print(f"   ✅ {producing_subsystems} subsystems produce metabolites for others")
        print(
            f"   ✅ {consuming_subsystems} subsystems consume metabolites from others"
        )
        print(f"   ✅ {unique_metabolites} unique metabolites mediate relationships")

    return relationships


def _translate_to_svg_format(metabolite_id: str) -> str:
    """Convert model metabolite ID to SVG format.

    Args:
        metabolite_id: Model metabolite ID (e.g., "MAM01802c")

    Returns:
        SVG metabolite ID (e.g., "m01802c")
    """
    if metabolite_id.startswith("MAM"):
        # Convert MAM01802c -> m01802c
        return "m" + metabolite_id[3:]
    return metabolite_id  # Return as-is if not in expected format

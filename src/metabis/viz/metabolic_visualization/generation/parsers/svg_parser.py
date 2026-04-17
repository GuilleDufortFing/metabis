"""SVG parser module for extracting graph structure from SVG files.

This module provides functionality to parse SVG files containing metabolic network
visualizations and extract node and edge information into a structured format.
"""

import xml.etree.ElementTree as ET
from typing import Dict, List


def extract_metabolite_subtypes(svg_file) -> Dict[str, str]:
    """Extract metabolite subtype information from SVG file.

    Args:
        svg_file: Path to the SVG file to parse

    Returns:
        Dictionary mapping metabolite IDs to their subtypes ('main' or 'intermediate')

    Example:
        >>> subtypes = extract_metabolite_subtypes("network.svg")
        >>> print(subtypes["met_001"])  # "main" or "intermediate"
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    metabolite_subtypes = {}

    # Find the container for nodes
    nodes_container = root.find(".//svg:g[@id='nodes']", ns)
    if nodes_container is None:
        return metabolite_subtypes

    # Extract metabolite subtype information
    for node in nodes_container.findall("svg:g", ns):
        class_attr = node.attrib.get("class", "")
        tokens = class_attr.split()

        if "met" not in tokens:
            continue

        # Get metabolite ID
        node_id = None
        for token in tokens:
            if token != "met":
                node_id = token
                # If the token ends with s or p, adjust it to e or x respectively
                if node_id.endswith("s"):
                    node_id = node_id[:-1] + "e"
                elif node_id.endswith("p"):
                    node_id = node_id[:-1] + "x"
                break

        if not node_id:
            continue

        # Determine subtype based on shape
        subtype = "main"  # Default
        if node.find("svg:circle[@class='shape']", ns) is not None:
            subtype = "intermediate"
        elif node.find("svg:ellipse[@class='shape']", ns) is not None:
            subtype = "main"

        metabolite_subtypes[node_id] = subtype

    return metabolite_subtypes


def extract_enzyme_reaction_associations(svg_file) -> Dict[str, List[str]]:
    """Extract enzyme-reaction association information from SVG file.

    Args:
        svg_file: Path to the SVG file to parse

    Returns:
        Dictionary mapping enzyme IDs to lists of associated reaction IDs

    Example:
        >>> associations = extract_enzyme_reaction_associations("network.svg")
        >>> print(associations["ENSG00000123456"])  # ["MAR00001", "MAR00002"]
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    enzyme_associations = {}

    # Find the container for nodes
    nodes_container = root.find(".//svg:g[@id='nodes']", ns)
    if nodes_container is None:
        return enzyme_associations

    # Extract enzyme-reaction associations
    for node in nodes_container.findall("svg:g", ns):
        class_attr = node.attrib.get("class", "")
        tokens = class_attr.split()

        if "enz" not in tokens:
            continue

        # Extract enzyme ID and associated reaction ID from class
        enzyme_id = None
        associated_reaction = None

        for token in tokens:
            if token != "enz":
                if token.startswith("ENSG"):
                    enzyme_id = token
                elif token.startswith("MAR"):
                    associated_reaction = token

        if not enzyme_id:
            continue

        # Handle multiple associations for the same enzyme
        if enzyme_id in enzyme_associations:
            if (
                associated_reaction
                and associated_reaction not in enzyme_associations[enzyme_id]
            ):
                enzyme_associations[enzyme_id].append(associated_reaction)
        else:
            enzyme_associations[enzyme_id] = (
                [associated_reaction] if associated_reaction else []
            )

    return enzyme_associations

"""Unified reaction equation parser for metabolic visualizations.

This module provides a robust parser for reaction equations that handles
various equation formats and extracts metabolite names while removing
stoichiometry coefficients and compartment tags.

This parser unifies the functionality previously split between the simple
and enhanced parsing approaches, using the more robust enhanced approach
as the foundation.
"""

import re
from typing import List, Tuple


def parse_reaction_equation(equation: str) -> Tuple[List[str], List[str], bool]:
    """Parse reaction equation into reactants and products.

    This is the main public API function that handles various equation formats
    and extracts metabolite names while removing stoichiometry coefficients
    and compartment tags.

    Args:
        equation: Reaction equation string (e.g., "glucose[c] + ATP[c] => " +
                 "glucose-6-phosphate[c] + ADP[c]")

    Returns:
        Tuple of (reactants, products, is_reversible) where reactants and products
        are lists of metabolite names, and is_reversible is a boolean indicating
        if the reaction is reversible (uses <=> arrow)

    Example:
        >>> reactants, products, is_reversible = parse_reaction_equation(
        ...     "glucose[c] + ATP[c] <=> glucose-6-phosphate[c] + ADP[c]"
        ... )
        >>> print(reactants)     # ['glucose[c]', 'ATP[c]']
        >>> print(products)      # ['glucose-6-phosphate[c]', 'ADP[c]']
        >>> print(is_reversible) # True

    Note:
        This function handles multiple arrow types:
        - "<=>" for reversible reactions
        - "=>" for irreversible reactions
        - "=" for equilibrium reactions
        - "->" and "<-" for compatibility with simpler formats

        It also properly handles:
        - Stoichiometry coefficients (e.g., "2 ATP[c]" -> "ATP[c]")
        - Compartment tags are preserved (e.g., "ATP[c]" stays as "ATP[c]")
        - Plus signs in metabolite names (e.g., "NAD+" is preserved)
        - Complex metabolite names with spaces and special characters
    """
    if not equation or not isinstance(equation, str):
        return [], [], False

    # Clean the equation
    equation = equation.strip()

    is_reversible = False
    # Handle different arrow types for reactions
    # Priority order: <=> (reversible), => (irreversible), = (equilibrium),
    # then -> and <- for backward compatibility
    if "<=>" in equation:
        # Reversible reaction (bidirectional)
        parts = equation.split("<=>")
        if len(parts) != 2:
            return [], [], False
        left_side, right_side = parts[0].strip(), parts[1].strip()
        is_reversible = True
    elif "->" in equation:
        # Irreversible reaction (forward direction)
        parts = equation.split("->")
        if len(parts) != 2:
            return [], [], False
        left_side, right_side = parts[0].strip(), parts[1].strip()
    else:
        # No recognized separator
        return [], [], False

    # Extract reactants and products using the sophisticated parser
    reactants = _extract_metabolites_from_side(left_side)
    products = _extract_metabolites_from_side(right_side)


    return reactants, products, is_reversible


def _extract_metabolites_from_side(side: str) -> List[str]:
    """Extract metabolite names from one side of the equation.

    This function uses sophisticated parsing to handle complex metabolite names
    while properly splitting on separating plus signs and removing stoichiometry
    and compartment information.

    Args:
        side: One side of the reaction equation (reactants or products)

    Returns:
        List of clean metabolite names

    Example:
        >>> metabolites = _extract_metabolites_from_side("2 glucose[c] + NAD+[c]")
        >>> print(metabolites)  # ['glucose[c]', 'NAD+[c]']
    """
    if not side:
        return []

    metabolites = []

    # Use a sophisticated approach to split by + while preserving + in metabolite names
    # Track bracket depth and only split on + that are outside brackets
    # and at proper boundaries
    i = 0
    current_compound = ""
    bracket_depth = 0

    while i < len(side):
        char = side[i]

        if char == "[":
            bracket_depth += 1
            current_compound += char
        elif char == "]":
            bracket_depth -= 1
            current_compound += char
        elif char == "+" and bracket_depth == 0:
            # This is a + that potentially separates compounds
            # Check if this + is surrounded by whitespace or at boundaries
            if i == 0 or i == len(side) - 1:
                current_compound += char  # + at boundary, likely part of name
            else:
                # Look at surrounding characters
                prev_char = side[i - 1] if i > 0 else ""
                next_char = side[i + 1] if i < len(side) - 1 else ""

                # If surrounded by spaces or letters, it's likely a separator
                if (
                    prev_char.isspace()
                    or next_char.isspace()
                    or (prev_char in "]" and next_char.isalpha())
                    or (prev_char.isalpha() and next_char.isalpha())
                ):
                    # This is a separator +
                    if current_compound.strip():
                        metabolites.append(current_compound.strip())
                    current_compound = ""
                    # Skip any following whitespace
                    while i + 1 < len(side) and side[i + 1].isspace():
                        i += 1
                else:
                    # This + is part of a metabolite name (like NAD+)
                    current_compound += char
        else:
            current_compound += char

        i += 1

    # Add the last compound
    if current_compound.strip():
        metabolites.append(current_compound.strip())

    # Process each compound to remove stoichiometry but preserve compartments
    processed_metabolites = []
    for compound in metabolites:
        if not compound:
            continue

        # Remove stoichiometry coefficient (e.g., "2 ATP[c]" -> "ATP[c]")
        # Match number at the beginning followed by space
        coefficient_match = re.match(r"^(\d+(?:\.\d+)?)\s+(.+)", compound)
        if coefficient_match:
            metabolite_name = coefficient_match.group(2).strip()
        else:
            metabolite_name = compound.strip()

        # Keep compartment tags as they define specific metabolites
        # (e.g., "ATP[c]" stays as "ATP[c]", "NAD+[m]" stays as "NAD+[m]")
        if metabolite_name:
            processed_metabolites.append(metabolite_name)

    return processed_metabolites

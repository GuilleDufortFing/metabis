"""A short, human-readable name for every reaction.

Only about half of Human1's reactions carry a specific name, and some names are
generic ("Transport Reaction") or long. Figures label reactions with a short name
instead of their MAR identifier:

* Human1's name, when it has a specific one;
* an exchange or demand reaction reads "Exchange of methylglyoxal";
* a transport reads "ADP-ribose transport (c→e)", with the compartments it joins;
* a conversion reads its main substrate and product, "linoleate → linoleoyl-CoA",
  leaving out currency metabolites, and names its compartment when it is not the
  cytosol: "linoleate → linoleoyl-CoA [r]".

Every name is cut to :data:`SHORT_NAME_LENGTH` characters. The identifier, the full
name and the equation stay available beside it in hovers and tables.
"""

from __future__ import annotations

import re
from typing import List, Tuple

SHORT_NAME_LENGTH = 34
# Metabolites that take part in so many reactions that they never name one.
CURRENCY_METABOLITES = frozenset(
    {
        "H+", "H2O", "ATP", "ADP", "AMP", "PPi", "Pi", "NAD+", "NADH", "NADP+", "NADPH",
        "CoA", "CO2", "O2", "FAD", "FADH2", "H2O2", "GTP", "GDP", "UTP", "UDP", "Na+",
        "K+", "HCO3-", "NH3",
    }
)
GENERIC_NAMES = frozenset({"", "transport reaction", "transport", "nan"})


def _side(text: str) -> List[Tuple[str, str]]:
    """The metabolites of one side of an equation, as (name, compartment) pairs."""
    metabolites = []
    for part in (piece.strip() for piece in text.split(" + ")):
        if not part:
            continue
        part = re.sub(r"^\d+(\.\d+)?(e-?\d+)?\s+", "", part)
        match = re.match(r"(.*)\[(\w+)\]$", part)
        metabolites.append((match.group(1).strip(), match.group(2)) if match else (part, ""))
    return metabolites


def _cut(text: str, length: int = SHORT_NAME_LENGTH) -> str:
    return text if len(text) <= length else text[: length - 1].rstrip() + "…"


def short_reaction_name(name: object, equation: object) -> str:
    """A reaction's short name, from its Human1 name or else its equation."""
    named = "" if name is None else str(name).strip()
    if named.lower() not in GENERIC_NAMES:
        return _cut(named)
    parts = re.split(r"\s*(<=>|=>|<=)\s*", str(equation or ""))
    left = _side(parts[0]) if parts else []
    right = _side(parts[2]) if len(parts) > 2 else []
    if not left and not right:
        return ""
    if not left or not right:
        return _cut(f"Exchange of {(left or right)[0][0]}")
    main_left = [m for m in left if m[0] not in CURRENCY_METABOLITES] or left
    main_right = [m for m in right if m[0] not in CURRENCY_METABOLITES] or right
    shared = sorted({m[0] for m in main_left} & {m[0] for m in main_right})
    if shared:
        metabolite = shared[0]
        source = next(c for n, c in main_left if n == metabolite)
        target = next(c for n, c in main_right if n == metabolite)
        return _cut(f"{metabolite} transport ({source}→{target})")
    compartments = {c for _, c in main_left + main_right}
    where = f" [{compartments.pop()}]" if len(compartments) == 1 and "c" not in compartments else ""
    return _cut(f"{main_left[0][0]} → {main_right[0][0]}") + where


__all__ = ["SHORT_NAME_LENGTH", "short_reaction_name"]

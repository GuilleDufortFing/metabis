"""Packaged compartment graph resource manifest and lookup helpers."""

from typing import Dict, List

COMPARTMENT_GRAPH_VARIANT = "human1"


COMPARTMENT_GRAPH_MANIFEST: Dict[str, Dict[str, str]] = {
    "cytosol_1": {
        "display_label": "Cytosol 1",
        "display_title": "Compartment: Cytosol 1",
    },
    "cytosol_2": {
        "display_label": "Cytosol 2",
        "display_title": "Compartment: Cytosol 2",
    },
    "cytosol_3": {
        "display_label": "Cytosol 3",
        "display_title": "Compartment: Cytosol 3",
    },
    "cytosol_4": {
        "display_label": "Cytosol 4",
        "display_title": "Compartment: Cytosol 4",
    },
    "cytosol_5": {
        "display_label": "Cytosol 5",
        "display_title": "Compartment: Cytosol 5",
    },
    "endoplasmic_reticulum": {
        "display_label": "Endoplasmic reticulum",
        "display_title": "Compartment: Endoplasmic reticulum",
    },
    "golgi_apparatus": {
        "display_label": "Golgi apparatus",
        "display_title": "Compartment: Golgi apparatus",
    },
    "lysosome": {
        "display_label": "Lysosome",
        "display_title": "Compartment: Lysosome",
    },
    "mitochondria": {
        "display_label": "Mitochondria",
        "display_title": "Compartment: Mitochondria",
    },
    "nucleus": {
        "display_label": "Nucleus",
        "display_title": "Compartment: Nucleus",
    },
    "peroxisome": {
        "display_label": "Peroxisome",
        "display_title": "Compartment: Peroxisome",
    },
}


COMPARTMENT_GRAPH_ALIASES = {
    "golgi": "golgi_apparatus",
}


def _normalize_compartment_lookup_key(element_name: str) -> str:
    return (
        str(element_name)
        .strip()
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace(" / ", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
        .lower()
    )


def list_compartment_resource_ids() -> List[str]:
    """Return the supported packaged compartment resource IDs."""
    return sorted(COMPARTMENT_GRAPH_MANIFEST)


def get_compartment_display_title(resource_id: str) -> str:
    """Return the human-readable plot title for a compartment resource."""
    return COMPARTMENT_GRAPH_MANIFEST[resource_id]["display_title"]


def normalize_compartment_resource_id(element_name: str) -> str:
    """Resolve a user-provided compartment name to a packaged resource ID."""
    available_ids = ", ".join(list_compartment_resource_ids())
    if element_name is None or not str(element_name).strip():
        raise ValueError(
            "Compartment graph requests require an explicit packaged resource ID. "
            f"Available compartment resources: {available_ids}"
        )

    normalized = _normalize_compartment_lookup_key(element_name)
    if normalized in COMPARTMENT_GRAPH_MANIFEST:
        return normalized

    alias = COMPARTMENT_GRAPH_ALIASES.get(normalized)
    if alias:
        return alias

    if normalized == "cytosol":
        raise ValueError(
            "Cytosol is split across five packaged Human1 compartment graphs. "
            "Use one of: cytosol_1, cytosol_2, cytosol_3, cytosol_4, cytosol_5"
        )

    if normalized == "extracellular":
        raise ValueError(
            "No packaged Human1 compartment graph is available for extracellular. "
            f"Available compartment resources: {available_ids}"
        )

    raise ValueError(
        f"Unknown compartment graph resource '{element_name}'. "
        f"Available compartment resources: {available_ids}"
    )

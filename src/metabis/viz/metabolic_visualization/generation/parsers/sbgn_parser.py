"""SBGN parser module for extracting graph structure from SBGN files.

This module provides functionality to parse SBGN (Systems Biology Graphical Notation)
files and extract raw graph structure data. The primary interface is
extract_raw_sbgn_data() which returns minimal data structures that can be converted
to domain objects by higher-level generators.

Key features:
- Raw data extraction without domain object creation
- HMR to MAR ID translation for reaction nodes
- Type-safe data structures with validation

Recommended usage:
    raw_glyphs, raw_connections = extract_raw_sbgn_data(sbgn_file, id_mapping)

"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple, TypedDict

import pandas as pd

# Note: Factory imports kept only for backward compatibility functions


# Raw data structures for SBGN parsing output
class RawGlyphData(TypedDict):
    """Raw glyph data extracted from SBGN XML.

    This represents the minimal data extracted from SBGN before
    any business logic or object creation is applied.
    """

    id: str  # Final ID after any translation
    xml_id: str  # Original XML ID from SBGN
    label: str  # Display label
    glyph_type: str  # Type: compartment, metabolite, gene, reaction
    x: float  # Center X coordinate
    y: float  # Center Y coordinate
    width: float  # Glyph width
    height: float  # Glyph height


class RawConnectionData(TypedDict):
    """Raw connection data extracted from SBGN XML.

    This represents the minimal connection data extracted from SBGN
    before any business logic or object creation is applied.
    """

    id: str  # Connection ID
    source: str  # Source glyph ID (after translation)
    target: str  # Target glyph ID (after translation)
    connection_type: str  # Type of connection (always 'metabolic' from SBGN)


class IDTranslationManager:
    """Manages ID translation from original SBGN XML IDs to final glyph IDs.

    This class ensures consistent ID translation throughout the parsing process,
    particularly for HMR to MAR reaction ID conversion. It works with raw data
    to maintain separation of concerns.
    """

    def __init__(self, hmr_to_mar_mapping: Optional[Dict[str, str]] = None):
        """Initialize the ID translation manager.

        Args:
            hmr_to_mar_mapping: Optional mapping from HMR to MAR reaction IDs
        """
        self.hmr_to_mar_mapping = hmr_to_mar_mapping or {}
        self.xml_id_to_final_id: Dict[str, str] = {}
        self.final_id_to_xml_id: Dict[str, str] = {}

    def register_glyph_translation(
        self, xml_id: str, label: str, glyph_type: str
    ) -> str:
        """Register a glyph's ID translation and return the final ID.

        Args:
            xml_id: Original ID from SBGN XML
            label: Glyph label (potentially in HMR format for reactions)
            glyph_type: Type of glyph (reaction, metabolite, gene, compartment)

        Returns:
            Final ID to use for the glyph
        """
        if glyph_type == "reaction":
            # For reactions, use translated label as final ID
            final_id = self._translate_reaction_id(label)
        else:
            # For non-reactions, use original XML ID
            final_id = xml_id

        # Store bidirectional mapping
        self.xml_id_to_final_id[xml_id] = final_id
        self.final_id_to_xml_id[final_id] = xml_id

        return final_id

    def translate_connection_reference(self, xml_id: str) -> str:
        """Translate a connection reference from XML ID to final ID.

        Args:
            xml_id: Original ID from SBGN XML (source or target)

        Returns:
            Final ID that should be used in connections
        """
        return self.xml_id_to_final_id.get(xml_id, xml_id)

    def _translate_reaction_id(self, hmr_label: str) -> str:
        """Translate reaction ID from HMR to MAR format.

        Args:
            hmr_label: Original HMR reaction label

        Returns:
            MAR reaction ID if mapping exists, otherwise original label
        """
        if self.hmr_to_mar_mapping and hmr_label in self.hmr_to_mar_mapping:
            return self.hmr_to_mar_mapping[hmr_label]
        return hmr_label


def extract_reaction_id_mapping(hmr_mar_file: str) -> Dict[str, str]:
    """Extract reaction ID mapping from HMR to MAR format.

    Args:
        hmr_mar_file: Path to the HMR-MAR mapping file

    Returns:
        Dictionary mapping HMR reaction IDs to MAR reaction IDs

    Example:
        >>> mapping = extract_reaction_id_mapping("hmr_mar_mapping.tsv")
        >>> print(mapping["HMR_0001"])  # "MAR00001"
    """
    return _parse_hmr_mar_mapping(hmr_mar_file)


# ============================================================================
# PRIMARY INTERFACE - Raw Data Extraction (Recommended)
# ============================================================================
# These functions provide the new, clean interface for extracting raw data
# from SBGN files without creating domain objects.


def extract_raw_sbgn_data(
    sbgn_file: str, id_mapping: Optional[Dict[str, str]] = None
) -> Tuple[List[RawGlyphData], List[RawConnectionData]]:
    """Extract raw data structures from SBGN file without creating domain objects.

    This function extracts minimal data from SBGN files and returns raw data
    structures that can be used by higher-level generators to create appropriate
    domain objects.

    Args:
        sbgn_file: Path to the SBGN XML file
        id_mapping: Optional dictionary mapping HMR reaction IDs to MAR reaction IDs

    Returns:
        Tuple of (raw_glyphs, raw_connections) where:
        - raw_glyphs: List of raw glyph data dictionaries
        - raw_connections: List of raw connection data dictionaries

    Example:
        >>> raw_glyphs, raw_connections = extract_raw_sbgn_data("network.sbgn")
        >>> print(len(raw_glyphs), len(raw_connections))  # 150 75
    """
    # Parse XML
    root = _parse_sbgn_xml(sbgn_file)

    # Create ID translation manager for consistent ID handling
    translation_manager = IDTranslationManager(id_mapping)

    # Extract raw glyph data
    raw_glyphs = _extract_raw_glyphs_from_xml(root, translation_manager)

    # Extract raw connection data
    raw_connections = _extract_raw_connections_from_xml(root, translation_manager)

    # Validate the extracted raw data
    # is_valid, errors = validate_raw_sbgn_data(raw_glyphs, raw_connections)
    # if not is_valid:
    #     raise ValueError(f"Invalid SBGN data extracted: {errors}")

    return raw_glyphs, raw_connections


def _parse_sbgn_xml(sbgn_file: str):
    """Parse SBGN XML file and return the root element.

    Args:
        sbgn_file: Path to the SBGN XML file

    Returns:
        XML root element with SBGN namespace defined
    """
    tree = ET.parse(sbgn_file)
    root = tree.getroot()
    return root


def _build_port_to_parent_mapping(root) -> Dict[str, str]:
    """Build mapping from port IDs to their parent node IDs.

    Args:
        root: SBGN XML root element

    Returns:
        Dictionary mapping port IDs to parent node IDs
    """
    ns = {"sbgn": "http://sbgn.org/libsbgn/0.2"}
    port_to_parent_map = {}

    for glyph in root.findall(".//sbgn:glyph", ns):
        glyph_id = glyph.get("id")
        original_class = glyph.get("class")

        # Record mapping from port IDs to parent node ID
        if original_class == "process":
            for port in glyph.findall(".//sbgn:port", ns):
                port_id = port.get("id")
                port_to_parent_map[port_id] = glyph_id

    return port_to_parent_map


def _classify_glyph_type(original_class: str, label: str) -> str:
    """Classify glyph type based on original class and label.

    Args:
        original_class: Original SBGN class attribute
        label: Glyph label text

    Returns:
        Classified glyph type string using standardized constants
    """
    if original_class == "compartment":
        return "compartment"
    elif label.startswith("m"):
        return "metabolite"
    elif label.startswith("ENS"):
        return "gene"
    else:
        return "reaction"


def _extract_raw_glyphs_from_xml(
    root, translation_manager: IDTranslationManager
) -> List[RawGlyphData]:
    """Extract raw glyph data from SBGN XML.

    Args:
        root: SBGN XML root element
        translation_manager: ID translation manager for consistent ID handling

    Returns:
        List of raw glyph data dictionaries
    """
    ns = {"sbgn": "http://sbgn.org/libsbgn/0.2"}
    raw_glyphs = []

    for glyph in root.findall(".//sbgn:glyph", ns):
        xml_id = glyph.get("id")
        original_class = glyph.get("class")

        # Get label
        label_elem = glyph.find(".//sbgn:label", ns)
        label = label_elem.get("text") if label_elem is not None else ""

        # Determine class based on original class or label prefix
        glyph_type = _classify_glyph_type(original_class, label)

        # Register ID translation and get final ID
        final_id = translation_manager.register_glyph_translation(
            xml_id, label, glyph_type
        )

        if glyph_type == "metabolite":
            label = "MAM" + label[1:] if label.startswith("m") else label
            # If label ends with s, transform it to e
            if label.endswith("s"):
                label = label[:-1] + "e"
            # If label ends with p, transform it to x
            elif label.endswith("p"):
                label = label[:-1] + "x"

        # Get position and size
        bbox = glyph.find(".//sbgn:bbox", ns)
        if bbox is not None:
            x = float(bbox.get("x"))
            y = float(bbox.get("y"))
            w = float(bbox.get("w"))
            h = float(bbox.get("h"))

            # Use center position
            center_x = x + w / 2
            center_y = y + h / 2

            # Create raw glyph data
            raw_glyph_data: RawGlyphData = {
                "id": final_id,
                "xml_id": xml_id,
                "label": label,
                "glyph_type": glyph_type,
                "x": center_x,
                "y": center_y,
                "width": w,
                "height": h,
            }

            raw_glyphs.append(raw_glyph_data)

    return raw_glyphs


def _extract_raw_connections_from_xml(
    root, translation_manager: IDTranslationManager
) -> List[RawConnectionData]:
    """Extract raw connection data from SBGN XML.

    Args:
        root: SBGN XML root element
        translation_manager: ID translation manager for consistent ID handling

    Returns:
        List of raw connection data dictionaries
    """
    ns = {"sbgn": "http://sbgn.org/libsbgn/0.2"}
    raw_connections = []

    # Build port-to-parent mapping with original XML IDs
    port_to_parent_map = _build_port_to_parent_mapping(root)

    for arc in root.findall(".//sbgn:arc", ns):
        arc_id = arc.get("id")
        source = arc.get("source")
        target = arc.get("target")

        # Replace port references with their parent node IDs (still original IDs)
        if source in port_to_parent_map:
            source = port_to_parent_map[source]
        if target in port_to_parent_map:
            target = port_to_parent_map[target]

        # Translate source and target to final IDs
        final_source = translation_manager.translate_connection_reference(source)
        final_target = translation_manager.translate_connection_reference(target)

        # Create raw connection data
        raw_connection_data: RawConnectionData = {
            "id": arc_id,
            "source": final_source,
            "target": final_target,
            "connection_type": "metabolic",  # All SBGN connections are metabolic
        }

        raw_connections.append(raw_connection_data)

    return raw_connections


def _parse_hmr_mar_mapping(hmr_mar_file):
    """Parse the HMR_con_MAR file to create a mapping from HMR IDs to MAR IDs."""
    try:
        # Read the file with pandas
        df = pd.read_csv(hmr_mar_file, sep="\t")

        # Create mapping from HMR_IDs to MAR_IDs
        hmr_to_mar = {}
        for _, row in df.iterrows():
            hmr_id = row.get("ID")
            mar_id = row.get("mar_id")

            # Skip if HMR ID is missing
            if not hmr_id or pd.isna(hmr_id):
                continue

            # If MAR ID is valid, use it
            if mar_id != "NA" and pd.notna(mar_id):
                hmr_to_mar[hmr_id] = mar_id
            else:
                # Generate MAR ID from HMR ID
                # Extract the number part from HMR_XXXX format
                if "_" in hmr_id:
                    number_part = hmr_id.split("_")[1]
                    # Create MAR ID with leading zero
                    generated_mar_id = f"MAR0{number_part}"
                    hmr_to_mar[hmr_id] = generated_mar_id

        return hmr_to_mar
    except Exception as e:
        print(f"Error parsing HMR_con_MAR file: {e}")
        return {}


# Validation functions for raw data structures
def validate_raw_glyph_data(glyph_data: RawGlyphData) -> bool:
    """Validate that raw glyph data has all required fields and correct types.

    Args:
        glyph_data: Raw glyph data to validate

    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        "id",
        "xml_id",
        "label",
        "glyph_type",
        "x",
        "y",
        "width",
        "height",
    ]

    # Check all required fields are present
    for field in required_fields:
        if field not in glyph_data:
            return False

    # Check types
    if not isinstance(glyph_data["id"], str) or not glyph_data["id"]:
        return False
    if not isinstance(glyph_data["xml_id"], str) or not glyph_data["xml_id"]:
        return False
    if not isinstance(glyph_data["label"], str):
        return False
    valid_types = ["compartment", "metabolite", "gene", "reaction"]
    if glyph_data["glyph_type"] not in valid_types:
        return False
    if not isinstance(glyph_data["x"], (int, float)):
        return False
    if not isinstance(glyph_data["y"], (int, float)):
        return False
    if not isinstance(glyph_data["width"], (int, float)) or glyph_data["width"] < 0:
        return False
    if not isinstance(glyph_data["height"], (int, float)) or glyph_data["height"] < 0:
        return False

    return True


def validate_raw_connection_data(connection_data: RawConnectionData) -> bool:
    """Validate that raw connection data has all required fields and correct types.

    Args:
        connection_data: Raw connection data to validate

    Returns:
        True if valid, False otherwise
    """
    required_fields = ["id", "source", "target", "connection_type"]

    # Check all required fields are present
    for field in required_fields:
        if field not in connection_data:
            return False

    # Check types and values
    if not isinstance(connection_data["id"], str) or not connection_data["id"]:
        return False
    if not isinstance(connection_data["source"], str) or not connection_data["source"]:
        return False
    if not isinstance(connection_data["target"], str) or not connection_data["target"]:
        return False
    if connection_data["connection_type"] not in ["metabolic"]:
        return False

    return True


def validate_raw_sbgn_data(
    raw_glyphs: List[RawGlyphData], raw_connections: List[RawConnectionData]
) -> Tuple[bool, List[str]]:
    """Validate complete raw SBGN data set.

    Args:
        raw_glyphs: List of raw glyph data
        raw_connections: List of raw connection data

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Validate all glyphs
    glyph_ids = set()
    for i, glyph in enumerate(raw_glyphs):
        if not validate_raw_glyph_data(glyph):
            errors.append(f"Invalid glyph data at index {i}: {glyph}")
        else:
            if glyph["id"] in glyph_ids:
                errors.append(f"Duplicate glyph ID: {glyph['id']}")
            glyph_ids.add(glyph["id"])

    # Validate all connections
    connection_ids = set()
    for i, connection in enumerate(raw_connections):
        if not validate_raw_connection_data(connection):
            errors.append(f"Invalid connection data at index {i}: {connection}")
        else:
            if connection["id"] in connection_ids:
                errors.append(f"Duplicate connection ID: {connection['id']}")
            connection_ids.add(connection["id"])

            # Check that source and target exist in glyphs
            if connection["source"] not in glyph_ids:
                source_id = connection["source"]
                errors.append(
                    f"Connection {connection['id']} references "
                    f"non-existent source: {source_id}"
                )
            if connection["target"] not in glyph_ids:
                target_id = connection["target"]
                errors.append(
                    f"Connection {connection['id']} references "
                    f"non-existent target: {target_id}"
                )

    return len(errors) == 0, errors

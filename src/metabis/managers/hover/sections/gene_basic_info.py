from typing import Any, Dict
from ..interfaces import HoverSection
from ..models import HoverContext, HoverData

class GeneBasicInfoSection(HoverSection):
    """Section for basic gene information (name, ID, etc.)."""

    def should_include(self, data: HoverData, context: HoverContext) -> bool:
        """Basic info is always included."""
        return True

    def build(self, data: HoverData, context: HoverContext) -> str:
        """Build basic information section."""
        if not data.get("is_in_model", False):
            element_id = data.get("gene_id", "Unknown")
            return f"<div>Gene {element_id} not found</div>"

        parts = []
        symbol = data.get("symbol", "N/A")
        parts.append(f"<div><strong>Gene:</strong> {symbol}</div>")

        if data.get("description"):
            parts.append(
                f"<div><strong>Description:</strong> {data['description']}</div>"
            )
        return "".join(parts)

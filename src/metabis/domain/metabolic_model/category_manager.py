"""Category management for metabolic subsystems.

This module provides the CategoryManager class which handles:
- Loading and reloading subsystem categories from files
- Dynamic color assignment based on category order
- Tracking whether custom category files are loaded

Categories are decoupled from model initialization to allow fast reloading
without recreating the entire metabolic model.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .loaders import load_subsystem_classification

if TYPE_CHECKING:
    from .model import MetabolicModel

# Fixed color palette (30 colors) - preserves existing colors, expands for more categories
CATEGORY_PALETTE = [
    # Original 20 colors (unchanged order for backward compatibility)
    "#E74C3C",  # Red - Central Carbon & Bioenergetics
    "#F1C40F",  # Yellow - Redox & Antioxidant Defense
    "#C0392B",  # Dark red - Carbohydrate Interconversions
    "#00B594",  # Teal - Short-Chain Fatty Acids
    "#0971B6",  # Blue - Fatty Acid Biosynthesis & Activation
    "#453792",  # Purple - Fatty Acid Oxidation
    "#9B59B6",  # Light purple - Polyunsaturated Fatty Acids & Eicosanoids
    "#E67E22",  # Orange - Membrane Lipids & Phospholipid Signaling
    "#27AE60",  # Green - Sterols, Steroid Hormones & Bile Acids
    "#F39C12",  # Yellow-orange - Amino Acid Metabolism
    "#FF69B4",  # Hot pink - Sulfur Amino Acids & One-Carbon Metabolism
    "#8B4513",  # Saddle brown - Nucleotide Metabolism
    "#C16C04",  # Brown-orange - Glycans & Glycosylation
    "#D35400",  # Dark orange - Vitamins, Cofactors & Heme
    "#FF1493",  # Deep pink - Fat-Soluble Vitamins & Isoprenoids
    "#1ABC9C",  # Teal - Xenobiotics & Detoxification
    "#708090",  # Slate gray - Membrane Transport & Exchange
    "#F17CEE",  # Orchid - Protein Processing & Turnover
    "#95A5A6",  # Gray - Model Utilities & Non-human Pathways
    "#34495E",  # Dark gray - Unknown
    # Extended colors (10 new)
    "#2ECC71",  # Emerald
    "#3498DB",  # Peter river blue
    "#9B2335",  # Jester red
    "#5D3FD3",  # Iris purple
    "#FF7F50",  # Coral
    "#20B2AA",  # Light sea green
    "#CD853F",  # Peru brown
    "#4169E1",  # Royal blue
    "#DA70D6",  # Orchid pink
    "#2F4F4F",  # Dark slate gray
]

FALLBACK_CATEGORY_COLOR = "#666666"  # Medium gray for categories beyond palette


class CategoryManager:
    """Manages subsystem categories and their color assignments.

    This class handles loading category assignments from files and provides
    color lookups based on category order. Categories can be reloaded at
    any time without recreating the metabolic model.

    Attributes:
        model: Reference to the parent MetabolicModel
        _category_order: List of category names in file order (for color assignment)
        _custom_file_loaded: Whether a custom (non-default) file is currently loaded
        _custom_file_path: Path to the custom file if one is loaded
        _default_file_path: Path to the default category file
    """

    def __init__(self, model: "MetabolicModel"):
        """Initialize the CategoryManager.

        Args:
            model: The parent MetabolicModel instance
        """
        self.model = model
        self._category_order: List[str] = []
        self._custom_file_loaded: bool = False
        self._custom_file_path: Optional[str] = None
        self._default_file_path: str = (
            model.resource_registry.get_subsystem_category_taxonomy_path(
                required=False
            )
        )

    def load_categories(
        self, file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load or reload categories from a file.

        This method updates the category assignments for all subsystems in the
        model. Subsystems not found in the category file will be assigned
        "Uncategorized".

        Args:
            file_path: Path to category file. If None, uses default file.

        Returns:
            Dict with loading statistics:
                - num_categories: Number of unique categories loaded
                - num_subsystems_updated: Number of subsystems that got a category
                - num_uncategorized: Number of subsystems set to "Uncategorized"
                - categories: List of category names in file order
        """
        # Determine which file to use
        if file_path is None:
            actual_path = self._default_file_path
            self._custom_file_loaded = False
            self._custom_file_path = None
        else:
            actual_path = file_path
            self._custom_file_loaded = True
            self._custom_file_path = file_path

        # Load category mapping from file
        # Returns: {subsystem_id: category_name}
        classification_map = load_subsystem_classification(actual_path)

        # Extract unique categories in file order
        # We need to preserve order, so we parse the file again for category order
        self._category_order = self._extract_category_order(actual_path)

        # Update all subsystems with their categories
        num_updated = 0
        num_uncategorized = 0

        for subsystem in self.model.subsystems.get_all():
            new_category = classification_map.get(subsystem.id, "Uncategorized")
            subsystem.category = new_category

            if new_category != "Uncategorized":
                num_updated += 1
            else:
                num_uncategorized += 1

        # Ensure "Uncategorized" is in the category order if there are uncategorized subsystems
        if num_uncategorized > 0 and "Uncategorized" not in self._category_order:
            self._category_order.append("Uncategorized")

        return {
            "num_categories": len(self._category_order),
            "num_subsystems_updated": num_updated,
            "num_uncategorized": num_uncategorized,
            "categories": list(self._category_order),
        }

    def _extract_category_order(self, file_path: str) -> List[str]:
        """Extract category names in file order.

        Args:
            file_path: Path to category file

        Returns:
            List of category names in the order they appear in the file
        """
        categories = []

        if not os.path.exists(file_path):
            return categories

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("###"):
                        category_name = line[3:].strip()
                        if category_name and category_name not in categories:
                            categories.append(category_name)
        except Exception as e:
            print(f"Warning: Error reading category file for order: {e}")

        return categories

    def get_category_color(self, category_name: str) -> str:
        """Get the color for a category based on its order in the file.

        Colors are assigned based on the category's position in _category_order.
        Categories beyond the palette size will get the fallback color.

        Args:
            category_name: Name of the category

        Returns:
            Hex color string (e.g., "#E74C3C")
        """
        if not category_name or category_name not in self._category_order:
            return FALLBACK_CATEGORY_COLOR

        index = self._category_order.index(category_name)

        if index < len(CATEGORY_PALETTE):
            return CATEGORY_PALETTE[index]
        else:
            return FALLBACK_CATEGORY_COLOR

    def get_categories(self) -> List[str]:
        """Get list of all current categories in file order.

        Returns:
            List of category names
        """
        return list(self._category_order)

    def get_category_colors(self) -> Dict[str, str]:
        """Get a dictionary mapping category names to colors.

        This provides backward compatibility with code that expects
        a CATEGORY_COLORS dict.

        Returns:
            Dict mapping category name to hex color string
        """
        return {
            category: self.get_category_color(category)
            for category in self._category_order
        }

    def is_custom_file_loaded(self) -> bool:
        """Check if a custom (non-default) category file was loaded.

        Returns:
            True if a custom file is loaded, False if using default
        """
        return self._custom_file_loaded

    def get_custom_file_path(self) -> Optional[str]:
        """Get path of custom file if one was loaded.

        Returns:
            Path to custom file, or None if using default
        """
        return self._custom_file_path

    def get_default_file_path(self) -> str:
        """Get path to the default category file.

        Returns:
            Path to the default subsystem category taxonomy file
        """
        return self._default_file_path

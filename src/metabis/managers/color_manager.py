from typing import Any, Dict, List, Tuple, Union

import pandas as pd

from bokeh.palettes import Greens, Oranges

from ..visual_config import theme
from ..analysis.distribution_binner import DistributionBinner
from .data_manager_mode import DataManagerMode


class ColorManager:
    """Handles all color calculations and palettes for metabolic visualizations.

    Supports multiple color configurations for different analysis modes and
    group colors.
    """

    def __init__(
        self,
        group_comparison_config: Dict[str, Any] = None,
        experiment_analysis_config: Dict[str, Any] = None,
        default_mode: DataManagerMode = DataManagerMode.GROUP_COMPARISON
    ):
        """Initialize ColorManager with mode-aware color configurations.

        Args:
            group_comparison_config: Configuration for GROUP_COMPARISON mode
            experiment_analysis_config: Configuration for EXPERIMENT_ANALYSIS mode
            default_mode: Default color mode to use (DataManagerMode enum or legacy string)
        """
        # Convert string mode to enum for backward compatibility
        if isinstance(default_mode, str):
            default_mode = DataManagerMode[default_mode.upper().replace(" ", "_")]

        # New multi-configuration mode
        self.configurations = {
            DataManagerMode.GROUP_COMPARISON: group_comparison_config
            or {
                "max_z_score": 4,
                "up_color": theme.UP_REGULATED,
                "down_color": theme.DOWN_REGULATED,
                "neutral_color": theme.NEUTRAL,
                "significant_pval_color": theme.SIGNIFICANT_PVAL,
            },
            DataManagerMode.EXPERIMENT_ANALYSIS: experiment_analysis_config
            or {
                "max_z_score": 4,
                "up_color": theme.ACTIVITY_UP,
                "down_color": theme.ACTIVITY_DOWN,
                "neutral_color": theme.NEUTRAL,
            },
        }

        # Set current mode and load configuration
        self.current_mode = default_mode
        self._group_colors = {}  # Storage for group colors
        self._distributions = {} # Storage for registered distribution binners and palettes
        self._load_current_configuration()

        # Define diverging color palettes for influence bars
        # Positive correlation (Green gradient) - darker greens first
        self.pos_influence_palette = list(reversed(Greens[9][1:]))
        # Negative correlation (Orange gradient) - darker first
        self.neg_influence_palette = list(reversed(Oranges[9][1:]))

    def _load_current_configuration(self):
        """Load the configuration for the current mode into instance variables."""
        if self.current_mode not in self.configurations:
            raise ValueError(f"Unknown color mode: {self.current_mode}")

        config = self.configurations[self.current_mode]

        # Set configuration values for current mode
        self.max_z_score = config["max_z_score"]
        self.up_color = config["up_color"]
        self.down_color = config["down_color"]
        self.neutral_color = config["neutral_color"]

        # Convert all colors to RGB tuples for efficient computation
        self._up_rgb = self._ensure_rgb(self.up_color)
        self._down_rgb = self._ensure_rgb(self.down_color)
        self._neutral_rgb = self._ensure_rgb(self.neutral_color)

    # ============================================================================
    # Distribution Management Methods
    # ============================================================================

    def register_distribution(
        self,
        mode: DataManagerMode,
        element_type: str,
        data: Union[pd.Series, pd.DataFrame],
        metric_name: str = "default",
        value_column: str = None,
        center_method: str = "zero",
        spread_method: str = "std",
        max_z: float = 4.0,
        n_bins: int = 30,
    ) -> None:
        """Register a distribution for color binning.

        Args:
            mode: The data manager mode (EXPERIMENT_ANALYSIS or GROUP_COMPARISON).
            element_type: The type of element (e.g., "reaction", "gene", "subsystem").
            data: The data distribution.
            metric_name: The name of the metric (e.g., "log2_fc", "default").
            value_column: Column name if data is a DataFrame.
            center_method: Method to calculate center ("mean", "median", "zero").
            spread_method: Method to calculate spread ("std", "mad").
            max_z: Max Z-score for binning.
            n_bins: Number of bins per direction.
        """
        # Create binner
        binner = DistributionBinner(
            data=data,
            value_column=value_column,
            center_method=center_method,
            spread_method=spread_method,
            max_z=max_z,
            n_bins=n_bins
        )

        # Get configuration for the specified mode
        config = self.configurations.get(mode)
        if not config:
            # Fallback to current config if mode not found
            config = self.configurations[self.current_mode]

        up_color = config["up_color"]
        down_color = config["down_color"]
        neutral_color = config["neutral_color"]
        
        # Handle p-value specific coloring
        is_pval = "pval" in metric_name.lower() or "p-value" in metric_name.lower()
        is_neg_log = "neg_log" in metric_name.lower() or "-log" in metric_name.lower()
        
        if is_pval:
            significant_color = config.get("significant_pval_color", theme.SIGNIFICANT_PVAL)
            if is_neg_log:
                # For -log(p-value), higher is more significant
                # Neutral remains neutral (low significance)
                # Up color becomes significant color (high significance)
                up_color = significant_color
            else:
                # For raw p-value, lower is more significant
                # Neutral becomes significant color (low p-value = high significance)
                # Up color becomes neutral (high p-value = low significance)
                # We swap them because the gradient generator goes from neutral -> up
                # But here we want significant -> neutral as value increases
                temp = neutral_color
                neutral_color = significant_color
                up_color = temp

        # Generate palettes
        # We generate n_bins + 1 colors to interpolate from neutral to max intensity
        # We skip the first one (neutral) for the bins 1..N
        pos_palette = self._generate_gradient_palette(neutral_color, up_color, n_bins + 1)[1:]
        neg_palette = self._generate_gradient_palette(neutral_color, down_color, n_bins + 1)[1:]
        
        # Create bin_colors map
        bin_colors = {0: neutral_color}
        
        for i in range(n_bins):
            # 1-based index for bins
            bin_idx = i + 1
            bin_colors[bin_idx] = pos_palette[i]
            bin_colors[-bin_idx] = neg_palette[i]

        # Store
        key = (mode, element_type, metric_name)
        self._distributions[key] = {
            "binner": binner,
            "bin_colors": bin_colors
        }

    def get_colors_from_distribution(
        self,
        mode: DataManagerMode,
        element_type: str,
        metric_name: str = "default",
        element_ids: Union[List[str], pd.Index] = None
    ) -> pd.Series:
        """Get colors for multiple elements based on a registered distribution.

        Args:
            mode: The data manager mode.
            element_type: The type of element.
            metric_name: The metric name.
            element_ids: List of element IDs to get colors for. If None, returns for all.

        Returns:
            pd.Series: Series of hex color strings indexed by element_ids.
        """
        key = (mode, element_type, metric_name)
        dist_info = self._distributions.get(key)
        
        if not dist_info:
            # Return neutral color for all requested IDs
            neutral = self.configurations[self.current_mode]["neutral_color"]
            if element_ids is not None:
                return pd.Series(neutral, index=element_ids)
            return pd.Series() # Empty series if no IDs and no distribution
            
        binner = dist_info["binner"]
        bin_colors = dist_info["bin_colors"]
        
        if element_ids is None:
            # Get for all in binner
            bins = binner.get_all_bins()
        else:
            # Get for specific IDs
            # Reindex to ensure we have all requested IDs, filling missing with 0 (neutral)
            bins = binner.get_all_bins().reindex(element_ids, fill_value=0).astype(int)
        
        # Map bins to colors
        # bin_colors is a dict {int: str}
        # We use map. If a bin is not in bin_colors (shouldn't happen if 0 is there), fill with neutral
        colors = bins.map(bin_colors)
        colors = colors.fillna(bin_colors[0])
        
        return colors

    def get_element_color_from_distribution(
        self,
        mode: DataManagerMode,
        element_type: str,
        element_id: str,
        metric_name: str = "default"
    ) -> str:
        """Get color for an element based on a registered distribution.

        Args:
            mode: The data manager mode.
            element_type: The type of element.
            element_id: The ID of the element.
            metric_name: The metric name.

        Returns:
            str: Hex color string.
        """
        key = (mode, element_type, metric_name)
        dist_info = self._distributions.get(key)
        
        if not dist_info:
            return self.configurations[self.current_mode]["neutral_color"]
            
        binner = dist_info["binner"]
        bin_colors = dist_info["bin_colors"]
        
        bin_idx = binner.get_bin(element_id)
        return bin_colors.get(bin_idx, bin_colors[0])

    def get_distribution_details(
        self,
        mode: DataManagerMode,
        element_type: str,
        element_id: str,
        metric_name: str = "default"
    ) -> Dict[str, Any]:
        """
        Get histogram data and colors for an element based on a registered distribution.

        Args:
            mode: The data manager mode.
            element_type: The type of element.
            element_id: The ID of the element.
            metric_name: The metric name.

        Returns:
            Dict containing:
            - histogram: List of dicts with bin, count, color, label
            - element_bin: The bin index of the element
            - element_color: The color of the element
            - element_value: The original value of the element
        """
        key = (mode, element_type, metric_name)
        dist_info = self._distributions.get(key)
        
        if not dist_info:
            # Return empty structure if no distribution found
            neutral_color = self.configurations[self.current_mode]["neutral_color"]
            return {
                "histogram": [],
                "element_bin": 0,
                "element_color": neutral_color,
                "element_value": None
            }
            
        binner = dist_info["binner"]
        bin_colors = dist_info["bin_colors"]
        
        # Get histogram data from binner
        raw_histogram = binner.get_histogram_data()
        
        # Enrich histogram with colors
        histogram = []
        for bin_data in raw_histogram:
            bin_idx = bin_data["bin"]
            color = bin_colors.get(bin_idx, bin_colors[0])
            histogram.append({
                **bin_data,
                "color": color
            })
            
        # Get element specific data
        element_bin = binner.get_bin(element_id)
        element_color = bin_colors.get(element_bin, bin_colors[0])
        element_value = binner.get_value(element_id)
        
        return {
            "histogram": histogram,
            "element_bin": element_bin,
            "element_color": element_color,
            "element_value": element_value
        }

    def _generate_gradient_palette(
        self,
        start_color: str,
        end_color: str,
        n_steps: int
    ) -> List[str]:
        """Generate a list of colors forming a gradient.

        Args:
            start_color: Start hex color.
            end_color: End hex color.
            n_steps: Number of steps.

        Returns:
            List[str]: List of hex color strings.
        """
        start_rgb = self._ensure_rgb(start_color)
        end_rgb = self._ensure_rgb(end_color)
        
        palette = []
        if n_steps <= 1:
            return [end_color]
            
        for i in range(n_steps):
            fraction = i / (n_steps - 1)
            color = self.gradient_color(start_rgb, end_rgb, fraction)
            palette.append(color)
            
        return palette

    # ============================================================================
    # Mode Management Methods
    # ============================================================================

    def set_mode(self, mode: Union[str, DataManagerMode]) -> None:
        """Switch color configuration mode.

        Args:
            mode: Color mode to switch to (DataManagerMode enum or legacy string)

        Raises:
            ValueError: If the mode is not available
        """
        # Convert string mode to enum for backward compatibility
        if isinstance(mode, str):
            if mode == "group_comparison":
                mode = DataManagerMode.GROUP_COMPARISON
            elif mode == "experiment_analysis":
                mode = DataManagerMode.EXPERIMENT_ANALYSIS
            else:
                mode = DataManagerMode.NO_EXPERIMENT

        if mode not in self.configurations:
            available_modes = list(self.configurations.keys())
            raise ValueError(
                f"Mode '{mode}' not available. Available modes: {available_modes}"
            )

        self.current_mode = mode
        self._load_current_configuration()

    def get_mode(self) -> DataManagerMode:
        """Get current color mode.

        Returns:
            DataManagerMode: Current color mode
        """
        return self.current_mode

    def get_available_modes(self) -> List[DataManagerMode]:
        """Get list of available color modes.

        Returns:
            List[DataManagerMode]: List of available color modes
        """
        return list(self.configurations.keys())

    def update_mode_configuration(
        self, mode: Union[str, DataManagerMode], config: Dict[str, Any]
    ) -> None:
        """Update configuration for specific mode.

        Args:
            mode: Mode to update configuration for (enum or legacy string)
            config: New configuration dictionary

        Raises:
            ValueError: If the mode is not available
        """
        # Convert string mode to enum for backward compatibility
        if isinstance(mode, str):
            if mode == "group_comparison":
                mode = DataManagerMode.GROUP_COMPARISON
            elif mode == "experiment_analysis":
                mode = DataManagerMode.EXPERIMENT_ANALYSIS
            else:
                mode = DataManagerMode.NO_EXPERIMENT

        if mode not in self.configurations:
            available_modes = list(self.configurations.keys())
            raise ValueError(
                f"Mode '{mode}' not available. Available modes: {available_modes}"
            )

        # Update the configuration
        self.configurations[mode].update(config)

        # If updating current mode, reload configuration
        if mode == self.current_mode:
            self._load_current_configuration()

    # ============================================================================
    # Group Color Management Methods
    # ============================================================================

    def set_group_colors(self, group_colors: Dict[str, str]) -> None:
        """Store group colors for violin plot coloring.

        Args:
            group_colors: Dictionary mapping group IDs to hex color strings
        """
        self._group_colors = group_colors.copy()

    def get_group_colors(self) -> Dict[str, str]:
        """Get stored group colors.

        Returns:
            Dict[str, str]: Dictionary mapping group IDs to hex color strings
        """
        return self._group_colors.copy()

    def get_group_color(self, group_id: str) -> str:
        """Get color for specific group.

        Args:
            group_id: ID of the group

        Returns:
            str: Hex color string for the group

        Raises:
            KeyError: If group_id is not found
        """
        if group_id not in self._group_colors:
            raise KeyError(f"Group '{group_id}' not found in group colors")
        return self._group_colors[group_id]

    def has_group_colors(self) -> bool:
        """Check if group colors are available.

        Returns:
            bool: True if group colors are stored, False otherwise
        """
        return bool(self._group_colors)

    def clear_group_colors(self) -> None:
        """Clear all stored group colors."""
        self._group_colors.clear()

    # ============================================================================
    # Group Color Generation Methods
    # ============================================================================

    def generate_group_colors(
        self,
        group_ids: List[str],
        palette_name: str = "turbo",
        custom_colors: Dict[str, str] = None,
    ) -> Dict[str, str]:
        """Generate colors for a list of group IDs.

        Args:
            group_ids: List of group identifiers to generate colors for
            palette_name: Name of the color palette to use ('turbo', 'viridis',
                'Category20', 'Set1', 'Set3')
            custom_colors: Optional dictionary of custom colors to override
                generated ones

        Returns:
            Dictionary mapping group IDs to hex color strings

        Raises:
            ValueError: If palette_name is not supported
        """
        if not group_ids:
            self._group_colors = {}
            return {}

        # Generate colors using specified palette
        generated_colors = self._generate_palette_colors(group_ids, palette_name)

        # Apply any custom color overrides
        if custom_colors:
            for group_id, color in custom_colors.items():
                if group_id in generated_colors:
                    generated_colors[group_id] = color

        # Store the generated colors
        self._group_colors = generated_colors
        return self._group_colors.copy()

    def regenerate_group_colors(
        self,
        group_ids: List[str] = None,
        palette_name: str = None,
        custom_colors: Dict[str, str] = None,
    ) -> Dict[str, str]:
        """Regenerate colors for existing or new groups.

        Args:
            group_ids: List of group identifiers. If None, uses existing groups
            palette_name: Name of the color palette. If None, uses 'turbo'
            custom_colors: Optional custom color overrides

        Returns:
            Dictionary mapping group IDs to hex color strings
        """
        # Use existing group IDs if none provided
        if group_ids is None:
            group_ids = list(self._group_colors.keys())

        # Use default palette if none specified
        if palette_name is None:
            palette_name = "turbo"

        return self.generate_group_colors(group_ids, palette_name, custom_colors)

    def _generate_palette_colors(
        self, group_ids: List[str], palette_name: str
    ) -> Dict[str, str]:
        """Generate colors from specified palette.

        Args:
            group_ids: List of group identifiers
            palette_name: Name of the color palette

        Returns:
            Dictionary mapping group IDs to hex color strings

        Raises:
            ValueError: If palette_name is not supported
        """
        n_colors = len(group_ids)

        # Define available palettes with dynamic imports
        def get_palette_colors(palette_name: str, n: int) -> List[str]:
            """Get colors from specified palette using dynamic imports."""
            if palette_name == "turbo":
                from bokeh.palettes import turbo

                # Custom override for the first two colors to ensure high contrast and aesthetics
                # Color 1: Deep Teal (Green-Blue) - Distinct from standard blue/green
                # Color 2: Vivid Orange - Distinct from red/fuchsia
                # These colors are chosen to be:
                # 1. High contrast with each other
                # 2. Distinct from the heatmap colors (Red/Blue/Fuchsia)
                # 3. Not too light (visible on white background)
                # 4. Aesthetic combination
                colors = turbo(max(3, n))
                if n >= 2:
                    colors = list(colors)
                    colors[0] = "#008080"  # Teal
                    colors[1] = "#FF8C00"  # Dark Orange
                return colors
            elif palette_name == "viridis":
                from bokeh.palettes import viridis

                return viridis(max(3, n))
            elif palette_name == "Category20":
                from bokeh.palettes import Category20

                return Category20[min(20, max(3, n))]
            elif palette_name == "Set1":
                from bokeh.palettes import Set1

                return Set1[min(9, max(3, n))]
            elif palette_name == "Set3":
                from bokeh.palettes import Set3

                return Set3[min(12, max(3, n))]
            else:
                raise ValueError(f"Unknown palette: {palette_name}")

        available_palettes = ["turbo", "viridis", "Category20", "Set1", "Set3"]

        if palette_name not in available_palettes:
            raise ValueError(
                f"Palette '{palette_name}' not supported. "
                f"Available palettes: {available_palettes}"
            )

        try:
            # Generate color palette
            palette = get_palette_colors(palette_name, n_colors)

            # Create mapping from group IDs to colors
            return dict(zip(group_ids, palette))

        except Exception as e:
            # Fallback to turbo if specific palette fails
            if palette_name != "turbo":
                print(
                    f"Warning: {palette_name} palette failed ({e}), "
                    f"falling back to turbo"
                )
                return self._generate_palette_colors(group_ids, "turbo")
            else:
                raise ValueError(f"Failed to generate colors: {e}")

    def get_available_palettes(self) -> List[str]:
        """Get list of available color palettes for groups.

        Returns:
            List of available palette names
        """
        return ["turbo", "viridis", "Category20", "Set1", "Set3"]

    def set_custom_group_color(self, group_id: str, color: str) -> None:
        """Override generated color for specific group.

        Args:
            group_id: ID of the group
            color: Hex color string to assign

        Raises:
            ValueError: If color format is invalid
        """
        # Validate color format
        try:
            self._ensure_rgb(color)
        except ValueError as e:
            raise ValueError(f"Invalid color format for group '{group_id}': {e}")

        # Set the custom color
        self._group_colors[group_id] = color

    # ============================================================================
    # Color Conversion and Utility Methods
    # ============================================================================

    def _ensure_rgb(
        self, color: Union[str, Tuple[int, int, int]]
    ) -> Tuple[int, int, int]:
        """Convert color to RGB tuple, detecting the input format.

        Args:
            color: Color as hex string or RGB tuple

        Returns:
            Tuple[int, int, int]: RGB color values
        """
        if isinstance(color, str):
            return self.hex_to_rgb(color)
        elif isinstance(color, (tuple, list)) and len(color) == 3:
            return tuple(int(c) for c in color)
        else:
            raise ValueError(
                f"Invalid color format: {color}. Expected hex string or RGB tuple."
            )

    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple.

        Args:
            hex_color: Hex color string (e.g., "#FF0000")

        Returns:
            Tuple[int, int, int]: RGB color values
        """
        hex_color = hex_color.lstrip("#")
        if len(hex_color) != 6:
            raise ValueError(f"Invalid hex color format: {hex_color}")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to hex color string.

        Args:
            rgb: RGB tuple (r, g, b)

        Returns:
            str: Hex color string
        """
        return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"

    @staticmethod
    def gradient_color(
        base_rgb: Tuple[int, int, int],
        target_rgb: Tuple[int, int, int],
        fraction: float,
    ) -> str:
        """Create a gradient color between two RGB colors.

        Args:
            base_rgb: Starting RGB color
            target_rgb: Target RGB color
            fraction: Interpolation fraction (0.0 to 1.0)

        Returns:
            str: Hex color string
        """
        # Clamp fraction between 0 and 1
        fraction = max(0.0, min(1.0, fraction))
        # Linear interpolation in RGB space
        r = int(base_rgb[0] + fraction * (target_rgb[0] - base_rgb[0]))
        g = int(base_rgb[1] + fraction * (target_rgb[1] - base_rgb[1]))
        b = int(base_rgb[2] + fraction * (target_rgb[2] - base_rgb[2]))
        return f"#{r:02X}{g:02X}{b:02X}"

    def get_effect_color(
        self,
        effect_value: float,
        neutral_color: Union[str, Tuple[int, int, int], None] = None,
        max_effect_value: float = None,
    ) -> str:
        """Get a color based on effect value (positive/negative).

        This is the primary color function that replaces get_metric_color.

        Args:
            effect_value: The effect value (typically z-score)
            neutral_color: Override neutral color for this calculation
            max_effect_value: Override max effect value for normalization

        Returns:
            str: Hex color code
        """
        # Handle NaN/None values
        if pd.isna(effect_value) or effect_value is None:
            effect_value = 0.0

        # Use provided max_effect_value or default to instance setting
        max_effect = (
            max_effect_value if max_effect_value is not None else self.max_z_score
        )

        # Use provided neutral_color or default to instance setting
        if neutral_color is not None:
            neutral_rgb = self._ensure_rgb(neutral_color)
        else:
            neutral_rgb = self._neutral_rgb

        # Normalize by effect value (configurable max effect value for full color)
        if max_effect > 0:
            norm_effect = min(abs(effect_value) / max_effect, 1.0)
        else:
            norm_effect = 0.0

        # Apply color gradient based on direction
        if effect_value > 0:
            # Positive effect: neutral to upregulated_color
            return self.gradient_color(neutral_rgb, self._up_rgb, norm_effect)
        elif effect_value < 0:
            # Negative effect: neutral to downregulated_color
            return self.gradient_color(neutral_rgb, self._down_rgb, norm_effect)
        else:
            # Zero effect: return neutral color
            return self.rgb_to_hex(neutral_rgb)

    def get_influence_color(
        self,
        correlation: float,
        influence: float,
        max_influence: float = 100.0,
        use_gray_scale: bool = False,
        gray_palette: List[str] = None,
    ) -> str:
        """Get color for gene influence visualization with detailed control.

        Args:
            correlation: The correlation value (-1 to 1)
            influence: The influence magnitude (e.g., pct_variance, 0 to max_influence)
            max_influence: The maximum possible influence value for scaling intensity
            use_gray_scale: If True, forces the use of the gray_palette
            gray_palette: Optional list of hex colors for grayscale

        Returns:
            str: Hex color code
        """
        # Handle NaN inputs
        if pd.isna(correlation):
            correlation = 0.0
        if pd.isna(influence):
            influence = 0.0

        # Clamp influence for intensity mapping
        clamped_influence = max(0.0, min(influence, max_influence))

        # Normalize influence magnitude (0 to 1) for intensity scaling
        if max_influence > 0:
            normalized_influence = clamped_influence / max_influence
        else:
            normalized_influence = 0

        # Select palette
        if use_gray_scale:
            if gray_palette:
                palette = gray_palette
                palette_size = len(gray_palette)
            else:
                # Fallback if gray palette not provided but requested
                return "#BEBEBE"  # Return a medium gray
        elif correlation >= 0:  # Positive or zero correlation
            palette = self.pos_influence_palette
            palette_size = len(self.pos_influence_palette)
        else:  # Negative correlation
            palette = self.neg_influence_palette
            palette_size = len(self.neg_influence_palette)

        if not palette:  # Handle empty palette case
            return "#000000"  # Return black if palette is empty

        # Calculate index based on normalized influence magnitude
        # Higher influence maps to higher index (typically darker color
        # in reversed palettes)
        color_index = min(int(normalized_influence * palette_size), palette_size - 1)

        return palette[color_index]

    def update_settings(
        self,
        max_z_score: int = None,
        up_color: Union[str, Tuple[int, int, int]] = None,
        down_color: Union[str, Tuple[int, int, int]] = None,
        neutral_color: Union[str, Tuple[int, int, int]] = None,
        mode: Union[str, DataManagerMode] = None,
    ):
        """Update color settings for current or specified mode.

        Args:
            max_z_score: Maximum z-score for full color intensity
            up_color: Color for positive/upregulated values
            down_color: Color for negative/downregulated values
            neutral_color: Color for neutral/zero values
            mode: Color mode to update (defaults to current mode)
        """
        target_mode = mode or self.current_mode

        # Prepare update dictionary
        updates = {}
        if max_z_score is not None:
            updates["max_z_score"] = max_z_score
        if up_color is not None:
            updates["up_color"] = up_color
        if down_color is not None:
            updates["down_color"] = down_color
        if neutral_color is not None:
            updates["neutral_color"] = neutral_color

        # Update the configuration
        if updates:
            self.update_mode_configuration(target_mode, updates)
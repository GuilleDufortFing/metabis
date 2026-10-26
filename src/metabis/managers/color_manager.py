from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from bokeh.palettes import Greens, Oranges

from ..visual_config import theme
from .color_distributions import (
    ColorDistributionRegistry,
    ensure_rgb,
    gradient_color,
    gradient_palette,
    hex_to_rgb,
    rgb_to_hex,
)
from .color_lookup import GraphColorLookup
from .group_palettes import group_palette
from .workspace_mode import AnalysisWorkspaceMode

# The group palette used unless another is asked for (see group_palettes).
DEFAULT_GROUP_PALETTE = "default"


class ColorManager:
    """Handles all color calculations and palettes for metabolic visualizations.

    Supports multiple color configurations for different analysis modes and
    group colors.

    The manager is the *Bokeh* half of colour: it generates the influence and
    categorical group palettes that need ``bokeh.palettes``. The two halves that
    do not -- reading a registered distribution back
    (:class:`~metabis.managers.color_lookup.GraphColorLookup`) and registering one
    (:class:`~metabis.managers.color_distributions.ColorDistributionRegistry`) --
    live beside it and are delegated to, not duplicated here, so a graph worker
    can hold them without holding Bokeh (AM10, M-12).
    """

    # Bokeh's continuous palettes (turbo, viridis) cannot return more than 256 colors.
    MAX_PALETTE_COLORS = 256

    def __init__(
        self,
        group_comparison_config: Dict[str, Any] = None,
        experiment_analysis_config: Dict[str, Any] = None,
        default_mode: AnalysisWorkspaceMode = AnalysisWorkspaceMode.GROUP_COMPARISON,
    ):
        """Initialize ColorManager with mode-aware color configurations.

        Args:
            group_comparison_config: Configuration for GROUP_COMPARISON mode
            experiment_analysis_config: Configuration for EXPERIMENT_ANALYSIS mode
            default_mode: Default color mode to use
        """
        # New multi-configuration mode
        configurations = {
            AnalysisWorkspaceMode.GROUP_COMPARISON: group_comparison_config
            or {
                "max_z_score": 4,
                "up_color": theme.UP_REGULATED,
                "down_color": theme.DOWN_REGULATED,
                "neutral_color": theme.NEUTRAL,
                "significant_pval_color": theme.SIGNIFICANT_PVAL,
            },
            AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS: experiment_analysis_config
            or {
                "max_z_score": 4,
                "up_color": theme.ACTIVITY_HIGH,
                "down_color": theme.ACTIVITY_LOW,
                "neutral_color": theme.ACTIVITY_MID,
            },
        }

        # Set current mode and load configuration. The registry owns the
        # configurations, the registered distributions, and the already-resolved
        # palettes; this manager reads them through it so there is exactly one
        # copy of each. Generating the palettes below is the half that needs
        # Bokeh, and is the reason this class still exists.
        self._registry = ColorDistributionRegistry(
            configurations,
            default_mode,
            # Diverging palettes for influence bars: green for positive
            # correlation, orange for negative, darker first.
            positive_influence_palette=list(reversed(Greens[9][1:])),
            negative_influence_palette=list(reversed(Oranges[9][1:])),
        )
        self._load_current_configuration()

    @property
    def configurations(self) -> Dict[AnalysisWorkspaceMode, Dict[str, Any]]:
        """The per-mode colour configurations, owned by the registry."""
        return self._registry.configurations

    @property
    def current_mode(self) -> AnalysisWorkspaceMode:
        """The mode whose configuration this manager currently paints with."""
        return self._registry.current_mode

    @current_mode.setter
    def current_mode(self, mode: AnalysisWorkspaceMode) -> None:
        self._registry.current_mode = mode

    @property
    def _distributions(self) -> Dict[Any, Dict[str, Any]]:
        """The registered distributions, owned by the registry."""
        return self._registry.distributions

    @property
    def _group_colors(self) -> Dict[str, str]:
        """The stored group colours, owned by the registry."""
        return self._registry.group_colors

    @_group_colors.setter
    def _group_colors(self, group_colors: Dict[str, str]) -> None:
        self._registry.group_colors = dict(group_colors)

    @property
    def pos_influence_palette(self) -> Tuple[str, ...]:
        """The generated palette for positive gene influence."""
        return self._registry.positive_influence_palette

    @property
    def neg_influence_palette(self) -> Tuple[str, ...]:
        """The generated palette for negative gene influence."""
        return self._registry.negative_influence_palette

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
        mode: AnalysisWorkspaceMode,
        element_type: str,
        data: Union[pd.Series, pd.DataFrame],
        metric_name: str = "default",
        value_column: str = None,
        center_method: str = "zero",
        spread_method: str = "std",
        max_z: float = 4.0,
        n_bins: int = 30,
        saturation_quantile: float = 0.98,
    ) -> None:
        """Register a distribution for color binning.

        Args:
            mode: The data manager mode (EXPERIMENT_ANALYSIS or GROUP_COMPARISON).
            element_type: The type of element (e.g., "reaction", "gene", "subsystem").
            data: The data distribution.
            metric_name: The name of the metric (e.g., "log2_fc", "default").
            value_column: Column name if data is a DataFrame.
            center_method: Method to calculate center ("mean", "median", "zero").
            spread_method: Method to calculate spread ("std", "mad", "saturating",
                or "asymmetric_saturating").
            max_z: Max Z-score for binning.
            n_bins: Number of bins per direction.
            saturation_quantile: Quantile of |value - center| that saturates the
                palette under the "saturating" spread method.
        """
        # Binning and gradient interpolation need no Bokeh, so they live in the
        # registry a graph worker can hold (AM10 slice 3b).
        self._registry.register_distribution(
            mode=mode,
            element_type=element_type,
            data=data,
            metric_name=metric_name,
            value_column=value_column,
            center_method=center_method,
            spread_method=spread_method,
            max_z=max_z,
            n_bins=n_bins,
            saturation_quantile=saturation_quantile,
        )

    def get_colors_from_distribution(
        self,
        mode: AnalysisWorkspaceMode,
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
        return self.color_lookup().get_element_colors_from_distribution(
            mode, element_type, metric_name, element_ids
        )

    def get_element_color_from_distribution(
        self,
        mode: AnalysisWorkspaceMode,
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
        return self.color_lookup().get_element_color_from_distribution(
            mode, element_type, element_id, metric_name
        )

    def get_color_for_value(
        self,
        mode: AnalysisWorkspaceMode,
        element_type: str,
        metric_name: str,
        value: float,
    ) -> str:
        """Colour a value as a registered distribution colours an element of it.

        For a value no element holds, such as a transporter family's Delta score
        on its subsystems' scale.
        """
        return self.color_lookup().get_color_for_value(mode, element_type, metric_name, value)

    def has_value(
        self,
        mode: AnalysisWorkspaceMode,
        element_type: str,
        element_id: str,
        metric_name: str = "default",
    ) -> bool:
        """Whether the registered distribution holds a finite value for this element."""
        return self.color_lookup().has_value(mode, element_type, element_id, metric_name)

    def get_color_limit(
        self,
        mode: AnalysisWorkspaceMode,
        element_type: str,
        metric_name: str = "default",
    ) -> Optional[float]:
        """Distance from the centre at which the registered palette saturates.

        Callers report this so the mapping from value to colour is auditable
        rather than implicit.
        """
        return self.color_lookup().get_color_limit(mode, element_type, metric_name)

    def get_distribution_details(
        self,
        mode: AnalysisWorkspaceMode,
        element_type: str,
        element_id: str,
        metric_name: str = "default"
    ) -> Dict[str, Any]:
        """Get histogram data and colors for an element based on a registered distribution.

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
        return self.color_lookup().get_distribution_details(
            mode, element_type, element_id, metric_name
        )

    def color_lookup(self) -> GraphColorLookup:
        """Capture the readable half of this manager as a plain frozen value.

        This is the M-12 (A') seam. The returned value answers every colour
        question graph enrichment asks, holds no Bokeh, and cannot be mutated by
        the worker that receives it -- so a graph job needs the colour *data*
        without needing the colour *manager*, and `bokeh.palettes` stops being
        reachable from graph calculation.

        Capture is a shallow read of state this manager already holds, so it is
        cheap enough to run on the document thread per graph render (measured at
        ~0 ms against the 208 ms run it feeds). The distributions are shared, not
        copied: they are append-only registrations that the document thread
        replaces rather than edits, and copying the binners would reintroduce the
        per-render tax ADR-0007 forbids.
        """
        return self._registry.lookup()

    def detached_color_registry(self) -> ColorDistributionRegistry:
        """Capture this manager's colour state as a registry a worker can own.

        The graph worker needs both halves -- it reads colours and it registers
        the distributions its own scoring produces -- so it gets a registry, not
        just a lookup. The copy is what makes that safe: registrations during the
        job stay in the job, exactly as the per-job ``deepcopy`` of this manager
        did, without the manager and therefore without Bokeh (AM10, M-12).
        """
        return self._registry.detached_copy()

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
        return gradient_palette(start_color, end_color, n_steps)

    # ============================================================================
    # Mode Management Methods
    # ============================================================================

    def set_mode(self, mode: AnalysisWorkspaceMode) -> None:
        """Switch color configuration mode.

        Args:
            mode: Color mode to switch to

        Raises:
            ValueError: If the mode is not available
        """
        if mode not in self.configurations:
            available_modes = list(self.configurations.keys())
            raise ValueError(
                f"Mode '{mode}' not available. Available modes: {available_modes}"
            )

        self.current_mode = mode
        self._load_current_configuration()

    def get_mode(self) -> AnalysisWorkspaceMode:
        """Get current color mode.

        Returns:
            AnalysisWorkspaceMode: Current color mode
        """
        return self.current_mode

    def get_available_modes(self) -> List[AnalysisWorkspaceMode]:
        """Get list of available color modes.

        Returns:
            List[AnalysisWorkspaceMode]: List of available color modes
        """
        return list(self.configurations.keys())

    def update_mode_configuration(
        self, mode: AnalysisWorkspaceMode, config: Dict[str, Any]
    ) -> None:
        """Update configuration for specific mode.

        Args:
            mode: Mode to update configuration for
            config: New configuration dictionary

        Raises:
            ValueError: If the mode is not available
        """
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

    def clear_distributions(self) -> None:
        """Clear every registered scientific color distribution."""
        self._registry.clear()

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
        return self.color_lookup().get_group_color(group_id)

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
        palette_name: str = DEFAULT_GROUP_PALETTE,
        custom_colors: Dict[str, str] = None,
    ) -> Dict[str, str]:
        """Generate colors for a list of group IDs.

        Args:
            group_ids: List of group identifiers to generate colors for
            palette_name: Name of the color palette to use ('default', 'turbo',
                'viridis', 'Category20', 'Set1', 'Set3')
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
            palette_name: Name of the color palette. If None, uses the default
            custom_colors: Optional custom color overrides

        Returns:
            Dictionary mapping group IDs to hex color strings
        """
        # Use existing group IDs if none provided
        if group_ids is None:
            group_ids = list(self._group_colors.keys())

        # Use default palette if none specified
        if palette_name is None:
            palette_name = DEFAULT_GROUP_PALETTE

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
            if palette_name == DEFAULT_GROUP_PALETTE:
                return group_palette(n)
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

        available_palettes = self.get_available_palettes()

        if palette_name not in available_palettes:
            raise ValueError(
                f"Palette '{palette_name}' not supported. "
                f"Available palettes: {available_palettes}"
            )

        try:
            # Bokeh's continuous palettes top out at MAX_PALETTE_COLORS entries, so ask
            # for at most that many and cycle them when there are more groups than colors.
            # Reusing a color is a legible degradation; raising here is not.
            palette = list(
                get_palette_colors(palette_name, min(n_colors, self.MAX_PALETTE_COLORS))
            )
            if not palette:
                raise ValueError(f"Palette '{palette_name}' returned no colors")

            colors = [palette[index % len(palette)] for index in range(n_colors)]
            return dict(zip(group_ids, colors))

        except Exception as e:
            # Fall back to the default palette if a specific palette fails
            if palette_name != DEFAULT_GROUP_PALETTE:
                print(
                    f"Warning: {palette_name} palette failed ({e}), "
                    f"falling back to the default"
                )
                return self._generate_palette_colors(group_ids, DEFAULT_GROUP_PALETTE)
            else:
                raise ValueError(f"Failed to generate colors: {e}")

    def get_available_palettes(self) -> List[str]:
        """Get list of available color palettes for groups.

        Returns:
            List of available palette names
        """
        return [DEFAULT_GROUP_PALETTE, "turbo", "viridis", "Category20", "Set1", "Set3"]

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
        return ensure_rgb(color)

    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple.

        Args:
            hex_color: Hex color string (e.g., "#FF0000")

        Returns:
            Tuple[int, int, int]: RGB color values
        """
        return hex_to_rgb(hex_color)

    @staticmethod
    def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to hex color string.

        Args:
            rgb: RGB tuple (r, g, b)

        Returns:
            str: Hex color string
        """
        return rgb_to_hex(rgb)

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
        return gradient_color(base_rgb, target_rgb, fraction)

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
        return self.color_lookup().get_influence_color(
            correlation, influence, max_influence, use_gray_scale, gray_palette
        )

    def update_settings(
        self,
        max_z_score: int = None,
        up_color: Union[str, Tuple[int, int, int]] = None,
        down_color: Union[str, Tuple[int, int, int]] = None,
        neutral_color: Union[str, Tuple[int, int, int]] = None,
        mode: AnalysisWorkspaceMode | None = None,
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

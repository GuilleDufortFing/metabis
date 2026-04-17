"""Data management for metabolic visualization.

This module provides the DataManager class which serves as the main facade
for data operations, including reaction data, gene data, and subsystem data.
It integrates with DataProvider for data retrieval and manages caching,
hover text generation, and color mapping.
"""

from typing import (
    Any,
    Dict,
    List,  # Added Optional for future use
    Optional,
)

# Initialize ExperimentAnalysis for within-group analysis
# No longer need GlyphClass import since we use string element types
from .cache_manager import CacheManager
from .color_manager import ColorManager
from .data_manager_mode import DataManagerMode
from ..visual_config import theme
from .data_provider import DataProvider
from .group_manager import GroupManager
from .hover_manager import HoverManager
from ..visual_config import theme
from ..analysis.subsystem_analysis import SubsystemAnalysis

class DataManager:
    """Handles data retrieval and statistical information for reactions and genes."""

    def __init__(
        self,
        metabolic_model=None,
        reaction_data=None,
        expression_data=None,
        experiment=None, # (legacy, can remove later)
        experiment_analysis=None,
        group_comparison=None,
        normalization_method: str = "mad",
        max_z_score: int = 6,
    ):
        """Initialize data provider with experiment and group comparison.

        Args:
            metabolic_model: MetabolicModel object containing metabolic network
                information
            reaction_data: ReactionData object containing flux data
            expression_data: ExpressionData object containing gene expression data
            experiment: Experiment object containing metabolic model, reaction
                and gene data
            experiment_analysis: Optional ExperimentAnalysis object for
                within-group analysis
            group_comparison: GroupComparison object containing statistical test
                results
        """
        self.metabolic_model = metabolic_model
        self.reaction_data = reaction_data
        self.expression_data = expression_data

        self.group_comparison = group_comparison
        self.experiment_analysis = experiment_analysis

        # Define default delta options
        default_delta_options = {
            "metric": "cohens_d",
            "transform": "sign",
            "numerator_filter": "significant",
            "denominator_filter": "all",
        }

        self.output_reaction_format = "mar_id"
        self.output_gene_format = "geneSymbols"
        self.output_metabolite_format = "metName"

        # Initialize GroupManager for group operations
        self.group_manager = GroupManager()

        # Initialize unified ColorManager with mode-aware configurations
        self.color_manager = ColorManager(
            group_comparison_config={
                "max_z_score": max_z_score,
                "up_color": theme.UP_REGULATED,
                "down_color": theme.DOWN_REGULATED,
                "neutral_color": theme.NEUTRAL,
            },
            experiment_analysis_config={
                "max_z_score": max_z_score,
                "up_color": theme.ACTIVITY_UP,
                "down_color": theme.ACTIVITY_DOWN,
                "neutral_color": theme.NEUTRAL,
            },
            default_mode="group_comparison",
        )
        # Initialize SubsystemAnalysis
        self.subsystem_analysis = SubsystemAnalysis(
            self,
            default_delta_config=default_delta_options,
        )
        self.cache_manager = CacheManager()

        # Determine and set the operation mode
        self.mode = self._determine_mode()

        # Initialize DataProvider with all data sources
        self.data_provider = DataProvider(
            metabolic_model=self.metabolic_model,
            reaction_data=self.reaction_data,
            expression_data=self.expression_data,
            group_comparison=self.group_comparison,
            experiment_analysis=self.experiment_analysis,
            group_manager=self.group_manager,
            subsystem_analysis=self.subsystem_analysis,
        )

        # Initialize HoverManager with unified color manager
        self.hover_manager = HoverManager(
            data_provider=self.data_provider,
            cache_manager=self.cache_manager,
            color_manager=self.color_manager,
            mode=self.mode,
        )

    # ============================================================================
    # Mode Management Methods
    # ============================================================================

    def _determine_mode(self) -> DataManagerMode:
        """Determine the appropriate operation mode based on available data.

        Returns:
            DataManagerMode: The determined mode based on available components
        """
        if self.group_comparison is not None:
            return DataManagerMode.GROUP_COMPARISON
        elif self.experiment_analysis is not None:
            return DataManagerMode.EXPERIMENT_ANALYSIS
        else:
            return DataManagerMode.NO_EXPERIMENT

    def get_mode(self) -> DataManagerMode:
        """Get the current operation mode.

        Returns:
            DataManagerMode: Current operation mode
        """
        return self.mode

    def set_mode(self, mode: DataManagerMode) -> None:
        """Set the operation mode manually.

        This allows switching between modes when multiple data sources are available.
        For example, you might have both GroupComparison and ExperimentAnalysis
        available and want to switch between them.

        Args:
            mode: The desired operation mode

        Raises:
            ValueError: If the requested mode is not available with current data
        """
        if mode == DataManagerMode.GROUP_COMPARISON and self.group_comparison is None:
            raise ValueError(
                "Cannot set GROUP_COMPARISON mode: no GroupComparison object available"
            )
        elif (
            mode == DataManagerMode.EXPERIMENT_ANALYSIS
            and self.experiment_analysis is None
        ):
            raise ValueError(
                "Cannot set EXPERIMENT_ANALYSIS mode: "
                "no ExperimentAnalysis object available"
            )

        self.mode = mode

        # Synchronize ColorManager mode with DataManagerMode
        if mode == DataManagerMode.GROUP_COMPARISON:
            self.color_manager.set_mode(DataManagerMode.GROUP_COMPARISON)
        elif mode == DataManagerMode.EXPERIMENT_ANALYSIS:
            self.color_manager.set_mode(DataManagerMode.EXPERIMENT_ANALYSIS)
        # NO_EXPERIMENT can use either, default to group_comparison
        else:
            self.color_manager.set_mode(DataManagerMode.GROUP_COMPARISON)

        # Clear any cached data that might be mode-dependent
        self.cache_manager.clear_cache()
        # Refresh hover builders for the new mode
        self.hover_manager.refresh_builders(mode)

    def get_available_modes(self) -> list[DataManagerMode]:
        """Get a list of modes that are available with the current data.

        Returns:
            list[DataManagerMode]: List of available modes
        """
        available_modes = [DataManagerMode.NO_EXPERIMENT]  # Always available

        if self.group_comparison is not None:
            available_modes.append(DataManagerMode.GROUP_COMPARISON)

        if self.experiment_analysis is not None:
            available_modes.append(DataManagerMode.EXPERIMENT_ANALYSIS)

        return available_modes

    def is_mode_available(self, mode: DataManagerMode) -> bool:
        """Check if a specific mode is available with current data.

        Args:
            mode: The mode to check

        Returns:
            bool: True if the mode is available
        """
        return mode in self.get_available_modes()

    def get_mode_description(self) -> str:
        """Get a human-readable description of the current mode.

        Returns:
            str: Description of current mode and its capabilities
        """
        if self.mode == DataManagerMode.GROUP_COMPARISON:
            return (
                "GROUP_COMPARISON mode: Provides between-group statistical "
                "comparisons, colors based on log2 fold changes, p-values, "
                "and Cohen's d"
            )
        elif self.mode == DataManagerMode.EXPERIMENT_ANALYSIS:
            return (
                "EXPERIMENT_ANALYSIS mode: Provides within-group analysis, "
                "colors based on within-group z-scores and relative activities"
            )
        else:
            return (
                "NO_EXPERIMENT mode: Basic metadata only, "
                "no statistical analysis or experimental data coloring"
            )

    # ============================================================================
    # Group Management Methods
    # ============================================================================

    def get_active_group(self) -> str:
        """Get the currently active group ID.

        Returns:
            str: ID of the currently active group

        Raises:
            ValueError: If no active group is set
        """
        return self.group_manager.get_active_group()

    def load_groups(
        self,
        criteria: str = "sample_prefix",
        columns: List[str] = None,
        fields: List[int] = None,
    ) -> Dict[str, List[str]]:
        """Organize samples into groups based on specified criteria.

        Args:
            criteria: Method to use for grouping samples. Supports:
                - "sample_prefix": Groups samples by prefix before first underscore
                - "adata_column": Groups samples by values in specified AnnData columns
                - "underscore_split": Groups samples by selected fields
                  after splitting by underscore
            columns: List of column names to use when criteria is "adata_column"
            fields: List of field indices (0-based) to use when criteria
                is "underscore_split"

        Returns:
            Dictionary containing grouping results and status information
        """
        # Get sample IDs from reaction data
        sample_ids = self.reaction_data.get_sample_ids()

        # Delegate to GroupManager for actual grouping logic
        grouping_result = self.group_manager.load_groups(
            sample_ids=sample_ids,
            criteria=criteria,
            columns=columns,
            fields=fields,
            metabolic_model=self.metabolic_model,
            expression_data=self.expression_data,
        )

        # If grouping was successful, generate colors via ColorManager
        if self.group_manager.has_groups():
            group_ids = self.group_manager.get_group_ids()
            group_colors = self.color_manager.generate_group_colors(
                group_ids, palette_name="turbo"
            )

        return grouping_result

    def set_analysis_group(self, group_id: str):
        """Set the current analysis group by ID.

        Args:
            group_id (str): ID of the group to set as active

        Raises:
            ValueError: If the group ID does not exist
        """
        # Delegate to GroupManager
        self.group_manager.set_active_group(group_id)

    def get_group_samples(self, group_id, exclude_group=None) -> List[str]:
        """Get samples associated with a group.

        Args:
            group_id (str): ID of the group. If "The rest", returns all
                samples not in any other group.
            exclude_group (str): Group to exclude when group_id is "The rest"

        Returns:
            List of sample IDs in the group

        Raises:
            ValueError: If requested group doesn't exist and isn't "The rest"
        """
        # Get all sample IDs for "The rest" functionality
        all_sample_ids = self.reaction_data.get_sample_ids()

        # Delegate to GroupManager
        return self.group_manager.get_group_samples(
            group_id=group_id,
            exclude_group=exclude_group,
            all_sample_ids=all_sample_ids,
        )

    def get_group_ids(self) -> List[str]:
        """Return the list of group IDs.

        Returns:
            List of group IDs from the experiment
        """
        return self.group_manager.get_group_ids()

    # ============================================================================
    # Group Color Methods (delegate to ColorManager)
    # ============================================================================

    def get_group_colors(self) -> Dict[str, str]:
        """Get all group colors.

        Returns:
            Dictionary mapping group IDs to hex color codes
        """
        return self.color_manager.get_group_colors()

    def get_group_color(self, group_id: str) -> str:
        """Get color for a specific group.

        Args:
            group_id: ID of the group

        Returns:
            Hex color code for the group
        """
        return self.color_manager.get_group_color(group_id)

    def regenerate_group_colors(
        self, palette_name: str = "turbo", custom_colors: Dict[str, str] = None
    ) -> Dict[str, str]:
        """Regenerate colors for all groups with new palette.

        Args:
            palette_name: Name of color palette to use
            custom_colors: Optional custom color overrides

        Returns:
            Dictionary mapping group IDs to new hex color codes
        """
        group_ids = self.get_group_ids()
        return self.color_manager.regenerate_group_colors(
            group_ids, palette_name, custom_colors
        )

    def set_group_color(self, group_id: str, color: str) -> None:
        """Set custom color for a specific group.

        Args:
            group_id: ID of the group
            color: Hex color code
        """
        self.color_manager.set_custom_group_color(group_id, color)

    # ============================================================================
    # Data Distribution Methods (delegate to DataProvider)
    # ============================================================================

    def get_metric_distribution_data(
        self, metric: str = "log2_fc", for_genes: bool = False
    ) -> Dict[str, Any]:
        """Get metric distribution data for color reference plots.

        Args:
            metric: The metric to get distribution for ('log2_fc', 'cohens_d', etc.)
            for_genes: If True, get gene data; if False, get reaction data

        Returns:
            Dict containing distribution data and metadata
        """
        return self.data_provider.get_metric_distribution_data(metric, for_genes)

    # ============================================================================
    # Update Methods
    # ============================================================================

    def update_experiment_analysis(self, experiment_analysis):  # Can delete
        """Update the experiment analysis object and refresh mode if needed.

        Args:
            experiment_analysis: New ExperimentAnalysis object
        """
        self.experiment_analysis = experiment_analysis

        # Update DataProvider with new experiment analysis
        self.data_provider.experiment_analysis = experiment_analysis

        # Re-determine mode in case this changes available options
        old_mode = self.mode
        new_mode = self._determine_mode()

        # If mode changed, update it and clear caches
        if old_mode != new_mode:
            self.mode = new_mode
            # Synchronize ColorManager mode
            if new_mode == DataManagerMode.EXPERIMENT_ANALYSIS:
                self.color_manager.set_mode(DataManagerMode.EXPERIMENT_ANALYSIS)
            else:
                self.color_manager.set_mode(DataManagerMode.GROUP_COMPARISON)
            self.cache_manager.clear_cache()
            # Refresh hover builders for the new mode
            self.hover_manager.refresh_builders(new_mode)

    def update_group_comparison(self, group_comparison):
        """Update the group comparison object and refresh mode if needed.

        Args:
            group_comparison: New GroupComparison object
        """
        self.group_comparison = group_comparison

        # Update SubsystemAnalysis
        self.subsystem_analysis.set_group_comparison(
            group_comparison
        )

        # Update DataProvider with new group comparison
        self.data_provider.group_comparison = group_comparison

        # Update HoverManager with new group comparison
        self.hover_manager.update_group_comparison(group_comparison)

        # Re-determine mode in case this changes available options
        old_mode = self.mode
        new_mode = self._determine_mode()

        # If mode changed, update it and clear caches
        if old_mode != new_mode:
            self.mode = new_mode
            # Synchronize ColorManager mode
            if new_mode == DataManagerMode.GROUP_COMPARISON:
                self.color_manager.set_mode(DataManagerMode.GROUP_COMPARISON)
            else:
                self.color_manager.set_mode(DataManagerMode.EXPERIMENT_ANALYSIS)
            self.cache_manager.clear_cache()
            # Refresh hover builders for the new mode
            self.hover_manager.refresh_builders(new_mode)

    def update_subsystem_delta_config(self, delta_config: Dict[str, Any]) -> bool:
        """
        Update the delta configuration for subsystem analysis.

        This updates the scoring configuration (metric, transform, filters)
        and recalculates all subsystem delta scores without recomputing
        base statistics.

        Args:
            delta_config: New configuration dict with keys:
                - 'metric': The metric to use (e.g., 'cohens_d', 'log2_fc')
                - 'transform': How to transform values ('sign', 'raw', 'abs')
                - 'numerator_filter': Filter for numerator ('all', 'significant')
                - 'denominator_filter': Filter for denominator ('all', 'significant')

        Returns:
            bool: True if delta stats were successfully recomputed,
                  False if base stats cache is empty.
        """
        success = self.subsystem_analysis.update_delta_config(delta_config)

        # Clear cached hover text that may depend on delta scores
        if success:
            self.cache_manager.clear_cache()

        return success

    # ============================================================================

    # Hover Methods
    # ============================================================================

    def get_element_hover(
        self,
        element_id: str,
        element_type: str,
        use_main_direction: bool = False,
        show_plot: bool = False,
        show_stats: bool = False,
        show_influence: bool = False,
        color_metric: str = "log2_fc",
        delta_options: Optional[Dict[str, Any]] = None,
        show_color_reference: bool = False,
        current_subsystem: Optional[str] = None,
        missing_reason: Optional[str] = None,
    ):
        """Get hover text for an element using HoverManager.

        This is the unified interface for all hover text generation.
        All hover functionality is now delegated to HoverManager.

        Args:
            element_id: ID of the element
            element_type: Type of element ("reaction", "gene", "subsystem")
            show_plot: Whether to include plot information
            show_stats: Whether to show statistics
            show_influence: Whether to show gene influence (for reactions)
            color_metric: Color metric to use
            show_color_reference: Whether to show color reference histogram
            current_subsystem: Active subsystem id/name for context-aware hovers
            missing_reason: If provided, indicates why data is missing

        Returns:
            str: Formatted hover text built by HoverManager
        """
        return self.hover_manager.get_element_hover(
            element_id=element_id,
            element_type=element_type,
            use_main_direction=use_main_direction,
            show_plot=show_plot,
            show_stats=show_stats,
            show_influence=show_influence,
            color_metric=color_metric,
            delta_options=delta_options,
            show_color_reference=show_color_reference,
            current_subsystem=current_subsystem,
            missing_reason=missing_reason,
        )

    # ============================================================================
    # Color Methods
    # ============================================================================

    def get_element_color(
        self,
        element_id: str,
        element_type: str,
        metric_name: str = "default",
        use_main_direction: bool = True,
        delta_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Get color for an element based on current mode.

        Args:
            element_id: ID of the element (reaction, gene, etc.)
            element_type: Type of element ("reaction", "gene", "subsystem")
            metric_name: Name of the metric to use for coloring

        Returns:
            str: Hex color code
        """
        # ColorManager automatically uses the correct configuration based on its mode
        if self.mode == DataManagerMode.GROUP_COMPARISON:
            # Check significance for reactions and genes
            if element_type in ["reaction", "gene"]:
                for_genes = (element_type == "gene")
                # For reactions, if using main direction, determine which direction to check
                if (
                    element_type == "reaction"
                    and use_main_direction
                    and self.experiment_analysis
                ):
                    base_id = element_id
                    if base_id.endswith("_back"):
                        base_id = base_id[:-5]

                    try:
                        direction = self.experiment_analysis.get_main_direction(
                            self.group_comparison.group_id_1, base_id
                        )
                        if direction == "backward":
                            element_id = f"{base_id}_back"
                        elif direction == "forward":
                            element_id = base_id
                        

                    except Exception:
                        # If we can't determine direction, stick with original ID
                        pass

                if self.group_comparison and not self.group_comparison.is_significant(
                    element_id, for_genes=for_genes
                ):
                    return theme.NOT_SIGNIFICANT

            # Handle subsystem coloring (reactions and genes)
            if element_type in ["subsystem", "subsystem_genes"]:
                for_genes = (element_type == "subsystem_genes")

                # Ensure distribution is up-to-date and registered
                # This triggers _precompute_all_delta_stats if delta_options changed
                self.data_provider.get_subsystem_data(
                    element_id,
                    for_genes=for_genes,
                    delta_config=delta_options,
                )

                # Determine the metric name to query
                target_metric = metric_name
                if delta_options:
                    target_metric = delta_options.get("metric", "default")
                elif metric_name == "default":
                    # Try to get from current config if available
                    current_config = self.subsystem_analysis._current_delta_config
                    if current_config:
                        target_metric = current_config.get("metric", "default")

                return self.color_manager.get_element_color_from_distribution(
                    mode=DataManagerMode.GROUP_COMPARISON,
                    element_type=element_type,
                    element_id=element_id,
                    metric_name=target_metric,
                )
                
            return self.color_manager.get_element_color_from_distribution(
                mode=DataManagerMode.GROUP_COMPARISON,
                element_type=element_type,
                element_id=element_id,
                metric_name=metric_name,
            )

        elif self.mode == DataManagerMode.EXPERIMENT_ANALYSIS:
            # Use the new distribution-based coloring
            target_metric = metric_name
            
            # If the metric provided is NOT a valid group ID, assume we want the active group
            # This handles "default", "z_score", "activity", or any other UI label
            if target_metric not in self.get_group_ids():
                try:
                    target_metric = self.get_active_group()
                except ValueError:
                    # If no active group, return neutral
                    return theme.NEUTRAL

            return self.color_manager.get_element_color_from_distribution(
                mode=DataManagerMode.EXPERIMENT_ANALYSIS,
                element_type=element_type,
                element_id=element_id,
                metric_name=target_metric,
            )
        else:
            # NO_EXPERIMENT mode - return neutral color
            return theme.NEUTRAL

    def get_influence_color(
        self, correlation: float, influence: float, max_influence: float = 100.0
    ) -> str:
        """Get color for gene influence visualization using ColorManager.

        Args:
            correlation: The correlation value (-1 to 1)
            influence: The influence magnitude
            max_influence: Maximum influence value for scaling

        Returns:
            str: Hex color code
        """
        # Influence coloring uses the unified color manager
        # The influence palettes are mode-independent
        return self.color_manager.get_influence_color(
            correlation, influence, max_influence
        )

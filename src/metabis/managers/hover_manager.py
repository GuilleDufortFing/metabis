"""Hover text management module.

This module provides a dedicated HoverManager class to handle all hover text
generation, caching, and mode-aware builder management. It consolidates hover
functionality that was previously scattered across DataManager.
"""

from typing import Any, Dict, Optional

from metabis.managers import data_manager

from .data_provider import DataProvider

from ..viz.metabolic_visualization.generation.data_structures.constants import VisualizationConstants

from .data_manager_mode import DataManagerMode
from .hover import presets
from .hover.models import HoverContext

class HoverManager:
    """Manages hover text generation and caching for metabolic visualizations.

    This class handles:
    1. Mode-aware hover builder initialization and management
    2. Context creation for hover building
    3. Element-specific hover text generation (genes, reactions, subsystems)
    4. Hover text caching integration
    5. Unified interface for all hover operations

    The class supports different visualization modes (GROUP_COMPARISON,
    EXPERIMENT_ANALYSIS, NO_EXPERIMENT) and automatically configures the
    appropriate hover builders and styling.
    """

    def __init__(
        self,
        data_provider,
        cache_manager,
        color_manager,
        mode: DataManagerMode = DataManagerMode.NO_EXPERIMENT,
    ):
        """Initialize the HoverManager.

        Args:
            data_provider: DataProvider instance for fetching element data
            cache_manager: CacheManager instance for hover text caching
            color_manager: Unified ColorManager instance with mode awareness
            mode: Current visualization mode
        """
        # Store dependencies
        self.data_provider = data_provider
        self.cache_manager = cache_manager
        self.color_manager = color_manager
        self.group_comparison = data_provider.group_comparison

        # Store mode and initialize builders
        self._mode = mode
        self._setup_hover_builders()

    @property
    def mode(self) -> DataManagerMode:
        return self._mode

    @mode.setter
    def mode(self, new_mode: DataManagerMode):
        if self._mode != new_mode:
            self._mode = new_mode
            self._setup_hover_builders()

    def _setup_hover_builders(self) -> None:
        """Initialize hover builders based on the current mode.

        Creates mode-appropriate hover builders for genes, reactions, and subsystems.
        Different modes use different styling and content strategies.
        """
        self.gene_hover_builder = presets.create_gene_builder_for_mode(self._mode)
        self.reaction_hover_builder = presets.create_reaction_builder_for_mode(self._mode)
        self.metabolite_hover_builder = presets.create_metabolite_builder_for_mode(self._mode)
        self.subsystem_hover_builder = presets.create_subsystem_builder_for_mode(self._mode)

        if self._mode == DataManagerMode.GROUP_COMPARISON:
            # Group Comparison styling
            style = (
                "font-family: Arial, sans-serif; font-size:12px; padding:10px; "
                "border:1px solid #ddd; border-radius:4px; background-color:rgba(255, 255, 255, 0.95); "
                "box-shadow: 0 2px 5px rgba(0,0,0,0.1); max-width: 700px;"
            )
            self.gene_hover_builder.set_container_style(style)
            self.reaction_hover_builder.set_container_style(style)
            self.metabolite_hover_builder.set_container_style(style)
            self.subsystem_hover_builder.set_container_style(style)

        elif self._mode == DataManagerMode.EXPERIMENT_ANALYSIS:
            # Experiment Analysis styling
            style = (
                "font-family: Arial, sans-serif; font-size:12px; padding:10px; "
                "border:1px solid #ddd; border-radius:4px; background-color:rgba(255, 255, 255, 0.95); "
                "box-shadow: 0 2px 5px rgba(0,0,0,0.1); max-width: 700px;"
            )
            self.gene_hover_builder.set_container_style(style)
            self.reaction_hover_builder.set_container_style(style)
            self.metabolite_hover_builder.set_container_style(style)
            self.subsystem_hover_builder.set_container_style(style)

        else:
            # No Experiment styling
            style = (
                "font-family: Arial, sans-serif; font-size:12px; padding:8px; "
                "border:1px solid #ddd; border-radius:4px; background-color:#f9f9f9;"
            )
            self.gene_hover_builder.set_container_style(style)
            self.reaction_hover_builder.set_container_style(style)
            self.metabolite_hover_builder.set_container_style(style)
            self.subsystem_hover_builder.set_container_style(style)

    def refresh_builders(self, new_mode: Optional[DataManagerMode] = None) -> None:
        """Refresh hover builders when mode changes.

        Args:
            new_mode: New mode to set (if None, uses current mode)
        """
        if new_mode is not None:
            self._mode = new_mode

        # Reinitialize builders with new mode
        self._setup_hover_builders()

    def _create_context(
        self,
        hover_type: str = "reaction",
        subsystem_metric: str = "log2_fc",
        main_direction: str = "forward",
        for_genes: bool = False,
        element_type: str = None,
        element_id: str = None,
        show_plot: bool = False,
        show_stats: bool = False,
        show_influence: bool = False,
        color_metric: str = "log2_fc",
        show_color_reference: bool = False,
        current_subsystem: Optional[str] = None,
    ) -> HoverContext:
        """Create context dictionary for hover building.

        The context provides mode-specific styling and configuration
        information to hover builders.

        Args:
            hover_type: Type of hover being built ("reaction", "gene", "subsystem")
            subsystem_metric: Metric being used for subsystem analysis
            for_genes: Whether this is for gene-level analysis
            element_type: Alternative way to specify element type
            show_plot: Whether to include plot information
            show_stats: Whether to show statistics
            show_influence: Whether to show gene influence (for reactions)
            color_metric: Color metric to use
            show_color_reference: Whether to show color reference histogram
            current_subsystem: Active subsystem id/name (for context-aware rendering)

        Returns:
            HoverContext: Context object with mode-specific configuration
        """
        # Determine element type
        if element_type is not None:
            hover_type = element_type

        # Resolve current category if possible
        current_category = None
        if current_subsystem:
            metabolic_model = getattr(self.data_provider, "metabolic_model", None)
            if metabolic_model is not None:
                try:
                    current_category = metabolic_model.get_subsystem_category(current_subsystem)
                except Exception:
                    current_category = None

        # Create base context
        context = HoverContext(
            mode=self._mode,
            hover_type=hover_type,
            element_type=hover_type,
            element_id=element_id,
            subsystem_metric=subsystem_metric,
            main_direction=main_direction,
            for_genes=for_genes,
            show_plot=show_plot,
            show_stats=show_stats,
            show_influence=show_influence,
            color_metric=color_metric,
            show_color_reference=show_color_reference,
            current_subsystem=current_subsystem,
            current_category=current_category,
            data_provider=self.data_provider,
            color_manager=self.color_manager,
        )

        # Add mode-specific context
        if self._mode == DataManagerMode.GROUP_COMPARISON:
            context.group_comparison = self.group_comparison
            if show_stats:
                context.show_comparison = True
            else:
                context.show_comparison = False

        elif self._mode == DataManagerMode.EXPERIMENT_ANALYSIS:
            context.show_comparison = False
            context.analysis_type = "within_group"

        else:  # NO_EXPERIMENT
            context.show_stats = False
            context.show_comparison = False

        return context

    def update_group_comparison(self, group_comparison):
        """Update the group comparison object.

        Args:
            group_comparison: New GroupComparison object
        """
        self.group_comparison = group_comparison

    # ============================================================================
    # Gene Hover Methods
    # ============================================================================

    def build_gene_hover(
        self,
        gene_id: str,
        show_plot: bool = False,
        show_stats: bool = False,
        color_metric: str = "log2_fc",
        show_color_reference: bool = False,
    ) -> str:
        """Build hover text for a gene (without caching)."""
        # Get comprehensive gene data
        gene_data = self.data_provider.get_gene_data(
            gene_id, include_stats=True, include_distributions=show_plot
        )

        # Create context for hover building
        context = self._create_context(
            element_type="gene",
            show_plot=show_plot,
            show_stats=show_stats,
            color_metric=color_metric,
            show_color_reference=show_color_reference,
        )

        # Build hover text and get color
        return self.gene_hover_builder.build(gene_data, context)

    def get_gene_hover(
        self,
        gene_id: str,
        show_plot: bool = False,
        show_stats: bool = False,
        color_metric: str = "log2_fc",
        show_color_reference: bool = False,
    ) -> str:
        """Get cached hover text for a gene, or build and cache if not exists."""
        hover_key = (
            f"gene_{gene_id}_plot_{show_plot}_stats_{show_stats}_metric_{color_metric}_"
            f"colorref_{show_color_reference}"
        )

        return self._get_cached_result(
            hover_key,
            lambda: self.build_gene_hover(
                gene_id, show_plot, show_stats, color_metric, show_color_reference
            ),
        )

    # ============================================================================
    # Reaction Hover Methods
    # ============================================================================

    def build_reaction_hover(
        self,
        reaction_id: str,
        main_direction: str = "forward",
        show_stats: bool = False,
        show_plot: bool = False,
        show_influence: bool = False,
        color_metric: str = "log2_fc",
        show_color_reference: bool = False,
    ) -> str:
        """Build hover text for a reaction (without caching)."""
        # Get comprehensive reaction data, which includes both directions

        reaction_id_clean = reaction_id.replace("_back", "")

        reaction_data = self.data_provider.get_reaction_data(
            reaction_id_clean,
            include_stats=show_stats,
            include_influence=show_influence,
            include_distributions=show_plot,
        )
        
        # Create context for hover building
        context = self._create_context(
            element_type="reaction",
            element_id=reaction_id,
            main_direction=main_direction,
            show_stats=show_stats,
            show_plot=show_plot,
            show_influence=show_influence,
            color_metric=color_metric,
            show_color_reference=show_color_reference,
        )

        # Build hover text and get color
        return self.reaction_hover_builder.build(reaction_data, context)

    def get_reaction_hover(
        self,
        reaction_id: str,
        use_main_direction: bool = False,
        show_stats: bool = False,
        show_plot: bool = False,
        show_influence: bool = False,
        color_metric: str = "log2_fc",
        show_color_reference: bool = False,
    ) -> str:
        """Get cached hover text for a reaction, or build and cache if not exists."""
        
        main_direction = None
        if self._mode == DataManagerMode.GROUP_COMPARISON:
            main_direction = self.data_provider.experiment_analysis.get_main_direction(
                self.group_comparison.group_id_1, reaction_id
            )
        else:
            direction = self.data_provider.experiment_analysis.get_main_direction(
                self.data_provider.group_manager.get_active_group(), reaction_id
            )

        if use_main_direction and main_direction is not None:
            if main_direction == "backward":
                reaction_id = f"{reaction_id}_back"

        hover_key = (
            f"reaction_{reaction_id}_stats_{show_stats}_plot_{show_plot}_"
            f"influence_{show_influence}_metric_{color_metric}_"
            f"colorref_{show_color_reference}"
        )

        return self._get_cached_result(
            hover_key,
            lambda: self.build_reaction_hover(
                reaction_id,
                main_direction=main_direction,
                show_stats=show_stats,
                show_plot=show_plot,
                show_influence=show_influence,
                color_metric=color_metric,
                show_color_reference=show_color_reference,
            ),
        )

    # ============================================================================
    # Metabolite Hover Methods
    # ============================================================================

    def build_metabolite_hover(
        self,
        metabolite_id: str,
        color_metric: str = "log2_fc",
        show_color_reference: bool = False,
        current_subsystem: Optional[str] = None,
    ) -> str:
        """Build hover text for a metabolite (without caching)."""
        # Get basic metabolite information from the model
        metabolite_data = self.data_provider.get_metabolite_data(
            metabolite_id
        )
        
        # Create context for hover building
        context = self._create_context(            
            element_type="metabolite",
            color_metric=color_metric,
            show_color_reference=show_color_reference,
            current_subsystem=current_subsystem,
        )

        # Build hover text
        return self.metabolite_hover_builder.build(metabolite_data, context)

    def get_metabolite_hover(
        self,
        metabolite_id: str,
        color_metric: str = "log2_fc",
        show_color_reference: bool = False,
        current_subsystem: Optional[str] = None,
    ) -> str:
        """Get cached hover text for a metabolite, or build and cache if not exists."""
        cs_key = current_subsystem or "none"
        hover_key = (
            f"metabolite_{metabolite_id}_metric_{color_metric}_"
            f"colorref_{show_color_reference}_cs_{cs_key}"
        )

        return self._get_cached_result(
            hover_key,
            lambda: self.build_metabolite_hover(
                metabolite_id, color_metric, show_color_reference, current_subsystem
            ),
        )

    # ============================================================================
    # Subsystem Hover Methods
    # ============================================================================

    def build_subsystem_hover(
        self,
        subsystem_name: str,
        subsystem_metric: str = "log2_fc",
        for_genes: bool = False,
        show_stats: bool = False,
        show_color_reference: bool = False,
        delta_options: Dict[str, Any] = None,
    ) -> str:
        """Build hover text for subsystem/pathway bars in group comparison plots."""
        # Get full subsystem data (includes totals and regulation dicts)
        subsystem_data = self.data_provider.get_subsystem_data(
            subsystem_name, for_genes, delta_options
        )

        if not subsystem_data.get("is_in_model", False):
            # If subsystem not found, return empty hover text
            return ""

        model_subsystem = self.data_provider.metabolic_model.get_subsystem(subsystem_name)
        display_name = model_subsystem.name if model_subsystem else subsystem_name

        subsystem_data["subsystem_display_name"] = display_name

        # Add/normalize fields expected by SubsystemSection without dropping dicts
        total_metric_values = subsystem_data.get("total_metric_values", {})
        regulation_scores = subsystem_data.get("regulation_scores", {})
        subsystem_data["subsystem_name"] = subsystem_name
        subsystem_data["impact_score"] = subsystem_data.get("impact_score", 0.0)
        subsystem_data["n_significant"] = subsystem_data.get(
            "significant_elements", 0
        )
        subsystem_data["n_total"] = subsystem_data.get("total_elements", 0)
        subsystem_data["regulation_score"] = regulation_scores.get(subsystem_metric, 0.0)
        subsystem_data["total_metric_value"] = total_metric_values.get(subsystem_metric, 0.0)

        # Create context for hover building
        context = self._create_context(
            hover_type="subsystem",
            subsystem_metric=subsystem_metric,
            color_metric=subsystem_metric,
            for_genes=for_genes,
            show_stats=show_stats,
            show_color_reference=show_color_reference,
        )

        # If Experiment Analysis mode, attach EA stats and group_id for consistency
        from .data_manager_mode import DataManagerMode
        if context.get("mode") == DataManagerMode.EXPERIMENT_ANALYSIS:
            try:
                subsystem_data["experiment_stats"] = self.data_provider.get_experiment_analysis_statistics(
                    subsystem_name, "subsystem", None
                )
            except Exception:
                pass
            active_group = getattr(self.data_provider.group_manager, "active_group", None)
            if active_group:
                context["group_id"] = active_group

        # Build hover text
        return self.subsystem_hover_builder.build(subsystem_data, context)

    def get_subsystem_hover(
        self,
        subsystem_name: str,
        subsystem_metric: str = "log2_fc",
        delta_options: Dict[str, Any] = None,
        for_genes: bool = False,
        show_stats: bool = False,
        show_color_reference: bool = False,
    ) -> str:
        """Get cached hover text for a subsystem/pathway.

        Builds and caches if not exists.
        """
        # Create a unique key for this subsystem hover (include for_genes in key)
        element_type = "gene" if for_genes else "reaction"

        # Build a string representation of delta_options for the cache key
        delta_str = ""
        if delta_options:
            # Sort keys for consistency
            delta_items = sorted(delta_options.items())
            delta_str = "_".join(f"{k}-{v}" for k, v in delta_items)
        hover_key = (
            f"subsystem_{subsystem_name}_{subsystem_metric}_{element_type}"
            f"_stats_{show_stats}_colorref_{show_color_reference}"
        )
        if delta_str:
            hover_key += f"_delta_{delta_str}"

        return self._get_cached_result(
            hover_key,
            lambda: self.build_subsystem_hover(
                subsystem_name=subsystem_name,
                subsystem_metric=subsystem_metric,
                for_genes=for_genes,
                show_stats=show_stats,
                show_color_reference=show_color_reference,
                delta_options=delta_options,
            ),
        )

    # ============================================================================
    # Unified Hover Interface
    # ============================================================================

    def get_element_hover(
        self,
        element_id: str,
        element_type: str,
        use_main_direction: bool = False,
        show_plot: bool = False,
        show_stats: bool = False,
        show_color_reference: bool = False,
        show_influence: bool = False,
        color_metric: str = "log2_fc",
        delta_options: Dict[str, Any] = None,
        current_subsystem: Optional[str] = None,
        missing_reason: Optional[str] = None,
    ):
        """Get hover text for an element using modern builders.

        Args:
            element_id: ID of the element
            element_type: Type of element ("reaction", "gene", "subsystem")
            use_main_direction: Whether to use main direction for reactions
            show_plot: Whether to include plot information
            show_stats: Whether to show statistics
            show_influence: Whether to show gene influence (for reactions)
            color_metric: Color metric to use
            show_color_reference: Whether to show color reference histogram
            current_subsystem: Active subsystem id/name for context-aware hovers
            missing_reason: Optional reason why data is missing (for basic hover)

        Returns:
            str: Formatted hover text built by modern hover builders
        """

        try:

            # Use modern hover builders based on element type
            if element_type == "reaction":
                return self.get_reaction_hover(
                    element_id,
                    use_main_direction=use_main_direction,
                    show_stats=show_stats,
                    show_plot=show_plot,
                    show_influence=show_influence,
                    color_metric=color_metric,
                    show_color_reference=show_color_reference,
                )


            elif element_type == "gene":
                return self.get_gene_hover(
                    element_id,
                    show_plot=show_plot,
                    show_stats=show_stats,
                    color_metric=color_metric,
                    show_color_reference=show_color_reference,
                )
            elif element_type == "subsystem":
                return self.get_subsystem_hover(
                    subsystem_name=element_id,
                    subsystem_metric=color_metric,
                    for_genes=False,
                    show_stats=show_stats,
                    show_color_reference=show_color_reference,
                    delta_options=delta_options,
                )
            elif element_type == "metabolite":
                # Metabolites don't have experimental stats; just return their hover
                return self.get_metabolite_hover(
                    element_id,
                    color_metric=color_metric,
                    show_color_reference=(False if missing_reason == "no_experimental_data" else show_color_reference),
                    current_subsystem=current_subsystem,
                )
            else:
                # For unknown element types, use basic builder with minimal data
                basic_data = {
                    "is_in_model": True,
                    "element_id": element_id,
                    "element_type": element_type,
                }
                context = self._create_context(element_type=element_type)

                # Use gene builder as fallback for basic elements
                return self.gene_hover_builder.build(basic_data, context)

        except Exception as e:
            # Fallback for any errors
            return f"{element_type.title()}: {element_id} (Error: {str(e)})"

    def _get_cached_result(self, hover_key: str, build_callback) -> str:
        """Generic method to handle caching logic for hover text.

        Args:
            hover_key: Unique key for the specific hover content
            build_callback: Function that returns the hover text if not cached

        Returns:
            str: The hover text (either from cache or newly built)
        """
        cache_key = self.cache_manager.get_cache_key(self.group_comparison)

        # If no cache key (no group comparison), build directly
        if cache_key is None:
            return build_callback()

        # Handle cache key changes
        self.cache_manager.invalidate_on_key_change(cache_key)

        # Try to get from cache
        cached_hover = self.cache_manager.get_cached_hover(cache_key, hover_key)
        if cached_hover is not None:
            return cached_hover

        # Build and cache
        hover_text = build_callback()
        self.cache_manager.cache_hover(cache_key, hover_key, hover_text)

        return hover_text

"""Hover text management module.

This module provides a dedicated HoverManager class to handle all hover text
generation, caching, and mode-aware builder management. It consolidates hover
functionality that was previously scattered across AnalysisWorkspace.
"""

import logging
from typing import Any, Dict, Optional

from .element_presentation import (
    comparison_direction_criterion,
    comparison_display_direction,
)
from .hover import presets
from .hover.models import HoverContext
from .hover.panel import PANEL_STYLE
from .hover.sections.subsystem_info import subsystem_hover_data
from .workspace_mode import AnalysisWorkspaceMode

logger = logging.getLogger(__name__)


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
        mode: AnalysisWorkspaceMode = AnalysisWorkspaceMode.NO_EXPERIMENT,
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
        self.group_comparison_artifact = data_provider.group_comparison_artifact

        # Store mode and initialize builders
        self._mode = mode
        self._setup_hover_builders()

    @property
    def mode(self) -> AnalysisWorkspaceMode:
        return self._mode

    @mode.setter
    def mode(self, new_mode: AnalysisWorkspaceMode):
        if self._mode != new_mode:
            self._mode = new_mode
            self._setup_hover_builders()

    def _setup_hover_builders(self) -> None:
        """Initialize hover builders based on the current mode.

        Creates mode-appropriate hover builders for genes, reactions, and subsystems.
        Different modes use different content strategies, in one panel style.
        """
        self.gene_hover_builder = presets.create_gene_builder_for_mode(self._mode)
        self.reaction_hover_builder = presets.create_reaction_builder_for_mode(self._mode)
        self.metabolite_hover_builder = presets.create_metabolite_builder_for_mode(self._mode)
        self.subsystem_hover_builder = presets.create_subsystem_builder_for_mode(self._mode)

        # Every mode draws its panels alike, as the subsystem graph's reactions and
        # genes are drawn in a comparison (panel.PANEL_STYLE).
        for builder in (
            self.gene_hover_builder,
            self.reaction_hover_builder,
            self.metabolite_hover_builder,
            self.subsystem_hover_builder,
        ):
            builder.set_container_style(PANEL_STYLE)

    def refresh_builders(self, new_mode: Optional[AnalysisWorkspaceMode] = None) -> None:
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
            direction_criterion=comparison_direction_criterion(
                getattr(self.data_provider, "subsystem_analysis", None)
            ),
            data_provider=self.data_provider,
            color_manager=self.color_manager,
        )

        # Add mode-specific context
        if self._mode == AnalysisWorkspaceMode.GROUP_COMPARISON:
            context.group_comparison = self.group_comparison_artifact
            if show_stats:
                context.show_comparison = True
            else:
                context.show_comparison = False

        elif self._mode == AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS:
            context.show_comparison = False
            context.analysis_type = "within_group"

        else:  # NO_EXPERIMENT
            context.show_stats = False
            context.show_comparison = False

        return context

    def update_group_comparison(self, artifact):
        """Update the installed comparison artifact used for hover.

        Args:
            artifact: New GroupComparisonArtifact, or None outside comparison mode.
        """
        self.group_comparison_artifact = artifact

    def _resolve_main_direction(self, reaction_id: str) -> Optional[str]:
        """Direction shown for a reversible reaction under the current mode.

        In comparison mode this is the shared reaction-direction map's selection,
        so the hover names the direction the arrowhead and the node colour were
        drawn from; in activity mode it follows the selected group's own summary.
        Previously the activity branch computed a direction and discarded it, so
        `use_main_direction` never took effect outside comparison mode.
        """
        experiment_analysis = getattr(self.data_provider, "experiment_analysis", None)
        if experiment_analysis is None:
            return None

        try:
            if self._mode == AnalysisWorkspaceMode.GROUP_COMPARISON and self.group_comparison_artifact:
                base_id = (
                    reaction_id[:-5]
                    if reaction_id.endswith("_back")
                    else reaction_id
                )
                return comparison_display_direction(
                    getattr(self.data_provider, "subsystem_analysis", None),
                    experiment_analysis,
                    self.group_comparison_artifact,
                    base_id,
                )
            activity_artifact = getattr(self.data_provider, "activity_artifact", None)
            if (
                self._mode == AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS
                and activity_artifact is not None
            ):
                return experiment_analysis.get_activity_direction(
                    activity_artifact, reaction_id
                )
        except Exception:
            return None

        return None

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
        main_direction = self._resolve_main_direction(reaction_id)

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

    def _build_activity_subsystem_hover(
        self,
        subsystem_name: str,
        show_stats: bool,
        show_color_reference: bool,
    ) -> str:
        """Build a subsystem hover from the activity context.

        Activity mode must not reach into SubsystemAnalysis: its statistics are
        delta scores over significant elements, which require a GroupComparison
        that does not exist here.
        """
        metabolic_model = self.data_provider.metabolic_model
        model_subsystem = (
            metabolic_model.get_subsystem(subsystem_name) if metabolic_model else None
        )
        if model_subsystem is None:
            return ""

        subsystem_data = {
            "is_in_model": True,
            "subsystem_name": subsystem_name,
            "subsystem_display_name": model_subsystem.name,
            "experiment_stats": self.data_provider.get_activity_statistics(
                subsystem_name, "subsystem"
            ),
        }

        context = self._create_context(
            hover_type="subsystem",
            element_id=subsystem_name,
            show_stats=show_stats,
            show_color_reference=show_color_reference,
        )
        return self.subsystem_hover_builder.build(subsystem_data, context)

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
        if self._mode == AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS:
            return self._build_activity_subsystem_hover(
                subsystem_name,
                show_stats=show_stats,
                show_color_reference=show_color_reference,
            )

        # Get full subsystem data (includes totals and net-direction dicts)
        subsystem_data = self.data_provider.get_subsystem_data(
            subsystem_name, for_genes, delta_options
        )

        if not subsystem_data.get("is_in_model", False):
            # If subsystem not found, return empty hover text
            return ""

        model_subsystem = self.data_provider.metabolic_model.get_subsystem(subsystem_name)
        display_name = model_subsystem.name if model_subsystem else subsystem_name
        subsystem_data = subsystem_hover_data(
            subsystem_data, subsystem_name=subsystem_name, display_name=display_name, metric=subsystem_metric
        )

        # Create context for hover building
        context = self._create_context(
            hover_type="subsystem",
            subsystem_metric=subsystem_metric,
            color_metric=subsystem_metric,
            for_genes=for_genes,
            show_stats=show_stats,
            show_color_reference=show_color_reference,
        )

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
            elif element_type in ("subsystem", "subsystem_genes"):
                # A subsystem read through its genes, as the overview's gene overlay draws it.
                return self.get_subsystem_hover(
                    subsystem_name=element_id,
                    subsystem_metric=color_metric,
                    for_genes=element_type == "subsystem_genes",
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
            # A hover must never break a render, but swallowing the failure silently
            # is how a broken data path reaches the user as card text while the
            # server log, the deterministic suites, and the mounted smokes stay green.
            logger.exception(
                "Hover build failed for %s '%s'; rendering an error card",
                element_type,
                element_id,
            )
            return f"{element_type.title()}: {element_id} (Error: {str(e)})"

    def _get_cached_result(self, hover_key: str, build_callback) -> str:
        """Generic method to handle caching logic for hover text.

        Args:
            hover_key: Unique key for the specific hover content
            build_callback: Function that returns the hover text if not cached

        Returns:
            str: The hover text (either from cache or newly built)
        """
        active_group = None
        group_manager = getattr(self.data_provider, "group_manager", None)
        if group_manager is not None:
            active_group = group_manager.get_active_group()
        activity_artifact = getattr(self.data_provider, "activity_artifact", None)
        cache_key = self.cache_manager.get_cache_key(
            self.group_comparison_artifact,
            mode=getattr(self.mode, "value", str(self.mode)),
            active_group=active_group,
            activity_key=activity_artifact.metric_key if activity_artifact else None,
        )

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

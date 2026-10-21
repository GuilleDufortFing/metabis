"""How one element is painted and which way it is drawn, without a workspace.

Graph enrichment asks two presentation questions per element: what colour is it,
and -- for a reversible reaction -- which direction is displayed. Both answers
were implemented on ``AnalysisWorkspace``, which is why a graph worker had to be
handed a whole reconstructed workspace to render anything (AM10 inventory,
part 2).

Neither answer actually needs a workspace. Both are functions of the same seven
plain collaborators: the mode, the installed comparison and activity artifacts,
the experiment analysis, the data provider, the subsystem analysis, and a colour
reader. This module states that by taking those collaborators as a protocol, so
``AnalysisWorkspace`` and ``GraphCalculationInput`` can both satisfy it and share
one implementation. The workspace delegates rather than keeping its own copy: a
second implementation would let a worker and the document thread paint the same
element differently, and no type check would see it (the MR-08 shape).

The protocol is structural and its member names match the workspace's existing
attributes, so the workspace passes ``self``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from ..analysis.activity_analysis import displayed_feature_id
from ..visual_config import theme
from .workspace_mode import AnalysisWorkspaceMode


class ElementPresentationContext(Protocol):
    """The collaborators an element's colour and direction are resolved from."""

    @property
    def mode(self) -> AnalysisWorkspaceMode: ...

    @property
    def experiment_analysis(self) -> Any: ...

    @property
    def group_comparison_artifact(self) -> Any: ...

    @property
    def activity_artifact(self) -> Any: ...

    @property
    def activity_overlay(self) -> Any: ...

    @property
    def data_provider(self) -> Any: ...

    @property
    def subsystem_analysis(self) -> Any: ...

    @property
    def color_manager(self) -> Any: ...


def resolve_element_color(
    context: ElementPresentationContext,
    element_id: str,
    element_type: str,
    metric_name: str = "default",
    use_main_direction: bool = True,
    delta_options: Optional[Dict[str, Any]] = None,
) -> str:
    """Colour for one element under the context's current mode.

    Args:
        context: The installed scientific and colour collaborators.
        element_id: ID of the element (reaction, gene, subsystem).
        element_type: Type of element ("reaction", "gene", "subsystem",
            "subsystem_genes").
        metric_name: Name of the metric to colour by.
        use_main_direction: Whether a reversible reaction is read in its
            displayed direction.
        delta_options: Subsystem delta configuration, when one applies.

    Returns:
        str: Hex color code.
    """
    if context.mode == AnalysisWorkspaceMode.GROUP_COMPARISON:
        comparison = context.group_comparison_artifact
        # Check significance for reactions and genes
        if element_type in ["reaction", "gene"]:
            for_genes = element_type == "gene"
            # For reactions, if using main direction, determine which direction
            # to check
            if (
                element_type == "reaction"
                and use_main_direction
                and context.experiment_analysis
            ):
                base_id = element_id
                if base_id.endswith("_back"):
                    base_id = base_id[:-5]

                try:
                    direction = comparison_display_direction(
                        context.subsystem_analysis,
                        context.experiment_analysis,
                        comparison,
                        base_id,
                    )
                    if direction == "backward":
                        element_id = f"{base_id}_back"
                    elif direction == "forward":
                        element_id = base_id
                except Exception:
                    # If we can't determine direction, stick with original ID
                    pass

            if comparison and not comparison.is_significant(
                element_id, for_genes=for_genes
            ):
                return theme.NOT_SIGNIFICANT

        # Handle subsystem coloring (reactions and genes)
        if element_type in ["subsystem", "subsystem_genes"]:
            for_genes = element_type == "subsystem_genes"

            # Ensure distribution is up-to-date and registered. This triggers
            # _precompute_all_delta_stats if delta_options changed.
            context.data_provider.get_subsystem_data(
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
                current_config = context.subsystem_analysis._current_delta_config
                if current_config:
                    target_metric = current_config.get("metric", "default")

            # A subsystem with no score has no data, and draws neutral grey
            # (Supplementary Section 1.5), not the white of a measured zero.
            if not context.color_manager.has_value(
                AnalysisWorkspaceMode.GROUP_COMPARISON,
                element_type,
                element_id,
                target_metric,
            ):
                return theme.NOT_SIGNIFICANT
            return context.color_manager.get_element_color_from_distribution(
                mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
                element_type=element_type,
                element_id=element_id,
                metric_name=target_metric,
            )

        return context.color_manager.get_element_color_from_distribution(
            mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
            element_type=element_type,
            element_id=element_id,
            metric_name=metric_name,
        )

    if context.mode == AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS:
        artifact = context.activity_artifact
        if artifact is None:
            return theme.NEUTRAL
        overlay = context.data_provider.activity_overlay_for_element(element_type)

        # Overview graphs use the selected overlay. Detailed subsystem graphs
        # resolve reaction and gene nodes to their own family through the provider.
        # Gray, not the neutral center of the scale, keeps unavailable evidence
        # from being read as "average activity".
        if not artifact.colors_element_type(overlay, element_type):
            return theme.ACTIVITY_UNAVAILABLE

        features = artifact.feature_result_for(overlay)
        if element_type in ("subsystem", "subsystem_genes"):
            row = artifact.subsystem_row(overlay, element_id)
            if row is None or row.score is None:
                # Not measured, or below the coverage threshold: say so with gray
                # rather than imply a neutral measurement.
                return theme.ACTIVITY_UNAVAILABLE
            lookup_id = element_id
            lookup_type = "subsystem"
        else:
            lookup_type = element_type
            lookup_id = element_id
            if element_type == "reaction" and use_main_direction:
                lookup_id = displayed_feature_id(features.activity_score, element_id)
            if lookup_id not in features.activity_score.index:
                return theme.ACTIVITY_UNAVAILABLE

        return context.color_manager.get_element_color_from_distribution(
            mode=AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS,
            element_type=lookup_type,
            element_id=lookup_id,
            metric_name=artifact.metric_key_for(overlay),
        )

    # NO_EXPERIMENT mode - return neutral color
    return theme.NEUTRAL


def installed_direction_map(subsystem_analysis: Any) -> Any:
    """The reaction-direction map currently installed, or None."""
    if subsystem_analysis is None:
        return None
    getter = getattr(subsystem_analysis, "get_reaction_direction_map", None)
    return getter() if getter is not None else None


def comparison_direction_criterion(subsystem_analysis: Any) -> Any:
    """The criterion the installed map actually selected directions with.

    Read from the map rather than from the workspace attribute so a surface that
    *names* the deciding population cannot name a different one than the map that
    chose the direction it is describing.
    """
    direction_map = installed_direction_map(subsystem_analysis)
    return direction_map.criterion if direction_map is not None else None


def comparison_display_direction(
    subsystem_analysis: Any,
    experiment_analysis: Any,
    comparison: Any,
    base_reaction_id: str,
) -> Optional[str]:
    """Displayed direction for one base reaction under an installed comparison.

    The answer is the installed :class:`~metabis.analysis.reaction_direction.
    ReactionDirectionMap` -- the same immutable selection Delta score and ORA
    consume -- so the arrowhead, the node colour, and the hover cannot disagree
    about which direction a reversible reaction is being read in.

    Reading the map rather than ``get_main_direction`` also fixes two things the
    per-group averages frame cannot express: the map honours the user's
    Target/Baseline **criterion**, and it is keyed by the comparison's own mean
    columns, so a Target composed of several base groups resolves. ``group_id_1``
    is ``"A + B"`` for such a union and is not a row of the averages frame at all.

    Falls back to the averages frame only when no map is installed, which is the
    pre-comparison state where there is nothing for the two to disagree about.
    """
    direction_map = installed_direction_map(subsystem_analysis)

    if direction_map is not None:
        entry = direction_map.get(base_reaction_id)
        return entry.selected_direction.value if entry is not None else None

    if experiment_analysis is None or comparison is None:
        return None
    return experiment_analysis.get_main_direction(comparison.group_id_1, base_reaction_id)


def resolve_display_direction(
    context: ElementPresentationContext,
    reaction_id: str,
) -> Optional[str]:
    """Direction shown for a reversible reaction under the current context."""
    if context.experiment_analysis is None:
        return None
    base_id = reaction_id[:-5] if reaction_id.endswith("_back") else reaction_id

    if context.mode == AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS and (
        context.activity_artifact
    ):
        # Single-group activity has no Target/Baseline, so the shared criterion
        # does not apply. Direction follows the selected group's own summary,
        # which may use a median while the averages table always used a mean.
        return context.experiment_analysis.get_activity_direction(
            context.activity_artifact, base_id
        )
    if context.mode == AnalysisWorkspaceMode.GROUP_COMPARISON and (
        context.group_comparison_artifact
    ):
        return comparison_display_direction(
            context.subsystem_analysis,
            context.experiment_analysis,
            context.group_comparison_artifact,
            base_id,
        )
    return None

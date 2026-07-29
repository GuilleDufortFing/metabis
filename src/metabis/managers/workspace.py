"""The Analysis Workspace facade for metabolic visualization.

This module provides the AnalysisWorkspace class which serves as the main facade
for data operations, including reaction data, gene data, and subsystem data.
It integrates with DataProvider for data retrieval and manages caching,
hover text generation, and color mapping.
"""

from dataclasses import replace
from math import isfinite
from typing import (
    Any,
    Dict,
    Iterable,
    List,  # Added Optional for future use
    Mapping,
    Optional,
    Tuple,
)
from uuid import uuid4

from ..analysis.activity_analysis import (
    ActivityArtifact,
    ActivityConfig,
    ActivityContext,
    ActivityOverlay,
)
from ..analysis.artifacts import ArtifactProvenance, fingerprint_json
from ..analysis.comparison_artifact import (
    ComparisonThresholds,
    GroupComparisonArtifact,
)
from ..analysis.dimensionality_reduction import ReductionArtifact
from ..analysis.group_comparison import register_comparison_distributions
from ..analysis.reaction_direction import (
    DEFAULT_REACTION_DIRECTION_CRITERION,
    ReactionDirectionCriterion,
)
from ..analysis.reaction_set_enrichment import ReactionSetEnrichmentArtifact
from ..analysis.set_enrichment import (
    SetEnrichmentArtifact,
    SetEnrichmentFeatureType,
)
from ..analysis.source_identity import SourceIdentityCache
from ..analysis.statistical_analysis import StatisticalAnalysis
from ..analysis.subsystem_analysis import SubsystemAnalysis, SubsystemDeltaArtifact
from ..analysis_workspace import (
    AnalysisSnapshot,
    RevisionKind,
    StableFrameReference,
    StableMatrixReference,
    StaleAnalysisResultError,
    WorkspaceProvenance,
    WorkspaceRevisions,
    normalize_revision_dependencies,
)
from ..domain.sample_metadata import MetadataAlignmentReport, SampleMetadata
from ..visual_config import theme

# Initialize ExperimentAnalysis for within-group analysis
# No longer need GlyphClass import since we use string element types
from .cache_manager import CacheManager
from .color_manager import ColorManager
from .data_provider import DataProvider
from .element_presentation import (
    resolve_display_direction,
    resolve_element_color,
)
from .group_manager import GroupManager
from .hover_manager import HoverManager
from .workspace_mode import AnalysisWorkspaceMode

_UNCHANGED = object()

# Revision dimensions an activity result depends on. A standalone activation
# captures provenance over exactly these, matching the graph worker's snapshot so
# an artifact built either way is judged stale by the same inputs.
_ACTIVITY_DEPENDENCIES = frozenset(
    {
        RevisionKind.DATASET,
        RevisionKind.METADATA,
        RevisionKind.GROUPS,
        RevisionKind.PREPROCESSING,
    }
)


class AnalysisWorkspace:
    """Handles data retrieval and statistical information for reactions and genes."""

    def __init__(
        self,
        metabolic_model=None,
        reaction_data=None,
        expression_data=None,
        experiment_analysis=None,
        sample_metadata=None,
        normalization_method: str = "mad",
        max_z_score: int = 6,
    ):
        """Initialize the Analysis Workspace inputs and installed analysis.

        Args:
            metabolic_model: MetabolicModel object containing metabolic network
                information
            reaction_data: ReactionData object containing flux data
            expression_data: ExpressionData object containing gene expression data
            experiment_analysis: Optional ExperimentAnalysis object for
                within-group analysis
        """
        self.metabolic_model = metabolic_model
        self.reaction_data = reaction_data
        self.expression_data = expression_data
        self._workspace_id = uuid4().hex
        self.dataset_revision = 0
        self.metadata_revision = 0
        self.group_revision = 0
        self.preprocessing_revision = 0
        self.threshold_revision = 0
        self.visualization_revision = 0
        self.sample_metadata = sample_metadata
        self.metadata_alignment_report = None

        # Record the inputs without hashing them: constructing a workspace is not
        # loading data into one. The server builds a detached workspace per graph
        # job as a worker facade over frames the session already holds, so hashing
        # here would tax every render. The dataset writers below do the hashing,
        # on the load that actually creates a new identity. See ADR-0007.
        self._source_identity = SourceIdentityCache()
        self._source_identity.record_dataset(
            metabolic_model=metabolic_model,
            reaction_data=reaction_data,
            expression_data=expression_data,
        )
        self._source_identity.record_metadata(sample_metadata)

        self.group_comparison_artifact: GroupComparisonArtifact | None = None
        self.reaction_set_enrichment_artifact: ReactionSetEnrichmentArtifact | None = None
        self._set_enrichment_artifacts: dict[
            SetEnrichmentFeatureType, SetEnrichmentArtifact
        ] = {}
        self.reaction_direction_criterion = DEFAULT_REACTION_DIRECTION_CRITERION
        self.experiment_analysis = experiment_analysis
        # Authoritative single-group activity result; None until activity mode is
        # applied. Every activity reader -- colouring, legend, hover, subsystem
        # ranking -- reads this artifact; no mutable context is installed beside it.
        self.activity_artifact: ActivityArtifact | None = None
        self.activity_overlay = ActivityOverlay.REACTION

        # Define default delta options
        default_delta_options = {
            "metric": "cohens_d",
            "transform": "sign",
            "numerator_filter": "significant",
            "denominator_filter": "all",
            "reaction_direction_criterion": self.reaction_direction_criterion.value,
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
                # Sequential ramp: white (low) -> amber (typical) -> purple (high).
                # The bin machinery interpolates neutral->up and neutral->down, so the
                # "neutral" centre is the typical activity, not an absence of signal.
                "up_color": theme.ACTIVITY_HIGH,
                "down_color": theme.ACTIVITY_LOW,
                "neutral_color": theme.ACTIVITY_MID,
            },
            default_mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
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
            group_comparison_artifact=self.group_comparison_artifact,
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

    @property
    def workspace_id(self) -> str:
        """Stable identity of this Analysis Workspace."""
        return self._workspace_id

    @property
    def source_fingerprints(self) -> Mapping[str, str]:
        """Portable identity of the loaded inputs, for artifacts to record.

        Reading this is free: each fingerprint was computed when its input was
        loaded. Artifact producers must read it here rather than fingerprint an
        input themselves, which would put a full-matrix hash on every run.
        """
        return self._source_identity.fingerprints

    def mark_dataset_changed(self) -> int:
        """Advance the cheap revision token used by derived-result caches."""
        self.dataset_revision += 1
        self.reaction_set_enrichment_artifact = None
        self._set_enrichment_artifacts.clear()
        return self.dataset_revision

    def mark_metadata_changed(self) -> int:
        """Advance the Sample Metadata revision token."""
        self.metadata_revision += 1
        self.reaction_set_enrichment_artifact = None
        self._set_enrichment_artifacts.clear()
        # Every metadata transition routes through here, after the new table is
        # installed, so this is the seam that re-identifies it.
        self._source_identity.identify_metadata(self.sample_metadata)
        return self.metadata_revision

    def mark_groups_changed(self) -> int:
        """Advance the group/sample-definition revision token."""
        self.group_revision += 1
        self.reaction_set_enrichment_artifact = None
        self._set_enrichment_artifacts.clear()
        return self.group_revision

    def mark_preprocessing_changed(self) -> int:
        """Advance the preprocessing revision token."""
        self.preprocessing_revision += 1
        self.reaction_set_enrichment_artifact = None
        self._set_enrichment_artifacts.clear()
        return self.preprocessing_revision

    def mark_thresholds_changed(
        self,
        feature_type: SetEnrichmentFeatureType | str | None = None,
    ) -> int:
        """Advance the Comparison Thresholds revision token."""
        self.threshold_revision += 1
        self.reaction_set_enrichment_artifact = None
        if feature_type is None:
            self._set_enrichment_artifacts.clear()
        else:
            self._set_enrichment_artifacts.pop(
                SetEnrichmentFeatureType(feature_type), None
            )
        return self.threshold_revision

    def mark_visualization_changed(self) -> int:
        """Advance the Visualization Context revision token."""
        self.visualization_revision += 1
        return self.visualization_revision

    def get_workspace_revisions(self) -> WorkspaceRevisions:
        """Return the complete immutable revision vector for this workspace."""
        return WorkspaceRevisions(
            dataset=int(self.dataset_revision),
            metadata=int(self.metadata_revision),
            groups=int(self.group_revision),
            preprocessing=int(self.preprocessing_revision),
            thresholds=int(self.threshold_revision),
            visualization=int(self.visualization_revision),
        )

    @staticmethod
    def _capture_frame(source, getter_name: str) -> StableFrameReference | None:
        if source is None:
            return None
        return StableFrameReference.capture(getattr(source, getter_name)())

    @staticmethod
    def _capture_expression_matrix(source) -> StableMatrixReference | None:
        """Capture expression through the storage-neutral ``matrix()`` seam (AM9).

        A sparse ``AnnData.X`` is referenced, not densified: the reference reads
        the sparse adapter and tracks only source identity and shape. Replacement
        is caught by the freshness check; a same-shape in-place edit of the
        resident buffer is prevented by the install-time freeze (M-11). Returns
        ``None`` when no expression matrix is loaded, matching the readers that
        already tolerate an unloaded expression input.
        """
        if source is None:
            return None
        matrix = source.matrix() if hasattr(source, "matrix") else None
        if matrix is None:
            return None
        return StableMatrixReference.capture(source)

    def _capture_group_memberships(
        self,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple(
            (str(group_id), tuple(map(str, self.get_group_samples(group_id))))
            for group_id in self.get_group_ids()
        )

    def capture_analysis_snapshot(
        self,
        dependencies: Iterable[RevisionKind | str] | None = None,
    ) -> AnalysisSnapshot:
        """Capture approved worker inputs without copying full data matrices.

        Args:
            dependencies: Revision dimensions that can affect the computation.
                Omitting it conservatively depends on every workspace revision.
        """
        provenance = WorkspaceProvenance(
            workspace_id=self.workspace_id,
            revisions=self.get_workspace_revisions(),
            dependencies=normalize_revision_dependencies(dependencies),
            reaction_frame=self._capture_frame(
                self.reaction_data, "get_reaction_df"
            ),
            expression_matrix=self._capture_expression_matrix(self.expression_data),
            group_memberships=self._capture_group_memberships(),
        )
        return AnalysisSnapshot(provenance=provenance)

    def require_current(self, provenance: WorkspaceProvenance) -> None:
        """Reject provenance that no longer matches this Analysis Workspace.

        Validation is read-only and is intended to run before the first result
        installation write.
        """
        if provenance.workspace_id != self.workspace_id:
            raise StaleAnalysisResultError(
                "The analysis result belongs to a different workspace"
            )

        current_revisions = self.get_workspace_revisions()
        for kind in provenance.dependencies:
            if provenance.revisions.value_for(kind) != current_revisions.value_for(kind):
                raise StaleAnalysisResultError(
                    f"The workspace {kind.value} revision changed during analysis"
                )

        data_dependencies = {
            RevisionKind.DATASET,
            RevisionKind.PREPROCESSING,
        }
        if provenance.dependencies & data_dependencies:
            current_reaction = self._capture_frame(
                self.reaction_data, "get_reaction_df"
            )
            current_expression = self._capture_expression_matrix(self.expression_data)
            if not self._frame_reference_matches(
                provenance.reaction_frame, current_reaction
            ) or not self._matrix_reference_matches(
                provenance.expression_matrix, current_expression
            ):
                raise StaleAnalysisResultError(
                    "An analysis input frame changed during analysis"
                )

        if (
            RevisionKind.GROUPS in provenance.dependencies
            and provenance.group_memberships != self._capture_group_memberships()
        ):
            raise StaleAnalysisResultError(
                "Group memberships changed during analysis"
            )

    @staticmethod
    def _frame_reference_matches(
        expected: StableFrameReference | None,
        current: StableFrameReference | None,
    ) -> bool:
        if expected is None or current is None:
            return expected is current
        return (
            expected.object_id == current.object_id
            and expected.shape == current.shape
            and expected.matches(current.read())
        )

    @staticmethod
    def _matrix_reference_matches(
        expected: StableMatrixReference | None,
        current: StableMatrixReference | None,
    ) -> bool:
        """Whether the expression source is unchanged since snapshot capture (AM9).

        Identity and shape catch *replacement* (a DATASET transaction installs a
        fresh ``ExpressionData``, giving a new source object). A same-shape
        in-place edit of the resident buffer cannot reach here: the install-time
        freeze (M-11) makes it raise at its source, so identity and shape are the
        complete storage-neutral freshness signal.
        """
        if expected is None or current is None:
            return expected is current
        return (
            expected.object_id == current.object_id
            and expected.shape == current.shape
        )

    def _replace_dataset_inputs(
        self,
        *,
        metabolic_model=_UNCHANGED,
        reaction_data=_UNCHANGED,
        expression_data=_UNCHANGED,
        preprocessing_changed: bool,
        clear_metadata: bool,
        output_reaction_format: str | None = None,
        output_gene_format: str | None = None,
        reset_output_formats: bool = False,
    ) -> None:
        """Commit one prepared input transition through the workspace seam."""
        next_model = (
            self.metabolic_model
            if metabolic_model is _UNCHANGED
            else metabolic_model
        )
        next_reaction = (
            self.reaction_data if reaction_data is _UNCHANGED else reaction_data
        )
        next_expression = (
            self.expression_data
            if expression_data is _UNCHANGED
            else expression_data
        )

        # Prepared inputs are authoritative only from this point onward. Provider
        # references are derived collaborator views, updated in the same operation.
        self.metabolic_model = next_model
        self.reaction_data = next_reaction
        self.expression_data = next_expression

        # Seal the replace-only contract (M-11, AM9): freeze the installed
        # expression matrix's value buffers so a same-shape in-place edit raises
        # at its source while a worker snapshot references it. Prevention was
        # chosen over id+shape detection; the freeze is idempotent, so a model
        # transition that leaves expression unchanged re-freezes harmlessly.
        if next_expression is not None:
            freeze = getattr(next_expression, "freeze", None)
            if callable(freeze):
                freeze()
        self.data_provider.metabolic_model = next_model
        self.data_provider.reaction_data = next_reaction
        self.data_provider.expression_data = next_expression

        # A load is where new input identity is created, so it is where identity
        # is paid for. Inputs whose object is unchanged keep the fingerprint they
        # already have; no later analysis, render, or export re-hashes them.
        self._source_identity.identify_dataset(
            metabolic_model=next_model,
            reaction_data=next_reaction,
            expression_data=next_expression,
        )

        if reset_output_formats:
            self.output_reaction_format = "mar_id"
            self.output_gene_format = "geneSymbols"
            self.output_metabolite_format = "metName"
        else:
            if output_reaction_format is not None:
                self.output_reaction_format = str(output_reaction_format)
            if output_gene_format is not None:
                self.output_gene_format = str(output_gene_format)

        self.mark_dataset_changed()
        if preprocessing_changed:
            self.mark_preprocessing_changed()

        if clear_metadata and (
            self.sample_metadata is not None
            or self.metadata_alignment_report is not None
        ):
            self.sample_metadata = None
            self.metadata_alignment_report = None
            self.mark_metadata_changed()

        self._invalidate_downstream_analysis(clear_groups=True)

    def apply_model_transition(self, prepared_model) -> None:
        """Install a transitioned Metabolic Model as one atomic DATASET change.

        This is the single seam for every live model change -- subsystem or
        compartment filter, filter reset/reload, and default or custom category
        replacement. The caller prepares a *fresh* model that already reflects
        the change (the alignment pattern: build the model locally, transform it,
        hand it here) and the workspace installs it, advances the DATASET
        revision, re-pays source identity at this writer, and invalidates every
        DATASET-dependent artifact and user group in one transaction (M-10).

        The model must be a fresh object, never the installed one mutated in
        place. Source identity is keyed on object identity (ADR-0007), so
        re-identifying a mutated original would hand back its stale fingerprint,
        and an in-flight worker still holding that same object could observe a
        torn read (MR-01). Rejecting an invalid model *before* any state changes
        also keeps the previous model, revisions, and installed artifacts
        coherent when a transition is refused.
        """
        if prepared_model is None:
            raise ValueError("A metabolic model is required for a model transition")
        if prepared_model is self.metabolic_model:
            raise ValueError(
                "A model transition must install a fresh model object, not the "
                "installed model mutated in place; source identity is keyed on "
                "object identity (ADR-0007)"
            )
        # A fresh model owns no pre-transition derived state. Clearing its private
        # operation cache keeps a copied-then-transitioned model from carrying an
        # entry computed against the pre-transition network (the filter/reload
        # methods do not clear it themselves; see am8-inventory.md).
        clear_memory_cache = getattr(prepared_model, "clear_memory_cache", None)
        if callable(clear_memory_cache):
            clear_memory_cache()
        self._replace_dataset_inputs(
            metabolic_model=prepared_model,
            preprocessing_changed=False,
            clear_metadata=False,
        )

    def replace_metabolic_model(self, metabolic_model) -> None:
        """Atomically replace the model and invalidate every derived result.

        A wholesale model replacement is a model transition with no measured
        input change, so it shares the one transaction seam.
        """
        self.apply_model_transition(metabolic_model)

    def replace_reaction_data(
        self,
        reaction_data,
        *,
        metabolic_model=_UNCHANGED,
        output_reaction_format: str | None = None,
        preprocessing_changed: bool = True,
    ) -> None:
        """Atomically install prepared Reaction Data and, optionally, its model."""
        if reaction_data is None:
            raise ValueError("Reaction Data is required; use reset_dataset() to clear it")
        if metabolic_model is None:
            raise ValueError("A metabolic model is required")
        self._replace_dataset_inputs(
            metabolic_model=metabolic_model,
            reaction_data=reaction_data,
            preprocessing_changed=preprocessing_changed,
            clear_metadata=True,
            output_reaction_format=output_reaction_format,
        )

    def replace_expression_data(
        self,
        expression_data,
        *,
        output_gene_format: str | None = None,
        preprocessing_changed: bool = True,
    ) -> None:
        """Atomically install prepared Expression Data."""
        if expression_data is None:
            raise ValueError("Expression Data is required; use reset_dataset() to clear it")
        self._replace_dataset_inputs(
            expression_data=expression_data,
            preprocessing_changed=preprocessing_changed,
            clear_metadata=True,
            output_gene_format=output_gene_format,
        )

    def reset_dataset(self) -> None:
        """Clear all measured inputs and derived state while retaining the model."""
        self._replace_dataset_inputs(
            reaction_data=None,
            expression_data=None,
            preprocessing_changed=True,
            clear_metadata=True,
            reset_output_formats=True,
        )

    def _invalidate_downstream_analysis(self, *, clear_groups: bool = False) -> None:
        """Invalidate every result derived from metadata or group membership."""
        if clear_groups:
            self.group_manager.clear_groups()
            self.group_revision += 1
        self.experiment_analysis = None
        self.group_comparison_artifact = None
        self.reaction_set_enrichment_artifact = None
        self._set_enrichment_artifacts.clear()
        self.activity_artifact = None
        self.data_provider.experiment_analysis = None
        self.data_provider.group_comparison_artifact = None
        self.data_provider.activity_artifact = None
        self.subsystem_analysis.set_group_comparison(None)
        self.subsystem_analysis._base_stats_cache = {}
        self.subsystem_analysis._delta_stats_cache = {}
        self.cache_manager.clear_cache()
        self.color_manager.clear_group_colors()
        self.color_manager.clear_distributions()
        self.mode = AnalysisWorkspaceMode.NO_EXPERIMENT
        self.hover_manager.update_group_comparison(None)
        self.hover_manager.refresh_builders(self.mode)

    def update_sample_metadata(
        self,
        metadata: SampleMetadata,
        *,
        alignment_policy: str = "strict",
    ) -> MetadataAlignmentReport:
        if self.reaction_data is None:
            raise ValueError("Reaction data must be loaded before sample metadata")
        expression_ids = (
            self.expression_data.get_sample_ids() if self.expression_data is not None else None
        )
        report = metadata.align(
            self.reaction_data.get_sample_ids(),
            expression_ids,
            policy=alignment_policy,
        )
        self.sample_metadata = metadata
        self.metadata_alignment_report = report
        self.mark_metadata_changed()
        self._invalidate_downstream_analysis(clear_groups=True)
        return report

    def clear_sample_metadata(self) -> None:
        self.sample_metadata = None
        self.metadata_alignment_report = None
        self.mark_metadata_changed()
        self._invalidate_downstream_analysis(clear_groups=True)

    def get_sample_display_label(self, sample_id: str) -> str:
        if self.sample_metadata is None:
            return str(sample_id)
        return self.sample_metadata.get_display_label(str(sample_id))

    def get_grouping_metadata(self) -> SampleMetadata | None:
        """Return standalone metadata or an equivalent view of h5ad observations."""
        if self.sample_metadata is not None:
            return self.sample_metadata
        adata = getattr(self.expression_data, "adata", None)
        if adata is not None and getattr(self.expression_data, "format", "").lower() == "h5ad":
            return SampleMetadata(adata.obs.copy())
        return None

    def load_groups_from_metadata(self, columns, *, include_missing=False):
        metadata = self.get_grouping_metadata()
        if metadata is None:
            raise ValueError("Load sample metadata before grouping by metadata columns")
        result = self.group_manager.load_groups_from_metadata(
            metadata,
            columns,
            sample_ids=self.reaction_data.get_sample_ids(),
            include_missing=include_missing,
        )
        self.group_revision += 1
        self._invalidate_downstream_analysis(clear_groups=False)
        if self.group_manager.has_groups():
            self.color_manager.generate_group_colors(self.get_group_ids(), palette_name="turbo")
        return result

    def define_group_from_predicates(self, name, predicates, *, replace: bool = False):
        metadata = self.get_grouping_metadata()
        if metadata is None:
            raise ValueError("Load sample metadata before defining predicate groups")
        group = self.group_manager.define_group_from_predicates(
            name,
            metadata,
            predicates,
            all_sample_ids=self.reaction_data.get_sample_ids(),
            replace=replace,
        )
        self.group_revision += 1
        self._invalidate_downstream_analysis(clear_groups=False)
        self.color_manager.generate_group_colors(self.get_group_ids(), palette_name="turbo")
        return group

    def define_group_union(self, name, source_group_ids, *, replace: bool = False):
        group = self.group_manager.define_group_union(
            name,
            source_group_ids,
            all_sample_ids=self.reaction_data.get_sample_ids(),
            replace=replace,
        )
        self.group_revision += 1
        self._invalidate_downstream_analysis(clear_groups=False)
        self.color_manager.generate_group_colors(self.get_group_ids(), palette_name="turbo")
        return group

    # ============================================================================
    # Mode Management Methods
    # ============================================================================

    def _determine_mode(self) -> AnalysisWorkspaceMode:
        """Determine the appropriate operation mode based on available data.

        Returns:
            AnalysisWorkspaceMode: The determined mode based on available components
        """
        if self.group_comparison_artifact is not None:
            return AnalysisWorkspaceMode.GROUP_COMPARISON
        elif self.experiment_analysis is not None:
            return AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS
        else:
            return AnalysisWorkspaceMode.NO_EXPERIMENT

    def get_mode(self) -> AnalysisWorkspaceMode:
        """Get the current operation mode.

        Returns:
            AnalysisWorkspaceMode: Current operation mode
        """
        return self.mode

    def set_mode(self, mode: AnalysisWorkspaceMode) -> None:
        """Set the operation mode manually.

        This allows switching between modes when multiple data sources are available.
        For example, you might have both GroupComparison and ExperimentAnalysis
        available and want to switch between them.

        Args:
            mode: The desired operation mode

        Raises:
            ValueError: If the requested mode is not available with current data
        """
        if mode == AnalysisWorkspaceMode.GROUP_COMPARISON and self.group_comparison_artifact is None:
            raise ValueError(
                "Cannot set GROUP_COMPARISON mode: no GroupComparison object available"
            )
        elif (
            mode == AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS
            and self.experiment_analysis is None
        ):
            raise ValueError(
                "Cannot set EXPERIMENT_ANALYSIS mode: "
                "no ExperimentAnalysis object available"
            )

        self.mode = mode

        # Synchronize ColorManager mode with AnalysisWorkspaceMode
        if mode == AnalysisWorkspaceMode.GROUP_COMPARISON:
            self.color_manager.set_mode(AnalysisWorkspaceMode.GROUP_COMPARISON)
        elif mode == AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS:
            self.color_manager.set_mode(AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS)
        # NO_EXPERIMENT can use either, default to group_comparison
        else:
            self.color_manager.set_mode(AnalysisWorkspaceMode.GROUP_COMPARISON)

        # Clear any cached data that might be mode-dependent
        self.cache_manager.clear_cache()
        # Refresh hover builders for the new mode
        self.hover_manager.refresh_builders(mode)

    def get_available_modes(self) -> list[AnalysisWorkspaceMode]:
        """Get a list of modes that are available with the current data.

        Returns:
            list[AnalysisWorkspaceMode]: List of available modes
        """
        available_modes = [AnalysisWorkspaceMode.NO_EXPERIMENT]  # Always available

        if self.group_comparison_artifact is not None:
            available_modes.append(AnalysisWorkspaceMode.GROUP_COMPARISON)

        if self.experiment_analysis is not None:
            available_modes.append(AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS)

        return available_modes

    def is_mode_available(self, mode: AnalysisWorkspaceMode) -> bool:
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
        if self.mode == AnalysisWorkspaceMode.GROUP_COMPARISON:
            return (
                "GROUP_COMPARISON mode: Provides between-group statistical "
                "comparisons, colors based on log2 fold changes, p-values, "
                "and Cohen's d"
            )
        elif self.mode == AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS:
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
    # Visualization Context (WP5)
    # ============================================================================

    def activate_activity_mode(
        self,
        config: ActivityConfig,
        *,
        overlay: ActivityOverlay = ActivityOverlay.REACTION,
        provenance: Optional[WorkspaceProvenance] = None,
        source_fingerprints: Optional[Mapping[str, str]] = None,
    ) -> ActivityContext:
        """Score one group, build its artifact, and make it the colouring result.

        Every precondition is checked here and reported as a user-facing error;
        callers must not fall back to another mode silently, because a mode that
        quietly changes under the user is how comparison statistics ended up
        painted onto a single-group view.

        The artifact is the authoritative result every reader consumes. A detached
        graph worker injects the input identity the document thread captured, so a
        render reads it rather than re-hashing the matrices (ADR-0007, MR-10). A
        standalone caller lets the workspace read its own -- paid for once at load.
        """
        if self.experiment_analysis is None:
            raise ValueError(
                "Load groups before using single-group activity coloring"
            )
        # A comparison-local Target/Baseline population names no base group; it
        # brings its own frozen membership, which the scoring validates against the
        # aligned matrix.
        if not config.has_explicit_population and not self.group_manager.validate_group_id(
            config.group_id
        ):
            available = ", ".join(map(str, self.get_group_ids())) or "none"
            raise ValueError(
                f"Group '{config.group_id}' does not exist. Available groups: {available}"
            )
        if overlay is ActivityOverlay.GENE and self.expression_data is None:
            raise ValueError(
                "The gene-expression overlay needs an expression matrix. Load one, "
                "or select the reaction activity-score overlay."
            )

        # compute_activity registers the colour distributions on this workspace's
        # colour manager, so the artifact install below does not register again.
        context = self.experiment_analysis.compute_activity(config)

        if provenance is None:
            provenance = self.capture_analysis_snapshot(_ACTIVITY_DEPENDENCIES).provenance
            source_fingerprints = self.source_fingerprints
        artifact = self.experiment_analysis.build_activity_artifact(
            context,
            provenance=provenance,
            source_fingerprints=dict(source_fingerprints or {}),
        )
        self.update_activity_artifact(
            artifact, overlay=overlay, register_colors=False
        )
        return context

    def _validate_activity_artifact_install(
        self,
        provenance: WorkspaceProvenance,
        artifact: ActivityArtifact,
    ) -> Any:
        """Validate a worker-produced activity artifact without mutating state."""
        self.require_current(provenance)
        analysis = self.experiment_analysis
        if analysis is None:
            raise ValueError("experiment_analysis not available in data_manager")
        if not isinstance(artifact, ActivityArtifact):
            raise TypeError("artifact must be an ActivityArtifact")
        config = artifact.config
        if config.has_explicit_population:
            # A comparison-local Target/Baseline population is not a persisted
            # group, so there is no group record to check. Its exact membership is
            # the artifact's own selected samples, which the artifact already
            # validated against the aligned reference universe.
            if not artifact.selected_sample_ids:
                raise ValueError(
                    f"Population '{config.group_id}' has no samples in the aligned "
                    "matrix"
                )
        else:
            if not self.group_manager.validate_group_id(config.group_id):
                raise ValueError(
                    f"Group '{config.group_id}' does not exist in experiment groups"
                )
            if config.group_id not in analysis.group_ids:
                raise ValueError(
                    f"Group '{config.group_id}' is not available in experiment_analysis"
                )
        for family in artifact.available_families:
            features = artifact.feature_result_for(family)
            if features.config != config:
                raise ValueError("Activity feature config does not match its artifact")
            if features.group_id != config.group_id:
                raise ValueError("Activity feature group does not match its artifact")
        if artifact.has_family(ActivityOverlay.GENE) and self.expression_data is None:
            raise ValueError("Gene activity evidence requires an expression matrix")

        return analysis

    def install_activity_artifact(
        self,
        provenance: WorkspaceProvenance,
        artifact: ActivityArtifact,
        *,
        overlay: ActivityOverlay = ActivityOverlay.REACTION,
    ) -> ActivityArtifact:
        """Install the authoritative activity artifact through the workspace seam.

        The artifact is the single source of truth for every activity reader --
        colouring, legend, hover, and the subsystem ranking; no mutable context is
        installed beside it. Provenance freshness and scientific preconditions are
        checked before the first write; the typed value and its provider mirror are
        committed before presentation, so a colour-adapter failure cannot leave
        scientific state split across owners.
        """
        analysis = self._validate_activity_artifact_install(provenance, artifact)
        self._require_artifact_source(provenance, artifact.metadata.source)
        if not artifact.has_family(overlay):
            raise ValueError(f"{overlay.value.title()} activity evidence is unavailable")

        analysis.install_activity_artifact(artifact)
        self._set_activity_artifact(artifact, overlay=overlay)
        self.refresh_activity_presentation(register_colors=True)
        return artifact

    def _set_activity_artifact(
        self,
        artifact: Optional[ActivityArtifact],
        *,
        overlay: ActivityOverlay = ActivityOverlay.REACTION,
    ) -> None:
        """Commit an activity artifact (or None) to state and the provider mirror."""
        self.activity_artifact = artifact
        self.data_provider.activity_artifact = artifact
        self.activity_overlay = overlay
        self.data_provider.activity_overlay = overlay

    def refresh_activity_presentation(self, *, register_colors: bool = True) -> None:
        """Refresh activity colouring, active group, and mode after an install.

        ``register_colors`` is False for a detached graph worker, whose colour
        manager already carries the distributions its own ``activate_activity_mode``
        registered; re-registering them there would add per-render cost (MR-10).
        """
        artifact = self.activity_artifact
        if artifact is None:
            raise ValueError("activity artifact not available in data_manager")
        analysis = self.experiment_analysis
        if analysis is None:
            raise ValueError("experiment_analysis not available in data_manager")

        # Set the active group and mode before publishing colours, so a failing
        # colour adapter leaves the scientific state (the installed artifact, the
        # active group, the mode) coherent -- presentation is the later phase. A
        # comparison-local population has no base-group record to activate; the
        # artifact remains the authoritative source for every activity reader.
        if not artifact.config.has_explicit_population:
            self.set_analysis_group(artifact.group_id)
        self.set_mode(AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS)
        if register_colors:
            # Publish the colour distributions from the installed artifact -- the
            # deleted context used to do this on the compute path.
            analysis.register_activity_artifact_distributions(artifact, force=True)

    def update_activity_artifact(
        self,
        artifact: ActivityArtifact,
        *,
        overlay: ActivityOverlay = ActivityOverlay.REACTION,
        register_colors: bool = True,
    ) -> ActivityArtifact:
        """Synchronous activity-artifact installation for standalone and worker use.

        A detached graph worker passes ``register_colors=False`` because its colour
        manager already registered the distributions when it computed the view.
        """
        self._set_activity_artifact(artifact, overlay=overlay)
        self.refresh_activity_presentation(register_colors=register_colors)
        return artifact

    def get_activity_artifact(self) -> ActivityArtifact | None:
        """Return the authoritative portable result for the active activity view."""
        return self.activity_artifact

    def install_subsystem_delta_artifact(
        self,
        provenance: WorkspaceProvenance,
        artifact: SubsystemDeltaArtifact,
    ) -> SubsystemDeltaArtifact:
        """Install comparison subsystem state and then refresh its presentation."""
        self.require_current(provenance)
        if not isinstance(artifact, SubsystemDeltaArtifact):
            raise TypeError("artifact must be a SubsystemDeltaArtifact")
        comparison = self.group_comparison_artifact
        if comparison is None:
            raise ValueError("group_comparison not available in data_manager")
        if self.data_provider.group_comparison_artifact is not comparison:
            raise ValueError("group_comparison provider mirror is not current")
        if self.subsystem_analysis.gc is not comparison:
            raise ValueError("subsystem analysis comparison is not current")

        self.subsystem_analysis.install_delta_artifact(
            artifact,
            register_presentation=False,
        )
        # set_mode synchronizes color mode, invalidates hover caches, and refreshes
        # hover builders without advancing the visualization input revision.
        self.set_mode(AnalysisWorkspaceMode.GROUP_COMPARISON)

        # Publish detached color inputs only after scientific caches are installed.
        self.subsystem_analysis.install_delta_artifact(
            artifact,
            register_presentation=True,
        )
        return artifact

    def apply_visualization_context(
        self,
        mode: AnalysisWorkspaceMode,
        *,
        activity_config: Optional[ActivityConfig] = None,
        activity_overlay: ActivityOverlay = ActivityOverlay.REACTION,
        provenance: Optional[WorkspaceProvenance] = None,
        source_fingerprints: Optional[Mapping[str, str]] = None,
    ) -> Optional[ActivityContext]:
        """Switch the graph coloring mode, raising instead of silently degrading."""
        if mode == AnalysisWorkspaceMode.GROUP_COMPARISON:
            if self.group_comparison_artifact is None:
                raise ValueError(
                    "Run a group comparison before using comparison coloring"
                )
            self._set_activity_artifact(None)
            self.set_mode(AnalysisWorkspaceMode.GROUP_COMPARISON)
            return None

        if mode == AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS:
            if activity_config is None:
                raise ValueError(
                    "Single-group activity mode requires an activity configuration"
                )
            return self.activate_activity_mode(
                activity_config,
                overlay=activity_overlay,
                provenance=provenance,
                source_fingerprints=source_fingerprints,
            )

        raise ValueError(f"Unsupported visualization mode: {mode}")

    def get_color_legend_text(self) -> str:
        """The sentence describing what the current graph colors mean."""
        if self.mode == AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS and self.activity_artifact:
            return self.activity_artifact.legend_text(self.activity_overlay)
        if self.mode == AnalysisWorkspaceMode.GROUP_COMPARISON and self.group_comparison_artifact:
            membership = self.group_comparison_artifact.membership
            return (
                f"Coloring Target '{membership.target_label}' minus Baseline "
                f"'{membership.baseline_label}'. Red/positive means higher in Target; "
                "blue/negative means lower in Target (higher in Baseline); gray means "
                "not significant under the selected thresholds."
            )
        return "No experimental coloring is applied."

    def get_display_direction(self, reaction_id: str) -> Optional[str]:
        """Direction shown for a reversible reaction under the current context.

        Delegated so a graph worker, which resolves this from a
        ``GraphCalculationInput`` and no workspace, cannot answer differently.
        """
        return resolve_display_direction(self, reaction_id)

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
        self.mark_groups_changed()
        self._invalidate_downstream_analysis(clear_groups=False)

        # If grouping was successful, generate colors via ColorManager
        if self.group_manager.has_groups():
            group_ids = self.group_manager.get_group_ids()
            self.color_manager.generate_group_colors(
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

    def get_group_samples(self, group_id) -> List[str]:
        """Get samples associated with a group.

        Args:
            group_id (str): ID of the group

        Returns:
            List of sample IDs in the group

        Raises:
            ValueError: If the requested group doesn't exist
        """
        # Delegate to GroupManager
        return self.group_manager.get_group_samples(group_id)

    def get_group_ids(self) -> List[str]:
        """Return the list of group IDs.

        Returns:
            List of group IDs from the experiment
        """
        return self.group_manager.get_group_ids()

    def get_group_records(self):
        """Return the canonical immutable sample-group records in definition order."""
        return self.group_manager.get_group_records()

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
    # Scientific Result Installation and Presentation Refresh
    # ============================================================================

    def _set_experiment_analysis(self, experiment_analysis) -> None:
        """Commit an analysis to authoritative state and provider mirrors."""
        self.experiment_analysis = experiment_analysis
        self.data_provider.experiment_analysis = experiment_analysis

        # Group summaries changed, so an activity result calculated from the
        # previous summaries is no longer scientifically valid.
        self.activity_artifact = None
        self.data_provider.activity_artifact = None

    def _set_group_comparison_artifact(self, artifact) -> None:
        """Commit a comparison artifact to authoritative state and the provider mirror.

        Installing a comparison invalidates any enrichment computed against the
        previous one, so it is cleared here rather than left to drift.
        """
        self.group_comparison_artifact = artifact
        self.reaction_set_enrichment_artifact = None
        self._set_enrichment_artifacts.clear()
        self.data_provider.group_comparison_artifact = artifact

    def install_experiment_analysis(
        self,
        provenance: WorkspaceProvenance,
        experiment_analysis,
    ):
        """Atomically install current group-summary analysis state.

        Provenance is checked before the first write. Presentation adapters are
        deliberately refreshed separately so their failure cannot roll back or
        leave the manager/provider scientific mirrors disagreeing.
        """
        self.require_current(provenance)
        self._set_experiment_analysis(experiment_analysis)
        return experiment_analysis

    def install_group_comparison_artifact(
        self,
        provenance: WorkspaceProvenance,
        artifact: GroupComparisonArtifact,
    ) -> GroupComparisonArtifact:
        """Install the authoritative comparison artifact through the workspace seam.

        The artifact is the single source of truth for every comparison reader
        (hover, plots, colouring, the significance report); no mutable view is
        installed beside it.
        """
        self.require_current(provenance)
        if not isinstance(artifact, GroupComparisonArtifact):
            raise TypeError("artifact must be a GroupComparisonArtifact")
        self._require_artifact_source(provenance, artifact.metadata.source)
        self._set_group_comparison_artifact(artifact)
        return artifact

    def get_group_comparison_artifact(self) -> GroupComparisonArtifact | None:
        """Return the installed authoritative comparison artifact, if any."""
        return self.group_comparison_artifact

    def install_reaction_set_enrichment_artifact(
        self,
        provenance: WorkspaceProvenance,
        artifact: ReactionSetEnrichmentArtifact,
    ) -> ReactionSetEnrichmentArtifact:
        """Install one current enrichment result for the authoritative comparison."""
        self.require_current(provenance)
        if not isinstance(artifact, ReactionSetEnrichmentArtifact):
            raise TypeError("artifact must be a ReactionSetEnrichmentArtifact")
        self._require_artifact_source(provenance, artifact.metadata.source)
        comparison = self.group_comparison_artifact
        if comparison is None:
            raise ValueError("A comparison artifact must be installed first")
        identity = artifact.metadata.scientific_identity
        if identity.get("comparison_artifact_id") != comparison.metadata.artifact_id:
            raise ValueError("Enrichment belongs to a different comparison artifact")
        if artifact.universe.membership != comparison.membership:
            raise ValueError("Enrichment membership differs from current comparison")
        if RevisionKind.THRESHOLDS not in artifact.metadata.source.dependencies:
            raise ValueError("Enrichment provenance must depend on comparison thresholds")
        threshold_identity = {
            "comparison_artifact_id": comparison.metadata.artifact_id,
            "reaction_thresholds": dict(comparison.reaction_thresholds.as_mapping()),
            "significant_reaction_ids": list(comparison.significant_reactions),
            "keep_reversed": comparison.keep_reversed,
        }
        if identity.get("comparison_threshold_fingerprint") != fingerprint_json(
            threshold_identity
        ):
            raise ValueError("Enrichment threshold identity is not current")
        if (
            artifact.config.reaction_direction_criterion
            is not self.reaction_direction_criterion
        ):
            raise ValueError("Enrichment reaction direction criterion is not current")

        self.reaction_set_enrichment_artifact = artifact
        return artifact

    def get_reaction_set_enrichment_artifact(
        self,
    ) -> ReactionSetEnrichmentArtifact | None:
        """Return the installed enrichment result without running analysis."""
        return self.reaction_set_enrichment_artifact

    def install_set_enrichment_artifact(
        self,
        provenance: WorkspaceProvenance,
        artifact: SetEnrichmentArtifact,
    ) -> SetEnrichmentArtifact:
        """Install one current reaction- or gene-set artifact transactionally."""
        self.require_current(provenance)
        if not isinstance(artifact, SetEnrichmentArtifact):
            raise TypeError("artifact must be a SetEnrichmentArtifact")
        self._require_artifact_source(provenance, artifact.metadata.source)
        comparison = self.group_comparison_artifact
        if comparison is None:
            raise ValueError("A comparison artifact must be installed first")
        artifact.require_current_comparison(comparison)
        if (
            artifact.feature_type is SetEnrichmentFeatureType.REACTION
            and artifact.config.reaction_direction_criterion
            is not self.reaction_direction_criterion
        ):
            raise ValueError(
                "Set-enrichment reaction direction criterion is not current"
            )
        self._set_enrichment_artifacts[artifact.feature_type] = artifact
        return artifact

    def get_set_enrichment_artifact(
        self,
        feature_type: SetEnrichmentFeatureType | str,
    ) -> SetEnrichmentArtifact | None:
        """Return the installed paired ORA artifact for one feature family."""
        return self._set_enrichment_artifacts.get(
            SetEnrichmentFeatureType(feature_type)
        )

    def set_reaction_direction_criterion(
        self,
        criterion: ReactionDirectionCriterion | str,
    ) -> ReactionDirectionCriterion:
        """Set the shared reaction direction and invalidate reaction summaries only."""
        selected = ReactionDirectionCriterion(criterion)
        if selected is self.reaction_direction_criterion:
            return selected
        self.reaction_direction_criterion = selected
        self.reaction_set_enrichment_artifact = None
        self._set_enrichment_artifacts.pop(SetEnrichmentFeatureType.REACTION, None)
        self.subsystem_analysis.invalidate_reaction_direction(selected)
        self.cache_manager.clear_cache()
        return selected

    def install_reduction_artifact(
        self,
        provenance: WorkspaceProvenance,
        artifact: ReductionArtifact,
    ) -> ReductionArtifact:
        """Install an authoritative reduction artifact through the workspace seam."""
        self.require_current(provenance)
        analysis = self.experiment_analysis
        if analysis is None:
            raise ValueError("experiment_analysis not available in data_manager")
        if not isinstance(artifact, ReductionArtifact):
            raise TypeError("artifact must be a ReductionArtifact")
        self._require_artifact_source(provenance, artifact.metadata.source)
        return analysis.install_reduction_artifact(artifact)

    @staticmethod
    def _require_artifact_source(
        provenance: WorkspaceProvenance,
        source: ArtifactProvenance,
    ) -> None:
        """Require portable metadata to still project from this live token.

        Freshness is scoped to the artifact's declared dependencies, so an
        artifact is stale only when an input it actually depends on moved. It is
        the same rule as ``require_current``: a whole-vector comparison would
        reject an installed reduction because a threshold changed elsewhere in
        the session.
        """
        if not source.matches_live(provenance):
            raise ValueError(
                "Artifact source provenance does not match the live workspace token"
            )

    def require_current_artifact(self, artifact) -> None:
        """Reject a portable artifact whose declared workspace inputs are stale.

        Export controls use this document-thread seam before publishing bytes. It
        projects a fresh live token through the artifact's own dependency set;
        source fingerprints stay immutable evidence rather than being recomputed
        by presentation code.
        """
        metadata = getattr(artifact, "metadata", None)
        source = getattr(metadata, "source", None)
        if not isinstance(source, ArtifactProvenance):
            raise TypeError("artifact must carry portable provenance metadata")
        provenance = self.capture_analysis_snapshot(source.dependencies).provenance
        self._require_artifact_source(provenance, source)

    def get_reduction_artifact(
        self,
        method: str,
        data_type: str,
    ) -> ReductionArtifact | None:
        """Return the installed authoritative reduction artifact, if any."""
        analysis = self.experiment_analysis
        if analysis is None:
            return None
        return analysis.get_reduction_artifact(method, data_type)

    def refresh_experiment_analysis_presentation(self) -> None:
        """Refresh color, mode, hover, and cache adapters after analysis commit."""
        analysis = self.experiment_analysis
        if analysis is None:
            raise ValueError("experiment_analysis not available in data_manager")

        analysis.register_presentation_distributions()
        self.set_mode(AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS)
        if not self.color_manager.get_group_colors():
            self.regenerate_group_colors()

    def refresh_group_comparison_presentation(
        self, *, register_colors: bool = True
    ) -> None:
        """Refresh comparison colouring, subsystem, hover, mode, and cache adapters.

        ``register_colors`` is False for a detached graph worker, whose colour
        manager already carries the distributions copied from the live session;
        re-registering them there would add per-render cost (MR-10).
        """
        artifact = self.group_comparison_artifact
        if artifact is None:
            raise ValueError("group_comparison not available in data_manager")

        if register_colors:
            # Register the colour distributions from the installed artifact -- the
            # deleted mutable view used to do this from the server. Genes only when
            # a gene test ran (i.e. expression was loaded).
            register_comparison_distributions(
                self, artifact.get_test_results(for_genes=False), for_genes=False
            )
            if artifact.gene_results is not None:
                register_comparison_distributions(
                    self, artifact.get_test_results(for_genes=True), for_genes=True
                )

        self.subsystem_analysis.set_group_comparison(artifact)
        self.hover_manager.update_group_comparison(artifact)
        self.set_mode(AnalysisWorkspaceMode.GROUP_COMPARISON)

    def update_subsystem_delta_config(self, delta_config: Dict[str, Any]) -> bool:
        """Update the delta configuration for subsystem analysis.

        This updates the scoring configuration (metric, transform, filters)
        and recalculates all subsystem delta scores without recomputing
        base statistics.

        Args:
            delta_config: New configuration dict with keys:
                - 'metric': The metric to use (e.g., 'cohens_d', 'log2_fc')
                - 'transform': How to transform values ('sign', 'raw', 'abs')
                - 'numerator_filter': Filter for numerator ('all', 'significant')
                - 'denominator_filter': Filter for denominator ('all', 'significant')
                - 'reaction_direction_criterion': Dominant Target/Baseline source
                  selection for reversible reactions

        Returns:
            bool: True if delta stats were successfully recomputed,
                  False if base stats cache is empty.
        """
        success = self.subsystem_analysis.update_delta_config(delta_config)

        # Clear cached hover text that may depend on delta scores
        if success:
            self.cache_manager.clear_cache()

        return success

    @staticmethod
    def _validated_comparison_threshold(
        name: str,
        value: Optional[float],
        *,
        maximum: Optional[float] = None,
    ) -> Optional[float]:
        """Normalize one threshold without touching installed comparison state."""
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"{name} threshold must be numeric")
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} threshold must be numeric") from exc
        if not isfinite(normalized) or normalized < 0:
            raise ValueError(f"{name} threshold must be a finite non-negative value")
        if maximum is not None and normalized > maximum:
            raise ValueError(f"{name} threshold must be at most {maximum:g}")
        return normalized

    def apply_comparison_thresholds(
        self,
        *,
        pval_threshold: Optional[float] = None,
        cohens_d_threshold: Optional[float] = None,
        log2_fc_threshold: Optional[float] = None,
        mean_difference_threshold: Optional[float] = None,
        keep_reversed: bool = False,
        for_genes: bool = False,
    ) -> Tuple[set, Dict[str, Any]]:
        """Apply significance thresholds as one workspace input transition.

        Validation completes before the comparison is mutated. A successful
        significance update advances ``threshold_revision`` exactly once, then
        recomputes threshold-dependent subsystem state and refreshes hover/cache
        presentation without replacing base result tables or group summaries.
        """
        artifact = self.group_comparison_artifact
        if artifact is None:
            raise ValueError("group_comparison not available in data_manager")
        if self.data_provider.group_comparison_artifact is not artifact:
            raise ValueError("group_comparison provider mirror is not current")
        if not isinstance(for_genes, bool):
            raise ValueError("for_genes must be a boolean")
        if not isinstance(keep_reversed, bool):
            raise ValueError("keep_reversed must be a boolean")
        if artifact.get_test_results(for_genes=for_genes) is None:
            element_type = "gene" if for_genes else "reaction"
            raise ValueError(f"No {element_type} test results are installed")

        validated = {
            "pval_threshold": self._validated_comparison_threshold(
                "p-value",
                pval_threshold,
                maximum=1.0,
            ),
            "cohens_d_threshold": self._validated_comparison_threshold(
                "Cohen's d",
                cohens_d_threshold,
            ),
            "log2_fc_threshold": self._validated_comparison_threshold(
                "log2 fold-change",
                log2_fc_threshold,
            ),
            "mean_difference_threshold": self._validated_comparison_threshold(
                "mean difference",
                mean_difference_threshold,
            ),
        }

        # Recompute significance for the thresholded family directly against the
        # artifact's own result frame -- no live comparison view is installed to
        # mutate. The other family's installed state is carried over unchanged.
        target_frame = artifact.get_test_results(for_genes=for_genes)
        analyzer = StatisticalAnalysis(
            test_type=artifact.test, correction_type=artifact.correction
        )
        significant_mask, threshold_info = analyzer.determine_significant_elements(
            target_frame,
            **validated,
            keep_reversed=keep_reversed,
        )
        significant = set(target_frame[significant_mask].index)
        target_frame["is_significant"] = target_frame.index.map(
            lambda element_id: element_id in significant
        )
        new_thresholds = ComparisonThresholds(
            pval=validated["pval_threshold"],
            cohens_d=validated["cohens_d_threshold"],
            log2_fc=validated["log2_fc_threshold"],
            mean_difference=validated["mean_difference_threshold"],
        )

        revisions = replace(
            self.get_workspace_revisions(),
            thresholds=self.threshold_revision + 1,
        )
        if for_genes:
            replacement_artifact = artifact.with_threshold_results(
                revisions=revisions,
                reaction_results=artifact.get_test_results(for_genes=False),
                reaction_thresholds=artifact.reaction_thresholds,
                significant_reactions=artifact.significant_reactions,
                gene_results=target_frame,
                gene_thresholds=new_thresholds,
                significant_genes=tuple(significant),
                keep_reversed=keep_reversed,
            )
        else:
            replacement_artifact = artifact.with_threshold_results(
                revisions=revisions,
                reaction_results=target_frame,
                reaction_thresholds=new_thresholds,
                significant_reactions=tuple(significant),
                gene_results=artifact.get_test_results(for_genes=True),
                gene_thresholds=artifact.gene_thresholds,
                significant_genes=artifact.significant_genes,
                keep_reversed=keep_reversed,
            )

        self.mark_thresholds_changed(
            SetEnrichmentFeatureType.GENE
            if for_genes
            else SetEnrichmentFeatureType.REACTION
        )
        self.group_comparison_artifact = replacement_artifact
        self.data_provider.group_comparison_artifact = replacement_artifact

        self.subsystem_analysis.set_group_comparison(replacement_artifact)
        self.hover_manager.update_group_comparison(replacement_artifact)
        self.set_mode(AnalysisWorkspaceMode.GROUP_COMPARISON)
        return significant, threshold_info

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

        Delegated to the shared resolver so a graph worker, which paints from a
        ``GraphCalculationInput`` and no workspace, cannot paint differently.

        Args:
            element_id: ID of the element (reaction, gene, etc.)
            element_type: Type of element ("reaction", "gene", "subsystem")
            metric_name: Name of the metric to use for coloring

        Returns:
            str: Hex color code
        """
        return resolve_element_color(
            self,
            element_id,
            element_type,
            metric_name=metric_name,
            use_main_direction=use_main_direction,
            delta_options=delta_options,
        )

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

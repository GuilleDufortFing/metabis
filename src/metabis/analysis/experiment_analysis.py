"""Experiment analysis module for within-group characterization.

This module provides comprehensive analysis capabilities for data_managers with
multiple groups, focusing on within-group analysis of reaction and subsystem
activities. Unlike GroupComparison which compares the same reaction/subsystem
between different groups, this module compares different reactions/subsystems
within the same group to identify relative activation patterns.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..analysis_workspace import WorkspaceProvenance
from ..domain.expression_matrix import DenseExpressionMatrix
from ..managers.workspace_mode import AnalysisWorkspaceMode
from .activity_analysis import (
    ActivityArtifact,
    ActivityConfig,
    ActivityContext,
    ActivityOverlay,
    calculate_feature_activity_from_matrix,
    calculate_subsystem_activity,
    sample_reference_distribution,
)
from .dimensionality_reduction import (
    DimensionalityReduction,
    ReductionArtifact,
    ReductionKey,
    ReductionResult,
)
from .reaction_direction import to_base_reaction_id

__all__ = [
    "ExperimentAnalysis",
    "ExperimentSummarySnapshot",
]


@dataclass(frozen=True)
class ExperimentSummarySnapshot:
    """Scientific summary state that can be moved between analysis instances."""

    reaction_averages: pd.DataFrame | None
    reaction_directions: pd.DataFrame | None
    subsystem_averages: pd.DataFrame | None
    gene_averages: pd.DataFrame | None
    completed: bool

    def __post_init__(self) -> None:
        if self.completed and (
            self.reaction_averages is None
            or self.reaction_directions is None
            or self.subsystem_averages is None
        ):
            raise ValueError(
                "A completed experiment summary snapshot requires reaction "
                "averages, reaction directions, and subsystem averages"
            )


class ExperimentAnalysis:
    """A class to analyze activity patterns within individual groups.

    This class provides methods to:
    1. Calculate average reaction activities for each group
    2. Calculate within-group z-scores showing relative activation of reactions
    3. Perform the same analysis at the subsystem level
    4. Compare reactions/subsystems within the same group

    Unlike GroupComparison which compares the same reaction between different groups,
    this class analyzes how different reactions compare within the same group,
    answering questions like "Which reactions are most active in group A?" or
    "How does glycolysis compare to oxidative phosphorylation within group B?"
    """

    def __init__(
        self,
        data_manager,
        auto_run: bool = True,
        *,
        register_distributions: bool = True,
    ):
        """Initialize the experiment analysis.

        Args:
            data_manager: An Experiment object with loaded data and defined groups
            auto_run: Whether to immediately run the default analysis pipeline

        Raises:
            ValueError: If data_manager is invalid or missing required data
        """
        self.data_manager = data_manager
        self.register_distributions = register_distributions

        # Results storage
        self._reaction_avgs: pd.DataFrame | None = None
        self._reaction_directions: pd.DataFrame | None = None
        self._gene_avgs: pd.DataFrame | None = None
        self._gene_group_avgs: pd.DataFrame | None = None
        self._subsystem_avgs: pd.DataFrame | None = None

        # Single-group activity contexts, keyed by ActivityConfig.metric_key
        self._activity_cache: Dict[str, ActivityContext] = {}
        self._activity_artifacts: dict[str, ActivityArtifact] = {}

        # Dimensionality reduction module
        self.dim_reduction = DimensionalityReduction()
        self._reduction_artifacts: dict[ReductionKey, ReductionArtifact] = {}

        # Analysis status tracking
        self._analysis_completed = False
        self._pca_completed = {"reactions": False, "expression": False}
        # Store basic info for validation
        self.group_ids = list(data_manager.get_group_ids())
        self.n_groups = len(self.group_ids)

        # Automatically run analysis like GroupComparison unless the caller
        # needs to supply explicit analysis parameters first.
        if auto_run:
            self.run_analysis()

    @classmethod
    def from_precomputed_results(
        cls,
        data_manager,
        *,
        reaction_averages: pd.DataFrame,
        reaction_directions: pd.DataFrame,
        subsystem_averages: pd.DataFrame,
        gene_averages: pd.DataFrame | None = None,
        reductions: Sequence[ReductionArtifact] = (),
    ) -> ExperimentAnalysis:
        """Build a completed analysis from immutable worker results.

        This is the public document-thread rehydration seam.  It installs all
        scientific summaries and reductions into a new analysis object without
        registering presentation distributions or mutating ``data_manager``.

        Reductions arrive as their artifacts, so a PCA computed here is
        replottable by the companion workflows that read the installed artifact.
        """
        analysis = cls.from_summary_snapshot(
            data_manager,
            ExperimentSummarySnapshot(
                reaction_averages=reaction_averages,
                reaction_directions=reaction_directions,
                subsystem_averages=subsystem_averages,
                gene_averages=gene_averages,
                completed=True,
            ),
        )
        for reduction in reductions:
            analysis.install_reduction_artifact(reduction)
        return analysis

    def snapshot_summaries(self) -> ExperimentSummarySnapshot:
        """Capture complete or in-progress scientific summary state."""
        return ExperimentSummarySnapshot(
            reaction_averages=self._reaction_avgs,
            reaction_directions=self._reaction_directions,
            subsystem_averages=self._subsystem_avgs,
            gene_averages=self._gene_avgs,
            completed=self._analysis_completed,
        )

    @classmethod
    def from_summary_snapshot(
        cls,
        data_manager,
        snapshot: ExperimentSummarySnapshot,
    ) -> ExperimentAnalysis:
        """Restore summary state without computing or writing presentation data."""
        analysis = cls(
            data_manager,
            auto_run=False,
            register_distributions=False,
        )
        analysis._reaction_avgs = snapshot.reaction_averages
        analysis._reaction_directions = snapshot.reaction_directions
        analysis._subsystem_avgs = snapshot.subsystem_averages
        analysis._gene_avgs = snapshot.gene_averages
        analysis._gene_group_avgs = snapshot.gene_averages
        analysis._analysis_completed = snapshot.completed
        return analysis

    def install_reduction_artifact(
        self, artifact: ReductionArtifact
    ) -> ReductionArtifact:
        """Install one authoritative reduction artifact into session state.

        The typed result cache keyed by ``ReductionKey`` keeps the coordinates by
        reference so ``get_reduction_result(s)`` and a worker-local ``_reuse``
        still find them; the artifact is the authoritative record the companion
        plots and export read.
        """
        if not isinstance(artifact, ReductionArtifact):
            raise TypeError("reduction artifact must be a ReductionArtifact")
        if artifact.key.method not in {"PCA", "T-SNE", "UMAP"}:
            raise ValueError(f"Unsupported reduction method: {artifact.key.method}")
        result = artifact.to_compute_result()
        self.dim_reduction.cache_result(result)
        self._reduction_artifacts.pop(artifact.key, None)
        self._reduction_artifacts[artifact.key] = artifact
        if artifact.key.method == "PCA":
            self._pca_completed[artifact.key.data_type] = True
        return artifact

    def get_reduction_artifacts(self) -> tuple[ReductionArtifact, ...]:
        """Return installed authoritative reduction artifacts."""
        return tuple(self._reduction_artifacts.values())

    def get_reduction_artifact(
        self,
        method: str,
        data_type: str,
    ) -> ReductionArtifact | None:
        """Return the most recently installed artifact for a workflow."""
        for artifact in reversed(self.get_reduction_artifacts()):
            if artifact.key.method == method and artifact.key.data_type == data_type:
                return artifact
        return None

    def get_reduction_results(self) -> tuple[ReductionResult, ...]:
        """Return the typed reduction results the calculator holds by key."""
        return tuple(self.dim_reduction.results.values())

    def get_reduction_result(
        self,
        method: str,
        data_type: str,
    ) -> ReductionResult | None:
        """Return the most recently installed typed result for a workflow."""
        for result in reversed(self.get_reduction_results()):
            if result.key.method == method and result.key.data_type == data_type:
                return result
        return None

    def get_summary_frames(
        self,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame | None,
    ]:
        """Return reaction, direction, subsystem, and optional gene summaries."""
        if (
            self._reaction_avgs is None
            or self._reaction_directions is None
            or self._subsystem_avgs is None
        ):
            raise ValueError("Experiment summary analysis is not complete")
        return (
            self._reaction_avgs,
            self._reaction_directions,
            self._subsystem_avgs,
            self._gene_avgs,
        )

    def register_presentation_distributions(self) -> None:
        """Register stored summaries with the presentation color adapter.

        Worker computation disables registration.  A document-thread caller uses
        this explicit method after provenance validation and scientific
        rehydration, keeping presentation work out of the worker boundary.
        """
        self.register_distributions = True
        summaries = (
            ("reaction", self._reaction_avgs),
            ("subsystem", self._subsystem_avgs),
            ("gene", self._gene_avgs),
        )
        for element_type, values in summaries:
            if values is not None:
                self._register_distributions(element_type, values)

    def _calculate_reaction_avgs(self) -> pd.DataFrame:
        """Calculate average reaction activities for each group.

        Returns:
            pd.DataFrame: DataFrame with groups as rows and reactions as columns,
                         containing average reaction activities
        """
        # Get reaction data
        reaction_df = self.data_manager.reaction_data.get_reaction_df()

        # Initialize results DataFrame
        group_avgs = pd.DataFrame(
            index=self.group_ids, columns=reaction_df.columns, dtype=float
        )

        # Calculate average for each group
        for group_id in self.group_ids:
            try:
                # Get samples for this group
                group_samples = self.data_manager.get_group_samples(group_id)

                # Calculate mean for this group
                group_data = reaction_df.loc[group_samples]
                group_avg = group_data.mean(axis=0)
                group_avgs.loc[group_id] = group_avg

            except Exception as e:
                print(f"Error processing group '{group_id}': {str(e)}")
                group_avgs.loc[group_id] = np.nan

        # Alias and register distributions for coloring
        self._reaction_avgs = group_avgs
        self._register_distributions("reaction", self._reaction_avgs)

        return group_avgs

    def _calculate_reaction_directions(self) -> pd.DataFrame:
        """Calculate main direction (forward/backward) for each reaction in each group.

        For each reaction in each group, determines whether the forward or backward
        direction is dominant based on average activities.

        Logic:
        - Only one direction measured: that direction wins.
        - Both measured: "forward" if activity(reaction) >= abs(activity(reaction_back)),
          "backward" otherwise.

        Columns are **base** reaction IDs, the union of the forward and ``_back``
        columns present. Keying them on the forward column instead dropped any
        reaction measured only in reverse - Compass blocks a direction often
        enough that 200 base reactions over 25 subsystems vanished from the graph,
        the hover, and subsystem enrichment on the bundled data alone (D-64).
        :func:`~metabis.analysis.reaction_direction.build_reaction_direction_map`
        resolves the same shape, so the two must not disagree.

        Returns:
            pd.DataFrame: Groups as rows, base reactions as columns, values are
                         "forward" or "backward"
        """
        reaction_avgs = self._reaction_avgs
        if reaction_avgs is None:
            raise RuntimeError("reaction averages must be calculated first")
        available = {str(reaction) for reaction in reaction_avgs.columns}
        base_reactions = sorted({to_base_reaction_id(item) for item in available})

        def _means(columns: List[str], present: np.ndarray) -> np.ndarray:
            values = np.zeros((len(self.group_ids), len(base_reactions)))
            if columns:
                measured = reaction_avgs.loc[self.group_ids, columns].to_numpy(
                    dtype=float
                )
                values[:, present] = np.where(np.isnan(measured), 0.0, measured)
            return values

        # Direction is decided per group x base reaction from the group means. The
        # comparison is vectorized over the whole matrix instead of reading and
        # writing one cell at a time, giving identical labels.
        has_forward = np.array([base in available for base in base_reactions])
        has_backward = np.array(
            [f"{base}_back" in available for base in base_reactions]
        )
        forward = _means(
            [base for base in base_reactions if base in available], has_forward
        )
        backward = _means(
            [f"{base}_back" for base in base_reactions if f"{base}_back" in available],
            has_backward,
        )

        # A reaction measured in one direction only takes that direction; where both
        # are measured the larger mean wins, and an exact tie goes to forward -
        # the same rule and the same tie-break as build_reaction_direction_map.
        both = (has_forward & has_backward)[np.newaxis, :]
        is_forward = np.where(
            both,
            forward >= np.abs(backward),
            has_forward[np.newaxis, :],
        )
        reaction_directions = pd.DataFrame(
            np.where(is_forward, "forward", "backward"),
            index=self.group_ids,
            columns=base_reactions,
        )

        # Store results
        self._reaction_directions = reaction_directions

        return reaction_directions

    def _calculate_subsystem_avgs(self) -> pd.DataFrame:
        """Calculate average subsystem activities for each group.

        For each group, calculates the average activity of all reactions
        within each subsystem.

        Returns:
            pd.DataFrame: DataFrame with groups as rows and subsystems as columns,
                         containing average subsystem activities
        """
        reaction_avgs = self._reaction_avgs
        if reaction_avgs is None:
            raise RuntimeError("reaction averages must be calculated first")

        # Get all subsystems
        model = self.data_manager.metabolic_model
        subsystems = model.get_subsystems()
        available_columns = set(reaction_avgs.columns)

        # Initialize results DataFrame
        subsystem_avgs = pd.DataFrame(
            index=self.group_ids, columns=subsystems, dtype=float
        )

        # get_reactions_for_subsystem does not depend on the group, so resolve each
        # subsystem's available reactions once and take every group's mean as one
        # vectorized column instead of reading and writing one cell per group.
        for subsystem in subsystems:
            reactions_in_subsystem = model.get_reactions_for_subsystem(subsystem)
            available_reactions = [
                reaction
                for reaction in reactions_in_subsystem
                if reaction in available_columns
            ]
            if available_reactions:
                # DataFrame.mean(axis=1) skips NaN, matching the per-group
                # Series.mean() the loop used.
                subsystem_avgs[subsystem] = reaction_avgs.loc[
                    self.group_ids, available_reactions
                ].mean(axis=1)
            else:
                subsystem_avgs[subsystem] = np.nan

        # Cache results
        self._subsystem_avgs = subsystem_avgs

        # Alias and register distributions for coloring
        self._register_distributions("subsystem", self._subsystem_avgs)

        return subsystem_avgs

    def _calculate_gene_avgs(self) -> pd.DataFrame:
        """Calculate average gene expression for each group.

        Returns:
            pd.DataFrame: DataFrame with groups as rows and genes as columns,
                         containing average gene expression
        """
        # Read expression through the matrix seam: identity is served without
        # densifying, and each group's median is taken over a bounded dense block
        # of that group's samples only -- never the whole matrix (AM6 slice 4).
        matrix = self.data_manager.expression_data.matrix()
        sample_ids = set(matrix.sample_ids)

        # Initialize results DataFrame
        gene_group_avgs = pd.DataFrame(
            index=self.group_ids, columns=matrix.feature_ids, dtype=float
        )

        # Calculate average for each group
        for group_id in self.group_ids:
            try:
                # Get samples for this group
                gene_group_samples = self.data_manager.get_group_samples(group_id)

                # Filter samples that exist in the data
                available_samples = [s for s in gene_group_samples if s in sample_ids]

                if not available_samples:
                    print(
                        f"Warning: No samples found for group '{group_id}' "
                        f"in expression data"
                    )
                    # Fill with NaN for this group
                    gene_group_avgs.loc[group_id] = np.nan
                    continue

                # median/MAD has no sparse reduction, so materialize just this
                # group's samples dense (bounded) and reduce over the rows.
                gene_group_data = matrix.select_samples(available_samples)
                gene_group_mean = gene_group_data.median(axis=0)
                gene_group_avgs.loc[group_id] = gene_group_mean

            except Exception as e:
                print(f"Error processing group '{group_id}': {str(e)}")
                gene_group_avgs.loc[group_id] = np.nan

        gene_group_avgs = np.log(gene_group_avgs + 1)

        # Store results
        self._gene_group_avgs = gene_group_avgs

        # Alias and register distributions for coloring
        self._gene_avgs = gene_group_avgs
        self._register_distributions("gene", self._gene_avgs)

        return gene_group_avgs

    def _get_data_for_reduction(self, data_type: str):
        """Get the expression-matrix seam and source name for a data type.

        Returns an ``ExpressionMatrix`` rather than a dense frame (AM6 slice 5):
        a sparse expression load ranks features and materializes only the bounded
        top-N block through the seam, never densifying the whole matrix. The
        reaction path wraps its already-dense frame in the dense adapter --
        reaction data is computed dense and untouched by AM6 -- so both reductions
        share one seam-driven implementation.
        """
        if data_type.lower() == "reactions":
            if self.data_manager.reaction_data is None:
                return None, "Reaction data not loaded."
            return (
                DenseExpressionMatrix(
                    self.data_manager.reaction_data.get_reaction_df()
                ),
                "reaction data",
            )
        elif data_type.lower() == "expression":
            if self.data_manager.expression_data is None:
                return None, "Expression data not loaded."
            matrix = self.data_manager.expression_data.matrix()
            if matrix is None:
                return None, "Expression data not loaded."
            return matrix, "expression data"
        return None, f"Invalid data_type: {data_type}"

    def calculate_pca(
        self,
        data_type: str = "reactions",
        n_components: Optional[int] = None,
        n_variable_features: int = 500,
    ) -> Dict[str, Any]:
        """Calculates PCA for the specified data type and stores the results."""
        matrix, source_name = self._get_data_for_reduction(data_type)
        if matrix is None:
            return {"status": "error", "messages": [source_name]}

        return self.dim_reduction.calculate_pca(
            matrix=matrix,
            data_type=data_type,
            data_source_name=source_name,
            n_components=n_components,
            n_variable_features=n_variable_features,
            preprocessing_revision=str(
                getattr(self.data_manager, "dataset_revision", 0)
            ),
            sample_revision=str(getattr(self.data_manager, "group_revision", 0)),
        )

    def calculate_tsne(
        self,
        data_type: str = "reactions",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_variable_features: int = 500,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Calculates t-SNE for the specified data type and stores the results."""
        matrix, source_name = self._get_data_for_reduction(data_type)
        if matrix is None:
            return {"status": "error", "messages": [source_name]}

        return self.dim_reduction.calculate_tsne(
            matrix=matrix,
            data_type=data_type,
            data_source_name=source_name,
            n_components=n_components,
            perplexity=perplexity,
            n_variable_features=n_variable_features,
            random_state=random_state,
            preprocessing_revision=str(
                getattr(self.data_manager, "dataset_revision", 0)
            ),
            sample_revision=str(getattr(self.data_manager, "group_revision", 0)),
        )

    def calculate_umap(
        self,
        data_type: str = "reactions",
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_variable_features: int = 500,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Calculates UMAP for the specified data type and stores the results."""
        matrix, source_name = self._get_data_for_reduction(data_type)
        if matrix is None:
            return {"status": "error", "messages": [source_name]}

        return self.dim_reduction.calculate_umap(
            matrix=matrix,
            data_type=data_type,
            data_source_name=source_name,
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_variable_features=n_variable_features,
            random_state=random_state,
            preprocessing_revision=str(
                getattr(self.data_manager, "dataset_revision", 0)
            ),
            sample_revision=str(getattr(self.data_manager, "group_revision", 0)),
        )

    def get_reaction_avgs(self) -> pd.DataFrame:
        """Get group reaction averages (average activities).

        Returns:
            pd.DataFrame: Group averages with groups as rows, reactions as columns
        """
        return self._reaction_avgs

    def get_reaction_directions(self) -> pd.DataFrame:
        """Get reaction directions for all groups.

        Returns the pre-computed main direction (forward/backward) for each reaction
        in each group. Only includes reactions that don't end with "_back".

        Returns:
            pd.DataFrame: Directions with groups as rows, reactions as columns,
                         values are "forward" or "backward"
        """
        return self._reaction_directions

    def get_main_direction(self, group_id: str, reaction_id: str) -> Optional[str]:
        """Get the main direction for a specific reaction in a group.

        Args:
            group_id: Group identifier
            reaction_id: Reaction identifier (must not end with "_back")

        Returns:
            Optional[str]: "forward" or "backward", or None if not found

        Raises:
            ValueError: If reaction_id ends with "_back" or analysis not completed
        """
        # Check if reaction_id ends with "_back"
        if reaction_id.endswith("_back"):
            reaction_id = reaction_id[:-5]  # Remove "_back" suffix

        # Ensure analysis has been run
        if not self._analysis_completed:
            raise ValueError("Analysis not yet completed. Call run_analysis() first.")

        # Check if results are available
        if self._reaction_directions is None:
            raise ValueError(
                "Reaction directions not calculated. Call run_analysis() first."
            )

        # Check if group exists
        if group_id not in self._reaction_directions.index:
            return None

        # Check if reaction exists
        if reaction_id not in self._reaction_directions.columns:
            return None

        # Return the pre-computed direction
        direction = self._reaction_directions.loc[group_id, reaction_id]

        # Handle potential NaN or None values
        if pd.isna(direction):
            return None

        return direction

    def get_subsystem_avgs(self) -> pd.DataFrame:
        """Get subsystem averages (average activities).

        Returns:
            pd.DataFrame: Subsystem averages with groups as rows, subsystems as columns
        """
        return self._subsystem_avgs

    def get_gene_avgs(self) -> pd.DataFrame:
        """Get gene expression averages for all groups.

        Returns:
            pd.DataFrame: Gene expression averages with groups as rows, genes as columns
        """
        return self._gene_group_avgs

    # ============================================================================
    # Single-Group Activity Methods (WP5)
    # ============================================================================

    def _activity_inputs(self, overlay: ActivityOverlay):
        """Return the expression-matrix seam and subsystem membership for an overlay.

        The gene overlay returns the seam over the loaded expression data, so a
        sparse-``X`` load is scored through bounded selections without densifying
        the whole matrix (AM6 slice 6). The reaction overlay wraps its already-dense
        frame in the dense adapter -- reaction data is computed dense, untouched by
        AM6 -- so both overlays feed one seam-driven scorer.
        """
        model = self.data_manager.metabolic_model
        subsystems = model.get_subsystems() if model is not None else []

        if overlay is ActivityOverlay.GENE:
            expression_data = getattr(self.data_manager, "expression_data", None)
            if expression_data is None:
                raise ValueError(
                    "The gene-expression overlay needs expression data. Load an "
                    "expression matrix or switch the overlay to reaction activity scores."
                )
            matrix = expression_data.matrix()
            if matrix is None:
                raise ValueError(
                    "The gene-expression overlay needs expression data. Load an "
                    "expression matrix or switch the overlay to reaction activity scores."
                )
            members = {
                subsystem: model.get_genes_for_subsystem(subsystem)
                for subsystem in subsystems
            }
            return matrix, members

        reaction_data = getattr(self.data_manager, "reaction_data", None)
        if reaction_data is None:
            raise ValueError("The reaction overlay needs a loaded reaction matrix")
        matrix = DenseExpressionMatrix(reaction_data.get_reaction_df())
        members = {
            subsystem: model.get_reactions_for_subsystem(subsystem)
            for subsystem in subsystems
        }
        return matrix, members

    def _activity_population(self, config: ActivityConfig) -> list[str]:
        """The exact samples one activity config summarizes.

        A comparison-local Target/Baseline population is not a persisted group, so
        its frozen membership travels on the config and is used verbatim; a base
        group is resolved from the Analysis Workspace as before. Scoring and the
        artifact's recorded selected samples both read this one rule.
        """
        if config.has_explicit_population:
            return list(config.sample_ids)
        return list(self.data_manager.get_group_samples(config.group_id))

    def compute_activity(self, config: ActivityConfig) -> ActivityContext:
        """Score reaction and optional gene families under one config.

        Results are cached per config *and* per data/group revision, so changing
        the group or summary always recomputes. Changing the overview overlay is
        presentation-only and reuses this same dual-family context.
        """
        if not config.has_explicit_population and config.group_id not in self.group_ids:
            raise ValueError(
                f"Group '{config.group_id}' does not exist. Available groups: "
                f"{', '.join(map(str, self.group_ids)) or 'none'}"
            )

        dataset_revision = int(getattr(self.data_manager, "dataset_revision", 0))
        group_revision = int(getattr(self.data_manager, "group_revision", 0))

        cached = self._activity_cache.get(config.metric_key)
        if (
            cached is not None
            and cached.dataset_revision == dataset_revision
            and cached.group_revision == group_revision
        ):
            return cached

        group_samples = self._activity_population(config)

        reaction_matrix, reaction_members = self._activity_inputs(
            ActivityOverlay.REACTION
        )
        reaction_features = calculate_feature_activity_from_matrix(
            reaction_matrix,
            group_samples,
            config,
            overlay=ActivityOverlay.REACTION,
        )
        reaction_subsystems = calculate_subsystem_activity(
            reaction_features, reaction_members
        )

        gene_features = None
        gene_subsystems = None
        gene_reference_summary: tuple[float, ...] = ()
        expression_data = getattr(self.data_manager, "expression_data", None)
        if expression_data is not None and expression_data.matrix() is not None:
            gene_matrix, gene_members = self._activity_inputs(ActivityOverlay.GENE)
            gene_features = calculate_feature_activity_from_matrix(
                gene_matrix,
                group_samples,
                config,
                overlay=ActivityOverlay.GENE,
            )
            gene_subsystems = calculate_subsystem_activity(
                gene_features,
                gene_members,
                use_main_direction=False,
            )
            gene_reference_summary = sample_reference_distribution(
                gene_features.activity_score
            )

        context = ActivityContext(
            config=config,
            reaction_features=reaction_features,
            reaction_subsystems=reaction_subsystems,
            gene_features=gene_features,
            gene_subsystems=gene_subsystems,
            dataset_revision=dataset_revision,
            group_revision=group_revision,
            reaction_reference_summary=sample_reference_distribution(
                reaction_features.activity_score
            ),
            gene_reference_summary=gene_reference_summary,
        )
        return self.install_activity_context(
            context,
            register_presentation=self.register_distributions,
        )

    def install_activity_context(
        self,
        context: ActivityContext,
        *,
        register_presentation: bool = True,
    ) -> ActivityContext:
        """Install one activity result and optionally publish its color scales."""
        self._activity_cache[context.metric_key] = context
        if register_presentation:
            self._register_activity_distributions(context, force=True)
        return context

    def build_activity_artifact(
        self,
        context: ActivityContext,
        *,
        provenance: WorkspaceProvenance,
        source_fingerprints: Mapping[str, str],
    ) -> ActivityArtifact:
        """Project a computed activity context into its portable artifact.

        The selected and reference samples are resolved from the same inputs the
        scoring read, by the same rule: a population sample only counts once it is
        in the aligned matrix, and every sample in that matrix is part of the
        aligned input universe. Asking a caller to supply the two lists invites an
        export that disagrees with the scores about which samples it summarized.

        Input identity is read from the Analysis Workspace, which paid for it at
        load; this never fingerprints the matrix (ADR-0007, MR-10).
        """
        matrix, _subsystem_members = self._activity_inputs(ActivityOverlay.REACTION)
        reference_sample_ids = matrix.sample_ids
        known = set(reference_sample_ids)
        group_samples = self._activity_population(context.config)
        return ActivityArtifact.build(
            context,
            provenance=provenance,
            source_fingerprints=source_fingerprints,
            selected_sample_ids=[
                str(sample) for sample in group_samples if str(sample) in known
            ],
            reference_sample_ids=[str(sample) for sample in reference_sample_ids],
        )

    def install_activity_artifact(self, artifact: ActivityArtifact) -> ActivityArtifact:
        """Retain one portable activity result as the authoritative export value."""
        if not isinstance(artifact, ActivityArtifact):
            raise TypeError("artifact must be an ActivityArtifact")
        self._activity_artifacts[artifact.config.metric_key] = artifact
        return artifact

    def get_activity_artifact(
        self, config: ActivityConfig | None = None
    ) -> ActivityArtifact | None:
        """Return the installed portable result for ``config`` or active context."""
        if config is not None:
            return self._activity_artifacts.get(config.metric_key)
        if not self._activity_cache:
            return None
        metric_key = next(reversed(self._activity_cache))
        return self._activity_artifacts.get(metric_key)

    def _register_activity_distributions(
        self,
        context: ActivityContext,
        *,
        force: bool = False,
    ) -> None:
        """Register activity color bins from a freshly computed context."""
        for overlay in (ActivityOverlay.REACTION, ActivityOverlay.GENE):
            if not context.has_family(overlay):
                continue
            self._publish_activity_distributions(
                binning=context.binning(),
                overlay=overlay.value,
                metric_key=context.config.metric_key_for(overlay),
                feature_scores=context.features_for(overlay).activity_score,
                subsystems=context.subsystems_for(overlay),
                force=force,
            )

    def register_activity_artifact_distributions(
        self,
        artifact: "ActivityArtifact",
        *,
        force: bool = True,
    ) -> None:
        """Register activity color bins from the installed artifact.

        The AM7 install path publishes colour scales from the authoritative
        artifact rather than a live context installed beside it. The values are
        identical -- the artifact projects the same binning, scores, and subsystem
        summary the context carried.
        """
        for overlay in artifact.available_families:
            self._publish_activity_distributions(
                binning=artifact.binning(),
                overlay=overlay.value,
                metric_key=artifact.metric_key_for(overlay),
                feature_scores=artifact.feature_result_for(overlay).activity_score,
                subsystems=artifact.subsystems_for(overlay),
                force=force,
            )

    def _publish_activity_distributions(
        self,
        *,
        binning: dict,
        overlay: str,
        metric_key: str,
        feature_scores: pd.Series,
        subsystems: pd.DataFrame,
        force: bool,
    ) -> None:
        """Register the feature and subsystem colour bins for one activity view."""
        color_manager = getattr(self.data_manager, "color_manager", None)
        if (not force and not self.register_distributions) or color_manager is None:
            return

        scores = feature_scores.dropna()
        if not scores.empty:
            color_manager.register_distribution(
                mode=AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS,
                element_type=overlay,
                metric_name=metric_key,
                data=scores,
                **binning,
            )

        subsystem_scores = (
            subsystems["activity_score"].dropna() if not subsystems.empty else None
        )
        if subsystem_scores is not None and not subsystem_scores.empty:
            color_manager.register_distribution(
                mode=AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS,
                element_type="subsystem",
                metric_name=metric_key,
                data=subsystem_scores,
                **binning,
            )

    def get_activity_direction(
        self, artifact: "ActivityArtifact", reaction_id: str
    ) -> str:
        """Displayed direction of a reversible reaction under the selected group."""
        return artifact.display_direction(reaction_id)

    def clear_activity_cache(self) -> None:
        self._activity_cache.clear()
        self._activity_artifacts.clear()

    # ============================================================================
    # Individual Element Statistics Methods
    # ============================================================================

    def get_reaction_statistics(
        self, reaction_id: str, group_id: str | None = None
    ) -> Dict[str, Any]:
        """Get statistical information for a specific reaction.

        Args:
            reaction_id: Reaction identifier
            group_id: Specific group to get statistics for (if None, uses active group)

        Returns:
            Dict containing reaction statistics for ExperimentAnalysis
        """
        # Ensure analysis has been run
        if not self._analysis_completed:
            return {
                "has_experiment_data": False,
                "reason": "Analysis not yet completed. Call run_analysis() first.",
            }

        # Get group ID
        if group_id is None:
            # Use active group from data manager if available
            try:
                group_id = self.data_manager.get_active_group()
            except Exception:
                # If no active group, use the first available group
                if self.group_ids:
                    group_id = self.group_ids[0]
                else:
                    return {
                        "has_experiment_data": False,
                        "reason": "No groups available",
                    }

        if group_id not in self.group_ids:
            return {
                "has_experiment_data": False,
                "reason": f"Group '{group_id}' not found in analysis",
            }

        try:
            # Get z-score and average activity
            averages = self.get_reaction_avgs()

            if reaction_id not in averages.columns:
                return {
                    "has_experiment_data": False,
                    "reason": f"Reaction '{reaction_id}' not found in analysis",
                }

            average_activity = averages.loc[group_id, reaction_id]

            average_activity = (
                float(average_activity) if not pd.isna(average_activity) else 0.0
            )

            # Get sample count for context
            try:
                group_samples = self.data_manager.get_group_samples(group_id)
                reaction_df = self.data_manager.reaction_data.get_reaction_df()
                sample_count = len([s for s in group_samples if s in reaction_df.index])
            except Exception:
                sample_count = 0

            return {
                "has_experiment_data": True,
                "group_id": group_id,
                "main_direction": self.get_main_direction(group_id, reaction_id),
                "average_activity": average_activity,
                "element_type": "reaction",
                "sample_count": sample_count,
                "mode": "EXPERIMENT_ANALYSIS",
            }

        except Exception as e:
            return {"has_experiment_data": False, "error": str(e)}

    def get_gene_statistics(
        self, gene_id: str, group_id: str | None = None
    ) -> Dict[str, Any]:
        """Get statistical information for a specific gene.

        Args:
            gene_id: gene identifier
            group_id: Specific group to get statistics for (if None, uses active group)

        Returns:
            Dict containing gene statistics for ExperimentAnalysis
        """
        # Ensure analysis has been run
        if not self._analysis_completed:
            return {
                "has_experiment_data": False,
                "reason": "Analysis not yet completed. Call run_analysis() first.",
            }

        # Get group ID
        if group_id is None:
            # Use active group from data manager if available
            try:
                group_id = self.data_manager.get_active_group()
            except Exception:
                # If no active group, use the first available group
                if self.group_ids:
                    group_id = self.group_ids[0]
                else:
                    return {
                        "has_experiment_data": False,
                        "reason": "No groups available",
                    }

        if group_id not in self.group_ids:
            return {
                "has_experiment_data": False,
                "reason": f"Group '{group_id}' not found in analysis",
            }

        try:
            # Get z-score and average activity
            averages = self.get_gene_avgs()

            average_activity = averages.loc[group_id, gene_id]

            # Handle NaN values
            average_activity = (
                float(average_activity) if not pd.isna(average_activity) else 0.0
            )

            # Get sample count for context
            try:
                group_samples = self.data_manager.get_group_samples(group_id)
                expresion_df = self.data_manager.expresion_data.get_expresion_df()
                sample_count = len(
                    [s for s in group_samples if s in expresion_df.index]
                )
            except Exception:
                sample_count = 0

            return {
                "has_experiment_data": True,
                "group_id": group_id,
                "average_activity": average_activity,
                "element_type": "reaction",
                "sample_count": sample_count,
                "mode": "EXPERIMENT_ANALYSIS",
            }

        except Exception as e:
            return {"has_experiment_data": False, "error": str(e)}

    def get_subsystem_statistics(
        self, subsystem_name: str, group_id: str | None = None
    ) -> Dict[str, Any]:
        """Get statistical information for a specific subsystem.

        Args:
            subsystem_name: Subsystem name
            group_id: Specific group to get statistics for (if None, uses active group)

        Returns:
            Dict containing subsystem statistics for ExperimentAnalysis
        """
        # Ensure analysis has been run
        if not self._analysis_completed:
            return {
                "has_experiment_data": False,
                "reason": "Analysis not yet completed. Call run_analysis() first.",
            }

        # Get group ID
        if group_id is None:
            try:
                group_id = self.data_manager.get_active_group()
            except Exception:
                if self.group_ids:
                    group_id = self.group_ids[0]
                else:
                    return {"has_stats": False, "reason": "No groups available"}

        if group_id not in self.group_ids:
            return {
                "has_stats": False,
                "reason": f"Group '{group_id}' not found in analysis",
            }

        try:
            # Get average activity
            averages = self.get_subsystem_avgs()

            average_activity = averages.loc[group_id, subsystem_name]

            # Handle NaN values
            average_activity = (
                float(average_activity) if not pd.isna(average_activity) else 0.0
            )

            return {
                "has_stats": True,
                "group_id": group_id,
                "average_activity": average_activity,
                "element_type": "subsystem",
                "mode": "EXPERIMENT_ANALYSIS",
            }

        except Exception as e:
            return {"has_stats": False, "error": str(e)}

    def clear_cache(self):
        """Clear all cached results to force recalculation."""
        self._reaction_directions = None
        self._subsystem_avgs = None
        print("Cache cleared. Next analysis calls will recalculate results.")

    def run_analysis(
        self,
        include_pca: bool = True,
        pca_data_types: List[str] | None = None,
        n_components: Optional[int] = None,
        n_variable_features: int = 500,
    ) -> Dict[str, Any]:
        """Run comprehensive analysis for all groups in the data_manager.

        This method orchestrates all analysis calculations, similar to GroupComparison's
        run_test() method. It performs all statistical analyses and optionally PCA,
        storing results for easy access through getter methods.

        Args:
            include_pca: Whether to include PCA analysis
            pca_data_types: List of data types for PCA ('reactions', 'expression')
                           If None, defaults to ['reactions']
            n_components: Number of principal components to keep for PCA
            n_variable_features: Number of most variable features for PCA

        Returns:
            Dict[str, Any]: Results dictionary with status and messages
        """
        results: Dict[str, Any] = {"status": "success", "messages": []}

        if pca_data_types is None:
            pca_data_types = ["reactions"]

        try:
            # Step 1: Calculate group reaction averages
            group_avgs = self._calculate_reaction_avgs()
            if group_avgs is not None:
                results["messages"].append(
                    f"Group reaction averages calculated: {group_avgs.shape}"
                )
            else:
                results["status"] = "error"
                results["messages"].append(
                    "Failed to calculate group reaction averages"
                )
                return results

            # Step 3: Calculate reaction directions
            reaction_directions = self._calculate_reaction_directions()
            if reaction_directions is not None:
                results["messages"].append(
                    f"Reaction directions calculated: {reaction_directions.shape}"
                )
            else:
                results["status"] = "warning"
                results["messages"].append("Failed to calculate reaction directions")

            # Step 4: Calculate subsystem averages
            subsystem_avgs = self._calculate_subsystem_avgs()
            if subsystem_avgs is not None:
                results["messages"].append(
                    f"Subsystem averages calculated: {subsystem_avgs.shape}"
                )
            else:
                results["status"] = "warning"
                results["messages"].append("Failed to calculate subsystem averages")

            # Step 6: Calculate group gene averages
            if getattr(self.data_manager, "expression_data", None):
                gene_group_avgs = self._calculate_gene_avgs()
                if gene_group_avgs is not None:
                    results["messages"].append(
                        f"Group gene expression averages calculated: {gene_group_avgs.shape}"
                    )
                else:
                    results["status"] = "warning"
                    results["messages"].append(
                        "Failed to calculate group gene averages"
                    )

            # Step 7: Calculate PCA if requested
            if include_pca:
                for data_type in pca_data_types:
                    pca_results = self.calculate_pca(
                        data_type=data_type,
                        n_components=n_components,
                        n_variable_features=n_variable_features,
                    )

                    if pca_results["status"] == "success":
                        self._pca_completed[data_type] = True
                        results["messages"].append(f"PCA completed for {data_type}")
                    else:
                        results["status"] = "warning"
                        results["messages"].append(
                            f"PCA failed for {data_type}: {pca_results.get('messages', [])}"
                        )
                        print(f"   ⚠ PCA failed for {data_type}")

            # Mark analysis as completed
            self._analysis_completed = True

        except Exception as e:
            self._analysis_completed = False
            results["status"] = "error"
            error_msg = f"Error during analysis execution: {str(e)}"
            results["messages"].append(error_msg)
            print(f"\n✗ {error_msg}")

        return results

    def _register_distributions(self, element_type: str, data_df: pd.DataFrame) -> None:
        """Register average activity distributions per group.

        Args:
            element_type: The type of element ("reaction", "gene", "subsystem").
            data_df: DataFrame containing average activities (index=groups, columns=elements).
        """
        if not self.register_distributions or not hasattr(
            self.data_manager, "color_manager"
        ):
            return

        try:
            for group_id in data_df.index:
                # Get the series for this group (indexed by element_id)
                group_data = data_df.loc[group_id]

                # Skip if group data is all NaN
                if group_data.isna().all():
                    continue

                self.data_manager.color_manager.register_distribution(
                    mode=AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS,
                    element_type=element_type,
                    metric_name=str(group_id),  # Ensure Key is string
                    data=group_data,
                    center_method="median",  # Center on median activity
                    spread_method="mad",  # Use robust spread
                )
        except Exception as e:
            print(f"Warning: Failed to register distributions for {element_type}: {e}")

    def reset_analysis(self) -> None:
        """Reset all analysis results and status flags.

        This clears all cached results and resets the analysis state,
        requiring run_analysis() to be called again.
        """
        # Clear cached results
        self._reaction_avgs = None
        self._reaction_directions = None
        self._subsystem_avgs = None
        self._gene_avgs = None
        self._activity_cache.clear()
        self._activity_artifacts.clear()

        # Clear dimensionality reduction results
        self.dim_reduction.clear_results()

        # Reset status flags
        self._analysis_completed = False
        self._pca_completed = {"reactions": False, "expression": False}

        print("Analysis reset completed. All results cleared.")

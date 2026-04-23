"""Experiment analysis module for within-group characterization.

This module provides comprehensive analysis capabilities for data_managers with
multiple groups, focusing on within-group analysis of reaction and subsystem
activities. Unlike GroupComparison which compares the same reaction/subsystem
between different groups, this module compares different reactions/subsystems
within the same group to identify relative activation patterns.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..managers.data_manager_mode import DataManagerMode
from .dimensionality_reduction import DimensionalityReduction


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

    def __init__(self, data_manager, auto_run: bool = True):
        """Initialize the experiment analysis.

        Args:
            data_manager: An Experiment object with loaded data and defined groups
            auto_run: Whether to immediately run the default analysis pipeline

        Raises:
            ValueError: If data_manager is invalid or missing required data
        """
        self.data_manager = data_manager

        # Results storage
        self._reaction_avgs = None
        self._reaction_directions = None
        self._gene_avgs = None
        self._subsystem_avgs = None

        # Dimensionality reduction module
        self.dim_reduction = DimensionalityReduction()

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

    @property
    def pca_results(self):
        return self.dim_reduction.pca_results

    @property
    def tsne_results(self):
        return self.dim_reduction.tsne_results

    @property
    def umap_results(self):
        return self.dim_reduction.umap_results

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
        direction is dominant based on average activities. Only processes reactions
        that don't end with "_back".

        Logic:
        - Direction is "forward" if reaction_id + "_back" doesn't exist
        - If "_back" version exists, compare activities:
          - "forward" if activity(reaction) > abs(activity(reaction_back))
          - "backward" otherwise

        Returns:
            pd.DataFrame: Groups as rows, reactions as columns (excluding "_back" reactions),
                         values are "forward" or "backward"
        """
        # Get all reactions (excluding "_back" reactions)
        all_reactions = self._reaction_avgs.columns
        forward_reactions = [r for r in all_reactions if not r.endswith("_back")]

        # Initialize results DataFrame
        reaction_directions = pd.DataFrame(
            index=self.group_ids, columns=forward_reactions, dtype=str
        )

        # Calculate direction for each group and reaction
        for group_id in self.group_ids:
            for reaction_id in forward_reactions:
                try:
                    # Get forward reaction activity
                    forward_activity = self._reaction_avgs.loc[group_id, reaction_id]

                    # Check if backward version exists
                    backward_reaction_id = f"{reaction_id}_back"

                    if backward_reaction_id not in all_reactions:
                        # No backward version, direction is forward
                        reaction_directions.loc[group_id, reaction_id] = "forward"
                    else:
                        # Get backward reaction activity
                        backward_activity = self._reaction_avgs.loc[
                            group_id, backward_reaction_id
                        ]

                        # Handle NaN values
                        if pd.isna(forward_activity):
                            forward_activity = 0.0
                        if pd.isna(backward_activity):
                            backward_activity = 0.0

                        # Compare activities
                        if forward_activity >= abs(backward_activity):
                            reaction_directions.loc[group_id, reaction_id] = "forward"
                        else:
                            reaction_directions.loc[group_id, reaction_id] = "backward"

                except Exception as e:
                    print(
                        f"    Error processing reaction '{reaction_id}' "
                        f"in group '{group_id}': {str(e)}"
                    )
                    reaction_directions.loc[group_id, reaction_id] = "forward"

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
        # Get all subsystems
        subsystems = self.data_manager.metabolic_model.get_subsystems()

        # Initialize results DataFrame
        subsystem_avgs = pd.DataFrame(
            index=self.group_ids, columns=subsystems, dtype=float
        )

        # Calculate average for each subsystem in each group
        for group_id in self.group_ids:
            for subsystem in subsystems:
                try:
                    # Get reactions in this subsystem
                    reactions_in_subsystem = (
                        self.data_manager.metabolic_model.get_reactions_for_subsystem(
                            subsystem
                        )
                    )

                    if not reactions_in_subsystem:
                        subsystem_avgs.loc[group_id, subsystem] = np.nan
                        continue

                    # Filter reactions that are available in the data
                    available_reactions = [
                        reaction
                        for reaction in reactions_in_subsystem
                        if reaction in self._reaction_avgs.columns
                    ]

                    if not available_reactions:
                        subsystem_avgs.loc[group_id, subsystem] = np.nan
                        continue

                    # Calculate mean activity for this subsystem
                    reaction_data = self._reaction_avgs.loc[
                        group_id, available_reactions
                    ]
                    subsystem_mean = reaction_data.mean()
                    subsystem_avgs.loc[group_id, subsystem] = subsystem_mean

                except Exception as e:
                    print(f"    Error processing subsystem '{subsystem}': {str(e)}")
                    subsystem_avgs.loc[group_id, subsystem] = np.nan

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
        # Get expression data
        expression_df = self.data_manager.expression_data.get_expression_df()

        # Initialize results DataFrame
        gene_group_avgs = pd.DataFrame(
            index=self.group_ids, columns=expression_df.columns, dtype=float
        )

        # Calculate average for each group
        for group_id in self.group_ids:
            try:
                # Get samples for this group
                gene_group_samples = self.data_manager.get_group_samples(group_id)

                # Filter samples that exist in the data
                available_samples = [s for s in gene_group_samples if s in expression_df.index]

                if not available_samples:
                    print(
                        f"Warning: No samples found for group '{group_id}' "
                        f"in expression data"
                    )
                    # Fill with NaN for this group
                    gene_group_avgs.loc[group_id] = np.nan
                    continue

                # Calculate mean for this group
                gene_group_data = expression_df.loc[available_samples]
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
        """Get dataframe and source name for a given data type."""
        if data_type.lower() == "reactions":
            if self.data_manager.reaction_data is None:
                return None, "Reaction data not loaded."
            return self.data_manager.reaction_data.get_reaction_df(), "reaction data"
        elif data_type.lower() == "expression":
            if self.data_manager.expression_data is None:
                return None, "Expression data not loaded."
            return self.data_manager.expression_data.get_expression_df(), "expression data"
        return None, f"Invalid data_type: {data_type}"

    def calculate_pca(
        self,
        data_type: str = "reactions",
        n_components: Optional[int] = None,
        n_variable_features: int = 500,
    ) -> Dict[str, Any]:
        """Calculates PCA for the specified data type and stores the results."""
        df, source_name = self._get_data_for_reduction(data_type)
        if df is None:
            return {"status": "error", "messages": [source_name]}

        return self.dim_reduction.calculate_pca(
            df=df,
            data_type=data_type,
            data_source_name=source_name,
            n_components=n_components,
            n_variable_features=n_variable_features,
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
        df, source_name = self._get_data_for_reduction(data_type)
        if df is None:
            return {"status": "error", "messages": [source_name]}

        return self.dim_reduction.calculate_tsne(
            df=df,
            data_type=data_type,
            data_source_name=source_name,
            n_components=n_components,
            perplexity=perplexity,
            n_variable_features=n_variable_features,
            random_state=random_state,
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
        df, source_name = self._get_data_for_reduction(data_type)
        if df is None:
            return {"status": "error", "messages": [source_name]}

        return self.dim_reduction.calculate_umap(
            df=df,
            data_type=data_type,
            data_source_name=source_name,
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_variable_features=n_variable_features,
            random_state=random_state,
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
            raise ValueError(
                "Analysis not yet completed. Call run_analysis() first."
            )

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
    # Individual Element Statistics Methods
    # ============================================================================

    def get_reaction_statistics(
        self, reaction_id: str, group_id: str = None
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
                    return {"has_experiment_data": False, "reason": "No groups available"}

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

    def get_gene_statistics(self, gene_id: str, group_id: str = None) -> Dict[str, Any]:
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
                    return {"has_experiment_data": False, "reason": "No groups available"}

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
                sample_count = len([s for s in group_samples if s in expresion_df.index])
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
        self, subsystem_name: str, group_id: str = None
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
        pca_data_types: List[str] = None,
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
        start_time = time.time()
        results = {"status": "success", "messages": []}

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
                    results["messages"].append("Failed to calculate group gene averages")

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
        if not hasattr(self.data_manager, "color_manager"):
            return

        try:
            for group_id in data_df.index:
                # Get the series for this group (indexed by element_id)
                group_data = data_df.loc[group_id]

                # Skip if group data is all NaN
                if group_data.isna().all():
                    continue

                self.data_manager.color_manager.register_distribution(
                    mode=DataManagerMode.EXPERIMENT_ANALYSIS,
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

        # Clear dimensionality reduction results
        self.dim_reduction.clear_results()

        # Reset status flags
        self._analysis_completed = False
        self._pca_completed = {"reactions": False, "expression": False}

        print("Analysis reset completed. All results cleared.")
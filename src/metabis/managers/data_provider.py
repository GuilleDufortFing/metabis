"""Data provider module for metabolic network visualization.

This module contains the DataProvider class which handles all data retrieval
and statistical information for reactions, genes, and subsystems. It serves
as the core data access layer for the metabis application.
"""

import math
from typing import Any, Dict, List, Optional

from ..analysis.activity_analysis import (
    SUMMARY_DESCRIPTIONS,
    ActivityArtifact,
    ActivityOverlay,
    displayed_feature_id,
)
from ..analysis.comparison_membership import ComparisonMembership


def _finite_or_none(value: Any) -> Optional[float]:
    """Coerce a possibly-missing numeric to a finite float, or None."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class DataProvider:
    """Handles data retrieval and statistical information for reactions and genes.

    This class contains all data retrieval logic extracted from AnalysisWorkspace.
    It provides methods to get basic data, statistical data, experiment analysis data,
    distributions, and comprehensive data for reactions, genes, and subsystems.
    """

    def __init__(
        self,
        metabolic_model=None,
        reaction_data=None,
        expression_data=None,
        group_comparison_artifact=None,
        experiment_analysis=None,
        group_manager=None,
        subsystem_analysis=None,
    ):
        """Initialize DataProvider with data sources and managers.

        Args:
            metabolic_model: MetabolicModel object containing metabolic network
                information
            reaction_data: ReactionData object containing flux data
            expression_data: ExpressionData object containing gene expression data
            group_comparison_artifact: GroupComparisonArtifact with computed statistical
                test results, or None outside comparison mode
            experiment_analysis: ExperimentAnalysis object for within-group analysis
            group_manager: GroupManager object for group operations
            subsystem_analysis: SubsystemAnalysis object for subsystem-level analysis
        """
        self.metabolic_model = metabolic_model
        self.reaction_data = reaction_data
        # (report, frozenset) memo for `_blocked_feature_ids`; see D-65.
        self._blocked_feature_cache: tuple | None = None
        self.expression_data = expression_data
        self.group_comparison_artifact = group_comparison_artifact
        self.experiment_analysis = experiment_analysis
        self.group_manager = group_manager
        self.subsystem_analysis = subsystem_analysis
        # Mirror of the installed activity artifact; None outside activity mode.
        self.activity_artifact: Optional[ActivityArtifact] = None
        self.activity_overlay = ActivityOverlay.REACTION
        # Detailed subsystem graphs display both measured families at once. The
        # overview/ranking still follows ``activity_overlay``. Graph calculation
        # scopes this flag to subsystem enrichment so ordinary workspace reads do
        # not silently change meaning.
        self.activity_dual_evidence = False

    def activity_overlay_for_element(self, element_type: str) -> ActivityOverlay:
        """Return the activity family that owns this element's node evidence."""
        if self.activity_dual_evidence:
            if (
                element_type in {"gene", "subsystem_genes"}
                and self.activity_artifact is not None
                and self.activity_artifact.has_family(ActivityOverlay.GENE)
            ):
                return ActivityOverlay.GENE
            if element_type == "reaction":
                return ActivityOverlay.REACTION
        return ActivityOverlay(self.activity_overlay)

    # ============================================================================
    # Basic Data Methods
    # ============================================================================

    def get_reaction_metabolic_model_data(self, reaction_id: str) -> Dict[str, Any]:
        """Get basic reaction metadata and information.

        Args:
            reaction_id: Reaction identifier

        Returns:
            Dict containing basic reaction information
        """
        # Get reaction metadata from experiment
        if reaction_id.endswith("_back"):
            reaction_id = reaction_id[:-5]  # Remove "_back" suffix

        reaction_metadata = self.metabolic_model.get_reaction_metadata(reaction_id)

        is_reversible = reaction_metadata.get("is_reversible", False)
        # Fall back to the loaded reaction matrix so `_back` link resolution does
        # not affect reversibility detection.
        if not is_reversible and self.reaction_data is not None:
            reaction_df = self.reaction_data.get_reaction_df()
            is_reversible = f"{reaction_id}_back" in reaction_df.columns

        if not reaction_metadata:
            return {
                "is_in_model": False,
                "reaction_id": reaction_id,
            }

        subsystem_category = reaction_metadata.get("subsystem_category")

        return {
            "is_in_model": True,
            "reaction_id": reaction_id,
            "display_label": reaction_metadata.get("display_label", reaction_id),
            "external_url": reaction_metadata.get("external_url", ""),
            "external_source": reaction_metadata.get("external_source", ""),
            "name": reaction_metadata.get("name", "N/A"),  # Fixed: was "rxnName"
            "subsystem": reaction_metadata.get("subsystems", ["N/A"])[0]
            if reaction_metadata.get("subsystems")
            else "N/A",  # Fixed: was "subSystem", now gets first subsystem
            "subsystem_category": subsystem_category,
            "formula": reaction_metadata.get(
                "formula", reaction_metadata.get("equation", "")
            ),  # Fixed: use "formula" or "equation"
            "equation": reaction_metadata.get("equation", ""),
            "is_reversible": is_reversible,
            "blocked_directions": self.get_blocked_directions(reaction_id),
            "metadata": reaction_metadata,
        }

    def _blocked_feature_ids(self) -> frozenset:
        """The blocked set as a membership-testable set, built once per input.

        Every reaction hover asks about it and a graph render builds thousands,
        so the tuple is turned into a set once and memoized against the report
        that produced it -- rebuilding a 2,879-element set per hover is exactly
        the per-render work ADR-0007/MR-10 exists to keep out.
        """
        report = getattr(self.reaction_data, "preprocessing_report", None)
        cached = getattr(self, "_blocked_feature_cache", None)
        if cached is not None and cached[0] is report:
            return cached[1]
        blocked = frozenset(getattr(report, "blocked_feature_ids", ()) or ())
        self._blocked_feature_cache = (report, blocked)
        return blocked

    def get_blocked_directions(self, reaction_id: str) -> tuple:
        """Which directions of a reaction Compass reports no flux for (D-65).

        A reaction dropped as blocked otherwise renders exactly like one that was
        never in the input, turning a Compass *result* ("no flux") into an
        *absence* ("no data"). Only reactions removed for a zero penalty qualify;
        a merely invariant one has a real, nonzero penalty.
        """
        blocked = self._blocked_feature_ids()
        if not blocked:
            return ()

        base_id = reaction_id[:-5] if reaction_id.endswith("_back") else reaction_id
        directions = []
        if base_id in blocked:
            directions.append("forward")
        if f"{base_id}_back" in blocked:
            directions.append("reverse")
        return tuple(directions)

    def get_gene_metabolic_model_data(self, gene_id: str) -> Dict[str, Any]:
        """Get basic gene metadata and information.

        Args:
            gene_id: Gene identifier

        Returns:
            Dict containing basic gene information
        """
        # Get gene metadata from experiment
        gene_metadata = self.metabolic_model.get_gene_metadata(gene_id)

        if not gene_metadata:
            return {"is_in_model": False, "gene_id": gene_id}

        return {
            "is_in_model": True,
            "gene_id": gene_id,
            "display_label": gene_metadata.get("display_label", gene_id),
            "external_url": gene_metadata.get("external_url", ""),
            "external_source": gene_metadata.get("external_source", ""),
            "symbol": gene_metadata.get(
                "geneSymbols", "N/A"
            ),  # This matches the metabolic model
            "description": gene_metadata.get(
                "geneNames", ""
            ),  # Fixed: was "description", now use "geneNames"
            "metadata": gene_metadata,
        }

    def get_metabolite_metabolic_model_data(self, metabolite_id: str) -> Dict[str, Any]:
        """Get basic metabolite metadata and information.

        Args:
            metabolite_id: Metabolite identifier

        Returns:
            Dict containing basic metabolite information
        """
        # Get metabolite metadata from experiment
        metabolite_metadata = self.metabolic_model.get_metabolite_metadata(
            metabolite_id
        )

        if not metabolite_metadata:
            return {"is_in_model": False, "metabolite_id": metabolite_id}

        return {
            "is_in_model": True,
            "metabolite_id": metabolite_id,
            "name": metabolite_metadata.get("name", "N/A"),
            "formula": metabolite_metadata.get("formula", ""),
            "charge": metabolite_metadata.get("charge", None),
            "compartment": metabolite_metadata.get("compartment", "N/A"),
            "metadata": metabolite_metadata,
        }

    # ============================================================================
    # Group Comparison Data Methods
    # ============================================================================

    def get_reaction_group_comparison_data(self, reaction_id: str) -> Dict[str, Any]:
        """Get statistical information for a reaction.

        Args:
            reaction_id: Reaction identifier

        Returns:
            Dict containing statistical information
        """
        if not self.group_comparison_artifact:
            return {"has_comparison_data": False}

        stats = self.group_comparison_artifact.get_statistical_info(
            reaction_id, for_genes=False
        )
        back_stats = self.group_comparison_artifact.get_statistical_info(
            reaction_id + "_back", for_genes=False
        )
        # Either direction is enough. Requiring the forward one reported "no
        # experimental data" for a reaction measured only in reverse -- the same
        # forward-keyed assumption as D-64, one layer up -- so a node that colours
        # from its reverse direction had a hover denying the data existed.
        if not stats and not back_stats:
            return {"has_comparison_data": False}

        return {
            "has_comparison_data": True,
            "group1": self.group_comparison_artifact.membership.target_label,
            "group2": self.group_comparison_artifact.membership.baseline_label,
            "forward": stats,
            "backward": back_stats,
        }

    def get_gene_group_comparison_data(self, gene_id: str) -> Dict[str, Any]:
        """Get statistical information for a gene.

        Args:
            gene_id: Gene identifier

        Returns:
            Dict containing statistical information
        """
        if not self.group_comparison_artifact:
            return {"has_comparison_data": False}

        stats = self.group_comparison_artifact.get_statistical_info(
            gene_id, for_genes=True
        )
        if not stats:
            return {"has_comparison_data": False}

        return {
            "has_comparison_data": True,
            "group1": self.group_comparison_artifact.membership.target_label,
            "group2": self.group_comparison_artifact.membership.baseline_label,
            "forward": stats,
        }

    # ============================================================================
    # Experiment Analysis Data Methods
    # ============================================================================

    def get_active_group(self) -> Optional[str]:
        """The group the current activity view is scoped to, if any."""
        if self.activity_artifact is not None:
            return self.activity_artifact.group_id
        if self.group_manager is not None:
            return self.group_manager.get_active_group()
        return None

    def get_activity_distributions(
        self, element_id: str, element_type: str
    ) -> Dict[str, Any]:
        """Distributions for the single-group activity hover violin.

        Two things on one axis, both on the input scale:

        * the group summary of *every* measured feature, which says where this one
          sits among all of them; and
        * this feature's own values across the group's samples, which says how
          consistent that position is.

        Comparison mode plots one violin per compared group. Here there is only one
        group, so the useful contrast is against the rest of the features.
        """
        artifact = self.activity_artifact
        if artifact is None:
            return {"has_distributions": False}
        overlay = self.activity_overlay_for_element(element_type)
        if not artifact.colors_element_type(overlay, element_type):
            return {"has_distributions": False}
        reference_summary = artifact.reference_summary_for(overlay)
        if not reference_summary:
            return {"has_distributions": False}

        features = artifact.feature_result_for(overlay)
        if element_type == "reaction":
            source = self.reaction_data
            feature_id = displayed_feature_id(features.activity_score, element_id)
            row = source.get_reaction_row(feature_id) if source else None
        elif element_type == "gene":
            source = self.expression_data
            feature_id = element_id
            row = source.get_gene_row(feature_id) if source else None
        else:
            return {"has_distributions": False}

        if row is None:
            return {"has_distributions": False}

        # The artifact's own selected samples are the exact population the scores
        # summarize, so the violin cannot disagree with the colour beside it. They
        # are also the only source for a comparison-local Target/Baseline
        # population, which has no base-group record to look up.
        group_samples = artifact.selected_sample_ids
        if not group_samples:
            return {"has_distributions": False}

        present = [sample for sample in group_samples if sample in row.index]
        values = [
            float(value)
            for value in row.loc[present].to_numpy(dtype=float)
            if math.isfinite(value)
        ]
        if not values:
            return {"has_distributions": False}

        return {
            "has_distributions": True,
            "is_activity": True,
            "group_id": artifact.group_id,
            "feature_id": feature_id,
            "reference_values": list(reference_summary),
            "reference_name": (
                "All reactions" if element_type == "reaction" else "All genes"
            ),
            "reference_count": len(features.activity_score),
            "element_values": values,
            "element_name": feature_id,
            "sample_count": len(values),
        }

    def _has_measurement(self, element_id: str, element_type: str) -> bool:
        """Is this element present in its own input matrix, whatever the overlay is?"""
        if element_type == "reaction":
            if self.reaction_data is None:
                return False
            columns = self.reaction_data.get_reaction_df().columns
            base_id = element_id[:-5] if element_id.endswith("_back") else element_id
            return base_id in columns or f"{base_id}_back" in columns
        if element_type == "gene":
            if self.expression_data is None:
                return False
            # A membership test needs only the feature labels, not the dense
            # matrix behind them (AM6).
            return element_id in self.expression_data.get_feature_ids()
        return False

    def get_activity_statistics(
        self, element_id: str, element_type: str
    ) -> Dict[str, Any]:
        """Get single-group activity statistics for the selected activity context.

        Returns the normalized group summary and enough provenance for a hover
        to state what the number means. Contains no p-value, Cohen's d or fold
        change: nothing is being compared.
        """
        artifact = self.activity_artifact
        if artifact is None:
            return {"has_experiment_data": False}

        config = artifact.config
        overlay = self.activity_overlay_for_element(element_type)
        features = artifact.feature_result_for(overlay)
        common = {
            "has_experiment_data": True,
            "mode": "EXPERIMENT_ANALYSIS",
            "is_activity": True,
            "group_id": config.group_id,
            "sample_count": features.sample_count,
            "reference_sample_count": features.reference_sample_count,
            "overlay": overlay.value,
            "summary": config.summary.value,
            "summary_description": SUMMARY_DESCRIPTIONS[config.summary],
            "element_type": element_type,
            "is_colored": artifact.colors_element_type(overlay, element_type),
        }

        if not artifact.colors_element_type(overlay, element_type):
            # The other overlay's glyphs are not scored, but they are not "missing"
            # either as long as the element exists in its own matrix. Saying they
            # have no data would dash every reaction whenever genes are overlaid.
            if not self._has_measurement(element_id, element_type):
                return {"has_experiment_data": False, "mode": "EXPERIMENT_ANALYSIS"}
            return {**common, "element_id": element_id, "activity_score": None}

        if element_type in ("subsystem", "subsystem_genes"):
            row = artifact.subsystem_row(overlay, element_id)
            if row is None:
                return {"has_experiment_data": False, "mode": "EXPERIMENT_ANALYSIS"}
            return {
                **common,
                "subsystem_id": element_id,
                "activity_score": row.score,
                "coverage": row.coverage,
                "measured_features": row.measured_reactions,
                "total_model_features": row.total_model_reactions,
                "positive_count": row.positive_count,
                "negative_count": row.negative_count,
                "minimum_coverage": config.minimum_subsystem_coverage,
                "aggregation": "median of measured member scores",
            }

        base_id = element_id[:-5] if element_id.endswith("_back") else element_id

        if element_type == "reaction" and overlay is ActivityOverlay.REACTION:
            shown_id = displayed_feature_id(features.activity_score, base_id)
            backward_id = f"{base_id}_back"
            if features.score(base_id) is None and features.score(backward_id) is None:
                return {"has_experiment_data": False, "mode": "EXPERIMENT_ANALYSIS"}
            return {
                **common,
                "element_id": shown_id,
                "main_direction": "backward"
                if shown_id.endswith("_back")
                else "forward",
                "activity_score": features.score(shown_id),
                "forward_activity_score": features.score(base_id),
                "backward_activity_score": features.score(backward_id),
                "is_reversible": backward_id in features.activity_score.index,
                "is_non_finite": shown_id in features.non_finite_feature_ids,
            }

        score = features.score(element_id)
        if element_id not in features.activity_score.index:
            return {"has_experiment_data": False, "mode": "EXPERIMENT_ANALYSIS"}
        return {
            **common,
            "element_id": element_id,
            "activity_score": score,
            "is_non_finite": element_id in features.non_finite_feature_ids,
        }

    def get_experiment_analysis_statistics(
        self, element_id: str, element_type: str, group_id: str = None
    ) -> Dict[str, Any]:
        """Get statistics for ExperimentAnalysis mode.

        Prefers the installed activity artifact when one is selected, and uses
        the canonical Experiment Summary for an explicitly requested group.

        Args:
            element_id: ID of the element
            element_type: Type of element
            group_id: Specific group to get statistics for

        Returns:
            Dict containing ExperimentAnalysis statistics
        """
        if self.activity_artifact is not None and group_id is None:
            return self.get_activity_statistics(element_id, element_type)

        if not self.experiment_analysis:
            return {"has_experiment_data": False}

        try:
            if element_type == "reaction":
                result = self.experiment_analysis.get_reaction_statistics(
                    element_id, group_id
                )
                if result.get("has_experiment_data"):
                    result["mode"] = "EXPERIMENT_ANALYSIS"
                return result

            elif element_type == "subsystem":
                result = self.experiment_analysis.get_subsystem_statistics(
                    element_id, group_id
                )
                if result.get("has_experiment_data"):
                    result["mode"] = "EXPERIMENT_ANALYSIS"
                return result

            elif element_type == "gene":
                result = self.experiment_analysis.get_gene_statistics(
                    element_id, group_id
                )
                if result.get("has_experiment_data"):
                    result["mode"] = "EXPERIMENT_ANALYSIS"
                return result

        except Exception as e:
            print(f"Error getting ExperimentAnalysis statistics for {element_id}: {e}")

        return {"has_experiment_data": False, "mode": "EXPERIMENT_ANALYSIS"}

    # ============================================================================
    # Distribution Data Methods
    # ============================================================================

    def _comparison_membership(self) -> Optional[ComparisonMembership]:
        """The exact samples the installed comparison ran on.

        Sides come from the comparison's own membership, never from current group
        state, so a plotted distribution cannot disagree with the statistics
        shown beside it after the groups are redefined.
        """
        return getattr(self.group_comparison_artifact, "membership", None)

    def get_reaction_flux_distributions(self, reaction_id: str) -> Dict[str, Any]:
        """Get flux distributions for both groups.

        Args:
            reaction_id: Reaction identifier

        Returns:
            Dict containing flux distributions
        """
        if not self.group_comparison_artifact or not self.reaction_data:
            return {"has_distributions": False}

        membership = self._comparison_membership()
        if membership is None:
            return {"has_distributions": False}

        group1_samples = membership.target_samples
        group2_samples = membership.baseline_samples

        if not group1_samples or not group2_samples:
            return {"has_distributions": False}

        try:
            reaction_row = self.reaction_data.get_reaction_row(reaction_id)
            if reaction_row is None:
                return {"has_distributions": False}

            # Extract distributions
            dist1 = [
                reaction_row[sample]
                for sample in group1_samples
                if sample in reaction_row.index
            ]
            dist2 = [
                reaction_row[sample]
                for sample in group2_samples
                if sample in reaction_row.index
            ]

            if not dist1 or not dist2:
                return {"has_distributions": False}

            return {
                "has_distributions": True,
                "group1_name": membership.target_label,
                "group2_name": membership.baseline_label,
                "group1_values": dist1,
                "group2_values": dist2,
            }

        except Exception as e:
            return {"has_distributions": False, "error": str(e)}

    def get_gene_expression_distributions(self, gene_id: str) -> Dict[str, Any]:
        """Get expression distributions for both groups.

        Args:
            gene_id: Gene identifier

        Returns:
            Dict containing expression distributions
        """
        if not self.group_comparison_artifact or not self.expression_data:
            return {"has_distributions": False}

        membership = self._comparison_membership()
        if membership is None:
            return {"has_distributions": False}

        group1_samples = membership.target_samples
        group2_samples = membership.baseline_samples

        if not group1_samples or not group2_samples:
            return {"has_distributions": False}

        try:
            gene_row = self.expression_data.get_gene_row(gene_id)
            if gene_row is None:
                return {"has_distributions": False}

            # Extract distributions
            dist1 = [
                gene_row[sample]
                for sample in group1_samples
                if sample in gene_row.index
            ]
            dist2 = [
                gene_row[sample]
                for sample in group2_samples
                if sample in gene_row.index
            ]

            if not dist1 or not dist2:
                return {"has_distributions": False}

            return {
                "has_distributions": True,
                "group1_name": membership.target_label,
                "group2_name": membership.baseline_label,
                "group1_values": dist1,
                "group2_values": dist2,
            }

        except Exception as e:
            return {"has_distributions": False, "error": str(e)}

    def get_metric_distribution_data(
        self, metric: str = "log2_fc", for_genes: bool = False
    ) -> Dict[str, Any]:
        """Get the distribution of a metric across all elements for color reference.

        Args:
            metric: The metric to get distribution for ('log2_fc', 'cohens_d', etc.)
            for_genes: If True, get gene data; if False, get reaction data

        Returns:
            Dict containing:
                - values: List of all metric values
                - metric_name: Human-readable metric name
                - min_value: Minimum value in distribution
                - max_value: Maximum value in distribution
                - has_data: Boolean indicating if data is available
        """
        if not self.group_comparison_artifact:
            return {"has_data": False, "error": "No group comparison data available"}

        try:
            # Get test results from GroupComparison
            test_results = self.group_comparison_artifact.get_test_results(
                for_genes=for_genes
            )

            if test_results is None or test_results.empty:
                return {"has_data": False, "error": "No test results available"}

            # Check if metric exists in the data
            if metric not in test_results.columns:
                return {
                    "has_data": False,
                    "error": f"Metric '{metric}' not found in test results",
                }

            # Extract metric values (excluding NaN values)
            metric_values = test_results[metric].dropna().tolist()

            if not metric_values:
                return {"has_data": False, "error": f"No valid {metric} values found"}

            # Get human-readable metric name
            metric_display_names = {
                "log2_fc": "Log2 Fold Change",
                "cohens_d": "Cohen's d",
                "mean_difference": "Mean Difference",
                "pvalue": "P-value",
            }
            metric_name = metric_display_names.get(
                metric, metric.replace("_", " ").title()
            )

            return {
                "has_data": True,
                "values": metric_values,
                "metric_name": metric_name,
                "min_value": min(metric_values),
                "max_value": max(metric_values),
                "element_type": "genes" if for_genes else "reactions",
            }

        except Exception as e:
            return {"has_data": False, "error": str(e)}

    # ============================================================================
    # Comprehensive Data Methods
    # ============================================================================

    def get_reaction_data(
        self,
        reaction_id: str,
        include_stats: bool = True,
        include_influence: bool = True,
        include_distributions: bool = False,
    ) -> Dict[str, Any]:
        """Get comprehensive reaction data including all available information.

        Args:
            reaction_id: Reaction identifier
            include_stats: Whether to include statistical information
            include_influence: Whether to include gene influence information
            include_distributions: Whether to include flux distributions

        Returns:
            Dict containing all requested reaction information
        """
        data = self.get_reaction_metabolic_model_data(reaction_id)

        if not data["is_in_model"]:
            return data

        if include_stats:
            data["comparison_stats"] = self.get_reaction_group_comparison_data(
                reaction_id
            )
            # Add experiment analysis stats for EXPERIMENT_ANALYSIS mode
            data["experiment_stats"] = self.get_experiment_analysis_statistics(
                reaction_id, "reaction", None
            )

        # TODO
        # if include_influence:
        #     data["influence"] = self.get_reaction_influence_data(
        #         reaction_id, data["subsystem"]
        #     )

        if include_distributions and self.activity_artifact is not None:
            # Activity mode has no second group to plot against, so it gets its own
            # reference-vs-feature distributions instead of the comparison violins.
            data["distributions"] = self.get_activity_distributions(
                reaction_id, "reaction"
            )
            return data

        if include_distributions:
            # Always attempt to fetch forward distributions first
            data["distributions"] = self.get_reaction_flux_distributions(reaction_id)

            # Try to fetch backward reaction distributions as well. Previously this
            # was gated on the presence of "backward" statistics which meant the
            # backward violin would be omitted when stats were not requested. We
            # should surface backward distributions when they exist regardless of
            # whether backward stats were computed.
            try:
                back_distributions = self.get_reaction_flux_distributions(
                    f"{reaction_id}_back"
                )
            except Exception:
                back_distributions = {"has_distributions": False}

            # Ensure the main distributions dict exists and attach backward
            if isinstance(data.get("distributions"), dict):
                data["distributions"]["backward"] = back_distributions

        return data

    def get_main_direction_significant_reactions(self) -> List[str]:

        group_id = self.group_comparison_artifact.group_id_1

        significant_reactions = self.group_comparison_artifact.get_significant_elements(
            for_genes=False
        )

        res = []

        for reaction_id in significant_reactions:
            # If reaction_id ends in "_back"
            if (
                reaction_id.endswith("_back")
                and self.experiment_analysis.get_main_direction(
                    group_id, reaction_id[:-5]
                )
                == "backward"
            ):
                res.append(reaction_id[:-5])
            # Else
            elif (
                not reaction_id.endswith("_back")
                and self.experiment_analysis.get_main_direction(group_id, reaction_id)
                == "forward"
            ):
                res.append(reaction_id)

        return res

    def get_gene_data(
        self,
        gene_id: str,
        include_stats: bool = True,
        include_distributions: bool = False,
    ) -> Dict[str, Any]:
        """Get comprehensive gene data including all available information.

        Args:
            gene_id: Gene identifier
            include_stats: Whether to include statistical information
            include_distributions: Whether to include expression distributions

        Returns:
            Dict containing all requested gene information
        """
        data = self.get_gene_metabolic_model_data(gene_id)

        if not data["is_in_model"]:
            return data

        if include_stats:
            data["comparison_stats"] = self.get_gene_group_comparison_data(gene_id)
            # Add experiment analysis stats for EXPERIMENT_ANALYSIS mode
            data["experiment_stats"] = self.get_experiment_analysis_statistics(
                gene_id, "gene", None
            )

        if include_distributions:
            if self.activity_artifact is not None:
                data["distributions"] = self.get_activity_distributions(gene_id, "gene")
            else:
                data["distributions"] = self.get_gene_expression_distributions(gene_id)

        return data

    def get_metabolite_data(
        self,
        metabolite_id: str,
    ) -> Dict[str, Any]:
        """Placeholder for future metabolite data retrieval.
        Get comprehensive metabolite data including all available information.

        Args:
            metabolite_id: Metabolite identifier

        Returns:
            Dict containing all requested metabolite information
        """
        data = self.get_metabolite_metabolic_model_data(metabolite_id)

        return data

    def get_subsystem_data(
        self,
        subsystem_name: str,
        for_genes: bool = False,
        delta_config: Optional[Dict[str, Any]] = None,
        compute_delta: bool = True,
    ) -> Dict[str, Any]:
        """Get subsystem-level statistics and information.

        Args:
            subsystem_name: Name of the subsystem/pathway
            for_genes: Whether to get gene-level data (True) or reaction-level (False)
            delta_config: Optional configuration dictionary for delta score calculation.
                Supported keys (all optional, defaults shown):
                    - metric: str = "log2_fc"
                    - transform: str = "sign"  # one of {"sign", "raw", "abs"}
                    - numerator_filter: str = "all"  # {"all", "significant"}
                    - denominator_filter: str = "all"  # {"all", "significant"}
            compute_delta: Whether to compute the delta_subsystem_score (default True)

        Returns:
            Dict containing subsystem information
        """
        if not self.subsystem_analysis:
            return {
                "subsystem_name": subsystem_name,
                "for_genes": for_genes,
                "is_in_model": False,
            }

        return self.subsystem_analysis.get_subsystem_statistics(
            subsystem_name=subsystem_name,
            for_genes=for_genes,
            delta_config=delta_config,
            compute_delta=compute_delta,
        )

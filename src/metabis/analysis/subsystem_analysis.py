from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..managers.workspace_mode import AnalysisWorkspaceMode
from .comparison_artifact import GroupComparisonArtifact
from .reaction_direction import (
    DEFAULT_REACTION_DIRECTION_CRITERION,
    ReactionDirectionCriterion,
    ReactionDirectionMap,
    ReactionDirectionSelection,
    build_reaction_direction_map,
)

__all__ = [
    "SubsystemAnalysis",
    "SubsystemDeltaArtifact",
]


@dataclass(frozen=True)
class SubsystemDeltaArtifact:
    """Detached subsystem delta state produced by a worker analysis."""

    base_stats: Mapping[Tuple[str, bool], Mapping[str, Any]]
    delta_stats: Mapping[Tuple[str, bool], Mapping[str, Any]]
    config: Mapping[str, Any] | None
    distributions: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    reaction_direction_map: ReactionDirectionMap | None = None

    def __post_init__(self) -> None:
        """Detach constructor inputs even when callers build an artifact directly."""
        object.__setattr__(self, "base_stats", _freeze_delta_value(self.base_stats))
        object.__setattr__(self, "delta_stats", _freeze_delta_value(self.delta_stats))
        object.__setattr__(self, "config", _freeze_delta_value(self.config))
        object.__setattr__(self, "distributions", _freeze_delta_value(self.distributions))
        if self.reaction_direction_map is not None and not isinstance(
            self.reaction_direction_map, ReactionDirectionMap
        ):
            raise TypeError("reaction_direction_map must be a ReactionDirectionMap")

    def thaw_base_stats(self) -> Dict[Tuple[str, bool], Dict[str, Any]]:
        return _thaw_delta_value(self.base_stats)

    def thaw_delta_stats(self) -> Dict[Tuple[str, bool], Dict[str, Any]]:
        return _thaw_delta_value(self.delta_stats)

    def thaw_config(self) -> Dict[str, Any] | None:
        return _thaw_delta_value(self.config)


def _freeze_delta_value(value: Any) -> Any:
    """Recursively detach built-in scientific values carried across a worker."""
    if isinstance(value, pd.DataFrame):
        raise TypeError("SubsystemDeltaArtifact cannot retain mutable DataFrames")
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_delta_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_delta_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_delta_value(item) for item in value)
    return deepcopy(value)


def _thaw_delta_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_delta_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_delta_value(item) for item in value]
    if isinstance(value, frozenset):
        return {_thaw_delta_value(item) for item in value}
    return deepcopy(value)


class SubsystemAnalysis:
    """Handles subsystem-level analysis for GroupComparison, including:
    - Subsystem statistics calculation
    - Delta score computation
    - Subsystem ranking
    """

    def __init__(
        self,
        data_manager=None,
        default_delta_config=None,
    ):
        """Initialize the SubsystemAnalysis module.

        Args:
            group_comparison_instance: The parent GroupComparison instance.
            metabolic_model: The metabolic model instance.
            reaction_data: The reaction data manager.
            expression_data: The expression data manager.
            default_delta_config: Default configuration for delta score calculation.
        """
        self.dm = data_manager
        self._gc = None  # Will be accessed via property
        self._base_stats_cache = {}

        self._current_delta_config = default_delta_config
        self._delta_stats_cache = {}
        self._reaction_direction_map: ReactionDirectionMap | None = None

    # Properties to access data through dm (avoids stale references)
    @property
    def gc(self):
        """The installed comparison artifact, from a local override or the workspace."""
        if self._gc is not None:
            return self._gc
        return self.dm.get_group_comparison_artifact() if self.dm else None

    @gc.setter
    def gc(self, value):
        """Set the comparison-artifact override."""
        self._gc = value

    @property
    def metabolic_model(self):
        """Get metabolic_model from data_manager."""
        return self.dm.metabolic_model if self.dm else None

    @property
    def reaction_data(self):
        """Get reaction_data from data_manager."""
        return self.dm.reaction_data if self.dm else None

    @property
    def expression_data(self):
        """Get expression_data from data_manager."""
        return self.dm.expression_data if self.dm else None

    @property
    def color_manager(self):
        """Get color_manager from data_manager."""
        return self.dm.color_manager if self.dm else None

    def set_group_comparison(
        self,
        group_comparison,
    ):
        """Update the GroupComparison instance and data context."""
        self._gc = group_comparison
        self._base_stats_cache = {}
        self._delta_stats_cache = {}
        self._reaction_direction_map = None
        if self._gc and self.metabolic_model:
            self._reaction_direction_map = self._build_reaction_direction_map()
            self._precompute_all_base_stats()
            if self._current_delta_config:
                self._precompute_all_delta_stats(self._current_delta_config)

    def update_delta_config(self, delta_config: Dict[str, Any]) -> bool:
        """Update the delta configuration and recompute delta stats.

        This recomputes base stats (to get current significant elements)
        and then computes delta stats with the new configuration.

        Args:
            delta_config: New configuration for delta score calculation.
                          Expected keys: 'metric', 'transform',
                          'numerator_filter', 'denominator_filter'

        Returns:
            bool: True if delta stats were successfully recomputed,
                  False if prerequisites are missing.
        """
        if not self.gc or not self.metabolic_model:
            return False

        self._current_delta_config = deepcopy(delta_config)
        self._reaction_direction_map = self._build_reaction_direction_map()
        # Recompute base stats to get current significant elements
        self._precompute_all_base_stats()

        # Now compute delta stats with the new config
        self._precompute_all_delta_stats(delta_config)
        return True

    @staticmethod
    def _collect_delta_distributions(
        delta_stats: Dict[Tuple[str, bool], Dict[str, Any]],
    ) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
        """Extract immutable color-distribution inputs from delta statistics."""
        scores_by_type: Dict[bool, Dict[str, float]] = {False: {}, True: {}}
        for (subsystem, for_genes), stats in delta_stats.items():
            if not stats.get("delta_subsystem_meta", {}).get("computed"):
                continue
            score = stats.get("delta_subsystem_score")
            if score is None:
                continue
            try:
                scores_by_type[for_genes][subsystem] = float(score)
            except (TypeError, ValueError):
                continue

        distributions = []
        for for_genes in (False, True):
            scores = scores_by_type[for_genes]
            if not scores:
                continue
            element_type = "subsystem_genes" if for_genes else "subsystem"
            distributions.append(
                (element_type, tuple(sorted(scores.items(), key=lambda item: item[0])))
            )
        return tuple(distributions)

    def _register_delta_distributions(
        self,
        distributions: tuple[tuple[str, tuple[tuple[str, float], ...]], ...],
        config: Dict[str, Any] | None,
    ) -> None:
        """Publish detached distribution inputs through the color adapter."""
        if self.color_manager is None:
            return
        metric_name = (config or {}).get("metric", "default")
        for element_type, scores in distributions:
            self.color_manager.register_distribution(
                mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
                element_type=element_type,
                data=pd.Series(dict(scores), dtype=float),
                metric_name=metric_name,
                center_method="zero",
                spread_method="std",
            )

    def get_delta_artifact(self) -> SubsystemDeltaArtifact:
        """Return a detached copy of current subsystem delta scientific state."""
        distributions = self._collect_delta_distributions(self._delta_stats_cache)
        return SubsystemDeltaArtifact(
            base_stats={
                key: {
                    name: deepcopy(value)
                    for name, value in stats.items()
                    if name != "_context"
                }
                for key, stats in self._base_stats_cache.items()
            },
            delta_stats=deepcopy(self._delta_stats_cache),
            config=deepcopy(self._current_delta_config),
            distributions=deepcopy(distributions),
            reaction_direction_map=self._reaction_direction_map,
        )

    def install_delta_artifact(
        self,
        artifact: SubsystemDeltaArtifact,
        *,
        register_presentation: bool = True,
    ) -> SubsystemDeltaArtifact:
        """Install detached delta state and optionally publish its color scales."""
        self._base_stats_cache = artifact.thaw_base_stats()
        self._delta_stats_cache = artifact.thaw_delta_stats()
        self._current_delta_config = artifact.thaw_config()
        self._reaction_direction_map = artifact.reaction_direction_map
        distributions = tuple(
            (element_type, tuple((name, float(score)) for name, score in scores))
            for element_type, scores in artifact.distributions
        )
        if register_presentation:
            self._register_delta_distributions(
                distributions,
                self._current_delta_config,
            )
        return artifact

    def get_reaction_direction_map(self) -> ReactionDirectionMap | None:
        """The exact shared selection every direction-aware surface reads.

        Built on demand when a comparison is available but nothing has installed
        the map yet. This is not an optimisation: the map is now what the graph
        draws, colours, and hovers, and it was previously built only by
        ``set_group_comparison`` / ``update_delta_config``. A graph worker that
        renders without first preparing comparison caches would otherwise fall
        back to the per-group averages frame and answer with the *Target*
        population regardless of the selected criterion -- silently, because the
        fallback returns a valid direction rather than an error (WP10/D-70).
        """
        if self._reaction_direction_map is None:
            self._reaction_direction_map = self._build_reaction_direction_map()
        return self._reaction_direction_map

    def invalidate_reaction_direction(
        self,
        criterion: ReactionDirectionCriterion,
    ) -> None:
        """Invalidate reaction-only Delta state after the shared criterion changes."""
        if not isinstance(criterion, ReactionDirectionCriterion):
            raise TypeError("criterion must be a ReactionDirectionCriterion")
        config = deepcopy(self._current_delta_config or {})
        config["reaction_direction_criterion"] = criterion.value
        self._current_delta_config = config
        # Rebuild rather than clear. The map is now also what the graph draws and
        # the hover reports, and a cleared map falls back to the per-group
        # averages frame -- which would answer with the *previous* criterion until
        # the next render rebuilt it. Rebuilding costs two mean-column reads.
        self._reaction_direction_map = self._build_reaction_direction_map()
        self._base_stats_cache = {
            key: value for key, value in self._base_stats_cache.items() if key[1]
        }
        self._delta_stats_cache = {
            key: value for key, value in self._delta_stats_cache.items() if key[1]
        }

    def _build_reaction_direction_map(self) -> ReactionDirectionMap | None:
        """Build the shared map once for the active comparison/configuration."""
        if not isinstance(self.gc, GroupComparisonArtifact):
            return None
        config = self._current_delta_config or {}
        criterion = ReactionDirectionCriterion(
            config.get(
                "reaction_direction_criterion",
                DEFAULT_REACTION_DIRECTION_CRITERION.value,
            )
        )
        return build_reaction_direction_map(self.gc, criterion)

    def _precompute_all_base_stats(self):
        """Pre-compute base statistics for all subsystems."""
        self._base_stats_cache = {}

        if not self.metabolic_model:
            return

        subsystems = self.metabolic_model.get_subsystems() or []

        for subsystem in subsystems:
            # Compute for reactions
            self._base_stats_cache[(subsystem, False)] = self._compute_base_stats(
                subsystem, for_genes=False
            )
            # Compute for genes
            self._base_stats_cache[(subsystem, True)] = self._compute_base_stats(
                subsystem, for_genes=True
            )

    def _precompute_all_delta_stats(self, delta_config: Dict[str, Any]):
        """Pre-compute delta statistics and distributions for all subsystems."""
        self._current_delta_config = deepcopy(delta_config)
        self._delta_stats_cache = {}

        if not self._base_stats_cache:
            return

        for key, base_stats in self._base_stats_cache.items():
            delta_stats = self._compute_delta_stats(
                base_stats,
                self._current_delta_config,
            )
            self._delta_stats_cache[key] = delta_stats

        self._register_delta_distributions(
            self._collect_delta_distributions(self._delta_stats_cache),
            self._current_delta_config,
        )

    def _get_valid_subsystem_elements(
        self,
        subsystem_name: str,
        metabolic_model,
        reaction_data,
        expression_data,
        for_genes: bool,
    ) -> List[str]:
        """Retrieve valid elements for the subsystem."""
        if for_genes:
            if expression_data is None:
                return []
            subsystem_elements = (
                metabolic_model.get_genes_for_subsystem(subsystem_name) or []
            )
            # Only the feature labels are needed; reading them off var_names avoids
            # densifying the whole matrix for a membership test (AM6).
            available_elements = set(expression_data.get_feature_ids())
        else:
            if reaction_data is None:
                return []
            base_elements = (
                metabolic_model.get_reactions_for_subsystem(subsystem_name) or []
            )
            available_elements = set(reaction_data.get_reaction_df().columns)
            if self._reaction_direction_map is not None:
                subsystem_elements = [
                    source
                    for element in base_elements
                    if (
                        source := self._reaction_direction_map.source_column_for(element)
                    )
                    is not None
                    and source in available_elements
                ]
            else:
                subsystem_elements = base_elements

        return [elem for elem in subsystem_elements if elem in available_elements]

    def _calculate_significance_stats(
        self, elements: List[str], for_genes: bool
    ) -> Dict[str, Any]:
        """Calculate significance statistics for the elements."""
        significant_set = self.gc.get_significant_elements(for_genes=for_genes)
        significant_in_subsystem = [elem for elem in elements if elem in significant_set]

        return {
            "significant_elements": len(significant_in_subsystem),
            "significant_percentage": (
                (len(significant_in_subsystem) / len(elements)) * 100 if elements else 0
            ),
            "significant_set": significant_set,
        }

    def _calculate_metric_aggregates(
        self, stats_frame: pd.DataFrame, metrics_to_process: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate aggregated metric values and net direction scores.

        The net direction score is the signed sum of a metric over the subsystem's
        elements divided by its absolute sum: a net share of effect magnitude in
        [-1, 1], where positive is net higher in Target and negative is net lower
        in Target (higher in Baseline). It is not a count of elements and it
        establishes no regulation (D-63).
        """
        total_metric_values: Dict[str, float] = {}
        net_direction_scores: Dict[str, float] = {}

        for metric in metrics_to_process:
            if metric not in stats_frame.columns or stats_frame.empty:
                total_metric_values[metric] = 0.0
                net_direction_scores[metric] = 0.0
                continue

            metric_series = stats_frame[metric].dropna().astype(float)
            if metric_series.empty:
                total_metric_values[metric] = 0.0
                net_direction_scores[metric] = 0.0
                continue

            total_sum = float(metric_series.sum())
            abs_sum = float(metric_series.abs().sum())
            total_metric_values[metric] = total_sum
            net_direction_scores[metric] = total_sum / abs_sum if abs_sum > 0 else 0.0

        return {
            "total_metric_values": total_metric_values,
            "net_direction_scores": net_direction_scores,
        }

    def _calculate_delta_score(
        self,
        stats_frame: pd.DataFrame,
        elements: List[str],
        significant_set: set,
        delta_options: Dict[str, Any],
        skipped_due_to_missing: int,
        direction_audit: tuple[ReactionDirectionSelection, ...] = (),
    ) -> Dict[str, Any]:
        """Calculate the delta subsystem score."""
        metric_for_delta = delta_options["metric"]
        numerator_sum = 0.0
        numerator_count = 0
        denominator_count = 0
        delta_score = 0.0
        skipped_elements = skipped_due_to_missing

        if metric_for_delta not in stats_frame.columns or stats_frame.empty:
            skipped_elements = len(elements)
        else:
            metric_series = stats_frame[metric_for_delta]
            valid_metric_series = metric_series.dropna().astype(float)
            skipped_elements += int(metric_series.isna().sum())

            if not valid_metric_series.empty:
                valid_index = list(valid_metric_series.index)
                significant_mask = [idx in significant_set for idx in valid_index]

                if delta_options["denominator_filter"] == "significant":
                    denominator_indices = [
                        idx
                        for idx, is_sig in zip(valid_index, significant_mask)
                        if is_sig
                    ]
                else:
                    denominator_indices = valid_index

                denominator_count = len(denominator_indices)

                if denominator_count > 0:
                    if delta_options["numerator_filter"] == "significant":
                        numerator_indices = [
                            idx
                            for idx, is_sig in zip(valid_index, significant_mask)
                            if is_sig
                        ]
                    else:
                        numerator_indices = valid_index

                    numerator_values = [
                        self._apply_delta_transform(
                            valid_metric_series[idx], delta_options["transform"]
                        )
                        for idx in numerator_indices
                    ]

                    numerator_sum = float(sum(numerator_values))
                    numerator_count = len(numerator_values)

                    delta_score = numerator_sum / denominator_count

        return {
            "delta_subsystem_score": delta_score,
            "delta_subsystem_meta": {
                "metric": metric_for_delta,
                "transform": delta_options["transform"],
                "numerator_filter": delta_options["numerator_filter"],
                "denominator_filter": delta_options["denominator_filter"],
                "numerator_sum": numerator_sum,
                "numerator_count": numerator_count,
                "denominator_count": denominator_count,
                "skipped_elements": skipped_elements,
                "computed": True,
                "reaction_direction_criterion": (
                    self._reaction_direction_map.criterion.value
                    if self._reaction_direction_map is not None
                    else None
                ),
                "reaction_direction_fingerprint": (
                    self._reaction_direction_map.fingerprint
                    if self._reaction_direction_map is not None
                    else None
                ),
                "selected_reaction_directions": tuple(
                    {
                        "base_reaction_id": entry.base_reaction_id,
                        "source_column": entry.source_column,
                        "selected_direction": entry.selected_direction.value,
                        "target_mean_activity": entry.target_mean_activity,
                        "baseline_mean_activity": entry.baseline_mean_activity,
                        "reason": entry.reason,
                    }
                    for entry in direction_audit
                ),
            },
        }


    @staticmethod
    def _apply_delta_transform(value: float, mode: str) -> float:
        """Apply a transform to a metric value for delta calculations."""
        if mode == "sign":
            if value > 0:
                return 1.0
            if value < 0:
                return -1.0
            return 0.0
        if mode == "raw":
            return float(value)
        if mode == "abs":
            return abs(float(value))
        raise ValueError(f"Unsupported transform mode '{mode}'")

    def _compute_base_stats(
        self,
        subsystem_name: str,
        for_genes: bool,
    ) -> Dict[str, Any]:
        """Calculate base statistics independent of delta options."""
        data = {
            "subsystem_name": subsystem_name,
            "for_genes": for_genes,
            "is_in_model": False,
        }

        if self.metabolic_model is None:
            return data

        test_results = self.gc.get_test_results(for_genes=for_genes)
        if test_results is None:
            return data

        # 1. Get Elements (reactions or genes)
        elements = self._get_valid_subsystem_elements(
            subsystem_name,
            self.metabolic_model,
            self.reaction_data,
            self.expression_data,
            for_genes,
        )
        if not elements:
            return data

        data["is_in_model"] = True
        data["total_elements"] = len(elements)

        # 2. Calculate Significance
        sig_stats = self._calculate_significance_stats(elements, for_genes)
        data["significant_elements"] = sig_stats["significant_elements"]
        data["significant_percentage"] = sig_stats["significant_percentage"]
        significant_set = sig_stats["significant_set"]

        # Prepare Stats Frame
        elements_with_stats = [elem for elem in elements if elem in test_results.index]
        stats_frame = (
            test_results.loc[elements_with_stats].copy()
            if elements_with_stats
            else pd.DataFrame(index=pd.Index([], name=test_results.index.name))
        )
        skipped_due_to_missing = len(elements) - len(elements_with_stats)

        # 3. Calculate Metric Aggregates
        metrics_to_process = ["log2_fc", "cohens_d", "mean_difference"]
        aggregates = self._calculate_metric_aggregates(stats_frame, metrics_to_process)

        data["total_metric_values"] = aggregates["total_metric_values"]
        data["net_direction_scores"] = aggregates["net_direction_scores"]

        # Store context for phase 2
        data["_context"] = {
            "stats_frame": stats_frame,
            "elements": elements,
            "significant_set": significant_set,
            "skipped_due_to_missing": skipped_due_to_missing,
            "direction_audit": (
                tuple(
                    entry
                    for element in elements
                    if (
                        entry := self._reaction_direction_map.get(element)
                    )
                    is not None
                )
                if not for_genes and self._reaction_direction_map is not None
                else ()
            ),
        }

        return data

    def _compute_delta_stats(
        self,
        base_stats: Dict[str, Any],
        delta_options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate delta score using base statistics."""
        # Create a copy that excludes _context (don't modify cached base_stats)
        context = base_stats.get("_context", None)
        result = {k: v for k, v in base_stats.items() if k != "_context"}

        if not context or not result.get("is_in_model"):
            result["delta_subsystem_score"] = None
            result["delta_subsystem_meta"] = {"computed": False}
            return result

        stats_frame = context["stats_frame"]
        elements = context["elements"]
        significant_set = context["significant_set"]
        skipped_due_to_missing = context["skipped_due_to_missing"]
        direction_audit = context.get("direction_audit", ())

        # 4. Calculate Delta Score
        delta_stats = self._calculate_delta_score(
            stats_frame,
            elements,
            significant_set,
            delta_options,
            skipped_due_to_missing,
            direction_audit,
        )

        # Combine results
        result["delta_subsystem_score"] = delta_stats["delta_subsystem_score"]
        result["delta_subsystem_meta"] = delta_stats["delta_subsystem_meta"]

        # Update total_metric_values with delta score
        if "total_metric_values" in result:
            result["total_metric_values"]["delta_subsystem_score"] = delta_stats[
                "delta_subsystem_score"
            ]

        return result

    def get_subsystem_statistics(
        self,
        subsystem_name: str,
        for_genes: bool = False,
        delta_config: Optional[Dict[str, Any]] = None,
        compute_delta: bool = True,
    ) -> Dict[str, Any]:
        """Public API to retrieve subsystem statistics."""
        base_stats = self._base_stats_cache.get((subsystem_name, for_genes))

        if base_stats is None:
            # Fallback if not cached (e.g. if precompute wasn't called or failed)
            base_stats = self._compute_base_stats(subsystem_name, for_genes)

        # Determine effective configuration
        effective_config = delta_config if delta_config is not None else self._current_delta_config

        if not compute_delta or effective_config is None:
            # Return a copy without the internal context
            result = base_stats.copy()
            if "_context" in result:
                del result["_context"]
            return result

        # Check if we need to recompute global delta stats
        # Only recompute global cache if we have a new config that differs from current
        if delta_config is not None and delta_config != self._current_delta_config:
            self._precompute_all_delta_stats(delta_config)

        # Retrieve from cache
        delta_stats = self._delta_stats_cache.get((subsystem_name, for_genes))

        # If not in cache or if we are using a specific config that might not be the global one
        # (Note: The logic above recomputes global if delta_config is passed.
        # If delta_config is None, we use effective_config (current global).
        # So we should check if we have a valid cached result for the effective config.)

        if delta_stats is None:
            delta_stats = self._compute_delta_stats(base_stats, effective_config)

        return delta_stats

    def rank_subsystems_by_delta_score(
        self,
        for_genes: bool = False,
    ) -> pd.DataFrame:
        """Rank subsystems by their pre-computed delta score.

        Args:
            for_genes: Whether to rank gene subsystems (True) or reaction subsystems (False).

        Returns:
            DataFrame containing subsystem statistics sorted by absolute delta score.
            ``net_direction_score`` is the net share of effect magnitude in
            [-1, 1] for ``impact_score_metric``: positive is net higher in Target,
            negative is net lower in Target (higher in Baseline). This frame
            reaches the user as a download, so no column may name a signed
            difference "regulation" (D-63).
        """
        subsystem_stats = []

        # Iterate over cached delta stats
        for key, stats in self._delta_stats_cache.items():
            subsystem_name, is_for_genes = key

            # Filter by element type
            if is_for_genes != for_genes:
                continue

            # Skip if delta score is not computed
            delta_score = stats.get("delta_subsystem_score")
            if delta_score is None:
                continue

            # Extract relevant fields
            meta = stats.get("delta_subsystem_meta", {})
            metric_name = meta.get("metric", "unknown")

            # Net share of effect magnitude for the metric the delta used
            net_direction_scores = stats.get("net_direction_scores", {})
            net_direction_score = net_direction_scores.get(metric_name, 0.0)

            stats_dict = {
                "subsystem": subsystem_name,
                "delta_score": delta_score,
                "abs_delta_score": abs(delta_score),
                "n_significant": stats.get("significant_elements", 0),
                "n_total": stats.get("total_elements", 0),
                "prop_significant": stats.get("significant_percentage", 0) / 100.0,
                "net_direction_score": net_direction_score,
                "impact_score_metric": metric_name,
            }

            subsystem_stats.append(stats_dict)

        if not subsystem_stats:
            # Return empty DataFrame with expected columns if no data
            return pd.DataFrame(columns=[
                "subsystem", "delta_score", "abs_delta_score",
                "n_significant", "n_total", "prop_significant",
                "net_direction_score", "impact_score_metric"
            ])

        # Create DataFrame and sort by absolute delta score
        stats_df = pd.DataFrame(subsystem_stats)
        return stats_df.sort_values("abs_delta_score", ascending=False)

from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from ..analysis_workspace import WorkspaceProvenance
from ..managers.workspace_mode import AnalysisWorkspaceMode
from .artifacts import (
    AnalysisArtifactMetadata,
    ArtifactDimension,
    ArtifactKind,
    PlainTable,
)
from .comparison_artifact import (
    ComparisonThresholds,
    GroupComparisonArtifact,
    MetricSummary,
)
from .adjustment import ComparisonAdjustment
from .comparison_membership import ComparisonMembership, GroupComparisonSpec
from .importance_analysis import (
    ImportanceAnalysis,
)
from .statistical_analysis import (
    SUPPORTED_TESTS,
    StatisticalAnalysis,
)


def register_comparison_distributions(
    data_manager, results: pd.DataFrame | None, *, for_genes: bool = False
) -> None:
    """Register a comparison's metric distributions into the color manager.

    Shared by the live-producer path and the artifact-install path so a
    comparison's colouring is registered identically whether it was just
    computed or read back from an installed :class:`GroupComparisonArtifact`.
    """
    if not hasattr(data_manager, "color_manager"):
        return

    if results is None:
        element_name = "gene" if for_genes else "reaction"
        raise ValueError(f"No computed {element_name} results to register")
    element_type = "gene" if for_genes else "reaction"

    metrics = [
        "log2_fc",
        "cohens_d",
        "mean_difference",
        "pval",
        "adjusted_pval",
        "neg_log_pval",
        "neg_log_adjusted_pval",
    ]

    for metric in metrics:
        if metric not in results.columns:
            continue
        data_manager.color_manager.register_distribution(
            mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
            element_type=element_type,
            metric_name=metric,
            data=results[metric],
            center_method="zero",
            spread_method="std",
        )


def _metric_summaries(
    metrics: Mapping[str, Mapping[str, float]],
) -> Tuple[MetricSummary, ...]:
    """Freeze normalization metrics into sorted, immutable summaries."""
    return tuple(
        MetricSummary(name, float(values["std"]), float(values["mean"]))
        for name, values in sorted(metrics.items())
    )


def _display_label_pairs(
    labels: Mapping[str, str] | None,
    feature_ids: Iterable[Any],
) -> Tuple[Tuple[str, str], ...]:
    """Pair every feature with its display label, defaulting to its canonical ID.

    A missing label is not an error: the canonical ID is a truthful label, and a
    label is never allowed to replace the ID it labels.
    """
    supplied = {str(key): str(value) for key, value in dict(labels or {}).items()}
    return tuple(
        sorted(
            (str(feature_id), supplied.get(str(feature_id), str(feature_id)))
            for feature_id in feature_ids
        )
    )


class GroupComparison:
    def __init__(self, *, spec: GroupComparisonSpec | None = None, membership: ComparisonMembership | None = None, test: str = "wilcoxon", correction: str = "fdr_bh", adjustment: ComparisonAdjustment | None = None) -> None:
        """Initialize a group comparison from an explicit specification or membership.

        Args:
            spec (GroupComparisonSpec): Explicit comparison selection (mode plus the
                one or two canonical group IDs).
            membership (ComparisonMembership): Explicit left/right sample membership.
                When omitted, `run_test` resolves it from the analysis workspace.
            test (str): Statistical test to use ('wilcoxon', 'ttest' or
                'moderated_t')
            adjustment (ComparisonAdjustment): Sample-sheet columns to adjust
                for. Only the moderated t-test accepts them.

        Exactly one of `spec` or `membership` is required; when both are given they
        must agree.

        Returns:
            None

        Raises:
            ValueError: If neither `spec` nor `membership` is given, if they
                disagree, or if the test is not supported
        """
        if membership is not None:
            if spec is not None and spec != membership.spec:
                raise ValueError("Comparison membership does not match its specification")
            spec = membership.spec
        elif spec is None:
            raise ValueError("GroupComparison requires an explicit spec or membership")
        self.spec = spec
        self.membership = membership
        self.group_id_1 = " + ".join(spec.target.base_group_ids)
        self.group_id_2 = (
            " + ".join(spec.baseline.base_group_ids)
            if spec.baseline is not None
            else "All other samples"
        )
        self.significant_reactions = set()
        self.back_significant_reactions = set()
        self.significant_genes = set()

        # Validate and store the statistical test
        if test.lower() not in SUPPORTED_TESTS:
            raise ValueError(
                f"Unsupported statistical test: {test}. Supported tests: {list(SUPPORTED_TESTS)}"
            )
        self.statistical_test = test.lower()
        if adjustment is not None and self.statistical_test != "moderated_t":
            raise ValueError("Only the moderated t-test can adjust for sample-sheet columns")
        self.adjustment = adjustment

        # Initialize the StatisticalAnalysis instance
        self.stats_analyzer = StatisticalAnalysis(test_type=test, correction_type=correction)

        # Initialize the ImportanceAnalysis instance with default parameters
        self.importance_analyzer = ImportanceAnalysis()

        # Store test results as single dataframes
        self._reaction_test_results = None  # Will be a DataFrame
        self._gene_test_results = None  # Will be a DataFrame

        # Stable defaults for effect-size normalization before results are computed.
        self._reaction_metrics_stats = {
            "cohens_d": {"std": 1.0, "mean": 0.0},
            "log2_fc": {"std": 1.0, "mean": 0.0},
            "mean_difference": {"std": 0.5, "mean": 0.0},
        }
        self._gene_metrics_stats = {
            "cohens_d": {"std": 1.0, "mean": 0.0},
            "log2_fc": {"std": 1.0, "mean": 0.0},
            "mean_difference": {"std": 0.5, "mean": 0.0},
        }

    @classmethod
    def from_computed_results(
        cls,
        *,
        spec: GroupComparisonSpec,
        test: str,
        correction: str,
        membership: ComparisonMembership | None = None,
        reaction_results: pd.DataFrame,
        reaction_metrics: Mapping[str, Mapping[str, float]],
        significant_reactions: Iterable[str],
        reaction_thresholds: Mapping[str, float | None],
        gene_results: pd.DataFrame | None = None,
        gene_metrics: Mapping[str, Mapping[str, float]] | None = None,
        significant_genes: Iterable[str] = (),
        gene_thresholds: Mapping[str, float | None] | None = None,
    ) -> "GroupComparison":
        """Construct a comparison from an already-computed scientific result.

        This is the public document-thread installation seam for worker results.
        The comparison owns its result tables and metric mappings after this call;
        callers do not need to populate implementation fields directly.
        """
        significant_gene_ids = tuple(significant_genes)
        if reaction_results is None:
            raise ValueError("Computed comparison requires reaction results")
        if gene_results is None and (
            gene_metrics is not None
            or significant_gene_ids
            or gene_thresholds is not None
        ):
            raise ValueError("Gene comparison state requires gene results")
        if gene_results is not None and (
            gene_metrics is None or gene_thresholds is None
        ):
            raise ValueError(
                "Computed gene results require metrics and thresholds"
            )

        comparison = cls(
            spec=spec,
            membership=membership,
            test=test,
            correction=correction,
        )
        comparison._reaction_test_results = reaction_results.copy()
        comparison._reaction_metrics_stats = deepcopy(dict(reaction_metrics))
        comparison.significant_reactions = set(significant_reactions)
        comparison._set_computed_thresholds(
            reaction_thresholds,
            for_genes=False,
        )

        if gene_results is not None:
            comparison._gene_test_results = gene_results.copy()
            comparison._gene_metrics_stats = deepcopy(dict(gene_metrics or {}))
            comparison.significant_genes = set(significant_gene_ids)
            comparison._set_computed_thresholds(
                gene_thresholds or {},
                for_genes=True,
            )
        return comparison

    def build_artifact(
        self,
        *,
        provenance: WorkspaceProvenance,
        source_fingerprints: Mapping[str, str],
        reaction_display_labels: Mapping[str, str] | None = None,
        gene_display_labels: Mapping[str, str] | None = None,
        keep_reversed: bool = False,
        warnings: Iterable[str] = (),
    ) -> GroupComparisonArtifact:
        """Project this computed comparison into its portable artifact.

        `source_fingerprints` is read from the Analysis Workspace, which paid for
        input identity once at load.  Computing it here instead would put a
        full-matrix hash on every comparison; see ADR-0007 and MR-10.

        Display labels are supplied by the caller that can see the metabolic
        model, so an exported row carries its label without the export reaching
        back into a live manager to look one up.
        """
        if self.membership is None:
            raise ValueError(
                "A comparison needs a resolved membership before it can become an "
                "artifact"
            )
        reaction_results = self.get_test_results(for_genes=False)
        if reaction_results is None:
            raise ValueError("This comparison has no reaction results to export")
        gene_results = self.get_test_results(for_genes=True)

        correction = self.stats_analyzer.correction_type
        configuration: Dict[str, Any] = {
            "correction": correction,
            "keep_reversed": bool(keep_reversed),
            "adjusted_for": list(self.adjustment.labels) if self.adjustment else [],
            "reaction_thresholds": dict(
                ComparisonThresholds.capture(
                    self.get_thresholds(for_genes=False)
                ).as_mapping()
            ),
            "gene_thresholds": (
                dict(
                    ComparisonThresholds.capture(
                        self.get_thresholds(for_genes=True)
                    ).as_mapping()
                )
                if gene_results is not None
                else None
            ),
        }
        metadata = AnalysisArtifactMetadata.from_workspace(
            kind=ArtifactKind.GROUP_COMPARISON,
            provenance=provenance,
            source_fingerprints=source_fingerprints,
            method=self.statistical_test,
            configuration=configuration,
            scientific_identity=self._artifact_identity(),
            dimensions=(
                ArtifactDimension("samples", len(self.membership.universe)),
                ArtifactDimension("target_samples", len(self.membership.target_samples)),
                ArtifactDimension(
                    "baseline_samples", len(self.membership.baseline_samples)
                ),
                ArtifactDimension("reactions", len(reaction_results)),
                ArtifactDimension(
                    "genes", len(gene_results) if gene_results is not None else 0
                ),
            ),
            warnings=tuple(warnings),
        )
        return GroupComparisonArtifact(
            metadata=metadata,
            membership=self.membership,
            test=self.statistical_test,
            correction=correction,
            reaction_results=PlainTable.capture(reaction_results),
            gene_results=(
                PlainTable.capture(gene_results) if gene_results is not None else None
            ),
            reaction_metrics=_metric_summaries(self.get_metrics_stats(for_genes=False)),
            gene_metrics=(
                _metric_summaries(self.get_metrics_stats(for_genes=True))
                if gene_results is not None
                else ()
            ),
            significant_reactions=tuple(sorted(map(str, self.significant_reactions))),
            significant_genes=(
                tuple(sorted(map(str, self.significant_genes)))
                if gene_results is not None
                else ()
            ),
            reaction_thresholds=ComparisonThresholds.capture(
                self.get_thresholds(for_genes=False)
            ),
            gene_thresholds=(
                ComparisonThresholds.capture(self.get_thresholds(for_genes=True))
                if gene_results is not None
                else None
            ),
            keep_reversed=bool(keep_reversed),
            reaction_display_labels=_display_label_pairs(
                reaction_display_labels, reaction_results.index
            ),
            gene_display_labels=(
                _display_label_pairs(gene_display_labels, gene_results.index)
                if gene_results is not None
                else ()
            ),
        )

    def _artifact_identity(self) -> Dict[str, Any]:
        """Return the membership coordinates that identify this comparison."""
        membership = self.membership
        return {
            "membership_fingerprint": membership.fingerprint,
            "mode": membership.mode.value,
            "target_base_group_ids": membership.target_base_group_ids,
            "baseline_base_group_ids": membership.baseline_base_group_ids,
            "target_samples": membership.target_samples,
            "baseline_samples": membership.baseline_samples,
            "universe": membership.universe,
            "excluded_base_group_ids": membership.excluded_base_group_ids,
        }

    def _set_computed_thresholds(
        self,
        thresholds: Mapping[str, float | None],
        *,
        for_genes: bool,
    ) -> None:
        self.set_thresholds(
            thresholds.get("pval"),
            thresholds.get("cohens_d"),
            thresholds.get("log2_fc"),
            thresholds.get("mean_difference"),
            for_genes=for_genes,
        )

    def replace_threshold_state_from(self, candidate: "GroupComparison") -> None:
        """Commit prevalidated threshold-dependent state while retaining identity.

        AnalysisWorkspace uses this after it has successfully built the matching
        immutable artifact, so presentation/provider references to this object do
        not need to be rewired and a failed pre-commit remains a no-op.
        """
        if not isinstance(candidate, GroupComparison):
            raise TypeError("candidate must be a GroupComparison")
        if (
            candidate.spec != self.spec
            or candidate.membership != self.membership
            or candidate.statistical_test != self.statistical_test
            or candidate.stats_analyzer.correction_type
            != self.stats_analyzer.correction_type
        ):
            raise ValueError("Threshold candidate belongs to a different comparison")

        reaction_results = candidate.get_test_results(for_genes=False)
        if reaction_results is None:
            raise ValueError("Threshold candidate has no reaction results")
        current_reactions = self.get_test_results(for_genes=False)
        if current_reactions is None or not current_reactions.index.equals(
            reaction_results.index
        ):
            raise ValueError("Threshold candidate reaction rows do not match")
        reaction_significance = reaction_results["is_significant"].copy()
        gene_results = candidate.get_test_results(for_genes=True)
        current_genes = self.get_test_results(for_genes=True)
        if (gene_results is None) != (current_genes is None):
            raise ValueError("Threshold candidate gene result availability differs")
        if gene_results is not None and not current_genes.index.equals(
            gene_results.index
        ):
            raise ValueError("Threshold candidate gene rows do not match")
        gene_significance = (
            gene_results["is_significant"].copy()
            if gene_results is not None
            else None
        )
        reaction_thresholds = dict(candidate.get_thresholds(for_genes=False))
        gene_thresholds = (
            dict(candidate.get_thresholds(for_genes=True))
            if gene_results is not None
            else None
        )
        significant_reactions = set(candidate.significant_reactions)
        significant_genes = set(candidate.significant_genes)

        current_reactions.loc[:, "is_significant"] = reaction_significance
        if current_genes is not None and gene_significance is not None:
            current_genes.loc[:, "is_significant"] = gene_significance
        self.significant_reactions = significant_reactions
        self.significant_genes = significant_genes
        self._set_computed_thresholds(reaction_thresholds, for_genes=False)
        if gene_thresholds is not None:
            self._set_computed_thresholds(gene_thresholds, for_genes=True)

    def _calculate_metrics_stats(
        self, results: pd.DataFrame, for_genes: bool = False
    ) -> None:
        """Calculate statistics for metrics normalization.

        Args:
            results (pd.DataFrame): Test results
            for_genes (bool): Whether the results are for genes (True) or reactions (False)

        Returns:
            None
        """
        # Use the StatisticalAnalysis class to calculate metrics stats
        calculated_stats = self.stats_analyzer.calculate_metrics_stats(results)

        # Update the appropriate metrics stats based on element type
        if for_genes:
            self._gene_metrics_stats.update(calculated_stats)
        else:
            self._reaction_metrics_stats.update(calculated_stats)

    def register_distributions(self, data_manager, *, for_genes: bool = False) -> None:
        """Register computed metric distributions with a presentation adapter.

        Args:
            data_manager: The data manager instance.
            for_genes: Register gene distributions instead of reaction distributions.
        """
        register_comparison_distributions(
            data_manager,
            self.get_test_results(for_genes=for_genes),
            for_genes=for_genes,
        )

    def get_metrics_stats(self, for_genes: bool = False) -> Dict[str, Dict[str, float]]:
        """Get the metrics statistics used for normalization.

        Args:
            for_genes (bool): Whether to get gene metrics (True) or reaction metrics (False)

        Returns:
            Dict[str, Dict[str, float]]: Dictionary of metrics statistics with structure:
                {
                    "metric_name": {
                        "mean": float_value,
                        "std": float_value
                    },
                    ...
                }
        """
        # Return the appropriate metrics stats dictionary
        if for_genes:
            return self._gene_metrics_stats
        else:
            return self._reaction_metrics_stats

    def resolve_membership(self, data_manager) -> ComparisonMembership:
        """Resolve and retain this comparison's explicit left/right membership.

        Args:
            data_manager: The analysis workspace holding canonical group records.

        Returns:
            ComparisonMembership: The exact samples the comparison will run on.
        """
        self.membership = data_manager.group_manager.resolve_comparison_membership(
            self.spec, data_manager.reaction_data.get_sample_ids()
        )
        return self.membership

    def run_test(
        self,
        data_manager,
        for_genes: bool = False,
        test: str = "wilcoxon",  # default test type
        correction: str = "fdr_bh",
        register_distributions: bool = True,
    ) -> None:
        """Run the statistical test on reaction or gene data.

        Args:
            data_manager (Experiment): The data_manager object containing data
            for_genes (bool): If True, run the test on gene data; otherwise on reaction data
            test (str): 'wilcoxon', 'ttest' or 'moderated_t' (default 'wilcoxon')

        Returns:
            None
        """
        membership = self.membership or self.resolve_membership(data_manager)
        group_1_samples = list(membership.target_samples)
        group_2_samples = list(membership.baseline_samples)
        covariates = (
            self.adjustment.design(group_1_samples + group_2_samples)
            if self.adjustment is not None
            else None
        )

        if for_genes:
            # The gene test compares only the two groups' samples, so materialize
            # just those rows dense through the seam -- a bounded block, never the
            # whole expression matrix (AM6 slice 4). ``comparison_test`` selects
            # exactly these samples after transposing, so the bounded frame yields
            # the identical result the full frame did.
            data_df = data_manager.expression_data.matrix().select_samples(
                group_1_samples + group_2_samples
            )
        else:
            data_df = data_manager.reaction_data.get_reaction_df()

        # Use StatisticalAnalysis to perform the comparison
        results = self.stats_analyzer.run_statistical_test(
            data_df,
            group_1_samples,
            group_2_samples,
            test_type=test,
            correction_type=correction,
            covariates=covariates,
        )

        # Add group information
        results["group_1"] = membership.target_label
        results["group_2"] = membership.baseline_label

        # Add IDs
        if for_genes:
            results["gene_id"] = results.index
            # Add subsystems information to the results DataFrame
            results["subsystems"] = results.index.map(
                lambda gid: ", ".join(
                    data_manager.metabolic_model.get_gene_metadata(gid).get(
                        "subsystems", ["Unknown"]
                    )
                )
            )
            self._gene_test_results = results
        else:
            results["reaction_id"] = results.index
            # Add subsystems information to the results DataFrame
            results["subsystems"] = results.index.map(
                lambda rid: ", ".join(
                    data_manager.metabolic_model.get_reaction_metadata(rid).get(
                        "subsystems", ["Unknown"]
                    )
                )
            )
            self._reaction_test_results = results

        # Default thresholds for all significant
        p_val_threshold = 1
        cohens_d_threshold = 0.0
        log2_fc_threshold = 0.0
        mean_difference_threshold = 0.0

        self.define_significant_elements(for_genes=for_genes,
            pval_threshold=p_val_threshold,
            cohens_d_threshold=cohens_d_threshold,
            log2_fc_threshold=log2_fc_threshold,
            mean_difference_threshold=mean_difference_threshold,
        )

        # Calculate statistics for effect-size normalization
        self._calculate_metrics_stats(results, for_genes=for_genes)

        # Register distributions for coloring
        if register_distributions:
            self.register_distributions(data_manager, for_genes=for_genes)



    def get_test_results(
        self, for_genes: bool = False
    ) -> Optional[pd.DataFrame]:
        """Get the test results for reactions or genes.

        Args:
            for_genes (bool): Return gene test results when True; reaction results otherwise.

        Returns:
            Optional[pd.DataFrame]: Test results for the requested element type and direction.
        """
        if for_genes:
            return self._gene_test_results
        else:
            return self._reaction_test_results

    def define_significant_elements(
        self,
        pval_threshold: Optional[float] = None,
        cohens_d_threshold: Optional[float] = None,
        log2_fc_threshold: Optional[float] = None,
        mean_difference_threshold: Optional[float] = None,
        keep_reversed: bool = False,
        for_genes: bool = False,
    ) -> Tuple[set, Dict[str, Any]]:
        """Identify significant reactions based on statistical thresholds.

        Args:
            pval_threshold (float, optional): P-value threshold for significance.
            cohens_d_threshold (float, optional): Minimum absolute Cohen's d value.
            log2_fc_threshold (float, optional): Minimum absolute log2 fold change.
            mean_difference_threshold (float, optional): Minimum absolute mean difference.
            keep_reversed (bool): Whether to always keep reactions with reversed flow
                as significant regardless of log2_fc value.
            for_genes (bool): Whether to identify significant genes instead of reactions.

        Returns:
            Tuple[set, Dict]: Tuple of (set of significant element IDs, threshold_info dict)

        Raises:
            ValueError: If reaction test hasn't been run yet
        """
        if for_genes:
            test_results = self._gene_test_results
        else:
            test_results = self._reaction_test_results

        # Check if test results exist
        # (If there are no test results then there shouldn't be rev test results either)
        if test_results is None:
            raise ValueError(
                "No test results found. Run the test first with run_test()."
            )

        # Use StatisticalAnalysis to determine significant elements
        significant_mask, threshold_info = (
            self.stats_analyzer.determine_significant_elements(
                test_results,
                pval_threshold=pval_threshold,
                cohens_d_threshold=cohens_d_threshold,
                log2_fc_threshold=log2_fc_threshold,
                mean_difference_threshold=mean_difference_threshold,
                keep_reversed=keep_reversed,
            )
        )

        # Store the significant elements based on the type (genes or reactions)
        if for_genes:
            self.significant_genes = set(test_results[significant_mask].index)
            self._gene_test_results["is_significant"] = self._gene_test_results.index.map(
                lambda gid: gid in self.significant_genes
            )
        else:
            self.significant_reactions = set(test_results[significant_mask].index)
            self._reaction_test_results[
                "is_significant"
            ] = self._reaction_test_results.index.map(
                lambda rid: rid in self.significant_reactions
            )

        self.set_thresholds(
            pval_threshold,
            cohens_d_threshold,
            log2_fc_threshold,
            mean_difference_threshold,
            for_genes,
        )

        return set(test_results[significant_mask].index), threshold_info

    def is_significant(
        self,
        element_id: str,
        for_genes: bool = False,
    ) -> bool:
        """Check if a specific element (reaction or gene) is significant.

        Args:
            element_id (str): ID of the element (reaction or gene)

        Returns:
            bool: True if the element is significant, False otherwise
        """
        if for_genes:
            return element_id in self.significant_genes
        return element_id in self.significant_reactions

    def set_thresholds(
        self,
        pval_threshold=None,
        cohens_d_threshold=None,
        log2_fc_threshold=None,
        mean_difference_threshold=None,
        for_genes=False,
    ):
        """Set threshold values for significance testing.

        Args:
            pval_threshold (float, optional): P-value threshold for significance.
            cohens_d_threshold (float, optional): Minimum absolute Cohen's d value.
            log2_fc_threshold (float, optional): Minimum absolute log2 fold change.
            mean_difference_threshold (float, optional): Minimum absolute mean difference.
            for_genes (bool): Whether to set thresholds for genes (True) or reactions (False).
        """
        if for_genes:
            self.adjusted_pval_threshold_gene = pval_threshold
            self.p_val_threshold_gene = pval_threshold
            self.neg_log_adj_pval_threshold_gene = (
                -np.log10(pval_threshold) if pval_threshold is not None else None
            )
            self.neg_log_pval_threshold_gene = (
                -np.log10(pval_threshold) if pval_threshold is not None else None
            )
            self.cohens_d_threshold_gene = cohens_d_threshold
            self.log2_fc_threshold_gene = log2_fc_threshold
            self.mean_difference_threshold_gene = mean_difference_threshold
        else:
            self.adjusted_pval_threshold_reaction = pval_threshold
            self.p_val_threshold_reaction = pval_threshold
            self.neg_log_adj_pval_threshold_reaction = (
                -np.log10(pval_threshold) if pval_threshold is not None else None
            )
            self.neg_log_pval_threshold_reaction = (
                -np.log10(pval_threshold) if pval_threshold is not None else None
            )
            self.cohens_d_threshold_reaction = cohens_d_threshold
            self.log2_fc_threshold_reaction = log2_fc_threshold
            self.mean_difference_threshold_reaction = mean_difference_threshold



    def get_thresholds(self, for_genes=False):
        """Get threshold values for significance testing.

        Args:
            for_genes (bool): Whether to get thresholds for genes (True) or reactions (False).

        Returns:
            dict: Dictionary containing the threshold values.
        """
        if for_genes:
            return {
                "adjusted_pval": self.adjusted_pval_threshold_gene,
                "pval": self.p_val_threshold_gene,
                "neg_log_adjusted_pval": self.neg_log_adj_pval_threshold_gene,
                "neg_log_pval": self.neg_log_pval_threshold_gene,
                "cohens_d": self.cohens_d_threshold_gene,
                "log2_fc": self.log2_fc_threshold_gene,
                "mean_difference": self.mean_difference_threshold_gene,
            }
        else:
            return {
                "adjusted_pval": self.adjusted_pval_threshold_reaction,
                "pval": self.p_val_threshold_reaction,
                "neg_log_adjusted_pval": self.neg_log_adj_pval_threshold_reaction,
                "neg_log_pval": self.neg_log_pval_threshold_reaction,
                "cohens_d": self.cohens_d_threshold_reaction,
                "log2_fc": self.log2_fc_threshold_reaction,
                "mean_difference": self.mean_difference_threshold_reaction,
            }

    def get_significant_elements(self, for_genes: bool = False, reverse = False) -> set:
        """Get the set of significant reactions or genes.

        Args:
            for_genes (bool): Whether to get significant genes (True) or reactions (False)

        Returns:
            set: Set of significant reaction or gene IDs
        """
        if for_genes:
            # If significant_genes is empty, return all genes
            return self.significant_genes
        else:
            return self.significant_reactions

    def get_statistical_info(
        self, element_id: str, for_genes: bool = False
    ) -> Dict[str, Any]:
        """Get statistical information for a specific element (reaction or gene).

        Args:
            element_id (str): ID of the element (reaction or gene)
            for_genes (bool): Whether to get gene info (True) or reaction info (False)

        Returns:
            Dict[str, Any]: Statistical information

        Raises:
            ValueError: If test hasn't been run yet
        """
        # Get appropriate test results based on element type
        if for_genes:
            results = self._gene_test_results
        else:
            results = self._reaction_test_results

        if results is None:
            raise ValueError(
                "No test results found. Run the test first with run_test()."
            )

        # Use StatisticalAnalysis to get element statistical info
        return self.stats_analyzer.get_element_statistical_info(
            results, element_id, self.group_id_1, self.group_id_2
        )

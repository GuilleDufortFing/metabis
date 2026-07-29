"""The typed input one graph calculation runs against.

Graph enrichment used to take "a data manager" -- in practice an
``AnalysisWorkspace``, and in the server's case one the graph worker
*reconstructed* per job: eight steps including six hand-assigned revision
counters written by server code, a re-created set of Sample Groups, a
``deepcopy`` of the colour manager, and an explicit rewiring of the hover
manager (AM10 inventory, part 2). None of that is what a graph needs. It is what
a graph needed *because* the parameter's type was a workspace.

``GraphCalculationInput`` is that parameter's real type. It names the ten
collaborator surfaces enrichment actually reads -- measured, not guessed -- and
nothing else. It carries no revision counters, no artifact installation seam, no
source-identity cache, and no way to write scientific state back. A caller that
holds one can render a graph; it cannot advance a dataset revision or install a
result, because those are not on it.

Two producers build one:

* :meth:`GraphCalculationInput.from_workspace` captures a live workspace's own
  collaborators on the document thread. Same objects, same thread, same
  behaviour -- this is the direct-library path, and it exists so the typed seam
  can land without changing what a library caller sees.
* :meth:`GraphCalculationInput.from_plain_values` builds the collaborators
  itself from plain artifacts and frames, with the input as their context. No
  workspace exists anywhere on that path, which is what M-12 asks for and what
  lets the server hand a worker plain values instead of rebuilding a workspace.

Both are Bokeh-free: the collaborators (``DataProvider``, ``HoverManager``,
``SubsystemAnalysis``, ``ColorDistributionRegistry``) import no Bokeh, which is
what makes M-12 (A') work -- hover and the data-provider lookups, the 46% and
3% of a render, stay on the worker over plain values, while the only
Bokeh-carrying surface (palette *generation*) never crosses the seam at all.

See ``MetabisReview/architecture-migration/am10-inventory.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from metabis.analysis.activity_analysis import (
    ActivityArtifact,
    ActivityConfig,
    ActivityOverlay,
)
from metabis.analysis.experiment_analysis import (
    ExperimentAnalysis,
    ExperimentSummarySnapshot,
)
from metabis.analysis.reaction_direction import DEFAULT_REACTION_DIRECTION_CRITERION
from metabis.analysis.subsystem_analysis import (
    SubsystemAnalysis,
    SubsystemDeltaArtifact,
)
from metabis.analysis_workspace import WorkspaceProvenance
from metabis.managers.cache_manager import CacheManager
from metabis.managers.color_distributions import ColorDistributionRegistry
from metabis.managers.data_provider import DataProvider
from metabis.managers.element_presentation import (
    resolve_display_direction,
    resolve_element_color,
)
from metabis.managers.group_manager import GroupDefinition, GroupManager
from metabis.managers.hover_manager import HoverManager
from metabis.managers.workspace_mode import AnalysisWorkspaceMode
from metabis.viz.set_enrichment_overview import (
    SetEnrichmentOverviewColors,
    build_set_enrichment_overview_colors,
)

DEFAULT_DELTA_CONFIG = {
    "metric": "cohens_d",
    "transform": "sign",
    "numerator_filter": "significant",
    "denominator_filter": "all",
    # The shared default, not a literal: this dict is the *graph worker's*
    # delta config, so a criterion spelled out here silently decided which
    # direction every rendered reaction was drawn, coloured, and hovered in
    # whenever the caller passed no explicit config (WP10/D-70).
    "reaction_direction_criterion": DEFAULT_REACTION_DIRECTION_CRITERION.value,
}


@dataclass(frozen=True)
class GraphCalculationInput:
    """Everything one graph calculation reads, and nothing it does not.

    The dataclass is frozen so enrichment cannot rebind a collaborator mid-run.
    The collaborators themselves are mutable -- a data provider caches, a
    subsystem analysis memoizes, a colour registry accepts the distributions the
    run computes -- and that is deliberate: what makes a worker's copy safe is
    that it *owns* them, not that they are immutable. See
    :meth:`from_plain_values`.
    """

    # --- Scientific inputs ------------------------------------------------
    metabolic_model: Any
    mode: AnalysisWorkspaceMode = AnalysisWorkspaceMode.NO_EXPERIMENT
    reaction_data: Any = None
    expression_data: Any = None
    group_comparison_artifact: Any = None
    activity_artifact: Any = None
    activity_overlay: ActivityOverlay = ActivityOverlay.REACTION
    activity_dual_evidence: bool = False
    experiment_analysis: Any = None
    # Worker-local subsystem/process overview colours when the Step-4 summary is
    # Set Enrichment. Populated by ``prepare_set_enrichment_overview`` and read by
    # subsystem/category glyph enrichment; ``None`` outside an ORA overview run.
    set_enrichment_overview: Optional[SetEnrichmentOverviewColors] = None

    # --- Collaborators ----------------------------------------------------
    group_manager: Any = None
    subsystem_analysis: Any = None
    data_provider: Any = None
    hover_manager: Any = None
    cache_manager: Any = None
    # A ColorDistributionRegistry, or the live ColorManager when captured from a
    # workspace on the document thread. Both answer the same six reads and accept
    # the same registration; neither is asked for a palette.
    color_manager: Any = None

    # --- Display formats --------------------------------------------------
    output_reaction_format: str = "mar_id"
    output_gene_format: str = "geneSymbols"
    output_metabolite_format: str = "metName"

    # --- Identity carried only for collaborators that key caches on it ----
    # These are plain numbers, not a workspace's revision counters: nothing here
    # can advance them, and no freshness decision is made against them. Result
    # freshness belongs to the coordinator that submitted the job (ADR-0002).
    dataset_revision: int = 0
    group_revision: int = 0

    # --- Producers --------------------------------------------------------

    @classmethod
    def from_workspace(
        cls,
        workspace: Any,
        *,
        mode: Optional[AnalysisWorkspaceMode] = None,
    ) -> "GraphCalculationInput":
        """Present a live workspace's own collaborators through the typed seam.

        This is the document-thread, direct-library producer. It deliberately
        shares the workspace's collaborators rather than copying them: on this
        path the caller *is* the document thread, and a copied colour registry
        would silently drop the distributions a render registers -- the live
        workspace would then paint the next render differently from this one.
        Detachment is :meth:`from_plain_values`'s job, where the worker owns
        every collaborator it touches.
        """
        return cls(
            metabolic_model=getattr(workspace, "metabolic_model", None),
            mode=mode if mode is not None else workspace.get_mode(),
            reaction_data=getattr(workspace, "reaction_data", None),
            expression_data=getattr(workspace, "expression_data", None),
            group_comparison_artifact=getattr(
                workspace, "group_comparison_artifact", None
            ),
            activity_artifact=getattr(workspace, "activity_artifact", None),
            activity_overlay=getattr(
                workspace, "activity_overlay", ActivityOverlay.REACTION
            ),
            experiment_analysis=getattr(workspace, "experiment_analysis", None),
            group_manager=getattr(workspace, "group_manager", None),
            subsystem_analysis=getattr(workspace, "subsystem_analysis", None),
            data_provider=getattr(workspace, "data_provider", None),
            hover_manager=getattr(workspace, "hover_manager", None),
            cache_manager=getattr(workspace, "cache_manager", None),
            color_manager=getattr(workspace, "color_manager", None),
            output_reaction_format=str(
                getattr(workspace, "output_reaction_format", "mar_id")
            ),
            output_gene_format=str(
                getattr(workspace, "output_gene_format", "geneSymbols")
            ),
            output_metabolite_format=str(
                getattr(workspace, "output_metabolite_format", "metName")
            ),
            dataset_revision=int(getattr(workspace, "dataset_revision", 0)),
            group_revision=int(getattr(workspace, "group_revision", 0)),
        )

    @classmethod
    def from_plain_values(
        cls,
        *,
        metabolic_model: Any,
        mode: AnalysisWorkspaceMode = AnalysisWorkspaceMode.NO_EXPERIMENT,
        reaction_data: Any = None,
        expression_data: Any = None,
        group_comparison_artifact: Any = None,
        activity_artifact: Any = None,
        activity_overlay: ActivityOverlay = ActivityOverlay.REACTION,
        color_registry: Optional[ColorDistributionRegistry] = None,
        group_memberships: Sequence[tuple[str, Sequence[str]]] = (),
        all_sample_ids: Optional[Sequence[str]] = None,
        active_group: Optional[str] = None,
        delta_config: Optional[Mapping[str, Any]] = None,
        output_reaction_format: str = "mar_id",
        output_gene_format: str = "geneSymbols",
        output_metabolite_format: str = "metName",
        dataset_revision: int = 0,
        group_revision: int = 0,
    ) -> "GraphCalculationInput":
        """Build the collaborators from plain values, with no workspace anywhere.

        This is the seam the server hands a graph worker. Everything it takes is
        a plain artifact, frame, membership list, or already-resolved colour --
        the caller assembles no managers and assigns no revisions, because the
        library owns what those collaborators are.

        The caller is expected to pass a ``color_registry`` obtained from
        ``ColorManager.detached_color_registry()`` on the document thread. It is
        the worker's own, so the distributions this run computes stay in this
        run.

        ``experiment_analysis`` is not a parameter: it needs this input as its
        context, so a caller that has a summary snapshot restores it against the
        returned value with ``ExperimentAnalysis.from_summary_snapshot`` and
        installs it with :meth:`with_experiment_analysis`.
        """
        group_manager = GroupManager()
        sample_ids = list(all_sample_ids or [])
        for group_id, samples in group_memberships:
            group_manager.define_group(
                group_id,
                list(samples),
                definition=GroupDefinition("custom"),
                all_sample_ids=sample_ids or None,
            )
        if active_group is not None:
            group_manager.set_active_group(active_group)

        registry = color_registry or ColorDistributionRegistry(
            configurations={}, current_mode=mode
        )
        registry.current_mode = mode

        # The collaborators take this input as their context, so it is built
        # first and they are attached afterwards. That is the one place a frozen
        # field is written, and it is written here rather than by any caller.
        calculation_input = cls(
            metabolic_model=metabolic_model,
            mode=mode,
            reaction_data=reaction_data,
            expression_data=expression_data,
            group_comparison_artifact=group_comparison_artifact,
            activity_artifact=activity_artifact,
            activity_overlay=activity_overlay,
            group_manager=group_manager,
            cache_manager=CacheManager(),
            color_manager=registry,
            output_reaction_format=output_reaction_format,
            output_gene_format=output_gene_format,
            output_metabolite_format=output_metabolite_format,
            dataset_revision=dataset_revision,
            group_revision=group_revision,
        )

        subsystem_analysis = SubsystemAnalysis(
            calculation_input,
            default_delta_config=dict(delta_config or DEFAULT_DELTA_CONFIG),
        )
        data_provider = DataProvider(
            metabolic_model=metabolic_model,
            reaction_data=reaction_data,
            expression_data=expression_data,
            group_comparison_artifact=group_comparison_artifact,
            experiment_analysis=None,
            group_manager=group_manager,
            subsystem_analysis=subsystem_analysis,
        )
        data_provider.activity_artifact = activity_artifact
        data_provider.activity_overlay = activity_overlay
        data_provider.activity_dual_evidence = False
        object.__setattr__(calculation_input, "subsystem_analysis", subsystem_analysis)
        object.__setattr__(calculation_input, "data_provider", data_provider)
        object.__setattr__(
            calculation_input,
            "hover_manager",
            HoverManager(
                data_provider=data_provider,
                cache_manager=calculation_input.cache_manager,
                color_manager=registry,
                mode=mode,
            ),
        )
        return calculation_input

    def with_experiment_analysis(self, analysis: Any) -> "GraphCalculationInput":
        """Attach a restored ``ExperimentAnalysis`` to this input, in place.

        The analysis is constructed *against* this input, so it cannot be a
        constructor argument. Returns ``self`` so a producer reads as one
        expression.
        """
        object.__setattr__(self, "experiment_analysis", analysis)
        if self.data_provider is not None:
            self.data_provider.experiment_analysis = analysis
        return self

    def restore_experiment_summary(
        self,
        summary: ExperimentSummarySnapshot,
    ) -> ExperimentAnalysis:
        """Restore worker-local analysis summaries against this input.

        The server captures only the plain summary value.  This method owns the
        collaborator wiring needed by activity scoring and hover generation, so
        a worker caller never reconstructs an Analysis Workspace or reaches into
        the input's provider graph.
        """
        analysis = ExperimentAnalysis.from_summary_snapshot(self, summary)
        self.with_experiment_analysis(analysis)
        analysis.register_presentation_distributions()
        return analysis

    def prepare_activity(
        self,
        config: ActivityConfig,
        *,
        overlay: ActivityOverlay = ActivityOverlay.REACTION,
        provenance: WorkspaceProvenance,
        source_fingerprints: Mapping[str, str],
    ) -> ActivityArtifact:
        """Compute and attach one worker-local activity artifact.

        This is calculation context, not an installation seam: the input and all
        collaborators it changes are owned by one worker.  Installing the
        returned artifact into the live Analysis Workspace remains a separate,
        provenance-checked document-thread operation.
        """
        analysis = self.experiment_analysis
        if analysis is None:
            raise ValueError("Activity graph input has no experiment summary")
        # A comparison-local Target/Baseline population is not a base group: its
        # frozen membership is on the config, so there is no group ID to look up.
        if (
            not config.has_explicit_population
            and config.group_id not in self.get_group_ids()
        ):
            available = ", ".join(map(str, self.get_group_ids())) or "none"
            raise ValueError(
                f"Group '{config.group_id}' does not exist. Available groups: "
                f"{available}"
            )
        if overlay is ActivityOverlay.GENE and self.expression_data is None:
            raise ValueError(
                "The gene-expression overlay needs an expression matrix. Load one, "
                "or select the reaction activity-score overlay."
            )

        existing = self.activity_artifact
        if existing is not None and existing.config == config:
            if not existing.has_family(overlay):
                raise ValueError(
                    f"{overlay.value.title()} activity evidence is unavailable"
                )
            object.__setattr__(self, "activity_overlay", overlay)
            self.data_provider.activity_artifact = existing
            self.data_provider.activity_overlay = overlay
            self._set_active_base_group(existing.config)
            return existing

        context = analysis.compute_activity(config)
        artifact = analysis.build_activity_artifact(
            context,
            provenance=provenance,
            source_fingerprints=source_fingerprints,
        )
        analysis.install_activity_artifact(artifact)
        object.__setattr__(self, "activity_artifact", artifact)
        object.__setattr__(self, "activity_overlay", overlay)
        self.data_provider.activity_artifact = artifact
        self.data_provider.activity_overlay = overlay
        self._set_active_base_group(artifact.config)
        return artifact

    def _set_active_base_group(self, config: ActivityConfig) -> None:
        """Point the group manager at the scored group, when one exists.

        A comparison-local population has no base-group record to activate. Every
        activity reader takes its values from the installed artifact, so the active
        base group is left untouched rather than being set to a label the group
        manager would reject.
        """
        if config.has_explicit_population:
            return
        self.group_manager.set_active_group(config.group_id)

    def prepare_comparison(self) -> SubsystemDeltaArtifact | None:
        """Prepare worker-local comparison caches and return their delta artifact."""
        artifact = self.group_comparison_artifact
        if artifact is None:
            return None
        self.subsystem_analysis.set_group_comparison(artifact)
        self.hover_manager.update_group_comparison(artifact)
        return self.subsystem_analysis.get_delta_artifact()

    def rank_subsystems(self, *, for_genes: bool = False):
        """Return comparison-mode subsystem ranking for one evidence family."""
        return self.subsystem_analysis.rank_subsystems_by_delta_score(
            for_genes=for_genes
        )

    def prepare_set_enrichment_overview(
        self,
        artifact: Any,
    ) -> SetEnrichmentOverviewColors:
        """Attach worker-local subsystem/process overview colours for one ORA family.

        This is calculation context, not an installation seam: the derived colour
        map is owned by this worker and read by subsystem glyph enrichment. The
        signed-log2 distribution is registered into this input's own colour
        registry, so a hover's color-reference histogram reads the same scale the
        glyph was coloured from. The installed :class:`SetEnrichmentArtifact` was
        already transactionally installed on the document thread; nothing here
        mutates live state. Detailed subsystem graphs never call this, so their
        reaction/gene node colours stay independent of the summary method.
        """
        overview = build_set_enrichment_overview_colors(artifact, self.color_manager)
        object.__setattr__(self, "set_enrichment_overview", overview)
        return overview

    # --- The surface graph enrichment reads -------------------------------

    def get_mode(self) -> AnalysisWorkspaceMode:
        """The analysis mode this graph is painted under."""
        return self.mode

    def get_group_comparison_artifact(self) -> Any:
        """The installed comparison artifact, or None outside comparison mode."""
        return self.group_comparison_artifact

    def get_element_color(
        self,
        element_id: str,
        element_type: str,
        metric_name: str = "default",
        use_main_direction: bool = True,
        delta_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Colour for one element, resolved exactly as the workspace resolves it."""
        return resolve_element_color(
            self,
            element_id,
            element_type,
            metric_name=metric_name,
            use_main_direction=use_main_direction,
            delta_options=delta_options,
        )

    def get_display_direction(self, reaction_id: str) -> Optional[str]:
        """Direction shown for a reversible reaction under this context."""
        return resolve_display_direction(self, reaction_id)

    def get_influence_color(
        self, correlation: float, influence: float, max_influence: float = 100.0
    ) -> str:
        """Colour for a gene-influence bar."""
        return self.color_manager.get_influence_color(
            correlation, influence, max_influence
        )

    def get_element_hover(self, *args: Any, **kwargs: Any) -> str:
        """Hover text for one element, built on this thread from plain values.

        Hover is 46% of a graph render and imports no Bokeh, which is why M-12
        (A') leaves it here rather than pre-resolving it on the document thread.
        """
        return self.hover_manager.get_element_hover(*args, **kwargs)

    # --- The surface the library's own collaborators read -----------------

    def get_active_group(self) -> str:
        """The active group ID."""
        return self.group_manager.get_active_group()

    def get_group_samples(self, group_id: str) -> List[str]:
        """Canonical sample IDs in one group."""
        return self.group_manager.get_group_samples(group_id)

    def get_group_ids(self) -> List[str]:
        """Every defined group ID."""
        return self.group_manager.get_group_ids()

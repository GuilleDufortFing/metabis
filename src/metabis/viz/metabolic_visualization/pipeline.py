"""Metabolic Graph Visualization Pipeline.

The pipeline owns the one graph execution: it resolves a request, generates the
structure, enriches it with experimental data, and returns a plain
:class:`~metabis.viz.metabolic_visualization.artifact.GraphArtifact`. It builds no
Bokeh model. Callers that want a plot render the artifact through the Bokeh adapter;
callers that want a table read the same artifact through the export adapter. Nobody
re-implements the execution. See ADR-0006.
"""

import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple

from metabis.analysis.artifacts import (
    AnalysisArtifactMetadata,
    ArtifactDimension,
    ArtifactKind,
)
from metabis.analysis_workspace import WorkspaceProvenance

from .artifact import (
    GraphArtifact,
    GraphColorMeaning,
    GraphRunMetadata,
    PipelinePhaseMetric,
    freeze_auxiliary,
    freeze_node,
    resolve_color_meaning,
    resolve_hover_detail,
)
from .calculation_input import GraphCalculationInput
from .enrichment.enricher import EnrichmentConfig, GraphEnricher
from .enrichment.focus_instance import FocusInstance
from .errors import GraphPipelineError
from .generation.generator import GraphGenerationConfig, MetabolicGraphGenerator
from .generation.utilities.compartment_resources import (
    get_compartment_display_title,
    normalize_compartment_resource_id,
)

if TYPE_CHECKING:
    # Rendering is a Bokeh adapter. Importing it here at runtime would make the
    # module that owns `build_artifact` -- the one graph execution, which runs on
    # a worker thread -- unimportable wherever Bokeh is absent, which is the
    # property ADR-0006 exists to guarantee. `run_full_pipeline` and
    # `_handle_output`, the two display methods, import it at call time instead.
    from .rendering.renderer import RenderingConfig

RUNTIME_LAYOUT_WARNING = "Runtime-generated layout was used; rendering may be slower."


def as_calculation_input(
    source: Any,
    *,
    activity_dual_evidence: bool = False,
) -> GraphCalculationInput | None:
    """Present whatever a caller supplied as the typed graph calculation input.

    Enrichment reads a ``GraphCalculationInput`` and nothing else, so a caller
    that still hands over an ``AnalysisWorkspace`` has it captured here, on the
    calling thread, rather than deeper down where a workspace would reach the
    enrichment implementation (AM10, M-12).
    """
    if source is None:
        return source
    calculation_input = (
        source
        if isinstance(source, GraphCalculationInput)
        else GraphCalculationInput.from_workspace(source)
    )
    if calculation_input.activity_dual_evidence == activity_dual_evidence:
        return calculation_input
    return replace(
        calculation_input,
        activity_dual_evidence=activity_dual_evidence,
    )


@dataclass(frozen=True)
class ResolvedGraphRequest:
    """Resolved graph request used by generation and enrichment."""

    generator_element_id: Optional[str]
    display_title: str
    focus_element_id: Optional[str] = None
    focus_element_type: Optional[str] = None


@dataclass
class PipelineRunState:
    """Mutable state captured for the most recent pipeline execution."""

    original_element_name: Optional[str] = None
    resolved_request: Optional[ResolvedGraphRequest] = None
    timings: Dict[str, float] = field(default_factory=dict)
    artifact: GraphArtifact | None = None
    metadata: GraphRunMetadata | None = None


def resolve_graph_request(
    metabolic_model,
    element_name: Optional[str],
    graph_type: str,
) -> ResolvedGraphRequest:
    """Resolve a user graph request into generation and focus identifiers."""
    if graph_type == "compartment":
        resource_id = normalize_compartment_resource_id(element_name)
        return ResolvedGraphRequest(
            generator_element_id=resource_id,
            display_title=get_compartment_display_title(resource_id),
            focus_element_id=resource_id,
            focus_element_type="compartment",
        )

    element_id, display_title = metabolic_model.retrieve_element_id(element_name, graph_type)

    if graph_type == "subsystem":
        return ResolvedGraphRequest(
            generator_element_id=element_id,
            display_title=display_title,
            focus_element_id=element_id,
            focus_element_type="subsystem",
        )

    return ResolvedGraphRequest(
        generator_element_id=element_id,
        display_title=display_title,
    )


@dataclass
class PipelineConfig:
    """Configuration for the metabolic graph visualization pipeline."""

    generation_config: Optional[GraphGenerationConfig] = None
    enrichment_config: Optional[EnrichmentConfig] = None
    rendering_config: Optional["RenderingConfig"] = None
    output_html: Optional[str] = None
    show_plot: bool = True
    enable_timing: bool = False

    def __post_init__(self):
        """Initialize default nested configurations if not provided."""
        if self.generation_config is None:
            self.generation_config = GraphGenerationConfig()

        if self.enrichment_config is None:
            self.enrichment_config = EnrichmentConfig()

        # `rendering_config` is deliberately left as None rather than defaulted
        # to `RenderingConfig()`: constructing one here would import Bokeh at
        # configuration time, in the module that owns `build_artifact`.
        # `render_graph_artifact` applies exactly this default when it receives
        # None, so the rendered result is unchanged.


class MetabolicGraphPipeline:
    """Pipeline for coordinating metabolic graph visualization phases.

    This class provides a unified interface for the three main phases:
    1. Graph Generation - Creating the basic graph structure
    2. Graph Enrichment - Adding data-driven styling and highlighting
    3. Graph Rendering - Creating the final Bokeh visualization
    """

    def __init__(self, metabolic_model, config: Optional[PipelineConfig] = None):
        """Initialize the pipeline.

        Args:
            metabolic_model: The metabolic model to use for graph generation
            config: Pipeline configuration. If None, defaults will be used.
        """
        self.metabolic_model = metabolic_model
        self.config = config or PipelineConfig()

        # Generation and enrichment are the pipeline's own phases. Rendering is not:
        # it is an adapter over the artifact, so the pipeline holds no renderer and
        # cannot construct a Bokeh model.
        self.generator = MetabolicGraphGenerator(
            metabolic_model, self.config.generation_config
        )
        self.enricher = GraphEnricher(self.config.enrichment_config)

        self._plot = None
        self._last_run_state = PipelineRunState()

    def build_artifact(
        self,
        element_name: Optional[str],
        graph_type: str,
        data_manager=None,
        *,
        provenance: WorkspaceProvenance | None = None,
        source_fingerprints: Mapping[str, str] | None = None,
    ) -> GraphArtifact:
        """Resolve, generate, and enrich one graph, and return it as plain data.

        This is the whole graph execution. It runs happily on a worker thread,
        because it constructs no Bokeh model: every decision a renderer or an
        export could need is made here and recorded on the returned artifact.

        When ``provenance`` and ``source_fingerprints`` are both supplied, the
        graph carries portable provenance like every other artifact family, so a
        node audit can state which workspace and which inputs it was drawn from.
        Both arrive as explicit parameters, captured on the document thread from
        the real, loaded workspace that paid for input identity at its load
        writer. ``build_artifact`` never reads identity off ``data_manager``: the
        server hands the graph worker a *detached* ``AnalysisWorkspace`` facade over
        frames the session already owns, and that facade recorded its inputs but
        never went through a load writer, so reading its ``source_fingerprints``
        would SHA-256 the whole reaction matrix, the whole expression matrix, and
        the entire model on every render -- the MR-10 regression ADR-0007 exists
        to prevent. A structure-only graph built with no workspace legitimately
        has no provenance, so its metadata stays ``None``.

        ``data_manager`` may be a ``GraphCalculationInput`` -- the typed seam
        enrichment actually reads -- or an ``AnalysisWorkspace``, which is
        captured into one here, before anything downstream can see a workspace.
        """
        calculation_input = as_calculation_input(
            data_manager,
            activity_dual_evidence=graph_type == "subsystem",
        )
        self._last_run_state = PipelineRunState(original_element_name=element_name)
        started = time.perf_counter()

        try:
            resolved_request = resolve_graph_request(
                self.metabolic_model,
                element_name,
                graph_type,
            )
            focus_instance = FocusInstance(
                self.metabolic_model,
                resolved_request.focus_element_id,
                element_type=resolved_request.focus_element_type,
            )
            self._last_run_state.resolved_request = resolved_request
        except Exception as exc:
            raise GraphPipelineError(
                f"Element '{element_name}' could not be resolved: {exc}",
                phase="resolve",
                element_id=element_name,
            ) from exc
        if self.config.enable_timing:
            self._record_timing("resolve_request", time.perf_counter() - started)

        try:
            glyphs, connections, generated_title = self.generate_graph(
                resolved_request.generator_element_id,
                graph_type,
            )
        except Exception as exc:
            if isinstance(exc, GraphPipelineError):
                raise
            raise GraphPipelineError(
                f"Graph generation failed: {exc}",
                phase="generate",
                resource_path=self.generator.last_resource_path,
                element_id=resolved_request.generator_element_id,
            ) from exc

        provider = (
            getattr(calculation_input, "data_provider", None)
            if calculation_input is not None
            else None
        )
        previous_dual_evidence = (
            getattr(provider, "activity_dual_evidence", False)
            if provider is not None
            else False
        )
        if provider is not None:
            provider.activity_dual_evidence = bool(
                calculation_input.activity_dual_evidence
            )
        try:
            enriched_glyphs, enriched_connections, auxiliary_data = self.enrich_graph(
                glyphs,
                connections,
                focus_instance,
                data_manager=calculation_input,
                graph_type=graph_type,
            )
        except Exception as exc:
            raise GraphPipelineError(
                f"Graph enrichment failed: {exc}",
                phase="enrich",
                resource_path=self.generator.last_resource_path,
                element_id=resolved_request.generator_element_id,
            ) from exc
        finally:
            if provider is not None:
                provider.activity_dual_evidence = previous_dual_evidence

        enrichment = self.config.enrichment_config
        timings = self._last_run_state.timings
        item_counts = {
            "generate_graph": len(glyphs) + len(connections),
            "enrich_graph": len(enriched_glyphs) + len(enriched_connections),
        }
        # Colour meaning, hover level, and warnings are resolved once, because the
        # provenance metadata records the same facts the artifact carries and the
        # two must not disagree about what this run was.
        color = resolve_color_meaning(
            calculation_input, color_metric=enrichment.color_metric
        )
        hover_detail = resolve_hover_detail(
            show_hover_distribution_plots=enrichment.show_hover_distribution_plots,
            show_hover_color_reference=enrichment.show_hover_color_reference,
        )
        warnings = (
            (RUNTIME_LAYOUT_WARNING,) if self.generator.used_runtime_layout else ()
        )
        artifact = GraphArtifact(
            graph_type=graph_type,
            element_id=resolved_request.generator_element_id,
            display_title=resolved_request.display_title or generated_title,
            nodes=tuple(freeze_node(glyph) for glyph in enriched_glyphs),
            edges=tuple(freeze_node(connection) for connection in enriched_connections),
            auxiliary=freeze_auxiliary(auxiliary_data),
            color=color,
            resource_path=self.generator.last_resource_path,
            resource_variant=self.generator.last_resource_variant,
            layout_engine=self.config.generation_config.layout_engine,
            used_runtime_layout=self.generator.used_runtime_layout,
            hover_detail=hover_detail,
            phases=tuple(
                PipelinePhaseMetric(name, elapsed, item_count=item_counts.get(name))
                for name, elapsed in timings.items()
            ),
            warnings=warnings,
            metadata=self._graph_metadata(
                graph_type=graph_type,
                element_id=resolved_request.generator_element_id,
                color=color,
                hover_detail=hover_detail,
                node_count=len(enriched_glyphs),
                edge_count=len(enriched_connections),
                warnings=warnings,
                provenance=provenance,
                source_fingerprints=source_fingerprints,
            ),
        )
        self._last_run_state.artifact = artifact
        return artifact

    def _graph_metadata(
        self,
        *,
        graph_type: str,
        element_id: Optional[str],
        color: GraphColorMeaning,
        hover_detail: str,
        node_count: int,
        edge_count: int,
        warnings: Tuple[str, ...],
        provenance: WorkspaceProvenance | None,
        source_fingerprints: Mapping[str, str] | None,
    ) -> AnalysisArtifactMetadata | None:
        """Project the live workspace's identity onto the graph, if a caller gave it.

        Input identity is *read* from the ``source_fingerprints`` the caller
        captured, never recomputed here: hashing a full matrix per render is the
        reverted MR-10 regression (ADR-0007). The method names how the graph was
        painted, the configuration is the generation and enrichment settings the
        run actually used, the scientific identity is the element that was drawn,
        and the dimensions are the node and edge counts -- which
        ``GraphArtifact`` validation requires to match the payload.

        Both parameters are needed to mint provenance; a graph built with no
        workspace has none, and does not invent one from an arbitrary manager.
        """
        if provenance is None or source_fingerprints is None:
            return None

        generation = self.config.generation_config
        enrichment = self.config.enrichment_config
        return AnalysisArtifactMetadata.from_workspace(
            kind=ArtifactKind.GRAPH,
            provenance=provenance,
            source_fingerprints=source_fingerprints,
            method=color.mode,
            configuration={
                "graph_type": graph_type,
                "layout_engine": generation.layout_engine,
                "use_human1_map": bool(generation.use_human1_map),
                "resource_variant": self.generator.last_resource_variant,
                "color_metric": color.metric,
                "color_saturation_limit": (
                    None
                    if color.saturation_limit is None
                    else float(color.saturation_limit)
                ),
                "delta_options": enrichment.delta_options,
                "hover_detail": hover_detail,
                "show_hover_stats": bool(enrichment.show_hover_stats),
                "show_hover_distribution_plots": bool(
                    enrichment.show_hover_distribution_plots
                ),
                "show_hover_color_reference": bool(
                    enrichment.show_hover_color_reference
                ),
                "highlight_significant_reactions": bool(
                    enrichment.highlight_significant_reactions
                ),
                "highlight_significant_genes": bool(
                    enrichment.highlight_significant_genes
                ),
                "highlight_gene_influence": bool(enrichment.highlight_gene_influence),
            },
            scientific_identity={
                "graph_type": graph_type,
                "element_id": element_id,
            },
            dimensions=(
                ArtifactDimension("nodes", node_count),
                ArtifactDimension("edges", edge_count),
            ),
            warnings=warnings,
        )

    def run_full_pipeline(
        self,
        element_name: str,
        graph_type: str,
        data_manager=None,
        *,
        provenance: WorkspaceProvenance | None = None,
        source_fingerprints: Mapping[str, str] | None = None,
    ):
        """Build the graph artifact and render it as a Bokeh plot.

        Args:
            element_name: Name of the subsystem, compartment, or filter to visualize
            graph_type: Type of graph to generate
                       ("subsystem", "compartment", "metabolic_network")
            data_manager: AnalysisWorkspace object with experimental data
            provenance: Live workspace freshness token captured on the document
                thread. Supplied with ``source_fingerprints`` to give the graph
                portable provenance; see ``build_artifact``.
            source_fingerprints: Portable input identity read from the workspace
                that paid for it at load, never recomputed here.

        Returns:
            Bokeh plot object
        """
        # Imported at call time, not module scope: this is the display half, and
        # `build_artifact` above must stay importable without Bokeh (ADR-0006).
        from .rendering.renderer import render_graph_artifact

        artifact = self.build_artifact(
            element_name,
            graph_type,
            data_manager=data_manager,
            provenance=provenance,
            source_fingerprints=source_fingerprints,
        )
        rendered = render_graph_artifact(artifact, self.config.rendering_config)

        self._plot = rendered.plot
        self._last_run_state.metadata = rendered.metadata
        self._handle_output(rendered.plot, self.config)

        return rendered.plot

    def _handle_output(self, plot: Any, config: PipelineConfig) -> None:
        """Write and/or display the rendered plot for a pipeline run."""
        if not (config.output_html or config.show_plot):
            return

        # Call-time import: see `run_full_pipeline`.
        from bokeh.plotting import output_file, show

        if config.output_html:
            output_file(config.output_html)

        if config.show_plot:
            show(plot)

    def generate_graph(
        self,
        element_id: str,
        graph_type: str,
    ) -> Tuple[Any, Any, str]:
        """Phase 1: Generate the basic graph structure.

        Args:
            element_id: Resolved identifier or filter used for generation
            graph_type: Type of graph to generate ("subsystem" or "pathway")

        Returns:
            Tuple of (glyphs, connections, display_title)
        """
        config = self.config

        if config.enable_timing:
            t_start = time.perf_counter()

        glyphs, connections, display_title = self.generator.create_graph(element_id, graph_type)

        # Store intermediate results
        self._raw_graph_data = (glyphs, connections, display_title)

        if config.enable_timing:
            elapsed = time.perf_counter() - t_start
            self._record_timing("generate_graph", elapsed)

        return glyphs, connections, display_title

    def enrich_graph(
        self,
        glyphs,
        connections,
        focus_instance: FocusInstance,
        data_manager=None,
        graph_type: str = None,
    ) -> Tuple[Any, Any, Any]:
        """Phase 2: Enrich graph with data-driven styling and highlighting.

        Args:
            glyphs: Raw glyph data from generation phase
            connections: Raw connection data from generation phase
            focus_instance: object that retrieves the IDs of the elements being visualized
            data_manager: AnalysisWorkspace object with experimental data
            graph_type: Type of graph being generated ("subsystem", "compartment", etc.)

        Returns:
            Tuple of (enriched_glyphs, enriched_connections, auxiliary_data)
        """
        config = self.config

        if config.enable_timing:
            t_start = time.perf_counter()

        enriched_glyphs, enriched_connections, auxiliary_data = self.enricher.enrich(
            glyphs=glyphs,
            connections=connections,
            focus_instance=focus_instance,
            data_manager=data_manager,
            config_override=config.enrichment_config,
            graph_type=graph_type,
        )

        # Store intermediate results
        self._enriched_graph_data = (
            enriched_glyphs,
            enriched_connections,
            auxiliary_data,
        )

        if config.enable_timing:
            elapsed = time.perf_counter() - t_start
            self._record_timing("enrich_graph", elapsed)

        return enriched_glyphs, enriched_connections, auxiliary_data

    def get_timings(self) -> Dict[str, float]:
        """Get timing information for pipeline phases.

        Returns:
            Dictionary of phase names to execution times in seconds
        """
        if self._last_run_state is None:
            return {}

        return dict(self._last_run_state.timings)

    def get_run_metadata(self) -> GraphRunMetadata | None:
        """Return telemetry for the last run. Only a rendered run has render timings."""
        return self._last_run_state.metadata if self._last_run_state else None

    def get_artifact(self) -> GraphArtifact | None:
        """Return the artifact the last run produced, rendered or not."""
        return self._last_run_state.artifact if self._last_run_state else None

    def _record_timing(self, phase_name: str, elapsed: float) -> None:
        """Persist a phase timing on the current run state."""
        if self._last_run_state is None:
            self._last_run_state = PipelineRunState()

        self._last_run_state.timings[phase_name] = elapsed


def create_metabolic_graph_pipeline(
    metabolic_model,
    config: Optional[PipelineConfig] = None,
) -> MetabolicGraphPipeline:
    """Create a MetabolicGraphPipeline from a pipeline configuration.

    Args:
        metabolic_model: The metabolic model to use.
        config: Pipeline configuration built from nested config objects. When
            omitted, the pipeline uses default nested configs.

    Returns:
        Configured MetabolicGraphPipeline instance
    """
    return MetabolicGraphPipeline(metabolic_model, config)

"""The plain result of graph analysis.

Everything a viewer or an export needs to know about a graph is decided here, while
the analysis runs, and recorded on the artifact: which nodes and edges exist, where
they sit, what they are painted, what that paint means, what went wrong, and how
long each phase took. An adapter translates those fields into Bokeh models or into
export rows. It decides nothing.

An artifact carries no Bokeh model, no ``AnalysisWorkspace``, and no live analysis object,
and building one imports no Bokeh. That is what lets the same result be rendered on
a document thread, written to a file, and asserted in a test. See ADR-0006.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd

from metabis.analysis.artifacts import AnalysisArtifactMetadata, ArtifactKind

GraphNode = Mapping[str, Any]
GraphEdge = Mapping[str, Any]

STRUCTURAL_MODE = "structural"


@dataclass(frozen=True)
class PipelinePhaseMetric:
    """How long one phase of graph analysis or rendering took."""

    name: str
    elapsed_seconds: float
    item_count: int | None = None
    output_bytes: int | None = None


@dataclass(frozen=True)
class GraphColorMeaning:
    """What a node's colour on this artifact means.

    Carried with the graph so an export can state the value-to-colour mapping
    without asking a widget what mode the screen happened to be in.
    """

    mode: str
    metric: str
    saturation_limit: float | None = None

    @property
    def is_structural(self) -> bool:
        """True when no experimental data coloured this graph."""
        return self.mode == STRUCTURAL_MODE


@dataclass(frozen=True)
class GraphArtifact:
    """A generated and enriched graph, with no renderer attached."""

    graph_type: str
    element_id: str | None
    display_title: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    auxiliary: Mapping[str, Mapping[str, Any]]
    color: GraphColorMeaning
    resource_path: str | None
    resource_variant: str
    layout_engine: str
    used_runtime_layout: bool
    hover_detail: str
    phases: tuple[PipelinePhaseMetric, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: AnalysisArtifactMetadata | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(freeze_node(node) for node in self.nodes))
        object.__setattr__(self, "edges", tuple(freeze_node(edge) for edge in self.edges))
        object.__setattr__(self, "auxiliary", freeze_auxiliary(self.auxiliary))
        if self.metadata is not None:
            if not isinstance(self.metadata, AnalysisArtifactMetadata):
                raise TypeError("metadata must be AnalysisArtifactMetadata or None")
            if self.metadata.kind is not ArtifactKind.GRAPH:
                raise ValueError("GraphArtifact metadata must have graph kind")
            dimensions = {
                dimension.name: dimension.size
                for dimension in self.metadata.dimensions
            }
            expected = {"nodes": self.node_count, "edges": self.edge_count}
            if dimensions != expected:
                raise ValueError(
                    "Graph artifact dimensions do not match its node and edge payload"
                )

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def node_ids(self) -> tuple[str, ...]:
        return tuple(str(node["id"]) for node in self.nodes)

    def nodes_of_class(self, glyph_class: str) -> tuple[GraphNode, ...]:
        return tuple(
            node for node in self.nodes if node.get("glyph_class") == glyph_class
        )


def freeze_node(node: Mapping[str, Any]) -> GraphNode:
    """Return a recursively read-only view of one node or edge."""
    return _deep_freeze(node)


def freeze_auxiliary(
    auxiliary: Mapping[str, Mapping[str, Any]] | None,
) -> Mapping[str, Mapping[str, Any]]:
    """Return a read-only view of the auxiliary overlay geometry."""
    if not auxiliary:
        return MappingProxyType({})
    return MappingProxyType(
        {str(key): _deep_freeze(value) for key, value in auxiliary.items()}
    )


def _deep_freeze(value: Any) -> Any:
    """Freeze built-in mutable containers without changing scalar payloads."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        return _deep_freeze(value.tolist())
    if isinstance(value, (pd.DataFrame, pd.Series, pd.Index)):
        raise TypeError(
            "GraphArtifact geometry cannot retain mutable pandas containers"
        )
    if isinstance(value, bytearray):
        return bytes(value)
    return value


def resolve_hover_detail(
    *,
    show_hover_distribution_plots: bool,
    show_hover_color_reference: bool,
) -> str:
    """Name the hover level the enrichment flags add up to."""
    if show_hover_distribution_plots:
        return "full"
    if show_hover_color_reference:
        return "medium"
    return "light"


def resolve_color_meaning(
    data_manager: Any,
    *,
    color_metric: str,
) -> GraphColorMeaning:
    """Record what this graph's colours encode, and where the palette saturates.

    A workspace with no experimental data colours nothing, and says so, rather than
    claiming a metric it never applied.
    """
    mode = getattr(data_manager, "mode", None)
    mode_value = getattr(mode, "value", None)
    if data_manager is None or mode_value is None or mode_value == "no_experiment":
        return GraphColorMeaning(mode=STRUCTURAL_MODE, metric=color_metric)

    color_manager = getattr(data_manager, "color_manager", None)
    saturation_limit = None
    if color_manager is not None:
        element_type = "reaction"
        artifact = getattr(data_manager, "activity_artifact", None)
        if artifact is not None and mode_value == "experiment_analysis":
            overlay = getattr(data_manager, "activity_overlay", "reaction")
            element_type = getattr(overlay, "value", str(overlay))
        try:
            saturation_limit = color_manager.get_color_limit(
                mode, element_type, color_metric
            )
        except (AttributeError, KeyError, TypeError):
            saturation_limit = None

    return GraphColorMeaning(
        mode=mode_value,
        metric=color_metric,
        saturation_limit=saturation_limit,
    )


@dataclass(frozen=True)
class GraphRunMetadata:
    """Telemetry for one graph run, including what rendering it cost.

    ``bokeh_model_count`` is a rendering fact, so it exists only once a renderer has
    actually built the models. An artifact that was never rendered has no count, and
    cannot invent one.
    """

    graph_type: str
    element_id: str | None
    display_title: str
    resource_path: str | None
    resource_variant: str
    layout_engine: str
    used_runtime_layout: bool
    glyph_count: int
    connection_count: int
    bokeh_model_count: int | None
    hover_detail: str
    phases: tuple[PipelinePhaseMetric, ...]
    warnings: tuple[str, ...]

    @classmethod
    def from_artifact(
        cls,
        artifact: GraphArtifact,
        *,
        bokeh_model_count: int | None = None,
        extra_phases: tuple[PipelinePhaseMetric, ...] = (),
    ) -> "GraphRunMetadata":
        return cls(
            graph_type=artifact.graph_type,
            element_id=artifact.element_id,
            display_title=artifact.display_title,
            resource_path=artifact.resource_path,
            resource_variant=artifact.resource_variant,
            layout_engine=artifact.layout_engine,
            used_runtime_layout=artifact.used_runtime_layout,
            glyph_count=artifact.node_count,
            connection_count=artifact.edge_count,
            bokeh_model_count=bokeh_model_count,
            hover_detail=artifact.hover_detail,
            phases=artifact.phases + extra_phases,
            warnings=artifact.warnings,
        )


__all__ = [
    "GraphArtifact",
    "GraphColorMeaning",
    "GraphEdge",
    "GraphNode",
    "PipelinePhaseMetric",
    "GraphRunMetadata",
    "STRUCTURAL_MODE",
    "freeze_auxiliary",
    "freeze_node",
    "resolve_color_meaning",
    "resolve_hover_detail",
]

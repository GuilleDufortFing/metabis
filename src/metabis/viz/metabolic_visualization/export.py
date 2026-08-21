"""Export adapter over a graph artifact.

The audit answers "what was actually drawn, and why is it that colour" for every
node in a rendered graph. It reads the same artifact the renderer reads, so an
exported row cannot disagree with the pixel: neither is derived from the other,
and both are derived from the analysis. Nothing here recomputes science.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pandas as pd

from .artifact import GraphArtifact

AUDIT_COLUMNS = (
    "node_id",
    "glyph_class",
    "label",
    "display_label",
    "original_id",
    "x",
    "y",
    "color",
    "alpha",
    "is_missing_data",
    "graph_type",
    "element_id",
    "resource_path",
    "resource_variant",
    "color_mode",
    "color_metric",
    "color_saturation_limit",
    "display_title",
    "layout_engine",
    "used_runtime_layout",
    "artifact_warnings",
    "artifact_id",
    "workspace_id",
    "dataset_fingerprint",
    "source_fingerprints",
    "raw_metric",
    "score",
    "significance",
    "missing_reason",
    "direction",
)

EDGE_AUDIT_COLUMNS = (
    "edge_index",
    "source_id",
    "target_id",
    "direction",
    "graph_type",
    "element_id",
    "artifact_id",
    "payload_json",
)

AUXILIARY_AUDIT_COLUMNS = (
    "layer",
    "graph_type",
    "element_id",
    "artifact_id",
    "payload_json",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _payload_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def graph_audit_rows(artifact: GraphArtifact) -> tuple[dict[str, Any], ...]:
    """One plain row per node, in the order the graph draws them."""
    metadata = artifact.metadata
    context = {
        "graph_type": artifact.graph_type,
        "element_id": artifact.element_id,
        "resource_path": artifact.resource_path,
        "resource_variant": artifact.resource_variant,
        "color_mode": artifact.color.mode,
        "color_metric": artifact.color.metric,
        "color_saturation_limit": artifact.color.saturation_limit,
        "display_title": artifact.display_title,
        "layout_engine": artifact.layout_engine,
        "used_runtime_layout": artifact.used_runtime_layout,
        "artifact_warnings": "\n".join(artifact.warnings),
        "artifact_id": metadata.artifact_id if metadata is not None else "",
        "workspace_id": metadata.source.workspace_id if metadata is not None else "",
        "dataset_fingerprint": (
            metadata.source.source_fingerprints["dataset"]
            if metadata is not None
            else ""
        ),
        "source_fingerprints": (
            json.dumps(
                dict(metadata.source.source_fingerprints),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            if metadata is not None
            else ""
        ),
    }
    return tuple(
        {
            "node_id": str(node["id"]),
            "glyph_class": str(node.get("glyph_class", "")),
            "label": str(node.get("label", "")),
            "display_label": str(node.get("display_label", node.get("label", ""))),
            # The canonical source identifier, where the glyph has one. A display
            # label never stands in for it.
            "original_id": str(node.get("original_id", "")),
            "x": float(node.get("x", 0.0)),
            "y": float(node.get("y", 0.0)),
            "color": str(node.get("color", "")),
            "alpha": float(node.get("alpha", 1.0)),
            "is_missing_data": bool(node.get("is_missing_data", False)),
            # Scientific values are copied only when enrichment recorded them on
            # the node.  The adapter never parses hover HTML or reruns analysis.
            "raw_metric": node.get("raw_metric"),
            "score": node.get("score"),
            "significance": node.get("significance"),
            "missing_reason": node.get("missing_reason"),
            "direction": node.get("direction"),
            **context,
        }
        for node in artifact.nodes
    )


def graph_audit_frame(artifact: GraphArtifact) -> pd.DataFrame:
    """The node audit as a table, with a stable column order."""
    return pd.DataFrame(list(graph_audit_rows(artifact)), columns=list(AUDIT_COLUMNS))


def graph_edge_frame(artifact: GraphArtifact) -> pd.DataFrame:
    """Every rendered edge in artifact order, retaining its complete payload."""
    artifact_id = artifact.metadata.artifact_id if artifact.metadata is not None else ""
    rows = []
    for index, edge in enumerate(artifact.edges):
        rows.append(
            {
                "edge_index": index,
                "source_id": str(edge.get("source", edge.get("start", ""))),
                "target_id": str(edge.get("target", edge.get("end", ""))),
                "direction": edge.get("direction"),
                "graph_type": artifact.graph_type,
                "element_id": artifact.element_id,
                "artifact_id": artifact_id,
                "payload_json": _payload_json(edge),
            }
        )
    return pd.DataFrame(rows, columns=list(EDGE_AUDIT_COLUMNS))


def graph_auxiliary_frame(artifact: GraphArtifact) -> pd.DataFrame:
    """Every auxiliary overlay layer with its complete immutable geometry payload."""
    artifact_id = artifact.metadata.artifact_id if artifact.metadata is not None else ""
    rows = [
        {
            "layer": str(layer),
            "graph_type": artifact.graph_type,
            "element_id": artifact.element_id,
            "artifact_id": artifact_id,
            "payload_json": _payload_json(payload),
        }
        for layer, payload in artifact.auxiliary.items()
    ]
    return pd.DataFrame(rows, columns=list(AUXILIARY_AUDIT_COLUMNS))


__all__ = [
    "AUDIT_COLUMNS",
    "AUXILIARY_AUDIT_COLUMNS",
    "EDGE_AUDIT_COLUMNS",
    "graph_audit_frame",
    "graph_audit_rows",
    "graph_auxiliary_frame",
    "graph_edge_frame",
]

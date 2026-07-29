"""Portable identity and provenance shared by analysis artifacts.

``WorkspaceProvenance`` is a live freshness token: it deliberately retains
matrix references and detached group memberships so an Analysis Workspace can
reject stale worker results.  Those process-local values do not belong in a
saved result.  This module projects that token into portable metadata while
leaving each result family free to own its scientific payload.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from metabis.analysis_workspace import (
    RevisionKind,
    WorkspaceProvenance,
    WorkspaceRevisions,
)
from metabis.version import __version__

_MANIFEST_SCHEMA_VERSION = 1
_REVISION_FIELDS = tuple(kind.value for kind in RevisionKind)

def fingerprint_json(value: Any) -> str:
    """Return a deterministic SHA-256 fingerprint for a JSON value.

    Mapping order is canonicalized and non-finite/non-JSON values are rejected,
    using the same value rules as artifact configuration.  The input is never
    changed.
    """
    frozen = _freeze_json(value, path="fingerprint")
    encoded = json.dumps(
        _thaw_json(frozen),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def fingerprint_frame(frame: pd.DataFrame) -> str:
    """Fingerprint a frame by its axes and logical values, invariant to storage.

    Axis values/order/names, column *logical* dtypes, and values participate. A
    sparse column and its numerically identical dense twin produce the **same**
    fingerprint: storage mode (sparse vs dense, the fill value, the stored-vs-
    implicit split) does not participate, only the values a reader would observe.

    This is deliberate (decision M-08). Source identity must not move when the
    same data is loaded in a different storage mode -- a dense CSV and a sparse
    h5ad of the same numbers share a dataset fingerprint, so making a workflow
    sparse-first does not invalidate provenance or caches across the AM5 seam.

    Columns are hashed one at a time, so no full-frame dense copy is ever held.
    Each sparse column is densified *before* hashing because
    ``pandas.util.hash_pandas_object`` is not storage-invariant for every subtype
    (a ``float32`` sparse column hashes differently from its dense twin); densifying
    the single column first makes the hash equal the dense frame's, bounded to one
    column rather than the whole matrix.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    def dense_columns():
        for position in range(frame.shape[1]):
            series = frame.iloc[:, position]
            dtype = series.dtype
            # The logical dtype: a SparseDtype's subtype is what a reader sees,
            # so float64 and Sparse[float64] are the same identity.
            logical_dtype = (
                dtype.subtype if isinstance(dtype, pd.SparseDtype) else dtype
            )
            if isinstance(dtype, pd.SparseDtype):
                series = series.sparse.to_dense()
            yield logical_dtype, series

    return _fingerprint_columns(frame.index, frame.columns, dense_columns())


def fingerprint_expression_matrix(matrix, *, chunk_features: int = 512) -> str:
    """Fingerprint an expression matrix without ever densifying it whole.

    Reproduces :func:`fingerprint_frame` byte for byte for the equivalent dense
    frame (the one ``matrix.to_dense_frame()`` would build), so a sparse load's
    dataset identity equals its dense twin's -- but at one *column block's* dense
    cost, never the whole matrix. Feature columns are densified a bounded block at
    a time through the seam's ``select_features``; each block is hashed and freed
    before the next. This is the AM6 slice-7 identity path for a sparse load, where
    building the full dense frame (or even a wide ``from_spmatrix`` frame, whose
    per-column overhead rivals dense) would defeat the memory win.

    The axes are taken from the seam's ``sample_ids``/``feature_ids``, which equal
    the dense frame's index/columns (``to_df`` builds them from the same labels).
    """
    feature_ids = list(matrix.feature_ids)
    index = pd.Index(matrix.sample_ids)
    columns = pd.Index(feature_ids)

    def dense_columns():
        for start in range(0, len(feature_ids), chunk_features):
            block_ids = feature_ids[start : start + chunk_features]
            block = matrix.select_features(block_ids)
            for feature_id in block_ids:
                column = block[feature_id]
                yield column.dtype, column

    return _fingerprint_columns(index, columns, dense_columns())


def _fingerprint_columns(index: pd.Index, columns: pd.Index, dense_columns) -> str:
    """Shared core: fingerprint a table from its axes and an iterator of dense columns.

    ``dense_columns`` yields ``(logical_dtype, values)`` in column order, each
    ``values`` a dense Series whose logical values are hashed (index excluded).
    Both the frame path and the bounded expression-matrix path go through here, so
    they cannot disagree on the byte layout.
    """
    digest = hashlib.sha256()
    digest.update(b"metabis.dataframe.v2\0")
    _update_index_fingerprint(digest, index, label=b"index")
    _update_index_fingerprint(digest, columns, label=b"columns")
    for position, (logical_dtype, series) in enumerate(dense_columns):
        digest.update(b"column\0")
        digest.update(str(position).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(logical_dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            pd.util.hash_pandas_object(series, index=False, categorize=False)
            .to_numpy(dtype="<u8", copy=False)
            .tobytes(order="C")
        )
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class PlainTable:
    """Immutable, renderer-independent capture of a pandas table."""

    index: tuple[object, ...]
    columns: tuple[object, ...]
    rows: tuple[tuple[object, ...], ...]
    index_name: object = None

    @classmethod
    def capture(cls, frame: pd.DataFrame) -> "PlainTable":
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        return cls(
            index=tuple(_freeze_table_value(value) for value in frame.index),
            columns=tuple(_freeze_table_value(value) for value in frame.columns),
            rows=tuple(
                tuple(_freeze_table_value(value) for value in row)
                for row in frame.itertuples(index=False, name=None)
            ),
            index_name=_freeze_table_value(frame.index.name),
        )

    def to_frame(self) -> pd.DataFrame:
        """Return a detached pandas adapter preserving empty table headers."""
        return pd.DataFrame(
            self.rows,
            index=pd.Index(self.index, name=self.index_name),
            columns=self.columns,
        )


class ArtifactKind(str, Enum):
    """A frozen discriminator for the supported scientific result families."""

    REDUCTION = "reduction"
    GROUP_COMPARISON = "group_comparison"
    ACTIVITY = "activity"
    REACTION_SET_ENRICHMENT = "reaction_set_enrichment"
    SET_ENRICHMENT = "set_enrichment"
    GRAPH = "graph"


@dataclass(frozen=True)
class ArtifactProvenance:
    """Portable projection of an Analysis Workspace freshness token.

    The complete revision vector is retained for audit. ``dependencies`` says
    which revision values participate in this artifact's scientific identity,
    while ``source_fingerprints`` supplies portable input identity. A canonical
    dataset fingerprint is mandatory; named metadata, model, and preprocessing
    identities may accompany it. Stable frame references and captured group
    memberships intentionally do not cross this boundary; result-specific
    artifacts carry canonical IDs or membership when their science requires them.
    """

    workspace_id: str
    revisions: WorkspaceRevisions
    dependencies: frozenset[RevisionKind]
    source_fingerprints: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.workspace_id, str) or not self.workspace_id:
            raise ValueError("Artifact provenance requires a workspace ID")
        if not isinstance(self.revisions, WorkspaceRevisions):
            raise TypeError("revisions must be WorkspaceRevisions")
        for name in _REVISION_FIELDS:
            value = getattr(self.revisions, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Revision '{name}' must be a non-negative integer")
        if not isinstance(self.dependencies, frozenset) or not all(
            isinstance(kind, RevisionKind) for kind in self.dependencies
        ):
            raise TypeError("dependencies must be a frozenset of RevisionKind values")
        fingerprints = _freeze_source_fingerprints(self.source_fingerprints)
        object.__setattr__(self, "source_fingerprints", fingerprints)

    @classmethod
    def from_workspace(
        cls,
        provenance: WorkspaceProvenance,
        *,
        source_fingerprints: Mapping[str, str],
    ) -> ArtifactProvenance:
        """Detach the portable fields from ``provenance``."""
        if not isinstance(provenance, WorkspaceProvenance):
            raise TypeError("provenance must be WorkspaceProvenance")
        return cls(
            workspace_id=provenance.workspace_id,
            revisions=provenance.revisions,
            dependencies=provenance.dependencies,
            source_fingerprints=source_fingerprints,
        )

    def matches_live(self, provenance: WorkspaceProvenance) -> bool:
        """Is this artifact still a current projection of the live workspace?

        Freshness is judged by *this artifact's* declared dependencies against
        the live revision vector, which is the rule ``require_current`` and
        ``artifact_id`` already use. Two things follow, and both are load-bearing:

        A counter the artifact does not depend on may move without making it
        stale -- applying a comparison threshold advances the threshold revision
        beside an installed reduction, and the reduction is no less current for it.

        The token's own dependency set is not compared, because it belongs to the
        caller's workflow, not to this artifact. A PCA computed by the group
        analysis declares that workflow's dependencies; the companion plots that
        replot it capture a narrower set. Requiring the two to agree would make
        an artifact installable only by the workflow that happened to build it.
        """
        if not isinstance(provenance, WorkspaceProvenance):
            raise TypeError("provenance must be WorkspaceProvenance")
        if self.workspace_id != provenance.workspace_id:
            return False
        return all(
            self.revisions.value_for(kind) == provenance.revisions.value_for(kind)
            for kind in self.dependencies
        )

    def to_manifest(self) -> dict[str, Any]:
        """Return a detached, JSON-safe provenance manifest."""
        return {
            "workspace_id": self.workspace_id,
            "revisions": {
                name: getattr(self.revisions, name) for name in _REVISION_FIELDS
            },
            "dependencies": sorted(kind.value for kind in self.dependencies),
            "source_fingerprints": dict(self.source_fingerprints),
        }

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> ArtifactProvenance:
        """Restore and validate portable provenance from a manifest."""
        if not isinstance(manifest, Mapping):
            raise TypeError("Artifact provenance manifest must be a mapping")
        revisions_manifest = _require_mapping(manifest, "revisions")
        if set(revisions_manifest) != set(_REVISION_FIELDS):
            raise ValueError("Artifact provenance must contain all workspace revisions")
        revisions: dict[str, int] = {}
        for name in _REVISION_FIELDS:
            value = revisions_manifest[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Revision '{name}' must be a non-negative integer")
            revisions[name] = value

        dependencies = manifest.get("dependencies")
        if not isinstance(dependencies, list) or not all(
            isinstance(value, str) for value in dependencies
        ):
            raise ValueError("Artifact provenance dependencies must be a JSON list")
        if len(dependencies) != len(set(dependencies)):
            raise ValueError("Artifact provenance dependencies must be unique")
        workspace_id = manifest.get("workspace_id")
        if not isinstance(workspace_id, str):
            raise ValueError("Artifact provenance workspace_id must be a string")
        source_fingerprints = _require_mapping(manifest, "source_fingerprints")
        return cls(
            workspace_id=workspace_id,
            revisions=WorkspaceRevisions(**revisions),
            dependencies=frozenset(RevisionKind(value) for value in dependencies),
            source_fingerprints=source_fingerprints,
        )

    def _identity_manifest(self) -> dict[str, Any]:
        """Return only revision values declared to affect the calculation."""
        return {
            "revisions": {
                kind.value: self.revisions.value_for(kind)
                for kind in sorted(self.dependencies, key=lambda item: item.value)
            },
            "dependencies": sorted(kind.value for kind in self.dependencies),
            "source_fingerprints": dict(self.source_fingerprints),
        }


@dataclass(frozen=True)
class ArtifactDimension:
    """One named result axis or count.

    ``label`` is presentation-only. It survives a manifest round trip but does
    not participate in stable identity; ``name`` is the canonical dimension ID.
    """

    name: str
    size: int
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Artifact dimension names cannot be empty")
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size < 0
        ):
            raise ValueError("Artifact dimension sizes must be non-negative integers")
        if self.label is not None and not isinstance(self.label, str):
            raise TypeError("Artifact dimension labels must be strings or None")

    def to_manifest(self) -> dict[str, Any]:
        """Return a JSON-safe dimension record."""
        return {"name": self.name, "size": self.size, "label": self.label}

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> ArtifactDimension:
        """Restore a dimension record from a manifest."""
        return cls(
            name=manifest["name"],
            size=manifest["size"],
            label=manifest.get("label"),
        )


@dataclass(frozen=True)
class AnalysisArtifactMetadata:
    """Common immutable metadata attached to a typed scientific result.

    Configuration and result-specific scientific identity are restricted to JSON
    values, recursively frozen, and sorted by mapping key. This gives callers a
    deterministic value independent of input dictionary and dimension order.
    """

    kind: ArtifactKind
    source: ArtifactProvenance
    method: str
    configuration: Mapping[str, Any]
    scientific_identity: Mapping[str, Any]
    dimensions: tuple[ArtifactDimension, ...]
    annotations: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    software_version: str = __version__
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ArtifactKind):
            raise TypeError("kind must be an ArtifactKind")
        if not isinstance(self.source, ArtifactProvenance):
            raise TypeError("source must be ArtifactProvenance")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("Analysis artifact methods cannot be empty")
        if not isinstance(self.configuration, Mapping):
            raise TypeError("configuration must be a mapping")
        if not isinstance(self.scientific_identity, Mapping):
            raise TypeError("scientific_identity must be a mapping")
        if not isinstance(self.annotations, Mapping):
            raise TypeError("annotations must be a mapping")

        configuration = _freeze_mapping(self.configuration, path="configuration")
        scientific_identity = _freeze_mapping(
            self.scientific_identity, path="scientific_identity"
        )
        annotations = _freeze_mapping(self.annotations, path="annotations")
        if not isinstance(self.dimensions, tuple):
            raise TypeError("dimensions must be a tuple")
        if not all(isinstance(item, ArtifactDimension) for item in self.dimensions):
            raise TypeError("dimensions must contain ArtifactDimension values")
        dimensions = tuple(sorted(self.dimensions, key=lambda item: item.name))
        dimension_names = [item.name for item in dimensions]
        if len(dimension_names) != len(set(dimension_names)):
            raise ValueError("Artifact dimension names must be unique")
        if not isinstance(self.warnings, tuple) or not all(
            isinstance(warning, str) for warning in self.warnings
        ):
            raise TypeError("warnings must be a tuple of strings")
        if not isinstance(self.software_version, str) or not self.software_version:
            raise ValueError("Analysis artifacts require a software version")

        object.__setattr__(self, "configuration", configuration)
        object.__setattr__(self, "scientific_identity", scientific_identity)
        object.__setattr__(self, "annotations", annotations)
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(self, "artifact_id", self._build_artifact_id())

    @classmethod
    def from_workspace(
        cls,
        *,
        kind: ArtifactKind,
        provenance: WorkspaceProvenance,
        source_fingerprints: Mapping[str, str],
        method: str,
        configuration: Mapping[str, Any],
        scientific_identity: Mapping[str, Any],
        dimensions: Sequence[ArtifactDimension],
        annotations: Mapping[str, Any] | None = None,
        warnings: Sequence[str] = (),
        software_version: str = __version__,
    ) -> AnalysisArtifactMetadata:
        """Build portable artifact metadata from a live workspace token."""
        return cls(
            kind=kind,
            source=ArtifactProvenance.from_workspace(
                provenance,
                source_fingerprints=source_fingerprints,
            ),
            method=method,
            configuration=configuration,
            scientific_identity=scientific_identity,
            dimensions=tuple(dimensions),
            annotations=annotations or {},
            warnings=tuple(warnings),
            software_version=software_version,
        )

    def to_manifest(self) -> dict[str, Any]:
        """Return a detached plain manifest accepted by ``json.dumps``."""
        return {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "artifact_id": self.artifact_id,
            "kind": self.kind.value,
            "source": self.source.to_manifest(),
            "method": self.method,
            "configuration": _thaw_json(self.configuration),
            "scientific_identity": _thaw_json(self.scientific_identity),
            "dimensions": [item.to_manifest() for item in self.dimensions],
            "annotations": _thaw_json(self.annotations),
            "warnings": list(self.warnings),
            "software_version": self.software_version,
        }

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> AnalysisArtifactMetadata:
        """Restore metadata and reject a manifest whose identity was altered."""
        if not isinstance(manifest, Mapping):
            raise TypeError("Analysis artifact manifest must be a mapping")
        schema_version = manifest.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != _MANIFEST_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported analysis artifact manifest schema")
        source_manifest = _require_mapping(manifest, "source")
        configuration = _require_mapping(manifest, "configuration")
        scientific_identity = _require_mapping(manifest, "scientific_identity")
        annotations_value = manifest.get("annotations", {})
        if not isinstance(annotations_value, Mapping):
            raise ValueError("Artifact annotations must be a JSON object")
        dimension_manifests = manifest.get("dimensions")
        warnings = manifest.get("warnings")
        if not isinstance(dimension_manifests, list):
            raise ValueError("Artifact dimensions must be a JSON list")
        if not isinstance(warnings, list) or not all(
            isinstance(warning, str) for warning in warnings
        ):
            raise ValueError("Artifact warnings must be a JSON list")
        if not all(isinstance(item, Mapping) for item in dimension_manifests):
            raise ValueError("Each artifact dimension must be a JSON object")

        kind = manifest.get("kind")
        if not isinstance(kind, str):
            raise ValueError("Artifact kind must be a string")
        method = manifest.get("method")
        if not isinstance(method, str):
            raise ValueError("Artifact method must be a string")
        software_version = manifest.get("software_version")
        if not isinstance(software_version, str):
            raise ValueError("Artifact software_version must be a string")
        artifact_id = manifest.get("artifact_id")
        if not isinstance(artifact_id, str):
            raise ValueError("Artifact artifact_id must be a string")

        restored = cls(
            kind=ArtifactKind(kind),
            source=ArtifactProvenance.from_manifest(source_manifest),
            method=method,
            configuration=configuration,
            scientific_identity=scientific_identity,
            dimensions=tuple(
                ArtifactDimension.from_manifest(item) for item in dimension_manifests
            ),
            annotations=annotations_value,
            warnings=tuple(warnings),
            software_version=software_version,
        )
        if artifact_id != restored.artifact_id:
            raise ValueError(
                "Analysis artifact manifest identity does not match content"
            )
        return restored

    def _build_artifact_id(self) -> str:
        identity = {
            "kind": self.kind.value,
            "source": self.source._identity_manifest(),
            "method": self.method,
            "configuration": _thaw_json(self.configuration),
            "scientific_identity": _thaw_json(self.scientific_identity),
            "dimensions": [
                {"name": item.name, "size": item.size} for item in self.dimensions
            ],
            "software_version": self.software_version,
        }
        encoded = json.dumps(
            identity,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"{self.kind.value}-{hashlib.sha256(encoded).hexdigest()}"


def _freeze_mapping(value: Mapping[str, Any], *, path: str) -> Mapping[str, Any]:
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{path} keys must be strings")
    frozen: dict[str, Any] = {}
    for key in sorted(value):
        frozen[key] = _freeze_json(value[key], path=f"{path}.{key}")
    return MappingProxyType(frozen)


def _freeze_source_fingerprints(value: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("source_fingerprints must be a mapping")
    if not all(isinstance(key, str) and key and key == key.strip() for key in value):
        raise ValueError("Source fingerprint names must be nonempty canonical strings")
    if not all(
        isinstance(fingerprint, str)
        and fingerprint
        and fingerprint == fingerprint.strip()
        for fingerprint in value.values()
    ):
        raise ValueError("Source fingerprints must be nonempty canonical strings")
    if "dataset" not in value:
        raise ValueError("Artifact provenance requires a canonical dataset fingerprint")
    return MappingProxyType({key: value[key] for key in sorted(value)})


def _freeze_json(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        return _freeze_mapping(value, path=path)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} contains a non-JSON value: {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _require_mapping(manifest: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = manifest.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Artifact manifest field '{key}' must be a JSON object")
    return value


def _update_index_fingerprint(
    digest: "hashlib._Hash", index: pd.Index, *, label: bytes
) -> None:
    digest.update(label)
    digest.update(b"\0")
    digest.update(type(index).__name__.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(tuple(index.names)).encode("utf-8"))
    digest.update(b"\0")
    if isinstance(index, pd.MultiIndex):
        digest.update(
            fingerprint_json(
                [
                    [
                        _json_scalar(item)
                        for item in index.get_level_values(level).tolist()
                    ]
                    for level in range(index.nlevels)
                ]
            ).encode("ascii")
        )
    else:
        digest.update(str(index.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            pd.util.hash_pandas_object(
                index, index=False, categorize=False
            ).to_numpy(dtype="<u8", copy=False).tobytes(order="C")
        )


def _json_scalar(value: Any) -> Any:
    """Give pandas/numpy scalar values a stable JSON representation."""
    if value is pd.NA:
        return {"type": "pd.NA"}
    if value is pd.NaT:
        return {"type": "pd.NaT"}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return {"type": "float", "value": "nan"}
        if math.isinf(value):
            return {"type": "float", "value": "inf" if value > 0 else "-inf"}
        return {"type": "float", "value": value}
    if value is None or isinstance(value, (str, bool, int)):
        return {"type": type(value).__name__, "value": value}
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return {"type": type(value).__name__, "value": value.isoformat()}
    return {"type": type(value).__name__, "value": repr(value)}


def _freeze_table_value(value: Any) -> Any:
    """Detach supported scalar/tuple table values without changing precision."""
    if value is None or value is pd.NA or value is pd.NaT:
        return value
    if isinstance(
        value,
        (
            str,
            bytes,
            bool,
            int,
            float,
            complex,
            np.generic,
            pd.Timestamp,
            pd.Timedelta,
        ),
    ):
        return value
    if isinstance(value, tuple):
        return tuple(_freeze_table_value(item) for item in value)
    raise TypeError(
        "PlainTable supports immutable scalar or tuple cells; "
        f"received {type(value).__name__}"
    )


__all__ = [
    "AnalysisArtifactMetadata",
    "ArtifactDimension",
    "ArtifactKind",
    "ArtifactProvenance",
    "PlainTable",
    "fingerprint_frame",
    "fingerprint_json",
]

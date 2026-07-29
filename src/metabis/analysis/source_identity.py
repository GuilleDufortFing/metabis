"""Portable identity of the inputs an Analysis Workspace has loaded.

A source fingerprint identifies an *input*, not a run. It is therefore computed
once, when that input is loaded or replaced, cached here, and read from here by
every artifact derived from it. No analysis run, render, or export may
fingerprint a full expression matrix, a full reaction matrix, or a whole
metabolic model: dataset identity does not change between renders, so computing
it there is answering a question nobody asked. See ADR-0007 and MR-10.

``SourceIdentityCache`` is the one place that rule is enforced. It recomputes a
component only when the object behind it is actually replaced, so re-reading
identity is free.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable

from ..domain.expression_matrix import SparseExpressionMatrix
from .artifacts import (
    fingerprint_expression_matrix,
    fingerprint_frame,
    fingerprint_json,
)

# Named identities folded into the canonical ``dataset`` fingerprint. Sample
# metadata is deliberately excluded: it advances its own revision axis, and a
# dataset does not stop being the same dataset when its labels change.
_DATASET_COMPONENTS = ("model", "reactions", "expression", "embeddings")

_MODEL_COLLECTIONS = (
    "reactions",
    "metabolites",
    "genes",
    "subsystems",
    "compartments",
)


def _fingerprint_expression(data: Any) -> str:
    """Fingerprint expression identity without densifying the whole matrix.

    A sparse load is fingerprinted through the bounded matrix seam
    (``fingerprint_expression_matrix``), one column block at a time, so a large
    sparse dataset is identified at a block's dense cost -- its hash byte-identical
    to the dense twin (M-08), AM6 slice 7. A dense load (or a source without the
    seam) already holds a dense frame, so it is fingerprinted directly.
    """
    matrix = data.matrix() if hasattr(data, "matrix") else None
    if isinstance(matrix, SparseExpressionMatrix):
        return fingerprint_expression_matrix(matrix)
    return fingerprint_frame(data.get_expression_df())


def model_identity(model: Any) -> dict[str, Any]:
    """Return the order-canonical scientific content of a metabolic model.

    Entity collections are sorted by content, so two models that were built in a
    different order but describe the same network share one identity.
    """
    identity: dict[str, Any] = {"class": type(model).__name__}
    for name in _MODEL_COLLECTIONS:
        collection = getattr(model, name, None)
        try:
            values = collection.get_all()
        except AttributeError:
            continue
        identity[name] = sorted(
            (_normalize_model_value(value) for value in values),
            key=fingerprint_json,
        )
    relationships = getattr(model, "subsystem_neighbor_relationships", None)
    if relationships is not None:
        identity["subsystem_neighbor_relationships"] = sorted(
            (_normalize_model_value(value) for value in relationships),
            key=fingerprint_json,
        )
    if "subsystems" not in identity:
        try:
            identity["subsystems"] = sorted(map(str, model.get_subsystems()))
        except AttributeError:
            pass

    get_categories = getattr(model, "get_categories", None)
    if callable(get_categories):
        categories = list(map(str, get_categories()))
        identity["category_order"] = categories
        get_category_color = getattr(model, "get_category_color", None)
        if callable(get_category_color):
            identity["category_colors"] = [
                [category, str(get_category_color(category))]
                for category in categories
            ]
    get_subsystems_by_category = getattr(model, "get_subsystems_by_category", None)
    if callable(get_subsystems_by_category):
        identity["subsystems_by_category"] = {
            str(category): sorted(map(str, subsystems))
            for category, subsystems in get_subsystems_by_category().items()
        }
    return identity


def _normalize_model_value(value: Any) -> Any:
    """Normalize entity content to order-canonical JSON values."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Enum):
        return _normalize_model_value(value.value)
    if is_dataclass(value):
        return {
            item.name: _normalize_model_value(getattr(value, item.name))
            for item in sorted(fields(value), key=lambda item: item.name)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_model_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_normalize_model_value(item) for item in value), key=fingerprint_json
        )
    if isinstance(value, (list, tuple)):
        return [_normalize_model_value(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): _normalize_model_value(item)
            for key, item in sorted(vars(value).items())
            if not str(key).startswith("_")
        }
    return str(value)


class SourceIdentityCache:
    """Fingerprints of the current inputs, computed at most once per input version.

    Recording an input is free; hashing it is not. The two are separate because
    *constructing* a workspace is not the same event as *loading* data into one.
    The server builds a detached ``AnalysisWorkspace`` per graph job as a worker facade
    over frames the session already owns, so a workspace constructor that hashed
    its inputs would put a full-matrix hash on every render -- the very
    regression ADR-0007 exists to prevent.

    So: the dataset writers ``warm()`` the cache, because a file arriving is a
    load and its identity is genuinely new. A facade over already-identified
    frames simply never reads ``fingerprints``, and pays nothing.
    """

    def __init__(self) -> None:
        # Current inputs: name -> (object, how to fingerprint it). Cheap to hold.
        self._sources: dict[str, tuple[Any, Callable[[Any], str]]] = {}
        # Already-paid fingerprints: name -> (the exact object hashed, its hash).
        self._computed: dict[str, tuple[Any, str]] = {}

    @property
    def fingerprints(self) -> Mapping[str, str]:
        """Portable input identity, always carrying a canonical ``dataset`` key.

        Any input recorded but not yet hashed is hashed here, once. A caller that
        has gone through a dataset writer finds everything already paid for.
        """
        named = {name: self._fingerprint(name) for name in self._sources}
        dataset = {
            name: named[name] for name in _DATASET_COMPONENTS if name in named
        }
        return MappingProxyType(
            {"dataset": fingerprint_json(dataset), **named}
        )

    def record_dataset(
        self,
        *,
        metabolic_model: Any,
        reaction_data: Any,
        expression_data: Any,
    ) -> None:
        """Note the measured inputs after one dataset transition. Hashes nothing."""
        self._record(
            "model",
            metabolic_model,
            lambda model: fingerprint_json(model_identity(model)),
        )
        self._record(
            "reactions",
            reaction_data,
            lambda data: fingerprint_frame(data.get_reaction_df()),
        )
        self._record(
            "expression",
            expression_data,
            _fingerprint_expression,
        )
        embedding_input = (
            getattr(expression_data, "embedding_input", None)
            if expression_data is not None
            else None
        )
        self._record(
            "embeddings",
            embedding_input,
            lambda value: value.content_fingerprint,
        )

    def record_metadata(self, sample_metadata: Any) -> None:
        """Note the sample metadata after one metadata transition. Hashes nothing."""
        self._record(
            "metadata",
            sample_metadata,
            lambda metadata: fingerprint_frame(metadata.table),
        )

    def identify_dataset(
        self,
        *,
        metabolic_model: Any,
        reaction_data: Any,
        expression_data: Any,
    ) -> None:
        """Record the measured inputs and pay for their identity now.

        This is the dataset load seam, so the cost lands on the load that created
        the new identity rather than on the next analysis or render.
        """
        self.record_dataset(
            metabolic_model=metabolic_model,
            reaction_data=reaction_data,
            expression_data=expression_data,
        )
        self._warm(_DATASET_COMPONENTS)

    def identify_metadata(self, sample_metadata: Any) -> None:
        """Record sample metadata and pay for its identity now.

        Only metadata identity is paid for here. A new metadata table does not
        make the matrices a new dataset, so it must not make anyone re-hash them.
        """
        self.record_metadata(sample_metadata)
        self._warm(("metadata",))

    def _warm(self, names: tuple[str, ...]) -> None:
        for name in names:
            if name in self._sources:
                self._fingerprint(name)

    def _record(
        self,
        name: str,
        source: Any,
        compute: Callable[[Any], str],
    ) -> None:
        if source is None:
            self._sources.pop(name, None)
            self._computed.pop(name, None)
            return
        self._sources[name] = (source, compute)

    def _fingerprint(self, name: str) -> str:
        source, compute = self._sources[name]
        known = self._computed.get(name)
        if known is not None and known[0] is source:
            return known[1]
        fingerprint = compute(source)
        self._computed[name] = (source, fingerprint)
        return fingerprint


__all__ = ["SourceIdentityCache", "model_identity"]

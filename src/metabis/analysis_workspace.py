"""Immutable snapshots and provenance for analysis workspace computations.

The workspace owns revision counters, while snapshots carry stable, read-only
references to large matrices and detached copies of small mutable inputs such as
group memberships.  Matrix references are deliberately not copied: callers must
replace workspace inputs rather than mutate captured frames in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

__all__ = [
    "RevisionKind",
    "ALL_REVISION_KINDS",
    "WorkspaceRevisions",
    "StableFrameReference",
    "StableMatrixReference",
    "WorkspaceProvenance",
    "AnalysisSnapshot",
    "StaleAnalysisResultError",
    "normalize_revision_dependencies",
]


class RevisionKind(str, Enum):
    """A workspace revision dimension that can affect an analysis result."""

    DATASET = "dataset"
    METADATA = "metadata"
    GROUPS = "groups"
    PREPROCESSING = "preprocessing"
    THRESHOLDS = "thresholds"
    VISUALIZATION = "visualization"


ALL_REVISION_KINDS = frozenset(RevisionKind)


@dataclass(frozen=True)
class WorkspaceRevisions:
    """The complete revision vector of an Analysis Workspace."""

    dataset: int = 0
    metadata: int = 0
    groups: int = 0
    preprocessing: int = 0
    thresholds: int = 0
    visualization: int = 0

    def value_for(self, kind: RevisionKind) -> int:
        """Return the counter represented by ``kind``."""
        return int(getattr(self, kind.value))


@dataclass(frozen=True)
class StableFrameReference:
    """A stable, read-only-by-contract reference to a matrix-like frame.

    Capturing a reference is constant-memory.  ``read`` detects replacement of
    the referenced object and shape changes to that object, but cannot detect an
    in-place value edit that preserves shape.  Workspace input APIs must therefore
    replace frames instead of modifying them after capture.
    """

    _frame: Any = field(repr=False, compare=False)
    object_id: int
    shape: tuple[int, int]

    @classmethod
    def capture(cls, frame: Any) -> StableFrameReference:
        """Capture ``frame`` without copying its matrix values."""
        shape = tuple(frame.shape)
        if len(shape) != 2:
            raise ValueError("Analysis input frames must be two-dimensional")
        return cls(frame, id(frame), (int(shape[0]), int(shape[1])))

    def read(self) -> Any:
        """Return the captured frame if its identity and shape are still stable."""
        if not self.matches(self._frame):
            raise StaleAnalysisResultError(
                "An analysis input frame changed after snapshot capture"
            )
        return self._frame

    def matches(self, frame: Any) -> bool:
        """Whether ``frame`` is the same object with the captured shape."""
        if frame is None or id(frame) != self.object_id:
            return False
        try:
            shape = tuple(frame.shape)
        except AttributeError:
            return False
        return shape == self.shape


@dataclass(frozen=True)
class StableMatrixReference:
    """A stable, read-only-by-contract reference to an expression matrix source.

    Where :class:`StableFrameReference` references a dense ``DataFrame``, this
    references the *source* that owns the storage-neutral ``matrix()`` seam (an
    ``ExpressionData``) and reads through it, so a sparse ``AnnData.X`` is never
    densified to capture or read a worker snapshot (AM9).  Capturing is
    constant-memory: it records the source's identity and matrix shape only.

    ``read`` returns the source's live ``ExpressionMatrix`` -- the sparse adapter
    for a sparse load, the dense one otherwise -- and rejects a shape change of
    the captured source.  Replacement of the source (a DATASET transaction
    installs a fresh ``ExpressionData``) is detected by the workspace freshness
    check, which compares this reference's identity against a fresh capture; a
    same-shape in-place edit of the resident ``X`` buffer is *prevented* by the
    workspace freezing those buffers on install (M-11), so it raises at the
    offending site rather than needing detection here.
    """

    _source: Any = field(repr=False, compare=False)
    object_id: int
    shape: tuple[int, int]

    @classmethod
    def capture(cls, source: Any) -> StableMatrixReference:
        """Capture ``source`` without densifying its matrix values."""
        shape = cls._shape_of(source)
        if shape is None:
            raise ValueError(
                "Cannot capture a reference to an unloaded expression matrix"
            )
        return cls(source, id(source), shape)

    def read(self) -> Any:
        """Return the source's storage-neutral matrix if its shape is stable."""
        matrix = self._source.matrix()
        if matrix is None or self._matrix_shape(matrix) != self.shape:
            raise StaleAnalysisResultError(
                "An analysis input matrix changed after snapshot capture"
            )
        return matrix

    @staticmethod
    def _shape_of(source: Any) -> tuple[int, int] | None:
        matrix = source.matrix()
        if matrix is None:
            return None
        return StableMatrixReference._matrix_shape(matrix)

    @staticmethod
    def _matrix_shape(matrix: Any) -> tuple[int, int]:
        return (len(matrix.sample_ids), len(matrix.feature_ids))


@dataclass(frozen=True)
class WorkspaceProvenance:
    """Workspace identity and dependencies carried by a derived result."""

    workspace_id: str
    revisions: WorkspaceRevisions
    dependencies: frozenset[RevisionKind]
    reaction_frame: StableFrameReference | None
    expression_matrix: StableMatrixReference | None
    group_memberships: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        normalized = frozenset(RevisionKind(kind) for kind in self.dependencies)
        object.__setattr__(self, "dependencies", normalized)
        memberships = tuple(
            (str(group_id), tuple(map(str, samples)))
            for group_id, samples in self.group_memberships
        )
        object.__setattr__(self, "group_memberships", memberships)


@dataclass(frozen=True)
class AnalysisSnapshot:
    """Approved immutable worker inputs captured from an Analysis Workspace."""

    provenance: WorkspaceProvenance

    @property
    def reaction_frame(self) -> StableFrameReference | None:
        """Stable reaction matrix reference, when reaction data is loaded."""
        return self.provenance.reaction_frame

    @property
    def expression_matrix(self) -> StableMatrixReference | None:
        """Storage-neutral expression matrix reference, when expression is loaded."""
        return self.provenance.expression_matrix

    @property
    def group_memberships(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Detached group memberships approved for worker computation."""
        return self.provenance.group_memberships

    def get_group_ids(self) -> tuple[str, ...]:
        """Return captured group identifiers in workspace order."""
        return tuple(group_id for group_id, _samples in self.group_memberships)

    def get_group_samples(self, group_id: str) -> tuple[str, ...]:
        """Return captured membership for ``group_id``."""
        for current, samples in self.group_memberships:
            if current == group_id:
                return samples
        raise ValueError(f"Group '{group_id}' not found in analysis snapshot")


class StaleAnalysisResultError(RuntimeError):
    """Raised when a result no longer matches its source Analysis Workspace."""


def normalize_revision_dependencies(
    dependencies: Iterable[RevisionKind | str] | None,
) -> frozenset[RevisionKind]:
    """Normalize a caller dependency declaration to an immutable set."""
    if dependencies is None:
        return ALL_REVISION_KINDS
    return frozenset(RevisionKind(kind) for kind in dependencies)

"""The expression-matrix seam (AM6 slices 2-3).

One interface over an expression matrix's identity and value operations, so an
analysis reads samples, features, a single feature column, or the full dense
block through the same seam whether the underlying storage is a dense frame or a
sparse ``AnnData``. The dense adapter reproduces exactly what the readers do
today -- slice the dense frame -- so introducing the seam changed no behaviour.
The sparse adapter (AM6 slice 3) implements the same interface without
densifying the whole matrix: identity and a single feature column are served off
the sparse storage, and only ``to_dense_frame`` materializes everything. The
numeric analyses (slices 4-5) migrate onto the seam's bounded operations.

Kept out of the package root deliberately (decision M-03): this is an internal
domain seam, not a public export.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "ExpressionMatrix",
    "DenseExpressionMatrix",
    "SparseExpressionMatrix",
    "DenseMaterializationError",
    "freeze_matrix_buffers",
    "is_sparse",
]


def is_sparse(matrix_x) -> bool:
    """True when an ``AnnData.X`` is a sparse matrix that needs densifying to read.

    Duck-typed to match the load path's convention (``expression_data.build``
    probes ``.data``/``.nnz`` rather than importing scipy): a scipy sparse matrix
    exposes ``toarray`` and ``nnz``; a dense ``ndarray`` exposes neither.
    """
    return hasattr(matrix_x, "toarray") and hasattr(matrix_x, "nnz")


# The value-buffer attribute names across the scipy sparse formats an h5ad load
# can yield: CSR/CSC carry ``data``/``indices``/``indptr``; COO carries
# ``data``/``row``/``col``. Freezing whichever are present covers every format
# without importing scipy or branching on the concrete class.
_SPARSE_BUFFER_ATTRS = ("data", "indices", "indptr", "row", "col")


def freeze_matrix_buffers(matrix_x) -> None:
    """Mark an ``AnnData.X`` matrix's value buffers read-only in place (M-11, AM9).

    The Analysis Workspace is replace-only after AM8: an installed matrix must
    not be edited in place while a worker snapshot references it. M-11 chose
    *prevention* over id+shape detection, so on install the workspace freezes the
    canonical ``X`` buffers here -- a scipy sparse matrix's ``data``/index arrays,
    or a dense ``ndarray`` -- and any same-shape in-place edit then raises a numpy
    ``ValueError`` at its source. This is the sparse expression store the AM9
    snapshot references; it does not (and cannot reliably) freeze a pandas
    ``DataFrame``, whose Copy-on-Write block manager defeats a buffer freeze, so
    dense-frame in-place mutation is forbidden by the static AM9 boundary guard
    instead. Idempotent: re-freezing an already-frozen buffer is a no-op.
    """
    if matrix_x is None:
        return
    if is_sparse(matrix_x):
        for attr in _SPARSE_BUFFER_ATTRS:
            buffer = getattr(matrix_x, attr, None)
            if isinstance(buffer, np.ndarray):
                buffer.flags.writeable = False
    elif isinstance(matrix_x, np.ndarray):
        matrix_x.flags.writeable = False


class DenseMaterializationError(ValueError):
    """A requested dense materialization would exceed its preflight budget (AM6 slice 6).

    Raised from ``to_dense_frame(max_bytes=...)`` *before* the dense block is
    allocated, using the shape/dtype estimate from ``estimate_dense_bytes`` -- so an
    over-budget densification is rejected rather than attempted and left to fail on
    allocation. A ``ValueError`` so the server's existing over-budget path (which
    already raises ``ValueError``) and generic error surfacing are unchanged, while
    a caller that wants the specific rejection can match this type and read the
    ``estimated_bytes``/``max_bytes`` it carries.
    """

    def __init__(self, estimated_bytes: int, max_bytes: int):
        self.estimated_bytes = int(estimated_bytes)
        self.max_bytes = int(max_bytes)
        super().__init__(
            f"Materializing this expression block densely needs about "
            f"{self.estimated_bytes / 1024**2:.1f} MiB; the budget is "
            f"{self.max_bytes / 1024**2:.1f} MiB."
        )


class ExpressionMatrix(ABC):
    """Read-only access to a (samples x features) expression matrix.

    Implementations expose the same operations over different storage. Every
    method returns the same values the dense frame would, so a caller cannot
    tell dense from sparse by its results -- only by its memory cost.

    An optional ``max_bytes`` budget preflight-gates every dense materialization
    the adapter performs -- the whole matrix (``to_dense_frame``) *and* the
    bounded sample/feature blocks (``select_samples``/``select_features``) -- so
    an over-budget block is refused before it is allocated (M-11, AM9). ``None``
    (the direct-library default) means no budget, so AM6's callers are unchanged;
    on the server the budget rides in from
    ``ExpressionData.dense_memory_budget_bytes`` through
    ``ExpressionData.matrix()``, so every worker's bounded block carries it.
    """

    def __init__(self, max_bytes: Optional[int] = None):
        self._max_bytes = max_bytes

    def _reject_if_over_budget(self, estimated_bytes: int) -> None:
        """Refuse a dense materialization whose preflight estimate exceeds the budget.

        The estimate is computed from shape and dtype only, so an over-budget
        block or matrix is rejected *before* it is allocated. With no budget
        configured this is a no-op.
        """
        if self._max_bytes is not None and estimated_bytes > self._max_bytes:
            raise DenseMaterializationError(estimated_bytes, self._max_bytes)

    @property
    @abstractmethod
    def sample_ids(self) -> List[str]:
        """Sample identifiers, in matrix (row) order."""

    @property
    @abstractmethod
    def feature_ids(self) -> List[str]:
        """Feature identifiers, in matrix (column) order."""

    @abstractmethod
    def feature_column(self, feature_id: str) -> Optional[pd.Series]:
        """One feature's values across all samples, or ``None`` if absent.

        The Series is indexed by sample id. A single-column read never needs the
        whole matrix dense, so a sparse adapter can serve it from one column.
        """

    @abstractmethod
    def select_samples(self, sample_ids: Sequence[str]) -> pd.DataFrame:
        """A bounded dense block: the given samples' rows, every feature.

        The rows are exactly ``sample_ids`` in that order; the columns are every
        feature, dense. This is the *bounded* materialization that group
        summaries and the two-group comparison need -- median/MAD have no sparse
        reduction, and the comparison touches only two groups -- so an analysis
        materializes one group (or two) dense, never the whole matrix. The block
        equals ``to_dense_frame().loc[list(sample_ids)]`` value for value; a
        sample absent from the matrix raises, as indexing the dense frame would.
        The block's own byte cost is preflight-gated by the adapter's budget, so
        an over-budget block is refused before allocation (M-11, AM9).
        """

    @abstractmethod
    def feature_variances(self) -> pd.Series:
        """Per-feature sample variance (ddof=1), indexed by feature id, in column order.

        The ranking input the reducers use to pick the most variable features.
        It is sparse-native: the sparse adapter computes each column's variance
        from the sparse column sums and sums-of-squares without densifying, so
        ranking never materializes the whole matrix. Values equal
        ``to_dense_frame().var()`` within floating-point tolerance; the selection
        depends on the ordering they induce, and well-separated variances rank
        identically whichever adapter serves them (AM6 slice 5).
        """

    @abstractmethod
    def select_features(self, feature_ids: Sequence[str]) -> pd.DataFrame:
        """A bounded dense block: every sample, the given features' columns.

        The column analogue of ``select_samples`` and the *bounded*
        materialization the sklearn reducers need -- a top-N variable-feature
        block, N bounded by the feature cap, never the whole matrix. The columns
        are exactly ``feature_ids`` in that order; the rows are every sample. The
        block equals ``to_dense_frame()[list(feature_ids)]`` value for value; a
        feature absent from the matrix raises, as indexing the dense frame would.
        The block's own byte cost is preflight-gated by the adapter's budget, so
        an over-budget block is refused before allocation (M-11, AM9).
        """

    @abstractmethod
    def estimate_dense_bytes(self) -> int:
        """Bytes a full dense materialization would allocate, from shape and dtype.

        Computed from ``(n_samples, n_features)`` and the value itemsize without
        allocating anything, so it can be checked *before* densifying. It equals
        the module-level :func:`metabis.domain.preprocessing.estimate_dense_bytes`
        for the same matrix, which the AM6 characterization pinned as accurate to
        within 0.1% of the realized dense frame -- a trustworthy preflight budget.
        """

    def to_dense_frame(self, max_bytes: Optional[int] = None) -> pd.DataFrame:
        """The full (samples x features) dense frame, optionally preflight-gated.

        The explicit full-densification escape hatch. When ``max_bytes`` is given
        and ``estimate_dense_bytes()`` exceeds it, this raises
        :class:`DenseMaterializationError` *before* allocating the block, so an
        over-budget densification is rejected rather than attempted (AM6 slice 6).
        With no explicit ``max_bytes`` it falls back to the adapter's own budget
        (``None`` for direct-library use, so it densifies unconditionally exactly
        as the earlier slices' callers expect; the server's configured budget on
        the seam otherwise). AM6's bounded slices move analyses onto
        ``select_samples``/``select_features`` so this is called only where the
        whole matrix is genuinely needed dense at once.
        """
        budget = self._max_bytes if max_bytes is None else max_bytes
        if budget is not None:
            estimated = self.estimate_dense_bytes()
            if estimated > budget:
                raise DenseMaterializationError(estimated, budget)
        return self._materialize_dense()

    @abstractmethod
    def _materialize_dense(self) -> pd.DataFrame:
        """Allocate and return the full dense frame, with no budget check."""


class DenseExpressionMatrix(ExpressionMatrix):
    """Adapter over an already-dense ``DataFrame`` -- today's behaviour, verbatim.

    It holds the frame by reference (no copy) and answers every operation by
    indexing it, so results are identical to the pre-seam readers.
    """

    def __init__(self, frame: pd.DataFrame, max_bytes: Optional[int] = None):
        super().__init__(max_bytes)
        self._frame = frame

    def _itemsize(self) -> int:
        cols = self._frame.shape[1]
        return int(
            self._frame.dtypes.iloc[0].itemsize if cols else np.dtype(float).itemsize
        )

    @property
    def sample_ids(self) -> List[str]:
        return self._frame.index.tolist()

    @property
    def feature_ids(self) -> List[str]:
        return self._frame.columns.tolist()

    def feature_column(self, feature_id: str) -> Optional[pd.Series]:
        if feature_id not in self._frame.columns:
            return None
        return self._frame[feature_id]

    def select_samples(self, sample_ids: Sequence[str]) -> pd.DataFrame:
        # Verbatim what the readers do today: slice the dense frame's rows -- but
        # the block (these rows x every feature) is preflight-budgeted first.
        sample_list = list(sample_ids)
        self._reject_if_over_budget(
            len(sample_list) * self._frame.shape[1] * self._itemsize()
        )
        return self._frame.loc[sample_list]

    def feature_variances(self) -> pd.Series:
        # Verbatim what the reduction did today: pandas per-column variance.
        return self._frame.var(numeric_only=True)

    def select_features(self, feature_ids: Sequence[str]) -> pd.DataFrame:
        # Verbatim what the reduction did today: slice the dense frame's columns
        # -- but the block (every sample x these features) is preflight-budgeted.
        feature_list = list(feature_ids)
        self._reject_if_over_budget(
            self._frame.shape[0] * len(feature_list) * self._itemsize()
        )
        return self._frame[feature_list]

    def estimate_dense_bytes(self) -> int:
        rows, cols = self._frame.shape
        return int(rows * cols * self._itemsize())

    def _materialize_dense(self) -> pd.DataFrame:
        # Already dense and already allocated at load; the budget check in the base
        # ``to_dense_frame`` is a no-op guard here (nothing new is allocated).
        return self._frame


class SparseExpressionMatrix(ExpressionMatrix):
    """Adapter over an ``AnnData`` whose ``X`` may be sparse (AM6 slice 3).

    Identity and a single feature column are served without densifying the whole
    matrix. Sample and feature ids read the AnnData's ``obs_names``/``var_names``
    -- the same labels ``to_df`` builds its axes from -- so they never touch
    ``X``. A feature column densifies exactly one column, through a one-column
    ``AnnData`` view's ``to_df``, so its dtype, index, and name match the full
    dense frame's column byte for byte while the whole matrix stays sparse.
    ``to_dense_frame`` is the one operation that materializes everything; AM6's
    later slices move analyses onto bounded operations so it is called only where
    the whole matrix is genuinely needed dense at once.

    It holds the ``AnnData`` by reference (no copy). It also serves a dense ``X``
    correctly -- ``to_df`` handles either storage -- but ``ExpressionData.matrix``
    only routes here when ``X`` is sparse; a dense load keeps the dense adapter.
    """

    def __init__(self, adata, max_bytes: Optional[int] = None):
        super().__init__(max_bytes)
        self._adata = adata

    def _itemsize(self) -> int:
        return int(np.dtype(getattr(self._adata.X, "dtype", float)).itemsize)

    @property
    def sample_ids(self) -> List[str]:
        return self._adata.obs_names.tolist()

    @property
    def feature_ids(self) -> List[str]:
        return self._adata.var_names.tolist()

    def feature_column(self, feature_id: str) -> Optional[pd.Series]:
        if feature_id not in self._adata.var_names:
            return None
        # One-column view -> ``to_df`` densifies a single column, never the whole
        # matrix, and reproduces the full frame's dtype/index/name exactly.
        return self._adata[:, feature_id].to_df()[feature_id]

    def select_samples(self, sample_ids: Sequence[str]) -> pd.DataFrame:
        # Row view -> ``to_df`` densifies only these samples' rows (a group's
        # worth), never the whole matrix, while ``X`` stays sparse. The rows come
        # out in the requested order and the frame's dtype/columns/index match the
        # full dense frame's, so it equals ``to_dense_frame().loc[sample_ids]``.
        # The block (these rows x every feature) is preflight-budgeted first.
        sample_list = list(sample_ids)
        self._reject_if_over_budget(
            len(sample_list) * self._adata.n_vars * self._itemsize()
        )
        return self._adata[sample_list, :].to_df()

    def feature_variances(self) -> pd.Series:
        # Sparse-native: var = (sum(x^2) - sum(x)^2 / n) / (n - 1), from the
        # column sums and sums-of-squares alone. ``X.multiply(X)`` stays sparse
        # and the zero entries contribute to neither sum, so no column or row of
        # the whole matrix is densified. The zeros' ``(0 - mean)^2`` terms fold
        # back in exactly through the algebra. Accumulate in float64 for stability
        # so the ranking matches the dense frame's ``var()`` (AM6 slice 5).
        matrix_x = self._adata.X
        feature_ids = self.feature_ids
        n_samples = matrix_x.shape[0]
        if n_samples < 2:
            # ddof=1 variance is undefined for n < 2; pandas returns NaN, so match it.
            return pd.Series(
                np.full(len(feature_ids), np.nan, dtype=float), index=feature_ids
            )
        column_sum = np.asarray(matrix_x.sum(axis=0, dtype=np.float64)).ravel()
        column_sum_of_squares = np.asarray(
            matrix_x.multiply(matrix_x).sum(axis=0, dtype=np.float64)
        ).ravel()
        variances = (
            column_sum_of_squares - (column_sum**2) / n_samples
        ) / (n_samples - 1)
        return pd.Series(variances, index=feature_ids)

    def select_features(self, feature_ids: Sequence[str]) -> pd.DataFrame:
        # Column view -> ``to_df`` densifies only these features' columns (the
        # top-N block), never the whole matrix, while ``X`` stays sparse. Columns
        # come out in the requested order and the frame's dtype/index match the
        # full dense frame's, so it equals ``to_dense_frame()[feature_ids]``.
        # The block (every sample x these features) is preflight-budgeted first.
        feature_list = list(feature_ids)
        self._reject_if_over_budget(
            self._adata.n_obs * len(feature_list) * self._itemsize()
        )
        return self._adata[:, feature_list].to_df()

    def estimate_dense_bytes(self) -> int:
        # Mirrors ``preprocessing.estimate_dense_bytes(adata)``: n_obs * n_vars *
        # itemsize, reading no values, so an over-budget densification is rejected
        # before ``to_df`` allocates the dense block.
        return int(self._adata.n_obs * self._adata.n_vars * self._itemsize())

    def _materialize_dense(self) -> pd.DataFrame:
        return self._adata.to_df()

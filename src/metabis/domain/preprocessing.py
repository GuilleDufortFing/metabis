"""Typed, auditable preprocessing for reaction and expression matrices."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Mapping, TypeAlias

import numpy as np
import pandas as pd

JSONScalar: TypeAlias = str | int | float | bool | None


class ReactionTransform(str, Enum):
    """Supported reaction-data transforms.

    These are the transforms that map an already-signed, already-comparable
    reaction matrix onto a new scale, and exactly one of them applies.

    :attr:`COMPASS_NEG_LOG` is **not** one of them. It is a supported serialized
    input alias for dataset declarations and stored configuration. Reading
    Compass penalties as activity scores is a *decoding* step -- it turns a
    penalty into a non-negative score and resolves the blocked-reaction sentinel
    -- which leaves a matrix that any of the transforms above can then be applied
    to. The alias is normalized at construction into
    :attr:`ReactionPreprocessingConfig.compass_activity`, with the transform set
    to :attr:`NONE`; downstream code therefore has one canonical two-stage
    representation.
    """

    NONE = "none"
    COMPASS_NEG_LOG = "compass_neg_log"
    SIGNED_CUBEROOT = "signed_cuberoot"
    RANK_PERCENTILE = "rank_percentile"
    QUANTILE = "quantile"


RankAxis: TypeAlias = Literal[
    "within_reaction_across_samples", "within_sample_across_reactions"
]
"""Which way a rank-based transform reads the matrix.

The two rank transforms are *not* variants of each other and neither replaces the
other. :attr:`ReactionTransform.RANK_PERCENTILE` ranks one reaction across the
samples, answering "where does this sample sit for this reaction"; it removes the
scale differences between reactions. :attr:`ReactionTransform.QUANTILE` ranks one
sample across the reactions and maps those ranks onto a common reference
distribution, answering "where does this reaction sit within this sample"; it
removes the scale differences between samples. Recording the axis in the
provenance is what keeps an exported result from being read along the wrong one.
"""


COMPASS_MINIMUM_RANGE = 1e-3
"""Upstream Compass's ``get_reaction_consistencies`` low-range cutoff.

Compass emits a penalty of exactly **0** for a reaction it could not make carry
any flux: with the maximal-flux constraint satisfied by the all-zero solution,
the minimized penalized flux is zero. A zero penalty therefore means *blocked*,
not *free* - the opposite of what ``-log(penalty + 1)`` implies, which would put
those reactions at the very top of the activity scale.

:func:`compass_negative_log` reads that zero as the sentinel it is and pins it to
the activity **floor**, so a blocked reaction that is blocked in every sample is
constant at the floor and this filter drops it, exactly as upstream does.

Note that dropping blocked reactions does *not* change any other reaction's
score. They sit at the top of the unshifted scale and the shift subtracts the
minimum, which the **largest** penalty sets; measured on the bundled Th17 data
the shift is 11.709256 whether or not they are included. The reason to drop them
is that they otherwise occupy a false ceiling of their own.
"""


@dataclass(frozen=True)
class ReactionPreprocessingConfig:
    """Configuration for one explicit reaction preprocessing pipeline.

    The pipeline has two independent stages, applied in this order:

    1. ``compass_activity`` -- decode Compass penalties into non-negative
       activity scores (see :func:`compass_negative_log`), dropping the reactions
       Compass could not make carry flux.
    2. ``transform`` -- one scale transform over whatever the first stage left.

    They are independent because they answer different questions. Stage 1 is
    about what the *numbers mean*: a Compass penalty is not an activity, and only
    the producer of the file knows that it is one. Stage 2 is about what scale to
    compare them on, and the answer does not depend on where the matrix came
    from. Composing them is therefore the normal case, not a special one --
    Compass activity scores can be quantile-normalized, rank-transformed, or
    cube-rooted exactly as any other non-negative matrix can.
    """

    transform: ReactionTransform = ReactionTransform.NONE
    filter_low_variance: bool = False
    minimum_feature_range: float = 0.0
    rank_axis: RankAxis | None = None
    compass_activity: bool = False

    def __post_init__(self) -> None:
        if self.minimum_feature_range < 0:
            raise ValueError("minimum_feature_range must be non-negative")
        # Normalize the supported serialized alias at the input boundary. Dataset
        # declarations and stored configuration may contain ``"compass_neg_log"``,
        # but analysis sees only the canonical two-stage representation.
        if self.transform is ReactionTransform.COMPASS_NEG_LOG:
            object.__setattr__(self, "compass_activity", True)
            object.__setattr__(self, "transform", ReactionTransform.NONE)
        if (
            self.transform is ReactionTransform.RANK_PERCENTILE
            and self.rank_axis
            not in (
                None,
                "within_reaction_across_samples",
            )
        ):
            raise ValueError(
                "rank percentile is defined across samples within reactions"
            )
        if (
            self.transform is ReactionTransform.QUANTILE
            and self.rank_axis
            not in (
                None,
                "within_sample_across_reactions",
            )
        ):
            raise ValueError(
                "quantile normalization is defined across reactions within samples"
            )
        # Each rank transform has exactly one legal axis, so a caller who did not
        # name it still gets it recorded: the provenance of an exported result
        # must say which way the ranking was read, not leave it to be inferred
        # from the transform name.
        if self.rank_axis is None:
            default_axis = {
                ReactionTransform.RANK_PERCENTILE: "within_reaction_across_samples",
                ReactionTransform.QUANTILE: "within_sample_across_reactions",
            }.get(self.transform)
            if default_axis is not None:
                object.__setattr__(self, "rank_axis", default_axis)

        # The Compass decoding is only meaningful together with the upstream
        # low-range filter, so a caller who asked for nothing gets it. Keeping every
        # reaction is still reachable: ask for the filter with a zero threshold,
        # which retains everything, and accept that blocked reactions will then sit
        # constant at the activity floor, contributing no signal to any comparison.
        if (
            self.compass_activity
            and not self.filter_low_variance
            and self.minimum_feature_range == 0.0
        ):
            object.__setattr__(self, "filter_low_variance", True)
            object.__setattr__(self, "minimum_feature_range", COMPASS_MINIMUM_RANGE)


class ExpressionInputState(str, Enum):
    """Declared state of an expression matrix before preprocessing."""

    RAW_COUNTS = "raw_counts"
    NORMALIZED = "normalized"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ExpressionPreprocessingConfig:
    """Independent expression normalization and log-transform settings."""

    normalize_total: bool = False
    log1p: bool = False
    input_state: ExpressionInputState = ExpressionInputState.UNKNOWN
    target_sum: float = 1_000_000.0

    def __post_init__(self) -> None:
        if self.target_sum <= 0:
            raise ValueError("target_sum must be positive")


@dataclass(frozen=True)
class PreprocessingReport:
    """Machine-readable provenance and data-quality summary."""

    method: str
    parameters: Mapping[str, JSONScalar]
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    zeros_before: int
    zeros_after: int
    non_finite_before: int
    non_finite_after: int
    filtered_feature_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    # ``input_storage`` records how the loaded matrix was stored before any
    # densification ("sparse" for a sparse-``X`` h5ad, "dense" otherwise).
    # ``conversions`` names each explicit dense materialization so provenance
    # records not just the numbers but how sparse input became dense.
    input_storage: str = "dense"
    conversions: tuple[str, ...] = ()
    # Features removed *because Compass reports no flux*, as opposed to merely
    # invariant features. A blocked reaction is a result, not an absence, and
    # only this subset may be described that way.
    blocked_feature_ids: tuple[str, ...] = ()

    @property
    def invariant_feature_ids(self) -> tuple[str, ...]:
        """Removed features whose penalty never varied but was not zero."""
        blocked = set(self.blocked_feature_ids)
        return tuple(
            feature for feature in self.filtered_feature_ids if feature not in blocked
        )


def _validate_numeric_finite(frame: pd.DataFrame, *, stage: str) -> None:
    values = frame.to_numpy(dtype=float, copy=False)
    count = int((~np.isfinite(values)).sum())
    if count:
        raise ValueError(f"{stage} contains {count} non-finite value(s)")


def _compass_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply ``-log1p`` to Compass penalties, without the shift.

    A penalty of exactly 0 is Compass's *blocked* sentinel, not a point on the
    penalty scale, so it is not transformed: ``-log1p(0) = 0`` is the **maximum**
    of this scale and would score a reaction that cannot carry flux as the most
    active one in the dataset. It is pinned to the scale's floor instead - the
    value of the largest penalty actually measured - because a blocked reaction
    is the least active, not the most.

    That conflates "cannot carry flux" with "worst observed penalty in this
    dataset", which compresses the effect size of a reaction blocked in only some
    samples. This is the conservative direction and preserves ordering.
    """
    numeric = frame.astype(float)
    negative_penalties = numeric < 0
    if negative_penalties.to_numpy().any():
        count = int(negative_penalties.to_numpy().sum())
        raise ValueError(
            "Compass activity scoring requires non-negative penalty values; "
            f"found {count} negative value(s)"
        )

    scores = -np.log1p(numeric)
    sentinel = numeric == 0
    if sentinel.to_numpy().any():
        measured = scores.to_numpy()[~sentinel.to_numpy()]
        # With nothing measured anywhere there is no scale to sit at the bottom
        # of; every value is the sentinel and they are all equal.
        floor = float(measured.min()) if measured.size else 0.0
        scores = scores.mask(sentinel, floor)
    return scores


def _shift_to_zero(frame: pd.DataFrame) -> pd.DataFrame:
    """Shift so the least active retained value is exactly zero."""
    if not frame.size:
        return frame
    return frame - float(frame.to_numpy().min())


def blocked_compass_reactions(frame: pd.DataFrame) -> tuple[str, ...]:
    """Reactions Compass never penalized, i.e. that it could not make carry flux.

    A penalty of exactly 0 in every sample is the signature of a blocked reaction:
    the LP satisfies the maximal-flux constraint with the all-zero solution, so the
    minimized penalized flux is zero. These are the *least* active reactions, even
    though ``-log(penalty + 1)`` would place them at the top of the scale.
    """
    zero_everywhere = (frame.astype(float) == 0).all(axis=0)
    return tuple(str(feature) for feature in frame.columns[zero_everywhere])


def partially_blocked_compass_reactions(frame: pd.DataFrame) -> tuple[str, ...]:
    """Reactions Compass blocked in some samples but not others.

    ``maximize_reaction`` optimises unit weight on the reaction subject to
    stoichiometry and the medium - expression never enters it - and its result is
    cached by reaction ID alone, shared across samples. Within a single Compass
    run blockedness therefore cannot vary by sample, so a reaction that is zero in
    some samples and positive in others means the matrix was concatenated from
    runs that did not share a medium, model, or Compass version.

    Such a reaction is retained and scored at the activity floor where it is
    blocked, but it is worth reporting: the two halves are not on a common scale.
    """
    numeric = frame.astype(float)
    is_zero = numeric == 0
    mixed = is_zero.any(axis=0) & ~is_zero.all(axis=0)
    return tuple(str(feature) for feature in frame.columns[mixed])


def compass_negative_log(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert Compass penalties to non-negative relative activity scores.

    Compass penalties are transformed with ``-log1p(x)`` and then shifted by the
    dataset-wide minimum, so lower penalties produce higher activity scores and the
    least-active value is zero. A penalty of exactly 0 is Compass's blocked
    sentinel and is pinned to the floor of that scale rather than transformed; see
    :func:`_compass_scores`.

    This is the transform alone. It does **not** drop the reactions blocked in
    every sample, which are constant at the floor and carry no signal;
    :func:`preprocess_reactions` does that, because dropping them changes which
    reactions the shift is computed over and so must happen before it. Prefer that
    entry point.
    """
    result = _shift_to_zero(_compass_scores(frame))
    _validate_numeric_finite(result, stage="Compass activity-score output")
    return result


def signed_cuberoot(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``sign(x) * abs(x) ** (1/3)`` while preserving zeros."""
    numeric = frame.astype(float)
    _validate_numeric_finite(numeric, stage="Signed cube-root input")
    return np.sign(numeric) * np.cbrt(np.abs(numeric))


def rank_percentile_by_feature(frame: pd.DataFrame) -> pd.DataFrame:
    """Percentile-rank each reaction across samples using average ties."""
    numeric = frame.astype(float)
    _validate_numeric_finite(numeric, stage="Rank-percentile input")
    return numeric.rank(axis=0, method="average", pct=True)


def quantile_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Map every sample onto one common reference distribution.

    The reference is the mean of the sorted values across samples: sort each
    sample's reactions, average position by position, and that average is the
    distribution every sample is mapped onto. A value is then replaced by the
    reference entry at its own rank within its sample, so after the transform
    every sample has an identical distribution and only the *ordering* of
    reactions within a sample carries information.

    Tied reactions take the **mean of the reference entries their tie spans**, so
    two equal measurements stay equal afterwards and a sample whose reactions are
    all equal lands on the mean of the whole reference rather than on its middle
    entry. This is the classic treatment; averaging the tied *rank* instead and
    interpolating agrees with it for a two-way tie but drifts for longer ones.

    Ties are also why "every sample ends with the same distribution" is exact only
    on tie-free input: each row is a partition-average of the same reference, so
    what always holds is that every sample ends with the reference's **total**. On
    a Compass penalty matrix, where a large block of reactions is exactly zero,
    that block collapses to one shared value per sample.

    This is the sample-wise counterpart of :func:`rank_percentile_by_feature`, not
    a replacement for it: that one ranks a reaction across samples, this one ranks
    a sample across reactions. See :data:`RankAxis`.

    Raises:
        ValueError: if the frame holds non-finite values, or fewer than two
            samples -- with one sample the "mean across samples" is that sample
            and the transform is a silent no-op rather than a normalization.
    """
    numeric = frame.astype(float)
    _validate_numeric_finite(numeric, stage="Quantile-normalization input")

    if numeric.shape[0] < 2:
        raise ValueError(
            "Quantile normalization needs at least two samples to build a common "
            f"reference distribution; got {numeric.shape[0]}"
        )
    if not numeric.shape[1]:
        return numeric.copy()

    values = numeric.to_numpy()
    reference = np.sort(values, axis=1).mean(axis=0)

    # A tie occupies a *range* of sorted positions, so each tied value takes the
    # mean of the reference over exactly that range. ``min``/``max`` ranks are its
    # first and last position (1-based), and the prefix sums turn the per-value
    # range mean into two lookups instead of a Python loop over every cell.
    first = numeric.rank(axis=1, method="min").to_numpy().astype(int)
    last = numeric.rank(axis=1, method="max").to_numpy().astype(int)
    cumulative = np.concatenate([[0.0], np.cumsum(reference)])
    mapped = (cumulative[last] - cumulative[first - 1]) / (last - first + 1)

    result = pd.DataFrame(mapped, index=numeric.index, columns=numeric.columns)
    _validate_numeric_finite(result, stage="Quantile-normalization output")
    return result


def filter_features_by_range(
    frame: pd.DataFrame, minimum_range: float
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Remove reaction columns whose range is below ``minimum_range``."""
    if minimum_range < 0:
        raise ValueError("minimum_range must be non-negative")
    ranges = frame.max(axis=0) - frame.min(axis=0)
    removed = tuple(str(feature) for feature in frame.columns[ranges < minimum_range])
    return frame.loc[:, ranges >= minimum_range].copy(), removed


def describe_reaction_pipeline(config: ReactionPreprocessingConfig) -> str:
    """Name the whole two-stage pipeline in one provenance-safe string.

    A composed pipeline needs a composed name: a result that was decoded from
    Compass penalties *and then* quantile-normalized is neither
    ``"compass_neg_log"`` nor ``"quantile"``, and recording either alone would let
    an exported file be read on the wrong scale. The two single-stage cases keep
    the names they always had, so existing provenance stays comparable.
    """
    stages = []
    if config.compass_activity:
        stages.append(ReactionTransform.COMPASS_NEG_LOG.value)
    if config.transform is not ReactionTransform.NONE or not stages:
        stages.append(config.transform.value)
    return "+".join(stages)


def preprocess_reactions(
    frame: pd.DataFrame, config: ReactionPreprocessingConfig
) -> tuple[pd.DataFrame, PreprocessingReport]:
    """Apply reaction preprocessing and return its complete provenance report."""
    original_index = frame.index.copy()
    values = frame.to_numpy(dtype=float, copy=False)
    zeros_before = int((values == 0).sum())
    non_finite_before = int((~np.isfinite(values)).sum())
    if non_finite_before:
        raise ValueError(
            f"Reaction input contains {non_finite_before} non-finite value(s)"
        )

    transforms = {
        ReactionTransform.NONE: lambda value: value.astype(float).copy(),
        ReactionTransform.SIGNED_CUBEROOT: signed_cuberoot,
        ReactionTransform.RANK_PERCENTILE: rank_percentile_by_feature,
        ReactionTransform.QUANTILE: quantile_normalize,
    }
    filtered: tuple[str, ...] = ()
    warnings: list[str] = []
    blocked: tuple[str, ...] = ()
    partially_blocked: tuple[str, ...] = ()
    global_offset = 0.0

    # Stage 1: decode Compass penalties into activity scores, if the file holds
    # them. Whatever this leaves is an ordinary non-negative matrix, which stage 2
    # transforms exactly as it would any other.
    decoded = frame
    if config.compass_activity:
        # Filter before shifting, as upstream Compass does, so the shift is
        # computed over exactly the reactions that survive into the result.
        blocked = blocked_compass_reactions(frame)
        partially_blocked = partially_blocked_compass_reactions(frame)
        scores = _compass_scores(frame)
        if config.filter_low_variance:
            scores, filtered = filter_features_by_range(
                scores, config.minimum_feature_range
            )
        # The offset the retained scores actually carry, taken from the values
        # being shifted rather than recomputed from the raw penalties.
        global_offset = -float(scores.to_numpy().min()) if scores.size else 0.0
        decoded = _shift_to_zero(scores)
        _validate_numeric_finite(decoded, stage="Compass activity-score output")

        surviving_blocked = [
            feature for feature in blocked if feature in decoded.columns
        ]
        if surviving_blocked:
            warnings.append(
                f"{len(surviving_blocked)} reaction(s) have a zero Compass penalty in "
                "every sample. Compass reports a zero penalty for a reaction it could "
                "not make carry flux, so these are blocked reactions: they are scored "
                "at the activity floor and are constant, contributing nothing to any "
                "comparison"
            )
        elif blocked:
            warnings.append(
                f"{len(blocked)} blocked reaction(s) with a zero Compass penalty in "
                "every sample were dropped"
            )
        surviving_partial = [
            feature for feature in partially_blocked if feature in decoded.columns
        ]
        if surviving_partial:
            warnings.append(
                f"{len(surviving_partial)} reaction(s) have a zero Compass penalty in "
                "some samples but not others. Blockedness cannot vary by sample within "
                "one Compass run, so this matrix was concatenated across runs; the "
                "blocked samples are scored at the activity floor, which compresses "
                "the reaction's effect size"
            )

    # Stage 2: one scale transform over the decoded matrix.
    result = transforms[config.transform](decoded)
    if config.filter_low_variance and not config.compass_activity:
        # With the Compass stage the filter has already run, on the scores and
        # before the shift, which is where upstream puts it; running it again here
        # would drop by a threshold that means something different on the
        # transformed scale.
        result, filtered = filter_features_by_range(
            result, config.minimum_feature_range
        )

    if not result.index.equals(original_index):
        raise RuntimeError("Reaction preprocessing changed or reordered sample IDs")

    output_values = result.to_numpy(dtype=float, copy=False)
    zeros_after = int((output_values == 0).sum())
    non_finite_after = int((~np.isfinite(output_values)).sum())
    if non_finite_after:
        raise ValueError(
            f"Reaction output contains {non_finite_after} non-finite value(s)"
        )

    parameters = {
        "filter_low_variance": config.filter_low_variance,
        "minimum_feature_range": config.minimum_feature_range,
        "rank_axis": config.rank_axis,
        "compass_activity": config.compass_activity,
    }
    if config.compass_activity:
        parameters["formula"] = "-log(x + 1) - min(-log(x + 1))"
        parameters["zero_penalty"] = "blocked sentinel, pinned to the activity floor"
        parameters["global_offset"] = global_offset
        parameters["blocked_reactions"] = len(blocked)
        parameters["partially_blocked_reactions"] = len(partially_blocked)

    report = PreprocessingReport(
        method=describe_reaction_pipeline(config),
        parameters=parameters,
        input_shape=frame.shape,
        output_shape=result.shape,
        zeros_before=zeros_before,
        zeros_after=zeros_after,
        non_finite_before=non_finite_before,
        non_finite_after=non_finite_after,
        filtered_feature_ids=filtered,
        warnings=tuple(warnings),
        blocked_feature_ids=blocked,
    )
    return result, report


def estimate_dense_bytes(adata, dtype: np.dtype | None = None) -> int:
    """Estimate bytes required to materialize an AnnData matrix densely."""
    resolved_dtype = dtype or getattr(adata.X, "dtype", np.dtype(float))
    return int(adata.n_obs * adata.n_vars * np.dtype(resolved_dtype).itemsize)

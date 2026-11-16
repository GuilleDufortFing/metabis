"""How a matrix is prepared before a reduction draws it as a map.

By default a reduction reads the features as loaded and keeps those of largest
variance, unscaled: right for data on one scale, such as log expression. Fluxes
span orders of magnitude, and a few large ones would then decide the whole map.
Three settings prepare them as Huang et al. prepared METAFlux's fluxes for their
map of TCGA lung tumours (Nat Commun 2023, Fig. 3b; Supplementary Methods 2-3):

* **Signed cube root**, ``sign(x) * |x|^(1/3)``: keeps each flux's direction
  and damps its range, as METAFlux's own tutorial normalizes fluxes.
* **Standardized variance**, Seurat's ``vst`` selection (Stuart et al., Cell
  2019): a local quadratic regression (loess, span 0.3) of log10 variance on
  log10 mean gives each feature the variance expected at its mean; the values,
  standardized by that expected spread and clipped at ``sqrt(n samples)``, are
  ranked by their variance. A feature whose mean is not positive has no log
  mean, so it cannot be ranked, as in Seurat.
* **Scaling**, as Seurat's ``ScaleData``: each feature centred to mean 0, divided
  by its standard deviation and clipped at 10, so every feature weighs the same.
"""

from __future__ import annotations

from typing import Literal, Sequence, Tuple

import numpy as np
import pandas as pd

FeatureTransform = Literal["none", "signed_cuberoot"]
FeatureSelection = Literal["variance", "standardized_variance"]
FEATURE_TRANSFORMS: Tuple[str, ...] = ("none", "signed_cuberoot")
FEATURE_SELECTIONS: Tuple[str, ...] = ("variance", "standardized_variance")
# Seurat's defaults: the loess span of its vst fit, and ScaleData's scale.max.
LOESS_SPAN = 0.3
SCALE_MAX = 10.0
# The loess is evaluated at this many points and interpolated between them, as R's
# loess interpolates its fit by default.
LOESS_POINTS = 100


def is_default(transform: str, selection: str, scale: bool) -> bool:
    """Whether the features are read as loaded: no transform, ranked by variance, unscaled."""
    return transform == "none" and selection == "variance" and not scale


def check_preparation(transform: str, selection: str) -> None:
    if transform not in FEATURE_TRANSFORMS:
        raise ValueError(f"Unknown feature transform {transform!r}; choose one of {FEATURE_TRANSFORMS}")
    if selection not in FEATURE_SELECTIONS:
        raise ValueError(f"Unknown feature selection {selection!r}; choose one of {FEATURE_SELECTIONS}")


def signed_cube_root(frame: pd.DataFrame) -> pd.DataFrame:
    """``sign(x) * |x|^(1/3)``, element by element; zero stays zero."""
    return np.sign(frame) * np.cbrt(frame.abs())


def _loess(x: np.ndarray, y: np.ndarray, span: float = LOESS_SPAN) -> np.ndarray:
    """A local quadratic fit of ``y`` on ``x`` with tricube weights, evaluated at ``x``.

    Each fit point uses the ``span`` share of the data nearest to it, as R's
    ``loess(degree = 2)`` does; the fit is computed on a grid over ``x`` and
    interpolated, as R interpolates its surface.
    """
    n = len(x)
    k = max(3, int(np.ceil(span * n)))
    grid = np.unique(np.quantile(x, np.linspace(0.0, 1.0, min(LOESS_POINTS, n))))
    fitted = np.empty(len(grid))
    for i, at in enumerate(grid):
        distance = np.abs(x - at)
        nearest = np.argpartition(distance, k - 1)[:k]
        reach = distance[nearest].max()
        weights = (1 - (distance[nearest] / reach) ** 3) ** 3 if reach > 0 else np.ones(k)
        centred = x[nearest] - at
        design = np.column_stack([np.ones(k), centred, centred**2])
        root = np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(design * root[:, None], y[nearest] * root, rcond=None)
        fitted[i] = coefficients[0]
    return np.interp(x, grid, fitted)


def standardized_variances(frame: pd.DataFrame) -> pd.Series:
    """Each feature's variance once standardized by the spread expected at its mean.

    Features with no variance, or a mean that is not positive, are left out:
    their mean has no logarithm, so the trend cannot place them.
    """
    mean = frame.mean()
    variance = frame.var()
    rankable = (variance > 0) & (mean > 0)
    if rankable.sum() < 3:
        return pd.Series(dtype=float)
    log_mean = np.log10(mean[rankable].to_numpy(dtype=float))
    log_variance = np.log10(variance[rankable].to_numpy(dtype=float))
    expected_sd = np.sqrt(10 ** _loess(log_mean, log_variance))
    values = frame.loc[:, rankable].to_numpy(dtype=float)
    standardized = (values - mean[rankable].to_numpy(dtype=float)) / expected_sd
    standardized = np.minimum(standardized, np.sqrt(len(frame)))
    return pd.Series(standardized.var(axis=0, ddof=1), index=frame.columns[rankable.to_numpy()])


def scale_features(values: np.ndarray) -> np.ndarray:
    """Centre each column to mean 0 and divide it by its standard deviation, clipped at 10."""
    centred = values - values.mean(axis=0)
    spread = values.std(axis=0, ddof=1)
    spread[spread == 0] = 1.0
    return np.clip(centred / spread, -SCALE_MAX, SCALE_MAX)


def prepare_features(
    frame: pd.DataFrame,
    n_features: int,
    *,
    transform: str = "none",
    selection: str = "variance",
    scale: bool = False,
) -> Tuple[np.ndarray, Sequence[str], str]:
    """The prepared (samples x features) block a reduction fits, its features, and how they were chosen.

    ``n_features`` of 0, or as many as there are, keeps every feature that can
    be ranked.
    """
    check_preparation(transform, selection)
    frame = frame.select_dtypes(include=[np.number])
    if transform == "signed_cuberoot":
        frame = signed_cube_root(frame)
    if selection == "standardized_variance":
        ranking = standardized_variances(frame)
        how = "standardized variance"
    else:
        ranking = frame.var().dropna()
        ranking = ranking[ranking > 0]
        how = "variance"
    if ranking.empty:
        raise ValueError("No feature varies across the samples, so none can be chosen for the map.")
    keep = ranking.index.tolist() if n_features <= 0 or n_features >= len(ranking) else ranking.nlargest(n_features).index.tolist()
    values = np.nan_to_num(frame[keep].to_numpy(dtype=float))
    if scale:
        values = scale_features(values)
    return values, keep, how


__all__ = [
    "FEATURE_SELECTIONS",
    "FEATURE_TRANSFORMS",
    "FeatureSelection",
    "FeatureTransform",
    "check_preparation",
    "is_default",
    "prepare_features",
    "scale_features",
    "signed_cube_root",
    "standardized_variances",
]

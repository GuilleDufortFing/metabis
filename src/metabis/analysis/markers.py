"""Markers: the features higher in one group than in every other.

As the OSCA book's ``scoreMarkers`` summarizes them (Amezquita et al., Nat
Methods 2020, "Marker gene detection"): every pair of groups gets an effect size,
and a group's markers are ranked by the smallest of its effects against the
others, a large value meaning "higher in this group than in all of the others".
The effect is Cohen's d, the difference of the two groups' means over their
average standard deviation, which OSCA names, with the AUC, the best general
choice. No p-value is given: OSCA leaves them out of marker detection, the
pairwise effects describing the groups rather than testing them.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

MARKER_COLUMNS = ("group", "feature_id", "min_cohens_d", "closest_group", "mean", "closest_mean")


def group_means(matrix: pd.DataFrame, group_samples: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Each feature's mean in each group: features (rows) by groups (columns), in order."""
    index = matrix.index.map(str)
    frame = matrix.set_axis(index, axis=0)
    return pd.DataFrame(
        {group: frame.loc[[str(s) for s in samples if str(s) in frame.index]].mean(axis=0) for group, samples in group_samples.items()}
    )


def pairwise_markers(matrix: pd.DataFrame, group_samples: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Every group's markers, ranked by their smallest Cohen's d against the other groups.

    ``matrix`` holds samples (rows) by features (columns); ``group_samples`` the
    samples of each group, in the order the result lists the groups. A feature
    is a group's marker when its smallest d is positive: higher there than in
    every other group. Each group's markers come strongest first, with the group
    the feature is closest to, the mean there, and the mean in the group itself.
    A feature with no spread in a pair is left out of that group's markers.
    """
    if len(group_samples) < 2:
        raise ValueError("Markers need at least two groups")
    frame = matrix.set_axis(matrix.index.map(str), axis=0)
    members = {group: [str(s) for s in samples if str(s) in frame.index] for group, samples in group_samples.items()}
    empty = [group for group, samples in members.items() if not samples]
    if empty:
        raise ValueError(f"No samples for {', '.join(empty)}")
    means = pd.DataFrame({group: frame.loc[samples].mean(axis=0) for group, samples in members.items()})
    variances = pd.DataFrame({group: frame.loc[samples].var(axis=0, ddof=1) for group, samples in members.items()})
    tables = []
    for group in members:
        others = [other for other in members if other != group]
        effects = pd.DataFrame(
            {
                other: (means[group] - means[other])
                / np.sqrt((variances[group] + variances[other]) / 2).replace(0, np.nan)
                for other in others
            }
        )
        complete = effects.dropna()
        smallest = complete.min(axis=1)
        mine = smallest[smallest > 0].sort_values(ascending=False, kind="mergesort")
        closest = complete.loc[mine.index].idxmin(axis=1)
        tables.append(
            pd.DataFrame(
                {
                    "group": group,
                    "feature_id": mine.index.astype(str),
                    "min_cohens_d": mine.to_numpy(dtype=float),
                    "closest_group": closest.to_numpy(),
                    "mean": means.loc[mine.index, group].to_numpy(dtype=float),
                    "closest_mean": [float(means.at[feature, other]) for feature, other in closest.items()],
                }
            )
        )
    return pd.concat(tables, ignore_index=True)[list(MARKER_COLUMNS)]


def top_markers(markers: pd.DataFrame, per_group: int) -> pd.DataFrame:
    """Each group's ``per_group`` strongest markers, groups in their order."""
    if per_group < 1:
        raise ValueError("Show at least one marker per group")
    return markers.groupby("group", sort=False, group_keys=False).head(per_group).reset_index(drop=True)


__all__ = ["MARKER_COLUMNS", "group_means", "pairwise_markers", "top_markers"]

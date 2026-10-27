"""Sample-sheet columns a comparison adjusts for.

Adjusting for a column enters it into the linear model beside the group, so the
group's effect is the difference between groups at equal values of that column
(the "remove unwanted variation" covariate of Leek et al. 2010 and limma's
additive design, Ritchie et al. 2015). Only the moderated t-test fits such a
model; the rank-sum and Welch tests compare raw values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

__all__ = ["AdjustmentColumn", "ComparisonAdjustment"]


@dataclass(frozen=True)
class AdjustmentColumn:
    """One sample-sheet column entered into the comparison's model."""

    name: str
    log10: bool = False

    @property
    def label(self) -> str:
        return f"log10 {self.name}" if self.log10 else self.name


@dataclass(frozen=True)
class ComparisonAdjustment:
    """The columns to adjust for and their values, indexed by sample.

    A numeric column enters as one slope, optionally on a log10 scale, as read
    depth or cell counts usually are. A text column enters as one indicator per
    level after the first, so each level gets its own offset.
    """

    columns: tuple[AdjustmentColumn, ...]
    values: pd.DataFrame

    def __post_init__(self) -> None:
        names = [column.name for column in self.columns]
        if not names:
            raise ValueError("An adjustment needs at least one column")
        if len(set(names)) != len(names):
            raise ValueError("Each column can be adjusted for once")
        missing = [name for name in names if name not in self.values.columns]
        if missing:
            raise ValueError(f"The sample sheet has no column {', '.join(missing)}")

    @classmethod
    def from_metadata(
        cls, metadata: pd.DataFrame, columns: Sequence[AdjustmentColumn]
    ) -> "ComparisonAdjustment":
        """Keep only the adjusted columns of a sample sheet indexed by sample."""
        columns = tuple(columns)
        names = [column.name for column in columns]
        values = metadata.loc[:, [name for name in names if name in metadata.columns]]
        values = values.copy()
        values.index = values.index.map(str)
        # Sample sheets are read as text: a column whose every value reads as a
        # number is a number.
        for name in values.columns:
            column = values[name]
            if is_numeric_dtype(column) or is_bool_dtype(column):
                continue
            numeric = pd.to_numeric(column, errors="coerce")
            if numeric[column.notna()].notna().all() and column.notna().any():
                values[name] = numeric
        return cls(columns=columns, values=values)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(column.label for column in self.columns)

    def design(self, samples: Sequence[str]) -> pd.DataFrame:
        """Numeric model columns for the tested samples, in their order."""
        samples = [str(sample) for sample in samples]
        missing = sorted(set(samples) - set(self.values.index))
        if missing:
            raise ValueError(
                f"The sample sheet has no row for {len(missing)} tested sample(s): "
                f"{', '.join(missing[:5])}"
            )
        blocks: list[pd.Series] = []
        for column in self.columns:
            raw = self.values.loc[samples, column.name]
            if raw.isna().any():
                raise ValueError(
                    f"{column.name} is empty for {int(raw.isna().sum())} tested sample(s)"
                )
            if is_numeric_dtype(raw) and not is_bool_dtype(raw):
                numeric = raw.astype(float)
                if column.log10:
                    if (numeric <= 0).any():
                        raise ValueError(
                            f"log10 {column.name} needs every value above zero"
                        )
                    numeric = np.log10(numeric)
                blocks.append(numeric.rename(column.label))
                continue
            if column.log10:
                raise ValueError(f"{column.name} is not numeric, so it has no log10")
            text = raw.astype(str)
            levels = sorted(text.unique())
            if len(levels) < 2:
                raise ValueError(
                    f"{column.name} has one value across these samples, so it "
                    "cannot be adjusted for"
                )
            for level in levels[1:]:
                blocks.append((text == level).astype(float).rename(f"{column.name}: {level}"))
        return pd.concat(blocks, axis=1)

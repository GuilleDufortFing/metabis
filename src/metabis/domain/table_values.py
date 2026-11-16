"""Plain messages for a matrix whose cells are not all numbers.

A loader reads a matrix whole, so one bad cell would otherwise surface as a NumPy
or pandas error, far from the file and with no hint of where to look. These
checks find the first bad cell and name it by its feature and its sample, which
hold whichever way round the file is written.
"""

from __future__ import annotations

import re
from typing import Hashable

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

# "1,5": a decimal comma, as a spreadsheet in many locales writes a number.
_DECIMAL_COMMA = re.compile(r"[-+]?\d+,\d+")

# The separators a table is read with, as a message names them.
SEPARATOR_NAMES = {"\t": "tabs", ",": "commas", ";": "semicolons"}


def text_cell(frame: pd.DataFrame) -> tuple[Hashable, Hashable, object] | None:
    """The first cell, column by column, holding something other than a number.

    Returns its row label, column label and value; a blank cell is not text (see
    :func:`non_finite_cells`).
    """
    for position in range(frame.shape[1]):
        column = frame.iloc[:, position]
        if is_bool_dtype(column):
            return frame.index[0], frame.columns[position], column.iloc[0]
        if is_numeric_dtype(column):
            continue
        parsed = pd.to_numeric(column, errors="coerce")
        text = column.notna().to_numpy() & parsed.isna().to_numpy()
        if text.any():
            row = int(np.argmax(text))
            return frame.index[row], frame.columns[position], column.iloc[row]
    return None


def describe_text_cell(what: str, feature_noun: str, feature: Hashable, sample: Hashable, value: object) -> str:
    """Say which cell of ``what`` ("The reaction file") is not a number."""
    hint = " Write decimals with a point, not a comma." if _DECIMAL_COMMA.fullmatch(str(value).strip()) else ""
    return f"{what} holds {str(value)!r} for {feature_noun} {feature} in sample {sample}, which is not a number.{hint}"


def non_finite_cells(values: np.ndarray) -> tuple[int, tuple[int, int]] | None:
    """How many cells are blank, NaN or infinite, and the position of the first."""
    bad = ~np.isfinite(values)
    count = int(bad.sum())
    if not count:
        return None
    row, column = np.argwhere(bad)[0]
    return count, (int(row), int(column))


def describe_non_finite(what: str, count: int, feature_noun: str, feature: Hashable, sample: Hashable) -> str:
    """Say how many cells of ``what`` are not finite, and where the first is."""
    cells = "1 cell is" if count == 1 else f"{count:,} cells are"
    first = "It is" if count == 1 else "The first is"
    return (
        f"{what} is incomplete: {cells} blank, NaN or infinite. {first} {feature_noun} {feature} in sample {sample}. "
        "Every cell must hold a number."
    )


def repeated_header_names(path: str, separator: str) -> list[str]:
    """The names a table's header gives more than one column, as the file writes them.

    pandas renames a repeated column (``S1`` and ``S1.1``), so a sample named twice
    along the header would otherwise pass as two samples.
    """
    header = pd.read_csv(path, sep=separator, header=None, nrows=1, dtype=str).iloc[0, 1:]
    names = pd.Index(header.dropna())
    return list(dict.fromkeys(names[names.duplicated()]))


def describe_repeated_samples(what: str, samples: list) -> str:
    """Say that ``what`` names some samples more than once."""
    more = f" (and {len(samples) - 1:,} more)" if len(samples) > 1 else ""
    return f"{what} names sample {samples[0]} more than once{more}. Each sample must appear once."


def renamed_repeats(columns, repeated: list) -> list:
    """The columns pandas renamed from the ``repeated`` names (``S1.1`` for a second ``S1``)."""
    patterns = [re.compile(rf"{re.escape(str(name))}\.\d+") for name in repeated]
    return [column for column in columns if any(pattern.fullmatch(str(column)) for pattern in patterns)]


def describe_repeats(noun: str, names: list) -> str:
    """Say that some ``noun``s appear more than once, and that the first of each was kept."""
    count = len(names)
    shown = ", ".join(map(str, names[:3])) + (", ..." if count > 3 else "")
    many = f"{count:,} {noun}s appear" if count != 1 else f"1 {noun} appears"
    return f"{many} more than once in the file ({shown}): the first of each was kept."

"""Validated, independently loaded sample metadata."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd

# The separators a sample sheet may use, preferred in this order on a tie.
METADATA_SEPARATORS = ("\t", ",", ";")


def _header_line(path: Path) -> str:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", errors="replace") as handle:
        return handle.readline()


def read_metadata_table(path: str | Path, *, separator: str | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Read a sample sheet as text, the same way for its preview and its load.

    A sheet saved by Excel starts with a byte-order mark, which would otherwise
    become part of its first column's name. The separator is the one of tab, comma
    and semicolon its header uses most, so a one-column sheet is not split on a
    letter, and a column name keeps no surrounding spaces.
    """
    path = Path(path)
    if separator is None:
        header = _header_line(path)
        separator = max(METADATA_SEPARATORS, key=lambda candidate: (header.count(candidate), -METADATA_SEPARATORS.index(candidate)))
    frame = pd.read_csv(path, sep=separator, dtype=object, encoding="utf-8-sig", nrows=nrows)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _some(samples: Sequence[str], count: int = 3) -> str:
    shown = ", ".join(list(samples)[:count])
    return shown + (", ..." if len(samples) > count else "")


def _samples(count: int) -> str:
    return "1 sample" if count == 1 else f"{count:,} samples"


@dataclass(frozen=True)
class MetadataAlignmentReport:
    metadata_samples: tuple[str, ...]
    reaction_samples: tuple[str, ...]
    expression_samples: tuple[str, ...]
    aligned_samples: tuple[str, ...]
    missing_from_metadata: tuple[str, ...]
    metadata_only: tuple[str, ...]
    missing_from_expression: tuple[str, ...]
    duplicate_ids: tuple[str, ...] = ()

    @property
    def is_strict_match(self) -> bool:
        return not (
            self.missing_from_metadata
            or self.metadata_only
            or self.missing_from_expression
            or self.duplicate_ids
        )


class SampleMetadata:
    """A metadata table indexed by immutable canonical sample IDs."""

    def __init__(self, table: pd.DataFrame, *, display_label_column: str | None = None):
        if not isinstance(table, pd.DataFrame):
            raise TypeError("metadata table must be a pandas DataFrame")
        frame = table.copy()
        ids = pd.Index(frame.index)
        normalized = ids.map(lambda value: "" if pd.isna(value) else str(value).strip())
        blank = [str(index + 2) for index, value in enumerate(normalized) if not value]
        if blank:
            raise ValueError(f"Sample IDs must be non-empty (rows: {', '.join(blank[:5])})")
        duplicates = tuple(dict.fromkeys(normalized[normalized.duplicated()].tolist()))
        if duplicates:
            raise ValueError(f"Sample IDs must be unique; duplicates: {', '.join(duplicates[:5])}")
        frame.index = normalized
        frame.index.name = ids.name or "sample_id"
        frame.columns = [str(column).strip() for column in frame.columns]
        if len(set(frame.columns)) != len(frame.columns):
            raise ValueError("Metadata column names must be unique")
        self.table = frame
        self.display_label_column: str | None = None
        self.set_display_label_column(display_label_column)

    @staticmethod
    def read_table(path: str | Path, *, separator: str | None = None, nrows: int | None = None) -> pd.DataFrame:
        """Read a sample sheet's rows as text, as :meth:`from_table` reads them (see :func:`read_metadata_table`)."""
        return read_metadata_table(path, separator=separator, nrows=nrows)

    @classmethod
    def from_table(
        cls,
        path: str | Path,
        *,
        separator: str | None = None,
        sample_id_column: str | None = None,
        display_label_column: str | None = None,
    ) -> "SampleMetadata":
        try:
            frame = read_metadata_table(path, separator=separator)
        except pd.errors.EmptyDataError as error:
            raise ValueError("The sample sheet is empty.") from error
        except UnicodeDecodeError as error:
            raise ValueError("The sample sheet is not text. Save it as CSV or TSV.") from error
        if frame.empty and not len(frame.columns):
            raise ValueError("The sample sheet has no columns.")
        id_column = sample_id_column or str(frame.columns[0])
        if id_column not in frame.columns:
            raise ValueError(f"The sample sheet has no column '{id_column}' to read the sample IDs from.")
        if display_label_column == id_column:
            # The ID column becomes the index on the next line, so it is not a
            # column any more by the time the label is resolved. Asking to be
            # labelled by the sample ID is what ``None`` already means, so honour
            # it rather than rejecting the sheet over a column that is present.
            display_label_column = None
        frame = frame.set_index(id_column, drop=True)
        return cls(frame, display_label_column=display_label_column)

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.table.columns)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(self.table.index)

    def set_display_label_column(self, column: str | None) -> None:
        if column in (None, ""):
            self.display_label_column = None
            return
        if column not in self.table.columns:
            raise ValueError(f"Display label column '{column}' was not found")
        self.display_label_column = column

    def get_display_label(self, sample_id: str) -> str:
        canonical = str(sample_id)
        if self.display_label_column is None or canonical not in self.table.index:
            return canonical
        value = self.table.at[canonical, self.display_label_column]
        if pd.isna(value) or not str(value).strip():
            return canonical
        return str(value).strip()

    def align(
        self,
        reaction_sample_ids: Sequence[str],
        expression_sample_ids: Sequence[str] | None = None,
        *,
        policy: Literal["strict", "partial"] = "strict",
    ) -> MetadataAlignmentReport:
        """Align the sheet's rows with the measured samples.

        ``partial`` accepts a sheet that leaves some samples out (they stay
        ungrouped) or describes others the data lacks, as long as it describes one
        of them; ``strict`` requires the sheet, the reaction data and any
        expression data to name the same samples.
        """
        if policy not in {"strict", "partial"}:
            raise ValueError("alignment policy must be 'strict' or 'partial'")
        reaction = tuple(map(str, reaction_sample_ids))
        expression = tuple(
            map(str, expression_sample_ids if expression_sample_ids is not None else ())
        )
        metadata_set, reaction_set, expression_set = set(self.sample_ids), set(reaction), set(expression)
        aligned = tuple(sample for sample in reaction if sample in metadata_set and (not expression or sample in expression_set))
        report = MetadataAlignmentReport(
            metadata_samples=self.sample_ids,
            reaction_samples=reaction,
            expression_samples=expression,
            aligned_samples=aligned,
            missing_from_metadata=tuple(sample for sample in reaction if sample not in metadata_set),
            metadata_only=tuple(sample for sample in self.sample_ids if sample not in reaction_set),
            missing_from_expression=tuple(sample for sample in reaction if expression and sample not in expression_set),
        )
        if not report.aligned_samples:
            raise ValueError(
                "No sample of the reaction data has a row in the sample sheet: the reaction data names its samples "
                f"{_some(reaction)}, and the sheet {_some(self.sample_ids)}. Choose the column that holds these IDs."
            )
        problems = []
        if policy == "strict" and report.missing_from_metadata:
            problems.append(
                f"{_samples(len(report.missing_from_metadata))} of the reaction data's {len(reaction):,} "
                f"{'has' if len(report.missing_from_metadata) == 1 else 'have'} no row in it "
                f"({_some(report.missing_from_metadata)})"
            )
        if policy == "strict" and report.metadata_only:
            problems.append(
                f"{len(report.metadata_only):,} of its rows name no sample of the reaction data ({_some(report.metadata_only)})"
            )
        if policy == "strict" and report.missing_from_expression:
            problems.append(
                f"{_samples(len(report.missing_from_expression))} of the reaction data "
                f"{'has' if len(report.missing_from_expression) == 1 else 'have'} no expression "
                f"({_some(report.missing_from_expression)})"
            )
        if problems:
            raise ValueError("The sample sheet does not match the data: " + "; ".join(problems) + ".")
        return report

"""Validated, independently loaded sample metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd


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

    @classmethod
    def from_table(
        cls,
        path: str | Path,
        *,
        separator: str | None = None,
        sample_id_column: str | None = None,
        display_label_column: str | None = None,
    ) -> "SampleMetadata":
        path = Path(path)
        if separator is None:
            separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else None
        frame = pd.read_csv(path, sep=separator, engine="python", dtype=object)
        if frame.empty and not len(frame.columns):
            raise ValueError("Metadata file has no columns")
        id_column = sample_id_column or str(frame.columns[0])
        if id_column not in frame.columns:
            raise ValueError(f"Sample ID column '{id_column}' was not found")
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
        policy: Literal["strict", "reaction_required"] = "strict",
    ) -> MetadataAlignmentReport:
        if policy not in {"strict", "reaction_required"}:
            raise ValueError("alignment policy must be 'strict' or 'reaction_required'")
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
        errors = list(report.missing_from_metadata)
        if policy == "strict":
            errors.extend(report.metadata_only)
            errors.extend(report.missing_from_expression)
        if errors:
            raise ValueError(
                "Metadata alignment failed: "
                f"{len(report.missing_from_metadata)} reaction samples missing metadata, "
                f"{len(report.metadata_only)} metadata-only samples, and "
                f"{len(report.missing_from_expression)} reaction samples missing expression"
            )
        return report

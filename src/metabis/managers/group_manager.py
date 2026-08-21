"""Validated sample group definitions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

import pandas as pd

from ..analysis.comparison_membership import (
    ComparisonMembership,
    GroupComparisonSpec,
    overlapping_samples,
)
from ..domain.sample_metadata import SampleMetadata


@dataclass(frozen=True)
class MetadataPredicate:
    column: str
    operator: Literal["equals", "in", "not_in", "is_missing"]
    values: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroupDefinition:
    source: Literal["prefix", "underscore", "adata", "metadata", "custom"]
    predicates: tuple[MetadataPredicate, ...] = ()
    source_group_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SampleGroup:
    id: str
    name: str
    sample_ids: tuple[str, ...]
    definition: GroupDefinition


@dataclass(frozen=True)
class GroupLoadResult:
    groups: tuple[SampleGroup, ...]
    unassigned_samples: tuple[str, ...]


class GroupManager:
    def __init__(self):
        self.group_records: dict[str, SampleGroup] = {}
        self.active_group: Optional[str] = None
        self.revision = 0

    def _replace(self, groups: dict[str, Sequence[str]], source: str) -> None:
        self.group_records = {
            name: SampleGroup(name, name, tuple(samples), GroupDefinition(source=source))
            for name, samples in groups.items()
        }
        self.active_group = None
        self.revision += 1

    @staticmethod
    def _require_underscored_sample_ids(sample_ids: Sequence[str], criteria: str) -> None:
        """Reject underscore-based criteria when no sample ID carries the delimiter."""
        if any("_" in str(sample_id) for sample_id in sample_ids):
            return
        example = str(sample_ids[0]) if sample_ids else "<none>"
        raise ValueError(
            f"The '{criteria}' criteria splits sample IDs on '_', but none of the "
            f"{len(sample_ids)} sample IDs contain '_' (for example '{example}'). "
            "Group by a metadata column instead, or rename the samples."
        )

    def load_groups(self, sample_ids: List[str], criteria="sample_prefix", columns=None, fields=None, metabolic_model=None, expression_data=None):
        if criteria == "sample_prefix":
            self._require_underscored_sample_ids(sample_ids, criteria)
            grouped = self._group_by_sample_prefix(sample_ids, metabolic_model)
            source = "prefix"
        elif criteria == "underscore_split":
            if not fields:
                raise ValueError("At least one field must be specified for underscore_split criteria")
            self._require_underscored_sample_ids(sample_ids, criteria)
            grouped = self._group_by_underscore_split(sample_ids, fields)
            source = "underscore"
        elif criteria == "adata_column":
            if not columns:
                raise ValueError("At least one column must be specified for adata_column criteria")
            if not expression_data or getattr(expression_data, "adata", None) is None:
                raise ValueError("adata_column criteria requires h5ad format expression data")
            metadata = SampleMetadata(expression_data.adata.obs.copy())
            grouped = self._group_by_metadata(sample_ids, metadata, columns)
            source = "adata"
        else:
            raise ValueError(f"Unsupported grouping criteria: {criteria}")
        if not grouped:
            raise ValueError(
                f"The '{criteria}' criteria produced no groups from {len(sample_ids)} samples. "
                "Choose a criteria that matches how these samples are named or annotated."
            )
        self._replace(grouped, source)
        return {"status": "success", "messages": [f"Created {len(grouped)} groups"]}

    def load_groups_from_metadata(self, metadata: SampleMetadata, columns: Sequence[str], *, sample_ids: Sequence[str] | None = None, include_missing: bool = False) -> GroupLoadResult:
        if not columns:
            raise ValueError("At least one metadata column is required")
        missing = [column for column in columns if column not in metadata.columns]
        if missing:
            raise ValueError(f"Metadata columns not found: {missing}")
        candidates = list(map(str, sample_ids or metadata.sample_ids))
        grouped = self._group_by_metadata(candidates, metadata, columns, include_missing)
        if not grouped:
            raise ValueError(
                f"Metadata columns {list(columns)} produced no groups from {len(candidates)} samples. "
                "Every sample is missing a value in at least one of these columns."
            )
        self._replace(grouped, "metadata")
        assigned = {sample for samples in grouped.values() for sample in samples}
        return GroupLoadResult(tuple(self.group_records.values()), tuple(sample for sample in candidates if sample not in assigned))

    @staticmethod
    def _group_by_metadata(sample_ids, metadata, columns, include_missing=False):
        grouped: dict[str, list[str]] = {}
        for sample_id in sample_ids:
            if sample_id not in metadata.table.index:
                continue
            # .loc with a list column key always yields a Series, including for a
            # single column, so never wrap it as if it were a scalar.
            values = metadata.table.loc[sample_id, list(columns)].tolist()
            if any(pd.isna(value) for value in values) and not include_missing:
                continue
            name = " | ".join("(missing)" if pd.isna(value) else str(value) for value in values)
            grouped.setdefault(name, []).append(sample_id)
        return grouped

    def define_group(self, name: str, sample_ids: Sequence[str], *, definition: GroupDefinition, all_sample_ids: Sequence[str] | None = None, replace: bool = False) -> SampleGroup:
        name = str(name).strip()
        if not name:
            raise ValueError("Group name must be non-empty")
        if name in self.group_records and not replace:
            raise ValueError(f"Group '{name}' already exists")
        normalized = tuple(map(str, sample_ids))
        if not normalized:
            raise ValueError("Group must contain at least one sample")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Group contains duplicate sample IDs")
        unknown = sorted(set(normalized) - set(map(str, all_sample_ids or normalized)))
        if unknown:
            raise ValueError(f"Unknown reaction samples: {unknown[:5]}")
        record = SampleGroup(name, name, normalized, definition)
        self.group_records[name] = record
        self.revision += 1
        return record

    def define_group_from_predicates(self, name: str, metadata: SampleMetadata, predicates: Sequence[MetadataPredicate], *, all_sample_ids: Sequence[str] | None = None, replace: bool = False) -> SampleGroup:
        if not predicates:
            raise ValueError("At least one predicate is required")
        mask = pd.Series(True, index=metadata.table.index)
        for predicate in predicates:
            if predicate.column not in metadata.columns:
                raise ValueError(f"Metadata column '{predicate.column}' was not found")
            values = metadata.table[predicate.column]
            requested = set(map(str, predicate.values))
            if predicate.operator == "equals":
                if len(requested) != 1:
                    raise ValueError("equals requires exactly one value")
                current = values.astype(str).isin(requested)
            elif predicate.operator == "in":
                current = values.astype(str).isin(requested)
            elif predicate.operator == "not_in":
                current = ~values.astype(str).isin(requested)
            elif predicate.operator == "is_missing":
                current = values.isna()
            else:
                raise ValueError(f"Unsupported predicate operator: {predicate.operator}")
            mask &= current
        allowed = set(map(str, all_sample_ids or metadata.sample_ids))
        matches = [sample for sample in metadata.sample_ids if mask.loc[sample] and sample in allowed]
        return self.define_group(name, matches, definition=GroupDefinition("custom", tuple(predicates)), all_sample_ids=allowed, replace=replace)

    def define_group_union(self, name: str, source_group_ids: Sequence[str], *, all_sample_ids: Sequence[str] | None = None, replace: bool = False) -> SampleGroup:
        if not self.group_records:
            raise ValueError("No source groups exist")
        missing = [group for group in source_group_ids if group not in self.group_records]
        if missing:
            raise ValueError(f"Union source groups not found: {missing}")
        members, seen = [], set()
        for group in source_group_ids:
            for sample in self.group_records[group].sample_ids:
                if sample not in seen:
                    seen.add(sample)
                    members.append(sample)
        return self.define_group(name, members, definition=GroupDefinition("custom", source_group_ids=tuple(source_group_ids)), all_sample_ids=all_sample_ids, replace=replace)

    def is_custom_group(self, name: str) -> bool:
        """True when the group came from a saved user definition rather than a grouping criteria."""
        record = self.group_records.get(name)
        return bool(record and record.definition.source == "custom")

    def get_overlap(self, group_1: str, group_2: str) -> tuple[str, ...]:
        return overlapping_samples(self.get_group_samples(group_1), self.get_group_samples(group_2))

    def validate_comparison(self, group_1: str, group_2: str) -> None:
        if group_1 == group_2:
            raise ValueError("Comparison groups must be different")
        overlap = self.get_overlap(group_1, group_2)
        if overlap:
            raise ValueError(f"Comparison groups overlap by {len(overlap)} samples: {', '.join(overlap[:5])}")

    def resolve_comparison_membership(self, spec: GroupComparisonSpec, aligned_sample_ids: Sequence[str]) -> ComparisonMembership:
        """Resolve comparison-local Target/Baseline population membership.

        Identity comes from canonical base-group records. Either side may compose
        several records; labels never replace their IDs or exact sample membership.
        """
        if not self.group_records:
            raise ValueError("No groups have been loaded")
        return ComparisonMembership.resolve(
            spec,
            group_samples={group_id: record.sample_ids for group_id, record in self.group_records.items()},
            labels={group_id: record.name for group_id, record in self.group_records.items()},
            aligned_sample_ids=aligned_sample_ids,
        )

    def get_rest_samples(self, selected_group_id: str, all_sample_ids: Sequence[str]) -> List[str]:
        selected = set(self.get_group_samples(selected_group_id))
        return [str(sample) for sample in all_sample_ids if str(sample) not in selected]

    def _group_by_sample_prefix(self, sample_ids, metabolic_model=None):
        grouped = {}
        for sample_id in sample_ids:
            match = re.match(r"([^_]+)_.*", sample_id)
            if match:
                grouped.setdefault(match.group(1), []).append(sample_id)
        return grouped

    @staticmethod
    def _group_by_underscore_split(sample_ids, fields):
        grouped = {}
        for sample_id in sample_ids:
            parts = sample_id.split("_")
            name = "_".join(parts[index] if 0 <= index < len(parts) else "" for index in fields)
            grouped.setdefault(name, []).append(sample_id)
        return grouped

    def set_active_group(self, group_id):
        if group_id not in self.group_records:
            raise ValueError(f"Group '{group_id}' does not exist in experiment groups")
        self.active_group = group_id

    def get_group_samples(self, group_id):
        if not self.group_records:
            raise ValueError("No groups have been loaded")
        record = self.group_records.get(group_id)
        if record is None:
            raise ValueError(f"Group '{group_id}' not found in experiment groups")
        return list(record.sample_ids)

    def get_group_records(self) -> tuple[SampleGroup, ...]:
        """Return the canonical immutable group records in definition order."""
        return tuple(self.group_records.values())

    def get_group_ids(self): return list(self.group_records)
    def get_active_group(self): return self.active_group
    def has_groups(self): return bool(self.group_records)
    def get_group_count(self): return len(self.group_records)
    def validate_group_id(self, group_id): return group_id in self.group_records
    def get_group_summary(self): return {group_id: len(record.sample_ids) for group_id, record in self.group_records.items()}
    def clear_groups(self):
        self.group_records = {}
        self.active_group = None
        self.revision += 1

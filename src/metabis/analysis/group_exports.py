"""Canonical tabular adapters for sample groups and comparison membership.

This is a supported public seam (ADR-0008): the names in ``__all__`` carry the
same compatibility promise as the other analysis seams, and a caller may depend
on the two exported column tuples as the frames' schema.

These adapters translate the immutable AM3 records; they do not resolve groups,
reselect samples, or interpret the legacy ``The rest`` label.  JSON cells are
used only for nested definition and universe values so CSV/TSV writers can retain
their exact order and structure without relying on Python ``repr`` output.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pandas as pd

from metabis.analysis.artifacts import fingerprint_json
from metabis.analysis.comparison_membership import ComparisonMembership
from metabis.managers.group_manager import GroupDefinition, SampleGroup

SAMPLE_GROUP_EXPORT_COLUMNS = (
    "group_id",
    "group_display_label",
    "definition_source",
    "definition_predicates",
    "definition_source_group_ids",
    "sample_order",
    "sample_id",
    "group_fingerprint",
)

COMPARISON_MEMBERSHIP_EXPORT_COLUMNS = (
    "comparison_fingerprint",
    "comparison_mode",
    "side",
    "base_group_ids",
    "side_label",
    "sample_order",
    "sample_id",
    "universe_sample_ids",
    "excluded_base_group_ids",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _definition_manifest(definition: GroupDefinition) -> dict[str, Any]:
    return {
        "source": definition.source,
        "predicates": [
            {
                "column": predicate.column,
                "operator": predicate.operator,
                "values": list(predicate.values),
            }
            for predicate in definition.predicates
        ],
        "source_group_ids": list(definition.source_group_ids),
    }


def sample_group_fingerprint(group: SampleGroup) -> str:
    """Digest canonical identity, definition, and exact ordered membership.

    The display label is deliberately excluded: ADR-0005 permits it to change
    without changing the group's scientific identity.
    """
    payload = {
        "group_id": group.id,
        "definition": _definition_manifest(group.definition),
        "sample_ids": list(group.sample_ids),
    }
    return fingerprint_json(payload)


def sample_groups_frame(groups: Sequence[SampleGroup]) -> pd.DataFrame:
    """Return one row per exact group sample, preserving group/sample order."""
    rows: list[dict[str, Any]] = []
    for group in groups:
        definition = _definition_manifest(group.definition)
        fingerprint = sample_group_fingerprint(group)
        for sample_order, sample_id in enumerate(group.sample_ids):
            rows.append(
                {
                    "group_id": group.id,
                    "group_display_label": group.name,
                    "definition_source": group.definition.source,
                    "definition_predicates": _canonical_json(
                        definition["predicates"]
                    ),
                    "definition_source_group_ids": _canonical_json(
                        definition["source_group_ids"]
                    ),
                    "sample_order": sample_order,
                    "sample_id": sample_id,
                    "group_fingerprint": fingerprint,
                }
            )
    return pd.DataFrame(rows, columns=list(SAMPLE_GROUP_EXPORT_COLUMNS))


def comparison_membership_frame(
    membership: ComparisonMembership,
) -> pd.DataFrame:
    """Return the exact Target/Baseline samples plus their resolution rule.

    ``universe_sample_ids`` and ``excluded_base_group_ids`` repeat on every row by
    design: each exported row remains auditable after filtering or splitting a
    CSV, while the ordered JSON arrays preserve the complete comparison rule.
    """
    universe = _canonical_json(list(membership.universe))
    exclusions = _canonical_json(list(membership.excluded_base_group_ids))
    sides = (
        (
            "target",
            membership.target_base_group_ids,
            membership.target_label,
            membership.target_samples,
        ),
        (
            "baseline",
            membership.baseline_base_group_ids,
            membership.baseline_label,
            membership.baseline_samples,
        ),
    )
    rows = [
        {
            "comparison_fingerprint": membership.fingerprint,
            "comparison_mode": membership.mode.value,
            "side": side,
            "base_group_ids": _canonical_json(list(base_group_ids)),
            "side_label": label,
            "sample_order": sample_order,
            "sample_id": sample_id,
            "universe_sample_ids": universe,
            "excluded_base_group_ids": exclusions,
        }
        for side, base_group_ids, label, sample_ids in sides
        for sample_order, sample_id in enumerate(sample_ids)
    ]
    return pd.DataFrame(rows, columns=list(COMPARISON_MEMBERSHIP_EXPORT_COLUMNS))


__all__ = [
    "COMPARISON_MEMBERSHIP_EXPORT_COLUMNS",
    "SAMPLE_GROUP_EXPORT_COLUMNS",
    "comparison_membership_frame",
    "sample_group_fingerprint",
    "sample_groups_frame",
]

"""Explicit Target/Baseline membership for one group comparison.

Step 2 creates atomic base groups.  A comparison then composes one or more of
those base groups into a Target side and, for pairwise comparisons, a Baseline
side.  Resolution freezes the selected base-group IDs, comparison-local labels,
and exact deduplicated sample membership before any analysis runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

REST_DISPLAY_LABEL = "All other samples"


class ComparisonMode(str, Enum):
    PAIRWISE = "pairwise"
    ONE_VS_REST = "one_vs_rest"


@dataclass(frozen=True)
class ComparisonSideSpec:
    """One comparison-local population composed from atomic base groups."""

    base_group_ids: tuple[str, ...]
    label: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.base_group_ids, str):
            raise TypeError("base_group_ids must be a sequence of group IDs")
        normalized = tuple(str(group_id).strip() for group_id in self.base_group_ids)
        if not normalized or any(not group_id for group_id in normalized):
            raise ValueError("Comparison side requires at least one base group")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Comparison side contains duplicate base-group IDs")
        object.__setattr__(self, "base_group_ids", normalized)
        object.__setattr__(self, "label", str(self.label).strip())


@dataclass(frozen=True)
class GroupComparisonSpec:
    """The user's Target/Baseline selection over canonical base-group IDs."""

    mode: ComparisonMode
    target: ComparisonSideSpec
    baseline: ComparisonSideSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target, ComparisonSideSpec):
            raise TypeError("target must be a ComparisonSideSpec")
        if self.baseline is not None and not isinstance(
            self.baseline, ComparisonSideSpec
        ):
            raise TypeError("baseline must be a ComparisonSideSpec or None")
        if self.mode == ComparisonMode.PAIRWISE and self.baseline is None:
            raise ValueError("Pairwise comparison requires a Baseline side")
        if self.mode == ComparisonMode.ONE_VS_REST and self.baseline is not None:
            raise ValueError("One-vs-rest comparison resolves Baseline automatically")

    @classmethod
    def pairwise(
        cls,
        target_base_group_ids: Sequence[str],
        baseline_base_group_ids: Sequence[str],
        *,
        target_label: str = "",
        baseline_label: str = "",
    ) -> "GroupComparisonSpec":
        """Build a pairwise spec from two base-group selections."""
        return cls(
            ComparisonMode.PAIRWISE,
            ComparisonSideSpec(_group_id_tuple(target_base_group_ids), target_label),
            ComparisonSideSpec(
                _group_id_tuple(baseline_base_group_ids), baseline_label
            ),
        )

    @classmethod
    def one_vs_rest(
        cls,
        target_base_group_ids: Sequence[str],
        *,
        target_label: str = "",
    ) -> "GroupComparisonSpec":
        """Build a spec whose Baseline is the aligned universe outside Target."""
        return cls(
            ComparisonMode.ONE_VS_REST,
            ComparisonSideSpec(_group_id_tuple(target_base_group_ids), target_label),
        )


def overlapping_samples(
    left: Sequence[str], right: Sequence[str]
) -> tuple[str, ...]:
    """Samples shared by two comparison sides, sorted."""
    return tuple(sorted(set(map(str, left)) & set(map(str, right))))


@dataclass(frozen=True)
class ComparisonMembership:
    """The exact Target and Baseline populations one comparison runs on.

    Base-group IDs and sample membership are scientific identity.  Labels are
    comparison-local presentation metadata and deliberately do not affect the
    fingerprint.  In one-vs-rest mode, Baseline is the aligned universe minus
    the deduplicated Target union.
    """

    mode: ComparisonMode
    target_base_group_ids: tuple[str, ...]
    baseline_base_group_ids: tuple[str, ...]
    target_label: str
    baseline_label: str
    target_samples: tuple[str, ...]
    baseline_samples: tuple[str, ...]
    universe: tuple[str, ...]
    excluded_base_group_ids: tuple[str, ...] = ()

    @classmethod
    def resolve(
        cls,
        spec: GroupComparisonSpec,
        *,
        group_samples: Mapping[str, Sequence[str]],
        aligned_sample_ids: Sequence[str],
        labels: Mapping[str, str] | None = None,
        rest_label: str | None = None,
    ) -> "ComparisonMembership":
        """Resolve a comparison spec against base groups and an aligned universe."""
        universe = tuple(str(sample_id) for sample_id in aligned_sample_ids)
        if not universe:
            raise ValueError("Comparison requires an aligned sample universe")
        if len(set(universe)) != len(universe):
            raise ValueError("Aligned sample universe contains duplicate sample IDs")

        display = {str(key): str(value) for key, value in dict(labels or {}).items()}
        target_samples = _union_membership(
            spec.target.base_group_ids,
            group_samples,
            universe,
            side_name="Target",
        )

        if spec.mode == ComparisonMode.ONE_VS_REST:
            selected = set(target_samples)
            baseline_samples = tuple(
                sample_id for sample_id in universe if sample_id not in selected
            )
            if not baseline_samples:
                raise ValueError(
                    "One-vs-rest comparison leaves no samples outside the Target "
                    "population"
                )
            baseline_base_group_ids: tuple[str, ...] = ()
            requested_rest_label = str(rest_label or "").strip()
            baseline_label = requested_rest_label or REST_DISPLAY_LABEL
            excluded_base_group_ids = spec.target.base_group_ids
        else:
            assert spec.baseline is not None
            if spec.target.base_group_ids == spec.baseline.base_group_ids:
                raise ValueError("Target and Baseline selections must be different")
            baseline_samples = _union_membership(
                spec.baseline.base_group_ids,
                group_samples,
                universe,
                side_name="Baseline",
            )
            overlap = overlapping_samples(target_samples, baseline_samples)
            if overlap:
                raise ValueError(
                    f"Target and Baseline overlap by {len(overlap)} samples: "
                    f"{', '.join(overlap[:5])}"
                )
            baseline_base_group_ids = spec.baseline.base_group_ids
            baseline_label = _side_label(spec.baseline, display)
            excluded_base_group_ids = ()

        return cls(
            mode=spec.mode,
            target_base_group_ids=spec.target.base_group_ids,
            baseline_base_group_ids=baseline_base_group_ids,
            target_label=_side_label(spec.target, display),
            baseline_label=baseline_label,
            target_samples=target_samples,
            baseline_samples=baseline_samples,
            universe=universe,
            excluded_base_group_ids=excluded_base_group_ids,
        )

    @property
    def spec(self) -> GroupComparisonSpec:
        """The comparison-local side selection resolved by this membership."""
        target = ComparisonSideSpec(self.target_base_group_ids, self.target_label)
        baseline = (
            ComparisonSideSpec(self.baseline_base_group_ids, self.baseline_label)
            if self.mode == ComparisonMode.PAIRWISE
            else None
        )
        return GroupComparisonSpec(self.mode, target, baseline)

    @property
    def fingerprint(self) -> str:
        """Stable digest of selected base groups and exact sample membership."""
        payload = {
            "mode": self.mode.value,
            "target_base_group_ids": self.target_base_group_ids,
            "baseline_base_group_ids": self.baseline_base_group_ids,
            "target_samples": self.target_samples,
            "baseline_samples": self.baseline_samples,
        }
        serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def _side_label(spec: ComparisonSideSpec, labels: Mapping[str, str]) -> str:
    if spec.label:
        return spec.label
    return " + ".join(labels.get(group_id, group_id) for group_id in spec.base_group_ids)


def _group_id_tuple(group_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(group_ids, str):
        raise TypeError("base-group selection must be a sequence, not a string")
    return tuple(group_ids)


def _union_membership(
    group_ids: Sequence[str],
    group_samples: Mapping[str, Sequence[str]],
    universe: Sequence[str],
    *,
    side_name: str,
) -> tuple[str, ...]:
    missing = [group_id for group_id in group_ids if group_id not in group_samples]
    if missing:
        raise ValueError(f"Base groups not found for {side_name}: {missing}")

    aligned = set(universe)
    seen: set[str] = set()
    members: list[str] = []
    for group_id in group_ids:
        for raw_sample_id in group_samples[group_id]:
            sample_id = str(raw_sample_id)
            if sample_id in aligned and sample_id not in seen:
                seen.add(sample_id)
                members.append(sample_id)
    resolved = tuple(members)
    if not resolved:
        raise ValueError(f"{side_name} population has no aligned samples")
    return resolved


__all__ = [
    "ComparisonMembership",
    "ComparisonMode",
    "ComparisonSideSpec",
    "GroupComparisonSpec",
    "REST_DISPLAY_LABEL",
    "overlapping_samples",
]

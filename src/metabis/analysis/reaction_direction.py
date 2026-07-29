"""Canonical reversible-reaction direction selection for comparison analyses.

The selected source column is a scientific input shared by subsystem Delta scores
and reaction-set enrichment.  Keeping the selection in one immutable artifact
prevents those analyses from independently choosing (or double-counting) a
forward/``_back`` pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from .artifacts import fingerprint_json
from .comparison_artifact import GroupComparisonArtifact


class ReactionDirectionCriterion(str, Enum):
    """Population whose mean activity chooses a reversible source column."""

    DOMINANT_IN_TARGET = "dominant_in_target"
    DOMINANT_IN_BASELINE = "dominant_in_baseline"


class ReactionSourceDirection(str, Enum):
    """Direction represented by an exact reaction result column."""

    FORWARD = "forward"
    BACKWARD = "backward"


DEFAULT_REACTION_DIRECTION_CRITERION = ReactionDirectionCriterion.DOMINANT_IN_BASELINE
"""The criterion every surface falls back to when the user has not chosen one.

Baseline dominance is the published behaviour: the displayed direction is the one
with the highest average activation in the *reference* group of the comparison.
Every default and every ``.get(..., <fallback>)`` reads this name rather than
spelling a member out, so the graph, the hover, Delta score, and ORA cannot drift
apart the way they did when four call sites each named ``DOMINANT_IN_TARGET``.
"""


@dataclass(frozen=True)
class ReactionDirectionSelection:
    """One audited exact-column choice for a unique base reaction."""

    base_reaction_id: str
    source_column: str
    selected_direction: ReactionSourceDirection
    target_mean_activity: float
    baseline_mean_activity: float
    criterion: ReactionDirectionCriterion
    reason: str


@dataclass(frozen=True)
class ReactionDirectionMap:
    """Immutable comparison-level source selection for unique base reactions."""

    criterion: ReactionDirectionCriterion
    membership_fingerprint: str
    entries: tuple[ReactionDirectionSelection, ...]
    fingerprint: str = field(init=False)
    _by_base: Mapping[str, ReactionDirectionSelection] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.criterion, ReactionDirectionCriterion):
            raise TypeError("criterion must be a ReactionDirectionCriterion")
        if not self.membership_fingerprint:
            raise ValueError("membership_fingerprint must not be empty")
        base_ids = tuple(entry.base_reaction_id for entry in self.entries)
        if base_ids != tuple(sorted(base_ids)) or len(base_ids) != len(set(base_ids)):
            raise ValueError("Reaction direction entries must have unique sorted base IDs")
        if any(entry.criterion is not self.criterion for entry in self.entries):
            raise ValueError("Every reaction direction entry must use the map criterion")
        object.__setattr__(
            self,
            "_by_base",
            MappingProxyType({entry.base_reaction_id: entry for entry in self.entries}),
        )
        manifest = {
            "criterion": self.criterion.value,
            "membership_fingerprint": self.membership_fingerprint,
            "entries": [
                {
                    "base_reaction_id": entry.base_reaction_id,
                    "source_column": entry.source_column,
                    "selected_direction": entry.selected_direction.value,
                    "target_mean_activity": entry.target_mean_activity,
                    "baseline_mean_activity": entry.baseline_mean_activity,
                    "reason": entry.reason,
                }
                for entry in self.entries
            ],
        }
        object.__setattr__(self, "fingerprint", fingerprint_json(manifest))

    def get(self, reaction_id: str) -> ReactionDirectionSelection | None:
        """Return the selection for a base or directional reaction ID."""
        return self._by_base.get(to_base_reaction_id(reaction_id))

    def source_column_for(self, reaction_id: str) -> str | None:
        """Return the selected exact result column for one base reaction."""
        entry = self.get(reaction_id)
        return entry.source_column if entry is not None else None


def to_base_reaction_id(reaction_id: str) -> str:
    """Collapse one exact directional column to its canonical base ID."""
    value = str(reaction_id)
    return value[:-5] if value.endswith("_back") else value


def build_reaction_direction_map(
    comparison: GroupComparisonArtifact,
    criterion: ReactionDirectionCriterion = DEFAULT_REACTION_DIRECTION_CRITERION,
) -> ReactionDirectionMap:
    """Choose one exact tested column per base reaction from population means.

    Means are compared directly in the population named by ``criterion``.  A
    lone direction wins and exact ties deterministically select forward.
    """
    if not isinstance(comparison, GroupComparisonArtifact):
        raise TypeError("comparison must be a GroupComparisonArtifact")
    if not isinstance(criterion, ReactionDirectionCriterion):
        raise TypeError("criterion must be a ReactionDirectionCriterion")

    results = comparison.results_frame("reaction")
    target_column = _mean_column(results.columns, "group_1_mean", "target_mean")
    baseline_column = _mean_column(results.columns, "group_2_mean", "baseline_mean")
    available = set(map(str, results.index))
    base_ids = sorted({to_base_reaction_id(item) for item in available})
    entries: list[ReactionDirectionSelection] = []

    for base_id in base_ids:
        forward = base_id if base_id in available else None
        backward_id = f"{base_id}_back"
        backward = backward_id if backward_id in available else None
        candidates = tuple(item for item in (forward, backward) if item is not None)
        if not candidates:  # Defensive: every base ID originated from one candidate.
            continue

        means = {
            source: (
                _finite_mean(results.at[source, target_column], source, "Target"),
                _finite_mean(results.at[source, baseline_column], source, "Baseline"),
            )
            for source in candidates
        }
        if forward is None:
            selected = backward
            reason = "only_backward_available"
        elif backward is None:
            selected = forward
            reason = "only_forward_available"
        else:
            mean_index = (
                0
                if criterion is ReactionDirectionCriterion.DOMINANT_IN_TARGET
                else 1
            )
            population = (
                "target"
                if criterion is ReactionDirectionCriterion.DOMINANT_IN_TARGET
                else "baseline"
            )
            forward_mean = means[forward][mean_index]
            backward_mean = means[backward][mean_index]
            if backward_mean > forward_mean:
                selected = backward
                reason = f"backward_selected_higher_{population}_mean"
            elif forward_mean > backward_mean:
                selected = forward
                reason = f"forward_selected_higher_{population}_mean"
            else:
                selected = forward
                reason = f"forward_selected_{population}_mean_tie"

        assert selected is not None
        target_mean, baseline_mean = means[selected]
        entries.append(
            ReactionDirectionSelection(
                base_reaction_id=base_id,
                source_column=selected,
                selected_direction=(
                    ReactionSourceDirection.BACKWARD
                    if selected.endswith("_back")
                    else ReactionSourceDirection.FORWARD
                ),
                target_mean_activity=target_mean,
                baseline_mean_activity=baseline_mean,
                criterion=criterion,
                reason=reason,
            )
        )

    return ReactionDirectionMap(
        criterion=criterion,
        membership_fingerprint=comparison.membership.fingerprint,
        entries=tuple(entries),
    )


def _mean_column(columns, internal_name: str, exported_name: str) -> str:
    if internal_name in columns:
        return internal_name
    if exported_name in columns:
        return exported_name
    raise ValueError(
        "Reaction results must contain normalized Target and Baseline mean activity "
        f"columns; missing '{internal_name}'"
    )


def _finite_mean(value, source_column: str, population: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Reaction '{source_column}' has a non-numeric {population} mean activity"
        ) from exc
    if not isfinite(result):
        raise ValueError(
            f"Reaction '{source_column}' has a non-finite {population} mean activity"
        )
    return result


__all__ = [
    "DEFAULT_REACTION_DIRECTION_CRITERION",
    "ReactionDirectionCriterion",
    "ReactionDirectionMap",
    "ReactionDirectionSelection",
    "ReactionSourceDirection",
    "build_reaction_direction_map",
    "to_base_reaction_id",
]

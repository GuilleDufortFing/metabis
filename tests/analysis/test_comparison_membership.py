"""Comparison sides are explicit sample membership, not a string sentinel."""

import pytest

from metabis import (
    ComparisonMembership,
    ComparisonMode,
    ComparisonSideSpec,
    GroupComparisonSpec,
)
from metabis.managers.group_manager import GroupDefinition, GroupManager

UNIVERSE = ["s1", "s2", "s3", "s4", "s5"]
GROUP_SAMPLES = {"treated": ["s2", "s1"], "control": ["s3", "s4"]}


def _spec(mode, group_1, group_2=None):
    if mode == ComparisonMode.ONE_VS_REST:
        return GroupComparisonSpec.one_vs_rest((group_1,))
    return GroupComparisonSpec.pairwise((group_1,), (group_2,))


def test_pairwise_membership_carries_both_sides_in_definition_order():
    membership = ComparisonMembership.resolve(
        _spec(ComparisonMode.PAIRWISE, "treated", "control"),
        group_samples=GROUP_SAMPLES,
        aligned_sample_ids=UNIVERSE,
    )

    assert membership.target_samples == ("s2", "s1")
    assert membership.baseline_samples == ("s3", "s4")
    assert membership.excluded_base_group_ids == ()
    assert membership.universe == tuple(UNIVERSE)


def test_one_vs_rest_records_the_aligned_universe_and_its_exclusion_rule():
    membership = ComparisonMembership.resolve(
        _spec(ComparisonMode.ONE_VS_REST, "treated"),
        group_samples=GROUP_SAMPLES,
        aligned_sample_ids=UNIVERSE,
    )

    # The right side is the universe minus the selected group, in universe order:
    # samples outside every saved group belong to the rest.
    assert membership.baseline_samples == ("s3", "s4", "s5")
    assert membership.baseline_base_group_ids == ()
    assert membership.excluded_base_group_ids == ("treated",)
    assert membership.baseline_label == "All other samples"


def test_one_vs_rest_without_remaining_samples_is_rejected():
    with pytest.raises(ValueError, match="no samples outside"):
        ComparisonMembership.resolve(
            _spec(ComparisonMode.ONE_VS_REST, "everything"),
            group_samples={"everything": UNIVERSE},
            aligned_sample_ids=UNIVERSE,
        )


def test_overlapping_and_identical_sides_are_rejected_at_the_comparison_seam():
    with pytest.raises(ValueError, match="overlap"):
        ComparisonMembership.resolve(
            _spec(ComparisonMode.PAIRWISE, "treated", "shared"),
            group_samples={**GROUP_SAMPLES, "shared": ["s1", "s5"]},
            aligned_sample_ids=UNIVERSE,
        )

    with pytest.raises(ValueError, match="must be different"):
        ComparisonMembership.resolve(
            _spec(ComparisonMode.PAIRWISE, "treated", "treated"),
            group_samples=GROUP_SAMPLES,
            aligned_sample_ids=UNIVERSE,
        )

    with pytest.raises(ValueError, match="not found"):
        ComparisonMembership.resolve(
            _spec(ComparisonMode.PAIRWISE, "treated", "missing"),
            group_samples=GROUP_SAMPLES,
            aligned_sample_ids=UNIVERSE,
        )


def test_display_names_label_a_comparison_without_changing_its_identity():
    spec = _spec(ComparisonMode.PAIRWISE, "treated", "control")
    anonymous = ComparisonMembership.resolve(
        spec, group_samples=GROUP_SAMPLES, aligned_sample_ids=UNIVERSE
    )

    renamed = ComparisonMembership.resolve(
        spec,
        group_samples=GROUP_SAMPLES,
        aligned_sample_ids=UNIVERSE,
        labels={"treated": "Compass treated", "control": "Vehicle"},
    )

    assert renamed.target_base_group_ids == ("treated",)
    assert renamed.baseline_base_group_ids == ("control",)
    assert renamed.target_label == "Compass treated"
    assert renamed.baseline_label == "Vehicle"
    # A rename must not invalidate a saved comparison, so it cannot move the fingerprint.
    assert renamed.fingerprint == anonymous.fingerprint


def test_membership_fingerprint_follows_the_samples_that_were_compared():
    spec = _spec(ComparisonMode.PAIRWISE, "treated", "control")
    original = ComparisonMembership.resolve(
        spec, group_samples=GROUP_SAMPLES, aligned_sample_ids=UNIVERSE
    )

    redefined = ComparisonMembership.resolve(
        spec,
        group_samples={**GROUP_SAMPLES, "control": ["s3", "s4", "s5"]},
        aligned_sample_ids=UNIVERSE,
    )

    assert redefined.fingerprint != original.fingerprint


def test_pairwise_sides_union_base_groups_and_deduplicate_samples():
    spec = GroupComparisonSpec(
        ComparisonMode.PAIRWISE,
        ComparisonSideSpec(("cd4", "cd8"), "T cells"),
        ComparisonSideSpec(("macrophage",), "Macrophages"),
    )

    membership = ComparisonMembership.resolve(
        spec,
        group_samples={
            "cd4": ("s1", "s2"),
            "cd8": ("s2", "s3"),
            "macrophage": ("s4", "s5"),
        },
        aligned_sample_ids=UNIVERSE,
    )

    assert membership.target_base_group_ids == ("cd4", "cd8")
    assert membership.baseline_base_group_ids == ("macrophage",)
    assert membership.target_samples == ("s1", "s2", "s3")
    assert membership.baseline_samples == ("s4", "s5")
    assert membership.target_label == "T cells"
    assert membership.baseline_label == "Macrophages"


def test_side_specs_reject_empty_duplicate_and_bare_string_selections():
    with pytest.raises(ValueError, match="at least one"):
        ComparisonSideSpec(())
    with pytest.raises(ValueError, match="duplicate"):
        ComparisonSideSpec(("cd4", "cd4"))
    with pytest.raises(TypeError, match="sequence"):
        GroupComparisonSpec.one_vs_rest("cd4")


def test_multi_group_one_vs_rest_uses_every_other_aligned_sample():
    membership = ComparisonMembership.resolve(
        GroupComparisonSpec.one_vs_rest(
            ("cd4", "cd8"), target_label="T cells"
        ),
        group_samples={"cd4": ("s1", "s2"), "cd8": ("s2", "s3")},
        aligned_sample_ids=UNIVERSE,
        rest_label="Non-T cells",
    )

    assert membership.target_samples == ("s1", "s2", "s3")
    assert membership.baseline_samples == ("s4", "s5")
    assert membership.baseline_base_group_ids == ()
    assert membership.excluded_base_group_ids == ("cd4", "cd8")
    assert membership.baseline_label == "Non-T cells"


def test_composite_side_labels_do_not_change_scientific_identity():
    base = GroupComparisonSpec.pairwise(("cd4", "cd8"), ("macrophage",))
    labelled = GroupComparisonSpec.pairwise(
        ("cd4", "cd8"),
        ("macrophage",),
        target_label="T cells",
        baseline_label="Myeloid",
    )
    groups = {
        "cd4": ("s1",),
        "cd8": ("s2",),
        "macrophage": ("s3", "s4"),
    }

    anonymous_membership = ComparisonMembership.resolve(
        base, group_samples=groups, aligned_sample_ids=UNIVERSE
    )
    labelled_membership = ComparisonMembership.resolve(
        labelled, group_samples=groups, aligned_sample_ids=UNIVERSE
    )

    assert labelled_membership.target_label == "T cells"
    assert labelled_membership.baseline_label == "Myeloid"
    assert labelled_membership.fingerprint == anonymous_membership.fingerprint


def test_composite_sides_reject_sample_overlap_after_union_resolution():
    with pytest.raises(ValueError, match="Target and Baseline overlap.*s2"):
        ComparisonMembership.resolve(
            GroupComparisonSpec.pairwise(("cd4", "cd8"), ("myeloid",)),
            group_samples={
                "cd4": ("s1",),
                "cd8": ("s2",),
                "myeloid": ("s2", "s3"),
            },
            aligned_sample_ids=UNIVERSE,
        )


def test_manager_resolves_membership_from_the_canonical_records():
    manager = GroupManager()
    manager.define_group("treated", ["s1", "s2"], definition=GroupDefinition("custom"))
    manager.define_group("control", ["s3", "s4"], definition=GroupDefinition("custom"))

    membership = manager.resolve_comparison_membership(
        _spec(ComparisonMode.ONE_VS_REST, "treated"), UNIVERSE
    )

    assert membership.target_samples == ("s1", "s2")
    assert membership.baseline_samples == ("s3", "s4", "s5")
    assert membership.spec == _spec(ComparisonMode.ONE_VS_REST, "treated")


def test_manager_resolves_comparison_local_composite_sides():
    manager = GroupManager()
    manager.define_group("cd4", ["s1"], definition=GroupDefinition("metadata"))
    manager.define_group("cd8", ["s2"], definition=GroupDefinition("metadata"))
    manager.define_group(
        "macrophage", ["s3", "s4"], definition=GroupDefinition("metadata")
    )

    membership = manager.resolve_comparison_membership(
        GroupComparisonSpec.pairwise(
            ("cd4", "cd8"),
            ("macrophage",),
            target_label="CD4 + CD8",
        ),
        UNIVERSE,
    )

    assert membership.target_samples == ("s1", "s2")
    assert membership.baseline_samples == ("s3", "s4")
    assert membership.target_label == "CD4 + CD8"
    assert membership.baseline_label == "macrophage"

import json

from metabis.analysis.comparison_membership import (
    ComparisonMembership,
    ComparisonMode,
    GroupComparisonSpec,
)
from metabis.analysis.group_exports import (
    COMPARISON_MEMBERSHIP_EXPORT_COLUMNS,
    SAMPLE_GROUP_EXPORT_COLUMNS,
    comparison_membership_frame,
    sample_group_fingerprint,
    sample_groups_frame,
)
from metabis.managers.group_manager import (
    GroupDefinition,
    MetadataPredicate,
    SampleGroup,
)


def test_sample_group_export_preserves_canonical_identity_definition_and_order():
    group = SampleGroup(
        id="stable-id",
        name="Readable label",
        sample_ids=("sample-2", "sample-1"),
        definition=GroupDefinition(
            source="custom",
            predicates=(MetadataPredicate("status", "in", ("case", "control")),),
            source_group_ids=("source-b", "source-a"),
        ),
    )

    frame = sample_groups_frame((group,))

    assert tuple(frame.columns) == SAMPLE_GROUP_EXPORT_COLUMNS
    assert frame["group_id"].tolist() == ["stable-id", "stable-id"]
    assert frame["group_display_label"].tolist() == [
        "Readable label",
        "Readable label",
    ]
    assert frame["sample_order"].tolist() == [0, 1]
    assert frame["sample_id"].tolist() == ["sample-2", "sample-1"]
    assert json.loads(frame.loc[0, "definition_predicates"]) == [
        {"column": "status", "operator": "in", "values": ["case", "control"]}
    ]
    assert json.loads(frame.loc[0, "definition_source_group_ids"]) == [
        "source-b",
        "source-a",
    ]
    assert set(frame["group_fingerprint"]) == {sample_group_fingerprint(group)}


def test_sample_group_fingerprint_ignores_display_only_rename_but_not_membership():
    definition = GroupDefinition("custom")
    original = SampleGroup("stable-id", "Original", ("s1", "s2"), definition)
    renamed = SampleGroup("stable-id", "Renamed", ("s1", "s2"), definition)
    reordered = SampleGroup("stable-id", "Renamed", ("s2", "s1"), definition)

    assert sample_group_fingerprint(original) == sample_group_fingerprint(renamed)
    assert sample_group_fingerprint(original) != sample_group_fingerprint(reordered)


def test_empty_sample_group_export_keeps_stable_headers():
    frame = sample_groups_frame(())

    assert frame.empty
    assert tuple(frame.columns) == SAMPLE_GROUP_EXPORT_COLUMNS


def test_pairwise_membership_export_is_the_exact_analysis_input():
    membership = ComparisonMembership.resolve(
        GroupComparisonSpec.pairwise(("left-id", "left-extra"), ("right-id",)),
        group_samples={
            "left-id": ("l2", "l1"),
            "left-extra": ("l1",),
            "right-id": ("r1", "r3"),
        },
        labels={"left-id": "Left label", "right-id": "Right label"},
        aligned_sample_ids=("r3", "l2", "outside", "l1", "r1"),
    )

    frame = comparison_membership_frame(membership)

    assert tuple(frame.columns) == COMPARISON_MEMBERSHIP_EXPORT_COLUMNS
    assert frame["side"].tolist() == [
        "target",
        "target",
        "baseline",
        "baseline",
    ]
    assert json.loads(frame.loc[0, "base_group_ids"]) == ["left-id", "left-extra"]
    assert json.loads(frame.loc[2, "base_group_ids"]) == ["right-id"]
    assert frame["side_label"].tolist() == [
        "Left label + left-extra",
        "Left label + left-extra",
        "Right label",
        "Right label",
    ]
    assert frame["sample_order"].tolist() == [0, 1, 0, 1]
    assert frame["sample_id"].tolist() == ["l2", "l1", "r1", "r3"]
    assert json.loads(frame.loc[0, "universe_sample_ids"]) == [
        "r3",
        "l2",
        "outside",
        "l1",
        "r1",
    ]
    assert json.loads(frame.loc[0, "excluded_base_group_ids"]) == []
    assert set(frame["comparison_fingerprint"]) == {membership.fingerprint}


def test_one_vs_rest_export_records_the_universe_and_exclusion_rule():
    membership = ComparisonMembership.resolve(
        GroupComparisonSpec.one_vs_rest(("selected",)),
        group_samples={"selected": ("s2", "s1")},
        labels={"selected": "Selected label"},
        aligned_sample_ids=("s1", "r1", "s2", "r2"),
    )

    frame = comparison_membership_frame(membership)

    assert frame["sample_id"].tolist() == ["s2", "s1", "r1", "r2"]
    assert [json.loads(value) for value in frame.loc[2:, "base_group_ids"]] == [
        [],
        [],
    ]
    assert frame.loc[2:, "side_label"].tolist() == [
        "All other samples",
        "All other samples",
    ]
    assert json.loads(frame.loc[0, "universe_sample_ids"]) == [
        "s1",
        "r1",
        "s2",
        "r2",
    ]
    assert json.loads(frame.loc[0, "excluded_base_group_ids"]) == ["selected"]


def test_membership_frame_keeps_headers_if_given_an_empty_but_valid_shape():
    # A resolved comparison cannot have an empty side. Constructing the immutable
    # record directly exercises the adapter's stable empty-table contract without
    # weakening the analysis resolver.
    membership = ComparisonMembership(
        mode=ComparisonMode.PAIRWISE,
        target_base_group_ids=("left",),
        baseline_base_group_ids=("right",),
        target_label="Left",
        baseline_label="Right",
        target_samples=(),
        baseline_samples=(),
        universe=("sample",),
    )

    frame = comparison_membership_frame(membership)

    assert frame.empty
    assert tuple(frame.columns) == COMPARISON_MEMBERSHIP_EXPORT_COLUMNS


def test_both_adapters_are_reachable_through_the_supported_public_surface():
    # The seam is now public (ADR-0008), so an application must be able to build
    # and export both frames without importing `metabis.managers`. Everything
    # here comes from the root namespace or the promoted module.
    import metabis
    from metabis.analysis import group_exports

    group = metabis.SampleGroup(
        id="cd4",
        name="CD4+ T cells",
        sample_ids=("s1", "s2"),
        definition=metabis.GroupDefinition(
            source="metadata",
            predicates=(metabis.MetadataPredicate("cell_type", "in", ("CD4",)),),
        ),
    )
    membership = metabis.ComparisonMembership.resolve(
        metabis.GroupComparisonSpec.one_vs_rest(("cd4",)),
        group_samples={"cd4": group.sample_ids},
        aligned_sample_ids=("s1", "s2", "s3"),
    )

    groups_frame = group_exports.sample_groups_frame((group,))
    membership_frame = group_exports.comparison_membership_frame(membership)

    assert tuple(groups_frame.columns) == group_exports.SAMPLE_GROUP_EXPORT_COLUMNS
    assert groups_frame["sample_id"].tolist() == ["s1", "s2"]
    assert (
        tuple(membership_frame.columns)
        == group_exports.COMPARISON_MEMBERSHIP_EXPORT_COLUMNS
    )
    assert membership_frame["sample_id"].tolist() == ["s1", "s2", "s3"]
    assert membership_frame["side"].tolist() == [
        "target",
        "target",
        "baseline",
    ]

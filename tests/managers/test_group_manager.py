import pandas as pd
import pytest

from metabis import GroupDefinition, MetadataPredicate, SampleMetadata
from metabis.managers.group_manager import GroupManager

SAMPLES = [
    "T_ctrl_1",
    "T_stim_1",
    "B_ctrl_1",
    "B_stim_1",
]


def test_groups_can_be_created_from_sample_prefix():
    manager = GroupManager()

    result = manager.load_groups(SAMPLES, criteria="sample_prefix")

    assert result["status"] == "success"
    assert manager.get_group_summary() == {"T": 2, "B": 2}


def test_groups_can_be_created_from_selected_sample_fields():
    manager = GroupManager()

    manager.load_groups(SAMPLES, criteria="underscore_split", fields=[0, 1])

    assert manager.get_group_ids() == ["T_ctrl", "T_stim", "B_ctrl", "B_stim"]


def test_active_group_lifecycle_and_one_vs_rest_selection():
    manager = GroupManager()
    manager.load_groups(SAMPLES, criteria="sample_prefix")

    manager.set_active_group("T")

    assert manager.get_active_group() == "T"
    assert manager.get_rest_samples("T", SAMPLES) == ["B_ctrl_1", "B_stim_1"]
    manager.clear_groups()
    assert manager.get_active_group() is None
    assert not manager.has_groups()


def test_metadata_intersections_predicates_unions_and_overlap_policy():
    metadata = SampleMetadata(
        pd.DataFrame(
            {
                "cell_type": ["B", "B", "CD4", "CD8"],
                "condition": ["healthy", "cancer", "healthy", "healthy"],
            },
            index=["B_h", "B_c", "T4", "T8"],
        )
    )
    manager = GroupManager()
    result = manager.load_groups_from_metadata(metadata, ["cell_type", "condition"])
    assert manager.get_group_summary()["B | healthy"] == 1
    assert result.unassigned_samples == ()

    cancer = manager.define_group_from_predicates(
        "B cancer",
        metadata,
        [MetadataPredicate("cell_type", "equals", ("B",)), MetadataPredicate("condition", "equals", ("cancer",))],
    )
    manager.define_group("CD4", ["T4"], definition=GroupDefinition("custom"))
    manager.define_group("CD8", ["T8"], definition=GroupDefinition("custom"))
    union = manager.define_group_union("T cells", ["CD4", "CD8"])

    assert cancer.sample_ids == ("B_c",)
    assert union.sample_ids == ("T4", "T8")
    assert manager.get_rest_samples("T cells", metadata.sample_ids) == ["B_h", "B_c"]

    manager.define_group("overlap", ["T8", "B_h"], definition=GroupDefinition("custom"))
    with pytest.raises(ValueError, match="overlap by 1 samples"):
        manager.validate_comparison("T cells", "overlap")


def test_membership_keeps_the_order_it_was_defined_with():
    manager = GroupManager()

    group = manager.define_group(
        "unsorted",
        ["T_stim_1", "B_ctrl_1", "T_ctrl_1"],
        definition=GroupDefinition("custom"),
        all_sample_ids=SAMPLES,
    )

    assert group.sample_ids == ("T_stim_1", "B_ctrl_1", "T_ctrl_1")
    assert manager.get_group_samples("unsorted") == ["T_stim_1", "B_ctrl_1", "T_ctrl_1"]


def test_union_of_overlapping_saved_groups_counts_each_sample_once():
    manager = GroupManager()
    manager.define_group("left", ["T_ctrl_1", "T_stim_1"], definition=GroupDefinition("custom"), all_sample_ids=SAMPLES)
    manager.define_group("right", ["T_stim_1", "B_ctrl_1"], definition=GroupDefinition("custom"), all_sample_ids=SAMPLES)

    union = manager.define_group_union("either", ["left", "right"], all_sample_ids=SAMPLES)

    assert union.sample_ids == ("T_ctrl_1", "T_stim_1", "B_ctrl_1")
    assert manager.get_rest_samples("either", SAMPLES) == ["B_stim_1"]


def test_records_are_the_only_group_store():
    manager = GroupManager()
    manager.load_groups(SAMPLES, criteria="sample_prefix")

    # The raw group dictionary is gone: records are the one representation, and a
    # returned membership is a copy that cannot write back into group state.
    assert not hasattr(manager, "groups")
    manager.get_group_samples("T").append("B_ctrl_1")

    assert manager.get_group_samples("T") == ["T_ctrl_1", "T_stim_1"]
    assert manager.get_group_ids() == ["T", "B"]


def test_clearing_groups_empties_the_records():
    manager = GroupManager()
    manager.load_groups(SAMPLES, criteria="sample_prefix")

    manager.clear_groups()

    assert manager.group_records == {}
    assert not manager.has_groups()


def test_empty_predicate_group_is_rejected():
    metadata = SampleMetadata(pd.DataFrame({"kind": ["A"]}, index=["S1"]))
    manager = GroupManager()
    with pytest.raises(ValueError, match="at least one sample"):
        manager.define_group_from_predicates(
            "missing", metadata, [MetadataPredicate("kind", "equals", ("B",))]
        )


def test_groups_can_be_created_from_a_single_metadata_column():
    """A one-column selection is the default UI state and must not be treated as a scalar."""
    metadata = SampleMetadata(
        pd.DataFrame(
            {"health": ["inflamed", "healthy", "inflamed", None]},
            index=["S1", "S2", "S3", "S4"],
        )
    )
    manager = GroupManager()

    result = manager.load_groups_from_metadata(metadata, ["health"])

    assert manager.get_group_summary() == {"inflamed": 2, "healthy": 1}
    # The sample with no value is reported rather than silently dropped.
    assert result.unassigned_samples == ("S4",)


def test_single_metadata_column_can_retain_missing_values():
    metadata = SampleMetadata(
        pd.DataFrame({"health": ["inflamed", None]}, index=["S1", "S2"])
    )
    manager = GroupManager()

    manager.load_groups_from_metadata(metadata, ["health"], include_missing=True)

    assert manager.get_group_summary() == {"inflamed": 1, "(missing)": 1}


def test_underscore_criteria_are_rejected_when_no_sample_id_has_an_underscore():
    dotted = ["N7.LPA.CCCACATG", "N9.EpiB.AAACGGCT"]
    manager = GroupManager()

    with pytest.raises(ValueError, match="none of the 2 sample IDs contain"):
        manager.load_groups(dotted, criteria="sample_prefix")

    with pytest.raises(ValueError, match="none of the 2 sample IDs contain"):
        manager.load_groups(dotted, criteria="underscore_split", fields=[0])

    assert not manager.has_groups()


def test_custom_groups_can_be_replaced_by_name():
    metadata = SampleMetadata(
        pd.DataFrame({"kind": ["A", "B", "A"]}, index=["S1", "S2", "S3"])
    )
    manager = GroupManager()

    manager.define_group_from_predicates(
        "picked", metadata, [MetadataPredicate("kind", "equals", ("A",))]
    )
    assert manager.is_custom_group("picked")

    # Saving the same name without replace is still refused...
    with pytest.raises(ValueError, match="already exists"):
        manager.define_group_from_predicates(
            "picked", metadata, [MetadataPredicate("kind", "equals", ("B",))]
        )

    # ...but an explicit replace redefines it in place.
    replaced = manager.define_group_from_predicates(
        "picked", metadata, [MetadataPredicate("kind", "equals", ("B",))], replace=True
    )

    assert replaced.sample_ids == ("S2",)
    assert manager.get_group_samples("picked") == ["S2"]


def test_metadata_derived_groups_are_not_custom():
    metadata = SampleMetadata(pd.DataFrame({"kind": ["A", "B"]}, index=["S1", "S2"]))
    manager = GroupManager()

    manager.load_groups_from_metadata(metadata, ["kind"])

    assert not manager.is_custom_group("A")


def test_metadata_grouping_without_any_usable_value_is_rejected():
    metadata = SampleMetadata(
        pd.DataFrame({"health": [None, None]}, index=["S1", "S2"])
    )
    manager = GroupManager()

    with pytest.raises(ValueError, match="produced no groups"):
        manager.load_groups_from_metadata(metadata, ["health"])

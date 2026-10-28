"""An analysis's comparisons: each chosen side against the other chosen sides."""

import pytest

from metabis.analysis.comparison_membership import ComparisonMembership, each_against_the_rest

GROUPS = {"H": ("h1", "h2"), "N": ("n1", "n2"), "I": ("i1", "i2"), "X": ("x1", "x2")}
UNIVERSE = [s for samples in GROUPS.values() for s in samples]


def test_two_sides_make_one_comparison():
    (spec,) = each_against_the_rest([("Inflamed", ["I"]), ("Healthy", ["H"])])

    membership = ComparisonMembership.resolve(spec, group_samples=GROUPS, aligned_sample_ids=UNIVERSE)
    assert (membership.target_label, membership.baseline_label) == ("Inflamed", "Healthy")
    assert membership.baseline_samples == ("h1", "h2")


def test_each_side_is_compared_with_the_other_chosen_sides_only():
    specs = each_against_the_rest([("Healthy", ["H"]), ("Non-inflamed", ["N"]), ("Inflamed", ["I"])])

    resolved = [ComparisonMembership.resolve(spec, group_samples=GROUPS, aligned_sample_ids=UNIVERSE) for spec in specs]
    assert [m.target_label for m in resolved] == ["Healthy", "Non-inflamed", "Inflamed"]
    assert resolved[2].baseline_label == "Healthy + Non-inflamed"
    # X, not chosen, is in no comparison: "the rest" is the rest of the chosen samples.
    assert set(resolved[2].baseline_samples) == {"h1", "h2", "n1", "n2"}


def test_a_composite_side_joins_its_groups_and_many_sides_are_counted():
    specs = each_against_the_rest([("Epithelium", ["H", "N"]), ("I", ["I"]), ("X", ["X"])])

    first = ComparisonMembership.resolve(specs[0], group_samples=GROUPS, aligned_sample_ids=UNIVERSE)
    assert first.target_samples == ("h1", "h2", "n1", "n2") and first.baseline_label == "I + X"
    many = each_against_the_rest([(name, [name]) for name in ["H", "N", "I", "X"]] + [("Y", ["Y"])])
    assert many[0].baseline.label == "the other 4 groups"


def test_sides_cannot_share_a_group_or_a_name():
    with pytest.raises(ValueError, match="both"):
        each_against_the_rest([("Epithelium", ["H", "N"]), ("Healthy", ["H"])])
    with pytest.raises(ValueError, match="name of its own"):
        each_against_the_rest([("A", ["H"]), ("A", ["N"])])
    with pytest.raises(ValueError, match="at least two"):
        each_against_the_rest([("A", ["H"])])

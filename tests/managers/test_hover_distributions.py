"""Hover distributions run on the comparison's explicit membership.

These pin the consumer AM3 broke: `DataProvider` re-derived the two comparison
sides from `GroupManager` at hover time, using the sample-exclusion arguments the
`ComparisonMembership` seam replaced.  The statistics beside the plot were
covered; the distributions under it were not, so the break only surfaced when a
node was hovered in a mounted session.
"""

import logging

import pandas as pd
import pytest

from metabis import (
    AnalysisWorkspace,
    ComparisonMembership,
    GroupComparison,
    GroupComparisonSpec,
)
from metabis.domain.expression_matrix import DenseExpressionMatrix
from metabis.managers.data_provider import DataProvider
from metabis.managers.group_manager import GroupDefinition, GroupManager
from metabis.managers.hover.models import HoverContext
from metabis.managers.hover.sections.distribution_plot import DistributionPlotSection
from metabis.managers.workspace_mode import AnalysisWorkspaceMode

SAMPLES = ["A1", "A2", "B1", "B2"]


class _ReactionData:
    def __init__(self):
        self.frame = pd.DataFrame({"R1": [1.0, 2.0, 4.0, 5.0]}, index=SAMPLES)

    def get_sample_ids(self):
        return list(self.frame.index)

    def get_reaction_df(self):
        return self.frame

    def get_reaction_row(self, reaction_id):
        if reaction_id not in self.frame.columns:
            return None
        return self.frame[reaction_id]


class _ExpressionData:
    def __init__(self):
        self.frame = pd.DataFrame({"G1": [10.0, 12.0, 30.0, 33.0]}, index=SAMPLES)

    def get_sample_ids(self):
        return list(self.frame.index)

    def get_expression_df(self):
        return self.frame

    def get_feature_ids(self):
        return [str(column) for column in self.frame.columns]

    def matrix(self):
        return DenseExpressionMatrix(self.frame)

    def get_gene_row(self, gene_id):
        if gene_id not in self.frame.columns:
            return None
        return self.frame[gene_id]


@pytest.fixture
def group_manager():
    manager = GroupManager()
    manager.define_group("A", ["A1", "A2"], definition=GroupDefinition("custom"))
    manager.define_group("B", ["B1", "B2"], definition=GroupDefinition("custom"))
    return manager


def _comparison(group_manager, spec):
    membership = group_manager.resolve_comparison_membership(spec, SAMPLES)
    return GroupComparison(membership=membership)


def _provider(group_manager, comparison):
    # The provider reads only the comparison's resolved membership for
    # distributions, so a membership-carrying comparison stands in for the
    # installed artifact here (both expose ``.membership``).
    return DataProvider(
        reaction_data=_ReactionData(),
        expression_data=_ExpressionData(),
        group_comparison_artifact=comparison,
        group_manager=group_manager,
    )


def test_pairwise_reaction_distributions_come_from_membership(group_manager):
    comparison = _comparison(
        group_manager, GroupComparisonSpec.pairwise(("A",), ("B",))
    )
    provider = _provider(group_manager, comparison)

    distributions = provider.get_reaction_flux_distributions("R1")

    assert distributions["has_distributions"] is True
    assert distributions["group1_values"] == [1.0, 2.0]
    assert distributions["group2_values"] == [4.0, 5.0]


def test_pairwise_gene_distributions_come_from_membership(group_manager):
    comparison = _comparison(
        group_manager, GroupComparisonSpec.pairwise(("A",), ("B",))
    )
    provider = _provider(group_manager, comparison)

    distributions = provider.get_gene_expression_distributions("G1")

    assert distributions["has_distributions"] is True
    assert distributions["group1_values"] == [10.0, 12.0]
    assert distributions["group2_values"] == [30.0, 33.0]


def test_distribution_labels_come_from_the_captured_membership(group_manager):
    """Display labels stay paired with the samples captured for the comparison."""
    membership = ComparisonMembership.resolve(
        GroupComparisonSpec.pairwise(("A",), ("B",)),
        group_samples={"A": ["A1", "A2"], "B": ["B1", "B2"]},
        aligned_sample_ids=SAMPLES,
        labels={"A": "Cases", "B": "Controls"},
    )
    provider = _provider(group_manager, GroupComparison(membership=membership))

    reaction_distributions = provider.get_reaction_flux_distributions("R1")
    gene_distributions = provider.get_gene_expression_distributions("G1")

    assert reaction_distributions["group1_name"] == "Cases"
    assert reaction_distributions["group2_name"] == "Controls"
    assert gene_distributions["group1_name"] == "Cases"
    assert gene_distributions["group2_name"] == "Controls"


def test_one_vs_rest_distributions_plot_the_resolved_rest(group_manager):
    """The Baseline label is display-only; samples come from the membership."""
    comparison = _comparison(
        group_manager, GroupComparisonSpec.one_vs_rest(("A",))
    )
    provider = _provider(group_manager, comparison)

    distributions = provider.get_reaction_flux_distributions("R1")

    assert distributions["group2_name"] == "All other samples"
    assert distributions["group1_values"] == [1.0, 2.0]
    assert distributions["group2_values"] == [4.0, 5.0]


def test_distributions_keep_the_samples_the_comparison_ran_on(group_manager):
    """A redefined group must not move the plotted samples under installed statistics."""
    comparison = _comparison(
        group_manager, GroupComparisonSpec.pairwise(("A",), ("B",))
    )
    provider = _provider(group_manager, comparison)

    group_manager.define_group(
        "A", ["A1", "B1"], definition=GroupDefinition("custom"), replace=True
    )

    distributions = provider.get_reaction_flux_distributions("R1")

    assert distributions["group1_values"] == [1.0, 2.0]


def test_a_comparison_without_membership_reports_no_distributions(group_manager):
    provider = _provider(
        group_manager,
        GroupComparison(spec=GroupComparisonSpec.pairwise(("A",), ("B",))),
    )

    assert provider.get_reaction_flux_distributions("R1") == {"has_distributions": False}
    assert provider.get_gene_expression_distributions("G1") == {"has_distributions": False}


class _Model:
    def get_reaction_metadata(self, reaction_id):
        return {"name": reaction_id, "subsystems": ["Glycolysis"], "genes": ["G1"]}

    def get_gene_metadata(self, gene_id):
        return {"symbol": gene_id, "reactions": ["R1"]}

    def get_subsystems(self):
        return ["Glycolysis"]

    def get_reactions_for_subsystem(self, _subsystem):
        return ["R1"]

    def get_genes_for_subsystem(self, _subsystem):
        return ["G1"]

    def get_subsystem_category(self, _subsystem):
        return None


@pytest.mark.parametrize(
    "element_id, element_type", [("R1", "reaction"), ("G1", "gene")]
)
def test_hovering_a_node_with_plots_does_not_render_an_error_card(
    element_id, element_type, install_comparison
):
    """The hover fallback turns any exception into card text, so assert on the card.

    `HoverManager.get_element_hover` catches everything and returns
    "<Type>: <id> (Error: ...)".  A broken data path therefore reaches the user as
    a rendered card rather than a failing test or a server traceback, which is how
    the group-signature change shipped green.
    """
    manager = AnalysisWorkspace(
        metabolic_model=_Model(),
        reaction_data=_ReactionData(),
        expression_data=_ExpressionData(),
    )
    manager.group_manager.define_group(
        "A", ["A1", "A2"], definition=GroupDefinition("custom"), all_sample_ids=SAMPLES
    )
    manager.group_manager.define_group(
        "B", ["B1", "B2"], definition=GroupDefinition("custom"), all_sample_ids=SAMPLES
    )
    comparison = GroupComparison(
        membership=manager.group_manager.resolve_comparison_membership(
            GroupComparisonSpec.pairwise(("A",), ("B",)), SAMPLES
        )
    )
    comparison.run_test(manager, test="ttest", correction="holm")
    comparison.run_test(manager, for_genes=True, test="ttest", correction="holm")
    install_comparison(manager, comparison)

    hover = manager.get_element_hover(
        element_id, element_type, show_plot=True, show_stats=True
    )

    assert "(Error:" not in hover
    assert "Plot error:" not in hover


class _ReverseOnlyReactionData(_ReactionData):
    """Compass reported no flux forward, so preprocessing dropped that column.

    A blocked direction is removed from the frame rather than kept as zeros (the
    Compass low-range filter), so the forward id is simply absent here.
    """

    def __init__(self):
        super().__init__()
        self.frame = pd.DataFrame({"R1_back": [1.0, 2.0, 4.0, 5.0]}, index=SAMPLES)


def _comparison_context():
    return HoverContext(
        mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
        hover_type="reaction",
        element_type="reaction",
        element_id="R1",
        show_plot=True,
    )


def _reverse_only_hover_data(group_manager):
    comparison = _comparison(
        group_manager, GroupComparisonSpec.pairwise(("A",), ("B",))
    )
    provider = DataProvider(
        metabolic_model=_Model(),
        reaction_data=_ReverseOnlyReactionData(),
        group_comparison_artifact=comparison,
        group_manager=group_manager,
    )
    return provider.get_reaction_data(
        "R1", include_stats=False, include_distributions=True
    )


def test_a_blocked_forward_direction_does_not_hide_the_measured_reverse(group_manager):
    """The violin section must ask about both directions, not just the forward one.

    ``get_reaction_data`` nests the backward direction inside the forward one, so
    a section that gates on the outer ``has_distributions`` gates on forward
    alone -- and a reverse-only reaction has no forward column at all. Its
    measured reverse direction then lost its violin with no message at all
    (WP10/D-65: a blocked direction is a result, not an absence).
    """
    data = _reverse_only_hover_data(group_manager)

    assert data["distributions"]["has_distributions"] is False
    assert data["distributions"]["backward"]["has_distributions"] is True
    assert DistributionPlotSection("reaction").should_include(
        data, _comparison_context()
    )


def test_a_reverse_only_reaction_renders_the_violin_it_has(group_manager):
    """One panel, drawn from the direction that has values, not from `forward`."""
    data = _reverse_only_hover_data(group_manager)

    plot = DistributionPlotSection("reaction").build(data, _comparison_context())

    assert "Plot error:" not in plot
    assert "<svg" in plot


def test_a_direction_blocked_in_both_senses_draws_nothing_rather_than_failing():
    """Retained-but-blocked columns are all zeros, which leaves no axis to share.

    Blocked directions are dropped by the Compass low-range filter, but a user who
    opts out of that filter keeps them as constant zeros. Excluding zeros from the
    shared axis then leaves nothing to take a minimum of.
    """
    zeros = {
        "has_distributions": True,
        "group1_name": "Target",
        "group2_name": "Baseline",
        "group1_values": [0.0, 0.0],
        "group2_values": [0.0, 0.0],
    }
    data = {"distributions": dict(zeros, backward=dict(zeros))}

    plot = DistributionPlotSection("reaction").build(data, _comparison_context())

    assert "Plot error:" not in plot


def test_the_comparison_violin_does_not_draw_single_group_activity_data():
    """Activity distributions carry a shape this section cannot plot.

    The hover chooses its violin section from the mode and its data from the
    provider, and the provider used to answer with the single-group shape
    whenever an activity artifact was installed. Skipping beats the red
    "Plot error: 'group2_values'" card that reached users.
    """
    data = {
        "distributions": {
            "has_distributions": True,
            "is_activity": True,
            "reference_values": [1.0, 2.0, 3.0],
            "element_values": [2.0, 2.5],
        }
    }

    assert not DistributionPlotSection("reaction").should_include(
        data, _comparison_context()
    )


def test_a_failed_hover_is_logged_and_not_only_rendered(caplog):
    """An error card must also reach the server log, or the mounted gate cannot see it."""
    manager = AnalysisWorkspace(metabolic_model=_Model(), reaction_data=_ReactionData())

    with caplog.at_level(logging.ERROR, logger="metabis.managers.hover_manager"):
        hover = manager.get_element_hover(None, "reaction", show_stats=True)

    assert "(Error:" in hover
    assert "Hover build failed" in caplog.text

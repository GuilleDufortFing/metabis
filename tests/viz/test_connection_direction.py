"""The arrowhead, the node colour, and the hover read one direction (WP10/D-69).

The graph used to decide which way to draw a reversible reaction from its own
private copy of the dominance rule, computed from the group's raw activity
averages. That copy disagreed with the rest of the graph twice over: it was keyed
on the forward column, so a reaction measured only in reverse drew forward while
its hover said "Reverse (more active direction)"; and it ignored the Step-4
Target/Baseline criterion entirely.

These tests pin the three surfaces to the same answer on the same input.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from metabis.analysis.artifacts import (
    AnalysisArtifactMetadata,
    ArtifactDimension,
    ArtifactKind,
    ArtifactProvenance,
    PlainTable,
)
from metabis.analysis.comparison_artifact import (
    ComparisonThresholds,
    GroupComparisonArtifact,
)
from metabis.analysis.comparison_membership import (
    ComparisonMembership,
    GroupComparisonSpec,
)
from metabis.analysis.reaction_direction import (
    DEFAULT_REACTION_DIRECTION_CRITERION,
    ReactionDirectionCriterion,
    build_reaction_direction_map,
)
from metabis.analysis_workspace import RevisionKind, WorkspaceRevisions
from metabis.managers.element_presentation import comparison_display_direction
from metabis.viz.metabolic_visualization.enrichment.utilities import (
    should_reverse_direction,
)
from metabis.viz.metabolic_visualization.generation.data_structures.core_types import (
    GlyphClass,
)


def _comparison(rows, *, target=("target",), baseline=("baseline",)):
    group_samples = {name: (f"{name}_s1", f"{name}_s2") for name in target + baseline}
    membership = ComparisonMembership.resolve(
        GroupComparisonSpec.pairwise(target, baseline),
        group_samples=group_samples,
        aligned_sample_ids=tuple(
            sample for samples in group_samples.values() for sample in samples
        ),
    )
    frame = pd.DataFrame.from_dict(rows, orient="index")
    metadata = AnalysisArtifactMetadata(
        kind=ArtifactKind.GROUP_COMPARISON,
        source=ArtifactProvenance(
            workspace_id="workspace",
            revisions=WorkspaceRevisions(dataset=1, groups=1),
            dependencies=frozenset({RevisionKind.DATASET, RevisionKind.GROUPS}),
            source_fingerprints={"dataset": "sha256:test"},
        ),
        method="ttest",
        configuration={"correction": "fdr_bh", "keep_reversed": False},
        scientific_identity={
            "membership_fingerprint": membership.fingerprint,
            "mode": membership.mode.value,
            "target_base_group_ids": membership.target_base_group_ids,
            "baseline_base_group_ids": membership.baseline_base_group_ids,
            "target_samples": membership.target_samples,
            "baseline_samples": membership.baseline_samples,
            "universe": membership.universe,
            "excluded_base_group_ids": membership.excluded_base_group_ids,
        },
        dimensions=(
            ArtifactDimension("samples", len(membership.universe)),
            ArtifactDimension("target_samples", len(membership.target_samples)),
            ArtifactDimension("baseline_samples", len(membership.baseline_samples)),
            ArtifactDimension("reactions", len(frame)),
            ArtifactDimension("genes", 0),
        ),
    )
    return GroupComparisonArtifact(
        metadata=metadata,
        membership=membership,
        test="ttest",
        correction="fdr_bh",
        reaction_results=PlainTable.capture(frame),
        gene_results=None,
        reaction_metrics=(),
        gene_metrics=(),
        significant_reactions=(),
        significant_genes=(),
        reaction_thresholds=ComparisonThresholds(None, None, None, None),
        gene_thresholds=None,
    )


class _SubsystemAnalysis:
    """Only the seam the direction resolvers read."""

    def __init__(self, direction_map):
        self._map = direction_map

    def get_reaction_direction_map(self):
        return self._map


def _direction(comparison, criterion, reaction_id):
    subsystem_analysis = _SubsystemAnalysis(
        build_reaction_direction_map(comparison, criterion)
    )
    return comparison_display_direction(
        subsystem_analysis, None, comparison, reaction_id
    )


METABOLITE = {"glyph_class": GlyphClass.METABOLITE, "label": "succinyl-CoA[m]"}
REACTION = {"glyph_class": GlyphClass.REACTION, "label": "MAR06414"}


def test_the_published_default_is_baseline_dominance():
    """The paper states the reference group decides, so nothing may default away."""
    assert (
        DEFAULT_REACTION_DIRECTION_CRITERION
        is ReactionDirectionCriterion.DOMINANT_IN_BASELINE
    )


def test_no_surface_carries_its_own_spelling_of_the_default():
    """Every default must be the shared constant, including string-valued ones.

    The graph worker's `DEFAULT_DELTA_CONFIG` spelled `"dominant_in_target"` as a
    literal, so a render that passed no explicit delta config built a Target map
    and drew, coloured, and hovered every reversible reaction from it — undoing
    the selected criterion for the exact path production uses (WP10/D-70).
    """
    from metabis.viz.metabolic_visualization.calculation_input import (
        DEFAULT_DELTA_CONFIG,
    )

    assert DEFAULT_DELTA_CONFIG["reaction_direction_criterion"] == (
        DEFAULT_REACTION_DIRECTION_CRITERION.value
    )


def test_the_direction_map_builds_on_demand_for_a_worker_that_never_prepared_one():
    """A missing map used to fall back to the Target-keyed averages frame.

    The fallback returns a valid direction rather than an error, so a worker that
    rendered without preparing comparison caches silently ignored the criterion.
    """
    from metabis.analysis.subsystem_analysis import SubsystemAnalysis

    comparison = _comparison(
        {
            "MAR06414": {"group_1_mean": 8.0, "group_2_mean": 1.0},
            "MAR06414_back": {"group_1_mean": 3.0, "group_2_mean": 9.0},
        }
    )
    analysis = SubsystemAnalysis(
        SimpleNamespace(
            metabolic_model=object(),
            reaction_data=None,
            expression_data=None,
            color_manager=None,
            get_group_comparison_artifact=lambda: comparison,
        ),
        default_delta_config=None,
    )

    # Nothing has called set_group_comparison or update_delta_config.
    direction_map = analysis.get_reaction_direction_map()

    assert direction_map is not None
    assert direction_map.criterion is DEFAULT_REACTION_DIRECTION_CRITERION
    assert direction_map.source_column_for("MAR06414") == "MAR06414_back"


def test_a_reverse_only_reaction_draws_backward_and_says_so():
    """The reported symptom: forward blocked, so only `_back` survives loading.

    `should_reverse_direction` used to require the forward column before it would
    consider reversing, so this reaction drew its arrows along the model's
    stoichiometry while its hover reported the reverse direction as the active
    one. 210 base reactions over 25 subsystems on the bundled Th17 data.
    """
    comparison = _comparison({"MAR06414_back": {"group_1_mean": 2.1, "group_2_mean": 2.0}})

    direction = _direction(comparison, DEFAULT_REACTION_DIRECTION_CRITERION, "MAR06414")

    assert direction == "backward"
    assert should_reverse_direction(METABOLITE, REACTION, direction) is True


@pytest.mark.parametrize(
    ("criterion", "expected_direction", "expected_reversed"),
    [
        (ReactionDirectionCriterion.DOMINANT_IN_TARGET, "forward", False),
        (ReactionDirectionCriterion.DOMINANT_IN_BASELINE, "backward", True),
    ],
)
def test_the_arrow_follows_the_selected_criterion(
    criterion, expected_direction, expected_reversed
):
    """A reaction whose dominant direction differs between the two populations.

    22 of the 1,270 two-directional reactions on the bundled Th17 data have this
    shape. The arrow ignored the criterion entirely and always read the Target.
    """
    comparison = _comparison(
        {
            "MAR06414": {"group_1_mean": 8.0, "group_2_mean": 1.0},
            "MAR06414_back": {"group_1_mean": 3.0, "group_2_mean": 9.0},
        }
    )

    direction = _direction(comparison, criterion, "MAR06414")

    assert direction == expected_direction
    assert should_reverse_direction(METABOLITE, REACTION, direction) is expected_reversed


def test_direction_resolves_for_a_target_composed_of_several_base_groups():
    """`group_id_1` is "a + b" for a union and is not a row of the averages frame.

    Resolving from the comparison's own mean columns is what makes a composed
    Target answerable at all; the per-group frame returned None for every
    reversible reaction in such a comparison.
    """
    comparison = _comparison(
        {
            "MAR06414": {"group_1_mean": 1.0, "group_2_mean": 2.0},
            "MAR06414_back": {"group_1_mean": 5.0, "group_2_mean": 9.0},
        },
        target=("target_a", "target_b"),
    )

    assert comparison.group_id_1 == "target_a + target_b"
    assert _direction(comparison, DEFAULT_REACTION_DIRECTION_CRITERION, "MAR06414") == (
        "backward"
    )


def test_the_back_spelling_resolves_to_the_same_answer():
    comparison = _comparison(
        {
            "MAR06414": {"group_1_mean": 1.0, "group_2_mean": 9.0},
            "MAR06414_back": {"group_1_mean": 5.0, "group_2_mean": 2.0},
        }
    )
    criterion = DEFAULT_REACTION_DIRECTION_CRITERION

    assert _direction(comparison, criterion, "MAR06414") == "forward"
    assert _direction(comparison, criterion, "MAR06414_back") == "forward"


def test_only_metabolite_reaction_edges_are_ever_flipped():
    """Enzyme and subsystem edges carry no stoichiometry and must not reverse."""
    gene = {"glyph_class": GlyphClass.GENE, "label": "OGDH"}

    assert should_reverse_direction(REACTION, gene, "backward") is False
    assert should_reverse_direction(METABOLITE, REACTION, "backward") is True
    assert should_reverse_direction(REACTION, METABOLITE, "backward") is True


def test_an_unresolved_direction_leaves_stoichiometry_alone():
    """A reaction absent from the map keeps the model's own arrow."""
    assert should_reverse_direction(METABOLITE, REACTION, None) is False
    assert should_reverse_direction(METABOLITE, REACTION, "forward") is False


def test_the_hover_names_the_direction_the_arrow_was_drawn_in():
    """The screenshot's defect, stated as an equality.

    The hover read the shared map through one path and the arrow re-derived it
    through another, so the panel could say "Reverse (more active direction)"
    above an arrow pointing forward.
    """
    from metabis.managers.hover_manager import HoverManager
    from metabis.managers.workspace_mode import AnalysisWorkspaceMode

    comparison = _comparison(
        {
            "MAR06414": {"group_1_mean": 8.0, "group_2_mean": 1.0},
            "MAR06414_back": {"group_1_mean": 3.0, "group_2_mean": 9.0},
        }
    )
    subsystem_analysis = _SubsystemAnalysis(
        build_reaction_direction_map(comparison, DEFAULT_REACTION_DIRECTION_CRITERION)
    )

    hover_manager = HoverManager.__new__(HoverManager)
    hover_manager._mode = AnalysisWorkspaceMode.GROUP_COMPARISON
    hover_manager.group_comparison_artifact = comparison
    hover_manager.data_provider = SimpleNamespace(
        experiment_analysis=object(),
        subsystem_analysis=subsystem_analysis,
    )

    hover_direction = hover_manager._resolve_main_direction("MAR06414")
    arrow_direction = comparison_display_direction(
        subsystem_analysis, None, comparison, "MAR06414"
    )

    assert hover_direction == "backward"
    assert hover_direction == arrow_direction
    assert should_reverse_direction(METABOLITE, REACTION, arrow_direction) is True
    # And the `_back` spelling the hover is handed for a backward node agrees.
    assert hover_manager._resolve_main_direction("MAR06414_back") == "backward"

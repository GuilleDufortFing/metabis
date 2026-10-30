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
from metabis.analysis.landscape import (
    FeatureLandscape,
    build_landscape,
    feature_landscape,
    marker_reactions,
    select_landscape_subsystems,
    select_varying_features,
    subsystem_delta,
    subsystem_delta_ranking,
    with_thresholds,
)
from metabis.analysis_workspace import RevisionKind, WorkspaceRevisions

GROUPS = {"A": ("a1", "a2"), "B": ("b1", "b2"), "C": ("c1", "c2")}
SUBSYSTEMS = {
    "glyc": ("Glycolysis", "Carbohydrate", ["R1", "R2", "R3", "R4"]),
    "tca": ("TCA cycle", "Energy", ["R5", "R6"]),
    "fao": ("Fatty acid oxidation", "Lipid", ["R7", "R8", "R9", "R10"]),
}
REACTIONS = [f"R{i}" for i in range(1, 11)]


class _Model:
    def get_subsystems(self):
        return list(SUBSYSTEMS)

    def get_reactions_for_subsystem(self, subsystem):
        return SUBSYSTEMS[subsystem][2]

    def get_subsystem_name(self, subsystem):
        return SUBSYSTEMS[subsystem][0]

    def get_subsystem_category(self, subsystem):
        return SUBSYSTEMS[subsystem][1]


class _Reactions:
    def get_reaction_df(self):
        return pd.DataFrame(0.0, index=[s for v in GROUPS.values() for s in v], columns=REACTIONS)


def _artifact(group, effects, pvalues):
    membership = ComparisonMembership.resolve(
        GroupComparisonSpec.one_vs_rest((group,)),
        group_samples=GROUPS,
        aligned_sample_ids=[s for v in GROUPS.values() for s in v],
    )
    frame = pd.DataFrame(
        {
            "cohens_d": effects,
            "adjusted_pval": pvalues,
            "log2_fc": effects,
            "mean_difference": effects,
            "group_1_mean": [1.0 + e for e in effects],
            "group_2_mean": [1.0] * len(effects),
            "reversed": False,
        },
        index=REACTIONS,
    )
    metadata = AnalysisArtifactMetadata(
        kind=ArtifactKind.GROUP_COMPARISON,
        source=ArtifactProvenance(
            workspace_id="workspace",
            revisions=WorkspaceRevisions(dataset=1, groups=1),
            dependencies=frozenset({RevisionKind.DATASET, RevisionKind.GROUPS}),
            source_fingerprints={"dataset": "sha256:test"},
        ),
        method="wilcoxon",
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
            ArtifactDimension("samples", 6),
            ArtifactDimension("target_samples", 2),
            ArtifactDimension("baseline_samples", 4),
            ArtifactDimension("reactions", len(frame)),
            ArtifactDimension("genes", 0),
        ),
    )
    return GroupComparisonArtifact(
        metadata=metadata,
        membership=membership,
        test="wilcoxon",
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


SIG, NOT = 0.01, 0.5
# A is higher almost everywhere; B is lower in glycolysis; C higher only in FAO.
ARTIFACTS = {
    "A": _artifact(
        "A",
        [1.5, 1.2, 0.9, 0.8, 0.7, 0.6, 2.5, 1.0, 1.2, 0.4],
        [SIG, SIG, SIG, SIG, SIG, SIG, SIG, SIG, SIG, NOT],
    ),
    "B": _artifact(
        "B",
        [-1.0, -1.1, -0.9, 0.1, -0.2, 0.3, -0.5, 0.1, -0.1, 0.2],
        [SIG, SIG, SIG, NOT, NOT, NOT, NOT, NOT, NOT, NOT],
    ),
    "C": _artifact(
        "C",
        [0.1, -0.1, 0.0, 0.2, 0.1, -0.2, 3.0, 0.9, 0.8, 0.7],
        [NOT, NOT, NOT, NOT, NOT, NOT, SIG, SIG, SIG, SIG],
    ),
}
THRESHOLDS = {"pval": 0.05, "cohens_d": None, "log2_fc": None, "mean_difference": None}


def _thresholded():
    return {label: with_thresholds(artifact, THRESHOLDS) for label, artifact in ARTIFACTS.items()}


def test_thresholds_decide_significance_on_a_detached_copy():
    decided = with_thresholds(ARTIFACTS["B"], THRESHOLDS)

    assert decided.significant_reactions == ("R1", "R2", "R3")
    assert decided.reaction_thresholds.pval == 0.05
    assert ARTIFACTS["B"].significant_reactions == ()


def test_subsystem_delta_is_signed_significant_count_over_all_reactions():
    table = subsystem_delta(_Model(), _Reactions(), with_thresholds(ARTIFACTS["B"], THRESHOLDS))

    assert table.loc["glyc", "delta"] == pytest.approx(-3 / 4)
    assert table.loc["glyc", "significant"] == 3
    assert table.loc["glyc", "reactions"] == 4
    assert table.loc["tca", "delta"] == 0.0
    assert table.loc["fao", "name"] == "Fatty acid oxidation"
    # What its node's hover says in the network: its significant share, its net
    # direction and how its Delta was computed.
    hover = table.loc["glyc", "hover"]
    assert "3 out of 4 (75.0%)" in hover and "Net direction" in hover and "= <strong>-0.750</strong>" in hover


def test_one_comparisons_delta_table_is_step_4s_ranking_strongest_first():
    table = subsystem_delta_ranking(_Model(), _Reactions(), with_thresholds(ARTIFACTS["B"], THRESHOLDS))
    delta = subsystem_delta(_Model(), _Reactions(), with_thresholds(ARTIFACTS["B"], THRESHOLDS))

    assert list(table.columns[:6]) == ["subsystem", "delta_score", "abs_delta_score", "n_significant", "n_total", "prop_significant"]
    assert table["abs_delta_score"].is_monotonic_decreasing
    glyc = table.set_index("subsystem").loc["glyc"]
    assert glyc["delta_score"] == pytest.approx(delta.loc["glyc", "delta"])
    assert (glyc["n_significant"], glyc["n_total"], glyc["prop_significant"]) == (3, 4, 0.75)


def test_landscape_places_every_group_side_by_side():
    landscape = build_landscape(_Model(), _Reactions(), _thresholded())

    assert list(landscape.delta.columns) == ["A", "B", "C"]
    assert list(landscape.hovers.columns) == ["A", "B", "C"] and "Net direction" in landscape.hovers.loc["glyc", "A"]
    assert landscape.delta.loc["glyc", "A"] == 1.0
    assert landscape.delta.loc["fao", "C"] == 1.0
    assert landscape.significant_reactions.to_dict() == {"A": 9, "B": 3, "C": 4}


def test_spread_rule_removes_each_groups_overall_level_and_orders_by_category():
    landscape = build_landscape(_Model(), _Reactions(), _thresholded())

    chosen = select_landscape_subsystems(landscape, rule="spread", top=2, min_reactions=2)

    # Glycolysis separates B from the others most; TCA, where only A moves, least.
    assert chosen == ["glyc", "fao"]
    assert select_landscape_subsystems(landscape, rule="spread", top=5, min_reactions=3) == [
        "glyc",
        "fao",
    ]


def test_category_and_chosen_rules():
    landscape = build_landscape(_Model(), _Reactions(), _thresholded())

    assert select_landscape_subsystems(landscape, rule="category", category="Energy", min_reactions=1) == ["tca"]
    assert select_landscape_subsystems(landscape, rule="chosen", chosen=["fao", "glyc"]) == ["glyc", "fao"]
    with pytest.raises(ValueError, match="No reactions in the data"):
        select_landscape_subsystems(landscape, rule="chosen", chosen=["missing"])


def test_the_feature_landscape_reads_each_comparisons_d_and_its_verdict():
    landscape = feature_landscape(_thresholded())

    assert list(landscape.effect.columns) == ["A", "B", "C"]
    assert landscape.effect.loc["R7", "C"] == 3.0
    assert bool(landscape.significant.loc["R1", "B"]) and not bool(landscape.significant.loc["R10", "A"])
    assert landscape.adjusted_pval.loc["R10", "C"] == SIG


def test_the_features_varying_most_across_groups_read_as_each_groups_block():
    landscape = feature_landscape(_thresholded())

    # R7 swings from -0.5 in B to 3.0 in C; then R1, R2 and R3, A against B.
    # A's block comes first, its largest d first, then C's.
    assert select_varying_features(landscape, top=4) == ["R1", "R2", "R3", "R7"]
    # R10 is significant only in C, and varies least.
    assert select_varying_features(landscape, top=10)[-1] == "R10"


def test_one_comparisons_features_rank_by_the_size_of_d_and_a_reverse_reaction_counts_once():
    index = ["R1", "R1_back", "R2", "R3", "R4"]
    landscape = FeatureLandscape(
        effect=pd.DataFrame({"A": [0.5, -0.5, -2.0, 1.0, 3.0]}, index=index),
        adjusted_pval=pd.DataFrame({"A": [SIG, SIG, SIG, SIG, NOT]}, index=index),
        significant=pd.DataFrame({"A": [True, True, True, True, False]}, index=index),
    )

    # R4 is not significant; R2 is largest, but lower, so last.
    assert select_varying_features(landscape, top=3) == ["R3", "R1", "R2"]
    assert len(select_varying_features(landscape, top=10)) == 3
    with pytest.raises(ValueError, match="at least one feature"):
        select_varying_features(landscape, top=0)


def test_markers_are_significant_higher_and_highest_in_their_group():
    markers = marker_reactions(_thresholded())

    a = markers[markers.group == "A"]
    c = markers[markers.group == "C"]
    assert set(markers.group) == {"A", "C"}
    # Ranked by the lead over the next group; R7 is C's, where C leads A.
    assert list(a.feature_id) == ["R1", "R2", "R3", "R4", "R5", "R9", "R6", "R8"]
    assert list(c.feature_id) == ["R7", "R10"]
    first = c.iloc[0]
    assert (first.cohens_d, first.next_group, first.next_cohens_d) == (3.0, "A", 2.5)
    assert first.margin == pytest.approx(0.5)


def _landscape(delta: dict, sizes: dict) -> "Landscape":
    from metabis.analysis.landscape import Landscape

    frame = pd.DataFrame(delta)
    names = {item: SUBSYSTEMS[item][0] if item in SUBSYSTEMS else item for item in frame.index}
    categories = {item: SUBSYSTEMS[item][1] if item in SUBSYSTEMS else "Other" for item in frame.index}
    return Landscape(
        delta=frame,
        significant=frame.abs().mul(4).round().astype(int),
        names=pd.Series(names),
        categories=pd.Series(categories),
        reactions=pd.Series(sizes),
        significant_reactions=pd.Series({group: 1 for group in frame.columns}),
    )


def test_shared_rows_take_each_familys_most_varying_in_turn():
    from metabis.analysis.landscape import select_shared_subsystems

    # Reaction Deltas swing wider: by size alone, both rows would be theirs.
    reactions = _landscape(
        {"A": {"glyc": 1.0, "tca": 0.5, "fao": 0.2, "x": 0.0}, "B": {"glyc": -1.0, "tca": -0.5, "fao": -0.2, "x": 0.0}},
        {"glyc": 6, "tca": 6, "fao": 6, "x": 6},
    )
    genes = _landscape(
        {"A": {"glyc": 0.0, "tca": 0.0, "fao": 0.0, "x": 0.3}, "B": {"glyc": 0.0, "tca": 0.0, "fao": 0.0, "x": -0.3}},
        {"glyc": 6, "tca": 6, "fao": 6, "x": 6},
    )

    rows = select_shared_subsystems([reactions, genes], top=2, min_reactions=5)

    # Glycolysis, the reactions' first, and x, the genes' first, though the reactions'
    # second and third vary more than any gene row.
    assert set(rows) == {"glyc", "x"}
    # Category, then name, as the single-landscape rule orders them.
    assert rows == sorted(rows, key=lambda item: (reactions.categories[item], reactions.names[item]))


def test_shared_rows_need_every_landscape_and_one_large_enough():
    from metabis.analysis.landscape import select_shared_subsystems

    reactions = _landscape({"A": {"glyc": 1.0, "tca": 0.5}}, {"glyc": 2, "tca": 6})
    genes = _landscape({"A": {"glyc": 0.2}}, {"glyc": 8})

    # TCA has no genes, so it cannot be a shared row; glycolysis is large in the genes.
    assert select_shared_subsystems([reactions, genes], rule="category", category="Carbohydrate", min_reactions=5) == ["glyc"]
    assert select_shared_subsystems([reactions, genes], rule="chosen", chosen=["glyc"]) == ["glyc"]
    with pytest.raises(ValueError, match="No features in the data"):
        select_shared_subsystems([reactions, genes], rule="chosen", chosen=["tca"])


def test_a_gene_landscape_reads_each_subsystems_genes(monkeypatch):
    from metabis.analysis import landscape as module

    asked = []

    def statistics(self, subsystem, for_genes=False):
        asked.append(for_genes)
        return {"is_in_model": True, "total_elements": 3, "significant_elements": 1, "delta_subsystem_score": 0.5}

    monkeypatch.setattr(module.SubsystemAnalysis, "get_subsystem_statistics", statistics)
    genes = build_landscape(_Model(), _Reactions(), {"A": with_thresholds(ARTIFACTS["A"], THRESHOLDS)}, for_genes=True)

    assert set(asked) == {True}
    assert genes.reactions.to_dict() == {"glyc": 3, "tca": 3, "fao": 3}
    # Counted over the comparison's significant genes, none here.
    assert genes.significant_reactions.to_dict() == {"A": 0}


def _ora(rows):
    columns = ["subsystem_id", "subsystem_name", "category", "direction", "signed_log2_enrichment_ratio", "adjusted_pval", "is_significant"]
    return pd.DataFrame(rows, columns=columns)


def test_the_enrichment_landscape_keeps_each_groups_significant_direction():
    from metabis.analysis.landscape import (
        enrichment_landscape,
        select_enriched_subsystems,
    )

    results = {
        "A": _ora([
            ["glyc", "Glycolysis", "Carbohydrate", "target_higher", 1.2, 0.01, True],
            ["glyc", "Glycolysis", "Carbohydrate", "target_lower", -0.3, 0.6, False],
            ["fao", "Fatty acid oxidation", "Lipid", "target_higher", 0.2, 0.7, False],
            ["fao", "Fatty acid oxidation", "Lipid", "target_lower", -0.4, 0.4, False],
        ]),
        "B": _ora([
            ["glyc", "Glycolysis", "Carbohydrate", "target_higher", 0.1, 0.9, False],
            ["glyc", "Glycolysis", "Carbohydrate", "target_lower", -2.0, 0.001, True],
        ]),
    }

    landscape = enrichment_landscape(results)

    assert landscape.enrichment.loc["glyc"].to_dict() == {"A": 1.2, "B": -2.0}
    # Not significant reads NaN, with the smaller adjusted p; B never tested FAO.
    assert pd.isna(landscape.enrichment.at["fao", "A"]) and landscape.adjusted_pval.at["fao", "A"] == 0.4
    assert landscape.tested.loc["fao"].to_dict() == {"A": True, "B": False}
    assert select_enriched_subsystems([landscape]) == ["glyc"]


def test_the_enriched_rows_can_be_the_most_significant_few():
    from metabis.analysis.landscape import (
        enrichment_landscape,
        select_enriched_subsystems,
    )

    results = {
        "A": _ora([
            ["glyc", "Glycolysis", "Carbohydrate", "target_higher", 1.2, 0.02, True],
            ["glyc", "Glycolysis", "Carbohydrate", "target_lower", -0.3, 0.6, False],
            ["tca", "TCA cycle", "Energy", "target_higher", 0.9, 0.001, True],
            ["tca", "TCA cycle", "Energy", "target_lower", -0.1, 0.9, False],
            ["fao", "Fatty acid oxidation", "Lipid", "target_higher", 0.2, 0.7, False],
            ["fao", "Fatty acid oxidation", "Lipid", "target_lower", -0.8, 0.04, True],
        ]),
    }
    landscape = enrichment_landscape(results)

    # The two smallest adjusted p, then drawn by category and name.
    assert select_enriched_subsystems([landscape], top=2) == ["glyc", "tca"]
    assert select_enriched_subsystems([landscape]) == ["glyc", "tca", "fao"]
    with pytest.raises(ValueError, match="at least one"):
        select_enriched_subsystems([landscape], top=0)


def test_one_comparisons_rows_take_the_strongest_delta_in_either_family():
    from metabis.analysis.landscape import select_comparison_subsystems

    reactions = _landscape({"A": {"glyc": 0.3, "tca": -0.9, "fao": 0.1, "x": 1.0}}, {"glyc": 6, "tca": 6, "fao": 6, "x": 2})
    genes = _landscape({"A": {"glyc": -0.95, "fao": 0.2}}, {"glyc": 6, "fao": 6})
    families = {"reaction": reactions, "gene": genes}

    # x swings furthest but is too small; glycolysis is strongest through its genes.
    assert select_comparison_subsystems(families, "A", top=2) == ["glyc", "tca"]
    assert select_comparison_subsystems({"reaction": reactions}, "A", top=2) == ["tca", "glyc"]
    assert select_comparison_subsystems(families, "A", rule="every", min_reactions=5) == ["glyc", "tca", "fao"]
    assert select_comparison_subsystems(families, "A", rule="category", category="Lipid") == ["fao"]
    assert select_comparison_subsystems(families, "A", rule="chosen", chosen=["x", "tca"]) == ["x", "tca"]
    with pytest.raises(ValueError, match="No features in the data"):
        select_comparison_subsystems(families, "A", rule="chosen", chosen=["nope"])


def test_one_comparisons_over_represented_rows_come_most_significant_first():
    from metabis.analysis.landscape import (
        enrichment_landscape,
        select_comparison_subsystems,
    )

    reactions = _landscape({"A": {"glyc": 0.3, "tca": -0.9, "fao": 0.1}}, {"glyc": 6, "tca": 6, "fao": 6})
    ora = enrichment_landscape({
        "A": _ora([
            ["glyc", "Glycolysis", "Carbohydrate", "target_higher", 1.2, 0.02, True],
            ["glyc", "Glycolysis", "Carbohydrate", "target_lower", -0.3, 0.6, False],
            ["fao", "Fatty acid oxidation", "Lipid", "target_higher", 0.2, 0.7, False],
            ["fao", "Fatty acid oxidation", "Lipid", "target_lower", -0.8, 0.004, True],
            ["tca", "TCA cycle", "Energy", "target_higher", 0.2, 0.5, False],
            ["tca", "TCA cycle", "Energy", "target_lower", -0.1, 0.9, False],
        ]),
    })

    rows = select_comparison_subsystems({"reaction": reactions}, "A", rule="ora", top=5, enrichment={"reaction": ora})

    assert rows == ["fao", "glyc"]


class _MembersModel(_Model):
    def get_genes_for_subsystem(self, subsystem):
        return {"glyc": ["G1", "G2", "G9"], "tca": ["G2"], "fao": []}[subsystem]


def test_a_subsystems_members_are_its_tested_features_one_direction_each():
    from metabis.analysis.landscape import subsystem_members
    from metabis.analysis.reaction_direction import ReactionDirectionCriterion

    frame = pd.DataFrame(
        {
            "cohens_d": [0.5, 0.2, 0.1, 0.3, 0.4],
            "adjusted_pval": [0.01] * 5,
            "group_1_mean": [1.0, 3.0, 1.0, 1.0, 1.0],
            "group_2_mean": [2.0, 0.5, 1.0, 1.0, 1.0],
        },
        index=["R1", "R1_back", "R2", "R5", "R6"],
    )
    from dataclasses import replace

    def resized(artifact, genes=None):
        sizes = {"reactions": len(frame), "genes": len(genes) if genes is not None else 0}
        metadata = replace(
            artifact.metadata,
            dimensions=tuple(
                ArtifactDimension(d.name, sizes.get(d.name, d.size), d.label) for d in artifact.metadata.dimensions
            ),
        )
        return replace(
            artifact,
            metadata=metadata,
            reaction_results=PlainTable.capture(frame),
            gene_results=PlainTable.capture(genes) if genes is not None else None,
            gene_thresholds=ComparisonThresholds(None, None, None, None) if genes is not None else None,
        )

    artifact = resized(ARTIFACTS["A"])
    genes = pd.DataFrame({"cohens_d": [0.1, 0.2]}, index=["G1", "G2"])

    by_baseline = subsystem_members(_MembersModel(), artifact)
    by_target = subsystem_members(_MembersModel(), artifact, criterion=ReactionDirectionCriterion.DOMINANT_IN_TARGET)

    # R1 counts once, in the direction more active in the population named.
    assert by_baseline == {"glyc": ("R1", "R2"), "tca": ("R5", "R6")}
    assert by_target == {"glyc": ("R1_back", "R2"), "tca": ("R5", "R6")}
    with_genes = resized(ARTIFACTS["A"], genes)
    assert subsystem_members(_MembersModel(), with_genes, for_genes=True) == {"glyc": ("G1", "G2"), "tca": ("G2",)}

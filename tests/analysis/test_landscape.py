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
    build_landscape,
    marker_reactions,
    select_landscape_subsystems,
    subsystem_delta,
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


def test_landscape_places_every_group_side_by_side():
    landscape = build_landscape(_Model(), _Reactions(), _thresholded())

    assert list(landscape.delta.columns) == ["A", "B", "C"]
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
    from metabis.analysis.landscape import enrichment_landscape, select_enriched_subsystems

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

"""Every group against the rest: Delta by subsystem and group, and marker reactions.

Comparing each group with all remaining samples and reading the results side by
side is the layout of Xu et al. (Cell Rep Med 2026, Figs. 3A-C); markers found
one group against the rest are standard in single-cell tools (Hao et al., Cell
2021). A group-by-pathway table is the core figure of single-cell metabolic
studies (Xiao, Dai and Locasale, Nat Commun 2019).

Each group's Delta is the overview's own: a :class:`SubsystemAnalysis` reads the
group's stored comparison, so a heatmap cell and the overview's bar for the same
result are the same number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .comparison_artifact import ComparisonThresholds, GroupComparisonArtifact
from .statistical_analysis import StatisticalAnalysis
from .subsystem_analysis import SubsystemAnalysis

__all__ = [
    "DEFAULT_LANDSCAPE_DELTA",
    "EnrichmentLandscape",
    "Landscape",
    "LandscapeRule",
    "build_landscape",
    "enrichment_landscape",
    "marker_reactions",
    "select_enriched_subsystems",
    "select_landscape_subsystems",
    "select_shared_subsystems",
    "subsystem_delta",
    "with_thresholds",
]

# Metabis's default Delta: the signs of the significant reactions' Cohen's d,
# summed and divided by every reaction of the subsystem.
DEFAULT_LANDSCAPE_DELTA: dict[str, Any] = {
    "metric": "cohens_d",
    "transform": "sign",
    "numerator_filter": "significant",
    "denominator_filter": "all",
}

LandscapeRule = Literal["spread", "category", "chosen"]


def _decide(artifact, for_genes, thresholds, keep_reversed):
    results = artifact.get_test_results(for_genes=for_genes)
    analyzer = StatisticalAnalysis(test_type=artifact.test, correction_type=artifact.correction)
    mask, _info = analyzer.determine_significant_elements(
        results,
        pval_threshold=thresholds.get("pval"),
        cohens_d_threshold=thresholds.get("cohens_d"),
        log2_fc_threshold=thresholds.get("log2_fc"),
        mean_difference_threshold=thresholds.get("mean_difference"),
        keep_reversed=keep_reversed,
    )
    results["is_significant"] = mask.to_numpy()
    significant = tuple(str(item) for item in results.index[mask.to_numpy()])
    return results, ComparisonThresholds.capture(thresholds), significant


def with_thresholds(
    artifact: GroupComparisonArtifact,
    thresholds: Mapping[str, float | None],
    gene_thresholds: Mapping[str, float | None] | None = None,
    *,
    keep_reversed: bool = False,
) -> GroupComparisonArtifact:
    """The artifact with its significance decided at the given thresholds.

    The copy is detached: it keeps the source revisions and is read, never
    installed. Each mapping takes ``pval``, ``cohens_d``, ``log2_fc`` and
    ``mean_difference`` to a value, or to ``None`` to leave one unset. Gene
    results keep their current calls unless ``gene_thresholds`` is given.
    """
    reactions, reaction_thresholds, significant_reactions = _decide(
        artifact, False, thresholds, keep_reversed
    )
    genes = artifact.get_test_results(for_genes=True)
    gene_state = (genes, artifact.gene_thresholds, tuple(artifact.significant_genes))
    if genes is not None and gene_thresholds is not None:
        gene_state = _decide(artifact, True, gene_thresholds, keep_reversed)
    return artifact.with_threshold_results(
        revisions=artifact.metadata.source.revisions,
        reaction_results=reactions,
        reaction_thresholds=reaction_thresholds,
        significant_reactions=significant_reactions,
        gene_results=gene_state[0],
        gene_thresholds=gene_state[1],
        significant_genes=gene_state[2],
        keep_reversed=keep_reversed,
    )


@dataclass(frozen=True)
class _DeltaInputs:
    """What a SubsystemAnalysis reads off its workspace, with no colour registry.

    A detached analysis must not register colour scales, or the landscape would
    repaint the overview of whichever result is in view.
    """

    metabolic_model: Any
    reaction_data: Any
    expression_data: Any = None
    color_manager: Any = None

    def get_group_comparison_artifact(self):
        return None


def subsystem_delta(
    metabolic_model,
    reaction_data,
    artifact: GroupComparisonArtifact,
    *,
    delta_config: Mapping[str, Any] | None = None,
    for_genes: bool = False,
    expression_data=None,
) -> pd.DataFrame:
    """Delta, size and significant count of every subsystem for one comparison.

    Rows are subsystem IDs; columns ``name``, ``category``, ``reactions``,
    ``significant`` and ``delta``. With ``for_genes`` the subsystem's genes are
    read instead, as the overview's gene overlay reads them, from
    ``expression_data``; ``reactions`` and ``significant`` then count genes.
    Subsystems with no reaction in the data are left out.
    """
    analysis = SubsystemAnalysis(
        _DeltaInputs(metabolic_model, reaction_data, expression_data),
        default_delta_config=dict(delta_config or DEFAULT_LANDSCAPE_DELTA),
    )
    analysis.set_group_comparison(artifact)
    rows = {}
    for subsystem in metabolic_model.get_subsystems() or []:
        stats = analysis.get_subsystem_statistics(subsystem, for_genes=for_genes)
        if not stats.get("is_in_model"):
            continue
        rows[str(subsystem)] = {
            "name": str(metabolic_model.get_subsystem_name(subsystem) or subsystem),
            "category": str(metabolic_model.get_subsystem_category(subsystem) or ""),
            "reactions": int(stats.get("total_elements", 0)),
            "significant": int(stats.get("significant_elements", 0)),
            "delta": float(stats.get("delta_subsystem_score") or 0.0),
        }
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=["name", "category", "reactions", "significant", "delta"]
    )


@dataclass(frozen=True)
class Landscape:
    """Delta for every subsystem (rows) and group (columns), with each row's facts.

    ``significant`` counts each subsystem's significant reactions per group;
    ``significant_reactions`` counts them over the whole model per group. In a
    gene landscape both count genes, and ``reactions`` the subsystem's genes.
    """

    delta: pd.DataFrame
    significant: pd.DataFrame
    names: pd.Series
    categories: pd.Series
    reactions: pd.Series
    significant_reactions: pd.Series


def build_landscape(
    metabolic_model,
    reaction_data,
    artifacts: Mapping[str, GroupComparisonArtifact],
    *,
    delta_config: Mapping[str, Any] | None = None,
    for_genes: bool = False,
    expression_data=None,
) -> Landscape:
    """Delta by subsystem for each group's comparison, side by side.

    ``artifacts`` maps a group's label to its comparison against the rest, with
    significance already decided (see :func:`with_thresholds`). With
    ``for_genes`` the subsystems' genes are read from ``expression_data``.
    """
    if not artifacts:
        raise ValueError("The landscape needs at least one comparison")
    tables = {
        label: subsystem_delta(
            metabolic_model,
            reaction_data,
            artifact,
            delta_config=delta_config,
            for_genes=for_genes,
            expression_data=expression_data,
        )
        for label, artifact in artifacts.items()
    }
    first = next(iter(tables.values()))
    delta = pd.DataFrame({label: table["delta"] for label, table in tables.items()})
    delta = delta.reindex(first.index).fillna(0.0)
    significant = pd.DataFrame(
        {label: table["significant"] for label, table in tables.items()}
    ).reindex(first.index).fillna(0).astype(int)
    return Landscape(
        delta=delta,
        significant=significant,
        names=first["name"],
        categories=first["category"],
        reactions=first["reactions"],
        significant_reactions=pd.Series(
            {
                label: len(artifact.significant_genes if for_genes else artifact.significant_reactions)
                for label, artifact in artifacts.items()
            }
        ),
    )


def select_landscape_subsystems(
    landscape: Landscape,
    *,
    rule: LandscapeRule = "spread",
    top: int = 25,
    min_reactions: int = 5,
    category: str | None = None,
    chosen: Sequence[str] = (),
) -> list[str]:
    """The subsystems the heatmap shows, ordered by category and then name.

    ``spread`` removes each group's mean Delta over the eligible subsystems, then
    keeps the ``top`` subsystems whose remaining values vary most across groups.
    Without that step a group that is higher everywhere (epithelium against
    immune cells) fills the ranking with its overall level. ``category`` keeps
    every eligible subsystem of one category; ``chosen`` keeps the given IDs.
    Subsystems with fewer than ``min_reactions`` reactions swing furthest on
    chance and are left out of the first two rules.
    """
    if top < 1:
        raise ValueError("Show at least one subsystem")
    if min_reactions < 1:
        raise ValueError("The minimum size is at least one reaction")
    eligible = landscape.reactions.index[landscape.reactions >= min_reactions]
    if rule == "spread":
        block = landscape.delta.loc[eligible]
        if block.empty:
            return []
        centred = block - block.mean(axis=0)
        spread = centred.var(axis=1, ddof=1) if block.shape[1] > 1 else centred.abs().iloc[:, 0]
        picked = list(spread.sort_values(ascending=False, kind="mergesort").index[:top])
    elif rule == "category":
        if not category:
            raise ValueError("Choose a category")
        picked = [item for item in eligible if landscape.categories[item] == category]
    elif rule == "chosen":
        unknown = [item for item in chosen if item not in landscape.delta.index]
        if unknown:
            raise ValueError(f"No reactions in the data for {', '.join(unknown)}")
        picked = list(dict.fromkeys(chosen))
    else:
        raise ValueError(f"Unknown subsystem rule {rule!r}")
    return sorted(picked, key=lambda item: (landscape.categories[item], landscape.names[item]))


def _spread(landscape: Landscape, rows: Sequence[str]) -> pd.Series:
    """How much each row's Delta varies across the groups, each group's mean removed."""
    block = landscape.delta.loc[list(rows)]
    if block.empty:
        return pd.Series(dtype=float)
    centred = block - block.mean(axis=0)
    return centred.var(axis=1, ddof=1) if block.shape[1] > 1 else centred.abs().iloc[:, 0]


def select_shared_subsystems(
    landscapes: Sequence[Landscape],
    *,
    rule: LandscapeRule = "spread",
    top: int = 25,
    min_reactions: int = 5,
    category: str | None = None,
    chosen: Sequence[str] = (),
) -> list[str]:
    """One row choice for landscapes read side by side, reactions beside genes.

    A row must be in every landscape. ``spread`` takes the subsystems varying
    most in each landscape in turn, its first, then the next one's first, and so
    on, since a family whose Deltas swing wider would otherwise fill every row.
    ``category`` keeps every subsystem of one category that is large enough in
    any landscape; ``chosen`` keeps the given IDs. Rows are ordered by category
    and then name, as :func:`select_landscape_subsystems` orders them.
    """
    if not landscapes:
        raise ValueError("Choose at least one landscape")
    if top < 1:
        raise ValueError("Show at least one subsystem")
    if min_reactions < 1:
        raise ValueError("The minimum size is at least one feature")
    first = landscapes[0]
    common = [item for item in first.delta.index if all(item in other.delta.index for other in landscapes[1:])]
    large = [item for item in common if any(other.reactions[item] >= min_reactions for other in landscapes)]
    if rule == "spread":
        rankings = []
        for landscape in landscapes:
            eligible = [item for item in common if landscape.reactions[item] >= min_reactions]
            spread = _spread(landscape, eligible)
            rankings.append(list(spread.sort_values(ascending=False, kind="mergesort").index))
        picked: list[str] = []
        for position in range(max((len(ranking) for ranking in rankings), default=0)):
            for ranking in rankings:
                if position < len(ranking) and ranking[position] not in picked and len(picked) < top:
                    picked.append(ranking[position])
    elif rule == "category":
        if not category:
            raise ValueError("Choose a category")
        picked = [item for item in large if first.categories[item] == category]
    elif rule == "chosen":
        unknown = [item for item in chosen if item not in common]
        if unknown:
            raise ValueError(f"No features in the data for {', '.join(unknown)}")
        picked = list(dict.fromkeys(chosen))
    else:
        raise ValueError(f"Unknown subsystem rule {rule!r}")
    return sorted(picked, key=lambda item: (first.categories[item], first.names[item]))


@dataclass(frozen=True)
class EnrichmentLandscape:
    """Over-representation for every tested subsystem (rows) and group (columns).

    ``enrichment`` is the signed log2 enrichment ratio of the significant
    direction, positive among the features higher in the group and negative
    among the lower, and NaN when neither is significant or the subsystem was
    not tested in that comparison; ``adjusted_pval`` is that direction's, or the
    smaller one when neither is significant; ``tested`` says whether the
    subsystem was large enough to be tested.
    """

    enrichment: pd.DataFrame
    adjusted_pval: pd.DataFrame
    tested: pd.DataFrame
    names: pd.Series
    categories: pd.Series


def enrichment_landscape(results: Mapping[str, pd.DataFrame]) -> EnrichmentLandscape:
    """Each group's paired over-representation, side by side.

    ``results`` maps a group's label to its ORA result table (a set-enrichment
    artifact's ``result_frame``): two rows per tested subsystem, one for the
    features higher in the group and one for the lower.
    """
    if not results:
        raise ValueError("The enrichment landscape needs at least one comparison")
    names: dict[str, str] = {}
    categories: dict[str, str] = {}
    enrichment: dict[str, dict[str, float]] = {}
    adjusted: dict[str, dict[str, float]] = {}
    for label, frame in results.items():
        enrichment[label], adjusted[label] = {}, {}
        for subsystem, rows in frame.groupby("subsystem_id", sort=False):
            subsystem = str(subsystem)
            names[subsystem] = str(rows["subsystem_name"].iloc[0])
            categories[subsystem] = str(rows["category"].iloc[0] or "")
            significant = rows[rows["is_significant"].astype(bool)]
            if len(significant):
                strongest = significant.iloc[significant["signed_log2_enrichment_ratio"].abs().argmax()]
                enrichment[label][subsystem] = float(strongest["signed_log2_enrichment_ratio"])
                adjusted[label][subsystem] = float(strongest["adjusted_pval"])
            else:
                enrichment[label][subsystem] = float("nan")
                adjusted[label][subsystem] = float(rows["adjusted_pval"].min())
    index = sorted(names, key=lambda item: (categories[item], names[item]))
    values = pd.DataFrame(enrichment).reindex(index)
    pvalues = pd.DataFrame(adjusted).reindex(index)
    tested = pd.DataFrame({label: [item in enrichment[label] for item in index] for label in results}, index=index)
    return EnrichmentLandscape(
        enrichment=values[list(results)],
        adjusted_pval=pvalues[list(results)],
        tested=tested,
        names=pd.Series(names).reindex(index),
        categories=pd.Series(categories).reindex(index),
    )


def select_enriched_subsystems(landscapes: Sequence[EnrichmentLandscape]) -> list[str]:
    """Every subsystem significant in some group of some landscape, by category and name."""
    picked: dict[str, tuple[str, str]] = {}
    for landscape in landscapes:
        for item in landscape.enrichment.index[landscape.enrichment.notna().any(axis=1)]:
            picked[item] = (landscape.categories[item], landscape.names[item])
    return sorted(picked, key=lambda item: picked[item])


def marker_reactions(artifacts: Mapping[str, GroupComparisonArtifact]) -> pd.DataFrame:
    """Reactions that mark one group: significant, higher, and highest there.

    A reaction is a group's marker when it is significant and higher in that
    group than in the rest, and the group's Cohen's d is the largest of all
    groups. Markers are ranked by the margin over the next group's d, so a wide
    lead comes first. ``artifacts`` carry significance already decided.
    """
    effects = pd.DataFrame(
        {
            label: artifact.get_test_results(for_genes=False)["cohens_d"].astype(float)
            for label, artifact in artifacts.items()
        }
    )
    pvalues = pd.DataFrame(
        {
            label: artifact.get_test_results(for_genes=False)["adjusted_pval"].astype(float)
            for label, artifact in artifacts.items()
        }
    ).reindex(effects.index)
    labels = {}
    for artifact in artifacts.values():
        labels.update(dict(artifact.reaction_display_labels))
    filled = effects.fillna(-np.inf)
    best = filled.idxmax(axis=1)
    rows = []
    for label, artifact in artifacts.items():
        significant = set(artifact.significant_reactions)
        mine = [
            feature
            for feature in effects.index
            if feature in significant and effects.at[feature, label] > 0 and best[feature] == label
        ]
        for feature in mine:
            others = filled.loc[feature].drop(label)
            next_group = str(others.idxmax()) if len(others) else ""
            next_effect = float(others.max()) if len(others) else float("nan")
            rows.append(
                {
                    "group": label,
                    "feature_id": str(feature),
                    "display_label": labels.get(str(feature), str(feature)),
                    "cohens_d": float(effects.at[feature, label]),
                    "adjusted_pval": float(pvalues.at[feature, label]),
                    "next_group": next_group,
                    "next_cohens_d": next_effect,
                    "margin": float(effects.at[feature, label]) - next_effect,
                }
            )
    table = pd.DataFrame(
        rows,
        columns=[
            "group",
            "feature_id",
            "display_label",
            "cohens_d",
            "adjusted_pval",
            "next_group",
            "next_cohens_d",
            "margin",
        ],
    )
    order = {label: position for position, label in enumerate(artifacts)}
    return (
        table.assign(_order=table["group"].map(order))
        .sort_values(["_order", "margin"], ascending=[True, False], kind="mergesort")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

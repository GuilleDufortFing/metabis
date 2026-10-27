"""Reactions against their genes, within one comparison.

A reaction can move with its genes, which the flux model then confirms, or move
while none of its genes does, because the change reaches it through the
network. Genes can also move with no reaction change when network constraints
absorb it. METAFlux compared predicted fluxes with expression (Huang et al., Nat
Commun 2023), and Delta-FBA links flux differences to differential expression
(Ravi and Gunawan, PLoS Comput Biol 2021).
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .comparison_artifact import GroupComparisonArtifact
from .reaction_direction import to_base_reaction_id

__all__ = ["CONCORDANCE_CLASSES", "reaction_gene_concordance"]

CONCORDANCE_CLASSES: dict[str, str] = {
    "follows": "Reaction changes with its genes",
    "network": "Reaction changes, its genes do not",
    "against": "Reaction changes against its genes",
    "genes": "Genes change, reaction does not",
    "neither": "Neither changes",
}


def reaction_gene_concordance(
    artifact: GroupComparisonArtifact,
    reactions_for_gene: Callable[[str], Iterable[str]],
) -> pd.DataFrame:
    """Classify every reaction with a tested gene against its genes' results.

    A reaction "follows" its genes when it is significant and at least one of
    its genes is significant with the same sign; "against" when its significant
    genes all move the other way; "network" when none of its genes is
    significant. Each row also carries the Cohen's d of the gene with the largest
    effect, for the scatter. Reactions with no tested gene are left out.
    """
    reactions = artifact.get_test_results(for_genes=False)
    genes = artifact.get_test_results(for_genes=True)
    if genes is None:
        raise ValueError("Load expression data and compare again to set reactions against genes")
    gene_effects = genes["cohens_d"].astype(float)
    significant_genes = set(artifact.get_significant_elements(for_genes=True))
    significant_reactions = set(artifact.get_significant_elements(for_genes=False))

    genes_of: dict[str, list[str]] = {}
    for gene in genes.index:
        for reaction in reactions_for_gene(str(gene)) or ():
            genes_of.setdefault(str(reaction), []).append(str(gene))

    rows = []
    for column in reactions.index:
        tested = genes_of.get(to_base_reaction_id(str(column)), [])
        if not tested:
            continue
        effects = gene_effects.loc[tested]
        strongest = str(effects.abs().fillna(-1.0).idxmax())
        reaction_effect = float(reactions.at[column, "cohens_d"])
        moving = [gene for gene in tested if gene in significant_genes]
        with_reaction = [
            gene for gene in moving if np.sign(gene_effects[gene]) == np.sign(reaction_effect)
        ]
        if str(column) in significant_reactions:
            status = "follows" if with_reaction else "against" if moving else "network"
        else:
            status = "genes" if moving else "neither"
        rows.append(
            {
                "feature_id": str(column),
                "reaction_effect": reaction_effect,
                "gene_effect": float(gene_effects[strongest]),
                "strongest_gene": strongest,
                "genes": len(tested),
                "significant_genes": len(moving),
                "status": status,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "feature_id",
            "reaction_effect",
            "gene_effect",
            "strongest_gene",
            "genes",
            "significant_genes",
            "status",
        ],
    )

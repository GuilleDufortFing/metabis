"""Notebook violin figures state the Target/Baseline orientation (D-51).

`plot_reaction_violins` is a root export, so a notebook user reaches it without
the server's labels.  These pin two things the WP7 language pass fixed: the
helper defaults no longer say "Group 1"/"Group 2", and the grid names each
violin by the side of the installed comparison it actually holds.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from bokeh.models import Plot

from metabis import (
    AnalysisWorkspace,
    GroupComparison,
    GroupComparisonSpec,
    MetabolicModel,
    ReactionData,
    ReactionPreprocessingConfig,
    plot_reaction_violins,
)
from metabis.viz import plot_violins

SUBSYSTEM_ID = "galactose_metabolism"
TARGET_LABEL = "Th17p cells"
BASELINE_LABEL = "Th17n cells"


@pytest.mark.parametrize(
    "function",
    [
        plot_violins.create_violin,
        plot_violins.create_html_violin_for_hover,
        plot_violins.create_static_violin_for_hover,
        plot_violins.create_matplotlib_violin_for_hover,
    ],
)
def test_violin_helpers_default_to_target_and_baseline(function):
    parameters = inspect.signature(function).parameters

    assert parameters["group1_name"].default == "Target"
    assert parameters["group2_name"].default == "Baseline"


def test_a_constant_side_renders_instead_of_raising():
    """A reversible reaction carrying no reverse flux is an all-zero side.

    `gaussian_kde` raises on its singular covariance, so the grid used to die on
    ordinary single-value flux data before any figure was returned.
    """
    plot = plot_violins.create_violin(
        dist1=[0.0, 0.0, 0.0, 0.0],
        dist2=[1.0, 2.0, 3.0, 4.0],
        color1="#d62728",
        color2="#1f77b4",
    )

    assert plot.xaxis[0].major_label_overrides == {1: "Target", 2: "Baseline"}


@pytest.fixture(scope="module")
def model() -> MetabolicModel:
    return MetabolicModel()


@pytest.fixture
def comparison_workspace(model, tmp_path_factory, install_comparison):
    """A real workspace with one labelled pairwise comparison installed."""
    reactions = model.get_reactions_for_subsystem(SUBSYSTEM_ID)
    samples = [f"A_{index}" for index in range(4)] + [
        f"B_{index}" for index in range(4)
    ]

    rng = np.random.default_rng(19)
    frame = pd.DataFrame(
        rng.normal(5.0, 2.0, size=(len(samples), len(reactions))),
        index=samples,
        columns=reactions,
    )
    frame.loc[frame.index[:4], reactions[0]] += 20.0

    path = tmp_path_factory.mktemp("violins") / "reactions.tsv"
    frame.to_csv(path, sep="\t")

    manager = AnalysisWorkspace(
        metabolic_model=model,
        reaction_data=ReactionData(
            str(path),
            "single",
            model,
            csv_separator="\t",
            samples_in="rows",
            preprocessing=ReactionPreprocessingConfig(),
        ),
    )
    manager.load_groups(criteria="sample_prefix")
    membership = manager.group_manager.resolve_comparison_membership(
        GroupComparisonSpec.pairwise(
            ("A",),
            ("B",),
            target_label=TARGET_LABEL,
            baseline_label=BASELINE_LABEL,
        ),
        list(frame.index),
    )
    comparison = GroupComparison(membership=membership)
    comparison.run_test(manager, test="ttest", correction="holm")

    artifact = install_comparison(manager, comparison)
    return manager, artifact


@pytest.mark.integration
def test_reaction_violins_name_each_side_of_the_installed_comparison(
    comparison_workspace,
):
    manager, artifact = comparison_workspace
    subsystem_label = artifact.get_test_results(for_genes=False)["subsystems"].iloc[0]

    grid = plot_reaction_violins(
        manager,
        subsystem=subsystem_label,
        return_plot=True,
    )

    figures = [
        model
        for model in grid.references()
        if isinstance(model, Plot) and model.xaxis
    ]
    assert figures, "the subsystem produced no violin plots"
    for plot in figures:
        assert plot.xaxis[0].major_label_overrides == {
            1: f"Target: {TARGET_LABEL}",
            2: f"Baseline: {BASELINE_LABEL}",
        }

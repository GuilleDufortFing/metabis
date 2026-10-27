"""`build_reduction_presentation` from a real loaded workspace (AM7 slice 4d).

The artifact-first reduction factories take a frozen ``ReductionPlotPresentation``
and never touch a manager. This is the production seam that builds that
presentation on the document thread, so the test drives the whole thing from a
workspace loaded the way the app loads one -- real model, real reaction frame,
real groups -- rather than a hand-built fixture (MR-12), and proves the
presentation the builder returns is one the four factories actually accept.
"""

from __future__ import annotations

import pandas as pd
import pytest
from bokeh.models import Plot

from metabis import (
    AnalysisWorkspace,
    ExperimentAnalysis,
    MetabolicModel,
    build_reduction_presentation,
    plot_pca_biplot_artifact,
    plot_pca_loadings_artifact,
    plot_pca_variance_artifact,
    plot_reduction_artifact,
)
from metabis.analysis_workspace import RevisionKind
from metabis.managers.group_manager import GroupDefinition

SAMPLES = ["s1", "s2", "s3", "s4", "s5", "s6"]


class _ReactionData:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def get_reaction_df(self) -> pd.DataFrame:
        return self.frame

    def get_sample_ids(self):
        return list(self.frame.index)


@pytest.fixture(scope="module")
def model() -> MetabolicModel:
    return MetabolicModel(use_cache=True)


def _loaded_workspace(model: MetabolicModel) -> tuple[AnalysisWorkspace, tuple[str, ...]]:
    reactions = tuple(model.get_reactions()[:8])
    frame = pd.DataFrame(
        {
            reaction: [float((i + col) % 5) + 0.25 * col for i in range(len(SAMPLES))]
            for col, reaction in enumerate(reactions)
        },
        index=SAMPLES,
    )
    manager = AnalysisWorkspace(metabolic_model=model)
    manager.replace_reaction_data(_ReactionData(frame), metabolic_model=model)
    analysis = ExperimentAnalysis(manager, auto_run=False)
    provenance = manager.capture_analysis_snapshot().provenance
    manager.install_experiment_analysis(provenance, analysis)
    manager.refresh_experiment_analysis_presentation()
    # Two groups; the second names a sample the matrix does not carry, which the
    # builder must drop before the factories reject the unknown member.
    definition = GroupDefinition(source="custom")
    manager.group_manager.define_group(
        "healthy", ["s1", "s2", "s3"], definition=definition
    )
    manager.group_manager.define_group(
        "disease", ["s4", "s5", "s6", "sX_not_in_matrix"], definition=definition
    )
    return manager, reactions


def _artifact(manager: AnalysisWorkspace):
    outcome = manager.experiment_analysis.calculate_pca(
        data_type="reactions", n_variable_features=100
    )
    assert outcome["status"] == "success", outcome["messages"]
    result = manager.experiment_analysis.get_reduction_results()[-1]
    provenance = manager.capture_analysis_snapshot(
        {RevisionKind.DATASET, RevisionKind.PREPROCESSING}
    ).provenance
    return result.build_artifact(
        provenance=provenance,
        source_fingerprints=dict(manager.source_fingerprints),
    )


def test_presentation_aligns_with_the_artifact_and_all_factories_accept_it(model):
    manager, _reactions = _loaded_workspace(model)
    artifact = _artifact(manager)

    presentation = build_reduction_presentation(manager, artifact, width=500, height=320)

    # Samples: exactly the artifact's, in order, with display labels.
    assert tuple(s.sample_id for s in presentation.samples) == artifact.sample_ids
    assert all(s.display_label for s in presentation.samples)
    # Features: exactly the artifact's, in order, with real reference labels.
    assert tuple(f.feature_id for f in presentation.features) == artifact.feature_names
    assert all(f.display_label for f in presentation.features)
    # The unknown group member was dropped, so no group references a sample the
    # matrix does not carry.
    known = set(artifact.sample_ids)
    assert {g.group_id for g in presentation.groups} == {"healthy", "disease"}
    for group in presentation.groups:
        assert set(group.sample_ids) <= known
        assert group.color
    assert "sX_not_in_matrix" not in {
        sample_id for g in presentation.groups for sample_id in g.sample_ids
    }

    # Alignment is proven by the factories accepting it (they validate strictly).
    for factory in (
        plot_reduction_artifact,
        plot_pca_variance_artifact,
        plot_pca_loadings_artifact,
        plot_pca_biplot_artifact,
    ):
        assert isinstance(factory(artifact, presentation), Plot)


def test_feature_hover_is_built_when_requested_and_skipped_otherwise(model):
    manager, _reactions = _loaded_workspace(model)
    artifact = _artifact(manager)

    with_hover = build_reduction_presentation(
        manager, artifact, include_feature_hover=True
    )
    without_hover = build_reduction_presentation(
        manager, artifact, include_feature_hover=False
    )

    assert any(feature.hover_html for feature in with_hover.features)
    assert all(feature.hover_html == "" for feature in without_hover.features)
    # Reference labels and URLs are cheap reads, so they are present either way.
    assert all(feature.external_url for feature in without_hover.features)
    assert without_hover.features[0].display_label == with_hover.features[0].display_label


def test_knobs_pass_through_to_the_presentation(model):
    manager, _reactions = _loaded_workspace(model)
    artifact = _artifact(manager)

    presentation = build_reduction_presentation(
        manager,
        artifact,
        show_sample_labels=True,
        loading_component_count=3,
        loading_feature_count=4,
        loading_sort_component=2,
        biplot_pc_x=1,
        biplot_pc_y=3,
        biplot_top_arrows=2,
        biplot_arrow_scale=0.9,
    )

    assert presentation.show_sample_labels is True
    assert presentation.loading_component_count == 3
    assert presentation.loading_feature_count == 4
    assert presentation.loading_sort_component == 2
    assert presentation.biplot_pc_x == 1
    assert presentation.biplot_pc_y == 3
    assert presentation.biplot_top_arrows == 2
    assert presentation.biplot_arrow_scale == 0.9


def test_a_feature_colours_the_samples_with_its_own_values(model):
    manager, reactions = _loaded_workspace(model)
    artifact = _artifact(manager)
    frame = manager.reaction_data.get_reaction_df()

    presentation = build_reduction_presentation(manager, artifact, colour_by=("reactions", reactions[2]))

    colouring = presentation.colouring
    assert colouring is not None and colouring.feature_id == reactions[2]
    assert colouring.values == tuple(frame.loc[list(artifact.sample_ids), reactions[2]])
    assert colouring.display_label and colouring.display_label != reactions[2]
    assert isinstance(plot_reduction_artifact(artifact, presentation), Plot)
    with pytest.raises(KeyError, match="No reaction"):
        build_reduction_presentation(manager, artifact, colour_by=("reactions", "not-a-reaction"))

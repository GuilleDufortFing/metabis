from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from bokeh.models import ColumnDataSource, Plot

from metabis.analysis.artifacts import (
    AnalysisArtifactMetadata,
    ArtifactDimension,
    ArtifactKind,
    ArtifactProvenance,
)
from metabis.analysis.dimensionality_reduction import (
    PCAArtifacts,
    ReductionArtifact,
    ReductionKey,
)
from metabis.analysis_workspace import RevisionKind, WorkspaceRevisions
from metabis.viz.plot_reduction_artifact import (
    ReductionFeaturePresentation,
    ReductionGroupPresentation,
    ReductionPlotPresentation,
    ReductionSamplePresentation,
    plot_pca_biplot_artifact,
    plot_pca_loadings_artifact,
    plot_pca_variance_artifact,
    plot_reduction_artifact,
)


def artifact():
    key = ReductionKey(
        method="PCA",
        data_type="reactions",
        n_variable_features=3,
        n_components=3,
        random_state=42,
        preprocessing_revision="dataset=1|preprocessing=2",
        sample_revision="1",
    )
    metadata = AnalysisArtifactMetadata(
        kind=ArtifactKind.REDUCTION,
        source=ArtifactProvenance(
            workspace_id="workspace",
            revisions=WorkspaceRevisions(dataset=1, preprocessing=2),
            dependencies=frozenset(
                {RevisionKind.DATASET, RevisionKind.PREPROCESSING}
            ),
            source_fingerprints={"dataset": "sha256:frame"},
        ),
        method="PCA",
        configuration={
            "data_type": "reactions",
            "parameters": {
                "n_components": 3,
                "n_variable_features": 3,
                "random_state": 42,
            },
            "preprocessing_revision": "dataset=1|preprocessing=2",
            "sample_revision": "1",
        },
        scientific_identity={
            "sample_ids": ["s1", "s2", "s3", "s4"],
            "feature_ids": ["R1", "R2", "R3"],
            "source": "computed",
        },
        dimensions=(
            ArtifactDimension("samples", 4),
            ArtifactDimension("features", 3),
            ArtifactDimension("components", 3),
        ),
    )
    return ReductionArtifact(
        metadata=metadata,
        key=key,
        coordinates=np.array(
            [[-2.0, -1.0, 0.1], [-1.0, -2.0, 0.2], [1.0, 2.0, 0.3], [2.0, 1.0, 0.4]]
        ),
        sample_ids=("s1", "s2", "s3", "s4"),
        feature_names=("R1", "R2", "R3"),
        data_source_name="Compass reactions",
        n_variable_features_used=3,
        n_components_retained=3,
        pca=PCAArtifacts(
            explained_variance_ratio=np.array([0.6, 0.3, 0.1]),
            loadings=np.array(
                [[0.9, 0.2, -0.1], [0.1, -0.8, 0.4], [0.2, 0.1, 0.7]]
            ),
        ),
        params=(
            ("n_components", 3),
            ("n_variable_features", 3),
            ("random_state", 42),
        ),
    )


def presentation(**overrides):
    values = {
        "samples": tuple(
            ReductionSamplePresentation(sample_id, f"Label {sample_id}")
            for sample_id in ("s1", "s2", "s3", "s4")
        ),
        "groups": (
            ReductionGroupPresentation("g1", "Healthy", "#112233", ("s1", "s2")),
            ReductionGroupPresentation("g2", "Disease", "#445566", ("s3", "s4")),
        ),
        "features": tuple(
            ReductionFeaturePresentation(
                feature_id,
                f"Display {feature_id}",
                f"<div>Hover {feature_id}</div>",
                f"https://example.invalid/{feature_id}",
            )
            for feature_id in ("R1", "R2", "R3")
        ),
        "width": 500,
        "height": 300,
        "show_sample_labels": True,
        "loading_component_count": 2,
        "loading_feature_count": 2,
        "loading_sort_component": 1,
        "biplot_pc_x": 1,
        "biplot_pc_y": 2,
        "biplot_top_arrows": 2,
    }
    values.update(overrides)
    return ReductionPlotPresentation(**values)


def sources(plot):
    return list(plot.select({"type": ColumnDataSource}))


def source_with(plot, field):
    return next(source for source in sources(plot) if field in source.data)


@pytest.mark.parametrize(
    "factory",
    (
        plot_reduction_artifact,
        plot_pca_variance_artifact,
        plot_pca_loadings_artifact,
        plot_pca_biplot_artifact,
    ),
)
def test_all_reduction_factories_are_artifact_first_and_return_fresh_models(factory):
    result = artifact()
    snapshot = presentation()
    coordinates = result.coordinates.copy()

    first = factory(result, snapshot)
    second = factory(result, snapshot)

    assert isinstance(first, Plot)
    assert isinstance(second, Plot)
    assert first is not second
    assert first.width == 500
    assert first.height == 300
    np.testing.assert_array_equal(result.coordinates, coordinates)
    assert result.coordinates.flags.writeable is False
    assert tuple(inspect.signature(factory).parameters) == ("artifact", "presentation")


def test_scatter_preserves_canonical_ids_display_names_colors_and_labels():
    plot = plot_reduction_artifact(artifact(), presentation())
    sample_sources = [source for source in sources(plot) if "sample_id" in source.data]
    sources_by_group = {
        source.data["group_id"][0]: source for source in sample_sources
    }

    assert set(sources_by_group) == {"g1", "g2"}
    assert list(sources_by_group["g1"].data["sample_id"]) == ["s1", "s2"]
    assert list(sources_by_group["g2"].data["sample_id"]) == ["s3", "s4"]
    assert list(sources_by_group["g1"].data["display_label"]) == [
        "Label s1",
        "Label s2",
    ]
    assert list(sources_by_group["g1"].data["group_id"]) == ["g1", "g1"]
    assert list(sources_by_group["g1"].data["group"]) == ["Healthy", "Healthy"]
    colors = [
        renderer.glyph.fill_color
        for renderer in plot.renderers
        if hasattr(renderer, "glyph") and hasattr(renderer.glyph, "fill_color")
    ]
    assert "#112233" in colors
    assert "#445566" in colors
    assert any(
        getattr(renderer.glyph, "text", None) == "display_label"
        for renderer in plot.renderers
        if hasattr(renderer, "glyph")
    )


def test_variance_uses_exact_artifact_values_and_selected_component_limit():
    plot = plot_pca_variance_artifact(
        artifact(), presentation(max_variance_components=2)
    )
    source = source_with(plot, "explained_variance")

    np.testing.assert_allclose(source.data["explained_variance"], [0.6, 0.3])
    np.testing.assert_allclose(source.data["cumulative_variance"], [0.6, 0.9])
    assert "first 2 of 3 PCs" in plot.title.text


def test_loadings_name_the_strongest_of_each_sign_per_component():
    plot = plot_pca_loadings_artifact(artifact(), presentation())
    source = source_with(plot, "loading")
    rows = list(reversed(plot.y_range.factors))

    # Two components, each a header row then its strongest positive and negative.
    assert rows[0] == "PC1 (60.0%)"
    assert "PC2 (30.0%)" in rows
    by_component = {
        (pc, feature): loading
        for pc, feature, loading in zip(
            source.data["PC"], source.data["feature_id"], source.data["loading"]
        )
    }
    assert by_component == {
        ("PC1", "R1"): 0.9,
        ("PC1", "R3"): -0.1,
        ("PC2", "R3"): 0.4,
        ("PC2", "R2"): -0.8,
    }
    assert all(label.startswith("Display ") for label in source.data["display_label"])
    assert all("Hover R" in hover for hover in source.data["hover"])
    assert "first 2 PCs" in plot.title.text


def test_identical_loadings_share_one_bar_that_counts_its_twins():
    tied = artifact()
    tied = replace(
        tied,
        pca=replace(
            tied.pca,
            loadings=np.array([[0.5, 0.5, -0.2], [0.1, -0.8, 0.4], [0.2, 0.1, 0.7]]),
        ),
    )
    plot = plot_pca_loadings_artifact(tied, presentation(loading_component_count=1))
    source = source_with(plot, "loading")

    assert list(source.data["feature_id"]) == ["R1", "R3"]
    assert source.data["Feature"][0] == "Display R1 and 1 identical"
    assert source.data["twins"][0] == "Display R2"


def test_biplot_uses_canonical_samples_and_limits_feature_arrows():
    plot = plot_pca_biplot_artifact(artifact(), presentation())
    sample_sources = [source for source in sources(plot) if "sample_id" in source.data]
    arrow_source = source_with(plot, "magnitude")

    assert sorted(
        sample_id
        for source in sample_sources
        for sample_id in source.data["sample_id"]
    ) == ["s1", "s2", "s3", "s4"]
    assert len(arrow_source.data["feature_id"]) == 2
    assert set(arrow_source.data["feature_id"]).issubset({"R1", "R2", "R3"})
    assert all(value.startswith("Display ") for value in arrow_source.data["display_label"])
    assert plot.xaxis[0].axis_label == "PC1 (60.0%)"
    assert plot.yaxis[0].axis_label == "PC2 (30.0%)"


def sample_label_renderers(plot):
    """Label renderers naming samples, not the feature arrows they sit among.

    Both label families draw the ``display_label`` field, so the sample ones are
    identified by the source they share with the scatter they annotate.
    """
    return [
        renderer
        for renderer in plot.renderers
        if getattr(getattr(renderer, "glyph", None), "text", None) == "display_label"
        and "sample_id" in renderer.data_source.data
    ]


def test_sample_labels_name_only_the_points_a_box_selection_picks():
    labelled = plot_pca_biplot_artifact(artifact(), presentation())
    unlabelled = plot_pca_biplot_artifact(artifact(), presentation(show_sample_labels=False))

    (labels,) = labelled.select({"name": "selection_labels"})
    # Nothing is named until a box selection picks points; hovers name every sample.
    assert labels.data_source.data["display_label"] == []
    (callback,) = {
        id(callback): callback
        for source in sources(labelled)
        if "sample_id" in source.data
        for callback in source.selected.js_property_callbacks["change:indices"]
    }.values()
    assert callback.args["labels"] is labels.data_source
    assert any(tool.__class__.__name__ == "BoxSelectTool" for tool in labelled.tools)
    assert not unlabelled.select({"name": "selection_labels"})
    # The arrow tips keep their own labels either way.
    assert [
        renderer.data_source
        for renderer in unlabelled.renderers
        if getattr(getattr(renderer, "glyph", None), "text", None) == "display_label"
    ] == [source_with(unlabelled, "magnitude")]


def test_the_biplot_draws_one_arrow_per_direction():
    from metabis.viz.plot_reduction_artifact import biplot_arrows

    # Three arrows point the same way, one the opposite way, one is too short.
    x = np.array([1.0, 0.95, 0.9, -0.8, 0.05])
    y = np.array([0.1, 0.12, 0.08, -0.1, 0.0])

    assert biplot_arrows(x, y, None) == [0, 3]
    assert biplot_arrows(x, y, 1) == [0]


def test_presentation_is_frozen_and_must_exactly_align_with_artifact():
    snapshot = presentation()
    with pytest.raises(FrozenInstanceError):
        snapshot.width = 1

    mismatched = replace(
        snapshot,
        samples=(ReductionSamplePresentation("other", "Other"),) + snapshot.samples[1:],
    )
    with pytest.raises(ValueError, match="samples must exactly align"):
        plot_reduction_artifact(artifact(), mismatched)


def test_non_pca_or_invalid_component_requests_fail_without_calculation_fallback():
    pca = artifact()
    non_pca = replace(
        pca,
        key=replace(pca.key, method="UMAP"),
        metadata=replace(pca.metadata, method="UMAP"),
        pca=None,
    )

    with pytest.raises(ValueError, match="requires a PCA"):
        plot_pca_variance_artifact(non_pca, presentation())
    with pytest.raises(ValueError, match="Invalid PC selection"):
        plot_pca_biplot_artifact(
            pca, presentation(biplot_pc_x=1, biplot_pc_y=4)
        )


def test_every_reduction_hover_is_the_shared_panel():
    """The hover panel the graphs' reactions and genes have, on every reduction figure."""
    from bokeh.models import HoverTool

    from metabis.managers.hover.panel import PANEL_STYLE

    opening = f"<div style='{PANEL_STYLE}'>"
    scatter = plot_reduction_artifact(artifact(), presentation())
    (sample_hover,) = scatter.select({"type": HoverTool})
    assert sample_hover.tooltips.startswith(opening)
    # A sample names its group in the group's colour, and its place by component.
    assert "color:@color" in sample_hover.tooltips and "<strong>PC1:</strong>" in sample_hover.tooltips
    assert set(source_with(scatter, "sample_id").data["color"]) <= {"#112233", "#445566"}
    (variance_hover,) = plot_pca_variance_artifact(artifact(), presentation()).select({"type": HoverTool})
    assert variance_hover.tooltips.startswith(opening)

    # A loading is added to its feature's own panel, in place of a second one round it.
    loadings = plot_pca_loadings_artifact(artifact(), presentation())
    hovers = list(source_with(loadings, "loading").data["hover"])
    assert hovers and all(hover.startswith(opening) and "Loading on PC" in hover for hover in hovers)
    assert all(hover.count(opening) == 1 for hover in hovers)

    biplot = plot_pca_biplot_artifact(artifact(), presentation())
    arrows = list(source_with(biplot, "x_end").data["hover"])
    assert all(hover.startswith(opening) and "Loading on PC2" in hover for hover in arrows)
    for tool in biplot.select({"type": HoverTool}):
        assert tool.tooltips.startswith(opening) or tool.tooltips == "@hover{safe}"

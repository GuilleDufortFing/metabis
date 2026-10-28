from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from bokeh.models import ColumnDataSource, LinearColorMapper, Plot

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
    FADED_ALPHA,
    FADED_SIZE,
    FEATURE_CHOICE,
    GROUP_FOCUS,
    add_arrow_colouring,
    FEATURE_ROWS,
    ReductionColouring,
    ReductionFeaturePresentation,
    ReductionGroupPresentation,
    ReductionPlotPresentation,
    ReductionSamplePresentation,
    plot_pca_biplot_artifact,
    plot_pca_components_artifact,
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


def _cells(plot):
    source = source_with(plot, "loading")
    return {
        (feature, component.split(" ")[0]): loading
        for feature, component, loading in zip(
            source.data["feature_id"], source.data["component"], source.data["loading"]
        )
    }


def test_loadings_are_a_heatmap_of_the_strongest_features_on_the_first_components():
    from bokeh.models import LinearColorMapper

    plot = plot_pca_loadings_artifact(artifact(), presentation())

    # One column per component with its share of variance, one row per feature.
    assert list(plot.x_range.factors) == ["PC1 (60.0%)", "PC2 (30.0%)"]
    assert list(reversed(plot.y_range.factors)) == ["Display R1", "Display R2"]
    assert _cells(plot) == {("R1", "PC1"): 0.9, ("R1", "PC2"): 0.1, ("R2", "PC1"): 0.2, ("R2", "PC2"): -0.8}
    # On ColorBrewer's brown, white and teal, symmetric about zero: never the red
    # and blue of a change, since a loading's sign flips with its component.
    mapper = plot.select_one({"type": LinearColorMapper})
    assert (mapper.low, mapper.high) == (-0.9, 0.9)
    colours = [colour.lower() for colour in mapper.palette]
    assert (colours[0], colours[-1]) == ("#543005", "#003c30") and "#f5f5f5" in colours
    assert not {"#0000ff", "#ff0000"} & set(colours)
    assert "ordered by |PC1|" in plot.title.text


def test_loadings_are_ordered_by_one_component_or_by_their_total():
    by_pc3 = plot_pca_loadings_artifact(artifact(), presentation(loading_sort_component=3))
    assert list(reversed(by_pc3.y_range.factors)) == ["Display R3", "Display R1"]

    total = plot_pca_loadings_artifact(
        artifact(), presentation(loading_sort_component=None, loading_component_count=3, loading_feature_count=3)
    )
    # Absolute loadings summed over PC1 to PC3: 1.2, 1.1 and 1.2.
    assert list(reversed(total.y_range.factors)) == ["Display R1", "Display R3", "Display R2"]
    assert "ordered by their total over these components" in total.title.text


def test_identical_loadings_share_one_row_that_counts_its_twins():
    tied = artifact()
    tied = replace(
        tied,
        pca=replace(
            tied.pca,
            loadings=np.array([[0.5, 0.5, -0.2], [0.1, -0.8, 0.4], [0.2, 0.1, 0.7]]),
        ),
    )
    plot = plot_pca_loadings_artifact(tied, presentation(loading_component_count=1))

    assert list(reversed(plot.y_range.factors)) == ["Display R1 and 1 alike", "Display R3"]
    hover = source_with(plot, "loading").data["hover"][0]
    assert "Loading alike" in hover and "Display R2" in hover


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
    tip_labels = [
        renderer.data_source
        for renderer in unlabelled.renderers
        if getattr(getattr(renderer, "glyph", None), "text", None) == "tip_label"
    ]
    assert tip_labels and all("magnitude" in source.data and "sample_id" not in source.data for source in tip_labels)


def test_the_biplot_draws_one_arrow_per_direction():
    from metabis.viz.plot_reduction_artifact import biplot_arrows

    # Three arrows point the same way, one the opposite way, one is too short.
    x = np.array([1.0, 0.95, 0.9, -0.8, 0.05])
    y = np.array([0.1, 0.12, 0.08, -0.1, 0.0])

    assert biplot_arrows(x, y, None) == [0, 3]
    assert biplot_arrows(x, y, 1) == [0]


def test_of_features_loading_alike_but_for_rounding_the_first_is_drawn():
    from metabis.viz.plot_reduction_artifact import biplot_arrows

    # Scored alike, two features' loadings differ by rounding alone, the second by
    # a hair longer: the first is drawn, the one the components' row is named after.
    x, y = np.array([0.6, 0.6 + 2e-16, -0.2]), np.array([0.3, 0.3, 0.9])
    assert biplot_arrows(x, y, None) == [2, 0]


def test_a_feature_loading_as_an_arrow_does_colours_that_arrow_rather_than_adding_one():
    data = artifact()
    # R1 is longer than R2 by rounding alone, as the row's first member and the
    # feature clicked in it are in Compass's tied reactions.
    loadings = np.array([[0.9 + 2e-16, 0.9, -0.1], [0.1, 0.1, 0.4], [0.2, 0.5, 0.7]])
    tied = replace(data, pca=replace(data.pca, loadings=loadings))
    colouring = ReductionColouring("R2", "Display R2", "reactions", (1.0, 2.0, 3.0, 4.0))

    plot = plot_reduction_artifact(
        tied, presentation(show_biplot_arrows=True, colouring=colouring, show_sample_labels=False)
    )

    (endpoints,) = [r for r in plot.renderers if r.name == "arrow_endpoints"]
    ids = list(endpoints.data_source.data["feature_id"])
    # R1 and R2 are one arrow, named after R2, which colours the samples.
    assert "R2" in ids and "R1" not in ids and len(ids) == len(set(ids))
    names = [
        label
        for renderer in plot.renderers
        if getattr(getattr(renderer, "glyph", None), "text", None) == "tip_label"
        for label in renderer.data_source.data["tip_label"]
    ]
    assert [name for name in names if name.startswith(("Display R1", "Display R2"))] == ["Display R2 (+1)"]


def test_an_arrow_leading_features_that_load_alike_is_named_with_their_count():
    from metabis.viz.plot_reduction_artifact import arrow_ties, biplot_arrows

    # Two features load alike on both axes: one arrow, named "(+1)" at its tip.
    x, y = np.array([0.7, 0.7, 0.1]), np.array([0.2, 0.2, 0.9])
    chosen = biplot_arrows(x, y, None)
    assert chosen == [2, 0] and arrow_ties(x, y, chosen) == [0, 1]


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

    # A cell's loading is added to its feature's own panel, in place of a second one round it.
    loadings = plot_pca_loadings_artifact(artifact(), presentation())
    (cell_hover,) = loadings.select({"type": HoverTool})
    assert cell_hover.tooltips.startswith(opening) and "Loading on @component" in cell_hover.tooltips
    hovers = list(source_with(loadings, "loading").data["hover"])
    assert hovers and all(opening not in hover and "Hover R" in hover for hover in hovers)

    biplot = plot_pca_biplot_artifact(artifact(), presentation())
    arrows = list(source_with(biplot, "x_end").data["hover"])
    assert all(hover.startswith(opening) and "Loading on PC2" in hover for hover in arrows)
    for tool in biplot.select({"type": HoverTool}):
        assert tool.tooltips.startswith(opening) or tool.tooltips == "@hover{safe}"


def test_a_pca_scatter_draws_the_chosen_components_and_arrows_only_when_asked():
    plain = plot_reduction_artifact(artifact(), presentation(biplot_pc_x=2, biplot_pc_y=3))
    arrows = plot_reduction_artifact(
        artifact(), presentation(biplot_pc_x=2, biplot_pc_y=3, show_biplot_arrows=True)
    )

    healthy = next(s for s in sources(plain) if "sample_id" in s.data and s.data["group_id"][0] == "g1")
    assert list(healthy.data["x"]) == [-1.0, -2.0]
    assert list(healthy.data["y"]) == [0.1, 0.2]
    assert plain.xaxis[0].axis_label == "PC2 (30.0%)"
    assert plain.yaxis[0].axis_label == "PC3 (10.0%)"
    assert not any("x_end" in s.data for s in sources(plain))
    assert len(source_with(arrows, "x_end").data["feature_id"]) == 2


def test_a_coloured_scatter_reads_on_the_sample_heatmaps_scale():
    # s4 is in no group: the scale is the grouped samples', as the heatmap's is,
    # and s4 lies far off it, clipped at three.
    groups = (
        ReductionGroupPresentation("g1", "Healthy", "#112233", ("s1", "s2")),
        ReductionGroupPresentation("g2", "Disease", "#445566", ("s3",)),
    )
    colouring = ReductionColouring("R9", "Lactate dehydrogenase", "reactions", (1.0, 2.0, 3.0, 10.0))

    plot = plot_reduction_artifact(artifact(), presentation(groups=groups, colouring=colouring))

    points = source_with(plot, "z")
    # Extremes last, so they are drawn over the rest.
    assert list(points.data["sample_id"]) == ["s2", "s1", "s3", "s4"]
    np.testing.assert_allclose(points.data["z"], [0.0, -1.0, 1.0, 3.0])
    assert list(points.data["group"]) == ["Healthy", "Healthy", "Disease", "Unassigned"]
    assert plot.title.text.endswith("coloured by Lactate dehydrogenase")
    assert not plot.legend or not plot.legend[0].items
    (bar,) = [item for item in plot.right if type(item).__name__ == "ColorBar"]
    assert bar.title == "Standardized score"
    assert (bar.color_mapper.low, bar.color_mapper.high) == (-3.0, 3.0)
    with pytest.raises(ValueError, match="one value per artifact sample"):
        plot_reduction_artifact(artifact(), presentation(colouring=replace(colouring, values=(1.0,))))


def test_the_components_figure_sets_variance_bars_over_the_loadings_columns():
    plot = plot_pca_components_artifact(
        artifact(),
        presentation(loading_component_count=3, loading_feature_count=3, loading_sort_component=None),
        plotted=(1, 3),
        feature_event="pick-feature",
        component_event="pick-component",
    )

    assert list(plot.x_range.factors) == ["PC1", "PC2", "PC3"]
    bars = source_with(plot, "top")
    # Bars stand on the rows, each in proportion to the first component's share.
    np.testing.assert_allclose(bars.data["top"], [3.5 + 5.0, 3.5 + 2.5, 3.5 + 5.0 / 6])
    assert list(bars.data["share"]) == ["60.0%", "30.0%", "10.0%"]
    cells = source_with(plot, "loading")
    assert len(cells.data["loading"]) == 9
    assert sorted(plot.yaxis[0].major_label_overrides.values()) == ["Display R1", "Display R2", "Display R3"]
    # PC1 and PC3 outlined in the logo's gradient, as short segments over each
    # column; the chosen feature's row empty until a click outlines it.
    choice = next(source for source in sources(plot) if source.name == FEATURE_CHOICE)
    (redraw,) = choice.js_property_callbacks["change:data"]
    assert list(choice.data["y"]) == []
    # The outline is redrawn from the rows the choice now holds, read from the
    # changed source itself: the callback's arguments are only the outline's.
    assert "cb_obj.data.y" in redraw.code and set(redraw.args) == {"outline", "left", "right"}
    boxes = sorted(
        (round(min(source.data["x0"]), 2), round(max(source.data["x0"]), 2))
        for source in sources(plot)
        if "x0" in source.data and len(source.data["x0"])
    )
    assert boxes == [(0.02, 0.98), (2.02, 2.98)]
    from metabis.viz.brand import gradient_color

    colours = {colour for source in sources(plot) if "x0" in source.data for colour in source.data["color"]}
    assert colours <= {gradient_color(step / 1000) for step in range(1001)}
    rows = next(tag for tag in plot.tags if getattr(tag, "name", None) == FEATURE_ROWS)
    assert sorted(rows.data["feature"]) == ["R1", "R2", "R3"]
    (callback,) = plot.js_event_callbacks["tap"]
    assert callback.args["feature_event"] == "pick-feature"
    assert callback.args["component_event"] == "pick-component"
    assert callback.args["data_type"] == "reactions"


def test_the_components_figure_names_long_rows_short_and_counts_alike_features():
    twins = replace(
        artifact(),
        pca=PCAArtifacts(
            explained_variance_ratio=np.array([0.6, 0.3, 0.1]),
            loadings=np.array([[0.9, 0.9, -0.1], [0.1, 0.1, 0.4], [0.2, 0.2, 0.7]]),
        ),
    )
    long_name = "A reaction named at far greater length than a row can hold"
    features = tuple(
        ReductionFeaturePresentation(f, long_name if f == "R1" else f"Display {f}", "", "")
        for f in ("R1", "R2", "R3")
    )

    plot = plot_pca_components_artifact(
        twins, presentation(features=features, loading_component_count=3, loading_feature_count=3)
    )

    labels = sorted(plot.yaxis[0].major_label_overrides.values())
    assert labels == ["A reaction named at far great\u2026 (+1)", "Display R3"]
    assert not plot.js_event_callbacks


def _layers(plot):
    """Each sample layer, in the order drawn: its fill, samples, opacities and sizes."""
    return [
        (
            renderer.glyph.fill_color,
            list(renderer.data_source.data["sample_id"]),
            list(renderer.data_source.data["alpha"]),
            list(renderer.data_source.data["size"]),
        )
        for renderer in plot.renderers
        if hasattr(renderer, "data_source") and "sample_id" in renderer.data_source.data
        and len(renderer.data_source.data["sample_id"])
    ]


def _focus(plot):
    (focus,) = [s for s in sources(plot) if s.name == GROUP_FOCUS]
    return focus


def test_highlighted_groups_are_drawn_over_the_rest_which_keep_their_colours_faint_and_small():
    plot = plot_reduction_artifact(
        artifact(), presentation(highlight_group_ids=("g2",), show_legend=False, show_sample_labels=False)
    )

    # The faded group first, so the highlighted one is drawn over it; neither turns grey.
    (faded, lit) = _layers(plot)
    full = lit[3][0]
    assert faded[:3] == ("#112233", ["s1", "s2"], [FADED_ALPHA] * 2) and faded[3] == [full * FADED_SIZE] * 2
    assert lit[:3] == ("#445566", ["s3", "s4"], [0.8] * 2) and lit[3] == [full] * 2
    assert not plot.legend or not plot.legend[0].items
    # The page moves the focus in the browser; the callback restyles both layers.
    focus = _focus(plot)
    assert list(focus.data["group_id"]) == ["g2"]
    (callback,) = focus.js_property_callbacks["change:data"]
    drawn = [r.data_source for r in plot.renderers if hasattr(r, "data_source") and "sample_id" in r.data_source.data]
    assert [id(source) for source in callback.args["points"]] == [id(source) for source in drawn]


def test_a_hidden_group_leaves_the_map_but_not_the_colour_scale():
    colouring = ReductionColouring("R9", "A reaction", "reactions", (1.0, 2.0, 3.0, 10.0))
    shown = plot_reduction_artifact(artifact(), presentation(colouring=colouring, show_sample_labels=False))
    hidden = plot_reduction_artifact(
        artifact(), presentation(colouring=colouring, hidden_group_ids=("g1",), show_sample_labels=False)
    )

    z_shown = dict(zip(source_with(shown, "z").data["sample_id"], source_with(shown, "z").data["z"]))
    z_hidden = dict(zip(source_with(hidden, "z").data["sample_id"], source_with(hidden, "z").data["z"]))
    assert set(z_hidden) == {"s3", "s4"}
    assert all(z_hidden[sample] == z_shown[sample] for sample in z_hidden)
    # On a level's scale: purple below the average, light grey at it, gold above.
    (mapper,) = [m for m in hidden.select({"type": LinearColorMapper})]
    assert (mapper.palette[0].upper(), mapper.palette[-1].upper()) == ("#542788", "#E6AB02") and "#E0E0E0" in [c.upper() for c in mapper.palette]


def test_a_highlighted_group_keeps_the_feature_colours_and_the_rest_fade_in_theirs():
    colouring = ReductionColouring("R9", "A reaction", "reactions", (1.0, 2.0, 3.0, 10.0))

    plot = plot_reduction_artifact(
        artifact(), presentation(colouring=colouring, highlight_group_ids=("g1",), show_sample_labels=False)
    )

    (faded, lit) = _layers(plot)
    # Both layers read the feature's level; only the opacity and size tell them apart.
    assert faded[0] == lit[0] and faded[0]["field"] == "z"
    assert sorted(faded[1]) == ["s3", "s4"] and faded[2] == [FADED_ALPHA] * 2
    assert sorted(lit[1]) == ["s1", "s2"] and lit[2] == [1.0] * 2
    assert list(_focus(plot).data["group_id"]) == ["g1"]


def test_a_sample_with_no_value_for_the_feature_is_hollow_not_grey():
    colouring = ReductionColouring("R9", "A reaction", "reactions", (1.0, float("nan"), 3.0, 10.0))

    plot = plot_reduction_artifact(artifact(), presentation(colouring=colouring, show_sample_labels=False))

    points = source_with(plot, "z")
    by_sample = dict(zip(points.data["sample_id"], zip(points.data["hollow"], points.data["edge"], points.data["edge_width"])))
    assert by_sample["s2"] == (True, "#606060", 1.0)
    assert not by_sample["s1"][0] and by_sample["s1"][1] != "#606060"
    (mapper,) = plot.select({"type": LinearColorMapper})
    assert mapper.nan_color == "#ffffff"


def test_the_feature_colouring_the_map_has_its_arrow_in_the_gradient_and_an_arrow_click_colours():
    colouring = ReductionColouring("R3", "Display R3", "reactions", (1.0, 2.0, 3.0, 4.0))
    plot = plot_reduction_artifact(
        artifact(),
        presentation(show_biplot_arrows=True, biplot_top_arrows=1, colouring=colouring, show_sample_labels=False),
    )

    (endpoints,) = [r for r in plot.renderers if r.name == "arrow_endpoints"]
    tips = endpoints.data_source
    # R3 is not the one automatic arrow, but it colours the samples, so it is drawn.
    assert "R3" in list(tips.data["feature_id"]) and len(tips.data["feature_id"]) == 2
    gradient = [s for s in sources(plot) if "x0" in s.data and len(s.data["x0"])]
    # It starts at the origin, in short segments along the gradient.
    assert gradient and (gradient[0].data["x0"][0], gradient[0].data["y0"][0]) == (0.0, 0.0)
    add_arrow_colouring(plot, "reactions", "colour-samples")
    (tap,) = [tool for tool in plot.tools if tool.__class__.__name__ == "TapTool"]
    endpoints = tap.renderers[0].data_source
    (callback,) = endpoints.selected.js_property_callbacks["change:indices"]
    assert callback.args["event"] == "colour-samples" and callback.args["data_type"] == "reactions"


def test_rows_the_map_draws_as_arrows_are_marked_in_the_components():
    plot = plot_pca_components_artifact(
        artifact(),
        presentation(loading_component_count=3, loading_feature_count=3),
        arrow_features=("R1",),
    )

    labels = set(plot.yaxis[0].major_label_overrides.values())
    assert "Display R1 \u279a" in labels and "Display R2" in labels


def test_the_components_show_the_plotted_ones_and_list_every_arrow():
    """The rows are the strongest summed over the shown components; the plotted
    components are always shown, and every feature the map draws as an arrow is
    listed, marked."""
    plot = plot_pca_components_artifact(
        artifact(),
        presentation(loading_component_count=2, loading_feature_count=2, loading_sort_component=None),
        plotted=(1, 3),
        arrow_features=("R2",),
        component_event="plot-component",
    )

    # PC3, plotted, is shown beyond the first two.
    assert list(plot.x_range.factors) == ["PC1", "PC2", "PC3"]
    labels = [plot.yaxis[0].major_label_overrides[y] for y in sorted(plot.yaxis[0].major_label_overrides, reverse=True)]
    # R1 and R3 load most over PC1 to PC3 (1.2 each); R2, an arrow, gets a row at the foot.
    assert labels == ["Display R1", "Display R3", "Display R2 \u279a"]
    # A click on the third bar plots PC3, the component in that column.
    (tap,) = plot.js_event_callbacks["tap"]
    assert tap.args["numbers"] == [1, 2, 3]


def test_an_arrow_outside_the_components_rows_gets_a_row_of_its_own():
    plot = plot_pca_components_artifact(
        artifact(), presentation(loading_component_count=2, loading_feature_count=1), arrow_features=("R3",)
    )

    labels = [plot.yaxis[0].major_label_overrides[y] for y in sorted(plot.yaxis[0].major_label_overrides, reverse=True)]
    # The one strongest row, then the arrow's own at the foot.
    assert labels == ["Display R1", "Display R3 \u279a"]

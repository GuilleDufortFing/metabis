"""The paired category plot reads one portable set-enrichment artifact."""

from dataclasses import replace

from bokeh.models import HoverTool

from metabis.analysis.artifacts import (
    AnalysisArtifactMetadata,
    ArtifactDimension,
    ArtifactKind,
    ArtifactProvenance,
    PlainTable,
)
from metabis.analysis.comparison_membership import (
    ComparisonMembership,
    GroupComparisonSpec,
)
from metabis.analysis.set_enrichment import (
    SetEnrichmentAnnotation,
    SetEnrichmentArtifact,
    SetEnrichmentConfig,
    SetEnrichmentFeatureType,
    SetEnrichmentUniverse,
    run_set_enrichment,
)
from metabis.analysis_workspace import RevisionKind, WorkspaceRevisions
from metabis.viz.plot_set_enrichment import (
    NOT_SIGNIFICANT_COLOR,
    plot_set_enrichment_category,
)


def _artifact(*, minimum_size=1) -> SetEnrichmentArtifact:
    membership = ComparisonMembership.resolve(
        GroupComparisonSpec.pairwise(("target",), ("baseline",)),
        group_samples={"target": ("t1", "t2"), "baseline": ("b1", "b2")},
        aligned_sample_ids=("t1", "t2", "b1", "b2"),
        labels={"target": "Target cells", "baseline": "Baseline cells"},
    )
    universe = SetEnrichmentUniverse(
        feature_type=SetEnrichmentFeatureType.GENE,
        tested_feature_ids=frozenset({"G1", "G2", "G3", "G4", "G5"}),
        target_higher_query_ids=frozenset({"G1", "G2"}),
        target_lower_query_ids=frozenset({"G4"}),
        subsystem_annotations=(
            SetEnrichmentAnnotation(
                "s1", "First pathway", ("G1", "G2", "G3"), "Metabolism"
            ),
            SetEnrichmentAnnotation(
                "s2", "Second pathway", ("G3", "G4", "G5"), "Metabolism"
            ),
        ),
    )
    result = run_set_enrichment(
        universe,
        SetEnrichmentConfig(
            minimum_measured_set_size=minimum_size,
            adjusted_pvalue_threshold=1.0,
        ),
    )
    metadata = AnalysisArtifactMetadata(
        kind=ArtifactKind.SET_ENRICHMENT,
        source=ArtifactProvenance(
            workspace_id="workspace",
            revisions=WorkspaceRevisions(dataset=1, groups=1, thresholds=1),
            dependencies=frozenset(
                {RevisionKind.DATASET, RevisionKind.GROUPS, RevisionKind.THRESHOLDS}
            ),
            source_fingerprints={"dataset": "sha256:dataset"},
        ),
        method="paired_hypergeometric_over_representation",
        configuration={"feature_type": "gene"},
        scientific_identity={
            "membership_fingerprint": membership.fingerprint,
            "feature_type": "gene",
        },
        dimensions=(ArtifactDimension("subsystems", 2),),
        warnings=result.warnings,
    )
    return SetEnrichmentArtifact(
        metadata=metadata,
        membership=membership,
        config=result.config,
        universe=result.universe,
        results=result.results,
        warnings=result.warnings,
        explanation=result.explanation,
    )


def test_category_plot_uses_one_dot_per_subsystem_without_recomputing(monkeypatch):
    artifact = _artifact()
    monkeypatch.setattr(
        "metabis.analysis.set_enrichment.run_set_enrichment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("plot recomputed enrichment")
        ),
    )

    plot = plot_set_enrichment_category(artifact)
    points = plot.select_one({"name": "set_enrichment_subsystem_points"})
    frame = points.data_source.data

    # A single dot per subsystem replaces the earlier paired points; the old
    # Target-higher/Target-lower glyphs no longer exist.
    assert not plot.select({"name": "set_enrichment_target_points"})
    assert not plot.select({"name": "set_enrichment_baseline_points"})
    assert set(frame["subsystem_id"]) == {"s1", "s2"}
    assert len(frame["subsystem_id"]) == 2
    # Each dot sits at the larger-magnitude direction, sign preserved.
    for target_visual, baseline_visual, dominant in zip(
        frame["target_visual"], frame["baseline_visual"], frame["dominant_visual"]
    ):
        expected = (
            baseline_visual
            if abs(baseline_visual) > abs(target_visual)
            else target_visual
        )
        assert dominant == expected
    # Significant subsystems take the shared signed-log2 gradient (hex colours),
    # not a single flat red/blue; the plot exposes no legend.
    assert all(str(color).startswith("#") for color in frame["dominant_color"])
    assert len(plot.legend) == 0
    assert "METABOLISM" in plot.y_range.factors
    assert plot.tags[-1] == {
        "artifact_id": artifact.metadata.artifact_id,
        "artifact_kind": "set_enrichment",
        "feature_type": "gene",
        "presentation": "paired_set_enrichment_category",
    }


def test_rows_are_ordered_within_category_by_largest_directional_magnitude():
    plot = plot_set_enrichment_category(_artifact())
    source = plot.select_one({"name": "set_enrichment_subsystem_points"}).data_source
    magnitudes = list(source.data["maximum_visual_magnitude"])

    assert magnitudes == sorted(magnitudes, reverse=True)


def test_category_plot_uses_the_rankings_rows():
    """Names in a column, a header row per category, and no category hue."""
    plot = plot_set_enrichment_category(
        _artifact(), category_colors={"Metabolism": "#2A9D8F"}
    )

    # Bokeh lists factors bottom up; read top down, the header opens its category.
    rows = list(reversed(plot.y_range.factors))
    assert rows[0] == "METABOLISM"
    assert set(rows[1:]) == {"First pathway", "Second pathway"}
    assert plot.yaxis[0].visible
    assert plot.ygrid[0].grid_line_color is None
    fills = {
        renderer.glyph.fill_color
        for renderer in plot.renderers
        if renderer.glyph.__class__.__name__ == "HBar"
    }
    assert "#2A9D8F" not in fills


def _points_hover(plot):
    points = plot.select_one({"name": "set_enrichment_subsystem_points"})
    (hover,) = [tool for tool in plot.select(type=HoverTool) if points in (tool.renderers or [])]
    return hover


def test_nonsignificant_subsystem_is_gray_and_hover_reports_both_results():
    artifact = _artifact()
    frame = artifact.results.to_frame()
    # A subsystem significant in neither direction is gray; its dot still carries
    # both directional results on hover.
    frame.loc[frame.subsystem_id == "s1", "is_significant"] = False
    artifact = replace(artifact, results=PlainTable.capture(frame))

    plot = plot_set_enrichment_category(artifact)
    renderer = plot.select_one({"name": "set_enrichment_subsystem_points"})
    source = renderer.data_source.data
    index = list(source["subsystem_id"]).index("s1")
    hover = _points_hover(plot)
    html = source["hover_html"][index]

    assert source["dominant_color"][index] == NOT_SIGNIFICANT_COLOR
    assert not source["significant"][index]
    assert hover.tooltips.startswith("@hover_html{safe}")
    assert "<strong>Color Reference:</strong>" in hover.tooltips
    assert "Target-higher" in html
    assert "Target-lower" in html
    assert "Adjusted p" in html
    assert "Overlap" in html
    assert "query" in html
    assert "Background N=5" in html
    assert "Not applicable to gene-set enrichment" in html


def test_every_dot_carries_the_gradient_color_reference_it_was_painted_from():
    """R-09. Supp. 1.5 claims this panel for every gradient plot.

    The category plot lost its legend when it adopted the shared signed-log2
    gradient, which left the claim untrue of the shipped server. The reference is
    built from the same distribution that chose the colours, and each dot marks
    its own value on it.
    """
    plot = plot_set_enrichment_category(_artifact())
    source = plot.select_one({"name": "set_enrichment_subsystem_points"}).data_source
    hover = _points_hover(plot)
    values = list(source.data["dominant_visual"])

    assert len(source.data["hover_html"]) > 0
    assert "<strong>Color Reference:</strong>" in hover.tooltips
    assert "Signed log2(ER) Distribution" in hover.tooltips
    assert hover.tooltips.count("<svg") == 1
    assert all("<svg" not in html for html in source.data["hover_html"])
    assert list(source.data["color_reference_value"]) == values
    assert len(set(source.data["color_reference_marker_x"])) == len(values)


def test_empty_eligible_artifact_and_wrong_input_remain_structural():
    plot = plot_set_enrichment_category(_artifact(minimum_size=99))
    renderer = plot.select_one({"name": "set_enrichment_subsystem_points"})
    assert len(renderer.data_source.data["subsystem_id"]) == 0

    try:
        plot_set_enrichment_category(object())
    except TypeError as exc:
        assert "SetEnrichmentArtifact" in str(exc)
    else:
        raise AssertionError("non-artifact input was accepted")

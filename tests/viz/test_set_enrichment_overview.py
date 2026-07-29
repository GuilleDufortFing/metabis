"""Subsystem overview colouring from one paired set-enrichment artifact.

WP6 stage 8: the metabolic overview paints subsystem glyphs with the same
diverging red/white/blue gradient delta uses, applied to signed
log2(enrichment ratio); a non-significant subsystem is gray. Process (category)
glyphs and detailed reaction/gene node colours are untouched.
"""

from dataclasses import replace

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
from metabis.managers.workspace_mode import AnalysisWorkspaceMode
from metabis.viz.metabolic_visualization.enrichment.glyph_enrichment import (
    enrich_subsystem_glyph_with_data,
)
from metabis.viz.set_enrichment_overview import (
    NOT_SIGNIFICANT_COLOR,
    SET_ENRICHMENT_SUBSYSTEM_METRIC,
    _comparison_registry,
    build_set_enrichment_overview_colors,
    build_subsystem_decisions,
    signed_log2_gradient_colors,
)


def _artifact(
    universe: SetEnrichmentUniverse | None = None,
    config: SetEnrichmentConfig | None = None,
) -> SetEnrichmentArtifact:
    membership = ComparisonMembership.resolve(
        GroupComparisonSpec.pairwise(("target",), ("baseline",)),
        group_samples={"target": ("t1", "t2"), "baseline": ("b1", "b2")},
        aligned_sample_ids=("t1", "t2", "b1", "b2"),
        labels={"target": "Target cells", "baseline": "Baseline cells"},
    )
    if universe is None:
        universe = SetEnrichmentUniverse(
            feature_type=SetEnrichmentFeatureType.GENE,
            tested_feature_ids=frozenset(
                {"G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"}
            ),
            target_higher_query_ids=frozenset({"G1", "G2"}),
            target_lower_query_ids=frozenset({"G4", "G5"}),
            subsystem_annotations=(
                SetEnrichmentAnnotation("s1", "First pathway", ("G1", "G2", "G3"), "Metabolism"),
                SetEnrichmentAnnotation("s2", "Second pathway", ("G4", "G5", "G6"), "Metabolism"),
                SetEnrichmentAnnotation("s3", "Third pathway", ("G3", "G6", "G7"), "Signalling"),
                SetEnrichmentAnnotation("s4", "Fourth pathway", ("G1", "G2", "G4", "G5"), "Signalling"),
            ),
        )
    if config is None:
        config = SetEnrichmentConfig(
            minimum_measured_set_size=1, adjusted_pvalue_threshold=0.05
        )
    result = run_set_enrichment(universe, config)
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
        dimensions=(ArtifactDimension("subsystems", 4),),
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


def _override(artifact, decisions):
    """Fix each (subsystem, direction) significance/visual for deterministic colours.

    ``decisions`` maps subsystem_id to ``(target_sig, target_log2, baseline_sig,
    baseline_log2)``.
    """
    frame = artifact.results.to_frame()
    for subsystem_id, (t_sig, t_log2, b_sig, b_log2) in decisions.items():
        higher = (frame.subsystem_id == subsystem_id) & (frame.direction == "target_higher")
        lower = (frame.subsystem_id == subsystem_id) & (frame.direction == "target_lower")
        frame.loc[higher, "is_significant"] = t_sig
        frame.loc[higher, "signed_log2_enrichment_ratio"] = t_log2
        frame.loc[lower, "is_significant"] = b_sig
        frame.loc[lower, "signed_log2_enrichment_ratio"] = b_log2
    return replace(artifact, results=PlainTable.capture(frame))


def test_single_signed_value_takes_the_larger_magnitude_direction():
    artifact = _override(
        _artifact(),
        {
            "s1": (True, 1.5, False, 0.0),  # only Target-higher -> positive
            "s2": (False, 0.0, True, -1.2),  # only Target-lower -> negative
            "s3": (True, 0.4, True, -1.8),  # both, Target-lower stronger -> negative
            "s4": (True, 2.1, True, -0.5),  # both, Target-higher stronger -> positive
        },
    )
    decisions = build_subsystem_decisions(artifact)

    assert decisions["s1"].signed_log2 == 1.5
    assert decisions["s2"].signed_log2 == -1.2
    assert decisions["s3"].signed_log2 == -1.8
    assert decisions["s4"].signed_log2 == 2.1
    assert decisions["s1"].significant and decisions["s3"].significant


def test_positive_negative_and_gray_use_the_delta_gradient_endpoints():
    artifact = _override(
        _artifact(),
        {
            "s1": (True, 2.0, False, 0.0),  # strongest positive -> red end
            "s2": (False, 0.0, True, -2.0),  # strongest negative -> blue end
            "s3": (False, 0.0, False, 0.0),  # neither -> gray
            "s4": (True, 0.05, False, 0.0),  # barely positive -> near white
        },
    )
    colors = build_set_enrichment_overview_colors(artifact, _comparison_registry())

    # Non-significant is exactly the shared gray, never a gradient colour.
    assert colors.for_subsystem("s3").fill_color == NOT_SIGNIFICANT_COLOR
    # The strongest positive is redder than a barely-positive one; the strongest
    # negative is bluer. Compare the dominant channel.
    red = colors.for_subsystem("s1").fill_color
    blue = colors.for_subsystem("s2").fill_color
    faint = colors.for_subsystem("s4").fill_color
    assert red != NOT_SIGNIFICANT_COLOR and blue != NOT_SIGNIFICANT_COLOR
    # white -> red drops the blue channel; white -> blue drops the red channel.
    assert _redness(red) > _redness(faint)  # stronger positive is more saturated
    assert _blueness(blue) > _blueness(red)  # negative leans blue, positive red


def test_distribution_is_registered_so_hover_reads_the_same_scale():
    registry = _comparison_registry()
    artifact = _override(
        _artifact(),
        {
            "s1": (True, 1.5, False, 0.0),
            "s2": (False, 0.0, True, -1.0),
            "s3": (False, 0.0, False, 0.0),
            "s4": (True, 0.6, False, 0.0),
        },
    )
    colors = build_set_enrichment_overview_colors(artifact, registry)

    details = registry.get_distribution_details(
        mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
        element_type="subsystem",
        element_id="s1",
        metric_name=SET_ENRICHMENT_SUBSYSTEM_METRIC,
    )
    assert details["element_value"] == 1.5
    # The glyph colour equals the distribution's colour for that element.
    assert colors.for_subsystem("s1").fill_color == details["element_color"]

    hover = colors.hover_for("s1", show_stats=True, show_color_reference=True)
    assert "Target-higher" in hover and "Target-lower" in hover
    assert "Color Reference" in hover and "log2(ER)" in hover


def _red(hex_color):
    return int(hex_color[1:3], 16)


def _blue(hex_color):
    return int(hex_color[5:7], 16)


def _redness(hex_color):
    # On a white -> red ramp the blue channel falls as saturation rises.
    return 255 - int(hex_color[5:7], 16)


def _blueness(hex_color):
    # On a white -> blue ramp the red channel falls as saturation rises.
    return 255 - int(hex_color[1:3], 16)


def test_empty_artifact_yields_empty_colors():
    artifact = _artifact()
    empty = artifact.results.to_frame().iloc[0:0]
    colors = build_set_enrichment_overview_colors(
        replace(artifact, results=PlainTable.capture(empty)), _comparison_registry()
    )
    assert dict(colors.subsystems) == {}


def test_signed_log2_gradient_colors_diverge_around_zero():
    import pandas as pd

    colors = signed_log2_gradient_colors(
        pd.Series({"a": 2.0, "b": -2.0, "c": 0.0}, dtype=float)
    )
    assert _red(colors["a"]) > _blue(colors["a"])  # positive -> red
    assert _blue(colors["b"]) > _red(colors["b"])  # negative -> blue


class _FocusStub:
    def __init__(self, model=None):
        self.model = model

    def has_element_type(self, _element_type):
        return False

    def get_ancestor_ids(self):
        return {}


class _DataManagerStub:
    """A calculation-input stand-in that fails if the delta path is used."""

    def __init__(self, overview):
        self.set_enrichment_overview = overview
        self.metabolic_model = None

    def get_element_hover(self, *args, **kwargs):
        raise AssertionError("delta hover must not run under set-enrichment overview")

    def get_element_color(self, *args, **kwargs):
        raise AssertionError("delta colour must not run under set-enrichment overview")


def _subsystem_glyph(subsystem_id):
    return {
        "glyph_class": "subsystem",
        "id": subsystem_id,
        "label": subsystem_id,
        "subsystem_name": subsystem_id,
        "reaction_count": 3,
        "metabolite_count": 2,
    }


def test_subsystem_glyph_paints_from_gradient_and_skips_the_delta_path():
    artifact = _override(
        _artifact(),
        {
            "s1": (True, 1.5, False, 0.0),
            "s3": (False, 0.0, False, 0.0),
        },
    )
    overview = build_set_enrichment_overview_colors(artifact, _comparison_registry())
    data_manager = _DataManagerStub(overview)

    significant = enrich_subsystem_glyph_with_data(
        _subsystem_glyph("s1"),
        alpha=0.9,
        subsystem_translations={},
        data_manager=data_manager,
        color_metric="log2_fc",
        delta_options=None,
        focus_instance=_FocusStub(),
    )
    assert significant["color"] == overview.for_subsystem("s1").fill_color
    assert significant["color"] != NOT_SIGNIFICANT_COLOR

    nonsignificant = enrich_subsystem_glyph_with_data(
        _subsystem_glyph("s3"),
        alpha=0.9,
        subsystem_translations={},
        data_manager=data_manager,
        color_metric="log2_fc",
        delta_options=None,
        focus_instance=_FocusStub(),
    )
    assert nonsignificant["color"] == NOT_SIGNIFICANT_COLOR


def test_subsystem_glyph_without_overview_uses_the_delta_path():
    class _DeltaManager:
        metabolic_model = None
        set_enrichment_overview = None

        def __init__(self):
            self.color_calls = 0

        def get_element_hover(self, *args, **kwargs):
            return "delta hover"

        def get_element_color(self, *args, **kwargs):
            self.color_calls += 1
            return "#FF00FF"

    manager = _DeltaManager()
    enriched = enrich_subsystem_glyph_with_data(
        _subsystem_glyph("s1"),
        alpha=0.9,
        subsystem_translations={},
        data_manager=manager,
        color_metric="log2_fc",
        delta_options={"metric": "cohens_d"},
        focus_instance=_FocusStub(),
    )

    assert manager.color_calls == 1
    assert enriched["color"] == "#FF00FF"
    assert enriched["hover"] == "delta hover"


def _ineligible_universe() -> SetEnrichmentUniverse:
    """A universe whose ``s_small`` subsystem has too few measured genes to test."""
    return SetEnrichmentUniverse(
        feature_type=SetEnrichmentFeatureType.GENE,
        tested_feature_ids=frozenset({"G1", "G2", "G3", "G4", "G5", "G6"}),
        target_higher_query_ids=frozenset({"G1", "G2"}),
        target_lower_query_ids=frozenset({"G5"}),
        subsystem_annotations=(
            SetEnrichmentAnnotation("s_big", "Big pathway", ("G1", "G2", "G3", "G4"), "Metabolism"),
            SetEnrichmentAnnotation("s_small", "Small pathway", ("G5", "G6"), "Metabolism"),
        ),
    )


def test_below_minimum_subsystem_is_excluded_from_overview_colors():
    # With a minimum of 3 measured genes, ``s_small`` (2 measured) is never tested,
    # so it must not appear in the overview colour map at all (D-59).
    artifact = _artifact(
        _ineligible_universe(),
        SetEnrichmentConfig(minimum_measured_set_size=3, adjusted_pvalue_threshold=0.05),
    )
    colors = build_set_enrichment_overview_colors(artifact, _comparison_registry())

    assert colors.minimum_measured_set_size == 3
    assert colors.for_subsystem("s_big") is not None
    assert colors.for_subsystem("s_small") is None


def test_ineligible_subsystem_paints_gray_and_never_reads_delta():
    # A subsystem the ORA never tested (below the minimum count, so absent from the
    # colour map) must paint the not-significant gray under the Set-Enrichment
    # summary, with an eligibility hover -- never the delta-score colour/hover
    # (D-59). The stub raises if the delta path is taken.
    artifact = _artifact(
        _ineligible_universe(),
        SetEnrichmentConfig(minimum_measured_set_size=3, adjusted_pvalue_threshold=0.05),
    )
    overview = build_set_enrichment_overview_colors(artifact, _comparison_registry())
    data_manager = _DataManagerStub(overview)

    enriched = enrich_subsystem_glyph_with_data(
        _subsystem_glyph("s_small"),
        alpha=0.9,
        subsystem_translations={},
        data_manager=data_manager,
        color_metric="log2_fc",
        delta_options={"metric": "cohens_d"},
        focus_instance=_FocusStub(),
    )

    assert enriched["color"] == NOT_SIGNIFICANT_COLOR
    assert "Not over-representation eligible" in enriched["hover"]
    assert "3 measured genes" in enriched["hover"]

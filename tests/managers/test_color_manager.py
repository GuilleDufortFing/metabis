import numpy as np
import pandas as pd
import pytest

from metabis.managers.color_manager import ColorManager
from metabis.managers.workspace_mode import AnalysisWorkspaceMode
from metabis.visual_config import theme


def test_group_colors_are_generated_for_a_normal_number_of_groups():
    manager = ColorManager()

    colors = manager.generate_group_colors(["A", "B", "C"])

    assert set(colors) == {"A", "B", "C"}
    assert len(set(colors.values())) == 3


def test_activity_ramp_runs_purple_through_grey_to_gold():
    """Activity mode places a level against the typical one, never as a change."""
    manager = ColorManager()
    # Two far outliers around a mass of typical values, so both extremes clip to
    # the ends of the ramp.
    scores = pd.Series({"low": -10.0, "high": 10.0} | {f"f{i}": 0.0 for i in range(100)})

    manager.register_distribution(
        mode=AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS,
        element_type="reaction",
        metric_name="activity",
        data=scores,
        center_method="zero",
        spread_method="std",
    )

    def color(element_id):
        return manager.get_element_color_from_distribution(
            mode=AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS,
            element_type="reaction",
            element_id=element_id,
            metric_name="activity",
        )

    assert color("high").upper() == theme.ACTIVITY_HIGH.upper()
    assert color("low").upper() == theme.ACTIVITY_LOW.upper()
    assert color("f0").upper() == theme.ACTIVITY_MID.upper()
    # The comparison heatmap's red/blue must not appear in an activity graph.
    assert theme.UP_REGULATED.upper() not in {color("high").upper(), color("low").upper()}


def _register(manager, scores, **overrides):
    options = dict(
        mode=AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS,
        element_type="reaction",
        metric_name="activity",
        data=scores,
        center_method="zero",
        spread_method="saturating",
    )
    options.update(overrides)
    manager.register_distribution(**options)

    def color(element_id):
        return manager.get_element_color_from_distribution(
            mode=AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS,
            element_type="reaction",
            element_id=element_id,
            metric_name="activity",
        )

    return color


def test_a_standard_deviation_scale_paints_concentrated_data_all_neutral():
    """The failure the saturating scale exists to prevent."""
    manager = ColorManager()
    # Real activity scores: bounded, and concentrated well inside one SD of zero.
    scores = pd.Series({f"r{i}": value for i, value in enumerate(np.linspace(-0.3, 0.3, 60))})

    color = _register(manager, scores, spread_method="std")

    # Under a 4-SD palette, the extremes of the data barely leave the grey centre.
    assert color("r0").upper() != theme.ACTIVITY_LOW.upper()
    assert color("r59").upper() != theme.ACTIVITY_HIGH.upper()


def test_the_saturating_scale_stretches_the_palette_over_the_data_that_exists():
    manager = ColorManager()
    scores = pd.Series({f"r{i}": value for i, value in enumerate(np.linspace(-0.3, 0.3, 60))})

    color = _register(manager, scores)

    # The same data now spans the whole ramp: least active is purple, most active
    # is gold, and the middle is the grey centre.
    assert color("r0").upper() == theme.ACTIVITY_LOW.upper()
    assert color("r59").upper() == theme.ACTIVITY_HIGH.upper()
    assert color("r30").upper() != color("r59").upper()

    limit = manager.get_color_limit(
        AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS, "reaction", "activity"
    )
    assert limit == pytest.approx(0.3, abs=0.02)


def test_one_extreme_outlier_cannot_flatten_every_other_colour():
    manager = ColorManager()
    scores = pd.Series(
        {f"r{i}": value for i, value in enumerate(np.linspace(-0.3, 0.3, 60))}
        | {"outlier": 25.0}
    )

    color = _register(manager, scores)

    # The outlier clips at the top of the ramp instead of dragging the scale out to
    # 25 and rendering every real difference invisible.
    assert color("outlier").upper() == theme.ACTIVITY_HIGH.upper()
    assert color("r59").upper() != color("r0").upper()
    assert (
        manager.get_color_limit(AnalysisWorkspaceMode.EXPERIMENT_ANALYSIS, "reaction", "activity")
        < 1.0
    )


def test_group_colors_cycle_instead_of_failing_beyond_the_palette_limit():
    """Bokeh palettes cap at 256 colors; more groups must still get a color."""
    group_ids = [f"G{index}" for index in range(ColorManager.MAX_PALETTE_COLORS + 40)]
    manager = ColorManager()

    colors = manager.generate_group_colors(group_ids, palette_name="turbo")

    assert len(colors) == len(group_ids)
    assert all(colors[group_id] for group_id in group_ids)
    # Colors repeat once the palette is exhausted, but nothing is left uncolored.
    assert len(set(colors.values())) == ColorManager.MAX_PALETTE_COLORS


def test_default_group_colors_are_okabe_ito_then_scanpy():
    """Up to eight groups take Okabe-Ito; more take scanpy's palettes, then repeat."""
    from metabis.managers.group_palettes import OKABE_ITO, SCANPY_20, SCANPY_102

    manager = ColorManager()

    def colors(n):
        return list(manager.generate_group_colors([f"G{index}" for index in range(n)]).values())

    assert colors(3) == list(OKABE_ITO[:3])
    assert colors(8) == list(OKABE_ITO)
    assert colors(9) == list(SCANPY_20[:9])
    assert colors(102) == list(SCANPY_102)
    beyond = colors(110)
    assert beyond[102:] == list(SCANPY_102[:8])

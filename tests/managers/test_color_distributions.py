"""Registering a colour distribution must work on a worker, without Bokeh.

``GraphColorLookup`` (slice 3a) let a graph worker *read* colours without holding
a ``ColorManager``. That was one step short of what enrichment actually does: a
worker also *registers* distributions while it runs -- activity scoring publishes
reaction/gene/subsystem activity distributions, and a subsystem graph asked for a
delta configuration the document thread has not precomputed registers that
distribution mid-enrichment. Measured on the real Human1 glycolysis workspace: 0
registrations during ``build_artifact`` when the configuration is already
installed, 1 when it is not.

So the write half has to be reachable without Bokeh too, and these tests pin the
two properties that makes true:

* registration genuinely runs with Bokeh unavailable, asserted against a blocked
  import in a fresh interpreter rather than a mock (AM10 slice 2 found a Bokeh
  edge every static scan called clean; only a runtime probe is evidence);
* a registry paints exactly what ``ColorManager`` paints, because the manager
  delegates here instead of keeping a second implementation. A reintroduced
  private copy on the manager would give a worker different colours from the
  document thread -- the MR-08 shadow-read shape, invisible to a type check.

See ``MetabisReview/architecture-migration/am10-inventory.md`` (M-12 A').
"""

import json

import pandas as pd
import pytest

from metabis.managers.color_distributions import (
    ColorDistributionRegistry,
    ensure_rgb,
    gradient_palette,
    hex_to_rgb,
    rgb_to_hex,
)
from metabis.managers.color_manager import ColorManager
from metabis.managers.workspace_mode import AnalysisWorkspaceMode
from tests.managers.test_color_lookup import run_without_bokeh

MODE = AnalysisWorkspaceMode.GROUP_COMPARISON

VALUES = pd.Series(
    {
        "MAR00001": -3.2,
        "MAR00002": -0.4,
        "MAR00003": 0.0,
        "MAR00004": 0.9,
        "MAR00005": 4.1,
    }
)


@pytest.fixture
def manager() -> ColorManager:
    return ColorManager()


# --- Agreement with the live manager -------------------------------------


def test_the_manager_registers_through_the_registry(manager):
    """One store, not two: what the manager registers is what the registry holds."""
    manager.register_distribution(
        mode=MODE, element_type="reaction", data=VALUES, metric_name="log2_fc"
    )

    assert manager._registry.distributions is manager._distributions
    assert (MODE, "reaction", "log2_fc") in manager._registry.distributions


def test_a_standalone_registry_paints_what_the_manager_paints(manager):
    """The same values, registered either way, must produce the same hex colours."""
    manager.register_distribution(
        mode=MODE, element_type="reaction", data=VALUES, metric_name="log2_fc"
    )
    registry = ColorDistributionRegistry(
        configurations={
            mode: dict(config) for mode, config in manager.configurations.items()
        },
        current_mode=manager.current_mode,
    )
    registry.register_distribution(
        mode=MODE, element_type="reaction", data=VALUES, metric_name="log2_fc"
    )

    registered = registry.distributions[(MODE, "reaction", "log2_fc")]
    assert registered["bin_colors"] == (
        manager._distributions[(MODE, "reaction", "log2_fc")]["bin_colors"]
    )
    for element_id in VALUES.index:
        assert registered["binner"].get_bin(element_id) == (
            manager._distributions[(MODE, "reaction", "log2_fc")]["binner"].get_bin(
                element_id
            )
        )


def test_a_p_value_metric_keeps_its_re_pointed_ramp(manager):
    """Significance runs one way, so a p-value palette is not the diverging one.

    Pins the branch this slice moved: raw p swaps the two ends of the ramp and
    -log(p) only replaces the up end.
    """
    for metric in ("pval", "neg_log_pval", "log2_fc"):
        manager.register_distribution(
            mode=MODE, element_type="reaction", data=VALUES, metric_name=metric
        )

    colors = {
        metric: manager._distributions[(MODE, "reaction", metric)]["bin_colors"]
        for metric in ("pval", "neg_log_pval", "log2_fc")
    }
    config = manager.configurations[MODE]

    # Raw p-value: the neutral centre becomes the significant colour.
    assert colors["pval"][0] == config["significant_pval_color"]
    # -log(p): the centre stays neutral and the positive end goes significant.
    assert colors["neg_log_pval"][0] == config["neutral_color"]
    assert colors["neg_log_pval"][30] == config["significant_pval_color"]
    # A plain effect metric is untouched by either branch.
    assert colors["log2_fc"][0] == config["neutral_color"]
    assert colors["log2_fc"][30] == config["up_color"]


def test_clearing_the_manager_clears_the_registry(manager):
    manager.register_distribution(
        mode=MODE, element_type="reaction", data=VALUES, metric_name="log2_fc"
    )
    manager.clear_distributions()

    assert manager._registry.distributions == {}


# --- Detachment ----------------------------------------------------------


def test_a_detached_copy_absorbs_a_workers_registrations(manager):
    """A worker registering mid-run must not repaint the document thread.

    This is what the per-job ``deepcopy(color_manager)`` buys today, without the
    manager and without the deep copy: the mapping is copied, the binners are
    shared because nothing mutates them after registration.
    """
    manager.register_distribution(
        mode=MODE, element_type="reaction", data=VALUES, metric_name="log2_fc"
    )
    worker = manager._registry.detached_copy()

    worker.register_distribution(
        mode=MODE, element_type="subsystem", data=VALUES, metric_name="cohens_d"
    )

    assert (MODE, "subsystem", "cohens_d") in worker.distributions
    assert (MODE, "subsystem", "cohens_d") not in manager._distributions
    # The registration it inherited is the same object, not a copy of it.
    assert (
        worker.distributions[(MODE, "reaction", "log2_fc")]["binner"]
        is manager._distributions[(MODE, "reaction", "log2_fc")]["binner"]
    )


def test_a_detached_copy_does_not_share_configuration(manager):
    worker = manager._registry.detached_copy()

    worker.configurations[MODE]["up_color"] = "#000000"

    assert manager.configurations[MODE]["up_color"] != "#000000"


# --- Colour arithmetic ---------------------------------------------------


def test_the_colour_helpers_round_trip():
    assert hex_to_rgb("#FF8000") == (255, 128, 0)
    assert rgb_to_hex((255, 128, 0)) == "#FF8000"
    assert ensure_rgb((255, 128, 0)) == (255, 128, 0)
    with pytest.raises(ValueError):
        hex_to_rgb("#FFF")
    with pytest.raises(ValueError):
        ensure_rgb(object())


def test_a_gradient_spans_both_endpoints():
    palette = gradient_palette("#000000", "#FFFFFF", 5)

    assert palette[0] == "#000000"
    assert palette[-1] == "#FFFFFF"
    assert len(palette) == 5
    assert gradient_palette("#000000", "#FFFFFF", 1) == ["#FFFFFF"]


# --- The property the module exists for ----------------------------------


def test_registration_runs_without_bokeh():
    """A worker must be able to publish a distribution it just computed."""
    result = run_without_bokeh(
        """
        import json

        import pandas as pd

        from metabis.managers.color_distributions import ColorDistributionRegistry
        from metabis.managers.color_lookup import GraphColorLookup
        from metabis.managers.workspace_mode import AnalysisWorkspaceMode

        mode = AnalysisWorkspaceMode.GROUP_COMPARISON
        registry = ColorDistributionRegistry(
            configurations={
                mode: {
                    "max_z_score": 6,
                    "up_color": "#D62728",
                    "down_color": "#1F77B4",
                    "neutral_color": "#F0F0F0",
                }
            },
            current_mode=mode,
        )
        registry.register_distribution(
            mode=mode,
            element_type="reaction",
            data=pd.Series({"MAR00001": -3.2, "MAR00003": 0.0, "MAR00005": 4.1}),
            metric_name="log2_fc",
        )
        lookup = GraphColorLookup(
            neutral_color="#F0F0F0",
            up_color="#D62728",
            down_color="#1F77B4",
            group_colors={},
            positive_influence_palette=(),
            negative_influence_palette=(),
            distributions=registry.distributions,
        )

        print(json.dumps({
            "up": lookup.get_element_color_from_distribution(
                mode, "reaction", "MAR00005", "log2_fc"
            ),
            "centre": lookup.get_element_color_from_distribution(
                mode, "reaction", "MAR00003", "log2_fc"
            ),
            "down": lookup.get_element_color_from_distribution(
                mode, "reaction", "MAR00001", "log2_fc"
            ),
        }))
        """
    )

    assert result.returncode == 0, result.stderr
    painted = json.loads(result.stdout.strip().splitlines()[-1])

    # And it painted what a Bokeh-carrying manager would have painted.
    manager = ColorManager(
        group_comparison_config={
            "max_z_score": 6,
            "up_color": "#D62728",
            "down_color": "#1F77B4",
            "neutral_color": "#F0F0F0",
        }
    )
    manager.register_distribution(
        mode=MODE,
        element_type="reaction",
        data=pd.Series({"MAR00001": -3.2, "MAR00003": 0.0, "MAR00005": 4.1}),
        metric_name="log2_fc",
    )
    assert painted == {
        key: manager.get_element_color_from_distribution(
            MODE, "reaction", element_id, "log2_fc"
        )
        for key, element_id in (
            ("up", "MAR00005"),
            ("centre", "MAR00003"),
            ("down", "MAR00001"),
        )
    }


def test_the_colour_manager_still_needs_bokeh_for_its_palettes():
    """Pins where the boundary is, so the test above cannot pass vacuously."""
    result = run_without_bokeh("import metabis.managers.color_manager")

    assert result.returncode != 0
    assert "Bokeh is unavailable" in result.stderr

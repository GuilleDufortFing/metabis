"""The colour lookup must be readable without Bokeh, and must not drift from the manager.

``GraphColorLookup`` exists so a graph worker can answer colour questions without
holding a ``ColorManager`` -- which is the sole remaining reason ``bokeh.palettes``
is reachable from anything an ``AnalysisWorkspace`` touches (AM10 inventory,
cause 3). Two properties therefore have to hold, and neither is implied by the
other.

The lookup must genuinely import and compute with Bokeh unavailable. That is
asserted against a real blocked import in a fresh interpreter, not a mock,
matching ``tests/architecture/test_graph_bokeh_boundary.py``: AM10 slice 2 found
a Bokeh dependency reached through a transitive edge that every static scan
declared clean, so only a runtime probe is evidence here.

And a captured lookup must paint exactly what the live manager paints. The
manager delegates to the lookup rather than keeping a second copy, so this is
structurally true -- but the delegation is what these tests pin, because a
reintroduced private implementation on the manager would silently give a graph
worker different colours from the document thread that captured it. That is the
MR-08 shape: a shadow read that no type check can see.

See ``MetabisReview/architecture-migration/am10-inventory.md`` (M-12 A').
"""

import dataclasses
import json
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from metabis.managers.color_lookup import GraphColorLookup
from metabis.managers.color_manager import ColorManager
from metabis.managers.workspace_mode import AnalysisWorkspaceMode

MODE = AnalysisWorkspaceMode.GROUP_COMPARISON


@pytest.fixture
def manager_with_distribution():
    """A manager carrying a real registered reaction distribution."""
    manager = ColorManager()
    values = pd.Series(
        {
            "MAR00001": -3.2,
            "MAR00002": -0.4,
            "MAR00003": 0.0,
            "MAR00004": 0.9,
            "MAR00005": 4.1,
        }
    )
    manager.register_distribution(
        mode=MODE, element_type="reaction", data=values, metric_name="log2_fc"
    )
    manager.set_group_colors({"treated": "#111111", "control": "#222222"})
    return manager, values


# --- Agreement with the live manager -------------------------------------


def test_a_captured_lookup_paints_what_the_manager_paints(manager_with_distribution):
    manager, values = manager_with_distribution
    lookup = manager.color_lookup()

    assert isinstance(lookup, GraphColorLookup)
    for element_id in values.index:
        assert lookup.get_element_color_from_distribution(
            MODE, "reaction", element_id, "log2_fc"
        ) == manager.get_element_color_from_distribution(
            MODE, "reaction", element_id, "log2_fc"
        )


def test_a_captured_lookup_agrees_on_distribution_details(manager_with_distribution):
    manager, _ = manager_with_distribution

    assert manager.color_lookup().get_distribution_details(
        MODE, "reaction", "MAR00005", "log2_fc"
    ) == manager.get_distribution_details(MODE, "reaction", "MAR00005", "log2_fc")


def test_a_captured_lookup_agrees_on_influence_colour(manager_with_distribution):
    manager, _ = manager_with_distribution
    lookup = manager.color_lookup()

    for correlation, influence in ((0.8, 40.0), (-0.8, 40.0), (0.0, 0.0), (0.5, 999.0)):
        assert lookup.get_influence_color(
            correlation, influence
        ) == manager.get_influence_color(correlation, influence)


def test_a_captured_lookup_agrees_on_group_colour(manager_with_distribution):
    manager, _ = manager_with_distribution

    assert manager.color_lookup().get_group_color("treated") == manager.get_group_color(
        "treated"
    )
    with pytest.raises(KeyError):
        manager.color_lookup().get_group_color("absent")


def test_an_unregistered_distribution_falls_to_the_neutral_colour():
    lookup = ColorManager().color_lookup()

    assert lookup.get_element_color_from_distribution(
        MODE, "reaction", "MAR00001", "log2_fc"
    ) == lookup.neutral_color
    assert lookup.get_color_limit(MODE, "reaction", "log2_fc") is None


# --- Detachment ----------------------------------------------------------


def test_the_lookup_is_frozen(manager_with_distribution):
    """A worker holding the lookup cannot repaint the document thread's colours."""
    manager, _ = manager_with_distribution
    lookup = manager.color_lookup()

    with pytest.raises(dataclasses.FrozenInstanceError):
        lookup.neutral_color = "#000000"
    with pytest.raises(TypeError):
        lookup.group_colors["treated"] = "#000000"
    with pytest.raises(TypeError):
        lookup.distributions[(MODE, "reaction", "log2_fc")] = {}


# --- The property the type exists for ------------------------------------

_BLOCK_BOKEH = """
import importlib.abc
import sys


class _BlockBokeh(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "bokeh" or name.startswith("bokeh."):
            raise ImportError("Bokeh is unavailable: " + name)
        return None


sys.meta_path.insert(0, _BlockBokeh())
"""


def run_without_bokeh(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_BOKEH + textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )


def test_the_colour_lookup_imports_and_computes_without_bokeh():
    """The whole point: colour answers on a worker thread, no Bokeh in reach.

    Note this imports ``metabis.managers.color_lookup`` directly. Importing it
    via ``color_manager`` would pull ``bokeh.palettes`` and prove the opposite of
    what is claimed -- the split is exactly what makes the lookup reachable.
    """
    result = run_without_bokeh(
        """
        import json

        import pandas as pd

        from metabis.analysis.distribution_binner import DistributionBinner
        from metabis.managers.color_lookup import GraphColorLookup
        from metabis.managers.workspace_mode import AnalysisWorkspaceMode

        mode = AnalysisWorkspaceMode.GROUP_COMPARISON
        binner = DistributionBinner(
            data=pd.Series({"MAR00001": -3.0, "MAR00002": 0.0, "MAR00003": 3.0})
        )
        lookup = GraphColorLookup(
            neutral_color="#CCCCCC",
            up_color="#FF0000",
            down_color="#0000FF",
            group_colors={"treated": "#111111"},
            positive_influence_palette=("#E5F5E0", "#31A354"),
            negative_influence_palette=("#FEE6CE", "#E6550D"),
            distributions={
                (mode, "reaction", "log2_fc"): {
                    "binner": binner,
                    "bin_colors": {0: "#CCCCCC", 1: "#FF0000", -1: "#0000FF"},
                }
            },
        )

        print(json.dumps({
            "element": lookup.get_element_color_from_distribution(
                mode, "reaction", "MAR00003", "log2_fc"
            ),
            "influence": lookup.get_influence_color(0.9, 100.0),
            "group": lookup.get_group_color("treated"),
            "missing": lookup.get_element_color_from_distribution(
                mode, "gene", "ENSG0", "log2_fc"
            ),
        }))
        """
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["influence"] == "#31A354"
    assert payload["group"] == "#111111"
    assert payload["missing"] == "#CCCCCC"


def test_the_colour_manager_still_needs_bokeh():
    """Pins *where* the boundary is, so the test above cannot pass vacuously."""
    result = run_without_bokeh("import metabis.managers.color_manager")

    assert result.returncode != 0
    assert "Bokeh is unavailable" in result.stderr

"""Computing a graph artifact must not require Bokeh.

The artifact is what a worker thread produces and what an export reads. If Bokeh
crept back into that half of the graph code, a worker could construct a model again
without anyone noticing until a session raced. The boundary is therefore asserted on
the source, not on a mocked call: rendering may import Bokeh, analysis may not.

Two kinds of guard live here, and both are needed.

The *static* guard scans each calculation module's own import statements. It is
cheap and it names the offending file, but it is per-module: it cannot see a
Bokeh dependency reached through an innocent-looking edge. For example, a
calculation module may import a pure helper from a presentation module whose own
module-level imports load Bokeh. Every directly scanned file can be clean while
the transitive import graph still violates the boundary.

The *runtime* guards assert what ADR-0006 promises, by
importing and computing with Bokeh genuinely unavailable. They caught the
transitive edge the static scan structurally could not.
"""

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import metabis.viz.metabolic_visualization as graph_package

GRAPH_ROOT = Path(graph_package.__file__).parent

# Everything a graph artifact is built from. Rendering is deliberately absent.
# `pipeline.py` owns `build_artifact`; its display methods import the Bokeh
# adapter at call time rather than at module scope.
ANALYSIS_MODULES = (
    "artifact.py",
    "export.py",
    "errors.py",
    "pipeline.py",
    "enrichment",
    "generation",
)


def analysis_sources():
    for entry in ANALYSIS_MODULES:
        target = GRAPH_ROOT / entry
        if target.is_dir():
            yield from sorted(target.rglob("*.py"))
        else:
            yield target


def module_level_imports(source: Path):
    """Imports executed when the module is imported.

    Deliberately *not* ``ast.walk``: an import inside a function body runs only
    when that function is called, which is the sanctioned way for a display
    method to reach the Bokeh adapter without making its module Bokeh-dependent
    (``pipeline._handle_output`` does exactly this). What matters here is what
    ``import`` alone pulls in.

    A call-time Bokeh import in a *calculation* path would slip past this, which
    is why the runtime guards below compute a real artifact with Bokeh blocked.
    """
    tree = ast.parse(source.read_text(), filename=str(source))
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module
        elif isinstance(node, ast.If):
            # `if TYPE_CHECKING:` blocks never execute.
            continue


@pytest.mark.parametrize(
    "source", list(analysis_sources()), ids=lambda path: path.name
)
def test_graph_analysis_modules_do_not_import_bokeh(source):
    offenders = [
        module
        for module in module_level_imports(source)
        if module == "bokeh" or module.startswith("bokeh.")
    ]

    assert offenders == [], f"{source.name} imports Bokeh: {offenders}"


def test_the_boundary_covers_the_modules_that_actually_build_an_artifact():
    """A guard that scans nothing passes for the wrong reason."""
    sources = list(analysis_sources())

    assert len(sources) > 10
    assert any(source.name == "artifact.py" for source in sources)
    assert any(source.name == "enricher.py" for source in sources)


def test_the_static_guard_covers_the_module_that_owns_build_artifact():
    """The module under suspicion must be in the scan, or the scan proves nothing.

    Excluding ``pipeline.py`` because it imports Bokeh would make the guard pass
    by omitting the calculation entry point it is meant to protect.
    """
    scanned = {source.name for source in analysis_sources()}

    assert "pipeline.py" in scanned
    assert "build_artifact" in (GRAPH_ROOT / "pipeline.py").read_text()


# --- Runtime guards: what ADR-0006 promises, asserted against a real import ---

# Bokeh is blocked with a meta-path finder rather than by uninstalling it, so the
# probe runs in the normal environment. Each probe gets a fresh interpreter: a
# blocked import leaves a half-initialized package in ``sys.modules``, which would
# make a second in-process probe fail for the wrong reason.
_BLOCK_BOKEH = """
import importlib.abc
import sys


class _BlockBokeh(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "bokeh" or name.startswith("bokeh."):
            raise ImportError("Bokeh is unavailable to graph calculation: " + name)
        return None


sys.meta_path.insert(0, _BlockBokeh())
"""


def run_without_bokeh(body: str) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh interpreter where importing Bokeh raises."""
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_BOKEH + textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )


def test_the_probe_can_import_a_bokeh_free_module():
    """The blocker must not fail everything, or the guards below prove nothing."""
    result = run_without_bokeh(
        """
        import json

        print(json.dumps({"ok": True}))
        """
    )

    assert result.returncode == 0, result.stderr
    assert '{"ok": true}' in result.stdout


def test_bokeh_is_really_blocked_in_the_probe():
    result = run_without_bokeh("import bokeh.plotting")

    assert result.returncode != 0
    assert "Bokeh is unavailable to graph calculation" in result.stderr


def test_graph_calculation_imports_without_bokeh():
    result = run_without_bokeh(
        """
        from metabis.viz.metabolic_visualization.pipeline import (
            MetabolicGraphPipeline,
        )

        print("imported", MetabolicGraphPipeline.__name__)
        """
    )

    assert result.returncode == 0, result.stderr


def test_a_graph_artifact_is_computed_without_bokeh():
    """The structure-only graph needs no workspace, so this isolates Bokeh alone.

    348 nodes and 230 edges is the Human1 glycolysis structure the packaged
    layout ships; the digest-level goldens live in
    ``tests/viz/test_graph_output_characterization.py``.
    """
    result = run_without_bokeh(
        """
        import json

        from metabis.domain.metabolic_model import MetabolicModel
        from metabis.viz.metabolic_visualization.generation.generator import (
            GraphGenerationConfig,
        )
        from metabis.viz.metabolic_visualization.pipeline import (
            MetabolicGraphPipeline,
            PipelineConfig,
        )

        pipeline = MetabolicGraphPipeline(
            MetabolicModel(use_cache=True),
            PipelineConfig(
                generation_config=GraphGenerationConfig(
                    use_human1_map=True,
                    layout_engine="dot",
                ),
                show_plot=False,
            ),
        )
        artifact = pipeline.build_artifact(
            "Glycolysis / Gluconeogenesis",
            "subsystem",
            data_manager=None,
        )
        print(json.dumps({"nodes": len(artifact.nodes), "edges": len(artifact.edges)}))
        """
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip()) == {"nodes": 348, "edges": 230}

import json

from metabis import AnalysisWorkspace, MetabolicModel, plot_metabolic_graph
from metabis.viz.metabolic_visualization.generation.utilities.cache_utils import (
    clear_graph_resource_cache,
    get_graph_resource_cache_info,
    load_visualization_cache,
)
from metabis.viz.metabolic_visualization.pipeline import (
    MetabolicGraphPipeline,
    PipelineConfig,
)


def test_cached_graph_resources_return_independent_mutable_copies(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text(json.dumps({"glyphs": [{"id": "g1", "style": {"color": "red"}}], "connections": []}))
    clear_graph_resource_cache()

    first_glyphs, _ = load_visualization_cache(str(path))
    first_glyphs[0]["style"]["color"] = "blue"
    second_glyphs, _ = load_visualization_cache(str(path))

    assert second_glyphs[0]["style"]["color"] == "red"
    assert get_graph_resource_cache_info().hits == 1


# `show` is imported inside `_handle_output` rather than at module scope, so that
# the module owning `build_artifact` stays importable without Bokeh (AM10/ADR-0006).
# These patch it where it is defined; the display behaviour itself is unchanged.
def test_show_plot_false_never_calls_bokeh_show(monkeypatch):
    calls = []
    monkeypatch.setattr("bokeh.plotting.show", calls.append)
    pipeline = MetabolicGraphPipeline.__new__(MetabolicGraphPipeline)
    pipeline._handle_output(object(), PipelineConfig(show_plot=False))
    assert calls == []


def test_show_plot_true_retains_notebook_compatibility(monkeypatch):
    calls = []
    monkeypatch.setattr("bokeh.plotting.show", calls.append)
    pipeline = MetabolicGraphPipeline.__new__(MetabolicGraphPipeline)
    plot = object()
    pipeline._handle_output(plot, PipelineConfig(show_plot=True))
    assert calls == [plot]


def test_no_output_and_no_show_does_not_even_import_bokeh(monkeypatch):
    """The worker-thread path must not reach Bokeh, not even to decide to skip it."""
    import sys

    def refuse(name, *args, **kwargs):
        if name == "bokeh" or name.startswith("bokeh."):
            raise AssertionError(f"_handle_output imported {name} with nothing to do")
        return original(name, *args, **kwargs)

    original = __import__
    monkeypatch.setattr("builtins.__import__", refuse)
    for module in [name for name in sys.modules if name.startswith("bokeh")]:
        monkeypatch.delitem(sys.modules, module, raising=False)

    pipeline = MetabolicGraphPipeline.__new__(MetabolicGraphPipeline)
    pipeline._handle_output(object(), PipelineConfig(show_plot=False, output_html=None))


def test_packaged_subsystem_reports_typed_phase_metadata():
    model = MetabolicModel(use_cache=True)
    manager = AnalysisWorkspace(metabolic_model=model)

    plot, metadata = plot_metabolic_graph(
        "Glycolysis / Gluconeogenesis",
        data_manager=manager,
        graph_type="subsystem",
        show_plot=False,
        enable_timing=True,
        return_metadata=True,
    )

    assert plot is not None
    assert metadata.graph_type == "subsystem"
    assert metadata.resource_path and metadata.resource_variant in {"human1", "standard"}
    assert metadata.glyph_count > 0
    assert metadata.connection_count > 0
    assert {phase.name for phase in metadata.phases} >= {
        "resolve_request", "generate_graph", "enrich_graph", "render_graph", "total_pipeline"
    }

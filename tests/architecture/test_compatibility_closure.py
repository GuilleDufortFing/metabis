"""Internal compatibility inputs do not create shadow interfaces."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

from metabis.analysis.importance_analysis import ImportanceAnalysis
from metabis.domain.metabolic_model.loaders import create_subsystems_from_reactions
from metabis.managers.color_manager import ColorManager
from metabis.managers.workspace_mode import AnalysisWorkspaceMode
from metabis.viz.metabolic_visualization.enrichment.connection_enrichment import (
    enrich_connections,
)
from metabis.viz.metabolic_visualization.enrichment.utilities import (
    prepare_translations,
)
from metabis.viz.metabolic_visualization.generation.data_structures.constants import (
    VisualizationConstants,
)
from metabis.viz.metabolic_visualization.generation.data_structures.core_types import (
    GlyphClass,
)

SRC = Path(__file__).resolve().parents[2] / "src" / "metabis"


def test_dead_graph_and_subsystem_parameters_are_deleted():
    subsystem_parameters = inspect.signature(
        create_subsystems_from_reactions
    ).parameters
    connection_parameters = inspect.signature(enrich_connections).parameters
    translation_parameters = inspect.signature(prepare_translations).parameters

    assert tuple(subsystem_parameters) == ("reactions",)
    assert "classification_file_path" not in subsystem_parameters
    assert "reaction_colors" not in connection_parameters
    assert tuple(translation_parameters) == ("glyphs", "data_manager")


def test_translation_preparation_has_no_dead_reaction_colour_projection():
    manager = SimpleNamespace(metabolic_model=None)
    reaction = {"glyph_class": GlyphClass.REACTION, "label": "R1"}

    translations = prepare_translations([reaction], manager)

    assert translations == {
        "metabolites": {},
        "genes": {},
        "reactions": {"R1": "R1"},
    }


def test_importance_analysis_exposes_only_the_canonical_analysis_methods():
    assert hasattr(ImportanceAnalysis, "analyze_single_reaction_influence")
    assert hasattr(ImportanceAnalysis, "analyze_subsystem_influence")
    assert not hasattr(ImportanceAnalysis, "run_single_reaction_analysis")
    assert not hasattr(ImportanceAnalysis, "run_subsystem_analysis")
    assert not hasattr(VisualizationConstants, "CATEGORY_PALETTE")


def test_colour_manager_mode_interfaces_are_enum_only():
    assert (
        inspect.signature(ColorManager).parameters["default_mode"].annotation
        is AnalysisWorkspaceMode
    )
    assert (
        inspect.signature(ColorManager.set_mode).parameters["mode"].annotation
        is AnalysisWorkspaceMode
    )
    assert (
        inspect.signature(ColorManager.update_mode_configuration)
        .parameters["mode"]
        .annotation
        is AnalysisWorkspaceMode
    )

    source_path = SRC / "managers" / "color_manager.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    mode_methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"__init__", "set_mode", "update_mode_configuration"}
    }
    string_coercions = [
        node
        for method in mode_methods.values()
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "isinstance"
        and len(node.args) > 1
        and isinstance(node.args[1], ast.Name)
        and node.args[1].id == "str"
    ]

    assert string_coercions == []

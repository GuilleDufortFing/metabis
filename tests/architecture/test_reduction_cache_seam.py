"""Keep the reduction artifact authoritative over its compute cache projection."""

import ast
from pathlib import Path

from metabis.analysis import dimensionality_reduction, experiment_analysis

REDUCTION_SOURCE = Path(dimensionality_reduction.__file__)
EXPERIMENT_SOURCE = Path(experiment_analysis.__file__)


def _attribute_path(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def test_compatibility_named_reduction_projection_is_removed():
    source = REDUCTION_SOURCE.read_text()

    assert "as_legacy_result" not in source
    assert hasattr(dimensionality_reduction.ReductionArtifact, "to_compute_result")


def test_callers_cannot_write_through_the_public_results_view():
    offenders: list[tuple[int, str]] = []
    for source_path in (REDUCTION_SOURCE, EXPERIMENT_SOURCE):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Subscript):
                    continue
                path = _attribute_path(target.value)
                if path and path.endswith(".results"):
                    offenders.append((target.lineno, path))

    assert offenders == []


def test_compute_cache_subscript_writes_are_local_to_cache_result():
    tree = ast.parse(REDUCTION_SOURCE.read_text(), filename=str(REDUCTION_SOURCE))
    offenders: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):
            if not isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                continue
            targets = child.targets if isinstance(child, ast.Assign) else [child.target]
            for target in targets:
                if not isinstance(target, ast.Subscript):
                    continue
                if _attribute_path(target.value) == "self._results" and node.name != "cache_result":
                    offenders.append((target.lineno, node.name))

    assert offenders == []

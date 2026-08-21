from pathlib import Path

import pytest

from metabis.resource_registry import ResourceRegistry
from metabis.viz.metabolic_visualization.generation.data_structures.validation import (
    validate_graph_data,
)
from metabis.viz.metabolic_visualization.generation.utilities.cache_utils import (
    load_visualization_cache,
)

GRAPH_ROOT = Path(ResourceRegistry().resources_root) / "graphs"
GRAPH_RESOURCES = sorted(GRAPH_ROOT.rglob("*.json"))


@pytest.mark.resource
@pytest.mark.parametrize(
    "resource_path",
    GRAPH_RESOURCES,
    ids=lambda path: str(path.relative_to(GRAPH_ROOT)),
)
def test_packaged_graph_resource_is_valid(resource_path):
    glyphs, connections = load_visualization_cache(str(resource_path))

    valid, errors = validate_graph_data(glyphs, connections)

    assert valid, f"{resource_path}: {'; '.join(errors[:10])}"


def test_graph_resource_inventory_is_not_empty():
    assert GRAPH_RESOURCES, f"No packaged graph resources found below {GRAPH_ROOT}"

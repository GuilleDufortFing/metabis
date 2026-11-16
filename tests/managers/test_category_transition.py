"""A change of subsystem categories is a change of presentation, not of data.

Categories group subsystems in the overview and its ranking; no analysis reads
them. Installing new ones keeps the data, the groups and every result current,
and moves only the VISUALIZATION revision.
"""

import pytest

from metabis import AnalysisWorkspace, MetabolicModel
from metabis.analysis.source_identity import fingerprint_model


@pytest.fixture(scope="module")
def model() -> MetabolicModel:
    return MetabolicModel(use_cache=True)


def test_categories_are_not_part_of_the_models_identity(model, tmp_path):
    regrouped = model.copy()
    # The taxonomy's format: a heading per category, its subsystems' IDs under it.
    path = tmp_path / "categories.txt"
    path.write_text("### Everything\n" + "".join(f"- {subsystem}\n" for subsystem in sorted(model.get_subsystems())))
    regrouped.load_categories(str(path))

    assert regrouped.get_categories() != model.get_categories()
    assert fingerprint_model(regrouped) == fingerprint_model(model)


def test_a_category_transition_moves_only_the_visualization_revision(model):
    workspace = AnalysisWorkspace(metabolic_model=model)
    before = workspace.get_workspace_revisions()
    fresh = model.copy()
    fresh.load_categories(None)

    workspace.apply_category_transition(fresh)

    after = workspace.get_workspace_revisions()
    assert workspace.metabolic_model is fresh
    assert after.dataset == before.dataset and after.groups == before.groups
    assert after.visualization == before.visualization + 1


def test_a_model_whose_network_changed_is_not_a_category_transition(model):
    workspace = AnalysisWorkspace(metabolic_model=model)
    changed = model.copy()
    assert changed.filter_by_subsystem_and_compartment(filters=["glycolysis_gluconeogenesis"]) == 0

    with pytest.raises(ValueError, match="Only the subsystem categories may change"):
        workspace.apply_category_transition(changed)
    assert workspace.metabolic_model is model

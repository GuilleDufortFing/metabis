"""Atomic input transitions owned by the Analysis Workspace."""

import pandas as pd
import pytest

from metabis import AnalysisWorkspace, AnalysisWorkspaceMode, GroupDefinition


class _ReactionData:
    def __init__(self, name: str):
        self.frame = pd.DataFrame({name: [1.0, 2.0]}, index=["A_1", "B_1"])

    def get_reaction_df(self):
        return self.frame

    def get_sample_ids(self):
        return list(self.frame.index)


class _ExpressionData:
    def __init__(self, name: str):
        self.frame = pd.DataFrame({name: [3.0, 4.0]}, index=["A_1", "B_1"])

    def get_expression_df(self):
        return self.frame

    def get_feature_ids(self):
        return [str(column) for column in self.frame.columns]


class _Model:
    def __init__(self, name: str):
        self.name = name


class _IdentityModel:
    """A model whose subsystems and private cache both participate in a test.

    ``model_identity`` folds ``get_subsystems()``, so two ``_IdentityModel``
    objects with different subsystems carry different dataset fingerprints -- the
    signal AM8 needs to prove a transition re-pays source identity. ``_cache`` and
    ``clear_memory_cache`` mirror the real model's private operation cache.
    """

    def __init__(self, subsystems, cache=None):
        self._subsystems = list(subsystems)
        self._cache = dict(cache or {})

    def get_subsystems(self):
        return list(self._subsystems)

    def clear_memory_cache(self):
        self._cache.clear()


def _manager_with_installed_state(model=None):
    model = _Model("old") if model is None else model
    reaction = _ReactionData("R_old")
    expression = _ExpressionData("G_old")
    manager = AnalysisWorkspace(
        metabolic_model=model,
        reaction_data=reaction,
        expression_data=expression,
    )
    manager.group_manager.define_group(
        "A",
        ["A_1"],
        definition=GroupDefinition("custom"),
        all_sample_ids=reaction.get_sample_ids(),
    )
    manager.group_manager.set_active_group("A")
    manager.sample_metadata = object()
    manager.metadata_alignment_report = object()

    experiment = object()
    comparison = object()
    activity = object()
    manager.experiment_analysis = experiment
    manager.group_comparison_artifact = comparison
    manager.activity_artifact = activity
    manager.data_provider.experiment_analysis = experiment
    manager.data_provider.group_comparison_artifact = comparison
    manager.data_provider.activity_artifact = activity
    manager.subsystem_analysis._gc = comparison
    manager.subsystem_analysis._base_stats_cache = {("S", False): {"old": True}}
    manager.subsystem_analysis._delta_stats_cache = {("S", False): {"old": True}}
    manager.cache_manager.cache_hover("context", "R_old", "cached")
    manager.color_manager.generate_group_colors(["A"])
    manager.color_manager.register_distribution(
        mode=AnalysisWorkspaceMode.GROUP_COMPARISON,
        element_type="reaction",
        data=pd.Series([1.0, 2.0], index=["R1", "R2"]),
    )
    manager.hover_manager.update_group_comparison(comparison)
    manager.mode = AnalysisWorkspaceMode.GROUP_COMPARISON
    return manager, model, reaction, expression


def _assert_derived_state_was_invalidated(manager):
    assert manager.get_group_ids() == []
    assert manager.group_manager.group_records == {}
    assert manager.group_manager.get_active_group() is None
    assert manager.experiment_analysis is None
    assert manager.group_comparison_artifact is None
    assert manager.activity_artifact is None
    assert manager.data_provider.experiment_analysis is None
    assert manager.data_provider.group_comparison_artifact is None
    assert manager.data_provider.activity_artifact is None
    assert manager.subsystem_analysis._base_stats_cache == {}
    assert manager.subsystem_analysis._delta_stats_cache == {}
    assert manager.cache_manager._hover_cache == {}
    assert manager.color_manager.get_group_colors() == {}
    assert manager.color_manager._distributions == {}
    assert manager.hover_manager.group_comparison_artifact is None
    assert manager.get_mode() is AnalysisWorkspaceMode.NO_EXPERIMENT


def test_reaction_and_model_replacement_is_one_atomic_workspace_transition():
    manager, _old_model, _old_reaction, expression = _manager_with_installed_state()
    new_model = _Model("new")
    new_reaction = _ReactionData("R_new")
    revisions_before = manager.get_workspace_revisions()

    manager.replace_reaction_data(
        new_reaction,
        metabolic_model=new_model,
        output_reaction_format="rxn_name",
    )

    assert manager.metabolic_model is new_model
    assert manager.reaction_data is new_reaction
    assert manager.expression_data is expression
    assert manager.data_provider.metabolic_model is new_model
    assert manager.data_provider.reaction_data is new_reaction
    assert manager.data_provider.expression_data is expression
    assert manager.output_reaction_format == "rxn_name"
    assert manager.dataset_revision == revisions_before.dataset + 1
    assert manager.preprocessing_revision == revisions_before.preprocessing + 1
    assert manager.group_revision == revisions_before.groups + 1
    assert manager.metadata_revision == revisions_before.metadata + 1
    assert manager.sample_metadata is None
    assert manager.metadata_alignment_report is None
    _assert_derived_state_was_invalidated(manager)


def test_reaction_replacement_can_clear_the_expression_in_the_same_transition():
    manager, _old_model, _old_reaction, _expression = _manager_with_installed_state()
    new_model = _Model("new")
    new_reaction = _ReactionData("R_new")
    revisions_before = manager.get_workspace_revisions()

    manager.replace_reaction_data(new_reaction, metabolic_model=new_model, clear_expression=True)

    assert manager.reaction_data is new_reaction
    assert manager.expression_data is None
    assert manager.data_provider.expression_data is None
    assert manager.sample_metadata is None
    assert manager.dataset_revision == revisions_before.dataset + 1
    _assert_derived_state_was_invalidated(manager)


def test_expression_replacement_preserves_other_inputs_and_bumps_once():
    manager, model, reaction, _old_expression = _manager_with_installed_state()
    new_expression = _ExpressionData("G_new")
    revisions_before = manager.get_workspace_revisions()

    manager.replace_expression_data(
        new_expression,
        output_gene_format="entrez",
    )

    assert manager.metabolic_model is model
    assert manager.reaction_data is reaction
    assert manager.expression_data is new_expression
    assert manager.data_provider.metabolic_model is model
    assert manager.data_provider.reaction_data is reaction
    assert manager.data_provider.expression_data is new_expression
    assert manager.output_gene_format == "entrez"
    assert manager.dataset_revision == revisions_before.dataset + 1
    assert manager.preprocessing_revision == revisions_before.preprocessing + 1
    assert manager.group_revision == revisions_before.groups + 1
    assert manager.metadata_revision == revisions_before.metadata + 1
    _assert_derived_state_was_invalidated(manager)


def test_model_only_replacement_does_not_claim_a_preprocessing_change():
    manager, _old_model, reaction, expression = _manager_with_installed_state()
    new_model = _Model("new")
    revisions_before = manager.get_workspace_revisions()

    manager.replace_metabolic_model(new_model)

    assert manager.metabolic_model is new_model
    assert manager.data_provider.metabolic_model is new_model
    assert manager.reaction_data is reaction
    assert manager.expression_data is expression
    assert manager.dataset_revision == revisions_before.dataset + 1
    assert manager.preprocessing_revision == revisions_before.preprocessing
    assert manager.group_revision == revisions_before.groups + 1
    assert manager.metadata_revision == revisions_before.metadata
    assert manager.sample_metadata is not None
    _assert_derived_state_was_invalidated(manager)


def test_full_dataset_reset_retains_model_and_clears_inputs_and_formats():
    manager, model, _reaction, _expression = _manager_with_installed_state()
    manager.output_reaction_format = "rxn_name"
    manager.output_gene_format = "entrez"
    revisions_before = manager.get_workspace_revisions()

    manager.reset_dataset()

    assert manager.metabolic_model is model
    assert manager.data_provider.metabolic_model is model
    assert manager.reaction_data is None
    assert manager.expression_data is None
    assert manager.data_provider.reaction_data is None
    assert manager.data_provider.expression_data is None
    assert manager.output_reaction_format == "mar_id"
    assert manager.output_gene_format == "geneSymbols"
    assert manager.dataset_revision == revisions_before.dataset + 1
    assert manager.preprocessing_revision == revisions_before.preprocessing + 1
    assert manager.group_revision == revisions_before.groups + 1
    assert manager.metadata_revision == revisions_before.metadata + 1
    _assert_derived_state_was_invalidated(manager)


# --- AM8: the model-transition seam ------------------------------------------
# A live model change (subsystem/compartment filter, filter reset, or category
# replacement) is one atomic DATASET transaction through apply_model_transition:
# a fresh model is installed, the DATASET revision advances, source identity is
# re-paid at the writer, and every DATASET-dependent artifact and user group is
# invalidated (M-10). Metadata is retained -- a model transition does not relabel
# samples -- and no preprocessing change is claimed.


def test_model_transition_installs_fresh_model_as_one_dataset_transaction():
    installed = _IdentityModel(["S_old"])
    manager, _old_model, reaction, expression = _manager_with_installed_state(
        model=installed
    )
    fresh = _IdentityModel(["S_new"])
    fingerprint_before = manager.source_fingerprints["dataset"]
    revisions_before = manager.get_workspace_revisions()

    manager.apply_model_transition(fresh)

    # The fresh model is authoritative on both the workspace and its provider.
    assert manager.metabolic_model is fresh
    assert manager.data_provider.metabolic_model is fresh
    # Measured inputs and sample metadata are untouched by a model transition.
    assert manager.reaction_data is reaction
    assert manager.expression_data is expression
    assert manager.sample_metadata is not None
    assert manager.metadata_alignment_report is not None
    # DATASET advances and groups are cleared; nothing else moves.
    assert manager.dataset_revision == revisions_before.dataset + 1
    assert manager.group_revision == revisions_before.groups + 1
    assert manager.preprocessing_revision == revisions_before.preprocessing
    assert manager.metadata_revision == revisions_before.metadata
    # Identity is re-paid at the writer: the dataset fingerprint follows the model.
    assert manager.source_fingerprints["dataset"] != fingerprint_before
    _assert_derived_state_was_invalidated(manager)


def test_model_transition_clears_the_fresh_models_private_cache():
    manager, _old_model, _reaction, _expression = _manager_with_installed_state(
        model=_IdentityModel(["S_old"])
    )
    fresh = _IdentityModel(["S_new"], cache={"stale": "entry"})

    manager.apply_model_transition(fresh)

    assert fresh._cache == {}


def test_model_transition_rejects_the_installed_object_in_place():
    manager, _old_model, reaction, _expression = _manager_with_installed_state(
        model=_IdentityModel(["S_old"])
    )
    installed = manager.metabolic_model
    fingerprint_before = manager.source_fingerprints["dataset"]
    revisions_before = manager.get_workspace_revisions()

    with pytest.raises(ValueError, match="fresh model object"):
        manager.apply_model_transition(installed)

    # A refused transition changes nothing: the installed model, revisions,
    # identity, and every derived result stay coherent.
    assert manager.metabolic_model is installed
    assert manager.get_workspace_revisions() == revisions_before
    assert manager.source_fingerprints["dataset"] == fingerprint_before
    assert manager.experiment_analysis is not None
    assert manager.group_comparison_artifact is not None
    assert manager.get_group_ids() == ["A"]


def test_model_transition_rejects_none_without_touching_state():
    manager, installed, _reaction, _expression = _manager_with_installed_state()
    revisions_before = manager.get_workspace_revisions()

    with pytest.raises(ValueError, match="required"):
        manager.apply_model_transition(None)

    assert manager.metabolic_model is installed
    assert manager.get_workspace_revisions() == revisions_before
    assert manager.experiment_analysis is not None
    assert manager.get_group_ids() == ["A"]


def test_replace_metabolic_model_shares_the_transition_seam():
    manager, _old_model, reaction, expression = _manager_with_installed_state()
    new_model = _Model("new")
    revisions_before = manager.get_workspace_revisions()

    manager.replace_metabolic_model(new_model)

    assert manager.metabolic_model is new_model
    assert manager.reaction_data is reaction
    assert manager.expression_data is expression
    assert manager.dataset_revision == revisions_before.dataset + 1
    assert manager.preprocessing_revision == revisions_before.preprocessing
    _assert_derived_state_was_invalidated(manager)

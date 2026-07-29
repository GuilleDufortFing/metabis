"""AM7: the public reload path replaces the server's private reset call.

The server used to reset a subsystem filter by calling
``MetabolicModel._initialize_all_data()`` directly. ``reload_all_data()`` is the
public inverse of ``filter_by_subsystem_and_compartment`` it now uses instead.
"""

from metabis import MetabolicModel


def test_reload_all_data_restores_the_full_model_after_a_filter():
    model = MetabolicModel(use_cache=True)
    full_reactions = set(model.get_reactions())
    assert full_reactions, "packaged model should load reactions"

    first_subsystem = model.get_subsystems()[0]
    model.filter_by_subsystem_and_compartment([first_subsystem])
    filtered_reactions = set(model.get_reactions())
    assert filtered_reactions < full_reactions, "filter should reduce the model"

    model.reload_all_data()

    assert set(model.get_reactions()) == full_reactions


def test_copy_is_a_fresh_independent_object():
    """AM8: a model transition installs a copy, so copy() must be independent."""
    model = MetabolicModel(use_cache=True)

    clone = model.copy()

    assert clone is not model
    # The category manager must point back at the copy, not the original, or a
    # category change on the copy would mutate the installed model.
    assert clone.category_manager is not model.category_manager
    assert clone.category_manager.model is clone


def test_copy_preserves_current_state():
    """The copy carries whatever the original currently is, not a fresh build."""
    model = MetabolicModel(use_cache=True)
    first_subsystem = model.get_subsystems()[0]
    model.filter_by_subsystem_and_compartment([first_subsystem])
    filtered_reactions = set(model.get_reactions())

    clone = model.copy()

    assert set(clone.get_reactions()) == filtered_reactions


def test_mutating_the_copy_leaves_the_original_untouched():
    """The transition applies its change to the copy; the installed model is safe."""
    model = MetabolicModel(use_cache=True)
    full_reactions = set(model.get_reactions())

    clone = model.copy()
    clone.filter_by_subsystem_and_compartment([clone.get_subsystems()[0]])

    assert set(clone.get_reactions()) < full_reactions
    assert set(model.get_reactions()) == full_reactions

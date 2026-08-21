"""The metabolite subsystem fan-out is memoized, and must stay correct under a model transition.

``get_metabolite_metadata`` reports which subsystems produce and consume a
metabolite. That is a fan-out over every incident reaction, and the degree
distribution is brutally skewed -- cytosolic water is incident to 3,352
reactions in Human1. Recomputing it per call made ``get_metabolite_data`` 36x
more expensive per call than its gene sibling and drove 225,294 entity lookups
while drawing one 348-node subsystem graph, because hover asks for the same
metabolite up to 36 times per render.

Memoizing is only safe because AM8 made the model replace-only: every transition
installs a *fresh* model object through ``apply_model_transition``, which clears
the model's memory cache, and ``test_installed_model_immutability.py`` forbids mutating an
installed model in place. These tests pin both halves -- that the memo returns
what the walk returned, and that clearing the cache really re-derives it -- so a
future change that reintroduces in-place model mutation fails here rather than
silently serving stale subsystem lists into hover text.

The equivalence half is asserted against a reference walk rather than a recorded
digest, so it stays meaningful if the packaged model is regenerated.
"""

import pytest

from metabis import MetabolicModel


@pytest.fixture(scope="module")
def model():
    return MetabolicModel(use_cache=True)


def reference_fanout(model, metabolite):
    """The per-call walk the memo replaced, resolved the slow way on purpose."""
    produced, consumed = set(), set()
    for attribute, sink in (("produced_by", produced), ("consumed_by", consumed)):
        for reaction_id in getattr(metabolite, attribute, []) or []:
            reaction = model.reactions.get(reaction_id, "primary")
            if reaction and getattr(reaction, "subsystem_id", None):
                subsystem = model.subsystems.get(reaction.subsystem_id, "primary")
                if subsystem and getattr(subsystem, "name", None):
                    sink.add(subsystem.name)
    return sorted(produced), sorted(consumed)


def hub_metabolites(model, limit=25):
    """The highest-degree metabolites, where the memo actually earns its keep."""
    scored = [
        (
            len(getattr(m, "produced_by", []) or [])
            + len(getattr(m, "consumed_by", []) or []),
            m,
        )
        for m in model.metabolites.get_all()
    ]
    scored.sort(key=lambda pair: -pair[0])
    return [metabolite for _, metabolite in scored[:limit]]


@pytest.mark.integration
def test_the_memo_reports_what_the_walk_reported(model):
    """Highest-degree metabolites, where a wrong index would show up first."""
    for metabolite in hub_metabolites(model):
        expected_produced, expected_consumed = reference_fanout(model, metabolite)
        metadata = model.get_metabolite_metadata(metabolite.id)

        assert metadata["produced_by_subsystems"] == expected_produced
        assert metadata["consumed_by_subsystems"] == expected_consumed


@pytest.mark.integration
def test_a_repeat_lookup_returns_the_same_value(model):
    """Hover asks for the same metabolite dozens of times per render."""
    metabolite = hub_metabolites(model, limit=1)[0]

    first = model.get_metabolite_metadata(metabolite.id)
    second = model.get_metabolite_metadata(metabolite.id)

    assert first["produced_by_subsystems"] == second["produced_by_subsystems"]
    assert first["consumed_by_subsystems"] == second["consumed_by_subsystems"]


@pytest.mark.integration
def test_clearing_the_memory_cache_re_derives_the_fanout(model):
    """The invalidation an AM8 model transition relies on.

    ``AnalysisWorkspace.apply_model_transition`` clears the prepared model's
    memory cache. If the fan-out did not live there, a filtered model would keep
    serving the unfiltered model's subsystem lists.
    """
    metabolite = hub_metabolites(model, limit=1)[0]
    before = model.get_metabolite_metadata(metabolite.id)["produced_by_subsystems"]

    assert model._cache.get("metabolite_subsystem_fanout")
    model.clear_memory_cache()
    assert not model._cache.get("metabolite_subsystem_fanout")

    assert model.get_metabolite_metadata(metabolite.id)["produced_by_subsystems"] == before


@pytest.mark.integration
def test_the_cached_fanout_cannot_be_mutated_by_a_caller(model):
    """One value serves every later hover, so it must not be writable."""
    metabolite = hub_metabolites(model, limit=1)[0]
    model.get_metabolite_metadata(metabolite.id)

    cached = model._cache["metabolite_subsystem_fanout"][metabolite.id]

    assert isinstance(cached[0], tuple) and isinstance(cached[1], tuple)
    with pytest.raises(TypeError):
        cached[0][0] = "Rewritten"


@pytest.mark.integration
def test_the_all_metabolites_branch_agrees_with_the_single_lookup(model):
    """Two code paths report subsystems; they must not disagree."""
    everything = model.get_metabolite_metadata()

    for metabolite in hub_metabolites(model, limit=10):
        single = model.get_metabolite_metadata(metabolite.id)
        assert everything[metabolite.id]["subsystems"] == sorted(
            set(single["produced_by_subsystems"]) | set(single["consumed_by_subsystems"])
        )

"""Detached subsystem delta artifact contracts."""

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from metabis.analysis.subsystem_analysis import SubsystemAnalysis


class _RecordingColorManager:
    def __init__(self):
        self.registrations = []

    def register_distribution(self, **registration):
        self.registrations.append(registration)


class _AnalysisWorkspace:
    def __init__(self):
        self.color_manager = _RecordingColorManager()


def _analysis_with_delta_state():
    analysis = SubsystemAnalysis(_AnalysisWorkspace())
    analysis._base_stats_cache = {
        ("Glycolysis", False): {
            "is_in_model": True,
            "nested": {"members": ["R1", "R2"]},
        },
        ("Glycolysis", True): {
            "is_in_model": True,
            "nested": {"members": ["G1"]},
        },
    }
    analysis._delta_stats_cache = {
        ("Glycolysis", False): {
            "delta_subsystem_score": 1.25,
            "delta_subsystem_meta": {"computed": True, "metric": "cohens_d"},
        },
        ("Glycolysis", True): {
            "delta_subsystem_score": -0.75,
            "delta_subsystem_meta": {"computed": True, "metric": "cohens_d"},
        },
    }
    analysis._current_delta_config = {
        "metric": "cohens_d",
        "filters": ["significant"],
    }
    return analysis


def test_delta_artifact_roundtrips_all_scientific_state():
    source = _analysis_with_delta_state()
    artifact = source.get_delta_artifact()
    target = SubsystemAnalysis(_AnalysisWorkspace())

    returned = target.install_delta_artifact(
        artifact,
        register_presentation=False,
    )
    installed = target.get_delta_artifact()

    assert returned is artifact
    assert installed.base_stats == artifact.base_stats
    assert installed.delta_stats == artifact.delta_stats
    assert installed.config == artifact.config
    assert installed.distributions == artifact.distributions
    assert target.dm.color_manager.registrations == []


def test_delta_artifact_and_installed_state_do_not_alias_source_or_each_other():
    source = _analysis_with_delta_state()
    artifact = source.get_delta_artifact()

    source._base_stats_cache[("Glycolysis", False)]["nested"]["members"].append("R3")
    source._delta_stats_cache[("Glycolysis", False)]["delta_subsystem_score"] = 99.0
    source._current_delta_config["filters"].append("all")

    assert artifact.base_stats[("Glycolysis", False)]["nested"]["members"] == (
        "R1",
        "R2",
    )
    assert artifact.delta_stats[("Glycolysis", False)]["delta_subsystem_score"] == 1.25
    assert artifact.config == {"metric": "cohens_d", "filters": ("significant",)}

    target = SubsystemAnalysis(_AnalysisWorkspace())
    target.install_delta_artifact(artifact, register_presentation=False)
    target._base_stats_cache[("Glycolysis", False)]["nested"]["members"].append("R4")
    target._delta_stats_cache[("Glycolysis", False)]["delta_subsystem_score"] = -99.0
    target._current_delta_config["filters"].append("all")

    assert artifact.base_stats[("Glycolysis", False)]["nested"]["members"] == (
        "R1",
        "R2",
    )
    assert artifact.delta_stats[("Glycolysis", False)]["delta_subsystem_score"] == 1.25
    assert artifact.config == {"metric": "cohens_d", "filters": ("significant",)}


def test_delta_artifact_install_registers_detached_presentation_distributions():
    artifact = _analysis_with_delta_state().get_delta_artifact()
    target = SubsystemAnalysis(_AnalysisWorkspace())

    target.install_delta_artifact(artifact, register_presentation=True)

    registrations = {
        registration["element_type"]: registration
        for registration in target.dm.color_manager.registrations
    }
    assert set(registrations) == {"subsystem", "subsystem_genes"}
    assert all(
        registration["metric_name"] == "cohens_d"
        for registration in registrations.values()
    )
    pd.testing.assert_series_equal(
        registrations["subsystem"]["data"],
        pd.Series({"Glycolysis": 1.25}, dtype=float),
    )
    pd.testing.assert_series_equal(
        registrations["subsystem_genes"]["data"],
        pd.Series({"Glycolysis": -0.75}, dtype=float),
    )


def test_delta_artifact_is_frozen():
    artifact = _analysis_with_delta_state().get_delta_artifact()

    with pytest.raises(FrozenInstanceError):
        artifact.config = None

    with pytest.raises(TypeError):
        artifact.base_stats[("Glycolysis", False)]["nested"]["members"] += ("R3",)
    with pytest.raises(TypeError):
        artifact.delta_stats[("Glycolysis", False)]["delta_subsystem_meta"][
            "computed"
        ] = False


def test_real_base_context_is_removed_from_the_cross_worker_artifact():
    source = _analysis_with_delta_state()
    source._base_stats_cache[("Glycolysis", False)]["_context"] = {
        "stats_frame": pd.DataFrame({"cohens_d": [1.25]}, index=["R1"]),
        "elements": ["R1"],
        "significant_set": {"R1"},
        "skipped_due_to_missing": 0,
    }

    artifact = source.get_delta_artifact()
    target = SubsystemAnalysis(_AnalysisWorkspace())
    target.install_delta_artifact(artifact, register_presentation=False)

    assert "_context" not in artifact.base_stats[("Glycolysis", False)]
    assert target.get_delta_artifact() == artifact

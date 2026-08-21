"""An artifact is stale only when an input it depends on moves (AM5).

Every artifact declares the revisions its science depends on, and both
``require_current`` and ``artifact_id`` already honour that declaration. The
installation seam did not: it compared the whole six-counter revision vector, so
a counter the artifact does not depend on made it un-installable.

Nothing caught it because no production caller ever re-installed a previously
built artifact -- the seams had no callers at all (MR-12). The reduction caller
is the one that does: the companion PCA plots reuse the installed reduction, and
a session that applies a comparison threshold in between advances the threshold
revision. A reduction does not depend on thresholds, and is no less current for
one having changed.
"""

from __future__ import annotations

import numpy as np
import pytest

from metabis import AnalysisWorkspace, ExperimentAnalysis
from metabis.analysis.dimensionality_reduction import ReductionKey, ReductionResult
from metabis.analysis_workspace import RevisionKind

REDUCTION_DEPENDENCIES = (RevisionKind.DATASET, RevisionKind.PREPROCESSING)


def _workspace() -> AnalysisWorkspace:
    manager = AnalysisWorkspace()
    analysis = ExperimentAnalysis(manager, auto_run=False)
    provenance = manager.capture_analysis_snapshot().provenance
    manager.install_experiment_analysis(provenance, analysis)
    manager.refresh_experiment_analysis_presentation()
    return manager


def _provenance(manager: AnalysisWorkspace):
    return manager.capture_analysis_snapshot(REDUCTION_DEPENDENCIES).provenance


def _artifact(manager: AnalysisWorkspace, provenance=None):
    result = ReductionResult(
        key=ReductionKey(
            method="PCA",
            data_type="reactions",
            n_variable_features=2,
            n_components=2,
            random_state=0,
            preprocessing_revision="1",
            sample_revision="1",
        ),
        transformed_data=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        sample_ids=("s1", "s2", "s3"),
        feature_names=("R1", "R2"),
        metadata={
            "explained_variance_ratio": np.array([0.7, 0.3]),
            "components": np.array([[0.6, 0.8], [0.8, -0.6]]),
            "n_variable_features_used": 2,
            "data_source_name": "reaction data",
            "n_components_retained": 2,
        },
    )
    return result.build_artifact(
        provenance=provenance or _provenance(manager),
        source_fingerprints=manager.source_fingerprints,
    )


def test_a_threshold_change_does_not_make_an_installed_reduction_stale():
    """The regression: applying a threshold locked out every PCA companion plot."""
    manager = _workspace()
    artifact = _artifact(manager)
    manager.install_reduction_artifact(_provenance(manager), artifact)

    manager.mark_thresholds_changed()

    # Reusing the installed reduction is what the loadings and biplot workflows
    # do, and the reduction depends on neither thresholds nor the visualization.
    reused = manager.get_reduction_artifact("PCA", "reactions")
    manager.require_current_artifact(reused)
    assert manager.install_reduction_artifact(_provenance(manager), reused) is reused


def test_a_preprocessing_change_still_makes_it_stale():
    """Scoping freshness to dependencies must not stop it rejecting a real change."""
    manager = _workspace()
    artifact = _artifact(manager)

    manager.mark_preprocessing_changed()

    with pytest.raises(ValueError, match="does not match the live workspace token"):
        manager.install_reduction_artifact(_provenance(manager), artifact)


def test_a_workflow_can_install_an_artifact_another_workflow_built():
    """The group analysis runs a PCA; the companion plots replot it.

    Those two workflows declare different dependency sets -- the group analysis
    also depends on groups and metadata. The artifact's own dependencies decide
    whether it is stale; requiring the caller's set to match would make an
    artifact installable only by the workflow that happened to build it.
    """
    manager = _workspace()
    broad = manager.capture_analysis_snapshot(
        (
            RevisionKind.DATASET,
            RevisionKind.METADATA,
            RevisionKind.GROUPS,
            RevisionKind.PREPROCESSING,
        )
    ).provenance
    artifact = _artifact(manager, provenance=broad)

    # Installed against the narrower token the reduction workflow captures.
    installed = manager.install_reduction_artifact(_provenance(manager), artifact)

    assert installed is artifact
    assert manager.get_reduction_artifact("PCA", "reactions") is artifact


def test_an_artifact_from_another_workspace_is_never_current():
    manager = _workspace()
    artifact = _artifact(manager)

    other = _workspace()

    with pytest.raises(ValueError, match="does not match the live workspace token"):
        other.install_reduction_artifact(_provenance(other), artifact)

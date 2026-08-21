from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metabis.analysis_workspace import RevisionKind

FIXTURES = Path(__file__).parent / "fixtures"

COMPARISON_INSTALL_DEPENDENCIES = frozenset(
    {
        RevisionKind.DATASET,
        RevisionKind.METADATA,
        RevisionKind.GROUPS,
        RevisionKind.PREPROCESSING,
    }
)


def install_comparison_artifact(
    manager, comparison, dependencies=COMPARISON_INSTALL_DEPENDENCIES
):
    """Build a comparison's artifact from a manager and install it.

    Since AM7 deleted the mutable comparison view, tests install the artifact the
    worker would build, so hover/plots/colouring read exactly the production seam.
    """
    snapshot = manager.capture_analysis_snapshot(dependencies)
    artifact = comparison.build_artifact(
        provenance=snapshot.provenance,
        source_fingerprints=manager.source_fingerprints,
    )
    manager.install_group_comparison_artifact(snapshot.provenance, artifact)
    manager.refresh_group_comparison_presentation()
    return artifact


@pytest.fixture
def install_comparison():
    """Return a callable that builds and installs a comparison's artifact."""
    return install_comparison_artifact


@pytest.fixture
def fixture_dir() -> Path:
    """Return the directory containing inspectable tabular fixtures."""
    return FIXTURES


@pytest.fixture
def reaction_frame() -> pd.DataFrame:
    """Return samples x reactions with zero, constant, negative, and non-finite cases."""
    return pd.read_csv(FIXTURES / "reaction_single.tsv", sep="\t", index_col=0)


@pytest.fixture
def metadata_frame() -> pd.DataFrame:
    """Return sample metadata with two factors and a named custom union."""
    return pd.read_csv(FIXTURES / "metadata.tsv", sep="\t", index_col=0)


def make_synthetic_flux_matrix(
    n_samples: int,
    n_reactions: int,
    zero_fraction: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Create deterministic samples x reactions benchmark input."""
    if not 0 <= zero_fraction <= 1:
        raise ValueError("zero_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n_samples, n_reactions))
    values[rng.random(values.shape) < zero_fraction] = 0.0
    return pd.DataFrame(
        values,
        index=[f"sample_{index:05d}" for index in range(n_samples)],
        columns=[f"MAR{index:05d}" for index in range(n_reactions)],
    )


@dataclass(frozen=True)
class TinyReaction:
    id: str
    reversible: bool
    subsystems: tuple[str, ...]


@pytest.fixture
def tiny_model() -> tuple[TinyReaction, ...]:
    """Return a coherent test-only model with overlapping subsystem membership."""
    return (
        TinyReaction("MAR00001", True, ("glycolysis", "shared pathway")),
        TinyReaction("MAR00002", False, ("glycolysis",)),
        TinyReaction("MAR00003", False, ()),
        TinyReaction("MAR00004", True, ("shared pathway",)),
    )

"""AM6 slice 7: the sparse load stays sparse through load, identity, and analysis.

Earlier slices moved every supported *analysis* reader onto the bounded matrix
seam; slice 7 removes the unconditional full-matrix ``to_df()`` at load, so the
dense frame is never built for a sparse input. These tests pin the two facts that
makes true end to end:

- **Identity is paid for without densifying.** A dataset load fingerprints its
  expression through ``identity_frame`` (a sparse-columned frame), whose hash is
  byte-identical to the dense twin's -- so provenance does not move -- while the
  full dense frame is never materialized (``df_expression_data`` stays ``None``).
- **A whole supported workflow (averages, PCA, activity) never densifies**, and
  its peak memory is a fraction of the dense frame it used to build.

The workspace is built the way the app builds one: constructed empty, then loaded
through ``replace_reaction_data``/``replace_expression_data`` so the identity
cache is warmed at the load writer (ADR-0007), exactly where a real session pays.
"""

import tracemalloc

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from metabis import (
    ActivityConfig,
    ExperimentAnalysis,
    ExpressionData,
    GroupSummary,
)
from metabis.analysis.artifacts import (
    fingerprint_expression_matrix,
    fingerprint_frame,
)
from metabis.analysis_workspace import RevisionKind, StaleAnalysisResultError
from metabis.domain.expression_matrix import SparseExpressionMatrix
from metabis.managers.group_manager import GroupDefinition
from metabis.managers.workspace import AnalysisWorkspace


def _sparse_expression(tmp_path, n_obs, n_vars, density=0.05, seed=3):
    samples = [f"s{i}" for i in range(n_obs)]
    genes = [f"g{j}" for j in range(n_vars)]
    rng = np.random.default_rng(seed)
    matrix = sparse.random(
        n_obs, n_vars, density=density, format="csr", random_state=rng, dtype=np.float32
    )
    path = tmp_path / "expr.h5ad"
    AnnData(
        X=matrix,
        obs=pd.DataFrame(index=samples),
        var=pd.DataFrame(index=genes),
    ).write_h5ad(path)
    expression = ExpressionData(
        path=str(path), gene_list=genes, data_format="h5ad"
    )
    assert isinstance(expression.matrix(), SparseExpressionMatrix)
    return expression, samples, genes


class _ReactionData:
    def __init__(self, samples):
        rng = np.random.default_rng(7)
        self.frame = pd.DataFrame(
            rng.normal(size=(len(samples), 3)),
            index=samples,
            columns=["R1", "R2", "R3"],
        )

    def get_reaction_df(self):
        return self.frame

    def get_sample_ids(self):
        return list(self.frame.index)


class _Model:
    def get_reaction_metadata(self, _reaction_id):
        return {"subsystems": ["Glycolysis"]}

    def get_gene_metadata(self, _gene_id):
        return {"subsystems": ["Glycolysis"]}

    def get_subsystems(self):
        return ["Glycolysis"]

    def get_reactions_for_subsystem(self, _subsystem):
        return ["R1", "R2", "R3"]

    def get_genes_for_subsystem(self, _subsystem):
        return []


def _manager(expression, samples) -> AnalysisWorkspace:
    model = _Model()
    manager = AnalysisWorkspace(metabolic_model=model)
    manager.replace_reaction_data(_ReactionData(samples), metabolic_model=model)
    manager.replace_expression_data(expression)
    half = len(samples) // 2
    manager.group_manager.define_group(
        "A", samples[:half], definition=GroupDefinition("custom")
    )
    manager.group_manager.define_group(
        "B", samples[half:], definition=GroupDefinition("custom")
    )
    return manager


def test_matrix_fingerprint_equals_the_dense_frame_fingerprint(tmp_path):
    """The bounded matrix fingerprint equals the dense twin's, byte for byte.

    A sparse load's identity is computed through ``fingerprint_expression_matrix``
    (bounded, one column block at a time) and must equal ``fingerprint_frame`` over
    the full dense frame, so provenance does not move when the load goes sparse.
    """
    expression, _samples, _genes = _sparse_expression(tmp_path, 60, 300)

    bounded = fingerprint_expression_matrix(expression.matrix())
    dense = fingerprint_frame(expression.get_expression_df())

    assert bounded == dense


def test_loading_a_sparse_dataset_pays_for_identity_without_densifying(tmp_path):
    """The load writer fingerprints expression without building the dense frame.

    ``replace_expression_data`` warms the source-identity cache at the load writer.
    Before slice 7 that hashed the dense ``to_df()`` frame and left it resident;
    now it hashes the sparse-columned ``identity_frame`` and ``df_expression_data``
    stays ``None`` -- and the dataset fingerprint equals the dense twin's, so
    provenance is unmoved.
    """
    expression, samples, _genes = _sparse_expression(tmp_path, 60, 300)
    dense_dataset_fingerprint = fingerprint_frame(expression.get_expression_df())
    # Reset to the sparse-first state the load starts from (the line above forced
    # the escape hatch, which caches; a fresh instance mirrors a real load).
    expression, samples, _genes = _sparse_expression(tmp_path, 60, 300)

    manager = _manager(expression, samples)

    assert expression.df_expression_data is None  # the load did not densify
    fingerprints = manager.source_fingerprints
    assert fingerprints["expression"] == dense_dataset_fingerprint


def test_a_supported_workflow_never_densifies_the_whole_matrix(tmp_path):
    """Gene averages, PCA, and activity all run without building the dense frame."""
    expression, samples, _genes = _sparse_expression(tmp_path, 80, 500)
    manager = _manager(expression, samples)

    analysis = ExperimentAnalysis(
        manager, auto_run=False, register_distributions=False
    )
    analysis._calculate_gene_avgs()
    analysis.calculate_pca(data_type="expression", n_variable_features=100)
    analysis.compute_activity(
        ActivityConfig(group_id="A", summary=GroupSummary.MEAN)
    )

    # Not one of them materialized the full dense frame.
    assert expression.df_expression_data is None
    assert isinstance(expression.matrix(), SparseExpressionMatrix)


def _dense_csv_twin(tmp_path, expression, samples, genes):
    """The same numbers as the sparse load, written as a dense CSV.

    Loading this exercises the pre-slice-7 storage path -- read dense, hold the
    whole frame resident -- so its peak is the fair comparison for the sparse
    path's peak on the identical workflow.
    """
    dense = expression.get_expression_df()
    path = tmp_path / "twin.csv"
    dense.to_csv(path)
    return ExpressionData(
        path=str(path), gene_list=genes, data_format="csv"
    )


def test_sparse_load_and_reduction_peaks_far_below_the_dense_path(tmp_path):
    """The material peak-memory win: the same workflow, sparse vs dense storage.

    A reduction materializes only the bounded top-N variable-feature block, so on
    the sparse path the whole load + PCA peaks at a small fraction of the dense
    path, which reads and holds the whole matrix resident. Both compute the same
    result (pinned elsewhere); this pins the memory difference.
    """
    n_obs, n_vars = 400, 6000  # dense frame ~9.2 MiB at float32
    sparse_expression, samples, genes = _sparse_expression(tmp_path, n_obs, n_vars)
    # Build the dense twin from the escape hatch, then reset to a fresh sparse load
    # so the measured region starts from the sparse-first state a real load leaves.
    dense_expression = _dense_csv_twin(tmp_path, sparse_expression, samples, genes)
    sparse_expression, samples, genes = _sparse_expression(tmp_path, n_obs, n_vars)

    def workflow(expression):
        manager = _manager(expression, samples)
        analysis = ExperimentAnalysis(
            manager, auto_run=False, register_distributions=False
        )
        analysis.calculate_pca(data_type="expression", n_variable_features=200)

    tracemalloc.start()
    tracemalloc.reset_peak()
    workflow(sparse_expression)
    _, sparse_peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    workflow(dense_expression)
    _, dense_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # The sparse path never built the whole dense frame...
    assert sparse_expression.df_expression_data is None
    # ...and the dense path (holding the whole matrix) peaks materially higher. A
    # generous 3x margin (measured ~19x) keeps the test robust to allocator noise.
    assert sparse_peak < dense_peak / 3, (
        f"sparse peak {sparse_peak / 1024**2:.1f} MiB not far below dense peak "
        f"{dense_peak / 1024**2:.1f} MiB"
    )


def test_snapshot_captures_sparse_expression_without_densifying(tmp_path):
    """AM9 slice 2b: the worker snapshot references the sparse matrix, not a dense frame.

    Before slice 2b, snapshot capture called ``get_expression_df()`` and referenced
    the full dense frame, so every worker inherited a densified matrix one seam
    after AM6 preserved sparsity. Now the storage-neutral reference reads the
    ``matrix()`` seam: capture and read serve the sparse adapter and its bounded
    operations without ever building the whole dense frame.
    """
    expression, samples, _genes = _sparse_expression(tmp_path, 60, 300)
    manager = _manager(expression, samples)

    snapshot = manager.capture_analysis_snapshot({RevisionKind.DATASET})
    matrix = snapshot.expression_matrix.read()

    assert isinstance(matrix, SparseExpressionMatrix)
    assert expression.df_expression_data is None  # capture/read did not densify
    # The worker reads a group's rows through the bounded seam, still no densify.
    block = matrix.select_samples(samples[:3])
    assert list(block.index) == samples[:3]
    assert expression.df_expression_data is None


def test_a_replaced_expression_source_is_caught_by_the_matrix_reference(tmp_path):
    """The storage-neutral reference is the backstop StableFrameReference gives reaction.

    The DATASET revision is the primary staleness signal, but replacing the source
    object (a fresh ``ExpressionData``, new identity) must fail freshness even when
    a revision was not advanced -- the same belt-and-suspenders the dense reaction
    reference provides.
    """
    expression, samples, _genes = _sparse_expression(tmp_path, 60, 300)
    manager = _manager(expression, samples)
    provenance = manager.capture_analysis_snapshot({RevisionKind.DATASET}).provenance

    manager.require_current(provenance)  # fresh state is current

    # Swap the source directly, without a DATASET transaction, so the revision
    # counter is unchanged and only the reference can detect the replacement.
    replacement, _samples, _genes = _sparse_expression(tmp_path, 60, 300)
    manager.expression_data = replacement

    with pytest.raises(StaleAnalysisResultError, match="input frame changed"):
        manager.require_current(provenance)


def test_a_captured_sparse_snapshot_reads_the_frozen_buffer(tmp_path):
    """Slice 2a + 2b together: the snapshot references the buffer the install froze.

    Freezing (slice 2a) prevents a same-shape in-place edit; referencing the sparse
    matrix (slice 2b) means a worker holding the snapshot reads exactly that frozen
    buffer, so it cannot observe a torn write.
    """
    expression, samples, genes = _sparse_expression(tmp_path, 40, 150)
    manager = _manager(expression, samples)
    snapshot = manager.capture_analysis_snapshot({RevisionKind.DATASET})

    matrix = snapshot.expression_matrix.read()
    assert isinstance(matrix, SparseExpressionMatrix)
    # A read through the snapshot still works...
    assert matrix.feature_column(genes[0]) is not None
    # ...while the buffer it references is the one the install froze.
    with pytest.raises(ValueError):
        expression.adata.X.data[0] = 0.0


def test_installing_expression_freezes_the_sparse_matrix_against_in_place_edits(tmp_path):
    """M-11 (AM9): the install writer seals the replace-only contract.

    The workspace is replace-only after AM8, so an installed matrix must not be
    edited in place while a worker snapshot references it. M-11 chose prevention:
    ``replace_expression_data`` freezes the canonical ``adata.X`` buffers, so a
    same-shape in-place edit raises at its source, while reads through the bounded
    seam are unaffected. This is the sparse store the AM9 snapshot reads.
    """
    expression, samples, genes = _sparse_expression(tmp_path, 40, 150)
    _manager(expression, samples)  # installs through replace_expression_data -> freeze

    matrix_x = expression.adata.X
    # A stored nonzero exists to edit; a same-shape in-place write is refused.
    assert matrix_x.data.size > 0
    with pytest.raises(ValueError):
        matrix_x.data[0] = 0.0
    with pytest.raises(ValueError):
        matrix_x.indices[0] = 0

    # Reads through the seam still work after freezing.
    seam = expression.matrix()
    assert isinstance(seam, SparseExpressionMatrix)
    assert seam.feature_column(genes[0]) is not None
    assert list(seam.select_samples(samples[:3]).index) == samples[:3]

    # Freezing is idempotent: re-installing an unchanged matrix re-freezes harmlessly.
    expression.freeze()
    with pytest.raises(ValueError):
        matrix_x.data[0] = 0.0

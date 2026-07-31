# Contributing to Metabis

This guide is for people changing Metabis and for anyone integrating against it
programmatically. The [README](README.md) covers installation and the demo
notebooks; everything here is the contract those tools are held to.

## Development setup

Follow the README installation steps, then install the development extra from
your clone:

```bash
pip install -e ".[dev]"
```

Install the example data before running anything that loads it:

```bash
python data/download_datasets.py
```

## Checks

Three gates must stay green.

```bash
python -m pytest
python -m ruff check .
python scripts/execute_demo_notebooks.py
```

`ruff check` is the enforced lint gate; `E501` and the `D` docstring rules are
deferred debt and are surfaced on demand with
`ruff check --extend-select D --select E501`.

The notebook gate executes each maintained notebook in a fresh kernel, fails on
the first cell error, and leaves the source notebooks unchanged. Pass
`--output-dir PATH` to retain executed copies for plot and table review.

## Supported public interface

Metabis promises stability only for the following namespaces. Depend on these;
treat everything else as implementation that may change without notice.

- **`metabis`** — the root namespace. Its `__all__` is the supported flat
  surface: domain inputs (`MetabolicModel`, `ReactionData`, `ExpressionData`,
  `SampleMetadata`, the preprocessing configs and functions), analysis entry
  points and typed results (`ExperimentAnalysis`, `GroupComparison`,
  `ReductionResult`, the activity results), canonical group and comparison
  identity (`SampleGroup`, `ComparisonMembership`, `ComparisonSideSpec`,
  `GroupComparisonSpec`, `ComparisonMode`), and the plotting functions. `from metabis import *` gives
  exactly `__all__`.
- **`metabis.analysis_workspace`** — the Analysis Workspace snapshot,
  provenance, revision, and stale-result contracts for capturing immutable
  inputs and installing results transactionally.
- **`metabis.analysis.h5ad_embeddings`** — discovery and artifact construction
  for detached PCA and UMAP inputs. The common discovery, alignment-error, and
  artifact-builder names are also available from `metabis`; the module exposes
  the complete closed import contract in its own `__all__`.
- **`metabis.viz.metabolic_visualization`** — the graph seam: `GraphArtifact`,
  its pipeline and configuration, the Bokeh rendering adapter, the audit/export
  frames, and the packaged-subsystem resource helpers.
- **`metabis.analysis.artifacts`** — common Analysis Artifact metadata,
  provenance, dimensions, immutable tables, and fingerprint helpers.
- **`metabis.analysis.comparison_artifact`** — `GroupComparisonArtifact`, its
  threshold and metric value types, and the canonical comparison export adapter.
- **`metabis.analysis.group_exports`** — the canonical tabular adapters for
  Sample Groups and Comparison Membership: stable group IDs, display labels,
  Group Definitions, exact sample order and group fingerprints
  (`sample_groups_frame`), and the resolved Target/Baseline samples with their
  mode, comparison-local labels, aligned universe, and one-vs-rest exclusions
  (`comparison_membership_frame`). The two exported column tuples are the
  frames' schema.
- **`metabis.analysis.reaction_set_enrichment`** — Reaction-Set Enrichment
  configuration, universe, annotations, artifact, calculation, result schema,
  and the Metabolic Model annotation adapter.
- **`metabis.analysis.set_enrichment`** — the shared paired-direction ORA schema,
  portable family artifact, deterministic Human1 reaction/GPR annotation
  adapters, and hypergeometric kernel used by reaction and gene workflows. It
  tests Target-higher and Target-lower queries together, corrects the combined
  hypothesis family, and supports family-specific workspace invalidation.
- The analysis-artifact and matrix seams: `ReductionArtifact` /
  `DimensionalityReduction`, `ActivityArtifact`, `SubsystemDeltaArtifact`,
  `ExperimentSummarySnapshot`, `CORRECTION_METHODS`, and the expression-matrix
  seam (`metabis.domain.expression_matrix`). Each such module names its
  supported surface in its own `__all__`.

Every module in these namespaces declares `__all__`; a name absent from an
`__all__` is internal even when the module is public. `metabis.managers`,
`metabis.resources`, `metabis.resource_registry`, and any other module not named
above are implementation.

Artifact families stay on their module interfaces rather than expanding the
root namespace. For example, a loaded Analysis Workspace can run and install
Reaction-Set Enrichment through public names only:

```python
from metabis.analysis.comparison_artifact import comparison_export_frame
from metabis.analysis.reaction_set_enrichment import (
    ReactionSetEnrichmentConfig,
    build_subsystem_annotations,
    run_reaction_set_enrichment,
)

annotations = build_subsystem_annotations(workspace.metabolic_model)
enrichment = run_reaction_set_enrichment(
    comparison_artifact,
    annotations,
    ReactionSetEnrichmentConfig(),
)
workspace.install_reaction_set_enrichment_artifact(provenance, enrichment)

comparison_table = comparison_export_frame(comparison_artifact, "reaction")
enrichment_table = enrichment.result_frame()
```

The paired family artifact is available for new integrations without traversing
model internals:

```python
from metabis.analysis.set_enrichment import (
    SetEnrichmentConfig,
    build_gene_set_annotations,
    run_gene_set_enrichment,
)

gene_sets = build_gene_set_annotations(workspace.metabolic_model)
gene_enrichment = run_gene_set_enrichment(
    comparison_artifact,
    gene_sets,
    SetEnrichmentConfig(),
)
workspace.install_set_enrichment_artifact(provenance, gene_enrichment)
gene_enrichment_table = gene_enrichment.result_frame()

from metabis import plot_set_enrichment_category

category_plot = plot_set_enrichment_category(gene_enrichment)
```

## Compatibility policy

Metabis follows [semantic versioning](https://semver.org/). A public name in an
`__all__` is removed or changed in signature only across a major version, and
only after at least one minor release that emits a `DeprecationWarning` naming
the removal version. New public surface may be added in a minor release. The
internal namespaces carry no such guarantee.

See [CHANGELOG.md](CHANGELOG.md) for the release-level summary.

Persisted reaction preprocessing may use `compass_neg_log`. This is a supported
input alias, not a second scale transform. Constructing
`ReactionPreprocessingConfig` normalizes it to the canonical form
`compass_activity=True, transform=ReactionTransform.NONE`. New code should use
that two-field form directly; another transform, including quantile
normalization, may then follow Compass activity decoding.

## Distribution check

Build both distribution formats and reject repository-local or review-process
material before testing an artifact:

```bash
python -m hatchling build
python scripts/check_package_contents.py dist/*.whl dist/*.tar.gz
```

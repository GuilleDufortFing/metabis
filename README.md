# Metabis

![Metabis graphical abstract](img/graphical_abstract.webp)

Metabis is an open-source Python library for metabolic data analysis, differential comparison, and metabolic graph visualization. It can be used as a standalone library in notebooks or Python scripts, and it is the analytical backbone behind the public Metabis platform at https://metabis.unav.edu/.

## Installation

Metabis requires Python 3.12. We recommend using conda and creating a dedicated Python 3.12 environment before installing Graphviz and the library.

### 1. Create and activate a conda environment

```bash
conda create -n metabis python=3.12
conda activate metabis
```

### 2. Install Graphviz

`pygraphviz` requires Graphviz to be available before the Python package install. Install it into the same environment with:

```bash
conda install -c conda-forge graphviz
```

### 3. Clone and install Metabis

```bash
git clone https://github.com/GuilleDufortFing/metabis.git
cd metabis
pip install --upgrade pip
pip install -e ".[dev]"
```

If you only need the library without notebook tooling, install from the cloned repository with:

```bash
pip install .
```

## Get Started

The recommended starting point is the staged demo notebook series under `demo/`:

- `demo/01_metabolic_model_and_data_loading.ipynb`
- `demo/02_groups_and_dimensionality_reduction.ipynb`
- `demo/03_group_statistical_comparison.ipynb`
- `demo/04_metabolic_graphs_and_subsystems.ipynb`
- `demo/05_multi_group_comparisons_smillie.ipynb`

Each notebook is self-contained, but the series is ordered from basic object loading to pathway-level interpretation. Together, notebooks `01` to `04` guide you through the main Metabis workflow step by step.

Notebook `05` then moves from the two-group Th17 example to the multi-group Smillie colon atlas, where the groups must be composed rather than simply named: Sample Groups over eight cell types with expression and reaction UMAPs, a Target built from several base groups, and base groups built from the intersection of two metadata columns.

When the notebooks are opened from this repository, they automatically add the local `src/` directory before importing `metabis`, so they can run directly from the checkout as long as the Python dependencies are installed.

If you prefer a single end-to-end case study, use
[demo/case_study.ipynb](demo/case_study.ipynb). It applies the same ordered,
artifact-first workflow to the Th17 example used in the Metabis server tutorial.

The example files the demo notebooks load are installed rather than tracked: they are large, and `data/*.csv`, `data/*.tsv`, and `data/*.h5ad` are gitignored. Install the complete set with:

```bash
python data/download_datasets.py
```

That downloads the author-maintained archive the Metabis server deploys, verifies its recorded SHA-256 digest, and writes the Th17 Compass files (`data/Th17_Compass_reactions.tsv`, `data/Th17_Compass_expression.tsv`, and the `data/Th17_Compass_metadata.tsv` sample sheet) together with the Smillie files notebook `05` loads (`data/smillie_clusters_pool_reactions.tsv`, `data/smillie_clusters_pool_expression_cpm.tsv`, and `data/smillie_metadata.tsv`). All are tab-delimited, as their `.tsv` suffix says.

The Th17 files can alternatively be rebuilt from the public Compass dataset:

```bash
python data/download_and_process_compass.py
```

The Smillie subset has no such generator, so the archive is the only way to obtain it.

Execute all six maintained notebooks without modifying their source files:

```bash
python scripts/execute_demo_notebooks.py
```

The command fails on the first cell error. Pass `--output-dir PATH` to retain
executed copies for plot and table review.

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

### Compatibility policy

Metabis follows [semantic versioning](https://semver.org/). A public name in an
`__all__` is removed or changed in signature only across a major version, and
only after at least one minor release that emits a `DeprecationWarning` naming
the removal version. New public surface may be added in a minor release. The
internal namespaces carry no such guarantee.

The first stable public-interface release is `1.0.0`. See
[CHANGELOG.md](CHANGELOG.md) for the release-level summary.

Persisted reaction preprocessing may use `compass_neg_log`. This is a supported
input alias, not a second scale transform. Constructing
`ReactionPreprocessingConfig` normalizes it to the canonical form
`compass_activity=True, transform=ReactionTransform.NONE`. New code should use
that two-field form directly; another transform, including quantile
normalization, may then follow Compass activity decoding.

### Distribution check

Build both distribution formats and reject repository-local or review-process
material before testing an artifact:

```bash
python -m hatchling build
python scripts/check_package_contents.py dist/*.whl dist/*.tar.gz
```

## Included Resources

Metabis ships with the packaged reference-model and graph resources needed for normal library use, so no separate resource checkout is required.
Third-party attribution and upstream license notices for bundled public resources and example data are collected in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Metabis is released under the [MIT License](LICENSE).

## Contact

For support or questions, please contact [gdufort@fing.edu.uy](mailto:gdufort@fing.edu.uy).

## Acknowledgements

- The Human1 reference model used by Metabis was obtained from [Human-GEM](https://github.com/SysBioChalmers/Human-GEM) and is distributed there under the [CC BY 4.0 license](https://github.com/SysBioChalmers/Human-GEM?tab=CC-BY-4.0-1-ov-file#readme).
- The Human1 maps used by Metabis were obtained from [Human-maps](https://github.com/SysBioChalmers/Human-maps) and are distributed there under the [CC BY 4.0 license](https://github.com/SysBioChalmers/Human-maps?tab=CC-BY-4.0-1-ov-file#readme).
- The Compass reaction-flux example data and study context are based on [Compass](https://github.com/wagnerlab-berkeley/Compass), which is distributed under the [BSD-3-Clause license](https://github.com/wagnerlab-berkeley/Compass?tab=BSD-3-Clause-1-ov-file#readme). The associated publication is [Wagner et al., Cell 2021](https://doi.org/10.1016/j.cell.2021.05.045).
- The Smillie example expression subset and sample metadata are derived from the processed data deposited as the public Single Cell Portal study [SCP259](https://singlecell.broadinstitute.org/single_cell/study/SCP259/intra-and-inter-cellular-rewiring-); its reaction matrix is Compass output computed from that subset. The [Single Cell Portal terms](https://singlecell.zendesk.com/hc/en-us/articles/360060798672-Terms-of-service) permit public-study data to be redistributed and reused, and limit it to research use rather than medical, clinical, or diagnostic decisions. The associated publication is [Smillie et al., Cell 2019](https://doi.org/10.1016/j.cell.2019.06.029).
- See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the bundled notice text carried with the public package.

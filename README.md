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

The archive is the single source for the example data, so the notebooks and the deployed Metabis server read byte-identical inputs.

## Public interface and versioning

Metabis follows [semantic versioning](https://semver.org/), and the first
stable public-interface release is `1.0.0`. Stability is promised for the
supported namespaces only; everything else is implementation that may change
without notice.

[CONTRIBUTING.md](CONTRIBUTING.md) lists those namespaces and states the
compatibility policy, and [CHANGELOG.md](CHANGELOG.md) carries the
release-level summary.

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

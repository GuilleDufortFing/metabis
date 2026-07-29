# Demo Notebooks

This folder contains the staged Metabis demo series plus a direct case study built around the generated Th17 Compass example dataset.

All notebooks are self-contained. You can run them independently, but notebooks `01` to `04` are designed to be read in order.

## Order

1. `01_metabolic_model_and_data_loading.ipynb`
   Introduces the Metabolic Dataset inputs (`MetabolicModel`, `ReactionData`,
   `ExpressionData`, and `SampleMetadata`) and the minimal `AnalysisWorkspace`.

2. `02_groups_and_dimensionality_reduction.ipynb`
   Defines metadata-driven Sample Groups, creates an `ExperimentAnalysis`, and
   installs and plots PCA, t-SNE, and UMAP Analysis Artifacts.

3. `03_group_statistical_comparison.ipynb`
   Resolves an explicit Target/Baseline Comparison Membership, installs the
   comparison Analysis Artifact, and walks through reaction and gene results,
   significance thresholds, histograms, and scatter plots.

4. `04_metabolic_graphs_and_subsystems.ipynb`
   Uses subsystem ranking, renderer-independent Graph Artifacts, and the Bokeh
   rendering adapter to move from global results to pathway-level interpretation.

## Case Study

`case_study.ipynb`
Applies the same ordered workflow directly to the Th17 example used in the
Metabis server tutorial. Unlike notebooks `01` to `04`, it assumes that
background and does not re-introduce the main library functionality step by
step.

## Prerequisites

- Install the repository and notebook dependencies with `pip install -e ".[dev]"`.
- Ensure Graphviz is installed before installing `pygraphviz`.
- Use a Jupyter-compatible environment such as VS Code notebooks.

The notebooks automatically add the local `src/` directory when they are run from this repository checkout, so an editable install is optional.

## Example Data

All notebooks in this folder load generated files from `metabis/data`:

- `Th17_Compass_reactions.tsv`
- `Th17_Compass_expression.tsv`
- `Th17_Compass_metadata.tsv`

To regenerate these files from the public Compass dataset, run:

```bash
python data/download_and_process_compass.py
```

The generator downloads the public Compass inputs and writes all three
tab-delimited files. The notebooks load them only from `metabis/data`.

## Execution gate

Run the complete maintained set in its documented order with:

```bash
python scripts/execute_demo_notebooks.py
```

The gate executes each notebook from the repository root in a fresh Jupyter
kernel, fails on the first cell error, and leaves the source notebooks
unchanged. Use `--output-dir PATH` to retain executed copies for visual review,
or pass one or more notebook filenames to run a focused subset.

If you want the single-notebook case study instead of the staged learning path,
use [`case_study.ipynb`](case_study.ipynb) in this folder.

# Demo Notebooks

This folder contains the staged Metabis demo series plus a direct publication-style case study built around the bundled Th17 Compass example dataset.

All notebooks are self-contained. You can run them independently, but notebooks `01` to `04` are designed to be read in order.

## Order

1. `01_metabolic_model_and_data_loading.ipynb`
   Introduces `MetabolicModel`, `ReactionData`, `ExpressionData`, and the minimal `DataManager` state.

2. `02_groups_and_dimensionality_reduction.ipynb`
   Defines groups with `DataManager.load_groups()`, creates an `ExperimentAnalysis`, and explores PCA, t-SNE, and UMAP views.

3. `03_group_statistical_comparison.ipynb`
   Introduces `GroupComparison`, explicitly sets the active analysis group, and then walks through reaction and gene test results, significance thresholds, histograms, and scatter plots.

4. `04_metabolic_graphs_and_subsystems.ipynb`
   Uses subsystem ranking and metabolic graph visualization to move from global comparison results to pathway-level interpretation.

## Case Study

`case_study.ipynb`
Applies the same ordered workflow directly to the Th17 example used in the publication and in the Metabis server tutorial. Unlike notebooks `01` to `04`, it assumes that background and does not re-introduce the main library functionality step by step.

## Prerequisites

- Install the repository dependencies.
- Ensure Graphviz is installed before installing `pygraphviz`.
- Use a Jupyter-compatible environment such as VS Code notebooks.

The notebooks automatically add the local `src/` directory when they are run from this repository checkout, so an editable install is optional.

## Example Data

All notebooks in this folder load the bundled files from `metabis/data`:

- `Th17_Compass_reactions.csv`
- `Th17_Compass_expression.csv`

To regenerate these files from the public Compass dataset, run:

```bash
python data/download_and_process_compass.py
```

The demo notebooks load these files only from `metabis/data`.

If you want the single-notebook case study instead of the staged learning path, use [`case_study.ipynb`](case_study.ipynb) in this folder.
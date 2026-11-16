# Demo Notebooks

Three notebooks, one for each tutorial of the Metabis server. Each runs its
tutorial's analysis with the library calls the server makes, so the numbers it
prints are the app's, and asserts them as it goes.

## Notebooks

1. `01_compass_th17.ipynb`
   Compass scores of pathogenic (Th17p) and non-pathogenic (Th17n) Th17 cells,
   from [Wagner et al., Cell 2021](https://doi.org/10.1016/j.cell.2021.05.045).
   It introduces the library as it goes: the model and the source model's
   reaction table, reaction and expression data, the sample sheet and the
   `AnalysisWorkspace`, groups and reductions, a Target/Baseline comparison,
   the Delta score, over-representation analysis, the network and pathway
   graphs, Metabolic Atlas maps, element details, single-group activity and
   adjustment for depth.

2. `02_smillie_pseudobulk.ipynb`
   Compass scores of seven colon cell types in healthy, non-inflamed and
   inflamed tissue, from [Smillie et al., Cell 2019](https://doi.org/10.1016/j.cell.2019.06.029):
   groups from two metadata columns, comparison sides joined from several
   groups, each group against the rest, the analysis overview and the maps.

3. `03_metaflux_tcga_lung.ipynb`
   METAFlux fluxes of 1,017 TCGA lung tumours, from
   [Huang et al., Nat Commun 2023](https://doi.org/10.1038/s41467-023-40457-w):
   a signed flux matrix read in single direction, values at a solver's
   tolerance read as zero, expression already on a log2 scale, log2 fold
   change, and a reaction map prepared as the paper prepared its own.

Read `01` first: `02` and `03` use what it introduces without introducing it
again.

## Prerequisites

- Install the repository and notebook dependencies with `pip install -e ".[dev]"`.
- Ensure Graphviz is installed before installing `pygraphviz`.
- Use a Jupyter-compatible environment such as VS Code notebooks.

The notebooks automatically add the local `src/` directory when they are run from this repository checkout, so an editable install is optional.

## Example Data

All notebooks in this folder load their example files from `metabis/data`:

- `Th17_Compass_reactions.tsv`, `Th17_Compass_expression.tsv` and
  `Th17_Compass_metadata.tsv` (notebook `01`)
- `smillie_pseudobulk_reactions.tsv`, `smillie_pseudobulk_expression_cp10k.tsv`
  and `smillie_pseudobulk_metadata.tsv` (notebook `02`)
- `metaflux_tcga_lung_reactions.tsv`, `metaflux_tcga_lung_expression.tsv` and
  `metaflux_tcga_lung_metadata.tsv` (notebook `03`)

Install every example file with:

```bash
python data/download_datasets.py
```

That script downloads the author-maintained archive the Metabis server deploys,
verifies its recorded SHA-256 digest, and extracts the files into `metabis/data`,
so the notebooks and the application read byte-identical inputs.

The archive is the single source for the example data, and the notebooks load
these files only from `metabis/data`. `THIRD_PARTY_NOTICES.md` gives each
example's source and terms.

## Execution gate

Run the complete maintained set in its documented order with:

```bash
python scripts/execute_demo_notebooks.py
```

The gate executes each notebook from the repository root in a fresh Jupyter
kernel, fails on the first cell error or failed assertion, and leaves the
source notebooks unchanged. Use `--output-dir PATH` to retain executed copies
for visual review, or pass one or more notebook filenames to run a focused
subset.

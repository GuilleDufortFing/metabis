# Third-Party Notices

Metabis source code is released under the MIT license in `LICENSE`.
Some bundled public resources and example data in this repository are derived
from upstream projects that remain under their own licenses and attribution
terms.

This notice applies to the public `metabis/` package tree. It does not change
the upstream licenses of third-party materials.

Metabis keeps its own project license in the repository-root `LICENSE` file.
Verbatim upstream license texts carried with the public package are collected in
`THIRD_PARTY_LICENSES/`.

## Human1-derived packaged resources

Within `src/metabis/resources`, Metabis redistributes Human1-derived material
adapted from the following upstream repositories:

- `src/metabis/resources/reference_model/`
- `src/metabis/resources/graphs/subsystem/human1/`
- `src/metabis/resources/graphs/compartment/`

### Human-GEM

- Upstream project: Human-GEM
- Source: https://github.com/SysBioChalmers/Human-GEM
- Upstream license: Creative Commons Attribution 4.0 International
  (`THIRD_PARTY_LICENSES/CC-BY-4.0.txt`)
- Metabis use: packaged Human1 reference-model tables and related exported
  runtime resources adapted from Human-GEM materials.
- Change notice: filenames, packaging layout, table exports, identifier
  crosswalks, and related preprocessing may differ from the upstream release.

### Human-maps

- Upstream project: Human-maps
- Source: https://github.com/SysBioChalmers/Human-maps
- Upstream license: Creative Commons Attribution-ShareAlike 4.0 International,
  as the map SVG files declare in their metadata
  (`THIRD_PARTY_LICENSES/CC-BY-SA-4.0.txt`). The repository's own LICENSE file
  is Creative Commons Attribution 4.0 International.
- Metabis use: packaged Human1 graph resources derived from Human-maps: the
  compartment and subsystem layouts (`resources/graphs/*/human1/*.json`), and
  Metabolic Atlas's link paths and subsystem bands beside them
  (`*.atlas.json`, written by `scripts/import_atlas_paths.py`). These derived
  files are shared under CC BY-SA 4.0.
- Change notice: upstream map assets were converted into Metabis runtime graph
  JSON resources and may include filtering, coordinate extraction,
  normalization, or curation; the link paths and bands are shifted into each
  layout's frame, paths to metabolites a layout does not draw are left out, and
  the glyphs a layout classes as reactions where Metabolic Atlas draws
  metabolites are listed, so the maps draw them as metabolites.

Use of these Human1-derived materials does not imply endorsement by the
Human-GEM or Human-maps authors or maintainers.

## Compass-based example data

The public example files below are based on the Th17 example distributed in the
Compass repository:

- `data/Th17_Compass_reactions.tsv`
- `data/Th17_Compass_expression.tsv`

- Upstream project: Compass
- Source: https://github.com/wagnerlab-berkeley/Compass
- Upstream license notice carried here: BSD 3-Clause
  (`THIRD_PARTY_LICENSES/BSD-3-Clause-Compass.txt`)
- Metabis use: example reaction-flux and expression files generated from the
  public Compass Th17 dataset for demos and documentation.
- Change notice: Metabis reformats the downloaded files, annotates samples with
  cell-type information, and writes project-specific output filenames.
- Associated publication: Wagner et al., Cell 2021,
  https://doi.org/10.1016/j.cell.2021.05.045

Use of these example materials does not imply endorsement by the Compass
authors, contributors, or copyright holders.

## Smillie-based example data

The public example files below are derived from the processed colon single-cell
data deposited as Single Cell Portal study SCP259:

- `data/smillie_clusters_pool_expression_cpm.tsv`
- `data/smillie_metadata.tsv`
- `data/smillie_clusters_pool_reactions.tsv`

- Upstream study: Smillie et al., intra- and inter-cellular rewiring of the
  human colon during ulcerative colitis
- Source: https://singlecell.broadinstitute.org/single_cell/study/SCP259/intra-and-inter-cellular-rewiring-
- Upstream terms: Broad Single Cell Portal terms of service,
  https://singlecell.zendesk.com/hc/en-us/articles/360060798672-Terms-of-service,
  which state that data submitted under a public study are available for
  unrestricted public view, redistribution, and reuse, and which limit portal
  content to research use rather than medical, clinical, or diagnostic use.
- Metabis use: an example expression subset and sample sheet for the multi-group
  demo notebook, plus a reaction matrix computed from that subset with Compass.
- Change notice: Metabis pools selected clusters, subsets cells, normalizes the
  expression matrix to CPM, and writes project-specific output filenames. The
  controlled-access raw sequencing data are not part of this material.
- Associated publication: Smillie et al., Cell 2019,
  https://doi.org/10.1016/j.cell.2019.06.029

Use of these example materials does not imply endorsement by the study authors,
contributors, or the Broad Institute.

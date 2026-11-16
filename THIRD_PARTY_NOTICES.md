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
- Source: https://github.com/SysBioChalmers/Human-GEM, at release v2.1.1
  (of 2026-10-07)
- Upstream license: Creative Commons Attribution 4.0 International
  (`THIRD_PARTY_LICENSES/CC-BY-4.0.txt`)
- Metabis use: packaged Human1 reference-model tables and related exported
  runtime resources adapted from Human-GEM materials. The reference-model
  tables are written from the release's `model` folder by
  `scripts/import_human_gem.py`.
- Change notice: filenames, packaging layout, table exports, identifier
  crosswalks, and related preprocessing may differ from the upstream release.

### Human-maps

- Upstream project: Human-maps
- Source: https://github.com/SysBioChalmers/Human-maps, at commit
  8ab97f21dd23cf3afe71f2acf385b6a0c08e4634 (its maps fitted to Human-GEM
  2.1.1, of 2026-10-28)
- Copyright notice, as the map SVG files carry it: Copyright Department of
  Biology and Biological Engineering, Chalmers University of Technology.
- Upstream license: Creative Commons Attribution-ShareAlike 4.0 International,
  as the map SVG files declare in their metadata
  (`THIRD_PARTY_LICENSES/CC-BY-SA-4.0.txt`). The repository's own LICENSE file
  is Creative Commons Attribution 4.0 International.
- Metabis use: packaged Human1 graph resources derived from Human-maps: the
  compartment and subsystem layouts (`resources/graphs/*/human1/*.json`), the
  transport maps of each membrane among the compartment layouts, and
  Metabolic Atlas's link paths and subsystem bands beside them, with a
  compartment map's metabolite labels (`*.atlas.json`), both written by
  `scripts/import_human_maps.py` from the maps' SBGN and SVG files. These
  derived files are shared under CC BY-SA 4.0.
- Change notice: upstream maps were converted into Metabis runtime graph JSON
  resources: reactions, metabolites, genes and their links are read from the
  SBGN files, and identifiers, shapes, compartment outlines, link paths, bands
  and labels from the SVG files at the same places; a reaction's touching gene
  boxes are grouped as one enzyme, and a compartment drawn as a line or a name
  alone becomes a box. Each metabolite's label lines, their offset from it, and
  whether it is drawn as a dot are kept under the layout glyph drawn in its
  place. A transport map's one subsystem band, under the whole map, is left
  out.

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

- `data/smillie_pseudobulk_expression_cp10k.tsv`
- `data/smillie_pseudobulk_metadata.tsv`
- `data/smillie_pseudobulk_reactions.tsv`

- Upstream study: Smillie et al., intra- and inter-cellular rewiring of the
  human colon during ulcerative colitis
- Source: https://singlecell.broadinstitute.org/single_cell/study/SCP259/intra-and-inter-cellular-rewiring-
- Upstream terms: Broad Single Cell Portal terms of service,
  https://singlecell.zendesk.com/hc/en-us/articles/360060798672-Terms-of-service,
  which state that data submitted under a public study are available for
  unrestricted public view, redistribution, and reuse, and which limit portal
  content to research use rather than medical, clinical, or diagnostic use.
- Metabis use: the Smillie example of the Metabis server and demo notebook
  `02`: pseudobulk expression profiles, their sample sheet, and the Compass
  reaction scores computed from them.
- Change notice: built as the Compass pseudobulk notebook builds them
  (https://compass-wagnerlab.readthedocs.io/en/latest/notebooks/pseudobulk.html):
  per biopsy, health state and cell type, the counts are summed and scaled to
  10,000 per profile, and Compass scores the profiles on RECON2. The
  controlled-access raw sequencing data are not part of this material.
- Associated publication: Smillie et al., Cell 2019,
  https://doi.org/10.1016/j.cell.2019.06.029

Use of these example materials does not imply endorsement by the study authors,
contributors, or the Broad Institute.

## METAFlux-based example data

The public example files below hold fluxes computed with METAFlux from the
open-access RNA-seq of The Cancer Genome Atlas lung tumours:

- `data/metaflux_tcga_lung_reactions.tsv`
- `data/metaflux_tcga_lung_expression.tsv`
- `data/metaflux_tcga_lung_metadata.tsv`

- Upstream projects: METAFlux (https://github.com/KChen-lab/METAFlux, MIT
  license, commit c904a9c) and TCGA-LUAD and TCGA-LUSC
  (https://portal.gdc.cancer.gov/projects/TCGA-LUAD,
  https://portal.gdc.cancer.gov/projects/TCGA-LUSC), read through UCSC Xena
  (https://xenabrowser.net/)
- Upstream terms: TCGA open-access data may be redistributed under the GDC data
  access policies,
  https://gdc.cancer.gov/access-data/data-access-processes-and-tools; research
  use only, not for medical, clinical, or diagnostic decisions.
- Metabis use: the METAFlux example of the Metabis server and demo notebook
  `03`: one primary tumour per patient, 516 adenocarcinomas and 501 squamous
  cell carcinomas, with their log2(TPM + 1) expression and a sample sheet.
- Change notice: the fluxes are METAFlux's bulk output with the human blood
  medium, written with project-specific filenames.
- Associated publication: Huang et al., Nat Commun 2023,
  https://doi.org/10.1038/s41467-023-40457-w

Use of these example materials does not imply endorsement by the METAFlux
authors or the TCGA Research Network.

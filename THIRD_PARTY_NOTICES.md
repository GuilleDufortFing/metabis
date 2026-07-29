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
- Upstream license: Creative Commons Attribution 4.0 International
  (`THIRD_PARTY_LICENSES/CC-BY-4.0.txt`)
- Metabis use: packaged Human1 graph resources derived from Human-maps.
- Change notice: upstream map assets were converted into Metabis runtime graph
  JSON resources and may include filtering, coordinate extraction,
  normalization, or curation.

Use of these Human1-derived materials does not imply endorsement by the
Human-GEM or Human-maps authors or maintainers.

## Compass-based example data

The public example files below are based on the Th17 example distributed in the
Compass repository:

- `data/Th17_Compass_reactions.tsv`
- `data/Th17_Compass_expression.tsv`
- `data/download_and_process_compass.py`

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

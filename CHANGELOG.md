# Changelog

All notable changes to Metabis are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Demo notebook `05_multi_group_comparisons_smillie.ipynb`: multi-group Sample
  Groups on the Smillie colon atlas, expression and reaction UMAPs, a Target
  composed of several base groups, and base groups built from the intersection
  of two metadata columns. The notebook execution gate now covers six notebooks.
- `data/download_datasets.py` installs every bundled example matrix from the
  author-maintained archive, verifying its recorded SHA-256 digest.
- Attribution and upstream terms for the Smillie example data in the README and
  in `THIRD_PARTY_NOTICES.md`.
- `add_click_behavior` accepts keyword-only `navigation_event` and
  `download_flag` for figures embedded without a Bokeh server
  (`bokeh.embed.json_item`). A subsystem tap dispatches the named `CustomEvent`
  on `window` with `detail = {element_id, label}`, and hover-panel downloads
  follow the named `window` flag, instead of driving a `Select` and a `Toggle`
  that such a figure would only carry as detached copies. Each replaces its
  widget argument; passing both raises `ValueError`. Existing calls are
  unchanged.
- `plot_statistical_scatterplots`, `plot_subsystem_comparison` and
  `plot_subsystem_activity_ranking` accept keyword-only `download_flag`, passed
  to `add_click_behavior`, so these figures embedded without a Bokeh server can
  download hover panels on tap.
- `statistical_results_columns` and `statistical_results_rows` return the
  columns and rows of the on-screen statistical results table, smallest adjusted
  p-value first, for callers that draw their own table instead of the Bokeh
  `DataTable` from `create_statistical_results_table`.
- Source-model reaction tables, `model_alignments/reactions_<MODEL>.tsv` for
  RECON1, RECON2, RECON2.2, RECON3D and HMR2.0: every reaction of the model,
  under the ID its data files carry, with the Human1 reaction it maps to and
  whether the two are written alike or reversed, read from both models'
  chemistry by `scripts/regenerate_model_alignments.py`.
  `MetabolicModel.get_source_reactions(name)` reads one, and
  `ReactionData(..., source_reactions=...)` maps the IDs it lists first and
  names, in `reversed_reactions`, the Human1 reactions whose backing source it
  writes the other way; `MetabolicModel.reverse_reactions(ids)` reverses them.
  `ReactionData.reaction_mapping` gives each kept source's Human1 reaction, and
  `MetabolicModel.get_alignment_values(name)` a model's per-reaction file.

### Changed

- The supported-interface definition, compatibility policy, and maintainer
  checks moved from the README into `CONTRIBUTING.md`. The README now covers
  installation, the demo notebooks, and the example data, and points at the
  contract from one short section.

### Removed

- `data/download_and_process_compass.py` and its `data/add_celltype_to_samples.py`
  helper. `data/download_datasets.py` is now the single documented route to the
  example data, so the notebooks and the deployed server cannot diverge on how
  the Th17 files were built.

### Fixed

- A source reaction now maps to Human1 through any one of the identifiers Human1
  lists for a reaction (`HMR_8572;RE0944C`) and through Recon3D's bracket
  spelling of a Recon1 or Recon2 compartment (`EX_glc(e)` as `EX_glc[e]`), ahead
  of the legacy BiGG aliases. The Compass Th17 example leaves 247 of its 6,941
  source reactions unmapped instead of 1,010.
- The reaction-orientation files under `model_interoperability/model_alignments`
  disagreed with the models they name: against Compass's Recon2, the RECON2 file
  called 723 of 881 reversed reactions aligned and 75 of its 233 reversals were
  not. The RECON1, RECON2, RECON2.2, RECON3D and HMR2.0 files are now read from
  each published source model's stoichiometry, paired as the loader pairs them,
  by `scripts/regenerate_model_alignments.py`, which pins every source by
  SHA-256 and can be rerun with `--check`. HMR1.0, with no public source, is
  unchanged and holds no reversal.
- A Human1 reaction is oriented by the source reaction that backs it in the
  loaded file, from the declared model's table, rather than by one value per
  Human1 reaction: two source reactions written opposite ways, as Recon2's
  r0634 and FAOXC101C8m are for MAR03146, are each read right. Four reactions
  of the bundled Smillie example were drawn reversed.
- With a source model declared, a reaction pairs with the Human1 reaction
  holding its exact chemistry when its identifier names another: Recon2's
  `EX_glc(e)` reaches glucose exchange, not the GD1beta exchange Human1's BiGG
  annotation files it under, and Recon1's `DOLP_Lter` a dolichyl-phosphate
  transport, as Recon1 means it, not the glucosyl transfer Recon2 reuses the ID
  for. An identifier naming a chemically unrelated reaction (Recon2's `CITL`,
  citrate lyase, filed under chitobiose hydrolysis) is left unmapped, and a
  reaction no identifier names maps by its chemistry. The Th17 example leaves
  208 reactions unmapped instead of 247.
- An SBML reaction ID is also read unescaped and without its prefix, so Compass
  output on Recon 2.2 (`R_EX_glc_LPAREN_e_RPAREN_`) maps.
- `DistributionBinner.bin_for_value` bins a value as an element of the
  distribution with that value is binned. An older definition later in the class
  shadowed it and ignored the spread method, so a transport family's Delta
  score could be coloured a bin away from a subsystem with the same score.

## [1.0.0] - 2026-07-27

This is the first stable release of the supported public interface.

### Added

- Analysis Workspace snapshots, revisions, source fingerprints, immutable
  Analysis Artifacts, provenance manifests, freshness checks, and transactional
  artifact installation.
- Metadata-first Sample Groups, explicit Target/Baseline Comparison Membership,
  canonical exports, and one-vs-rest comparison support.
- Dense and sparse expression paths with bounded materialization and load
  preflight checks.
- Detached import of PCA and UMAP coordinates from `h5ad`, including optional
  PCA loadings and explained-variance metadata.
- Reaction- and gene-set enrichment artifacts, subsystem activity and Delta
  analysis, reproducible graph artifacts, and packaged subsystem layouts.
- Five executable demo notebooks and a deterministic notebook execution gate.

### Changed

- Reaction preprocessing now represents Compass activity-score decoding and the
  optional scale transform as independent controls.
- The serialized `compass_neg_log` value is retained as an input alias and
  normalizes to the canonical two-field preprocessing configuration.
- Public interfaces are explicitly declared by `__all__`, with specialized
  artifact families exposed through documented module seams.
- Graph calculation is independent of Bokeh rendering and consumes immutable
  worker inputs.

### Fixed

- Direction-aware reaction statistics, blocked-reaction reporting, sparse input
  handling, imported-embedding identity, graph layout quality, and plot/export
  consistency across the supported workflows.

[1.0.0]: https://github.com/GuilleDufortFing/metabis/releases/tag/v1.0.0

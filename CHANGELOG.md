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

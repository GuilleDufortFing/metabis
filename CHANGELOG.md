# Changelog

All notable changes to Metabis are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases follow [Semantic Versioning](https://semver.org/).

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

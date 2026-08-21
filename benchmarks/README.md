# Revision benchmark protocol

The benchmark runner produces `revision-benchmark.json` and a Markdown summary. It uses deterministic synthetic matrices, requires no network access, and keeps benchmark execution outside normal `pytest` runs.

Run from the public-library repository:

```bash
python benchmarks/benchmark_revision.py --tier tiny --output-dir benchmark-results
```

The runner currently measures reaction-input generation, cubic-root normalization, prefix group creation, and a reaction-only Welch comparison. Later work packages add expression alignment, experiment summaries, PCA/UMAP/t-SNE, threshold updates, reaction-set enrichment, graph rendering, and plot exports using the same schema.

## Dataset tiers

| Tier | Source | Purpose |
| --- | --- | --- |
| tiny | deterministic synthetic fixture | smoke and regression checks |
| medium | deterministic synthetic matrix | expose scaling behavior without external data |
| large | deterministic synthetic 5,000 × 8,000 matrix | exercise configured memory/time limits reproducibly |

The runner does not currently consume a real publication or multi-cell-type dataset. Real-data reviewer evidence is a separately provisioned release profile tracked as A-05; archive it externally with a dataset ID/version and redistribution terms. Do not commit large inputs or ad-hoc generated result directories.

## Required workloads

The frozen revision artifact must cover reaction load/mapping, expression load/alignment, groups and summaries, PCA/UMAP/t-SNE, reaction-only and reaction-plus-expression comparisons, threshold updates, reaction-set enrichment, overview rendering, small/medium/large subsystem rendering at every hover level, and PNG/SVG/PDF export.

For each scenario record wall and CPU time, peak RSS, payload bytes, dimensions/counts, sparsity/zero fraction, cold/warm cache state, status, and failure phase. Browser runs additionally record first status paint, time to interaction, and second-session responsiveness.

## Environment evidence

Keep the library and server commits, Python and dependency versions, CPU and RAM, OS, Graphviz, browser/driver versions, and server upload/WebSocket/resource flags alongside every frozen artifact.

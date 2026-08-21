# Test suite

Run the deterministic, network-free library suite from this directory's parent:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache PYTHONPATH=src \
  conda run -n metabis python -m pytest
```

Benchmark scenarios are separate from the default suite:

```bash
python benchmarks/benchmark_revision.py --tier tiny --output-dir benchmark-results
```

## Revision work-package map

| Package | Test location |
| --- | --- |
| WP1 normalization/statistics/cache correctness | `tests/domain/`, `tests/analysis/` |
| WP2 metadata and custom groups | `tests/domain/test_sample_metadata.py`, `tests/managers/test_metadata_groups.py` |
| WP3 precomputed embeddings and exports | `tests/domain/test_embedding_data.py`, `tests/viz/test_embedding_exports.py` |
| WP4 rendering, jobs, and performance | `tests/viz/test_graph_resources.py`, server integration/browser suites |
| WP5 single-group activity | `tests/analysis/test_experiment_analysis.py` |
| WP6 reaction-set enrichment | `tests/analysis/test_reaction_set_enrichment.py` |
| WP7 downloads and interpretation | `tests/viz/test_exports.py` |
| WP8 formats and scope | `tests/domain/test_format_detection.py` |

Paths without a test module yet are the stable destinations for their later work packages.

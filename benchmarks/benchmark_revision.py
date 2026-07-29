#!/usr/bin/env python3
"""Deterministic benchmark runner for the Metabis revision evidence pipeline."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from metabis.analysis.statistical_analysis import comparison_test
from metabis.domain.reaction_data import apply_cbrt_normalization
from metabis.managers.group_manager import GroupManager

TIERS = {
    "tiny": {"samples": 16, "reactions": 24, "zero_fraction": 0.25},
    "medium": {"samples": 500, "reactions": 800, "zero_fraction": 0.70},
    "large": {"samples": 5_000, "reactions": 8_000, "zero_fraction": 0.90},
}


def _git_commit(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def capture_environment(repo: Path) -> dict[str, Any]:
    """Capture reproducibility metadata without contacting external services."""
    dependency_names = (
        "anndata",
        "bokeh",
        "numpy",
        "pandas",
        "scanpy",
        "scipy",
        "statsmodels",
        "umap-learn",
    )
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "commit": _git_commit(repo),
        "cpu_model": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "total_ram_gb": _total_ram_gb(),
        "graphviz": _command_version(["dot", "-V"]),
        "browser": os.environ.get("METABIS_BROWSER_VERSION", "not-run"),
        "driver": os.environ.get("METABIS_DRIVER_VERSION", "not-run"),
        "server_flags": os.environ.get("METABIS_SERVER_FLAGS", "not-supplied"),
        "dependencies": {
            name: _package_version(name) for name in dependency_names
        },
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _command_version(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return (result.stdout or result.stderr).strip()
    except (OSError, subprocess.CalledProcessError):
        return "not-installed"


def _total_ram_gb() -> float | str:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return round(page_size * page_count / (1024**3), 2)
    except (AttributeError, OSError, ValueError):
        return "unknown"


def peak_rss_mb() -> float:
    """Return process peak resident memory in MiB on Linux or macOS."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return value / divisor


def make_synthetic_flux_matrix(
    n_samples: int,
    n_reactions: int,
    zero_fraction: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Create deterministic samples x reactions input without a large fixture file."""
    import numpy as np

    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n_samples, n_reactions))
    values[rng.random(values.shape) < zero_fraction] = 0.0
    return pd.DataFrame(
        values,
        index=[f"group{index % 2}_sample_{index:05d}" for index in range(n_samples)],
        columns=[f"MAR{index:05d}" for index in range(n_reactions)],
    )


def measure_scenario(
    scenario: str,
    dataset: str,
    dimensions: dict[str, Any],
    workload: Callable[[], Any],
) -> tuple[dict[str, Any], Any]:
    """Measure a workload and always return a machine-readable status record."""
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    tracemalloc.start()
    result = None
    error = None
    status = "success"
    try:
        result = workload()
    except Exception as exc:  # benchmark evidence must retain failures
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    record = {
        "scenario": scenario,
        "dataset": dataset,
        "dimensions": dimensions.copy(),
        "timings_seconds": {
            "wall": time.perf_counter() - started_wall,
            "cpu": time.process_time() - started_cpu,
        },
        "peak_rss_mb": peak_rss_mb(),
        "traced_peak_mb": traced_peak / (1024 * 1024),
        "payload_bytes": 0,
        "status": status,
        "failure_phase": scenario if error else None,
        "error": error,
    }
    return record, result


def write_artifacts(artifact: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write JSON evidence plus a concise Markdown summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "revision-benchmark.json"
    markdown_path = output_dir / "revision-benchmark.md"
    encoded = json.dumps(artifact, indent=2, sort_keys=True)
    json_path.write_text(encoded + "\n", encoding="utf-8")

    rows = [
        "# Metabis revision benchmark",
        "",
        f"Dataset: `{artifact['dataset']}`",
        "",
        "| Scenario | Status | Wall (s) | CPU (s) | Peak RSS (MiB) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for record in artifact["results"]:
        rows.append(
            "| {scenario} | {status} | {wall:.6f} | {cpu:.6f} | {rss:.1f} |".format(
                scenario=record["scenario"],
                status=record["status"],
                wall=record["timings_seconds"]["wall"],
                cpu=record["timings_seconds"]["cpu"],
                rss=record["peak_rss_mb"],
            )
        )
    markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return json_path, markdown_path


def run_benchmarks(tier: str, output_dir: Path) -> dict[str, Any]:
    """Run the implemented deterministic baseline scenarios for one tier."""
    config = TIERS[tier]
    dataset = f"synthetic-{tier}-v1"
    dimensions = {
        "samples": config["samples"],
        "reactions": config["reactions"],
        "genes": 0,
        "groups": 2,
        "zero_fraction": config["zero_fraction"],
    }
    results = []

    generation_record, frame = measure_scenario(
        "synthetic-reaction-input",
        dataset,
        dimensions,
        lambda: make_synthetic_flux_matrix(
            n_samples=config["samples"],
            n_reactions=config["reactions"],
            zero_fraction=config["zero_fraction"],
        ),
    )
    results.append(generation_record)
    if frame is not None:
        generation_record["payload_bytes"] = int(frame.memory_usage(index=True).sum())

    if frame is not None:
        normalization_record, normalized = measure_scenario(
            "reaction-cubic-root-normalization",
            dataset,
            dimensions,
            lambda: apply_cbrt_normalization(frame),
        )
        normalization_record["payload_bytes"] = int(normalized.memory_usage(index=True).sum())
        results.append(normalization_record)

        group_record, manager = measure_scenario(
            "group-creation-from-prefix",
            dataset,
            dimensions,
            lambda: _create_groups(frame.index.tolist()),
        )
        group_record["dimensions"]["groups"] = manager.get_group_count() if manager else 0
        results.append(group_record)

        group_a = frame.index[::2].tolist()
        group_b = frame.index[1::2].tolist()
        statistics_record, statistics = measure_scenario(
            "reaction-only-welch-comparison",
            dataset,
            dimensions,
            lambda: comparison_test(frame, group_a, group_b, test="ttest", correction="holm"),
        )
        if statistics is not None:
            statistics_record["payload_bytes"] = int(
                statistics.memory_usage(index=True).sum()
            )
        results.append(statistics_record)

    repo = Path(__file__).resolve().parents[1]
    artifact = {
        "schema_version": 1,
        "dataset": dataset,
        "dimensions": dimensions,
        "environment": capture_environment(repo),
        "results": results,
    }
    write_artifacts(artifact, output_dir)
    return artifact


def _create_groups(sample_ids: list[str]) -> GroupManager:
    manager = GroupManager()
    manager.load_groups(sample_ids, criteria="sample_prefix")
    return manager


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=TIERS, default="tiny")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    artifact = run_benchmarks(args.tier, args.output_dir)
    return 0 if all(item["status"] == "success" for item in artifact["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

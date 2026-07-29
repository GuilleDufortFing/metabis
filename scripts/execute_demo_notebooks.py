#!/usr/bin/env python3
"""Execute the maintained demo notebooks without modifying the source files."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIRECTORY = REPOSITORY_ROOT / "demo"
MAINTAINED_NOTEBOOKS = (
    "01_metabolic_model_and_data_loading.ipynb",
    "02_groups_and_dimensionality_reduction.ipynb",
    "03_group_statistical_comparison.ipynb",
    "04_metabolic_graphs_and_subsystems.ipynb",
    "case_study.ipynb",
)
RUNTIME_DIRECTORY = Path(tempfile.gettempdir()) / "metabis-notebook-runtime"


def configure_runtime_directories() -> None:
    """Keep notebook runtime caches out of user-specific and source directories."""
    runtime_paths = {
        "IPYTHONDIR": RUNTIME_DIRECTORY / "ipython",
        "JUPYTER_CONFIG_DIR": RUNTIME_DIRECTORY / "jupyter",
        "MPLCONFIGDIR": RUNTIME_DIRECTORY / "matplotlib",
        "NUMBA_CACHE_DIR": RUNTIME_DIRECTORY / "numba",
    }
    for variable, path in runtime_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(variable, str(path))
    # A fresh Matplotlib cache announces itself on stderr from the first notebook
    # even though nothing failed. Build it once before kernel output is captured.
    with contextlib.redirect_stderr(io.StringIO()):
        from matplotlib import font_manager

        font_manager.findSystemFonts()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the maintained Metabis demo notebooks in their documented "
            "order and fail on the first cell error."
        )
    )
    parser.add_argument(
        "notebooks",
        nargs="*",
        choices=MAINTAINED_NOTEBOOKS,
        help="Optional notebook filenames; defaults to the complete maintained set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Keep executed notebooks in this directory. By default they are "
            "written to a temporary directory and removed after the gate passes."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Maximum execution time per cell in seconds (default: 900).",
    )
    return parser.parse_args()


def require_example_data() -> None:
    required = (
        "Th17_Compass_reactions.tsv",
        "Th17_Compass_expression.tsv",
        "Th17_Compass_metadata.tsv",
    )
    missing = [
        name for name in required if not (REPOSITORY_ROOT / "data" / name).is_file()
    ]
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing generated demo data: {names}. Run "
            "`python data/download_and_process_compass.py` first."
        )


def execute_notebook(source: Path, destination: Path, timeout: int) -> float:
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
        record_timing=False,
    )
    started = time.monotonic()
    client.execute(cwd=str(REPOSITORY_ROOT))
    elapsed = time.monotonic() - started
    nbformat.validate(notebook)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination)
    return elapsed


def run(notebook_names: tuple[str, ...], output_directory: Path, timeout: int) -> None:
    require_example_data()
    for name in notebook_names:
        source = DEMO_DIRECTORY / name
        if not source.is_file():
            raise FileNotFoundError(f"Maintained notebook is missing: {source}")
        print(f"Executing {name} ...", flush=True)
        elapsed = execute_notebook(source, output_directory / name, timeout)
        print(f"Passed {name} ({elapsed:.1f}s)", flush=True)


def main() -> int:
    args = parse_args()
    configure_runtime_directories()
    notebook_names = tuple(args.notebooks) or MAINTAINED_NOTEBOOKS
    try:
        if args.output_dir is not None:
            run(notebook_names, args.output_dir.resolve(), args.timeout)
        else:
            with tempfile.TemporaryDirectory(prefix="metabis-notebooks-") as temporary:
                run(notebook_names, Path(temporary), args.timeout)
    except Exception as error:
        print(f"Notebook execution gate failed: {error}", file=sys.stderr)
        return 1
    print(f"Passed {len(notebook_names)} demo notebook(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

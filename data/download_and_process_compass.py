#!/usr/bin/env python3
"""
Download and process the Compass Th17 dataset.

The downloaded example inputs are based on the public Compass repository:
https://github.com/wagnerlab-berkeley/Compass
Compass is distributed under the BSD-3-Clause license. The associated
publication is Wagner et al., Cell 2021:
https://doi.org/10.1016/j.cell.2021.05.045

Steps:
1) Download reactions.tsv, linear_gene_expression_matrix.tsv, and cell_metadata.csv
2) Run add_celltype_to_samples.py to annotate sample names with cell types
3) Convert processed TSVs to CSVs:
    - Th17_Compass_reactions.csv
    - Th17_Compass_expression.csv
4) Remove original downloaded files and intermediate TSVs
"""
import hashlib
import subprocess
import sys
from pathlib import Path

try:
    import requests
    import pandas as pd
except ImportError as e:
    print("Missing dependency:", e)
    print("Please install required packages: requests, pandas")
    sys.exit(1)


RAW_URLS = {
    "reactions.tsv": "https://raw.githubusercontent.com/wagnerlab-berkeley/Compass/master/docs/notebooks/extdata/Th17/reactions.tsv",
    "linear_gene_expression_matrix.tsv": "https://raw.githubusercontent.com/wagnerlab-berkeley/Compass/master/docs/notebooks/extdata/Th17/linear_gene_expression_matrix.tsv",
    "cell_metadata.csv": "https://raw.githubusercontent.com/wagnerlab-berkeley/Compass/master/docs/notebooks/extdata/Th17/cell_metadata.csv",
}


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_files(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for fname, url in RAW_URLS.items():
        dest = data_dir / fname
        print(f"Downloading {fname} -> {dest}")
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            sys.exit(1)
        dest.write_bytes(r.content)
        print(f"Saved {fname} ({dest.stat().st_size:,} bytes, sha256={sha256sum(dest)[:12]}…)")


def run_processing_script(data_dir: Path) -> None:
    # Use subprocess to run the existing script so its CLI/prints remain intact
    script = data_dir / "add_celltype_to_samples.py"
    if not script.exists():
        print(f"Error: Processing script not found at {script}")
        sys.exit(1)
    print("\nRunning add_celltype_to_samples.py …")
    # Run with the same Python interpreter
    cmd = [sys.executable, str(script)]
    try:
        result = subprocess.run(cmd, cwd=str(data_dir), check=True, capture_output=True, text=True)
        # Stream outputs afterward to keep logs available
        print(result.stdout)
        if result.stderr:
            print("[stderr]", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Processing script failed with exit code", e.returncode)
        print(e.stdout)
        print(e.stderr)
        sys.exit(e.returncode)


def convert_tsv_to_csv(data_dir: Path) -> None:
    # Map intermediate TSVs to final CSV names
    mapping = {
        "reactions_with_celltype.tsv": "Th17_Compass_reactions.csv",
        "linear_gene_expression_matrix_with_celltype.tsv": "Th17_Compass_expression.csv",
    }
    for src_name, out_name in mapping.items():
        src = data_dir / src_name
        out = data_dir / out_name
        if not src.exists():
            print(f"Error: Expected processed file not found: {src}")
            sys.exit(1)
        print(f"Converting {src.name} -> {out.name} (tab-delimited)")
        try:
            df = pd.read_csv(src, sep='\t', index_col=0)
            # Write final outputs as tab-delimited, keeping requested .csv filenames
            df.to_csv(out, sep='\t')
        except Exception as e:
            print(f"Error converting {src} to CSV: {e}")
            sys.exit(1)
        print(f"Saved {out.name} ({out.stat().st_size:,} bytes)")


def cleanup_files(data_dir: Path) -> None:
    # Remove the original downloads and intermediate TSVs
    to_remove = [
        data_dir / "reactions.tsv",
        data_dir / "linear_gene_expression_matrix.tsv",
        data_dir / "cell_metadata.csv",
        data_dir / "reactions_with_celltype.tsv",
        data_dir / "linear_gene_expression_matrix_with_celltype.tsv",
    ]
    print("\nCleaning up original and intermediate files …")
    removed = 0
    for p in to_remove:
        try:
            if p.exists():
                p.unlink()
                removed += 1
                print(f"Removed {p.name}")
        except Exception as e:
            print(f"Warning: Could not remove {p}: {e}")
    print(f"Cleanup complete. Files removed: {removed}")


def main():
    data_dir = Path(__file__).resolve().parent
    print("=== Compass Th17: Download and Process ===")
    print(f"Data directory: {data_dir}")

    download_files(data_dir)
    run_processing_script(data_dir)
    convert_tsv_to_csv(data_dir)
    cleanup_files(data_dir)

    final_react = data_dir / "Th17_Compass_reactions.csv"
    final_expr = data_dir / "Th17_Compass_expression.csv"

    print("\n=== Done ===")
    print("Created files:")
    print(f" - {final_react}")
    print(f" - {final_expr}")


if __name__ == "__main__":
    main()

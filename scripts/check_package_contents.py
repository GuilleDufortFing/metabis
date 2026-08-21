"""Reject repository-local and review-process files from built distributions."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

FORBIDDEN_SEGMENTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "benchmark-results",
        "browser-profiles",
        "metabisreview",
        "manuscript",
        "rebuttal",
        "submission",
    }
)
FORBIDDEN_SUFFIXES = frozenset(
    {".aux", ".log", ".out", ".pyc", ".pyo", ".zip"}
)


def archive_names(path: Path) -> tuple[str, ...]:
    """Return member names from a wheel/zip or gzipped source distribution."""
    if path.suffix in {".whl", ".zip"}:
        with zipfile.ZipFile(path) as archive:
            return tuple(archive.namelist())
    if path.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path, mode="r:gz") as archive:
            return tuple(member.name for member in archive.getmembers())
    raise ValueError(f"Unsupported distribution archive: {path}")


def forbidden_members(path: Path) -> tuple[str, ...]:
    """Return archive members that violate the public-package content policy."""
    rejected: list[str] = []
    for name in archive_names(path):
        member = PurePosixPath(name)
        segments = {part.casefold() for part in member.parts}
        if segments & FORBIDDEN_SEGMENTS or member.suffix.casefold() in FORBIDDEN_SUFFIXES:
            rejected.append(name)
    return tuple(sorted(rejected))


def main() -> int:
    """Validate every distribution archive named on the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()

    failures: list[str] = []
    for archive in args.archives:
        rejected = forbidden_members(archive)
        if rejected:
            failures.extend(f"{archive}: {member}" for member in rejected)
        else:
            print(f"{archive}: package content clean")
    if failures:
        print("Forbidden package members:")
        print("\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

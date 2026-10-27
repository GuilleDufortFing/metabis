#!/usr/bin/env python3
"""Download and install the bundled Metabis example datasets.

The dataset matrices are too large for Git. This script downloads the author-
maintained archive from the FING cloud share and extracts it beside this file:

    python data/download_datasets.py

It installs the same files the Metabis server ships, so a demo notebook and the
deployed application read byte-identical inputs:

    - Th17_Compass_reactions.tsv, Th17_Compass_expression.tsv,
      Th17_Compass_metadata.tsv  (notebooks 01 to 04 and the Th17 case study)
    - smillie_clusters_pool_reactions.tsv,
      smillie_clusters_pool_expression_cpm.tsv, smillie_metadata.tsv
      (the Smillie multi-group notebook)
    - smillie_pseudobulk_reactions.tsv,
      smillie_pseudobulk_expression_cp10k.tsv, smillie_pseudobulk_metadata.tsv
      (the server's Smillie example: Compass's pseudobulk notebook rebuilt)

The archive is the single source for these files, so the notebooks and the
deployed application read byte-identical inputs.

The public share page itself returns HTML. ``DOWNLOAD_URL`` therefore uses the
Nextcloud ``/download`` endpoint, which currently serves ``MetabisDatasets.zip``.
The archive is downloaded to a temporary file, checked against the recorded
SHA-256 digest, safely extracted, and then removed.

The Metabis server provisions the same archive with its own copy of this script.
Publishing a new archive therefore means updating ``DOWNLOAD_URL`` and
``ARCHIVE_SHA256`` in both files.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

DATA_DIR = Path(__file__).resolve().parent
DOWNLOAD_URL = "https://nube.fing.edu.uy/index.php/s/FSRdpzGAiAAiyJF/download"
ARCHIVE_NAME = "MetabisDatasets.zip"
ARCHIVE_SHA256 = "f87fc820d0bd07fd6615ef6f8409f83a8e4b1db61c59c4c084dc6229d68ee1f8"
CHUNK_SIZE = 1024 * 1024


def download_archive(url: str, destination: Path) -> str:
    """Stream ``url`` to ``destination`` and return its SHA-256 digest."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Metabis dataset provisioner"},
    )
    digest = hashlib.sha256()
    downloaded = 0

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with destination.open("wb") as output:
                while chunk := response.read(CHUNK_SIZE):
                    output.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    print(
                        f"\rDownloaded {downloaded / 1024**2:.1f} MiB",
                        end="",
                        flush=True,
                    )
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc

    print()
    return digest.hexdigest()


def _safe_archive_members(archive: zipfile.ZipFile, output_dir: Path):
    """Return regular archive members whose paths stay inside ``output_dir``."""
    output_root = output_dir.resolve()
    members: list[tuple[zipfile.ZipInfo, Path]] = []

    for member in archive.infolist():
        relative = PurePosixPath(member.filename)
        if member.is_dir() or relative.parts[:1] == ("__MACOSX",):
            continue
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.name.startswith("._")
        ):
            raise RuntimeError(f"Unsafe path in dataset archive: {member.filename!r}")

        file_type = stat.S_IFMT(member.external_attr >> 16)
        if file_type == stat.S_IFLNK:
            raise RuntimeError(f"Symlink in dataset archive: {member.filename!r}")
        if file_type not in (0, stat.S_IFREG):
            raise RuntimeError(
                f"Unsupported entry in dataset archive: {member.filename!r}"
            )

        destination = (output_dir / Path(*relative.parts)).resolve()
        if destination != output_root and output_root not in destination.parents:
            raise RuntimeError(f"Unsafe path in dataset archive: {member.filename!r}")
        members.append((member, destination))

    if not members:
        raise RuntimeError("The dataset archive contains no files")
    return members


def extract_archive(archive_path: Path, output_dir: Path) -> list[Path]:
    """Safely and atomically extract dataset files into ``output_dir``."""
    if not zipfile.is_zipfile(archive_path):
        raise RuntimeError(f"Downloaded file is not a ZIP archive: {archive_path}")

    installed: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        members = _safe_archive_members(archive, output_dir)
        for member, destination in members:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.part")
            try:
                with archive.open(member) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=CHUNK_SIZE)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            installed.append(destination)
    return installed


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ARCHIVE_NAME}")
    print(f"Source: {DOWNLOAD_URL}")
    print(f"Destination: {DATA_DIR}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".MetabisDatasets-",
            suffix=".zip",
            dir=DATA_DIR,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)

        digest = download_archive(DOWNLOAD_URL, temporary_path)
        if digest != ARCHIVE_SHA256:
            raise RuntimeError(
                "Dataset archive checksum mismatch: "
                f"expected {ARCHIVE_SHA256}, received {digest}"
            )
        print(f"Verified SHA-256: {digest}")

        installed = extract_archive(temporary_path, DATA_DIR)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"Installed {len(installed)} dataset file(s):")
    for path in installed:
        print(f" - {path.relative_to(DATA_DIR)} ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

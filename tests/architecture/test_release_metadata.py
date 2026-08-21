"""Keep the stable release version and its public documentation coherent."""

from __future__ import annotations

import runpy
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_version_source_matches_release_documentation():
    version = runpy.run_path(PROJECT_ROOT / "src/metabis/version.py")["__version__"]
    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert version == "1.0.0"
    assert f"## [{version}] - " in changelog
    assert f"stable public-interface release is `{version}`" in readme


def test_package_metadata_uses_the_canonical_version_source():
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
        metadata = tomllib.load(file)

    assert metadata["project"]["dynamic"] == ["version"]
    assert metadata["project"]["requires-python"] == ">=3.12"
    assert metadata["tool"]["hatch"]["version"]["path"] == "src/metabis/version.py"

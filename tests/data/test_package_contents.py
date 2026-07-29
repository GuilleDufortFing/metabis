"""Protect built distributions from repository-local and review-process files."""

from __future__ import annotations

import importlib.util
import tarfile
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_package_contents.py"
SPEC = importlib.util.spec_from_file_location("check_package_contents", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PACKAGE_CONTENTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE_CONTENTS)


def test_wheel_content_guard_rejects_review_and_cache_files(tmp_path):
    archive = tmp_path / "metabis.whl"
    with zipfile.ZipFile(archive, "w") as wheel:
        wheel.writestr("metabis/__init__.py", "")
        wheel.writestr("MetabisReview/status.md", "")
        wheel.writestr("metabis/__pycache__/module.pyc", "")

    assert PACKAGE_CONTENTS.forbidden_members(archive) == (
        "MetabisReview/status.md",
        "metabis/__pycache__/module.pyc",
    )


def test_sdist_content_guard_rejects_repository_metadata(tmp_path):
    source = tmp_path / ".git"
    source.write_text("gitdir: local", encoding="utf-8")
    archive = tmp_path / "metabis.tar.gz"
    with tarfile.open(archive, "w:gz") as sdist:
        sdist.add(source, arcname="metabis-1.0.0/.git")

    assert PACKAGE_CONTENTS.forbidden_members(archive) == (
        "metabis-1.0.0/.git",
    )

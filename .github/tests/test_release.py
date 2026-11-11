"""Test release rejection paths without importing astroimred or its dependencies."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "check_release", _ROOT / "scripts/check_release.py"
)
assert _SPEC is not None and _SPEC.loader is not None
release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Create a minimal, valid release tree."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "astroimred"\nversion = "1.2.3"\n'
        'dependencies = ["numpy>=1.24"]\n'
        '[build-system]\nrequires = ["uv_build>=0.9.18,<0.10.0"]\n',
        encoding="utf-8",
    )
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## 1.2.3 - 2026-09-07\n\n- Release change.\n"
        "\n## 1.2.2 - 2026-09-01\n\n- Older change.\n",
        encoding="utf-8",
    )
    return tmp_path


def test_release_extracts_only_matching_notes(project: Path) -> None:
    assert release.check_release(project, tag="v1.2.3") == (
        "1.2.3",
        "- Release change.\n",
    )


def test_rehearsal_accepts_pending_changes(project: Path) -> None:
    (project / "CHANGELOG.md").write_text(
        "## Unreleased\n\n- Pending.\n", encoding="utf-8"
    )
    assert release.check_release(project) == ("1.2.3", "- Pending.\n")


@pytest.mark.parametrize("tag", ["v1.2.4", "1.2.3", "v1.2.3rc1", "v01.2.3", ""])
def test_rejects_wrong_tags(project: Path, tag: str) -> None:
    with pytest.raises(ValueError, match="tag"):
        release.check_release(project, tag=tag)


@pytest.mark.parametrize("version", ["1.2.3rc1", "01.2.3", "1.2", "1.2.3+local"])
def test_rejects_nonstable_versions(project: Path, version: str) -> None:
    manifest = project / "pyproject.toml"
    manifest.write_text(
        manifest.read_text().replace('"1.2.3"', f'"{version}"'), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="stable"):
        release.check_release(project)


@pytest.mark.parametrize(
    "notes",
    [
        "## Unreleased\n\n- Pending.\n\n## 1.2.3 - 2026-09-07\n\n- Change.\n",
        "## 1.2.3 - 2026-09-07\n",
        "## 1.2.3 - 2026-09-07\n\n### Added\n",
        "## 1.2.3 - 2026-02-30\n\n- Change.\n",
        "## 1.2.3 - soon\n\n- Change.\n",
        "## 1.2.2 - 2026-09-07\n\n- Change.\n",
        "## 1.2.3 - 2026-09-07\n\n- A.\n\n## 1.2.3 - 2026-09-08\n\n- B.\n",
        "## Unreleased\n\n## Unreleased\n",
    ],
)
def test_rejects_unready_release_notes(project: Path, notes: str) -> None:
    (project / "CHANGELOG.md").write_text(notes, encoding="utf-8")
    with pytest.raises(ValueError):
        release.check_release(project, tag="v1.2.3")


def test_rejects_direct_dependencies(project: Path) -> None:
    manifest = project / "pyproject.toml"
    manifest.write_text(
        manifest.read_text().replace("numpy>=1.24", "numpy @ file:///sibling/numpy"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="registry"):
        release.check_release(project)


def test_clean_check_detects_untracked_files(project: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(project)], check=True)
    with pytest.raises(ValueError, match="pending changes"):
        release.check_release(project, require_clean=True)


def test_cli_outputs_and_failure(project: Path, tmp_path: Path) -> None:
    output = tmp_path / "output"
    notes = tmp_path / "notes"
    assert (
        release.main(
            [
                "--root",
                str(project),
                "--tag",
                "v1.2.3",
                "--github-output",
                str(output),
                "--notes-output",
                str(notes),
            ]
        )
        == 0
    )
    assert output.read_text() == "version=1.2.3\n"
    assert notes.read_text() == "- Release change.\n"
    assert release.main(["--root", str(project), "--tag", "v9.0.0"]) == 1

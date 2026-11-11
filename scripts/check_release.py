"""Check the version and changelog before building or publishing astroimred."""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import tomllib
from collections.abc import Sequence
from datetime import date
from pathlib import Path

_VERSION = r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"


def check_release(
    root: Path, *, tag: str | None = None, require_clean: bool = False
) -> tuple[str, str]:
    """Validate a rehearsal or tagged release without importing the package.

    Parameters
    ----------
    root : pathlib.Path
        Repository containing `pyproject.toml` and `CHANGELOG.md`.
    tag : str or None, optional
        Exact ``vX.Y.Z`` tag. `None` allows pending `Unreleased` notes.
    require_clean : bool, optional
        Reject staged, unstaged, and untracked files.

    Returns
    -------
    version : str
        Stable package version.
    notes : str
        Validated changelog section, with a final newline when nonempty.

    Raises
    ------
    ValueError
        Metadata, dependency sources, release notes, or Git state are invalid.
    OSError
        A required file or Git executable cannot be accessed.
    """
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    if project["name"] != "astroimred":
        raise ValueError("pyproject.toml: expected package name astroimred")
    version = project["version"]
    if not isinstance(version, str) or re.fullmatch(_VERSION, version) is None:
        raise ValueError("pyproject.toml: version must be stable X.Y.Z")
    if tag is not None and tag != f"v{version}":
        raise ValueError(f"tag {tag!r} must match v{version}")

    # uv-only overrides are disabled in CI; published requirements must stand alone.
    requirements = list(project.get("dependencies", []))
    requirements.extend(metadata["build-system"]["requires"])
    for group in project.get("optional-dependencies", {}).values():
        requirements.extend(group)
    for requirement in requirements:
        if "@" in requirement.split(";", 1)[0]:
            raise ValueError(f"Use a registry dependency instead of {requirement!r}")

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = list(re.finditer(r"^##[ \t]+(.+?)[ \t]*$", changelog, re.MULTILINE))
    sections: dict[str, str] = {}
    for index, heading in enumerate(headings):
        title = heading.group(1)
        if title in sections:
            raise ValueError(f"CHANGELOG.md: duplicate section {title!r}")
        end = (
            headings[index + 1].start() if index + 1 < len(headings) else len(changelog)
        )
        sections[title] = changelog[heading.end() : end].strip()
    releases = [
        (title, body)
        for title, body in sections.items()
        if title.startswith(f"{version} - ")
    ]
    if len(releases) > 1:
        raise ValueError(f"CHANGELOG.md: duplicate release for {version}")
    if tag is not None:
        if sections.get("Unreleased", "").strip():
            raise ValueError(
                "Move Unreleased notes into the dated release section before tagging"
            )
        if not releases:
            raise ValueError(f"CHANGELOG.md: expected ## {version} - YYYY-MM-DD")
        title, notes = releases[0]
        release_date = title.removeprefix(f"{version} - ")
        if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", release_date) is None:
            raise ValueError("CHANGELOG.md: release date must be YYYY-MM-DD")
        date.fromisoformat(release_date)
        if not any(
            line.strip() and not line.lstrip().startswith("#")
            for line in notes.splitlines()
        ):
            raise ValueError("CHANGELOG.md: release notes must be nonempty")
    elif "Unreleased" in sections:
        notes = sections["Unreleased"]
    elif releases:
        notes = releases[0][1]
    else:
        raise ValueError(
            "CHANGELOG.md: expected Unreleased or a matching release section"
        )

    if require_clean:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )
        if result.stdout:
            raise ValueError("Commit or remove pending changes before releasing")
    return version, f"{notes}\n" if notes else ""


def main(argv: Sequence[str] | None = None) -> int:
    """Validate metadata and optionally save outputs for GitHub Actions.

    Parameters
    ----------
    argv : sequence of str or None, optional
        CLI arguments; `None` uses process arguments.

    Returns
    -------
    int
        Zero on success, one on invalid metadata or inaccessible files.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--tag")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--notes-output", type=Path)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args(argv)
    try:
        version, notes = check_release(
            args.root, tag=args.tag, require_clean=args.require_clean
        )
        if args.notes_output:
            args.notes_output.write_text(notes, encoding="utf-8")
        if args.github_output:
            with args.github_output.open("a", encoding="utf-8") as stream:
                stream.write(f"version={version}\n")
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        logging.error("Release validation failed: %s", error)
        return 1
    logging.info("Validated astroimred %s", version)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())

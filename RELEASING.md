# Release astroimred

**Run a manual rehearsal, then push the matching `vX.Y.Z` tag to publish.**
Use the exact commit that passed the rehearsal.

| Event | Checks | Publication |
| --- | --- | --- |
| Branch push or pull request | CI and installed distributions | None |
| Manual `release.yml` | Same checks; dated notes required when run on a tag | None |
| Matching `vX.Y.Z` tag push | Same checks, plus dated release notes | PyPI, then GitHub release |

## Shared convention

Keep publication in one `release.yml`. The convention combines ImCombiners'
tag trigger, reducers' reusable CI and early release-note checks, and pqfilt's
installed-artifact checks. A GitHub release records successful publication;
creating one does not trigger publishing.

| Package type | Artifacts | Registries |
| --- | --- | --- |
| `astroimred`: pure Python, `uv_build` | One universal wheel and sdist | PyPI |
| `astroapers`: Maturin, public Rust API | Per-Python/platform wheels and sdist | PyPI and crates.io |

Publish the exact Python files that passed installation tests. Rust publication
packages the checked commit again with Cargo. Use registry dependencies and
temporary OIDC credentials. Never synchronize `~/.venvs/ysbpy`.

## One-time setup

1. Confirm the GitHub repository is `ysBach/astroimred`. Older clones can retain
   the former `fitsmgmt` remote; inspect `git remote -v` and update it if needed:

   ```sh
   git remote set-url origin https://github.com/ysBach/astroimred.git
   ```

   The publisher job permits only `ysBach/astroimred`. A repository rename also
   requires changing that guard and the PyPI trusted-publisher settings.

2. Create a GitHub environment named `pypi`. Restrict deployment to **tags**
   matching `v*`. For fully automatic releases after tagging, do not configure
   required environment reviewers. Protect tag creation with a repository ruleset.

3. Add a trusted publisher in
   [PyPI Publishing settings](https://pypi.org/manage/project/astroimred/settings/publishing/):

   | Setting | Value |
   | --- | --- |
   | Project | `astroimred` |
   | Owner / repository | `ysBach` / `astroimred` |
   | Workflow filename | `release.yml` |
   | Environment | `pypi` |

   For a project that does not exist yet, use a
   [pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
   No permanent PyPI token belongs in GitHub secrets.

4. Enable Actions and require the CI build and installed-distribution checks
   on `main`. Select their actual check names after the first hosted run.
   Repository and registry settings are external setup; committing these files
   does not configure them.

## Prepare and rehearse

1. Select the next unused version in `pyproject.toml`. Move `Unreleased` notes
   into one nonempty `## X.Y.Z - YYYY-MM-DD` section with the real release date.
   Leave `Unreleased` empty or remove it. The checker rejects prerelease tags,
   mismatches, duplicate sections, and pending notes on tagged releases.

2. Activate the shared environment for local release-tool checks:

   ```sh
   source ~/.venvs/ysbpy/bin/activate
   uv run --active --no-sync python scripts/check_release.py
   uv run --active --no-sync pytest .github/tests -q
   ```

   Release tooling requires Python 3.11+. With a project `.venv`, omit `--active`.
   Do not run `uv sync` or `uv pip sync` against the shared environment.
   `uv.lock` is intentionally untracked here: the version comes from
   `pyproject.toml`, and CI resolves published dependencies for fresh installs.

3. Review and commit the intended files, including new files. Push the commit
   and start a rehearsal:

   ```sh
   uv run --active --no-sync python scripts/check_release.py --require-clean
   git push origin main
   gh workflow run release.yml --ref main
   gh run list --workflow release.yml
   ```

4. Inspect the run with `gh run view RUN_ID`. All checks must pass; PyPI and
   GitHub release jobs must be skipped. Record the successful run's full commit
   SHA with `gh run view RUN_ID --json headSha --jq .headSha`.

## Publish

1. Check out that exact commit. Replace `REHEARSED_COMMIT_SHA` and `X.Y.Z`:

   ```sh
   git switch --detach REHEARSED_COMMIT_SHA
   uv run --active --no-sync python scripts/check_release.py --tag vX.Y.Z --require-clean
   git tag -a vX.Y.Z -m "Release X.Y.Z"
   git push origin vX.Y.Z
   git switch main
   ```

   **Pushing the tag publishes automatically.** A version bump or ordinary push
   does not publish. Versions are selected by the maintainer, not by CI.

2. Check the tag-triggered Actions run, the PyPI version page, and the GitHub
   release. Its wheel, sdist, and notes come from the same successful run.

## Coverage and recovery

### Check the tested artifacts

CI builds a `py3-none-any` wheel and sdist.

| Platform | Python coverage |
| --- | --- |
| Linux | 3.11–3.14; also installs the sdist on 3.14 |
| macOS ARM64 | 3.13 |
| Windows x86-64 | 3.13 |

Each install uses the `full` extra in a fresh environment. Checks cover isolated
imports, license/FITS handling, CLI help, and the Python suite. Release-tool
tests and Ruff checks cover `scripts/` and `.github/tests/`; existing application
formatting is outside this publishing change. These are configured targets,
not evidence of compatibility until hosted runs pass.

`UV_NO_SOURCES=true` disables developer overrides. Dependencies must be available
from registries; a test that needs unpublished sibling changes must fail until
those dependencies are released. CI intentionally does not substitute sibling
checkouts. The same retained wheel and sdist are uploaded without rebuilding.

### Recover without changing published files

- **Before uploading:** fix the problem, commit, and rehearse again. Source or
  artifact changes after a tag push require a new version/tag; never move a
  published tag.
- **Upload/network/setup failure:** fix the external issue and rerun only failed
  jobs in the original tag-push run. Keep its original artifacts. `uv publish`
  skips identical existing files and rejects conflicting bytes; see
  [uv retry behavior](https://docs.astral.sh/uv/guides/package/).
- **GitHub release failure after PyPI succeeded:** rerun the failed job if no
  GitHub release exists. If one exists, inspect it and restore missing assets
  from the original run. Do not rebuild files or republish changed bytes.

PyPI publication and GitHub release creation are separate operations; a partial
release is possible. Retain the original successful artifacts during recovery.

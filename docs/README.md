# Build and publish the documentation

**Edit QMD pages or Python docstrings, then push to `main`.**
After one-time Pages setup, Actions builds and publishes
<https://ysbach.github.io/astroimred/> automatically.

## Local preview

1. Install [Quarto 1.9.37](https://quarto.org/docs/download/). Choose an
   environment from the repository root:

   **Reuse the project `.venv`:**

   ```sh
   uv pip install --python .venv/bin/python -r docs/requirements-quarto.txt -e ".[full]"
   ```

   **Or isolate documentation tools in `.venv-docs`:**

   ```sh
   uv venv --python 3.13 .venv-docs
   uv pip install --python .venv-docs/bin/python -r docs/requirements-quarto.txt -e ".[full]"
   ```

   `--python` selects the environment explicitly, regardless of activation.
   `.venv-docs` is optional. Never run `uv sync` or `uv pip sync` against
   `~/.venvs/ysbpy`.

2. Build API pages and execute the offline tutorials:

   ```sh
   make docs-build
   ```

   This defaults to `.venv-docs`. For `.venv`, use
   `make docs-build DOCS_PYTHON=.venv/bin/python`. Any prepared environment
   works with `DOCS_PYTHON=/absolute/path/to/python`.
   Output is `docs/quarto/_site/`; `make` never installs or synchronizes packages.

3. Preview while editing:

   ```sh
   make docs-preview
   ```

   Add the same `DOCS_PYTHON` argument when using `.venv` or another environment.
   Open the local URL printed by Quarto. After changing Python docstrings,
   restart the preview or run `make docs-api` to regenerate reference pages.

## One-time GitHub Pages setup

1. In `ysBach/astroimred`, open **Settings → Pages → Build and deployment**.
   Select **GitHub Actions** as the source.
2. Ensure Actions is enabled. Permit the `github-pages` environment to deploy
   from `main`. If you require automatic publishing, omit required reviewers.
3. Commit the site sources and workflow, then push to `main` or run the `Docs`
   workflow manually on `main`. Check its build and deploy jobs.

| Event | Build | Deploy |
| --- | --- | --- |
| Pull request | API and executable tutorials | Never |
| Push to `main` | API and executable tutorials | After a successful build |
| Manual run on `main` | API and executable tutorials | After a successful build |
| Manual run elsewhere or on a fork | API and executable tutorials | Never |

Deployment uses GitHub's Pages artifact and temporary OIDC identity. No personal
token or `gh-pages` branch is needed. Committing the workflow does not change
repository Settings; the one-time setup above is still required.

## Authoring convention

### Write for quick reading without losing detail

Apply the **I Have ADHD** style to maintained guides, API docstrings, examples,
changelogs, and contributor docs:

1. **Lead with the task or result.** Put the first useful command or example
   near the start; remove preambles and repeated conclusions.
2. **Make the structure visible.** Use short paragraphs, descriptive headings,
   numbered procedures, and tables for comparisons. Split long lists into
   meaningful groups of at most five items.
3. **Preserve the contract.** Keep arguments, defaults, units, shapes, return
   values, equations, examples, references, and mask/NaN/error behavior.
   Move detail into a focused section rather than deleting it.
4. **Keep limitations beside the relevant step.** Distinguish measured results,
   assumptions, and unknowns. Never invent timings or guarantees for brevity.
5. **Keep API docstrings in NumPy style.** Use single backticks for objects and
   arguments, double backticks for code expressions. Keep paragraphs and lists
   readable when rendered. Reference pages do not need conversational closing
   actions or time estimates.

For each changelog update, give a compact before/after example:
``arg=old`` produced one behavior; ``arg=new`` now produces another.
When handing off a change, include a commit-ready sentence and the files to stage.

### Keep the established reading path

The Sphinx sources are short FITS and reduction API outlines. The website adds
companion guides without replacing them. No tracked tutorial notebooks exist
at the time this reading path was prepared.

| Established source | Website companion |
| --- | --- |
| `docs/index.rst`, `docs/api.rst` | FITS quickstart, learning path, generated API |
| `docs/reduction/api/preproc.rst` | Calibration tutorial and preprocessing API |
| `docs/reduction/api/crrej.rst` | Cosmic-ray API linked from calibration |
| `docs/reduction/api/cli.rst` | Command-line guide |

Expand these established topics first. Proposal documents and experimental
work stay outside the main reading path. Maintain QMD navigation and companion
pages alongside the original Sphinx outlines.

### Edit maintained sources

| Content | Maintained source |
| --- | --- |
| Landing page and tutorials | `docs/quarto/*.qmd`, `docs/quarto/tutorials/*.qmd` |
| Function signatures and descriptions | Python docstrings |
| Navigation, API groups, execution policy | `docs/quarto/_quarto.yml` |
| Builder versions | `docs/requirements-quarto.txt` |

quartodoc reads NumPy-style docstrings from the current checkout. Most entries
use static inspection; `imcombine` needs a module import because its docstring
is assembled at runtime. API pages, search data, figures, and HTML are generated
and ignored by Git.

### Check the rendered output

`make docs-api` also converts legacy RST math, notes, warnings, and code
blocks in generated pages to Quarto markup. For example,
``.. warning::`` becomes a warning callout. Fenced code remains literal.
The local quartodoc renderer preserves paragraph breaks, lists, and nested
blocks in parameter descriptions instead of flattening them into one line.
Empty directives and unsupported directive options fail the build; simplify
the source docstring or explicitly extend the docs adapter when needed.

Small synthetic examples execute on every build. `freeze: false` prevents
code-only package updates from reusing old results. A failing example fails
the build and leaves the published site unchanged. Live services and large
observational datasets do not belong in the default build.

The older Sphinx tree remains available as historical source. The Quarto site
is the maintained documentation entry point. Package releases continue through
`release.yml`; documentation publishing is independent and follows `main`.

## Why Quarto rather than Marimo?

Quarto provides the shared navigation, search, prose, code output, and API
reference used in `imcombiners` and `astroapers`.
[Its static sites deploy through GitHub Pages](https://quarto.org/docs/publishing/github-pages.html).
Marimo is useful for reactive experiments; its static HTML exports do not
rerun Python, while browser execution depends on
[Pyodide-compatible packages](https://docs.marimo.io/guides/wasm/).
Keep such experiments optional rather than making them prerequisites for
reading a package reference.

Use the same tooling with different emphasis: `astroimred` starts from FITS
workflows, `ndfiltrs` starts from API semantics, and `skyloc` starts from
tutorials with explicit external-data requirements.

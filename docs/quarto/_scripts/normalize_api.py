"""Convert supported legacy RST markup in generated API pages to Quarto."""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path


def _next_fence(line: str, fence: str | None) -> str | None:
    """Track Markdown fences without treating shorter markers as closers."""
    marker = re.match(r"^[ \t]*(`{3,}|~{3,})(.*)$", line)
    if marker is None:
        return fence
    token, suffix = marker.groups()
    if fence is None:
        return token
    if token[0] == fence[0] and len(token) >= len(fence) and not suffix.strip():
        return None
    return fence


def normalize_api(markdown: str) -> str:
    """Convert simple RST block directives to Quarto Markdown.

    Parameters
    ----------
    markdown : str
        Generated API text. Supported directives are ``math``, ``note``,
        ``warning``, and ``code-block`` with `text` or `python` language.

    Returns
    -------
    str
        Markdown with display math, callouts, and code fences. Existing
        fenced code and ordinary Markdown remain unchanged.

    Raises
    ------
    ValueError
        A supported directive is empty or uses options or unsupported arguments.

    Notes
    -----
    This docs build adapter handles the supported block forms only. Other
    directive types remain unchanged. Repeated normalization has no effect.
    """
    lines = markdown.splitlines(keepends=True)
    output: list[str] = []
    fence: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        previous_fence = fence
        fence = _next_fence(line, fence)
        directive = re.match(
            r"^([ \t]*)\.\. (math|note|warning|code-block)::(.*)$", line
        )
        if previous_fence is not None or fence is not None or directive is None:
            output.append(line)
            index += 1
            continue

        indent, kind, argument = directive.groups()
        argument = argument.strip()
        if (kind == "code-block" and argument not in {"text", "python"}) or (
            kind != "code-block" and argument
        ):
            raise ValueError(f"Unsupported RST {kind} argument: {argument!r}.")
        index += 1
        indented: list[str] = []
        while index < len(lines):
            candidate = lines[index]
            prefix = candidate[: len(candidate) - len(candidate.lstrip(" \t"))]
            if candidate.strip() and len(prefix.expandtabs()) <= len(
                indent.expandtabs()
            ):
                break
            indented.append(prefix.expandtabs() + candidate[len(prefix) :])
            index += 1
        body = textwrap.dedent("".join(indented)).strip("\n")
        if not body.strip() or re.match(r":[\w-]+:", body.lstrip()):
            raise ValueError(
                f"RST {kind} needs a nonempty indented body without options."
            )

        if kind == "math":
            opening = closing = "$$"
        elif kind == "code-block":
            longest = max((len(run) for run in re.findall(r"`+", body)), default=0)
            closing = "`" * max(3, longest + 1)
            opening = closing + argument
        else:
            body = normalize_api(body).rstrip("\n")
            longest = max(
                (len(run) for run in re.findall(r"(?m)^\s*(:{3,})", body)),
                default=0,
            )
            closing = ":" * max(3, longest + 1)
            opening = f"{closing} {{.callout-{kind}}}"
        output.append(f"{indent}{opening}\n")
        output.extend(f"{indent}{row}\n" for row in body.splitlines())
        output.append(f"{indent}{closing}\n\n")
    return "".join(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("api_dir", type=Path, help="Generated quartodoc API directory.")
    args = parser.parse_args()
    for page in sorted(args.api_dir.glob("*.qmd")):
        original = page.read_text(encoding="utf-8")
        normalized = normalize_api(original)
        if normalized != original:
            page.write_text(normalized, encoding="utf-8")

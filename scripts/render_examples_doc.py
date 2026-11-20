"""Render the public Sphinx examples page from example READMEs."""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
OUTPUT_PATH = ROOT / "doc" / "examples.md"


_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")
_ALERT_RE = re.compile(r"^> \[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*$")


def _demote_headings_outside_fences(content: str) -> str:
    """Demote every Markdown heading by one level, leaving fenced code untouched.

    ``#`` comment lines inside TOML, Python and shell code blocks are not
    headings, so the published code stays identical to the shipped files.
    """
    out: list[str] = []
    fence: str | None = None
    for line in content.splitlines(keepends=True):
        match = _FENCE_RE.match(line)
        if fence is None and match:
            fence = match.group(1)
        elif (
            fence is not None and line.startswith(fence[0] * len(fence)) and line.strip() == line.strip()[0] * len(line.strip())
        ):
            fence = None
        elif fence is None and line.startswith("#"):
            line = "#" + line
        out.append(line)
    return "".join(out)


def _convert_github_alerts(content: str) -> str:
    """Turn GitHub ``> [!NOTE]`` alert blocks into MyST admonitions.

    MyST renders the GitHub syntax as a literal ``[!NOTE]`` blockquote.
    """
    out: list[str] = []
    lines = content.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        match = _ALERT_RE.match(lines[i].rstrip("\n"))
        if not match:
            out.append(lines[i])
            i += 1
            continue
        body: list[str] = []
        i += 1
        while i < len(lines) and lines[i].startswith(">"):
            body.append(lines[i][2:] if lines[i].startswith("> ") else lines[i][1:])
            i += 1
        out.append(f":::{{{match.group(1).lower()}}}\n")
        out.extend(body)
        if body and not body[-1].endswith("\n"):
            out.append("\n")
        out.append(":::\n")
    return "".join(out)


def render_examples_page() -> str:
    """Return the complete generated ``doc/examples.md`` content."""
    header = (
        "(examples_link)=\n"
        "\n"
        "# Examples\n"
        "\n"
        "Example files for **jQMC** are found at\n"
        "<https://github.com/jqmc-project/jQMC/tree/main/examples>.\n"
        "\n"
    )

    # Put the curated interface/feature index before the individual tutorial
    # bodies, so that readers find the twist, stress, MPC or open-shell
    # tutorial by need.
    index_content = (EXAMPLES_DIR / "README.md").read_text("utf-8")
    index_content = re.sub(r"\A# Examples\s*\n+", "", index_content, count=1)
    # Index links into the manual are written relative to examples/
    # (``../doc/notes/x.md``); the generated page lives in doc/.
    index_content = re.sub(r"(?<!!)(\[[^\]]*\])\(\.\./doc/([^)]+)\)", r"\1(\2)", index_content)

    sections: list[str] = []
    seen_footnotes: set[str] = set()
    for readme in sorted(EXAMPLES_DIR.glob("jqmc-*/README.md")):
        example_name = readme.parent.name
        rel_dir = f"../examples/{example_name}"
        content = readme.read_text("utf-8")

        content = _demote_headings_outside_fences(content)
        content = _convert_github_alerts(content)

        def fix_image(match: re.Match[str], directory: str = rel_dir) -> str:
            """Rewrite a local image relative to the generated page."""
            alt, path = match.group(1), match.group(2)
            if path.startswith(("http://", "https://")):
                return match.group(0)
            return f"![{alt}]({directory}/{path})"

        content = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", fix_image, content)
        # README links into the manual are written relative to the example
        # directory (``../../doc/notes/x.md``); the generated page lives in doc/.
        content = re.sub(r"(?<!!)(\[[^\]]*\])\(\.\./\.\./doc/([^)]+)\)", r"\1(\2)", content)

        def deduplicate_footnote(match: re.Match[str]) -> str:
            """Keep the first definition of a cross-example footnote."""
            label = match.group(1)
            if label in seen_footnotes:
                return ""
            seen_footnotes.add(label)
            return match.group(0)

        content = re.sub(
            r"^\[\^([^\]]+)\]:.*$",
            deduplicate_footnote,
            content,
            flags=re.MULTILINE,
        )
        content = re.sub(r"^(```\w+):[^\s`]+", r"\1", content, flags=re.MULTILINE)
        sections.append(content.rstrip())

    return header + index_content.rstrip() + "\n\n" + "\n\n".join(sections) + "\n"


def write_examples_page() -> bool:
    """Write the generated page and return whether it changed."""
    rendered = render_examples_page()
    if OUTPUT_PATH.exists() and OUTPUT_PATH.read_text("utf-8") == rendered:
        return False
    OUTPUT_PATH.write_text(rendered, "utf-8")
    return True


def main() -> int:
    """Render the page, or fail without writing when ``--check`` is used."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if doc/examples.md is stale.")
    args = parser.parse_args()

    rendered = render_examples_page()
    current = OUTPUT_PATH.read_text("utf-8") if OUTPUT_PATH.exists() else None
    if args.check:
        if current != rendered:
            print("doc/examples.md is stale; run: python scripts/render_examples_doc.py")
            return 1
        return 0

    if current != rendered:
        OUTPUT_PATH.write_text(rendered, "utf-8")
        print(f"Updated {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

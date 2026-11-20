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

import re
import os
from pathlib import Path

INCLUDE_MARKER = re.compile(r"<!--\s*include:\s*(.+?)\s*-->", re.MULTILINE)
INCLUDE_BLOCK = re.compile(
    r"(<!--\s*include:\s*(.+?)\s*-->[ \t]*\n(?:[ \t]*\n)*)(```[\w:.-]*\n)([\s\S]*?)(^```[ \t]*$)",
    re.MULTILINE,
)

_HEADER_END = "# POSSIBILITY OF SUCH DAMAGE.\n"


def _project_header() -> str:
    """The project licence header: the comment block that opens this file."""
    text = Path(__file__).read_text("utf-8")
    header = text[: text.index(_HEADER_END) + len(_HEADER_END)]
    if not header.startswith("# Copyright (C) ") or any(not line.startswith("#") for line in header.splitlines()):
        raise ValueError(f"{__file__} does not open with the project licence header")
    return header


PROJECT_HEADER = _project_header()


def strip_project_header(content: str, source: Path) -> str:
    """Remove the project licence header and the blank line after it.

    The README listings show the code without the licence header that every
    shipped Python file carries; the rest of the file is listed unchanged.
    """
    count = content.count(PROJECT_HEADER)
    if count == 0:
        return content
    start = content.index(PROJECT_HEADER)
    end = start + len(PROJECT_HEADER)
    if count > 1 or (start and content[start - 1] != "\n") or content[end : end + 1] != "\n":
        raise ValueError(f"{source}: expected one project licence header at the start of a line, followed by a blank line")
    return content[:start] + content[end + 1 :]


def included_text(readme: Path, spec: str) -> str:
    """Read a file, or a named docs region of it, without importing its code.

    The project licence header is not part of the included text.
    """
    filename, separator, region = spec.partition("#")
    source = readme.parent / filename
    content = strip_project_header(source.read_text("utf-8"), source)
    if separator:
        begin = f"# docs: {region}-begin\n"
        end = f"# docs: {region}-end"
        if content.count(begin) != 1 or content.count(end) != 1:
            raise ValueError(f"{source}: expected one begin/end pair for {region}")
        content = content.split(begin, 1)[1].split(end, 1)[0]
    return content.rstrip() + "\n"


def sync_readme(readme_path):
    """Reads a README file, looks for <!-- include: filepath --> markers,
    and updates the following code block with the content of the file.
    """
    try:
        with open(readme_path) as f:
            content = f.read()
    except FileNotFoundError:
        print(f"File not found: {readme_path}")
        return

    # Regex to find the marker and the following code block
    # Matches:
    # 1. <!-- include: path/to/file --> (with optional whitespace)
    # 2. The opening fence of the code block (e.g. ```python)
    # 3. The content inside the code block (non-greedy)
    # 4. The closing fence (```)
    pattern = INCLUDE_BLOCK
    for marker in INCLUDE_MARKER.finditer(content):
        if pattern.match(content, marker.start()) is None:
            raise ValueError(f"{readme_path}: include {marker.group(1)} has no fenced block")

    def replace_chunk(match):
        marker_line = match.group(1)
        rel_path = match.group(2).strip()
        opening_fence = match.group(3)
        closing_fence = match.group(5)

        # Resolve path relative to README location
        readme_dir = os.path.dirname(readme_path)
        source_path = os.path.join(readme_dir, rel_path.partition("#")[0])

        if not os.path.exists(source_path):
            print(f"Warning: Source file not found: {source_path} (referenced in {readme_path})")
            return match.group(0)  # Return original text if file not found

        try:
            new_content = included_text(Path(readme_path), rel_path)
        except Exception as e:
            print(f"Error reading {source_path}: {e}")
            return match.group(0)

        return f"{marker_line}{opening_fence}{new_content}{closing_fence}"

    new_content = pattern.sub(replace_chunk, content)

    if new_content != content:
        with open(readme_path, "w") as f:
            f.write(new_content)
        print(f"Updated {readme_path}")
    else:
        print(f"No changes needed for {readme_path}")


if __name__ == "__main__":
    # Find all README.md in examples/
    # You can adjust the root path as needed
    root = Path("examples")
    if not root.exists():
        # Fallback if running from a different directory
        root = Path(".")

    print("Scanning for README.md files...")
    for readme in root.rglob("README.md"):
        sync_readme(readme)

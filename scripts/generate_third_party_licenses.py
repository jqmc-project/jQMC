"""Write LICENSES/THIRD_PARTY_RUST_CRATES.txt from the Rust crate's resolved dependencies.

The output is plain text (not Markdown) because the licence texts are copied
verbatim, including their non-ASCII characters.

The jqmc_rs extension links its normal (non-dev, non-build) Rust dependencies
statically, so the wheel carries their code. This lists every such crate with
its version, declared licence and repository, followed by the licence and
notice files that the crate itself ships. It reads ``cargo metadata`` (offline,
from the local cargo registry), so run it after the dependencies are fetched:

    python scripts/generate_third_party_licenses.py          # write
    python scripts/generate_third_party_licenses.py --check  # fail if stale
"""

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
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRATE_DIR = ROOT / "jqmc" / "rs"
OUTPUT = ROOT / "LICENSES" / "THIRD_PARTY_RUST_CRATES.txt"
LICENSE_FILE_PREFIXES = ("LICENSE", "LICENCE", "COPYING", "COPYRIGHT", "NOTICE", "UNLICENSE")


def _metadata() -> dict:
    result = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--offline", "--locked"],
        cwd=CRATE_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _linked_packages(metadata: dict) -> list[dict]:
    """Packages reachable from the root through normal dependency edges, root excluded."""
    packages = {package["id"]: package for package in metadata["packages"]}
    nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
    root = metadata["resolve"]["root"]
    seen: set[str] = set()
    stack = [root]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        for dep in nodes[node_id]["deps"]:
            if any(kind["kind"] is None for kind in dep["dep_kinds"]):
                stack.append(dep["pkg"])
    seen.discard(root)
    return sorted((packages[node_id] for node_id in seen), key=lambda p: (p["name"], p["version"]))


def _license_files(package: dict) -> list[Path]:
    """Every licence or notice file anywhere in the crate source, e.g. of vendored C code."""
    directory = Path(package["manifest_path"]).parent
    files = [
        path for path in sorted(directory.rglob("*")) if path.is_file() and path.name.upper().startswith(LICENSE_FILE_PREFIXES)
    ]
    if package.get("license_file"):
        declared = directory / package["license_file"]
        if declared.is_file() and declared not in files:
            files.append(declared)
    return files


def render() -> str:
    packages = _linked_packages(_metadata())
    lines = [
        "# Third-party Rust crates linked into jqmc_rs",
        "",
        "The `jqmc_rs` extension of jQMC statically links Rust crates from the list",
        "below. The list is a conservative cross-target inventory: the normal",
        "dependencies of `jqmc/rs/Cargo.lock` for every target, so it also contains",
        "procedural macros (used at build time) and crates for platforms a given wheel",
        "does not target. Each crate is distributed under its own licence, given here",
        "as declared by the crate, followed by every licence and notice file in the",
        "crate's source, with its path relative to the crate.",
        "",
        "Generated by `scripts/generate_third_party_licenses.py` from `jqmc/rs/Cargo.lock`;",
        "do not edit by hand.",
        "",
        "| Crate | Version | Licence | Repository |",
        "|---|---|---|---|",
    ]
    for package in packages:
        licence = package.get("license") or "see licence file"
        lines.append(f"| {package['name']} | {package['version']} | {licence} | {package.get('repository') or ''} |")
    # Identical licence texts (for example the Apache-2.0 text most crates ship)
    # are printed once, with every crate that ships them.
    texts: dict[str, list[str]] = {}
    without_file = []
    for package in packages:
        files = _license_files(package)
        if not files:
            without_file.append(f"{package['name']} {package['version']} ({package.get('license') or 'no licence declared'})")
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            relative = path.relative_to(Path(package["manifest_path"]).parent).as_posix()
            texts.setdefault(text, []).append(f"{package['name']} {package['version']} ({relative})")
    lines += ["", "## Crates that ship no licence file", ""]
    lines += [f"- {entry}" for entry in without_file] or ["- none"]
    for index, (text, owners) in enumerate(texts.items(), start=1):
        lines += ["", f"## Licence text {index}", "", "Shipped by: " + ", ".join(owners) + ".", "", "```text", text, "```"]
    return "\n".join(line.rstrip() for line in lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed file is not current")
    args = parser.parse_args()
    text = render()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != text:
            print(f"{OUTPUT.relative_to(ROOT)} is stale; run scripts/generate_third_party_licenses.py", file=sys.stderr)
            return 1
        return 0
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

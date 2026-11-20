#!/usr/bin/env python3
"""Generate the complete GDD reference pages for the ordinary Sphinx manual."""

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

import posixpath
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "doc/gdd_audit"
sys.path.insert(0, str(ROOT))

_LINK = re.compile(r"(\]\()([^\s)]+)(\))")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_HTML_ANCHOR = re.compile(r'^<a id="([^"]+)"></a>$')
_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
_SLUG_CLEAN = re.compile(r"[^\w\u4e00-\u9fff\- ]")


def _repository_url() -> str:
    """Use the checkout's ordinary repository links, including public releases."""
    try:
        remote = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "https://github.com/jqmc-project/jQMC"
    if remote.startswith("git@github.com:"):
        remote = "https://github.com/" + remote.split(":", 1)[1]
    parsed = urlsplit(remote)
    if parsed.hostname == "github.com":
        return "https://github.com/" + parsed.path.strip("/").removesuffix(".git")
    return "https://github.com/jqmc-project/jQMC"


def _page_name(name: str) -> str:
    return str(Path(name).with_name("index.md")) if Path(name).name == "README.md" else name


def _heading_text(markup: str) -> str:
    """Return the inline text MyST uses when it creates a heading slug."""
    markup = re.sub(r"!?\[([^]]*)\]\([^)]+\)", r"\1", markup)
    markup = re.sub(r"`+([^`]+?)`+", r"\1", markup)
    return markup.replace("*", "").replace("~", "")


def _page_anchor_slugs(content: str) -> tuple[dict[str, list[tuple[int, str]]], set[str]]:
    """Map generated HTML anchors to the following MyST heading slugs."""
    anchors: dict[str, list[tuple[int, str]]] = {}
    slugs: set[str] = set()
    pending: list[tuple[str, int]] = []
    fence = None
    offset = 0
    for line_with_end in content.splitlines(keepends=True):
        line = line_with_end.rstrip("\r\n")
        marker = _FENCE.match(line)
        if fence is not None:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
            offset += len(line_with_end)
            continue
        if marker:
            fence = marker[1]
            offset += len(line_with_end)
            continue
        if anchor := _HTML_ANCHOR.fullmatch(line):
            pending.append((anchor[1], offset))
            offset += len(line_with_end)
            continue
        if heading := _HEADING.match(line):
            base = _SLUG_CLEAN.sub("", _heading_text(heading[2]).lower().replace(" ", "-"))
            slug = base
            number = 1
            while slug in slugs:
                slug = f"{slug}-{number}"
                number += 1
            slugs.add(slug)
            for name, position in pending:
                anchors.setdefault(name, []).append((position, slug))
            pending.clear()
        elif line.strip() and pending:
            raise ValueError(f"generated anchor(s) {pending!r} are not followed by a level 1-4 heading")
        offset += len(line_with_end)
    if pending:
        raise ValueError(f"generated anchor(s) {pending!r} have no following heading")
    return anchors, slugs


def _sphinx_fragment_links(outputs: dict[str, str]) -> dict[str, str]:
    """Point generated fragments at MyST heading IDs instead of raw HTML IDs."""
    page_anchors = {name: _page_anchor_slugs(content)[0] for name, content in outputs.items()}
    owners: dict[str, list[str]] = {}
    for name, anchors in page_anchors.items():
        for anchor in anchors:
            owners.setdefault(anchor, []).append(name)

    def owner_for(source: str, target: str, anchor: str) -> str | None:
        candidates = owners.get(anchor, [])
        if not candidates:
            return None
        target_path = Path(target)
        expected_parent = target_path.with_suffix("") if len(target_path.parts) == 2 else target_path.parent
        local = [name for name in candidates if Path(name).parent == expected_parent]
        if len(local) == 1:
            return local[0]
        return candidates[0] if len(candidates) == 1 else None

    rewritten: dict[str, str] = {}
    for name, content in outputs.items():
        line_offset = 0

        def replace(match: re.Match[str]) -> str:
            href = match[2]
            parsed = urlsplit(href)
            if parsed.scheme or parsed.netloc or not parsed.fragment:
                return match[0]
            target = posixpath.normpath(posixpath.join(posixpath.dirname(name), unquote(parsed.path))) if parsed.path else name
            owner = target if parsed.fragment in page_anchors.get(target, {}) else owner_for(name, target, parsed.fragment)
            if owner is None:
                return match[0]
            path = parsed.path if owner == target else posixpath.relpath(owner, posixpath.dirname(name))
            choices = page_anchors[owner][parsed.fragment]
            fragment = choices[0][1]
            if owner == name:
                link_position = line_offset + match.start()
                fragment = next((slug for position, slug in choices if position > link_position), fragment)
            return match[1] + urlunsplit(("", "", path, parsed.query, fragment)) + match[3]

        result = []
        fence = None
        for line in content.splitlines(keepends=True):
            marker = _FENCE.match(line)
            if fence is not None:
                result.append(line)
                if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                    fence = None
            elif marker:
                fence = marker[1]
                result.append(line)
            else:
                result.append(_LINK.sub(replace, line))
            line_offset += len(line)
        rewritten[name] = "".join(result)
    return rewritten


def _manual_links(content: str, source: str, destination: str, pages: dict[str, str], repository: str) -> str:
    """Keep audit references local and link repository files to their normal URLs."""

    def replace(match: re.Match[str]) -> str:
        href = match[2]
        parsed = urlsplit(href)
        if parsed.scheme or parsed.netloc or not parsed.path:
            return match[0]
        target = posixpath.normpath(posixpath.join(posixpath.dirname(source), unquote(parsed.path)))
        if target in pages:
            path = posixpath.relpath(pages[target], posixpath.dirname(destination))
            href = urlunsplit(("", "", path, parsed.query, parsed.fragment))
        else:
            kind = "tree" if parsed.path.endswith("/") or (ROOT / target).is_dir() else "blob"
            href = repository + f"/{kind}/main/" + quote(target, safe="/")
            if parsed.query:
                href += "?" + parsed.query
            if parsed.fragment:
                href += "#" + parsed.fragment
        return match[1] + href + match[3]

    result = []
    fence = None
    for line in content.splitlines(keepends=True):
        marker = _FENCE.match(line)
        if fence is not None:
            result.append(line)
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
        elif marker:
            fence = marker[1]
            result.append(line)
        else:
            result.append(_LINK.sub(replace, line))
    return "".join(result)


def write_manual_pages() -> list[Path]:
    """Refresh ignored manual sources from all complete, rich GDD renderings."""
    from gdd.engine.derived_cache import prepare_derived

    prepare_derived(ROOT, sys.executable)
    from gdd.engine.guarantee_store import load_module
    from scripts.gdd_audit_views import GENERATED_VIEW_ROOT
    from scripts.render_gdd_derivatives_ledger import CONTRACT_PATH, render
    from scripts.render_gdd_ssot_ledgers import render_all

    _local_views, rich_views = render_all(with_generated_views=True)
    source_pages = {f"{GENERATED_VIEW_ROOT.as_posix()}/{name}": _page_name(name) for name in rich_views}
    source_pages.update({f"gdd/ledger/{name}": _page_name(name) for name in rich_views})
    requirements = "derivatives-requirements.md"
    source_pages["gdd/derived/derivatives-requirements.md"] = requirements
    source_pages["gdd/ssot/guarantees/README.md"] = requirements
    contents = {_page_name(name): (f"{GENERATED_VIEW_ROOT.as_posix()}/{name}", text) for name, text in rich_views.items()}
    if len(contents) != len(rich_views) or requirements in contents:
        raise ValueError("GDD manual page names collide")
    contents[requirements] = ("gdd/ssot/guarantees/README.md", render(load_module(CONTRACT_PATH)))

    # Directory entry pages also make dossier-directory links ordinary Sphinx links.
    directories = {""}
    for name in contents:
        directories.update(parent.as_posix() for parent in Path(name).parents if parent.as_posix() != ".")
    for directory in directories:
        name = posixpath.join(directory, "index.md")
        if name not in contents:
            title = "GDD reference" if not directory else "GDD reference: " + directory.replace("/", " / ")
            contents[name] = ("gdd/derived/human-ledger/" + name, f"# {title}\n")
        for root in (GENERATED_VIEW_ROOT.as_posix(), "gdd/ledger"):
            source_pages[posixpath.normpath(posixpath.join(root, directory))] = name
            source_pages[posixpath.join(root, directory, "index.md")] = name

    repository = _repository_url()
    outputs = {}
    for name, (source, content) in contents.items():
        text = _manual_links(content, source, name, source_pages, repository)
        if Path(name).name == "index.md":
            directory = posixpath.dirname(name)
            children = sorted(
                posixpath.relpath(other.removesuffix(".md"), directory)
                for other in contents
                if other != name
                and (
                    posixpath.dirname(other) == directory
                    or (other.endswith("/index.md") and posixpath.dirname(posixpath.dirname(other)) == directory)
                )
            )
            if children:
                text += "\n```{toctree}\n:hidden:\n:maxdepth: 1\n\n" + "\n".join(children) + "\n```\n"
        outputs[name] = text

    outputs = _sphinx_fragment_links(outputs)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_ROOT.rglob("*.md"):
        if path.relative_to(OUTPUT_ROOT).as_posix() not in outputs:
            path.unlink()
    for name, text in outputs.items():
        path = OUTPUT_ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return [OUTPUT_ROOT / name for name in sorted(outputs)]


if __name__ == "__main__":
    print(f"Generated {len(write_manual_pages())} GDD manual pages")

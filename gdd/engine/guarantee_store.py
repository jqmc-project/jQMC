"""Single loader for the guarantee SSoT (one file per guarantee, issue 205).

Layout
------
``gdd/ssot/guarantees/<module>.yaml``
    The *module file*. It keeps every module-level section (``document``,
    ``enforcement_map``, ``symbol_contract``, ...) and replaces each guarantee
    section by a split marker::

        guarantees:
          layout: one-file-per-guarantee

``gdd/ssot/guarantees/<module>/<GUARANTEE-ID>.yaml``
    One guarantee of the ``guarantees`` section: the mapping of one list item
    of the assembled document.

``gdd/ssot/guarantees/<module>/<section>/<GUARANTEE-ID>.yaml``
    One row of another split section (only ``module_evidence`` exists).

The assembled module document is the mapping a single-file ledger would
parse to: same keys in the same order, each split section a list of guarantee
mappings sorted by guarantee id. New guarantees therefore never edit the
module file, which is what lets two branches that each add a guarantee to the
same module merge without conflict.

Every reader of guarantee ledgers goes through :func:`load_module` (working
tree) or :func:`load_module_at_ref` (a git revision). The logical ledger name
stays the module-file path, so references such as
``gdd/ssot/guarantees/jqmc_rs.yaml`` in other SSoT documents keep naming the
same ledger. Only the split layout is read: a module file whose guarantee
section is still a monolithic inline list is rejected with an error that
names the expected per-guarantee files (an empty list holds no guarantee and
is accepted).

The loader fails closed: a guarantee file whose name differs from its id, a
duplicate id (across files, sections or modules), a guarantee directory or
section directory without a declaring module file, unknown marker keys and
stray files are all errors.
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

import hashlib
import os
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
GUARANTEES_RELATIVE = "gdd/ssot/guarantees"
GUARANTEE_DIR = ROOT / GUARANTEES_RELATIVE
SPLIT_SECTIONS = ("guarantees", "module_evidence")
SPLIT_LAYOUT = "one-file-per-guarantee"
MARKER_KEYS = frozenset({"layout"})
# The module file is named after the module; README.md is a generated view.
NON_LEDGER_FILES = frozenset({"README.md"})

_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


class GuaranteeLayoutError(ValueError):
    """The split guarantee layout is inconsistent; never silently assembled."""


def _parse(label: str, text: str, loader: Any = None) -> Any:
    try:
        return yaml.load(text, Loader=loader or _LOADER)
    except yaml.YAMLError as exc:
        raise GuaranteeLayoutError(f"{label}: invalid YAML: {exc}") from exc


def module_name(module_relative: str) -> str:
    """``gdd/ssot/guarantees/jqmc_rs.yaml`` -> ``jqmc_rs``."""
    return PurePosixPath(module_relative).stem


def module_directory_relative(module_relative: str) -> str:
    path = PurePosixPath(module_relative)
    return (path.parent / path.stem).as_posix()


def section_directory_relative(module_relative: str, section: str) -> str:
    base = module_directory_relative(module_relative)
    return base if section == "guarantees" else f"{base}/{section}"


def guarantee_relative(module_relative: str, section: str, guarantee_id: str) -> str:
    """Repository-relative path of one split guarantee file."""
    return f"{section_directory_relative(module_relative, section)}/{guarantee_id}.yaml"


def guarantee_path(module_path: Path, section: str, guarantee_id: str) -> Path:
    directory = module_path.parent / module_path.stem
    if section != "guarantees":
        directory = directory / section
    return directory / f"{guarantee_id}.yaml"


def is_split_marker(value: Any) -> bool:
    return isinstance(value, dict) and value.get("layout") == SPLIT_LAYOUT


def _section_files(module_relative: str, files: Iterable[str]) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Group files under ``<module>/`` by section; return (sections, stray files)."""
    prefix = module_directory_relative(module_relative) + "/"
    sections: dict[str, dict[str, str]] = {}
    stray: list[str] = []
    for relative in sorted(files):
        if not relative.startswith(prefix):
            continue
        parts = relative[len(prefix) :].split("/")
        if len(parts) == 1:
            section, name = "guarantees", parts[0]
        elif len(parts) == 2 and parts[0] in SPLIT_SECTIONS and parts[0] != "guarantees":
            section, name = parts
        else:
            stray.append(relative)
            continue
        if not name.endswith(".yaml") or name == ".yaml":
            stray.append(relative)
            continue
        sections.setdefault(section, {})[name[: -len(".yaml")]] = relative
    return sections, stray


def assemble_module(
    module_relative: str,
    module_text: str,
    read: Any,
    files: Iterable[str],
    *,
    loader: Any = None,
) -> dict[str, Any]:
    """Assemble one module document.

    ``read(relative) -> str`` returns a file's text and ``files`` lists every
    repository-relative file below the module directory. ``loader`` is the
    YAML loader class (default ``CSafeLoader``); strict callers may pass a
    duplicate-key-rejecting loader, which then applies to every file.
    """
    return assemble_document(
        module_relative,
        _parse(module_relative, module_text, loader),
        lambda relative: _parse(relative, read(relative), loader),
        files,
    )


def assemble_document(
    module_relative: str,
    document: Any,
    load_row: Any,
    files: Iterable[str],
) -> dict[str, Any]:
    """Assemble from an already parsed module file; ``load_row(relative)`` parses one guarantee file."""
    if not isinstance(document, dict):
        raise GuaranteeLayoutError(f"{module_relative}: YAML document must be a mapping")
    sections, stray = _section_files(module_relative, files)
    if stray:
        raise GuaranteeLayoutError(f"{module_relative}: unexpected files in the guarantee directory: {stray}")
    assembled: dict[str, Any] = {}
    seen: dict[str, str] = {}
    for key, value in document.items():
        if key not in SPLIT_SECTIONS or not is_split_marker(value):
            if key in SPLIT_SECTIONS and isinstance(value, list) and value:
                raise GuaranteeLayoutError(
                    f"{module_relative}: section {key} is a monolithic inline list; every guarantee must be its own "
                    f"file {section_directory_relative(module_relative, key)}/<ID>.yaml and the module file must hold "
                    f"the split marker {{layout: {SPLIT_LAYOUT}}}"
                )
            if key in SPLIT_SECTIONS and sections.get(key):
                raise GuaranteeLayoutError(
                    f"{module_relative}: section {key} is not split but split guarantee files exist: "
                    f"{sorted(sections[key].values())}"
                )
            assembled[key] = value
            continue
        unknown = sorted(set(value) - MARKER_KEYS)
        if unknown:
            raise GuaranteeLayoutError(f"{module_relative}: unknown split-marker keys in {key}: {unknown}")
        available = sections.get(key, {})
        rows: list[dict[str, Any]] = []
        for guarantee_id in sorted(available):
            relative = available[guarantee_id]
            row = load_row(relative)
            if not isinstance(row, dict):
                raise GuaranteeLayoutError(f"{relative}: a guarantee file must hold one mapping")
            if row.get("id") != guarantee_id:
                raise GuaranteeLayoutError(f"{relative}: file name does not match its guarantee id {row.get('id')!r}")
            if guarantee_id in seen:
                raise GuaranteeLayoutError(f"{relative}: duplicate guarantee id {guarantee_id} (also {seen[guarantee_id]})")
            seen[guarantee_id] = relative
            rows.append(row)
        assembled[key] = rows
    orphaned = sorted(
        relative
        for section, rows in sections.items()
        if not is_split_marker(document.get(section))
        for relative in rows.values()
    )
    if orphaned:
        raise GuaranteeLayoutError(f"{module_relative}: guarantee files of an undeclared section: {orphaned}")
    return assembled


# --------------------------------------------------------------------------
# Working tree


def _absolute(path: Path) -> Path:
    """Absolute and normalized, without resolving symlinks (shadow roots link real directories)."""
    return Path(os.path.normpath(Path(path).absolute()))


def _relative(path: Path, root: Path) -> str:
    try:
        return _absolute(path).relative_to(_absolute(root)).as_posix()
    except ValueError:
        return path.as_posix()


def _module_root(module_path: Path) -> Path:
    """Infer the repository root of ``.../gdd/ssot/guarantees/<m>.yaml``."""
    parts = _absolute(module_path).parts
    if len(parts) >= 4 and parts[-4:-1] == ("gdd", "ssot", "guarantees"):
        return Path(*parts[:-4])
    return _absolute(module_path).parent


def module_files(module_path: Path) -> list[Path]:
    """The module file followed by every file below its guarantee directory (sorted)."""
    directory = module_path.parent / module_path.stem
    extra = sorted(path for path in directory.rglob("*") if path.is_file()) if directory.is_dir() else []
    return [module_path, *extra]


def module_texts(module_path: Path) -> tuple[tuple[str, str], ...]:
    """``(relative path, text)`` of every file that makes up one module ledger."""
    root = _module_root(module_path)
    return tuple((_relative(path, root), path.read_text(encoding="utf-8")) for path in module_files(module_path))


def assemble_texts(texts: tuple[tuple[str, str], ...], *, loader: Any = None) -> dict[str, Any]:
    (module_relative, module_text), *rest = texts
    by_path = dict(rest)
    return assemble_module(module_relative, module_text, by_path.__getitem__, by_path, loader=loader)


def load_module(module_path: str | Path, *, loader: Any = None) -> dict[str, Any]:
    """Return the assembled module document (the pre-split parse) from the working tree."""
    return assemble_texts(module_texts(Path(module_path)), loader=loader)


def load_ssot_document(path: str | Path, *, loader: Any = None) -> Any:
    """Parse any SSoT YAML file; a guarantee module ledger is assembled from its split files."""
    path = Path(path)
    if is_module_ledger(path):
        return load_module(path, loader=loader)
    return yaml.load(path.read_text(encoding="utf-8"), Loader=loader or _LOADER)


def module_sha256(module_path: Path) -> str:
    """Digest of the exact reviewed bytes of every file in one module ledger.

    A module without split files (a single-file or synthetic ledger) has the
    digest of its single file's bytes.
    """
    files = module_files(module_path)
    if len(files) == 1:
        return hashlib.sha256(files[0].read_bytes()).hexdigest()
    root = _module_root(module_path)
    digest = hashlib.sha256()
    for path in files:
        digest.update(_relative(path, root).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def module_paths(root: Path = ROOT) -> list[Path]:
    """Every module ledger file (``gdd/ssot/guarantees/*.yaml``), sorted."""
    return sorted((Path(root) / GUARANTEES_RELATIVE).glob("*.yaml"))


def iter_modules(root: Path = ROOT) -> Iterator[tuple[Path, dict[str, Any]]]:
    """``(module path, assembled document)`` for every module, after layout validation."""
    errors = layout_errors(root)
    if errors:
        raise GuaranteeLayoutError("; ".join(errors))
    for path in module_paths(root):
        yield path, load_module(path)


def is_split_guarantee_file(path: str | Path) -> bool:
    """True for a file below ``gdd/ssot/guarantees/<module>/`` (one guarantee, not a document)."""
    parts = _absolute(Path(path)).parts
    for index in range(len(parts) - 3):
        if parts[index : index + 3] == ("gdd", "ssot", "guarantees"):
            return len(parts) - index > 4
    return False


def is_module_ledger(path: str | Path) -> bool:
    """True for ``<root>/gdd/ssot/guarantees/<module>.yaml`` (not a split guarantee file)."""
    resolved = _absolute(Path(path))
    return resolved.suffix == ".yaml" and resolved.parts[-4:-1] == ("gdd", "ssot", "guarantees")


def layout_errors(root: Path = ROOT) -> list[str]:
    """Whole-layout validation (orphans, duplicates, names); empty when consistent."""
    root = Path(root)
    base = root / GUARANTEES_RELATIVE
    if not base.is_dir():
        return [f"{GUARANTEES_RELATIVE}: missing guarantee directory"]
    errors: list[str] = []
    modules = {path.stem for path in module_paths(root)}
    for child in sorted(base.iterdir()):
        if child.is_dir():
            if child.name not in modules:
                errors.append(f"{_relative(child, root)}: orphan guarantee directory without module file {child.name}.yaml")
        elif child.suffix != ".yaml" and child.name not in NON_LEDGER_FILES:
            errors.append(f"{_relative(child, root)}: unexpected file in the guarantee directory")
    owners: dict[str, str] = {}
    for path in module_paths(root):
        relative = _relative(path, root)
        try:
            document = load_module(path)
        except (OSError, GuaranteeLayoutError) as exc:
            errors.append(str(exc))
            continue
        for section in SPLIT_SECTIONS:
            for row in document.get(section) or ():
                row_id = row.get("id") if isinstance(row, dict) else None
                if not isinstance(row_id, str):
                    continue
                where = guarantee_relative(relative, section, row_id)
                if row_id in owners:
                    errors.append(f"duplicate guarantee id {row_id}: {owners[row_id]} and {where}")
                else:
                    owners[row_id] = where
    return errors


# --------------------------------------------------------------------------
# Git revisions (trusted-base comparisons)


class _GitTree:
    def __init__(self, ref: str, cwd: Path) -> None:
        self.ref = ref
        self.cwd = Path(cwd)
        listed = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", ref, "--", GUARANTEES_RELATIVE],
            cwd=self.cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        if listed.returncode:
            raise GuaranteeLayoutError(listed.stderr.strip() or f"cannot list {GUARANTEES_RELATIVE} at {ref}")
        self.files = listed.stdout.splitlines()
        self._cache: dict[str, str] = {}

    def module_relatives(self) -> list[str]:
        prefix = GUARANTEES_RELATIVE + "/"
        return sorted(
            path for path in self.files if path.startswith(prefix) and "/" not in path[len(prefix) :] and path.endswith(".yaml")
        )

    def read(self, relative: str) -> str:
        if relative not in self._cache:
            shown = subprocess.run(
                ["git", "show", f"{self.ref}:{relative}"], cwd=self.cwd, text=True, capture_output=True, check=False
            )
            if shown.returncode:
                raise GuaranteeLayoutError(shown.stderr.strip() or f"cannot read {self.ref}:{relative}")
            self._cache[relative] = shown.stdout
        return self._cache[relative]

    def prefetch(self, relatives: Iterable[str]) -> None:
        wanted = [relative for relative in relatives if relative not in self._cache]
        if not wanted:
            return
        batch = subprocess.run(
            ["git", "cat-file", "--batch"],
            cwd=self.cwd,
            input="".join(f"{self.ref}:{relative}\n" for relative in wanted).encode(),
            capture_output=True,
            check=False,
        )
        if batch.returncode:
            return
        data = batch.stdout
        position = 0
        for relative in wanted:
            end = data.index(b"\n", position)
            header = data[position:end].split()
            position = end + 1
            if len(header) != 3 or header[1] != b"blob":
                continue
            size = int(header[2])
            self._cache[relative] = data[position : position + size].decode("utf-8")
            position += size + 1


_TREES: dict[tuple[str, str], _GitTree] = {}


def _tree(ref: str, cwd: Path) -> _GitTree:
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=cwd, text=True, capture_output=True, check=False
    )
    if resolved.returncode:
        raise GuaranteeLayoutError(resolved.stderr.strip() or f"cannot resolve {ref}")
    commit = resolved.stdout.strip()
    key = (commit, str(Path(cwd).resolve()))
    if key not in _TREES:
        _TREES[key] = _GitTree(commit, cwd)
    return _TREES[key]


def module_relatives_at_ref(ref: str, cwd: Path = ROOT) -> list[str]:
    """Module ledger paths (not split guarantee files) present at ``ref``."""
    return _tree(ref, cwd).module_relatives()


def load_module_at_ref(ref: str, module_relative: str, cwd: Path = ROOT, *, loader: Any = None) -> dict[str, Any] | None:
    """Assembled module document at a git revision; ``None`` when the module is absent there."""
    tree = _tree(ref, cwd)
    if module_relative not in tree.files:
        return None
    prefix = module_directory_relative(module_relative) + "/"
    files = [path for path in tree.files if path.startswith(prefix)]
    tree.prefetch([module_relative, *files])
    return assemble_module(module_relative, tree.read(module_relative), tree.read, files, loader=loader)

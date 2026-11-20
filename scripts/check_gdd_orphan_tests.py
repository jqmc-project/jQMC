#!/usr/bin/env python3
"""Static CI gate: every test function is registered in a GDD ledger or declared.

The orphan-test contract closes the last anti-vacuity hole of the module
ledgers: a test that exists but is registered nowhere proves nothing to the
contract, and a contract obligation whose test silently disappears must turn
the gate red. Every collected test function under ``tests/`` must therefore be

- registered as evidence in some ledger under ``gdd/ssot/guarantees/``, or
- inside a declared exempt root (out-of-scope packages), or
- inside a declared exempt file (derivative-contract evidence hosts and the
  GDD framework self-tests), or
- explicitly listed in ``pending_unregistered`` (the reviewed shrink-only
  backlog).

Stale declarations are themselves errors: a pending entry that no longer
exists, or that became registered, must be removed from the registry.
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

import ast
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REGISTRY = ROOT / "gdd/ssot/test_registry.yaml"
GUARANTEES = ROOT / "gdd/ssot/guarantees"
TEST_ROOTS = ("tests",)


def _registered_functions() -> set[tuple[str, str]]:
    registered: set[tuple[str, str]] = set()

    def harvest(value: Any) -> None:
        if isinstance(value, dict):
            node = value.get("pytest_node")
            if isinstance(node, str):
                parts = node.split("::")
                registered.add((parts[0], parts[-1]))
            for child in value.values():
                harvest(child)
        elif isinstance(value, list):
            for child in value:
                harvest(child)

    from gdd.engine.guarantee_store import load_module

    for path in sorted(GUARANTEES.glob("*.yaml")):
        harvest(load_module(path))
    return registered


def _collected_functions() -> dict[str, list[str]]:
    collected: dict[str, list[str]] = {}
    for root in TEST_ROOTS:
        for path in sorted((ROOT / root).rglob("test_*.py")):
            rel = path.relative_to(ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            names: list[str] = []

            def walk(body: list[ast.stmt]) -> None:
                for node in body:
                    if isinstance(node, ast.ClassDef):
                        walk(node.body)
                    elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                        names.append(node.name)

            walk(tree.body)
            if names:
                collected[rel] = names
    return collected


def qualified_test_functions(source: str, relative: str) -> list[str]:
    """Static definition identities, not expanded pytest parameter cases."""
    tree = ast.parse(source, filename=relative)
    names: list[str] = []

    def walk(body: list[ast.stmt], chain: list[str]) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                walk(node.body, chain + [node.name])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                names.append("::".join([relative] + chain + [node.name]))

    walk(tree.body, [])
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate test definition identity in {relative}")
    return sorted(names)


# --- Lane / CWD classification, enforced STATICALLY -------------------------
#
# tests/conftest.py already refuses an unclassified test, but only at COLLECTION
# time, and under xdist that refusal kills the worker: the failure then surfaces
# as ``INTERNALERROR> assert not crashitem`` naming whatever unrelated item that
# worker happened to hold. One unmarked test can therefore stop every light
# test of its module from running in ANY CI lane while the red job reads as a
# resource crash -- and if one of those tests is registered GDD evidence, the
# obligation is nominally covered by a test that cannot execute.
#
# This check enforces the same rule before commit instead of at runtime.
_TIER_MARKERS = ("light", "heavy")
_CWD_MARKERS = ("with_chdir", "without_chdir")


def _mark_names(nodes: list[ast.expr]) -> set[str]:
    """Marker names in a decorator list / pytestmark value (``pytest.mark.X``)."""
    names: set[str] = set()
    for node in nodes:
        current = node.func if isinstance(node, ast.Call) else node
        parts: list[str] = []
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        parts.reverse()
        if len(parts) >= 3 and parts[0] == "pytest" and parts[1] == "mark":
            names.add(parts[2])
    return names


def _pytestmark_names(body: list[ast.stmt]) -> set[str]:
    """Marker names contributed by a module- or class-level ``pytestmark``."""
    names: set[str] = set()
    for node in body:
        targets = node.targets if isinstance(node, ast.Assign) else ([node.target] if isinstance(node, ast.AnnAssign) else [])
        if not any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in targets):
            continue
        value = node.value
        if value is None:
            continue
        names |= _mark_names(list(value.elts) if isinstance(value, (ast.List, ast.Tuple)) else [value])
    return names


def _defers_to_parametrize_rows(decorators: list[ast.expr]) -> bool:
    """True when the tier may be supplied per ``pytest.param`` row.

    conftest allows the tier to come from ``pytest.param(..., marks=...)``.
    Which rows carry which marker is not always a static question: the argvalues
    are frequently a module-level table (``@parametrize("a, b", _PARAMS)``), and
    the marks then live inside that variable. Treat any such case as
    unresolvable and leave it to the runtime check -- this gate must never
    produce a false positive, because its whole purpose is to be trusted enough
    to block a commit.
    """
    for node in decorators:
        if not isinstance(node, ast.Call) or "parametrize" not in _mark_names([node.func]) | {getattr(node.func, "attr", "")}:
            continue
        for argvalue in node.args[1:]:
            if not isinstance(argvalue, (ast.List, ast.Tuple)):
                return True  # a table we cannot read: rows may carry marks
        if "marks" in ast.dump(node):
            return True
    return False


def _classification_errors() -> list[str]:
    errors: list[str] = []
    for root in TEST_ROOTS:
        for path in sorted((ROOT / root).rglob("test_*.py")):
            rel = path.relative_to(ROOT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            module_marks = _pytestmark_names(tree.body)

            def walk(body: list[ast.stmt], inherited: set[str], chain: list[str]) -> None:
                for node in body:
                    if isinstance(node, ast.ClassDef):
                        walk(
                            node.body,
                            inherited | _pytestmark_names(node.body) | _mark_names(node.decorator_list),
                            chain + [node.name],
                        )
                    elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                        if _defers_to_parametrize_rows(node.decorator_list):
                            continue
                        marks = inherited | _mark_names(node.decorator_list)
                        node_id = "::".join([rel] + chain + [node.name])
                        if not marks & set(_TIER_MARKERS):
                            errors.append(f"missing light/heavy marker: {node_id}")
                        elif set(_TIER_MARKERS) <= marks:
                            errors.append(f"carries BOTH light and heavy (ambiguous tier): {node_id}")
                        if not marks & set(_CWD_MARKERS):
                            errors.append(f"missing with_chdir/without_chdir marker: {node_id}")

            walk(tree.body, module_marks, [])
    return errors


def check() -> list[str]:
    errors: list[str] = _classification_errors()
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    exempt_roots = [item["root"] for item in registry.get("exempt_roots", [])]
    exempt_files = {item["path"] for item in registry.get("exempt_files", [])}
    pending: set[tuple[str, str]] = set()
    for item in registry.get("pending_unregistered", []):
        for name in item.get("functions", []):
            pending.add((item["path"], name))

    registered = _registered_functions()
    collected = _collected_functions()
    collected_pairs = {(rel, name) for rel, names in collected.items() for name in names}

    for path in exempt_files:
        if not (ROOT / path).is_file():
            errors.append(f"exempt file does not exist: {path}")
        if any((path, name) in registered for name in collected.get(path, [])):
            registered_here = sorted(name for name in collected.get(path, []) if (path, name) in registered)
            errors.append(f"exempt file has registered evidence (move it out of exempt_files): {path}: {registered_here[:3]}")
    if len(exempt_files) != len(registry.get("exempt_files", [])):
        errors.append("duplicate exempt file declaration")
    for pair in sorted(pending):
        if pair not in collected_pairs:
            errors.append(f"stale pending entry (test no longer exists): {pair[0]}::{pair[1]}")
        if pair in registered:
            errors.append(f"stale pending entry (test is now registered): {pair[0]}::{pair[1]}")

    orphans: list[str] = []
    for rel, names in sorted(collected.items()):
        if any(rel == root or rel.startswith(root + "/") for root in exempt_roots):
            continue
        if rel in exempt_files:
            continue
        for name in names:
            if (rel, name) in registered or (rel, name) in pending:
                continue
            orphans.append(f"{rel}::{name}")
    if orphans:
        errors.append(f"orphan tests (register in a ledger or declare in the registry): {len(orphans)}")
        errors.extend(f"  orphan: {node}" for node in orphans[:40])
    return errors


def main() -> int:
    errors = check()
    if errors:
        print("GDD orphan-test check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    registry = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    pending_total = sum(len(item.get("functions", [])) for item in registry.get("pending_unregistered", []))
    print(
        "GDD orphan-test check passed "
        f"({len(_registered_functions())} registered functions, {pending_total} reviewed pending, 0 orphans)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

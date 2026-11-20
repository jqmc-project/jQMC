"""Static discovery of derivative production boundaries.

The guarantee SSoT declares the expected boundary.  This module independently
observes source references and direct autodiff transforms; it never generates or
shrinks the scientific obligation universe.
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
import importlib.util
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


DERIVATIVE_PACKAGE = "jqmc.derivatives"
AUTODIFF_TRANSFORMS = frozenset(
    {
        "jax.grad",
        "jax.hessian",
        "jax.jacfwd",
        "jax.jacrev",
        "jax.jvp",
        "jax.linearize",
        "jax.value_and_grad",
        "jax.vjp",
    }
)


@dataclass(frozen=True, order=True)
class SourceLocation:
    """One source observation used for diagnostics, not contract identity."""

    path: str
    scope: str
    line: int


@dataclass(frozen=True, order=True)
class AutodiffCallsite:
    """Stable semantic identity of one direct autodiff transform in a consumer."""

    path: str
    scope: str
    transform: str
    operand: str

    @property
    def id(self) -> str:
        return f"{self.path}::{self.scope}::{self.transform}::{self.operand}"


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Lambda):
        return "<lambda>"
    return "<dynamic>"


def _module_name(root: Path, path: Path) -> tuple[str, str]:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    module = ".".join(parts)
    package = module if is_package else module.rpartition(".")[0]
    return module, package


def _known_derivative_modules(root: Path) -> set[str]:
    modules = {DERIVATIVE_PACKAGE}
    package_root = root / "jqmc/derivatives"
    for path in package_root.rglob("*.py"):
        module, _package = _module_name(root, path)
        modules.add(module)
    return modules


def _resolve_from_import(package: str, level: int, module: str | None) -> str:
    if level == 0:
        return module or ""
    relative = "." * level + (module or "")
    return importlib.util.resolve_name(relative, package)


def _import_aliases(tree: ast.Module, package: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                local = item.asname or item.name.split(".", 1)[0]
                target = item.name if item.asname else item.name.split(".", 1)[0]
                previous = aliases.setdefault(local, target)
                if previous != target:
                    raise ValueError(f"ambiguous import alias {local}: {previous} vs {target}")
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from_import(package, node.level, node.module)
            for item in node.names:
                if item.name == "*":
                    raise ValueError(f"wildcard import cannot be audited at line {node.lineno}")
                local = item.asname or item.name
                target = f"{base}.{item.name}" if base else item.name
                previous = aliases.setdefault(local, target)
                if previous != target:
                    raise ValueError(f"ambiguous import alias {local}: {previous} vs {target}")
    return aliases


def _resolve_reference(name: str, aliases: dict[str, str]) -> str:
    head, separator, tail = name.partition(".")
    target = aliases.get(head)
    if target is None:
        return name
    return f"{target}.{tail}" if separator else target


def _scope_by_node(tree: ast.Module) -> dict[int, str]:
    scopes: dict[int, str] = {}

    def walk(node: ast.AST, stack: tuple[str, ...]) -> None:
        next_stack = stack
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            next_stack = (*stack, node.name)
        scopes[id(node)] = ".".join(next_stack) if next_stack else "<module>"
        for child in ast.iter_child_nodes(node):
            walk(child, next_stack)

    walk(tree, ())
    return scopes


def _production_source_paths(root: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted((root / "jqmc").rglob("*.py"))
        if "obsolete" not in path.relative_to(root).parts and "derivatives" not in path.relative_to(root).parts[1:2]
    )


def discover_external_derivative_references(root: Path) -> dict[str, tuple[SourceLocation, ...]]:
    """Find every source reference crossing into ``jqmc.derivatives``.

    All production Python sources are scanned except the derivative package itself
    and the explicitly obsolete tree.  Module imports are plumbing and are not
    returned; every referenced symbol below that module boundary is returned.
    """

    known_modules = _known_derivative_modules(root)
    observations: dict[str, list[SourceLocation]] = defaultdict(list)
    for path in _production_source_paths(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        _module, package = _module_name(root, path)
        aliases = _import_aliases(tree, package)
        scopes = _scope_by_node(tree)
        parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

        for node in ast.walk(tree):
            raw = ""
            if isinstance(node, ast.Attribute):
                parent = parents.get(id(node))
                if isinstance(parent, ast.Attribute) and parent.value is node:
                    continue
                raw = _dotted_name(node)
            elif isinstance(node, ast.Name) and node.id in aliases:
                parent = parents.get(id(node))
                if isinstance(parent, ast.Attribute) and parent.value is node:
                    continue
                raw = node.id
            if not raw:
                continue
            target = _resolve_reference(raw, aliases)
            if not target.startswith(f"{DERIVATIVE_PACKAGE}.") or target in known_modules:
                continue
            relative = path.relative_to(root).as_posix()
            observations[target].append(SourceLocation(relative, scopes[id(node)], node.lineno))

    return {target: tuple(sorted(set(locations))) for target, locations in sorted(observations.items())}


def discover_direct_autodiff_calls(root: Path, consumer_paths: tuple[str, ...]) -> tuple[AutodiffCallsite, ...]:
    """Find direct JAX differentiation transforms in declared consumers."""

    discovered: list[AutodiffCallsite] = []
    for relative in consumer_paths:
        path = root / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        _module, package = _module_name(root, path)
        aliases = _import_aliases(tree, package)
        scopes = _scope_by_node(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            transform = _resolve_reference(_dotted_name(node.func), aliases)
            if transform not in AUTODIFF_TRANSFORMS:
                continue
            operand = _dotted_name(node.args[0]) if node.args else "<missing>"
            discovered.append(
                AutodiffCallsite(
                    path=relative,
                    scope=scopes[id(node)],
                    transform=transform,
                    operand=operand,
                )
            )
    ids = [item.id for item in discovered]
    if len(ids) != len(set(ids)):
        duplicates = sorted(item for item in set(ids) if ids.count(item) > 1)
        raise ValueError(f"direct autodiff callsite identities are not unique: {duplicates}")
    return tuple(sorted(discovered))

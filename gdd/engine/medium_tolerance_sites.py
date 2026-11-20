"""Static inventory of test call sites that can select the ``medium`` XLA-Rust tier.

``jqmc/_setting.py`` states that the ``medium`` backend-consistency envelope
(fp32 atol 1e-4 / rtol 2e-3) is a temporary exception for amplified
force/stress/parameter-derivative aggregates and deterministic MCMC/GFMC
end-to-end parity, and MUST NOT be used by ordinary leaf-kernel parity.
This module finds every call of
``get_tolerance_xla_rust_consistency`` under ``tests/`` (including import
aliases and invocations of nested ``functools.partial`` bindings) whose
effective ``level`` argument can be ``"medium"``. Partial constructors alone
are not invocations; an explicit call-site keyword overrides its partial
default. A level or argument pack that cannot be resolved is medium-capable
(fail closed), as are calls through known aliases with opaque bindings.

Sites are keyed by ``(path, enclosing function qualname)``; the reviewed
allow-list lives in ``gdd.engine.medium_tolerance_allowlist``.
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
from pathlib import Path

from gdd.engine.static_values import UNKNOWN, DictValue, ModuleResolver

RESOLVER = "get_tolerance_xla_rust_consistency"


def _resolver_aliases(tree: ast.Module) -> set[str]:
    names = {RESOLVER}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == RESOLVER and alias.asname:
                    names.add(alias.asname)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
                targets, value = [node.target], node.value
            else:
                continue
            if not _resolver_reference(value, names):
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in names:
                    names.add(target.id)
                    changed = True
    return names


def _resolver_reference(node: ast.AST | None, aliases: set[str]) -> bool:
    return (isinstance(node, ast.Name) and node.id in aliases) or (isinstance(node, ast.Attribute) and node.attr == RESOLVER)


class _ResolverCalls:
    """Follow lexical aliases and nested partials without executing test code.

    Each result is a sequence of partial constructors, innermost first. A
    conditional binding contributes all possibilities; unresolved bindings of
    a known resolver alias fail closed. Constructors alone are not call sites.
    """

    def __init__(self, resolver: ModuleResolver):
        self.resolver = resolver
        self.aliases = _resolver_aliases(resolver.tree)
        self.partial_names = {"partial"}
        self.partial_modules = {"functools"}
        for node in ast.walk(resolver.tree):
            if isinstance(node, ast.ImportFrom) and node.module == "functools":
                self.partial_names.update(a.asname or a.name for a in node.names if a.name == "partial")
            elif isinstance(node, ast.Import):
                self.partial_modules.update(a.asname or a.name for a in node.names if a.name == "functools")
        changed = True
        while changed:
            changed = False
            for node in ast.walk(resolver.tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) or node.value is None:
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                for target in targets:
                    if not isinstance(target, ast.Name):
                        continue
                    for names, matches in (
                        (self.aliases, self.is_resolver(value)),
                        (self.partial_names, self.is_partial(value)),
                        (self.partial_modules, isinstance(value, ast.Name) and value.id in self.partial_modules),
                    ):
                        if matches and target.id not in names:
                            names.add(target.id)
                            changed = True
        # Accessing a partial's public mutable keyword dictionary escapes this
        # bounded reader, including aliases of that dictionary. Fail closed.
        self.escaped_partials: set[ast.Call] = set()
        escaped = set()
        for node in ast.walk(resolver.tree):
            if isinstance(node, ast.Attribute) and node.attr == "keywords":
                chains = self.chains(node.value)
                if chains is not UNKNOWN:
                    escaped.update(partial for chain in chains for partial in chain)
        self.escaped_partials = escaped

    def is_resolver(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Call) and self.is_partial(node.func) and node.args:
            return self.is_resolver(node.args[0])
        if isinstance(node, ast.IfExp):
            return self.is_resolver(node.body) or self.is_resolver(node.orelse)
        return _resolver_reference(node, self.aliases)

    def is_partial(self, node: ast.AST) -> bool:
        return (isinstance(node, ast.Name) and node.id in self.partial_names) or (
            isinstance(node, ast.Attribute)
            and node.attr == "partial"
            and isinstance(node.value, ast.Name)
            and node.value.id in self.partial_modules
        )

    def _scope(self, node: ast.AST):
        parent = self.resolver.parents.get(node)
        while parent is not None:
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.Module)):
                return parent
            parent = self.resolver.parents.get(parent)
        return self.resolver.tree

    def chains(self, node: ast.AST, seen=frozenset()):
        if node in seen:
            return UNKNOWN
        seen = seen | {node}
        if isinstance(node, ast.Attribute) and node.attr == RESOLVER:
            return [()]
        if isinstance(node, ast.Call) and self.is_partial(node.func) and node.args:
            if node in self.escaped_partials:
                return UNKNOWN
            inner = self.chains(node.args[0], seen)
            return UNKNOWN if inner is UNKNOWN else [chain + (node,) for chain in inner]
        if isinstance(node, ast.IfExp):
            left, right = self.chains(node.body, seen), self.chains(node.orelse, seen)
            return UNKNOWN if left is UNKNOWN or right is UNKNOWN else left + right
        if not isinstance(node, ast.Name):
            return []
        if node.id not in self.aliases:
            return []
        scope = self._scope(node)
        use_scope = scope
        while scope is not None:
            candidates = []
            for statement in ast.walk(scope):
                if self._scope(statement) is not scope:
                    continue
                if scope is use_scope and (getattr(statement, "end_lineno", 0), getattr(statement, "end_col_offset", 0)) >= (
                    node.lineno,
                    node.col_offset,
                ):
                    continue
                if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                    if any(isinstance(t, ast.Name) and t.id == node.id for t in targets):
                        candidates.append(statement)
                elif isinstance(statement, (ast.Import, ast.ImportFrom)):
                    if any((a.asname or a.name.split(".")[0]) == node.id for a in statement.names):
                        candidates.append(statement)
            if candidates:
                candidates.sort(key=lambda item: item.lineno)
                # A straight-line assignment supersedes earlier bindings. Keep
                # conditional alternatives after it instead of guessing a branch.
                latest = 0
                for i, item in enumerate(candidates):
                    if self.resolver.parents.get(item) is scope:
                        latest = i
                out = []
                for item in candidates[latest:]:
                    if isinstance(item, ast.ImportFrom):
                        if any(a.name == RESOLVER and (a.asname or a.name) == node.id for a in item.names):
                            out.append(())
                        continue
                    if isinstance(item, ast.Import):
                        continue
                    if item.value is None:
                        return UNKNOWN if node.id in self.aliases else []
                    resolved = self.chains(item.value, seen)
                    if resolved is UNKNOWN or (not resolved and node.id in self.aliases):
                        return UNKNOWN
                    out.extend(resolved)
                return out
            if scope is self.resolver.tree:
                break
            if any(arg.arg == node.id for arg in ast.walk(scope.args) if isinstance(arg, ast.arg)):
                return UNKNOWN
            scope = self._scope(scope)
        return [()] if node.id in self.aliases else []


def _partial_levels(call: ast.Call, chain: tuple[ast.Call, ...], resolver: ModuleResolver):
    """Append positional arguments and override partial keyword defaults.

    A later explicit level overrides even an unresolved earlier keyword
    default. Unknown packs at the invocation remain medium-capable.
    """
    missing = object()
    levels = frozenset({missing})
    args = []
    for layer in (*chain, call):
        args.extend(layer.args[1:] if layer is not call else layer.args)
        supplied = set()
        unknown = False
        guaranteed = False
        for keyword in layer.keywords:
            if keyword.arg == "level":
                guaranteed = True
                values = resolver.values(keyword.value)
                if values is UNKNOWN:
                    unknown = True
                else:
                    supplied.update(values)
            elif keyword.arg is None:
                packs = _pack_values(keyword.value, resolver)
                if packs is UNKNOWN or any(not isinstance(pack, DictValue) for pack in packs):
                    unknown = True
                    continue
                pack_guaranteed = True
                for pack in packs:
                    try:
                        supplied.add(pack.lookup("level"))
                    except KeyError:
                        pack_guaranteed = False
                guaranteed |= pack_guaranteed
        if unknown:
            levels = UNKNOWN
        elif supplied:
            if not guaranteed:
                if levels is UNKNOWN:
                    continue
                supplied.update(levels)
            levels = frozenset(supplied)
    if levels is UNKNOWN:
        return UNKNOWN
    # Reuse the ordinary argument reader, retaining original AST parents for
    # name/parameter resolution. It fails closed for unknown positional packs.
    positional = _levels(ast.Call(func=call.func, args=args, keywords=[]), resolver)
    if positional is UNKNOWN:
        return UNKNOWN
    return (levels - {missing}) | (positional if missing in levels else frozenset())


def _pack_values(node: ast.AST, resolver: ModuleResolver):
    """Literal initializers cannot prove a pack's contents after mutation."""
    names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
    changed = True
    while changed:
        changed = False
        for statement in ast.walk(resolver.tree):
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)) or statement.value is None:
                continue
            if not isinstance(statement.value, ast.Name):
                continue
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            aliases = {target.id for target in targets if isinstance(target, ast.Name)} | {statement.value.id}
            if aliases & names and not aliases <= names:
                names.update(aliases)
                changed = True
    for statement in ast.walk(resolver.tree):
        if (
            isinstance(statement, ast.Subscript)
            and isinstance(statement.ctx, (ast.Store, ast.Del))
            and isinstance(statement.value, ast.Name)
            and statement.value.id in names
        ):
            return UNKNOWN
        if (
            isinstance(statement, ast.Call)
            and isinstance(statement.func, ast.Attribute)
            and isinstance(statement.func.value, ast.Name)
            and statement.func.value.id in names
            and statement.func.attr not in {"get", "items", "keys", "values", "copy", "count", "index"}
        ):
            return UNKNOWN
    return resolver.values(node)


def _levels(call: ast.Call, resolver: ModuleResolver):
    """Resolve the level argument, not the container of a ``**kwargs`` pack.

    Packs with unknown contents are medium-capable. In particular, a resolved
    dictionary is not itself a level string, and an empty pack must not erase
    an explicit ``level='medium'`` argument.
    """
    levels: set = set()
    if any(isinstance(arg, ast.Starred) for arg in call.args):
        positional = _pack_values(ast.Tuple(elts=call.args, ctx=ast.Load()), resolver)
        if positional is UNKNOWN:
            return UNKNOWN
        for arguments in positional:
            if len(arguments) >= 2:
                levels.add(arguments[1])
    elif len(call.args) >= 2:
        positional = resolver.values(call.args[1])
        if positional is UNKNOWN:
            return UNKNOWN
        levels.update(positional)
    for keyword in call.keywords:
        if keyword.arg == "level":
            values = resolver.values(keyword.value)
            if values is UNKNOWN:
                return UNKNOWN
            levels.update(values)
        elif keyword.arg is None:
            packs = _pack_values(keyword.value, resolver)
            if packs is UNKNOWN or any(not isinstance(pack, DictValue) for pack in packs):
                return UNKNOWN
            for pack in packs:
                try:
                    levels.add(pack.lookup("level"))
                except KeyError:
                    pass
    return frozenset(levels or {"strict"})


def _qualname(resolver: ModuleResolver, node: ast.AST) -> str:
    parts = []
    current = resolver.parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            parts.append(current.name)
        current = resolver.parents.get(current)
    return ".".join(reversed(parts)) or "<module>"


def source_medium_sites(source: str, relative: str) -> dict[tuple[str, str], list[int]]:
    """``(path, qualname) -> [line, ...]`` of medium-capable calls in one module."""
    tree = ast.parse(source, filename=relative)
    resolver = ModuleResolver(tree)
    calls = _ResolverCalls(resolver)
    sites: dict[tuple[str, str], list[int]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        chains = calls.chains(node.func)
        if chains is not UNKNOWN and not chains:
            continue
        candidates = (
            [UNKNOWN]
            if chains is UNKNOWN
            else [_partial_levels(node, chain, resolver) if chain else _levels(node, resolver) for chain in chains]
        )
        if all(values is not UNKNOWN and "medium" not in values for values in candidates):
            continue
        sites.setdefault((relative, _qualname(resolver, node)), []).append(node.lineno)
    return sites


def medium_sites(root: Path) -> dict[tuple[str, str], list[int]]:
    """Every medium-capable resolver call site under ``tests/``."""
    sites: dict[tuple[str, str], list[int]] = {}
    for path in sorted((root / "tests").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if RESOLVER not in source:
            continue
        relative = path.relative_to(root).as_posix()
        for key, lines in source_medium_sites(source, relative).items():
            sites.setdefault(key, []).extend(lines)
    return sites

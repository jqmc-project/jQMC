"""Protect source-level skip capabilities against an independently trusted base.

This is a change guard, not permission for an existing skip to close an
obligation. Runtime completion still requires every obligated case to pass.
No generated inventory can approve a new skip or a changed skip condition.
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
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json


_KINDS = {"skip", "skipif", "xfail", "importorskip", "skip_mixed_precision"}
_SCOPES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_CONTROLS = (
    ast.If,
    ast.While,
    ast.IfExp,
    ast.For,
    ast.AsyncFor,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.TryStar,
    ast.ExceptHandler,
    ast.Match,
    ast.match_case,
)


def _structure(node: object) -> str:
    """Interpreter-independent structural dump used for every skip digest.

    ``ast.dump`` is not stable across Python versions (3.13 omits None and
    empty-list fields; 3.12 added empty ``type_params``), so frozen digests
    hashed from it differ by interpreter. This keeps every node type, field
    name and value that ``ast.dump(include_attributes=False)`` keeps, and
    omits only fields that are None or an empty list on every version.
    """
    if isinstance(node, ast.AST):
        fields = (
            f"{name}={_structure(value)}"
            for name, value in ast.iter_fields(node)
            if value is not None and not (isinstance(value, list) and not value)
        )
        return f"{type(node).__name__}({', '.join(fields)})"
    if isinstance(node, list):
        return "[" + ", ".join(_structure(value) for value in node) + "]"
    return repr(node)


def skip_inventory(source: str) -> Counter[tuple[str, str, str]]:
    """Retain lexical alias/condition dependencies, not comments or line numbers.

    A conservative backward slice includes all assignments in a referenced
    lexical scope. Changing such a dependency requires review even if a full
    Python interpreter could prove that particular write irrelevant. Dynamic
    imports and arbitrary external factories are not evaluated or approved.
    """
    # The inventory is a pure function of the source text: identical texts
    # (for example an unchanged trusted-base and candidate file, or the
    # re-inventory of a reviewed new file) are analysed once.  Each caller
    # receives a fresh Counter.
    return Counter(dict(_skip_inventory_items(source)))


@lru_cache(maxsize=1024)
def _skip_inventory_items(source: str) -> tuple[tuple[tuple[str, str, str], int], ...]:
    return tuple(_skip_inventory(source).items())


def _skip_inventory(source: str) -> Counter[tuple[str, str, str]]:
    tree = ast.parse(source)
    parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    writes: dict[tuple[int, str], list[ast.AST | str]] = {}

    owners: dict[int, ast.AST | None] = {}

    def owner(node):
        # Memoized per node of this (immutable, alive) tree.
        identity = id(node)
        if identity in owners:
            return owners[identity]
        node = parents.get(identity)
        while node is not None and not isinstance(node, _SCOPES):
            node = parents.get(id(node))
        owners[identity] = node
        return node

    def bind(scope, name, value):
        writes.setdefault((id(scope), name), []).append(value)

    for node in ast.walk(tree):
        scope = owner(node)
        if isinstance(node, ast.Import):
            for alias in node.names:
                bind(scope, alias.asname or alias.name.split(".")[0], alias.name if alias.asname else alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bind(scope, alias.asname or alias.name, f"{'.' * node.level}{node.module or ''}.{alias.name}")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if isinstance(target, ast.Name):
                    bind(scope, target.id, node.value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bind(scope, node.name, node)

    def bindings(name, scope):
        while scope is not None:
            values = writes.get((id(scope), name))
            if values is not None:
                return values
            scope = owner(scope)
        return []

    def classes(node, scope, seen=frozenset()):
        """Resolve local class receivers without constructing or importing them."""
        key = ("class", id(node), id(scope))
        if key in seen:
            return []
        seen = seen | {key}
        if isinstance(node, ast.ClassDef):
            return [node]
        if isinstance(node, ast.Call):
            return classes(node.func, scope, seen)
        if not isinstance(node, ast.Name):
            return []
        # The receiver may be called something other than self/cls, or be
        # captured in a nested local function. Follow its defining method.
        enclosing = scope
        while enclosing is not None:
            if isinstance(enclosing, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = enclosing.args.posonlyargs + enclosing.args.args
                parent = owner(enclosing)
                if arguments and node.id == arguments[0].arg and isinstance(parent, ast.ClassDef):
                    return [parent]
            enclosing = owner(enclosing)
        return [
            cls
            for value in bindings(node.id, scope)
            if not isinstance(value, str)
            for cls in classes(value, owner(value), seen)
        ]

    def members(cls, attribute, seen=frozenset()):
        if id(cls) in seen:
            return []
        direct = writes.get((id(cls), attribute))
        if direct is not None:
            return direct
        return [
            member
            for base in cls.bases
            for parent in classes(base, owner(cls))
            for member in members(parent, attribute, seen | {id(cls)})
        ]

    # ``names`` re-walks a helper body for every reference to it.  A result
    # depends on ``seen`` only through the cycle keys its own traversal can
    # test, so it is memoized on (node, scope, seen restricted to a superset
    # of those keys).  ``checked_keys`` computes that superset by the same
    # traversal without pruning; ``classes``/``members`` never read ``seen``.
    reachable_keys: dict[tuple[int, int], frozenset] = {}
    function_names: dict[tuple[int, int, frozenset], frozenset] = {}

    def checked_keys(node, scope):
        identity = (id(node), id(scope))
        cached = reachable_keys.get(identity)
        if cached is not None:
            return cached
        keys = set()
        pending = [(node, scope)]
        visited = set()
        while pending:
            current, current_scope = pending.pop()
            state = (id(current), id(current_scope))
            if state in visited:
                continue
            visited.add(state)
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                keys.add(("function", id(current)))
                for child in ast.walk(current):
                    if isinstance(child, ast.Call):
                        pending.append((child.func, owner(child)))
                    elif isinstance(child, ast.Return) and child.value is not None:
                        pending.append((child.value, owner(child)))
                if isinstance(current, ast.Lambda):
                    pending.append((current.body, current_scope))
            elif isinstance(current, ast.Call):
                pending.append((current.func, current_scope))
                if current.args:
                    pending.append((current.args[0], current_scope))
            elif isinstance(current, ast.Attribute):
                pending.append((current.value, current_scope))
                for cls in classes(current.value, current_scope):
                    for member in members(cls, current.attr):
                        if not isinstance(member, str):
                            pending.append((member, owner(member)))
            elif isinstance(current, ast.Name):
                keys.add((id(current_scope), current.id))
                for value in bindings(current.id, current_scope):
                    if not isinstance(value, str):
                        pending.append((value, owner(value)))
        result = frozenset(keys)
        reachable_keys[identity] = result
        return result

    def names(node, scope, seen=frozenset()):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            memo_key = (id(node), id(scope), seen & checked_keys(node, scope))
            cached = function_names.get(memo_key)
            if cached is None:
                cached = frozenset(function_reference_names(node, scope, seen))
                function_names[memo_key] = cached
            return set(cached)
        return reference_names(node, scope, seen)

    def function_reference_names(node, scope, seen):
        # Calling an already-existing helper or marker factory is a new
        # capability use even if its definition did not change. Follow
        # local calls/returns conservatively; do not execute user code.
        key = ("function", id(node))
        if key in seen:
            return set()
        seen = seen | {key}
        result = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                result.update(names(child.func, owner(child), seen))
            elif isinstance(child, ast.Return) and child.value is not None:
                result.update(names(child.value, owner(child), seen))
        if isinstance(node, ast.Lambda):
            result.update(names(node.body, scope, seen))
        return result

    def reference_names(node, scope, seen):
        if isinstance(node, ast.Call):
            if names(node.func, scope, seen) & {"functools.partial"} and node.args:
                return names(node.args[0], scope, seen)
            return names(node.func, scope, seen)
        if isinstance(node, ast.Attribute):
            result = {f"{name}.{node.attr}" for name in names(node.value, scope, seen)}
            for cls in classes(node.value, scope):
                for member in members(cls, node.attr):
                    result.update({member} if isinstance(member, str) else names(member, owner(member), seen))
            return result
        if not isinstance(node, ast.Name):
            return set()
        key = (id(scope), node.id)
        if key in seen:
            return set()
        values = bindings(node.id, scope)
        return (
            {
                name
                for value in values
                for name in ({value} if isinstance(value, str) else names(value, owner(value), seen | {key}))
            }
            if values
            else {node.id}
        )

    def dependencies(node, seen=frozenset()):
        rows = []
        for child in ast.walk(node):
            if not isinstance(child, ast.Name) or not isinstance(child.ctx, ast.Load):
                continue
            scope = owner(child)
            key = (id(scope), child.id)
            if key in seen:
                continue
            for value in bindings(child.id, scope):
                if isinstance(value, str):
                    rows.append((child.id, value))
                else:
                    rows.append((child.id, _structure(value), dependencies(value, seen | {key})))
        return sorted(rows, key=repr)

    inventory = Counter()
    seen_sites = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Name, ast.Attribute)):
            continue
        scope = owner(node)
        resolved = names(node, scope)
        kinds = {
            name.rsplit(".", 1)[-1] for name in resolved if name.startswith("pytest.") and name.rsplit(".", 1)[-1] in _KINDS
        }
        if not kinds:
            continue
        parent = parents.get(id(node))
        site = parent if isinstance(parent, ast.Call) and parent.func is node else node
        controls, scopes = [], []
        child = site
        while (context := parents.get(id(child))) is not None:
            if isinstance(context, _CONTROLS):
                # The same predicate on the opposite arm has opposite meaning.
                # Retain loop/handler/match headers too, without pinning unrelated
                # statements in the body or their source locations.
                arm = next(
                    (
                        name
                        for name, value in ast.iter_fields(context)
                        if value is child or (isinstance(value, list) and child in value)
                    ),
                    "unknown",
                )
                header = []
                for field in ("test", "iter", "target", "type", "subject", "pattern", "guard", "items"):
                    value = getattr(context, field, None)
                    for part in value if isinstance(value, list) else [value]:
                        if isinstance(part, ast.AST):
                            header.append((field, _structure(part), dependencies(part)))
                controls.append((type(context).__name__, arm, header))
            elif isinstance(context, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                scopes.append(context.name)
            child = context
        qualified = ".".join(reversed(scopes)) or "<module>"
        payload = repr((_structure(site), dependencies(site), controls))
        digest = hashlib.sha256(payload.encode()).hexdigest()
        for kind in kinds:
            if (id(site), kind) not in seen_sites:
                seen_sites.add((id(site), kind))
                inventory[(qualified, kind, digest)] += 1
    return inventory


def skip_change_errors(before: str, after: str, path: str, *, baseline_exists: bool = True) -> list[str]:
    """Reject every added or changed skip capability or condition.

    Callers must prove absence in the baseline before passing False. The
    default is deliberately conservative, including for an empty source.
    """
    added = skip_inventory(after) - skip_inventory(before)
    return [
        f"{path}::{scope}: added or changed {kind} capability/condition requires explicit review ({count} site(s), {digest[:12]})"
        for (scope, kind, digest), count in sorted(added.items())
    ]


def precision_inventory(derived: dict) -> Counter[tuple[str, str, str, str]]:
    """Read the trusted/realized numeric case universe, independently of skip prose.

    This is not approval of a derived binding: freshness, source hashes and
    authored minima are still checked by the ordinary materializer/contract
    gates. It prevents re-generation from deleting a previously owed mode.
    """
    result = Counter()
    if not isinstance(derived, dict) or not isinstance(derived.get("guarantees"), dict):
        raise ValueError("precision protection requires a guarantee mapping")
    for guarantee, payload in derived["guarantees"].items():
        if not isinstance(guarantee, str) or not isinstance(payload, dict) or not isinstance(payload.get("nodes"), dict):
            raise ValueError("precision protection requires named node mappings")
        for node, entry in payload["nodes"].items():
            if not isinstance(node, str) or not isinstance(entry, dict) or not isinstance(entry.get("case_binding"), dict):
                raise ValueError("precision protection requires named case bindings")
            binding = entry["case_binding"]
            execution = binding["execution"]
            cases = binding["fixture"]["concrete_cases"]
            if not isinstance(execution, dict) or not isinstance(cases, list) or not cases:
                raise ValueError(f"{node}: precision protection requires concrete cases")
            for case in cases:
                if not isinstance(case, dict) or not isinstance(case.get("selector"), dict):
                    raise ValueError(f"{node}: precision protection requires case selectors")
                selector = json.dumps(case["selector"], sort_keys=True, allow_nan=False, separators=(",", ":"))
                modes = case.get("precision_modes", execution["precision_modes"])
                if not isinstance(modes, list) or not modes or len(set(modes)) != len(modes):
                    raise ValueError(f"{node}: malformed precision modes")
                for mode in modes:
                    if mode not in {"full", "mixed", "not-applicable"}:
                        raise ValueError(f"{node}: unknown precision mode {mode!r}")
                    if mode != "not-applicable":
                        result[(guarantee, node, selector, mode)] += 1
    return result


def precision_removal_errors(before: dict, after: dict, path: str) -> list[str]:
    """A marker reason, alternate guarantee or new pin cannot erase a trusted case/mode."""
    old, new = precision_inventory(before), precision_inventory(after)
    candidates = defaultdict(list)
    for (guarantee, node, selector, mode), count in new.items():
        candidates[(guarantee, node, mode)].extend([json.loads(selector)] * count)
    requests = defaultdict(list)
    for (guarantee, node, selector, mode), count in old.items():
        requests[(guarantee, node, mode)].extend([selector] * count)
    removed = Counter()
    for (guarantee, node, mode), selectors in requests.items():
        realized = candidates[(guarantee, node, mode)]
        # Adding an explicit axis for an already-implicit fixture setting is
        # strengthening, not deletion. Preserve every old tuple and mode with
        # an injective match; one realized case cannot cover duplicate demands.
        edges = []
        for encoded in selectors:
            required = json.loads(encoded)
            edges.append(
                [
                    index
                    for index, candidate in enumerate(realized)
                    if required.keys() <= candidate.keys()
                    and json.dumps(
                        {key: candidate[key] for key in required}, sort_keys=True, allow_nan=False, separators=(",", ":")
                    )
                    == encoded
                ]
            )
        assigned = {}

        def assign(request, seen):
            for candidate in edges[request]:
                if candidate in seen:
                    continue
                seen.add(candidate)
                if candidate not in assigned or assign(assigned[candidate], seen):
                    assigned[candidate] = request
                    return True
            return False

        for request, selector in enumerate(selectors):
            if not assign(request, set()):
                removed[(guarantee, node, selector, mode)] += 1
    return [
        f"{path}::{guarantee}::{node}: removed trusted {mode} obligation for {selector} ({count} case(s)); explicit review required"
        for (guarantee, node, selector, mode), count in sorted(removed.items())
    ]

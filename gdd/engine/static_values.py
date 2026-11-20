"""Bounded static value resolution for small literal-driven test expressions.

The GDD static gates sometimes need to know which literal values an
expression in a test can take -- for example which integer indices a
``counters[idx]`` subscript reads, or which tolerance level a
``level=level`` keyword passes. This module resolves such expressions from
the module source alone, without importing or executing the test.

It is deliberately narrow and fail-closed: anything it cannot follow (an
arbitrary call, an attribute of a runtime object, a name bound by an
unsupported statement) resolves to ``UNKNOWN``, and every caller treats
``UNKNOWN`` as "cannot be proved" rather than guessing. Supported forms are
constants, tuple/list/set/dict displays (with starred splices), names bound
by plain or tuple-unpacking assignment, ``for`` loops and comprehensions,
function parameters (pytest ``parametrize`` values for tests, argument
expressions at same-module call sites for helpers, and defaults), subscripts,
``.items()/.values()/.keys()``, ``range``, ``list/tuple/set``,
``pytest.param``, ``+ - *`` on integers and tuples, and conditional
expressions (both branches).
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
import itertools
from typing import Any


class _Unknown:
    def __repr__(self) -> str:
        return "UNKNOWN"


UNKNOWN = _Unknown()
_PRODUCT_LIMIT = 4096
_DEPTH_LIMIT = 24


class DictValue(tuple):
    """A hashable dict display: a tuple of ``(key, value)`` pairs."""

    def lookup(self, key: Any) -> Any:
        for item_key, item_value in self:
            if item_key == key:
                return item_value
        raise KeyError(key)


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _iterate(value: Any) -> list[Any] | _Unknown:
    if isinstance(value, DictValue):
        return [key for key, _ in value]
    if isinstance(value, (tuple, frozenset, str)):
        return list(value)
    return UNKNOWN


class ModuleResolver:
    """Resolve expressions inside one parsed module."""

    def __init__(self, tree: ast.Module) -> None:
        self.tree = tree
        self.functions: dict[str, ast.FunctionDef] = {}
        self.parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                self.parents[child] = node
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions[node.name] = node
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.functions[f"{node.name}.{item.name}"] = item

    # ------------------------------------------------------------------ scope
    def enclosing_function(self, node: ast.AST) -> ast.FunctionDef | None:
        current = self.parents.get(node)
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                return current if not isinstance(current, ast.Lambda) else self.enclosing_function(current)
            current = self.parents.get(current)
        return None

    def _bindings(self, name: str, scope: ast.AST) -> list[tuple[str, ast.AST, Any]]:
        """Binding sites of ``name`` directly owned by ``scope`` (not nested defs)."""
        found: list[tuple[str, ast.AST, Any]] = []

        def visit(node: ast.AST) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    continue
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        path = _target_path(target, name)
                        if path is not None:
                            found.append(("assign", child.value, path))
                elif isinstance(child, ast.AnnAssign) and child.value is not None:
                    path = _target_path(child.target, name)
                    if path is not None:
                        found.append(("assign", child.value, path))
                elif isinstance(child, (ast.AugAssign, ast.NamedExpr)):
                    target = child.target
                    if isinstance(target, ast.Name) and target.id == name:
                        found.append(("opaque", child, ()))
                elif isinstance(child, (ast.For, ast.AsyncFor)):
                    path = _target_path(child.target, name)
                    if path is not None:
                        found.append(("iterate", child.iter, path))
                elif isinstance(child, (ast.With, ast.AsyncWith)):
                    for item in child.items:
                        if item.optional_vars is not None and _target_path(item.optional_vars, name) is not None:
                            found.append(("opaque", child, ()))
                elif isinstance(child, (ast.Import, ast.ImportFrom)):
                    for alias in child.names:
                        if (alias.asname or alias.name.split(".")[0]) == name:
                            found.append(("opaque", child, ()))
                elif isinstance(child, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                    pass  # comprehension targets are resolved from the use site
                visit(child)

        visit(scope)
        return found

    # ------------------------------------------------------------- evaluation
    def values(self, node: ast.AST, depth: int = 0) -> frozenset | _Unknown:
        """All values ``node`` can take, or ``UNKNOWN``."""
        if depth > _DEPTH_LIMIT:
            return UNKNOWN
        if isinstance(node, ast.Constant):
            return frozenset([node.value])
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return self._sequence(node.elts, depth)
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                return UNKNOWN
            keys = [self.values(key, depth + 1) for key in node.keys]
            vals = [self.values(value, depth + 1) for value in node.values]
            if any(part is UNKNOWN for part in keys + vals):
                return UNKNOWN
            options = keys + vals
            if _product_size(options) > _PRODUCT_LIMIT:
                return UNKNOWN
            out = set()
            count = len(keys)
            for combo in itertools.product(*options):
                out.add(DictValue(zip(combo[:count], combo[count:])))
            return frozenset(out)
        if isinstance(node, ast.Name):
            return self._name(node, depth)
        if isinstance(node, ast.Subscript):
            bases = self.values(node.value, depth + 1)
            keys = self.values(node.slice, depth + 1)
            if bases is UNKNOWN or keys is UNKNOWN:
                return UNKNOWN
            out = set()
            for base in bases:
                for key in keys:
                    try:
                        if isinstance(base, DictValue):
                            out.add(base.lookup(key))
                        elif isinstance(base, tuple) and isinstance(key, int) and not isinstance(key, bool):
                            out.add(base[key])
                        else:
                            return UNKNOWN
                    except (KeyError, IndexError):
                        continue
            return frozenset(out)
        if isinstance(node, ast.IfExp):
            first = self.values(node.body, depth + 1)
            second = self.values(node.orelse, depth + 1)
            if first is UNKNOWN or second is UNKNOWN:
                return UNKNOWN
            return first | second
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            left = self.values(node.left, depth + 1)
            right = self.values(node.right, depth + 1)
            if left is UNKNOWN or right is UNKNOWN or len(left) * len(right) > _PRODUCT_LIMIT:
                return UNKNOWN
            out = set()
            for a, b in itertools.product(left, right):
                try:
                    if isinstance(node.op, ast.Add):
                        out.add(a + b)
                    elif isinstance(node.op, ast.Sub):
                        out.add(a - b)
                    else:
                        out.add(a * b)
                except TypeError:
                    return UNKNOWN
            return frozenset(out)
        if isinstance(node, ast.Call):
            return self._call(node, depth)
        return UNKNOWN

    def _sequence(self, elts: list[ast.AST], depth: int) -> frozenset | _Unknown:
        parts: list[list[tuple]] = []
        for element in elts:
            if isinstance(element, ast.Starred):
                inner = self.values(element.value, depth + 1)
                if inner is UNKNOWN:
                    return UNKNOWN
                spliced = []
                for value in inner:
                    items = _iterate(value)
                    if items is UNKNOWN:
                        return UNKNOWN
                    spliced.append(tuple(items))
                parts.append(spliced)
            else:
                inner = self.values(element, depth + 1)
                if inner is UNKNOWN:
                    return UNKNOWN
                parts.append([(value,) for value in inner])
        if _product_size(parts) > _PRODUCT_LIMIT:
            return UNKNOWN
        return frozenset(tuple(itertools.chain.from_iterable(combo)) for combo in itertools.product(*parts))

    def _call(self, node: ast.Call, depth: int) -> frozenset | _Unknown:
        name = call_name(node.func)
        if isinstance(node.func, ast.Attribute) and name in {"items", "values", "keys"} and not node.args:
            bases = self.values(node.func.value, depth + 1)
            if bases is UNKNOWN:
                return UNKNOWN
            out = set()
            for base in bases:
                if not isinstance(base, DictValue):
                    return UNKNOWN
                if name == "items":
                    out.add(tuple(tuple(pair) for pair in base))
                elif name == "values":
                    out.add(tuple(value for _, value in base))
                else:
                    out.add(tuple(key for key, _ in base))
            return frozenset(out)
        if isinstance(node.func, ast.Name) and name == "range" and 1 <= len(node.args) <= 3 and not node.keywords:
            parts = [self.values(arg, depth + 1) for arg in node.args]
            if any(part is UNKNOWN for part in parts) or _product_size(parts) > _PRODUCT_LIMIT:
                return UNKNOWN
            out = set()
            for combo in itertools.product(*parts):
                if not all(isinstance(value, int) for value in combo):
                    return UNKNOWN
                out.add(tuple(range(*combo)))
            return frozenset(out)
        if isinstance(node.func, ast.Name) and name in {"list", "tuple", "set", "frozenset", "sorted"} and len(node.args) == 1:
            inner = self.values(node.args[0], depth + 1)
            if inner is UNKNOWN:
                return UNKNOWN
            out = set()
            for value in inner:
                items = _iterate(value)
                if items is UNKNOWN:
                    return UNKNOWN
                out.add(tuple(items))
            return frozenset(out)
        if name == "param" and isinstance(node.func, ast.Attribute) and call_name(node.func.value) == "pytest":
            return self._sequence(list(node.args), depth)
        return UNKNOWN

    def _name(self, node: ast.Name, depth: int) -> frozenset | _Unknown:
        name = node.id
        # Comprehension targets bound around the use site.
        current: ast.AST | None = node
        while current is not None:
            parent = self.parents.get(current)
            if isinstance(parent, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                for generator in parent.generators:
                    path = _target_path(generator.target, name)
                    if path is not None:
                        return self._destructure(self.values(generator.iter, depth + 1), path, iterate=True)
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                break
            current = parent
        function = self.enclosing_function(node)
        scopes: list[ast.AST] = [function] if function is not None else []
        scopes.append(self.tree)
        for scope in scopes:
            bindings = self._bindings(name, scope)
            parameter = _parameter(scope, name) if scope is not self.tree else None
            if not bindings and parameter is None:
                continue
            out: set = set()
            for kind, source, path in bindings:
                if kind == "opaque":
                    return UNKNOWN
                resolved = self._destructure(self.values(source, depth + 1), path, iterate=(kind == "iterate"))
                if resolved is UNKNOWN:
                    return UNKNOWN
                out |= resolved
            if parameter is not None:
                resolved = self._parameter_values(scope, name, depth)
                if resolved is UNKNOWN:
                    return UNKNOWN
                out |= resolved
            return frozenset(out)
        return UNKNOWN

    def _destructure(self, values: frozenset | _Unknown, path: tuple, *, iterate: bool) -> frozenset | _Unknown:
        if values is UNKNOWN:
            return UNKNOWN
        candidates: list[Any] = []
        for value in values:
            if iterate:
                items = _iterate(value)
                if items is UNKNOWN:
                    return UNKNOWN
                candidates.extend(items)
            else:
                candidates.append(value)
        out = set()
        for value in candidates:
            for index in path:
                if not isinstance(value, tuple) or index >= len(value):
                    return UNKNOWN
                value = value[index]
            out.add(value)
        return frozenset(out)

    def _parameter_values(self, function: ast.AST, name: str, depth: int) -> frozenset | _Unknown:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return UNKNOWN
        positional = [arg.arg for arg in function.args.posonlyargs + function.args.args]
        out: set = set()
        found = False
        # pytest parametrize values
        for decorator in function.decorator_list:
            if not (
                isinstance(decorator, ast.Call) and call_name(decorator.func) == "parametrize" and len(decorator.args) >= 2
            ):
                continue
            names_node = decorator.args[0]
            names = self.values(names_node, depth + 1)
            if names is UNKNOWN or len(names) != 1:
                return UNKNOWN
            (spec,) = names
            argnames = [part.strip() for part in spec.split(",")] if isinstance(spec, str) else list(spec)
            if name not in argnames:
                continue
            found = True
            rows = self.values(decorator.args[1], depth + 1)
            if rows is UNKNOWN:
                return UNKNOWN
            column = argnames.index(name)
            for table in rows:
                for row in _iterate(table) if not isinstance(table, str) else [table]:
                    if len(argnames) == 1:
                        if isinstance(row, tuple) and len(row) == 1:
                            row = row[0]
                        out.add(row)
                    else:
                        if not isinstance(row, tuple) or column >= len(row):
                            return UNKNOWN
                        out.add(row[column])
        if found:
            return frozenset(out)
        # same-module call sites of a helper
        qualname = function.name
        calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == qualname
        ]
        index = positional.index(name) if name in positional else None
        defaults = function.args.defaults
        default_offset = len(positional) - len(defaults)
        for call in calls:
            argument = None
            for keyword in call.keywords:
                if keyword.arg == name:
                    argument = keyword.value
            if argument is None and index is not None and index < len(call.args):
                if any(isinstance(arg, ast.Starred) for arg in call.args[: index + 1]):
                    return UNKNOWN
                argument = call.args[index]
            if argument is None:
                if index is not None and index >= default_offset:
                    argument = defaults[index - default_offset]
                else:
                    return UNKNOWN
            resolved = self.values(argument, depth + 1)
            if resolved is UNKNOWN:
                return UNKNOWN
            out |= resolved
            found = True
        if not found:
            return UNKNOWN
        return frozenset(out)


def _target_path(target: ast.AST, name: str, prefix: tuple = ()) -> tuple | None:
    if isinstance(target, ast.Name):
        return prefix if target.id == name else None
    if isinstance(target, (ast.Tuple, ast.List)):
        for index, element in enumerate(target.elts):
            if isinstance(element, ast.Starred):
                if _target_path(element.value, name) is not None:
                    return None  # starred unpack is not modelled
                continue
            found = _target_path(element, name, (*prefix, index))
            if found is not None:
                return found
    return None


def _parameter(function: ast.AST, name: str) -> ast.arg | None:
    if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return None
    arguments = function.args
    for arg in arguments.posonlyargs + arguments.args + arguments.kwonlyargs:
        if arg.arg == name:
            return arg
    return None


def _product_size(parts: list) -> int:
    size = 1
    for part in parts:
        size *= max(len(part), 1)
        if size > _PRODUCT_LIMIT:
            return size
    return size

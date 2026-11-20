"""Describe comparison tolerance sources without executing scientific code.

These descriptions are observations, NOT approved comparison requirements or
runtime evidence.  In particular a library default, helper parameter, computed
expression or unresolved binding is not a centrally governed scalar merely
because a central symbol occurs somewhere below it.  Consumers must match an
independently reviewed requirement and the values used by the actual invocation.
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
import copy
from dataclasses import dataclass
import hashlib
from pathlib import PurePosixPath
from typing import Any


_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_SCOPES = (*_FUNCTIONS, ast.ClassDef, ast.Module, ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_CENTRAL_MODULES = {"jqmc._setting", "jqmc._precision"}
_RESOLVERS = {
    "jqmc._precision.get_tolerance",
    "jqmc._precision.get_tolerance_min",
    "jqmc._precision.get_tolerance_reduction_order",
    "jqmc._precision.get_tolerance_xla_rust_consistency",
    "jqmc._setting.get_tolerance_gfmc_fast_pbc",
    "jqmc._setting.get_tolerance_mixed_fidelity",
}
# Parameter positions, not numeric default values.  An omitted argument remains
# explicitly ungoverned until an authored contract describes its semantics.
_SIGNATURES = {
    "numpy.testing.assert_array_equal": {},
    "numpy.testing.assert_allclose": {"rtol": 2, "atol": 3},
    "numpy.allclose": {"rtol": 2, "atol": 3},
    "numpy.isclose": {"rtol": 2, "atol": 3},
    "jax.numpy.allclose": {"rtol": 2, "atol": 3},
    "jax.numpy.isclose": {"rtol": 2, "atol": 3},
    "numpy.testing.assert_almost_equal": {"decimal": 2},
    "numpy.testing.assert_array_almost_equal": {"decimal": 2},
    "numpy.testing.assert_array_max_ulp": {"maxulp": 2},
    "pytest.approx": {"rel": 1, "abs": 2},
    "math.isclose": {"rel_tol": None, "abs_tol": None},
}
_TOLERANCE_NAMES = {name for signature in _SIGNATURES.values() for name in signature}
_MUTATORS = {"append", "extend", "insert", "pop", "clear", "remove", "update", "reverse", "sort", "__setitem__"}


def _syntax(node: ast.AST) -> str:
    # Copy only when the normalization below changes something: without an
    # empty f-string constant the deep copy is structurally identical to the
    # node itself, and ``ast.dump`` does not mutate its argument.
    normalized = node
    if any(
        isinstance(child, ast.JoinedStr)
        and any(isinstance(value, ast.Constant) and value.value == "" for value in child.values)
        for child in ast.walk(node)
    ):
        normalized = copy.deepcopy(node)
        for child in ast.walk(normalized):
            if isinstance(child, ast.JoinedStr):
                child.values = [value for value in child.values if not (isinstance(value, ast.Constant) and value.value == "")]
    try:
        dumped = ast.dump(normalized, include_attributes=False, show_empty=True)
    except TypeError:
        dumped = ast.dump(normalized, include_attributes=False)
    return dumped.replace(", type_params=[]", "")


def _fragment(node: ast.AST) -> dict[str, str]:
    """Keep unresolved helpers sensitive to edits without inlining their bodies."""
    syntax = _syntax(node)
    if len(syntax) <= 512:
        return {"syntax": syntax}
    return {"syntax_sha256": hashlib.sha256(syntax.encode()).hexdigest(), "node_type": type(node).__name__}


def _names(node: ast.AST):
    if isinstance(node, ast.Name):
        yield node.id, ()
    elif isinstance(node, (ast.Tuple, ast.List)):
        for index, child in enumerate(node.elts):
            for name, indices in _names(child):
                yield name, (index, *indices)


@dataclass(frozen=True)
class _Write:
    node: ast.AST
    value: ast.AST | None = None
    indices: tuple[int, ...] = ()
    imported: str | None = None
    unsupported: str | None = None


class ToleranceSourceIndex:
    """Lexical, conservative provenance for comparison calls in one Python module.

    Straight-line aliases, destructuring and definitions dominating the same
    branch are followed. Ambiguous joins, closure state and loop-carried writes
    are recorded as unresolved; they are not guessed from a variable's name.
    This does not prove that a mutable object or a global was unchanged at run
    time. That is deliberately the responsibility of invocation-bound evidence.
    """

    def __init__(self, tree: ast.Module, path: str):
        self.tree = tree
        self.path = path
        self.parents = {id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        self.nodes = {(type(node), node.lineno, node.col_offset): node for node in ast.walk(tree) if hasattr(node, "lineno")}
        self._descriptions: dict[tuple[int, int], dict[str, Any]] | None = None
        self._inequalities: dict[tuple[int, int], dict[str, Any]] | None = None
        self._helper_call_ordinals: dict[tuple[int, int], int] | None = None
        self._observation_addresses: dict[tuple[type, int, int], dict[str, Any]] | None = None
        self.writes: dict[int, dict[str, list[_Write]]] = {}
        self.declarations: dict[int, set[str]] = {}
        self.wildcard_imports: dict[int, list[ast.ImportFrom]] = {}
        for node in ast.walk(tree):
            scope = self._scope(node)
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                self.declarations.setdefault(id(scope), set()).update(node.names)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for name, indices in _names(target):
                        self._add(scope, name, _Write(node, node.value, indices))
                    if isinstance(target, (ast.Attribute, ast.Subscript)):
                        self._mutation(scope, target, node)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self._add(
                        scope,
                        alias.asname or alias.name.split(".")[0],
                        _Write(node, imported=alias.name if alias.asname else alias.name.split(".")[0]),
                    )
            elif isinstance(node, ast.ImportFrom):
                package = list(PurePosixPath(path).with_suffix("").parts[:-1])
                prefix = package[: len(package) - node.level + 1] if node.level else []
                module = ".".join([*prefix, *(node.module or "").split(".")]).strip(".")
                for alias in node.names:
                    if alias.name == "*":
                        self.wildcard_imports.setdefault(id(scope), []).append(node)
                        continue
                    self._add(scope, alias.asname or alias.name, _Write(node, imported=f"{module}.{alias.name}"))
            elif isinstance(node, (*_FUNCTIONS, ast.ClassDef)):
                self._add(scope, node.name, _Write(node, unsupported="local callable or class"))
            elif isinstance(node, (ast.AugAssign, ast.NamedExpr, ast.For, ast.AsyncFor)):
                for name, _ in _names(node.target):
                    self._add(scope, name, _Write(node, unsupported="non-straight-line binding"))
                self._mutation(scope, node.target, node)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars is not None:
                        for name, _ in _names(item.optional_vars):
                            self._add(scope, name, _Write(node, unsupported="context-manager binding"))
            elif isinstance(node, ast.ExceptHandler) and node.name:
                self._add(scope, node.name, _Write(node, unsupported="exception binding"))
            elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
                self._add(scope, node.name, _Write(node, unsupported="pattern-bound value"))
            elif isinstance(node, ast.MatchMapping) and node.rest:
                self._add(scope, node.rest, _Write(node, unsupported="pattern-bound value"))
            elif isinstance(node, ast.Delete):
                for target in node.targets:
                    for name, _ in _names(target):
                        self._add(scope, name, _Write(node, unsupported="deleted binding"))
                    self._mutation(scope, target, node)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _MUTATORS:
                self._mutation(scope, node.func, node)

    def _scope(self, node: ast.AST):
        while (node := self.parents.get(id(node))) is not None:
            if isinstance(node, _SCOPES):
                return node
        return self.tree

    def comparison_aliases(self, node: ast.AST, actual: str, reference: str) -> bool:
        """Recognize aliases of one value, without equating separate executions.

        This is a bounded source check, not a proof of oracle independence.
        Ambiguous writes retain their unresolved provenance and are not claimed
        to be aliases. Calls carry their source addresses: two calls to the same
        function may intentionally test determinism and are not one execution.
        """
        scope = self._scope(node)
        index = _OperandSourceIndex(self)
        expressions = [ast.parse(text, mode="eval").body for text in (actual, reference)]
        values = [index.expression(expression, scope=scope, before=node) for expression in expressions]

        def ambiguous(value, *, callable_position=False):
            if isinstance(value, dict):
                if value.get("kind") == "unresolved" and not callable_position:
                    return True
                return any(ambiguous(child, callable_position=key == "function") for key, child in value.items())
            if isinstance(value, list):
                return any(ambiguous(child) for child in value)
            return False

        def observed_value(value):
            if isinstance(value, dict):
                return value.get("kind") in {"evaluated-call", "evaluated-read", "parameter"} or any(
                    observed_value(child) for child in value.values()
                )
            return isinstance(value, list) and any(observed_value(child) for child in value)

        # A source constant compared with its declared literal is a legitimate
        # configuration guard, not an attempt at an independent runtime oracle.
        return values[0] == values[1] and observed_value(values[0]) and not any(ambiguous(value) for value in values)

    def _add(self, scope: ast.AST, name: str, write: _Write):
        self.writes.setdefault(id(scope), {}).setdefault(name, []).append(write)

    def unreachable_comparison(self, node: ast.AST) -> bool:
        """Recognize a constant-dead if arm without executing candidate code.

        Follow only immutable, function-local lexical aliases. Dynamic calls,
        global/nonlocal state, imported values and ambiguous bindings stay unknown;
        this is not a general proof of branch reachability or fixture coverage.
        """
        matched = self.nodes.get((type(node), getattr(node, "lineno", None), getattr(node, "col_offset", None)))
        if matched is None or (matched is not node and _syntax(matched) != _syntax(node)):
            return False

        def truth(description):
            if description.get("kind") == "literal":
                try:
                    return bool(ast.literal_eval(description["value"]))
                except (ValueError, SyntaxError):
                    return None
            if description.get("kind") != "expression":
                return None
            fields = description["fields"]
            operation = fields.get("op", {}).get("operation")
            if description["operation"] == "UnaryOp" and operation == "Not":
                value = truth(fields["operand"])
                return None if value is None else not value
            if description["operation"] == "BoolOp":
                values = [truth(value) for value in fields["values"]]
                if operation == "And":
                    return False if False in values else True if all(value is True for value in values) else None
                if operation == "Or":
                    return True if True in values else False if all(value is False for value in values) else None
            return None

        cursor = matched
        while (parent := self.parents.get(id(cursor))) is not None:
            if isinstance(parent, (*_FUNCTIONS, ast.Lambda)):
                break
            if isinstance(parent, ast.If):
                index = _BranchSourceIndex(self, self._scope(parent.test))
                condition = truth(index.expression(parent.test))
                if (condition is False and cursor in parent.body) or (condition is True and cursor in parent.orelse):
                    return True
            cursor = parent
        return False

    def _mutation(self, scope: ast.AST, target: ast.AST, node: ast.AST):
        while isinstance(target, (ast.Attribute, ast.Subscript)):
            target = target.value
        if isinstance(target, ast.Name):
            self._add(scope, target.id, _Write(node, unsupported="mutated binding"))

    def _qualified_scope(self, node: ast.AST) -> str:
        names = []
        while node is not self.tree:
            if isinstance(node, (*_FUNCTIONS, ast.ClassDef)):
                names.append(node.name)
            node = self._scope(node)
        return ".".join(reversed(names)) or "<module>"

    def observation_address(self, node: ast.AST) -> dict[str, Any]:
        """Identify syntax within its qualified symbol, separately from tracing coordinates.

        Ordinals include all syntax of the same kind, not just observations
        reachable from one particular test. Shared helpers thus keep the same
        identity in every caller. Locations and full expressions remain part
        of the binding digest; a stable ID is not a certificate of content.
        """
        if self._observation_addresses is None:
            addresses = {}
            counts: dict[tuple[str, str], int] = {}
            for child in sorted(self.nodes.values(), key=lambda item: (item.lineno, item.col_offset)):
                scope = self._qualified_scope(self._scope(child))
                kind = type(child).__name__.lower()
                counter = (scope, kind)
                counts[counter] = counts.get(counter, 0) + 1
                addresses[(type(child), child.lineno, child.col_offset)] = {
                    "id": f"{self.path}::{scope}#{kind}-{counts[counter]}",
                    "source_location": {"path": self.path, "line": child.lineno, "column": child.col_offset},
                }
            self._observation_addresses = addresses
        return copy.deepcopy(self._observation_addresses[(type(node), node.lineno, node.col_offset)])

    def _controls(self, node: ast.AST, scope: ast.AST):
        result = []
        while node is not scope:
            parent = self.parents.get(id(node))
            if parent is None:
                break
            if isinstance(
                parent,
                (
                    ast.If,
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                    ast.Try,
                    ast.TryStar,
                    ast.With,
                    ast.AsyncWith,
                    ast.Match,
                    ast.match_case,
                ),
            ):
                for field, value in ast.iter_fields(parent):
                    if isinstance(value, list) and node in value:
                        result.append((id(parent), field))
            node = parent
        return tuple(reversed(result))

    def _unresolved(self, reason: str, node: ast.AST, **extra) -> dict[str, Any]:
        return {"kind": "unresolved", "reason": reason, **_fragment(node), **extra}

    def _parameter(self, scope: ast.AST, name: str, seen):
        if not isinstance(scope, _FUNCTIONS):
            return None
        positional = [*scope.args.posonlyargs, *scope.args.args]
        defaults = [None] * (len(positional) - len(scope.args.defaults)) + list(scope.args.defaults)
        pairs = [*zip(positional, defaults), *zip(scope.args.kwonlyargs, scope.args.kw_defaults)]
        pairs.extend((parameter, None) for parameter in (scope.args.vararg, scope.args.kwarg) if parameter is not None)
        for parameter, default in pairs:
            if parameter.arg == name:
                result = {"kind": "parameter", "scope": self._qualified_scope(scope), "name": name}
                if default is not None:
                    result["default"] = self.expression(default, scope=self._scope(scope), before=scope, _seen=seen)
                return result
        return None

    def _name(self, node: ast.Name, scope: ast.AST, before: ast.AST, seen):
        key = (id(scope), node.id, id(before))
        if key in seen:
            return self._unresolved("cyclic binding", node)
        seen = seen | {key}
        if not isinstance(scope, (*_FUNCTIONS, ast.Module, ast.ClassDef)):
            return self._unresolved("unsupported lexical scope", node)
        if node.id in self.declarations.get(id(scope), ()):
            return self._unresolved("explicit global/nonlocal state requires invocation evidence", node)
        writes = self.writes.get(id(scope), {}).get(node.id, [])
        use_path = self._controls(before, scope)
        loops = [parent for parent in self._ancestors(before) if isinstance(parent, (ast.For, ast.AsyncFor, ast.While))]
        if any(
            loop in self._ancestors(write.node)
            and (write.node.lineno, write.node.col_offset)
            >= (getattr(before, "lineno", float("inf")), getattr(before, "col_offset", 0))
            for loop in loops
            for write in writes
        ):
            return self._unresolved("loop-carried binding", node, writes=[_fragment(write.node) for write in writes])
        compatible = []
        for write in writes:
            path = self._controls(write.node, scope)
            # A write on the opposite arm of the same if cannot reach this use.
            if any(control == other and arm != other_arm for control, arm in path for other, other_arm in use_path):
                continue
            if (write.node.lineno, write.node.col_offset) < (
                getattr(before, "lineno", float("inf")),
                getattr(before, "col_offset", 0),
            ):
                if write.node not in self._ancestors(before):
                    compatible.append(write)
        if compatible:
            write = max(compatible, key=lambda row: (row.node.lineno, row.node.col_offset))
            for wildcard in self.wildcard_imports.get(id(scope), ()):
                if (
                    (write.node.lineno, write.node.col_offset)
                    < (wildcard.lineno, wildcard.col_offset)
                    < (getattr(before, "lineno", float("inf")), getattr(before, "col_offset", 0))
                ):
                    return self._unresolved("wildcard import may replace binding", wildcard, name=node.id)
            path = self._controls(write.node, scope)
            if not all(control in use_path for control in path):
                return self._unresolved("control-dependent binding", write.node, name=node.id)
            if write.unsupported:
                return self._unresolved(write.unsupported, write.node, name=node.id)
            if write.imported:
                return self._reference(write.imported)
            if write.value is None:
                return self._unresolved("annotation without a value", write.node)
            value = self.expression(write.value, scope=scope, before=write.node, _seen=seen)
            for index in write.indices:
                if value["kind"] in {"tuple", "list"} and index < len(value["items"]):
                    value = value["items"][index]
                else:
                    value = {"kind": "item", "value": value, "index": index}
            return value
        parameter = self._parameter(scope, node.id, seen)
        if parameter is not None:
            return parameter
        if writes:
            return self._unresolved("local binding used before assignment", node)
        outer = self._scope(scope)
        while isinstance(outer, (*_FUNCTIONS, ast.ClassDef)):
            if isinstance(outer, _FUNCTIONS):
                bindings = self.writes.get(id(outer), {}).get(node.id, [])
                parameter = self._parameter(outer, node.id, seen)
                if bindings or parameter is not None or node.id in self.declarations.get(id(outer), ()):
                    return self._unresolved(
                        "closure state requires invocation evidence",
                        node,
                        scope=self._qualified_scope(outer),
                        bindings=[_fragment(write.node) for write in bindings],
                        parameter=parameter,
                    )
            outer = self._scope(outer)
        if scope is self.tree:
            return self._unresolved("name has no source binding", node)
        # Global initializers have run before pytest invokes a test. This is a
        # source description, not a claim that globals cannot be monkeypatched.
        return self._name(node, outer, ast.Pass(), seen)

    def _ancestors(self, node: ast.AST):
        result = []
        while (node := self.parents.get(id(node))) is not None:
            result.append(node)
        return result

    @staticmethod
    def _reference(name: str):
        module = name.rpartition(".")[0]
        return {"kind": "central-reference" if module in _CENTRAL_MODULES else "imported-reference", "name": name}

    def expression(self, node: ast.AST, *, scope=None, before=None, _seen=frozenset()) -> dict[str, Any]:
        """Return a JSON-compatible description, retaining unsupported constructs."""
        scope = self._scope(node) if scope is None else scope
        before = node if before is None else before
        if isinstance(node, ast.Name):
            return self._name(node, scope, before, _seen)
        if isinstance(node, ast.Constant):
            return {"kind": "literal", "type": type(node.value).__name__, "value": repr(node.value)}
        if isinstance(
            node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Await, ast.Yield, ast.YieldFrom)
        ):
            return self._unresolved("deferred or comprehension expression requires invocation evidence", node)

        def resolve(child):
            return self.expression(child, scope=scope, before=before, _seen=_seen)

        if isinstance(node, ast.Attribute):
            value = resolve(node.value)
            if value["kind"] in {"imported-reference", "central-reference"}:
                return self._reference(f"{value['name']}.{node.attr}")
            return {"kind": "attribute", "value": value, "name": node.attr}
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            return {"kind": type(node).__name__.lower(), "items": [resolve(child) for child in node.elts]}
        if isinstance(node, ast.Call):
            inlined = self._pure_return_resolver_call(node, scope, before, _seen)
            if inlined is not None:
                return inlined
            function = resolve(node.func)
            return {
                "kind": "central-resolver-call"
                if function.get("name") in _RESOLVERS and function["kind"] == "central-reference"
                else "call",
                "function": function,
                "args": [resolve(argument) for argument in node.args],
                "keywords": [{"name": keyword.arg, "value": resolve(keyword.value)} for keyword in node.keywords],
            }
        # Preserve the operation AND recursively resolved inputs, e.g. scaling
        # a central tolerance is not described as the original named scalar.
        fields = {}
        for key, value in ast.iter_fields(node):
            if key == "ctx":
                continue
            fields[key] = (
                [resolve(child) if isinstance(child, ast.AST) else child for child in value]
                if isinstance(value, list)
                else resolve(value)
                if isinstance(value, ast.AST)
                else value
            )
        return {"kind": "expression", "operation": type(node).__name__, "fields": fields}

    def _pure_return_resolver_call(self, call: ast.Call, scope, before, seen) -> dict[str, Any] | None:
        """Describe a call of a module helper that only returns a central resolver call.

        This is deliberately narrow and fails closed (returns ``None`` so the
        ordinary unresolved description is kept). It applies only to an
        undecorated module-level ``def`` bound once in its module, called by its
        plain name with plain positional/keyword arguments, whose single
        statement after an optional docstring is ``return <central resolver
        call>``. The caller's argument descriptions replace exactly the helper
        parameters that call reads (a declared default when the caller omits
        one). No arithmetic, branch or side effect can hide in such a helper.
        Its syntax is recorded, so editing the helper changes the description.
        This does not move or approve the resolver's own source site.
        """
        if (
            not isinstance(call.func, ast.Name)
            or any(isinstance(argument, ast.Starred) for argument in call.args)
            or any(keyword.arg is None for keyword in call.keywords)
        ):
            return None
        writes = self.writes.get(id(self.tree), {}).get(call.func.id, [])
        if len(writes) != 1 or type(writes[0].node) is not ast.FunctionDef:
            return None
        if any(call.func.id in names for names in self.declarations.values()):
            return None  # a ``global`` rebinding elsewhere could replace the helper at run time
        helper = writes[0].node
        if self._scope(helper) is not self.tree or self._name(call.func, scope, before, seen) != self._unresolved(
            "local callable or class", helper, name=call.func.id
        ):
            return None
        signature = helper.args
        if helper.decorator_list or signature.posonlyargs or signature.vararg or signature.kwarg or signature.kwonlyargs:
            return None
        body = helper.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if not isinstance(body[0].value.value, str):
                return None
            body = body[1:]
        if len(body) != 1 or not isinstance(body[0], ast.Return) or not isinstance(body[0].value, ast.Call):
            return None
        returned = self.expression(body[0].value, scope=helper, before=body[0], _seen=seen)
        if returned.get("kind") != "central-resolver-call":
            return None
        names = [parameter.arg for parameter in signature.args]
        if len(call.args) > len(names):
            return None
        bound = dict(zip(names, call.args))
        for keyword in call.keywords:
            if keyword.arg not in names or keyword.arg in bound:
                return None
            bound[keyword.arg] = keyword.value
        defaults = set(names[len(names) - len(signature.defaults) :])
        qualified = self._qualified_scope(helper)
        missing = []

        def substitute(value):
            if isinstance(value, dict):
                if value.get("kind") == "parameter" and value.get("scope") == qualified:
                    if value["name"] in bound:
                        return self.expression(bound[value["name"]], scope=scope, before=before, _seen=seen)
                    if value["name"] in defaults and "default" in value:
                        return value["default"]
                    missing.append(value["name"])
                    return value
                return {key: substitute(child) for key, child in value.items()}
            if isinstance(value, list):
                return [substitute(child) for child in value]
            return value

        described = substitute(returned)
        if missing:
            return None
        return {**described, "pure_return_helper": {"name": helper.name, **_fragment(helper)}}

    def comparison(self, call: ast.Call) -> dict[str, Any] | None:
        """Describe all tolerance arguments, including partial/defaulted calls."""
        function = self.expression(call.func)
        signature = _SIGNATURES.get(function.get("name")) if function["kind"] == "imported-reference" else None
        explicit = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg in _TOLERANCE_NAMES}
        leaf = (
            call.func.attr
            if isinstance(call.func, ast.Attribute)
            else call.func.id
            if isinstance(call.func, ast.Name)
            else None
        )
        if signature is None and not explicit and leaf not in {name.rsplit(".", 1)[-1] for name in _SIGNATURES}:
            return None
        forwarded = [self.expression(keyword.value) for keyword in call.keywords if keyword.arg is None]
        starred = any(isinstance(argument, ast.Starred) for argument in call.args)
        arguments = {}
        for name, position in (signature or dict.fromkeys(explicit)).items():
            if name in explicit:
                arguments[name] = self.expression(explicit[name])
            elif forwarded or starred:
                arguments[name] = self._unresolved("forwarded arguments require call binding", call)
            elif position is not None and position < len(call.args):
                arguments[name] = self.expression(call.args[position])
            else:
                arguments[name] = {"kind": "library-default", "parameter": name}
        if function.get("name") == "pytest.approx" and signature is not None:
            # pytest uses None as a sentinel, not a numeric tolerance. Unlike
            # NumPy, an explicit non-None abs suppresses an omitted/None rel.
            # Do not execute central imports or assume a helper/global is
            # non-None: describe that condition, leaving invocation proof to
            # the consumer. Forwarded arguments remain unresolved above.
            for name, value in tuple(arguments.items()):
                if value == {"kind": "literal", "type": "NoneType", "value": "None"}:
                    arguments[name] = {"kind": "library-default", "parameter": name, "specified_as": value}
            if arguments["rel"]["kind"] == "library-default" and arguments["abs"]["kind"] != "library-default":
                arguments["rel"] = {
                    **arguments["rel"],
                    "kind": "conditional-library-default",
                    "condition": "abs-is-None",
                }
        result = {"comparator": function, "arguments": arguments}
        if signature is None:
            result["unresolved_signature"] = True
        if forwarded:
            result["forwarded_keywords"] = forwarded
        return result

    def comparisons(self) -> list[dict[str, Any]]:
        """List comparison descriptions with lexical ordinals and separate locations."""
        return copy.deepcopy(list(self._comparison_index().values()))

    def _comparison_index(self) -> dict[tuple[int, int], dict[str, Any]]:
        if self._descriptions is not None:
            return self._descriptions
        rows = {}
        counts: dict[tuple[str, str], int] = {}
        for call in sorted(
            (node for node in ast.walk(self.tree) if isinstance(node, ast.Call)),
            key=lambda node: (node.lineno, node.col_offset),
        ):
            row = self.comparison(call)
            if row is None:
                continue
            scope = self._qualified_scope(self._scope(call))
            exact = row["comparator"].get("name") == "numpy.testing.assert_array_equal" or (
                isinstance(call.func, (ast.Name, ast.Attribute))
                and (call.func.id if isinstance(call.func, ast.Name) else call.func.attr) == "assert_array_equal"
            )
            category = "exact-comparison" if exact else "comparison"
            counter = (scope, category)
            counts[counter] = counts.get(counter, 0) + 1
            rows[(call.lineno, call.col_offset)] = {
                "site": f"{self.path}::{scope}#{category}-{counts[counter]}",
                "location": {"line": call.lineno, "column": call.col_offset},
                **row,
            }
        self._descriptions = rows
        return rows

    def observation(self, node: ast.AST) -> dict[str, Any]:
        """Bind a typed verdict to its actual AST, not another call on a nearby line."""
        matched = self.nodes.get((type(node), getattr(node, "lineno", None), getattr(node, "col_offset", None)))
        if matched is None or (matched is not node and _syntax(matched) != _syntax(node)):
            return {"status": "unresolved", "reason": "verdict AST does not match the indexed source", "comparisons": []}
        descriptions = self._comparison_index()
        rows = []
        for child in ast.walk(matched):
            if isinstance(child, ast.Call):
                row = descriptions.get((child.lineno, child.col_offset))
                if row is not None:
                    rows.append(copy.deepcopy(row))
        rows.sort(key=lambda row: (row["location"]["line"], row["location"]["column"]))
        if not rows:
            return {"status": "unresolved", "reason": "verdict has no described comparison call", "comparisons": []}
        return {"status": "source-described", "comparisons": rows}

    def _callable_name(self, node: ast.AST) -> str | None:
        """Resolve a magnitude/reduction callable; an unbound bare name is a builtin."""
        description = self.expression(node)
        if description["kind"] in {"imported-reference", "central-reference"}:
            return description["name"]
        if (
            isinstance(node, ast.Name)
            and description["kind"] == "unresolved"
            and description.get("reason") == "name has no source binding"
        ):
            return f"builtins.{node.id}"
        return None

    def _inequality_index(self) -> dict[tuple[int, int], dict[str, Any]]:
        """Hand-written ``assert`` inequalities in a reviewed form.

        Ordinals count every asserted single comparison in a scope, recognized
        or not, so rewriting one unrecognized site does not renumber the others.
        """
        if self._inequalities is not None:
            return self._inequalities
        from gdd.engine import inequality_receipts

        rows = {}
        counts: dict[str, int] = {}
        for assertion in sorted(
            (node for node in ast.walk(self.tree) if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)),
            key=lambda node: (node.lineno, node.col_offset),
        ):
            scope = self._qualified_scope(self._scope(assertion))
            counts[scope] = counts.get(scope, 0) + 1
            test = assertion.test
            aliases = {}
            # Only immediately preceding straight-line single-name assignments
            # in this exact statement list can define a precomputed deviation
            # or scale. No control-flow join, helper or arbitrary interpreter.
            parent = self.parents.get(id(assertion))
            assignments = {}
            preceding = []
            if parent is not None:
                for _, statements in ast.iter_fields(parent):
                    if not isinstance(statements, list) or assertion not in statements:
                        continue
                    for previous in reversed(statements[: statements.index(assertion)]):
                        if not (
                            isinstance(previous, ast.Assign)
                            and len(previous.targets) == 1
                            and isinstance(previous.targets[0], ast.Name)
                        ):
                            break
                        preceding.append(previous)
                        assignments.setdefault(previous.targets[0].id, previous)
            expanded = copy.copy(test)
            if isinstance(test.left, ast.Name) and test.left.id in assignments:
                expanded.left = assignments[test.left.id].value
                aliases["deviation"] = test.left.id
            bound = test.comparators[0]
            if (
                isinstance(bound, ast.BinOp)
                and isinstance(bound.op, ast.Add)
                and isinstance(bound.right, ast.BinOp)
                and isinstance(bound.right.op, ast.Mult)
                and isinstance(bound.right.right, ast.Name)
                and bound.right.right.id in assignments
            ):
                expanded_bound = copy.copy(bound)
                expanded_bound.right = copy.copy(bound.right)
                expanded_bound.right.right = assignments[bound.right.right.id].value
                expanded.comparators = [expanded_bound]
                aliases["scale"] = bound.right.right.id
            if aliases:
                earliest = min(assignments[name].lineno for name in aliases.values())
                intervening = [statement for statement in preceding if statement.lineno >= earliest]
                # An intervening assignment/call can change an operand after
                # its deviation was evaluated. Only the selected pure aliases
                # themselves may occur between their definitions and assertion.
                if len(intervening) != len(aliases) or any(
                    statement.targets[0].id not in aliases.values() for statement in intervening
                ):
                    continue
            recognized = inequality_receipts.recognize(expanded, self._callable_name)
            if recognized is None:
                continue
            test = assertion.test
            rows[(test.lineno, test.col_offset)] = {
                "site": f"{self.path}::{scope}#{inequality_receipts.SITE_CATEGORY}-{counts[scope]}",
                "location": {"line": test.lineno, "column": test.col_offset},
                "comparator": dict(inequality_receipts.SOURCE_COMPARATOR),
                "form": recognized["form"],
                "operator": recognized["operator"],
                "symbols": {role: ast.unparse(recognized[role]) for role in ("actual", "reference", "atol", "rtol")},
                "arguments": {role: self.expression(recognized[role]) for role in ("atol", "rtol")},
            }
            if "scale_reference" in recognized:
                rows[(test.lineno, test.col_offset)]["symbols"]["scale_reference"] = ast.unparse(recognized["scale_reference"])
                rows[(test.lineno, test.col_offset)]["scale_floor"] = recognized["scale_floor"]
            for conversion in ("deviation_float", "scale_float"):
                if recognized.get(conversion):
                    rows[(test.lineno, test.col_offset)][conversion] = True
            if aliases:
                rows[(test.lineno, test.col_offset)]["expression_aliases"] = aliases
        self._inequalities = rows
        return rows

    def inequality_observation(self, node: ast.AST) -> dict[str, Any] | None:
        """Source description of a recognized asserted inequality, or None."""
        matched = self.nodes.get((type(node), getattr(node, "lineno", None), getattr(node, "col_offset", None)))
        if not isinstance(matched, ast.Assert) or (matched is not node and _syntax(matched) != _syntax(node)):
            return None
        row = self._inequality_index().get((matched.test.lineno, matched.test.col_offset))
        if row is None:
            return None
        return {"status": "source-described", "comparisons": [copy.deepcopy(row)]}

    def helper_inequality_observation(self, node: ast.AST) -> dict[str, Any] | None:
        """Bind one direct call to a pure, module-local single-assert helper.

        This is argument substitution, not a helper interpreter. Defaults,
        decorators, forwarding, branches and nested helper calls are excluded.
        Every formal is explicit; only the four recognized inequality operands
        acquire roles. Their tolerance provenance belongs to the caller.
        """
        from gdd.engine import inequality_receipts

        call = self.nodes.get((type(node), getattr(node, "lineno", None), getattr(node, "col_offset", None)))
        if not isinstance(call, ast.Call) or _syntax(call) != _syntax(node) or not isinstance(call.func, ast.Name):
            return None
        writes = self.writes.get(id(self.tree), {}).get(call.func.id, ())
        if len(writes) != 1 or type(writes[0].node) is not ast.FunctionDef:
            return None
        helper = writes[0].node
        scope = self._scope(call)
        if (
            self._scope(helper) is not self.tree
            or self._name(call.func, scope, call, set())
            != self._unresolved("local callable or class", helper, name=helper.name)
            or any(helper.name in names for names in self.declarations.values())
        ):
            return None
        signature = helper.args
        if (
            helper.decorator_list
            or signature.defaults
            or signature.posonlyargs
            or signature.vararg
            or signature.kwarg
            or signature.kwonlyargs
            or any(isinstance(argument, ast.Starred) for argument in call.args)
            or any(keyword.arg is None for keyword in call.keywords)
        ):
            return None
        body = helper.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        if len(body) != 1 or not isinstance(body[0], ast.Assert):
            return None
        recognized = inequality_receipts.recognize(body[0].test, self._callable_name)
        if (
            recognized is None
            or any(key in recognized for key in ("scale_reference", "deviation_float", "scale_float"))
            or any(not isinstance(recognized[role], ast.Name) for role in ("actual", "reference", "atol", "rtol"))
        ):
            return None
        names = [parameter.arg for parameter in signature.args]
        if len(call.args) > len(names):
            return None
        bound = dict(zip(names, call.args))
        for keyword in call.keywords:
            if keyword.arg not in names or keyword.arg in bound:
                return None
            bound[keyword.arg] = keyword.value
        if set(bound) != set(names) or any(
            recognized[role].id not in bound for role in ("actual", "reference", "atol", "rtol")
        ):
            return None
        # Count local named calls even when their body ceases to be supported:
        # a source fault must not renumber unrelated approved call sites.
        if self._helper_call_ordinals is None:
            self._helper_call_ordinals = {}
            counts: dict[int, int] = {}
            local_functions = {
                name
                for name, bindings in self.writes.get(id(self.tree), {}).items()
                if any(type(write.node) is ast.FunctionDef for write in bindings)
            }
            for child in sorted(
                (
                    child
                    for child in ast.walk(self.tree)
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id in local_functions
                ),
                key=lambda child: (child.lineno, child.col_offset),
            ):
                owner = id(self._scope(child))
                counts[owner] = counts.get(owner, 0) + 1
                self._helper_call_ordinals[(child.lineno, child.col_offset)] = counts[owner]
        row = {
            "site": f"{self.path}::{self._qualified_scope(scope)}#helper-inequality-{self._helper_call_ordinals[(call.lineno, call.col_offset)]}",
            "location": {"line": call.lineno, "column": call.col_offset},
            "comparator": dict(inequality_receipts.SOURCE_COMPARATOR),
            "form": recognized["form"],
            "operator": recognized["operator"],
            "symbols": {role: recognized[role].id for role in ("actual", "reference", "atol", "rtol")},
            "arguments": {role: self.expression(bound[recognized[role].id]) for role in ("atol", "rtol")},
            "helper": {
                "name": helper.name,
                "line": helper.lineno,
                "syntax_sha256": hashlib.sha256(_syntax(helper).encode()).hexdigest(),
                "call_position": [call.lineno, call.end_lineno, call.col_offset, call.end_col_offset],
            },
        }
        return {
            "actual": ast.unparse(bound[recognized["actual"].id]),
            "reference": ast.unparse(bound[recognized["reference"].id]),
            "provenance": {"status": "source-described", "comparisons": [row]},
        }


class _BranchSourceIndex(ToleranceSourceIndex):
    """Do not mistake a module initializer or captured variable for a constant."""

    def __init__(self, source: ToleranceSourceIndex, scope: ast.AST):
        self.__dict__ = source.__dict__.copy()
        self.branch_scope = scope
        self.captured = {name for node in ast.walk(scope) if isinstance(node, ast.Nonlocal) for name in node.names}

    def _name(self, node: ast.Name, scope: ast.AST, before: ast.AST, seen):
        if scope is not self.branch_scope or node.id in self.captured:
            return self._unresolved("global/nonlocal branch state requires execution evidence", node)
        if not self.writes.get(id(scope), {}).get(node.id):
            return self._unresolved("branch value is not a function-local constant", node)
        return super()._name(node, scope, before, seen)


class _OperandSourceIndex(ToleranceSourceIndex):
    """Read-only lexical index view retaining identity of each evaluated call."""

    def __init__(self, source: ToleranceSourceIndex):
        self.__dict__.update(source.__dict__)

    def expression(self, node: ast.AST, **kwargs) -> dict[str, Any]:
        if isinstance(node, ast.Call):
            # A NumPy copy changes storage, not the observed numerical values.
            # Resolve the imported callable instead of trusting its spelling;
            # an unrelated local ``copy`` function is not a value alias.
            function = ToleranceSourceIndex.expression(self, node.func, **kwargs)
            if (
                function.get("kind") == "imported-reference"
                and function.get("name") == "numpy.copy"
                and len(node.args) == 1
                and not node.keywords
            ):
                return self.expression(node.args[0], **kwargs)
            # This identity is used only for an in-memory equality decision;
            # it is never serialized into a binding, receipt or source digest.
            # Do not recursively expand a computation's arguments: alias
            # identity needs the evaluation site, not its transitive dataflow.
            return {"kind": "evaluated-call", "call_identity": id(node)}
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            # Two reads separated by a driver step can observe different state,
            # including properties. Only a stored alias is the same read.
            return {"kind": "evaluated-read", "read_identity": id(node)}
        return super().expression(node, **kwargs)


def observation_location(observation: dict[str, Any]) -> tuple[str, int, int]:
    """Resolve tracing metadata without treating a symbol ID as a line address."""
    label = observation["id"]
    if "::" in label:
        location = observation.get("source_location")
        if (
            not isinstance(location, dict)
            or set(location) != {"path", "line", "column"}
            or location["path"] != label.split("::", 1)[0]
            or type(location["line"]) is not int
            or location["line"] < 1
            or type(location["column"]) is not int
            or location["column"] < 0
        ):
            raise ValueError(f"symbol-anchored observation lacks valid source_location: {label}")
        return location["path"], location["line"], location["column"]
    # Legacy explicit scratch specs only, not an alias for new receipt IDs.
    source, line, column = label.rsplit(":", 2)
    line, column = int(line), int(column)
    if not source or line < 1 or column < 0:
        raise ValueError("invalid legacy observation source location")
    return source, line, column


def observation_provenance_errors(observation: dict[str, Any]) -> list[str]:
    """Validate source linkage, without claiming that its policy was approved."""
    label = observation.get("id", "<missing-id>")
    provenance = observation.get("tolerance_provenance")
    if not isinstance(provenance, dict) or provenance.get("status") != "source-described":
        return [f"{label}: tolerance comparison lacks matched source provenance"]
    comparisons = provenance.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        return [f"{label}: tolerance provenance has no comparison calls"]
    try:
        source, start_line, start_column = observation_location(observation)
        end_line = observation["end_lineno"]
        if type(end_line) is not int or end_line < start_line:
            raise ValueError("invalid source range")
    except (AttributeError, KeyError, TypeError, ValueError):
        return [f"{label}: tolerance observation lacks a valid source range"]
    errors = []
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            errors.append(f"{label}: malformed tolerance source comparison")
            continue
        function = comparison.get("comparator", {})
        name = function.get("name") if isinstance(function, dict) else None
        signature = _SIGNATURES.get(name) if isinstance(name, str) and function.get("kind") == "imported-reference" else None
        location = comparison.get("location", {})
        line, column = (location.get("line"), location.get("column")) if isinstance(location, dict) else (None, None)
        if (
            not isinstance(comparison.get("site"), str)
            or not comparison["site"].startswith(source + "::")
            or type(line) is not int
            or type(column) is not int
            or not start_line <= line <= end_line
            or column < 0
            or (line == start_line and column < start_column)
        ):
            errors.append(f"{label}: tolerance provenance belongs to a different source site")
        if signature is None or comparison.get("unresolved_signature"):
            errors.append(f"{label}: tolerance comparator signature is unresolved")
            continue
        arguments = comparison.get("arguments")
        if not isinstance(arguments, dict) or set(arguments) != set(signature):
            errors.append(f"{label}: tolerance source arguments omit or invent comparator parameters")
        elif any(not isinstance(value, dict) or not value.get("kind") for value in arguments.values()):
            errors.append(f"{label}: malformed tolerance argument provenance")
    return errors

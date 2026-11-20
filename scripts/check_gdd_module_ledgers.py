#!/usr/bin/env python3
"""Static CI gate for governed per-module guarantee ledgers.

Validates every governed ledger under ``gdd/ssot/guarantees/`` (the derivative
ledger through its ``module_evidence`` section): guarantee kinds and ID
prefixes against the verdict contract, that every verdict method is known and
legal for its guarantee's class, well-formed evidence groups without a node
registered twice, and that every declared evidence node exists with exactly
the pinned pytest parameter expansion. It reads the ledgers and the verdict
contract only; whether a test verifies its claim is judged by the test and by
review. The GDD preflight runs it on every pull request.
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
import ast
import itertools
import math
import sys
from pathlib import Path
from types import CodeType
from typing import Any
import weakref

import yaml


ROOT = Path(__file__).resolve().parents[1]
GUARANTEE_DIR = ROOT / "gdd/ssot/guarantees"
VERDICTS_PATH = ROOT / "gdd/ssot/verdicts.yaml"
sys.path.insert(0, str(ROOT))

from gdd.engine.ledger import ledger_cache, load_ledger
from gdd.engine.parameter_requirements import parameter_case_errors


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


class _Opaque:
    """Inert stand-in for imported symbols used inside parametrize values.

    Attribute access and calls return another opaque token, so expressions like
    ``np.array([...])`` evaluate to a countable list element. It deliberately
    has no ``__len__``: an opaque value used AS the case list still fails loudly.
    """

    def __init__(self, expression: str = "unresolved") -> None:
        self.gdd_expression = expression

    def __getattr__(self, name: str) -> _Opaque:
        return _Opaque(f"{self.gdd_expression}.{name}")

    def __call__(self, *args: Any, **kwargs: Any) -> _Opaque:
        arguments = [*(repr(value) for value in args), *(f"{key}={value!r}" for key, value in kwargs.items())]
        return _Opaque(f"{self.gdd_expression}({', '.join(arguments)})")

    def __format__(self, spec: str) -> str:
        return self.gdd_expression

    def __repr__(self) -> str:
        return self.gdd_expression

    def _binary(self, operator: str, other: Any) -> _Opaque:
        return _Opaque(f"({self.gdd_expression} {operator} {other!r})")

    def _reflected(self, operator: str, other: Any) -> _Opaque:
        # ``other <op> self``: the left operand renders on the left.
        return _Opaque(f"({other!r} {operator} {self.gdd_expression})")

    def __add__(self, other: Any) -> _Opaque:
        return self._binary("+", other)

    __radd__ = __add__

    def __sub__(self, other: Any) -> _Opaque:
        return self._binary("-", other)

    def __rsub__(self, other: Any) -> _Opaque:
        return self._reflected("-", other)

    def __mul__(self, other: Any) -> _Opaque:
        return self._binary("*", other)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> _Opaque:
        return self._binary("/", other)

    def __rtruediv__(self, other: Any) -> _Opaque:
        return self._reflected("/", other)

    def __neg__(self) -> _Opaque:
        return _Opaque(f"-{self.gdd_expression}")

    def __pos__(self) -> _Opaque:
        return _Opaque(f"+{self.gdd_expression}")

    def __bool__(self) -> bool:
        # Unresolvable feature flags evaluate falsy so environment-conditioned
        # case lists pin the full (canonical) collection, not a reduced lane.
        return False


class _ParamShim:
    """Inert pytest.param stand-in for evaluating case-list expressions."""

    mark = _Opaque("pytest.mark")

    @staticmethod
    def param(*values, marks=(), id=None):
        return _ParamValue(values, id, marks)


class _NumpyShim(_Opaque):
    """Static NumPy stand-in with exact scalar constants used in pytest IDs."""

    pi = math.pi

    def __init__(self) -> None:
        super().__init__("numpy")


class _ParamValue(tuple):
    """Tuple-compatible pytest.param value retaining its explicit collection ID."""

    def __new__(cls, values: tuple[Any, ...], parameter_id: str | None, marks: Any = ()):
        instance = super().__new__(cls, values)
        instance.gdd_parameter_id = parameter_id
        if not isinstance(marks, (list, tuple, set, frozenset)):
            marks = (marks,)
        instance.gdd_marks = tuple(str(mark) for mark in marks)
        return instance


# Expressions that ``ast.literal_eval`` rejected, with their compiled code
# (``None`` while ``compile`` itself rejects the node, as it does for a store
# context).  Both steps are pure functions of the never-mutated syntax node, so
# a repeated evaluation skips the known literal_eval failure and evals the
# identical code object in a fresh namespace (or recompiles, raising exactly
# as before).  Weak keys: an entry lives exactly as long as its node.
_COMPILED_NON_LITERALS: weakref.WeakKeyDictionary[ast.AST, CodeType | None] = weakref.WeakKeyDictionary()
_NOT_SEEN = object()


def _literal(node: ast.AST, constants: dict[str, Any]) -> Any:
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "range" and all(isinstance(argument, ast.Constant) for argument in node.args):
            return list(range(*[argument.value for argument in node.args]))
        if node.func.id == "list" and len(node.args) == 1:
            return list(_literal(node.args[0], constants))
    memoize = isinstance(node, ast.AST)
    code = _COMPILED_NON_LITERALS.get(node, _NOT_SEEN) if memoize else _NOT_SEEN
    if code is _NOT_SEEN:
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError):
            if memoize:
                _COMPILED_NON_LITERALS[node] = None
            code = compile(ast.Expression(node), "<parametrize>", "eval")
            if memoize:
                _COMPILED_NON_LITERALS[node] = code
    elif code is None:
        code = compile(ast.Expression(node), "<parametrize>", "eval")
        _COMPILED_NON_LITERALS[node] = code
    namespace = {
        "__builtins__": {},
        "itertools": itertools,
        "range": range,
        "pytest": _ParamShim,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "frozenset": frozenset,
        "sorted": sorted,
        "len": len,
        "zip": zip,
        "enumerate": enumerate,
        **constants,
    }
    return eval(code, namespace)


_MODULE_CONSTANT_CACHE: dict[str, dict[str, Any] | None] = {}
_EVAL_CONTEXT_CACHE: dict[str, tuple[ast.Module, dict[str, Any]]] = {}


def _tests_module_constants(module: str, cache: dict[str, dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """Best-effort constants of a repository module (tests or jqmc) named in an import."""
    cache = _MODULE_CONSTANT_CACHE
    if module in cache:
        return cache[module]
    path = ROOT / (module.replace(".", "/") + ".py")
    constants: dict[str, Any] | None = None
    if (module.startswith(("tests.", "jqmc.")) or module == "jqmc") and path.is_file():
        cache[module] = None  # recursion guard while resolving this module's imports
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        package = module.rsplit(".", 1)[0] if "." in module else None
        constants = _module_constants(tree, seed=_import_bindings(tree, package=package))
    cache[module] = constants
    return constants


def _import_bindings(tree: ast.Module, package: str | None = None) -> dict[str, Any]:
    """Names bound by imports: tests-tree constants where resolvable, opaque shims otherwise."""
    bindings: dict[str, Any] = {}
    provided = {"itertools", "pytest", "range"}  # real objects in the eval namespace
    cache: dict[str, dict[str, Any]] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                if name not in provided:
                    bindings[name] = _NumpyShim() if alias.name == "numpy" else _Opaque(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                source = _tests_module_constants(node.module or "", cache)
            elif package:
                # Relative import inside the tests tree: resolve against the
                # importing file's package so sibling constants (case lists,
                # fixture tables) stay countable.
                source = _tests_module_constants(f"{package}.{node.module}" if node.module else package, cache)
            else:
                source = None
            for alias in node.names:
                name = alias.asname or alias.name
                if source is not None and alias.name in source:
                    bindings[name] = source[alias.name]
                elif name not in provided:
                    bindings[name] = _Opaque(f"{node.module}.{alias.name}" if node.module else alias.name)
    return bindings


def _length_preserving_helper(fn: ast.FunctionDef) -> bool:
    """True for one-parameter helpers that return exactly one row per input item.

    Recognized shapes: a single return of a comprehension over the parameter
    without filters, or ``rows = []; for item in param: rows.append(...);
    return rows`` with exactly one unconditional append.
    """
    params = [argument.arg for argument in fn.args.args]
    if len(params) != 1:
        return False
    body = [n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    if len(body) == 1 and isinstance(body[0], ast.Return):
        value = body[0].value
        if isinstance(value, (ast.ListComp, ast.GeneratorExp)) and len(value.generators) == 1:
            generator = value.generators[0]
            return not generator.ifs and isinstance(generator.iter, ast.Name) and generator.iter.id == params[0]
        return False
    if len(body) == 3 and isinstance(body[0], ast.Assign) and isinstance(body[1], ast.For) and isinstance(body[2], ast.Return):
        target = body[0].targets[0]
        loop, returned = body[1], body[2].value
        if not (isinstance(target, ast.Name) and isinstance(returned, ast.Name) and returned.id == target.id):
            return False
        if not (isinstance(loop.iter, ast.Name) and loop.iter.id == params[0]):
            return False
        if any(isinstance(n, (ast.If, ast.For, ast.While)) for n in loop.body):
            return False
        appends = [
            n
            for n in loop.body
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == "append"
            and isinstance(n.value.func.value, ast.Name)
            and n.value.func.value.id == target.id
        ]
        return len(appends) == 1
    return False


def _list_passthrough(values: Any, *args: Any, **kwargs: Any) -> list[Any]:
    return list(values)


def _nullary_return(fn: ast.FunctionDef) -> ast.AST | None:
    """The return expression of a zero-argument single-return helper, else None."""
    if fn.args.args or fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg:
        return None
    body = [n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None:
        return body[0].value
    return None


def _parameterized_return(fn: ast.FunctionDef) -> ast.AST | None:
    """The return expression of a side-effect-free single-return helper."""
    if fn.args.vararg is not None or fn.args.kwarg is not None:
        return None
    arguments = [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]
    if not arguments:
        return None
    body = [node for node in fn.body if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))]
    if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None:
        return body[0].value
    return None


class _NullaryHelper:
    """Callable shim deferring evaluation of a nullary helper's return expression."""

    def __init__(self, name: str, expression: ast.AST) -> None:
        self.gdd_expression = name
        self._expression = expression
        self._constants: dict[str, Any] = {}

    def bind(self, constants: dict[str, Any]) -> None:
        self._constants = constants

    def __call__(self) -> Any:
        try:
            return _literal(self._expression, self._constants)
        except Exception:
            return _Opaque("unresolved-helper-return")


class _ParameterizedReturnHelper:
    """Restricted evaluator for local helpers such as ``_p(x) -> (...)``."""

    def __init__(self, function: ast.FunctionDef, expression: ast.AST) -> None:
        self.gdd_expression = function.name
        self._function = function
        self._expression = expression
        self._constants: dict[str, Any] = {}

    def bind(self, constants: dict[str, Any]) -> None:
        self._constants = constants

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        positional = [*self._function.args.posonlyargs, *self._function.args.args]
        names = [argument.arg for argument in positional]
        required = len(positional) - len(self._function.args.defaults)
        if len(args) > len(positional):
            raise TypeError("too many positional arguments")
        bound = dict(zip(names, args, strict=False))
        for argument, default in zip(positional[required:], self._function.args.defaults, strict=True):
            if argument.arg not in bound:
                bound[argument.arg] = _literal(default, self._constants)
        bound.update(kwargs)
        missing = [argument.arg for argument in positional[:required] if argument.arg not in bound]
        if missing:
            raise TypeError(f"missing helper arguments: {missing}")
        for argument, default in zip(self._function.args.kwonlyargs, self._function.args.kw_defaults, strict=True):
            if argument.arg not in bound:
                if default is None:
                    raise TypeError(f"missing keyword-only helper argument: {argument.arg}")
                bound[argument.arg] = _literal(default, self._constants)
        realized = _literal(self._expression, {**self._constants, **bound})
        if isinstance(realized, _Opaque):
            # Preserve a short, runtime-resolvable helper call in the human
            # ledger instead of expanding the helper's implementation into a
            # page-wide production-constructor expression.
            arguments = [*(repr(value) for value in args), *(f"{key}={value!r}" for key, value in kwargs.items())]
            return _Opaque(f"{self._function.name}({', '.join(arguments)})")
        return realized


def _module_constants(tree: ast.Module, seed: dict[str, Any] | None = None) -> dict[str, Any]:
    constants: dict[str, Any] = dict(seed or {})
    constants.setdefault("__file__", _Opaque("__file__"))
    constants.setdefault("__name__", _Opaque("__name__"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and _length_preserving_helper(node):
            constants.setdefault(node.name, _list_passthrough)
        elif isinstance(node, ast.FunctionDef) and _nullary_return(node) is not None:
            helper = _NullaryHelper(node.name, _nullary_return(node))
            helper.bind(constants)  # live reference: later constants are visible
            constants.setdefault(node.name, helper)
        elif isinstance(node, ast.FunctionDef) and _parameterized_return(node) is not None:
            helper = _ParameterizedReturnHelper(node, _parameterized_return(node))
            helper.bind(constants)  # live reference: later constants are visible
            constants.setdefault(node.name, helper)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            constants.setdefault(node.name, _Opaque(node.name))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if node.value is None:
            continue
        try:
            literal = _literal(node.value, constants)
        except Exception:
            if isinstance(node.value, ast.Dict) and all(
                isinstance(key, ast.Constant) for key in node.value.keys if key is not None
            ):
                literal = {key.value: None for key in node.value.keys if key is not None}
            else:
                continue
        for target in targets:
            if isinstance(target, ast.Name):
                constants[target.id] = literal
    return constants


def _find_function(
    tree: ast.Module, class_chain: list[str], function_name: str
) -> tuple[ast.FunctionDef, list[ast.ClassDef]] | None:
    body: list[ast.stmt] = tree.body
    class_nodes: list[ast.ClassDef] = []
    for class_name in class_chain:
        classes = [node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name]
        if len(classes) != 1:
            return None
        class_nodes.append(classes[0])
        body = classes[0].body
    matches = [node for node in body if isinstance(node, ast.FunctionDef) and node.name == function_name]
    return (matches[0], class_nodes) if len(matches) == 1 else None


def _fixture_param_multiplier(
    tree: ast.Module,
    class_nodes: list[ast.ClassDef],
    function: ast.FunctionDef,
    constants: dict[str, Any],
) -> int:
    """Collection multiplier from parametrized fixtures requested by the test.

    Walks the fixture-request closure (function args -> fixture args -> ...)
    across the module body and the enclosing class chain, multiplying the
    ``params`` lengths of every distinct parametrized fixture reached.
    """

    def fixture_defs(body: list[ast.stmt]) -> dict[str, tuple[ast.FunctionDef, ast.AST | None]]:
        found: dict[str, tuple[ast.FunctionDef, ast.AST | None]] = {}
        for node in body:
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                call = decorator if isinstance(decorator, ast.Call) else None
                name = _dotted_name(call.func if call else decorator)
                if not name.endswith("fixture"):
                    continue
                params = None
                if call is not None:
                    for keyword in call.keywords:
                        if keyword.arg == "params":
                            params = keyword.value
                found[node.name] = (node, params)
        return found

    fixtures = fixture_defs(tree.body)
    for class_node in class_nodes:
        fixtures.update(fixture_defs(class_node.body))

    seen: set[str] = set()
    multiplier = 1

    def visit(argument_names: list[str]) -> None:
        nonlocal multiplier
        for argument in argument_names:
            if argument in fixtures and argument not in seen:
                seen.add(argument)
                fixture_node, params = fixtures[argument]
                if params is not None:
                    multiplier *= len(_literal(params, constants))
                visit([a.arg for a in fixture_node.args.args if a.arg not in ("self", "request")])

    visit([a.arg for a in function.args.args if a.arg != "self"])
    return multiplier


def _parameter_case_count(function: ast.FunctionDef, constants: dict[str, Any], class_nodes: list[ast.ClassDef] = ()) -> int:
    count = 1
    decorators = [d for node in class_nodes for d in node.decorator_list] + list(function.decorator_list)
    for decorator in decorators:
        if not isinstance(decorator, ast.Call) or not _dotted_name(decorator.func).endswith(".parametrize"):
            continue
        if len(decorator.args) < 2:
            raise ValueError(f"line {decorator.lineno}: parametrize has no values")
        values = _literal(decorator.args[1], constants)
        count *= len(values)
    return count


def _actual_cases(pytest_node: str, errors: list[str]) -> int | None:
    parts = pytest_node.split("::")
    if len(parts) < 2:
        errors.append(f"invalid pytest node {pytest_node!r}")
        return None
    path = ROOT / parts[0]
    if not path.is_file():
        errors.append(f"missing evidence file {parts[0]}")
        return None
    key = parts[0]
    if key not in _EVAL_CONTEXT_CACHE:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        package = str(Path(key).parent).replace("/", ".")
        _EVAL_CONTEXT_CACHE[key] = (tree, _module_constants(tree, seed=_import_bindings(tree, package=package)))
    tree, constants = _EVAL_CONTEXT_CACHE[key]
    found = _find_function(tree, parts[1:-1], parts[-1])
    if found is None:
        errors.append(f"pytest function does not exist: {pytest_node}")
        return None
    function, class_nodes = found
    try:
        return _parameter_case_count(function, constants, class_nodes) * _fixture_param_multiplier(
            tree, class_nodes, function, constants
        )
    except Exception as exc:
        errors.append(f"{pytest_node}: cannot evaluate pytest parameter count: {exc}")
        return None


def _governed_ledgers() -> list[Path]:
    """Governed ledgers and their guarantee-section key.

    The derivative contract keeps its universe sections under its own schema;
    its ordinary module-test evidence lives in ``module_evidence`` and is
    pinned here exactly like every module ledger's ``guarantees``.
    """
    ledgers = []
    for path in sorted(GUARANTEE_DIR.glob("*.yaml")):
        document = load_ledger(path)
        if document.get("document", {}).get("lifecycle") == "placeholder":
            continue
        section = "module_evidence" if path.name == "derivatives.yaml" else "guarantees"
        ledgers.append((path, section))
    return ledgers


def check() -> tuple[list[str], int, int]:
    errors: list[str] = []
    verdicts = yaml.safe_load(VERDICTS_PATH.read_text(encoding="utf-8"))
    prefix_by_kind = {row["kind"]: row["prefix"] for row in verdicts["classes"]}
    methods_by_id = {row["id"]: row for row in verdicts["verdict_methods"]}

    ledger_count = 0
    guarantee_count = 0
    # The test/guarantee relation is many-to-many (gdd/README.md): a node may
    # be evidence of several guarantees, each with its own module contract
    # (module_receipts.load_node_contracts). Only the same node registered
    # twice under one guarantee is a duplicate.
    for path, section in _governed_ledgers():
        ledger_count += 1
        relative = path.relative_to(ROOT).as_posix()
        document = load_ledger(path)
        for guarantee in document.get(section, []):
            guarantee_count += 1
            label = f"{relative}: {guarantee.get('id')}"
            kind = guarantee.get("kind")
            if kind not in prefix_by_kind:
                errors.append(f"{label}: kind {kind!r} is not a guarantee class from the verdict contract")
            elif not str(guarantee.get("id", "")).startswith(prefix_by_kind[kind] + "-"):
                errors.append(f"{label}: ID prefix does not match its class ({kind} requires {prefix_by_kind[kind]}-)")
            declared_methods = guarantee.get("verdict_methods", [])
            if not declared_methods:
                errors.append(f"{label}: every guarantee must assign verdict methods")
            for method_id in declared_methods:
                method = methods_by_id.get(method_id)
                if method is None:
                    errors.append(f"{label}: unknown verdict method {method_id}")
                elif kind not in method.get("classes", []):
                    errors.append(f"{label}: verdict method {method_id} is not legal for class {kind!r}")
            total = 0
            evidence = guarantee.get("required_evidence")
            if not isinstance(evidence, list) or not evidence:
                errors.append(f"{label}: required_evidence must be a nonempty list of verdict-method groups")
                continue
            seen_nodes: set[str] = set()
            for group in evidence:
                method_id = group.get("verdict_method")
                if method_id not in declared_methods:
                    errors.append(f"{label}: evidence group uses undeclared verdict method {method_id!r}")
                for node in group.get("nodes", []):
                    pytest_node = node.get("pytest_node", "")
                    if pytest_node in seen_nodes:
                        errors.append(f"{label}: pytest node registered twice in this guarantee: {pytest_node}")
                    seen_nodes.add(pytest_node)
                    errors.extend(
                        f"{label}: {pytest_node}: {failure}"
                        for failure in parameter_case_errors(
                            node,
                            node.get("case_binding", {}).get("fixture", {}).get("concrete_cases"),
                            execution=node.get("case_binding", {}).get("execution"),
                        )
                    )
                    actual = _actual_cases(pytest_node, errors)
                    if actual is None:
                        continue
                    if node.get("expected_cases") != actual:
                        errors.append(
                            f"{label}: {pytest_node} expects {node.get('expected_cases')} cases, source defines {actual}"
                        )
                    total += actual
            if guarantee.get("scope", {}).get("expected_obligations") != total:
                errors.append(
                    f"{label}: scope.expected_obligations={guarantee.get('scope', {}).get('expected_obligations')} "
                    f"but the registered nodes expand to {total}"
                )
    return errors, ledger_count, guarantee_count


def main() -> int:
    # One parse of each ledger and derived binding per run.
    with ledger_cache():
        return _main()


def _main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    errors, ledgers, guarantees = check()
    if errors:
        print("GDD module-ledger check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"GDD module-ledger check passed ({ledgers} governed module ledgers, {guarantees} guarantees).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

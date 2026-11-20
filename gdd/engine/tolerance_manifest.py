#!/usr/bin/env python3
"""Static manifest of the central numerical settings (issue 227).

The protected-contract gate must reject a change that weakens a central
tolerance of ``jqmc/_setting.py`` or ``jqmc/_precision.py`` without a
reviewer's approval, while a tightening or an unrelated data edit passes. The strength
of every GDD guarantee is the tolerance its evidence compares with, so these
two modules are reviewed files; this module decides what in them is judged by
machine and what by a human.

Data is machine-judged. A *data binding* is a module-level store or delete
through names and subscripts whose value is built only from constants,
containers, arithmetic, comparisons, names and subscripts (``_DATA_NODES``):
``ATOL_X = 1e-8``, ``_TOLERANCE["strict"]["float64"] = (1e-8, 1e-6)``,
``FACTOR *= 0.5``, ``del ATOL_OLD``. Those statements are evaluated statically,
in order, by a small interpreter of exactly those forms with Python's own
semantics (aliases share storage, ``|=``/``+=`` mutate a dict or list in place;
nothing is imported, no code of the module runs), and every leaf value is
recorded under a stable key. The base's and the candidate's manifests are then compared: a tolerance,
ULP or condition-number budget may only tighten, a digit count or floor may
only grow, every other reviewed setting must stay equal, nothing may
disappear, and new entries are free. The direction follows the name
(``TOLERANCE_TOKENS``, ``PLACES_TOKENS``) or the documented override
(``SETTING_KINDS``).

Code is human-judged. Everything else in the two modules -- definitions,
lambdas, calls, attributes, imports, compound statements, ``:=``, ``global``
-- is code. The module's code shape (its AST without data bindings, docstrings
and annotations) is compared with the base's; if it differs, the gate reports
one line, ``<module>: settings code changed: <shape id>``, which stays red
until the change is reviewed and merged into the base. The id is the hash of the candidate's code shape. What
a resolver computes, in which
context, with which cache: that is what the reviewer reads in the diff.

Nothing here tries to decide from Python source what a value "really" is: an
unknown construct is code by definition and goes to review. Inline numeric
tolerance literals in tests stay forbidden (check_no_inline_tolerances), so
the tests consume these reviewed values.
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
import hashlib
import math
import numbers
import operator
import subprocess
from pathlib import Path
from typing import Any

SETTINGS_FILE = "jqmc/_setting.py"
PRECISION_FILE = "jqmc/_precision.py"
SETTING_MODULES = (SETTINGS_FILE, PRECISION_FILE)

# Name tokens that fix the direction of a numeric setting. A tolerance, ULP or
# condition-number budget admits more as it grows; a digit count admits more as
# it shrinks. Names whose role is inverted or ambiguous (a non-vacuity or
# distinctness threshold) are compared exactly, like every other setting.
TOLERANCE_TOKENS = frozenset({"ATOL", "RTOL", "TOL", "TOLERANCE", "MAXULP", "KAPPA"})
PLACES_TOKENS = frozenset({"DECIMAL", "SIGNIFICANT"})
EXACT_TOKENS = frozenset({"NONVACUITY", "DISTINCTNESS", "DISTINCTION"})
# Named budgets whose names carry no tolerance token but whose direction is
# documented at their definition. ``tolerance`` admits more as it grows (the
# coherent AO bias verdict is ``|kappa| <= fraction * eps + multiple * eps64``
# with the standard error admitted up to a fraction of that bound; the
# S-Stiefel check is ``gamma_k <= safety * operation * n * eps / (1 - ...)``).
# ``floor`` admits more as it shrinks (the kappa cap is
# ``rtol / ((slope * degree + offset) * eps)``; a normality test on fewer
# samples resolves less).
SETTING_KINDS: dict[str, str] = {
    "AO_VALUE_COHERENT_BIAS_EPS_FRACTION": "tolerance",
    "AO_VALUE_COHERENT_BIAS_FP64_EPS_MULTIPLE": "tolerance",
    "AO_VALUE_COHERENT_BIAS_STANDARD_ERROR_FRACTION": "tolerance",
    "S_STIEFEL_BACKWARD_ERROR_SAFETY_FACTOR": "tolerance",
    "S_STIEFEL_BACKWARD_ERROR_OPERATION_FACTOR": "tolerance",
    "EVALUATION_KAPPA_ROUNDING_OFFSET": "floor",
    "EVALUATION_KAPPA_ROUNDING_SLOPE": "floor",
    "RS_NORMAL_SANITY_N": "floor",
}


class ManifestError(Exception):
    """The settings of a tree cannot be read; the gate fails closed."""


# --- what is data, what is code -------------------------------------------------

# The only statements of a settings module that are data, not code: a store or
# delete through names and subscripts whose value is built from constants,
# containers, arithmetic, comparisons, names and subscripts. Everything else --
# definitions, lambdas, calls, attributes, imports, compound statements, ``:=``,
# ``global`` -- is code.
_DATA_NODES = (
    ast.Constant,
    ast.Tuple,
    ast.List,
    ast.Set,
    ast.Dict,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Name,
    ast.Subscript,
    ast.Slice,
    ast.Starred,
    ast.operator,
    ast.unaryop,
    ast.boolop,
    ast.cmpop,
    ast.expr_context,
)


def is_data_binding(statement: ast.stmt) -> bool:
    """Whether a module-level statement only binds data the manifest records (see ``_DATA_NODES``)."""
    if isinstance(statement, ast.Assign):
        parts: list[ast.AST] = [*statement.targets, statement.value]
    elif isinstance(statement, (ast.AugAssign, ast.AnnAssign)):
        parts = [statement.target, *([statement.value] if statement.value is not None else [])]
    elif isinstance(statement, ast.Delete):
        parts = list(statement.targets)
    else:
        return False
    return all(isinstance(node, _DATA_NODES) for part in parts for node in ast.walk(part))


class _CodeShape(ast.NodeTransformer):
    """The reviewed code of a settings module: without data bindings, docstrings and annotations."""

    def visit_Module(self, node: ast.Module) -> ast.AST:
        node.body = [statement for statement in node.body if not is_data_binding(statement)]
        return self._strip(node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.annotation = None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        node.annotation = ast.Constant(value=None)
        return self.generic_visit(node)

    def _strip(self, node: Any) -> Any:
        if hasattr(node, "returns"):
            node.returns = None
        body = list(node.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:] or [ast.Pass()]
        node.body = body
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._strip(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._strip(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return self._strip(node)


def code_shape(source: str) -> str:
    """A stable text of a settings module's reviewed code (raises SyntaxError on an unparsable module)."""
    return ast.dump(_CodeShape().visit(ast.parse(source)), include_attributes=False)


def code_shape_id(shape: str) -> str:
    return hashlib.sha256(shape.encode("utf-8")).hexdigest()[:12]


def settings_code_change_errors(before_source: str, after_source: str, relative: str) -> list[str]:
    """The reported line for a settings module whose reviewed code differs from the trusted base.

    Values are machine-judged by the manifest; code is human-judged: any change
    to the module outside its data bindings, docstrings and annotations needs
    a reviewer's approval, whatever it computes. The line names the candidate's code
    shape, so the approval covers exactly that code and a further change needs
    a further approval.
    """
    try:
        before, after = code_shape(before_source), code_shape(after_source)
    except SyntaxError as exc:
        return [f"cannot parse settings module {relative}: {exc}"]
    if before == after:
        return []
    return [f"{relative}: settings code changed: {code_shape_id(after)}"]


# --- static evaluation of the data bindings ------------------------------------------


class Ref:
    """A value the static evaluator cannot compute: a name bound by code, or an expression on one.

    Recorded as the text of the expression and compared exactly, so an alias of
    a function or of an imported object is a reviewed setting whose text may
    not change; the code behind it is reviewed through the code shape.
    """

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return f"<{self.text}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Ref) and other.text == self.text

    def __hash__(self) -> int:
        return hash(self.text)


_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitAnd: operator.and_,
    ast.BitXor: operator.xor,
}
# ``x op= y`` calls the in-place operator: it mutates a mutable ``x`` (a dict or list
# reached through any alias) and returns it, or returns a new immutable value.
_INPLACE = {
    ast.Add: operator.iadd,
    ast.Sub: operator.isub,
    ast.Mult: operator.imul,
    ast.Div: operator.itruediv,
    ast.FloorDiv: operator.ifloordiv,
    ast.Mod: operator.imod,
    ast.Pow: operator.ipow,
    ast.LShift: operator.ilshift,
    ast.RShift: operator.irshift,
    ast.BitOr: operator.ior,
    ast.BitAnd: operator.iand,
    ast.BitXor: operator.ixor,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_, ast.Invert: operator.invert}
_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


def _root(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Subscript, ast.Starred)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


class _Evaluator:
    """Evaluates the data bindings of a module in order; anything it cannot compute becomes a ``Ref``."""

    def __init__(self) -> None:
        self.env: dict[str, Any] = {}

    # -- expressions ---------------------------------------------------------------

    def value(self, node: ast.expr) -> Any:
        try:
            return self._value(node)
        except Exception:  # noqa: BLE001 - the evaluator is total; what it cannot compute is a reviewed text
            return Ref(ast.unparse(node))

    def _value(self, node: ast.expr) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.env[node.id] if node.id in self.env else Ref(node.id)
        if isinstance(node, ast.Tuple):
            return tuple(self._sequence(node.elts))
        if isinstance(node, ast.List):
            return self._sequence(node.elts)
        if isinstance(node, ast.Set):
            return set(self._sequence(node.elts))
        if isinstance(node, ast.Dict):
            result: dict[Any, Any] = {}
            for key, value in zip(node.keys, node.values, strict=True):
                if key is None:
                    spread = self._value(value)
                    if not isinstance(spread, dict):
                        raise TypeError("** of a non-dict")
                    result.update(spread)
                else:
                    result[self._value(key)] = self._value(value)
            return result
        if isinstance(node, ast.Subscript):
            container = self._value(node.value)
            index = self._value(node.slice)
            if isinstance(container, Ref) or isinstance(index, Ref):
                raise TypeError("subscript of a reference")
            return container[index]
        if isinstance(node, ast.Slice):
            parts = [self._value(part) if part is not None else None for part in (node.lower, node.upper, node.step)]
            return slice(*parts)
        if isinstance(node, ast.BinOp):
            left, right = self._value(node.left), self._value(node.right)
            if isinstance(left, Ref) or isinstance(right, Ref):
                raise TypeError("arithmetic on a reference")
            return _BINARY[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp):
            operand = self._value(node.operand)
            if isinstance(operand, Ref):
                raise TypeError("arithmetic on a reference")
            return _UNARY[type(node.op)](operand)
        if isinstance(node, ast.BoolOp):
            values = [self._value(value) for value in node.values]
            if any(isinstance(value, Ref) for value in values):
                raise TypeError("boolean operation on a reference")
            result = values[0]
            for value in values[1:]:
                result = (result and value) if isinstance(node.op, ast.And) else (result or value)
            return result
        if isinstance(node, ast.Compare):
            left = self._value(node.left)
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                right = self._value(comparator)
                if isinstance(left, Ref) or isinstance(right, Ref):
                    raise TypeError("comparison with a reference")
                if not _COMPARE[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            test = self._value(node.test)
            if isinstance(test, Ref):
                raise TypeError("condition on a reference")
            return self._value(node.body) if test else self._value(node.orelse)
        raise TypeError(f"not a data expression: {type(node).__name__}")

    def _sequence(self, elements: list[ast.expr]) -> list[Any]:
        out: list[Any] = []
        for element in elements:
            if isinstance(element, ast.Starred):
                spread = self._value(element.value)
                if isinstance(spread, Ref):
                    raise TypeError("* of a reference")
                out.extend(spread)
            else:
                out.append(self._value(element))
        return out

    # -- statements ------------------------------------------------------------------

    def bind(self, statement: ast.stmt) -> None:
        if isinstance(statement, ast.Assign):
            value = self.value(statement.value)
            for target in statement.targets:
                self.store(target, value)
        elif isinstance(statement, ast.AnnAssign):
            if statement.value is not None:
                self.store(statement.target, self.value(statement.value))
        elif isinstance(statement, ast.AugAssign):
            current, operand = self.value(statement.target), self.value(statement.value)
            try:
                if isinstance(current, Ref) or isinstance(operand, Ref):
                    raise TypeError("augmented assignment on a reference")
                self.store(statement.target, _INPLACE[type(statement.op)](current, operand))
            except Exception:  # noqa: BLE001
                self.store(statement.target, Ref(ast.unparse(statement)))
        elif isinstance(statement, ast.Delete):
            for target in statement.targets:
                self.delete(target)

    def store(self, target: ast.expr, value: Any) -> None:
        if isinstance(target, ast.Name):
            self.env[target.id] = value
        elif isinstance(target, ast.Starred):
            self.store(target.value, value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            self._unpack(target, value)
        elif isinstance(target, ast.Subscript):
            container, index = self.value(target.value), self.value(target.slice)
            try:
                if isinstance(container, Ref) or isinstance(index, Ref):
                    raise TypeError("store through a reference")
                container[index] = value
            except Exception:  # noqa: BLE001
                self._mark(target, f"{ast.unparse(target)} <- {value!r}")

    def _unpack(self, target: ast.Tuple | ast.List, value: Any) -> None:
        if isinstance(value, Ref) or not isinstance(value, (tuple, list)):
            for element in target.elts:
                self.store(element, Ref(f"{ast.unparse(target)} <- {value!r}"))
            return
        items = list(value)
        starred = [index for index, element in enumerate(target.elts) if isinstance(element, ast.Starred)]
        if len(starred) == 1 and len(items) >= len(target.elts) - 1:
            position = starred[0]
            tail = len(target.elts) - position - 1
            for element, item in zip(target.elts[:position], items[:position], strict=True):
                self.store(element, item)
            self.store(target.elts[position], items[position : len(items) - tail])
            for element, item in zip(target.elts[position + 1 :], items[len(items) - tail :], strict=True):
                self.store(element, item)
        elif not starred and len(items) == len(target.elts):
            for element, item in zip(target.elts, items, strict=True):
                self.store(element, item)
        else:
            for element in target.elts:
                self.store(element, Ref(f"{ast.unparse(target)} <- {value!r}"))

    def delete(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self.env.pop(target.id, None)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self.delete(element)
        elif isinstance(target, ast.Subscript):
            container, index = self.value(target.value), self.value(target.slice)
            try:
                if isinstance(container, Ref) or isinstance(index, Ref):
                    raise TypeError("delete through a reference")
                del container[index]
            except Exception:  # noqa: BLE001
                self._mark(target, f"del {ast.unparse(target)}")

    def _mark(self, target: ast.AST, text: str) -> None:
        """A store or delete the evaluator cannot perform turns the whole binding into a reviewed text."""
        root = _root(target)
        if root is not None:
            self.env[root] = Ref(text)


def data_bindings(source: str) -> dict[str, Any]:
    """The values of a module's data bindings after its top-level data statements ran, in order."""
    evaluator = _Evaluator()
    for statement in ast.parse(source).body:
        if is_data_binding(statement):
            evaluator.bind(statement)
    return evaluator.env


# --- recording ------------------------------------------------------------------------


def value_kind(name: str) -> str:
    """``tolerance`` (may only shrink), ``places`` (may only grow) or ``setting`` (exact)."""
    tokens = set(name.upper().split("_")) - {""}
    if tokens & EXACT_TOKENS:
        return "setting"
    if tokens & PLACES_TOKENS:
        return "places"
    if tokens & TOLERANCE_TOKENS:
        return "tolerance"
    return "setting"


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


class _Members(list):
    """Marker for a set-valued binding: members are keys, so adding one is free."""


def _plain(value: Any) -> Any:
    """A JSON-like copy of a value: numbers, strings, booleans, None, lists, dicts, set members; references as text."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Ref):
        return repr(value)
    if _is_number(value):
        return int(value) if isinstance(value, numbers.Integral) else float(value)
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return _Members(sorted((_plain(item) for item in value), key=repr))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return repr(value)


def flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    """Record every leaf of ``value`` under ``prefix[key]...`` keys (set members as ``prefix{member}``)."""
    if isinstance(value, _Members):
        for member in value:
            out[f"{prefix}{{{member}}}"] = member
    elif isinstance(value, dict):
        for key in sorted(value):
            flatten(f"{prefix}[{key}]", value[key], out)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            flatten(f"{prefix}[{index}]", item, out)
    else:
        out[prefix] = value


def settings_manifest(source: str, relative: str) -> dict[str, dict[str, Any]]:
    """``{"<relative>::<leaf key>": {"kind", "value"}}`` for every data binding of a settings module."""
    try:
        bindings = data_bindings(source)
    except SyntaxError as exc:
        raise ManifestError(f"cannot parse settings module {relative}: {exc}") from exc
    entries: dict[str, dict[str, Any]] = {}
    for name, value in bindings.items():
        if name.startswith("__"):
            continue
        kind = SETTING_KINDS.get(name, value_kind(name))
        leaves: dict[str, Any] = {}
        flatten(name, _plain(value), leaves)
        for key, leaf in leaves.items():
            entries[f"{relative}::{key}"] = {"kind": kind if _is_number(leaf) else "setting", "value": leaf}
    return entries


# --- comparison -----------------------------------------------------------------------


def _violation(kind: str, before: Any, after: Any) -> str | None:
    """The kind whose direction ``after`` violates against ``before`` (or ``None`` when admissible)."""
    if _is_number(before) and _is_number(after):
        if math.isnan(before) or math.isnan(after):
            return None if math.isnan(before) and math.isnan(after) else kind
        if kind == "tolerance":
            return kind if after > before else None
        if kind in ("places", "floor"):
            return kind if after < before else None
        return kind if after != before else None
    return kind if type(before) is not type(after) or before != after else None


_LABELS = {
    "tolerance": ("tolerance loosened", "tolerance removed"),
    "places": ("decimal places reduced", "decimal places removed"),
    "floor": ("floor lowered", "floor removed"),
    "setting": ("reviewed setting changed", "reviewed setting removed"),
}


def manifest_errors(base: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]]) -> list[str]:
    """One line per weakening, change or removal; tightening and additions are free."""
    errors = []
    for key, before in sorted(base.items()):
        relative, _, display = key.partition("::")
        after = candidate.get(key)
        kind = before["kind"]
        if after is None:
            errors.append(f"{relative}: {_LABELS[kind][1]}: {display} {before['value']!r}")
            continue
        detail = f"{display} {before['value']!r} -> {after['value']!r}"
        if after["kind"] != kind:
            errors.append(f"{relative}: reviewed setting changed: {detail}")
            continue
        violated = _violation(kind, before["value"], after["value"])
        if violated:
            errors.append(f"{relative}: {_LABELS[violated][0]}: {detail}")
    return errors


# --- the gate ----------------------------------------------------------------------------


def _blob(base_ref: str, relative: str, git_cwd: Path) -> str | None:
    """The trusted base's copy of a file, or ``None`` when the base has no such file."""
    result = subprocess.run(["git", "show", f"{base_ref}:{relative}"], cwd=git_cwd, capture_output=True, check=False)
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        if "does not exist" in detail or "exists on disk, but not in" in detail:
            return None
        raise ManifestError(f"trusted-base settings {base_ref}:{relative} are unreadable: {detail}")
    return result.stdout.decode("utf-8")


def settings_contract_errors(base_ref: str, candidate_root: Path, *, git_cwd: Path) -> list[str]:
    """Compare the settings modules of the candidate with the trusted base.

    Returns the reported lines (``tolerance loosened``, ``reviewed setting
    changed``, ``... removed``, ``settings code changed: <id>``), or an
    ``cannot evaluate the settings manifest: ...`` line when a
    module cannot be read or parsed.
    """
    errors: list[str] = []
    for relative in SETTING_MODULES:
        path = candidate_root / relative
        try:
            before = _blob(base_ref, relative, git_cwd)
        except ManifestError as exc:
            return [f"cannot evaluate the settings manifest: {exc}"]
        if before is None:
            continue
        if not path.is_file():
            return [f"cannot evaluate the settings manifest: {relative} was deleted from the candidate"]
        after = path.read_text(encoding="utf-8")
        if after == before:
            continue
        try:
            base, candidate = settings_manifest(before, relative), settings_manifest(after, relative)
        except ManifestError as exc:
            return [f"cannot evaluate the settings manifest: {exc}"]
        errors.extend(manifest_errors(base, candidate))
        errors.extend(settings_code_change_errors(before, after, relative))
    return errors

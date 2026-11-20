#!/usr/bin/env python3
"""Pre-commit guard: reject local numerical comparison budgets.

The semantic pass rejects asserted residual upper bounds independently of their
magnitude or variable name, and explicit comparator budgets in tests and
production. It follows same-file aliases, arithmetic, literal argument packs,
and simple local helper returns. Unmodified central constants/helper results
remain accepted. Conventional reference magnitudes (including a unit scale
floor) are distinct from multiplying a tolerance by a local numeric factor.
Non-vacuity lower bounds, exact zero/sign assertions, finite-difference inputs,
and non-asserted fixture filters are not numerical error budgets.

This is bounded static analysis, not execution of arbitrary Python: unresolved
imported helper implementations and implicit comparator defaults are outside
its provenance proof. Production scanning is restricted to explicit comparator
tolerance arguments; it does not ban numerical algorithm/input parameters.
New residual-budget/tolerance-budget findings cannot be waived by ``tol-ok``.

Project policy (see the top-of-file comment of ``jqmc/_setting.py``): every
tolerance value used in
``tests/`` must come from a named constant or helper in ``jqmc/_setting.py``.
Inline ``pytest.approx(abs=1e-14)``, ``assert rel < 1e-8``, ``rtol = 1e-6``
(indirected into a later assert) etc. are all banned -- a bit-exact
assertion breaks mixed-precision runs, and the central constants in
``_setting.py`` carry the precision policy.

This is an AST-based checker (``ast.parse`` each file, no regex over source
text) so it cannot be fooled by a literal inside a comment or docstring, and
it also runs these four heuristic checks (they do not limit the semantic
pass described above):

  1. **Bare comparisons** ``expr < LITERAL`` / ``expr <= LITERAL`` (or the
     mirrored ``LITERAL > expr`` / ``LITERAL >= expr``) where ``LITERAL`` is
     a small (``0 < |literal| <= 1e-3``) float/int constant and ``expr``
     "reads like" an error metric -- its Name/Attribute identifiers contain
     a token from ``{resid, residual, rel, err, error, diff, delta, dev,
     norm, l2, maxabs, mismatch}``, OR it is a ``np.abs(...).max()``-style
     chain (both an ``abs`` and a ``max`` identifier appear in the chain).
     Only the "error must be below the bound" direction is flagged; the
     mirrored "value must be above a floor" idiom (``imag > 1e-3``,
     ``jastrow > 0.05``, a non-vacuity / genericity guard) is a different
     idiom and is deliberately NOT flagged even when it happens to use one
     of the tokens above, because the literal there is a floor, not a
     tolerance.
  2. **``pytest.approx`` / ``np.isclose`` / ``math.isclose`` calls** whose
     tolerance-position arguments are numeric literals: the ``abs=`` /
     ``rel=`` / ``atol=`` / ``rtol=`` / ``rel_tol=`` / ``abs_tol=`` keyword
     slots, and the positional ``rtol``/``atol`` slots of ``np.isclose`` /
     the positional ``rel_tol`` slot of ``math.isclose``. A bare
     ``pytest.approx(EXPECTED_LITERAL)`` with no tolerance kwarg is flagged
     only when ``EXPECTED_LITERAL`` itself is tolerance-sized
     (``<= 1e-3``) or the call sits directly inside a ``Compare`` whose
     other side reads like an error metric (see 1) -- otherwise it is
     benign metadata equality (``pytest.approx(75.0)`` checking an electron
     count) and is left alone.
  3. **Function-scope tolerance-named assignments** ``NAME = FLOATLIT``
     where ``NAME`` (split on ``_``) has a word-part exactly equal to one
     of ``{tol, atol, rtol, eps, threshold, tolerance}``, AND ``NAME`` is
     later referenced inside an ``assert`` in the *same* function (catches
     the ``tol = 1e-10`` / ``... assert rel < tol`` indirection pattern).
     A tolerance-named local that is only ever passed as a builder kwarg
     (``build_ewald_data(..., tol=ewald_tol)``) and never touches an
     ``assert`` is NOT flagged -- that is a legitimate builder parameter,
     not a comparison tolerance.
  4. **``atol=`` / ``rtol=`` / ``abs=`` keyword arguments** to ANY call
     (not just ``approx``/``isclose``) whose value is a numeric literal --
     an AST keyword-argument scan, so it cannot be triggered by a docstring that merely mentions ``atol=0``.

NaN agreement: ``equal_nan=``/``nan_ok=`` with anything but ``False``
(or a positional ``equal_nan``) is rejected unless the enclosing function is in
``tests/_nan_guard.py:NAN_AGREEMENT_ALLOWLIST``; ``# tol-ok`` does not apply.

Allowlist: ``# tol-ok: <reason>`` can suppress findings of the four heuristic
checks only, not semantic residual/tolerance budgets or NaN agreement.

Examples accepted when they contain no local budget rescaling:
  * Named imports / references: ``atol=ATOL_PYTHON_FP64_TIGHT`` (kwarg
    value is a ``Name``, not a numeric ``Constant``).
  * Helper return values: ``atol, rtol = get_tol_hdf5_roundtrip("strict")``
    (assignment target is a tuple, not a single tolerance-named ``Name``;
    and the value is a ``Call``, not a float ``Constant``).
  * Builder kwargs (``ewald_tol=1e-12``, ``tol=1e-9`` passed to a
    ``build_*``/projection function): these are keyword arguments to an
    arbitrary call, which class 4 only flags for the literal keywords
    ``atol``/``rtol``/``abs`` -- not ``tol``/``ewald_tol``/etc.
  * FD steps (``h = 1e-3``), seeds, ``dt``, ``scale``: none of these
    identifiers match the class-3 tolerance-name token set.
  * Comments / docstrings: the AST has no ``Constant`` nodes there at all.

History: this script was a bash + ``grep -nP`` one-liner, then a single
``re.compile`` regex over ``abs=``/``atol=``/``rtol=`` kwargs. The regex
missed every bare-comparison (``assert rel < 1e-8``) and indirected
(``tol = 1e-10; ...; assert x < tol``) tolerance literal in the suite --
this AST rewrite closes those gaps without regressing the kwarg check it
already had.
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
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared vocabulary
# ---------------------------------------------------------------------------

# Class 1: identifiers that "read like" an error metric.
_ERROR_TOKENS = frozenset(
    {
        "resid",
        "residual",
        "rel",
        "err",
        "error",
        "diff",
        "delta",
        "dev",
        "norm",
        "l2",
        "maxabs",
        "mismatch",
    }
)

# Class 3: identifiers that "read like" a tolerance variable.
_TOL_NAME_TOKENS = frozenset({"tol", "atol", "rtol", "eps", "threshold", "tolerance"})

# Class 1 magnitude gate; the semantic assertion pass has no
# magnitude cutoff and also rejects loosened large residual budgets.
_TOL_MAGNITUDE_MAX = 1.0e-3

# Class 4: kwarg names that are ALWAYS flagged when given a numeric literal,
# regardless of magnitude.
_TOLERANCE_KWARGS = frozenset({"atol", "rtol", "abs", "maxulp", "decimal", "significant"})

# Class 2: extra tolerance-position kwargs recognised on approx/isclose.
_APPROX_ISCLOSE_TOL_KWARGS = frozenset({"abs", "rel", "atol", "rtol", "rel_tol", "abs_tol"})

_CAMEL_RE = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")

USAGE_HINT = """
Project policy: every tolerance literal (comparison bound, pytest.approx /
np.isclose argument, or a tol/atol/rtol/eps-named local later used in an
assert) in tests/ must come from a named constant or helper in
`jqmc/_setting.py`.

To fix:
  1. Pick a fitting constant from `jqmc/_setting.py` (or add a new one
     with a short docstring explaining the use case).
  2. Import it into the test file:
        from jqmc._setting import ATOL_PYTHON_FP64_TIGHT  # etc.
  3. Replace the inline literal:
        atol=1e-15            ->   atol=ATOL_PYTHON_FP64_TIGHT
        assert rel < 1e-8     ->   assert rel < RTOL_SOME_NAMED_CONSTANT
        tol = 1e-10            ->   tol = RTOL_SOME_NAMED_CONSTANT

Residual/tolerance-budget findings cannot be suppressed with `# tol-ok`.
Preserve numerical values when centralizing a bound. A physical/fixture bound
should use a clearly named central physical constant, not pretend to be an
agreement tolerance. Activation lower bounds and finite-difference inputs are
distinct. Production files are checked for explicit comparator budgets only.
This bounded static checker does not prove arbitrary imported helper behavior
or implicit comparator defaults.

See the top-of-file comment of `jqmc/_setting.py` for the rule and
rationale.
"""


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _split_ident(name: str) -> list[str]:
    """Split a snake_case / camelCase identifier into lowercase word-parts."""
    out: list[str] = []
    for piece in name.split("_"):
        if not piece:
            continue
        sub = _CAMEL_RE.findall(piece)
        out.extend(s.lower() for s in (sub or [piece]))
    return out


def _float_value(node: ast.expr) -> float | None:
    """Return the literal numeric value of *node*, or None if not a literal."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _float_value(node.operand)
        if inner is None:
            return None
        return -inner if isinstance(node.op, ast.USub) else inner
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    return None


def _identifier_parts(node: ast.AST) -> list[str]:
    """All lowercase word-parts from every Name/Attribute inside *node*."""
    parts: list[str] = []
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            parts.extend(_split_ident(n.id))
        elif isinstance(n, ast.Attribute):
            parts.extend(_split_ident(n.attr))
    return parts


def _has_error_token(node: ast.AST) -> bool:
    return any(p in _ERROR_TOKENS for p in _identifier_parts(node))


def _has_abs_max_chain(node: ast.AST) -> bool:
    """True if an ``abs`` and a ``max`` identifier both appear in *node*.

    Catches ``np.abs(a - b).max()`` style error-metric expressions whose
    top-level Name is not itself an error-token, e.g. a bare ``.max()``.
    """
    names = {p for p in _identifier_parts(node)}
    return "abs" in names and "max" in names


def _call_target_name(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def _call_root_name(call: ast.Call) -> str | None:
    """The leftmost Name in a dotted call target.

    E.g. ``pytest`` in ``pytest.approx(...)``, or None for a bare
    ``approx(...)`` call (e.g. after ``from pytest import approx``).
    """
    f = call.func
    while isinstance(f, ast.Attribute):
        f = f.value
    if isinstance(f, ast.Name):
        return f.id
    return None


# ---------------------------------------------------------------------------
# Finding record
# ---------------------------------------------------------------------------


class Finding:
    __slots__ = ("cls", "detail", "lineno")

    def __init__(self, lineno: int, cls: str, detail: str) -> None:
        self.lineno = lineno
        self.cls = cls
        self.detail = detail


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.findings: list[Finding] = []
        # Set of FunctionDef/AsyncFunctionDef nodes already processed for
        # class 3, so nested functions are not double-scanned as part of
        # their enclosing function's body.
        self._scanned_funcs: set[int] = set()

    def visit_Module(self, node: ast.Module) -> None:
        semantic = _ToleranceDataflow()
        semantic.visit(node)
        self.findings.extend(semantic.findings)
        self.generic_visit(node)

    # -- class 5 (solver tol= feeding an asserted residual) -----------------
    def _check_solver_tol_kwargs(self, func: ast.FunctionDef) -> None:
        """Flag ``tol=<literal>`` solver kwargs in functions that assert on a
        variable assigned from the same call's result.

        The failure class: ``..., residual, _ = solve(...,
        tol=1.0e-11)`` followed by ``assert residual <= SYMBOL`` -- the request
        and the assertion silently diverge unless both come from one central
        symbol. Builder kwargs (``build_ewald_data(..., tol=...)``) whose
        results never reach an assert are untouched.
        """
        assigned_from_tol_call: dict[str, int] = {}
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            tol_kw = next(
                (
                    kw
                    for kw in node.value.keywords
                    if kw.arg in ("tol", "rel_tol", "abs_tol") and _float_value(kw.value) is not None
                ),
                None,
            )
            if tol_kw is None:
                continue
            for target in node.targets:
                elements = target.elts if isinstance(target, ast.Tuple) else [target]
                for element in elements:
                    if isinstance(element, ast.Name):
                        assigned_from_tol_call[element.id] = tol_kw.value.lineno
        if not assigned_from_tol_call:
            return
        for node in ast.walk(func):
            if not isinstance(node, ast.Assert):
                continue
            for name_node in ast.walk(node):
                if isinstance(name_node, ast.Name) and name_node.id in assigned_from_tol_call:
                    self.findings.append(
                        Finding(
                            assigned_from_tol_call[name_node.id],
                            5,
                            f"solver tol= literal feeds asserted result {name_node.id!r}; "
                            "use a jqmc._setting symbol for both the request and the assertion",
                        )
                    )
                    return

    # -- class 1 + class 2 (compare-context) --------------------------------
    def visit_Compare(self, node: ast.Compare) -> None:
        if len(node.ops) == 1:
            self._check_bare_comparison(node)
            self._check_approx_in_compare(node)
        self.generic_visit(node)

    def _check_bare_comparison(self, node: ast.Compare) -> None:
        op = node.ops[0]
        left, right = node.left, node.comparators[0]
        lit = _float_value(right)
        if lit is not None and isinstance(op, (ast.Lt, ast.LtE)):
            expr = left
        else:
            lit = _float_value(left)
            if lit is not None and isinstance(op, (ast.Gt, ast.GtE)):
                expr = right
            else:
                return
        if not (0 < abs(lit) <= _TOL_MAGNITUDE_MAX):
            return
        if _has_error_token(expr) or _has_abs_max_chain(expr):
            self.findings.append(Finding(node.lineno, "bare-compare", f"literal={lit!r}"))

    def _check_approx_in_compare(self, node: ast.Compare) -> None:
        sides = [node.left, *node.comparators]
        for i, side in enumerate(sides):
            if not (isinstance(side, ast.Call) and _call_target_name(side) == "approx"):
                continue
            if self._approx_has_tol_kwarg(side):
                continue  # already flagged by visit_Call
            if len(side.args) != 1:
                continue
            lit = _float_value(side.args[0])
            if lit is None:
                continue
            if 0 < abs(lit) <= _TOL_MAGNITUDE_MAX:
                continue  # already flagged by visit_Call
            others = [s for j, s in enumerate(sides) if j != i]
            if any(_has_error_token(o) or _has_abs_max_chain(o) for o in others):
                self.findings.append(Finding(side.lineno, "approx-sibling", "approx(literal) vs error-like sibling"))

    @staticmethod
    def _approx_has_tol_kwarg(call: ast.Call) -> bool:
        return any(kw.arg in ("abs", "rel") for kw in call.keywords)

    # -- class 2 (direct) + class 4 ------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        target = _call_target_name(node)
        root = _call_root_name(node)

        if target == "approx" and root in (None, "pytest"):
            self._check_approx_call(node)
        elif target == "isclose" and root in (None, "np", "numpy", "math"):
            self._check_isclose_call(node, is_math=(root == "math"))
        else:
            self._check_generic_tolerance_kwargs(node)
        if target == "assert_array_max_ulp" and len(node.args) > 2:
            limit = _float_value(node.args[2])
            if limit is not None:
                self.findings.append(Finding(node.args[2].lineno, "ulp-positional", f"maxulp={limit!r}"))
        if target in {"assert_almost_equal", "assert_array_almost_equal", "assert_approx_equal"} and len(node.args) > 2:
            limit = _float_value(node.args[2])
            if limit is not None:
                keyword = "significant" if target == "assert_approx_equal" else "decimal"
                self.findings.append(Finding(node.args[2].lineno, "precision-positional", f"{keyword}={limit!r}"))

        self.generic_visit(node)

    def _check_approx_call(self, node: ast.Call) -> None:
        for kw in node.keywords:
            if kw.arg in ("abs", "rel"):
                lit = _float_value(kw.value)
                if lit is not None:
                    self.findings.append(Finding(kw.value.lineno, "approx-kwarg", f"{kw.arg}={lit!r}"))
        if not node.args:
            return
        if any(kw.arg in ("abs", "rel") for kw in node.keywords):
            return  # kwarg literal already reported (or is a Name -> fine)
        if len(node.args) != 1:
            return
        lit = _float_value(node.args[0])
        if lit is not None and 0 < abs(lit) <= _TOL_MAGNITUDE_MAX:
            self.findings.append(Finding(node.args[0].lineno, "approx-expected", f"expected={lit!r}"))

    def _check_isclose_call(self, node: ast.Call, *, is_math: bool) -> None:
        tol_kwargs = ("rel_tol", "abs_tol") if is_math else ("rtol", "atol")
        for kw in node.keywords:
            if kw.arg in tol_kwargs:
                lit = _float_value(kw.value)
                if lit is not None:
                    self.findings.append(Finding(kw.value.lineno, "isclose-kwarg", f"{kw.arg}={lit!r}"))
        # positional tolerance slots: math.isclose(a, b, rel_tol, abs_tol);
        # np.isclose(a, b, rtol, atol)
        for idx in (2, 3):
            if idx < len(node.args):
                lit = _float_value(node.args[idx])
                if lit is not None:
                    self.findings.append(Finding(node.args[idx].lineno, "isclose-positional", f"arg[{idx}]={lit!r}"))

    def _check_generic_tolerance_kwargs(self, node: ast.Call) -> None:
        for kw in node.keywords:
            if kw.arg in _TOLERANCE_KWARGS:
                lit = _float_value(kw.value)
                if lit is not None:
                    self.findings.append(Finding(kw.value.lineno, "kwarg", f"{kw.arg}={lit!r}"))

    # -- class 3 --------------------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_solver_tol_kwargs(node)
        self._scan_function_scope_tolerances(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scan_function_scope_tolerances(node)
        self.generic_visit(node)

    def _scan_function_scope_tolerances(self, func: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        # A name can be (re-)assigned more than once in the same function
        # (e.g. an if/else ladder setting ``tol`` differently per branch) --
        # keep every assignment so each one is checked/reported independently.
        candidates: dict[str, list[ast.Assign]] = {}
        for stmt in ast.walk(func):
            if stmt is func:
                continue
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # nested function scans itself independently
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if not any(p in _TOL_NAME_TOKENS for p in _split_ident(target.id)):
                continue
            lit = _float_value(stmt.value)
            if lit is None:
                continue
            candidates.setdefault(target.id, []).append(stmt)

        if not candidates:
            return

        used_in_assert: set[str] = set()
        for stmt in ast.walk(func):
            if isinstance(stmt, ast.Assert):
                for n in ast.walk(stmt):
                    if isinstance(n, ast.Name) and n.id in candidates:
                        used_in_assert.add(n.id)

        for name in used_in_assert:
            for assign in candidates[name]:
                lit = _float_value(assign.value)
                self.findings.append(Finding(assign.lineno, "named-tol-assign", f"{name}={lit!r}"))


class _ToleranceDataflow(ast.NodeVisitor):
    """Bounded lexical tracking at numerical-comparison sinks, independent of scale.

    Upper bounds on absolute values, norms, differences and their reductions
    are residual budgets even when a variable is not named ``error``. Ordinary
    physical intervals, zero/sign tests, lower non-vacuity floors and FD steps
    are not such sinks. Absolute physical caps use explicit central physical
    constants, not comment exemptions; magnitude is never an exemption.

    This is deliberately not an arbitrary Python interpreter: imported policy
    constants and helper results are retained, while local aliases, literal
    containers/casts and arithmetic in tolerance positions are inspected.
    Existing checks and the separate fail-closed NaN policy remain unchanged.
    """

    def __init__(self, *, comparators_only: bool = False) -> None:
        self.findings: list[Finding] = []
        self.scopes: list[dict[str, list[ast.AST]]] = []
        self.assertions: list[set[int]] = []
        self.comparators_only = comparators_only

    def _lookup(self, name: str) -> list[ast.AST]:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return []

    @staticmethod
    def _bindings(node: ast.AST) -> dict[str, list[ast.AST]]:
        bindings: dict[str, list[ast.AST]] = {}

        def bind(target, value):
            if isinstance(target, ast.Name):
                bindings.setdefault(target.id, []).append(value)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for index, item in enumerate(target.elts):
                    bind(item, ast.Subscript(value=value, slice=ast.Constant(index), ctx=ast.Load()))

        pending = list(ast.iter_child_nodes(node))
        while pending:
            child = pending.pop()
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bindings.setdefault(child.name, []).append(child)
                continue
            if isinstance(child, (ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    bind(target, child.value)
            elif isinstance(child, (ast.AnnAssign, ast.NamedExpr)) and child.value is not None:
                bind(child.target, child.value)
            elif isinstance(child, ast.AugAssign):
                # Keep the literal contribution, rather than forgetting a later
                # scale change to an initially central tolerance.
                bind(child.target, child.value)
            elif isinstance(child, ast.ImportFrom):
                for alias in child.names:
                    if alias.asname and alias.asname != alias.name:
                        bind(ast.Name(id=alias.asname), ast.Name(id=alias.name))
                    else:
                        bindings.setdefault(alias.name, [])
            pending.extend(ast.iter_child_nodes(child))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = [*node.args.posonlyargs, *node.args.args]
            for argument in [*args, *node.args.kwonlyargs]:
                bindings.setdefault(argument.arg, [])
            for argument, default in zip(args[-len(node.args.defaults) :], node.args.defaults, strict=False):
                bind(ast.Name(id=argument.arg), default)
            for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True):
                if default is not None:
                    bind(ast.Name(id=argument.arg), default)
        return bindings

    def _enter(self, node):
        self.scopes.append(self._bindings(node))
        relevant: set[int] = set()

        def dependencies(part, seen=frozenset()):
            if id(part) in relevant:
                return
            seen = seen | {id(part)}
            relevant.add(id(part))
            for value in self._expand(part):
                if value is not part:
                    dependencies(value, seen)
            for value in ast.iter_child_nodes(part):
                dependencies(value, seen)

        pending = list(ast.iter_child_nodes(node))
        while pending:
            child = pending.pop()
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Assert):
                dependencies(child.test)
            pending.extend(ast.iter_child_nodes(child))
        self.assertions.append(relevant)
        self.generic_visit(node)
        self.assertions.pop()
        self.scopes.pop()

    visit_Module = _enter
    visit_FunctionDef = _enter
    visit_AsyncFunctionDef = _enter
    visit_ClassDef = _enter
    visit_Lambda = _enter

    def _expand(self, node: ast.AST, seen=frozenset()):
        if id(node) in seen:
            yield node
            return
        seen = seen | {id(node)}
        if isinstance(node, ast.Name) and node.id not in seen:
            values = self._lookup(node.id)
            if values:
                for value in values:
                    yield from self._expand(value, seen | {node.id})
                return
        if isinstance(node, ast.Subscript):
            for value in self._expand(node.value, seen):
                if isinstance(value, (ast.Tuple, ast.List)) and isinstance(node.slice, ast.Constant):
                    index = node.slice.value
                    if type(index) is int and -len(value.elts) <= index < len(value.elts):
                        yield from self._expand(value.elts[index], seen)
                        return
                if isinstance(value, ast.Dict) and isinstance(node.slice, ast.Constant):
                    for key, item in zip(value.keys, value.values, strict=True):
                        if isinstance(key, ast.Constant) and key.value == node.slice.value:
                            yield from self._expand(item, seen)
                            return
        yield node

    def _target(self, node: ast.AST) -> set[str]:
        return {
            value.id if isinstance(value, ast.Name) else value.attr
            for value in self._expand(node)
            if isinstance(value, (ast.Name, ast.Attribute))
        }

    def _reference_scale(self, node: ast.AST, seen=frozenset()) -> bool:
        """Recognize magnitude scales, not arbitrary numeric tolerance multipliers.

        A unit floor on an actual magnitude is the conventional relative-error
        scale. In contrast ``max(ATOL, 1)`` is an absolute budget replacement.
        Scientific values inside ``abs(reference)`` are not tolerance literals.
        """
        if id(node) in seen:
            return False
        seen = seen | {id(node)}
        if isinstance(node, ast.Name):
            # A zero-initialized running maximum of actual magnitudes remains
            # a reference scale; its self-reference is not a numeric budget.
            values = self._lookup(node.id)
            magnitudes = []
            for value in values:
                if _float_value(value) == 0:
                    continue
                if isinstance(value, ast.Call) and self._target(value.func) & {"max"}:
                    magnitudes.extend(
                        argument for argument in value.args if not (isinstance(argument, ast.Name) and argument.id == node.id)
                    )
                else:
                    break
            else:
                if magnitudes and all(self._reference_scale(item, seen) for item in magnitudes):
                    return True
        for value in self._expand(node):
            if value is not node and self._reference_scale(value, seen):
                return True
            if isinstance(value, ast.Call):
                names = self._target(value.func)
                if names & {"abs", "absolute", "fabs", "norm"} and value.args:
                    if not self._scalar_budget_expression(value.args[0]):
                        return True
                if names & {"max", "amax", "min", "amin", "mean", "float", "asarray"}:
                    operands = list(value.args)
                    if isinstance(value.func, ast.Attribute) and not value.args:
                        operands.append(value.func.value)
                    scales = [self._reference_scale(item, seen) for item in operands]
                    if any(scales) and all(scale or _float_value(item) == 1 for scale, item in zip(scales, operands)):
                        return True
            if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
                if (_float_value(value.left) == 1 and self._reference_scale(value.right, seen)) or (
                    _float_value(value.right) == 1 and self._reference_scale(value.left, seen)
                ):
                    return True
        return False

    def _scalar_budget_expression(self, node: ast.AST, seen=frozenset()) -> bool:
        """Do not mistake a literal/central budget wrapped in abs for data scale."""
        if id(node) in seen:
            return False
        seen = seen | {id(node)}
        for value in self._expand(node):
            if value is not node and self._scalar_budget_expression(value, seen):
                return True
            if _float_value(value) is not None:
                return True
            if isinstance(value, ast.Name) and any(token in _TOL_NAME_TOKENS for token in _split_ident(value.id)):
                return True
            if isinstance(value, (ast.BinOp, ast.UnaryOp)):
                if all(
                    self._scalar_budget_expression(child, seen)
                    for child in ast.iter_child_nodes(value)
                    if isinstance(child, ast.expr)
                ):
                    return True
            if isinstance(value, ast.Call) and self._target(value.func) & {"len"}:
                return True
            if isinstance(value, ast.Call) and self._target(value.func) & {
                "float",
                "int",
                "float32",
                "float64",
                "array",
                "asarray",
            }:
                if value.args and self._scalar_budget_expression(value.args[0], seen):
                    return True
        return False

    def _literal_budget(self, node: ast.AST, seen=frozenset()) -> bool:
        if id(node) in seen:
            return False
        if self._reference_scale(node):
            return False
        seen = seen | {id(node)}
        for value in self._expand(node):
            if value is not node:
                if self._literal_budget(value, seen):
                    return True
            elif _float_value(value) is not None:
                return True
            elif isinstance(value, (ast.BinOp, ast.UnaryOp, ast.IfExp, ast.Tuple, ast.List)):
                children = (value.body, value.orelse) if isinstance(value, ast.IfExp) else ast.iter_child_nodes(value)
                if any(self._literal_budget(child, seen) for child in children if isinstance(child, ast.expr)):
                    return True
            elif isinstance(value, ast.Call) and self._target(value.func) & {
                "float",
                "int",
                "float32",
                "float64",
                "asarray",
                "array",
                "max",
                "min",
                "abs",
                "absolute",
                "fabs",
            }:
                if any(self._literal_budget(argument, seen) for argument in value.args):
                    return True
            elif isinstance(value, ast.Call):
                if self._target(value.func) & {"len"}:
                    return True
                for function in self._expand(value.func):
                    if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                        scope = self._bindings(function)
                        for arg, supplied in zip((*function.args.posonlyargs, *function.args.args), value.args):
                            scope[arg.arg] = [supplied]
                        for keyword in value.keywords:
                            if keyword.arg is not None:
                                scope[keyword.arg] = [keyword.value]
                            else:
                                for pack in self._expand(keyword.value):
                                    if isinstance(pack, ast.Dict):
                                        for key, item in zip(pack.keys, pack.values):
                                            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                                                scope[key.value] = [item]
                        pending = [function.body] if isinstance(function, ast.Lambda) else list(function.body)
                        returns = []
                        while pending:
                            statement = pending.pop()
                            if isinstance(function, ast.Lambda) and statement is function.body:
                                returns.append(statement)
                            elif isinstance(statement, ast.Return) and statement.value is not None:
                                returns.append(statement.value)
                            elif not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                                pending.extend(ast.iter_child_nodes(statement))
                        self.scopes.append(scope)
                        try:
                            if any(self._literal_budget(expression, seen) for expression in returns):
                                return True
                        finally:
                            self.scopes.pop()
            elif isinstance(value, ast.Subscript) and isinstance(value.value, ast.Call):
                if self._literal_budget(value.value, seen):
                    return True
            elif (isinstance(value, ast.Attribute) and value.attr in {"size", "shape"}) or (
                isinstance(value, ast.Subscript) and isinstance(value.value, ast.Attribute) and value.value.attr == "shape"
            ):
                return True
        return False

    def _residual(self, node: ast.AST, seen=frozenset()) -> bool:
        if id(node) in seen:
            return False
        seen = seen | {id(node)}
        for value in self._expand(node):
            if value is not node and self._residual(value, seen):
                return True
            if isinstance(value, ast.Name) and any(part in _ERROR_TOKENS for part in _split_ident(value.id)):
                return True
            if isinstance(value, ast.Attribute) and any(part in _ERROR_TOKENS for part in _split_ident(value.attr)):
                return True
            if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Sub):
                return True
            if isinstance(value, ast.Call):
                names = self._target(value.func)
                if names & {"abs", "absolute", "fabs", "norm"}:
                    return True
                if not names & {
                    "max",
                    "amax",
                    "min",
                    "amin",
                    "sum",
                    "mean",
                    "sqrt",
                    "real",
                    "imag",
                    "float",
                    "asarray",
                    "array",
                    "all",
                    "any",
                }:
                    continue
                operands = list(value.args)
                if isinstance(value.func, ast.Attribute):
                    operands.append(value.func.value)
                if any(self._residual(part, seen) for part in operands):
                    return True
            elif isinstance(value, (ast.Attribute, ast.BinOp, ast.UnaryOp)):
                if any(self._residual(part, seen) for part in ast.iter_child_nodes(value) if isinstance(part, ast.expr)):
                    return True
        return False

    def _zero(self, node: ast.AST) -> bool:
        values = tuple(self._expand(node))
        return bool(values) and all(_float_value(value) == 0 for value in values)

    def _integer_grid(self, node: ast.AST, seen=frozenset()) -> bool:
        """Recognize index-grid support checks, not floating residual values."""
        if id(node) in seen:
            return False
        seen = seen | {id(node)}
        for value in self._expand(node):
            if value is not node and self._integer_grid(value, seen):
                return True
            if isinstance(value, ast.Subscript) and self._integer_grid(value.value, seen):
                return True
            if isinstance(value, ast.Call):
                names = self._target(value.func)
                if names & {"arange", "indices"}:
                    return True
                if names & {"abs", "absolute", "max", "amax", "min", "amin", "stack", "meshgrid", "reshape"}:
                    operands = list(value.args)
                    if isinstance(value.func, ast.Attribute) and value.func.attr == "reshape":
                        operands.append(value.func.value)
                    if any(self._integer_grid(item, seen) for item in operands):
                        return True
        return False

    def _integer_extent(self, node: ast.AST, seen=frozenset()) -> bool:
        if id(node) in seen:
            return False
        seen = seen | {id(node)}
        for value in self._expand(node):
            if value is not node and self._integer_extent(value, seen):
                return True
            if isinstance(value, ast.Call) and self._target(value.func) & {"int"} and value.args:
                argument = value.args[0]
                if isinstance(argument, ast.Call) and self._target(argument.func) & {"ceil", "floor"}:
                    return True
            if isinstance(value, ast.BinOp) and isinstance(value.op, (ast.Add, ast.Sub)):
                if self._integer_extent(value.left, seen) and type(_float_value(value.right)) in (float, int):
                    return True
        return False

    def visit_Compare(self, node):
        # Each link is directional: a lower activation floor is not an error
        # budget merely because its magnitude resembles a numerical tolerance.
        if self.comparators_only or not any(id(node) in scope for scope in self.assertions):
            self.generic_visit(node)
            return
        for left, op, right in zip((node.left, *node.comparators), node.ops, node.comparators):
            if isinstance(op, (ast.Lt, ast.LtE)):
                expression, bound = left, right
            elif isinstance(op, (ast.Gt, ast.GtE)):
                expression, bound = right, left
            else:
                continue
            index_support = self._integer_extent(bound) and self._integer_grid(expression)
            if self._residual(expression) and self._literal_budget(bound) and not self._zero(bound) and not index_support:
                self.findings.append(
                    Finding(node.lineno, "residual-budget", "literal or locally rescaled upper residual bound")
                )
        self.generic_visit(node)

    def visit_Call(self, node):
        names = self._target(node.func)
        if self.comparators_only and not names & {
            "approx",
            "isclose",
            "allclose",
            "assert_allclose",
            "assert_array_max_ulp",
            "assert_almost_equal",
            "assert_array_almost_equal",
            "assert_approx_equal",
        }:
            self.generic_visit(node)
            return
        extra = _APPROX_ISCLOSE_TOL_KWARGS if names & {"approx", "isclose", "allclose", "assert_allclose"} else frozenset()
        for keyword in node.keywords:
            if keyword.arg in _TOLERANCE_KWARGS | extra and self._literal_budget(keyword.value):
                self.findings.append(
                    Finding(node.lineno, "tolerance-budget", f"{keyword.arg}= contains a local numeric budget")
                )
            elif keyword.arg is None:
                for pack in self._expand(keyword.value):
                    if isinstance(pack, ast.Dict):
                        for key, value in zip(pack.keys, pack.values, strict=True):
                            if (
                                isinstance(key, ast.Constant)
                                and key.value in _TOLERANCE_KWARGS | extra
                                and self._literal_budget(value)
                            ):
                                self.findings.append(
                                    Finding(
                                        node.lineno,
                                        "tolerance-budget",
                                        f"**kwargs {key.value}= contains a local numeric budget",
                                    )
                                )
        slots = (2, 3) if names & {"isclose", "allclose", "assert_allclose"} else ()
        if names & {"assert_array_max_ulp", "assert_almost_equal", "assert_array_almost_equal", "assert_approx_equal"}:
            slots = (2,)
        arguments = []
        for argument in node.args:
            if isinstance(argument, ast.Starred):
                packs = list(self._expand(argument.value))
                if len(packs) == 1 and isinstance(packs[0], (ast.Tuple, ast.List)):
                    arguments.extend(packs[0].elts)
                else:
                    arguments.append(argument)
            else:
                arguments.append(argument)
        for index in slots:
            if len(arguments) > index and self._literal_budget(arguments[index]):
                self.findings.append(
                    Finding(node.lineno, "tolerance-budget", f"positional tolerance {index} contains a local numeric budget")
                )
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _is_under_tests(path: Path) -> bool:
    """``path`` is somewhere under a ``tests/`` directory at any depth."""
    return "tests" in path.parts


def _is_allowlist_line(line_text: str) -> bool:
    return "# tol-ok:" in line_text or "#tol-ok:" in line_text


def _statement_ranges(tree: ast.AST) -> list[tuple[int, int]]:
    """``(start_lineno, end_lineno)`` for every statement in *tree*.

    Used to find the smallest enclosing statement of a finding so the
    allowlist comment can be looked up anywhere in that statement's line
    span -- not just the single physical line the literal sits on. This
    matters because ``ruff format`` may wrap a long ``pytest.approx(...)``
    call across lines, moving the trailing ``# tol-ok:`` comment (attached
    to the closing paren) onto a different physical line than the literal
    argument itself.
    """
    out: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt):
            end = getattr(node, "end_lineno", None)
            if end is not None:
                out.append((node.lineno, end))
    return out


def _allowlisted(lines: list[str], finding_lineno: int, stmt_ranges: list[tuple[int, int]]) -> bool:
    # Smallest enclosing statement range containing the finding's line.
    enclosing = [r for r in stmt_ranges if r[0] <= finding_lineno <= r[1]]
    end = min((r[1] for r in enclosing), default=finding_lineno)
    end = min(end, len(lines))
    return any(_is_allowlist_line(lines[lineno - 1]) for lineno in range(finding_lineno, end + 1))


def _scan(path: Path, *, production: bool = False) -> list[tuple[int, str, Finding]]:
    """Return ``[(lineno, line_text, finding), ...]`` for every violation."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return []
    lines = text.splitlines()
    visitor = _ToleranceDataflow(comparators_only=True) if production else _Visitor()
    visitor.visit(tree)
    stmt_ranges = _statement_ranges(tree)

    hits: list[tuple[int, str, Finding]] = []
    seen_lines: set[int] = set()
    for finding in sorted(visitor.findings, key=lambda f: f.lineno):
        if finding.lineno in seen_lines:
            continue  # collapse multiple findings on one physical line
        line_text = lines[finding.lineno - 1] if 0 < finding.lineno <= len(lines) else ""
        if finding.cls not in {"residual-budget", "tolerance-budget"} and _allowlisted(lines, finding.lineno, stmt_ranges):
            continue
        seen_lines.add(finding.lineno)
        hits.append((finding.lineno, line_text, finding))
    return hits


# ---------------------------------------------------------------------------
# NaN-agreement opt-ins
# ---------------------------------------------------------------------------
#
# ``tests/_nan_guard.py`` makes NaN == NaN a failure in every numpy.testing
# comparison at run time. This static half rejects any NEW spelling that asks for
# NaN agreement -- ``equal_nan=`` / ``nan_ok=`` with anything but the literal
# ``False``, or a positional ``equal_nan`` slot -- unless the enclosing function is
# a reviewed entry of ``NAN_AGREEMENT_ALLOWLIST`` in that module. ``# tol-ok``
# comments do NOT suppress these findings.

NAN_GUARD_MODULE = Path(__file__).resolve().parents[1] / "tests" / "_nan_guard.py"
_NAN_OPT_IN_KWARGS = frozenset({"equal_nan", "nan_ok"})
# Positional index of ``equal_nan`` in the comparators that accept it.
_NAN_POSITIONAL_SLOT = {"assert_allclose": 4, "allclose": 4, "isclose": 4, "array_equal": 2}
# Unlike numpy.testing assertions, these comparators have no runtime NaN guard.
# Their default is safe, but an unresolved argument pack can override it.
_NAN_UNWRAPPED_COMPARATORS = frozenset({"allclose", "isclose", "array_equal", "approx"})


def load_nan_allowlist(module: Path | None = None) -> dict[tuple[str, str], str]:
    """Read the literal ``NAN_AGREEMENT_ALLOWLIST`` without importing the module (fail closed)."""
    module = NAN_GUARD_MODULE if module is None else module
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == "NAN_AGREEMENT_ALLOWLIST" for t in targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict) or not all(
                isinstance(k, tuple)
                and len(k) == 2
                and all(isinstance(part, str) and part for part in k)
                and isinstance(v, str)
                and v.strip()
                for k, v in value.items()
            ):
                raise ValueError(f"{module}: NAN_AGREEMENT_ALLOWLIST must map (file, qualname) to a non-empty reason")
            return value
    raise ValueError(f"{module}: NAN_AGREEMENT_ALLOWLIST literal not found")


class _NanOptInVisitor(ast.NodeVisitor):
    """Collect ``(lineno, qualname, detail)`` for every NaN-agreement opt-in call."""

    def __init__(self) -> None:
        self.scope: list[str] = []
        self.findings: list[tuple[int, str, str]] = []
        self.comparator_aliases: dict[str, str] = {}

    def _comparator_name(self, expression: ast.expr) -> str | None:
        if isinstance(expression, ast.Attribute):
            return expression.attr
        if isinstance(expression, ast.Name):
            return self.comparator_aliases.get(expression.id, expression.id)
        return None

    def prime_comparator_aliases(self, tree: ast.AST) -> None:
        """Discover static aliases before examining deferred function bodies.

        Module imports/assignments may occur after a test definition but before
        its execution. Use a monotone, conservative file-wide alias inventory;
        unresolved rebinding never erases a possible comparator identity.
        """
        nodes = tuple(ast.walk(tree))
        previous_size = -1
        while len(self.comparator_aliases) != previous_size:
            previous_size = len(self.comparator_aliases)
            for node in nodes:
                if isinstance(node, ast.ImportFrom):
                    self.visit_ImportFrom(node)
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    name = self._comparator_name(node.value)
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    if name in _NAN_UNWRAPPED_COMPARATORS:
                        for target in targets:
                            if isinstance(target, ast.Name):
                                self.comparator_aliases.setdefault(target.id, name)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            if alias.name in _NAN_UNWRAPPED_COMPARATORS:
                self.comparator_aliases.setdefault(alias.asname or alias.name, alias.name)

    def visit_Assign(self, node):
        name = self._comparator_name(node.value)
        if name in _NAN_UNWRAPPED_COMPARATORS:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.comparator_aliases.setdefault(target.id, name)
        # Keep previously identified aliases conservatively on rebinding. This
        # scanner is not a general Python data-flow interpreter; it must not
        # silently exempt a comparator because of an unresolved assignment.
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        name = self._comparator_name(node.value)
        if isinstance(node.target, ast.Name) and name in _NAN_UNWRAPPED_COMPARATORS:
            self.comparator_aliases.setdefault(node.target.id, name)
        self.generic_visit(node)

    @staticmethod
    def _safe_keyword_pack(value: ast.expr) -> bool:
        """Only a literal mapping with known safe keys can bypass runtime inspection."""
        if not isinstance(value, ast.Dict):
            return False
        for key, item in zip(value.keys, value.values, strict=True):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                return False
            if key.value in _NAN_OPT_IN_KWARGS and not (isinstance(item, ast.Constant) and item.value is False):
                return False
        return True

    def _enter(self, node, name: str, is_function: bool) -> None:
        self.scope.append(name)
        if is_function:
            self.scope.append("<locals>")
        self.generic_visit(node)
        if is_function:
            self.scope.pop()
        self.scope.pop()

    def visit_FunctionDef(self, node):
        self._enter(node, node.name, True)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self._enter(node, node.name, False)

    def visit_Lambda(self, node):
        self._enter(node, "<lambda>", True)

    def _qualname_of_call(self) -> str:
        # co_qualname of the frame executing the call: drop the trailing "<locals>".
        parts = self.scope[:-1] if self.scope and self.scope[-1] == "<locals>" else self.scope
        return ".".join(parts) if parts else "<module>"

    def visit_Call(self, node):
        name = self._comparator_name(node.func)
        for keyword in node.keywords:
            if keyword.arg in _NAN_OPT_IN_KWARGS and not (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is False
            ):
                self.findings.append((node.lineno, self._qualname_of_call(), f"{keyword.arg}={ast.unparse(keyword.value)}"))
            elif keyword.arg is None and name in _NAN_UNWRAPPED_COMPARATORS and not self._safe_keyword_pack(keyword.value):
                self.findings.append(
                    (node.lineno, self._qualname_of_call(), f"unresolved **kwargs on {name}: {ast.unparse(keyword.value)}")
                )
        if name in _NAN_UNWRAPPED_COMPARATORS and any(isinstance(arg, ast.Starred) for arg in node.args):
            self.findings.append((node.lineno, self._qualname_of_call(), f"unresolved *args on {name}"))
        slot = _NAN_POSITIONAL_SLOT.get(name or "")
        if slot is not None and len(node.args) > slot:
            value = node.args[slot]
            if not (isinstance(value, ast.Constant) and value.value is False):
                self.findings.append((node.lineno, self._qualname_of_call(), f"positional equal_nan={ast.unparse(value)}"))
        self.generic_visit(node)


def nan_opt_in_findings(path: Path, allowlist: dict[tuple[str, str], str], relative: str | None = None) -> list[str]:
    """Return one message per NaN-agreement opt-in in *path* not covered by *allowlist*."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []
    if relative is None:
        try:
            relative = path.resolve().relative_to(NAN_GUARD_MODULE.parents[1]).as_posix()
        except ValueError:
            relative = path.as_posix()
    visitor = _NanOptInVisitor()
    visitor.prime_comparator_aliases(tree)
    visitor.visit(tree)
    return [
        f"{relative}:{lineno}: NaN-agreement opt-in {detail} in {qualname} is not in "
        "tests/_nan_guard.py:NAN_AGREEMENT_ALLOWLIST"
        for lineno, qualname, detail in visitor.findings
        if (relative, qualname) not in allowlist
    ]


def main(argv: list[str]) -> int:
    # The commit hook must enforce the same repository-wide contract as CI's
    # ``pre-commit run --all-files``.  With pre-commit's default filename
    # passing, an existing violation in a file brought in by a merge could be
    # missed whenever that file was not part of the current staged diff.
    if not argv:
        argv = [str(path) for root in ("tests", "jqmc", "jqmc_workflow") for path in sorted(Path(root).rglob("*.py"))]
        argv.extend(str(path) for path in sorted(Path("jqmc/rs/src").rglob("*.rs")))

    bad = False
    try:
        nan_allowlist = load_nan_allowlist()
    except (OSError, SyntaxError, ValueError) as exc:
        print(f"ERROR: cannot read the NaN-agreement allow-list: {exc}", file=sys.stderr)
        return 1
    for arg in argv:
        path = Path(arg)
        if path.suffix == ".rs":
            # Also support direct invocation from an arbitrary working directory.
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from rust_tolerance_lint import scan_rust

            try:
                rust_hits = scan_rust(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                print(f"ERROR: cannot inspect Rust test budgets in {path}: {exc}", file=sys.stderr)
                bad = True
                continue
            for lineno, detail in rust_hits:
                print(f"ERROR: {path}:{lineno}: {detail}", file=sys.stderr)
                bad = True
            continue
        production = not _is_under_tests(path) and bool({"jqmc", "jqmc_workflow"} & set(path.parts))
        central = Path(__file__).resolve().parents[1] / "jqmc" / "_setting.py"
        if path.suffix != ".py" or (not _is_under_tests(path) and not production) or path.resolve() == central:
            continue
        nan_hits = [] if production else nan_opt_in_findings(path, nan_allowlist)
        if nan_hits:
            bad = True
            for message in nan_hits:
                print(f"ERROR: {message}", file=sys.stderr)
        hits = _scan(path, production=production)
        if not hits:
            continue
        bad = True
        print(f"ERROR: inline tolerance literal(s) found in {path}:", file=sys.stderr)
        for lineno, line, finding in hits:
            print(f"{lineno}:[{finding.cls}:{finding.detail}] {line}", file=sys.stderr)
    if bad:
        print(USAGE_HINT, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

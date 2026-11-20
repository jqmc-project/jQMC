"""Bounded source and runtime checks for centrally scaled NumPy spacing bounds.

The registry owns scientific symbols, case selectors and reference scales.  This
adapter understands only scalar addition and literal-table lookup; arbitrary
Python expressions and user-defined numeric objects are never evaluated.
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
import math
from typing import Any, Callable


POLICY = "CMP-SOURCE-SCALED-SPACING-001"
# NumPy is imported inside the runtime functions below, not at module scope:
# static GDD consumers import this module and must stay free of numerical
# imports (same pattern as inequality_receipts).


def _number(value: Any, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError("spacing values must be finite built-in scalars")
    return float(value)


def binding(definition: dict[str, Any]) -> dict[str, str]:
    """Validate the small registered adapter, not an arbitrary resolver."""
    result = definition.get("tolerance-binding")
    if (
        definition.get("id") != POLICY
        or definition.get("method") != "allclose"
        or not isinstance(result, dict)
        or set(result) != {"multiplier-symbol", "rtol-symbol", "scale-dtype", "spacing-function"}
        or result["scale-dtype"] != "float64"
        or result["spacing-function"] != "numpy.spacing"
        or any(
            not isinstance(result[key], str)
            or not result[key].startswith("jqmc._setting.")
            or not result[key].removeprefix("jqmc._setting.").isidentifier()
            for key in ("multiplier-symbol", "rtol-symbol")
        )
    ):
        raise ValueError("unsupported registered spacing adapter")
    return result


def _selector(value: Any) -> tuple:
    if not isinstance(value, dict) or not value or any(not isinstance(key, str) for key in value):
        raise ValueError("spacing selector must be a nonempty mapping")
    if any(type(item) not in (str, int, bool) for item in value.values()):
        raise ValueError("spacing selectors require string, integer or boolean values")
    return tuple((key, type(item).__name__, item) for key, item in sorted(value.items()))


def case_scales(disposition: dict[str, Any]) -> list[dict[str, Any]]:
    """Require a finite, unambiguous, same-axis table of reviewed scales."""
    if not isinstance(disposition, dict) or set(disposition) != {"policy", "case_scales"} or disposition["policy"] != POLICY:
        raise ValueError("spacing disposition requires policy and case_scales")
    rows = disposition["case_scales"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("spacing case scales must be nonempty")
    seen, axes = set(), None
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"selector", "scale"}:
            raise ValueError("spacing case requires selector and scale")
        key = _selector(row["selector"])
        if key in seen or (axes is not None and axes != set(row["selector"])):
            raise ValueError("duplicate or overlapping spacing case selectors")
        seen.add(key)
        axes = set(row["selector"])
        _number(row["scale"], positive=True)
    return rows


def selected_scale(disposition: dict[str, Any], selector: dict[str, Any]) -> float:
    rows = case_scales(disposition)
    if not isinstance(selector, dict):
        raise ValueError("spacing requires a concrete parameter case")
    matched = [
        row
        for row in rows
        if all(
            key in selector and type(selector[key]) is type(value) and selector[key] == value
            for key, value in row["selector"].items()
        )
    ]
    if len(matched) != 1:
        raise ValueError("spacing case is missing or ambiguous")
    return _number(matched[0]["scale"], positive=True)


def coverage_errors(disposition: dict[str, Any], selectors: list[dict[str, Any]]) -> list[str]:
    """All collected cases must match and no declared scale row may be dead."""
    try:
        rows = case_scales(disposition)
        if not isinstance(selectors, list) or not selectors:
            raise ValueError("spacing case universe is missing")
        for selector in selectors:
            selected_scale(disposition, selector)
        for row in rows:
            if not any(
                all(
                    key in case and type(case[key]) is type(value) and case[key] == value
                    for key, value in row["selector"].items()
                )
                for case in selectors
            ):
                raise ValueError("declared spacing case is absent from collection")
    except (KeyError, TypeError, ValueError) as exc:
        return [str(exc)]
    return []


def resolved_basis(disposition, definition, selector, symbol_reader: Callable[[str], Any]) -> dict[str, Any]:
    adapter = binding(definition)
    multiplier = _number(symbol_reader(adapter["multiplier-symbol"]), positive=True)
    rtol = _number(symbol_reader(adapter["rtol-symbol"]))
    if rtol < 0:
        raise ValueError("spacing relative tolerance must be nonnegative")
    return {
        "scale": selected_scale(disposition, selector),
        "multiplier": multiplier,
        "rtol": rtol,
        "function": "numpy.spacing",
        "dtype": "float64",
    }


def resolve(disposition, definition, selector, symbol_reader: Callable[[str], Any]) -> dict[str, float]:
    import numpy as np

    basis = resolved_basis(disposition, definition, selector, symbol_reader)
    atol = float(basis["multiplier"] * np.spacing(np.float64(basis["scale"])))
    if not math.isfinite(atol) or atol <= 0:
        raise ValueError("resolved spacing bound must be finite and positive")
    return {"atol": atol, "rtol": basis["rtol"]}


def _fields(value, operation):
    if (
        not isinstance(value, dict)
        or set(value) != {"kind", "operation", "fields"}
        or value["kind"] != "expression"
        or value["operation"] != operation
    ):
        raise ValueError("unsupported spacing source expression")
    return value["fields"]


def _description(value, selector, scope):
    if not isinstance(value, dict):
        raise ValueError("unresolved spacing source")
    if value.get("kind") == "literal":
        if set(value) != {"kind", "type", "value"}:
            raise ValueError("unsupported spacing literal")
        result = ast.literal_eval(value["value"])
        if type(result) not in (str, int, float, bool):
            raise ValueError("unsupported spacing literal type")
        return result
    if value.get("kind") == "parameter":
        if set(value) != {"kind", "scope", "name"} or value["scope"] != scope or value["name"] not in selector:
            raise ValueError("spacing reads an unowned parameter")
        result = selector[value["name"]]
        if type(result) not in (str, int, bool):
            raise ValueError("unsupported spacing parameter")
        return result
    if value.get("operation") == "BinOp":
        fields = _fields(value, "BinOp")
        if set(fields) != {"left", "op", "right"} or _fields(fields["op"], "Add") != {}:
            raise ValueError("spacing scale permits addition only")
        return _number(_description(fields["left"], selector, scope)) + _number(_description(fields["right"], selector, scope))
    if value.get("operation") == "Subscript":
        fields = _fields(value, "Subscript")
        if set(fields) != {"value", "slice"}:
            raise ValueError("unsupported spacing lookup")
        table = _fields(fields["value"], "Dict")
        if (
            set(table) != {"keys", "values"}
            or not isinstance(table["keys"], list)
            or not isinstance(table["values"], list)
            or len(table["keys"]) != len(table["values"])
        ):
            raise ValueError("unsupported spacing table")
        keys = []
        for key in table["keys"]:
            if not isinstance(key, dict) or key.get("kind") != "literal":
                raise ValueError("spacing table requires literal keys")
            key = _description(key, selector, scope)
            if type(key) not in (str, int, bool) or key in keys:
                raise ValueError("duplicate or unsupported spacing table key")
            keys.append(key)
        key = _description(fields["slice"], selector, scope)
        hits = [index for index, item in enumerate(keys) if type(key) is type(item) and item == key]
        if len(hits) != 1:
            raise ValueError("spacing table lacks the selected case")
        return _description(table["values"][hits[0]], selector, scope)
    raise ValueError("unsupported spacing scale source")


def validate_source(arguments, disposition, definition, selector, scope) -> None:
    """Match the unchanged source formula and its scale, not just the result."""
    adapter = binding(definition)
    if not isinstance(arguments, dict) or set(arguments) != {"atol", "rtol"}:
        raise ValueError("spacing requires explicit atol and rtol")
    if arguments["rtol"] != {"kind": "central-reference", "name": adapter["rtol-symbol"]}:
        raise ValueError("spacing relative tolerance must use its central symbol")
    fields = _fields(arguments["atol"], "BinOp")
    if (
        set(fields) != {"left", "op", "right"}
        or _fields(fields["op"], "Mult") != {}
        or fields["left"] != {"kind": "central-reference", "name": adapter["multiplier-symbol"]}
    ):
        raise ValueError("spacing multiplier must use its central symbol")
    call = fields["right"]
    if (
        not isinstance(call, dict)
        or set(call) != {"kind", "function", "args", "keywords"}
        or call["kind"] != "call"
        or call["function"] != {"kind": "imported-reference", "name": "numpy.spacing"}
        or call["keywords"] != []
        or not isinstance(call["args"], list)
        or len(call["args"]) != 1
    ):
        raise ValueError("spacing requires a direct NumPy spacing call")
    if _number(_description(call["args"][0], selector, scope), positive=True) != selected_scale(disposition, selector):
        raise ValueError("spacing source scale differs from its reviewed case")


def _runtime_value(expression, local, global_):
    if isinstance(expression, ast.Constant):
        value = expression.value
    elif isinstance(expression, ast.Name):
        namespace = local if expression.id in local else global_
        if expression.id not in namespace:
            raise ValueError("spacing runtime name is unavailable")
        value = namespace[expression.id]
    elif isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        return _number(_runtime_value(expression.left, local, global_)) + _number(
            _runtime_value(expression.right, local, global_)
        )
    elif isinstance(expression, ast.Subscript):
        table = _runtime_value(expression.value, local, global_)
        key = _runtime_value(expression.slice, local, global_)
        if type(table) is not dict or type(key) not in (str, int, bool):
            raise ValueError("spacing runtime lookup requires a built-in dictionary")
        if any(type(item) not in (str, int, bool) for item in table):
            raise ValueError("spacing runtime table has unsupported keys")
        matches = [item for item in table if type(item) is type(key) and item == key]
        if len(matches) != 1:
            raise ValueError("spacing runtime table lacks the selected case")
        value = table[matches[0]]
    else:
        raise ValueError("unsupported spacing runtime expression")
    if type(value) not in (str, int, float, bool, dict):
        raise ValueError("spacing runtime input is not a built-in scalar or table")
    return value


def capture(expression: str, local: dict[str, Any], global_: dict[str, Any]) -> dict[str, Any]:
    """Read the actual statement-entry scale without eval or arbitrary calls.

    The observation expression is snapshot-bound elsewhere.  Requiring direct
    names for the two central constants and the known NumPy function keeps this
    runtime reader narrower than Python's expression language.
    """
    import numpy as np

    try:
        call = ast.parse(expression, mode="eval").body
        if not isinstance(call, ast.Call) or any(item.arg is None for item in call.keywords):
            raise ValueError("spacing capture requires an explicit comparator call")
        kwargs = {item.arg: item.value for item in call.keywords}
        if len(kwargs) != len(call.keywords):
            raise ValueError("duplicate spacing comparator keyword")
        atol, rtol = kwargs["atol"], kwargs["rtol"]
        if (
            not isinstance(atol, ast.BinOp)
            or not isinstance(atol.op, ast.Mult)
            or not isinstance(atol.left, ast.Name)
            or not isinstance(rtol, ast.Name)
        ):
            raise ValueError("unsupported spacing runtime formula")
        spacing = atol.right
        if not isinstance(spacing, ast.Call) or len(spacing.args) != 1 or spacing.keywords:
            raise ValueError("unsupported spacing runtime call")
        if isinstance(spacing.func, ast.Name):
            fn = local.get(spacing.func.id, global_.get(spacing.func.id))
        elif isinstance(spacing.func, ast.Attribute) and isinstance(spacing.func.value, ast.Name):
            module = local.get(spacing.func.value.id, global_.get(spacing.func.value.id))
            if module is not np or spacing.func.attr != "spacing":
                raise ValueError("spacing runtime function is not NumPy spacing")
            fn = vars(module).get("spacing")
        else:
            raise ValueError("unsupported spacing runtime callable")
        if fn is not np.spacing:
            raise ValueError("spacing runtime function was substituted")
        multiplier = _number(_runtime_value(atol.left, local, global_), positive=True)
        scale = _number(_runtime_value(spacing.args[0], local, global_), positive=True)
        relative = _number(_runtime_value(rtol, local, global_))
        if relative < 0:
            raise ValueError("spacing runtime relative tolerance is negative")
        return {"scale": scale, "multiplier": multiplier, "rtol": relative, "function": "numpy.spacing", "dtype": "float64"}
    except (KeyError, TypeError, ValueError, SyntaxError, OverflowError) as exc:
        return {"capture_error": str(exc)}


def runtime_errors(record, disposition, definition, selector, symbol_reader) -> list[str]:
    """Require the measured input scale even when multiple scales share an ULP."""
    try:
        expected = resolved_basis(disposition, definition, selector, symbol_reader)
        if not isinstance(record, dict) or set(record) != set(expected):
            raise ValueError("spacing runtime basis is missing or malformed")
        for key in ("scale", "multiplier", "rtol"):
            if _number(record[key]) != expected[key]:
                raise ValueError(f"spacing runtime {key} differs from its reviewed basis")
        if any(record[key] != expected[key] for key in ("function", "dtype")):
            raise ValueError("spacing runtime function or dtype differs from its reviewed basis")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return [str(exc)]
    return []

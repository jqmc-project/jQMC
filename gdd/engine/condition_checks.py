"""Replay a declared conditioning assertion against its captured numeric inputs.

Only a small arithmetic language is accepted. No test-source ``eval``, imports,
object methods, or caller-supplied functions are executed by the receipt join.
Unknown syntax or missing data is a contract error, not successful evidence.
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
import keyword
import operator
from typing import Any, Mapping

import numpy as np


_FUNCTIONS = {
    "np.all": np.all,
    "np.max": np.max,
    "np.min": np.min,
    "np.abs": np.abs,
    "np.isfinite": np.isfinite,
    "max": max,
    "min": min,
    "abs": abs,
    "len": len,
}
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_COMPARISONS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_MAX_INPUT_SIZE = 65536


def condition_iteration_errors(declaration: Any, checks: Any, observations: Any, constraints: Any) -> list[str]:
    """Bind a repeated conditioning assertion to one authored comparison's iteration domain.

    The comparison owns the finite labels and numerical tolerance policy. A
    conditioning receipt must reuse that domain and budget, rather than
    declaring a second independently mutable allowance.
    """
    if declaration is None:
        return []
    if not isinstance(declaration, dict) or not declaration:
        return ["condition_check_iterations must be a nonempty mapping"]
    specs = {row["id"]: row for row in observations}
    errors = []
    for identity, link in declaration.items():
        try:
            if not isinstance(identity, str) or not isinstance(link, dict) or set(link) != {"comparison"}:
                raise ValueError("conditioning iteration requires an assertion identity and exactly comparison")
            if not isinstance(checks, dict) or not isinstance(link["comparison"], str) or link["comparison"] not in checks:
                raise ValueError("conditioning iteration references an unknown executable comparison")
            spec = specs.get(identity, {})
            if spec.get("expression") not in constraints or spec.get("required_execution") is not True:
                raise ValueError("conditioning iteration requires its mandatory conditioning assertion")
            if not {"atol", "rtol"} <= condition_symbols(spec["expression"]):
                raise ValueError("conditioning iteration assertion must consume both linked tolerance inputs")
            check = checks[link["comparison"]]
            iterations = check.get("iterations")
            if not isinstance(iterations, dict) or set(iterations) != {"variable", "required_values"}:
                raise ValueError("conditioning iteration requires the linked comparison's finite iterations")
            variable, values = iterations["variable"], iterations["required_values"]
            if (
                not isinstance(variable, str)
                or not variable.isidentifier()
                or keyword.iskeyword(variable)
                or not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) or not value for value in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError("conditioning iteration requires distinct nonempty string labels")
            matching = [
                observation
                for observation in observations
                if any(
                    c["site"] == check.get("site") for c in observation.get("tolerance_provenance", {}).get("comparisons", ())
                )
            ]
            if (
                len(matching) != 1
                or matching[0].get("required_execution") is not True
                or not spec.get("required_in_invocation")
                or spec["required_in_invocation"] != matching[0].get("required_in_invocation")
            ):
                raise ValueError("conditioning assertion and linked comparison require the same mandatory invocation")
            if check.get("comparator") != "numpy.testing.assert_allclose":
                raise ValueError("conditioning iteration requires an explicit atol/rtol allclose comparison")
        except (KeyError, TypeError, ValueError, SyntaxError) as exc:
            errors.append(f"{identity}: {exc}")
    return errors


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name(node.value)}.{node.attr}"
    raise ValueError("condition operand must be a name or a data attribute")


def condition_symbols(expression: str) -> set[str]:
    """List numeric inputs, excluding names used solely as allowed operators."""
    symbols: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.Name, ast.Attribute)):
            symbols.add(_name(node))
        elif isinstance(node, ast.Call):
            if _name(node.func) not in _FUNCTIONS or node.keywords:
                raise ValueError("unsupported function or keyword in conditioning assertion")
            for argument in node.args:
                visit(argument)
        elif isinstance(node, (ast.Expression, ast.Compare, ast.BinOp, ast.UnaryOp, ast.BoolOp)):
            for child in ast.iter_child_nodes(node):
                if not isinstance(child, (ast.cmpop, ast.operator, ast.unaryop, ast.boolop)):
                    visit(child)
        elif not isinstance(node, ast.Constant) or not isinstance(node.value, (bool, int, float)):
            raise ValueError(f"unsupported conditioning syntax: {type(node).__name__}")

    visit(ast.parse(expression, mode="eval"))
    return symbols


def _encode(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype.kind not in "biufc" or not 0 < array.size <= _MAX_INPUT_SIZE or not np.all(np.isfinite(array)):
        raise ValueError("conditioning input must be a bounded-size finite numeric value")
    result = {"dtype": str(array.dtype), "shape": list(array.shape), "real": array.real.tolist()}
    if np.iscomplexobj(array):
        result["imag"] = array.imag.tolist()
    return result


def capture_condition_inputs(expression: str, local: Mapping[str, Any], global_: Mapping[str, Any]) -> dict[str, Any]:
    """Copy actual assertion operands before the assertion executes."""
    result = {}
    for symbol in sorted(condition_symbols(expression)):
        root, *attributes = symbol.split(".")
        if root in local:
            value = local[root]
        elif root in global_:
            value = global_[root]
        else:
            raise ValueError(f"conditioning input was not observed: {symbol}")
        for attribute in attributes:
            # Do not run arbitrary descriptors while observing a test frame.
            try:
                value = vars(value)[attribute]
            except (TypeError, KeyError) as exc:
                raise ValueError(f"conditioning data attribute was not observed: {symbol}") from exc
        result[symbol] = _encode(value)
    return result


def condition_input_errors(expression: str, inputs: Any) -> list[str]:
    """Verify the exact operands and re-evaluate the bounded assertion language."""
    try:
        if not isinstance(inputs, dict) or set(inputs) != condition_symbols(expression):
            raise ValueError("conditioning input symbol set differs from the assertion")
        values = {}
        for symbol, record in inputs.items():
            if not isinstance(record, dict) or set(record) not in (
                {"dtype", "shape", "real"},
                {"dtype", "shape", "real", "imag"},
            ):
                raise ValueError(f"malformed conditioning input: {symbol}")
            dtype = np.dtype(record["dtype"])
            if dtype.kind not in "biufc" or ((dtype.kind == "c") != ("imag" in record)):
                raise ValueError(f"invalid conditioning input dtype: {symbol}")
            array = np.asarray(record["real"])
            if "imag" in record:
                imag = np.asarray(record["imag"])
                if imag.shape != array.shape:
                    raise ValueError(f"conditioning real/imag shapes differ: {symbol}")
                array = array + 1j * imag
            if list(array.shape) != record["shape"]:
                raise ValueError(f"conditioning input shape mismatch: {symbol}")
            array = array.astype(dtype)
            if not 0 < array.size <= _MAX_INPUT_SIZE or not np.all(np.isfinite(array)):
                raise ValueError(f"invalid conditioning input values: {symbol}")
            if _encode(array) != record:
                raise ValueError(f"conditioning input is not canonical for its dtype: {symbol}")
            values[symbol] = array

        def evaluate(node: ast.AST) -> Any:
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, (ast.Name, ast.Attribute)):
                return values[_name(node)]
            if isinstance(node, ast.Call):
                return _FUNCTIONS[_name(node.func)](*(evaluate(argument) for argument in node.args))
            if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
                return _BINARY[type(node.op)](evaluate(node.left), evaluate(node.right))
            if isinstance(node, ast.UnaryOp):
                operations = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_}
                if type(node.op) in operations:
                    return operations[type(node.op)](evaluate(node.operand))
            if isinstance(node, ast.Compare):
                left = evaluate(node.left)
                checks = []
                for operation, right_node in zip(node.ops, node.comparators, strict=True):
                    right = evaluate(right_node)
                    checks.append(_COMPARISONS[type(operation)](left, right))
                    left = right
                result = checks[0]
                for check in checks[1:]:
                    result = np.logical_and(result, check)
                return result
            if isinstance(node, ast.BoolOp):
                results = [bool(evaluate(value)) for value in node.values]
                return all(results) if isinstance(node.op, ast.And) else any(results)
            raise ValueError(f"unsupported conditioning operation: {type(node).__name__}")

        result = np.asarray(evaluate(ast.parse(expression, mode="eval").body))
        if result.shape != () or result.dtype.kind != "b" or not bool(result):
            raise ValueError("observed inputs do not satisfy the declared conditioning assertion")
    except (ValueError, KeyError, TypeError, SyntaxError, OverflowError, ZeroDivisionError) as exc:
        return [str(exc)]
    return []

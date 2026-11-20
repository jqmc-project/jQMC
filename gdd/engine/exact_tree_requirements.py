"""Finite, authored array-leaf obligations for recursive exact state comparisons.

Empty leaves are structural evidence, not successful numerical comparisons.
The array comparator retains the guarded suite's assert_array_equal semantics:
NaNs are rejected and dtype identity is not a new requirement.
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
import json
import keyword
import math
from collections import Counter


POLICY = "CMP-SOURCE-ARRAY-EQUALITY-001"
COMPARATOR = "numpy.testing.assert_array_equal"


def _selector_key(selection):
    """Canonical JSON keeps booleans distinct from integer parameter values."""
    if not isinstance(selection, dict) or any(
        not isinstance(key, str) or not key or type(value) not in (str, int, bool) for key, value in selection.items()
    ):
        raise ValueError("tree case requires named concrete parameter values")
    return json.dumps(selection, sort_keys=True, separators=(",", ":"))


def source_signature_errors(source, comparison):
    """Require the reviewed two-operand/default API, not ignored strict kwargs."""
    try:
        location = comparison["location"]
        calls = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and (node.lineno, node.col_offset) == (location["line"], location["column"])
        ]
        if len(calls) != 1:
            return ["exact tree comparator source location is not one call"]
        call = calls[0]
        if len(call.args) != 2 or any(isinstance(value, ast.Starred) for value in call.args):
            return ["exact tree comparator requires exactly two explicit operands"]
        if any(keyword.arg != "err_msg" for keyword in call.keywords):
            return ["exact tree comparator source must retain the default comparison semantics"]
    except (KeyError, TypeError, ValueError, SyntaxError):
        return ["exact tree comparator source signature is unresolved"]
    return []


def validate_tree_cases(declaration, *, cases=None, selector=None):
    """Validate a finite case/path/shape inventory; never derive it from receipts."""
    if not isinstance(declaration, dict) or set(declaration) != {
        "path_variable",
        "cases",
    }:
        raise ValueError("tree cases require exactly path_variable and cases")
    variable = declaration["path_variable"]
    if not isinstance(variable, str) or not variable.isidentifier() or keyword.iskeyword(variable):
        raise ValueError("tree path variable must be a local identifier")
    rows = declaration["cases"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("tree cases require a nonempty explicit case list")
    selected = None
    declared = []
    runtime_key = _selector_key(selector) if selector is not None else None
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"selector", "leaves"}:
            raise ValueError("tree case requires exactly selector and leaves")
        selection, leaves = row["selector"], row["leaves"]
        selection_key = _selector_key(selection)
        if selection_key in declared:
            raise ValueError("tree case selectors must be unique")
        declared.append(selection_key)
        if not isinstance(leaves, dict) or not leaves:
            raise ValueError("tree case requires a nonempty path inventory")
        for path, shape in leaves.items():
            if not isinstance(path, str) or not path:
                raise ValueError("tree leaf path must be a nonempty string")
            if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
                raise ValueError("tree leaf requires an exact nonnegative operand shape")
        if not any(math.prod(shape) > 0 for shape in leaves.values()):
            raise ValueError("empty tree leaves cannot replace nonempty array evidence")
        if selection_key == runtime_key:
            selected = leaves
    if cases is not None:
        collected = [_selector_key(case["selector"]) for case in cases]
        if len(set(collected)) != len(collected) or set(collected) != set(declared):
            raise ValueError("tree cases must exactly cover the collected parameter selectors")
    if selector is not None and selected is None:
        raise ValueError("tree case has no exact matching runtime selector")
    return selected


def capture_array_leaf(actual, reference, namespace, declaration):
    """Observe the original comparator's semantics, keeping structural leaves distinct."""
    # NumPy is imported here, not at module scope: static GDD consumers
    # (case-binding derivation, ledger rendering) import this module and must
    # stay free of numerical imports. Only runtime leaf capture needs NumPy.
    import numpy as np

    variable = declaration["path_variable"]
    value = namespace.get(variable)
    record = {
        "tree_path": {
            "variable": variable,
            "value": value if isinstance(value, str) else None,
        }
    }
    try:
        left, right = np.asarray(actual), np.asarray(reference)
        record["operand_shapes"] = {
            "actual": list(left.shape),
            "reference": list(right.shape),
        }
        record["operand_dtypes"] = {
            "actual": str(left.dtype),
            "reference": str(right.dtype),
        }
        if left.dtype.kind not in "biufc" or right.dtype.kind not in "biufc":
            record["operand_check"] = "unsupported-operands"
            return record
        if np.any(np.isnan(left)) or np.any(np.isnan(right)):
            record["operand_check"] = "nan"
            return record
        # Replay independently from the comparator under observation. A noop
        # substitution must never certify mismatched values. The explicit leaf
        # inventory excludes scalar broadcasting; dtype identity is not imposed.
        if left.shape != right.shape:
            record["operand_check"] = "invalid-shape"
        elif not left.size:
            record["operand_check"] = "empty"
        else:
            record["operand_check"] = "passed" if bool(np.all(left == right)) else "value-mismatch"
    except AssertionError:
        record["operand_check"] = "value-mismatch"
    except (TypeError, ValueError, OverflowError):
        record["operand_check"] = "unsupported-operands"
    return record


def tree_call_errors(call, declaration, *, selector):
    """Validate one comparator call against its authored nonempty/structural leaf."""
    if not isinstance(selector, dict) or not isinstance(call, dict):
        return ["exact tree call requires a concrete selector and call record"]
    try:
        leaves = validate_tree_cases(declaration, selector=selector)
    except (KeyError, TypeError, ValueError) as exc:
        return [str(exc)]
    path = call.get("tree_path")
    if (
        not isinstance(path, dict)
        or set(path) != {"variable", "value"}
        or path["variable"] != declaration["path_variable"]
        or not isinstance(path["value"], str)
        or path["value"] not in leaves
    ):
        return ["undeclared or malformed exact tree leaf path"]
    expected_shape = leaves[path["value"]]
    errors = []
    shapes = call.get("operand_shapes")
    if (
        not isinstance(shapes, dict)
        or set(shapes) != {"actual", "reference"}
        or any(
            not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape) for shape in shapes.values()
        )
        or shapes != {"actual": expected_shape, "reference": expected_shape}
    ):
        errors.append("exact tree leaf operand shapes differ from the authored path")
    dtypes = call.get("operand_dtypes")
    if (
        not isinstance(dtypes, dict)
        or set(dtypes) != {"actual", "reference"}
        or any(not isinstance(value, str) or not value for value in dtypes.values())
    ):
        errors.append("exact tree leaf operand dtypes were not observed")
    # Dtype identity is deliberately NOT required by assert_array_equal(strict=False).
    expected_status = "empty" if math.prod(expected_shape) == 0 else "passed"
    if call.get("operand_check") != expected_status:
        errors.append("exact tree leaf does not satisfy its array comparison semantics")
    if call.get("arguments") != {} or call.get("comparator") != COMPARATOR or call.get("outcome") != "normal":
        errors.append("exact tree leaf comparator did not finish with the authored default arguments")
    return errors


def tree_coverage_errors(check, invocations, *, selector):
    """Require every leaf once within THIS test receipt, never across receipts.

    Existing invocation/caller/executed-observation validation must also run.
    This supplies the finite missing-leaf detector that source call graphs alone
    cannot provide for loops over a nested state tree.
    """
    if not isinstance(selector, dict) or not isinstance(invocations, list):
        return ["exact tree coverage requires a concrete selector and invocation list"]
    declaration = check["tree_cases"]
    try:
        leaves = validate_tree_cases(declaration, selector=selector)
    except (KeyError, TypeError, ValueError) as exc:
        return [str(exc)]
    seen = Counter()
    errors = []
    for invocation in invocations:
        if not isinstance(invocation, dict) or not isinstance(invocation.get("tolerance_calls", []), list):
            errors.append("malformed exact tree comparison invocation")
            continue
        for attempt in invocation.get("tolerance_calls", []):
            if not isinstance(attempt, dict):
                errors.append("malformed exact tree comparison attempt")
                continue
            if attempt.get("site") != check["site"]:
                continue
            calls = attempt.get("calls")
            if not isinstance(calls, list) or len(calls) != 1:
                errors.append("exact tree leaf requires one actual comparator call per invocation")
                continue
            call = calls[0]
            errors.extend(tree_call_errors(call, declaration, selector=selector))
            path_record = call.get("tree_path") if isinstance(call, dict) else None
            path = path_record.get("value") if isinstance(path_record, dict) else None
            if isinstance(path, str):
                seen[path] += 1
    if seen != Counter({path: 1 for path in leaves}):
        errors.append("exact tree comparison requires each authored path exactly once in this receipt")
    return errors

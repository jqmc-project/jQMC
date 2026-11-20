"""Finite component receipts for zone-derived absolute and central zero relative tolerance.

No scientific budget is introduced: each component retains strict zone ``atol``
and the central exact-zero ``rtol``. Repeated comparisons identify actual tuple
members, rather than trusting a loop counter or a mere number of calls.
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
from typing import Any


POLICY = "CMP-ZONE-ATOL-ZERO-RTOL-001"
ZERO_SYMBOL = "jqmc._setting.RTOL_BIT_EXACT"


def binding(definition: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "registry-id": "TOL-ZONE-MIN-001",
        "fixed-arguments": {"level": "strict"},
        "rtol-symbol": ZERO_SYMBOL,
    }
    if (
        definition.get("id") != POLICY
        or definition.get("method") != "allclose"
        or definition.get("tolerance-binding") != expected
        or definition.get("allowed-precision-modes") != ["full", "mixed"]
    ):
        raise ValueError("zone/zero adapter must retain strict zone atol and central zero rtol in both modes")
    return expected


def components(disposition: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(disposition, dict) or set(disposition) != {"policy", "components"} or disposition["policy"] != POLICY:
        raise ValueError("zone/zero disposition requires only policy and components")
    rows = disposition["components"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("zone/zero components must be a nonempty ordered inventory")
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or set(row) != {"index", "zones"}
            or type(row["index"]) is not int
            or row["index"] != index
            or not isinstance(row["zones"], list)
            or not row["zones"]
            or any(type(zone) is not str or not zone for zone in row["zones"])
            or len(set(row["zones"])) != len(row["zones"])
        ):
            raise ValueError("zone/zero components require consecutive indices and distinct zone names")
    return rows


def validate_loop(loop: Any, rows: list[dict[str, Any]]) -> None:
    if loop is None:
        if len(rows) != 1:
            raise ValueError("multiple components require an explicit tuple loop")
        return
    if not isinstance(loop, dict) or set(loop) != {"sequence", "value", "index"}:
        raise ValueError("component loop requires sequence, value and optional index variable")
    names = [loop["sequence"], loop["value"]] + ([] if loop["index"] is None else [loop["index"]])
    if (
        len(rows) < 2
        or any(type(name) is not str or not name.isidentifier() or keyword.iskeyword(name) for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("component loop requires distinct local names and multiple components")


def scopes(definition: dict[str, Any]) -> list[dict[str, Any]]:
    binding(definition)
    rows = definition.get("allowed-bindings")
    if not isinstance(rows, list) or not rows:
        raise ValueError("zone/zero policy requires a finite reviewed binding inventory")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "pytest_node",
            "site",
            "verdict_method",
            "precision",
            "selectors",
            "components",
            "component_loop",
        }:
            raise ValueError("zone/zero binding fields differ from the reviewed schema")
        for name in ("pytest_node", "site", "verdict_method"):
            if type(row[name]) is not str or not row[name]:
                raise ValueError("zone/zero bindings require explicit node, site and verdict method")
        # Paths contain dots: compare qualification separately, not by globally replacing them.
        path, qualified = row["site"].split("::", 1)
        owner = path + "::" + qualified.split("#", 1)[0].replace(".", "::")
        if owner != row["pytest_node"] or row["precision"] not in {"full", "mixed"}:
            raise ValueError("zone/zero site must belong to its declared node and precision")
        key = (row["pytest_node"], row["site"], row["verdict_method"], row["precision"])
        if key in seen:
            raise ValueError("duplicate zone/zero binding")
        seen.add(key)
        entries = components({"policy": POLICY, "components": row["components"]})
        validate_loop(row["component_loop"], entries)
        selectors = row["selectors"]
        if not isinstance(selectors, list) or not selectors:
            raise ValueError("zone/zero binding requires concrete parameter cases")
        selector_keys = []
        for selector in selectors:
            if (
                not isinstance(selector, dict)
                or not selector
                or any(type(name) is not str or type(value) not in (str, bool, int) for name, value in selector.items())
            ):
                raise ValueError("zone/zero selectors must be explicit scalar mappings")
            selector_keys.append(tuple((name, type(value).__name__, value) for name, value in sorted(selector.items())))
        if len(set(selector_keys)) != len(selector_keys) or any(set(value) != set(selectors[0]) for value in selectors):
            raise ValueError("zone/zero selectors must be distinct and use the same axes")
    return rows


def validate_scope(definition, check, disposition, pytest_node, method, precision, cases, selector):
    matches = [
        row
        for row in scopes(definition)
        if row["pytest_node"] == pytest_node
        and row["site"] == check["site"]
        and row["verdict_method"] == method
        and row["precision"] == precision
    ]
    if len(matches) != 1:
        raise ValueError("zone/zero comparison is outside its reviewed node/site/method/mode binding")
    row = matches[0]
    if row["components"] != disposition["components"] or row["component_loop"] != check.get("component_loop"):
        raise ValueError("zone/zero component inventory differs from its reviewed binding")
    if check["comparator"] != "numpy.testing.assert_allclose":
        raise ValueError("zone/zero requires its explicit allclose comparator")
    if cases is not None:
        current = [case["selector"] for case in cases]
        if len(current) != len(row["selectors"]) or any(value not in row["selectors"] for value in current):
            raise ValueError("zone/zero case universe differs from the reviewed cases")
        if len({repr(sorted(value.items())) for value in current}) != len(current):
            raise ValueError("zone/zero case universe contains duplicates")
    elif not isinstance(selector, dict) or not any(
        set(selector) == set(value)
        and all(type(selector[name]) is type(item) and selector[name] == item for name, item in value.items())
        for value in row["selectors"]
    ):
        raise ValueError("zone/zero requires one of its exact reviewed parameter cases")


def _description(value, index, loop):
    if value.get("kind") == "literal":
        return ast.literal_eval(value["value"])
    if value.get("kind") == "expression" and value.get("operation") == "IfExp":
        fields = value.get("fields", {})
        test = fields.get("test", {})
        compare = test.get("fields", {})
        variable = (loop or {}).get("index")
        if (
            not variable
            or set(fields) != {"test", "body", "orelse"}
            or test.get("kind") != "expression"
            or test.get("operation") != "Compare"
            or set(compare) != {"left", "ops", "comparators"}
            or compare["left"].get("kind") != "unresolved"
            or compare["left"].get("syntax") != f"Name(id='{variable}', ctx=Load())"
            or compare["ops"] != [{"kind": "expression", "operation": "Eq", "fields": {}}]
            or len(compare["comparators"]) != 1
        ):
            raise ValueError("unsupported component-dependent zone expression")
        pivot = _description(compare["comparators"][0], index, loop)
        if type(pivot) is not int:
            raise ValueError("component zone selection requires an integer pivot")
        choices = [_description(fields[arm], index, loop) for arm in ("body", "orelse")]
        return choices[0 if index == pivot else 1]
    raise ValueError("unresolved zone/zero source expression")


def validate_source(source, check, disposition, source_text):
    rows = components(disposition)
    loop = check.get("component_loop")
    validate_loop(loop, rows)
    if source.get("forwarded_keywords") or source.get("unresolved_signature"):
        raise ValueError("zone/zero requires an explicit resolved comparator signature")
    args = source.get("arguments", {})
    if set(args) != {"atol", "rtol"} or args["rtol"] != {"kind": "central-reference", "name": ZERO_SYMBOL}:
        raise ValueError("zone/zero source must use the central zero relative tolerance")
    absolute = args["atol"]
    if absolute.get("kind") != "item" or type(absolute.get("index")) is not int or absolute["index"] != 0:
        raise ValueError("zone/zero absolute tolerance must be resolver component zero")
    call = absolute.get("value", {})
    if (
        call.get("kind") != "central-resolver-call"
        or call.get("function") != {"kind": "central-reference", "name": "jqmc._precision.get_tolerance"}
        or call.get("keywords") != []
        or len(call.get("args", [])) != 2
    ):
        raise ValueError("zone/zero requires the reviewed single-zone resolver")
    for row in rows:
        if [_description(call["args"][0], row["index"], loop)] != row["zones"] or _description(
            call["args"][1], row["index"], loop
        ) != "strict":
            raise ValueError("zone/zero source zones or level differ for a declared component")
    tree = ast.parse(source_text)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    location = source["location"]
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and (node.lineno, node.col_offset) == (location["line"], location["column"])
    ]
    if len(matches) != 1:
        raise ValueError("zone/zero source comparison location is ambiguous")
    node = matches[0]
    loops = []
    while node in parents and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        node = parents[node]
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            loops.append(node)
    if loop is None:
        if loops:
            raise ValueError("singleton zone/zero comparison cannot be placed in a loop")
        return
    if len(loops) != 1 or not isinstance(loops[0], ast.For):
        raise ValueError("component comparison requires exactly its declared for loop")
    node = loops[0]
    iterator, target = node.iter, node.target
    if loop["index"] is not None:
        if (
            not isinstance(iterator, ast.Call)
            or not isinstance(iterator.func, ast.Name)
            or iterator.func.id != "enumerate"
            or len(iterator.args) != 1
            or iterator.keywords
            or not isinstance(target, ast.Tuple)
            or len(target.elts) != 2
            or not isinstance(target.elts[0], ast.Name)
            or target.elts[0].id != loop["index"]
        ):
            raise ValueError("component index must enumerate the entire declared tuple")
        iterator, target = iterator.args[0], target.elts[1]
    if (
        not isinstance(iterator, ast.Call)
        or not isinstance(iterator.func, ast.Name)
        or iterator.func.id != "zip"
        or len(iterator.args) != 2
        or any(not isinstance(arg, ast.Name) for arg in iterator.args)
        or iterator.args[1].id != loop["sequence"]
        or iterator.args[0].id == loop["sequence"]
        or len(iterator.keywords) != 1
        or iterator.keywords[0].arg != "strict"
        or not isinstance(iterator.keywords[0].value, ast.Constant)
        or iterator.keywords[0].value.value is not True
        or not isinstance(target, ast.Tuple)
        or len(target.elts) != 2
        or any(not isinstance(value, ast.Name) for value in target.elts)
        or target.elts[1].id != loop["value"]
        or target.elts[0].id == loop["value"]
    ):
        raise ValueError("component loop must zip complete distinct tuples with strict=True")
    actual = ast.parse(check["actual"]["expression"], mode="eval").body
    if not isinstance(actual, ast.Call) or len(actual.args) != 1 or ast.unparse(actual.args[0]) != loop["value"]:
        raise ValueError("component comparison must consume its actual tuple member")


def _local_shape_value(expression, local):
    if isinstance(expression, ast.Name):
        return local[expression.id]
    if isinstance(expression, ast.Attribute):
        if expression.attr not in {"shape", "num_ao"}:
            raise ValueError("unsupported fixture shape attribute")
        return getattr(_local_shape_value(expression.value, local), expression.attr)
    if isinstance(expression, ast.Subscript) and isinstance(expression.slice, ast.Constant):
        if type(expression.slice.value) is not int:
            raise ValueError("fixture shape indices must be integers")
        return _local_shape_value(expression.value, local)[expression.slice.value]
    raise ValueError("reference dimensions must come from fixture/input shape metadata")


def capture(check, local, actual, reference):
    """Capture concrete identity in the tuple; duplicated or aliased members fail closed."""
    # NumPy is imported here, not at module scope: static GDD consumers import
    # this module and must stay free of numerical imports.
    import numpy as np

    try:
        loop = check.get("component_loop")
        actual_expression = ast.parse(check["actual"]["expression"], mode="eval").body
        reference_expression = ast.parse(check["reference"]["expression"], mode="eval").body
        if (
            not isinstance(actual_expression, ast.Call)
            or ast.unparse(actual_expression.func) != "np.imag"
            or len(actual_expression.args) != 1
            or not isinstance(actual_expression.args[0], ast.Name)
            or not isinstance(reference_expression, ast.Call)
            or ast.unparse(reference_expression.func) != "np.zeros"
            or len(reference_expression.args) != 1
            or not isinstance(reference_expression.args[0], ast.Tuple)
        ):
            raise ValueError("zone/zero operands must retain imaginary value and fixture-shaped zero array")
        value = local[actual_expression.args[0].id]
        expected_shape = [_local_shape_value(dimension, local) for dimension in reference_expression.args[0].elts]
        if any(type(size) is not int or size <= 0 for size in expected_shape):
            raise ValueError("zone/zero reference shape must be nonempty fixture/input dimensions")
        index, length = 0, 1
        if loop is not None:
            sequence = local[loop["sequence"]]
            if type(sequence) is not tuple or len({id(item) for item in sequence}) != len(sequence):
                raise ValueError("component sequence must be a tuple of distinct objects")
            found = [index for index, item in enumerate(sequence) if item is value]
            if len(found) != 1 or value is not local[loop["value"]]:
                raise ValueError("component value does not uniquely identify a tuple member")
            index, length = found[0], len(sequence)
            if loop["index"] is not None and (type(local.get(loop["index"])) is not int or local[loop["index"]] != index):
                raise ValueError("declared component index disagrees with tuple identity")
        return {
            "index": index,
            "length": length,
            "component_loop": loop,
            "operand_matches": bool(np.array_equal(np.asarray(actual), np.imag(value))),
            "operand_shapes": {"actual": list(np.shape(actual)), "reference": list(np.shape(reference))},
            "expected_shape": expected_shape,
            "reference_zero": bool(np.all(np.asarray(reference) == 0)),
        }
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        return {"capture_error": str(exc)}


def captured_index(record, check, disposition):
    rows = components(disposition)
    if (
        not isinstance(record, dict)
        or set(record)
        != {"index", "length", "component_loop", "operand_matches", "operand_shapes", "expected_shape", "reference_zero"}
        or type(record["index"]) is not int
        or record["index"] not in range(len(rows))
        or type(record["length"]) is not int
        or record["length"] != len(rows)
        or record["component_loop"] != check.get("component_loop")
        or record["operand_matches"] is not True
        or record["reference_zero"] is not True
        or not isinstance(record["expected_shape"], list)
        or not record["expected_shape"]
        or any(type(size) is not int or size <= 0 for size in record["expected_shape"])
        or record["operand_shapes"] != {"actual": record["expected_shape"], "reference": record["expected_shape"]}
    ):
        raise ValueError("missing, substituted or invalid tuple component receipt")
    return record["index"]

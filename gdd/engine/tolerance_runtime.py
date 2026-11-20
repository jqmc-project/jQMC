"""Capture actual comparison arguments at their registered Python call sites.

No scientific implementation or comparator is monkeypatched. Code identity is
captured when this module is imported, before a test can replace the callable.
Each statement attempt is distinct, including repeated calls on one loop line.
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

import dis
import math
from typing import Any

import numpy as np
from numpy.testing._private import utils as _np_testing_utils

from gdd.engine import exact_tree_requirements, inequality_receipts, spacing_requirements, zone_zero_requirements
from gdd.engine.fixture_observations import input_object_snapshot


# Code identity comes from NumPy's private implementation module, not the public
# ``numpy.testing`` namespace: the test suite's NaN-agreement guard
# (tests/_nan_guard.py) replaces the public names with wrappers that always call
# these originals, so the traced frame is still the original comparator.
_ASSERT_ALLCLOSE_CODE = _np_testing_utils.assert_allclose.__code__
_ASSERT_ARRAY_EQUAL_CODE = _np_testing_utils.assert_array_equal.__code__
_ALLCLOSE_CODE = np.allclose.__wrapped__.__code__
_ISCLOSE_CODE = np.isclose.__wrapped__.__code__
_DECIMAL_CODES = {
    _np_testing_utils.assert_almost_equal.__code__: "numpy.testing.assert_almost_equal",
    _np_testing_utils.assert_array_almost_equal.__code__: "numpy.testing.assert_array_almost_equal",
}
_COMPARATORS = {
    _ASSERT_ALLCLOSE_CODE: ("numpy.testing.assert_allclose", ("actual", "desired")),
    _ASSERT_ARRAY_EQUAL_CODE: ("numpy.testing.assert_array_equal", _ASSERT_ARRAY_EQUAL_CODE.co_varnames[:2]),
    _ALLCLOSE_CODE: ("numpy.allclose", ("a", "b")),
    _ISCLOSE_CODE: ("numpy.isclose", ("a", "b")),
    **{code: (name, ("actual", "desired")) for code, name in _DECIMAL_CODES.items()},
}
_EXACT_OPERAND_NAMES = _ASSERT_ARRAY_EQUAL_CODE.co_varnames[:2]
_NORMAL_RETURN = {"RETURN_VALUE", "RETURN_CONST", "INSTRUMENTED_RETURN_VALUE", "INSTRUMENTED_RETURN_CONST"}


def _operand_check(values: dict[str, Any]) -> str:
    """Apply registered element-wise semantics, without NumPy's scalar broadcasting/NaN waiver."""
    try:
        actual, reference = np.asarray(values["actual"]), np.asarray(values["desired"])
        if actual.shape != reference.shape or actual.dtype.kind not in "biufc" or reference.dtype.kind not in "biufc":
            return "invalid-shape-or-dtype"
        if np.any(np.isnan(actual)) or np.any(np.isnan(reference)):
            return "nan"
        atol, rtol = values["atol"], values["rtol"]
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in (atol, rtol)):
            return "invalid-tolerance"
        if not actual.size:
            # Not a numerical pass. Completion needs an authored case/shape
            # match and a separate nonempty comparison in the same test.
            return "empty"
        # Cast integer inputs before subtraction to avoid overflow. Preserve
        # complex magnitudes and identical signed infinities explicitly.
        dtype = np.result_type(actual.dtype, reference.dtype, np.float64)
        a, b = actual.astype(dtype), reference.astype(dtype)
        finite = np.isfinite(a) & np.isfinite(b)
        if np.any(a[~finite] != b[~finite]):
            return "infinity-mismatch"
        if not np.all(np.abs(a[finite] - b[finite]) <= atol + rtol * np.abs(b[finite])):
            return "value-mismatch"
        return "passed"
    except (KeyError, TypeError, ValueError, OverflowError):
        return "unsupported-operands"


def _exact_operand_check(values: dict[str, Any]) -> str:
    """Match observe_exact's numeric value semantics, never scalar broadcasting or raw bits."""
    try:
        actual, reference = (np.asarray(values[name]) for name in _EXACT_OPERAND_NAMES)
        if actual.shape != reference.shape or actual.dtype != reference.dtype or actual.dtype.kind not in "biufc":
            return "invalid-shape-or-dtype"
        if np.any(np.isnan(actual)) or np.any(np.isnan(reference)):
            return "nan"
        if not actual.size:
            return "empty"
        # No subtraction or float cast: adjacent large integers must remain distinct.
        return "passed" if np.all(actual == reference) else "value-mismatch"
    except (KeyError, TypeError, ValueError, OverflowError):
        return "unsupported-operands"


def _decimal_operand_check(values: dict[str, Any]) -> str:
    """Check the decimal API's strict absolute bound, not an invented allclose pair.

    The registered numerical contract rejects broadcasting, empty arrays and
    NaNs even where the NumPy convenience API would allow them. Both decimal
    APIs use the magnitude bound for finite complex inputs as well.
    """
    try:
        actual, reference = np.asarray(values["actual"]), np.asarray(values["desired"])
        if actual.shape != reference.shape or actual.dtype.kind not in "biufc" or reference.dtype.kind not in "biufc":
            return "invalid-shape-or-dtype"
        digits = values["decimal"]
        if type(digits) is not int or not 0 <= digits <= 308:
            return "invalid-tolerance"
        if np.any(np.isnan(actual)) or np.any(np.isnan(reference)):
            return "nan"
        if not actual.size:
            return "empty"
        dtype = np.result_type(actual.dtype, reference.dtype, np.float64)
        a, b = actual.astype(dtype), reference.astype(dtype)
        finite = np.isfinite(a) & np.isfinite(b)
        if np.any(a[~finite] != b[~finite]):
            return "infinity-mismatch"
        bound = 1.5 * 10.0 ** (-digits)
        return "passed" if np.all(np.abs(a[finite] - b[finite]) < bound) else "value-mismatch"
    except (KeyError, TypeError, ValueError, OverflowError):
        return "unsupported-operands"


class ToleranceRuntime:
    def __init__(self, checks, observations, *, snapshot_sites=(), chain_sites=None):
        self.snapshot_sites = frozenset(snapshot_sites)
        self.chain_sites = chain_sites or {}
        self.requirements = {check["site"]: check for check in (checks or {}).values()}
        required = set(self.requirements)
        self.specs = {
            observation["id"]: [
                row for row in observation.get("tolerance_provenance", {}).get("comparisons", ()) if row["site"] in required
            ]
            for observation in observations
        }
        self.spacing_sources = {
            row["site"]: observation["expression"]
            for observation in observations
            for row in self.specs.get(observation["id"], ())
            if any(
                disposition.get("policy") == spacing_requirements.POLICY
                for disposition in self.requirements[row["site"]]["precision"].values()
            )
        }
        self.statements: dict[int, list[dict[str, Any]]] = {}
        self.pending: dict[tuple[int, str], dict[str, Any]] = {}
        self.frames: dict[int, dict[str, Any]] = {}
        self.positions: dict[Any, tuple] = {}
        self.statement_entries: dict[tuple[Any, int, int], int | None] = {}
        self.comparison_lines: dict[tuple[str, int], set[str]] = {}
        for observation_id, rows in self.specs.items():
            for spec in rows:
                key = (observation_id.split("::", 1)[0], spec["location"]["line"])
                self.comparison_lines.setdefault(key, set()).add(observation_id)

    def begin_statement(self, frame, observation_id):
        for spec in self.specs.get(observation_id, ()):
            location = spec["location"]
            key = (frame.f_code, location["line"], location["column"])
            if key not in self.statement_entries:
                self.statement_entries[key] = next(
                    (
                        instruction.offset
                        for instruction in dis.get_instructions(frame.f_code)
                        if (instruction.positions.lineno, instruction.positions.col_offset) == key[1:]
                    ),
                    None,
                )
            # A multiline call revisits its first source line after evaluating
            # arguments. Only entry into the expression starts a new attempt;
            # the next loop iteration reaches that entry again. Never discard
            # empty attempts, which may be genuine substituted/noop calls.
            if frame.f_lasti != self.statement_entries[key]:
                continue
            attempt = {"site": spec["site"], "observation_id": observation_id, "calls": []}
            self.statements.setdefault(id(frame), []).append(attempt)
            self.pending[(id(frame), spec["site"])] = attempt
            if spec["site"] in self.spacing_sources:
                attempt["spacing_basis"] = spacing_requirements.capture(
                    self.spacing_sources[spec["site"]], frame.f_locals, frame.f_globals
                )
            if spec["comparator"] == inequality_receipts.SOURCE_COMPARATOR and "helper" not in spec:
                # A hand-written inequality has no comparator frame to trace.
                # Its operands and tolerance are read from this frame on entry
                # to the comparison expression and the reviewed form replayed.
                call = inequality_receipts.capture(spec, frame.f_locals, frame.f_globals)
                self._record_iteration(call, spec["site"], frame)
                attempt["calls"].append(call)

    def observe_helper_call(self, frame):
        """Capture explicit arguments from a verified direct helper frame only."""
        caller = frame.f_back
        if caller is None or id(caller) not in self.statements:
            return
        positions = self.positions.setdefault(caller.f_code, tuple(caller.f_code.co_positions()))
        offset = caller.f_lasti // 2
        position = positions[offset] if 0 <= offset < len(positions) else None
        for rows in self.specs.values():
            for spec in rows:
                helper = spec.get("helper")
                if helper is None or tuple(helper["call_position"]) != position:
                    continue
                attempt = self.pending.get((id(caller), spec["site"]))
                if attempt is None or frame.f_code.co_filename != caller.f_code.co_filename:
                    continue
                if frame.f_code not in inequality_receipts.helper_code_candidates(
                    frame.f_code.co_filename, helper["name"], helper["line"], helper["syntax_sha256"]
                ):
                    continue
                call = inequality_receipts.capture(spec, frame.f_locals, {})
                self._record_iteration(call, spec["site"], caller)
                attempt["calls"].append(call)

    def _record_iteration(self, call, site, frame):
        iterations = self.requirements[site].get("iterations")
        if iterations is not None:
            variable = iterations["variable"]
            value = frame.f_locals.get(variable)
            call["iteration"] = {
                "variable": variable,
                "value": value if isinstance(value, str) else {"invalid_type": type(value).__name__},
            }

    def finish_invocation(self, frame):
        frame_id = id(frame)
        for key in tuple(self.pending):
            if key[0] == frame_id:
                del self.pending[key]
        return self.statements.pop(frame_id, [])

    def observe_call(self, frame, event):
        if event != "call" or frame.f_code not in _COMPARATORS:
            return None
        comparator, operand_names = _COMPARATORS[frame.f_code]
        exact = frame.f_code is _ASSERT_ARRAY_EQUAL_CODE
        decimal = frame.f_code in _DECIMAL_CODES
        caller = frame.f_back
        while caller is not None and id(caller) not in self.statements:
            # NumPy's decimal scalar API recursively handles components and
            # delegates arrays to its array API. Only the outer authored call
            # is an attempt; library-internal calls cannot impersonate it.
            if id(caller) in self.frames:
                return None
            caller = caller.f_back
        if caller is None:
            return None
        positions = self.positions.setdefault(caller.f_code, tuple(caller.f_code.co_positions()))
        position = positions[caller.f_lasti // 2]
        for rows in self.specs.values():
            for spec in rows:
                if spec["comparator"] != {"kind": "imported-reference", "name": comparator}:
                    continue
                if (position[0], position[2]) != (spec["location"]["line"], spec["location"]["column"]):
                    continue
                attempt = self.pending.get((id(caller), spec["site"]))
                if attempt is None:
                    continue
                values = frame.f_locals
                arguments = {}
                for name in () if exact else (("decimal",) if decimal else ("atol", "rtol")):
                    value = values.get(name)
                    # JSON-safe actual scalar values, not recomputed resolver outputs.
                    if decimal and isinstance(value, np.integer):
                        value = int(value)
                    elif isinstance(value, np.floating):
                        value = float(value)
                    arguments[name] = (
                        value
                        if type(value) in (int, float) and math.isfinite(value)
                        else {"invalid_type": type(value).__name__}
                    )
                actual_value, reference_value = (values.get(name) for name in operand_names)
                call = {
                    "comparator": comparator,
                    "arguments": arguments,
                    "outcome": "entered",
                    "operand_check": _exact_operand_check(values)
                    if exact
                    else (_decimal_operand_check if decimal else _operand_check)(
                        dict(actual=actual_value, desired=reference_value, **arguments)
                    ),
                }
                if spec["site"] in self.spacing_sources:
                    call["spacing_basis"] = attempt.get("spacing_basis")
                if any(
                    disposition.get("policy") == zone_zero_requirements.POLICY
                    for disposition in self.requirements[spec["site"]]["precision"].values()
                ):
                    call["zone_zero_component"] = zone_zero_requirements.capture(
                        self.requirements[spec["site"]], caller.f_locals, actual_value, reference_value
                    )
                if exact or "empty_cases" in self.requirements[spec["site"]]:
                    try:
                        call["operand_shapes"] = {
                            "actual": list(np.shape(actual_value)),
                            "reference": list(np.shape(reference_value)),
                        }
                        if exact:
                            call["operand_dtypes"] = {
                                "actual": str(np.asarray(actual_value).dtype),
                                "reference": str(np.asarray(reference_value).dtype),
                            }
                    except (ValueError, TypeError, OverflowError) as exc:
                        call["operand_shapes"] = {"capture_error": str(exc)}
                if "tree_cases" in self.requirements[spec["site"]]:
                    call.update(
                        exact_tree_requirements.capture_array_leaf(
                            actual_value,
                            reference_value,
                            caller.f_locals,
                            self.requirements[spec["site"]]["tree_cases"],
                        )
                    )
                if spec["site"] in self.snapshot_sites:
                    try:
                        call["operand_snapshots"] = {
                            "actual": input_object_snapshot(np.asarray(actual_value)),
                            "reference": input_object_snapshot(np.asarray(reference_value)),
                        }
                    except (ValueError, TypeError, AssertionError) as exc:
                        call["operand_snapshot_error"] = str(exc)
                if spec["site"] in self.chain_sites:
                    from gdd.engine.canonical_chain import CanonicalStreamingCapture

                    captures = {
                        id(value): value for value in caller.f_locals.values() if type(value) is CanonicalStreamingCapture
                    }
                    try:
                        if len(captures) != 1:
                            raise ValueError("canonical comparison requires one unambiguous local capture")
                        capture = next(iter(captures.values()))
                        call["canonical_chain_step"] = capture.comparison_context(self.chain_sites[spec["site"]])
                    except (ValueError, KeyError, AttributeError) as exc:
                        call["canonical_chain_error"] = str(exc)
                self._record_iteration(call, spec["site"], caller)
                attempt["calls"].append(call)
                self.frames[id(frame)] = call
                frame.f_trace_lines = False
                return self.trace_return
        return None

    def trace_return(self, frame, event, arg):
        if event == "return":
            call = self.frames.pop(id(frame), None)
            if call is not None:
                opcode = dis.opname[frame.f_code.co_code[frame.f_lasti]]
                if call["comparator"] == "numpy.isclose":
                    result = np.asarray(arg)
                    returned_success = result.dtype.kind == "b" and result.size > 0 and bool(np.all(result))
                elif call["comparator"] == "numpy.allclose":
                    returned_success = isinstance(arg, (bool, np.bool_)) and bool(arg)
                else:
                    returned_success = arg is None
                call["outcome"] = "normal" if opcode in _NORMAL_RETURN and returned_success else "failed"
        return self.trace_return

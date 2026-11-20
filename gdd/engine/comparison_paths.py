"""Source-bound conditional comparison scopes, shared by tracing and replay.

Branch membership is observed from Python control flow, never by re-evaluating
the condition or guessing its outcome from a parameter label.
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
import builtins
import re
from typing import Any


NUMERICAL_KINDS = {"tolerance-comparison", "exact-comparison", "numerical-bound"}
UNTRACED_COMPARISON_SCOPES = (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def function_scope(function: ast.FunctionDef | ast.AsyncFunctionDef, source: str) -> str:
    first = min([function.lineno, *(decorator.lineno for decorator in function.decorator_list)])
    return f"{source}:{first}:{function.name}"


def terminal_return(suite: list[ast.stmt]) -> bool:
    """A suite ending in a one-line ``return`` (bare or with a value).

    The runtime binds the exit to the return statement's line, so a return
    statement spanning several lines cannot name one source exit.
    """
    return bool(suite) and isinstance(suite[-1], ast.Return) and suite[-1].lineno == suite[-1].end_lineno


def assertion_guard(node: ast.AST) -> ast.Call | None:
    """``if not <comparator call>: raise AssertionError(...)`` -- an assertion written as a guard.

    Returns the comparator call. The guard's test is evaluated on every
    execution of the statement and its only suite fails it, exactly like
    ``assert <comparator call>, <message>``.
    """
    if (
        not isinstance(node, ast.If)
        or node.orelse
        or len(node.body) != 1
        or not isinstance(node.body[0], ast.Raise)
        or not isinstance(node.test, ast.UnaryOp)
        or not isinstance(node.test.op, ast.Not)
        or not isinstance(node.test.operand, ast.Call)
    ):
        return None
    exception = node.body[0].exc
    exception = exception.func if isinstance(exception, ast.Call) else exception
    if not isinstance(exception, ast.Name) or exception.id != "AssertionError":
        return None
    return node.test.operand


def nonrecovering_try(node: ast.AST) -> bool:
    """Recognize cleanup/diagnostic regions that cannot recover an exception.

    A normal invocation must complete the body, else and finally suites. An
    early return is still checked against their comparison obligations. More
    general recovery, exception groups and suspending handlers need explicit
    execution semantics, not an optimistic static approximation.
    """
    return isinstance(node, ast.Try) and all(
        handler.body
        and isinstance(handler.body[-1], ast.Raise)
        and not any(
            isinstance(child, (ast.Return, ast.Yield, ast.YieldFrom, ast.Await, ast.Break, ast.Continue))
            for statement in handler.body
            for child in ast.walk(statement)
        )
        for handler in node.handlers
    )


def conditional_execution(
    node: ast.AST, parents: dict[int, ast.AST], source: str, test: ast.AST | None, *, allow_try_body: bool = False
) -> dict | None:
    """Describe enclosing if suites up to the actual containing function.

    Non-recovering cleanup/diagnostic regions preserve normal-call obligations.
    Recovering exception regions and suspending functions are not covered.
    """
    path = []
    cursor = node
    owner = None
    while (parent := parents.get(id(cursor))) is not None:
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner = parent
            break
        if isinstance(parent, (ast.TryStar, ast.ExceptHandler, *UNTRACED_COMPARISON_SCOPES)):
            return None
        if isinstance(parent, ast.Try) and not nonrecovering_try(parent) and not (allow_try_body and cursor in parent.body):
            return None
        if isinstance(parent, ast.If):
            arm = "body" if cursor in parent.body else "orelse" if cursor in parent.orelse else None
            if arm is None and cursor is parent.test and assertion_guard(parent) is not None:
                # The comparator of an assertion guard is evaluated with its
                # statement; the guard is not a selected suite.
                cursor = parent
                continue
            if arm is None:
                return None
            path.append((parent, arm))
        cursor = parent
    if owner is None or not path or any(isinstance(child, (ast.Yield, ast.YieldFrom, ast.Await)) for child in ast.walk(owner)):
        return None
    function = function_scope(owner, source)
    parent_scope = function
    conditions = []
    for condition, arm in reversed(path):
        identity = f"{source}:{condition.lineno}:{condition.col_offset}:if"
        descriptor = {
            "id": identity,
            "parent": parent_scope,
            "condition": [condition.lineno, condition.test.end_lineno],
            "body": [condition.body[0].lineno, condition.body[-1].end_lineno],
            "orelse": [condition.orelse[0].lineno, condition.orelse[-1].end_lineno] if condition.orelse else [],
        }
        # Only a terminal one-line return (bare or with a value) in a direct
        # function-level suite has the bounded reachability semantics below:
        # the selected suite still performs all of its own comparisons.
        # Nested/premature returns, cleanup regions and multi-line return
        # statements retain the strict contract.
        if parents.get(id(condition)) is owner:
            returns = {
                name: suite[-1].lineno
                for name, suite in (("body", condition.body), ("orelse", condition.orelse))
                if terminal_return(suite)
            }
            if returns:
                descriptor["terminal_returns"] = returns
        conditions.append(descriptor)
        parent_scope = f"{identity}/{arm}"
    return {"function": function, "test_scope": owner is test, "conditions": conditions, "scope": parent_scope}


EXECUTION_SCOPE_KEYS = ("required_in_invocation", "conditional_execution", "required_execution")


def comparison_scope_function(observation: dict[str, Any]) -> str | None:
    """The function whose invocations carry this observation's execution obligation."""
    conditional = observation.get("conditional_execution")
    return observation.get("required_in_invocation") or (conditional.get("function") if isinstance(conditional, dict) else None)


def comparison_call_closure(observations, function: str) -> set[str]:
    """Functions an invocation of ``function`` reaches through ordinary comparison-helper calls.

    Expected-error calls are excluded: their exception is accepted by their own
    ``pytest.raises`` context and never propagates further up.
    """
    edges: dict[str, set[str]] = {}
    for row in observations:
        helper = row.get("helper_call") if isinstance(row, dict) else None
        if isinstance(helper, dict) and "expected_error" not in helper and isinstance(helper.get("callee"), str):
            edges.setdefault(comparison_scope_function(row), set()).add(helper["callee"])
    seen: set[str] = set()
    stack = [function]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(edges.get(current, ()))
    return seen


def comparison_scope_contracts(observations) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build exact per-scope obligations; reject inconsistent source metadata."""
    observations = tuple(observations)
    scopes: dict[str, dict[str, Any]] = {}
    conditions: dict[str, dict[str, Any]] = {}

    def scope(identity, function):
        if not isinstance(identity, str) or not identity or not isinstance(function, str) or not function:
            raise ValueError("invalid comparison scope")
        value = scopes.setdefault(
            identity, {"function": function, "comparisons": set(), "conditions": set(), "calls": {}, "test": False}
        )
        if value["function"] != function:
            raise ValueError("comparison scope belongs to conflicting functions")
        return value

    observation_index = {row["id"]: row for row in observations if isinstance(row, dict) and isinstance(row.get("id"), str)}
    for observation in observations:
        helper = observation.get("helper_call")
        if helper is not None:
            if (
                observation.get("kind") != "comparison-helper-call"
                or not isinstance(helper, dict)
                or set(helper) - {"expected_error", "negated"} != {"callee", "position"}
                or "negated" in helper
                and (helper["negated"] is not True or "expected_error" in helper)
            ):
                raise ValueError("malformed comparison helper call")
            if (
                not isinstance(helper["callee"], str)
                or not helper["callee"]
                or not isinstance(helper["position"], list)
                or len(helper["position"]) != 4
                or any(type(value) is not int for value in helper["position"])
            ):
                raise ValueError("invalid comparison helper target/position")
            if "expected_error" in helper:
                expected = helper["expected_error"]
                if not isinstance(expected, dict) or set(expected) != {
                    "observation",
                    "exception",
                    "match",
                    "end_lineno",
                    "failure_sites",
                }:
                    raise ValueError("malformed expected-error helper contract")
                failures = expected["failure_sites"]
                if (
                    not isinstance(failures, list)
                    or not failures
                    or any(not isinstance(identity, str) for identity in failures)
                    or len(failures) != len(set(failures))
                ):
                    raise ValueError("invalid expected-error failure-site inventory")
                # A comparison failure may occur in the helper itself or in any
                # comparison procedure it calls normally; the exception then
                # propagates back through each linked call (exceptional chain).
                reachable = comparison_call_closure(observations, helper["callee"])
                if any(
                    observation_index.get(identity, {}).get("kind") not in NUMERICAL_KINDS
                    or comparison_scope_function(observation_index[identity]) not in reachable
                    for identity in failures
                ):
                    raise ValueError("expected-error failure sites must name numerical comparisons reachable from its helper")
                exception = getattr(builtins, expected["exception"], None) if isinstance(expected["exception"], str) else None
                context = observation_index.get(expected["observation"], {})
                if (
                    not isinstance(exception, type)
                    or not issubclass(exception, BaseException)
                    or context.get("kind") != "expected-error"
                    or context.get("expected_error_context")
                    != {"function": comparison_scope_function(observation), "end_lineno": expected["end_lineno"]}
                    or any(context.get(key) != observation.get(key) for key in EXECUTION_SCOPE_KEYS)
                    or type(expected["end_lineno"]) is not int
                    or expected["end_lineno"] < helper["position"][1]
                    or expected["match"] is not None
                    and not isinstance(expected["match"], str)
                ):
                    raise ValueError("invalid expected-error helper contract")
                if expected["match"] is not None:
                    try:
                        re.compile(expected["match"])
                    except re.error as exc:
                        raise ValueError("invalid expected-error message pattern") from exc
            selected = observation.get("required_in_invocation") or observation.get("conditional_execution", {}).get("scope")
            function = observation.get("required_in_invocation") or observation.get("conditional_execution", {}).get("function")
            if selected is None:
                raise ValueError("comparison helper call lacks a supported caller scope")
            if not (0 < helper["position"][0] <= helper["position"][1] and min(helper["position"][2:]) >= 0):
                raise ValueError("invalid comparison helper source bounds")
            declared = scope(selected, function)["calls"]
            if observation["id"] in declared and declared[observation["id"]] != helper:
                raise ValueError("conflicting comparison helper call")
            declared[observation["id"]] = helper
            scope(helper["callee"], helper["callee"])
        if "expected_rejection" in observation:
            # A comparator that is the only statement of a pytest.raises
            # (AssertionError) suite is negative evidence: its verdict is the
            # accepted rejection, carried by the context's own scope.
            context = (
                observation_index.get(observation["expected_rejection"], {})
                if isinstance(observation["expected_rejection"], str)
                else {}
            )
            context_function = comparison_scope_function(context)
            if (
                observation.get("kind") not in NUMERICAL_KINDS
                or any(key in observation for key in EXECUTION_SCOPE_KEYS)
                or "helper_call" in observation
                or context.get("kind") != "expected-error"
                or context_function is None
                or not isinstance(context.get("expected_error_context"), dict)
                or set(context["expected_error_context"]) != {"function", "end_lineno"}
                or context["expected_error_context"]["function"] != context_function
                or type(context["expected_error_context"]["end_lineno"]) is not int
            ):
                raise ValueError("invalid expected comparison rejection")
        if identity := observation.get("required_in_invocation"):
            scope(identity, identity)["comparisons"].add(observation["id"])
            scope(identity, identity)["test"] |= observation.get("required_execution") is True
        conditional = observation.get("conditional_execution")
        if conditional is None:
            continue
        if not isinstance(conditional, dict) or set(conditional) != {"function", "test_scope", "conditions", "scope"}:
            raise ValueError("malformed conditional comparison contract")
        function = conditional["function"]
        if type(conditional["test_scope"]) is not bool:
            raise ValueError("conditional test-scope flag must be boolean")
        scope(function, function)["test"] |= conditional["test_scope"]
        path = conditional["conditions"]
        if not isinstance(path, list) or not path:
            raise ValueError("conditional comparison lacks its source path")
        parent = function
        for index, descriptor in enumerate(path):
            if not isinstance(descriptor, dict) or set(descriptor) - {"terminal_returns"} != {
                "id",
                "parent",
                "condition",
                "body",
                "orelse",
            }:
                raise ValueError("malformed conditional source region")
            identity = descriptor["id"]
            if not isinstance(identity, str) or not identity or descriptor["parent"] != parent:
                raise ValueError("conditional source ancestry mismatch")
            for arm in ("condition", "body", "orelse"):
                bounds = descriptor[arm]
                if arm == "orelse" and bounds == []:
                    continue
                if not isinstance(bounds, list) or len(bounds) != 2 or any(type(value) is not int for value in bounds):
                    raise ValueError("invalid conditional source bounds")
                if not 0 < bounds[0] <= bounds[1]:
                    raise ValueError("invalid conditional source bounds")
            if "terminal_returns" in descriptor:
                returns = descriptor["terminal_returns"]
                if descriptor["parent"] != function or not isinstance(returns, dict) or not returns:
                    raise ValueError("invalid terminal-return scope")
                if any(
                    arm not in {"body", "orelse"} or type(line) is not int or not descriptor[arm] or line != descriptor[arm][1]
                    for arm, line in returns.items()
                ):
                    raise ValueError("invalid terminal-return source bounds")
            if identity in conditions and conditions[identity] != descriptor:
                raise ValueError("conflicting conditional source regions")
            conditions[identity] = descriptor
            scope(parent, function)["conditions"].add(identity)
            for arm in ("body", "orelse"):
                if descriptor[arm]:
                    scope(f"{identity}/{arm}", function)
            parent = path[index + 1]["parent"] if index + 1 < len(path) else conditional["scope"]
            if parent not in {f"{identity}/{arm}" for arm in ("body", "orelse") if descriptor[arm]}:
                raise ValueError("conditional comparison has an invalid selected suite")
        scope(parent, function)["comparisons"].add(observation["id"])
    numerical_ids = {row["id"] for row in observations if row.get("kind") in NUMERICAL_KINDS}
    verifiable = {contract["function"] for contract in scopes.values() if contract["comparisons"] & numerical_ids}
    while True:
        expanded = verifiable | {
            contract["function"]
            for contract in scopes.values()
            if any(call["callee"] in verifiable for call in contract["calls"].values())
        }
        if expanded == verifiable:
            break
        verifiable = expanded
    if any(call["callee"] not in verifiable for contract in scopes.values() for call in contract["calls"].values()):
        raise ValueError("comparison helper target has no numerical comparison contract")
    return scopes, conditions


def synthetic_comparison_invocations(observations) -> list[dict[str, Any]]:
    """Fabricate all-branch records only for completion-gate self-test fixtures."""
    observations = tuple(observations)
    scopes, conditions = comparison_scope_contracts(observations)
    observation_index = {row["id"]: row for row in observations}

    records = []

    def cyclic(identity, active):
        # A nested elif/else suite can contain recursive alternatives while
        # still providing a finite leaf. Only an unavoidable cycle blocks it.
        contract = scopes[identity]
        return any(spec["callee"] in active for spec in contract["calls"].values()) or any(
            conditions[condition]["orelse"] and all(cyclic(f"{condition}/{arm}", active) for arm in ("body", "orelse"))
            for condition in contract["conditions"]
        )

    def content(identity, active, base_case=False):
        contract = scopes[identity]
        result = {"executed": sorted(contract["comparisons"])}
        if contract["calls"]:
            if base_case and any(spec["callee"] in active for spec in contract["calls"].values()):
                raise ValueError("synthetic helper recursion has no finite base case")
            result["calls"] = [
                {
                    "call": call,
                    "invocation": invocation(spec["callee"], active, spec.get("expected_error"), spec.get("negated")),
                }
                for call, spec in sorted(contract["calls"].items())
            ]
        if contract["conditions"]:
            result["branches"] = []
            for condition in sorted(contract["conditions"]):
                arms = [
                    arm
                    for arm in ("body", "orelse")
                    if conditions[condition][arm] and not (base_case and cyclic(f"{condition}/{arm}", active))
                ]
                if not arms:
                    if conditions[condition]["orelse"]:
                        raise ValueError("synthetic helper recursion has no finite branch")
                    result["branches"].append({"condition": condition, "arm": "fallthrough", "executed": []})
                else:
                    result["branches"].extend(
                        {"condition": condition, "arm": arm, **content(f"{condition}/{arm}", active, base_case)} for arm in arms
                    )
        return result

    def line_of(identity):
        row = observation_index[identity]
        if row.get("helper_call"):
            return row["helper_call"]["position"][0]
        line = row.get("source_location", {}).get("line")
        if type(line) is not int:
            raise ValueError("synthetic exceptional invocation requires source line coordinates")
        return line

    def target_scope(identity):
        row = observation_index[identity]
        conditional = row.get("conditional_execution")
        return conditional["scope"] if conditional else row["required_in_invocation"]

    def active_path(identity):
        conditional = observation_index[identity].get("conditional_execution")
        if not conditional:
            return []
        path = conditional["conditions"]
        return [
            (descriptor["id"], (path[index + 1]["parent"] if index + 1 < len(path) else conditional["scope"]).rsplit("/", 1)[1])
            for index, descriptor in enumerate(path)
        ]

    def prefix_content(identity, stop, path, target, child, active):
        """Contents of an invocation suite whose execution stopped at ``target`` on line ``stop``.

        Obligations before the stop are completed; the suite on the active
        path is the last decision record and is itself only a prefix.
        """
        contract = scopes[identity]
        result = {"executed": sorted(key for key in contract["comparisons"] if key != target and line_of(key) < stop)}
        if contract["calls"]:
            result["calls"] = [
                {
                    "call": call,
                    "invocation": invocation(spec["callee"], active, spec.get("expected_error"), spec.get("negated")),
                }
                for call, spec in sorted(contract["calls"].items())
                if call != target and spec["position"][0] < stop
            ]
            if target in contract["calls"]:
                result["calls"].append({"call": target, "invocation": child})
        if contract["conditions"]:
            result["branches"] = []
            selected = path[0] if path else None
            for condition in sorted(contract["conditions"]):
                if selected is not None and condition == selected[0]:
                    continue
                if conditions[condition]["condition"][0] >= stop:
                    continue
                arm = "body" if conditions[condition]["body"] else "orelse"
                result["branches"].append({"condition": condition, "arm": arm, **content(f"{condition}/{arm}", active)})
            if selected is not None:
                condition, arm = selected
                result["branches"].append(
                    {
                        "condition": condition,
                        "arm": arm,
                        **prefix_content(f"{condition}/{arm}", stop, path[1:], target, child, active),
                    }
                )
        return result

    def failure_chain(callee, failure_sites):
        """Shortest ordinary-call path from ``callee`` to a declared failure site."""
        callers = {}
        queue = [callee]
        seen = {callee}
        while queue:
            function = queue.pop(0)
            sites = sorted(
                (line_of(key), key)
                for key in failure_sites
                if (
                    observation_index[key].get("required_in_invocation")
                    or observation_index[key]["conditional_execution"]["function"]
                )
                == function
            )
            if sites:
                # The last site keeps the longest completed prefix (strongest fixture).
                chain = [(function, sites[-1][1])]
                while chain[0][0] != callee:
                    caller, call = callers[chain[0][0]]
                    chain.insert(0, (caller, call))
                return chain
            for scope_identity, contract in sorted(scopes.items()):
                if contract["function"] != function:
                    continue
                for call, spec in sorted(contract["calls"].items()):
                    if "expected_error" in spec or spec["callee"] in seen:
                        continue
                    seen.add(spec["callee"])
                    callers[spec["callee"]] = (function, call)
                    queue.append(spec["callee"])
        raise ValueError("synthetic expected-error helper has no reachable failure site")

    def exceptional(identity, expected_error, active):
        exception_type = getattr(builtins, expected_error["exception"])
        name = f"{exception_type.__module__}.{exception_type.__qualname__}"
        message = regex_example(expected_error["match"]) if expected_error["match"] is not None else "synthetic failure"
        if not issubclass(AssertionError, exception_type):
            # An expected non-assertion error (error disposition) is raised by
            # the code under test before any comparison of the helper ran.
            line = int(identity.rsplit(":", 2)[1])
            contract = scopes[identity]
            record = {
                "scope": identity,
                "executed": [],
                "exception": {"type": name, "message": message, "failed": None, "line": line},
            }
            if contract["calls"]:
                record["calls"] = []
            if contract["conditions"]:
                record["branches"] = []
            return append(record)
        chain = failure_chain(identity, expected_error["failure_sites"])
        child = None
        for function, target in reversed(chain):
            stop = line_of(target)
            record = {
                "scope": function,
                **prefix_content(function, stop, active_path(target), target, child, active | {function}),
                "exception": {"type": name, "message": message, "failed": target},
            }
            if target_scope(target) is None:
                raise ValueError("synthetic failure site lacks an execution scope")
            child = append(record)
        return child

    def append(record):
        index = len(records)
        bind_helper_call_parents(record, records, index)
        records.append(record)
        return index

    def invocation(identity, active, expected_error=None, negated=False):
        if expected_error is not None:
            return exceptional(identity, expected_error, active)
        record = {"scope": identity, **content(identity, active | {identity}, identity in active)}
        if negated:
            # A negated predicate call expects a false return: the returned
            # bound is not positive evidence of this invocation.
            def strip(suite):
                suite["executed"] = [key for key in suite["executed"] if not observation_index[key].get("truthy_return")]
                for branch in suite.get("branches", ()):
                    strip(branch)

            strip(record)
        return append(record)

    for identity, scope in sorted(scopes.items()):
        if scope["function"] == identity:
            invocation(identity, set())
    return records


def regex_example(pattern: str) -> str:
    """A message matched by ``pattern``, for synthetic self-test receipts only.

    Supports the regular subset used by ``pytest.raises(match=...)`` in the
    suite (literals, classes, wildcards, repeats, groups, alternation and
    anchors); anything else is rejected rather than approximated.
    """
    try:
        from re import _parser as parser  # Python >= 3.11
    except ImportError:  # pragma: no cover - older interpreters
        import sre_parse as parser

    def emit(items) -> str:
        out = []
        for op, value in items:
            name = str(op)
            if name == "LITERAL":
                out.append(chr(value))
            elif name == "NOT_LITERAL":
                out.append("a" if value != ord("a") else "b")
            elif name == "ANY":
                out.append("x")
            elif name in {"MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT"}:
                low, _high, inner = value
                out.append(emit(inner) * low)
            elif name == "SUBPATTERN":
                out.append(emit(value[-1]))
            elif name == "ATOMIC_GROUP":
                out.append(emit(value))
            elif name == "BRANCH":
                out.append(emit(value[1][0]))
            elif name == "IN":
                first_op, first_value = value[0]
                if str(first_op) == "LITERAL":
                    out.append(chr(first_value))
                elif str(first_op) == "RANGE":
                    out.append(chr(first_value[0]))
                elif str(first_op) == "CATEGORY":
                    out.append({"CATEGORY_DIGIT": "0", "CATEGORY_SPACE": " "}.get(str(first_value), "a"))
                else:
                    raise ValueError("synthetic expected-error message cannot satisfy its pattern")
            elif name != "AT":
                raise ValueError("synthetic expected-error message cannot satisfy its pattern")
        return "".join(out)

    text = emit(parser.parse(pattern))
    if re.search(pattern, text) is None:
        raise ValueError("synthetic expected-error message cannot satisfy its pattern")
    return text


def bind_helper_call_parents(record, invocations, parent_index):
    """Store the reciprocal caller identity on each successfully linked child."""
    for call in record.get("calls", ()):
        child = call["invocation"]
        if type(child) is int and 0 <= child < parent_index:
            invocations[child]["caller"] = {"call": call["call"], "invocation": parent_index}
    for branch in record.get("branches", ()):
        bind_helper_call_parents(branch, invocations, parent_index)

"""Test-side closure of authored mixed-fidelity primaries over real engine output.

The lexical index establishes a source comparison, not an execution contract.
Both independent views must agree; this helper does not approve numerical
budgets, infer authored operands, or substitute for runtime comparison receipts.
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
from pathlib import Path

import yaml

from gdd.engine import case_binding_inventory
from gdd.engine.comparison_requirements import requirement_errors
from gdd.engine.tolerance_provenance import ToleranceSourceIndex
from scripts import check_gdd_module_ledgers

METHOD = "VM-NUM-MIXED-FIDELITY-001"


def lexical_comparisons(root: Path, path: str) -> dict:
    """Describe real source sites with the same public index used by GDD."""
    tree = ast.parse((root / path).read_text(encoding="utf-8"))
    index = ToleranceSourceIndex(tree, path)
    sites = {}
    calls = {(node.lineno, node.col_offset): node for node in ast.walk(tree) if isinstance(node, ast.Call)}
    for row in index.comparisons():
        location = row["location"]
        call = calls[location["line"], location["column"]]
        sites[row["site"]] = (
            ast.unparse(call.args[0]) if call.args else None,
            ast.unparse(call.args[1]) if len(call.args) > 1 else None,
        )
    for node in ast.walk(tree):
        described = (
            index.helper_inequality_observation(node)
            if isinstance(node, ast.Call)
            else index.inequality_observation(node)
            if isinstance(node, ast.Assert)
            else None
        )
        if described is None:
            continue
        provenance = described.get("provenance", described)
        for row in provenance["comparisons"]:
            symbols = row.get("symbols", {})
            sites[row["site"]] = (
                described.get("actual", symbols.get("actual")),
                described.get("reference", symbols.get("reference")),
            )
    return sites


def _explicit_assertion_helpers(root: Path, key: str, observations: list, cases: list, reached: set, precisions: tuple) -> set:
    """Follow synchronous source-resolved calls that execute real assertions.

    A diagnostic result may be assigned, returned or aggregated without
    suppressing the assertion. Reuse the engine's lexical resolver and case
    forwarding; a definition, callback or lazy/recovered call is not execution.
    """
    path, *parts = key.split("::")
    index = case_binding_inventory._module_index(root / path)
    test, classes = index.test(parts)
    nodes = index.assertion_dependencies(index.reachable(test, classes), include_callbacks=False)
    functions, parents, enclosing, resolve = case_binding_inventory._comparison_function_graph(nodes, Path(path))
    test_scope = case_binding_inventory.function_scope(test, path)
    edges = []
    for caller, (source, function) in functions.items():
        if isinstance(function, ast.AsyncFunctionDef) or any(
            isinstance(child, (ast.Yield, ast.YieldFrom, ast.Await)) for child in ast.walk(function)
        ):
            continue
        for call in ast.walk(function):
            if not isinstance(call, ast.Call) or enclosing(call, (ast.FunctionDef, ast.AsyncFunctionDef)) is not function:
                continue
            statement = parents.get(id(call))
            while statement is not None and not isinstance(statement, ast.stmt):
                statement = parents.get(id(statement))
            if not isinstance(statement, (ast.Expr, ast.Assign, ast.Return)):
                continue
            target = resolve(caller, call)
            if target not in functions:
                continue
            # A swallowed/expected exception, lazy expression or recovered
            # branch cannot establish ordinary positive primary execution.
            cursor = call
            supported = True
            while (cursor := parents.get(id(cursor))) is not None and cursor is not function:
                if isinstance(
                    cursor,
                    (
                        ast.Try,
                        ast.TryStar,
                        ast.With,
                        ast.AsyncWith,
                        ast.Lambda,
                        ast.BoolOp,
                        ast.IfExp,
                        ast.GeneratorExp,
                        ast.ListComp,
                        ast.SetComp,
                        ast.DictComp,
                    ),
                ):
                    supported = False
            if supported:
                edges.append((id(call), caller, target))
    per_case = []
    for precision, case in ((mode, case) for mode in precisions for case in cases):
        if precision not in case.get("precision_modes", ["full", "mixed"]):
            continue
        selected = {
            id(call)
            for call, _, _ in case_binding_inventory._case_reachable_calls(
                nodes, case["selector"], index.precision_constants[precision]
            )
        }
        reachable = set(reached)
        while True:
            expanded = reachable | {
                target for call, caller, target in edges if call in selected and (caller == test_scope or caller in reachable)
            }
            for row in observations:
                conditional = row.get("conditional_execution", {})
                caller = row.get("required_in_invocation") or conditional.get("function")
                if (
                    row.get("kind") == "comparison-helper-call"
                    and not row.get("unsupported_execution")
                    and "expected_error" not in row["helper_call"]
                    and caller in expanded
                ):
                    expanded.add(row["helper_call"]["callee"])
            if expanded == reachable:
                break
            reachable = expanded
        per_case.append(reachable)
    return set.intersection(*per_case) if per_case else set()


def primary_binding_errors(root: Path, node: dict, *, cases: int | None = None, method: str = METHOD) -> list[str]:
    """Require authored nonempty checks, lexical operands and actual node binding."""
    key = node["pytest_node"]
    checks = node.get("executable_checks")
    if checks is None and method != METHOD:
        return []  # Existing unmigrated mathematical registrations remain unmigrated.
    if not isinstance(checks, dict) or not checks:
        # requirement_errors(None) intentionally preserves legacy distinction;
        # the mixed-fidelity primary obligation is stricter than that default.
        return [f"{key}: missing authored mixed-fidelity primary"]
    count_errors = []
    count = cases if cases is not None else check_gdd_module_ledgers._actual_cases(key, count_errors)
    if count_errors or count is None:
        return [f"{key}: case inventory: {message}" for message in count_errors]
    binding = case_binding_inventory.derive_case_binding(key, count, method, "full-wavefunction")
    observations = binding["comparison"]["typed_observations"]
    # A definition reachable through an escaped callback is still lexical
    # evidence only. Follow the engine's explicit comparison-procedure calls.
    reachable_helpers = set()
    while True:
        expanded = set(reachable_helpers)
        for observation in observations:
            if observation.get("kind") != "comparison-helper-call" or observation.get("unsupported_execution"):
                continue
            conditional = observation.get("conditional_execution", {})
            caller = observation.get("required_in_invocation") or conditional.get("function")
            if observation.get("required_execution") or conditional.get("test_scope") or caller in reachable_helpers:
                expanded.add(observation["helper_call"]["callee"])
        if expanded == reachable_helpers:
            break
        reachable_helpers = expanded
    indexed = {}
    errors = []
    explicit_helpers = {}
    for identity, check in checks.items():
        site = check.get("site", "")
        path = site.split("::", 1)[0]
        if path not in indexed:
            indexed[path] = lexical_comparisons(root, path)
        source = indexed[path].get(site)
        if source is None:
            errors.append(f"{key}: {identity}: lexical comparison site is missing: {site}")
        elif source != (check.get("actual", {}).get("expression"), check.get("reference", {}).get("expression")):
            errors.append(f"{key}: {identity}: lexical operand expressions differ: {site}")
        matched = [
            observation
            for observation in observations
            if any(row["site"] == site for row in observation.get("tolerance_provenance", {}).get("comparisons", ()))
        ]
        if len(matched) != 1:
            errors.append(f"{key}: {identity}: primary is absent or ambiguous in actual node observations: {site}")
        else:
            observation = matched[0]
            conditional = observation.get("conditional_execution", {})
            caller = observation.get("required_in_invocation") or conditional.get("function")
            if observation.get("unsupported_execution"):
                errors.append(f"{key}: {identity}: unsupported primary execution: {observation['unsupported_execution']}")
            elif not (observation.get("required_execution") or conditional.get("test_scope") or caller in reachable_helpers):
                precisions = tuple(sorted(check.get("precision", {})))
                if precisions not in explicit_helpers:
                    explicit_helpers[precisions] = _explicit_assertion_helpers(
                        root, key, observations, binding["fixture"]["concrete_cases"], reachable_helpers, precisions
                    )
                if caller not in explicit_helpers[precisions]:
                    errors.append(f"{key}: {identity}: primary has no supported invocation from the test: {site}")
    errors.extend(
        f"{key}: {message}"
        for message in requirement_errors(
            checks,
            observations,
            node.get("required_precision_modes", ["full", "mixed"]),
            root=root,
            method=method,
            pytest_node=key,
            cases=binding["fixture"]["concrete_cases"],
        )
    )
    return errors


def authored_mixed_fidelity_nodes(root: Path, *, method: str = METHOD):
    """Read the authoritative YAML, so regeneration cannot erase the obligation."""
    for path in sorted((root / "gdd/ssot/guarantees").rglob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for evidence in data.get("required_evidence", ()):
            if not isinstance(evidence, dict):
                continue
            if evidence.get("verdict_method") == method:
                for node in evidence.get("nodes", ()):
                    yield data["id"], node


def authored_full_math_on_migrated_nodes(root: Path):
    """Retain existing authored full AD comparisons on nodes with a NUM partition."""
    migrated = {node["pytest_node"] for _, node in authored_mixed_fidelity_nodes(root)}
    for guarantee, node in authored_mixed_fidelity_nodes(root, method="VM-MATH-AD-ORACLE-001"):
        if (
            node["pytest_node"] in migrated
            and "full" in node.get("required_precision_modes", ["full", "mixed"])
            and node.get("executable_checks") is not None
        ):
            yield guarantee, node

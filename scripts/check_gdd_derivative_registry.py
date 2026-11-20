#!/usr/bin/env python3
"""Pin the derivative registry 1:1 to the recorder call sites of the tests (issue 789).

Source-only (no pytest, JAX or jqmc import). A test definition that directly
calls one of the fixed recorder entry points below emits derivative receipts,
so it must be a registered emitter of ``gdd/engine/derivative_registry.py``;
every registered definition must exist and directly call one. The GFMC
force/stress functions emit from their module's autouse fixture: each reviewed
spec must name an existing test function of that module, and those functions
are the only fixture emitters. A contract test that calls an entry point with
synthetic inputs on purpose carries ``gdd_synthetic_derivative_receipts`` and
is not an emitter. Direct calls only: there is no reachability analysis.
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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REGISTRY = "gdd/engine/derivative_registry.py"
SYNTHETIC_MARKER = "gdd_synthetic_derivative_receipts"
# The fixed recorder entry points: calling one directly from a test emits receipts.
ENTRY_POINTS = frozenset(
    {
        "gdd.engine.evidence.emit_completion_record",
        "tests.test_jqmc_kernel._gdd_adapters.case_receipt",
        "tests.test_jqmc_kernel._gdd_fd_adapters.record_fd_case",
        "tests.test_jqmc_kernel._generated_matrix_base.run_matrix_cell",
        "tests.test_jqmc_kernel.test_analytic_coverage_evidence._run_parameter_leaf_evidence",
        "tests.test_jqmc_kernel.test_analytic_coverage_evidence._record_validity_receipt",
        "tests.test_jqmc_kernel.test_analytic_coverage_evidence._record_continuous_regime_receipt",
        "tests.test_jqmc_kernel.test_analytic_coverage_evidence.test_numerical_regime_evidence",
        "tests.test_jqmc_kernel.test_gfmc_analytic_coverage_evidence._record_gfmc_regime_receipt",
        "tests.test_jqmc_kernel.test_gfmc_analytic_coverage_evidence.test_gfmc_numerical_regime_evidence",
        "tests.test_jqmc_kernel.test_gfmc_analytic_coverage_evidence.test_gfmc_output_liveness",
        "tests.test_jqmc_kernel.test_gfmc_analytic_coverage_evidence.test_gfmc_invalid_non_local_move",
        "tests.test_jqmc_kernel.test_mixed_precision._record_mixed_route_receipt",
        "tests.test_rs_kernel._gdd_rust_adapters.mirror_upstream_receipt",
        "tests.test_rs_kernel._gdd_rust_adapters.record_rust_bound_case",
        "tests.test_rs_kernel.test_rs_analytic_matrix._assert_close",
        "tests.test_rs_kernel.test_rs_analytic_matrix_rRe._assert_close",
    }
)


def _module_name(relative: str) -> str:
    return relative[: -len(".py")].replace("/", ".")


def _evaluate(node: ast.AST, names: dict[str, str]):
    """The literal registry table: constants, f-strings over module constants, tuples and dicts."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if isinstance(node, ast.JoinedStr):
        return "".join(_evaluate(value, names) for value in node.values)
    if isinstance(node, ast.FormattedValue) and node.conversion == -1 and node.format_spec is None:
        return _evaluate(node.value, names)
    if isinstance(node, ast.Tuple):
        return tuple(_evaluate(value, names) for value in node.elts)
    if isinstance(node, ast.Dict) and None not in node.keys:
        result = {}
        for key, value in zip(node.keys, node.values, strict=True):
            key = _evaluate(key, names)
            if key in result:
                raise ValueError(f"{REGISTRY}: duplicate registered definition {key}")
            result[key] = _evaluate(value, names)
        return result
    raise ValueError(f"{REGISTRY}: STATIC_EMITTERS must be literal, found {ast.dump(node)[:80]}")


def registry_table(root: Path = ROOT) -> tuple[dict[str, tuple[str, str]], str]:
    """``STATIC_EMITTERS`` and ``GFMC_FORCE_MODULE`` read from the registry source."""
    tree = ast.parse((root / REGISTRY).read_text(encoding="utf-8"))
    names: dict[str, str] = {}
    table = gfmc = None
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            name = statement.targets[0].id
            if name == "STATIC_EMITTERS":
                table = _evaluate(statement.value, names)
            elif isinstance(statement.value, (ast.Constant, ast.JoinedStr)):
                try:
                    names[name] = _evaluate(statement.value, names)
                except ValueError:
                    continue
                if name == "GFMC_FORCE_MODULE":
                    gfmc = names[name]
    if not isinstance(table, dict) or gfmc is None:
        raise ValueError(f"{REGISTRY}: STATIC_EMITTERS or GFMC_FORCE_MODULE is missing")
    return table, gfmc


def _imports(statements, module: str) -> dict[str, str]:
    """Local name -> qualified name for the imports among ``statements`` (one level)."""
    bound = {}
    package = module.rsplit(".", 1)[0]
    for statement in statements:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                bound[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
        elif isinstance(statement, ast.ImportFrom):
            base = statement.module or ""
            if statement.level:
                base = ".".join([*package.split(".")[: len(package.split(".")) - statement.level + 1], base]).strip(".")
            for alias in statement.names:
                bound[alias.asname or alias.name] = f"{base}.{alias.name}"
    return bound


def _qualified(call: ast.Call, module: str, bound: dict[str, str], defined: set[str]) -> str | None:
    function = call.func
    if isinstance(function, ast.Name):
        if function.id in bound:
            return bound[function.id]
        return f"{module}.{function.id}" if function.id in defined else None
    if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name) and function.value.id in bound:
        return f"{bound[function.value.id]}.{function.attr}"
    return None


def _has_marker(function: ast.AST, marker: str) -> bool:
    for decorator in getattr(function, "decorator_list", ()):
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Attribute) and target.attr == marker:
            return True
    return False


def test_definitions(tree: ast.Module):
    """``(qualname, function node)`` of every module-level test and ``Test*`` class method."""
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name.startswith("test"):
            yield statement.name, statement
        elif isinstance(statement, ast.ClassDef) and statement.name.startswith("Test"):
            for child in statement.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test"):
                    yield f"{statement.name}::{child.name}", child


def recorder_call_sites(root: Path = ROOT) -> tuple[dict[str, list[str]], set[str], dict[str, set[str]]]:
    """Test definitions that directly call an entry point, synthetic ones, and every test per file."""
    calling: dict[str, list[str]] = {}
    synthetic: set[str] = set()
    tests: dict[str, set[str]] = {}
    for path in sorted((root / "tests").rglob("test_*.py")):
        relative = path.relative_to(root).as_posix()
        module = _module_name(relative)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module_bound = _imports(tree.body, module)
        defined = {statement.name for statement in tree.body if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))}
        tests[relative] = set()
        for qualname, function in test_definitions(tree):
            definition = f"{relative}::{qualname}"
            tests[relative].add(qualname)
            bound = {**module_bound, **_imports([node for node in ast.walk(function) if isinstance(node, ast.stmt)], module)}
            called = sorted(
                {
                    name
                    for node in ast.walk(function)
                    if isinstance(node, ast.Call) and (name := _qualified(node, module, bound, defined)) in ENTRY_POINTS
                }
            )
            if not called:
                continue
            if _has_marker(function, SYNTHETIC_MARKER):
                synthetic.add(definition)
            else:
                calling[definition] = called
    return calling, synthetic, tests


def registry_errors(root: Path = ROOT, gfmc_functions=None) -> list[str]:
    errors = []
    try:
        table, gfmc_module = registry_table(root)
    except (OSError, SyntaxError, ValueError) as exc:
        return [str(exc)]
    if gfmc_functions is None:
        from gdd.engine.derivative_requirements import _GFMC_FUNCTION_SPECS

        gfmc_functions = [spec.function_name for spec in _GFMC_FUNCTION_SPECS]
    calling, synthetic, tests = recorder_call_sites(root)
    for definition in sorted(set(calling) - set(table)):
        errors.append(f"{definition} calls the recorder entry point(s) {calling[definition]} but is not a registered emitter")
    for definition in sorted(set(table) - set(calling)):
        errors.append(f"registered emitter {definition} has no direct recorder call (or does not exist)")
    for definition in sorted(synthetic & set(table)):
        errors.append(f"{definition} is registered but marked {SYNTHETIC_MARKER}")
    gfmc_tests = tests.get(gfmc_module, set())
    if len(set(gfmc_functions)) != len(gfmc_functions):
        errors.append("the reviewed GFMC function specs name a function twice")
    for name in sorted(set(gfmc_functions) - gfmc_tests):
        errors.append(f"GFMC spec {name} names no test function of {gfmc_module}")
    for name in sorted(set(gfmc_functions)):
        if f"{gfmc_module}::{name}" in table or f"{gfmc_module}::{name}" in calling:
            errors.append(f"{gfmc_module}::{name} emits from its fixture and must not also be a direct emitter")
    return errors


def main() -> int:
    errors = registry_errors()
    if errors:
        print("GDD derivative registry contract failed:\n- " + "\n- ".join(errors), file=sys.stderr)
        return 1
    table, _gfmc = registry_table()
    print(f"GDD derivative registry: {len(table)} direct emitters match their recorder call sites 1:1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Static CI gate for the derivative GDD contract.

This checker intentionally does not import jqmc, JAX, pytest, or the evidence
modules.  It is safe to run in the small pre-commit environment and checks the
contract against source syntax and the live, dependency-free requirement
expansion. Runtime closure is checked separately by
``tests/test_jqmc_kernel/test_gdd_derivatives_contract.py``.
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
from itertools import combinations, product
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "gdd/ssot/guarantees/derivatives.yaml"
SHELL_STRUCTURES = {"closed-shell", "open-shell"}
SPIN_RESTRICTIONS = {"not-applicable", "spin-restricted", "spin-unrestricted"}
sys.path.insert(0, str(ROOT))

from gdd.engine.production_entry_discovery import (
    discover_direct_autodiff_calls,
    discover_external_derivative_references,
)
from gdd.engine.ledger import load_ledger
from gdd.engine.fd_comparison_contract import fd_source_contract_errors
from gdd.engine.derivative_inactivity import inactivity_contract_errors


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _literal(node: ast.AST, constants: dict[str, Any]) -> Any:
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    return ast.literal_eval(node)


def _module_constants(tree: ast.Module) -> dict[str, Any]:
    constants: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if value is None:
            continue
        try:
            literal = ast.literal_eval(value)
        except (ValueError, TypeError):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                constants[target.id] = literal
    return constants


def _assignment(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constants = _module_constants(tree)
    if name not in constants:
        raise ValueError(f"{path.relative_to(ROOT)} does not define literal {name}")
    return constants[name]


def _find_test(tree: ast.Module, class_name: str | None, function_name: str) -> ast.FunctionDef | None:
    body: list[ast.stmt] = tree.body
    if class_name is not None:
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
        if len(classes) != 1:
            return None
        body = classes[0].body
    matches = [node for node in body if isinstance(node, ast.FunctionDef) and node.name == function_name]
    return matches[0] if len(matches) == 1 else None


def _parameter_case_count(function: ast.FunctionDef, constants: dict[str, Any]) -> int:
    count = 1
    for decorator in function.decorator_list:
        if not isinstance(decorator, ast.Call) or not _dotted_name(decorator.func).endswith(".parametrize"):
            continue
        if len(decorator.args) < 2:
            raise ValueError(f"line {decorator.lineno}: parametrize has no values")
        values = _literal(decorator.args[1], constants)
        count *= len(values)
    return count


def _test_source_tokens(function: ast.FunctionDef) -> set[str]:
    tokens = {_dotted_name(node) for node in ast.walk(function) if isinstance(node, (ast.Name, ast.Attribute))}
    tokens.add(function.name)
    return {token.lower() for token in tokens if token}


def _check_selector_mappings(
    value: Any,
    dimensions: dict[str, Any],
    label: str,
    errors: list[str],
) -> None:
    """Require every ``*_by_<axis>`` map to cover its declared selector exactly."""
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str) and "_by_" in key:
                axis = key.rsplit("_by_", 1)[1]
                declared = dimensions.get(axis)
                expected = set(declared if isinstance(declared, list) else [declared])
                actual = set(child) if isinstance(child, dict) else set()
                if not expected or None in expected or actual != expected:
                    errors.append(
                        f"{label}: {key} must map every {axis} selector exactly; "
                        f"expected={sorted(map(str, expected))}, actual={sorted(map(str, actual))}"
                    )
            _check_selector_mappings(child, dimensions, label, errors)
    elif isinstance(value, list):
        for child in value:
            _check_selector_mappings(child, dimensions, label, errors)


def _check_fd_node(node: dict[str, Any], errors: list[str]) -> None:
    label = f"FD evidence {node.get('id', '<missing-id>')}"
    required = {
        "id",
        "pytest_node",
        "fixture",
        "case_binding",
        "derivative",
        "finite_difference",
        "precision",
    }
    missing = sorted(required - set(node))
    if missing:
        errors.append(f"{label}: missing fields {missing}")
        return
    fixture = node["fixture"]
    if not isinstance(fixture, dict) or not {"builder_id", "input_provenance", "dimensions"} <= set(fixture):
        errors.append(f"{label}: fixture must declare builder_id, input_provenance, and dimensions")
    case_binding = node["case_binding"]
    binding_keys = {"input", "sources", "wavefunction", "electron_configuration", "comparison"}
    if not isinstance(case_binding, dict) or set(case_binding) != binding_keys:
        errors.append(f"{label}: case_binding must contain exactly {sorted(binding_keys)}")
    else:
        input_binding = case_binding["input"]
        if not isinstance(input_binding, dict) or input_binding.get("kind") not in {
            "parameter-set",
            "committed-artifact",
        }:
            errors.append(f"{label}: case_binding.input must declare a supported kind")
        expected_kind = {
            "deterministic-in-test-synthetic": "parameter-set",
            "committed-input-derived": "committed-artifact",
        }.get(fixture.get("input_provenance"))
        if expected_kind is None or input_binding.get("kind") != expected_kind:
            errors.append(
                f"{label}: audit input kind {input_binding.get('kind')!r} does not match "
                f"fixture provenance {fixture.get('input_provenance')!r}"
            )
        for key in ("sources", "wavefunction", "electron_configuration", "comparison"):
            if not isinstance(case_binding[key], dict) or not case_binding[key]:
                errors.append(f"{label}: case_binding.{key} must be a nonempty mapping")
        electrons = case_binding["electron_configuration"]
        if electrons.get("mode") == "required":
            for key in ("position_source", "generator", "configurations", "cond_G"):
                if key not in electrons:
                    errors.append(f"{label}: required electron configuration omits {key}")
        wavefunction = case_binding["wavefunction"]
        shell = wavefunction.get("shell_structure")
        restriction = wavefunction.get("spin_restriction")
        if shell is not None and shell not in SHELL_STRUCTURES:
            errors.append(f"{label}: invalid wavefunction.shell_structure {shell!r}")
        if restriction is not None and restriction not in SPIN_RESTRICTIONS:
            errors.append(f"{label}: invalid wavefunction.spin_restriction {restriction!r}")
        electron_counts = electrons.get("electron_counts")
        if isinstance(electron_counts, dict):
            up = electron_counts.get("up")
            down = electron_counts.get("down")
            if isinstance(up, int) and isinstance(down, int):
                expected_shell = "closed-shell" if up == down else "open-shell"
                if shell is None:
                    errors.append(f"{label}: literal electron counts require wavefunction.shell_structure")
                elif shell != expected_shell:
                    errors.append(
                        f"{label}: wavefunction.shell_structure={shell!r} contradicts "
                        f"electron_counts={electron_counts}; expected {expected_shell!r}"
                    )
                if restriction is None:
                    errors.append(f"{label}: literal electron counts require wavefunction.spin_restriction")
                elif wavefunction.get("scope") in {"full-wavefunction", "determinant-component"} and (
                    (restriction == "not-applicable") != (down == 0)
                ):
                    errors.append(
                        f"{label}: wavefunction.spin_restriction={restriction!r} contradicts electron_counts={electron_counts}"
                    )
                elif wavefunction.get("scope") not in {"full-wavefunction", "determinant-component"} and (
                    restriction != "not-applicable"
                ):
                    errors.append(f"{label}: a component without determinant orbitals has no spin restriction")
        dimensions = fixture.get("dimensions", {}) if isinstance(fixture, dict) else {}
        _check_selector_mappings(case_binding, dimensions, label, errors)
        source_cases = case_binding["sources"].get("cases")
        if source_cases is not None:
            if not isinstance(source_cases, list) or not source_cases:
                errors.append(f"{label}: case_binding.sources.cases must be a nonempty list")
            else:
                selectors = [case.get("when") for case in source_cases if isinstance(case, dict)]
                axes = {key for selector in selectors if isinstance(selector, dict) for key in selector}
                if len(axes) != 1:
                    errors.append(f"{label}: source cases must partition exactly one selector axis")
                else:
                    axis = next(iter(axes))
                    declared = dimensions.get(axis)
                    expected = set(declared if isinstance(declared, list) else [declared])
                    actual = {
                        selector[axis] for selector in selectors if isinstance(selector, dict) and set(selector) == {axis}
                    }
                    if actual != expected or len(selectors) != len(actual):
                        errors.append(
                            f"{label}: source cases must cover {axis} exactly once; "
                            f"expected={sorted(map(str, expected))}, actual={sorted(map(str, actual))}"
                        )
    if node["precision"] != "full":
        errors.append(f"{label}: FD correctness evidence must use full precision")

    parts = str(node["pytest_node"]).split("::")
    if len(parts) not in (2, 3):
        errors.append(f"{label}: invalid pytest_node {node['pytest_node']!r}")
        return
    relative = Path(parts[0])
    path = ROOT / relative
    if not path.is_file():
        errors.append(f"{label}: missing evidence file {relative}")
        return
    class_name = parts[1] if len(parts) == 3 else None
    function_name = parts[-1]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = _find_test(tree, class_name, function_name)
    if function is None:
        errors.append(f"{label}: pytest function does not exist: {node['pytest_node']}")
        return

    decorators = {_dotted_name(item.func if isinstance(item, ast.Call) else item) for item in function.decorator_list}
    if not ({"pytest.mark.light", "pytest.mark.heavy"} & decorators):
        errors.append(f"{label}: evidence carries neither the light nor the heavy marker")
    forbidden = {name for name in decorators if name.endswith((".skip", ".skipif", ".xfail"))}
    if forbidden:
        errors.append(f"{label}: evidence contains a green escape marker: {sorted(forbidden)}")
    availability_markers = {name for name in decorators if name.startswith("requires_")}
    if availability_markers and not fixture.get("availability_probe"):
        errors.append(
            f"{label}: conditional fixture marker {sorted(availability_markers)} requires a blocking availability_probe"
        )

    try:
        actual_cases = _parameter_case_count(function, _module_constants(tree))
    except (ValueError, TypeError) as exc:
        errors.append(f"{label}: cannot evaluate pytest parameter count: {exc}")
    else:
        required_ids = _derive_fd_receipt_ids(node, errors)
        if actual_cases != len(required_ids):
            errors.append(
                f"{label}: declared selectors require {len(required_ids)} pytest cases, source defines {actual_cases}"
            )

    tokens = _test_source_tokens(function)
    has_ad = any("autodiff" in token or token.endswith(("jax.grad", "jax.jacrev")) or "_ad" in token for token in tokens)
    has_fd = any("numerical" in token or "debug" in token or "_fd" in token or "fd_" in token for token in tokens)
    has_comparison = any(token.endswith("assert_allclose") for token in tokens)
    if not (has_ad and has_fd and has_comparison):
        errors.append(f"{label}: source no longer visibly contains AD, FD, and assert_allclose routes")


def _derive_fd_receipt_ids(node: dict[str, Any], errors: list[str]) -> set[str]:
    """Expand one FD evidence node's id_dimensions into its receipt-ID universe."""
    label = f"FD evidence {node.get('id', '<missing-id>')}"
    dimensions = node.get("fixture", {}).get("dimensions", {})
    id_dimensions = node.get("id_dimensions")
    if not isinstance(id_dimensions, list) or not id_dimensions:
        errors.append(f"{label}: id_dimensions must list the dimensions encoded in receipt IDs")
        return set()
    if list(id_dimensions) != sorted(id_dimensions) or len(id_dimensions) != len(set(id_dimensions)):
        errors.append(f"{label}: id_dimensions must be unique and sorted: {id_dimensions}")
        return set()
    unknown = sorted(set(id_dimensions) - set(dimensions))
    if unknown:
        errors.append(f"{label}: id_dimensions reference undeclared fixture dimensions {unknown}")
        return set()
    combos: list[tuple[tuple[str, Any], ...]] = [()]
    for key in id_dimensions:
        values = dimensions[key] if isinstance(dimensions[key], list) else [dimensions[key]]
        combos = [prefix + ((key, value),) for prefix in combos for value in values]
    derived = {f"FD::{node['id']}::" + ",".join(f"{key}={value}" for key, value in combo) for combo in combos}
    if len(derived) != len(combos):
        errors.append(f"{label}: duplicate selector values collapse distinct FD declarations")
    return derived


def _check_fd_receipt_id_universe(nodes: list[dict[str, Any]], errors: list[str]) -> None:
    """Validate authored selectors, not a second hand-copied ID inventory."""
    seen: set[str] = set()
    for node in nodes:
        identifiers = _derive_fd_receipt_ids(node, errors)
        if seen & identifiers:
            errors.append(f"FD receipt IDs are duplicated: {sorted(seen & identifiers)}")
        seen.update(identifiers)
    if not seen:
        errors.append("FD receipt requirement universe must not be empty")


def _class_string_constants(path: Path, class_name: str) -> set[str]:
    """Every string literal inside one class body of a source file (AST, no import)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    if len(classes) != 1:
        raise ValueError(f"{path.relative_to(ROOT)} must define exactly one {class_name}")
    return {node.value for node in ast.walk(classes[0]) if isinstance(node, ast.Constant) and isinstance(node.value, str)}


def _inventory_row_ids(path: Path, name: str) -> list[str]:
    """First-column ID of every row in one engine inventory tuple (AST, no import)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name and isinstance(node.value, ast.Tuple):
            ids = []
            for row in node.value.elts:
                if not isinstance(row, ast.Tuple) or not row.elts:
                    raise ValueError(f"{name} row is not a tuple with a leading ID")
                first = row.elts[0]
                if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
                    raise ValueError(f"{name} row has a non-literal leading ID")
                ids.append(first.value)
            return ids
    raise ValueError(f"{path.relative_to(ROOT)} does not define tuple {name}")


def _check_engine_vocabularies(contract: dict[str, Any], errors: list[str]) -> None:
    """Verify contract vocabularies and B/C/P/R inventories against the engine source."""
    engine = ROOT / "gdd/engine/derivative_requirements.py"
    common = contract.get("common_contract", {})
    try:
        outcome_constants = _class_string_constants(engine, "ExpectedOutcome")
        proof_constants = _class_string_constants(engine, "ProofKind")
        requirement_constants = _class_string_constants(engine, "Requirement")
    except (OSError, SyntaxError, ValueError) as exc:
        errors.append(str(exc))
        return
    declared_outcomes = {str(value).replace("-", "/") for value in common.get("allowed_outcomes", [])}
    if not declared_outcomes or not declared_outcomes <= outcome_constants:
        errors.append(
            f"common_contract.allowed_outcomes differ from the engine ExpectedOutcome vocabulary: "
            f"{sorted(declared_outcomes - outcome_constants)}"
        )
    declared_proofs = {str(value).replace("-", "_") for value in common.get("proof_kinds", [])}
    if not declared_proofs or not declared_proofs <= proof_constants:
        errors.append(
            f"common_contract.proof_kinds differ from the engine ProofKind vocabulary: "
            f"{sorted(declared_proofs - proof_constants)}"
        )
    declared_families = {item.get("family") for item in contract.get("guarantees", []) if item.get("family") is not None}
    engine_families = {value for value in requirement_constants if value and not any(ch.isspace() for ch in value)}
    if not declared_families <= engine_families:
        errors.append(
            f"guarantee family tokens are unknown to the engine Requirement contract: "
            f"{sorted(declared_families - engine_families)}"
        )

    inventory = ROOT / "gdd/engine/derivative_inventory.py"
    inventories = {
        "reviewed_branch_inventory": "PRODUCTION_BRANCH_INVENTORY",
        "reviewed_validity_inventory": "PRODUCTION_VALIDITY_CONSTRAINTS",
        "reviewed_parameter_leaf_inventory": "PARAMETER_LEAF_INVENTORY",
        "reviewed_numerical_regime_inventory": "PRODUCTION_NUMERICAL_REGIMES",
    }
    for table in inventories.values():
        try:
            row_ids = _inventory_row_ids(inventory, table)
        except (OSError, SyntaxError, ValueError) as exc:
            errors.append(str(exc))
            continue
        if not row_ids or len(row_ids) != len(set(row_ids)):
            errors.append(f"engine table {table} must contain nonempty, unique requirement IDs")


def _require_path(value: str, label: str, errors: list[str]) -> None:
    if not (ROOT / value).exists():
        errors.append(f"{label}: missing path {value}")


def _top_level_definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}


def _assignment_string_values(path: Path, assignment_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == assignment_name for target in targets):
            matches.append(node.value)
    if len(matches) != 1 or matches[0] is None:
        raise ValueError(f"{path.relative_to(ROOT)} must define exactly one {assignment_name}")
    return {node.value for node in ast.walk(matches[0]) if isinstance(node, ast.Constant) and isinstance(node.value, str)}


def _check_production_entry_discovery(contract: dict[str, Any], guarantees: list[dict[str, Any]], errors: list[str]) -> None:
    declaration = contract.get("domain", {}).get("production_entry_discovery", {})
    if declaration.get("operation") != "derivatives.production_entry_discovery.v1":
        errors.append("production entry discovery operation is missing or unsupported")
        return

    authority = declaration.get("authority_model", {})
    if authority != {
        "expected-boundary": "this-SSoT",
        "observed-boundary": "source-AST-discovery",
        "obligation-universe-source": "guarantees-and-fixture-contracts-only",
        "rule": authority.get("rule"),
    } or not authority.get("rule"):
        errors.append("production discovery must keep SSoT expectations independent from source observations")

    source_scope = declaration.get("source_scope", {})
    consumers = source_scope.get("declared-consumer-paths", [])
    consumer_paths = tuple(item.get("path", "") for item in consumers)
    if len(consumer_paths) != len(set(consumer_paths)) or any(not path for path in consumer_paths):
        errors.append("production derivative consumer paths must be present and unique")
    for path in consumer_paths:
        _require_path(path, "production derivative consumer", errors)
    declared_consumers = {consumer for item in consumers for consumer in item.get("consumers", [])}
    family_consumers = {
        consumer
        for family in contract.get("domain", {}).get("derivative_families", {}).values()
        for consumer in family.get("consumers", [])
    }
    if declared_consumers != family_consumers:
        errors.append(
            "production derivative consumer source mapping differs from the guarantee consumer universe: "
            f"mapped={sorted(declared_consumers)}, guarantees={sorted(family_consumers)}"
        )

    external = declaration.get("external_reference_contract", {})
    public_entries = {
        entry
        for family in contract.get("domain", {}).get("derivative_families", {}).values()
        for entry in family.get("public_entries", [])
    }
    reviewed_support = external.get("reviewed-non-public-references", [])
    support_targets = [item.get("target") for item in reviewed_support]
    if None in support_targets or len(support_targets) != len(set(support_targets)):
        errors.append("reviewed non-public derivative references must have unique targets")
    overlap = public_entries & set(support_targets)
    if overlap:
        errors.append(f"public derivative entries cannot also be non-public references: {sorted(overlap)}")
    expected_targets = public_entries | set(support_targets)
    known_guarantees = {item["id"] for item in guarantees}
    allowed_roles = {"contract-vocabulary", "production-adapter", "result-adapter"}
    for item in reviewed_support:
        if item.get("role") not in allowed_roles:
            errors.append(f"{item.get('target')}: invalid non-public derivative role {item.get('role')!r}")
        item_guarantees = set(item.get("guarantees", []))
        if not item_guarantees or not item_guarantees <= known_guarantees:
            errors.append(f"{item.get('target')}: unknown or empty guarantee binding {sorted(item_guarantees)}")

    try:
        observed = discover_external_derivative_references(ROOT)
    except (OSError, SyntaxError, ValueError) as exc:
        errors.append(f"production derivative source discovery failed: {exc}")
    else:
        observed_targets = set(observed)
        missing = sorted(expected_targets - observed_targets)
        unknown = sorted(observed_targets - expected_targets)
        if missing:
            errors.append(f"production derivative references disappeared without SSoT review: {missing}")
        if unknown:
            details = {target: [f"{loc.path}:{loc.line} ({loc.scope})" for loc in observed[target]] for target in unknown}
            errors.append(f"unmapped production derivative references discovered: {details}")

    direct = declaration.get("direct_autodiff_contract", {})
    reviewed_calls = direct.get("reviewed-callsites", [])
    expected_call_ids = [item.get("id") for item in reviewed_calls]
    if None in expected_call_ids or len(expected_call_ids) != len(set(expected_call_ids)):
        errors.append("reviewed direct-autodiff callsites must have unique semantic IDs")
    for item in reviewed_calls:
        role = item.get("role")
        activation = item.get("activation")
        if role not in {"explicitly-disabled-alternate-backend", "debug-oracle"}:
            errors.append(f"{item.get('id')}: invalid direct-autodiff role {role!r}")
        if role == "explicitly-disabled-alternate-backend" and activation != "ANALYTIC_GRAD_ENABLED-is-false":
            errors.append(f"{item.get('id')}: alternate backend must require an explicit analytical-disable switch")
        if role == "debug-oracle" and activation != "debug-class-only":
            errors.append(f"{item.get('id')}: debug oracle must remain debug-class-only")
        if item.get("guarantee") not in known_guarantees:
            errors.append(f"{item.get('id')}: unknown guarantee {item.get('guarantee')!r}")
    try:
        observed_calls = discover_direct_autodiff_calls(ROOT, consumer_paths)
    except (OSError, SyntaxError, ValueError) as exc:
        errors.append(f"direct-autodiff source discovery failed: {exc}")
    else:
        observed_call_ids = {item.id for item in observed_calls}
        expected_call_set = set(expected_call_ids)
        missing = sorted(expected_call_set - observed_call_ids)
        unknown = sorted(observed_call_ids - expected_call_set)
        if missing:
            errors.append(f"reviewed direct-autodiff callsites disappeared: {missing}")
        if unknown:
            errors.append(f"unmapped direct-autodiff callsites discovered: {unknown}")


def _check_existing_test_integration(contract: dict[str, Any], errors: list[str]) -> None:
    integration = contract["existing_test_integration"]
    matrix = integration.get("generated_continuous_matrix", {})
    if matrix.get("evidence_collection") != "connected":
        errors.append("generated continuous matrix evidence collection must remain connected")

    matrix_path = ROOT / "tests/test_jqmc_kernel/_generated_matrix_base.py"
    matrix_definitions = _top_level_definitions(matrix_path)
    for symbol in ("MatrixCaseBinding", "matrix_case_binding", "run_matrix_cell"):
        if symbol not in matrix_definitions:
            errors.append(f"generated matrix GDD adapter is missing {symbol}")

    geminal_axis = matrix.get("geminal_fixture_axis", {})
    expected_gem_tokens = ("SD", "AGPn", "AGP", "AGP_MO")
    if tuple(geminal_axis.get("expected-tokens", ())) != expected_gem_tokens:
        errors.append("generated matrix must declare SD, AGPn, AO-AGP, and MO-AGP fixture tokens")
    try:
        actual_gem_tokens = _assignment(ROOT / "gdd/engine/derivative_matrix.py", "_GEMS")
    except (OSError, SyntaxError, ValueError) as exc:
        errors.append(str(exc))
    else:
        if actual_gem_tokens != expected_gem_tokens:
            errors.append(f"generated matrix geminal axis is {actual_gem_tokens}, expected {expected_gem_tokens}")
    expected_mo_semantics = {"determinant-kind": "AGP", "representation": "MO"}
    if geminal_axis.get("semantics", {}).get("AGP_MO") != expected_mo_semantics:
        errors.append("AGP_MO must be declared as determinant kind AGP in MO representation")
    molecular_matrix_path = ROOT / "tests/test_jqmc_kernel/test_analytic_grad_matches_autograd_generated_molecular.py"
    molecular_definitions = _top_level_definitions(molecular_matrix_path)
    if "test_mo_agp_fixture_axis_is_explicit_and_buildable" not in molecular_definitions:
        errors.append("generated matrix is missing its explicit MO-AGP structural fixture test")

    engine_path = ROOT / "gdd/engine/evidence.py"
    engine_definitions = _top_level_definitions(engine_path)
    required_engine = {
        "InputBinding",
        "ElectronConfigurationBinding",
        "ComparisonBinding",
        "ObservationBinding",
        "CaseReceipt",
        "CompositeCaseReceipt",
        "emit_completion_record",
        "observe_electron_configurations",
        "observe_no_electron_configuration",
        "observe_allclose",
        "observe_exact",
        "observe_non_numerical",
    }
    if not required_engine <= engine_definitions:
        errors.append(f"GDD receipt engine is missing {sorted(required_engine - engine_definitions)}")

    targeted = integration.get("targeted_cases", {})
    if targeted.get("evidence_collection") != "connected":
        errors.append("targeted derivative integration must keep evidence collection connected for every obligation")
    manifest_definitions = _top_level_definitions(ROOT / "tests/test_jqmc_kernel/analytic_coverage_registration.py")
    if "targeted_binding_issues" not in manifest_definitions:
        errors.append("targeted derivative integration is missing its binding completion operation")
    gfmc = integration.get("gfmc_broad_matrix", {})
    if gfmc.get("evidence_collection") != "connected":
        errors.append("GFMC broad integration must keep evidence collection connected for every obligation")
    from gdd.engine.derivative_reference_forms import reference_form_errors

    errors.extend(reference_form_errors(gfmc))
    if "gfmc_matrix_binding_issues" not in manifest_definitions:
        errors.append("GFMC broad integration is missing its binding completion operation")
    regimes = integration.get("numerical_regimes", {})
    if regimes.get("evidence_collection") != "connected":
        errors.append("numerical-regime integration must keep evidence collection connected for every obligation")
    if "regime_binding_issues" not in manifest_definitions:
        errors.append("numerical-regime integration is missing its binding completion operation")
    fd = integration.get("fd_autodiff_sentinels", {})
    if fd.get("evidence_collection") != "connected":
        errors.append("FD/autodiff integration must keep evidence collection connected for every declared selector")
    if "fd_binding_issues" not in manifest_definitions:
        errors.append("FD/autodiff integration is missing its binding completion operation")
    mixed = integration.get("mixed_precision_routes", {})
    if mixed.get("evidence_collection") != "connected":
        errors.append("mixed-precision integration must keep evidence collection connected for every route")
    if "mixed_binding_issues" not in manifest_definitions:
        errors.append("mixed-precision integration is missing its binding completion operation")
    rust = integration.get("rust_xla_parity", {})
    if rust.get("evidence_collection") != "connected":
        errors.append("Rust/XLA evidence collection must remain connected through cross-worker aggregation")
    if rust.get("explicit-not-applicable") != 0:
        errors.append("Rust/XLA integration must require all upstream obligations with no N/A subtraction")

    # The collected cases close the exact universes (collection_closure) and
    # every registered case checks its own receipts at teardown (derivative_registry).
    completion = integration.get("cross_worker_completion", {})
    if completion.get("evidence_collection") != "connected":
        errors.append("derivative evidence collection must be connected")
    if completion.get("required_lanes") != ["xla-full", "mixed", "rust-full", "rust-mixed"]:
        errors.append("derivative completion lanes are incorrect")
    for key in ("collection_closure", "derivative_registry", "registry_checker"):
        if not (ROOT / str(completion.get(key, ""))).is_file():
            errors.append(f"derivative completion is missing its {key}")
    from gdd.engine.derivative_closure import PINNED_UNIVERSE_SHA256

    if (completion.get("exact_universes") or {}).get("lane-sha256") != PINNED_UNIVERSE_SHA256:
        errors.append("the per-lane derivative universe sha256 pins differ from gdd.engine.derivative_closure")
    if integration.get("remaining_existing_tests") != {"evidence_collection": "connected", "scopes": []}:
        errors.append("all existing derivative test scopes must have connected evidence collection")

    source_registry = yaml.safe_load((ROOT / matrix["input_binding"]["source_registry"]).read_text(encoding="utf-8"))
    known_source_ids = set(source_registry.get("sources", {}))
    try:
        bound_source_ids = _assignment_string_values(matrix_path, "_GDD_SOURCE_ID_BY_STEM")
    except (OSError, SyntaxError, ValueError) as exc:
        errors.append(str(exc))
    else:
        unknown_sources = sorted({value for value in bound_source_ids if value.startswith("WF-SRC-")} - known_source_ids)
        if unknown_sources:
            errors.append(f"generated matrix uses unknown committed source IDs {unknown_sources}")

    electron = matrix.get("electron_configuration", {})
    if electron.get("seed") != 0 or electron.get("configuration-count") != 2:
        errors.append("generated matrix must bind the actual seed=0, two-walker electron batch")
    if electron.get("geminal-matrix-conditioning") != "not-limited":
        errors.append("generated matrix cannot claim a cond(G) bound it does not enforce")
    try:
        harness_route = _assignment(matrix_path, "_MATRIX_PRODUCTION_ROUTE")
    except (OSError, SyntaxError, ValueError) as exc:
        errors.append(str(exc))
    else:
        declared_route = electron.get("production-route-capture")
        if not declared_route or declared_route != harness_route:
            errors.append(f"generated matrix route capture differs: contract={declared_route!r}, harness={harness_route!r}")
    harness_path = ROOT / "tests/test_jqmc_kernel/_grad_fidelity_harness.py"
    harness_tree = ast.parse(harness_path.read_text(encoding="utf-8"), filename=str(harness_path))
    draw_batch = _find_test(harness_tree, None, "_draw_batch")
    if draw_batch is None or "np.random.pcg64" not in _test_source_tokens(draw_batch):
        errors.append("generated electron batch must instantiate the contracted PCG64 algorithm explicitly")

    comparisons = yaml.safe_load((ROOT / "gdd/ssot/verdicts.yaml").read_text(encoding="utf-8"))
    policy_ids = {row["id"] for row in comparisons["comparison_policies"]}
    matrix_comparison = matrix.get("comparison", {})
    matrix_policy_ids = {matrix_comparison.get(precision, {}).get("policy-id") for precision in ("full", "mixed")}
    if None in matrix_policy_ids or not matrix_policy_ids <= policy_ids:
        errors.append("generated matrix references an unknown comparison policy")

    for relative in matrix.get("pytest_modules", []):
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"generated matrix pytest module is missing: {relative}")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        function = _find_test(tree, None, "test_matrix_cell")
        if function is None:
            errors.append(f"{relative} has no test_matrix_cell")
            continue
        tokens = _test_source_tokens(function)
        if not {"run_matrix_cell", "receipt.status"} <= tokens:
            errors.append(f"{relative} does not consume and assert the runtime GDD receipt")


def _check_live_requirement_snapshot(baseline: dict[str, Any], errors: list[str]) -> None:
    """Validate declared semantics and live expansion, never archival digest pins."""
    try:
        from gdd.engine import derivative_requirements as requirements
        from gdd.engine import derivative_matrix as matrix

        cases = requirements.REQUIRED_CASES
        if not cases or len({case.id for case in cases}) != len(cases):
            errors.append("live derivative requirements must have nonempty, unique IDs")
        for case in cases:
            if set(case.required_precision) != {"full", "mixed"}:
                errors.append(f"{case.id}: analytical derivatives must cover full and mixed")
        # Expansion is not an authority to silently drop or rewrite individual
        # requirements: compare with the current declarative builders, not a
        # frozen serialized list. Adding a genuine declaration needs no repin.
        builders = (
            requirements._matrix_requirements,
            requirements._leaf_requirements,
            requirements._regime_requirements,
            requirements._disposition_requirements,
            requirements._gfmc_matrix_requirements,
            requirements._gfmc_contract_requirements,
        )
        declared = {case.id: case for builder in builders for case in builder()}
        observed = {case.id: case for case in cases}
        if declared != observed:
            errors.append("live derivative requirements differ from their declared case/leaf/proof/liveness expansion")

        from gdd.engine.guarantee_store import load_module

        contract = load_module(CONTRACT_PATH)
        domain = contract["domain"]
        required_aos = set(domain["continuous_matrix_required_ao_families"])
        realized_aos = {cell[0] for cell, _heavy in matrix._generate_matrix()}
        if not required_aos or not required_aos <= realized_aos:
            errors.append(f"derivative matrix misses declared AO families: {sorted(required_aos - realized_aos)}")
        for family in domain["derivative_families"].values():
            if family["family"] in {"ln_Psi", "e_L"}:
                if missing := set(family["axes"]) - set(requirements._PHYSICAL_AXES):
                    errors.append(f"{family['family']}: derivative expansion misses declared axes: {sorted(missing)}")

        # These are independent scientific minimum tuples, not the old hash of
        # the source-derived matrix. Extra valid source cases remain permitted.
        specs = {spec.function_name: spec for spec in requirements._GFMC_FUNCTION_SPECS}
        for group in domain["gfmc_required_parameter_sets"]:
            function = group["pytest_node"].split("::")[-1]
            spec = specs.get(function)
            if spec is None or (
                spec.family != group["family"]
                or spec.proof.value != group["proof"]
                or not set(group["targets"]) <= set(spec.targets)
            ):
                errors.append(f"{function}: GFMC declared family/proof/targets are missing")
                continue
            rows = requirements._gfmc_parameter_rows(function)
            for required in group["parameters"]:
                if required not in rows:
                    errors.append(f"{function}: GFMC required scientific input tuple is missing: {required}")
    except (ImportError, OSError, SyntaxError, ValueError, TypeError, KeyError, AttributeError, NameError) as exc:
        errors.append(f"cannot expand live derivative requirements: {exc}")


def _check_reviewed_matrix_coverage(errors: list[str]) -> None:
    """Check the declared pairwise and selected interactions, not a full Cartesian claim."""
    from gdd.engine import derivative_matrix as matrix

    candidates = matrix._candidates()
    selected = [tuple(cell) for cell, _heavy in matrix._generate_matrix()]
    if len(set(selected)) != len(selected) or not set(selected) <= set(candidates):
        errors.append("derivative matrix contains duplicate or inadmissible cells")
        return
    for axes in combinations(range(5), 2):
        required = {tuple(cell[index] for index in axes) for cell in candidates}
        realized = {tuple(cell[index] for index in axes) for cell in selected}
        missing = required - realized
        if missing:
            errors.append(f"derivative matrix misses reviewed pairwise interaction {axes}: {sorted(missing)}")
    twist_required = set(product(matrix._JASTROW_CHANNEL_TOKENS, matrix._ECPS))
    twist_realized = {(cell[1], cell[2]) for cell in selected if cell[3] == "twist"}
    if missing := twist_required - twist_realized:
        errors.append(f"derivative matrix misses reviewed twist/Jastrow/potential interaction: {sorted(missing)}")
    periodic_required = set(product(("cart_pgto", "sphe_pgto"), ("J1", "J2", "J3poly", "DTN")))
    periodic_realized = {(cell[0], cell[1]) for cell in selected if cell[2:4] == ("ECP", "pbc_gamma")}
    if missing := periodic_required - periodic_realized:
        errors.append(f"derivative matrix misses reviewed periodic ECP/pGTO/Jastrow interaction: {sorted(missing)}")


def _check_required_matrix_sources(errors: list[str]) -> None:
    """Missing fixture files fail explicitly without changing the obligation universe."""
    from gdd.engine.derivative_matrix import _cell_id, _generate_matrix, _resolve, _stem_available

    for cell, _heavy in _generate_matrix():
        declared = _resolve(cell, require_sources=False)
        if declared is None:
            errors.append(f"required derivative cell has no declared builder: {_cell_id(cell)}")
        elif not _stem_available(declared[0]):
            errors.append(f"required derivative fixture source is missing: {declared[0]} for {_cell_id(cell)}")


def _check_rust_empty_allowlists(errors: list[str]) -> None:
    """Enforce the reviewed empty parity-gap inventories without importing Rust/JAX."""
    names = {"_DIVERGENCE_ALLOWLIST", "_XLA_ORACLE_GAP_ALLOWLIST"}
    for filename in ("test_rs_analytic_matrix.py", "test_rs_analytic_matrix_rRe.py"):
        path = ROOT / "tests/test_rs_kernel" / filename
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            errors.append(f"cannot inspect Rust parity allowlists: {exc}")
            continue
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for name in sorted(names):
            definitions = []
            for node in tree.body:
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
                )
                if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                    definitions.append(node)
            if len(definitions) != 1 or not isinstance(definitions[0].value, ast.Dict) or definitions[0].value.keys:
                errors.append(f"{filename}: {name} must have exactly one empty literal dictionary definition")
                continue
            definition = definitions[0]
            # This deliberately supports the existing read-only idioms, not
            # arbitrary Python data flow. An alias or unknown use must not hide
            # a later mutation behind the apparently empty initializer.
            for node in ast.walk(tree):
                if not isinstance(node, ast.Name) or node.id != name:
                    continue
                parent = parents[node]
                if parent is definition and node is getattr(definition, "target", None):
                    continue
                if parent is definition and isinstance(definition, ast.Assign) and definition.targets == [node]:
                    continue
                read_only = isinstance(node.ctx, ast.Load) and (
                    (isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not))
                    or (isinstance(parent, ast.comprehension) and parent.iter is node)
                    or (
                        isinstance(parent, ast.Attribute)
                        # A copy is a new mutable allowlist: accepting it would
                        # let later writes evade this initializer/use check.
                        and parent.attr in {"get", "keys", "items", "values"}
                        and isinstance(parents.get(parent), ast.Call)
                        and parents[parent].func is parent
                    )
                )
                if not read_only:
                    errors.append(f"{filename}:{node.lineno}: {name} has an unreviewed write/alias/use")


def check() -> list[str]:
    errors: list[str] = inactivity_contract_errors(ROOT)
    try:
        contract = load_ledger(CONTRACT_PATH)
    except (OSError, yaml.YAMLError) as exc:
        return [f"cannot load {CONTRACT_PATH.relative_to(ROOT)}: {exc}"]
    if not isinstance(contract, dict):
        return ["derivative contract must be a YAML mapping"]

    required_top = {
        "schema_version",
        "document",
        "enforcement",
        "operation_ids",
        "baseline",
        "domain",
        "common_contract",
        "existing_test_integration",
        "assurance_chain",
        "fixture_sets",
        "guarantees",
    }
    missing_top = sorted(required_top - set(contract))
    if missing_top:
        errors.append(f"missing top-level fields: {missing_top}")
        return errors
    if contract["schema_version"] != 1:
        errors.append("schema_version must be 1")
    if contract["document"].get("lifecycle") != "cutover":
        errors.append("derivative contract lifecycle must use the human-readable cutover state")
    runtime = contract["enforcement"]["runtime"]
    for key in ("collection_closure", "in_test_self_check"):
        if not (ROOT / str(runtime.get(key, ""))).is_file():
            errors.append(f"derivative runtime enforcement is missing its {key}")

    _check_existing_test_integration(contract, errors)

    guarantees = contract["guarantees"]
    fixtures = contract["fixture_sets"]
    guarantee_ids = [item.get("id") for item in guarantees]
    fixture_ids = [item.get("id") for item in fixtures]
    if None in guarantee_ids or len(guarantee_ids) != len(set(guarantee_ids)):
        errors.append("guarantee IDs must be present and unique")
    expected_guarantee_ids = {
        "NUM-DERIV-FD-AUTODIFF-001",
        "MATH-DERIV-LOG-WAVEFUNCTION-001",
        "MATH-DERIV-LOCAL-ENERGY-001",
        "MATH-DERIV-CONVENTIONAL-ALGORITHM-001",
        "MATH-DERIV-LOAD-BALANCED-ALGORITHM-001",
        "NUM-DERIV-MIXED-001",
        "IMPL-DERIV-RUST-XLA-001",
        "OPS-DERIV-CLOSURE-001",
    }
    if set(guarantee_ids) != expected_guarantee_ids:
        errors.append("derivative guarantee IDs differ from the descriptive reviewed vocabulary")
    verdict_path = ROOT / "gdd/ssot/verdicts.yaml"
    try:
        verdict_registry = yaml.safe_load(verdict_path.read_text(encoding="utf-8"))["verdict_methods"]
        comparisons_doc = yaml.safe_load(verdict_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, KeyError, TypeError) as exc:
        errors.append(f"cannot load the verdict-method registry: {exc}")
        verdict_registry = []
    else:
        known_comparison_policies = {row["id"] for row in comparisons_doc["comparison_policies"]}
        known_observation_policies = {row["id"] for row in comparisons_doc["observation_policies"]}
        method_ids = [row.get("id") for row in verdict_registry]
        if None in method_ids or len(method_ids) != len(set(method_ids)):
            errors.append("verdict-method IDs must be present and unique")
        for row in verdict_registry:
            unknown_cmp = sorted(set(row.get("comparison_policies", [])) - known_comparison_policies)
            unknown_obs = sorted(set(row.get("observation_policies", [])) - known_observation_policies)
            if unknown_cmp or unknown_obs:
                errors.append(
                    f"{row.get('id')}: verdict method references unknown pass-criteria policies: "
                    f"comparison={unknown_cmp}, observation={unknown_obs}"
                )
            if (
                not row.get("comparison_policies")
                and not row.get("observation_policies")
                and row.get("id") != "VM-OPS-SET-CLOSURE-001"
            ):
                errors.append(f"{row.get('id')}: verdict method declares no pass criterion")
    methods_by_id = {row.get("id"): row for row in verdict_registry}

    taxonomy_path = ROOT / "gdd/ssot/verdicts.yaml"
    try:
        taxonomy = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))
        prefix_by_kind = {row["kind"]: row["prefix"] for row in taxonomy["classes"]}
    except (OSError, yaml.YAMLError, KeyError, TypeError) as exc:
        errors.append(f"cannot load the guarantee class taxonomy: {exc}")
        prefix_by_kind = {}
    if prefix_by_kind and set(prefix_by_kind) != {"physical", "mathematical", "numerical", "implementation", "operational"}:
        errors.append("guarantee class taxonomy must define exactly the five reviewed classes")
    for guarantee in guarantees:
        kind = guarantee.get("kind")
        if prefix_by_kind:
            if kind not in prefix_by_kind:
                errors.append(f"{guarantee.get('id')}: kind {kind!r} is not a guarantee class from the taxonomy")
            elif not str(guarantee.get("id", "")).startswith(prefix_by_kind[kind] + "-"):
                errors.append(
                    f"{guarantee.get('id')}: ID prefix does not match its class ({kind} requires {prefix_by_kind[kind]}-)"
                )
    for guarantee in guarantees:
        declared_methods = guarantee.get("verdict_methods", [])
        if not declared_methods or len(declared_methods) != len(set(declared_methods)):
            errors.append(f"{guarantee.get('id')}: every guarantee must assign unique verdict methods")
        for method_id in declared_methods:
            method = methods_by_id.get(method_id)
            if method is None:
                errors.append(f"{guarantee.get('id')}: unknown verdict method {method_id}")
            elif guarantee.get("kind") not in method.get("classes", []):
                errors.append(
                    f"{guarantee.get('id')}: verdict method {method_id} is not legal for class "
                    f"{guarantee.get('kind')!r} (allowed: {method.get('classes')})"
                )
    for guarantee in guarantees:
        if not guarantee.get("name") or not guarantee.get("plain_language"):
            errors.append(f"{guarantee.get('id')}: human-readable name and plain_language are required")
        if "certification" in guarantee:
            errors.append(f"{guarantee.get('id')}: per-run certification must not be used as contract lifecycle")
        allowed_families = {"ln_Psi", "e_L", "e_L_t", "e_L_n"}
        for field, value in (
            ("family", guarantee.get("family")),
            ("family_selector", guarantee.get("scope", {}).get("family_selector")),
        ):
            if value is not None and value not in allowed_families:
                errors.append(f"{guarantee.get('id')}: {field} must use a descriptive production token, not {value!r}")
    if None in fixture_ids or len(fixture_ids) != len(set(fixture_ids)):
        errors.append("fixture-set IDs must be present and unique")

    _check_production_entry_discovery(contract, guarantees, errors)
    _check_engine_vocabularies(contract, errors)

    expected_chain = [
        (1, "XLA-finite-difference-equals-XLA-autodiff", "DERIV-FIX-FD-AUTODIFF-001"),
        (2, "XLA-autodiff-equals-XLA-analytical", "DERIV-FIX-AUTODIFF-ANALYTICAL-001"),
        (3, "XLA-analytical-equals-Rust-analytical", "DERIV-FIX-XLA-RUST-001"),
    ]
    actual_chain = [(item.get("order"), item.get("relation"), item.get("fixture_set")) for item in contract["assurance_chain"]]
    if actual_chain != expected_chain:
        errors.append(f"assurance_chain differs from the required three ordered relations: {actual_chain}")
    for item in contract["assurance_chain"]:
        references = [item.get("guarantee")] if "guarantee" in item else item.get("guarantees", [])
        unknown = sorted(set(references) - set(guarantee_ids))
        if unknown:
            errors.append(f"assurance-chain guarantee references are unknown: {unknown}")
        if item.get("fixture_set") not in fixture_ids:
            errors.append(f"assurance-chain fixture reference is unknown: {item.get('fixture_set')}")
    for guarantee in guarantees:
        fixture_id = guarantee.get("scope", {}).get("fixture_set")
        if fixture_id is not None and fixture_id not in fixture_ids:
            errors.append(f"{guarantee.get('id')}: scope fixture reference is unknown: {fixture_id}")
        if guarantee.get("id") == "OPS-DERIV-CLOSURE-001":
            required = guarantee.get("scope", {}).get("required_guarantees", [])
            unknown = sorted(set(required) - set(guarantee_ids))
            if unknown:
                errors.append(f"OPS-DERIV-CLOSURE-001 references unknown guarantees: {unknown}")

    # A document header has no decisions field.
    if "decisions" in contract["document"]:
        errors.append("document header must not contain a decisions field (document.decisions)")

    baseline = contract["baseline"]
    eps_semantics = (
        contract.get("domain", {}).get("axis_semantics", {}).get("stored_token_eps", {}).get("dispatch_by_boundary", {})
    )
    expected_eps_semantics = {
        "molecular": ("molecular-affine", False),
        "periodic": ("cell-strain", True),
    }
    if set(eps_semantics) != set(expected_eps_semantics):
        errors.append("stored eps must dispatch to exactly molecular and periodic semantic axes")
    for boundary, (semantic_axis, physical_stress) in expected_eps_semantics.items():
        declaration = eps_semantics.get(boundary, {})
        if (
            declaration.get("semantic_axis") != semantic_axis
            or declaration.get("physical_stress_observable") is not physical_stress
            or declaration.get("rust_xla_parity_required") is not True
        ):
            errors.append(
                f"stored eps {boundary} semantics must be {semantic_axis}, "
                f"physical_stress_observable={physical_stress}, and Rust/XLA parity required"
            )
    path_fields = [
        (baseline["required_manifest"]["path"], "required manifest"),
        (baseline["registered_evidence"]["path"], "registered evidence"),
        (baseline["static_completion_gate"]["path"], "XLA completion gate"),
        (baseline["rust_manifest"]["path"], "Rust manifest"),
        (baseline["rust_completion_gate"]["path"], "Rust completion gate"),
        (contract["enforcement"]["runtime"]["pytest_node"], "runtime GDD gate"),
    ]
    for value, label in path_fields:
        _require_path(value, label, errors)

    from gdd.engine.derivative_requirements import REQUIRED_CASES

    required_families = {case.family for case in REQUIRED_CASES}
    declared_families = {item["family"] for item in guarantees if item.get("family")}
    if required_families != declared_families:
        errors.append("every live derivative family must have exactly its declared guarantee family")
    if baseline["rust_manifest"]["explicit_not_applicable"] != 0:
        errors.append("Rust must require the complete XLA universe with zero N/A subtraction")

    _check_live_requirement_snapshot(baseline, errors)
    _check_reviewed_matrix_coverage(errors)
    _check_required_matrix_sources(errors)
    _check_rust_empty_allowlists(errors)

    fd_fixture = next((item for item in fixtures if item.get("id") == "DERIV-FIX-FD-AUTODIFF-001"), None)
    if fd_fixture is None:
        errors.append("missing DERIV-FIX-FD-AUTODIFF-001")
    else:
        reviewed = fd_fixture.get("reviewed_evidence", {})
        nodes = reviewed.get("nodes", [])
        ids = [node.get("id") for node in nodes]
        node_ids = [node.get("pytest_node") for node in nodes]
        if None in ids or len(ids) != len(set(ids)):
            errors.append("FD evidence IDs must be present and unique")
        if None in node_ids or len(node_ids) != len(set(node_ids)):
            errors.append("FD pytest nodes must be present and unique")
        for node in nodes:
            _check_fd_node(node, errors)
        _check_fd_receipt_id_universe(nodes, errors)
        errors.extend(fd_source_contract_errors(root=ROOT))

    return errors


def main() -> int:
    errors = check()
    if errors:
        print("Derivative GDD contract check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Derivative GDD contract check passed (live required matrices, zero Rust N/A, authored FD selectors).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

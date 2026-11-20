"""Pytest-side registration adapter for the derivative GDD contract.

The reviewed requirement universe (``REQUIRED_CASES``) lives in
``gdd.engine.derivative_requirements``; this adapter discovers the actual
pytest registrations (``NUMERICAL_CASES``) from the evidence modules' real
``@pytest.mark.parametrize`` marks and compares the two sides.  CI still
requires the requirement and registration ID sets to match exactly; a removed
pytest axis/quantity/cell becomes a registration gap, and a matrix edit still
requires an explicit snapshot review upstream in the engine.
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
from collections import Counter
from itertools import product
from pathlib import Path

import yaml

from gdd.engine.derivative_requirements import (  # noqa: F401  (re-exported contract API)
    _stable_parameter_value,
    APPROVED_GFMC_MATRIX_SHA256,
    APPROVED_MATRIX_SHA256,
    REQUIRED_CASES,
    ExpectedOutcome,
    NumericalCase,
    ProofKind,
    RegistrationDiff,
    Requirement,
    SentinelKind,
    _FAMILY_QUANTITY,
    _GFMC_FUNCTION_SPECS,
    _QUANTITY_FAMILY,
    _VALIDITY_REQUIREMENT_PARTS,
    _gfmc_case_id,
    _gfmc_parameter_rows,
    _matrix_requirement_id,
    gfmc_broad_matrix_covers_both_precisions,
    gfmc_function_inventory_diff,
    gfmc_matrix_hash,
    gfmc_matrix_payload,
    gfmc_matrix_snapshot_matches,
    matrix_snapshot_matches,
    requirements_by_id,
)
from tests.test_jqmc_kernel import _generated_matrix_base as _matrix
from tests.test_jqmc_kernel import test_analytic_coverage_evidence as _evidence
from tests.test_jqmc_kernel import test_analytic_grad_matches_autograd_generated_molecular as _matrix_molecular
from tests.test_jqmc_kernel import test_analytic_grad_matches_autograd_generated_periodic as _matrix_periodic
from tests.test_jqmc_kernel import test_analytic_grad_matches_autograd_generated_twist as _matrix_twist
from tests.test_jqmc_kernel import test_gfmc_analytic_coverage_evidence as _gfmc_targeted
from tests.test_jqmc_kernel import test_mixed_precision as _mixed

_MATRIX_TEST_MODULES = (
    ("test_analytic_grad_matches_autograd_generated_molecular.py", _matrix_molecular),
    ("test_analytic_grad_matches_autograd_generated_periodic.py", _matrix_periodic),
    ("test_analytic_grad_matches_autograd_generated_twist.py", _matrix_twist),
)


_EVIDENCE_FILE = "tests/test_jqmc_kernel/test_analytic_coverage_evidence.py"
_GFMC_EVIDENCE_FILE = "tests/test_jqmc_kernel/test_gfmc_analytic_coverage_evidence.py"


def _parameters(**values) -> tuple[tuple[str, str], ...]:
    """Callspec values of one registered case, in a stable display form."""
    return tuple((name, str(value)) for name, value in values.items())


def _has_skip_mixed_precision_marker(module) -> bool:
    return any(mark.name == "skip_mixed_precision" for mark in getattr(module, "pytestmark", ()))


def matrix_modules_cover_both_precisions() -> bool:
    """Whether every ln_Psi/e_L analytical-vs-autograd matrix runs in both precision lanes."""
    return all(not _has_skip_mixed_precision_marker(module) for _filename, module in _MATRIX_TEST_MODULES)


def _pytest_param_values(function, parameter_name: str) -> tuple:
    """Read one parameter axis from the function's actual pytest marks."""
    matching = [
        mark for mark in function.pytestmark if mark.name == "parametrize" and mark.args and mark.args[0] == parameter_name
    ]
    if len(matching) != 1:
        raise ValueError(f"expected one {parameter_name!r} parametrization, found {len(matching)}")
    values = []
    for value in matching[0].args[1]:
        parameter_values = getattr(value, "values", None)
        values.append(parameter_values[0] if parameter_values is not None else value)
    return tuple(values)


def _discovered_numerical_cases() -> tuple[NumericalCase, ...]:
    cases = []
    for filename, module in _MATRIX_TEST_MODULES:
        function = module.test_matrix_cell
        cells = _pytest_param_values(function, "cell")
        quantities = _pytest_param_values(function, "quantity")
        axes = _pytest_param_values(function, "axis")
        for cell, quantity, axis in product(cells, quantities, axes):
            cell = tuple(cell)
            family = _QUANTITY_FAMILY[quantity]
            case_id = _matrix_requirement_id(family, cell, axis)
            binding = _matrix.matrix_case_binding(cell, quantity, axis)
            if binding.obligation_id != case_id:
                raise ValueError(f"matrix GDD binding identity mismatch for {case_id}")
            cases.append(
                NumericalCase(
                    requirement_id=case_id,
                    pytest_definition=f"tests/test_jqmc_kernel/{filename}::test_matrix_cell",
                    pytest_parameters=_parameters(cell=_matrix._cell_id(cell), quantity=quantity, axis=axis),
                    builder_available=_matrix._buildable(cell),
                    registered_precision=("full", "mixed"),
                    proof=ProofKind.DIRECT_AD,
                    input_binding_id=binding.input_binding.binding_id,
                    electron_configuration_binding_id=binding.electron_configuration.binding_id,
                    comparison_binding_id=binding.comparison.binding_id,
                    runtime_receipt_operation="gdd.engine.evidence.CaseReceipt.validate",
                )
            )
    return tuple(cases)


_LEAF_SOURCES_BY_SCENARIO = _evidence._LEAF_SOURCE_IDS_BY_SCENARIO


def _discovered_targeted_cases() -> tuple[NumericalCase, ...]:
    """Discover requirement evidence from the actual targeted pytest marks."""
    cases = []

    leaf_function = _evidence.test_parameter_leaf_evidence
    leaf_families = _pytest_param_values(leaf_function, "family")
    leaf_scenarios = _pytest_param_values(leaf_function, "scenario")
    for scenario, family in product(leaf_scenarios, leaf_families):
        for source_id in _LEAF_SOURCES_BY_SCENARIO[scenario]:
            cases.append(
                NumericalCase(
                    requirement_id=_evidence.leaf_obligation_id(family, source_id),
                    pytest_definition=f"{_EVIDENCE_FILE}::test_parameter_leaf_evidence",
                    pytest_parameters=_parameters(scenario=scenario, family=family),
                    builder_available=True,
                    registered_precision=("full", "mixed"),
                    proof=ProofKind.DIRECT_AD,
                    liveness_checked=True,
                    input_binding_id=f"INPUT::{family}|leaf|{source_id}",
                    electron_configuration_binding_id=f"ELECTRONS::{family}|leaf|{source_id}",
                    comparison_binding_id=f"COMPARISON::{family}|leaf|{source_id}",
                    runtime_receipt_operation="gdd.engine.evidence.CaseReceipt.validate",
                )
            )

    validity_function = _evidence.test_validity_disposition
    validity_families = _pytest_param_values(validity_function, "family")
    dispositions = _pytest_param_values(validity_function, "disposition")
    for disposition, family in product(dispositions, validity_families):
        outcome, source_id = _VALIDITY_REQUIREMENT_PARTS[disposition]
        cases.append(
            NumericalCase(
                requirement_id=f"{family}|{outcome}|{source_id}|{disposition}",
                pytest_definition=f"{_EVIDENCE_FILE}::test_validity_disposition",
                pytest_parameters=_parameters(disposition=disposition, family=family),
                builder_available=True,
                registered_precision=("full", "mixed"),
                proof=None,
                input_binding_id=f"INPUT::{family}|{outcome}|{source_id}|{disposition}",
                electron_configuration_binding_id=f"ELECTRONS::{family}|{outcome}|{source_id}|{disposition}",
                comparison_binding_id=(
                    f"N-A::{family}::{disposition}" if outcome == "n-a" else f"ERROR::{family}::{disposition}"
                ),
                runtime_receipt_operation="gdd.engine.evidence.CaseReceipt.validate",
            )
        )

    if callable(getattr(_evidence, "test_v1_potential_independence_identity", None)):
        cases.append(
            NumericalCase(
                requirement_id=_evidence.IDENTITY_OBLIGATION_ID,
                pytest_definition=f"{_EVIDENCE_FILE}::test_v1_potential_independence_identity",
                pytest_parameters=(),
                builder_available=True,
                registered_precision=("full", "mixed"),
                proof=ProofKind.IDENTITY,
                input_binding_id="INPUT::ln_Psi|identity|B20|potential-independent",
                electron_configuration_binding_id="ELECTRONS::ln_Psi|identity|B20|potential-independent",
                comparison_binding_id="EXACT::ln_Psi|identity|B20|potential-independent::{precision-mode}",
                runtime_receipt_operation="gdd.engine.evidence.CaseReceipt.validate",
            )
        )

    regime_function = _evidence.test_numerical_regime_evidence
    regime_requirements = _pytest_param_values(regime_function, "requirement_id")
    for requirement_id in regime_requirements:
        is_composite = requirement_id.endswith("|R03")
        cases.append(
            NumericalCase(
                requirement_id=requirement_id,
                pytest_definition=f"{_EVIDENCE_FILE}::test_numerical_regime_evidence",
                pytest_parameters=_parameters(requirement_id=requirement_id),
                builder_available=True,
                registered_precision=("full", "mixed"),
                proof=ProofKind.DIRECT_AD,
                input_binding_id=(f"COMPOSITE-INPUT::{requirement_id}" if is_composite else f"INPUT::{requirement_id}"),
                electron_configuration_binding_id=(
                    f"COMPOSITE-ELECTRONS::{requirement_id}" if is_composite else f"ELECTRONS::{requirement_id}"
                ),
                comparison_binding_id=(f"COMPOSITE::{requirement_id}" if is_composite else f"COMPARISON::{requirement_id}"),
                runtime_receipt_operation=(
                    "gdd.engine.evidence.CompositeCaseReceipt.validate"
                    if is_composite
                    else "gdd.engine.evidence.CaseReceipt.validate"
                ),
            )
        )
    return tuple(cases)


def _discovered_gfmc_matrix_cases() -> tuple[NumericalCase, ...]:
    """Discover each expanded GFMC direct/identity pytest registration."""
    cases = []
    for spec in _GFMC_FUNCTION_SPECS:
        for parameters, target in product(_gfmc_parameter_rows(spec.function_name), spec.targets):
            case_id = _gfmc_case_id(spec, parameters, target)
            cases.append(
                NumericalCase(
                    requirement_id=case_id,
                    pytest_definition=f"tests/test_jqmc_kernel/test_gfmc_force_analytic.py::{spec.function_name}",
                    pytest_parameters=tuple((key, repr(value)) for key, value in sorted(parameters.items())),
                    builder_available=True,
                    registered_precision=("full", "mixed"),
                    proof=spec.proof,
                    input_binding_id=f"INPUT::{case_id}",
                    electron_configuration_binding_id=f"ELECTRONS::{case_id}",
                    comparison_binding_id=f"COMPARISON::{case_id}",
                    runtime_receipt_operation="gdd.engine.evidence.CaseReceipt.validate",
                )
            )
    return tuple(cases)


def _discovered_gfmc_targeted_cases() -> tuple[NumericalCase, ...]:
    cases = []
    families = _pytest_param_values(_gfmc_targeted.test_gfmc_output_liveness, "family")
    for family in families:
        cases.append(
            NumericalCase(
                requirement_id=_gfmc_targeted.output_obligation_id(family),
                pytest_definition=f"{_GFMC_EVIDENCE_FILE}::test_gfmc_output_liveness",
                pytest_parameters=_parameters(family=family),
                builder_available=True,
                registered_precision=("full", "mixed"),
                proof=ProofKind.DIRECT_AD,
                liveness_checked=True,
                input_binding_id=f"INPUT::{family}|output|B24",
                electron_configuration_binding_id=f"ELECTRONS::{family}|output|B24",
                comparison_binding_id=f"LIVENESS::{family}|output|B24",
                runtime_receipt_operation="gdd.engine.evidence.CaseReceipt.validate",
            )
        )

    families = _pytest_param_values(_gfmc_targeted.test_gfmc_invalid_non_local_move, "family")
    for family in families:
        cases.append(
            NumericalCase(
                requirement_id=_gfmc_targeted.invalid_move_obligation_id(family),
                pytest_definition=f"{_GFMC_EVIDENCE_FILE}::test_gfmc_invalid_non_local_move",
                pytest_parameters=_parameters(family=family),
                builder_available=True,
                registered_precision=("full", "mixed"),
                proof=None,
                input_binding_id=f"INPUT::{family}|error|C10|unknown-non-local-move",
                electron_configuration_binding_id=f"ELECTRONS::{family}|error|C10|unknown-non-local-move",
                comparison_binding_id=f"ERROR::{family}|error|C10|unknown-non-local-move",
                runtime_receipt_operation="gdd.engine.evidence.CaseReceipt.validate",
            )
        )

    requirement_ids = _pytest_param_values(_gfmc_targeted.test_gfmc_numerical_regime_evidence, "requirement_id")
    for requirement_id in requirement_ids:
        cases.append(
            NumericalCase(
                requirement_id=requirement_id,
                pytest_definition=f"{_GFMC_EVIDENCE_FILE}::test_gfmc_numerical_regime_evidence",
                pytest_parameters=_parameters(requirement_id=requirement_id),
                builder_available=True,
                registered_precision=("full", "mixed"),
                proof=ProofKind.DIRECT_AD,
                input_binding_id=f"INPUT::{requirement_id}",
                electron_configuration_binding_id=f"ELECTRONS::{requirement_id}",
                comparison_binding_id=f"COMPARISON::{requirement_id}",
                runtime_receipt_operation="gdd.engine.evidence.CaseReceipt.validate",
            )
        )
    return tuple(cases)


NUMERICAL_CASES = (
    _discovered_numerical_cases()
    + _discovered_targeted_cases()
    + _discovered_gfmc_matrix_cases()
    + _discovered_gfmc_targeted_cases()
)


def generated_matrix_binding_issues() -> tuple[str, ...]:
    """Return generated cases missing any binding needed for a runtime receipt."""
    issues = []
    matrix_cases = [case for case in NUMERICAL_CASES if "|matrix|" in case.requirement_id]
    required_matrix_ids = {case.id for case in REQUIRED_CASES if case.category == "matrix"}
    registered_matrix_ids = {case.requirement_id for case in matrix_cases}
    for missing in sorted(required_matrix_ids - registered_matrix_ids):
        issues.append(f"missing-matrix-case:{missing}")
    for extra in sorted(registered_matrix_ids - required_matrix_ids):
        issues.append(f"unknown-matrix-case:{extra}")
    for case in matrix_cases:
        if not case.input_binding_id:
            issues.append(f"missing-input-binding:{case.requirement_id}")
        if not case.electron_configuration_binding_id:
            issues.append(f"missing-electron-binding:{case.requirement_id}")
        if not case.comparison_binding_id:
            issues.append(f"missing-comparison-binding:{case.requirement_id}")
        if not case.runtime_receipt_operation:
            issues.append(f"missing-runtime-receipt:{case.requirement_id}")
    return tuple(sorted(issues))


def registration_diff() -> RegistrationDiff:
    """Compare the reviewed requirement IDs with actual pytest registrations."""
    required_ids = {requirement.id for requirement in REQUIRED_CASES}
    counts = Counter(case.requirement_id for case in NUMERICAL_CASES)
    registered_ids = set(counts)
    unavailable = tuple(sorted(case.requirement_id for case in NUMERICAL_CASES if not case.builder_available))
    requirements = requirements_by_id()
    unique_cases = {case.requirement_id: case for case in NUMERICAL_CASES if counts[case.requirement_id] == 1}
    comparable_ids = required_ids & set(unique_cases)
    return RegistrationDiff(
        missing=tuple(sorted(required_ids - registered_ids)),
        extra=tuple(sorted(registered_ids - required_ids)),
        duplicates=tuple(sorted(case_id for case_id, count in counts.items() if count != 1)),
        unavailable=unavailable,
        precision_mismatch=tuple(
            sorted(
                requirement_id
                for requirement_id in comparable_ids
                if not set(requirements[requirement_id].required_precision)
                <= set(unique_cases[requirement_id].registered_precision)
            )
        ),
        proof_mismatch=tuple(
            sorted(
                requirement_id
                for requirement_id in comparable_ids
                if requirements[requirement_id].proof != unique_cases[requirement_id].proof
            )
        ),
        missing_liveness=tuple(
            sorted(
                requirement_id
                for requirement_id in comparable_ids
                if requirements[requirement_id].liveness_required and not unique_cases[requirement_id].liveness_checked
            )
        ),
    )


def family_completion_issues(family: str) -> tuple[str, ...]:
    """Return mechanical manifest defects that keep a family from ``finished``."""
    if family not in ("ln_Psi", "e_L", "e_L_t", "e_L_n"):
        raise ValueError(f"unsupported completion family: {family}")
    issues = []
    if family in _FAMILY_QUANTITY and not matrix_snapshot_matches():
        issues.append("reviewed matrix snapshot mismatch")
    if family in _FAMILY_QUANTITY and not matrix_modules_cover_both_precisions():
        issues.append("ln_Psi/e_L autograd matrix does not run in both precision lanes")
    if family in ("e_L_t", "e_L_n") and not gfmc_matrix_snapshot_matches():
        issues.append("reviewed GFMC matrix snapshot mismatch")
    if family in ("e_L_t", "e_L_n"):
        missing_functions, extra_functions = gfmc_function_inventory_diff()
        if missing_functions:
            issues.append(f"missing GFMC evidence functions: {', '.join(missing_functions)}")
        if extra_functions:
            issues.append(f"unreviewed GFMC evidence functions: {', '.join(extra_functions)}")
        if not gfmc_broad_matrix_covers_both_precisions():
            issues.append("GFMC autograd matrix does not run in both precision lanes")
    diff = registration_diff()
    for label, requirement_ids in (
        ("missing registration", diff.missing),
        ("extra registration", diff.extra),
        ("duplicate registration", diff.duplicates),
        ("unavailable builder", diff.unavailable),
        ("precision-lane mismatch", diff.precision_mismatch),
        ("proof mismatch", diff.proof_mismatch),
        ("missing liveness contract", diff.missing_liveness),
    ):
        family_ids = tuple(requirement_id for requirement_id in requirement_ids if requirement_id.startswith(f"{family}|"))
        if family_ids:
            issues.append(f"{label}: {', '.join(family_ids)}")
    return tuple(issues)


def _subset_registration_issues(cases, required_ids):
    """Compare actual identities with live requirements rather than a stored count."""
    counts = Counter(case.requirement_id for case in cases)
    issues = [f"missing-case:{identifier}" for identifier in sorted(required_ids - counts.keys())]
    issues.extend(f"unexpected-case:{identifier}" for identifier in sorted(counts.keys() - required_ids))
    issues.extend(f"duplicate-case:{identifier}" for identifier, count in sorted(counts.items()) if count != 1)
    return issues


def targeted_binding_issues() -> tuple[str, ...]:
    """Return missing bindings for the required leaf/validity/output/identity obligations."""
    targeted_categories = {"leaf", "validity", "output", "identity"}
    lookup = {requirement.id: requirement for requirement in REQUIRED_CASES}
    cases = [case for case in NUMERICAL_CASES if lookup[case.requirement_id].category in targeted_categories]
    required_ids = {requirement.id for requirement in REQUIRED_CASES if requirement.category in targeted_categories}
    issues = _subset_registration_issues(cases, required_ids)
    for case in cases:
        missing = [
            name
            for name, value in (
                ("input", case.input_binding_id),
                ("electron", case.electron_configuration_binding_id),
                ("check", case.comparison_binding_id),
                ("receipt", case.runtime_receipt_operation),
            )
            if not value
        ]
        if missing:
            issues.append(f"{case.requirement_id}:{','.join(missing)}")
    return tuple(issues)


def gfmc_matrix_binding_issues() -> tuple[str, ...]:
    """Return missing bindings for the required GFMC direct/identity obligations."""
    lookup = {requirement.id: requirement for requirement in REQUIRED_CASES}
    cases = [case for case in NUMERICAL_CASES if lookup[case.requirement_id].category in {"gfmc-direct", "gfmc-identity"}]
    required_ids = {
        requirement.id for requirement in REQUIRED_CASES if requirement.category in {"gfmc-direct", "gfmc-identity"}
    }
    issues = _subset_registration_issues(cases, required_ids)
    for case in cases:
        if not all(
            (
                case.input_binding_id,
                case.electron_configuration_binding_id,
                case.comparison_binding_id,
                case.runtime_receipt_operation,
            )
        ):
            issues.append(f"{case.requirement_id}:incomplete-binding")
    return tuple(issues)


def regime_binding_issues() -> tuple[str, ...]:
    """Return missing bindings for the required continuous/GFMC regime obligations."""
    lookup = {requirement.id: requirement for requirement in REQUIRED_CASES}
    cases = [case for case in NUMERICAL_CASES if lookup[case.requirement_id].category == "regime"]
    required_ids = {requirement.id for requirement in REQUIRED_CASES if requirement.category == "regime"}
    issues = _subset_registration_issues(cases, required_ids)
    for case in cases:
        if not all(
            (
                case.input_binding_id,
                case.electron_configuration_binding_id,
                case.comparison_binding_id,
                case.runtime_receipt_operation,
            )
        ):
            issues.append(f"{case.requirement_id}:incomplete-binding")
    return tuple(issues)


def fd_binding_issues() -> tuple[str, ...]:
    """Return reviewed FD nodes missing a concrete runtime receipt call."""
    contract_path = Path(__file__).resolve().parents[2] / "gdd/ssot/guarantees/derivatives.yaml"
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    fixture = next(item for item in contract["fixture_sets"] if item["id"] == "DERIV-FIX-FD-AUTODIFF-001")
    reviewed = fixture["reviewed_evidence"]
    issues = []
    root = Path(__file__).resolve().parents[2]
    for node in reviewed["nodes"]:
        parts = node["pytest_node"].split("::")
        tree = ast.parse((root / parts[0]).read_text(encoding="utf-8"))
        scope: ast.AST = tree
        for name in parts[1:]:
            scope = next(
                candidate
                for candidate in getattr(scope, "body", ())
                if isinstance(candidate, (ast.ClassDef, ast.FunctionDef)) and candidate.name == name
            )
        calls = [
            call
            for call in ast.walk(scope)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "record_fd_case"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value == node["id"]
        ]
        if len(calls) != 1:
            issues.append(f"{node['id']}:runtime-receipt-call-count={len(calls)}")
            continue
        required_keywords = {
            "fixture_parameters",
            "coordinates",
            "compared_pairs",
            "atol",
            "rtol",
            "stencil",
            "step",
            "direction",
        }
        actual_keywords = {keyword.arg for keyword in calls[0].keywords}
        if missing := sorted(required_keywords - actual_keywords):
            issues.append(f"{node['id']}:missing-{','.join(missing)}")
    return tuple(issues)


def mixed_binding_issues() -> tuple[str, ...]:
    """Return defects in the six continuous and four GFMC mixed-route receipts."""
    path = Path(__file__).with_name("test_mixed_precision.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    issues = []
    expected = {
        "test_mixed_production_continuous_analytical_smoke": (6, "continuous", ("quantity", "route")),
        "test_mixed_production_gfmc_analytical_smoke": (4, "gfmc", ("family", "route")),
    }
    for function_name, (case_count, route_class, parameter_names) in expected.items():
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
        calls = [
            call
            for call in ast.walk(function)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "_record_mixed_route_receipt"
        ]
        if len(calls) != 1:
            issues.append(f"{function_name}:receipt-call-count={len(calls)}")
        elif not any(
            keyword.arg == "route_class" and isinstance(keyword.value, ast.Constant) and keyword.value.value == route_class
            for keyword in calls[0].keywords
        ):
            issues.append(f"{function_name}:wrong-route-class")
        runtime_function = getattr(_mixed, function_name)
        parameter_count = 1
        for parameter_name in parameter_names:
            parameter_count *= len(_pytest_param_values(runtime_function, parameter_name))
        if parameter_count != case_count:
            issues.append(f"{function_name}:case-count={parameter_count}")
    return tuple(issues)


# -- derivative registry key functions (issue 789) ----------------------------
#
# ``gdd.engine.derivative_registry`` maps every emitting XLA/MIXED definition to
# one of these. Each derives the receipts one collected case emits from its real
# callspec values, with the same id helpers the test body uses; they return
# ``(universe keys, upstream receipts)`` and build no node id.


def matrix_cell_keys(definition, params, precision):
    return [_matrix.matrix_obligation_id(params["cell"], params["quantity"], params["axis"])], []


def leaf_keys(definition, params, precision):
    family, scenario = params["family"], params["scenario"]
    return [_evidence.leaf_obligation_id(family, source_id) for source_id in _LEAF_SOURCES_BY_SCENARIO[scenario]], []


def validity_keys(definition, params, precision):
    return [_evidence.validity_obligation_id(params["family"], params["disposition"])], []


def identity_keys(definition, params, precision):
    return [_evidence.IDENTITY_OBLIGATION_ID], []


def regime_keys(definition, params, precision):
    # The regime tests are parametrized by the obligation id itself.
    return [params["requirement_id"]], []


def gfmc_output_keys(definition, params, precision):
    return [_gfmc_targeted.output_obligation_id(params["family"])], []


def gfmc_invalid_keys(definition, params, precision):
    return [_gfmc_targeted.invalid_move_obligation_id(params["family"])], []


def gfmc_force_keys(definition, params, precision):
    """The autouse fixture's receipts: one per output target of the reviewed spec."""
    function_name = definition.rsplit("::", 1)[1]
    (spec,) = (row for row in _GFMC_FUNCTION_SPECS if row.function_name == function_name)
    return [_gfmc_case_id(spec, params, target) for target in spec.targets], []


def mixed_continuous_keys(definition, params, precision):
    return [_mixed.mixed_route_obligation_id("continuous", params["quantity"], params["route"])], []


def mixed_gfmc_keys(definition, params, precision):
    return [_mixed.mixed_route_obligation_id("gfmc", params["family"], params["route"])], []

"""Mechanical coverage contract for the Rust analytical-derivative parity suite.

The Python analytical implementation owns the physical fixture universe.  Rust is a
second implementation of that production path, so its parity tests must not maintain a
smaller, hand-copied universe.  This module derives the required cells from
``tests.test_jqmc_kernel.analytic_coverage_manifest`` and discovers the registrations in
the two production Rust matrices.  The two sides are intentionally independent: changing
the analytical universe or deleting a Rust parametrization produces a diff.

The stored ``eps`` token is split semantically by boundary: periodic cells prove physical
cell-strain parity and molecular cells prove the non-observable molecular-affine capability.
Every upstream obligation is blocking; the Rust/XLA contract has no N/A subtraction.
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

import hashlib
import json
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from gdd.engine import derivative_requirements as _analytical
from tests.test_jqmc_kernel import analytic_coverage_registration as _registration
from tests.test_jqmc_kernel import gen_coverage_readme as _inventory
from tests.test_jqmc_kernel import _generated_matrix_base as _matrix
from tests.test_rs_kernel import test_gfmc_analytic_matrix as _rust_gfmc
from tests.test_rs_kernel import test_route_parity_param_deriv as _rust_route
from tests.test_rs_kernel import test_rs_analytic_matrix as _rust_c
from tests.test_rs_kernel import test_rs_analytic_matrix_rRe as _rust_geometry


FAMILIES = ("ln_Psi", "e_L")
FAMILY_QUANTITY = {"ln_Psi": "ln_Psi", "e_L": "e_L"}
QUANTITY_FAMILY = {value: key for key, value in FAMILY_QUANTITY.items()}
AXES = ("c", "r", "R", "eps")

# This digest is over the upstream analytical requirement contract (IDs, proof and
# precision metadata included), not over Rust's registrations.  A main-side inventory
# or semantic-contract change therefore requires explicit Rust review before this
# snapshot may be updated.
APPROVED_UPSTREAM_REQUIREMENTS_SHA256 = "7c0ec7bfcc22d2b50dcc8ed3f2ee5875cacb17df3b6c9c91b5b1a0be4ae21ebe"


@dataclass(frozen=True)
class ContinuousRegistrationDiff:
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    duplicate: tuple[str, ...]
    unavailable: tuple[str, ...]
    block_mismatch: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not (self.missing or self.extra or self.duplicate or self.unavailable or self.block_mismatch)


@dataclass(frozen=True)
class GfmcRegistrationDiff:
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    duplicate: tuple[str, ...]
    function_mismatch: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not (self.missing or self.extra or self.duplicate or self.function_mismatch)


def upstream_requirements_hash() -> str:
    payload = [
        {
            "id": requirement.id,
            "family": requirement.family,
            "category": requirement.category,
            "source_id": requirement.source_id,
            "outcome": requirement.expected_outcome.value,
            "proof": requirement.proof.value if requirement.proof is not None else None,
            "precision": requirement.required_precision,
            "leaves": requirement.expected_leaves,
            "liveness": requirement.liveness_required,
        }
        for requirement in _analytical.REQUIRED_CASES
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def upstream_snapshot_matches() -> bool:
    return upstream_requirements_hash() == APPROVED_UPSTREAM_REQUIREMENTS_SHA256


def _case_id(cell: tuple[str, ...], family: str, axis: str) -> str:
    return f"{family}|matrix|{_matrix._cell_id(cell)}|axis={axis}"


def required_continuous_case_ids() -> tuple[str, ...]:
    """Every upstream ln_Psi/e_L matrix value, including molecular-affine parity."""
    return tuple(
        requirement.id
        for requirement in _analytical.REQUIRED_CASES
        if requirement.category == "matrix" and requirement.family in FAMILIES
    )


def _parameter_values(param) -> tuple:
    return tuple(param.values)


def _pytest_axis(function, name: str) -> tuple:
    marks = [mark for mark in function.pytestmark if mark.name == "parametrize" and mark.args and mark.args[0] == name]
    if len(marks) != 1:
        raise ValueError(f"{function.__name__}: expected one {name!r} parametrization, found {len(marks)}")
    return tuple(value.values[0] if getattr(value, "values", None) is not None else value for value in marks[0].args[1])


def discovered_continuous_case_ids() -> tuple[str, ...]:
    """Discover actual Rust matrix registrations from pytest's parameter objects."""
    found: list[str] = []
    # c is split by active parameter block; collapse it to the physical cell after
    # checking the complete block set separately in ``continuous_registration_diff``.
    seen_c: set[tuple[tuple[str, ...], str]] = set()
    for param in _rust_c._PARAMS:
        cell, _block, quantity = _parameter_values(param)
        key = (tuple(cell), quantity)
        if key not in seen_c:
            found.append(_case_id(tuple(cell), QUANTITY_FAMILY[quantity], "c"))
            seen_c.add(key)
    for param in _rust_geometry._PARAMS:
        cell, axis, quantity = _parameter_values(param)
        found.append(_case_id(tuple(cell), QUANTITY_FAMILY[quantity], axis))
    return tuple(found)


def _expected_c_blocks(cell: tuple[str, ...]) -> frozenset[str]:
    """Derive c-axis owners from the upstream P01-P15 parameter inventory.

    In particular, J3 radial leaves are not inferred from Rust's current ``_active_blocks``
    helper.  P09/P10 require a simple GTO/pGTO J3 owner and P14/P15 require the GTO
    child of a composite J3 owner; omitting ``J3Basis`` therefore remains visible.
    """
    ao, jas, _ecp, _bc, _gem = _matrix._split_cell(cell)
    blocks = {"Det"}
    if ao in _rust_c._RADIAL_AO:
        blocks.add("Basis")
    if jas in {"J1", "J2", "J3poly", "DTN"}:
        blocks.update({"J1", "J2"})
    if jas == "J3poly":
        blocks.add("J3")
        if ao in _rust_c._RADIAL_AO:
            blocks.add("J3Basis")
    if jas == "DTN":
        blocks.add("DTN")
    return frozenset(blocks)


def continuous_registration_diff() -> ContinuousRegistrationDiff:
    required = set(required_continuous_case_ids())
    discovered = discovered_continuous_case_ids()
    counts = Counter(discovered)
    registered = set(discovered)

    actual_blocks: dict[tuple[tuple[str, ...], str], set[str]] = defaultdict(set)
    for param in _rust_c._PARAMS:
        cell, block, quantity = _parameter_values(param)
        actual_blocks[(tuple(cell), quantity)].add(block)
    block_mismatch = []
    for cell, _heavy in _matrix._generate_matrix():
        for quantity in FAMILY_QUANTITY.values():
            expected = _expected_c_blocks(tuple(cell))
            actual = frozenset(actual_blocks[(tuple(cell), quantity)])
            if actual != expected:
                block_mismatch.append(
                    f"{_matrix._cell_id(tuple(cell))}/{quantity}: expected={sorted(expected)}, actual={sorted(actual)}"
                )

    unavailable = tuple(
        sorted(
            _matrix._cell_id(tuple(cell)) for cell, _heavy in _matrix._generate_matrix() if not _matrix._buildable(tuple(cell))
        )
    )
    return ContinuousRegistrationDiff(
        missing=tuple(sorted(required - registered)),
        extra=tuple(sorted(registered - required)),
        duplicate=tuple(sorted(case_id for case_id, count in counts.items() if count != 1)),
        unavailable=unavailable,
        block_mismatch=tuple(sorted(block_mismatch)),
    )


def required_gfmc_case_ids() -> tuple[str, ...]:
    """All main e_L_t/e_L_n direct-AD and estimator-identity numerical obligations."""
    return tuple(
        requirement.id
        for requirement in _analytical.REQUIRED_CASES
        if requirement.family in {"e_L_t", "e_L_n"} and requirement.category in {"gfmc-direct", "gfmc-identity"}
    )


def discovered_gfmc_case_ids() -> tuple[str, ...]:
    """Expand the actual Rust pytest registrations to their consumed output axes."""
    found: list[str] = []
    for param in _rust_gfmc._PARAMS:
        spec, parameters = _parameter_values(param)
        found.extend(_analytical._gfmc_case_id(spec, parameters, target) for target in spec.targets)
    return tuple(found)


def gfmc_registration_diff() -> GfmcRegistrationDiff:
    required = set(required_gfmc_case_ids())
    discovered = discovered_gfmc_case_ids()
    counts = Counter(discovered)
    registered = set(discovered)

    expected_functions = {spec.function_name for spec in _analytical._GFMC_FUNCTION_SPECS}
    actual_functions = {_parameter_values(param)[0].function_name for param in _rust_gfmc._PARAMS}
    return GfmcRegistrationDiff(
        missing=tuple(sorted(required - registered)),
        extra=tuple(sorted(registered - required)),
        duplicate=tuple(sorted(case_id for case_id, count in counts.items() if count != 1)),
        function_mismatch=tuple(
            sorted(
                [f"missing:{name}" for name in expected_functions - actual_functions]
                + [f"extra:{name}" for name in actual_functions - expected_functions]
            )
        ),
    )


def required_rust_case_ids() -> tuple[str, ...]:
    """Every XLA obligation, with no Rust-side N/A subtraction."""
    return tuple(requirement.id for requirement in _analytical.REQUIRED_CASES)


def discovered_leaf_case_ids() -> tuple[str, ...]:
    scenarios = _pytest_axis(_rust_route.test_parameter_leaf_liveness_matches_main_oracle, "scenario")
    families = _pytest_axis(_rust_route.test_parameter_leaf_liveness_matches_main_oracle, "family")
    return tuple(
        f"{family}|leaf|{source_id}"
        for scenario in scenarios
        for family in families
        for source_id in _registration._LEAF_SOURCES_BY_SCENARIO[scenario]
    )


def discovered_v1_v2_regime_case_ids() -> tuple[str, ...]:
    found = list(
        _pytest_axis(
            _rust_geometry.test_rs_numerical_regime_matches_main_autograd,
            "requirement_id",
        )
    )
    leaf_ids = set(discovered_leaf_case_ids())
    if all(f"{family}|leaf|P{i:02d}" in leaf_ids for family in FAMILIES for i in range(1, 16)):
        found.extend(f"{family}|regime|R03" for family in FAMILIES)

    route_patterns = {pattern for pattern, _basis, _heavy in _rust_route._REAL_CELLS}
    if {"mol_h2_cart_sd_jnone", "mol_cart_sd_j12"} <= route_patterns:
        found.extend(f"{family}|regime|R09" for family in FAMILIES)
    return tuple(found)


def discovered_v1_v2_validity_case_ids() -> tuple[str, ...]:
    families = _pytest_axis(_rust_route.test_parameter_validity_disposition_matches_main, "family")
    dispositions = _pytest_axis(_rust_route.test_parameter_validity_disposition_matches_main, "disposition")
    return tuple(
        f"{family}|{_analytical._VALIDITY_REQUIREMENT_PARTS[disposition][0]}|"
        f"{_analytical._VALIDITY_REQUIREMENT_PARTS[disposition][1]}|{disposition}"
        for disposition in dispositions
        for family in families
    )


def discovered_gfmc_targeted_case_ids() -> tuple[str, ...]:
    regimes = _pytest_axis(_rust_gfmc.test_gfmc_numerical_regime_matches_rust, "requirement_id")
    output_families = _pytest_axis(_rust_gfmc.test_gfmc_output_liveness_matches_rust, "family")
    validity_families = _pytest_axis(
        _rust_gfmc.test_gfmc_invalid_non_local_move_matches_public_contract,
        "family",
    )
    return (
        tuple(regimes)
        + tuple(f"{family}|output|B24" for family in output_families)
        + tuple(f"{family}|error|C10|unknown-non-local-move" for family in validity_families)
    )


def rust_required_registration_diff() -> GfmcRegistrationDiff:
    """One-to-one registration gate for all 979 required Rust obligations."""
    required = set(required_rust_case_ids())
    discovered = (
        discovered_continuous_case_ids()
        + discovered_leaf_case_ids()
        + discovered_v1_v2_regime_case_ids()
        + discovered_v1_v2_validity_case_ids()
        + ("ln_Psi|identity|B20|potential-independent",)
        + discovered_gfmc_case_ids()
        + discovered_gfmc_targeted_case_ids()
    )
    counts = Counter(discovered)
    registered = set(discovered)
    identity_present = callable(getattr(_rust_route, "test_v1_potential_independence_native_contract", None))
    return GfmcRegistrationDiff(
        missing=tuple(sorted(required - registered)),
        extra=tuple(sorted(registered - required)),
        duplicate=tuple(sorted(case_id for case_id, count in counts.items() if count != 1)),
        function_mismatch=(() if identity_present else ("missing:ln_Psi-potential-identity",)),
    )


def upstream_inventory_is_the_one_reviewed_here() -> bool:
    """Pin the named leaf/regime vocabularies imported from main.

    The numerical Rust evidence is housed in the production parity modules, but its
    required vocabulary must expand immediately when main adds a new P/R row.
    """
    leaf_ids = tuple(row[0] for row in _inventory.PARAMETER_LEAF_INVENTORY)
    regime_ids = tuple(row[0] for row in _inventory.PRODUCTION_NUMERICAL_REGIMES)
    return leaf_ids == tuple(f"P{i:02d}" for i in range(1, 16)) and regime_ids == tuple(f"R{i:02d}" for i in range(1, 11))


def rust_gdd_binding_issues() -> tuple[str, ...]:
    """Return gaps in the runtime-fragment layer for all 979 required Rust obligations."""
    issues = []
    diff = rust_required_registration_diff()
    if not diff.complete:
        issues.append(
            f"registration:missing={len(diff.missing)},extra={len(diff.extra)},"
            f"duplicate={len(diff.duplicate)},functions={len(diff.function_mismatch)}"
        )
    if set(required_rust_case_ids()) != {requirement.id for requirement in _analytical.REQUIRED_CASES}:
        issues.append("required-universe")
    root = Path(__file__).resolve().parents[2]
    required_markers = {
        "tests/test_rs_kernel/test_rs_analytic_matrix.py": {"record_rust_matrix_fragment"},
        "tests/test_rs_kernel/test_rs_analytic_matrix_rRe.py": {
            "record_rust_matrix_fragment",
            "mirror_upstream_receipt",
        },
        "tests/test_rs_kernel/test_route_parity_param_deriv.py": {"record_rust_bound_case"},
        "tests/test_rs_kernel/test_gfmc_analytic_matrix.py": {
            "record_rust_bound_case",
            "mirror_upstream_receipt",
        },
    }
    for relative, markers in required_markers.items():
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        for marker in sorted(markers - calls):
            issues.append(f"{relative}:missing-{marker}")
    return tuple(issues)


def required_runtime_fragment_keys() -> tuple[str, ...]:
    """Return the exact non-derived Rust fragment keys expected across all workers."""
    keys = []
    for param in _rust_c._PARAMS:
        cell, block, quantity = _parameter_values(param)
        upstream = _case_id(tuple(cell), QUANTITY_FAMILY[quantity], "c")
        keys.append(f"{upstream}::{block}")
    for param in _rust_geometry._PARAMS:
        cell, axis, quantity = _parameter_values(param)
        upstream = _case_id(tuple(cell), QUANTITY_FAMILY[quantity], axis)
        keys.append(f"{upstream}::{axis}")
    keys.extend(f"{upstream}::{upstream.rsplit('|', 1)[-1]}" for upstream in discovered_leaf_case_ids())
    keys.extend(
        f"{upstream}::regime"
        for upstream in _pytest_axis(
            _rust_geometry.test_rs_numerical_regime_matches_main_autograd,
            "requirement_id",
        )
    )
    keys.extend(
        f"{family}|{_analytical._VALIDITY_REQUIREMENT_PARTS[disposition][0]}|"
        f"{_analytical._VALIDITY_REQUIREMENT_PARTS[disposition][1]}|{disposition}::{disposition}"
        for disposition in _pytest_axis(_rust_route.test_parameter_validity_disposition_matches_main, "disposition")
        for family in _pytest_axis(_rust_route.test_parameter_validity_disposition_matches_main, "family")
    )
    keys.append("ln_Psi|identity|B20|potential-independent::c-r-R")
    for param in _rust_gfmc._PARAMS:
        spec, parameters = _parameter_values(param)
        keys.extend(f"{_analytical._gfmc_case_id(spec, parameters, target)}::{target}" for target in spec.targets)
    regimes = _pytest_axis(_rust_gfmc.test_gfmc_numerical_regime_matches_rust, "requirement_id")
    keys.extend(f"{upstream}::regime" for upstream in regimes)
    output_families = _pytest_axis(_rust_gfmc.test_gfmc_output_liveness_matches_rust, "family")
    keys.extend(f"{family}|output|B24::output" for family in output_families)
    validity_families = _pytest_axis(
        _rust_gfmc.test_gfmc_invalid_non_local_move_matches_public_contract,
        "family",
    )
    keys.extend(f"{family}|error|C10|unknown-non-local-move::validity" for family in validity_families)
    if len(keys) != len(set(keys)):
        raise AssertionError("Rust runtime fragment inventory contains duplicates")
    return tuple(sorted(keys))


# -- derivative registry key functions (issue 789) ----------------------------
#
# Node-level emitters of every Rust fragment: ``gdd.engine.derivative_registry``
# maps each emitting Rust definition to one of these. Each derives, from the
# case's real callspec values and with the same id helpers the test uses, the
# fragments it emits and the upstream XLA case receipts it replays (the targeted
# Rust tests call main's evidence function, whose own receipt is emitted too).
# They return ``(fragment keys, upstream receipts)``. ln_Psi/e_L R03 and R09 have
# no Rust receipt: they stay derived from their leaf fragments.


def _fragment(upstream: str, component: str) -> str:
    from gdd.engine.derivative_registry import fragment_key

    return fragment_key(upstream, component)


def _upstream_xla():
    from tests.test_jqmc_kernel import test_analytic_coverage_evidence as evidence
    from tests.test_jqmc_kernel import test_gfmc_analytic_coverage_evidence as gfmc_evidence

    return evidence, gfmc_evidence


def matrix_c_fragment_keys(definition, params, precision):
    upstream = _matrix.matrix_obligation_id(params["cell"], params["quantity"], "c")
    return [_fragment(upstream, params["block"])], []


def matrix_geometry_fragment_keys(definition, params, precision):
    upstream = _matrix.matrix_obligation_id(params["cell"], params["quantity"], params["axis"])
    return [_fragment(upstream, params["axis"])], []


def regime_fragment_keys(definition, params, precision):
    upstream = params["requirement_id"]
    return [_fragment(upstream, "regime")], [upstream]


def leaf_fragment_keys(definition, params, precision):
    evidence, _ = _upstream_xla()
    family, scenario = params["family"], params["scenario"]
    return [
        _fragment(evidence.leaf_obligation_id(family, source_id), source_id)
        for source_id in evidence._LEAF_SOURCE_IDS_BY_SCENARIO[scenario]
    ], []


def validity_fragment_keys(definition, params, precision):
    evidence, _ = _upstream_xla()
    disposition = params["disposition"]
    return [_fragment(evidence.validity_obligation_id(params["family"], disposition), disposition)], []


def identity_fragment_keys(definition, params, precision):
    evidence, _ = _upstream_xla()
    return [_fragment(evidence.IDENTITY_OBLIGATION_ID, "c-r-R")], []


def gfmc_matrix_fragment_keys(definition, params, precision):
    spec, parameters = params["spec"], params["parameters"]
    return [_fragment(_analytical._gfmc_case_id(spec, parameters, target), target) for target in spec.targets], []


def gfmc_output_fragment_keys(definition, params, precision):
    _, gfmc_evidence = _upstream_xla()
    upstream = gfmc_evidence.output_obligation_id(params["family"])
    return [_fragment(upstream, "output")], [upstream]


def gfmc_invalid_fragment_keys(definition, params, precision):
    _, gfmc_evidence = _upstream_xla()
    upstream = gfmc_evidence.invalid_move_obligation_id(params["family"])
    return [_fragment(upstream, "validity")], [upstream]

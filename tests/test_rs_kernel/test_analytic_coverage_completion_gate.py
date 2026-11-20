"""Blocking completion gates for Rust analytical-derivative parity coverage."""

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

import pytest

from tests.test_rs_kernel.analytic_coverage_manifest import (
    continuous_registration_diff,
    gfmc_registration_diff,
    required_rust_case_ids,
    rust_required_registration_diff,
    discovered_leaf_case_ids,
    discovered_v1_v2_regime_case_ids,
    discovered_gfmc_targeted_case_ids,
)
from gdd.engine.derivative_requirements import REQUIRED_CASES

pytestmark = [pytest.mark.light, pytest.mark.without_chdir, pytest.mark.design_pin]


def test_upstream_analytical_requirements_have_rust_registrations():
    """Rust supplies live evidence for upstream requirements without a second snapshot pin."""
    assert rust_required_registration_diff().complete


def test_rust_continuous_matrix_registration_is_complete():
    diff = continuous_registration_diff()
    assert diff.complete, (
        "Rust log-wavefunction/local-energy parity is not a complete mirror of main's analytical fixture matrix. "
        "A missing J3Basis owner means P09/P10/P14/P15 are not compared; a missing physical "
        "case means a main analytical result has no Rust equality proof. "
        f"missing={diff.missing}, extra={diff.extra}, duplicate={diff.duplicate}, "
        f"unavailable={diff.unavailable}, block_mismatch={diff.block_mismatch}"
    )


def test_rust_gfmc_matrix_registration_is_complete():
    diff = gfmc_registration_diff()
    assert diff.complete, (
        "Rust conventional/load-balanced algorithm parity is not a complete mirror of main's direct-AD and "
        "estimator-identity matrix. Every registered row executes the authoritative "
        "main oracle with the analytical entry replaced by Rust; deleting a function, "
        "parameter value, or consumed output axis must therefore block CI. "
        f"missing={diff.missing}, extra={diff.extra}, duplicate={diff.duplicate}, "
        f"function_mismatch={diff.function_mismatch}"
    )


def test_every_required_rust_obligation_has_blocking_evidence():
    diff = rust_required_registration_diff()
    assert diff.complete, (
        "Rust must register every main analytical obligation with no N/A subtraction. "
        "This includes molecular-affine parity, P01-P15 liveness, R01-R10 "
        "preconditions, validity/error outcomes, both algorithm outputs, and all direct/identity "
        "rows. Updating only a digest cannot satisfy this one-to-one gate. "
        f"missing={diff.missing}, extra={diff.extra}, duplicate={diff.duplicate}, "
        f"function_mismatch={diff.function_mismatch}"
    )


def test_rust_contract_has_no_not_applicable_derivative_obligations():
    assert set(required_rust_case_ids()) == {requirement.id for requirement in REQUIRED_CASES}
    assert rust_required_registration_diff().complete


def test_upstream_leaf_and_regime_requirements_have_native_evidence():
    """Every upstream parameter leaf and numerical regime has actual Rust test evidence."""
    required = {requirement.id for requirement in REQUIRED_CASES if requirement.category in {"leaf", "regime"}}
    observed = (
        set(discovered_leaf_case_ids()) | set(discovered_v1_v2_regime_case_ids()) | set(discovered_gfmc_targeted_case_ids())
    )
    assert required <= observed


@pytest.mark.parametrize(
    "filename",
    [
        "test_rs_analytic_matrix.py",
        "test_rs_analytic_matrix_rRe.py",
        "test_gfmc_analytic_matrix.py",
        "test_route_parity_param_deriv.py",
    ],
)
def test_rust_analytic_matrices_have_no_green_escape(filename):
    path = Path(__file__).with_name(filename)
    tree = ast.parse(path.read_text())

    def is_pytest_escape(node: ast.AST) -> bool:
        if not isinstance(node, ast.Attribute) or node.attr not in {"skip", "xfail", "skipif"}:
            return False
        owner = node.value
        if isinstance(owner, ast.Name):
            return owner.id == "pytest"
        return (
            isinstance(owner, ast.Attribute)
            and owner.attr == "mark"
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "pytest"
        )

    escapes = [node.lineno for node in ast.walk(tree) if is_pytest_escape(node)]

    assert not escapes, f"{filename} can turn parity obligations green at lines {escapes}"

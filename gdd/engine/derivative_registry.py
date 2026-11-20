"""Node-level registry of every derivative receipt a pytest case emits (issue 789).

Every derivative key of the exact universes (XLA full and mixed, FD, the MIXED
production routes, Rust fragments) is an attribute of the collected pytest case
that emits it. ``EMITTERS`` maps each emitting test definition to its family and
to the key function that derives the case's keys from its real callspec values,
with the same helpers the test itself uses to build its obligation ids; the
exact node id is the collected item's own ``nodeid``. No id string is built.

Two consumers:

* collection (preflight and the PR structure check, ``GDD_EXECUTION_CONTRACT_SCOPE=1``): the
  execution recorder annotates each collected row with its ``derivative_keys``;
  ``gdd.engine.derivative_closure.collection_closure_errors`` closes the
  universes over those rows (``scripts/check_gdd_required_cases.py``);
* execution: ``DerivativeSelfCheck`` fails a passed case at teardown unless the
  receipts it emitted, classified by the aggregator's ``classify`` in the case's
  lane, are exactly its registered ``(lane, key)`` pairs, once each, and each
  receipt passes the existing validation; it then attests the count in the
  execution record.

``scripts/check_gdd_derivative_registry.py`` statically pins this registry 1:1
to the recorder call sites in the tests. Rust R03/R09 of ln_Psi/e_L have no
receipt: they stay derived from their leaf fragments.
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

import importlib
from collections import Counter
from functools import cache
from typing import Any

import pytest

_K = "tests/test_jqmc_kernel/"
_R = "tests/test_rs_kernel/"
_XLA = "tests.test_jqmc_kernel.analytic_coverage_registration"
_FD = "tests.test_jqmc_kernel._gdd_fd_adapters"
_RUST = "tests.test_rs_kernel.analytic_coverage_manifest"

# Evidence lane of each family per precision mode; a missing precision means the
# family emits nothing in that mode.
FAMILY_LANES = {
    "xla": {"full": "xla-full", "mixed": "mixed"},
    "fd": {"full": "xla-full"},
    "mixed-route": {"mixed": "mixed"},
    "rust": {"full": "rust-full", "mixed": "rust-mixed"},
}

# definition -> (family, "module:key_function"). Literal on purpose: the static
# registry checker reads it without importing JAX or the tests.
STATIC_EMITTERS = {
    f"{_K}test_analytic_grad_matches_autograd_generated_molecular.py::test_matrix_cell": ("xla", f"{_XLA}:matrix_cell_keys"),
    f"{_K}test_analytic_grad_matches_autograd_generated_periodic.py::test_matrix_cell": ("xla", f"{_XLA}:matrix_cell_keys"),
    f"{_K}test_analytic_grad_matches_autograd_generated_twist.py::test_matrix_cell": ("xla", f"{_XLA}:matrix_cell_keys"),
    f"{_K}test_analytic_coverage_evidence.py::test_parameter_leaf_evidence": ("xla", f"{_XLA}:leaf_keys"),
    f"{_K}test_analytic_coverage_evidence.py::test_validity_disposition": ("xla", f"{_XLA}:validity_keys"),
    f"{_K}test_analytic_coverage_evidence.py::test_v1_potential_independence_identity": ("xla", f"{_XLA}:identity_keys"),
    f"{_K}test_analytic_coverage_evidence.py::test_numerical_regime_evidence": ("xla", f"{_XLA}:regime_keys"),
    f"{_K}test_gfmc_analytic_coverage_evidence.py::test_gfmc_output_liveness": ("xla", f"{_XLA}:gfmc_output_keys"),
    f"{_K}test_gfmc_analytic_coverage_evidence.py::test_gfmc_invalid_non_local_move": ("xla", f"{_XLA}:gfmc_invalid_keys"),
    f"{_K}test_gfmc_analytic_coverage_evidence.py::test_gfmc_numerical_regime_evidence": ("xla", f"{_XLA}:regime_keys"),
    f"{_K}test_mixed_precision.py::test_mixed_production_continuous_analytical_smoke": (
        "mixed-route",
        f"{_XLA}:mixed_continuous_keys",
    ),
    f"{_K}test_mixed_precision.py::test_mixed_production_gfmc_analytical_smoke": ("mixed-route", f"{_XLA}:mixed_gfmc_keys"),
    f"{_K}test_atomic_orbital.py::TestComputeAOsGradLaplacian::test_AOs_grad_auto_vs_numerical_molecular": (
        "fd",
        f"{_FD}:fd_keys",
    ),
    f"{_K}test_atomic_orbital.py::TestComputeAOsGradLaplacian::test_AOs_grad_auto_vs_numerical_periodic": (
        "fd",
        f"{_FD}:fd_keys",
    ),
    f"{_K}test_atomic_orbital.py::TestComputeAOsGradLaplacian::test_AOs_laplacian_auto_vs_numerical_molecular": (
        "fd",
        f"{_FD}:fd_keys",
    ),
    f"{_K}test_atomic_orbital.py::TestComputeAOsGradLaplacian::test_AOs_laplacian_auto_vs_numerical_periodic": (
        "fd",
        f"{_FD}:fd_keys",
    ),
    f"{_K}test_molecular_orbital.py::TestMOsPeriodic::test_MOs_periodic_grad_auto_vs_numerical": ("fd", f"{_FD}:fd_keys"),
    f"{_K}test_molecular_orbital.py::TestMOsPeriodic::test_MOs_periodic_laplacian_auto_vs_numerical": ("fd", f"{_FD}:fd_keys"),
    f"{_K}test_jastrow_dtn.py::TestDTNGradLaplacian::test_numerical_and_auto_grads_and_laplacian_Jastrow_dtn_molecular": (
        "fd",
        f"{_FD}:fd_keys",
    ),
    f"{_K}test_jastrow_dtn.py::TestDTNGradLaplacian::test_numerical_and_auto_grads_and_laplacian_Jastrow_dtn_periodic": (
        "fd",
        f"{_FD}:fd_keys",
    ),
    f"{_K}test_jastrow_dtn.py::TestDTNPhaseB::test_dtn_position_grad_matches_fd": ("fd", f"{_FD}:fd_keys"),
    f"{_K}test_ao_basis_optimization.py::TestAOBasisOptimizationMisc::test_j3_exponent_gradient_finite_diff": (
        "fd",
        f"{_FD}:fd_keys",
    ),
    f"{_K}test_ao_basis_optimization.py::TestAOBasisOptimizationMisc::test_j3_coefficient_gradient_finite_diff": (
        "fd",
        f"{_FD}:fd_keys",
    ),
    f"{_K}test_ao_basis_optimization.py::TestAOBasisOptimizationMisc::test_geminal_exponent_gradient_finite_diff": (
        "fd",
        f"{_FD}:fd_keys",
    ),
    f"{_K}test_integration_mcmc_stress.py::TestMCMCStressAD::test_stress_HF_fd_vs_ad_per_walker": ("fd", f"{_FD}:fd_keys"),
    f"{_K}test_integration_qe_periodic.py::TestQEPeriodicStressFDvsAD::test_stress_HF_fd_vs_ad_per_walker": (
        "fd",
        f"{_FD}:fd_keys",
    ),
    f"{_K}test_integration_qe_periodic.py::TestQEPeriodicForceFDvsAD::test_force_HF_fd_vs_ad_per_walker": (
        "fd",
        f"{_FD}:fd_keys",
    ),
    f"{_R}test_rs_analytic_matrix.py::test_rs_analytic_matrix": ("rust", f"{_RUST}:matrix_c_fragment_keys"),
    f"{_R}test_rs_analytic_matrix_rRe.py::test_rs_analytic_matrix_rRe": ("rust", f"{_RUST}:matrix_geometry_fragment_keys"),
    f"{_R}test_rs_analytic_matrix_rRe.py::test_rs_numerical_regime_matches_main_autograd": (
        "rust",
        f"{_RUST}:regime_fragment_keys",
    ),
    f"{_R}test_route_parity_param_deriv.py::test_parameter_leaf_liveness_matches_main_oracle": (
        "rust",
        f"{_RUST}:leaf_fragment_keys",
    ),
    f"{_R}test_route_parity_param_deriv.py::test_parameter_validity_disposition_matches_main": (
        "rust",
        f"{_RUST}:validity_fragment_keys",
    ),
    f"{_R}test_route_parity_param_deriv.py::test_v1_potential_independence_native_contract": (
        "rust",
        f"{_RUST}:identity_fragment_keys",
    ),
    f"{_R}test_gfmc_analytic_matrix.py::test_gfmc_analytic_matrix_matches_rust": ("rust", f"{_RUST}:gfmc_matrix_fragment_keys"),
    f"{_R}test_gfmc_analytic_matrix.py::test_gfmc_numerical_regime_matches_rust": ("rust", f"{_RUST}:regime_fragment_keys"),
    f"{_R}test_gfmc_analytic_matrix.py::test_gfmc_output_liveness_matches_rust": ("rust", f"{_RUST}:gfmc_output_fragment_keys"),
    f"{_R}test_gfmc_analytic_matrix.py::test_gfmc_invalid_non_local_move_matches_public_contract": (
        "rust",
        f"{_RUST}:gfmc_invalid_fragment_keys",
    ),
}
# The GFMC force/stress evidence functions emit from their module's autouse
# fixture, one definition per reviewed GFMC function spec.
GFMC_FORCE_MODULE = f"{_K}test_gfmc_force_analytic.py"
SELFCHECK_PLUGIN = "gdd-derivative-selfcheck"
SYNTHETIC_MARKER = "gdd_synthetic_derivative_receipts"


@cache
def emitters() -> dict[str, tuple[str, str]]:
    """Every registered emitting definition: the literal table plus one entry per GFMC spec."""
    from gdd.engine.derivative_requirements import _GFMC_FUNCTION_SPECS

    table = dict(STATIC_EMITTERS)
    for spec in _GFMC_FUNCTION_SPECS:
        definition = f"{GFMC_FORCE_MODULE}::{spec.function_name}"
        if definition in table:
            raise ValueError(f"duplicate derivative emitter {definition}")
        table[definition] = ("xla", f"{_XLA}:gfmc_force_keys")
    return table


@cache
def _key_function(target: str):
    module, name = target.split(":")
    return getattr(importlib.import_module(module), name)


def derivative_keys(definition: str, params: dict[str, Any], precision: str) -> dict[str, Any] | None:
    """The registered derivative receipts of one case, or None for an unregistered definition.

    ``keys`` are the universe keys of the lane; ``upstream`` are the XLA case
    receipts a Rust parity case emits while replaying its upstream obligation.
    """
    entry = emitters().get(definition)
    if entry is None:
        return None
    family, target = entry
    lane = FAMILY_LANES[family].get(precision)
    if lane is None:
        return {"lane": None, "keys": [], "upstream": []}
    keys, upstream = _key_function(target)(definition, dict(params), precision)
    if not keys:
        raise ValueError(f"{definition}: a registered case derives no derivative key from {params}")
    return {"lane": lane, "keys": sorted(keys), "upstream": sorted(upstream)}


def fragment_key(upstream_obligation_id: str, component_id: str) -> str:
    """The universe key of one Rust fragment (``required_rust_fragments``' format)."""
    return f"{upstream_obligation_id}::{component_id}"


# The XLA lane a Rust lane's upstream replay receipts belong to (same precision).
UPSTREAM_LANES = {"rust-full": "xla-full", "rust-mixed": "mixed"}


def classify_receipt(record: dict[str, Any], lane: str) -> tuple[str, str] | None:
    """The ``(lane, key)`` an emitted receipt closes, with the aggregator's own ``classify``.

    A primary receipt is classified in its case's lane. In a Rust lane a
    non-fragment receipt is an upstream replay: it must classify as an XLA
    receipt of the same precision (``MATH-DERIV-*`` guarantee), exactly as it
    would in that XLA lane. None means the receipt closes nothing.
    """
    from gdd.engine.derivative_closure import classify

    if lane in UPSTREAM_LANES and record.get("record_kind") != "rust-fragment":
        return classify({**record, "lane": UPSTREAM_LANES[lane]})
    return classify({**record, "lane": lane})


def _definition(item) -> str:
    return item.nodeid.split("[", 1)[0]


class DerivativeSelfCheck:
    """Fail a passed case unless it emitted exactly its registered receipts (teardown).

    Every emitted receipt is classified with the aggregator's ``classify`` into
    the case's lane (an upstream replay into the XLA lane of the same
    precision); the multiset of classified ``(lane, key)`` pairs must equal the
    registered one, so a receipt with the right id but the wrong guarantee,
    kind or lane closes nothing and fails the case. Each receipt must also pass
    ``validate_record``.

    Active in every session of the test tree, authoritative or not: receipts are
    collected in-process from ``emit_completion_record`` whether or not they are
    written. On success the case is attested; the execution recorder writes the
    attestation into the case's execution record and the in-job verifier
    requires it for every planned emitting case.
    """

    def __init__(self, registry=derivative_keys):
        self.registry = registry
        self.attestations: dict[str, dict[str, Any]] = {}
        self._emitted: list[dict[str, Any]] | None = None
        self._call_passed = False

    def _sink(self, record: dict[str, Any]) -> None:
        if self._emitted is None:
            raise AssertionError("a derivative receipt was emitted outside any pytest case")
        self._emitted.append(record)

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item):
        from gdd.engine import evidence

        self._emitted, self._call_passed = [], False
        evidence._SELFCHECK_SINK = self._sink

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_makereport(self, item, call):
        report = yield
        if report.when == "call":
            self._call_passed = report.passed
        return report

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_teardown(self, item, nextitem):
        # Fixtures (including emitting autouse fixtures) finalize inside the yield.
        try:
            result = yield
        finally:
            from gdd.engine import evidence

            evidence._SELFCHECK_SINK = None
            emitted, self._emitted = self._emitted or [], None
        self.check(item, emitted)
        return result

    def check(self, item, emitted: list[dict[str, Any]]) -> None:
        if item.get_closest_marker(SYNTHETIC_MARKER) is not None:
            return
        params = dict(getattr(getattr(item, "callspec", None), "params", {}))
        precision = item.config.getoption("precision_mode", "full")
        expected = self.registry(_definition(item), params, precision)
        if expected is None:
            if emitted:
                raise AssertionError(f"{item.nodeid} emits derivative receipts but is not a registered emitter")
            return
        if not self._call_passed:
            return  # a failed or skipped case is not passing evidence; it is never attested
        lane = expected["lane"]
        want = Counter((lane, key) for key in expected["keys"])
        want.update((UPSTREAM_LANES.get(lane, lane), key) for key in expected["upstream"])
        got = Counter(classify_receipt(record, lane) for record in emitted)
        if got != want:
            missing = sorted((want - got).elements())
            extra = sorted((got - want).elements(), key=str)
            raise AssertionError(
                f"{item.nodeid}: classified derivative receipts differ from the registry "
                f"(None = a receipt that closes nothing in its lane): missing={missing[:10]}, extra={extra[:10]}"
            )
        from gdd.engine.derivative_closure import validate_record

        for record in emitted:
            classified = classify_receipt(record, lane)
            try:
                validate_record({**record, "lane": classified[0]})
            except (AssertionError, ValueError, TypeError, KeyError) as exc:
                raise AssertionError(f"{item.nodeid}: invalid derivative receipt {classified}: {exc}") from None
        if expected["keys"]:
            # The attested count is the planned universe-key count of the case
            # (its upstream replay receipts were checked above, not counted).
            self.attestations[item.nodeid] = {"lane": expected["lane"], "keys": len(expected["keys"])}


def register(config) -> None:
    """Install the self-check in a session of the test tree (tests/conftest.py)."""
    config.pluginmanager.register(DerivativeSelfCheck(), SELFCHECK_PLUGIN)

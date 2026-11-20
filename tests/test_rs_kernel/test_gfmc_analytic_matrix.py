"""Rust parity mirror of main's complete GFMC analytical-derivative matrix.

The authoritative fixture/parameter universe and the independent autograd oracles live
in ``tests.test_jqmc_kernel.test_gfmc_force_analytic``.  Re-copying that universe here
would let the Rust suite silently lag main, so each row below invokes the corresponding
main test function while replacing only its production analytical entry point with the
native Rust implementation.

Consequently a direct-AD row compares Rust to the same independent autograd expression
used by main.  For the e_L_n identity rows only the ``n`` entry point is replaced: the test
therefore compares Rust GFMC_n to the untouched XLA GFMC_t implementation instead of
becoming a Rust-vs-Rust tautology.  Unsupported native combinations raise and keep CI
red; there is deliberately no skip, xfail, allowlist, or JAX fallback in this module.
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

import numpy as np
import pytest

import jqmc_rs
from jqmc.atomic_orbital import is_complex_aos
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
from jqmc._precision import get_tolerance_xla_rust_consistency
from gdd.engine import derivative_requirements as _upstream_manifest
from tests.test_jqmc_kernel import test_gfmc_force_analytic as _upstream_tests
from tests.test_jqmc_kernel import test_gfmc_analytic_coverage_evidence as _upstream_evidence
from tests.test_rs_kernel._gdd_rust_adapters import mirror_upstream_receipt
from tests.test_rs_kernel._gdd_rust_adapters import record_rust_bound_case
from tests.test_jqmc_kernel._gdd_adapters import committed_input

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]

_WEIGHTS = np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=np.float64)
_GRID_POINTS = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=np.float64)
_NN = 1
_NATIVE_DERIVATIVE_ENTRIES = (
    "_gfmc_grad_e_L_ae",
    "_gfmc_grad_e_L_ecp",
    "_gfmc_grad_e_L_ecp_pbc",
    "_gfmc_grad_e_L_ecp_pbc_complex",
    "_gfmc_stress_ae",
    "_gfmc_stress_ecp_jnone",
    "_gfmc_stress_ecp_withj",
    "_gfmc_stress_ecp_pbc_jnone",
    "_gfmc_stress_ecp_pbc_withj",
    "_gfmc_stress_ecp_pbc_complex",
)
_GDD_RUST_CONTEXT = None


def _install_force_stress_parity_tolerance(monkeypatch, target_module):
    """Use the dedicated force/stress tier for the imported XLA oracle checks.

    The upstream matrix owns its ordinary XLA analytical-vs-AD tolerances.  In
    this mirror the analytical side is replaced by an independent Rust
    implementation, so the comparison must use the dedicated XLA/Rust policy.
    ``medium`` is the mixed force/stress tier; in full precision it
    resolves to the same values as strict.
    """
    original_assert = getattr(target_module, "_NP_ASSERT_ALLCLOSE", np.testing.assert_allclose)
    atol, rtol = get_tolerance_xla_rust_consistency(
        (
            "ao_eval",
            "ao_grad_lap",
            "det_eval",
            "wf_kinetic",
            "coulomb",
            "ewald",
            "jastrow_eval",
            "jastrow_grad_lap",
        ),
        "medium",
    )

    def assert_rust_xla_close(actual, desired, *args, **kwargs):
        del args, kwargs
        original_assert(actual, desired, atol=atol, rtol=rtol)
        if _GDD_RUST_CONTEXT is not None:
            _GDD_RUST_CONTEXT["assertions"].append((actual, desired))

    if hasattr(target_module, "_NP_ASSERT_ALLCLOSE"):
        monkeypatch.setattr(target_module, "_NP_ASSERT_ALLCLOSE", assert_rust_xla_close)
    else:
        monkeypatch.setattr(target_module.np.testing, "assert_allclose", assert_rust_xla_close)
    return assert_rust_xla_close, atol, rtol


def _stable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_stable(item) for item in value]
    return repr(value)


def _row_id(spec, parameters) -> str:
    payload = {
        "family": spec.family,
        "function": spec.function_name,
        "parameters": {key: _stable(value) for key, value in sorted(parameters.items())},
        "proof": spec.proof.value,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
    return f"{spec.family}-{spec.function_name}-{digest}"


_PARAMS = tuple(
    pytest.param(spec, parameters, id=_row_id(spec, parameters))
    for spec in _upstream_manifest._GFMC_FUNCTION_SPECS
    for parameters in _upstream_manifest._gfmc_parameter_rows(spec.function_name)
)


def _is_complex(ham) -> bool:
    orb_up = ham.wavefunction_data.geminal_data.orb_data_up_spin
    ao_leaf = orb_up.aos_data if hasattr(orb_up, "aos_data") else orb_up
    return bool(is_complex_aos(ao_leaf))


def _has_jastrow(ham) -> bool:
    jd = ham.wavefunction_data.jastrow_data
    return any(
        getattr(jd, name) is not None
        for name in (
            "jastrow_one_body_data",
            "jastrow_two_body_data",
            "jastrow_three_body_data",
            "jastrow_dtn_data",
        )
    )


def _rust_positions(ham, r_up, r_dn, RT, alat, non_local_move="tmove"):
    """Dispatch one public GFMC position derivative to the native kernel family."""
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    cp = ham.coulomb_potential_data
    ru = np.asarray(r_up, dtype=np.float64)
    rd = np.asarray(r_dn, dtype=np.float64)
    rt = np.asarray(RT, dtype=np.float64)
    if non_local_move not in {None, "tmove", "dltmove"}:
        raise ValueError(f"unsupported non_local_move={non_local_move!r}")
    if bool(cp.ecp_flag) and non_local_move is None:
        raise ValueError("unsupported non_local_move=None for ECP")
    dltmove = non_local_move == "dltmove"
    pbc = bool(ham.structure_data.pbc_flag)
    if _GDD_RUST_CONTEXT is not None:
        _GDD_RUST_CONTEXT["route_calls"].append(("positions", ham, r_up, r_dn))

    if pbc and _is_complex(ham):
        return jqmc_rs._gfmc_grad_e_L_ecp_pbc_complex(gd, jd, cp, ru, rd, alat, rt, _WEIGHTS, _GRID_POINTS, _NN, dltmove)
    if pbc:
        # Despite its name, production intentionally uses this core for
        # both ECP and bare-AE PBC: empty ECP arrays reduce it to the AE expression.
        return jqmc_rs._gfmc_grad_e_L_ecp_pbc(gd, jd, cp, ru, rd, alat, rt, _WEIGHTS, _GRID_POINTS, _NN, dltmove)
    if bool(cp.ecp_flag):
        return jqmc_rs._gfmc_grad_e_L_ecp(gd, jd, cp, ru, rd, alat, rt, _WEIGHTS, _GRID_POINTS, _NN, dltmove)
    return jqmc_rs._gfmc_grad_e_L_ae(gd, jd, cp, ru, rd, alat, rt)


def _rust_strain(ham, r_up, r_dn, RT, alat, non_local_move="tmove"):
    """Dispatch one public GFMC strain derivative to the native kernel family."""
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    cp = ham.coulomb_potential_data
    ru = np.asarray(r_up, dtype=np.float64)
    rd = np.asarray(r_dn, dtype=np.float64)
    rt = np.asarray(RT, dtype=np.float64)
    if non_local_move not in {None, "tmove", "dltmove"}:
        raise ValueError(f"unsupported non_local_move={non_local_move!r}")
    if bool(cp.ecp_flag) and non_local_move is None:
        raise ValueError("unsupported non_local_move=None for ECP")
    dltmove = non_local_move == "dltmove"
    pbc = bool(ham.structure_data.pbc_flag)
    if _GDD_RUST_CONTEXT is not None:
        _GDD_RUST_CONTEXT["route_calls"].append(("strain", ham, r_up, r_dn))

    if pbc and _is_complex(ham):
        return jqmc_rs._gfmc_stress_ecp_pbc_complex(gd, jd, cp, ru, rd, alat, rt, _WEIGHTS, _GRID_POINTS, _NN, dltmove)
    if pbc and _has_jastrow(ham):
        return jqmc_rs._gfmc_stress_ecp_pbc_withj(gd, jd, cp, ru, rd, alat, rt, _WEIGHTS, _GRID_POINTS, _NN, dltmove)
    if pbc:
        if dltmove:
            raise AssertionError("Rust PBC-jnone GFMC stress has no dltmove argument")
        return jqmc_rs._gfmc_stress_ecp_pbc_jnone(gd, cp, ru, rd, alat, rt, _WEIGHTS, _GRID_POINTS, _NN)
    if bool(cp.ecp_flag):
        if _has_jastrow(ham):
            return jqmc_rs._gfmc_stress_ecp_withj(
                gd,
                jd,
                cp,
                ru,
                rd,
                alat,
                rt,
                _WEIGHTS,
                _GRID_POINTS,
                _NN,
                dltmove,
            )
        return jqmc_rs._gfmc_stress_ecp_jnone(gd, cp, ru, rd, alat, rt, _WEIGHTS, _GRID_POINTS, _NN, dltmove)
    return jqmc_rs._gfmc_stress_ae(gd, jd, cp, ru, rd, alat, rt)


@pytest.mark.parametrize("spec,parameters", _PARAMS)
def test_gfmc_analytic_matrix_matches_rust(spec, parameters, monkeypatch):
    """Run one authoritative main GFMC oracle row with its Rust implementation."""
    global _GDD_RUST_CONTEXT
    _GDD_RUST_CONTEXT = {"assertions": [], "route_calls": []}
    parity_assert, atol, rtol = _install_force_stress_parity_tolerance(monkeypatch, _upstream_tests)
    native_calls: list[str] = []
    for name in _NATIVE_DERIVATIVE_ENTRIES:
        original = getattr(jqmc_rs, name)

        def recorded(*args, _name=name, _original=original, **kwargs):
            native_calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(jqmc_rs, name, recorded)

    if spec.proof is _upstream_manifest.ProofKind.DIRECT_AD:
        prefix = "grad_e_L_t" if spec.family == "e_L_t" else "grad_e_L_n"
        monkeypatch.setattr(_upstream_tests.gfmc, f"{prefix}_wrt_positions", _rust_positions)
        monkeypatch.setattr(_upstream_tests.gfmc, f"{prefix}_wrt_strain", _rust_strain)
    else:
        # Keep GFMC_t untouched on identity rows.  This makes the existing n==t
        # assertion a Rust-GFMC_n == XLA-GFMC_t parity assertion.  Main correctly
        # uses array_equal for XLA n-vs-t because both delegate to one expression;
        # Rust and XLA are independent floating-point implementations and are not
        # bit-identical, so replace only that assertion with the ordinary fp64
        # kernel-parity tolerance (the production functions remain untouched).
        monkeypatch.setattr(_upstream_tests.gfmc, "grad_e_L_n_wrt_positions", _rust_positions)
        monkeypatch.setattr(_upstream_tests.gfmc, "grad_e_L_n_wrt_strain", _rust_strain)

        def assert_rust_xla_close(actual, desired, *args, **kwargs):
            del args, kwargs
            parity_assert(actual, desired, atol=atol, rtol=rtol)

        monkeypatch.setattr(_upstream_tests, "_NP_ASSERT_ARRAY_EQUAL", assert_rust_xla_close)

    test_function = getattr(_upstream_tests, spec.function_name)
    test_function(**parameters)
    assert native_calls, (
        f"{spec.function_name}{parameters}: the Rust parity row completed without invoking "
        "a native GFMC derivative entry point; a JAX fallback would make the comparison vacuous"
    )
    assert _GDD_RUST_CONTEXT["assertions"] and _GDD_RUST_CONTEXT["route_calls"]
    source_ids, artifact_format = _upstream_tests._gdd_source_for_case(spec.function_name, parameters)
    _route, _ham, r_up, r_dn = _GDD_RUST_CONTEXT["route_calls"][0]
    zones = (
        "ao_eval",
        "ao_grad_lap",
        "det_eval",
        "wf_kinetic",
        "coulomb",
        "ewald",
        "jastrow_eval",
        "jastrow_grad_lap",
    )
    for target in spec.targets:
        obligation_id = _upstream_manifest._gfmc_case_id(spec, parameters, target)
        target_assertions = _upstream_tests._gdd_assertions_for_target(_GDD_RUST_CONTEXT["assertions"], spec.targets, target)
        record_rust_bound_case(
            upstream_obligation_id=obligation_id,
            component_id=target,
            pytest_node_id="tests/test_rs_kernel/test_gfmc_analytic_matrix.py::test_gfmc_analytic_matrix_matches_rust",
            input_binding=committed_input(
                obligation_id,
                source_ids=source_ids,
                artifact_format=artifact_format,
                builder_operation=spec.function_name,
            ),
            r_up=r_up,
            r_dn=r_dn,
            generator_operation=spec.function_name,
            seed=None,
            compared_pairs=tuple(target_assertions),
            zones=zones,
            level="medium",
            atol=atol,
            rtol=rtol,
        )
    _GDD_RUST_CONTEXT = None


_TARGETED_REGIME_PARAMS = tuple(
    pytest.param(requirement_id, id=requirement_id) for requirement_id in _upstream_evidence._REGIME_REQUIREMENTS
)


def _install_targeted_rust_family(family, monkeypatch):
    """Replace one e_L_t/e_L_n public family and make n==t a tolerant Rust/XLA check."""
    parity_assert, atol, rtol = _install_force_stress_parity_tolerance(monkeypatch, _upstream_evidence)
    prefix = "grad_e_L_t" if family == "e_L_t" else "grad_e_L_n"
    monkeypatch.setattr(_upstream_tests.gfmc, f"{prefix}_wrt_positions", _rust_positions)
    monkeypatch.setattr(_upstream_tests.gfmc, f"{prefix}_wrt_strain", _rust_strain)

    if family == "e_L_n":
        t_positions = _upstream_evidence.gfmc.grad_e_L_t_wrt_positions

        def rust_n_xla_t_identity(
            ham,
            r_up,
            r_dn,
            RT,
            alat,
            *,
            non_local_move="tmove",
            return_receipt_pairs=False,
        ):
            native = _rust_positions(ham, r_up, r_dn, RT, alat, non_local_move)
            xla = t_positions(ham, r_up, r_dn, RT, alat, non_local_move=non_local_move)
            pairs = []
            for left, right in zip(native, xla, strict=True):
                parity_assert(left, right, atol=atol, rtol=rtol)
                pairs.append((left, right))
            return (native, tuple(pairs)) if return_receipt_pairs else native

        monkeypatch.setattr(_upstream_evidence, "_assert_n_t_identity", rust_n_xla_t_identity)


@pytest.mark.parametrize("family", _upstream_evidence._FAMILIES)
def test_gfmc_output_liveness_matches_rust(family, monkeypatch):
    """Main's shaped, finite, nonzero e_L_t/e_L_n outputs must come from Rust."""
    captured = []
    original = _upstream_evidence.case_receipt

    def capture(**kwargs):
        receipt = original(**kwargs)
        captured.append(receipt)
        return receipt

    monkeypatch.setattr(_upstream_evidence, "case_receipt", capture)
    _install_targeted_rust_family(family, monkeypatch)
    _upstream_evidence.test_gfmc_output_liveness(family)
    assert len(captured) == 1
    mirror_upstream_receipt(
        captured[0],
        component_id="output",
        pytest_node_id="tests/test_rs_kernel/test_gfmc_analytic_matrix.py::test_gfmc_output_liveness_matches_rust",
    )


@pytest.mark.parametrize("family", _upstream_evidence._FAMILIES)
def test_gfmc_invalid_non_local_move_matches_public_contract(family, monkeypatch):
    """The Rust route preserves main's C10 validation instead of boolean-coercing typos."""
    captured = []
    original = _upstream_evidence.case_receipt

    def capture(**kwargs):
        receipt = original(**kwargs)
        captured.append(receipt)
        return receipt

    monkeypatch.setattr(_upstream_evidence, "case_receipt", capture)
    _install_targeted_rust_family(family, monkeypatch)
    _upstream_evidence.test_gfmc_invalid_non_local_move(family)
    assert len(captured) == 1
    mirror_upstream_receipt(
        captured[0],
        component_id="validity",
        pytest_node_id=(
            "tests/test_rs_kernel/test_gfmc_analytic_matrix.py::test_gfmc_invalid_non_local_move_matches_public_contract"
        ),
    )


@pytest.mark.parametrize("requirement_id", _TARGETED_REGIME_PARAMS)
def test_gfmc_numerical_regime_matches_rust(requirement_id, monkeypatch):
    """Reuse main's R01-R10 precondition checks with Rust as the derivative DUT."""
    captured = []
    original = _upstream_evidence.case_receipt

    def capture(**kwargs):
        receipt = original(**kwargs)
        captured.append(receipt)
        return receipt

    monkeypatch.setattr(_upstream_evidence, "case_receipt", capture)
    family = requirement_id.split("|", 1)[0]
    _install_targeted_rust_family(family, monkeypatch)
    _upstream_evidence.test_gfmc_numerical_regime_evidence(requirement_id)
    assert len(captured) == 1
    mirror_upstream_receipt(
        captured[0],
        component_id="regime",
        pytest_node_id="tests/test_rs_kernel/test_gfmc_analytic_matrix.py::test_gfmc_numerical_regime_matches_rust",
    )

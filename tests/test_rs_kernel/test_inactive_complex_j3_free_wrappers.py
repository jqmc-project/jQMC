"""Free native real wrappers with an INACTIVE (all-zero ``j_matrix``) J3 over a complex plane-wave child.

Issue #170 (after #140 / PR #143 and #157). An all-zero ``j_matrix`` makes the three-body Jastrow
factor identically zero, but the remaining free real wrappers re-read the J3 orbital through the
real ``OrbData`` parser unconditionally (``parse_jastrow_full``, the ECP ``parse_j3`` and three
direct reads). When the J3 orbital is an MO over a composite of a real pGTO and a COMPLEX plane-wave
child, that parser drops the complex child, so the orbital has fewer AO rows than MO columns and
the wrappers panicked with ``ndarray: index out of bounds``; ``grad_ln_Psi_jastrow_3b_jmatrix`` and
``compute_grads_and_laplacian_jastrow_three_body`` panicked for an ACTIVE complex-child J3 too.

Fixture: the committed Li Gamma ccECP cell (spherical pGTO single determinant, J1 + J2 + DTN;
``rs_ham("pbc_gamma_sphepgto_jdtn")``) and its AGP form (``to_AGP``) for the ``lambda`` wrappers,
with an identity-MO-wrapped (pGTO s, plane-wave) composite J3 orbital (the #140 / #157 witness
orbital). Children: a complex plane wave with a nonzero G vector (``complex_pw_G``), a complex
constant ``k = 0`` plane wave (``complex_pw_0``), and a real constant plane wave (``real_pw``) as the
control that the real parser represents.

Independent references:

* values, energies and non-J3 derivatives of an inactive J3: the same native call with
  ``jastrow_three_body_data=None`` (J3 is identically 0), compared exactly;
* the J3 ``j_matrix`` derivatives: for a complex child (inactive or active) the documented
  ``NotImplementedError`` hand-off to the JAX route; for the representable inactive control, the
  XLA ``jax.grad`` of ``compute_Jastrow_three_body`` (its derivative is nonzero although the J3 is
  inactive);
* the J3 gradients / Laplacians of an inactive J3: exactly zero (no J3).

Precision (#163): the J3-free identities and the exact zeros compare one precision
mode with itself (the inactive J3 and the J3-free call of the same route, mixed vs mixed or full vs
full), so they run in both sessions. The XLA comparison is analytical vs AD, so it is judged in
full only; the mixed session instead compares the ``grad_ln_Psi_jastrow_3b_jmatrix`` block of the
mixed route with the same route rebuilt in full (``tests/_precision_fidelity.py``, the reviewed
``parameter_derivative`` fidelity budget) and checks that an out-of-budget change is rejected.
The ``NotImplementedError`` dispositions run in both sessions.
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

import functools

import jax
import numpy as np
import pytest

import jqmc_rs
from jqmc._precision import get_tolerance_xla_rust_consistency, is_mixed_precision_enabled
from jqmc._setting import get_tolerance_mixed_fidelity
from jqmc.atomic_orbital import AOs_cart_pgto_data, AOs_composite_data, AOs_plane_wave_complex_data, AOs_plane_wave_data
from jqmc.determinant import to_AGP
from jqmc.jastrow_factor import Jastrow_three_body_data, compute_Jastrow_three_body
from jqmc.jqmc_mcmc import _NV_BASE_MESH, Nv_default
from jqmc.atomic_orbital import is_complex_aos
from jqmc.molecular_orbital import MOs_data
from tests._precision_fidelity import mixed_vs_full, out_of_budget_mixed_block
from tests.test_rs_kernel._rs_fixtures import rs_ham

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.with_chdir

_WEIGHTS = np.asarray(_NV_BASE_MESH[Nv_default].weights, dtype=np.float64)
_GRID = np.asarray(_NV_BASE_MESH[Nv_default].grid_points, dtype=np.float64)
_CONFIG_SEED = 7
_CHILDREN = ["complex_pw_G", "complex_pw_0", "real_pw"]
_COMPLEX_CHILDREN = ["complex_pw_G", "complex_pw_0"]
_J3_REFUSAL = "supported only by the complex native path"

# (geminal form, call(geminal, jastrow, w)) for every free real wrapper that consumes the J3 only
# as part of the wavefunction value; w carries the Coulomb data and the electron configuration.
_VALUE_ROUTES = {
    "grad_T_pbc_wrt_R": ("SD", lambda g, j, w: jqmc_rs.grad_T_pbc_wrt_R(g, j, w["r_up"], w["r_dn"])),
    "grad_T_pbc_wrt_strain": ("SD", lambda g, j, w: jqmc_rs.grad_T_pbc_wrt_strain(g, j, w["r_up"], w["r_dn"])),
    "grad_T_dtn_pbc_wrt_strain": ("SD", lambda g, j, w: jqmc_rs.grad_T_dtn_pbc_wrt_strain(g, j, w["r_up"], w["r_dn"])),
    "grad_eL_wrt_mo_coeff": ("SD", lambda g, j, w: jqmc_rs.grad_eL_wrt_mo_coeff(g, j, w["r_up"], w["r_dn"])),
    "grad_eL_ecp_wrt_mo_coeff": (
        "SD",
        lambda g, j, w: jqmc_rs.grad_eL_ecp_wrt_mo_coeff(g, j, w["cp"], w["r_up"], w["r_dn"], _WEIGHTS, _GRID),
    ),
    "grad_eL_wrt_lambda": ("AGP", lambda g, j, w: jqmc_rs.grad_eL_wrt_lambda(g, j, w["r_up"], w["r_dn"])),
    "grad_eL_ecp_wrt_lambda": (
        "AGP",
        lambda g, j, w: jqmc_rs.grad_eL_ecp_wrt_lambda(g, j, w["cp"], w["r_up"], w["r_dn"], _WEIGHTS, _GRID),
    ),
    "grad_eL_ecp_wrt_lambda_diag": (
        "AGP",
        lambda g, j, w: jqmc_rs.grad_eL_ecp_wrt_lambda_diag(g, j, w["cp"], w["r_up"], w["r_dn"], _WEIGHTS, _GRID),
    ),
    "grad_eL_wrt_jastrow_1b_param": (
        "SD",
        lambda g, j, w: jqmc_rs.grad_eL_wrt_jastrow_1b_param(g, j, w["r_up"], w["r_dn"]),
    ),
    "grad_eL_wrt_jastrow_2b_param": (
        "SD",
        lambda g, j, w: jqmc_rs.grad_eL_wrt_jastrow_2b_param(g, j, w["r_up"], w["r_dn"]),
    ),
    "grad_eL_ecp_wrt_jastrow_1b_param": (
        "SD",
        lambda g, j, w: jqmc_rs.grad_eL_ecp_wrt_jastrow_1b_param(g, j, w["cp"], w["r_up"], w["r_dn"], _WEIGHTS, _GRID),
    ),
    "grad_eL_ecp_wrt_jastrow_2b_param": (
        "SD",
        lambda g, j, w: jqmc_rs.grad_eL_ecp_wrt_jastrow_2b_param(g, j, w["cp"], w["r_up"], w["r_dn"], _WEIGHTS, _GRID),
    ),
    "grad_eL_dtn_kinetic_params": ("SD", lambda g, j, w: jqmc_rs.grad_eL_dtn_kinetic_params(g, j, w["r_up"], w["r_dn"])),
    "grad_eL_ecp_dtn_params": (
        "SD",
        lambda g, j, w: jqmc_rs.grad_eL_ecp_dtn_params(g, j, w["cp"], w["r_up"], w["r_dn"], _WEIGHTS, _GRID),
    ),
    "grad_eL_wrt_radial": ("SD", lambda g, j, w: jqmc_rs.grad_eL_wrt_radial(g, w["r_up"], w["r_dn"], jastrow_data=j)),
    "grad_eL_ecp_wrt_radial": (
        "SD",
        lambda g, j, w: jqmc_rs.grad_eL_ecp_wrt_radial(g, j, w["cp"], w["r_up"], w["r_dn"], _WEIGHTS, _GRID),
    ),
}

# The free real J3 ``j_matrix``-derivative wrappers.
_JMATRIX_ROUTES = {
    "grad_eL_wrt_jastrow_3b_jmatrix": lambda g, j, w: jqmc_rs.grad_eL_wrt_jastrow_3b_jmatrix(g, j, w["r_up"], w["r_dn"]),
    "grad_eL_ecp_wrt_jastrow_3b_jmatrix": lambda g, j, w: jqmc_rs.grad_eL_ecp_wrt_jastrow_3b_jmatrix(
        g, j, w["cp"], w["r_up"], w["r_dn"], _WEIGHTS, _GRID
    ),
    "grad_ln_Psi_jastrow_3b_jmatrix": lambda g, j, w: jqmc_rs.grad_ln_Psi_jastrow_3b_jmatrix(
        j.jastrow_three_body_data, w["r_up"], w["r_dn"]
    ),
}


def _j3(structure, child: str, *, active: bool) -> Jastrow_three_body_data:
    """The identity-MO-wrapped (pGTO s, plane-wave child) composite J3 of #140 / #157."""
    gauss = AOs_cart_pgto_data.build(
        structure_data=structure,
        nucleus_index=(0,),
        num_ao=1,
        num_ao_prim=1,
        orbital_indices=(0,),
        exponents=np.array([1.0]),
        coefficients=np.array([1.0]),
        angular_momentums=(0,),
        polynominal_order_x=(0,),
        polynominal_order_y=(0,),
        polynominal_order_z=(0,),
    )
    reciprocal = 2.0 * np.pi * np.linalg.inv(np.asarray(structure.cell)).T
    if child == "complex_pw_G":
        j3_child = AOs_plane_wave_complex_data.build(structure, np.stack([np.zeros(3), reciprocal[0]]), k_point=(0.0, 0.0, 0.0))
    elif child == "complex_pw_0":
        j3_child = AOs_plane_wave_complex_data.build(structure, np.zeros((1, 3)), k_point=(0.0, 0.0, 0.0))
    else:
        j3_child = AOs_plane_wave_data.build(structure, np.zeros((1, 3)), np.zeros(1))
    n_ao = 1 + int(j3_child.num_ao)
    basis = AOs_composite_data.build((gauss, j3_child))
    dtype = np.complex128 if is_complex_aos(basis) else np.float64
    orb = MOs_data.build(aos_data=basis, mo_coefficients=np.eye(n_ao, dtype=dtype))
    j_matrix = np.zeros((n_ao, n_ao + 1))
    if active:
        block = np.arange(1, n_ao * n_ao + 1, dtype=np.float64).reshape(n_ao, n_ao) / (10.0 * n_ao * n_ao)
        j_matrix[:, :-1] = block + block.T
        j_matrix[:, -1] = 0.05 * np.arange(1, n_ao + 1)
    j3 = Jastrow_three_body_data(orb_data=orb, j_matrix=j_matrix)
    j3.sanity_check()
    return j3


def _build_witness(child: str) -> dict:
    """Geminals, Jastrows (inactive / active / no J3), Coulomb data and one configuration."""
    ham = rs_ham("pbc_gamma_sphepgto_jdtn")
    st = ham.structure_data
    wf = ham.wavefunction_data
    jastrow = wf.jastrow_data
    geminal = wf.geminal_data
    rng = np.random.default_rng(_CONFIG_SEED)
    cell = np.asarray(st.cell, dtype=np.float64)
    return {
        "SD": geminal,
        "AGP": to_AGP(geminal),
        "inactive": jastrow.replace(jastrow_three_body_data=_j3(st, child, active=False)),
        "active": jastrow.replace(jastrow_three_body_data=_j3(st, child, active=True)),
        "bare": jastrow.replace(jastrow_three_body_data=None),
        "cp": ham.coulomb_potential_data,
        "r_up": rng.uniform(0.0, 1.0, (geminal.num_electron_up, 3)) @ cell,
        "r_dn": rng.uniform(0.0, 1.0, (geminal.num_electron_dn, 3)) @ cell,
    }


_witness = functools.lru_cache(maxsize=None)(_build_witness)


def _leaves(value) -> list[np.ndarray]:
    """Flatten a wrapper result (array, tuple or nested dict / tuple) into its array leaves."""
    if isinstance(value, dict):
        return [leaf for key in sorted(value) for leaf in _leaves(value[key])]
    if isinstance(value, (tuple, list)):
        return [leaf for part in value for leaf in _leaves(part)]
    return [np.asarray(value)]


@pytest.mark.light
@pytest.mark.parametrize("route", sorted(_VALUE_ROUTES))
@pytest.mark.parametrize("child", _CHILDREN)
def test_free_real_wrapper_inactive_complex_child_j3_equals_j3_free(child, route):
    """A free real wrapper with an inactive J3 returns exactly the J3-free result.

    The wrappers consume the J3 only through the wavefunction (values, kinetic / ECP energies,
    forces, strains and non-J3 parameter derivatives), so an all-zero ``j_matrix`` must give the
    result of the same call with ``jastrow_three_body_data=None``, never a panic for the complex
    children. The J3-free reference is finite and nonzero for every route. ``real_pw`` is
    the control. Each session compares within its own precision mode.
    """
    w = _witness(child)
    gem_kind, call = _VALUE_ROUTES[route]
    geminal = w[gem_kind]
    ref = _leaves(call(geminal, w["bare"], w))
    got = _leaves(call(geminal, w["inactive"], w))
    assert [g.shape for g in got] == [r.shape for r in ref]
    for g in got:
        assert np.all(np.isfinite(g)), route
    assert any(np.any(r != 0.0) for r in ref), route
    for g, r in zip(got, ref, strict=True):
        np.testing.assert_array_equal(g, r, err_msg=f"{child}: {route}")


@pytest.mark.light
@pytest.mark.parametrize("route", sorted(_VALUE_ROUTES))
@pytest.mark.parametrize("child", _COMPLEX_CHILDREN)
def test_free_real_wrapper_active_complex_child_j3_refused(child, route):
    """An ACTIVE complex-child J3 on a free real wrapper raises ``NotImplementedError``, never panics."""
    w = _witness(child)
    gem_kind, call = _VALUE_ROUTES[route]
    with pytest.raises(NotImplementedError, match=_J3_REFUSAL):
        call(w[gem_kind], w["active"], w)


@pytest.mark.light
@pytest.mark.parametrize("activity", ["inactive", "active"])
@pytest.mark.parametrize("route", sorted(_JMATRIX_ROUTES))
@pytest.mark.parametrize("child", _COMPLEX_CHILDREN)
def test_free_real_j3_jmatrix_derivative_complex_child_refused(child, route, activity):
    """The free real J3 ``j_matrix`` derivatives refuse a complex-child J3, inactive or active.

    The derivative of an inactive J3 is nonzero (products of J3 orbital values the real parser
    cannot evaluate), so neither a truncated orbital nor an omitted block is correct: the wrappers
    raise ``NotImplementedError`` (the hand-off to the JAX route) instead of panicking.
    """
    w = _witness(child)
    with pytest.raises(NotImplementedError, match=_J3_REFUSAL):
        _JMATRIX_ROUTES[route](w["SD"], w[activity], w)


@pytest.mark.light
def test_free_real_j3_jmatrix_derivative_inactive_representable_j3_matches_xla():
    """An inactive J3 the real parser represents keeps its nonzero ``j_matrix`` derivative.

    Control for the refusal above (the ``real_pw`` child): ``grad_ln_Psi_jastrow_3b_jmatrix`` of the
    all-zero ``j_matrix`` equals the XLA ``jax.grad`` of ``compute_Jastrow_three_body`` and is
    nonzero, and the kinetic / ECP ``j_matrix`` wrappers return finite arrays of the
    ``j_matrix`` shape. Mixed session: the ``grad_ln_Psi_jastrow_3b_jmatrix`` block (the one judged
    against AD in full) stays within the ``parameter_derivative`` fidelity budget of the same route
    rebuilt in full precision, and an out-of-budget change of the mixed block is rejected.
    """
    w = _witness("real_pw")
    j3 = w["inactive"].jastrow_three_body_data
    shape = np.asarray(j3.j_matrix).shape
    outs = {name: _leaves(call(w["SD"], w["inactive"], w)) for name, call in _JMATRIX_ROUTES.items()}
    for name, out in outs.items():
        assert len(out) == 1 and out[0].shape == shape, name
        assert np.all(np.isfinite(out[0])), name
    if is_mixed_precision_enabled():

        def evaluate(case: dict) -> np.ndarray:
            return _leaves(_JMATRIX_ROUTES["grad_ln_Psi_jastrow_3b_jmatrix"](case["SD"], case["inactive"], case))[0]

        mixed_block, full_block = mixed_vs_full(
            lambda: _build_witness("real_pw"),
            evaluate,
            configure_native=jqmc_rs.configure_native_precision,
            native_policy_snapshot=jqmc_rs.native_precision_support,
        )
        atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
        assert np.all(np.isfinite(mixed_block)) and np.all(np.isfinite(full_block))
        assert np.abs(full_block).max() > atol
        assert np.abs(mixed_block - full_block).max() <= atol + rtol * np.abs(full_block).max()
        mutated = out_of_budget_mixed_block(mixed_block, full_block, atol=atol, rtol=rtol)
        assert not np.abs(mutated - full_block).max() <= atol + rtol * np.abs(full_block).max()
        return
    got = outs["grad_ln_Psi_jastrow_3b_jmatrix"][0]
    ref = np.asarray(jax.grad(lambda d: compute_Jastrow_three_body(d, w["r_up"], w["r_dn"]))(j3).j_matrix)
    assert np.any(ref != 0.0)
    atol, rtol = get_tolerance_xla_rust_consistency(("jastrow_eval",), level="medium")
    np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("child", _CHILDREN)
def test_free_real_j3_grads_and_laplacian_inactive_j3_zero_active_complex_refused(child):
    """``compute_grads_and_laplacian_jastrow_three_body``: inactive J3 -> exact zeros; active complex -> refused.

    An all-zero ``j_matrix`` makes the J3 identically zero, so its electron gradients and
    Laplacians are exactly 0 (the J3-free value) with shapes ``(n_up, 3)``, ``(n_dn, 3)``,
    ``(n_up,)``, ``(n_dn,)``. An active complex-child J3 raises ``NotImplementedError``. Before the
    fix both panicked for the complex children.
    """
    w = _witness(child)
    n_up, n_dn = w["r_up"].shape[0], w["r_dn"].shape[0]
    got = _leaves(jqmc_rs.compute_grads_and_laplacian_jastrow_three_body(w["inactive"], w["r_up"], w["r_dn"]))
    assert [g.shape for g in got] == [(n_up, 3), (n_dn, 3), (n_up,), (n_dn,)]
    for g in got:
        assert np.all(np.isfinite(g))
        np.testing.assert_array_equal(g, np.zeros_like(g))
    if child in _COMPLEX_CHILDREN:
        with pytest.raises(NotImplementedError, match=_J3_REFUSAL):
            jqmc_rs.compute_grads_and_laplacian_jastrow_three_body(w["active"], w["r_up"], w["r_dn"])

"""Rust-vs-XLA ECP non-local force/stress parity with an electron ON a quadrature direction.

The ECP non-local force and stress need the Legendre derivative ``P_l'(cos theta)``.  The
form ``l (P_{l-1}(x) - x P_l(x)) / (1 - x^2)`` is ``0/0 = NaN`` at ``x = +-1``.  ``x = +-1`` is reached deterministically in GFMC right
after every accepted T-move / DLT-move, because the moved electron then sits exactly on a
direction of the current quadrature mesh.  Both backends use the tabulated derivative
polynomial (``coulomb_potential::legendre_deriv`` / ``ecp_force._legendre_derivative_tablated``).

Each row evaluates a native entry point and its XLA production twin at

* ``generic``: electrons scattered around the nuclei, a generic mesh rotation;
* ``on_mesh``: the same, but one up electron placed at ``R_I + r v`` with ``v`` the first
  mesh direction of the current rotation, so ``cos theta`` is exactly ``+1`` for that
  direction and exactly ``-1`` for its antipode (``icosahedron_sym_mesh_Nv12`` points 0 and 1
  are ``(0, 0, +-1)``; the rotation is a signed permutation, so ``v`` stays exact).

Both sides must be finite (NaN never counts as agreement: ``equal_nan=False``), then agree
at the medium XLA-Rust tier.  Systems carry a live non-local ``l >= 1`` channel (for
``l = 0`` the derivative term is identically zero): the committed Cl2 ccECP (non-local s and
p on both atoms), and the committed Li Gamma / k025-twist cells with a non-local p projector
added (the Li ccECP is s-only non-local; its local channel is kept, retagged ``l = 2``).
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

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import jqmc_rs
from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc.atomic_orbital import is_complex_aos
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
from jqmc.derivatives import assemble
from jqmc.derivatives import gfmc as xla_gfmc
from jqmc.jqmc_mcmc import _geminal_inv_batched

from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]

_W = np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=np.float64)
_GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=np.float64)
_ALAT = 0.30
_NN = 1
_ZONES = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb", "ewald", "jastrow_eval", "jastrow_grad_lap")

# Signed permutation (x, y, z) -> (y, z, x): an exact rotation, so the rotated
# mesh direction gp[0] @ RT = (1, 0, 0) is exact and cos theta hits +-1 exactly.
_RT_EXACT = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])


def _rt_generic() -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(11).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def _cl2():
    return _build_ham_for_cell("Cl2_ecp_ccpvtz_cart.h5", True, None, "SD")


def _with_live_p(ham):
    """Li ccECP (s non-local, p local) -> s and p non-local, local retagged l = 2.

    The added p projector reuses the s projector's radial terms.  The local channel keeps
    its radial terms; only its tag moves from ``l_max + 1 = 1`` to ``2``.
    """
    cpd = ham.coulomb_potential_data
    ang = np.asarray(cpd.ang_moms, dtype=int)
    lmax1 = np.asarray(cpd.max_ang_mom_plus_1, dtype=int)
    nuc = np.asarray(cpd.nucleus_index, dtype=int)
    assert np.all(lmax1 == 1), "fixture precondition: Li ccECP with an s-only non-local part"
    s = ang == 0
    new_ang = np.where(ang == 1, 2, ang).tolist() + [1] * int(s.sum())
    cat = lambda f: list(getattr(cpd, f)) + [getattr(cpd, f)[i] for i in np.flatnonzero(s)]  # noqa: E731
    new = cpd.replace(
        max_ang_mom_plus_1=tuple(int(v) for v in lmax1 + 1),
        num_ecps=len(new_ang),
        ang_moms=tuple(new_ang),
        nucleus_index=tuple(int(v) for v in cat("nucleus_index")),
        exponents=tuple(float(v) for v in cat("exponents")),
        coefficients=tuple(float(v) for v in cat("coefficients")),
        powers=tuple(int(v) for v in cat("powers")),
    )
    new.sanity_check()
    assert set(nuc.tolist()) == set(np.asarray(new.nucleus_index)[np.asarray(new.ang_moms) == 1].tolist())
    return ham.replace(coulomb_potential_data=new)


_SYSTEMS = {
    "cl2_mol": _cl2,
    "li_gamma_p": lambda: _with_live_p(rs_ham("pbc_gamma_sphepgto_j12")),
    "li_twist_p": lambda: _with_live_p(rs_ham("twist_sphepgto_j12")),
}


def _assert_live_high_l(ham):
    cpd = ham.coulomb_potential_data
    ang = np.asarray(cpd.ang_moms, dtype=int)
    nuc = np.asarray(cpd.nucleus_index, dtype=int)
    lmax1 = np.asarray(cpd.max_ang_mom_plus_1, dtype=int)
    live = (ang < lmax1[nuc]) & (ang >= 1) & (np.asarray(cpd.coefficients, dtype=float) != 0.0)
    assert live.any(), "fixture precondition: a non-zero non-local l >= 1 projector"


def _configuration(ham, kind):
    """(r_up, r_dn, RT). ``on_mesh`` puts up electron 0 on the first rotated mesh direction."""
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(5)
    ru = R[np.arange(nu) % len(R)] + rng.normal(scale=0.7, size=(nu, 3))
    rd = R[np.arange(nd) % len(R)] + rng.normal(scale=0.7, size=(nd, 3))
    if kind == "generic":
        RT = _rt_generic()
    else:
        RT = _RT_EXACT
        v = _GP[0] @ RT
        assert np.array_equal(np.abs(v), [1.0, 0.0, 0.0]) and np.array_equal(np.abs(_GP[1] @ RT)[0], 1.0)
        ru[0] = R[0] + 0.75 * v
    return ru, rd, RT


def _check(rust, xla, label):
    rust, xla = np.asarray(rust, dtype=np.float64), np.asarray(xla, dtype=np.float64)
    assert np.all(np.isfinite(rust)), f"{label}: Rust not finite: {rust}"
    assert np.all(np.isfinite(xla)), f"{label}: XLA not finite: {xla}"
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="medium")
    np.testing.assert_allclose(rust, xla, atol=atol, rtol=rtol, equal_nan=False, err_msg=label)


def _is_complex(ham):
    orb_up = ham.wavefunction_data.geminal_data.orb_data_up_spin
    return bool(is_complex_aos(orb_up.aos_data if hasattr(orb_up, "aos_data") else orb_up))


def _rust_force(ham, ru, rd, RT, dlt):
    wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
    args = (wf.geminal_data, wf.jastrow_data, cp, ru, rd, _ALAT, RT, _W, _GP, _NN, dlt)
    if not bool(ham.structure_data.pbc_flag):
        return jqmc_rs._gfmc_grad_e_L_ecp(*args)
    if _is_complex(ham):
        return jqmc_rs._gfmc_grad_e_L_ecp_pbc_complex(*args)
    return jqmc_rs._gfmc_grad_e_L_ecp_pbc(*args)


def _rust_stress(ham, ru, rd, RT, dlt):
    wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
    args = (wf.geminal_data, wf.jastrow_data, cp, ru, rd, _ALAT, RT, _W, _GP, _NN, dlt)
    if not bool(ham.structure_data.pbc_flag):
        return jqmc_rs._gfmc_stress_ecp_withj(*args)
    if _is_complex(ham):
        return jqmc_rs._gfmc_stress_ecp_pbc_complex(*args)
    return jqmc_rs._gfmc_stress_ecp_pbc_withj(*args)


_MOVES = ("tmove", "dltmove")
_KINDS = ("generic", "on_mesh")


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("non_local_move", _MOVES)
@pytest.mark.parametrize("system", sorted(_SYSTEMS))
def test_gfmc_ecp_force_rust_vs_xla_on_mesh(system, non_local_move, kind):
    """GFMC HF force d e_L / d{R, r_up, r_dn}: native kernel == XLA, finite on a mesh direction."""
    ham = _SYSTEMS[system]()
    _assert_live_high_l(ham)
    ru, rd, RT = _configuration(ham, kind)
    got = _rust_force(ham, ru, rd, RT, non_local_move == "dltmove")
    ref = xla_gfmc.grad_e_L_t_wrt_positions(ham, ru, rd, RT, _ALAT, non_local_move=non_local_move)
    for name, g, r in zip(("R", "r_up", "r_dn"), got, ref, strict=True):
        _check(g, r, f"{system}/{non_local_move}/{kind}: force {name}")


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("non_local_move", _MOVES)
@pytest.mark.parametrize("system", sorted(_SYSTEMS))
def test_gfmc_ecp_stress_rust_vs_xla_on_mesh(system, non_local_move, kind):
    """GFMC HF stress d e_L / d eps: native kernel == XLA, finite on a mesh direction."""
    ham = _SYSTEMS[system]()
    _assert_live_high_l(ham)
    ru, rd, RT = _configuration(ham, kind)
    got = _rust_stress(ham, ru, rd, RT, non_local_move == "dltmove")
    ref = xla_gfmc.grad_e_L_t_wrt_strain(ham, ru, rd, RT, _ALAT, non_local_move=non_local_move)
    _check(got, ref, f"{system}/{non_local_move}/{kind}: stress")


@pytest.mark.parametrize("kind", _KINDS)
def test_vmc_ecp_force_rust_vs_xla_on_mesh(kind):
    """MCMC HF force d e_L / d{R, r_up, r_dn} on Cl2: ``vmc_grad_e_L_ecp`` == ``assemble.compute_grad_e_L``."""
    ham = _cl2()
    _assert_live_high_l(ham)
    ru, rd, RT = _configuration(ham, kind)
    wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
    _, ginv, _, _ = _geminal_inv_batched(wf.geminal_data, jnp.asarray(ru)[None], jnp.asarray(rd)[None])
    ginv = np.asarray(ginv[0], dtype=np.float64)
    got = jqmc_rs.vmc_grad_e_L_ecp(wf.geminal_data, wf.jastrow_data, cp, ru, rd, _W, _GP @ RT, ginv)
    ref_R = assemble.compute_grad_e_L(ham, jnp.asarray(ru), jnp.asarray(rd), RT, wrt="R", geminal_inv=jnp.asarray(ginv))
    ref_u, ref_d = assemble.compute_grad_e_L(ham, jnp.asarray(ru), jnp.asarray(rd), RT, wrt="r", geminal_inv=jnp.asarray(ginv))
    for name, g, r in zip(("R", "r_up", "r_dn"), got, (ref_R, ref_u, ref_d), strict=True):
        _check(g, r, f"cl2_mol/{kind}: MCMC force {name}")


@pytest.mark.parametrize("non_local_move", _MOVES)
def test_gfmc_ecp_native_force_stress_keep_mixed_fidelity_on_mesh(non_local_move):
    """Native GFMC ECP HF force and stress on Cl2 (``on_mesh``) keep Mixed-versus-Full fidelity.

    Without a supplied inverse the native ECP mesh base must assemble G from the fused
    value/grad/lap value table that the kinetic forward inverts, not from the standalone
    fp32-arithmetic value table: otherwise G^{-1} and the fp64 AO gradients describe
    two different geminal matrices and the stress loses about 3e-2 in mixed.  One full
    physical source is serialized and reloaded under each policy, with the native precision
    policy switched alongside; each output array is one homogeneous block under the existing
    ``parameter_derivative`` budget, with an isolated excessive-error control.
    """
    from jqmc._setting import get_tolerance_mixed_fidelity
    from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block
    from tests.test_rs_kernel._native_precision_fidelity import full_source_hamiltonian_factory

    dlt = non_local_move == "dltmove"

    def native(ham):
        _assert_live_high_l(ham)
        ru, rd, RT = _configuration(ham, "on_mesh")
        force_R, force_up, force_dn = _rust_force(ham, ru, rd, RT, dlt)
        return {
            "nuclear": np.asarray(force_R, dtype=np.float64),
            "electron-up": np.asarray(force_up, dtype=np.float64),
            "electron-down": np.asarray(force_dn, dtype=np.float64),
            "strain": np.asarray(_rust_stress(ham, ru, rd, RT, dlt), dtype=np.float64),
        }

    mixed, full = mixed_vs_full(
        full_source_hamiltonian_factory(_cl2),
        native,
        configure_native=jqmc_rs.configure_native_precision,
        native_policy_snapshot=jqmc_rs.native_precision_support,
    )
    assert set(mixed) == set(full) == {"nuclear", "electron-up", "electron-down", "strain"}
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    blocks = mixed_fidelity_blocks(mixed, full)
    assert len(blocks) == 4
    for mixed_block, full_block in blocks:
        assert np.max(np.abs(mixed_block - full_block)) <= atol + rtol * np.max(np.abs(full_block)), (
            f"{non_local_move}: {np.max(np.abs(mixed_block - full_block)):.3e} > {atol + rtol * np.max(np.abs(full_block)):.3e}"
        )
        changed = out_of_budget_mixed_block(mixed_block, full_block, atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            assert np.max(np.abs(changed - full_block)) <= atol + rtol * np.max(np.abs(full_block))

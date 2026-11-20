"""Core-level parity: periodic bare-AE GFMC HF force/stress.

A periodic (PBC) run with ``ecp_flag=False`` (bare all-electron) is force/stress-servable
by the native GFMC chunk: ``_resolve_native_gfmc_chunk`` has no PBC/ECP family gate.
The periodic HF-force/stress Rust CORES the chunk folds in
(``_gfmc_grad_e_L_ecp_pbc`` / ``_gfmc_stress_ecp_pbc_{jnone,withj}``, the exact
kernels ``run_gfmc_projection_{t,n}`` use) are not ECP-specific: they take generic
local/non-local ECP arrays sourced from ``coulomb_potential_data``, and
``Coulomb_potential_data._nucleus_index_local_part`` / ``_exponents_local_part`` / ... /
``_effective_charges`` degrade to EMPTY arrays (ECP contribution identically zero) /
bare atomic-number charges when ``ecp_flag=False``.

This file calls these cores directly on bare-AE periodic hamiltonians (jnone / J12 /
DTN, force AND stress) and requires them to match the XLA analytic S'-reduction oracle
(``jqmc.derivatives.gfmc.grad_e_L_t_wrt_{positions,strain}``; measured ~1e-13..1e-14).
The DRIVER-LEVEL pin that the dispatch engages the native chunk on a periodic bare-AE
force run lives in ``test_gfmc_chain_modes.py::test_gfmc_native_chunk_gate_matrix``. A
fixture (twist, MPC, non-pGTO stress leaf) that diverges is a genuinely missing kernel
and needs its own gate, not a bare-AE family gate.
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

import jax.numpy as jnp

import jqmc_rs

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc._setting import ATOL_FLOOR_MEDIUM, RTOL_FLOOR_MEDIUM
from jqmc.derivatives import gfmc as dgfmc
from jqmc.jqmc_mcmc import _geminal_inv_batched, _NV_BASE_MESH, Nv_default

from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell

pytestmark = [pytest.mark.with_chdir]

_W = np.asarray(_NV_BASE_MESH[Nv_default].weights, float)
_GP = np.asarray(_NV_BASE_MESH[Nv_default].grid_points, float)
_ZONES = ("ao_eval", "mo_eval", "mo_grad", "det_grad_lap", "wf_kinetic", "coulomb", "ewald")

# (h5 stem, with_jastrow, jastrow spec) -- all periodic (PBC) bare-AE (ecp_flag=False)
# fixtures. jnone / J12 / DTN span the
# jnone + withJ core signatures (``_gfmc_stress_ecp_pbc_{jnone,withj}``) and the DTN
# leg the withJ stress folds in.
_PATTERNS = {
    "pbc_ae_sphe_jnone": ("H2_pbc_ae_sphe.h5", False, None),
    "pbc_ae_cart_j12": ("H4_pbc_ae_cart.h5", True, None),
    "pbc_ae_cart_jdtn": ("H4_pbc_ae_cart.h5", True, {"dtn": True}),
}


def _ham(pattern):
    stem, with_jastrow, spec = _PATTERNS[pattern]
    return _build_ham_for_cell(stem, with_jastrow, spec, "SD")


def _config(ham, seed=2, scale=0.6):
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    ru = R[rng.integers(0, len(R), nu)] + rng.normal(scale=scale, size=(nu, 3))
    rd = R[rng.integers(0, len(R), nd)] + rng.normal(scale=scale, size=(nd, 3))
    return np.ascontiguousarray(ru), np.ascontiguousarray(rd)


def _ginv(ham, ru, rd):
    _, g, _, _ = _geminal_inv_batched(ham.wavefunction_data.geminal_data, jnp.asarray(ru)[None], jnp.asarray(rd)[None])
    return np.asarray(g[0])


def _tol():
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="medium")
    return max(atol, ATOL_FLOOR_MEDIUM), max(rtol, RTOL_FLOOR_MEDIUM)


def _assert_ecp_flag_false(ham):
    assert not bool(ham.coulomb_potential_data.ecp_flag), "fixture must be bare-AE (ecp_flag=False)"
    assert bool(ham.structure_data.pbc_flag), "fixture must be periodic"


@pytest.mark.light
@pytest.mark.parametrize("pattern", sorted(_PATTERNS))
def test_ungate_b2_pbc_ae_force_green(pattern):
    """The native "ECP" periodic HF-force core, called directly on a bare-AE
    periodic hamiltonian, matches the XLA S'-reduction oracle for GFMC_t (see the
    module docstring)."""
    ham = _ham(pattern)
    _assert_ecp_flag_false(ham)
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _config(ham)
    alat = 0.3
    RT = np.eye(3)

    n_R, n_ru, n_rd = jqmc_rs._gfmc_grad_e_L_ecp_pbc(gd, jd, cp, ru, rd, alat, RT, _W, _GP, 1, False)
    n_R, n_ru, n_rd = np.asarray(n_R), np.asarray(n_ru), np.asarray(n_rd)

    ginv = _ginv(ham, ru, rd)
    x_R, x_ru, x_rd = dgfmc.grad_e_L_t_wrt_positions(
        ham, jnp.asarray(ru), jnp.asarray(rd), jnp.asarray(RT), alat, geminal_inv=jnp.asarray(ginv)
    )
    x_R, x_ru, x_rd = np.asarray(x_R), np.asarray(x_ru), np.asarray(x_rd)

    atol, rtol = _tol()
    np.testing.assert_allclose(n_R, x_R, atol=atol, rtol=rtol, err_msg=f"{pattern} PBC bare-AE HF force dR")
    np.testing.assert_allclose(n_ru, x_ru, atol=atol, rtol=rtol, err_msg=f"{pattern} PBC bare-AE HF force dr_up")
    np.testing.assert_allclose(n_rd, x_rd, atol=atol, rtol=rtol, err_msg=f"{pattern} PBC bare-AE HF force dr_dn")


@pytest.mark.light
@pytest.mark.parametrize("pattern", sorted(_PATTERNS))
def test_ungate_b2_pbc_ae_stress_green(pattern):
    """The native "ECP" periodic HF-stress cores, called directly on a bare-AE
    periodic hamiltonian, match the XLA S'-reduction oracle for GFMC_t (see the
    module docstring)."""
    ham = _ham(pattern)
    _assert_ecp_flag_false(ham)
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _config(ham)
    alat = 0.3
    RT = np.eye(3)
    with_jastrow = _PATTERNS[pattern][1]

    if with_jastrow:
        n_s = jqmc_rs._gfmc_stress_ecp_pbc_withj(gd, jd, cp, ru, rd, alat, RT, _W, _GP, 1, False)
    else:
        n_s = jqmc_rs._gfmc_stress_ecp_pbc_jnone(gd, cp, ru, rd, alat, RT, _W, _GP, 1)
    n_s = np.asarray(n_s)

    ginv = _ginv(ham, ru, rd)
    x_s = np.asarray(
        dgfmc.grad_e_L_t_wrt_strain(ham, jnp.asarray(ru), jnp.asarray(rd), jnp.asarray(RT), alat, geminal_inv=jnp.asarray(ginv))
    )

    atol, rtol = _tol()
    assert n_s.shape == x_s.shape == (3, 3)
    np.testing.assert_allclose(n_s, x_s, atol=atol, rtol=rtol, err_msg=f"{pattern} PBC bare-AE HF stress")

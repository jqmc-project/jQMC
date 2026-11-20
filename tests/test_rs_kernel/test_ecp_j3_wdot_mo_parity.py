"""Native ECP-mesh J3 contraction with a non-identity MO-wrapped J3 orbital (#368).

The production real MCMC chunk (``run_energy_chunk_batch`` with ``native_rng``)
evaluates the ECP non-local J3 ratios through ``coulomb_potential::ecp_j3_wdot_all``:
one electron-fixed raw-AO weight row per electron, obtained by folding the J3
single-move exponent through the MO coefficient matrix ``C`` of the J3 orbital.
The molecular Rust fixtures only reach this fold with a raw-AO J3 orbital
(``mo_coefficients = None``) or with an ECP that has no live non-local channel
(H2), so a wrong orientation of ``C`` was invisible there.

This module drives the production chunk on water ccECP (live O non-local
channels) with a J3 orbital wrapped in ``MOs_data`` whose ``C`` is either a
random orthogonal square matrix or a random rectangular matrix. Proposal
normals and rotated quadrature meshes are injected through the test-only seam
``test_injected_inputs`` (production native arithmetic, no alternate route), and
``want_steps`` returns the post-step walker configurations. Every stored native
``e_L`` must equal an independent XLA ``compute_local_energy`` recomputed from
scratch at the same configuration with the same mesh rotation. The route
predicate counters prove that every native local energy took the J3 w-dot fold.

The open-shell water-cation row retains four up and three down occupied MOs.
It also exercises the determinant w-dot fold's unpaired columns (#412), which
closed shells and single-spin ECP atoms cannot distinguish. Its XLA non-local
reference is checked against full determinant recomputation on the same mesh.
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jqmc_rs  # noqa: E402

from jqmc._precision import get_tolerance_min, precision_policy_snapshot  # noqa: E402
from jqmc._setting import Nv_default  # noqa: E402
from jqmc.coulomb_potential import (  # noqa: E402
    _compute_ecp_non_local_parts_nearest_neighbors_debug,
    compute_ecp_non_local_parts_nearest_neighbors_fast_update,
)
from jqmc.determinant import SD_data  # noqa: E402
from jqmc.hamiltonians import compute_local_energy  # noqa: E402
from jqmc.jastrow_factor import Jastrow_three_body_data  # noqa: E402
from jqmc.jqmc_mcmc import _NV_BASE_MESH, _geminal_inv_batched  # noqa: E402
from jqmc.molecular_orbital import MOs_data  # noqa: E402
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell  # noqa: E402
from tests.test_rs_kernel.test_chain_modes import _assert_value_parity  # noqa: E402

pytestmark = [pytest.mark.with_chdir]

_MESH = _NV_BASE_MESH[Nv_default]
_W = np.asarray(_MESH.weights, float)
_GP = np.asarray(_MESH.grid_points, float)
_NW = 2
_STEPS = 3
_SPREAD = 0.5
# The determinant w-dot fold has a distinct unpaired-column contraction.
_DET_WDOT_ENGAGED, _DET_WDOT_DECLINED = 0, 1
# ``route_predicate_counters()`` index of the (engaged, declined) ``ecp_j3_wdot_ok`` counts.
_J3_WDOT_ENGAGED, _J3_WDOT_DECLINED = 2, 3


def _water_mo_j3_ham(n_mo):
    """Water ccECP SD + J1/J2 + an active J3 on a non-identity MO wrapper of its GTO basis."""
    ham = _build_ham_for_cell("water_ccecp_ccpvqz.h5", True, {"j3": "sphe_gto"}, "SD")
    jd = ham.wavefunction_data.jastrow_data
    aos = jd.jastrow_three_body_data.orb_data
    n_ao = int(aos.num_ao)
    rng = np.random.default_rng(368)
    if n_mo == n_ao:
        coefficients, _ = np.linalg.qr(rng.normal(size=(n_ao, n_ao)))
    else:
        coefficients = rng.normal(size=(n_mo, n_ao)) / np.sqrt(n_ao)
    j_matrix = 0.05 * rng.normal(size=(n_mo, n_mo + 1))
    j_matrix[:, :n_mo] = 0.5 * (j_matrix[:, :n_mo] + j_matrix[:, :n_mo].T)
    j3 = Jastrow_three_body_data(
        orb_data=MOs_data(num_mo=n_mo, aos_data=aos, mo_coefficients=coefficients),
        j_matrix=j_matrix,
    )
    ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd.replace(jastrow_three_body_data=j3)))
    ham.sanity_check()
    # Non-vacuity of the input class: molecular, live non-local ECP channel, and a
    # J3 MO matrix that is neither the identity nor symmetric (so any transposed
    # or dropped fold of C changes the energy).
    assert not ham.structure_data.pbc_flag
    cp = ham.coulomb_potential_data
    assert cp.ecp_flag
    l_local = np.asarray(cp.max_ang_mom_plus_1)[np.asarray(cp.nucleus_index)]
    live_non_local = (np.asarray(cp.ang_moms) != l_local) & (np.asarray(cp.coefficients) != 0.0)
    assert np.any(live_non_local), "the fixture needs a live non-local ECP channel (H2 ccECP has none)"
    assert np.any(j_matrix != 0)
    if n_mo == n_ao:
        assert not np.allclose(coefficients, coefficients.T)
    assert not np.allclose(coefficients[:, :n_mo], np.eye(n_mo))
    return ham


def _rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _initial_walkers(ham):
    gd = ham.wavefunction_data.geminal_data
    positions = np.asarray(ham.structure_data.positions, float)
    centre = positions[int(np.argmax(np.asarray(ham.structure_data.atomic_numbers)))]  # the O nucleus
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(7)
    r_up = centre + rng.normal(scale=_SPREAD, size=(_NW, nu, 3))
    r_dn = centre + rng.normal(scale=_SPREAD, size=(_NW, nd, 3))
    return r_up, r_dn


def _run_native_chunk(ham, rotations, normals):
    """One production real energy chunk with injected normals/rotated meshes; returns (e_L, r_up_steps, r_dn_steps)."""
    gd = ham.wavefunction_data.geminal_data
    cp = ham.coulomb_potential_data
    ctx = jqmc_rs.build_obs_ctx(
        gd,
        ham.wavefunction_data.jastrow_data,
        cp,
        False,
        False,
        False,
        precision_policy=precision_policy_snapshot(),
    )
    positions = np.asarray(ham.structure_data.positions, float)
    charges = np.asarray(ham.structure_data.atomic_numbers, float) - np.asarray(cp.z_cores, float)
    r_up, r_dn = _initial_walkers(ham)
    g, gi, _, _ = _geminal_inv_batched(gd, jnp.asarray(r_up), jnp.asarray(r_dn))
    keys = np.asarray([[34356 + w, 778899 + 3 * w] for w in range(_NW)], dtype=np.uint32)
    grids = np.ascontiguousarray(np.stack([np.stack([_GP @ rot for rot in rotations])] * _NW))
    jqmc_rs.gfmc_route_counters_reset()
    out = ctx.run_energy_chunk_batch(
        positions,
        charges,
        r_up,
        r_dn,
        np.asarray(g, float),
        np.asarray(gi, float),
        2.0,
        1.0e-6,
        keys,
        np.ascontiguousarray(normals),
        grids,
        _W,
        int(gd.num_electron_up + gd.num_electron_dn),
        None,
        0,
        native_rng=True,
        random_mesh=True,
        base_grid_points=_GP,
        n_steps_fast=_STEPS,
        want_steps=True,
        test_injected_inputs=True,
    )
    counters = list(jqmc_rs.route_predicate_counters())
    assert (counters[_DET_WDOT_ENGAGED], counters[_DET_WDOT_DECLINED]) == (2 * _NW * _STEPS, 0), (
        f"the native determinant w-dot fold did not run on every local energy: {counters}"
    )
    # One ``ecp_j3_wdot_ok`` decision per spin per native local energy, all engaged.
    assert (counters[_J3_WDOT_ENGAGED], counters[_J3_WDOT_DECLINED]) == (2 * _NW * _STEPS, 0), (
        f"the native J3 w-dot fold did not run on every local energy: {counters}"
    )
    e_l = np.asarray(out["e_L"], float)
    r_up_steps = np.asarray(out["r_up_steps"], float)
    r_dn_steps = np.asarray(out["r_dn_steps"], float)
    assert e_l.shape == (_NW, _STEPS) and np.all(np.isfinite(e_l))
    assert r_up_steps.shape[:2] == (_NW, _STEPS) and r_dn_steps.shape[:2] == (_NW, _STEPS)
    assert int(np.sum(np.asarray(out["accepted"]))) > 0, "no proposal was accepted; the chunk did not move"
    return e_l, r_up_steps, r_dn_steps


@pytest.mark.heavy
@pytest.mark.parametrize("shape", ["square-orthogonal", "rectangular", "rectangular-open-shell-4up-3dn"])
def test_ecp_mesh_j3_mo_wrapped_chunk_energy_matches_xla(shape, monkeypatch):
    """Native chunk e_L with a non-identity MO-wrapped J3 equals XLA recomputed at every stored step."""
    n_ao = 114  # water ccECP cc-pVQZ spherical GTO basis (asserted below)
    ham = _water_mo_j3_ham(n_ao if shape == "square-orthogonal" else 40)
    assert int(ham.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data.aos_data.num_ao) == n_ao
    open_shell = shape == "rectangular-open-shell-4up-3dn"
    if open_shell:
        # Remove one occupied down-spin MO from the committed neutral-water SD;
        # retain its basis, ECP and Jastrow parameters without fitting the oracle.
        gd = ham.wavefunction_data.geminal_data
        down = gd.orb_data_dn_spin
        geminal = SD_data(
            num_electron_up=4,
            num_electron_dn=3,
            orb_data_up_spin=gd.orb_data_up_spin,
            orb_data_dn_spin=MOs_data(num_mo=3, aos_data=down.aos_data, mo_coefficients=down.mo_coefficients[:3]),
            time_reversal=True,
        )
        ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=geminal))
        ham.sanity_check()
        assert geminal.num_electron_up > geminal.num_electron_dn > 0

    rng = np.random.default_rng(20261104)
    rotations = [_rotation(rng) for _ in range(_STEPS)]
    gd = ham.wavefunction_data.geminal_data
    normals = rng.standard_normal((_NW, _STEPS, int(gd.num_electron_up + gd.num_electron_dn)))

    monkeypatch.delenv("JQMC_FLAG_RUST_ON_CPU", raising=False)
    e_native, r_up_steps, r_dn_steps = _run_native_chunk(ham, rotations, normals)

    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
    e_xla = np.asarray(
        [
            [
                float(
                    compute_local_energy(
                        ham, jnp.asarray(r_up_steps[w, k]), jnp.asarray(r_dn_steps[w, k]), jnp.asarray(rotations[k])
                    )
                )
                for k in range(_STEPS)
            ]
            for w in range(_NW)
        ]
    )
    assert np.all(np.isfinite(e_xla))
    if open_shell:
        # Anchor the fast-update XLA reference to a full determinant at every
        # quadrature point for one stored configuration (NN=1: the O ECP).
        ru, rd = r_up_steps[0, 0], r_dn_steps[0, 0]
        _, inverse, _, _ = _geminal_inv_batched(gd, jnp.asarray(ru[None]), jnp.asarray(rd[None]))
        v_fast = compute_ecp_non_local_parts_nearest_neighbors_fast_update(
            ham.coulomb_potential_data,
            ham.wavefunction_data,
            jnp.asarray(ru),
            jnp.asarray(rd),
            jnp.asarray(rotations[0]),
            inverse[0],
            NN=1,
            Nv=Nv_default,
        )[3]
        v_debug = _compute_ecp_non_local_parts_nearest_neighbors_debug(
            ham.coulomb_potential_data,
            ham.wavefunction_data,
            ru,
            rd,
            rotations[0],
            NN=1,
            Nv=Nv_default,
        )[3]
        atol, rtol = get_tolerance_min(["coulomb", "jastrow_eval", "jastrow_ratio", "det_eval", "det_ratio"], "strict")
        assert np.isfinite(v_debug) and abs(v_debug) > atol, "the open-shell non-local ECP reference must be live"
        np.testing.assert_allclose(v_fast, v_debug, atol=atol, rtol=rtol)
    # The production chunk carries its Sherman--Morrison and J3 ratio states across
    # steps while XLA recomputes from scratch: the reviewed MCMC driver envelope.
    _assert_value_parity(e_native, e_xla, f"native ECP-mesh J3 w-dot fold through a {shape} MO matrix")

"""Native periodic real-Gamma GFMC ECP force and stress keep Mixed-versus-Full fidelity (#758).

The native GFMC periodic ECP non-local force and stress (``_gfmc_grad_e_L_ecp_pbc``,
``_gfmc_stress_ecp_pbc_withj``) run without a supplied geminal inverse.  Their geminal base
(``gem_forward_pbc``, the base of ``MeshDetFastPbc``) contracts ``G^{-1}`` with fp64 orbital
derivative tables, so ``G``, its inverse and every moved-electron candidate column of the
rank-one det ratio must come from one value table: the fused convention (fp64 body, completed
AO value rounded to ``ao_eval``), never the zone-local fp32-arithmetic value kernels.

* ``sphe`` (the committed ``silicon.h5`` spherical AOs): the exact base, whose candidate
  columns used the standalone fp32 value kernel against a fused-table ``G``;
* ``cart`` (the same wavefunction converted exactly to Cartesian AOs with
  ``Geminal_data.to_cartesian``): the FAST screened base, whose ``G`` came from the fp32 sweep
  values against fp64 FAST derivative tables.

The Si8 cell (ccECP with non-local s and p, real Gamma, J1+J2) is built and converted in full,
serialized once, and reloaded under each policy with the native precision policy switched
alongside.
Each output array is one homogeneous block under the existing central
``parameter_derivative`` budget, with an isolated excessive-error control.
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

jax.config.update("jax_enable_x64", True)

import jqmc_rs
from jqmc._setting import get_tolerance_mixed_fidelity
from jqmc.atomic_orbital import AOs_cart_pgto_data
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12

from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel._native_precision_fidelity import full_source_hamiltonian_factory

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]

_W = np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=np.float64)
_GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=np.float64)
_ALAT = 0.30
_NN = 1


def _silicon(ao_form):
    ham = _build_ham_for_cell("silicon.h5", True, None, "SD")
    if ao_form == "sphe":
        return ham
    wf = ham.wavefunction_data
    return ham.replace(
        wavefunction_data=wf.replace(geminal_data=wf.geminal_data.to_cartesian(), jastrow_data=wf.jastrow_data.to_cartesian())
    )


# Signed permutation (x, y, z) -> (y, z, x): an exact rotation, so the rotated first mesh
# direction is exactly (1, 0, 0) (the ``on_mesh`` configuration of the Legendre endpoint rows).
_RT_EXACT = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])


def _configuration(ham, kind):
    """Electrons scattered around the nuclei; ``generic``: a generic proper mesh rotation,
    ``on_mesh``: the exact rotation with up electron 0 on its first rotated mesh direction."""
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(5)
    ru = R[np.arange(nu) % len(R)] + rng.normal(scale=0.7, size=(nu, 3))
    rd = R[np.arange(nd) % len(R)] + rng.normal(scale=0.7, size=(nd, 3))
    if kind == "on_mesh":
        ru[0] = R[0] + 0.75 * (_GP[0] @ _RT_EXACT)
        return ru, rd, _RT_EXACT
    q, r = np.linalg.qr(np.random.default_rng(11).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return ru, rd, (q if np.linalg.det(q) > 0 else -q)


@pytest.mark.parametrize("kind", ["generic", "on_mesh"])
@pytest.mark.parametrize("non_local_move", ["tmove", "dltmove"])
@pytest.mark.parametrize("ao_form", ["sphe", "cart"])
def test_gfmc_ecp_native_pbc_force_stress_keep_mixed_fidelity(ao_form, non_local_move, kind):
    """Native real-Gamma GFMC ECP HF force and stress on Si8: mixed against an independent full rebuild."""
    dlt = non_local_move == "dltmove"

    def native(ham):
        assert bool(ham.structure_data.pbc_flag)
        cpd = ham.coulomb_potential_data
        assert cpd.ecp_flag
        ang = np.asarray(cpd.ang_moms, dtype=int)
        lmax1 = np.asarray(cpd.max_ang_mom_plus_1, dtype=int)[np.asarray(cpd.nucleus_index, dtype=int)]
        assert np.any((ang >= 1) & (ang < lmax1)), "fixture precondition: a non-local l >= 1 projector"
        orb_up = ham.wavefunction_data.geminal_data.orb_data_up_spin
        aos = orb_up.aos_data if hasattr(orb_up, "aos_data") else orb_up
        assert isinstance(aos, AOs_cart_pgto_data) == (ao_form == "cart"), type(aos)
        ru, rd, RT = _configuration(ham, kind)
        wf = ham.wavefunction_data
        args = (wf.geminal_data, wf.jastrow_data, cpd, ru, rd, _ALAT, RT, _W, _GP, _NN, dlt)
        force_R, force_up, force_dn = jqmc_rs._gfmc_grad_e_L_ecp_pbc(*args)
        return {
            "nuclear": np.asarray(force_R, dtype=np.float64),
            "electron-up": np.asarray(force_up, dtype=np.float64),
            "electron-down": np.asarray(force_dn, dtype=np.float64),
            "strain": np.asarray(jqmc_rs._gfmc_stress_ecp_pbc_withj(*args), dtype=np.float64),
        }

    mixed, full = mixed_vs_full(
        full_source_hamiltonian_factory(lambda: _silicon(ao_form)),
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
            f"{ao_form} {non_local_move} {kind}: {np.max(np.abs(mixed_block - full_block)):.3e} > {atol + rtol * np.max(np.abs(full_block)):.3e}"
        )
        changed = out_of_budget_mixed_block(mixed_block, full_block, atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            assert np.max(np.abs(changed - full_block)) <= atol + rtol * np.max(np.abs(full_block))

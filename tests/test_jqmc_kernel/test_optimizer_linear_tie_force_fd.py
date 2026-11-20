"""Generalized force of linear-tie (symmetric) blocks vs a physical-direction FD.

``MCMC.get_dln_WF`` projects the stored per-sample derivative of every
``linear_tie`` block (the J3 ``j_matrix`` and the AGP ``lambda_matrix``) onto
its symmetric subspace, so that ``f``, ``S`` and the step respect the tie.
The stored per-entry derivative ``d ln|Psi| / d c_ij`` treats ``c_ij`` and
``c_ji`` as independent and is not symmetric per sample, so without the
projection ``get_gF`` returns the unsymmetrized covariance on the off-diagonal
entries.

The reference never touches the production projection. Per sample it takes a
central finite difference of ``ln|Psi|`` along the PHYSICAL tied direction
``D``: ``E_ij + E_ji`` for an off-diagonal entry of the paired square block,
``E_ii`` on the diagonal, ``E_ij`` on the free last J3 column. The coupling
matrix is rebuilt directly with ``Jastrow_three_body_data.replace`` /
``AGP_data.replace``. The storage unit vector of an off-diagonal entry is half
of ``E_ij + E_ji``, so the expected storage-unit force is
``-2 cov(O_D, e_L) / 2`` there and ``-2 cov(O_D, e_L)`` elsewhere.

The samples are injected into an MCMC in the same way as
``complex_opt_ground_truth_harness._Case.injected_mcmc``.

Accuracy (measured; worst |f_impl - f_ref| over each block, both modes):

* ``get_gF`` with one sample per jackknife bin returns exactly
  ``(1 - 1/(N-1)^2)`` times the plain covariance at unit weights. At N=400
  that is 6.3e-6 relative, at most 2.7e-6 absolute here.
* ``ln|Psi|`` is affine in ``j_matrix``, so the J3 central difference has no
  truncation error at any step. The step 1.0 keeps the mixed-mode fp32
  ``jastrow_eval`` round-off below 3e-8 relative; at 1e-5 that round-off
  alone gives 2e-2 relative error in mixed mode.
* ``ln|Psi|`` is smooth in ``lambda_matrix`` (``det_eval`` stays fp64 in
  mixed mode), and the step 1e-5 adds 2e-9 relative.

The registered FD-vs-analytic pair (atol 1e-5, rtol 5e-5) clears the clean
error by about 4x. Disabling the projection gives errors of 3.9e-3 (J3) and
1.5e-1 (lambda) absolute, 300x and more above the bound.
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
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose

from jqmc._setting import ATOL_FD_VS_ANALYTIC_TIGHT, RTOL_FD_VS_ANALYTIC_TIGHT
from jqmc.derivatives import bridge
from jqmc.determinant import to_AGP
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_three_body_data
from jqmc.jqmc_mcmc import MCMC
from jqmc.molecular_orbital import MOs_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests.test_jqmc_kernel.complex_opt_ground_truth_harness import _RT_IDENTITY, _eL_batched, _ln_batched

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]

_TREXIO = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")
_N_SAMPLES = 400
_MASK = {"opt_J1_param": False, "opt_J2_param": False, "opt_J3_param": True, "opt_geminal": True}
# J3 is affine in j_matrix (exact central difference at any step); lambda is not.
_FD_STEP = {"j3_matrix": 1.0, "lambda_matrix": 1e-5}


@functools.lru_cache(maxsize=1)
def _case():
    """H2 AE cc-pVDZ, identity-MO J3 and closed-shell AGP at a generic symmetric point."""
    structure, aos, _mo_up, _mo_dn, sd, coulomb = read_trexio_file(
        os.path.join(_TREXIO, "H2_ae_ccpvdz_cart.h5"), store_tuple=True
    )
    rng = np.random.default_rng(7)
    n = int(aos.num_ao)
    jm = 0.1 * rng.standard_normal((n, n + 1))
    jm[:, :n] = 0.5 * (jm[:, :n] + jm[:, :n].T)
    j3 = Jastrow_three_body_data(orb_data=MOs_data(num_mo=n, aos_data=aos, mo_coefficients=np.eye(n)), j_matrix=jm)
    agp = to_AGP(sd)
    lam = np.asarray(agp.lambda_matrix)
    pert = rng.standard_normal(lam.shape)
    agp = agp.replace(lambda_matrix=lam + 0.1 * np.max(np.abs(lam)) * 0.5 * (pert + pert.T))
    wf = Wavefunction_data(jastrow_data=Jastrow_data(jastrow_three_body_data=j3), geminal_data=agp)
    ham = Hamiltonian_data(structure_data=structure, wavefunction_data=wf, coulomb_potential_data=coulomb)
    blocks = wf.get_variational_blocks(state_token=("linear-tie-fd", 0), **_MASK)

    pos = np.asarray(structure._positions_cart_np)
    r_up = pos[0][None, None, :] + rng.standard_normal((_N_SAMPLES, 1, 3))
    r_dn = pos[1][None, None, :] + rng.standard_normal((_N_SAMPLES, 1, 3))
    e_L = np.asarray(_eL_batched(ham, r_up, r_dn, _RT_IDENTITY))
    o_map = bridge._reduce_c_param_grads(
        wf, bridge.batched_grad_ln_Psi(wrt="c", mask=_MASK)(ham, r_up, r_dn), _N_SAMPLES, _MASK
    )

    mcmc = MCMC(
        hamiltonian_data=ham,
        num_walkers=1,
        comput_log_WF_param_deriv=True,
        comput_e_L_param_deriv=False,
        comput_position_deriv=False,
        random_discretized_mesh=False,
    )
    mcmc._MCMC__stored_w_L = np.ones((_N_SAMPLES, 1))
    mcmc._MCMC__stored_e_L = e_L.reshape(_N_SAMPLES, 1)
    mcmc._MCMC__stored_log_WF_param_grads = {
        name: np.asarray(arr).reshape(_N_SAMPLES, 1, *np.shape(arr)[1:]) for name, arr in o_map.items()
    }
    mcmc._MCMC__mcmc_counter = _N_SAMPLES
    f_impl, _ = mcmc.get_gF(num_mcmc_warmup_steps=0, num_mcmc_bin_blocks=_N_SAMPLES, blocks=blocks)
    return wf, blocks, r_up, r_dn, e_L, np.asarray(f_impl)


@pytest.mark.parametrize("block_name", ["j3_matrix", "lambda_matrix"])
def test_tied_block_force_matches_physical_direction_fd(block_name):
    """get_gF on a tied block equals -2 cov(O_D, e_L), halved per storage unit off the diagonal."""
    wf, blocks, r_up, r_dn, e_L, f_all = _case()
    offset = 0
    for block in blocks:
        if block.name == block_name:
            break
        offset += block.size
    assert block.name == block_name and block.tangent_spec.kind == "linear_tie"

    if block_name == "j3_matrix":
        j3 = wf.jastrow_data.jastrow_three_body_data
        base = np.asarray(j3.j_matrix)

        def wf_at(mat):
            return wf.replace(jastrow_data=wf.jastrow_data.replace(jastrow_three_body_data=j3.replace(j_matrix=mat)))
    else:
        agp = wf.geminal_data
        base = np.asarray(agp.lambda_matrix)

        def wf_at(mat):
            return wf.replace(geminal_data=agp.replace(lambda_matrix=mat))

    h = _FD_STEP[block_name]
    n_paired = base.shape[0]
    de = e_L - np.mean(e_L)
    f_ref = np.zeros(base.shape)
    for i, j in np.ndindex(base.shape):
        direction = np.zeros(base.shape)
        direction[i, j] = 1.0
        paired_offdiag = j < n_paired and j != i
        if paired_offdiag:
            direction[j, i] = 1.0
        ln_plus = np.asarray(_ln_batched(wf_at(base + h * direction), r_up, r_dn))
        ln_minus = np.asarray(_ln_batched(wf_at(base - h * direction), r_up, r_dn))
        o_d = (ln_plus - ln_minus) / (2.0 * h)
        f_d = -2.0 * np.mean((o_d - np.mean(o_d)) * de)
        f_ref[i, j] = 0.5 * f_d if paired_offdiag else f_d

    f_impl = f_all[offset : offset + block.size].reshape(base.shape)
    assert_allclose(
        f_impl,
        f_ref,
        rtol=RTOL_FD_VS_ANALYTIC_TIGHT,
        atol=ATOL_FD_VS_ANALYTIC_TIGHT,
        err_msg=f"get_gF on the tied {block_name} block != physical-direction FD force",
    )

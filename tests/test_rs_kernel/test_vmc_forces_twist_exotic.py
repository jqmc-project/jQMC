"""Layer-1 equivalence: COMPLEX-twist NON-pGTO leaf VMC HF + Pulay force.

The pure plane-wave / pure B-spline / pgto(+)pw / pgto(+)bspline twist (k != Gamma) leaves
carry the complex PW + B-spline ELECTRON-axis kinetic-force term
(``complex_det::complex_det_kinetic_grads`` + ``bspline::AosBsplineComplexData::det_egrad_fields``)
and the complex force entry / det-Pulay handle a 0-gto leaf (``complex_gto_home_atoms`` /
``gto_slice_cm``). Each row asserts native (``vmc_grad_e_L_pbc_complex`` /
``vmc_grad_ln_psi_pbc_complex``) == the JAX ``assemble.compute_grad_e_L`` / ``compute_grad_ln_Psi``
oracle at the periodic-force fp floor across the supported complex AO leaves.
the complex ELECTRON-axis kinetic HF force (the Re(A)Re(B) != Re(A*B) trap).

Heavy: the leaves are built from the committed espresso QE projection samples (no bench-h5).
Kept in a SEPARATE module from ``test_vmc_forces.py`` (which is light-tier) because the QE
projection build is heavy and a test must carry EXACTLY one of light / heavy.
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

import os

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import jqmc_rs  # noqa: E402

from jqmc._setting import (  # noqa: E402
    ATOL_FLOOR_MEDIUM,
    RTOL_FLOOR_MEDIUM,
)
from jqmc.derivatives import assemble  # noqa: E402
from jqmc.jqmc_mcmc import _geminal_inv_batched, _NV_BASE_MESH, Nv_default  # noqa: E402
from tests._test_data_cache import qe_example_files_root

pytestmark = [pytest.mark.with_chdir, pytest.mark.heavy]

_MESH = _NV_BASE_MESH[Nv_default]
_W = np.asarray(_MESH.weights, float)
_GP = np.asarray(_MESH.grid_points, float)
_ATOL_PBC, _RTOL_PBC = ATOL_FLOOR_MEDIUM, RTOL_FLOOR_MEDIUM

_QE_DIR = os.fspath(qe_example_files_root())
_QE_TWIST_SAVE = os.path.join(_QE_DIR, "Li_k025", "out", "li_k025.save")
_qe_twist_ok = os.path.isfile(os.path.join(_QE_TWIST_SAVE, "data-file-schema.xml")) and os.path.isfile(
    os.path.join(_QE_TWIST_SAVE, "Li.ccECP.upf")
)

_TWIST_EXOTIC = [
    ("twist_pw", "twist_pw"),  # AOs_plane_wave_complex_data (pure)
    ("twist_bspline", "twist_bspline"),  # AOs_bspline_complex_data (pure)
    ("twist_pgto_pw", "twist_pgto_pw"),  # composite pgto-complex + plane_wave_complex
    ("twist_pgto_bspline", "twist_pgto_bspline"),  # composite pgto-complex + bspline_complex
]


def _build(kind):
    from tests.test_rs_kernel.test_rust_dispatch_equivalence import _build_ham_qe

    return _build_ham_qe(kind)


def _cfg(ham, seed, scale=0.6):
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    ru = R[rng.integers(0, len(R), nu)] + rng.normal(scale=scale, size=(nu, 3))
    rd = R[rng.integers(0, len(R), nd)] + rng.normal(scale=scale, size=(nd, 3))
    return ru, rd


def _geminv(ham, ru, rd):
    _, ginv, _, _ = _geminal_inv_batched(ham.wavefunction_data.geminal_data, jnp.asarray(ru)[None], jnp.asarray(rd)[None])
    return np.asarray(ginv[0], dtype=np.complex128)


def _jax_hf(ham, ru, rd, ginv):
    RT = np.eye(3)
    dR = np.asarray(
        assemble.compute_grad_e_L(ham, jnp.asarray(ru), jnp.asarray(rd), RT, wrt="R", geminal_inv=jnp.asarray(ginv))
    )
    dru, drd = assemble.compute_grad_e_L(ham, jnp.asarray(ru), jnp.asarray(rd), RT, wrt="r", geminal_inv=jnp.asarray(ginv))
    return dR, np.asarray(dru), np.asarray(drd)


def _jax_pulay(ham, ru, rd, ginv):
    dR = np.asarray(assemble.compute_grad_ln_Psi(ham, jnp.asarray(ru), jnp.asarray(rd), wrt="R", geminal_inv=jnp.asarray(ginv)))
    dru, drd = assemble.compute_grad_ln_Psi(ham, jnp.asarray(ru), jnp.asarray(rd), wrt="r", geminal_inv=jnp.asarray(ginv))
    return dR, np.asarray(dru), np.asarray(drd)


@pytest.mark.parametrize("fid,kind", _TWIST_EXOTIC, ids=[f[0] for f in _TWIST_EXOTIC])
@pytest.mark.parametrize("seed", [0, 3])
def test_vmc_force_twist_exotic(fid, kind, seed):
    """Vmc force twist exotic."""
    if not _qe_twist_ok:
        pytest.skip("Li_k025 QE fixture not present")
    ham = _build(kind)
    wf = ham.wavefunction_data
    gd, jd, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    ru, rd = _cfg(ham, seed)
    ginv = _geminv(ham, ru, rd)
    # HF force (real, Re-projected). The electron-axis (dru/drd) is the complex-leaf kinetic term.
    dR_j, dru_j, drd_j = _jax_hf(ham, ru, rd, ginv)
    nR, nu, nd = jqmc_rs.vmc_grad_e_L_pbc_complex(gd, jd, cp, ru, rd, _W, _GP, None)
    np.testing.assert_allclose(np.asarray(nR), dR_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{fid} HF dR")
    np.testing.assert_allclose(np.asarray(nu), dru_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{fid} HF dr_up")
    np.testing.assert_allclose(np.asarray(nd), drd_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{fid} HF dr_dn")
    # Pulay force
    pR_j, pru_j, prd_j = _jax_pulay(ham, ru, rd, ginv)
    pR, pu, pd = jqmc_rs.vmc_grad_ln_psi_pbc_complex(gd, jd, ru, rd)
    np.testing.assert_allclose(np.asarray(pR), pR_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{fid} Pulay dR")
    np.testing.assert_allclose(np.asarray(pu), pru_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{fid} Pulay dr_up")
    np.testing.assert_allclose(np.asarray(pd), prd_j, atol=_ATOL_PBC, rtol=_RTOL_PBC, err_msg=f"{fid} Pulay dr_dn")

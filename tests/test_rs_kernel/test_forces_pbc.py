"""Layer-1: PBC kinetic nuclear force d T/d R (Rust forces_pbc.rs) vs the
XLA analytical force implementation. REAL Gamma-point sphe-pgto / composite
pgto+pw. jnone AND with-Jastrow (J1/J2; orbital-Hessian cross term + J1 PBC
smooth-dist force).

Fixtures are committed-fixture patterns via ``tests.test_rs_kernel._rs_fixtures.rs_ham``.

The default Rust periodic AO evaluators use the exact full image sum, like
XLA. Screened/cutoff evaluation remains only in explicitly named fast routes.
These parity rows therefore use the same dedicated XLA--Rust consistency
tolerance as molecular rows; pytest's full/mixed precision mode is the only
tolerance selector.
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

import jqmc_rs  # noqa: E402
import numpy as np
import pytest

from jqmc._precision import get_tolerance_xla_rust_consistency  # noqa: E402
from jqmc._precision import get_tolerance_xla_rust_consistency as _d8_backend_tolerance
from jqmc.derivatives.force import grad_T_wrt_R_withJ  # noqa: E402
from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = pytest.mark.with_chdir

# name -> canonical rs_ham pattern; comment retains the original bench-h5 provenance.
# Per-row cost tier: the sphe-pgto rows (larger AO count / active J3) each take
# ~10-20s locally (dominated by jax.grad compile of compute_kinetic_energy under
# _inject_active_positions, not the Rust side) -> heavy. The composite rows stay
# under 5s -> light.
_ROWS = {
    "sphepgto_jnone": ("pbc_gamma_sphepgto_jnone", "heavy"),  # single sphe-pgto, jnone
    "composite_jnone": ("pbc_composite_jnone", "light"),  # composite cart-pgto+PW, jnone
    "composite_j12": ("pbc_composite_j12", "light"),  # composite + J1/J2
    "sphepgto_j12": ("pbc_gamma_sphepgto_j12", "heavy"),  # sphe-pgto + J1/J2
    "sphepgto_j123": ("pbc_gamma_sphepgto_j123", "heavy"),  # sphe-pgto + J1/J2/J3
}
_TIER_MARKS = {"light": pytest.mark.light, "heavy": pytest.mark.heavy}
_ROW_PARAMS = [pytest.param(name, marks=_TIER_MARKS[tier]) for name, (_pattern, tier) in _ROWS.items()]

# Zones crossed by grad_T_pbc_wrt_R: the AO cotangent cutoff (ao_eval/ao_grad_lap),
# the determinant path it feeds (det_eval), and the kinetic-energy assembly
# (wf_kinetic) that amplifies the AO-side truncation into the force.
_ZONES = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic")


def _rand_j3(ham, seed=7):
    wf = ham.wavefunction_data
    j3 = wf.jastrow_data.jastrow_three_body_data
    n = np.asarray(j3.j_matrix).shape[0]
    jm = np.random.default_rng(seed).normal(scale=0.2, size=(n, n + 1))
    # Production invariant: the J3 coupling block is symmetric (bosonic Jastrow);
    # Jastrow_three_body_data.sanity_check RAISES otherwise, and several jqmc
    # streaming/ratio kernels exploit it (j3_mat.T @ x == j3_mat @ x). A .replace()
    # bypasses sanity_check, so symmetrize here or we test an invalid regime.
    jm[:, :-1] = 0.5 * (jm[:, :-1] + jm[:, :-1].T)
    return wf.jastrow_data.replace(jastrow_three_body_data=j3.replace(j_matrix=jm))


@pytest.mark.parametrize("name", _ROW_PARAMS)
def test_grad_T_pbc_R(name):
    """Grad T PBC R."""
    pattern, _tier = _ROWS[name]
    ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    if pattern == "pbc_gamma_sphepgto_j123":
        wf = wf.replace(jastrow_data=_rand_j3(ham))
    gd = wf.geminal_data
    jd = wf.jastrow_data
    R = np.asarray(ham.structure_data.positions, float)
    nu, nd = gd.num_electron_up, gd.num_electron_dn
    atol, rtol = _d8_backend_tolerance(_ZONES, level="medium")
    for seed in range(2):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        rng = np.random.default_rng(seed)
        ru = R[np.arange(nu) % len(R)] + rng.normal(scale=0.7, size=(nu, 3))
        rd = R[np.arange(nd) % len(R)] + rng.normal(scale=0.7, size=(nd, 3))
        ana = jqmc_rs.grad_T_pbc_wrt_R(gd, jd, ru, rd)
        ref = grad_T_wrt_R_withJ(gd, jd, ru, rd)
        ana_np, ref_np = np.asarray(ana), np.asarray(ref)
        d = np.abs(ana_np - ref_np).max()
        # dynamic-AO-cutoff tradeoff (see module docstring): loosest zone atol,
        # plus a relative term scaled by the force magnitude itself.
        assert d <= atol + rtol * max(1.0, np.abs(ref_np).max()), (name, seed, d)


@pytest.mark.light
def test_grad_T_pbc_R_composite_j3():
    """PBC kinetic NUCLEAR force dT/dR for a J3 orbital that is an MO-wrapped
    pgto(+)PW COMPOSITE (``_J3_KIND['composite'] == 'mo'`` -> ``MOs_data(AOs_composite_
    data)``): the J3-force/stress-on-composite case. The pGTO child
    contributes its atom-centered J3 kinetic force; the plane-wave child (dAO/dR = 0)
    contributes zero to the force but still enters the pGTO rows' cotangent through the
    symmetric j_matrix coupling. Oracle: XLA analytical kinetic force."""
    from tests.test_jqmc_kernel._generated_matrix_base import _J3_KIND, _build_ham_for_cell

    ham = _build_ham_for_cell("QE:Li_gamma_pgto+pw", True, {"j3": _J3_KIND["composite"]}, "SD")
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    R = np.asarray(ham.structure_data.positions, float)
    nu, nd = gd.num_electron_up, gd.num_electron_dn
    atol, rtol = _d8_backend_tolerance(_ZONES, level="medium")
    for seed in range(2):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        rng = np.random.default_rng(seed)
        ru = R[np.arange(nu) % len(R)] + rng.normal(scale=0.7, size=(nu, 3))
        rd = R[np.arange(nd) % len(R)] + rng.normal(scale=0.7, size=(nd, 3))
        ana = np.asarray(jqmc_rs.grad_T_pbc_wrt_R(gd, jd, ru, rd))
        ref = np.asarray(grad_T_wrt_R_withJ(gd, jd, ru, rd))
        d = np.abs(ana - ref).max()
        assert d <= atol + rtol * max(1.0, np.abs(ref).max()), (seed, d)


@pytest.mark.light
@pytest.mark.parametrize("withj", [False, True])
def test_grad_T_pbc_R_bspline(withj):
    """PBC kinetic NUCLEAR force dT/dR for a B-spline geminal (QE-projection Li_gamma, ECP),
    jnone and J1/J2. The determinant Pulay is exactly zero (control points sit on a fixed
    cell-fractional grid, dAO/dR=0), so grad_T_pbc_wrt_R must (a) NOT panic on the B-spline
    leaf and (b) return only the J1 periodic smooth-distance Pulay (basis-independent). The
    B-spline electron-axis kinetic force is a separate target (test_gfmc_forces)."""
    from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell

    if withj:
        ham = _build_ham_for_cell("QEPROJ:bspline:Li_gamma:ECP", True, None, "SD")  # J1/J2
    else:
        ham = rs_ham("bspline_gamma_jnone")
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    jd = wf.jastrow_data
    R = np.asarray(ham.structure_data.positions, float)
    nu, nd = gd.num_electron_up, gd.num_electron_dn
    atol, rtol = _d8_backend_tolerance(_ZONES, level="medium")
    for seed in range(2):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        rng = np.random.default_rng(seed)
        ru = R[np.arange(nu) % len(R)] + rng.normal(scale=0.7, size=(nu, 3))
        rd = R[np.arange(nd) % len(R)] + rng.normal(scale=0.7, size=(nd, 3))
        ana = np.asarray(jqmc_rs.grad_T_pbc_wrt_R(gd, jd, ru, rd))
        ref = np.asarray(grad_T_wrt_R_withJ(gd, jd, ru, rd))
        d = np.abs(ana - ref).max()
        assert d <= atol + rtol * max(1.0, np.abs(ref).max()), (withj, seed, d)

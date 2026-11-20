"""FAST ECP radial cut on the real GFMC projection mesh (IMPL-RS-GFMC-FAST-ECP-RADIAL-CUT-001).

On the real FAST GFMC mesh (``gfmc::compute_ecp_nonlocal_mesh_fast``) an (electron,
non-local term) pair whose radial factor ``|v_t(r)| = |c r^(p-2) exp(-e r^2)|`` at the
distance to the electron's nearest ECP nucleus is below
``FAST_ECP_NONLOCAL_RADIAL_CUT`` (jqmc._setting, the VMC FAST rule) contributes
nothing: the mesh evaluates the truncated Hamiltonian H' with ``v_t = 0`` there.
Electrons left with only zero amplitudes are then skipped exactly
(IMPL-RS-GFMC-ECP-ZERO-GROUP-SKIP-001). The test-only hook
``JQMC_RS_GFMC_ECP_NO_RADIAL_CUT`` disables the cut for an A/B comparison.

On the water ccECP fixture, seeded configurations spread wide enough that some
electrons lie beyond the O s-channel cut radius, with random mesh rotations, the
single-configuration FAST LRDMC local energy ``jqmc_rs._gfmc_lrdmc_ecp(fast=True)``
is evaluated with and without the cut. Asserted (tmove and dltmove):

* the kept / dropped pair counts (``jqmc_rs.gfmc_route_counters`` 70 / 71) equal
  the pairs whose ``|v_t|`` (recomputed here in NumPy) is at or above / below EPS,
  and the uncut run counts none;
* per configuration (tmove), the local energy and the rate ``b_L`` change by at
  most the a-posteriori bound ``B = sum_dropped |v_t| (2l+1) sum_v w_v |P_l(cos_v)|
  |Psi(r'_v)/Psi(r)|`` (each dropped pair's own exact contribution bound, with the
  wavefunction ratio evaluated independently by ``jqmc.wavefunction``), plus the
  central reduction-order floor; and ``B <= EPS sum_dropped (2l+1) sum_v w_v
  |P_l| |Psi'/Psi|``;
* non-vacuity: pairs with ``0 < |v_t| < EPS`` are dropped and pairs are kept.
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
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jqmc_rs  # noqa: E402

from jqmc._precision import get_tolerance_reduction_order  # noqa: E402
from jqmc._setting import FAST_ECP_NONLOCAL_RADIAL_CUT  # noqa: E402
from jqmc.wavefunction import evaluate_wavefunction  # noqa: E402
from tests.test_rs_kernel._rs_fixtures import rs_ham  # noqa: E402
from tests.test_rs_kernel.test_gfmc_projection import _EL_ZONES_J, _MESH_GP, _MESH_W  # noqa: E402

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

_HOOK = "JQMC_RS_GFMC_ECP_NO_RADIAL_CUT"
_N_CONFIGS = 6
# Std (bohr) of the electron offsets around a random nucleus: wide enough that
# some electrons lie beyond the O s-channel cut radius (~1.6 bohr).
_SPREAD = 1.4
_ALAT = 0.30


def _rotation(rng):
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _legendre(l, x):
    return np.polynomial.legendre.legval(x, [0.0] * l + [1.0])


def _pairs(cp, positions, r):
    """Per electron: (nearest ECP nucleus, |r - R|, [(term, v_t)])."""
    nuc = np.asarray(cp._nucleus_index_non_local_part, int)
    e = np.asarray(cp._exponents_non_local_part, float)
    c = np.asarray(cp._coefficients_non_local_part, float)
    p = np.asarray(cp._powers_non_local_part, float)
    ecp_nuclei = np.unique(nuc)
    out = []
    for x in r:
        d = np.linalg.norm(positions[ecp_nuclei] - x, axis=1)
        ia = int(ecp_nuclei[int(np.argmin(d))])
        rr = float(np.linalg.norm(positions[ia] - x))
        terms = [(t, c[t] * rr ** (p[t] - 2.0) * np.exp(-e[t] * rr * rr)) for t in np.nonzero(nuc == ia)[0]]
        out.append((ia, rr, terms))
    return out


def _local_energy(ham, r_up, r_dn, rt, dltmove, cut, monkeypatch):
    if cut:
        monkeypatch.delenv(_HOOK, raising=False)
    else:
        monkeypatch.setenv(_HOOK, "1")
    wf = ham.wavefunction_data
    jqmc_rs.gfmc_route_counters_reset()
    e_l, b_l, _ = jqmc_rs._gfmc_lrdmc_ecp(
        wf.geminal_data,
        wf.jastrow_data,
        ham.coulomb_potential_data,
        r_up,
        r_dn,
        _ALAT,
        rt,
        _MESH_W,
        _MESH_GP,
        1,
        dltmove,
        fast=True,
    )
    counters = jqmc_rs.gfmc_route_counters()
    return float(e_l), float(b_l), int(counters[70]), int(counters[71])


@pytest.mark.parametrize("pattern", ["mol_ecp_sphe_j12"])
@pytest.mark.parametrize("dltmove", [False, True])
def test_gfmc_fast_ecp_radial_cut_within_a_posteriori_bound(pattern, dltmove, monkeypatch):
    """The GFMC FAST radial cut drops exactly the sub-EPS pairs and moves e_L / b_L within their own bound."""
    ham = rs_ham(pattern)
    assert not bool(ham.structure_data.pbc_flag) and bool(ham.coulomb_potential_data.ecp_flag), pattern
    cp = ham.coulomb_potential_data
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    positions = np.asarray(ham.structure_data.positions, float)
    ang_mom = np.asarray(cp._ang_mom_non_local_part, int)
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    atol, rtol = get_tolerance_reduction_order(_EL_ZONES_J + ("jastrow_ratio",))
    rng = np.random.default_rng(20261105)
    psi = jax.jit(lambda ru, rd: evaluate_wavefunction(wf, ru, rd))
    nonzero_dropped = total_kept = 0
    for config in range(_N_CONFIGS):
        r_up = positions[rng.integers(0, len(positions), nu)] + rng.normal(scale=_SPREAD, size=(nu, 3))
        r_dn = positions[rng.integers(0, len(positions), nd)] + rng.normal(scale=_SPREAD, size=(nd, 3))
        rt = np.ascontiguousarray(_rotation(rng))
        e_c, b_c, kept_c, dropped_c = _local_energy(ham, r_up, r_dn, rt, dltmove, True, monkeypatch)
        e_n, b_n, kept_n, dropped_n = _local_energy(ham, r_up, r_dn, rt, dltmove, False, monkeypatch)
        tag = f"{pattern} dltmove={dltmove} config {config}"
        assert np.isfinite(e_c) and np.isfinite(e_n) and np.isfinite(b_c) and np.isfinite(b_n), tag
        assert (kept_n, dropped_n) == (0, 0), f"{tag}: the uncut route counted pairs"
        pairs = _pairs(cp, positions, r_up) + _pairs(cp, positions, r_dn)
        mags = np.array([abs(v) for _, _, terms in pairs for _, v in terms])
        want_dropped = int(np.count_nonzero(mags < FAST_ECP_NONLOCAL_RADIAL_CUT))
        assert (kept_c, dropped_c) == (mags.size - want_dropped, want_dropped), (
            f"{tag}: (kept, dropped) = {(kept_c, dropped_c)}, expected {(mags.size - want_dropped, want_dropped)}"
        )
        total_kept += kept_c
        if dltmove:
            continue
        # A-posteriori bound of the dropped contributions (tmove: vfull = ang * Psi'/Psi).
        gp = _MESH_GP @ rt
        psi0 = float(psi(r_up, r_dn))
        dropped_terms = []
        cut_terms = []
        for k, (ia, rr, terms) in enumerate(pairs):
            dropped = [(t, v) for t, v in terms if abs(v) < FAST_ECP_NONLOCAL_RADIAL_CUT and v != 0.0]
            if not dropped:
                continue
            nonzero_dropped += len(dropped)
            is_up = k < nu
            i = k if is_up else k - nu
            x = (r_up if is_up else r_dn)[i]
            for v_idx in range(len(_MESH_W)):
                d = gp[v_idx]
                cosv = float(np.dot(x - positions[ia], d) / (rr * np.linalg.norm(d)))
                xp = positions[ia] + rr * d
                ru, rd = r_up.copy(), r_dn.copy()
                (ru if is_up else rd)[i] = xp
                ratio = abs(float(psi(ru, rd)) / psi0)
                for t, v in dropped:
                    shape = (2 * ang_mom[t] + 1) * _MESH_W[v_idx] * abs(_legendre(ang_mom[t], cosv)) * ratio
                    dropped_terms.append(abs(v) * shape)
                    cut_terms.append(FAST_ECP_NONLOCAL_RADIAL_CUT * shape)
        bound = float(np.sum(dropped_terms))
        cut_scaled = float(np.sum(cut_terms))
        assert bound <= cut_scaled, f"{tag}: a-posteriori bound {bound:.3e} exceeds the EPS-scaled sum {cut_scaled:.3e}"
        floor_e = atol + rtol * abs(e_n)
        floor_b = atol + rtol * abs(b_n)
        assert abs(e_c - e_n) <= bound + floor_e, f"{tag}: |de_L| = {abs(e_c - e_n):.3e} > {bound:.3e} + {floor_e:.1e}"
        assert abs(b_c - b_n) <= bound + floor_b, f"{tag}: |db_L| = {abs(b_c - b_n):.3e} > {bound:.3e} + {floor_b:.1e}"
    assert total_kept > 0, f"{pattern}: every pair dropped"
    if not dltmove:
        assert nonzero_dropped > 0, f"{pattern}: vacuous cut (no pair with 0 < |v_t| < EPS dropped)"

"""FAST ECP non-local radial cut (IMPL-RS-FAST-ECP-NONLOCAL-RADIAL-CUT-001).

On the FAST Rust route of ``compute_ecp_nonlocal_parts_all_pairs`` (mesh AO cutoff
plus the Sherman-Morrison inverse: the native production chunk) an (electron,
non-local term) pair whose radial factor ``|v_t(r)| = |c r^(p-2) exp(-e r^2)|`` is
below ``FAST_ECP_NONLOCAL_RADIAL_CUT`` (jqmc._setting, 1e-12 Ha) skips its whole
quadrature. That is exactly a modified Hamiltonian with the semilocal radial
functions truncated where ``|v_t| < EPS``; with the rotation-averaged quadrature
equal to the angular projector, every energy expectation moves by at most
``N_e * n_t * EPS`` (n_t = non-local terms per nucleus).

On the committed water ccECP fixture, for seeded configurations with random mesh
rotations, the test-only probe ``_ecp_nonlocal_fast_radial_cut_probe`` evaluates
the FAST route with and without the cut. Asserted: the Rust constant equals the
central one; the kept / dropped pair counts equal the pairs whose ``|v_t|``
(recomputed here in NumPy from the ECP tables, nearest ECP nucleus per electron)
is at or above / below EPS; some pairs are dropped and some kept (non-vacuity); and
the mean change of the non-local energy over the configurations is within
``N_e * n_t * EPS`` plus the central reduction-order floor. The strong pointwise
decomposition (cut value = uncut value minus the independently recomputed dropped
contributions) is the native test
``coulomb_potential::tests::fast_ecp_radial_cut_drops_exactly_sub_bound_pairs``.
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

from jqmc._setting import ATOL_REDUCTION_ORDER, FAST_ECP_NONLOCAL_RADIAL_CUT  # noqa: E402
from jqmc.jqmc_mcmc import _NV_BASE_MESH, Nv_default  # noqa: E402
from tests.test_rs_kernel._rs_fixtures import rs_ham  # noqa: E402

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

_MESH = _NV_BASE_MESH[Nv_default]
_W = np.ascontiguousarray(np.asarray(_MESH.weights, float))
_GP = np.asarray(_MESH.grid_points, float)
_N_CONFIGS = 16
# Std (bohr) of the electron offsets around a randomly chosen nucleus: wide enough
# that some electrons lie beyond the O s-channel cut radius (~1.6 bohr).
_SPREAD = 1.4
_PATTERNS = ("mol_ecp_sphe_j12",)


def _rotation(rng):
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _radial_factors(cp, positions, r):
    """|v_t| of every (electron, term of its nearest ECP nucleus) pair, independently of Rust."""
    nuc = np.asarray(cp._nucleus_index_non_local_part, int)
    e = np.asarray(cp._exponents_non_local_part, float)
    c = np.asarray(cp._coefficients_non_local_part, float)
    p = np.asarray(cp._powers_non_local_part, float)
    ecp_nuclei = np.unique(nuc)
    out = []
    for x in r:
        d = np.linalg.norm(positions[ecp_nuclei] - x, axis=1)
        k = int(np.argmin(d))
        rr = float(d[k])
        sel = nuc == ecp_nuclei[k]
        out.append(np.abs(c[sel] * rr ** (p[sel] - 2.0) * np.exp(-e[sel] * rr * rr)))
    return np.concatenate(out) if out else np.zeros(0)


@pytest.mark.parametrize("pattern", _PATTERNS)
def test_fast_ecp_radial_cut_moves_energy_within_bound(pattern):
    """The FAST radial cut drops exactly the sub-EPS pairs and moves the mean ECP energy by <= N_e n_t EPS."""
    assert jqmc_rs.fast_ecp_nonlocal_radial_cut() == FAST_ECP_NONLOCAL_RADIAL_CUT
    ham = rs_ham(pattern)
    assert not bool(ham.structure_data.pbc_flag) and bool(ham.coulomb_potential_data.ecp_flag), pattern
    gd = ham.wavefunction_data.geminal_data
    jd = ham.wavefunction_data.jastrow_data
    cp = ham.coulomb_potential_data
    positions = np.asarray(ham.structure_data.positions, float)
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    nuc = np.asarray(cp._nucleus_index_non_local_part, int)
    assert nuc.size > 0, f"{pattern}: no non-local ECP terms"
    n_t = int(np.bincount(nuc).max())
    bound = (nu + nd) * n_t * FAST_ECP_NONLOCAL_RADIAL_CUT
    rng = np.random.default_rng(20261105)
    shifts = []
    total_kept = total_dropped = 0
    for config in range(_N_CONFIGS):
        r_up = positions[rng.integers(0, len(positions), nu)] + rng.normal(scale=_SPREAD, size=(nu, 3))
        r_dn = positions[rng.integers(0, len(positions), nd)] + rng.normal(scale=_SPREAD, size=(nd, 3))
        gp = np.ascontiguousarray(_GP @ _rotation(rng))
        v_cut, kept, dropped = jqmc_rs._ecp_nonlocal_fast_radial_cut_probe(
            gd, jd, cp, r_up, r_dn, _W, gp, FAST_ECP_NONLOCAL_RADIAL_CUT
        )
        v_full, kept0, dropped0 = jqmc_rs._ecp_nonlocal_fast_radial_cut_probe(gd, jd, cp, r_up, r_dn, _W, gp, 0.0)
        assert (kept0, dropped0) == (0, 0), f"{pattern} config {config}: the uncut route counted pairs"
        assert np.isfinite(v_cut) and np.isfinite(v_full), f"{pattern} config {config}: non-finite ECP energy"
        mags = np.concatenate([_radial_factors(cp, positions, r_up), _radial_factors(cp, positions, r_dn)])
        want_dropped = int(np.count_nonzero(mags < FAST_ECP_NONLOCAL_RADIAL_CUT))
        assert (kept, dropped) == (mags.size - want_dropped, want_dropped), (
            f"{pattern} config {config}: (kept, dropped) = {(kept, dropped)}, expected {(mags.size - want_dropped, want_dropped)}"
        )
        shifts.append(v_cut - v_full)
        total_kept += kept
        total_dropped += dropped
    assert total_dropped > 0 and total_kept > 0, f"{pattern}: vacuous cut ({total_kept} kept, {total_dropped} dropped)"
    mean_shift = abs(float(np.mean(shifts)))
    assert mean_shift <= bound + ATOL_REDUCTION_ORDER, (
        f"{pattern}: mean ECP energy shift {mean_shift:.3e} > N_e n_t EPS = {bound:.3e} (+ floor {ATOL_REDUCTION_ORDER:.1e}); "
        f"max |shift| {float(np.max(np.abs(shifts))):.3e}"
    )

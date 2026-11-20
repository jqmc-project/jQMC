"""NaN propagation through the GFMC projection.

A NaN acceptance ratio must REJECT in MCMC (``mcmc.rs::accept_clamp``: ``jnp.min``
propagates and ``b < NaN`` is False); a degenerate determinant ratio that is absorbed
instead can march the Rust walker to ~300 Bohr while XLA keeps it put. ``gfmc.rs`` has
the same class of sites, where a bare ``f64::min`` / ``f64::max`` (IEEE minNum/maxNum,
which ABSORB the NaN) would disagree with XLA's ``jnp.minimum`` / ``jnp.maximum``,
which PROPAGATE:

  * the FN/SP split ``el.min(0.0)`` / ``el.max(0.0)`` (kinetic elements AND the ECP
    non-local amplitudes, on every reference chain and on the production chunk),
  * the ZV composition ``zv.max(ei)``,
  * the escape-time clamp ``tau_left.min(log(1-xi)/b_L)``.

The categorical hop index follows ``cdf.partition_point(|c| c < u)`` at both hop sites,
which returns 0 on an all-NaN cdf, as ``jnp.searchsorted(all-NaN cdf, u) == 0`` does; a
linear scan ``if acc >= u`` would fall through to ``k = total - 1``.

Injection (poison deterministically)
------------------------------------
One NaN in a single MO coefficient poisons every determinant on both engines, so every
discretized kinetic element -- the quantity that feeds the FN/SP clamps, ``b_L`` and the
hop cdf -- is NaN for XLA and Rust alike.  No randomness, no tuned degeneracy.

What a NaN-absorbing native chain does on this exact fixture (measured):

  * ``GFMC_n``: XLA's cdf is all-NaN, ``searchsorted`` returns 0, and it hops electron 0.
    An absorbing Rust chain sums the NaNs to 0 (``fsum``), its linear scan never
    satisfies ``acc >= u`` and falls to ``k = total - 1``, hopping the LAST candidate
    instead -- a different walker out of the same interval (``r_up[0]`` unchanged on the
    Rust side vs shifted by ``alat`` per sub-step on the XLA side).
  * ``GFMC_t``: worse -- ``b_L`` absorbed to exactly ``0.0`` gives
    ``tau_left.min(log(1-xi)/0.0) == -inf``, ``tau_left -= -inf`` grows without bound and
    the native escape-time loop DOES NOT TERMINATE.  XLA gets ``tau_update = NaN``,
    ``tau_left = NaN``, and its ``while tau_left > 0.0`` exits after one sub-step.  The
    GFMC_t case below therefore also pins TERMINATION.
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

import jqmc_rs  # noqa: F401  (import side effect: the extension must be loadable)

pytestmark = [pytest.mark.with_chdir]

from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_rs_kernel.test_gfmc_projection import (
    _MESH_GP,
    _MESH_W,
    _cfg,
    _ctx,
    _xla_interval_n,
    _xla_interval_t,
)

_ALAT = 0.30
_TAU = 0.10
_E_SCF = -1.0
_NSUB = 2
_CFG_SEED = 3
_KEY_SEED = 7
_NO_INJECTED_ROTATIONS = np.zeros((0, 3, 3))  # random_mesh=False -> R^T = I


def _poison(ham):
    """One NaN in a single up-spin MO coefficient -> every determinant NaN on BOTH
    engines -> every discretized kinetic element (the FN/SP + cdf input) NaN."""
    gd = ham.wavefunction_data.geminal_data
    mo = np.asarray(gd.orb_data_up_spin.mo_coefficients, dtype=np.float64).copy()
    mo[0, 0] = np.nan
    orb = gd.orb_data_up_spin.replace(mo_coefficients=mo)
    wf = ham.wavefunction_data.replace(geminal_data=gd.replace(orb_data_up_spin=orb))
    return ham.replace(wavefunction_data=wf)


def _setup(pattern):
    ham = _poison(rs_ham(pattern))
    r_up, r_dn = _cfg(rs_ham(pattern), _CFG_SEED)
    return ham, r_up, r_dn, jax.random.PRNGKey(_KEY_SEED)


@pytest.mark.light
@pytest.mark.parametrize("pattern", ["mol_cart_sd_jnone", "mol_ecp_cart_jnone"])
def test_gfmc_n_nan_poisoned_interval_matches_xla(pattern):
    """GFMC_n: on an all-NaN cdf the native hop must land where XLA's ``searchsorted``
    lands (index 0), and the NaN must reach ``w_L`` / ``e_L`` / ``V_diag``.

    A regression absorbs the NaNs and hops ``total - 1`` instead: a
    DIFFERENT walker, silently, out of an interval XLA marks NaN.
    """
    ham, r_up, r_dn, key = _setup(pattern)
    ref = _xla_interval_n(ham, r_up, r_dn, _ALAT, _E_SCF, _NSUB, key, "tmove")
    got = _ctx(ham).run_gfmc_projection_n(
        r_up,
        r_dn,
        _ALAT,
        _E_SCF,
        _NSUB,
        int(key[0]),
        int(key[1]),
        _NO_INJECTED_ROTATIONS,
        _MESH_W,
        _MESH_GP,
        random_mesh=False,
        nn=1,
    )
    # Precondition: the poisoned cell is real -- XLA itself goes NaN.
    assert np.isnan(ref["e_L"]) and np.isnan(ref["w_L"]), (pattern, ref["e_L"], ref["w_L"])
    # The NaN must PROPAGATE natively too (absorbing it is the defect).
    assert np.isnan(float(got["e_L"])), f"native e_L absorbed the NaN: {float(got['e_L'])}"
    assert np.isnan(float(got["w_L"])), f"native w_L absorbed the NaN: {float(got['w_L'])}"
    assert np.isnan(float(got["V_diag"])), f"native V_diag absorbed the NaN: {float(got['V_diag'])}"
    # The categorical index on an all-NaN cdf must agree: exact walker equality.
    np.testing.assert_array_equal(np.asarray(got["r_up"]), np.asarray(ref["r_up"]))
    np.testing.assert_array_equal(np.asarray(got["r_dn"]), np.asarray(ref["r_dn"]))


@pytest.mark.light
@pytest.mark.parametrize("pattern", ["mol_cart_sd_jnone", "mol_ecp_cart_jnone"])
def test_gfmc_t_nan_poisoned_interval_terminates_like_xla(pattern):
    """GFMC_t: ``b_L`` must stay NaN so ``tau_update`` is NaN and the escape-time loop
    exits after ONE sub-step, exactly as XLA's ``while tau_left > 0.0`` does.

    A regression absorbs ``b_L`` to ``0.0``, ``tau_left.min(log(1-xi)/0.0) == -inf`` and the
    native loop runs forever -- so this test also pins termination (a regression here
    hangs rather than fails; the measured behaviour is in the module docstring).
    """
    ham, r_up, r_dn, key = _setup(pattern)
    ref = _xla_interval_t(ham, r_up, r_dn, _ALAT, _TAU, key, "tmove")
    got = _ctx(ham).run_gfmc_projection_t(
        r_up,
        r_dn,
        _ALAT,
        _TAU,
        int(key[0]),
        int(key[1]),
        _MESH_W,
        _MESH_GP,
        random_mesh=False,
        nn=1,
    )
    assert np.isnan(ref["e_L"]), pattern  # precondition: XLA goes NaN
    assert int(got["n_substeps"]) == ref["n_substeps"], (pattern, int(got["n_substeps"]), ref["n_substeps"])
    assert np.isnan(float(got["e_L"])), f"native e_L absorbed the NaN: {float(got['e_L'])}"
    assert np.isnan(float(got["w_L"])), f"native w_L absorbed the NaN: {float(got['w_L'])}"
    np.testing.assert_array_equal(np.asarray(got["r_up"]), np.asarray(ref["r_up"]))
    np.testing.assert_array_equal(np.asarray(got["r_dn"]), np.asarray(ref["r_dn"]))

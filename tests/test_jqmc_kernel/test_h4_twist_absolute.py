"""Absolute interacting H4 twist energy, independently assembled in the active mode."""

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

import jax
import numpy as np
import pytest

from jqmc import atomic_orbital
from jqmc._precision import get_tolerance_min
from tests.test_jqmc_kernel.test_periodic_invariants import _pattern_ham, _draw_well_conditioned_cfg, _eL_jax, _ginv
from tests.test_jqmc_kernel._h4_twist_absolute_reference import kinetic_value, ewald_value

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]
_ZONES = ("ao_eval", "ao_grad_lap", "jastrow_eval", "jastrow_grad_lap", "wf_kinetic", "coulomb", "ewald", "local_energy")


def _case():
    ham = _pattern_ham("h4_twist_j12")
    up, down = _draw_well_conditioned_cfg(ham, seed=7)
    reference = np.asarray(kinetic_value(ham.wavefunction_data, up, down) + ewald_value(ham, up, down))
    return ham, up, down, reference


def test_interacting_h4_twist_local_energy_matches_independent_absolute_value():
    """The exact interacting H4 twist fixture has an independently derived complex e_L.

    H4_pbc_ae_k025_cart.h5, SD with opposite spin Bloch k, conventional J1+J2,
    two up/two down electrons; use the periodic-invariant fixture unchanged.
    One image-offset configuration uses PCG64(700000+attempt), at most 300
    proposals selected only by cond(G)<1e4. Each electron uses a nucleus image
    at 0/2/5 lattice hops and normal(scale=0.6) jitter; down[0] is placed near
    up[0] with normal(scale=0.8) jitter. Independent Cartesian primitive normalization,
    polynomial Gaussian sums, integer-image Bloch phases, MO contractions,
    determinant and periodic J1/J2 expressions supply Psi. Its complex Hessian
    yields kinetic energy, with independent NumPy Ewald for all point charges.
    Shared inputs are primitive coefficients, declared image indices and MO
    coefficients, not production AO/Jastrow/derivative/energy computations.
    Both modes use their active zone arithmetic and the unchanged central
    strict comparison; no finite differences, cross-mode oracle or MC average.
    """
    ham, up, down, reference = _case()
    actual = np.asarray(_eL_jax(ham, up, down, _ginv(ham, up, down)))
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    assert np.iscomplexobj(actual) and np.iscomplexobj(reference)
    assert abs(reference.imag) > atol + rtol * abs(reference.real)
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)


def test_interacting_h4_absolute_anchor_rejects_wrong_bloch_momentum(monkeypatch):
    """Scaling only production Bloch phases by 1.01 breaks the fixed-input absolute anchor.

    Both image weights and outside-cell restoration use the altered momentum.
    The independent reference derives phases from the unmodified stored k and
    integer images. The unmutated production result must pass first; changing
    phase geometry must then fail at the same existing strict tolerance.
    """
    ham, up, down, reference = _case()
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    actual = np.asarray(_eL_jax(ham, up, down, _ginv(ham, up, down)))
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)
    cls = atomic_orbital._ComplexBlochPGTOMixin
    original_fold, original_restore = cls._fold_image_axis, cls._restore_phase

    def damaged(self):
        k = 1.01 * np.asarray(self.k_point)
        phases = np.exp(2j * np.pi * (np.asarray(self._image_indices) @ k))
        return self.replace(k_point=tuple(k), _image_phase_factors=phases)

    monkeypatch.setattr(cls, "_fold_image_axis", lambda self, arr, zone: original_fold(damaged(self), arr, zone))
    monkeypatch.setattr(cls, "_restore_phase", lambda self, coords, zone: original_restore(damaged(self), coords, zone))
    jax.clear_caches()
    changed = np.asarray(_eL_jax(ham, up, down, _ginv(ham, up, down)))
    assert np.isfinite(changed)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(changed, reference, atol=atol, rtol=rtol, equal_nan=False)

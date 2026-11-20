"""Native complex VMC energy of the Li ccECP plane-wave twist fixtures against the independent reference."""

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
import jqmc_rs

from jqmc import atomic_orbital
from jqmc._precision import get_tolerance_min
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
from tests.test_jqmc_kernel._complex_ecp_reference import complex_local_energy_reference
from tests.test_jqmc_kernel.test_complex_pw_twist_absolute import PATTERNS, case, require_live_twist
from tests.test_jqmc_kernel.test_periodic_invariants import _ginv

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]
# Same existing zones as the shared XLA oracle. Keep this tuple local so the
# static tolerance resolver can verify the native comparator's exact budget.
_ZONES = (
    "ao_eval",
    "ao_grad_lap",
    "mo_eval",
    "jastrow_eval",
    "jastrow_grad_lap",
    "det_eval",
    "wf_kinetic",
    "coulomb",
    "ewald",
    "local_energy",
)


def _native(ham, up, down, inverse):
    weights = np.asarray(icosahedron_sym_mesh_Nv12.weights, dtype=np.float64)
    points = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, dtype=np.float64)
    return complex(
        *jqmc_rs.compute_local_energy_fast_complex(ham, up, down, weights, points, np.asarray(inverse, dtype=np.complex128))
    )


def _compare_components(actual, expected):
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    assert np.isfinite(actual) and np.isfinite(expected)
    for part in (np.real, np.imag):
        np.testing.assert_allclose(part(actual), part(expected), atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_native_pw_twist_vmc_energy_matches_independent_absolute_value(pattern):
    """Native composite_twist_j12 / pw_twist_j12 complex e_L equals the shared independent reference.

    The exact fixtures, seed-7 image-offset configuration, cond(G) < 1e4,
    identity Nv12 nearest-ECP mesh and explicit plane-wave / pGTO / Hessian /
    Ewald / quadrature reference of the XLA guarantee are reused, not
    reimplemented for Rust. ``compute_local_energy_fast_complex`` with the
    geminal inverse is compared with it, real and imaginary components
    separately, in each precision mode at the existing strict zone policy. The
    imaginary kinetic and nonlocal ECP parts must be active.
    """
    ham, up, down, RT = case(pattern)
    require_live_twist(ham, up, down, RT)
    expected = complex_local_energy_reference(ham, up, down, RT)
    actual = _native(ham, up, down, _ginv(ham, up, down))
    _compare_components(actual, expected)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_native_pw_twist_absolute_value_rejects_omitted_plane_wave_twist(pattern, monkeypatch):
    """The native energy with plane-wave wave vectors G instead of G + k fails the absolute anchor.

    The native marshaller reads the production ``_gk_vectors_jnp`` of every
    complex plane-wave leaf. After an unmutated passing baseline, that property
    is changed to drop the twist; the reference keeps the stored Miller
    triples and twist, so the same strict comparison must fail in either mode.
    """
    ham, up, down, RT = case(pattern)
    expected = complex_local_energy_reference(ham, up, down, RT)
    inverse = _ginv(ham, up, down)
    _compare_components(_native(ham, up, down, inverse), expected)
    cls = atomic_orbital.AOs_plane_wave_complex_data
    original = cls._gk_vectors_jnp

    def untwisted(self):
        return original.fget(self.replace(k_point=(0.0, 0.0, 0.0)))

    monkeypatch.setattr(cls, "_gk_vectors_jnp", property(untwisted))
    try:
        changed = _native(ham, up, down, inverse)
    finally:
        monkeypatch.undo()
        jax.clear_caches()
    assert np.isfinite(changed)
    with pytest.raises(AssertionError):
        _compare_components(changed, expected)

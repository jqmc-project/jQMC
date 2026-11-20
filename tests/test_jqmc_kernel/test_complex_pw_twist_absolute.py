"""Independent absolute complex VMC energy of the Li ccECP plane-wave twist fixtures.

The periodic-invariant patterns ``composite_twist_j12`` (QE Li_k025 projected
onto windowed Cartesian pGTO plus 40 complex plane waves) and ``pw_twist_j12``
(the same save projected onto 48 complex plane waves), each with the matrix
J1 + J2 Jastrow and the Li ccECP, are otherwise covered only by native-versus-native
invariants and Rust/XLA parity. Here the complex VMC local energy H Psi / Psi
of each fixture is compared with the shared independent reference of
``_complex_ecp_reference``: explicit complex plane waves rebuilt from the
stored Miller triples and fractional twist, explicit primitive Gaussian/image
Bloch sums for the pGTO child, MO and determinant contractions, periodic J1/J2,
a complex Hessian kinetic energy, a NumPy Ewald sum and explicit radial /
Legendre twelve-point nonlocal quadrature before any real projection.
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

import ast
import inspect
import textwrap

import jax
import numpy as np
import pytest

from jqmc import atomic_orbital
from jqmc import coulomb_potential as coulomb
from jqmc._precision import get_tolerance_min
from jqmc.determinant import compute_geminal_all_elements
from jqmc.hamiltonians import compute_local_energy, compute_local_energy_fast
from tests.test_jqmc_kernel._complex_ecp_reference import complex_local_energy_reference, ecp_terms, kinetic_value
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_jqmc_kernel.test_periodic_invariants import _COND_MAX, _draw_well_conditioned_cfg, _ginv

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]
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
# The same builder stems as ``tests/test_rs_kernel/test_periodic_invariants._pattern_ham``
# (that module imports the native extension, so the mapping is restated here).
_PATTERN_STEMS = {
    "composite_twist_j12": "QEPROJ:pgto+pw:Li_k025:ECP",
    "pw_twist_j12": "QEPROJ:pw:Li_k025:ECP",
}
PATTERNS = tuple(_PATTERN_STEMS)


def _plane_wave_leaves(ham):
    """Every complex plane-wave AO leaf of both spin channels."""
    leaves = []
    gd = ham.wavefunction_data.geminal_data
    for spin in ("orb_data_up_spin", "orb_data_dn_spin"):
        aos = getattr(gd, spin).aos_data
        for leaf in getattr(aos, "children", (aos,)):
            if isinstance(leaf, atomic_orbital.AOs_plane_wave_complex_data):
                leaves.append(leaf)
    return leaves


def case(pattern):
    """The unchanged fixture, the seed-7 image-offset configuration and the identity mesh rotation."""
    ham = _build_ham_for_cell(_PATTERN_STEMS[pattern], True, None, "SD")
    up, down = _draw_well_conditioned_cfg(ham, seed=7)
    G = np.asarray(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, up, down))
    cond_G = np.linalg.cond(G)
    assert cond_G < _COND_MAX
    leaves = _plane_wave_leaves(ham)
    assert len(leaves) == 2 and all(np.any(np.asarray(leaf.k_point) != 0) for leaf in leaves)
    return ham, up, down, np.eye(3)


def require_live_twist(ham, up, down, RT):
    """The twist must reach both the kinetic energy and the nonlocal ECP as imaginary parts."""
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    full_terms = ecp_terms(ham, up, down, RT)[-1]
    assert np.all(np.isfinite(full_terms))
    assert np.max(abs(np.imag(full_terms))) > atol + rtol * np.max(abs(np.real(full_terms)))
    kinetic = complex(kinetic_value(ham.wavefunction_data, up, down))
    assert np.isfinite(kinetic)
    assert abs(kinetic.imag) > atol + rtol * abs(kinetic.real)


def _compare(actual, reference):
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("route", ["standard", "fast"])
@pytest.mark.parametrize("pattern", PATTERNS)
def test_pw_twist_vmc_energy_matches_independent_absolute_value(pattern, route):
    """composite_twist_j12 / pw_twist_j12 VMC energy equals the independent complex H Psi / Psi.

    QE Li_k025 (k = (1/4, 1/4, 1/4)) projected as in the periodic-invariant
    patterns (composite: windowed Cartesian pGTO plus 40 plane waves; pw: 48
    plane waves), time-reversal SD, J1 + J2, Li ccECP; one up and one down
    electron at the seed-7 image-offset configuration (PCG64(700000+attempt),
    at most 300 proposals, cond(G) < 1e4), identity rotation, nearest nucleus,
    twelve-point quadrature. The reference rebuilds every plane wave from the
    stored Miller triples and fractional twist, every pGTO primitive with its
    integer-image Bloch phase, the determinant, J1/J2, the complex Hessian
    kinetic energy, the NumPy Ewald sum and the explicit nonlocal quadrature;
    no production AO, Jastrow, ECP or energy evaluator supplies it. Shared
    inputs are the Miller triples, twist, primitive data, image indices and MO
    coefficients. The standard (fresh) and fast (reused inverse) consumers are
    each compared in their own precision mode at the existing strict zone
    policy; the imaginary kinetic and ECP parts must be active.
    """
    ham, up, down, RT = case(pattern)
    require_live_twist(ham, up, down, RT)
    expected = complex_local_energy_reference(ham, up, down, RT)
    actual = (
        compute_local_energy(ham, up, down, RT)
        if route == "standard"
        else compute_local_energy_fast(ham, up, down, RT, _ginv(ham, up, down))
    )
    _compare(actual, expected)


@pytest.mark.parametrize("fault", ["omit-plane-wave-twist", "force-real-ecp"])
@pytest.mark.parametrize("pattern", PATTERNS)
def test_pw_twist_absolute_value_rejects_production_fault(pattern, fault, monkeypatch):
    """Dropping the plane-wave twist or the imaginary ECP part breaks the absolute anchor.

    The unchanged fixture and independent reference must first pass with the
    fast VMC consumer. ``omit-plane-wave-twist`` makes the production complex
    plane-wave wave vector G instead of G + k (values, gradients and
    Laplacians); ``force-real-ecp`` applies the fixed-phase real projection in
    the shared fast nonlocal ECP kernel. The reference keeps the stored twist
    and the unprojected quadrature, so each fault must fail the same strict
    comparison in either precision mode.
    """
    ham, up, down, RT = case(pattern)
    reference = complex_local_energy_reference(ham, up, down, RT)
    inverse = _ginv(ham, up, down)
    _compare(compute_local_energy_fast(ham, up, down, RT, inverse), reference)
    if fault == "omit-plane-wave-twist":
        cls = atomic_orbital.AOs_plane_wave_complex_data
        original = cls._gk_vectors_jnp

        def untwisted(self):
            return original.fget(self.replace(k_point=(0.0, 0.0, 0.0)))

        monkeypatch.setattr(cls, "_gk_vectors_jnp", property(untwisted))
    else:
        name = "_compute_ecp_non_local_parts_nearest_neighbors_fast_update_impl"
        source = textwrap.dedent(inspect.getsource(getattr(coulomb, name)))
        old = "if _is_complex_wf and project_real:"
        assert source.count(old) == 1
        namespace = dict(vars(coulomb))
        exec(compile(ast.parse(source.replace(old, "if _is_complex_wf:")), "<pw-twist-ecp-fault>", "exec"), namespace)
        monkeypatch.setattr(coulomb, name, namespace[name])
    jax.clear_caches()
    try:
        changed = compute_local_energy_fast(ham, up, down, RT, inverse)
    finally:
        # Do not leave the faulty traces in the compilation cache for later tests.
        monkeypatch.undo()
        jax.clear_caches()
    assert np.isfinite(changed)
    with pytest.raises(AssertionError):
        _compare(changed, reference)

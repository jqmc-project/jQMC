"""Conventional molecular Jastrow cusp normalization against independent radial formulas."""

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
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import jastrow_factor as jastrow
from jqmc._precision import get_dtype_np, get_tolerance_min
from jqmc.structure import Structure_data

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _radial_samples():
    """Nine fixed noncoincident positions; no seed, rejection or determinant condition selection."""
    dtype = get_dtype_np("jastrow_eval")
    radii = np.asarray([1e-7, 1e-5, 1e-3], dtype=dtype)
    directions = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=dtype)
    return radii, directions


def _radial_decay(form, b, scale, radius):
    """Independent same-mode slope envelope with unit right limit, without production helpers."""
    dtype = get_dtype_np("jastrow_eval")
    x = dtype(b) * dtype(scale) * dtype(radius)
    if form == "exp":
        return np.exp(-x)
    if form == "pade":
        return dtype(1) / (dtype(1) + x) ** 2
    raise ValueError(form)


@pytest.mark.parametrize("form", ["exp", "pade"])
@pytest.mark.parametrize("charge", [1, 8])
@pytest.mark.parametrize("b", [0.5, 1.25])
@pytest.mark.parametrize("spin", ["up", "down"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_j1_ae_radial_profile_has_physical_cusp(form, charge, b, spin):
    """AE molecular J1 has the -Z cusp normalization in both exp and Pade radial profiles.

    One synthetic H or O nucleus is at the origin, with zero removed core
    electrons and one up or one down electron. No TREXIO, orbitals, determinant
    or cond(G) selection is involved. Probe distances 1e-7, 1e-5 and 1e-3 Bohr
    along +x, -y and +z, with b=0.5 or 1.25 and no random configurations.

    Independently differentiate -A*u(c*r), where A=(2Z)^(3/4), c=(2Z)^(1/4).
    Its radial derivative is -Z*exp(-b*c*r) or -Z/(1+b*c*r)^2. Compare both
    AD of the actual forward kernel and the actual analytical gradient to
    that closed form at every positive distance and direction. This samples
    the cusp-normalized profile, not a finite-difference or an AD at the
    nondifferentiable Cartesian origin. NumPy uses the current Jastrow zone
    dtype and both modes retain existing strict element-wise tolerances.
    ECP, periodic geometry and complete-wavefunction cusp are outside scope.
    """
    structure = Structure_data(
        pbc_flag=False,
        positions=((0.0, 0.0, 0.0),),
        atomic_numbers=(charge,),
        element_symbols=("H" if charge == 1 else "O",),
        atomic_labels=("nucleus",),
    )
    data = jastrow.Jastrow_one_body_data(
        jastrow_1b_param=b, jastrow_1b_type=form, structure_data=structure, core_electrons=(0,)
    )
    data.sanity_check()
    radii, directions = _radial_samples()
    dtype = get_dtype_np("jastrow_eval")
    scale = dtype(2 * charge) ** dtype(0.25)
    atol, rtol = get_tolerance_min(("jastrow_eval", "jastrow_grad_lap"), "strict")
    for radius_index in range(3):
        radius = radii[radius_index]
        expected = -dtype(charge) * _radial_decay(form, b, scale, radius)
        for direction_index in range(3):
            sample_label = str(3 * radius_index + direction_index)
            direction = directions[direction_index]
            position = jnp.asarray((radius * direction)[None, :])
            empty = jnp.empty((0, 3), dtype=position.dtype)
            r_up_carts, r_dn_carts = (position, empty) if spin == "up" else (empty, position)
            ad_gradients = jax.grad(jastrow.compute_Jastrow_one_body, argnums=(1, 2))(data, r_up_carts, r_dn_carts)
            analytical_gradients = jastrow.compute_grads_and_laplacian_Jastrow_one_body(data, r_up_carts, r_dn_carts)
            index = 0 if spin == "up" else 1
            ad_slope = np.dot(np.asarray(ad_gradients[index])[0], direction)
            analytical_slope = np.dot(np.asarray(analytical_gradients[index])[0], direction)
            np.testing.assert_allclose(ad_slope, expected, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label)
            np.testing.assert_allclose(analytical_slope, expected, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label)


@pytest.mark.parametrize("form", ["exp", "pade"])
@pytest.mark.parametrize("b", [0.5, 1.25])
@pytest.mark.parametrize("spin_pair", ["up-down", "up-up", "down-down"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_j2_radial_profile_has_antiparallel_cusp_and_spin_independent_form(form, b, spin_pair):
    """Conventional J2 enforces the opposite-spin half-cusp and uses the same profile for equal spins.

    Use exactly two synthetic electrons, one fixed at the origin and the
    other at 1e-7, 1e-5 or 1e-3 Bohr along +x, -y or +z. The three sectors
    are up-down (1,1), up-up (2,0), down-down (0,2), with exp/Pade and b=0.5
    or 1.25. There are no nuclei, files, determinant, cond(G) limit or RNG.

    Compare the moved electron's radial derivative from actual forward AD
    and analytical gradients to independent same-mode NumPy formulas:
    exp(-b*r)/2 or 1/[2*(1+b*r)^2]. These have right limit 1/2. This is the
    physical opposite-spin cusp, but deliberately NOT the exact same-spin
    1/4 cusp: the established conventional ansatz is spin-independent.
    No FD, statistical acceptance or full-versus-mixed comparison is used.
    PBC, DTN, Rust and complete-wavefunction cusp are outside this test.
    """
    data = jastrow.Jastrow_two_body_data(jastrow_2b_param=b, jastrow_2b_type=form)
    data.sanity_check()
    radii, directions = _radial_samples()
    dtype = get_dtype_np("jastrow_eval")
    atol, rtol = get_tolerance_min(("jastrow_eval", "jastrow_grad_lap"), "strict")
    for radius_index in range(3):
        radius = radii[radius_index]
        expected = dtype(0.5) * _radial_decay(form, b, 1, radius)
        for direction_index in range(3):
            sample_label = str(3 * radius_index + direction_index)
            direction = directions[direction_index]
            position = jnp.asarray((radius * direction)[None, :])
            origin = jnp.zeros((1, 3), dtype=position.dtype)
            empty = jnp.empty((0, 3), dtype=position.dtype)
            if spin_pair == "up-down":
                r_up_carts, r_dn_carts = position, origin
            elif spin_pair == "up-up":
                r_up_carts, r_dn_carts = jnp.concatenate((position, origin)), empty
            else:
                r_up_carts, r_dn_carts = empty, jnp.concatenate((position, origin))
            ad_gradients = jax.grad(jastrow.compute_Jastrow_two_body, argnums=(1, 2))(data, r_up_carts, r_dn_carts)
            analytical_gradients = jastrow.compute_grads_and_laplacian_Jastrow_two_body(data, r_up_carts, r_dn_carts)
            index = 1 if spin_pair == "down-down" else 0
            ad_slope = np.dot(np.asarray(ad_gradients[index])[0], direction)
            analytical_slope = np.dot(np.asarray(analytical_gradients[index])[0], direction)
            np.testing.assert_allclose(ad_slope, expected, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label)
            np.testing.assert_allclose(analytical_slope, expected, atol=atol, rtol=rtol, equal_nan=False, err_msg=sample_label)


@pytest.mark.parametrize("part", ["j1", "j2"])
@pytest.mark.parametrize("fault", ["sign", "half"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_cusp_profile_verdict_rejects_wrong_forward_normalization(monkeypatch, part, fault):
    """A passing physical profile rejects a sign or half-factor defect in the actual forward route."""
    if part == "j1":
        test_j1_ae_radial_profile_has_physical_cusp("exp", 8, 0.5, "up")
        name = "compute_Jastrow_one_body"
    else:
        test_j2_radial_profile_has_antiparallel_cusp_and_spin_independent_form("pade", 0.5, "up-down")
        name = "compute_Jastrow_two_body"
    original = getattr(jastrow, name)
    factor = -1 if fault == "sign" else 0.5

    def corrupted(*args, **kwargs):
        return factor * original(*args, **kwargs)

    monkeypatch.setattr(jastrow, name, corrupted)
    with pytest.raises(AssertionError):
        if part == "j1":
            test_j1_ae_radial_profile_has_physical_cusp("exp", 8, 0.5, "up")
        else:
            test_j2_radial_profile_has_antiparallel_cusp_and_spin_independent_form("pade", 0.5, "up-down")

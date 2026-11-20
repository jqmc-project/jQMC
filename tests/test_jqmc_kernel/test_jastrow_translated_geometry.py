"""Fixed translated near-nucleus J1 derivative profiles."""

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

import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import jastrow_factor as jastrow
from jqmc._precision import get_dtype_np, get_tolerance_min
from jqmc.structure import Structure_data

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.mark.parametrize("form", ["exp", "pade"])
@pytest.mark.parametrize("boundary", ["molecular", "cubic"])
@pytest.mark.parametrize("spin", ["up", "down"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_j1_translated_near_nucleus_derivatives(form, boundary, spin):
    """J1 keeps the physical displacement before zone rounding at a translated AE nucleus.

    One synthetic H nucleus lies at (0.2,0.3,0.4) Bohr. Three same-spin
    electrons are at displacements (1,2,3), (-3,1,2), (2,-3,1) times 1e-7
    Bohr; the other spin sector is empty. Use exp/Pade with b=1, open
    geometry or an 8-Bohr cubic cell. No files, RNG, determinant, cond(G)
    screening, finite differences or stochastic acceptance is involved.

    Independent NumPy radial/chain-rule formulas evaluate gradients and
    Laplacians in the active Jastrow zone. In the cubic case the reference
    differentiates the exact sine distance, not its small-radius limit.
    The same fp64 geometry input is differenced before the zone cast on
    both sides. Actual analytical and autodiff helpers must independently
    match every component with the existing strict zone policy, including
    mixed mode; this is not a full-versus-mixed comparison. Oblique-cell
    metrics, ECP and complete-wavefunction cusps are outside scope.
    """
    center = np.asarray([0.2, 0.3, 0.4], dtype=np.float64)
    points = center + np.asarray([[1, 2, 3], [-3, 1, 2], [2, -3, 1]], dtype=np.float64) * 1e-7
    pbc = boundary == "cubic"
    structure = Structure_data.build(
        positions=center[None],
        atomic_numbers=(1,),
        element_symbols=("H",),
        atomic_labels=("H1",),
        pbc_flag=pbc,
        vec_a=(8.0, 0.0, 0.0) if pbc else (),
        vec_b=(0.0, 8.0, 0.0) if pbc else (),
        vec_c=(0.0, 0.0, 8.0) if pbc else (),
    )
    data = jastrow.Jastrow_one_body_data.build(
        jastrow_1b_param=1.0,
        jastrow_1b_type=form,
        structure_data=structure,
        core_electrons=(0,),
    )
    dtype = get_dtype_np("jastrow_grad_lap")
    displacement = (points - center).astype(dtype)
    if pbc:
        angle = dtype(np.pi) * displacement / dtype(8)
        sine = np.sin(angle)
        scaled_sine = dtype(8) / dtype(np.pi) * sine
        cosine = np.cos(angle)
        radius = np.sqrt(np.sum(scaled_sine * scaled_sine, axis=1))
        distance_gradient = scaled_sine * cosine / radius[:, None]
        norm_squared = np.sum(distance_gradient * distance_gradient, axis=1)
        lap_distance = (np.sum(cosine * cosine - sine * sine, axis=1) - norm_squared) / radius
    else:
        radius = np.sqrt(np.sum(displacement * displacement, axis=1))
        distance_gradient = displacement / radius[:, None]
        norm_squared = np.ones_like(radius)
        lap_distance = dtype(2) / radius
    coefficient = dtype(2) ** dtype(0.25)
    if form == "exp":
        first = -np.exp(-coefficient * radius)
        second = -coefficient * first
    else:
        denominator = dtype(1) + coefficient * radius
        first = -dtype(1) / (denominator * denominator)
        second = dtype(2) * coefficient / denominator**3
    expected_gradient = first[:, None] * distance_gradient
    expected_laplacian = second * norm_squared + first * lap_distance
    r_up_carts, r_dn_carts = jnp.asarray(points), jnp.empty((0, 3), dtype=jnp.float64)
    if spin == "down":
        r_up_carts, r_dn_carts = r_dn_carts, r_up_carts
    index = 0 if spin == "up" else 1
    analytic = jastrow.compute_grads_and_laplacian_Jastrow_one_body(data, r_up_carts, r_dn_carts)
    automatic = jastrow._compute_grads_and_laplacian_Jastrow_one_body_auto(data, r_up_carts, r_dn_carts)
    analytical_gradient, analytical_laplacian = np.asarray(analytic[index]), np.asarray(analytic[index + 2])
    autodiff_gradient, autodiff_laplacian = np.asarray(automatic[index]), np.asarray(automatic[index + 2])
    atol, rtol = get_tolerance_min(["jastrow_eval", "jastrow_grad_lap"], "strict")
    np.testing.assert_allclose(analytical_gradient, expected_gradient, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(analytical_laplacian, expected_laplacian, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(autodiff_gradient, expected_gradient, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(autodiff_laplacian, expected_laplacian, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("route", ["analytical", "automatic"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_translated_j1_contract_rejects_coordinate_precast(monkeypatch, route):
    """A passing near-nucleus profile rejects rounding coordinates before subtraction."""
    test_j1_translated_near_nucleus_derivatives("exp", "molecular", "up")
    name = (
        "compute_grads_and_laplacian_Jastrow_one_body"
        if route == "analytical"
        else "_compute_grads_and_laplacian_Jastrow_one_body_auto"
    )
    original = getattr(jastrow, name)

    def precast(data, up, down):
        return original(data, up.astype(jnp.float32).astype(up.dtype), down.astype(jnp.float32).astype(down.dtype))

    monkeypatch.setattr(jastrow, name, precast)
    with pytest.raises(AssertionError):
        test_j1_translated_near_nucleus_derivatives("exp", "molecular", "up")

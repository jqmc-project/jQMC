"""Fixed-input nonlocal ECP references, independent of debug and ratio kernels."""

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

import math

import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_dtype_jnp, get_dtype_np, get_tolerance_min
from jqmc.atomic_orbital import AOs_sphe_gto_data
from jqmc.coulomb_potential import (
    Coulomb_potential_data,
    compute_ecp_non_local_parts_nearest_neighbors,
    compute_ecp_non_local_parts_nearest_neighbors_fast_update,
    compute_ecp_non_local_parts_nearest_neighbors_fast_update_holomorphic,
    octahedron_sym_mesh_Nv18,
)
from jqmc.determinant import SD_data, compute_geminal_all_elements
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data
from jqmc.molecular_orbital import MOs_data
from jqmc.structure import Structure_data
from jqmc.wavefunction import Wavefunction_data


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _rotation(rotation):
    if rotation == "identity":
        return np.eye(3)
    if rotation != "tilted":
        raise ValueError("unknown fixed rotation")
    a, b = 0.41, 0.27
    return np.asarray([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]) @ np.asarray(
        [[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]]
    )


def _quadrature_18():
    """Authored coordinates/weights, not read from the implementation's mesh."""
    q = 1 / np.sqrt(2)
    directions = np.asarray(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
            [q, q, 0],
            [q, -q, 0],
            [-q, q, 0],
            [-q, -q, 0],
            [q, 0, q],
            [q, 0, -q],
            [-q, 0, q],
            [-q, 0, -q],
            [0, q, q],
            [0, -q, q],
            [0, q, -q],
            [0, -q, -q],
        ]
    )
    return directions, np.asarray([1 / 30] * 6 + [1 / 15] * 12)


def _gaussian_orbital_ratio(proposed, current, centre, ell, magnetic, exponent, value_dtype):
    """Psi(proposed)/Psi(current) of one real solid-harmonic Gaussian SD, evaluated per zone.

    Authored from the closed form; no production AO, MO, determinant or ratio
    code is called. ``value_dtype`` is the dtype of the orbital values: the
    ``ao_eval`` dtype for the zone reference (float32 in mixed), or the fp64
    geometry dtype. The stored exponent is rounded to ``value_dtype`` (AO
    parameters are stored in the AO zone). The exponent argument, harmonic
    polynomial and normalization are formed from the fp64 geometry and rounded
    once to ``value_dtype``; the exponential and value products run in
    ``value_dtype``. The one-orbital MO/determinant ratio is taken in
    ``det_eval`` and returned in the ``coulomb`` dtype that weights the projector.
    """
    points = np.concatenate([np.asarray(current, dtype=float)[None, :], np.asarray(proposed, dtype=float)])
    x, y, z = (points - np.asarray(centre, dtype=points.dtype)).T
    radius_squared = x * x + y * y + z * z
    stored = np.asarray(exponent, dtype=value_dtype).astype(points.dtype)
    root3 = np.sqrt(3.0)
    harmonic = {
        (0, 0): np.ones_like(x),
        (1, -1): y,
        (1, 0): z,
        (1, 1): x,
        (2, -2): root3 * x * y,
        (2, -1): root3 * y * z,
        (2, 0): (3 * z * z - radius_squared) / 2,
        (2, 1): root3 * x * z,
        (2, 2): root3 * (x * x - y * y) / 2,
    }[(ell, magnetic)]
    normalization = (2 * stored / np.pi) ** 0.75 * (4 * stored) ** (ell / 2) / np.sqrt((1, 1, 3)[ell])
    values = (
        jnp.asarray(normalization, dtype=value_dtype)
        * jnp.exp(jnp.asarray(-stored * radius_squared, dtype=value_dtype))
        * jnp.asarray(harmonic, dtype=value_dtype)
    )
    assert values.dtype == jnp.dtype(value_dtype)
    determinants = np.asarray(values, dtype=get_dtype_np("det_eval"))
    return np.asarray(determinants[1:] / determinants[0], dtype=get_dtype_np("coulomb"))


def _projector_hamiltonian(ell, magnetic, *, displaced=False, channel=None):
    """Synthetic one-electron SD, explicit ECP terms and finite local potentials."""
    centres = np.asarray([[0.0, 0.0, 0.0], [1.5, 0.2, -0.1]]) if displaced else np.zeros((1, 3))
    n_nuclei = len(centres)
    structure = Structure_data.build(
        positions=centres,
        atomic_numbers=(1,) * n_nuclei,
        element_symbols=("H",) * n_nuclei,
        atomic_labels=tuple(f"H{i + 1}" for i in range(n_nuclei)),
        pbc_flag=False,
    )
    orbital = AOs_sphe_gto_data.build(
        structure_data=structure,
        nucleus_index=(int(displaced),),
        num_ao=1,
        num_ao_prim=1,
        orbital_indices=(0,),
        exponents=np.asarray([0.8]),
        coefficients=np.ones(1),
        angular_momentums=(ell,),
        magnetic_quantum_numbers=(magnetic,),
    )
    mos = MOs_data.build(aos_data=orbital, mo_coefficients=np.ones((1, 1)))
    geminal = SD_data.build(num_electron_up=1, num_electron_dn=0, mos_up=mos, mos_dn=mos, time_reversal=False)
    wavefunction = Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data())
    if displaced:
        coefficients = [0.0] * 7
        coefficients[channel] = (-1.0 if channel % 2 else 1.0) * (1 + channel / 5)
        exponents = [0.7 + k / 10 for k in range(7)]
        # Both local potentials include exp(-a r^2)/r to cancel -1/r.
        coulomb = Coulomb_potential_data(
            structure_data=structure,
            ecp_flag=True,
            z_cores=(0.0, 0.0),
            max_ang_mom_plus_1=(7, 0),
            num_ecps=9,
            nucleus_index=(0,) * 8 + (1,),
            ang_moms=tuple(range(8)) + (0,),
            coefficients=tuple(coefficients) + (1.0, 1.0),
            exponents=tuple(exponents) + (0.4, 0.4),
            powers=(2,) * 7 + (1, 1),
        )
    else:
        coulomb = Coulomb_potential_data(
            structure_data=structure,
            ecp_flag=True,
            z_cores=(0.0,),
            max_ang_mom_plus_1=(3,),
            num_ecps=4,
            nucleus_index=(0,) * 4,
            ang_moms=(0, 1, 2, 3),
            coefficients=(1.3, 2.1, -0.8, 1.0),
            exponents=(0.7, 0.9, 1.2, 0.4),
            powers=(0, 1, 2, 1),
        )
    coulomb.sanity_check()
    return Hamiltonian_data(structure_data=structure, coulomb_potential_data=coulomb, wavefunction_data=wavefunction)


@pytest.mark.parametrize("ell,magnetic", [(0, 0), (1, -1), (1, 0), (1, 1), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)])
@pytest.mark.parametrize("rotation", ["identity", "tilted"])
@pytest.mark.gdd_electron_configuration(
    mode="required",
    position_source="explicit",
    reason="One fixed electronic configuration; no sampling or conditioning rejection.",
)
def test_s_p_d_projector_has_analytic_radial_eigenvalue(ell, magnetic, rotation):
    """The s/p/d ECP projector equals its radial eigenvalue for every real harmonic.

    Synthetic parameter-set Hamiltonian, no TREXIO file: one pseudized H centre,
    one up electron and no down electrons, molecular SD/MO, one spherical Gaussian
    of the selected (ell, m), no Jastrow. A finite local ECP cancels the nuclear
    singularity. The fixed electron is (0.31, -0.27, 0.49); G is 1 by 1 with
    finite nonzero entry, so cond(G)=1 and no screen is applied. Nv=18 and both
    fixed rotations integrate products of s/p/d harmonics exactly. Full and mixed
    use their own Coulomb strict policy; no FD or statistical tolerance is used.

    Same-precision reference: the eigenvalue V_l(r) is an exact constant,
    but production reaches it through orbital values evaluated in ao_eval
    (float32 in mixed). The reference is therefore V_l(r) plus V_l(r) times the
    shift of the authored 18-point projection when its orbital ratios are
    evaluated per zone (_gaussian_orbital_ratio: values in ao_eval, ratio in
    det_eval, projector in coulomb) instead of in the fp64 geometry dtype. In
    full both projections run the same fp64 arithmetic, the shift is exactly 0
    and the reference is the unchanged analytic eigenvalue.
    """
    ham = _projector_hamiltonian(ell, magnetic)
    r_up = np.asarray([[0.31, -0.27, 0.49]])
    r_down = np.empty((0, 3))
    g = np.asarray(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, r_up, r_down))
    assert g.shape == (1, 1) and np.all(np.isfinite(g)) and g[0, 0] != 0
    cond_g = np.linalg.cond(g)
    assert cond_g == 1
    rotation_matrix = _rotation(rotation)
    result = compute_ecp_non_local_parts_nearest_neighbors(
        ham.coulomb_potential_data,
        ham.wavefunction_data,
        jnp.asarray(r_up),
        jnp.asarray(r_down),
        jnp.asarray(rotation_matrix),
        NN=1,
        Nv=18,
    )
    actual = np.asarray(result[-1])
    radius = np.sqrt(np.sum(r_up * r_up))
    eigenvalue = (1.3, 2.1, -0.8)[ell] * radius ** ((0, 1, 2)[ell] - 2) * np.exp(-(0.7, 0.9, 1.2)[ell] * radius**2)
    # The public RT argument rotates row vectors on the right.
    directions = _quadrature_18()[0] @ rotation_matrix
    weights = np.asarray(_quadrature_18()[1], dtype=get_dtype_np("coulomb"))
    legendre = np.polynomial.legendre.legval(directions @ (r_up[0] / radius), [0] * ell + [1])

    def projection(value_dtype):
        ratio = _gaussian_orbital_ratio(radius * directions, r_up[0], np.zeros(3), ell, magnetic, 0.8, value_dtype)
        return np.sum((2 * ell + 1) * weights * legendre * ratio)

    shift = projection(get_dtype_jnp("ao_eval")) - projection(r_up.dtype)
    reference = np.asarray(eigenvalue + eigenvalue * shift, dtype=get_dtype_np("coulomb"))
    assert np.isfinite(actual) and np.isfinite(reference)
    atol, rtol = get_tolerance_min(("coulomb",), "strict")
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)


_RADIAL_POWER_ELECTRONS = np.asarray([[0.31, -0.27, 0.49], [-0.37, 0.26, 0.58]])
_RADIAL_POWER_TERM = (1.3, 0.7)  # (c, a) of the l=0 non-local term


def _radial_power_hamiltonian(power):
    """One pseudized H centre, one up and one down electron in a nodeless s Gaussian.

    The ECP has a single non-local l=0 term ``(c, a, n) = (1.3, 0.7, power)`` and
    the local tag l=1. Each electron's quadrature points lie on its own sphere
    about the only nucleus, so the s-orbital ratio is 1 at every point and the
    l=0 projector reduces to the radial factor ``c r^(n-2) exp(-a r^2)``.
    """
    structure = Structure_data.build(
        positions=np.zeros((1, 3)), atomic_numbers=(1,), element_symbols=("H",), atomic_labels=("H1",), pbc_flag=False
    )
    orbital = AOs_sphe_gto_data.build(
        structure_data=structure,
        nucleus_index=(0,),
        num_ao=1,
        num_ao_prim=1,
        orbital_indices=(0,),
        exponents=np.asarray([0.8]),
        coefficients=np.ones(1),
        angular_momentums=(0,),
        magnetic_quantum_numbers=(0,),
    )
    mos = MOs_data.build(aos_data=orbital, mo_coefficients=np.ones((1, 1)))
    geminal = SD_data.build(num_electron_up=1, num_electron_dn=1, mos_up=mos, mos_dn=mos, time_reversal=False)
    wavefunction = Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data())
    coulomb = Coulomb_potential_data(
        structure_data=structure,
        ecp_flag=True,
        z_cores=(0.0,),
        max_ang_mom_plus_1=(1,),
        num_ecps=2,
        nucleus_index=(0, 0),
        ang_moms=(0, 1),
        coefficients=(_RADIAL_POWER_TERM[0], 1.0),
        exponents=(_RADIAL_POWER_TERM[1], 0.4),
        powers=(power, 1),
    )
    coulomb.sanity_check()
    return Hamiltonian_data(structure_data=structure, coulomb_potential_data=coulomb, wavefunction_data=wavefunction)


@pytest.mark.parametrize("power", [0, 1, 2, 3])
@pytest.mark.parametrize("route", ["nearest", "fast_update", "fast_update_holomorphic"])
@pytest.mark.parametrize("rotation", ["identity", "tilted"])
@pytest.mark.gdd_electron_configuration(
    mode="required",
    position_source="explicit",
    reason="One fixed two-electron configuration; no sampling or conditioning rejection.",
)
def test_nonlocal_radial_power_matches_closed_form_on_every_route(power, route, rotation):
    """Every XLA non-local route evaluates c r^(n-2) exp(-a r^2) with the term's own power n.

    Synthetic parameter-set Hamiltonian (_radial_power_hamiltonian): one pseudized
    H centre at the origin, a nodeless s Gaussian SD with one up electron at
    (0.31, -0.27, 0.49) and one down electron at (-0.37, 0.26, 0.58), no Jastrow,
    a single non-local l=0 term with power n in {0, 1, 2, 3} (n = 2 is the ccECP
    control). Routes: the nearest-neighbour evaluator, and the fast-update kernel
    (real and holomorphic public wrappers) used by compute_ecp_coulomb_potential
    and the MCMC/GFMC fast paths, with A_old_inv from a NumPy inverse of G.
    Nv=18 and the fixed rotations are those of the s/p/d eigenvalue test.

    Reference: sum over both electrons of the closed-form radial factor, shifted
    by the authored 18-point s projection evaluated per zone (orbital
    values in ao_eval, ratio in det_eval) minus the fp64-geometry projection;
    the shift is exactly 0 in full. Coulomb strict policy in full and mixed. The
    closed form at n != 2 differs from the n = 2 value by more than the tolerance
    (asserted), so a route that ignores the stored power fails.
    """
    ham = _radial_power_hamiltonian(power)
    r_up = _RADIAL_POWER_ELECTRONS[:1]
    r_down = _RADIAL_POWER_ELECTRONS[1:]
    g = np.asarray(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, r_up, r_down))
    assert g.shape == (1, 1) and np.all(np.isfinite(g)) and g[0, 0] != 0
    assert np.linalg.cond(g) == 1
    rotation_matrix = _rotation(rotation)
    arguments = (
        ham.coulomb_potential_data,
        ham.wavefunction_data,
        jnp.asarray(r_up),
        jnp.asarray(r_down),
        jnp.asarray(rotation_matrix),
    )
    if route == "nearest":
        result = compute_ecp_non_local_parts_nearest_neighbors(*arguments, NN=1, Nv=18)
    else:
        fast_update = {
            "fast_update": compute_ecp_non_local_parts_nearest_neighbors_fast_update,
            "fast_update_holomorphic": compute_ecp_non_local_parts_nearest_neighbors_fast_update_holomorphic,
        }[route]
        result = fast_update(*arguments, A_old_inv=jnp.asarray(np.linalg.inv(g)), NN=1, Nv=18)
    actual = np.real(np.asarray(result[-1]))

    coefficient, exponent = _RADIAL_POWER_TERM
    directions = _quadrature_18()[0] @ rotation_matrix
    weights = np.asarray(_quadrature_18()[1], dtype=get_dtype_np("coulomb"))

    def radial_factor(radius, n):
        return coefficient * radius ** (n - 2) * np.exp(-exponent * radius**2)

    reference = 0.0
    for electron in _RADIAL_POWER_ELECTRONS:
        radius = np.linalg.norm(electron)

        def projection(value_dtype, electron=electron, radius=radius):
            ratio = _gaussian_orbital_ratio(radius * directions, electron, np.zeros(3), 0, 0, 0.8, value_dtype)
            return np.sum(weights * ratio)

        shift = projection(get_dtype_jnp("ao_eval")) - projection(electron.dtype)
        reference += radial_factor(radius, power) * (1 + shift)
    reference = np.asarray(reference, dtype=get_dtype_np("coulomb"))
    atol, rtol = get_tolerance_min(("coulomb",), "strict")
    if power != 2:
        control = sum(radial_factor(np.linalg.norm(electron), 2) for electron in _RADIAL_POWER_ELECTRONS)
        assert abs(reference - control) > atol + rtol * abs(reference), "the power must change the radial factor"
    assert np.isfinite(actual) and np.isfinite(reference)
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("channel", [0, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("position", ["first", "second"])
@pytest.mark.parametrize("rotation", ["identity", "tilted"])
@pytest.mark.gdd_electron_configuration(
    mode="required", position_source="explicit", reason="Each named position is fixed; no sampling or conditioning rejection."
)
def test_each_ecp_channel_matches_independent_discrete_projection(channel, position, rotation):
    """Each nonzero l=0..6 channel matches an independent per-grid Gaussian projection.

    Synthetic molecular Hamiltonian, no artifact: two pseudized H centres at
    (0,0,0) and (1.5,0.2,-0.1), one up electron/down empty, SD/MO of a normalized
    s Gaussian (alpha=0.8) centred on the second nucleus, no Jastrow. Positions
    are first=(0.31,-0.27,0.49) or second=(-0.37,0.26,0.58); no conditioning
    rejection. G is finite, nonzero and 1 by 1; cond(G)=1 is asserted.
    NN=1 selects the first nucleus. Its selected channel alone is
    nonzero. The closed-form Gaussian ratio uses no production wavefunction
    evaluator and is evaluated where production evaluates it:
    orbital values in ao_eval (float32 in mixed), the SD ratio in det_eval,
    and Legendre, weights and radial factor in coulomb (_gaussian_orbital_ratio).
    Legendre evaluation and the authored 18-point mesh are independent.
    Per-grid values and the sum must both agree under the existing Coulomb
    strict policy, in full/mixed separately. This guarantees the discrete rule,
    not continuous high-l integration accuracy.
    """
    ham = _projector_hamiltonian(0, 0, displaced=True, channel=channel)
    point = (0.31, -0.27, 0.49) if position == "first" else (-0.37, 0.26, 0.58)
    r_up = np.asarray([point])
    r_down = np.empty((0, 3))
    g = np.asarray(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, r_up, r_down))
    assert g.shape == (1, 1) and np.all(np.isfinite(g)) and g[0, 0] != 0
    cond_g = np.linalg.cond(g)
    assert cond_g == 1
    rotation_matrix = _rotation(rotation)
    result = compute_ecp_non_local_parts_nearest_neighbors(
        ham.coulomb_potential_data,
        ham.wavefunction_data,
        jnp.asarray(r_up),
        jnp.asarray(r_down),
        jnp.asarray(rotation_matrix),
        NN=1,
        Nv=18,
    )
    actual_grid = np.asarray(result[-2])
    actual_sum = np.asarray(result[-1])
    directions, weights = _quadrature_18()
    # The public RT argument rotates row vectors on the right.
    directions = directions @ rotation_matrix
    radius = np.linalg.norm(r_up[0])
    proposed = radius * directions
    centre = np.asarray([1.5, 0.2, -0.1])
    ratio = _gaussian_orbital_ratio(proposed, r_up[0], centre, 0, 0, 0.8, get_dtype_jnp("ao_eval"))
    cosine = directions @ (r_up[0] / radius)
    legendre = np.polynomial.legendre.legval(cosine, [0] * channel + [1])
    coefficient = (-1.0 if channel % 2 else 1.0) * (1 + channel / 5)
    radial = coefficient * np.exp(-(0.7 + channel / 10) * radius**2)
    reference_grid = np.asarray(radial * (2 * channel + 1) * weights * legendre * ratio, dtype=get_dtype_np("coulomb"))
    reference_sum = np.sum(reference_grid)
    assert np.all(np.isfinite(actual_grid)) and np.isfinite(actual_sum)
    atol, rtol = get_tolerance_min(("coulomb",), "strict")
    assert np.any(np.abs(reference_grid) > atol), "the selected ECP channel must be excited"
    np.testing.assert_allclose(actual_grid, reference_grid, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(actual_sum, reference_sum, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize(
    "powers",
    [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 0, 3),
        (0, 0, 4),
        (0, 0, 5),
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, 2),
        (0, 1, 3),
        (0, 1, 4),
        (0, 2, 0),
        (0, 2, 1),
        (0, 2, 2),
        (0, 2, 3),
        (0, 3, 0),
        (0, 3, 1),
        (0, 3, 2),
        (0, 4, 0),
        (0, 4, 1),
        (0, 5, 0),
        (1, 0, 0),
        (1, 0, 1),
        (1, 0, 2),
        (1, 0, 3),
        (1, 0, 4),
        (1, 1, 0),
        (1, 1, 1),
        (1, 1, 2),
        (1, 1, 3),
        (1, 2, 0),
        (1, 2, 1),
        (1, 2, 2),
        (1, 3, 0),
        (1, 3, 1),
        (1, 4, 0),
        (2, 0, 0),
        (2, 0, 1),
        (2, 0, 2),
        (2, 0, 3),
        (2, 1, 0),
        (2, 1, 1),
        (2, 1, 2),
        (2, 2, 0),
        (2, 2, 1),
        (2, 3, 0),
        (3, 0, 0),
        (3, 0, 1),
        (3, 0, 2),
        (3, 1, 0),
        (3, 1, 1),
        (3, 2, 0),
        (4, 0, 0),
        (4, 0, 1),
        (4, 1, 0),
        (5, 0, 0),
    ],
)
def test_octahedral_rule_integrates_monomials_through_degree_five(powers):
    """The production 18-point rule reproduces every spherical monomial through degree 5.

    No wavefunction or electron configuration is involved. The production mesh
    points/weights are the input. The reference is the normalized sphere integral:
    odd powers integrate to zero; even powers follow the double-factorial formula.
    This is the exactness condition needed for s/p/d projector eigenvalues, not a
    claim about arbitrary high-degree functions. Both precision lanes use the
    existing Coulomb strict policy without Monte Carlo error bars.
    """
    points = np.asarray(octahedron_sym_mesh_Nv18.grid_points)
    weights = np.asarray(octahedron_sym_mesh_Nv18.weights)
    actual = np.sum(weights * np.prod(points ** np.asarray(powers), axis=1))
    if any(power % 2 for power in powers):
        reference = 0.0
    else:
        numerator = math.prod(math.factorial(power) // (2 ** (power // 2) * math.factorial(power // 2)) for power in powers)
        degree = sum(powers)
        denominator = math.factorial(degree + 1) // (2 ** (degree // 2) * math.factorial(degree // 2))
        reference = numerator / denominator
    atol, rtol = get_tolerance_min(("coulomb",), "strict")
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)

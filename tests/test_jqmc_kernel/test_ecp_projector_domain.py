"""XLA ECP capability boundaries are checked before numerical projection."""

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

from jqmc import coulomb_potential as cp
from jqmc._precision import get_dtype_np, get_tolerance_min
from jqmc.determinant import compute_geminal_all_elements
from jqmc.hamiltonians import Hamiltonian_data
from tests.test_jqmc_kernel.test_ecp_projector_reference import _projector_hamiltonian


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _domain_hamiltonian(lmax, coefficient):
    """Two nuclei with different local tags; only the highest projector is active."""
    ham = _projector_hamiltonian(0, 0, displaced=True, channel=0)
    count = lmax + 3
    coulomb = ham.coulomb_potential_data.replace(
        max_ang_mom_plus_1=(lmax + 1, 0),
        num_ecps=count,
        nucleus_index=(0,) * (count - 1) + (1,),
        ang_moms=tuple(range(lmax + 2)) + (0,),
        coefficients=(0.0,) * lmax + (coefficient, 1.0, 1.0),
        exponents=(0.7,) * (lmax + 1) + (0.4, 0.4),
        powers=(2,) * (lmax + 1) + (1, 1),
    )
    return Hamiltonian_data(
        structure_data=ham.structure_data,
        coulomb_potential_data=coulomb,
        wavefunction_data=ham.wavefunction_data,
    )


def _evaluate(route, ham):
    """Use real public entries and a fixed, nonsingular one-electron SD."""
    r_up_carts = jnp.asarray([[0.31, -0.27, 0.49]])
    r_dn_carts = jnp.empty((0, 3))
    arguments = (ham.coulomb_potential_data, ham.wavefunction_data, r_up_carts, r_dn_carts)
    if route == "nearest-neighbor":
        return cp.compute_ecp_non_local_parts_nearest_neighbors(*arguments, jnp.eye(3), NN=2, Nv=18)
    if route == "holomorphic":
        return cp.compute_ecp_non_local_parts_nearest_neighbors_holomorphic(*arguments, jnp.eye(3), NN=2, Nv=18)
    if route == "fast-update":
        inverse = jnp.linalg.inv(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, r_up_carts, r_dn_carts))
        return cp.compute_ecp_non_local_parts_nearest_neighbors_fast_update(*arguments, jnp.eye(3), inverse, NN=2, Nv=18)
    if route == "all-pairs":
        return cp.compute_ecp_non_local_parts_all_pairs(*arguments, jnp.eye(3), Nv=18)
    if route == "provided-quadrature":
        mesh = cp.octahedron_sym_mesh_Nv18
        return cp.compute_ecp_non_local_part_all_pairs_jax_weights_grid_points(*arguments, mesh.weights, mesh.grid_points)
    raise ValueError(f"unknown ECP entry {route}")


@pytest.mark.parametrize("lmax", [6, 7, 8])
@pytest.mark.parametrize("coefficient", [0.0, 1.0])
@pytest.mark.gdd_electron_configuration(mode="not-required", reason="Construction and exact ECP table metadata only.")
def test_domain_fixture_realizes_requested_channels(lmax, coefficient):
    """The built ECP table realizes every declared angular-channel and coefficient case.

    The actual synthetic Hamiltonian must contain contiguous non-local tags
    0..lmax and distinct per-nucleus local tags. Only the highest non-local term
    carries the selected coefficient (zero or one); both local terms remain one.
    Exact array equality checks the built object, not labels or collected counts.
    This test inspects construction only and does not evaluate electron positions.
    """
    ham = _domain_hamiltonian(lmax, coefficient)
    coulomb = ham.coulomb_potential_data
    expected_tags = np.asarray(list(range(lmax + 2)) + [0])
    expected_coefficients = np.asarray([0.0] * lmax + [coefficient, 1.0, 1.0])
    expected_local_tags = np.asarray([lmax + 1, 0])
    np.testing.assert_array_equal(np.asarray(coulomb.ang_moms), expected_tags)
    np.testing.assert_array_equal(np.asarray(coulomb.coefficients), expected_coefficients)
    np.testing.assert_array_equal(np.asarray(coulomb.max_ang_mom_plus_1), expected_local_tags)


@pytest.mark.parametrize("route", ["nearest-neighbor", "holomorphic", "fast-update", "all-pairs", "provided-quadrature"])
@pytest.mark.parametrize("lmax,coefficient", [(7, 1.0), (7, 0.0), (8, 1.0), (8, 0.0)])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_xla_rejects_unsupported_projector_before_evaluation(route, lmax, coefficient):
    """Every XLA ECP entry rejects l=7/8, including zero-coefficient projectors.

    The synthetic molecular SD has one up electron, no down electrons, one s
    Gaussian, no Jastrow, and two nuclei with distinct local tags. No TREXIO or
    random positions are used. The nonzero 1x1 geminal has cond(G)=1 without
    screening. A complete channel table passes backend-neutral sanity checking;
    XLA must raise NotImplementedError rather than return NaN. This is a
    capability-rejection check, not a numerical-tolerance comparison.
    """
    ham = _domain_hamiltonian(lmax, coefficient)
    ham.coulomb_potential_data.sanity_check()
    with pytest.raises(NotImplementedError, match=r"XLA.*non-local.*l=0\.\.6"):
        _evaluate(route, ham)


@pytest.mark.parametrize("ell", [0, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("cosine", [-1.0, -0.75, -0.25, 0.0, 0.25, 0.75, 1.0])
@pytest.mark.gdd_electron_configuration(mode="not-required", reason="Scalar polynomial argument, not electron coordinates.")
def test_tabulated_legendre_matches_independent_rodrigues_formula(ell, cosine):
    """All seven XLA Legendre arms match an independent Rodrigues construction.

    Fixed scalar cosine inputs cover both endpoints, zero and interior values.
    The reference differentiates (x*x-1)**ell as a NumPy Polynomial and divides
    by 2**ell * ell!, rather than copying tabulated coefficients or a debug/Rust
    recurrence. Both values use this test's Coulomb precision and central strict
    tolerance. No wavefunction, electronic positions, AO approximation, angular
    quadrature or full-versus-mixed comparison is involved.
    """
    argument = np.asarray(cosine, dtype=get_dtype_np("coulomb"))
    polynomial = (np.polynomial.Polynomial([-1, 0, 1]) ** ell).deriv(ell) / (2**ell * math.factorial(ell))
    reference = np.asarray(polynomial(argument), dtype=get_dtype_np("coulomb"))
    actual = np.asarray(cp._legendre_tablated(ell, jnp.asarray(argument)))
    atol, rtol = get_tolerance_min(["coulomb"], "strict")
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol)


@pytest.mark.parametrize("route", ["nearest-neighbor", "holomorphic", "fast-update", "all-pairs", "provided-quadrature"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_xla_accepts_l6_with_local_tag7(route):
    """Non-local l=6 stays evaluable when its local tag is 7 and its neighbour's is 0.

    The same fixed synthetic one-electron SD as the rejection cases is used.
    All returned entries must be finite and the projector array nonempty.
    This boundary-acceptance check does not assert continuous high-l quadrature
    accuracy; independent discrete projector-reference tests own that claim.
    """
    ham = _domain_hamiltonian(6, 1.0)
    ham.coulomb_potential_data.sanity_check()
    result = _evaluate(route, ham)
    assert np.asarray(result[2]).size > 0
    if route == "provided-quadrature":
        assert np.asarray(result[0]).shape == (7, 1, 18, 1, 3)
        assert np.asarray(result[1]).shape == (7, 0, 18, 0, 3)
        assert np.asarray(result[2]).shape == (7, 1, 18)
        assert np.asarray(result[3]).shape == (7, 0, 18)
    else:
        points = 18 if route == "all-pairs" else 36
        assert np.asarray(result[0]).shape == (points, 1, 3)
        assert np.asarray(result[1]).shape == (points, 0, 3)
        assert np.asarray(result[2]).shape == (points,)
    for value in result:
        assert np.all(np.isfinite(np.asarray(value)))


@pytest.mark.parametrize("route", ["nearest-neighbor", "holomorphic", "fast-update", "all-pairs", "provided-quadrature"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_xla_rejects_declared_padding_above_supported_degree(route):
    """Declaring l=8 is rejected even if the table has no l=7/8 terms to inspect.

    The fixed one-electron SD is unchanged. The ECP declaration is deliberately
    inconsistent: actual non-local terms stop at l=6, but the local tag is 9.
    Direct replacement bypasses sanity_check; the XLA entry must still reject
    this metadata before padding unsupported zero-coefficient projectors.
    """
    ham = _domain_hamiltonian(6, 1.0)
    coulomb = ham.coulomb_potential_data.replace(
        max_ang_mom_plus_1=(9, 0),
        ang_moms=tuple(range(7)) + (9, 0),
    )
    ham = ham.replace(coulomb_potential_data=coulomb)
    with pytest.raises(NotImplementedError, match=r"XLA.*non-local.*l=0\.\.6"):
        _evaluate(route, ham)

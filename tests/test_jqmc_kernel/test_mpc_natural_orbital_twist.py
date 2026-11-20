"""MPC natural-orbital density of an AO-basis AGP at a complex twist.

``AGP_data.natural_orbital_density_coefficients_jnp`` factorises the AO pairing
matrix for the MPC one-body density. The AO geminal is bilinear, so with
``conj(C) S C^T = I`` and ``lambda = C_up^T diag(L) C_dn`` the matrix
``S_up^{1/2} lambda (S_dn^{1/2})^T`` has the singular values ``L``; the right
factor is a transpose, as in ``AGP_data.convert_from_AOs_to_MOs``. A Slater
determinant converted to the AO basis must therefore come back with occupations
exactly one on the occupied orbitals and zero elsewhere, and with the same
per-spin density as its MOs. At a complex twist ``S_dn = conj(S_up)`` is not
real and the adjoint form ``S_up^{1/2} lambda S_dn^{1/2}`` gives wrong
occupations; at Gamma the two forms coincide.
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

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min
from jqmc.atomic_orbital import compute_AOs, compute_overlap_matrix, is_complex_aos
from jqmc.determinant import AGP_data, SD_data, to_AGP
from jqmc.molecular_orbital import compute_MOs
from jqmc.trexio_wrapper import read_trexio_file

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_TREXIO_DIR = Path(__file__).resolve().parents[1] / "trexio_example_files"
# (fixture, MOs): the stored PySCF MOs, and for H4 also two random complex
# S-orthonormal MOs on the same leaf, whose occupied span is far from its complex
# conjugate (the stored k = 1/4 spans are nearly conjugation invariant, which hides
# a conjugated dn singular-vector factor below the strict tolerance).
_TWIST_CASES = [
    ("H4_pbc_ae_k025_cart.h5", "stored"),
    ("H2_pbc_ae_k025_sphe.h5", "stored"),
    ("H4_pbc_ae_k025_cart.h5", "random-complex"),
]
_GAMMA_FIXTURES = ["H4_pbc_ae_cart.h5", "H2_pbc_ae_sphe.h5"]


def _random_complex_slater(geminal):
    """Same leaves and electron counts, up MOs replaced by random complex S-orthonormal rows.

    ``C = Q^T (S^{-1/2})^T`` with a QR-orthonormal complex ``Q`` gives
    ``conj(C) S C^T = Q^H Q = I``; the dn channel is re-derived by time reversal.
    """
    mos_up = geminal.orb_data_up_spin
    overlap = np.asarray(compute_overlap_matrix(mos_up.aos_data))
    overlap = 0.5 * (overlap + overlap.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    inverse_sqrt = (eigenvectors / np.sqrt(eigenvalues)[None, :]) @ eigenvectors.conj().T
    rng = np.random.default_rng(41)
    shape = (overlap.shape[0], int(mos_up.num_mo))
    q_matrix = np.linalg.qr(rng.normal(size=shape) + 1j * rng.normal(size=shape))[0]
    coefficients = q_matrix.T @ inverse_sqrt.T
    return SD_data.build(
        num_electron_up=geminal.num_electron_up,
        num_electron_dn=geminal.num_electron_dn,
        mos_up=mos_up.replace(mo_coefficients=coefficients),
    )


def _slater_and_ao_agp(name, mos="stored"):
    """``(SD geminal, its AO-basis AGP)`` of a committed periodic TREXIO fixture."""
    geminal = read_trexio_file(str(_TREXIO_DIR / name), store_tuple=True)[4]
    if mos == "random-complex":
        geminal = _random_complex_slater(geminal)
    return geminal, AGP_data.convert_from_MOs_to_AOs(to_AGP(geminal))


def _hermitian_sqrt(overlap):
    overlap = 0.5 * (overlap + overlap.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    return (eigenvectors * np.sqrt(eigenvalues)[None, :]) @ eigenvectors.conj().T


def _paired_factor_singular_values(ao_agp, *, transpose_right):
    """Singular values of ``S_up^{1/2} lambda R`` with ``R = (S_dn^{1/2})^T`` or ``S_dn^{1/2}``."""
    sqrt_up = _hermitian_sqrt(np.asarray(compute_overlap_matrix(ao_agp.orb_data_up_spin)))
    sqrt_dn = _hermitian_sqrt(np.asarray(compute_overlap_matrix(ao_agp.orb_data_dn_spin)))
    paired = np.asarray(ao_agp.lambda_matrix)[:, : ao_agp.orb_num_dn]
    right = sqrt_dn.T if transpose_right else sqrt_dn
    return np.linalg.svd(sqrt_up @ paired @ right, compute_uv=False)


def _expected_occupations(geminal, size):
    expected = np.zeros(size, dtype=np.float64)
    expected[: int(geminal.num_electron_dn)] = 1.0
    return expected


def _density(aos_data, coefficients, occupations, r_carts):
    values = np.asarray(coefficients) @ np.asarray(compute_AOs(aos_data, jnp.asarray(r_carts)))
    return np.asarray(occupations) @ np.abs(values) ** 2


def _check_natural_orbitals(geminal, ao_agp, frozen_overlap):
    aos_up, coeff_up, occ_up, aos_dn, coeff_dn, occ_dn = ao_agp.natural_orbital_density_coefficients_jnp(
        frozen_overlap=frozen_overlap
    )
    atol, rtol = get_tolerance_min(("det_eval",), "strict")
    num_occupied = int(geminal.num_electron_dn)
    identity = np.eye(num_occupied)
    for occupations, coefficients, mos in (
        (occ_up, coeff_up, geminal.orb_data_up_spin),
        (occ_dn, coeff_dn, geminal.orb_data_dn_spin),
    ):
        occupations = np.asarray(occupations, dtype=np.float64)
        assert np.all(np.isfinite(occupations))
        np.testing.assert_allclose(occupations, _expected_occupations(geminal, occupations.size), atol=atol, rtol=rtol)
        # The occupied natural orbitals are S-orthonormal and span the stored MOs:
        # conj(P) S P^T = I and conj(P) S C^T is unitary (all singular values 1).
        overlap = np.asarray(compute_overlap_matrix(mos.aos_data))
        occupied = np.asarray(coefficients)[:num_occupied]
        stored = np.asarray(mos.mo_coefficients)
        np.testing.assert_allclose(occupied.conj() @ overlap @ occupied.T, identity, atol=atol, rtol=rtol)
        cross = np.linalg.svd(occupied.conj() @ overlap @ stored.T, compute_uv=False)
        np.testing.assert_allclose(cross, np.ones(num_occupied), atol=atol, rtol=rtol)

    rng = np.random.default_rng(29)
    cell = np.asarray(geminal.orb_data_up_spin.aos_data.structure_data.cell, dtype=np.float64)
    r_carts = rng.uniform(-0.5, 1.5, size=(16, 3)) @ cell
    atol, rtol = get_tolerance_min(("ao_eval", "mo_eval", "det_eval"), "strict")
    for aos_data, coefficients, occupations, mos in (
        (aos_up, coeff_up, occ_up, geminal.orb_data_up_spin),
        (aos_dn, coeff_dn, occ_dn, geminal.orb_data_dn_spin),
    ):
        mo_values = np.asarray(compute_MOs(mos, jnp.asarray(r_carts)))
        reference = np.sum(np.abs(mo_values) ** 2, axis=0)
        density = _density(aos_data, coefficients, occupations, r_carts)
        assert np.all(np.isfinite(density)) and np.all(np.isfinite(reference))
        np.testing.assert_allclose(density, reference, atol=atol, rtol=rtol)


@pytest.mark.parametrize("overlap_route", ["direct", "frozen"])
@pytest.mark.parametrize("trexio_file,mos", _TWIST_CASES)
def test_twist_ao_agp_natural_orbitals_recover_the_slater_determinant(trexio_file, mos, overlap_route):
    """At k = 1/4 the AO-AGP of a Slater determinant has occupations 1 (occupied) and 0.

    Complex Bloch pGTO leaves (Cartesian H4, spherical H2), up leaf at +k and dn leaf
    at -k, with the stored MOs and (H4) random complex S-orthonormal MOs. Occupations
    equal 1 / 0 within the strict det_eval policy, the occupied natural orbitals are
    S-orthonormal and span the stored MOs (all singular values of conj(P) S C^T are 1),
    and the per-spin MPC density from the natural orbitals equals the density of the stored MOs at 16
    points inside and outside the home cell, with the overlap computed directly or
    passed in as the hoisted ``frozen_overlap``.
    """
    geminal, ao_agp = _slater_and_ao_agp(trexio_file, mos)
    assert is_complex_aos(ao_agp.orb_data_up_spin)
    frozen_overlap = None
    if overlap_route == "frozen":
        frozen_overlap = (
            np.asarray(compute_overlap_matrix(ao_agp.orb_data_up_spin)),
            np.asarray(compute_overlap_matrix(ao_agp.orb_data_dn_spin)),
        )
    _check_natural_orbitals(geminal, ao_agp, frozen_overlap)


@pytest.mark.parametrize("trexio_file,mos", _TWIST_CASES)
def test_twist_adjoint_right_factor_is_rejected(trexio_file, mos):
    """The untransposed form ``S_up^{1/2} lambda S_dn^{1/2}`` fails the occupation check at the twist.

    The overlap is genuinely complex there; the transposed form reproduces the occupations
    within the strict det_eval policy and the adjoint form does not.
    """
    geminal, ao_agp = _slater_and_ao_agp(trexio_file, mos)
    atol, rtol = get_tolerance_min(("det_eval",), "strict")
    assert np.max(np.abs(np.imag(np.asarray(compute_overlap_matrix(ao_agp.orb_data_up_spin))))) > atol
    correct = _paired_factor_singular_values(ao_agp, transpose_right=True)
    adjoint = _paired_factor_singular_values(ao_agp, transpose_right=False)
    expected = _expected_occupations(geminal, correct.size)
    np.testing.assert_allclose(correct, expected, atol=atol, rtol=rtol)
    assert np.all(np.isfinite(adjoint))
    assert not np.allclose(adjoint, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("trexio_file", _GAMMA_FIXTURES)
def test_gamma_ao_agp_natural_orbitals_are_unchanged(trexio_file):
    """At Gamma the overlap is real, both forms agree, and the occupations are 1 / 0.

    Real pGTO leaves (Cartesian H4, spherical H2): the production occupations and
    densities satisfy the same checks as at the twist, and the transposed and adjoint
    singular values coincide within the strict det_eval policy, so the choice of form
    does not change the Gamma result.
    """
    geminal, ao_agp = _slater_and_ao_agp(trexio_file)
    assert not is_complex_aos(ao_agp.orb_data_up_spin)
    _check_natural_orbitals(geminal, ao_agp, None)
    atol, rtol = get_tolerance_min(("det_eval",), "strict")
    np.testing.assert_allclose(
        _paired_factor_singular_values(ao_agp, transpose_right=True),
        _paired_factor_singular_values(ao_agp, transpose_right=False),
        atol=atol,
        rtol=rtol,
    )

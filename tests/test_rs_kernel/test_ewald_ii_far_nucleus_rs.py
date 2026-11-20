"""Rust ion-ion Ewald kernels with a nucleus given several cells outside the home cell (issue #52).

Rust counterpart of ``tests/test_jqmc_kernel/test_ewald_ii_far_nucleus.py``. The Rust ii energy
(``jqmc_rs.compute_ewald_potential_ii``), force (``jqmc_rs.grad_Eii_wrt_R``, the kernel the GFMC /
MCMC periodic force drivers call) and stress (``jqmc_rs.grad_Eii_wrt_strain``) at the far-shifted
NaCl geometry are compared with backend-independent references:

* energy: the literature NaCl Madelung value ``-4 alpha_M``;
* force / stress: fourth-order central FD of the Rust ii energy of the same perturbed crystal
  written with every nucleus inside the home cell. Strained Ewald tables are rebuilt from the
  fixed integer index sets with ``_ewald_tables_from_cell``.

A second test repeats the check on the strongly skewed cell basis of the XLA file (issue #530),
where the nucleus fold ``fold_parallelepiped`` is active and its basis orientation is observable.
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
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)

import jqmc_rs  # noqa: E402
from jqmc._precision import get_tolerance  # noqa: E402
from jqmc.coulomb_potential import _ewald_tables_from_cell  # noqa: E402
from tests.test_jqmc_kernel.test_ewald_ii_far_nucleus import (  # noqa: E402
    ALPHA_M_NACL,
    DISPLACEMENT_PATTERN,
    FAR_SHIFTS,
    NACL_SITES,
    SKEWED_CELL,
    SKEWED_CHARGES,
    SKEWED_DISPLACEMENT_PATTERN,
    SKEWED_FAR_SHIFTS,
    SKEWED_FRAC_SITES,
    central_fd4,
    far_positions,
    make_ion_cell,
    make_nacl_cell,
    skewed_far_positions,
)

pytestmark = pytest.mark.without_chdir


def _rust_inputs(coulomb, ewald_data):
    return (
        float(ewald_data.eta),
        np.asarray(ewald_data.real_image_list, dtype=np.float64),
        np.asarray(ewald_data.g_vectors, dtype=np.float64),
        np.asarray(ewald_data.g_factors, dtype=np.float64),
        np.asarray(coulomb._effective_charges, dtype=np.float64),
        np.asarray(ewald_data.structure_data.positions, dtype=np.float64),
        np.asarray(ewald_data.structure_data.cell, dtype=np.float64),
    )


def _rust_energy(coulomb, ewald_data) -> float:
    eta, images, g_vectors, g_factors, charges, positions, cell = _rust_inputs(coulomb, ewald_data)
    return float(jqmc_rs.compute_ewald_potential_ii(charges, positions, eta, images, g_vectors, g_factors, cell))


@pytest.mark.light
@pytest.mark.parametrize("shift_name", list(FAR_SHIFTS))
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Ion-ion Ewald energy, force and stress; no electron coordinates enter E_ii."
)
def test_rust_ewald_ii_far_nucleus_energy_force_stress_are_translation_invariant(shift_name: str):
    """Rust E_ii, dE_ii/dR and dE_ii/deps are unchanged when a nucleus is given several cells away."""
    atol, rtol = get_tolerance("ewald", "strict")
    shift = FAR_SHIFTS[shift_name]

    coulomb, ewald_data = make_nacl_cell(far_positions(NACL_SITES, shift))
    e_rust = _rust_energy(coulomb, ewald_data)
    np.testing.assert_allclose(e_rust, -4.0 * ALPHA_M_NACL, atol=atol, rtol=rtol)

    base = NACL_SITES + DISPLACEMENT_PATTERN
    coulomb_in, ewald_in = make_nacl_cell(base)
    eta, _, _, _, charges, _, cell_in = _rust_inputs(coulomb_in, ewald_in)

    def rust_energy_in_cell(positions: np.ndarray, strain: np.ndarray | None = None) -> float:
        F = np.eye(3) if strain is None else np.eye(3) + strain
        cell = cell_in @ F.T
        images, g_vectors, g_factors, _ = _ewald_tables_from_cell(
            jnp.asarray(eta), ewald_in._real_image_indices_jnp, ewald_in._g_indices_jnp, jnp.asarray(cell)
        )
        return float(
            jqmc_rs.compute_ewald_potential_ii(
                charges,
                np.asarray(positions) @ F.T,
                eta,
                np.asarray(images),
                np.asarray(g_vectors),
                np.asarray(g_factors),
                cell,
            )
        )

    fd_force = central_fd4(lambda P: rust_energy_in_cell(P), np.array(base))
    fd_stress = central_fd4(lambda eps: rust_energy_in_cell(base, eps), np.zeros((3, 3)))
    # Non-vacuity: every ion feels a force and the stress is nonzero at the comparison tolerance.
    assert np.all(np.abs(fd_force).max(axis=1) > atol)
    assert np.abs(fd_stress).max() > atol

    coulomb_far, ewald_far = make_nacl_cell(far_positions(base, shift))
    eta_f, images_f, g_vectors_f, g_factors_f, charges_f, positions_f, cell_f = _rust_inputs(coulomb_far, ewald_far)
    force_rust = np.asarray(jqmc_rs.grad_Eii_wrt_R(eta_f, images_f, g_vectors_f, g_factors_f, charges_f, positions_f, cell_f))
    stress_rust = np.asarray(
        jqmc_rs.grad_Eii_wrt_strain(eta_f, images_f, g_vectors_f, g_factors_f, charges_f, positions_f, cell_f)
    )
    np.testing.assert_allclose(force_rust, fd_force, atol=atol, rtol=rtol)
    np.testing.assert_allclose(stress_rust, fd_stress, atol=atol, rtol=rtol)


def _rust_in_cell_energy_fn(coulomb_in, ewald_in):
    """Rust ii energy of the in-cell crystal at ``positions`` under affine strain (tables rebuilt per strain)."""
    eta, _, _, _, charges, _, cell_in = _rust_inputs(coulomb_in, ewald_in)

    def energy(positions: np.ndarray, strain: np.ndarray | None = None) -> float:
        F = np.eye(3) if strain is None else np.eye(3) + strain
        cell = cell_in @ F.T
        images, g_vectors, g_factors, _ = _ewald_tables_from_cell(
            jnp.asarray(eta), ewald_in._real_image_indices_jnp, ewald_in._g_indices_jnp, jnp.asarray(cell)
        )
        return float(
            jqmc_rs.compute_ewald_potential_ii(
                charges,
                np.asarray(positions) @ F.T,
                eta,
                np.asarray(images),
                np.asarray(g_vectors),
                np.asarray(g_factors),
                cell,
            )
        )

    return energy


@pytest.mark.light
@pytest.mark.parametrize("shift_name", list(SKEWED_FAR_SHIFTS))
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Ion-ion Ewald energy, force and stress; no electron coordinates enter E_ii."
)
def test_rust_ewald_ii_far_nucleus_skewed_basis_force_stress_are_translation_invariant(shift_name: str):
    """On the strongly skewed cell basis, Rust E_ii, dE_ii/dR and dE_ii/deps ignore a far-shifted nucleus.

    The Rust ii energy at the far-shifted geometry equals the Rust ii energy of the same crystal written
    with every nucleus inside the home cell; the Rust ii force and stress kernels at the far-shifted
    geometry equal fourth-order central FD of that in-cell energy.
    """
    atol, rtol = get_tolerance("ewald", "strict")
    shift = SKEWED_FAR_SHIFTS[shift_name]

    base = SKEWED_FRAC_SITES @ SKEWED_CELL + SKEWED_DISPLACEMENT_PATTERN
    coulomb_in, ewald_in = make_ion_cell(base, SKEWED_CELL, SKEWED_CHARGES)
    rust_energy_in_cell = _rust_in_cell_energy_fn(coulomb_in, ewald_in)
    e_in_cell = rust_energy_in_cell(base)
    fd_force = central_fd4(lambda P: rust_energy_in_cell(P), np.array(base))
    fd_stress = central_fd4(lambda eps: rust_energy_in_cell(base, eps), np.zeros((3, 3)))
    # Non-vacuity: every ion feels a force and the stress is nonzero at the comparison tolerance.
    assert np.all(np.abs(fd_force).max(axis=1) > atol)
    assert np.abs(fd_stress).max() > atol

    coulomb_far, ewald_far = make_ion_cell(skewed_far_positions(base, shift), SKEWED_CELL, SKEWED_CHARGES)
    eta_f, images_f, g_vectors_f, g_factors_f, charges_f, positions_f, cell_f = _rust_inputs(coulomb_far, ewald_far)
    e_rust = _rust_energy(coulomb_far, ewald_far)
    force_rust = np.asarray(jqmc_rs.grad_Eii_wrt_R(eta_f, images_f, g_vectors_f, g_factors_f, charges_f, positions_f, cell_f))
    stress_rust = np.asarray(
        jqmc_rs.grad_Eii_wrt_strain(eta_f, images_f, g_vectors_f, g_factors_f, charges_f, positions_f, cell_f)
    )
    np.testing.assert_allclose(e_rust, e_in_cell, atol=atol, rtol=rtol)
    np.testing.assert_allclose(force_rust, fd_force, atol=atol, rtol=rtol)
    np.testing.assert_allclose(stress_rust, fd_stress, atol=atol, rtol=rtol)

"""Ion-ion Ewald channel with a nucleus given several cells outside the home cell (issue #52).

A periodic lattice energy is invariant when any nucleus is translated by a lattice vector. The
ion-ion Ewald real-space sum adds a truncated image list ``T_n`` to the pair displacement
``R_I - R_J``; :py:func:`jqmc.coulomb_potential._enumerate_real_images` sizes that list for
``|R_I - R_J| <= sum_i |a_i|`` only, so the pair displacement must be folded before the image sum.
Without the fold, moving one ion of the 8-ion NaCl cell by ``3a(1,1,1)`` or ``6a(1,0,0)`` shifts
E_ii by +1.5e-2 Ha and the ion-ion force / stress by ~0.1.

Independent references:

* energy: the literature NaCl Madelung constant, ``E_ii = -4 alpha_M`` for the fixture
  (``R_NN = 1`` Bohr, ``Z = +-1``);
* force and stress: fourth-order central finite differences of the ion-ion energy of the SAME
  (perturbed) crystal written with every nucleus inside the home cell, where the image list is
  complete. By lattice-translation invariance the derivatives at the far-shifted geometry must
  equal these.

A second test repeats the force / stress check on a strongly skewed (non-reduced) cell basis with
a far-shifted nucleus (issue #530). On a diagonal cell ``inv(cell) == inv(cell.T)``, and on mild
triclinic cells with in-cell ions the fold rounds every fractional difference to zero, so neither
regime can tell the fold ``d - round(d @ inv(cell)) @ cell`` from a transposed-inverse variant.

This file covers the XLA / numpy paths (``compute_ewald_potential_ii``, the cached
``ewald_data.ii_energy`` precompute, ``_compute_ewald_potential_ii_debug``, the analytic
``grad_Eii_wrt_R`` / ``grad_Eii_wrt_strain`` and the AD force / stress of the energy). The Rust
kernels are covered by ``tests/test_rs_kernel/test_ewald_ii_far_nucleus_rs.py``.
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

from jqmc._precision import get_tolerance  # noqa: E402
from jqmc.coulomb_potential import (  # noqa: E402
    Coulomb_potential_data,
    _compute_ewald_potential_ii_debug,
    build_ewald_data,
    compute_ewald_potential_ii,
)
from jqmc.derivatives import ewald as ewald_derivatives  # noqa: E402
from jqmc.structure import Structure_data  # noqa: E402

pytestmark = [pytest.mark.without_chdir]

#: NaCl rocksalt Madelung constant (literature; Borwein-Glasser-McPhedran).
ALPHA_M_NACL = 1.747564594633
#: Lattice constant of the 8-ion simple-cubic NaCl cell (Bohr); R_NN = A_NACL / 2 = 1.
A_NACL = 2.0
NACL_SITES = np.array(
    [[0, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float64
) * (A_NACL / 2.0)
NACL_CHARGES = (1, 1, 1, 1, -1, -1, -1, -1)
#: Asymmetric displacement pattern (Bohr, times 0.07) so that no force or stress entry vanishes by
#: symmetry in the derivative checks.
DISPLACEMENT_PATTERN = 0.07 * np.array(
    [[1, 0, 0], [0, -1, 2], [0, 0, 0], [1, 1, 0], [0, 0, -1], [2, 0, 1], [0, 0, 0], [-1, 0, 1]], dtype=np.float64
)
#: Index of the ion that is moved far outside the home cell.
SHIFTED_ION = 1
#: Lattice translations (in units of the cubic lattice constant) applied to SHIFTED_ION.
FAR_SHIFTS = {"3a_111": (3, 3, 3), "6a_100": (6, 0, 0)}
#: Fourth-order central-difference step (Bohr / dimensionless strain).
FD_STEP = 1.0e-3


#: Strongly skewed (non-reduced) cell basis, rows a1, a2, a3 in Bohr: the basis of the repository's
#: skew-geometry tests, ((5,0,0),(4.5,2,0),(4,0.7,3)) / 2. It is far from orthogonal, so fractional
#: coordinates in this basis differ strongly from Cartesian projections.
SKEWED_CELL = 0.5 * np.array([[5.0, 0.0, 0.0], [4.5, 2.0, 0.0], [4.0, 0.7, 3.0]], dtype=np.float64)
#: Fractional sites of a neutral four-ion crystal in SKEWED_CELL and its charges.
SKEWED_FRAC_SITES = np.array([[0.05, 0.10, 0.15], [0.55, 0.60, 0.20], [0.30, 0.70, 0.65], [0.80, 0.25, 0.85]], dtype=np.float64)
SKEWED_CHARGES = (1, 1, -1, -1)
#: Asymmetric Cartesian perturbation (Bohr) of the skewed-cell ions.
SKEWED_DISPLACEMENT_PATTERN = 0.07 * np.array([[1, 0, 0], [0, -1, 2], [1, 1, 0], [0, 0, -1]], dtype=np.float64)
#: Integer lattice translations (in units of the SKEWED_CELL rows) applied to SHIFTED_ION.
SKEWED_FAR_SHIFTS = {"skew_3_m2_4": (3, -2, 4), "skew_m4_5_m1": (-4, 5, -1)}


def make_ion_cell(positions: np.ndarray, cell: np.ndarray, charges: tuple[int, ...]) -> tuple[Coulomb_potential_data, object]:
    """Build a bare-ion periodic cell (rows of ``cell`` = lattice vectors) with its Ewald tables (tol 1e-12)."""
    n_ion = len(charges)
    structure = Structure_data.build(
        pbc_flag=True,
        positions=np.asarray(positions, dtype=np.float64),
        vec_a=tuple(float(x) for x in cell[0]),
        vec_b=tuple(float(x) for x in cell[1]),
        vec_c=tuple(float(x) for x in cell[2]),
        atomic_numbers=charges,
        element_symbols=("X",) * n_ion,
        atomic_labels=tuple(f"X{i}" for i in range(n_ion)),
    )
    coulomb = Coulomb_potential_data(structure_data=structure, ecp_flag=False)
    ewald_data = build_ewald_data(structure, coulomb, tol=1e-12)
    return coulomb.replace(ewald_data=ewald_data), ewald_data


def make_nacl_cell(positions: np.ndarray) -> tuple[Coulomb_potential_data, object]:
    """Build the 8-ion NaCl SC cell at ``positions`` with its Ewald tables (tol 1e-12)."""
    return make_ion_cell(positions, A_NACL * np.eye(3), NACL_CHARGES)


def skewed_far_positions(base: np.ndarray, shift: tuple[int, int, int]) -> np.ndarray:
    """Copy of ``base`` with SHIFTED_ION translated by the SKEWED_CELL lattice vector ``shift @ SKEWED_CELL``."""
    positions = np.array(base, dtype=np.float64, copy=True)
    positions[SHIFTED_ION] += np.asarray(shift, dtype=np.float64) @ SKEWED_CELL
    return positions


def far_positions(base: np.ndarray, shift: tuple[int, int, int]) -> np.ndarray:
    """Copy of ``base`` with SHIFTED_ION translated by the lattice vector ``shift * A_NACL``."""
    positions = np.array(base, dtype=np.float64, copy=True)
    positions[SHIFTED_ION] += A_NACL * np.asarray(shift, dtype=np.float64)
    return positions


def strained_structure(structure: Structure_data, positions: np.ndarray, strain: np.ndarray) -> Structure_data:
    """Structure with nuclei and cell deformed by ``F = I + strain`` (affine strain)."""
    F = jnp.eye(3) + jnp.asarray(strain)
    return structure.replace(
        positions=jnp.asarray(positions) @ F.T,
        vec_a=F @ jnp.asarray(structure.vec_a),
        vec_b=F @ jnp.asarray(structure.vec_b),
        vec_c=F @ jnp.asarray(structure.vec_c),
    )


def central_fd4(energy, x0: np.ndarray) -> np.ndarray:
    """Fourth-order central finite-difference gradient of ``energy`` at ``x0`` (any shape)."""
    grad = np.zeros_like(x0)
    for idx in np.ndindex(x0.shape):
        values = []
        for k in (-2, -1, 1, 2):
            x = np.array(x0, copy=True)
            x[idx] += k * FD_STEP
            values.append(float(energy(x)))
        grad[idx] = (values[0] - 8.0 * values[1] + 8.0 * values[2] - values[3]) / (12.0 * FD_STEP)
    return grad


@pytest.mark.light
@pytest.mark.parametrize("shift_name", list(FAR_SHIFTS))
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Ion-ion Ewald energy, force and stress; no electron coordinates enter E_ii."
)
def test_ewald_ii_far_nucleus_energy_force_stress_are_translation_invariant(shift_name: str):
    """E_ii, dE_ii/dR and dE_ii/deps do not change when a nucleus is given several cells away.

    Energy: ``compute_ewald_potential_ii`` (XLA), ``ewald_data.ii_energy`` (numpy precompute
    stored by ``build_ewald_data``; the Rust production path recomputes ii from the current
    structure) and ``_compute_ewald_potential_ii_debug`` at the far-shifted NaCl geometry all
    equal ``-4 alpha_M``. Force / stress: the analytic
    ``grad_Eii_wrt_R`` / ``grad_Eii_wrt_strain`` and the AD derivatives of the XLA energy at the
    far-shifted, perturbed geometry equal fourth-order central FD of the in-cell energy.
    """
    atol, rtol = get_tolerance("ewald", "strict")
    shift = FAR_SHIFTS[shift_name]
    e_madelung = -4.0 * ALPHA_M_NACL

    # --- energy on the exact rocksalt geometry --------------------------------------------
    coulomb, ewald_data = make_nacl_cell(far_positions(NACL_SITES, shift))
    e_xla = float(compute_ewald_potential_ii(coulomb, ewald_data))
    e_cached = float(ewald_data.ii_energy)
    e_debug = float(_compute_ewald_potential_ii_debug(coulomb, ewald_data))
    np.testing.assert_allclose(e_xla, e_madelung, atol=atol, rtol=rtol)
    np.testing.assert_allclose(e_cached, e_madelung, atol=atol, rtol=rtol)
    np.testing.assert_allclose(e_debug, e_madelung, atol=atol, rtol=rtol)

    # --- force and stress on a perturbed geometry -----------------------------------------
    base = NACL_SITES + DISPLACEMENT_PATTERN
    coulomb_in, ewald_in = make_nacl_cell(base)
    structure_in = ewald_in.structure_data

    def energy_in_cell(positions: np.ndarray, strain: np.ndarray | None = None) -> jax.Array:
        sd = (
            structure_in.replace(positions=jnp.asarray(positions))
            if strain is None
            else strained_structure(structure_in, positions, strain)
        )
        return compute_ewald_potential_ii(coulomb_in, ewald_in.replace(structure_data=sd))

    fd_force = central_fd4(lambda P: energy_in_cell(P), np.array(base))
    fd_stress = central_fd4(lambda eps: energy_in_cell(base, eps), np.zeros((3, 3)))
    # Non-vacuity: every ion feels a force and the stress is nonzero at the comparison tolerance.
    assert np.all(np.abs(fd_force).max(axis=1) > atol)
    assert np.abs(fd_stress).max() > atol

    coulomb_far, ewald_far = make_nacl_cell(far_positions(base, shift))
    structure_far = ewald_far.structure_data
    positions_far = np.asarray(structure_far.positions, dtype=np.float64)

    force_analytic = np.asarray(ewald_derivatives.grad_Eii_wrt_R(coulomb_far, ewald_far))
    force_ad = np.asarray(
        jax.grad(
            lambda P: compute_ewald_potential_ii(
                coulomb_far, ewald_far.replace(structure_data=structure_far.replace(positions=P))
            )
        )(jnp.asarray(positions_far))
    )
    stress_analytic = np.asarray(ewald_derivatives.grad_Eii_wrt_strain(coulomb_far, ewald_far))
    stress_ad = np.asarray(
        jax.grad(
            lambda eps: compute_ewald_potential_ii(
                coulomb_far, ewald_far.replace(structure_data=strained_structure(structure_far, positions_far, eps))
            )
        )(jnp.zeros((3, 3)))
    )
    np.testing.assert_allclose(force_analytic, fd_force, atol=atol, rtol=rtol)
    np.testing.assert_allclose(force_ad, fd_force, atol=atol, rtol=rtol)
    np.testing.assert_allclose(stress_analytic, fd_stress, atol=atol, rtol=rtol)
    np.testing.assert_allclose(stress_ad, fd_stress, atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("shift_name", list(SKEWED_FAR_SHIFTS))
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Ion-ion Ewald energy, force and stress; no electron coordinates enter E_ii."
)
def test_ewald_ii_far_nucleus_skewed_basis_force_stress_are_translation_invariant(shift_name: str):
    """On a strongly skewed cell basis, a far-shifted nucleus leaves E_ii and its R / strain derivatives unchanged.

    Four ions (Z = +1, +1, -1, -1) in SKEWED_CELL with an asymmetric perturbation; one ion is translated
    by the lattice vector ``shift @ SKEWED_CELL``. The XLA energy, the cached numpy ``ii_energy`` and the
    ``_debug`` oracle at the far-shifted geometry equal the XLA energy of the same crystal written with
    every nucleus inside the home cell. The analytic ``grad_Eii_wrt_R`` / ``grad_Eii_wrt_strain`` and
    the AD derivatives of the XLA energy at the far-shifted geometry equal fourth-order central FD of
    the in-cell energy.
    """
    atol, rtol = get_tolerance("ewald", "strict")
    shift = SKEWED_FAR_SHIFTS[shift_name]

    base = SKEWED_FRAC_SITES @ SKEWED_CELL + SKEWED_DISPLACEMENT_PATTERN
    positions_far = skewed_far_positions(base, shift)
    # Regime check: some folded pair displacement has a fractional coordinate beyond 1/2 in this basis
    # AND the fractional coordinates taken with the transposed inverse round differently, so the fold
    # is active and its orientation is observable (both are false on the cubic NaCl cell).
    pair = positions_far[:, None, :] - positions_far[None, :, :]
    frac = pair @ np.linalg.inv(SKEWED_CELL)
    frac_transposed = pair @ np.linalg.inv(SKEWED_CELL.T)
    assert np.any(np.abs(frac) > 0.5)
    assert np.any(np.round(frac) != np.round(frac_transposed))

    coulomb_in, ewald_in = make_ion_cell(base, SKEWED_CELL, SKEWED_CHARGES)
    structure_in = ewald_in.structure_data

    def energy_in_cell(positions: np.ndarray, strain: np.ndarray | None = None) -> jax.Array:
        sd = (
            structure_in.replace(positions=jnp.asarray(positions))
            if strain is None
            else strained_structure(structure_in, positions, strain)
        )
        return compute_ewald_potential_ii(coulomb_in, ewald_in.replace(structure_data=sd))

    e_in_cell = float(energy_in_cell(base))
    fd_force = central_fd4(lambda P: energy_in_cell(P), np.array(base))
    fd_stress = central_fd4(lambda eps: energy_in_cell(base, eps), np.zeros((3, 3)))
    # Non-vacuity: every ion feels a force and the stress is nonzero at the comparison tolerance.
    assert np.all(np.abs(fd_force).max(axis=1) > atol)
    assert np.abs(fd_stress).max() > atol

    coulomb_far, ewald_far = make_ion_cell(positions_far, SKEWED_CELL, SKEWED_CHARGES)
    structure_far = ewald_far.structure_data
    positions_far = np.asarray(structure_far.positions, dtype=np.float64)

    e_xla = float(compute_ewald_potential_ii(coulomb_far, ewald_far))
    e_cached = float(ewald_far.ii_energy)
    e_debug = float(_compute_ewald_potential_ii_debug(coulomb_far, ewald_far))
    np.testing.assert_allclose(e_xla, e_in_cell, atol=atol, rtol=rtol)
    np.testing.assert_allclose(e_cached, e_in_cell, atol=atol, rtol=rtol)
    np.testing.assert_allclose(e_debug, e_in_cell, atol=atol, rtol=rtol)

    force_analytic = np.asarray(ewald_derivatives.grad_Eii_wrt_R(coulomb_far, ewald_far))
    force_ad = np.asarray(
        jax.grad(
            lambda P: compute_ewald_potential_ii(
                coulomb_far, ewald_far.replace(structure_data=structure_far.replace(positions=P))
            )
        )(jnp.asarray(positions_far))
    )
    stress_analytic = np.asarray(ewald_derivatives.grad_Eii_wrt_strain(coulomb_far, ewald_far))
    stress_ad = np.asarray(
        jax.grad(
            lambda eps: compute_ewald_potential_ii(
                coulomb_far, ewald_far.replace(structure_data=strained_structure(structure_far, positions_far, eps))
            )
        )(jnp.zeros((3, 3)))
    )
    np.testing.assert_allclose(force_analytic, fd_force, atol=atol, rtol=rtol)
    np.testing.assert_allclose(force_ad, fd_force, atol=atol, rtol=rtol)
    np.testing.assert_allclose(stress_analytic, fd_stress, atol=atol, rtol=rtol)
    np.testing.assert_allclose(stress_ad, fd_stress, atol=atol, rtol=rtol)

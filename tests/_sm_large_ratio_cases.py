"""Shared inputs and independent references for the large-ratio carried-inverse tests (issue #127).

An accepted single-electron move whose determinant ratio ``R`` is huge (an
electron returning from a deep orbital tail) must leave a carried inverse that
still inverts the carried geminal, so that the NEXT move's ratio and Metropolis
decision are right. The references here never use a production ratio,
inverse-update, AS or proposal helper:

* ``water``: committed water ccECP cc-pVQZ (4 up / 4 dn), no Jastrow. PCG64 seed
  2025 places every electron at a nucleus + 0.5 N(0, 1); electron 0 of the moved
  spin is then put 28 bohr out along +x from nucleus 0. Move 1 brings it to
  12 bohr (exact log10|R| ~ 15.5, exact acceptance 1); move 2 takes it to 13 bohr
  (exact acceptance ~0.16, rejected by a uniform draw above 0.25). The
  reference rebuilds G at every configuration and takes ``numpy.linalg.slogdet``.
* ``power``: one up electron, one normalized Cartesian GTO x^12 exp(-r^2) on a
  bare Z = 7 nucleus. Move 1 takes x from 0.01 to 0.3 (|R| ~ 5e17 > 1/eps,
  exact acceptance 1); move 2 back to x = 0.02 (exact acceptance ~1e-28). The
  reference is the closed form ln psi = ln N + 12 ln|x| - r^2 with an lgamma
  normalization. The values stay inside the float32 range (|psi| >= 1e-26), so
  both precision modes apply.
* ``x82``: the same model with x^82 (issue #111); used only at an exactly
  underflowed walker (G = 0 in both precision modes) for the inverse refresh.
* ``cpower``: the complex native sweep. Neutral He (Z = 2)
  in a periodic cubic cell of side 100 bohr, Bloch twist (0.25, 0.125, 0), no
  Jastrow, one up and one down electron sharing the genuinely complex orbital
  phi = c N12 x^12 (1 + i x) exp(-r^2) (two normalized Cartesian primitives,
  time_reversal=False). The moved electron sits at (x, 0.3, 0.4), the other at
  (0.4, 0.3, 0.4). Every point lies in the home cell, where the wrap-restore
  phase is 1, and the nearest lattice image is ~100 bohr away (its Gaussian is
  exactly 0), so G = phi(r_up) conj(phi(r_dn)) in closed form (the down
  channel is the opposite-twist partner conj(phi)).
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

from math import exp, lgamma, log, sqrt
from pathlib import Path

import numpy as np

from jqmc.atomic_orbital import AOs_cart_gto_data, build_AOs_cart_pgto_complex_data
from jqmc.coulomb_potential import Coulomb_potential_data, build_ewald_data
from jqmc.determinant import SD_data, compute_geminal_all_elements
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data
from jqmc.molecular_orbital import MOs_data
from tests._opposite_spin_twist_cases import opposite_twist_partner
from jqmc.structure import Structure_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

TREXIO_DIR = Path(__file__).resolve().parent / "trexio_example_files"

# Proposal scale of every large-ratio test (the transition density is evaluated
# with the same Dt in the reference).
DT = 2.0
POWER_Z = 7
POWER_EXPONENT = 1.0
POWER_L = 12
# Fixed uniform draws of the XLA twin (move 1 always accepts; move 2 must reject).
XLA_DRAWS = (0.955, 0.2927)
# Native keys are searched so that the second draw is at least this far above
# the exact move-2 acceptance (~0.16), i.e. the expected rejection has margin.
NATIVE_MIN_SECOND_DRAW = 0.25

# Move 1 of every case must exceed this ratio magnitude, 2**48 = 1 / (16 eps)
# of float64: the plain rank-1 update then keeps at most about one significant
# digit of the moved column / row (none at all beyond 1 / eps).
LARGE_RATIO_WITNESS = 2.0**48

WATER_DISTANCES = (28.0, 12.0, 13.0)
POWER_X = (0.01, 0.3, 0.02)


def water_case(spin: str):
    """Water ham, nuclei, valence charges and the start/move-1/move-2 configurations."""
    structure, _, _, _, geminal, coulomb = read_trexio_file(str(TREXIO_DIR / "water_ccecp_ccpvqz.h5"), store_tuple=True)
    ham = Hamiltonian_data(
        structure_data=structure,
        wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data()),
        coulomb_potential_data=coulomb,
    )
    nuclei = np.asarray(structure.positions, dtype=np.float64)
    charges = np.asarray(structure.atomic_numbers, dtype=np.float64) - np.asarray(coulomb.z_cores, dtype=np.float64)
    rng = np.random.default_rng(2025)
    up = nuclei[rng.integers(0, len(nuclei), geminal.num_electron_up)] + rng.normal(size=(geminal.num_electron_up, 3)) * 0.5
    dn = nuclei[rng.integers(0, len(nuclei), geminal.num_electron_dn)] + rng.normal(size=(geminal.num_electron_dn, 3)) * 0.5
    configurations = []
    for distance in WATER_DISTANCES:
        u, d = up.copy(), dn.copy()
        (u if spin == "up" else d)[0] = nuclei[0] + np.array([distance, 0.0, 0.0])
        configurations.append((u, d))
    return ham, nuclei, charges, configurations


def power_model(power: int):
    """One up electron in a single normalized Cartesian GTO x^power exp(-r^2) on a bare N nucleus."""
    structure = Structure_data.build(
        positions=np.zeros((1, 3)),
        pbc_flag=False,
        atomic_numbers=(POWER_Z,),
        element_symbols=("N",),
        atomic_labels=("N1",),
    )
    aos = AOs_cart_gto_data.build(
        structure_data=structure,
        nucleus_index=(0,),
        num_ao=1,
        num_ao_prim=1,
        orbital_indices=(0,),
        exponents=(POWER_EXPONENT,),
        coefficients=(1.0,),
        angular_momentums=(power,),
        polynominal_order_x=(power,),
        polynominal_order_y=(0,),
        polynominal_order_z=(0,),
    )
    up = MOs_data(num_mo=1, aos_data=aos, mo_coefficients=np.array([[1.0]]))
    dn = MOs_data(num_mo=0, aos_data=aos, mo_coefficients=np.empty((0, 1)))
    geminal = SD_data(num_electron_up=1, num_electron_dn=0, orb_data_up_spin=up, orb_data_dn_spin=dn)
    ham = Hamiltonian_data(
        structure_data=structure,
        wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data()),
        coulomb_potential_data=Coulomb_potential_data(structure_data=structure),
    )
    return ham, np.zeros((1, 3)), np.array([float(POWER_Z)])


def power_case():
    """The x^12 model and its start/move-1/move-2 configurations (one up electron, no down)."""
    ham, nuclei, charges = power_model(POWER_L)
    configurations = [(np.array([[x, 0.3, 0.4]]), np.zeros((0, 3))) for x in POWER_X]
    return ham, nuclei, charges, configurations


def power_log_psi(position: np.ndarray, power: int = POWER_L) -> float:
    """Closed-form ln|psi| of the normalized x^power exp(-a r^2) primitive (no production code)."""
    a = POWER_EXPONENT
    ln_norm = -0.5 * (lgamma(power + 0.5) - (power + 0.5) * log(2 * a) + 2 * (lgamma(0.5) - 0.5 * log(2 * a)))
    return ln_norm + power * log(abs(float(position[0]))) - a * float(position @ position)


COMPLEX_Z = 2
COMPLEX_CELL = 100.0
COMPLEX_TWIST = (0.25, 0.125, 0.0)
COMPLEX_OTHER = (0.4, 0.3, 0.4)
# Moves of the complex detector of the exact moved column / row: 0.01 -> 0.3
# (|R| ~ 5.07e17, exact acceptance 1) -> 0.02 (exact acceptance ~6.8e-29).
COMPLEX_X = (0.01, 0.3, 0.02)


def _ln_primitive_norm(power: int) -> float:
    """ln N_p of x^p exp(-r^2): N_p^-2 = Gamma(p + 1/2) 2^(-p-1/2) (pi / 2) (closed form, no production code)."""
    return -0.5 * (lgamma(power + 0.5) - (power + 0.5) * log(2.0) + 2 * (lgamma(0.5) - 0.5 * log(2.0)))


def _complex_mo() -> tuple[np.ndarray, float]:
    """MO coefficients (1, i N12/N13) / sqrt(1 + (N12/N13)^2) and ln of the prefactor c N12 of phi."""
    q = exp(_ln_primitive_norm(12) - _ln_primitive_norm(13))
    c = 1.0 / sqrt(1.0 + q * q)
    return np.array([[c, 1j * q * c]]), log(c) + _ln_primitive_norm(12)


def complex_power_model():
    """The periodic complex He model of ``cpower`` (Hamiltonian, nuclei, valence charges)."""
    side = COMPLEX_CELL
    structure = Structure_data.build(
        positions=np.zeros((1, 3)),
        pbc_flag=True,
        vec_a=[side, 0.0, 0.0],
        vec_b=[0.0, side, 0.0],
        vec_c=[0.0, 0.0, side],
        atomic_numbers=(COMPLEX_Z,),
        element_symbols=("He",),
        atomic_labels=("He1",),
    )
    aos = build_AOs_cart_pgto_complex_data(
        structure,
        k_point=COMPLEX_TWIST,
        nucleus_index=(0, 0),
        num_ao=2,
        num_ao_prim=2,
        orbital_indices=(0, 1),
        exponents=np.array([POWER_EXPONENT, POWER_EXPONENT]),
        coefficients=np.array([1.0, 1.0]),
        angular_momentums=(12, 13),
        polynominal_order_x=(12, 13),
        polynominal_order_y=(0, 0),
        polynominal_order_z=(0, 0),
    )
    mos = MOs_data.build(aos_data=aos, mo_coefficients=_complex_mo()[0])
    geminal = SD_data.build(
        num_electron_up=1, num_electron_dn=1, mos_up=mos, mos_dn=opposite_twist_partner(mos), time_reversal=False
    )
    coulomb = Coulomb_potential_data.build(structure_data=structure)
    coulomb = coulomb.replace(ewald_data=build_ewald_data(structure, coulomb, n_electron=2))
    ham = Hamiltonian_data(
        structure_data=structure,
        wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data()),
        coulomb_potential_data=coulomb,
    )
    return ham, np.zeros((1, 3)), np.array([float(COMPLEX_Z)])


def complex_power_case(spin: str, xs=COMPLEX_X):
    """The complex model and the configurations of the moved electron at x in ``xs`` (other electron fixed)."""
    ham, nuclei, charges = complex_power_model()
    other = np.array([COMPLEX_OTHER])
    configurations = []
    for x in xs:
        moved_electron = np.array([[x, 0.3, 0.4]])
        configurations.append((moved_electron, other.copy()) if spin == "up" else (other.copy(), moved_electron))
    return ham, nuclei, charges, configurations


def complex_phi(position: np.ndarray) -> complex:
    """Closed-form phi = c N12 x^12 (1 + i x) exp(-r^2), evaluated in the log domain for the magnitude."""
    x = float(position[0])
    ln_magnitude = _complex_mo()[1] + 12 * log(abs(x)) - float(position @ position)
    return exp(ln_magnitude) * (1.0 + 1j * x)


def complex_log_abs_psi(configuration) -> float:
    """ln|phi(r_up) phi(r_dn)| in closed form."""
    total = 0.0
    for position in (configuration[0][0], configuration[1][0]):
        x = float(position[0])
        total += _complex_mo()[1] + 12 * log(abs(x)) + 0.5 * log(1.0 + x * x) - float(position @ position)
    return total


def width(position: np.ndarray, nuclei: np.ndarray, charges: np.ndarray) -> float:
    """Kato-cusp proposal width f_l of the nearest nucleus (independent numpy formula)."""
    distances = np.linalg.norm(nuclei - position, axis=1)
    atom = int(np.argmin(distances))
    radius = float(distances[atom])
    charge = float(charges[atom])
    return (radius + charge ** (-2)) / (radius + 1)


def moved(configuration, spin: str) -> np.ndarray:
    up, dn = configuration
    return (up if spin == "up" else dn)[0]


def injected_normal(before, after, spin: str, nuclei, charges) -> float:
    """Normal draw that the proposal ``r + normal * f_l(r) * Dt`` (axis x) maps onto ``after``."""
    return float((moved(after, spin)[0] - moved(before, spin)[0]) / (width(moved(before, spin), nuclei, charges) * DT))


def proposed(before, normal: float, spin: str, nuclei, charges):
    """The proposal of one x-axis move, recomputed in numpy."""
    up, dn = before[0].copy(), before[1].copy()
    position = (up if spin == "up" else dn)[0]
    position[0] += normal * width(moved(before, spin), nuclei, charges) * DT
    return up, dn


def log_abs_psi(ham, configuration, model: str) -> float:
    """Reference ln|Psi|: rebuilt-G slogdet (water) or the closed form (power and complex power models)."""
    if model == "power":
        return power_log_psi(configuration[0][0])
    if model == "cpower":
        return complex_log_abs_psi(configuration)
    if model == "cpair":
        return complex_pair_log_abs_psi(configuration)
    matrix = np.asarray(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, *configuration))
    sign, value = np.linalg.slogdet(matrix)
    assert sign != 0, "reference precondition: invertible rebuilt geminal"
    return float(value)


def log_abs_ratio(ham, before, after, model: str) -> float:
    """Reference ln|Psi(after)/Psi(before)|."""
    return log_abs_psi(ham, after, model) - log_abs_psi(ham, before, model)


def exact_acceptance(ham, before, after, spin: str, nuclei, charges, model: str) -> float:
    """min(1, |Psi(after)/Psi(before)|^2 T(after->before)/T(before->after)), all in the log domain."""
    a, b = moved(before, spin), moved(after, spin)
    fa, fb = width(a, nuclei, charges) * DT, width(b, nuclei, charges) * DT
    log_transition = log(fa / fb) - float(np.sum((b - a) ** 2)) / 2 * (fb ** (-2) - fa ** (-2))
    log_ratio = 2 * (log_abs_psi(ham, after, model) - log_abs_psi(ham, before, model))
    return float(np.exp(min(0.0, log_ratio + log_transition)))


def reference_geminal(ham, configuration, model: str) -> np.ndarray:
    """G rebuilt from the positions (water) or the closed-form 1x1 G (power: psi; cpower: phi(r_up) conj(phi(r_dn)))."""
    if model == "power":
        return np.array([[np.exp(power_log_psi(configuration[0][0])) * np.sign(configuration[0][0][0]) ** POWER_L]])
    if model == "cpower":
        # The down channel is the opposite-twist partner: conj(phi).
        return np.array([[complex_phi(configuration[0][0]) * np.conj(complex_phi(configuration[1][0]))]])
    return np.asarray(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, *configuration))


def row_scaled(matrix: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divide each row by the reference row's max-norm (G rows of far electrons are tiny)."""
    scale = np.max(np.abs(reference), axis=1, keepdims=True)
    return matrix / scale, reference / scale


def column_scaled(matrix: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divide each column by the reference column's max-norm (G^-1 columns of far electrons are huge)."""
    scale = np.max(np.abs(reference), axis=0, keepdims=True)
    return matrix / scale, reference / scale


# ``cpair``: the complex native sweep with a 2x2
# geminal, so that the rank-1 term of the accepted-move update (not only the
# exact moved column / row) decides the carried inverse. Neutral Be (Z = 4) in
# the periodic cubic cell / twist of ``cpower``, no Jastrow, time_reversal=False,
# two up and two down electrons in the two MOs
#     phi0 = c N12 x^12 (1 + i x) exp(-r^2)   (the ``cpower`` orbital)
#     phi1 = N13 x^13 exp(-r^2)               (the normalized x^13 primitive).
# The moved electron (index 0 of its spin) sits at (x, 0.3, 0.4); the other
# electron of that spin at PAIR_OTHER_SAME, the opposite spin at PAIR_OPPOSITE.
# The down channel is the opposite-twist partner conj(phi_k).
# With Phi[k, i] = phi_k(r_i), G = Phi_up^T conj(Phi_dn) and
#     det Phi = c N12 N13 x_0^12 x_1^12 (x_1 - x_0) exp(-r_0^2 - r_1^2),
# all in closed form; every point lies in the home cell (phase 1, images 0).
PAIR_Z = 4
PAIR_OTHER_SAME = (0.4, 0.3, 0.4)
PAIR_OPPOSITE = ((0.5, 0.2, 0.1), (0.7, 0.1, 0.3))
PAIR_X_TARGET = 0.3


def complex_pair_model():
    """The periodic complex 2x2 Be model of ``cpair`` (Hamiltonian, nuclei, valence charges)."""
    side = COMPLEX_CELL
    structure = Structure_data.build(
        positions=np.zeros((1, 3)),
        pbc_flag=True,
        vec_a=[side, 0.0, 0.0],
        vec_b=[0.0, side, 0.0],
        vec_c=[0.0, 0.0, side],
        atomic_numbers=(PAIR_Z,),
        element_symbols=("Be",),
        atomic_labels=("Be1",),
    )
    aos = build_AOs_cart_pgto_complex_data(
        structure,
        k_point=COMPLEX_TWIST,
        nucleus_index=(0, 0),
        num_ao=2,
        num_ao_prim=2,
        orbital_indices=(0, 1),
        exponents=np.array([POWER_EXPONENT, POWER_EXPONENT]),
        coefficients=np.array([1.0, 1.0]),
        angular_momentums=(12, 13),
        polynominal_order_x=(12, 13),
        polynominal_order_y=(0, 0),
        polynominal_order_z=(0, 0),
    )
    coefficients = np.vstack([_complex_mo()[0], np.array([[0.0, 1.0]])]).astype(np.complex128)
    mos = MOs_data.build(aos_data=aos, mo_coefficients=coefficients)
    geminal = SD_data.build(
        num_electron_up=2, num_electron_dn=2, mos_up=mos, mos_dn=opposite_twist_partner(mos), time_reversal=False
    )
    coulomb = Coulomb_potential_data.build(structure_data=structure)
    coulomb = coulomb.replace(ewald_data=build_ewald_data(structure, coulomb, n_electron=4))
    ham = Hamiltonian_data(
        structure_data=structure,
        wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data()),
        coulomb_potential_data=coulomb,
    )
    return ham, np.zeros((1, 3)), np.array([float(PAIR_Z)])


def complex_pair_case(spin: str, x_start: float):
    """The ``cpair`` model and the start / target configurations of the moved electron (x_start -> 0.3)."""
    ham, nuclei, charges = complex_pair_model()
    opposite = np.array(PAIR_OPPOSITE)
    configurations = []
    for x in (x_start, PAIR_X_TARGET):
        same = np.array([[x, 0.3, 0.4], PAIR_OTHER_SAME])
        configurations.append((same, opposite.copy()) if spin == "up" else (opposite.copy(), same))
    return ham, nuclei, charges, configurations


def complex_pair_orbitals(positions: np.ndarray) -> np.ndarray:
    """Closed-form Phi[k, i] = phi_k(r_i) of ``cpair`` (magnitudes formed in the log domain)."""
    ln_c_n12 = _complex_mo()[1]
    ln_n13 = _ln_primitive_norm(13)
    phi = np.empty((2, len(positions)), dtype=np.complex128)
    for i, position in enumerate(np.asarray(positions, dtype=np.float64)):
        x, r2 = float(position[0]), float(position @ position)
        phi[0, i] = exp(ln_c_n12 + 12 * log(x) - r2) * (1.0 + 1j * x)
        phi[1, i] = exp(ln_n13 + 13 * log(x) - r2)
    return phi


def complex_pair_geminal(configuration) -> np.ndarray:
    """Closed-form G = Phi_up^T conj(Phi_dn) of ``cpair`` (the down channel is conj(phi_k))."""
    return complex_pair_orbitals(configuration[0]).T @ np.conj(complex_pair_orbitals(configuration[1]))


def complex_pair_inverse(configuration) -> np.ndarray:
    """Closed-form G^-1 = conj(Phi_dn)^-1 Phi_up^-T of ``cpair`` from the 2x2 adjugates (no matrix inversion routine)."""

    def inverse_2x2(m: np.ndarray) -> np.ndarray:
        return np.array([[m[1, 1], -m[0, 1]], [-m[1, 0], m[0, 0]]]) / (m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])

    up, dn = (complex_pair_orbitals(positions) for positions in configuration)
    dn = np.conj(dn)  # the down channel is the opposite-twist partner: conj(phi_k)
    return inverse_2x2(dn) @ inverse_2x2(up).T


def complex_pair_log_abs_psi(configuration) -> float:
    """ln|det Phi_up det Phi_dn| of ``cpair`` in closed form (see the model comment)."""
    ln_prefactor = _complex_mo()[1] + _ln_primitive_norm(13)
    total = 0.0
    for positions in configuration:
        (x0, x1), r2 = positions[:, 0], np.sum(positions * positions, axis=1)
        total += ln_prefactor + 12 * (log(x0) + log(x1)) + log(abs(x1 - x0)) - float(r2[0] + r2[1])
    return total

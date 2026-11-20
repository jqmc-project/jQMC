"""Shared inputs and independent references for the AS-factor robustness tests (issues #131 and #134).

The Attaccalite-Sorella factor ``R_AS = (S F)^(-3/8)`` (``S`` the minimum squared
row/column norm of the geminal ``G``, ``F = ||G^-1||_F^2``) is invariant under
scaling one row (up electron) or column (down electron) of ``G``. Two
non-robust evaluations made it 0:

* #131 (current state): a far electron makes one row of ``G`` smaller than
  ~1e-154, ``S`` underflows to 0, ``F`` overflows, ``S * F = 0 * inf`` gave
  ``R_AS = 0`` and the walker was frozen (every proposal rejected) at any
  ``epsilon_AS`` on XLA and at ``epsilon_AS > 0`` natively; ``w_L`` was NaN or 0.
* #134 (proposed state): for a tiny determinant ratio the proposed row / column
  ``G_k + (G'_k - G_k)`` (XLA) or the incremental cross norms
  ``n2 + 2 G v + v^2`` (native) cancelled to 0, ``R_AS' = 0`` and the AS ratio
  ``eps / 0 = inf`` forced the acceptance of a move with exact probability
  below 1e-16.

Inputs (no Jastrow, Dt = 2):

* ``power``: one up electron in one normalized Cartesian GTO x^12 exp(-r^2) on a
  bare Z = 7 nucleus. ``G`` is 1x1, so ``R_AS = 1`` exactly at every
  configuration and ln psi = ln N + 12 ln|x| - r^2 in closed form.
* ``h2``: committed H2 all-electron cc-pVDZ (1 up / 1 down, 1x1 ``G``), electron
  0 of either spin moved along +x.
* ``water``: committed water ccECP cc-pVQZ (4 up / 4 down), PCG64 seed 2025 start.

The references never use a production ratio, inverse-update, AS or proposal
helper: ``numpy.linalg.slogdet`` of the rebuilt ``G`` (or the closed form) for
the wave-function ratio, the Kato-width Gaussian transition density in numpy,
and :func:`reference_ln_as_factor`, a row-rescaled log-domain evaluation of
``R_AS`` with numpy's inverse of the rescaled matrix.
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

from math import lgamma, log
from pathlib import Path

import numpy as np

from jqmc.atomic_orbital import AOs_cart_gto_data
from jqmc.coulomb_potential import Coulomb_potential_data
from jqmc.determinant import SD_data, compute_geminal_all_elements
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data
from jqmc.molecular_orbital import MOs_data
from jqmc.structure import Structure_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

TREXIO_DIR = Path(__file__).resolve().parent / "trexio_example_files"

DT = 2.0
POWER_Z = 7
POWER_EXPONENT = 1.0
POWER_L = 12
XLA_DRAWS = (0.01, 0.5, 0.99)

# A walker whose geminal row is below sqrt(tiny) ~ 1.5e-154 has an underflowing S.
UNDERFLOW_ROW_NORM = 1.0e-154
# Row scale whose square (~1e-120) is still far inside the linear S * F range but
# negligible against every other contribution, i.e. already in the scale-invariant limit.
LINEAR_RANGE_ROW_SCALE = 1.0e-60
# Determinant-ratio magnitudes at which the incremental proposed norms lose every
# digit: |R|^2 < eps (native cross norms) and |R| < eps (XLA additive row).
NATIVE_CANCELLATION_RATIO = 1.0e-8
XLA_CANCELLATION_RATIO = 1.0e-16


def _hamiltonian(structure, geminal, coulomb):
    return Hamiltonian_data(
        structure_data=structure,
        wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data()),
        coulomb_potential_data=coulomb,
    )


def power_model(power: int = POWER_L):
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
    ham = _hamiltonian(structure, geminal, Coulomb_potential_data(structure_data=structure))
    return ham, np.zeros((1, 3)), np.array([float(POWER_Z)])


def power_configuration(x: float):
    return np.array([[x, 0.3, 0.4]]), np.zeros((0, 3))


def h2_model():
    """H2 all-electron cc-pVDZ (Cartesian), no Jastrow, and its start configuration."""
    structure, _, _, _, geminal, coulomb = read_trexio_file(str(TREXIO_DIR / "H2_ae_ccpvdz_cart.h5"), store_tuple=True)
    ham = _hamiltonian(structure, geminal, coulomb)
    nuclei = np.asarray(structure.positions, dtype=np.float64)
    charges = np.asarray(structure.atomic_numbers, dtype=np.float64)
    start = (np.array([nuclei[0] + [0.0, 0.2, 0.1]]), np.array([nuclei[1] + [0.0, -0.1, 0.2]]))
    return ham, nuclei, charges, start


def water_model():
    """Water ccECP cc-pVQZ, no Jastrow, valence charges and the PCG64(2025) start configuration."""
    structure, _, _, _, geminal, coulomb = read_trexio_file(str(TREXIO_DIR / "water_ccecp_ccpvqz.h5"), store_tuple=True)
    ham = _hamiltonian(structure, geminal, coulomb)
    nuclei = np.asarray(structure.positions, dtype=np.float64)
    charges = np.asarray(structure.atomic_numbers, dtype=np.float64) - np.asarray(coulomb.z_cores, dtype=np.float64)
    rng = np.random.default_rng(2025)
    up = nuclei[rng.integers(0, len(nuclei), geminal.num_electron_up)] + rng.normal(size=(geminal.num_electron_up, 3)) * 0.5
    dn = nuclei[rng.integers(0, len(nuclei), geminal.num_electron_dn)] + rng.normal(size=(geminal.num_electron_dn, 3)) * 0.5
    return ham, nuclei, charges, (up, dn)


def power_log_psi(position: np.ndarray, power: int = POWER_L) -> float:
    """Closed-form ln|psi| of the normalized x^power exp(-a r^2) primitive (no production code)."""
    a = POWER_EXPONENT
    ln_norm = -0.5 * (lgamma(power + 0.5) - (power + 0.5) * log(2 * a) + 2 * (lgamma(0.5) - 0.5 * log(2 * a)))
    return ln_norm + power * log(abs(float(position[0]))) - a * float(position @ position)


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


def normal_for_shift(before, shift: float, spin: str, nuclei, charges) -> float:
    """Normal draw that moves electron 0 of ``spin`` by ``shift`` bohr along x."""
    return float(shift / (width(moved(before, spin), nuclei, charges) * DT))


def proposed(before, normal: float, spin: str, nuclei, charges):
    """The proposal of one x-axis move, recomputed in numpy."""
    up, dn = before[0].copy(), before[1].copy()
    position = (up if spin == "up" else dn)[0]
    position[0] += normal * width(moved(before, spin), nuclei, charges) * DT
    return up, dn


def reference_geminal(ham, configuration, model: str) -> np.ndarray:
    """G rebuilt from the positions, or the closed-form 1x1 G = psi (power model)."""
    if model == "power":
        x = configuration[0][0]
        return np.array([[np.exp(power_log_psi(x)) * np.sign(x[0]) ** POWER_L]])
    return np.asarray(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, *configuration), dtype=np.float64)


def reference_ln_as_factor(matrix: np.ndarray) -> float:
    """Independent ln R_AS = -3/8 (ln S + ln F) of ``matrix``, evaluated with per-row rescaling.

    ``G = D Gs`` with ``D`` the row max-norms, so ``G^-1 = Gs^-1 D^-1`` and every
    logarithm is formed from the well-scaled ``Gs`` (numpy inverse) and ``ln D``;
    nothing underflows or overflows. ``-inf`` for an exactly zero row.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (1, 1):
        return 0.0 if matrix[0, 0] != 0.0 else -np.inf
    scale = np.max(np.abs(matrix), axis=1)
    if np.any(scale == 0.0):
        return -np.inf
    scaled = matrix / scale[:, None]
    ln_scale = np.log(scale)
    with np.errstate(divide="ignore"):
        ln_rows = 2.0 * ln_scale + np.log(np.sum(scaled**2, axis=1))
        ln_cols = np.logaddexp.reduce(2.0 * ln_scale[:, None] + 2.0 * np.log(np.abs(scaled)), axis=0)
        ln_inverse_terms = 2.0 * np.log(np.abs(np.linalg.inv(scaled))) - 2.0 * ln_scale[None, :]
    ln_f = np.logaddexp.reduce(ln_inverse_terms.ravel())
    ln_s = min(np.min(ln_rows), np.min(ln_cols))
    return float(-3.0 / 8.0 * (ln_s + ln_f))


def ln_abs_psi(ham, configuration, model: str) -> float:
    """Reference ln|Psi|: rebuilt-G slogdet or the closed form (power model); -inf for a singular G."""
    if model == "power":
        return power_log_psi(configuration[0][0])
    sign, value = np.linalg.slogdet(reference_geminal(ham, configuration, model))
    return float(value) if sign != 0 else -np.inf


def exact_acceptance(ham, before, after, spin: str, nuclei, charges, model: str, epsilon: float) -> float:
    """AS-guided acceptance min(1, |Psi'/Psi|^2 T'/T (g(R_AS')/g(R_AS))^2), g(R) = max(R, eps)/R, in the log domain.

    With ``epsilon == 0`` the guiding function is Psi itself (g = 1). A proposal with
    Psi' = 0 has acceptance 0.
    """
    a, b = moved(before, spin), moved(after, spin)
    fa, fb = width(a, nuclei, charges) * DT, width(b, nuclei, charges) * DT
    ln_transition = log(fa / fb) - float(np.sum((b - a) ** 2)) / 2 * (fb ** (-2) - fa ** (-2))
    ln_new, ln_old = ln_abs_psi(ham, after, model), ln_abs_psi(ham, before, model)
    assert np.isfinite(ln_old), "reference precondition: the current state has Psi != 0"
    if not np.isfinite(ln_new):
        return 0.0
    ln_guide = 0.0
    if epsilon > 0.0:
        ln_eps = log(epsilon)
        ln_g = [max(0.0, ln_eps - reference_ln_as_factor(reference_geminal(ham, c, model))) for c in (before, after)]
        ln_guide = ln_g[1] - ln_g[0]
    return float(np.exp(min(0.0, 2.0 * (ln_new - ln_old + ln_guide) + ln_transition)))


def case(name: str):
    """(ham, nuclei, charges, model, spin, before, normal) of one named single-move witness.

    * ``power-underflow`` (#131): x^12 model, x = 1e-14 -> 0.3; |G| ~ 4e-171 before, exact acceptance 1.
    * ``power-cancel`` (#134): x^12 model, x = 1.0 -> 0.05; |R| ~ 1e-15.6, exact acceptance ~4e-31.
    * ``h2-<spin>-<d>`` (#134): H2 electron 0 of ``spin`` moved +d bohr along x
      (d = 12: log10|R| ~ -8.6; d = 17: ~ -16.3).
    * ``water-up-30`` (#134): water up electron 0 moved +30 bohr along x (log10|R| ~ -23).
    """
    if name.startswith("power"):
        ham, nuclei, charges = power_model()
        x0, x1 = (1.0e-14, 0.3) if name == "power-underflow" else (1.0, 0.05)
        before = power_configuration(x0)
        return ham, nuclei, charges, "power", "up", before, normal_for_shift(before, x1 - x0, "up", nuclei, charges)
    if name.startswith("h2"):
        _, spin, distance = name.split("-")
        ham, nuclei, charges, before = h2_model()
        return ham, nuclei, charges, "h2", spin, before, normal_for_shift(before, float(distance), spin, nuclei, charges)
    if name == "water-up-30":
        ham, nuclei, charges, before = water_model()
        return ham, nuclei, charges, "water", "up", before, normal_for_shift(before, 30.0, "up", nuclei, charges)
    raise KeyError(name)

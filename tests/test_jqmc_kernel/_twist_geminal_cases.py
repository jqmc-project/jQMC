"""Complex k != Gamma geminal cases and an independent complex wavefunction reference.

The cases share one committed all-electron periodic fixture,
``H4_pbc_ae_k025_cart.h5`` (four H atoms in a cubic 15.1 bohr cell, twist
``k_frac = (0.25, 0, 0)``, complex Bloch AO leaves, two up and two down electrons),
with the reviewed J1+J2 recipe of ``_generated_matrix_base._build_ham_for_cell``.
No ECP and no J3 at the twist.

``CASES`` selects the geminal representation:

* ``sd_tr``      committed SD, ``time_reversal=True`` (Convention B, down = conj(up) on
  the ``-k`` leaf); the MO block is a Wirtinger pair.
* ``sd_untied``  SD with ``time_reversal=False``; the down MO coefficients receive a fixed
  complex perturbation so the two spin channels are genuinely independent holomorphic
  blocks.
* ``agp``        the reviewed full-rank dense AO AGP lift of the SD (complex lambda).
* ``agpn``       a genuine natural-orbital AGPn with four MOs for two pairs: the two
  occupied MOs plus two fixed complex rows, S-retracted, ``time_reversal=True`` and the
  non-uniform real occupations ``LAMBDA_DIAG``. With more MOs than pairs the occupations
  mix the orbitals, so ``d ln Psi / d lambda_k`` is complex (with only two MOs it would be
  exactly ``1 / lambda_k``).

``OPEN_SHELL_CASES`` re-spin the same four electrons as three up and one down
(``OPEN_SHELL_ELECTRONS``) so the unpaired geminal columns are live:

* ``agp_open``   the ``agp`` paired lambda plus two complex unpaired AO columns (the second
  occupied MO row and one fixed complex row), ``time_reversal=True`` (restricted open shell).
* ``agpn_open``  the ``agpn`` natural orbitals and occupations plus a fixed complex
  ``lambda_unpaired`` block of shape ``(4, 2)``.

``ln_psi_reference`` evaluates ``ln Psi`` (complex logarithm) from the production AO values
only: MO contraction, geminal assembly, determinant and the Jastrow exponent are written
here from the representation's defining formula, so a real projection anywhere in the
production geminal / determinant / kinetic chain moves the production side only.
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

import os

import jax
import jax.numpy as jnp
import numpy as np

from jqmc.atomic_orbital import compute_AOs
from jqmc.determinant import AGP_data, AGPn_data, SD_data, _smetric_polar_retract
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import compute_Jastrow_part
from jqmc.molecular_orbital import MOs_data
from jqmc.wavefunction import Wavefunction_data

STEM = "H4_pbc_ae_k025_cart.h5"
CASES = ("sd_tr", "sd_untied", "agp", "agpn")
OPEN_SHELL_CASES = ("agp_open", "agpn_open")
OPEN_SHELL_ELECTRONS = (3, 1)
LAMBDA_DIAG = (1.0, 0.5, 0.25, 0.125)
_TREXIO = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")


def _with_geminal(hamiltonian: Hamiltonian_data, geminal) -> Hamiltonian_data:
    wavefunction = hamiltonian.wavefunction_data.replace(geminal_data=geminal)
    return hamiltonian.replace(wavefunction_data=wavefunction)


def _complex_rows(rng: np.random.Generator, shape: tuple[int, ...], scale: float) -> np.ndarray:
    return scale * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def build_case(kind: str) -> Hamiltonian_data:
    """Return the Hamiltonian of one twist geminal case (deterministic, no RNG state leaks)."""
    from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell

    if kind == "agp":
        hamiltonian = _build_ham_for_cell(STEM, True, None, "AGP")
        hamiltonian.sanity_check()
        return hamiltonian
    hamiltonian = _build_ham_for_cell(STEM, True, None, "SD")
    sd = hamiltonian.wavefunction_data.geminal_data
    n_up_open, n_dn_open = OPEN_SHELL_ELECTRONS
    if kind == "agp_open":
        agp = _build_ham_for_cell(STEM, True, None, "AGP").wavefunction_data.geminal_data
        occupied = np.asarray(sd.orb_data_up_spin.mo_coefficients)
        extra = _complex_rows(np.random.Generator(np.random.PCG64(13)), (1, occupied.shape[1]), 0.5 * np.max(np.abs(occupied)))
        unpaired = np.vstack([occupied[1:2], extra]).T
        lambda_matrix = np.hstack([np.asarray(agp.lambda_matrix), unpaired]).astype(np.complex128)
        open_agp = AGP_data(
            num_electron_up=n_up_open,
            num_electron_dn=n_dn_open,
            orb_data_up_spin=agp.orb_data_up_spin,
            orb_data_dn_spin=agp.orb_data_dn_spin,
            lambda_matrix=lambda_matrix,
            time_reversal=True,
        )
        hamiltonian = _with_geminal(hamiltonian, open_agp)
        hamiltonian.sanity_check()
        return hamiltonian
    if kind == "sd_tr":
        pass
    elif kind == "sd_untied":
        dn = np.asarray(sd.orb_data_dn_spin.mo_coefficients)
        perturbation = _complex_rows(np.random.Generator(np.random.PCG64(7)), dn.shape, 0.05 * np.max(np.abs(dn)))
        hamiltonian = _with_geminal(
            hamiltonian,
            sd.replace(
                time_reversal=False,
                orb_data_dn_spin=sd.orb_data_dn_spin.replace(mo_coefficients=dn + perturbation),
            ),
        )
    elif kind in ("agpn", "agpn_open"):
        occupied = np.asarray(sd.orb_data_up_spin.mo_coefficients)
        extra = _complex_rows(np.random.Generator(np.random.PCG64(11)), occupied.shape, 0.5 * np.max(np.abs(occupied)))
        up_aos = sd.orb_data_up_spin.aos_data
        c_up = _smetric_polar_retract(np.vstack([occupied, extra]), up_aos)
        n_mo = c_up.shape[0]
        mos_up = MOs_data(num_mo=n_mo, aos_data=up_aos, mo_coefficients=np.ascontiguousarray(c_up))
        mos_dn = MOs_data(
            num_mo=n_mo,
            aos_data=sd.orb_data_dn_spin.aos_data,
            mo_coefficients=np.ascontiguousarray(np.conj(c_up)),
        )
        if kind == "agpn":
            agpn = AGPn_data.build(
                num_electron_up=int(sd.num_electron_up),
                num_electron_dn=int(sd.num_electron_dn),
                mos_up=mos_up,
                mos_dn=mos_dn,
                time_reversal=True,
            )
        else:
            unpaired = _complex_rows(np.random.Generator(np.random.PCG64(17)), (n_mo, n_up_open - n_dn_open), 0.5)
            agpn = AGPn_data.build(
                num_electron_up=n_up_open,
                num_electron_dn=n_dn_open,
                mos_up=mos_up,
                mos_dn=mos_dn,
                lambda_unpaired=unpaired,
                time_reversal=True,
            )
        hamiltonian = _with_geminal(hamiltonian, agpn.replace(lambda_diag=np.asarray(LAMBDA_DIAG, dtype=np.float64)))
    else:
        raise ValueError(f"unknown twist geminal case {kind!r}")
    hamiltonian.sanity_check()
    return hamiltonian


def reference_parameters(wavefunction: Wavefunction_data) -> dict[str, jax.Array]:
    """Complex parameters of ``ln_psi_reference`` that the production blocks map onto.

    ``cu`` / ``cd`` are the up / down MO rows (for a time-reversal block the down rows
    equal ``conj(cu)`` and are differentiated as the independent Wirtinger partner);
    ``lam`` is the AGP ambient lambda or the AGPn occupations (embedded as complex).
    """
    geminal = wavefunction.geminal_data
    parameters: dict[str, jax.Array] = {}
    if isinstance(geminal, (SD_data, AGPn_data)):
        parameters["cu"] = jnp.asarray(geminal.orb_data_up_spin.mo_coefficients, dtype=jnp.complex128)
        parameters["cd"] = jnp.asarray(geminal.orb_data_dn_spin.mo_coefficients, dtype=jnp.complex128)
    if isinstance(geminal, AGPn_data):
        parameters["lam"] = jnp.asarray(geminal.lambda_diag, dtype=jnp.complex128)
    if isinstance(geminal, AGP_data):
        parameters["lam"] = jnp.asarray(geminal.lambda_matrix, dtype=jnp.complex128)
    return parameters


def reference_geminal(wavefunction: Wavefunction_data, r_up, r_dn, parameters=None) -> jax.Array:
    """Geminal matrix from its defining formula over the production AO values."""
    geminal = wavefunction.geminal_data
    parameters = reference_parameters(wavefunction) if parameters is None else parameters
    ao_up = compute_AOs(geminal.orb_data_up_spin.aos_data if "cu" in parameters else geminal.orb_data_up_spin, r_up)
    ao_dn = compute_AOs(geminal.orb_data_dn_spin.aos_data if "cd" in parameters else geminal.orb_data_dn_spin, r_dn)
    x_up = parameters["cu"] @ ao_up if "cu" in parameters else ao_up.astype(jnp.complex128)
    x_dn = parameters["cd"] @ ao_dn if "cd" in parameters else ao_dn.astype(jnp.complex128)
    if isinstance(geminal, SD_data):
        return x_up[: geminal.num_occ_dn].T @ x_dn
    if isinstance(geminal, AGPn_data):
        paired = jnp.diag(parameters["lam"]) @ x_dn
        unpaired = np.asarray(geminal.lambda_unpaired)
        if unpaired.size:
            paired = jnp.hstack([paired, jnp.asarray(unpaired, dtype=jnp.complex128)])
        return x_up.T @ paired
    columns = parameters["lam"][:, : x_dn.shape[0]] @ x_dn
    if parameters["lam"].shape[1] > x_dn.shape[0]:
        # Open shell: the trailing lambda columns are the unpaired orbitals in the up basis.
        columns = jnp.hstack([columns, parameters["lam"][:, x_dn.shape[0] :]])
    return x_up.T @ columns


def ln_psi_reference(wavefunction: Wavefunction_data, r_up, r_dn, parameters=None) -> jax.Array:
    """Complex ``ln Psi = J + ln det G`` with the geminal of :func:`reference_geminal`."""
    jastrow = compute_Jastrow_part(wavefunction.jastrow_data, r_up, r_dn)
    return jastrow.astype(jnp.complex128) + jnp.log(jnp.linalg.det(reference_geminal(wavefunction, r_up, r_dn, parameters)))


def psi_reference(wavefunction: Wavefunction_data, r_up, r_dn) -> jax.Array:
    """Complex ``Psi``; its phase is the phase of ``det G`` (the Jastrow is real)."""
    return jnp.exp(ln_psi_reference(wavefunction, r_up, r_dn))


def kinetic_reference(wavefunction: Wavefunction_data, r_up, r_dn) -> jax.Array:
    """Per-electron ``-1/2 lap Psi / Psi`` from the complex Hessian of ``psi_reference``."""
    n_up = r_up.shape[0]
    positions = jnp.concatenate((jnp.asarray(r_up), jnp.asarray(r_dn)))

    def psi(points):
        return psi_reference(wavefunction, points[:n_up], points[n_up:])

    hessian = jax.jacfwd(jax.jacfwd(psi))(positions)
    return -0.5 * jnp.einsum("ijij->i", hessian) / psi(positions)

"""Down-determinant conjugation leaves every real fixed-phase kernel quantity unchanged (XLA).

For an LSDA Slater determinant with a real Jastrow factor and a real
Hamiltonian (no vector potential, no spin-orbit coupling), the state whose
down-spin orbitals are the stored beta orbitals at ``+k`` and the state whose
down-spin orbitals are their complex conjugates (the ``-k`` orbitals of the
same spin channel) agree configuration by configuration in ``|Psi|``, in the
real local energy, in every real non-local ECP candidate and in every real
LRDMC projection term. The imaginary parts of down-electron terms change
sign; they are not compared. The orbital values, gradients and Laplacians
must be exact conjugates, so a payload or twist mismatch on the conjugated
leaf cannot pass through the energy-level comparison.
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

import numpy as np
import pytest

from jqmc._precision import get_tolerance_min
from jqmc._setting import ATOL_NONVACUITY_DISTINCTION, RTOL_NONVACUITY_DISTINCTION
from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors
from jqmc.hamiltonians import compute_local_energy
from jqmc.jqmc_gfmc import _compute_V_elements_t
from jqmc.molecular_orbital import compute_MOs, compute_MOs_grad, compute_MOs_laplacian
from jqmc.wavefunction import (
    compute_discretized_kinetic_energy,
    compute_kinetic_energy_all_elements,
    evaluate_ln_wavefunction,
)
from tests._opposite_spin_twist_cases import KINDS, configurations, conjugate_partner, convert, rotation

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]

_ALAT = 0.30
_ORBITAL_ZONES = ("ao_eval", "ao_bloch_eval", "ao_grad_lap", "mo_eval", "mo_grad", "mo_lap")
_ENERGY_ZONES = _ORBITAL_ZONES + (
    "jastrow_eval",
    "jastrow_grad_lap",
    "jastrow_ratio",
    "det_eval",
    "det_grad_lap",
    "det_ratio",
    "wf_eval",
    "wf_kinetic",
    "wf_ratio",
    "coulomb",
    "ewald",
    "local_energy",
)


def _terms(ham, up, dn, rt):
    wf = ham.wavefunction_data
    dn_orb = wf.geminal_data.orb_data_dn_spin
    t_up, t_dn = compute_kinetic_energy_all_elements(wf, up, dn)
    _, _, v_nl, _ = compute_ecp_non_local_parts_nearest_neighbors(
        coulomb_potential_data=ham.coulomb_potential_data, wavefunction_data=wf, r_up_carts=up, r_dn_carts=dn, RT=rt
    )
    _, _, hops = compute_discretized_kinetic_energy(alat=_ALAT, wavefunction_data=wf, r_up_carts=up, r_dn_carts=dn, RT=rt)
    out = {
        "dn_value": np.asarray(compute_MOs(dn_orb, dn)),
        "dn_grad": np.stack([np.asarray(g) for g in compute_MOs_grad(dn_orb, dn)]),
        "dn_lap": np.asarray(compute_MOs_laplacian(dn_orb, dn)),
        "ln_abs_psi": np.asarray(evaluate_ln_wavefunction(wf, up, dn)),
        "T_up": np.asarray(t_up),
        "T_dn": np.asarray(t_dn),
        "e_L": np.asarray(compute_local_energy(ham, up, dn, rt)),
        "V_nonlocal": np.asarray(v_nl),
        "hops": np.asarray(hops),
    }
    for move in ("tmove", "dltmove"):
        out[move] = np.asarray(_compute_V_elements_t(ham, up, dn, rt, move, _ALAT))
    return out


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    directory = tmp_path_factory.mktemp("lsda_twist_pair")
    return lambda kind: convert(kind, directory)


@pytest.mark.parametrize("kind", KINDS)
def test_down_conjugation_preserves_real_fixed_phase_terms(kind, converted):
    """Orbitals conjugate exactly; |Psi|, real e_L, ECP candidates and LRDMC terms agree.

    Down-electron orbital values, gradients and Laplacians of the partner are
    the conjugates of the converted ones, up-electron kinetic terms are equal
    and down-electron ones are conjugate. ``ln|Psi|``, ``Re e_L``, every real
    non-local ECP candidate, the real discretized-kinetic hop amplitudes with
    their fixed-node clipping signs, and the real diagonal, off-diagonal and
    measurement LRDMC energies for the T-move and DLT-move are equal.
    """
    ham = converted(kind)
    partner = conjugate_partner(ham)
    g = ham.wavefunction_data.geminal_data
    assert not g.time_reversal and g.num_electron_up > g.num_electron_dn
    o_atol, o_rtol = get_tolerance_min(_ORBITAL_ZONES, "strict")
    e_atol, e_rtol = get_tolerance_min(_ENERGY_ZONES, "strict")
    rt = rotation()
    for up, dn in configurations(ham):
        a, b = _terms(ham, up, dn, rt), _terms(partner, up, dn, rt)
        assert np.max(np.abs(np.imag(a["dn_value"]))) > o_atol
        # Genuinely distinct spin channels: the down orbitals are neither the
        # first N_dn up orbitals nor their conjugates.
        up_at_dn = np.asarray(compute_MOs(g.orb_data_up_spin, dn))[: g.num_electron_dn]
        for candidate in (up_at_dn, np.conj(up_at_dn)):
            assert not np.allclose(
                np.abs(candidate), np.abs(a["dn_value"]), atol=ATOL_NONVACUITY_DISTINCTION, rtol=RTOL_NONVACUITY_DISTINCTION
            )
        for key in ("dn_value", "dn_grad", "dn_lap"):
            np.testing.assert_allclose(b[key], np.conj(a[key]), atol=o_atol, rtol=o_rtol, err_msg=key)
        np.testing.assert_allclose(b["T_up"], a["T_up"], atol=e_atol, rtol=e_rtol, err_msg="T_up")
        np.testing.assert_allclose(b["T_dn"], np.conj(a["T_dn"]), atol=e_atol, rtol=e_rtol, err_msg="T_dn")
        assert np.max(np.abs(a["V_nonlocal"])) > e_atol
        for key in ("ln_abs_psi", "e_L", "V_nonlocal", "hops", "tmove", "dltmove"):
            np.testing.assert_allclose(np.real(b[key]), np.real(a[key]), atol=e_atol, rtol=e_rtol, err_msg=key)
        np.testing.assert_array_equal(np.real(b["hops"]) < 0.0, np.real(a["hops"]) < 0.0)

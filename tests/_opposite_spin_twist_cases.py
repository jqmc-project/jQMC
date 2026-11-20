"""Common-twist versus opposite-twist down-spin determinants built from one LSDA input.

The QE ``O-lsda_k025`` fixture (ccECP oxygen, ``k = (1/4, 1/4, 1/4)``,
``N_up = 4 > N_dn = 2``) is converted once per AO kind with an active real
J2/J3 Jastrow. Its converted down-spin orbitals are paired with their whole
complex conjugate: the opposite-sign ``with_negated_k()`` leaf with
conjugated coefficients (an involution, so the pair is the same whichever
member the converter emits). The two states share the up channel, the
Jastrow factor and the Hamiltonian; they differ only by conjugation of the
complete down determinant.
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

from pathlib import Path

import numpy as np

from jqmc.determinant import SD_data
from jqmc.hamiltonians import Hamiltonian_data
from tests._test_data_cache import qe_example_files_root

KINDS = ("pgto", "pw", "pgto+pw", "bspline", "pgto+bspline")
_SAVE = qe_example_files_root() / "O-lsda_k025" / "out" / "o_k025.save"


def convert(kind: str, directory: Path) -> Hamiltonian_data:
    """Standalone QE conversion with J2 and an active, deterministic real J3."""
    from jqmc.jqmc_qe import run_qe2jqmc

    out = directory / f"ham_{kind.replace('+', '_')}.h5"
    j3 = "ao-small" if kind.startswith("pgto") else "mo"
    qe = {"qe_save_dir": str(_SAVE), "kind": kind, "upf_paths": [], "ecutwfc": None, "alpha_min": None}
    run_qe2jqmc(qe, {"j1_parameter": None, "j2_parameter": 1.0, "j3_basis_type": j3}, str(out))
    ham = Hamiltonian_data.load_from_hdf5(str(out))
    jd = ham.wavefunction_data.jastrow_data
    j3_data = jd.jastrow_three_body_data
    rng = np.random.default_rng(922)
    matrix = rng.uniform(-0.05, 0.05, size=np.shape(j3_data.j_matrix))
    matrix[:, :-1] = 0.5 * (matrix[:, :-1] + matrix[:, :-1].T)
    jd = jd.replace(jastrow_three_body_data=j3_data.replace(j_matrix=matrix))
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd))


def conjugate_partner(ham: Hamiltonian_data) -> Hamiltonian_data:
    """The same state with the complete down determinant conjugated.

    ``conj(phi(r)) = sum_mu conj(C_mu) conj(chi_mu(r))`` and
    ``conj(chi_mu) = with_negated_k()(chi_mu)`` for every Bloch AO kind, so the
    pair is (coefficients, leaf) -> (conj coefficients, negated-k leaf). The
    dataclass is built directly: the partner of a supported state is the other
    spin-twist convention, which ``SD_data.build`` may refuse by policy.
    """
    g = ham.wavefunction_data.geminal_data
    dn = g.orb_data_dn_spin
    partner = dn.replace(aos_data=dn.aos_data.with_negated_k(), mo_coefficients=np.conj(np.asarray(dn.mo_coefficients)))
    new_g = SD_data(
        num_electron_up=g.num_electron_up,
        num_electron_dn=g.num_electron_dn,
        orb_data_up_spin=g.orb_data_up_spin,
        orb_data_dn_spin=partner,
        time_reversal=g.time_reversal,
    )
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=new_g))


def configurations(ham: Hamiltonian_data) -> list[tuple[np.ndarray, np.ndarray]]:
    """Two configurations with every down electron in a distinct regime.

    The first down electron is in the ECP core region, so its non-local
    candidates are active. The second is close to a cell face, where the
    periodic image sum of every AO kind carries an O(1) Bloch phase, so an
    orbital that is conjugated only partly (coefficients, leaf or twist label)
    differs visibly from the true conjugate.
    """
    g = ham.wavefunction_data.geminal_data
    center = np.asarray(ham.structure_data.positions, dtype=np.float64)[0]
    rng = np.random.default_rng(23)
    out = []
    for face in ((4.6, 0.3, -0.2), (-0.4, 4.7, 0.5)):
        up = center + rng.normal(scale=1.0, size=(g.num_electron_up, 3))
        dn = np.stack([center + rng.normal(scale=0.4, size=3), center + np.asarray(face)])
        out.append((up, dn))
    return out


def rotation() -> np.ndarray:
    """A fixed, non-axis-aligned lattice / quadrature orientation."""
    t = 0.37
    return np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])


def opposite_twist_partner(orbitals):
    """The complete complex conjugate of ``orbitals``: its down-spin form at the opposite twist.

    Accepts ``MOs_data`` (dense or implicit identity) or a bare AO leaf; real /
    Gamma inputs are returned unchanged up to the identity conjugation.
    """
    from jqmc.molecular_orbital import MOs_data

    if not isinstance(orbitals, MOs_data):
        return orbitals.with_negated_k() if hasattr(orbitals, "with_negated_k") else orbitals
    aos = orbitals.aos_data.with_negated_k() if hasattr(orbitals.aos_data, "with_negated_k") else orbitals.aos_data
    if orbitals.identity:
        return MOs_data.build_identity(aos_data=aos)
    return orbitals.replace(aos_data=aos, mo_coefficients=np.conj(np.asarray(orbitals.mo_coefficients)))

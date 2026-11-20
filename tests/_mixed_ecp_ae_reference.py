"""Mixed ECP / all-electron two-nucleus molecule and its independent references.

Shared by the issue #20/#21 detectors: the XLA/Rust non-local energy test
(``tests/test_rs_kernel/test_ecp_local.py``) and the XLA ECP force test
(``tests/test_jqmc_kernel/test_jqmc_derivatives.py``).

Two nuclei, ONE of which carries an ECP; the other is all-electron (no ECP
rows at all). Two up electrons, one next to each nucleus, in a 2x2 Slater
determinant of single-primitive s Gaussians (identity MO matrix, so the AO
normalisation cancels in every ratio). The non-local projector of an electron
lives on its NN nearest ECP nuclei; an all-electron nucleus projects nothing
and must not displace the ECP centre from that set. With a single ECP
nucleus every NN >= 1 therefore gives the untruncated sum over electrons
    V_nl = sum_e sum_l V_l(r_eI) (2l+1) sum_v w_v P_l(cos theta_ev) Psi(r_e -> R_I + r_eI u_v) / Psi,
evaluated below by hand on an authored octahedral rule (u_v = +-x, +-y, +-z,
w_v = 1/6), Legendre P_0 = 1, P_1 = cos, and a closed-form determinant. No
production nearest-set, padding, mesh or wavefunction code is used.
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

MIXED_AE_POSITIONS = np.array([[0.0, 0.0, 0.0], [1.5, 0.2, -0.1]])
MIXED_AE_EXPONENTS = np.array([0.8, 0.9])
# Electron 0 is nearest nucleus 1 and electron 1 nearest nucleus 0.
MIXED_AE_R_UP = np.array([[1.2, 0.3, 0.0], [0.4, -0.2, 0.3]])
# Non-local channels (l, coefficient, exponent, power) and the local channel of the ECP nucleus.
MIXED_AE_NONLOCAL = ((0, 1.3, 0.7, 2), (1, -0.6, 0.9, 2))
MIXED_AE_LOCAL = (2, 1.0, 0.4, 1)


def mixed_ecp_ae_system(ecp_atom):
    """Build the two-nucleus molecule with the ECP on nucleus ``ecp_atom`` only; return (Coulomb data, wavefunction)."""
    from jqmc.atomic_orbital import AOs_sphe_gto_data
    from jqmc.coulomb_potential import Coulomb_potential_data
    from jqmc.determinant import SD_data
    from jqmc.jastrow_factor import Jastrow_data
    from jqmc.molecular_orbital import MOs_data
    from jqmc.structure import Structure_data
    from jqmc.wavefunction import Wavefunction_data

    structure = Structure_data.build(
        positions=MIXED_AE_POSITIONS,
        atomic_numbers=(1, 1),
        element_symbols=("H", "H"),
        atomic_labels=("H1", "H2"),
        pbc_flag=False,
    )
    aos = AOs_sphe_gto_data.build(
        structure_data=structure,
        nucleus_index=(0, 1),
        num_ao=2,
        num_ao_prim=2,
        orbital_indices=(0, 1),
        exponents=MIXED_AE_EXPONENTS,
        coefficients=np.ones(2),
        angular_momentums=(0, 0),
        magnetic_quantum_numbers=(0, 0),
    )
    mos = MOs_data.build(aos_data=aos, mo_coefficients=np.eye(2))
    geminal = SD_data.build(num_electron_up=2, num_electron_dn=0, mos_up=mos, mos_dn=mos, time_reversal=False)
    wf = Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data())
    terms = MIXED_AE_NONLOCAL + (MIXED_AE_LOCAL,)
    max_ang_mom_plus_1 = [0, 0]
    max_ang_mom_plus_1[ecp_atom] = MIXED_AE_LOCAL[0]
    cp = Coulomb_potential_data(
        structure_data=structure,
        ecp_flag=True,
        z_cores=(0.0, 0.0),
        max_ang_mom_plus_1=tuple(max_ang_mom_plus_1),
        num_ecps=len(terms),
        nucleus_index=(ecp_atom,) * len(terms),
        ang_moms=tuple(t[0] for t in terms),
        coefficients=tuple(t[1] for t in terms),
        exponents=tuple(t[2] for t in terms),
        powers=tuple(t[3] for t in terms),
    )
    cp.sanity_check()
    return cp, wf


def mixed_ecp_ae_reference(ecp_atom):
    """Independent projection, evaluated entirely in float64 (the ideal physical value)."""
    directions = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    weights = np.full(6, 1.0 / 6.0)
    exponents = np.asarray(MIXED_AE_EXPONENTS, dtype=np.float64)

    def psi(r):
        sq = np.sum((r[:, None, :] - MIXED_AE_POSITIONS[None, :, :]) ** 2, axis=-1)
        return np.linalg.det(np.exp(-exponents[None, :] * sq))

    centre = MIXED_AE_POSITIONS[ecp_atom]
    psi0 = psi(MIXED_AE_R_UP)
    total = 0.0
    for e in range(len(MIXED_AE_R_UP)):
        d = MIXED_AE_R_UP[e] - centre
        r = np.linalg.norm(d)
        for l, c, a, p in MIXED_AE_NONLOCAL:
            v_l = c * r ** (p - 2) * np.exp(-a * r * r)
            for u, w in zip(directions, weights):
                moved = MIXED_AE_R_UP.copy()
                moved[e] = centre + r * u
                legendre = 1.0 if l == 0 else float(np.dot(d / r, u))
                total += v_l * (2 * l + 1) * legendre * w * psi(moved) / psi0
    return total


def mixed_ecp_ae_potential_jnp(r, nuclei, ecp_atom):
    """Independent V_ecp (local + non-local) of the mixed ECP/AE molecule as a jnp function.

    Same authored octahedral rule, Legendre P_0/P_1 and closed-form 2x2 determinant
    as ``mixed_ecp_ae_reference``, but written over the electron positions ``r`` AND
    the nuclear positions ``nuclei`` (AO centres and the ECP centre move with them),
    so ``jax.grad`` gives the exact d V_ecp / d r and d V_ecp / d R with the
    quadrature rotation held fixed. No production ECP, nearest-set, mesh,
    wavefunction or force code is used; JAX supplies only the differentiation.
    """
    import jax.numpy as jnp

    directions = jnp.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=jnp.float64)
    exponents = jnp.asarray(MIXED_AE_EXPONENTS, dtype=jnp.float64)

    def psi(rr):
        m = jnp.exp(-exponents[None, :] * jnp.sum((rr[:, None, :] - nuclei[None, :, :]) ** 2, axis=-1))
        return m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]

    centre = nuclei[ecp_atom]
    psi0 = psi(r)
    _, c_loc, a_loc, p_loc = MIXED_AE_LOCAL
    total = 0.0
    for e in range(r.shape[0]):
        d = r[e] - centre
        dist = jnp.sqrt(jnp.sum(d * d))
        total = total + c_loc * dist ** (p_loc - 2) * jnp.exp(-a_loc * dist * dist)
        for l, c, a, p in MIXED_AE_NONLOCAL:
            v_l = c * dist ** (p - 2) * jnp.exp(-a * dist * dist)
            for v in range(directions.shape[0]):
                moved = r.at[e].set(centre + dist * directions[v])
                legendre = 1.0 if l == 0 else jnp.dot(d / dist, directions[v])
                total = total + v_l * (2 * l + 1) * legendre * psi(moved) / psi0 / directions.shape[0]
    return total

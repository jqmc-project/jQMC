"""Hoist the spherical->Cartesian basis conversion OUT of the per-walker jit.

The analytic R/eps (force/stress) paths take Cartesian-GTO AO derivatives, so for a
spherical-GTO wavefunction they call ``orb_to_cart_pieces`` -> the production
``_aos_sphe_to_cart``. That conversion is a host-side STRUCTURAL op (shell grouping,
``np.allclose`` on exponents, the c2s transform matrix) which cannot run under
``jit`` with a traced hamiltonian (it converts traced exponents via ``np.asarray``).

But the basis is STATIC -- exponents / angular momenta / the c2s transform never
change during optimization or across walkers (only positions and the variational
params do). So convert once, EAGERLY, before building the ``jit(vmap(...))`` closure:
``MOs_data.to_cartesian`` rebuilds the MO over a Cartesian AO basis while preserving
the molecular-orbital *functions* (radial-independent per-shell transform), so the
orbital values / grads / laplacians -- hence every gradient, including the variational
``lambda_unpaired`` / Jastrow blocks and the physical forces / stress -- are unchanged.
After this, ``orb_to_cart_pieces`` hits the Cartesian (idempotent) branch and the
R/eps paths jit-compile. Idempotent on already-Cartesian inputs.

The c (O_k / dE_L/dc) axis never needs this -- it uses the MO value/grad/lap kernels
directly (jit-safe already), so this hoist only matters for spherical R / eps.
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

from jqmc.hamiltonians import Hamiltonian_data


def cartesianize_orbitals(hamiltonian_data: Hamiltonian_data) -> Hamiltonian_data:
    """Return ``hamiltonian_data`` with its spherical orbital leaves converted to
    Cartesian (geminal MOs + J3 basis), keeping the geminal type / variational
    parametrization. Call ONCE before the per-walker jit. No-op if already Cartesian,
    or if the basis is periodic/exotic (those convert per-leaf inside the kernels)."""
    wf = hamiltonian_data.wavefunction_data

    gd = wf.geminal_data
    up_orbital = gd.orb_data_up_spin
    up_aos = up_orbital.aos_data if type(up_orbital).__name__ == "MOs_data" else up_orbital
    host_cartesian_transform_types = (
        "AOs_cart_gto_data",
        "AOs_sphe_gto_data",
        "AOs_sphe_pgto_data",
        "AOs_sphe_pgto_complex_data",
    )
    if type(up_aos).__name__ not in host_cartesian_transform_types:
        return hamiltonian_data  # periodic/exotic: per-leaf cart conversion happens in the kernels
    if type(gd.orb_data_up_spin).__name__ == "MOs_data":
        new_gd = gd.replace(
            orb_data_up_spin=gd.orb_data_up_spin.to_cartesian(),
            orb_data_dn_spin=gd.orb_data_dn_spin.to_cartesian(),
        )
    else:  # bare-AO geminal (AGP): the geminal-level convert (transforms lambda_matrix)
        new_gd = gd.to_cartesian()

    jd = wf.jastrow_data
    j3 = jd.jastrow_three_body_data
    new_jd = jd.replace(jastrow_three_body_data=j3.to_cartesian()) if j3 is not None else jd

    return hamiltonian_data.replace(wavefunction_data=wf.replace(geminal_data=new_gd, jastrow_data=new_jd))

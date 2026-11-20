"""The per-nucleus core-electron count a fixture's own pseudopotential removes.

``Jastrow_one_body_data`` uses ``core_electrons`` as ``Z_eff = Z - core_electrons``
and, through ``c = (2 Z_eff)^(1/4)``, it sets the electron-nucleus cusp constant
that the J1 value, its gradient and Laplacian, the kinetic energy, and every
force / parameter derivative built on them inherit.

A hardcoded ``tuple([0.0] * n_atom)`` in a test builder means ``Z_eff == Z``, the
one value at which the whole subtraction is a no-op -- including for pseudized
systems, where it makes a silicon atom behave like bare Z = 14 and a chlorine like
bare Z = 17. Measured on Cl2/ccECP, J1 alone then reaches -188.3 and drags the composed Jastrow to ``exp(J) ~ 1e-65``. The value belongs
to the Hamiltonian the fixture already carries, not to an assumption about it,
and production threads exactly this ``z_cores -> core_electrons`` link
(``qe_to_jqmc.py``).

Reading it from the data also removes a trap the file names set: pseudization is
NOT visible in a stem name. ``H2_ecp_ccpvdz_cart.h5`` and ``H_ecp_ccpvdz_cart.h5``
carry ``ecp_flag=True`` but ``z_cores == 0`` (hydrogen ccECP removes no core), so
zero is correct there, while ``lithium.h5``, ``lithium_k025.h5``,
``lithium_twist_*.h5``, ``silicon.h5`` and ``water_ccecp_ccpvqz.h5`` carry no
``_ecp_`` in their names and remove 2, 2, 2, 10 and 2 electrons respectively.
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

from typing import Any


def core_electrons_of(coulomb_potential_data: Any, n_atom: int) -> tuple[float, ...]:
    """Return the per-nucleus core-electron counts, as ``init_jastrow_one_body_data`` wants them.

    All-electron fixtures carry ``ecp_flag=False`` and an EMPTY ``z_cores``, so the
    all-zero tuple is produced for them -- this helper is an identity for every
    all-electron system and only changes pseudized ones.
    """
    if not getattr(coulomb_potential_data, "ecp_flag", False):
        return tuple([0.0] * n_atom)
    z_cores = tuple(float(z) for z in coulomb_potential_data.z_cores)
    if len(z_cores) != n_atom:
        raise ValueError(f"z_cores has {len(z_cores)} entries but the structure has {n_atom} atoms.")
    return z_cores

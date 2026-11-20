"""jqmc: a JAX-based quantum Monte Carlo package.

The public API is exactly the names listed in ``__all__`` (and lazily re-exported here):
the drivers (:class:`MCMC`, :class:`GFMC_t`, :class:`GFMC_n`), the user-assembled data
containers (structure / Hamiltonian / wavefunction / Jastrow / geminal / one-particle
bases), the ``read_trexio_file`` loader, the ``cli_parameters`` schema, and the
evaluation kernels for atomic and molecular orbitals and for the Hamiltonian terms
(bare Coulomb channels, ECP local and non-local parts) at a given electron
configuration. Everything else in the sub-modules is internal machinery (fast-update
states, builders, file-format parsers) that requires knowledge of jqmc internals to use,
and is intentionally NOT part of the public API -- import it explicitly from its
sub-module if you must, but it may change without notice.

The Sphinx API reference documents exactly this ``__all__`` (``automodule:: jqmc``), so a
new sub-module is private-by-default: it is disclosed only once its public names are added
here.

Re-exports are LAZY (PEP 562 ``__getattr__``): ``import jqmc`` does not eagerly import any
sub-module, so importing the package is cheap and free of import-time side effects (heavy
JAX / MPI setup happens only when a public name is first accessed). ``from jqmc import MCMC``
triggers the load on demand.
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

import importlib
from typing import Any

# public name -> sub-module it lives in (relative to this package)
_PUBLIC = {
    "MCMC": "jqmc_mcmc",
    "GFMC_t": "jqmc_gfmc",
    "GFMC_n": "jqmc_gfmc",
    "Structure_data": "structure",
    "Coulomb_potential_data": "coulomb_potential",
    "Hamiltonian_data": "hamiltonians",
    "Wavefunction_data": "wavefunction",
    "Jastrow_data": "jastrow_factor",
    "Jastrow_one_body_data": "jastrow_factor",
    "Jastrow_two_body_data": "jastrow_factor",
    "Jastrow_three_body_data": "jastrow_factor",
    "Jastrow_DTN_data": "jastrow_factor",
    "SD_data": "determinant",
    "AGP_data": "determinant",
    "AGPn_data": "determinant",
    "to_AGPn": "determinant",
    "read_trexio_file": "trexio_wrapper",
    "cli_parameters": "jqmc_miscs",
    # one-particle basis containers and their evaluation kernels
    "AOs_cart_gto_data": "atomic_orbital",
    "AOs_sphe_gto_data": "atomic_orbital",
    "AOs_cart_pgto_data": "atomic_orbital",
    "AOs_sphe_pgto_data": "atomic_orbital",
    "AOs_cart_pgto_complex_data": "atomic_orbital",
    "AOs_sphe_pgto_complex_data": "atomic_orbital",
    "AOs_plane_wave_data": "atomic_orbital",
    "AOs_plane_wave_complex_data": "atomic_orbital",
    "AOs_bspline_data": "atomic_orbital",
    "AOs_bspline_complex_data": "atomic_orbital",
    "AOs_bspline_tiled_complex_data": "atomic_orbital",
    "AOs_composite_data": "atomic_orbital",
    "compute_AOs": "atomic_orbital",
    "compute_AOs_grad": "atomic_orbital",
    "compute_AOs_laplacian": "atomic_orbital",
    "MOs_data": "molecular_orbital",
    "compute_MOs": "molecular_orbital",
    "compute_MOs_grad": "molecular_orbital",
    "compute_MOs_laplacian": "molecular_orbital",
    # Hamiltonian terms at a given electron configuration
    "compute_bare_coulomb_potential": "coulomb_potential",
    "compute_bare_coulomb_potential_el_el": "coulomb_potential",
    "compute_bare_coulomb_potential_el_ion": "coulomb_potential",
    "compute_bare_coulomb_potential_ion_ion": "coulomb_potential",
    "compute_ecp_local_parts_all_pairs": "coulomb_potential",
    "compute_ecp_non_local_parts_all_pairs": "coulomb_potential",
    "compute_ecp_non_local_parts_nearest_neighbors": "coulomb_potential",
}

__all__ = list(_PUBLIC)


#: Version attributes, resolved on first access by ``jqmc._build_info`` (a git
#: subprocess for a source checkout), so ``import jqmc`` itself stays side-effect free.
_BUILD_INFO_ATTRIBUTES = {"__version__": "version", "__git_commit__": "git_commit"}


def __getattr__(name: str) -> Any:
    """Lazily import a public name from its sub-module on first access (PEP 562)."""
    if name in _BUILD_INFO_ATTRIBUTES:
        from ._build_info import build_info

        return getattr(build_info(), _BUILD_INFO_ATTRIBUTES[name])
    module = _PUBLIC.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)

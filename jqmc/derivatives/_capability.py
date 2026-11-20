"""PRODUCTION capability helpers for analytic gradients (NOT a dispatch gate).

The analytic dispatch (``__init__._grad_bundled`` with ``analytic=True``) is STRICT:
it always attempts the analytic path and an unimplemented axis/leaf raises
``NotImplementedError`` -- there is NO silent ``_auto`` fallback (``_auto`` is reached
only when a caller explicitly passes ``analytic=False`` as a test oracle). So this
module does NOT gate the analytic path. It provides only the production HELPERS the
driver / assemble actually branch on:

  * ``jit_ao_serveable`` -- the jit-bridge may only serve the AO leaves in
    ``SUPPORTED_AO_TYPES_JIT``; other leaves route through the eager dispatch.
  * ``_orb_no_radial_basis`` / ``_j3_basis_opt_orb_ok`` -- the assemble J3
    basis-opt guard uses these with ``SUPPORTED_J3_BASIS_OPT_AO``,
    ``NO_BASIS_PARAM_AO``, and ``COMPOSITE_BASIS_OPT_GTO_AO`` to distinguish a
    legitimately empty basis gradient from an unsupported radial basis.

Analytic coverage is not tracked here. It is checked behaviorally through strict
dispatch and analytic-versus-autograd parity. The live sets below are extended by
the assemble module at import time.
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

from logging import getLogger

from jqmc.atomic_orbital import AOs_base_data
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.molecular_orbital import MOs_data

logger = getLogger("jqmc").getChild(__name__)

# Extended by assemble at import time. AO types outside the JIT-safe set use the
# eager analytic dispatch.
SUPPORTED_AO_TYPES_JIT: set[str] = set()
# AO leaf types whose J3 radial exponent/coefficient gradients are analytic.
# MO-backed leaves are unwrapped by _j3_basis_opt_orb_ok. Composite leaves use
# COMPOSITE_BASIS_OPT_GTO_AO with parameter-free PW/B-spline siblings.
SUPPORTED_J3_BASIS_OPT_AO: set[str] = set()
SUPPORTS_FAST: bool = False
SUPPORTS_NONORTHO: bool = True


def _ao_type_names(aos: AOs_base_data) -> list[str]:
    """Class names of an AO leaf, descending into composite children."""
    tn = type(aos).__name__
    if tn == "AOs_composite_data":
        out: list[str] = []
        for child in aos.children:
            out += _ao_type_names(child)
        return out
    return [tn]


# AO leaf types with no optimizable radial parameters. They contribute an empty
# gradient when combined with an optimizable GTO/pGTO child.
NO_BASIS_PARAM_AO: set[str] = {
    "AOs_plane_wave_data",
    "AOs_plane_wave_complex_data",
    "AOs_bspline_data",
    "AOs_bspline_complex_data",
}
# GTO/pGTO child types accepted inside a composite basis-gradient route. Every
# other child must belong to NO_BASIS_PARAM_AO.
COMPOSITE_BASIS_OPT_GTO_AO: set[str] = {
    "AOs_cart_gto_data",
    "AOs_sphe_gto_data",
    "AOs_cart_pgto_data",
    "AOs_sphe_pgto_data",
    "AOs_cart_pgto_complex_data",
    "AOs_sphe_pgto_complex_data",
}


def _orb_no_radial_basis(orb_data: AOs_base_data | MOs_data) -> bool:
    """True iff the (MO-unwrapped) orbital leaf carries NO optimizable radial basis
    parameters at all -- a single ``NO_BASIS_PARAM_AO`` leaf (plane-wave / B-spline,
    whose basis is fixed by the kinetic cutoff / spline grid) or a composite whose
    children are ALL such paramless leaves (no GTO/pGTO child).

    For such a leaf the basis-parameter gradient (``opt_lambda_basis_*`` /
    ``opt_J3_basis_*``) is legitimately EMPTY/ZERO -- ``_auto`` (autograd) produces no
    such gradient entry either -- so the assemble guard must SKIP-to-zero rather than
    raise. This is DISTINCT from a leaf that DOES carry GTO/pGTO radial params of an
    unsupported variant (``_j3_basis_opt_orb_ok`` False while this also returns
    False), which is the genuine silent-wrong case the guard still raises on.
    Mirrors the ``NO_BASIS_PARAM_AO`` branch of the composite rules above."""
    aos = orb_data.aos_data if type(orb_data).__name__ == "MOs_data" else orb_data
    return all(t in NO_BASIS_PARAM_AO for t in _ao_type_names(aos))


def _j3_basis_opt_orb_ok(orb_data: AOs_base_data | MOs_data) -> bool:
    """True iff the three-body Jastrow orbital's basis-parameter grad is analytic.

    MO-backed orbitals are checked through their AO container. A single leaf must
    belong to ``SUPPORTED_J3_BASIS_OPT_AO``. A composite must contain at least
    one child in ``COMPOSITE_BASIS_OPT_GTO_AO``; every remaining child must be
    either another supported GTO/pGTO child or a parameter-free plane-wave or
    B-spline child. Real and complex pGTO children follow the same rule.
    """
    aos = orb_data.aos_data if type(orb_data).__name__ == "MOs_data" else orb_data
    if type(aos).__name__ == "AOs_composite_data":
        names = _ao_type_names(aos)
        if not any(t in COMPOSITE_BASIS_OPT_GTO_AO for t in names):
            return False
        return all((t in COMPOSITE_BASIS_OPT_GTO_AO) or (t in NO_BASIS_PARAM_AO) for t in names)
    return type(aos).__name__ in SUPPORTED_J3_BASIS_OPT_AO


def jit_ao_serveable(hamiltonian_data: Hamiltonian_data) -> bool:
    """True iff every AO leaf is in the JIT-SAFE SUPPORTED_AO_TYPES_JIT. The production MCMC driver
    runs the analytic backend through the jit bridge, which has only been verified for that AO set;
    other leaves (exotic PW/bspline analytic kernels are eager) route through the eager dispatch."""
    try:
        gd = hamiltonian_data.wavefunction_data.geminal_data
        for od in (gd.orb_data_up_spin, gd.orb_data_dn_spin):
            aos = od.aos_data if type(od).__name__ == "MOs_data" else od
            if any(t not in SUPPORTED_AO_TYPES_JIT for t in _ao_type_names(aos)):
                return False
        return True
    except Exception:
        # Keep the probe non-raising, but surface structural regressions before
        # using the eager fallback.
        logger.warning("jit_ao_serveable probe raised; falling back to the eager (non-jit) route.", exc_info=True)
        return False

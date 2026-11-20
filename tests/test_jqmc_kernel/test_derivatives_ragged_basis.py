"""Non-shell-major (ragged) periodic pGTO basis support -- permutation oracle.

A QE-converted periodic spherical-pGTO Hamiltonian is NOT shell-major: on some atoms two
same-``l`` shells are stored with their ``2l+1`` magnetic components INTERLEAVED
(``[m0_A, m0_B, m1_A, m1_B, ...]``), so a single shell's AOs are non-contiguous and
the two shells differ only in their radial exponents. The analytic FORCE / STRESS /
drift path converts the spherical AOs to Cartesian inside the traced-ham
``jit(vmap)`` bridge via ``_aos_sphe_to_cart`` -> ``_aos_sphe_to_cart_traced`` ->
``_ragged_structural_shell_proxy``; a proxy that recovers the shell grouping by
assuming shell-major CONTIGUITY raises
``RuntimeError: ... incomplete shell ... Non-shell-major basis layout is unsupported``.

These tests build a small periodic sphe-pGTO Hamiltonian (``lithium.h5``, 2 s- + 2
p-shells per atom), then a PHYSICALLY IDENTICAL copy whose home-AO ordering is
permuted so the same-``l`` shells interleave (mirroring the QE layout), consistently
permuting the geminal MO-coefficient columns, the J3 orbital basis, and the J3
``j_matrix`` rows+columns. Because the wavefunction is unchanged, every observable
(``e_L`` value, HF+Pulay force ``wrt=R``, stress ``wrt=eps``, drift ``wrt=r``) on the
permuted ham must EQUAL the reference at the strict same-engine floor -- a perfect
oracle. This module covers the JAX analytic bridge; the native
Rust companion lives in ``tests/test_rs_kernel/test_derivatives_ragged_basis.py``.
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

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import get_tolerance_min  # noqa: E402
from jqmc._setting import ATOL_DEBUG_VS_PRODUCTION, RTOL_DEBUG_VS_PRODUCTION  # noqa: E402
from jqmc.derivatives import bridge  # noqa: E402
from jqmc.hamiltonians import compute_local_energy  # noqa: E402
from jqmc.jqmc_mcmc import _geminal_inv_batched  # noqa: E402
from jqmc.molecular_orbital import MOs_data  # noqa: E402
from tests.test_jqmc_kernel._grad_fidelity_harness import _build_ham  # noqa: E402
from tests.test_jqmc_kernel._ragged_permute import (  # noqa: E402
    interleave_perm,
    molecular_view,
    permute_aos,
)

# Imported by the native companion tests/test_rs_kernel/test_derivatives_ragged_basis.py.
_ZONES = ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb", "jastrow_eval", "jastrow_grad_lap")
_ATOL, _RTOL = get_tolerance_min(_ZONES, "strict")
# The JAX permuted/reference comparison below applies the fp64 strict values
# (1e-8, 1e-6) in BOTH precision modes: the two hams are physically
# identical, so the stricter bound is kept in mixed and declared as one reviewed
# central pair (ATOL/RTOL_DEBUG_VS_PRODUCTION) instead of a mode-resolved zone.


# --------------------------------------------------------------------------- #
# Permutation machinery (physically identical -> perfect oracle). The AO-leaf
# permutation helpers are shared with the AO-level property suite
# (``_ragged_permute``); this module adds the wavefunction-level plumbing
# (MO-coefficient columns + J3 ``j_matrix`` rows/cols).
# --------------------------------------------------------------------------- #
def _perm_for(ham) -> list[int]:
    aos = ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data
    return interleave_perm(molecular_view(aos))


def _permute_ham(ham, perm: list[int]):
    """Return a PHYSICALLY IDENTICAL ham with home-AO order permuted by ``perm``:
    geminal MO bases (+ MO-coef columns), and J3 orbital basis (+ ``j_matrix`` rows/cols)."""
    wf = ham.wavefunction_data
    gd = wf.geminal_data

    def perm_mo(mo: MOs_data) -> MOs_data:
        new_aos = permute_aos(mo.aos_data, perm)
        newC = np.asarray(mo.mo_coefficients)[:, perm]
        return MOs_data(num_mo=mo.num_mo, aos_data=new_aos, mo_coefficients=newC)

    new_gd = gd.replace(orb_data_up_spin=perm_mo(gd.orb_data_up_spin), orb_data_dn_spin=perm_mo(gd.orb_data_dn_spin))

    jd = wf.jastrow_data
    j3 = jd.jastrow_three_body_data
    if j3 is not None:
        p = np.asarray(perm)
        jm = np.asarray(j3.j_matrix)
        new_jm = np.empty_like(jm)
        new_jm[:, :-1] = jm[np.ix_(p, p)]  # square coupling block: rows AND cols
        new_jm[:, -1] = jm[p, -1]  # J1-like final column: rows only
        orb = j3.orb_data
        new_orb = perm_mo(orb) if type(orb).__name__ == "MOs_data" else permute_aos(orb, perm)
        new_j3 = j3.replace(orb_data=new_orb, j_matrix=new_jm)
        jd = jd.replace(jastrow_three_body_data=new_j3)

    return ham.replace(wavefunction_data=wf.replace(geminal_data=new_gd, jastrow_data=jd))


# --------------------------------------------------------------------------- #
# Fixtures: reference + permuted (non-shell-major) hams, cached per spec.
# --------------------------------------------------------------------------- #
_SPECS = {
    "jnone": (False, None),  # determinant sphe-pGTO basis permuted (no Jastrow)
    "j123": (True, {"j3": "pgto"}),  # + J3 sphe-pGTO orbital basis permuted
}


def _make_pair(spec_key):
    with_j, spec = _SPECS[spec_key]
    ref = _build_ham("lithium.h5", with_j, spec)
    perm = _perm_for(ref)
    assert perm != list(range(len(perm))), "interleave permutation must be non-trivial"
    return ref, _permute_ham(ref, perm)


_NW = 2


def _cfg_batch(ham, seed=0):
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    ru = R[rng.integers(0, len(R), (_NW, nu))] + rng.normal(scale=0.6, size=(_NW, nu, 3))
    rd = R[rng.integers(0, len(R), (_NW, nd))] + rng.normal(scale=0.6, size=(_NW, nd, 3))
    return jnp.asarray(ru), jnp.asarray(rd)


def _ginv_batch(ham, ru_b, rd_b):
    _, ginv, _, _ = _geminal_inv_batched(ham.wavefunction_data.geminal_data, ru_b, rd_b)
    return jnp.asarray(ginv)


def _cfg_single(ham, seed=0):
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)
    ru = R[rng.integers(0, len(R), nu)] + rng.normal(scale=0.6, size=(nu, 3))
    rd = R[rng.integers(0, len(R), nd)] + rng.normal(scale=0.6, size=(nd, 3))
    return jnp.asarray(ru), jnp.asarray(rd)


# --------------------------------------------------------------------------- #
# JAX analytic path: permuted == reference at strict floor.
#
# The R / eps / r axes are driven through the PRODUCTION bridge closures
# (``jit(vmap(...))`` with the hamiltonian on a ``None`` axis, so its basis
# exponents become TRACERS) -- the exact route where the traced-ham c2s twin
# (``_aos_sphe_to_cart_traced`` -> ``_ragged_structural_shell_proxy``) runs and
# where the non-shell-major bug lives. (A direct eager ``assemble`` call would
# route through the concrete ``_match_shell`` path and never hit the proxy.)
# --------------------------------------------------------------------------- #
def _jax_axis(ham, axis, seed=0):
    if axis == "eL":  # value path (layout-free): eager local energy is the natural probe
        ru, rd = _cfg_single(ham, seed)
        return np.asarray(compute_local_energy(ham, ru, rd, jnp.eye(3)))
    ru_b, rd_b = _cfg_batch(ham, seed)
    ginv = _ginv_batch(ham, ru_b, rd_b)
    try:
        if axis == "r":
            f = bridge.batched_grad_ln_Psi(wrt="r", mask=None, with_geminal_inv=True, analytic=True)
            out = f(ham, ru_b, rd_b, ginv)
        else:  # R (HF+Pulay force) or eps (stress) of e_L
            RT = jnp.broadcast_to(jnp.eye(3), (_NW, 3, 3))
            f = bridge.batched_grad_e_L(wrt=axis, mask=None, with_geminal_inv=True, analytic=True)
            out = f(ham, ru_b, rd_b, RT, ginv)
    finally:
        bridge.cache_clear()
    return np.concatenate([np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(out)])


@pytest.mark.parametrize("spec_key", ["jnone", "j123"])
@pytest.mark.parametrize("axis", ["eL", "R", "eps", "r"])
def test_jax_permuted_equals_reference(spec_key, axis):
    """JAX engine: e_L value, HF+Pulay force (R), stress (eps), drift (r) on the permuted
    non-shell-major ham EQUAL the shell-major reference (R/eps/r must not raise the
    ragged-proxy ``RuntimeError``; e_L agrees in any case since the value path is layout-free)."""
    ref, perm = _make_pair(spec_key)
    a = _jax_axis(ref, axis)
    b = _jax_axis(perm, axis)
    np.testing.assert_allclose(
        b,
        a,
        atol=ATOL_DEBUG_VS_PRODUCTION,
        rtol=RTOL_DEBUG_VS_PRODUCTION,
        err_msg=f"{spec_key}/{axis} permuted != reference",
    )

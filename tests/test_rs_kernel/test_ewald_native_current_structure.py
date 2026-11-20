"""Native Rust Ewald energy follows the CURRENT structure of a Hamiltonian (issue #373).

``build_ewald_data`` caches Cartesian tables (``real_image_list``, ``g_vectors``,
``g_factors``) and the ion-ion energy ``ii_energy`` for the structure it is given. The
XLA kernels never read those caches at runtime: they rebuild the tables in-trace from
the static integer index sets and ``ewald_data.structure_data`` and evaluate the
ion-ion energy at its positions. A native route that read the caches would, after a
Python-API change of the structure (an ion displaced in every ``Structure_data`` leaf
with the repository's ``tree_map`` idiom, or the cell strained with
``Hamiltonian_data._apply_strain``), differ silently from XLA by the change of E_ii
(8.1e-4 Ha) and, for a strain, of the e-e tables (2.6e-3 Ha).

For both changes, on the committed periodic lithium fixture ``pbc_gamma_sphepgto_j12``:

1. the stale cached ``ii_energy`` differs from the ion-ion energy of the changed
   structure (non-vacuity: the cache really is stale);
2. the XLA Ewald energy ``ii + ei + ee`` of the changed Hamiltonian equals
   ``ref.numpy_ewald_energy``, a textbook Ewald sum of the ion and electron point
   charges with its own eta, cutoffs and image lists (independent anchor of the
   reference backend on the changed structure);
3. the Rust-minus-XLA local-energy gap of the changed Hamiltonian equals the gap of the
   same Hamiltonian after the checkpoint-load Ewald rebuild
   (``_apply_ewald_rebuild_post_load``, which rebuilds every cache from the current
   structure; Rust/XLA parity of freshly built Hamiltonians is pinned by
   ``test_ecp_local.py`` / ``test_ewald.py``), for the native free local energy and
   for the production native observable context (``build_obs_ctx(...).e_L``, the
   MCMC e_L route). Both sides of the comparison see identical wavefunction and
   electron inputs, so the fp32 kinetic rounding of the mixed lane cancels and the
   residual is the fp64 ``ewald`` zone.

Full and mixed precision are each judged with their own tolerances.
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jqmc_rs  # noqa: E402
from jqmc._precision import get_tolerance_xla_rust_consistency, precision_policy_snapshot  # noqa: E402
from jqmc._setting import (  # noqa: E402
    ATOL_EWALD_VS_EXTERNAL_REFERENCE,
    ATOL_NONVACUITY_DISTINCTION,
    RTOL_EWALD_VS_EXTERNAL_REFERENCE,
    RTOL_NONVACUITY_DISTINCTION,
)
from jqmc.coulomb_potential import (  # noqa: E402
    compute_ewald_potential_ee,
    compute_ewald_potential_ei,
    compute_ewald_potential_ii,
)
from jqmc.hamiltonians import _apply_ewald_rebuild_post_load, compute_local_energy  # noqa: E402
from jqmc.jqmc_mcmc import _NV_BASE_MESH, Nv_default, _geminal_inv_batched  # noqa: E402
from jqmc.structure import Structure_data  # noqa: E402
from tests.test_jqmc_kernel import _noncubic_oracles as ref  # noqa: E402
from tests.test_rs_kernel._rs_fixtures import rs_ham  # noqa: E402

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_PATTERN = "pbc_gamma_sphepgto_j12"
#: Displacement of ion 0 (Bohr), the issue's reproducer.
_DISPLACEMENT = np.array([0.25, 0.10, -0.20])
#: Deformation gradient F = I + eps (about 2 % strain, shear included), the issue's reproducer.
_STRAIN = np.array([[0.02, 0.01, 0.0], [0.01, -0.015, 0.005], [0.0, 0.005, 0.01]])
_MESH = _NV_BASE_MESH[Nv_default]


def _host(tree):
    """Copy every JAX array leaf to numpy (the native parsers take host arrays)."""
    return jax.tree_util.tree_map(lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, tree)


def _displaced(ham):
    """Move ion 0 in every ``Structure_data`` leaf (the ``_displaced`` idiom of the force tests)."""
    d = np.zeros((int(ham.structure_data.natom), 3))
    d[0] = _DISPLACEMENT

    def move(x):
        if isinstance(x, Structure_data) and np.asarray(x.positions).shape == d.shape:
            return x.replace(positions=np.asarray(x.positions) + d)
        return x

    return jax.tree_util.tree_map(move, ham, is_leaf=lambda x: isinstance(x, Structure_data))


def _changed(ham, case):
    """Return the changed Hamiltonian and the map applied to Cartesian points."""
    if case == "displaced_ion":
        return _displaced(ham), lambda r: r
    F = np.eye(3) + _STRAIN
    return _host(ham._apply_strain(jnp.asarray(F))), lambda r: np.asarray(r) @ F.T


def _electrons(ham, seed=0, scale=0.6):
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    R = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(seed)
    r_up = R[np.arange(nu) % len(R)] + rng.normal(scale=scale, size=(nu, 3))
    r_dn = R[np.arange(nd) % len(R)] + rng.normal(scale=scale, size=(nd, 3))
    return r_up, r_dn


def _rust_free_e_L(ham, r_up, r_dn):
    w = np.asarray(_MESH.weights, dtype=np.float64)
    gp = np.asarray(_MESH.grid_points, dtype=np.float64)
    return float(jqmc_rs.compute_local_energy(ham, r_up, r_dn, w, gp))


def _rust_ctx_e_L(ham, r_up, r_dn):
    gd = ham.wavefunction_data.geminal_data
    ctx = jqmc_rs.build_obs_ctx(
        gd,
        ham.wavefunction_data.jastrow_data,
        ham.coulomb_potential_data,
        False,
        True,
        False,
        precision_policy=precision_policy_snapshot(),
    )
    _, ginv, _, _ = _geminal_inv_batched(gd, jnp.asarray(r_up)[None], jnp.asarray(r_dn)[None])
    w = np.asarray(_MESH.weights, dtype=np.float64)
    gp = np.asarray(_MESH.grid_points, dtype=np.float64)
    return float(ctx.e_L(r_up, r_dn, w, gp, np.asarray(ginv[0], dtype=np.float64)))


@pytest.mark.parametrize("case", ["displaced_ion", "strained_cell"])
def test_native_local_energy_follows_current_structure(case):
    """The Rust-XLA e_L gap of a displaced/strained Hamiltonian equals the gap with rebuilt Ewald caches."""
    base = rs_ham(_PATTERN)
    ham, to_changed = _changed(base, case)
    r_up, r_dn = (to_changed(r) for r in _electrons(base))
    cp = ham.coulomb_potential_data
    ed = cp.ewald_data

    # (1) Non-vacuity: the cached ion-ion energy is that of the unchanged structure.
    e_ii = float(compute_ewald_potential_ii(cp, ed))
    stale = float(ed.ii_energy)
    assert abs(stale - e_ii) > ATOL_NONVACUITY_DISTINCTION + RTOL_NONVACUITY_DISTINCTION * abs(e_ii), (stale, e_ii)

    # (2) Independent anchor: XLA ii + ei + ee of the changed Hamiltonian equals a textbook Ewald sum.
    ei_up, ei_dn = compute_ewald_potential_ei(cp, jnp.asarray(r_up), jnp.asarray(r_dn), ed)
    e_ewald = (
        e_ii
        + float(np.sum(np.asarray(ei_up)))
        + float(np.sum(np.asarray(ei_dn)))
        + float(compute_ewald_potential_ee(jnp.asarray(r_up), jnp.asarray(r_dn), ed))
    )
    x = np.concatenate([np.asarray(ed.structure_data.positions, dtype=np.float64), r_up, r_dn])
    q = np.concatenate([np.asarray(cp._effective_charges, dtype=np.float64), -np.ones(len(r_up) + len(r_dn))])
    e_ewald_ref = ref.numpy_ewald_energy(np.asarray(ed.structure_data.cell, dtype=np.float64), x, q)
    np.testing.assert_allclose(
        e_ewald, e_ewald_ref, atol=ATOL_EWALD_VS_EXTERNAL_REFERENCE, rtol=RTOL_EWALD_VS_EXTERNAL_REFERENCE
    )

    # (3) Backend parity on the changed Hamiltonian, free native e_L and the production ctx e_L:
    # the Rust - XLA gap must not depend on whether the Ewald caches were rebuilt for the
    # current structure. Identical wavefunction/electron inputs on both sides cancel the
    # mixed-lane kinetic rounding; the residual lives in the fp64 ewald zone.
    rebuilt = _host(_apply_ewald_rebuild_post_load(ham))
    e_L_xla = float(compute_local_energy(ham, r_up, r_dn, jnp.eye(3)))
    e_L_xla_rebuilt = float(compute_local_energy(rebuilt, r_up, r_dn, jnp.eye(3)))
    gap_free = _rust_free_e_L(ham, r_up, r_dn) - e_L_xla
    gap_free_rebuilt = _rust_free_e_L(rebuilt, r_up, r_dn) - e_L_xla_rebuilt
    gap_ctx = _rust_ctx_e_L(ham, r_up, r_dn) - e_L_xla
    gap_ctx_rebuilt = _rust_ctx_e_L(rebuilt, r_up, r_dn) - e_L_xla_rebuilt
    atol, rtol = get_tolerance_xla_rust_consistency(("ewald",), level="strict")
    np.testing.assert_allclose(gap_free, gap_free_rebuilt, atol=atol, rtol=rtol)
    np.testing.assert_allclose(gap_ctx, gap_ctx_rebuilt, atol=atol, rtol=rtol)

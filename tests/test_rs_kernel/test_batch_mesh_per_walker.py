"""Per-walker rotated ECP meshes on the native per-step MCMC batch entries.

The production per-step route (``MCMC.run`` whenever forces, stress or
parameter derivatives are requested, and every complex-twist observable step)
draws ONE Haar rotation per walker per step (``_jit_vmap_generate_RTs``), builds
``gp_all = base_mesh @ RTs`` with shape ``(nw, n_grid, 3)`` and hands it to the
native ``NativeObsCtx`` batch entries.  Walker ``w`` of every result must be
evaluated on ``gp_all[w]``.

Older fixed-configuration tests feed these entries either one walker or an
identity rotation for every walker, so ``gp_all[w] == gp_all[0]`` and a native
entry that reads walker 0's mesh for every walker passes them (#347).  Here
three walkers receive three distinct production rotations in one call, and each
walker's native value is compared with XLA evaluated with that walker's own
rotation:

* ``e_L_batch`` / ``e_L_complex_batch`` vs ``compute_local_energy_fast``;
* ``grad_e_L_ecp_batch`` / ``grad_e_L_pbc_batch`` /
  ``grad_e_L_pbc_complex_batch`` vs ``assemble.compute_grad_e_L(wrt="R"|"r")``;
* ``grad_e_L_eps_pbc_batch`` / ``grad_e_L_eps_pbc_complex_batch`` vs
  ``assemble.compute_grad_e_L(wrt="eps")``;
* real and complex ``dE_L/dc`` through the production assembler
  ``_rs_native_c_param_grads(..., rts_b=RTs)`` (the ``pd_el_ecp_*_batch``
  entries or the complex per-walker adapter) vs
  ``assemble.compute_grad_e_L(wrt="c")``.

Non-vacuity is asserted per observable: the XLA value obtained when every
walker uses walker 0's rotation (exactly what a walker-index defect computes)
must differ from the per-walker reference beyond the comparison tolerance.
Full and mixed precision are judged separately with the dedicated XLA--Rust
consistency tolerances already used by the per-walker observable suite.

dE_L/dc (issue #717): full precision compares native and XLA directly. In mixed
precision each route is compared, at the same budget, with an independently
rebuilt full-precision reference: the XLA route evaluated on a Hamiltonian
reloaded under the full policy from the serialized full-precision input
(``full_source_hamiltonian_factory`` + ``mixed_vs_full``), at the same walkers
and rotations. The mixed values come from a Hamiltonian reloaded under the mixed
policy from the same source. Native and XLA evaluate the fp32 Jastrow drift with
separate implementations, so their mixed values can differ by twice one route's
fp32 error while each route stays within its budget of the fp64 value; comparing
each route with fp64 also identifies which route is wrong.
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

import jqmc_rs
from jqmc._precision import get_tolerance_xla_rust_consistency, is_mixed_precision_enabled, precision_policy_snapshot
from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_ZERO
from jqmc.derivatives import assemble
from jqmc.derivatives.bridge import _MASK_DEFAULTS, _reduce_c_param_grads
from jqmc.hamiltonians import compute_local_energy_fast
from jqmc.jqmc_mcmc import (
    _NV_BASE_MESH,
    _PARAM_BLOCK_TO_MASK_KW,
    Nv_default,
    _RsComplexPdCtx,
    _jit_vmap_generate_RTs,
    _rs_native_c_param_grads,
)
from tests._precision_fidelity import mixed_vs_full
from tests.test_rs_kernel._native_precision_fidelity import full_source_hamiltonian_factory
from tests.test_rs_kernel._rs_fixtures import rs_energy_ham
from tests.test_rs_kernel.test_mcmc_observables import _geminal_inv, _ham, _is_complex, _walker_batch

pytestmark = [pytest.mark.with_chdir]

_NW = 3
_ROTATION_SEED = 7
_MESH = _NV_BASE_MESH[Nv_default]
_W = np.asarray(_MESH.weights, float)
_GP = np.asarray(_MESH.grid_points, float)

# Same zone sets as ``test_mcmc_observables`` (e_L and its position/parameter derivatives).
_ZONES_EL = (
    "ao_eval",
    "mo_eval",
    "det_eval",
    "jastrow_eval",
    "wf_kinetic",
    "ao_grad_lap",
    "det_grad_lap",
    "jastrow_grad_lap",
    "coulomb",
    "ewald",
    "jastrow_ratio",
)
_ZONES_GRAD = _ZONES_EL + ("mo_grad",)

# ECP fixtures with a live non-local part: molecular water (real), real Gamma
# periodic, complex twist k=(1/4,1/4,1/4).
_CASES = [
    pytest.param("mol_ecp_sphe_j12", marks=pytest.mark.light, id="mol_ecp_sphe_j12"),
    pytest.param("pbc_gamma_sphepgto_j12", marks=pytest.mark.heavy, id="pbc_gamma_sphepgto_j12"),
    pytest.param("twist_sphepgto_j12", marks=pytest.mark.heavy, id="twist_sphepgto_j12"),
]


def _distinct_rotations(nw):
    """Production per-walker rotations: ``_jit_vmap_generate_RTs`` on split keys."""
    keys = jax.random.split(jax.random.PRNGKey(_ROTATION_SEED), nw)
    rts = np.asarray(_jit_vmap_generate_RTs(keys), float)
    for w in range(nw):
        np.testing.assert_allclose(rts[w] @ rts[w].T, np.eye(3), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        assert np.linalg.det(rts[w]) > 0.0
    for a in range(nw):
        for b in range(a + 1, nw):
            assert not np.allclose(rts[a], rts[b]), f"walkers {a} and {b} share a rotation"
    return rts


def _assert_detects_shared_mesh(per_walker, shared, atol, rtol, label):
    """The walker-0 mesh value must differ from the per-walker value at the tolerance."""
    per_walker = np.asarray(per_walker)
    shared = np.asarray(shared)
    assert not np.allclose(np.real(shared), np.real(per_walker), atol=atol, rtol=rtol) or not np.allclose(
        np.imag(shared), np.imag(per_walker), atol=atol, rtol=rtol
    ), f"{label}: a walker-0 mesh is indistinguishable from per-walker meshes at atol={atol:g}, rtol={rtol:g}"


def _assert_parts_close(got, ref, atol, rtol, label):
    got = np.asarray(got)
    ref = np.asarray(ref)
    assert got.shape == ref.shape, f"{label}: shape {got.shape} != {ref.shape}"
    np.testing.assert_allclose(np.real(got), np.real(ref), atol=atol, rtol=rtol, err_msg=f"{label} (Re)")
    np.testing.assert_allclose(np.imag(got), np.imag(ref), atol=atol, rtol=rtol, err_msg=f"{label} (Im)")


def _xla_c_param_grads(ham, ru_b, rd_b, rts, mask):
    """XLA per-walker dE_L/dc blocks, walker ``w`` on rotation ``rts[w]``."""
    wf = ham.wavefunction_data
    nw = ru_b.shape[0]
    ginv_b = _geminal_inv_b(ham, ru_b, rd_b)
    per_walker = [
        assemble.compute_grad_e_L(
            ham,
            jnp.asarray(ru_b[w]),
            jnp.asarray(rd_b[w]),
            rts[w],
            wrt="c",
            mask=mask,
            geminal_inv=jnp.asarray(ginv_b[w]),
        )
        for w in range(nw)
    ]
    stacked = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *per_walker)
    flat = _reduce_c_param_grads(wf, stacked, nw, mask)
    return {n: np.asarray(v) for n, v in flat.items() if mask.get(_PARAM_BLOCK_TO_MASK_KW.get(n, ""), True)}


def _geminal_inv_b(ham, ru_b, rd_b):
    ginv_b = np.stack([_geminal_inv(ham, ru_b[w], rd_b[w]) for w in range(ru_b.shape[0])])
    return ginv_b.astype(np.complex128) if _is_complex(ham) else ginv_b


def _c_param_grads(ham, ru_b, rd_b, rts, mask):
    """``(native, xla)`` per-walker dE_L/dc blocks of ``ham`` under the active policy.

    Native: the production assembler ``_rs_native_c_param_grads(..., rts_b=rts)`` on a
    context built from ``ham`` with the active policy snapshot. XLA projects real
    parameters but keeps both components of complex orbital coefficients, so a native
    block is projected to its real part exactly where the XLA block is real.
    """
    wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
    cplx = _is_complex(ham)
    pbc = bool(ham.structure_data.pbc_flag)
    nw = ru_b.shape[0]
    ctx = jqmc_rs.build_obs_ctx(
        wf.geminal_data,
        wf.jastrow_data,
        cp,
        cplx,
        pbc,
        pbc,  # want_stress
        want_force=True,
        want_gfmc=False,
        want_param_deriv=True,
        precision_policy=precision_policy_snapshot(),
    )
    param_ctx = _RsComplexPdCtx(wf.geminal_data, wf.jastrow_data, cp, ctx) if cplx else ctx
    native = _rs_native_c_param_grads(
        param_ctx,
        ham,
        [ru_b[w] for w in range(nw)],
        [rd_b[w] for w in range(nw)],
        mask,
        nw,
        quantity="e_L",
        ecp_flag=True,
        ecp_w=_W,
        ecp_gp=_GP,
        rts_b=rts,
        complex_twist=cplx,
        geminal_inv=_geminal_inv_b(ham, ru_b, rd_b) if cplx else None,
    )
    xla = _xla_c_param_grads(ham, ru_b, rd_b, rts, mask)
    native = {n: np.asarray(v) for n, v in native.items()}
    for n in native:
        if cplx and n in xla and not np.iscomplexobj(xla[n]):
            native[n] = np.real(native[n])
    return native, xla


@pytest.mark.parametrize("pattern", _CASES)
def test_native_batch_entries_use_each_walkers_rotated_mesh(pattern):
    """Native per-step batch entries evaluate walker ``w`` on ``base_mesh @ RT[w]``."""
    ham = _ham(pattern)
    wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
    assert bool(cp.ecp_flag), f"{pattern}: fixture has no ECP mesh"
    cplx = _is_complex(ham)
    pbc = bool(ham.structure_data.pbc_flag)
    ru_b, rd_b = _walker_batch(ham, nw=_NW, seed=0)
    ru_b, rd_b = np.ascontiguousarray(ru_b), np.ascontiguousarray(rd_b)
    rts = _distinct_rotations(_NW)
    shared_rts = np.stack([rts[0]] * _NW)
    gp_all = _GP @ rts  # (nw, n_grid, 3), as jqmc_mcmc builds ``_gp_all``
    assert gp_all.shape == (_NW, _GP.shape[0], 3)
    ginv_b = np.stack([_geminal_inv(ham, ru_b[w], rd_b[w]) for w in range(_NW)])
    if cplx:
        ginv_b = ginv_b.astype(np.complex128)
    ctx = jqmc_rs.build_obs_ctx(
        wf.geminal_data,
        wf.jastrow_data,
        cp,
        cplx,
        pbc,
        pbc,  # want_stress
        want_force=True,
        want_gfmc=False,
        want_param_deriv=True,
        precision_policy=precision_policy_snapshot(),
    )

    # ---- e_L ----------------------------------------------------------------
    # Per-walker e_L: strict in full, reviewed medium envelope in mixed
    # (as ``test_mcmc_walker_batch_e_L``).
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_EL, level="medium" if is_mixed_precision_enabled() else "strict")

    def _xla_e_L(rot):
        return np.asarray([complex(compute_local_energy_fast(ham, ru_b[w], rd_b[w], rot[w], ginv_b[w])) for w in range(_NW)])

    if cplx:
        native = np.asarray(ctx.e_L_complex_batch(ru_b, rd_b, _W, gp_all, ginv_b))
    else:
        native = np.asarray(ctx.e_L_batch(ru_b, rd_b, _W, gp_all, ginv_b))
    reference = _xla_e_L(rts)
    _assert_detects_shared_mesh(reference, _xla_e_L(shared_rts), atol, rtol, f"{pattern} e_L")
    _assert_parts_close(native.astype(np.complex128), reference, atol, rtol, f"{pattern} per-walker e_L")

    # ---- HF force (dE_L/dR, dE_L/dr_up, dE_L/dr_dn) -------------------------
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_GRAD, level="medium")

    def _xla_grad(wrt, rot):
        out = []
        for w in range(_NW):
            g = assemble.compute_grad_e_L(
                ham, jnp.asarray(ru_b[w]), jnp.asarray(rd_b[w]), rot[w], wrt=wrt, geminal_inv=jnp.asarray(ginv_b[w])
            )
            out.append(tuple(np.asarray(np.real(x), float) for x in g) if wrt == "r" else np.asarray(np.real(g), float))
        if wrt == "r":
            return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])
        return np.stack(out)

    if cplx:
        force = ctx.grad_e_L_pbc_complex_batch(ru_b, rd_b, _W, gp_all, ginv_b)
    elif pbc:
        force = ctx.grad_e_L_pbc_batch(ru_b, rd_b, _W, gp_all, ginv_b)
    else:
        force = ctx.grad_e_L_ecp_batch(ru_b, rd_b, _W, gp_all, ginv_b)
    g_R, g_up, g_dn = (np.asarray(x, float) for x in force)
    ref_R = _xla_grad("R", rts)
    ref_up, ref_dn = _xla_grad("r", rts)
    _assert_detects_shared_mesh(ref_R, _xla_grad("R", shared_rts), atol, rtol, f"{pattern} HF force dR")
    _assert_parts_close(g_R, ref_R, atol, rtol, f"{pattern} HF force dR")
    _assert_parts_close(g_up, ref_up, atol, rtol, f"{pattern} HF force dr_up")
    _assert_parts_close(g_dn, ref_dn, atol, rtol, f"{pattern} HF force dr_dn")

    # ---- HF stress (periodic) -------------------------------------------------
    if pbc:
        if cplx:
            stress = np.asarray(ctx.grad_e_L_eps_pbc_complex_batch(ru_b, rd_b, _W, gp_all, ginv_b), float)
        else:
            stress = np.asarray(ctx.grad_e_L_eps_pbc_batch(ru_b, rd_b, _W, gp_all, ginv_b), float)
        ref_eps = _xla_grad("eps", rts)
        _assert_detects_shared_mesh(ref_eps, _xla_grad("eps", shared_rts), atol, rtol, f"{pattern} HF stress")
        _assert_parts_close(stress, ref_eps, atol, rtol, f"{pattern} HF stress")

    # ---- dE_L/dc through the production assembler --------------------------
    # Full: direct native-vs-XLA parity. Mixed: each route against an independently
    # rebuilt full-precision (fp64) reference of the same serialized input and the same
    # walkers / rotations, at the same budget (#717). Two separate fp32 implementations
    # of the Jastrow drift differ by up to twice one route's fp32 error, so comparing them
    # with each other cannot tell which route is wrong; each route against fp64 can.
    if cplx:
        # Fixed positions and quadrature allow the strict parity policy here.
        atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_GRAD, level="strict")
    mask = dict(_MASK_DEFAULTS)
    native_c, ref_c = _c_param_grads(ham, ru_b, rd_b, rts, mask)
    shared_c = _xla_c_param_grads(ham, ru_b, rd_b, shared_rts, mask)
    assert native_c, f"{pattern}: native c-axis assembler emitted no block"
    assert set(native_c) == set(ref_c), f"{pattern}: native blocks {sorted(native_c)} != XLA blocks {sorted(ref_c)}"
    blocks = sorted(ref_c)
    _assert_detects_shared_mesh(
        np.concatenate([np.ravel(ref_c[b]) for b in blocks]),
        np.concatenate([np.ravel(shared_c[b]) for b in blocks]),
        atol,
        rtol,
        f"{pattern} dE_L/dc",
    )
    for block in blocks:
        got = np.asarray(native_c[block])
        assert got.shape[0] == _NW, f"{pattern} {block}: walker axis is not axis 0 ({got.shape})"
        assert got.shape == np.shape(ref_c[block]), f"{pattern} {block}: shape {got.shape} != {np.shape(ref_c[block])}"
        if cplx:
            assert np.isfinite(got).all(), f"{pattern} {block}: non-finite native derivative"
            _assert_detects_shared_mesh(ref_c[block], shared_c[block], atol, rtol, f"{pattern} dE_L/dc block={block}")

    if not is_mixed_precision_enabled():
        for block in blocks:
            _assert_parts_close(native_c[block], ref_c[block], atol, rtol, f"{pattern} dE_L/dc block={block}")
        return

    mixed, full = mixed_vs_full(
        full_source_hamiltonian_factory(lambda: rs_energy_ham(pattern)),
        lambda fresh: _c_param_grads(fresh, ru_b, rd_b, rts, mask),
        configure_native=jqmc_rs.configure_native_precision,
        native_policy_snapshot=jqmc_rs.native_precision_support,
    )
    mixed_native, mixed_xla = mixed
    full_native, full_xla = full
    assert set(mixed_native) == set(mixed_xla) == set(full_native) == set(full_xla) == set(blocks)
    for block in blocks:
        assert np.isfinite(full_xla[block]).all(), f"{pattern} {block}: non-finite fp64 reference"
        for route, value in (("native", mixed_native[block]), ("XLA", mixed_xla[block])):
            _assert_parts_close(value, full_xla[block], atol, rtol, f"{pattern} mixed {route} dE_L/dc vs fp64 block={block}")

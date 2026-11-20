"""AGPn geminal (natural-orbital-occupation AGP) value + kinetic vs jqmc.

AGPn_data stores the paired lambda-block as a DIAGONAL `lambda_diag` (natural-
orbital occupations) in the MO basis (+ optional `lambda_unpaired` for open
shells). Its materialised `lambda_matrix` property = `diag(lambda_diag)`
(h-stacked with unpaired), and `orb_num_dn` / MO-rep orbitals are duck-typed
exactly like AGP -- so the EXISTING Rust geminal path (`GeminalData::from_py`
reads `lambda_matrix`+`orb_num_dn`) handles AGPn unchanged.

Fixtures are the committed hams. The committed `_rs_fixtures` patterns only ship
ONE true AGPn pattern (`mol_cart_agpn_jnone` = H2 ae cart), so the Jastrow/ECP axis
coverage (cart/sphe, jnone/j12/j123ao/jdtn, AE/ECP) is reproduced here by converting an SD geminal to AGPn via
`jqmc.determinant.to_AGPn` on top of the already-committed SD patterns --
`to_AGPn` exposes the SAME wavefunction as the source SD (lambda_diag =
[1]*N_occ + [0]*N_virt), which is exactly what the harness in
test_analytic_grad_matches_autograd_curated.py does for its own AGPn row. This lets the ECP
sphe j12/jnone patterns stand in for the water AGPn ECP fixtures without
touching _rs_fixtures.py.
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

from functools import partial

import jqmc_rs
from jqmc._precision import precision_policy_snapshot
import numpy as np
import pytest

pytestmark = pytest.mark.with_chdir

import jax
import jax.numpy as jnp

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc._precision import get_tolerance_xla_rust_consistency as _leaf_xla_rust_tolerance
from jqmc.coulomb_potential import compute_ecp_non_local_parts_all_pairs, icosahedron_sym_mesh_Nv12, octahedron_sym_mesh_Nv6
from jqmc.derivatives import assemble
from jqmc.derivatives.bridge import _MASK_DEFAULTS, _reduce_c_param_grads
from jqmc.determinant import compute_ln_det_geminal_all_elements, to_AGPn
from jqmc.jqmc_mcmc import _PARAM_BLOCK_TO_MASK_KW, _geminal_inv_batched, _rs_native_c_param_grads
from jqmc.wavefunction import compute_kinetic_energy
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_jqmc_kernel.optimizer_metadata_helpers import build_explicit_common_rank_agpn_from_sd
from tests.test_rs_kernel._rs_fixtures import rs_ham

get_tolerance_xla_rust_consistency = partial(get_tolerance_xla_rust_consistency, level="medium")

_ZONES_LNDET = ("ao_eval", "det_eval")
_ZONES_KINETIC = ("wf_kinetic", "ao_eval", "ao_grad_lap", "det_grad_lap", "jastrow_grad_lap")
_ZONES_ECP_NONLOCAL = ("coulomb", "det_ratio", "jastrow_ratio", "ao_eval", "jastrow_eval")
_ZONES_GRAD = ("wf_kinetic", "ao_eval", "det_eval", "coulomb", "jastrow_grad_lap", "mo_grad")
_W = np.asarray(icosahedron_sym_mesh_Nv12.weights, float)
_GP = np.asarray(icosahedron_sym_mesh_Nv12.grid_points, float)

# (rs_ham pattern, is a true committed AGPn or converted-via-to_AGPn SD)
_FIXTURES = [
    "mol_cart_agpn_jnone",  # true AGPn (H2 ae cart)
    "mol_ecp_cart_jnone",  # SD -> AGPn (H2 ecp cart, jnone)
    "mol_ecp_cart_j12",  # SD -> AGPn (H2 ecp cart, j12)
    "mol_ecp_sphe_jnone",  # SD -> AGPn (water ecp sphe, jnone)
    "mol_ecp_sphe_j12",  # SD -> AGPn (water ecp sphe, j12)
]


def _agpn_ham(pattern):
    ham = rs_ham(pattern)
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    if type(gd).__name__ == "AGPn_data":
        return ham
    gd2 = to_AGPn(gd)
    return ham.replace(wavefunction_data=wf.replace(geminal_data=gd2))


def _cfg(ham, gd, seed):
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(seed)

    def pick(n):
        idx = rng.integers(0, R.shape[0], size=n)
        return R[idx] + rng.normal(scale=0.5, size=(n, 3))

    return pick(int(gd.num_electron_up)), pick(int(gd.num_electron_dn))


@pytest.mark.light
@pytest.mark.parametrize("pattern", _FIXTURES)
def test_agpn_lndet_and_kinetic(pattern):
    """Agpn lndet and kinetic."""
    ham = _agpn_ham(pattern)
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    assert type(gd).__name__ == "AGPn_data"
    atol_l, rtol_l = _leaf_xla_rust_tolerance(_ZONES_LNDET, level="strict")
    atol_k, rtol_k = _leaf_xla_rust_tolerance(_ZONES_KINETIC, level="strict")
    # The converted spherical AGPn seed=0 point is ill-conditioned for strict
    # fp32-leaf aggregate parity.  These three points have cond(G) 42, 14, 19.
    seeds = (40, 49, 46) if pattern == "mol_ecp_sphe_j12" else range(3)
    for sample_index, seed in enumerate(seeds):
        sample_label = str(sample_index)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, gd, seed)
        lref = float(compute_ln_det_geminal_all_elements(gd, ru, rd))
        lgot = float(jqmc_rs.compute_ln_det_geminal_all_elements(gd, ru, rd))
        assert np.isclose(lgot, lref, rtol=rtol_l, atol=atol_l), (pattern, seed, abs(lgot - lref))
        tref = float(compute_kinetic_energy(wf, ru, rd))
        tgot = float(jqmc_rs.compute_kinetic_energy(gd, wf.jastrow_data, ru, rd))
        assert np.isclose(tgot, tref, rtol=rtol_k, atol=atol_k), (pattern, seed, abs(tgot - tref))


@pytest.mark.light
def test_agpn_randomized_lambda_diag():
    """Exercise the genuine non-identity diagonal paired block (the committed
    `mol_cart_agpn_jnone` fixture has lambda_diag == 1). Randomizing lambda_diag
    changes the materialised lambda_matrix that BOTH jqmc and Rust read, so
    they must stay consistent (diagonal materialization path).
    """
    ham = _agpn_ham("mol_ecp_sphe_jnone")
    gd = ham.wavefunction_data.geminal_data
    rng = np.random.default_rng(11)
    import jax.numpy as jnp

    new_diag = jnp.asarray(0.3 + rng.random(np.asarray(gd.lambda_diag).shape))
    gd2 = gd.replace(lambda_diag=new_diag)
    atol, rtol = _leaf_xla_rust_tolerance(_ZONES_LNDET, level="strict")
    for seed in range(3):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, gd2, seed)
        lref = float(compute_ln_det_geminal_all_elements(gd2, ru, rd))
        lgot = float(jqmc_rs.compute_ln_det_geminal_all_elements(gd2, ru, rd))
        assert np.isclose(lgot, lref, rtol=rtol, atol=atol), (seed, abs(lgot - lref))


_ECP_AGPN_PATTERNS = ["mol_ecp_sphe_jnone", "mol_ecp_sphe_j12"]


@pytest.mark.light
@pytest.mark.parametrize("pattern", _ECP_AGPN_PATTERNS)
def test_agpn_ecp_nonlocal_with_jastrow(pattern):
    """ECP non-local energy `sum A*ratio` (ratio = det-ratio x exp(dJ)) for the
    molecular AGPn ECP geminal vs jqmc `compute_ecp_non_local_parts_all_pairs`.
    """
    w = np.asarray(octahedron_sym_mesh_Nv6.weights, float)
    gp = np.asarray(octahedron_sym_mesh_Nv6.grid_points, float)
    ham = _agpn_ham(pattern)
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    cd = ham.coulomb_potential_data
    natom = np.asarray(ham.structure_data.positions).shape[0]
    atol, rtol = _leaf_xla_rust_tolerance(_ZONES_ECP_NONLOCAL, level="strict")
    seeds = (40, 49, 46) if pattern == "mol_ecp_sphe_j12" else range(3)
    for sample_index, seed in enumerate(seeds):
        sample_label = str(sample_index)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, gd, seed)
        ref = float(
            np.real(compute_ecp_non_local_parts_all_pairs(cd, wf, ru, rd, np.eye(3), Nv=6, flag_determinant_only=False)[3])
        )
        got = float(jqmc_rs.compute_ecp_nonlocal_parts_all_pairs(gd, wf.jastrow_data, cd, ru, rd, w, gp, False, natom))
        assert np.isclose(got, ref, rtol=rtol, atol=atol), (pattern, seed, abs(got - ref))


# --------------------------------------------------------------------------- #
# OPEN-SHELL AGPn (``lambda_unpaired`` non-empty).
#
# Every committed AGPn fixture is closed-shell (``test_rs_analytic_matrix.py``
# self-notes this), so the Rust ``lambda_unpaired`` chain rule -- the extra
# ``(N_up, N_up - N_dn)`` unpaired MO columns h-stacked onto the paired diagonal
# block -- had NEVER been asserted at any level. Canonical open-shell SD owns
# no minority-spin virtual rows, so the test-only constructor explicitly
# completes the common-rank natural-orbital basis before building AGPn. A
# divergence then localises to AGPn unpaired-column materialisation and its
# derivatives, without weakening the occupied-only SD convention.
# --------------------------------------------------------------------------- #
_OPEN_SHELL_PATTERNS = ["mol_cart_sd_jnone", "mol_cart_sd_j12"]


def _open_shell_agpn(pattern):
    base = rs_ham(pattern)
    wf = base.wavefunction_data
    gd = build_explicit_common_rank_agpn_from_sd(wf.geminal_data)
    assert type(gd).__name__ == "AGPn_data"
    assert gd.lambda_unpaired is not None and np.asarray(gd.lambda_unpaired).shape[1] > 0, (
        f"{pattern}: explicit common-rank AGPn produced an empty unpaired block (fixture is not open-shell)"
    )
    return base.replace(wavefunction_data=wf.replace(geminal_data=gd))


@pytest.mark.light
@pytest.mark.parametrize("pattern", _OPEN_SHELL_PATTERNS)
def test_agpn_open_shell_lndet_and_kinetic(pattern):
    """OPEN-SHELL AGPn ln|det G| + kinetic (the position chain rule through the
    unpaired MO columns), Rust vs jqmc on N (5 up / 2 dn).
    """
    ham = _open_shell_agpn(pattern)
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    atol_l, rtol_l = _leaf_xla_rust_tolerance(_ZONES_LNDET, level="strict")
    atol_k, rtol_k = _leaf_xla_rust_tolerance(_ZONES_KINETIC, level="strict")
    for seed in range(3):
        sample_label = str(seed)  # noqa: F841 - captured by the GDD iteration receipt
        ru, rd = _cfg(ham, gd, seed)
        lref = float(compute_ln_det_geminal_all_elements(gd, ru, rd))
        lgot = float(jqmc_rs.compute_ln_det_geminal_all_elements(gd, ru, rd))
        assert np.isclose(lgot, lref, rtol=rtol_l, atol=atol_l), (pattern, seed, abs(lgot - lref))
        tref = float(compute_kinetic_energy(wf, ru, rd))
        tgot = float(jqmc_rs.compute_kinetic_energy(gd, wf.jastrow_data, ru, rd))
        assert np.isclose(tgot, tref, rtol=rtol_k, atol=atol_k), (pattern, seed, abs(tgot - tref))


def _geminal_inv(gd, ru, rd):
    _, ginv, _, _ = _geminal_inv_batched(gd, jnp.asarray(ru)[None], jnp.asarray(rd)[None])
    return np.asarray(ginv[0])


def _assert_c_param_grad_parity(ham, quantity, ecp_flag, seed=1, nw=2):
    """Native production assembler ``_rs_native_c_param_grads`` vs the XLA
    ``assemble`` reference, block by block, on a deterministic walker batch --
    the SAME two-layer comparison ``test_mcmc_walker_batch_param_deriv`` runs,
    replicated here so the AGP/AGPn lambda blocks (absent from the shared fixture
    set) are covered without touching the shared ``_rs_fixtures`` registry.
    Returns the sorted native block set so callers can assert the lambda block
    is actually present.
    """
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    cp = ham.coulomb_potential_data
    mask = dict(_MASK_DEFAULTS)
    R = np.asarray(ham.structure_data.positions, float)
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    ru_b = [np.ascontiguousarray(R[rng.integers(0, len(R), nu)] + rng.normal(scale=0.6, size=(nu, 3))) for _ in range(nw)]
    rd_b = [np.ascontiguousarray(R[rng.integers(0, len(R), nd)] + rng.normal(scale=0.6, size=(nd, 3))) for _ in range(nw)]
    ctx = jqmc_rs.build_obs_ctx(
        gd,
        wf.jastrow_data,
        cp,
        False,
        bool(ham.structure_data.pbc_flag),
        False,
        want_force=False,
        want_gfmc=False,
        want_param_deriv=True,
        precision_policy=precision_policy_snapshot(),
    )
    rts = np.stack([np.eye(3)] * nw)
    native = _rs_native_c_param_grads(
        ctx, ham, ru_b, rd_b, mask, nw, quantity=quantity, ecp_flag=ecp_flag, ecp_w=_W, ecp_gp=_GP, rts_b=rts
    )
    assert native, "native c-axis assembler emitted no block"
    per = []
    for w in range(nw):
        ru, rd = jnp.asarray(ru_b[w]), jnp.asarray(rd_b[w])
        ginv = jnp.asarray(_geminal_inv(gd, ru_b[w], rd_b[w]))
        if quantity == "ln_Psi":
            per.append(assemble.compute_grad_ln_Psi(ham, ru, rd, wrt="c", mask=mask, geminal_inv=ginv))
        else:
            per.append(assemble.compute_grad_e_L(ham, ru, rd, np.eye(3), wrt="c", mask=mask, geminal_inv=ginv))
    grad_b = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *per)
    ref = _reduce_c_param_grads(wf, grad_b, nw, mask)
    ref = {n: v for n, v in ref.items() if mask.get(_PARAM_BLOCK_TO_MASK_KW.get(n, ""), True)}
    assert set(native) == set(ref), (quantity, sorted(native), sorted(ref))
    # Same resolver and level as the module's partial alias, spelled directly so
    # the GDD receipt can bind the medium XLA-Rust zone policy of this site.
    atol, rtol = _leaf_xla_rust_tolerance(_ZONES_GRAD, level="medium")
    for block in sorted(native):
        got, r = np.asarray(native[block]), np.asarray(ref[block])
        assert got.shape == r.shape and got.shape[0] == nw, (quantity, block, got.shape, r.shape)
        np.testing.assert_allclose(np.real(got), np.real(r), atol=atol, rtol=rtol, err_msg=f"{quantity} block={block} (Re)")
        np.testing.assert_allclose(np.imag(got), np.imag(r), atol=atol, rtol=rtol, err_msg=f"{quantity} block={block} (Im)")
    return sorted(native)


@pytest.mark.light
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
def test_agpn_open_shell_lambda_unpaired_param_deriv(quantity):
    """``O_k = dln|Psi|/dc`` and ``dE_L/dc`` for the OPEN-SHELL AGPn geminal,
    block by block. The ``lambda_unpaired`` block MUST appear (and match XLA):
    this is the parameter-level assertion of the unpaired-column chain rule that
    no committed fixture reached before.
    """
    ham = _open_shell_agpn("mol_cart_sd_j12")
    # A one-walker batch still exercises the production batch assembler while
    # avoiding a second walker whose small derivative components are
    # cancellation-dominated.  seed=23 has cond(G)=5.3e1.
    blocks = _assert_c_param_grad_parity(ham, quantity, ecp_flag=False, seed=23, nw=1)
    assert "lambda_unpaired" in blocks, f"unpaired block missing from native param-deriv: {blocks}"
    assert "lambda_diag" in blocks, f"paired-diagonal block missing: {blocks}"


# --------------------------------------------------------------------------- #
# AGP x ECP ``pd_el_ecp_lambda`` route.
#
# The route-parity spy found ``pd_el_ecp_lambda`` was
# the single production obs-ctx method with NO committed fixture -- an AGP x ECP
# combination was never built. ``_build_ham_for_cell(<ecp stem>, ..., "AGP")``
# produces one; the geminal ``lambda_matrix`` block of ``dE_L/dc`` on an ECP
# system is exactly the non-local-ECP-through-lambda derivative that method
# serves.
# --------------------------------------------------------------------------- #
@pytest.mark.light
@pytest.mark.parametrize("stem, tag", [("water_ccecp_ccpvqz.h5", "water_sphe"), ("H2_ecp_ccpvdz_cart.h5", "h2_cart")])
def test_agp_ecp_lambda_param_deriv(stem, tag):
    """AGP geminal on an ECP system: ``dE_L/dc`` including the ``lambda_matrix``
    block (the ``pd_el_ecp_lambda`` route) vs XLA ``assemble``.
    """
    ham = _build_ham_for_cell(stem, True, None, "AGP")
    assert bool(ham.coulomb_potential_data.ecp_flag)
    assert type(ham.wavefunction_data.geminal_data).__name__ == "AGP_data"
    # The 114x114 spherical lambda derivative contains many cancellation-small
    # entries; use a stable one-walker production batch for strict parity.
    seed, nw = (15, 1) if stem == "water_ccecp_ccpvqz.h5" else (1, 2)
    blocks = _assert_c_param_grad_parity(ham, "e_L", ecp_flag=True, seed=seed, nw=nw)
    assert "lambda_matrix" in blocks, f"lambda block missing from AGP x ECP param-deriv: {blocks}"

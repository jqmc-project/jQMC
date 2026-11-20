"""Long-chain carried-vs-fresh gate for the Rust GFMC determinant grad/lap caches.

Inside one LRDMC projection interval the native drivers keep
a ``MeshFastState`` (real) or ``MeshFastStateC`` (complex twist) whose determinant
derivative cache is seeded once and advanced after every hop: the real route
carries the derivative columns and the ``winvup/winvdo`` contraction rows by the
same Sherman-Morrison factors as ``G^-1``; the complex route carries its own
fp64 kinetic-family inverse ``kin_ginv`` and value state. The per-electron
``grad ln|Det|`` / ``lap ln|Det|`` read from those caches feed the continuum
kinetic term of every sub-step.

``NativeObsCtx.debug_det_grad_lap_chain`` drives the PRODUCTION state through a
prescribed hop list (``advance_after_hop`` + the per-sub-step contraction) and
returns, every ``_CHECK_EVERY`` hops, the carried result next to

* ``fresh``: a new state built at the same configuration, whose first
  contraction seeds every table and forms the contraction weights by full
  matrix products (the from-scratch reference of the whole chain), and
* ``twin``: the matrix-product contraction of the CARRIED derivative tables
  with the carried inverse and value state (real: ``contract_det_grad_lap_cache``;
  complex: ``contract_det_grad_lap_cache_complex_matmul``), which isolates the
  carried contraction weights from the rest of the chain.

The hop
list is the deterministic LRDMC-like schedule of
``tests/_det_gradlap_chain_fixtures.py`` (``N_HOPS`` hops, one electron per spin
frozen), so any per-hop corruption of a carried quantity accumulates.

Tolerance, each mode judged within its own precision:

* carried vs twin, every case: ``get_tolerance_min(_ZONES_LNDET, "strict")``
  (same tables, same inverse; only the carried weights differ);
* carried vs fresh, molecular and complex-twist caches: the same pair.
  The ln|Det| derivatives cross the ``ao_eval`` zone (value state and ``G^-1``),
  so in mixed precision the pair resolves to the fp32 zone tier, exactly as the
  sibling carried-vs-fresh and carried-vs-rebuilt verdicts of
  ``test_lrdmc_mesh_chain_rust.py`` (``get_tolerance_min(zones, "strict")``); in
  full precision it is the fp64 ``det_grad_lap`` strict pair. The context is
  built with the active precision policy, so the mixed lane judges mixed arithmetic;
* carried vs the XLA ``compute_grads_and_laplacian_ln_Det`` at the same
  configuration, every ``_ANCHOR_EVERY`` hops: an independent implementation,
  so a defect shared by the carried and the fresh Rust contraction (both use
  the same final kernel) cannot pass. Molecular: the strict XLA-Rust
  consistency pair of the ln|Det| zones; periodic: the same strict pair raised
  to the periodic route-parity floor ``ATOL/RTOL_FLOOR_MEDIUM`` that the other
  periodic Rust-vs-XLA derivative parity tests apply;
* carried vs fresh, real periodic caches: ``get_tolerance_gfmc_fast_pbc(_ZONES_LNDET)``, the
  documented periodic fast-interval floor (the fp32 zone envelope in mixed, as in
  the sibling ``test_lrdmc_mesh_chain_rust.py``). The carried value column of a moved
  electron comes from the screened fast-cut folded kernel
  (``update_det_grad_lap_cache_and_value_folded``) while a fresh state seeds its
  values with the single-position value kernel; their ~1e-9 value-class
  difference reaches the Sherman-Morrison carry, which is the fast-chain class
  that floor was measured for.
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
import pytest

import jqmc_rs
from jqmc._precision import get_tolerance, get_tolerance_min, get_tolerance_xla_rust_consistency, precision_policy_snapshot
from jqmc._setting import ATOL_FLOOR_MEDIUM, RTOL_FLOOR_MEDIUM, get_tolerance_gfmc_fast_pbc
from jqmc.determinant import compute_grads_and_laplacian_ln_Det
from tests._det_gradlap_chain_fixtures import (
    N_HOPS,
    NEAR_SINGULAR_MIN_COND,
    build_case,
    hop_schedule,
    near_singular_schedule,
    start_configuration,
)

pytestmark = [pytest.mark.with_chdir]

_CHECK_EVERY = 10
# Every checkpoint: with one ao_eval value convention in the carried
# geminal the mixed lane meets the strict XLA-Rust pair at every checkpoint, and a
# hopped value column in a different convention is red within a few hundred hops.
_ANCHOR_EVERY = 10
# Zones of the per-electron ln|Det| derivatives (as in test_grads_laplacian.py).
_ZONES_LNDET = ("ao_eval", "ao_grad_lap", "det_grad_lap")
_NAMES = ("grad_up", "grad_dn", "lap_up", "lap_dn")

# case -> (complex ctx, tolerance family)
_CASES = {
    "sd_real_mol_open": (False, "strict"),
    "agp_mo_real_mol_open": (False, "strict"),
    "agp_real_mol_open": (False, "strict"),
    "sd_real_pbc_open": (False, "fast_pbc"),
    "agp_real_pbc_open": (False, "fast_pbc"),
    "sd_complex_twist_open": (True, "strict"),
    "agp_complex_twist_open": (True, "strict"),
    "sd_complex_bspline_twist": (True, "strict"),
    # Multi-electron B-spline (O-lsda, 4 up / 2 dn).
    "sd_real_bspline_open": (False, "fast_pbc"),
    "sd_complex_bspline_open": (True, "strict"),
}


@pytest.fixture
def _chain_tmp(tmp_path_factory):
    # Per case: with a module-scoped directory a later case would load the HDF5
    # an earlier case converted from the committed QE save, so its own execution
    # would never reach that save and its receipt would depend on test order.
    return str(tmp_path_factory.mktemp("det_gradlap_chain_rust"))


def _tolerance(family):
    if family == "fast_pbc":
        return get_tolerance_gfmc_fast_pbc(_ZONES_LNDET)
    return get_tolerance_min(_ZONES_LNDET, "strict")


def _anchor_tolerance(ham):
    if bool(ham.structure_data.pbc_flag):
        # Periodic Rust-vs-XLA derivative parity: the route-parity floor, as in
        # test_periodic_lsda.py / test_grads_laplacian_far.py (the Rust periodic
        # derivative kernels screen far-image tails; measured here up to
        # ~1.3e-5 relative on fresh states, independent of the carry).
        atol, rtol = get_tolerance_xla_rust_consistency(_ZONES_LNDET, level="strict")
        return max(atol, ATOL_FLOOR_MEDIUM), max(rtol, RTOL_FLOOR_MEDIUM)
    return get_tolerance_xla_rust_consistency(_ZONES_LNDET, level="strict")


def _ctx(ham, complex_ctx):
    wf = ham.wavefunction_data
    return jqmc_rs.build_obs_ctx(
        wf.geminal_data,
        wf.jastrow_data,
        ham.coulomb_potential_data,
        complex_ctx,
        bool(ham.structure_data.pbc_flag),
        False,
        want_force=False,
        want_gfmc=True,
        precision_policy=precision_policy_snapshot(),
    )


def _chain_rows(ham, complex_ctx, n_hops, hops=None, check_every=_CHECK_EVERY):
    r_up, r_dn = start_configuration(ham)
    if hops is None:
        hops = hop_schedule(ham, r_up, r_dn, n_hops=n_hops)
    checkpoints = sorted(set(range(0, n_hops + 1, check_every)) | {n_hops})
    rows = _ctx(ham, complex_ctx).debug_det_grad_lap_chain(
        r_up,
        r_dn,
        [h[0] for h in hops],
        [h[1] for h in hops],
        np.ascontiguousarray([h[2] for h in hops], dtype=np.float64),
        checkpoints,
    )
    assert [int(row["hop"]) for row in rows] == checkpoints
    return hops, rows


@pytest.mark.light
@pytest.mark.parametrize("case", sorted(_CASES))
def test_rust_det_gradlap_cache_long_chain_matches_fresh(case, _chain_tmp):
    """Carried Rust det grad/lap equals a fresh state at every checkpoint of N_HOPS."""
    complex_ctx, family = _CASES[case]
    ham = build_case(case, _chain_tmp)
    hops, rows = _chain_rows(ham, complex_ctx, N_HOPS)
    assert {is_up for is_up, _, _ in hops} == {True, False}, f"{case}: the chain must move both spins"
    checks = (("fresh", _tolerance(family)), ("twin", _tolerance("strict")))
    geminal = ham.wavefunction_data.geminal_data
    anchor_atol, anchor_rtol = _anchor_tolerance(ham)
    r_up, r_dn = (np.array(r) for r in start_configuration(ham))
    applied = 0
    n_anchor = 0
    for row in rows:
        hop = int(row["hop"])
        while applied < hop:
            is_up, e, new = hops[applied]
            (r_up if is_up else r_dn)[e] = new
            applied += 1
        if hop % _ANCHOR_EVERY == 0 or hop == N_HOPS:
            n_anchor += 1
            reference = compute_grads_and_laplacian_ln_Det(geminal, r_up, r_dn)
            for name, got, ref in zip(_NAMES, row["carried"], reference):
                got = np.asarray(got)
                ref = np.asarray(ref)
                label = f"{case} hop {hop}: {name} carried vs XLA"
                assert got.shape == ref.shape, label
                assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref)), f"{label}: non-finite"
                np.testing.assert_allclose(got, ref, atol=anchor_atol, rtol=anchor_rtol, err_msg=label)
        for reference, (atol, rtol) in checks:
            for name, got, ref in zip(_NAMES, row["carried"], row[reference]):
                got = np.asarray(got)
                ref = np.asarray(ref)
                label = f"{case} hop {row['hop']}: {name} carried vs {reference}"
                assert np.iscomplexobj(got) == complex_ctx
                assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref)), f"{label}: non-finite"
                np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, err_msg=label)
    assert n_anchor >= N_HOPS // _ANCHOR_EVERY


# Cases with two or more up electrons (the excursion moves up electron 1 onto 0).
_NEAR_SINGULAR_CASES = sorted(case for case in _CASES if case != "sd_complex_bspline_twist")


@pytest.mark.light
@pytest.mark.parametrize("case", _NEAR_SINGULAR_CASES)
def test_rust_det_gradlap_cache_near_singular_excursions_match_twin(case, _chain_tmp):
    """Carried contraction weights equal the matrix-product route through near-singular hops.

    Along :func:`near_singular_schedule` (a same-spin pair repeatedly brought
    ``1e-5`` bohr apart, ``cond(G) >= NEAR_SINGULAR_MIN_COND``, and separated
    again) the Sherman-Morrison factors that advance the carried contraction
    weights -- the real ``winvup/winvdo`` rows and the complex
    ``m kin_ginv`` / ``O_up kin_ginv^T`` -- are as large as a walk can make
    them. After EVERY hop the carried result must equal the matrix-product
    contraction of the same carried derivative tables with the same carried
    inverse and value state (the complex matrix-product route / the real
    ``contract_det_grad_lap_cache``): the A/B of the incremental weights against
    their from-scratch rebuild, strict fp64 ``det_grad_lap`` pair, each precision
    mode within itself.

    The fresh-state comparison of :func:`test_rust_det_gradlap_cache_long_chain_matches_fresh`
    is deliberately not repeated here: across such excursions the carried
    Sherman-Morrison INVERSE itself (shared by the incremental and the
    matrix-product contraction, and by the value chain) departs from a fresh
    inverse at the ``eps * cond(G)`` class, which is the carried-inverse drift
    class, not a property of the contraction weights tested here.
    """
    complex_ctx, _family = _CASES[case]
    ham = build_case(case, _chain_tmp)
    r_up, r_dn = start_configuration(ham)
    hops, cond_max = near_singular_schedule(ham, r_up, r_dn)
    assert cond_max >= NEAR_SINGULAR_MIN_COND, f"{case}: excursion cond(G)={cond_max:.3g} is not near-singular"
    _, rows = _chain_rows(ham, complex_ctx, len(hops), hops=hops, check_every=1)
    assert len(rows) == len(hops) + 1
    # Deliberately the fp64 det_grad_lap pair in both modes: the residual here is the fp64
    # carried-inverse drift at cond(G) ~ 2.5e9; the fp32 ln|Det| zone family would hide it.
    atol, rtol = get_tolerance("det_grad_lap", "strict")
    for row in rows:
        for name, got, ref in zip(_NAMES, row["carried"], row["twin"]):
            got = np.asarray(got)
            ref = np.asarray(ref)
            label = f"{case} hop {row['hop']}: {name} carried vs twin"
            assert np.iscomplexobj(got) == complex_ctx
            assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref)), f"{label}: non-finite"
            np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, err_msg=label)

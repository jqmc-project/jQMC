"""The "8 sets" analytic-vs-``_auto`` integration harness +
a gap inventory.

The "8 sets" are ``{ln_Psi, e_L} x {c, r, R, eps}``, exercised through the PRODUCTION
bridge closures (``derivatives/bridge.py``) -- the SAME ``jit(vmap(...))``-over-a-
TRACED-ham factories the MCMC/GFMC drivers build (``jqmc_mcmc._cached_grad_*``), so
the harness is ROUTE-FAITHFUL: traced-ham + vmap-over-a-walker-batch + running-SM
``geminal_inv`` threading + ``mpc_ctx``, exactly the production shape::

    a = bridge.batched_grad_<q>(wrt=axis, ..., analytic=True )(ham, r_up, r_dn[, RT], ginv)
    b = bridge.batched_grad_<q>(wrt=axis, ..., analytic=False)(ham, r_up, r_dn[, RT], ginv)
    tree-allclose(a, b)          # tight fp64 tol; full precision only (see Precision note below)

``a == b`` is switch-ON (``analytic=True``, the assembled analytic path) vs switch-OFF
(``analytic=False``, the ``_auto`` autograd twin) THROUGH THE SAME FACTORY -- both get
the identical ``wrt``/``mask``/``mpc_ctx``/``geminal_inv``, so the comparison is
apples-to-apples: whatever branches are frozen/pruned are frozen on both. ``wrt`` selects
the target axis -- for the c-axis the ``_C_MASK`` (every parameter block on EXCEPT the two
geminal radial-basis blocks, which ``with_param_grad_mask`` makes mutually exclusive with
``opt_geminal``; see ``_C_MASK``); for ``r``/``R``/``eps`` the physical-axis switch. Running this through the
production ``jit(vmap)`` bridge (not the eager public entry) is ALSO what keeps the harness
off the eager-microcompile-flood XLA-accumulation crash (the eager per-axis path floods
thousands of tiny compiles per call). Per-fixture ``bridge.cache_clear()`` teardown drops
the inner-jit LRU + XLA programs so walking 13 distinct hams does not exhaust LLVM mmap.

``analytic=True`` is the assembled analytic path; it raises ``NotImplementedError``
on a flagged axis it does not implement (NO silent ``_auto`` fallback).
This harness does NOT fill any analytic gap itself: where the analytic path raises,
the ``(fixture, quantity, axis)`` combination is recorded in the GAP inventory and the
test FAILS hard (RED, not xfail -- an unimplemented axis is a mission remainder for the
orchestrator to implement, not something to hide). The session-scoped
``_gap_inventory_report`` fixture prints the accumulated gaps at teardown, giving the
complete remaining-gap inventory across xdist workers.

Precision: the two numerical tests carry full-only exclusions. The shared
``_assert_axis_allclose`` comparator also supports same-mode mixed comparisons
used by the separately registered generated matrix and does not self-skip.
The raw composite radial-leaf test has its own mixed exclusion.
The flag-splitting/empty-mask contract has no numerical comparison and runs in
both precision modes. Passing it does not certify a skipped numerical test.

The shared machinery (ham builders, bridge routing, the ``_assert_axis_allclose`` comparator, the
axis-flag helpers, the fixture builders, and the GAP-inventory fixtures) lives in the sibling
``_grad_fidelity_harness`` module; this file keeps only the curated ``_FIXTURES`` list and the
three ``test_*`` functions.
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

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

from jqmc import derivatives
from jqmc._setting import ATOL_AUTO_VS_ANALYTIC_DERIV, RTOL_AUTO_VS_ANALYTIC_DERIV
from jqmc.derivatives import bridge
from tests.test_jqmc_kernel._grad_fidelity_harness import (
    _NW,
    _QE_STEM,
    _assert_axis_allclose,
    _batched_geminal_inv,
    _build_ham,
    _clear_bridge_caches,  # noqa: F401  autouse fixture re-exported for collection
    _draw,
    _draw_batch,
    _flags_for_axis,
    _gap_inventory_report,  # noqa: F401  session autouse fixture re-exported for collection
    _record_gap,
    requires_qe,
)

# --- fixture matrix (LIGHT): molecular cart / sphe / ECP GTO + light periodic
# (real Gamma & complex twist), with and without an analytic-polynomial Jastrow.
# (id, trexio stem, with_jastrow, spec). ``spec`` is an optional dict extending the
# wavefunction so the c-axis exercises the two known remaining analytic gaps that the
# base matrix (J1+J2, MO/SD geminal, no MPC) does NOT hit:
#   * ``"j3"``: attach an ACTIVE three-body Jastrow whose orbital leaf is of the given
#     type -- ``"pgto"`` / ``"complex"`` (reuse the fixture's own AO leaf) or ``"mo"``
#     (that AO leaf wrapped in an ``MOs_data``). The c-axis mask turns on
#     ``opt_J3_basis_exp/coeff``; that block is analytic ONLY for a single-leaf REAL
#     cart/sphe GTO J3 orbital (assemble/_capability), so a pgto / complex-twist / MO
#     J3 orbital surfaces gap #33 (J3-basis on a non-cart/sphe-GTO leaf).
#   * ``"dense_agp"``: force the MO/SD geminal into a dense-lambda AO-AGP
#     (``convert_from_MOs_to_AOs``) -- the #35 AO-AGP shape.
#   * ``"mpc"``: thread the MPC computational context (``mpc_ctx = (grid_shape, eta,
#     frozen_overlap)``, built by ``_mpc_ctx_for`` exactly as ``jqmc_mcmc`` does) into BOTH bridge
#     closures -- the analytic assemble AND the MPC-aware ``_auto`` twin key off ``mpc_ctx`` (dense-AGP
#     + MPC = gap #35, AO-AGP MPC-c). There is NO ``mpc_flag`` ham attribute.
# Reuses the committed trexio sources -- no h5. ---
_FIXTURES = [
    ("H2_cart_J", "H2_ae_ccpvdz_cart.h5", True, None),  # molecular cart-GTO, closed-shell, J1+J2
    ("H2_cart_noJ", "H2_ae_ccpvdz_cart.h5", False, None),  # ... no Jastrow
    ("N_cart_J", "N_ae_ccpvdz_cart.h5", True, None),  # molecular cart-GTO, OPEN-shell, J1+J2
    ("H2_sphe_J", "H2_ae_ccpvdz_sphe.h5", True, None),  # molecular sphe-GTO
    ("H2_ecp_cart_J", "H2_ecp_ccpvdz_cart.h5", True, None),  # molecular ECP (local + non-local)
    ("H2_ecp_cart_noJ", "H2_ecp_ccpvdz_cart.h5", False, None),  # ... no Jastrow
    ("Li_pbc_gamma_J", "lithium.h5", True, None),  # periodic real Gamma, ECP sphe-pgto, J1+J2
    # periodic real Gamma ECP + DTN (u/chi/f): exercises the periodic-ECP dE_L/dc DTN block
    # THROUGH THE PRODUCTION BRIDGE (the bridge is the only route that catches bridge-specific
    # bugs, so the route fidelity is asserted here, not via compute_grad_* directly).
    ("Li_pbc_gamma_JDTN", "lithium.h5", True, {"dtn": True}),
    ("Li_pbc_twist_J", "lithium_k025.h5", True, None),  # periodic complex twist (k != 0)
    # --- complex-twist + DTN polynomial Jastrow (u/chi/f). The twist x DTN x ECP x e_L/c cell was a
    # gate over-claim (dtn.grad_T_dtn_wrt_param complex scan-carry / species-scatter dtypes), fixed
    # by result_type(M, GD); this row exercises that cell PLUS the DTN physical axes on twist.
    ("Li_pbc_twist_JDTN", "lithium_k025.h5", True, {"dtn": True}),
    # --- #33: J3-basis optimization beyond the single-leaf cart/sphe GTO orbital -----
    # pGTO J3 orbital (periodic real Gamma) and complex-twist pGTO J3 orbital: c-axis
    # opt_J3_basis + all physical axes analytic (filled #33 subcases).
    ("Li_pbc_gamma_J3pgto", "lithium.h5", True, {"j3": "pgto"}),
    ("Li_pbc_twist_J3cplx", "lithium_k025.h5", True, {"j3": "complex"}),
    # --- MO-backed J3 on a PLAIN (non dense-AO-AGP) molecular geminal ------------------
    # The default molecular geminal is a plain SD (MO leaves); an MO-backed J3 (cart-GTO leaf
    # wrapped in an identity MOs_data) exercises the MO->bare-AO J3 fold (_j3_mo_to_bare_ao)
    # on the R / eps (force / stress) axes for a geminal that is NOT the dense-AO-AGP pairing.
    # The fold is geminal-independent, so all 8 sets
    # (ln_Psi/e_L x c/r/R/eps) must match _auto -- guards against re-refusing MO-J3 Pulay.
    ("H2_SD_J3MO", "H2_ae_ccpvdz_cart.h5", True, {"j3": "mo"}),
    # --- dense AO-AGP on the QE pgto+pw COMPOSITE (well-conditioned; see the note at
    # _qe_composite_pieces). MO-backed J3 whose underlying AO leaf is the COMPOSITE itself:
    # covers the #33 subcase (composite J3-basis: value O_k + kinetic dT/dc contract
    # per child; PW children carry no radial params) AND the composite J3 physical axes
    # (drift / force / stress descend to the children, PW rows R-independent /
    # affine-value-invariant). Paired with the dense AO-AGP geminal (the MO-J3 R/eps fold
    # ``_j3_mo_to_bare_ao`` is verified for the dense-AGP pairing only). NOTE: the flat
    # block dict does NOT publish j3_basis_* for a composite (GTO-only accessor), so the
    # c-axis compare covers the other 6 blocks; the composite J3-basis grads are verified
    # at the raw grad-pytree leaf level in test_composite_j3_basis_grads_vs_auto below.
    pytest.param("Li_AGP_J3MO", _QE_STEM, True, {"dense_agp": True, "j3": "mo"}, marks=requires_qe),
    # --- #35: dense AO-AGP + run-time MPC on the composite --------------------------
    # The e_L/c axis runs the analytic dT/dc + mpc.grad_Vmpc_ee_wrt_c_AGP (explicit
    # singular-vector VJP) against the MPC-aware _auto twin (hoisted frozen overlap):
    # on this well-conditioned composite the natural-orbital SVs are all distinct, so the
    # SVD derivative is well-defined and a==b holds -- the combination that was xfailed
    # ("no valid a==b reference") on the bare-pgto basis. Same MO-backed composite J3.
    pytest.param("Li_AGP_MPC_J3MO", _QE_STEM, True, {"dense_agp": True, "mpc": True, "j3": "mo"}, marks=requires_qe),
]
_FIXTURE_IDS = [(f.values[0] if hasattr(f, "values") else f[0]) for f in _FIXTURES]


@pytest.mark.light
@pytest.mark.skip_mixed_precision(
    reason="analytical-vs-autograd is an fp64 correctness oracle; mixed precision uses production analytical smoke tests"
)
@pytest.mark.parametrize("fixture_id,stem,with_jastrow,spec", _FIXTURES, ids=_FIXTURE_IDS)
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
@pytest.mark.parametrize("axis", ["c", "r", "R", "eps"])
def test_8set_analytic_vs_auto(fixture_id, stem, with_jastrow, spec, quantity, axis):
    """One of the 8 sets on one fixture, ROUTE-FAITHFUL: switch-ON (analytic, assembled) == switch-OFF
    (``_auto`` autograd twin) THROUGH THE PRODUCTION BRIDGE -- ``jit(vmap)`` over a walker batch with a
    TRACED ham, running-SM ``geminal_inv`` threading, and ``mpc_ctx`` -- exactly the driver's shape.

    Where analytic=True raises NotImplementedError (at trace time) the combination is a documented
    analytic gap -> recorded in the GAP inventory, then re-raised as a hard failure (RED, not xfail;
    it is a mission remainder for the orchestrator to implement, not something this harness hides).
    Both closures come from the SAME factory with the SAME ``wrt``/``mask``/``mpc_ctx``/
    ``geminal_inv``, differing ONLY in ``analytic=True`` vs ``False`` (the a==b switch).
    """
    ham = _build_ham(stem, with_jastrow, spec)
    r_up, r_dn = _draw_batch(ham)
    ginv = _batched_geminal_inv(ham, r_up, r_dn)

    # a = switch-ON (analytic) closure; b = switch-OFF (_auto) closure -- SAME factory, SAME route.
    # c-axis: the ``_C_MASK`` (all blocks minus the conflicting geminal-basis pair); physical axes
    # ignore ``mask``. Both closures get the identical mask so any frozen block is frozen on both.
    c_mask = _flags_for_axis("c") if axis == "c" else None
    if quantity == "ln_Psi":
        make = lambda an: bridge.batched_grad_ln_Psi(wrt=axis, mask=c_mask, with_geminal_inv=True, analytic=an)  # noqa: E731
        call = lambda f: f(ham, r_up, r_dn, ginv)  # noqa: E731
    else:
        RT = jnp.broadcast_to(jnp.eye(3), (_NW, 3, 3))
        make = lambda an: bridge.batched_grad_e_L(  # noqa: E731
            wrt=axis, mask=c_mask, with_geminal_inv=True, analytic=an
        )
        call = lambda f: f(ham, r_up, r_dn, RT, ginv)  # noqa: E731

    # analytic path: NotImplementedError (raised during the jit trace) == a remaining analytic gap.
    # Record it in the GAP inventory for teardown reporting, then FAIL hard -- a mission remainder
    # the orchestrator must implement, not an xfail escape.
    try:
        a = call(make(True))
    except NotImplementedError as e:
        _record_gap(fixture_id, quantity, axis, str(e))
        pytest.fail(f"GAP {fixture_id} {quantity}/{axis}: analytic NotImplementedError -- {str(e).splitlines()[0]}")

    b = call(make(False))  # ground-truth reference: the _auto autograd twin through the same route
    _assert_axis_allclose(ham, axis, a, b, cell_id=fixture_id, quantity=quantity)


@pytest.mark.light
@pytest.mark.skip_mixed_precision(
    reason="analytical-vs-autograd is an fp64 correctness oracle; mixed precision uses production analytical smoke tests"
)
@requires_qe
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
def test_composite_j3_basis_grads_vs_auto(quantity):
    """Composite J3 basis: the analytic per-child radial exp/coeff grads (value O_k via
    basis.grad_J3_wrt_basis + kinetic dT/dc via basis.grad_T_wrt_J3_basis, both contracting
    per composite child through basis._basis_contract_*_general / ao_deriv.j3_basis_derivs)
    equal the ``_auto`` autograd grads. The flat block dict does NOT publish ``j3_basis_*``
    for a composite J3 orbital (``Jastrow_data.collect_param_grads`` is GTO-only), so the
    8-set c-axis compare above never sees these grads -- compare the RAW grad-pytree radial
    leaves of every GTO/pGTO child directly (PW children carry no radial params)."""
    ham = _build_ham(_QE_STEM, True, {"dense_agp": True, "j3": "mo"})
    r_up, r_dn = _draw(ham)
    flags = _flags_for_axis("c")
    if quantity == "ln_Psi":
        a = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=True)["c"]
        b = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=False)["c"]
    else:
        a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), flags, analytic=True)["c"]
        b = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), flags, analytic=False)["c"]
    ga = a.jastrow_data.jastrow_three_body_data.orb_data.aos_data  # grad pytree: MO wrapper -> composite
    gb = b.jastrow_data.jastrow_three_body_data.orb_data.aos_data
    # Full-precision only (skipped in mixed above): tight fp64 route floor, same as the 8-set axes.
    atol, rtol = ATOL_AUTO_VS_ANALYTIC_DERIV, RTOL_AUTO_VS_ANALYTIC_DERIV
    compared = 0
    for i, (ca, cb) in enumerate(zip(ga.children, gb.children)):
        if "gto" not in type(ca).__name__:
            continue
        np.testing.assert_allclose(
            np.asarray(ca.radial.exponents),
            np.asarray(cb.radial.exponents),
            atol=atol,
            rtol=rtol,
            err_msg=f"{quantity} composite J3-basis child{i} d/d exponents",
        )
        np.testing.assert_allclose(
            np.asarray(ca.radial.coefficients),
            np.asarray(cb.radial.coefficients),
            atol=atol,
            rtol=rtol,
            err_msg=f"{quantity} composite J3-basis child{i} d/d coefficients",
        )
        compared += 1
    assert compared > 0, "no GTO child compared -- the composite J3-basis check would be vacuous"


@pytest.mark.light
@pytest.mark.parametrize("axis", ["c", "r", "R", "eps"])
def test_flag_split_selects_requested_axis(axis):
    """Contract (no numeric compare -> not precision-sensitive): the unified interface
    returns a dict keyed exactly by the requested axis, the c-axis is implied by the
    parameter-block flags (no ``opt_c``), and the ``_auto`` path honours ``flags``."""
    ham = _build_ham("H2_ae_ccpvdz_cart.h5", with_jastrow=True)
    r_up, r_dn = _draw(ham)
    flags = _flags_for_axis(axis)

    out = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=False)
    assert set(out) == {axis}, f"{axis}: expected dict keyed by {{{axis!r}}}, got {sorted(out)}"
    outL = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), flags, analytic=False)
    assert set(outL) == {axis}

    # c-axis is implied by any parameter block being True; the empty flag set selects nothing.
    assert set(derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, {}, analytic=False)) == set()
    assert "opt_c" not in _flags_for_axis("c")

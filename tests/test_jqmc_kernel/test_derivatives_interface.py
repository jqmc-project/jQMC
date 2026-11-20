"""The unified ``opt_flags``/``analytic`` gradient interface.

Covers the interface CONTRACT (opt_flag vocabulary +
splitting into the whitelist mask + axes, axis bundling, ``analytic`` routing,
strict no-``_auto`` error), plus an ``analytic=False`` equivalence against the
live per-axis ``_auto`` path so the bundle is proven to carry the same
gradients. Real analytic-vs-``_auto`` numerics across the 8 sets are in the
integration harness, not here.
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

from jqmc._setting import ATOL_BIT_EXACT, RTOL_BIT_EXACT
from jqmc import derivatives
from jqmc.derivatives import _auto

from .test_jqmc_derivatives import _build_ham, _draw

_STEM = "H2_ae_ccpvdz_cart.h5"


def _full_mask(**on):
    """The complete whitelist mask: named blocks True, every other block False --
    exactly what ``_split_flags`` produces for the c-axis."""
    return {k: bool(on.get(k, False)) for k in derivatives._PARAM_FLAG_KEYS}


def _true_keys(mask):
    return {k for k, v in mask.items() if v}


def _tree_allclose(a, b, *, rtol=RTOL_BIT_EXACT, atol=ATOL_BIT_EXACT):
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=rtol, atol=atol)


# ---------------------------------------------------------------- flag logic
@pytest.mark.light
def test_split_flags_param_block_requests_c():
    """Split flags param block requests c."""
    pm, axes = derivatives._split_flags({"opt_geminal": True})
    assert axes == ["c"]
    assert pm == _full_mask(opt_geminal=True)  # complete mask, only geminal True
    assert _true_keys(pm) == {"opt_geminal"}


@pytest.mark.light
def test_split_flags_all_false_param_no_c():
    # Whitelist: a block that is only ever False requests nothing -> no c-axis.
    """Split flags all false param no c."""
    pm, axes = derivatives._split_flags({"opt_geminal": False, "opt_J1_param": False})
    assert axes == []
    assert _true_keys(pm) == set()


@pytest.mark.light
def test_split_flags_no_opt_c_key():
    # The c-axis is implied by parameter blocks, so opt_c is not a valid flag.
    """Split flags no opt c key."""
    with pytest.raises(ValueError, match="unknown gradient flags"):
        derivatives._split_flags({"opt_c": True})


@pytest.mark.light
def test_split_flags_physical_axes_only():
    """Split flags physical axes only."""
    pm, axes = derivatives._split_flags({"opt_r": True, "opt_R": True, "opt_eps": True})
    assert axes == ["r", "R", "eps"]  # deterministic order, no c
    assert _true_keys(pm) == set()


@pytest.mark.light
def test_split_flags_combined_order():
    """Split flags combined order."""
    _, axes = derivatives._split_flags({"opt_eps": True, "opt_r": True, "opt_geminal": True})
    assert axes == ["c", "r", "eps"]  # c first, then physical axes in canonical order


@pytest.mark.light
def test_split_flags_unknown_key_raises():
    """Split flags unknown key raises."""
    with pytest.raises(ValueError, match="unknown gradient flags"):
        derivatives._split_flags({"opt_bogus": True})


@pytest.mark.light
@pytest.mark.parametrize("axis", ["c", "r", "R", "eps"])
def test_curated_flag_split_selects_requested_axis(axis):
    """Each requested axis is the sole result key; empty parameter flags select nothing.

    Use the curated H2 cartesian all-electron fixture with J1/J2 and the same
    seed-zero configuration. Both log-wavefunction and local-energy outputs
    obey the axis contract in each precision mode. This checks dictionary
    structure and flags, not numerical derivative accuracy.
    """
    from tests.test_jqmc_kernel._grad_fidelity_harness import (
        _build_ham as _curated_build_ham,
        _draw as _curated_draw,
        _flags_for_axis,
    )

    ham = _curated_build_ham("H2_ae_ccpvdz_cart.h5", with_jastrow=True)
    r_up, r_dn = _curated_draw(ham)
    flags = _flags_for_axis(axis)
    out = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=False)
    assert set(out) == {axis}, f"{axis}: expected dict keyed by {{{axis!r}}}, got {sorted(out)}"
    outL = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), flags, analytic=False)
    assert set(outL) == {axis}
    assert set(derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, {}, analytic=False)) == set()
    assert "opt_c" not in _flags_for_axis("c")


# ------------------------------------------------------- analytic=False bundle
@pytest.mark.light
def test_bundled_auto_matches_per_axis_ln_psi():
    """Bundled auto matches per axis ln psi."""
    ham = _build_ham(_STEM)
    r_up, r_dn = _draw(ham)
    mask = _full_mask(opt_geminal=True)
    out = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, {"opt_geminal": True, "opt_r": True}, analytic=False)
    assert set(out) == {"c", "r"}
    # each bundled axis == the live per-axis _auto call with the SAME whitelist mask
    _tree_allclose(out["c"], _auto.compute_grad_ln_Psi(ham, r_up, r_dn, wrt="c", mask=mask))
    _tree_allclose(out["r"], _auto.compute_grad_ln_Psi(ham, r_up, r_dn, wrt="r", mask=mask))


@pytest.mark.light
def test_bundled_auto_matches_per_axis_e_l():
    """Bundled auto matches per axis e l."""
    ham = _build_ham(_STEM)
    r_up, r_dn = _draw(ham)
    RT = jnp.eye(3)
    mask = _full_mask(opt_J1_param=True)
    out = derivatives.compute_grad_e_L(ham, r_up, r_dn, RT, {"opt_J1_param": True, "opt_r": True}, analytic=False)
    assert set(out) == {"c", "r"}
    _tree_allclose(out["c"], _auto.compute_grad_e_L(ham, r_up, r_dn, RT, wrt="c", mask=mask))
    _tree_allclose(out["r"], _auto.compute_grad_e_L(ham, r_up, r_dn, RT, wrt="r", mask=mask))


@pytest.mark.light
def test_bundled_mask_is_honored():
    # Two flag sets that both request c (J1 on) but differ on the geminal block
    # must give different c-axis grads (the mask reaches the kernel).
    """Bundled mask is honored."""
    ham = _build_ham(_STEM)
    r_up, r_dn = _draw(ham)
    g_on = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, {"opt_J1_param": True, "opt_geminal": True}, analytic=False)["c"]
    g_off = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, {"opt_J1_param": True, "opt_geminal": False}, analytic=False)["c"]
    on_leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(g_on)]
    off_leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(g_off)]
    assert any(not np.array_equal(a, b) for a, b in zip(on_leaves, off_leaves))


# -------------------------------------------------- analytic=True routing/strict
@pytest.mark.light
def test_analytic_true_raises_when_unsupported(monkeypatch):
    # Strict contract: analytic=True ALWAYS attempts the analytic path
    # and NEVER falls back to _auto. Post gate-removal there is no capability pre-check, so
    # a genuinely-unimplemented axis/leaf surfaces as a NotImplementedError raised by the
    # deep kernel (or the mask-conditional basis-opt guard in assemble.py). Simulate that by
    # forcing the analytic kernel to raise, and assert the dispatch PROPAGATES it rather than
    # silently routing to _auto.
    """Analytic true raises when unsupported."""
    ham = _build_ham(_STEM)
    r_up, r_dn = _draw(ham)

    def _boom(*a, **k):
        raise NotImplementedError("simulated unimplemented analytic axis")

    monkeypatch.setattr(derivatives, "_analytic_axis", _boom)
    with pytest.raises(NotImplementedError, match="simulated unimplemented analytic axis"):
        derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, {"opt_r": True}, analytic=True)


@pytest.mark.light
def test_analytic_true_routes_to_assemble(monkeypatch):
    # The analytic numeric kernel runs under ``jax.jit`` (``_jitted_assemble``,
    # an lru_cached closure over ``_assemble.compute_grad_*``). Two consequences for this
    # routing seam: (1) the stub executes at TRACE time, so its return must be a valid JAX
    # value (a plain ``object()`` sentinel raises TypeError under jit) -- use a distinctive
    # constant array instead and compare by VALUE; (2) the lru_cache must be cleared BEFORE
    # (an earlier test may have compiled the REAL kernel for the same static key, which
    # would silently bypass the stub) and AFTER (the stub-traced closure must not leak into
    # later tests). The ``called`` side-effect still fires at trace time.
    """Analytic true routes to assemble."""
    ham = _build_ham(_STEM)
    r_up, r_dn = _draw(ham)
    sentinel = jnp.full((3,), 7.5)
    called = {}

    def _stub(h, ru, rd, *, wrt, mask, fast, geminal_inv):
        called[wrt] = mask
        return sentinel

    monkeypatch.setattr(derivatives._assemble, "compute_grad_ln_Psi", _stub)
    derivatives._jitted_assemble.cache_clear()
    try:
        out = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, {"opt_geminal": True, "opt_r": True}, analytic=True)
        assert set(out) == {"c", "r"}
        np.testing.assert_array_equal(np.asarray(out["c"]), np.asarray(sentinel))
        np.testing.assert_array_equal(np.asarray(out["r"]), np.asarray(sentinel))
        # both axes receive the SAME complete whitelist mask (only geminal True)
        assert called["c"] == _full_mask(opt_geminal=True)
        assert called["r"] == _full_mask(opt_geminal=True)
    finally:
        derivatives._jitted_assemble.cache_clear()


# --------------------------------------------------------------- misc contract
@pytest.mark.light
def test_missing_flags_and_wrt_raises():
    """Missing flags and wrt raises."""
    ham = _build_ham(_STEM)
    r_up, r_dn = _draw(ham)
    with pytest.raises(TypeError):
        derivatives.compute_grad_ln_Psi(ham, r_up, r_dn)


@pytest.mark.light
def test_legacy_wrt_form_removed():
    # The public dispatch has no ``wrt=``/``mask=`` keyword form -- calling it must
    # raise TypeError naturally (unexpected keyword argument).
    """Legacy wrt form removed."""
    ham = _build_ham(_STEM)
    r_up, r_dn = _draw(ham)
    with pytest.raises(TypeError):
        derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, **{"wrt": "r"})
    with pytest.raises(TypeError):
        derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), **{"wrt": "r"})
    with pytest.raises(TypeError):
        derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, {"opt_r": True}, analytic=False, mask=None)


@pytest.mark.light
def test_param_grad_mask_defaults_single_source():
    """Drift-catch: the param-block default convention lives in
    ONE place -- ``derivatives.bridge._MASK_DEFAULTS`` (parameter blocks True, "basis"
    blocks False). Verify every duplicated declaration still mirrors it exactly, so a
    new block cannot silently default to a zero gradient.
    """
    import inspect

    from jqmc.derivatives.bridge import _MASK_DEFAULTS
    from jqmc.jqmc_mcmc import MCMC, _PARAM_BLOCK_TO_MASK_KW
    from jqmc.wavefunction import Wavefunction_data

    # (a) the single source itself obeys the "basis" predicate over the flag vocabulary.
    assert _MASK_DEFAULTS == {k: ("basis" not in k) for k in derivatives._PARAM_FLAG_KEYS}

    # (b) ``with_param_grad_mask`` literal signature defaults == the single source.
    sig_defaults = {
        n: p.default
        for n, p in inspect.signature(Wavefunction_data.with_param_grad_mask).parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    assert sig_defaults == _MASK_DEFAULTS

    # (c) the MCMC block-name namespace bridges bijectively onto the flag vocabulary and
    # its defaults derive from the single source.
    assert set(_PARAM_BLOCK_TO_MASK_KW.values()) == set(derivatives._PARAM_FLAG_KEYS)
    default_flags = MCMC._MCMC__default_param_grad_flags()
    assert default_flags == {block: _MASK_DEFAULTS[kw] for block, kw in _PARAM_BLOCK_TO_MASK_KW.items()}

"""Deterministic Rust-vs-XLA VALUE parity of the production MCMC driver on exotic AO leaves.

``test_rust_dispatch_equivalence`` proves that the native routes RUN on every periodic AO
leaf, but its stored-trajectory contract is shape/dtype/finite only, because production
Rust owns its proposal RNG and the two chains are independent.  This module closes that
gap: the XLA proposal normals (and, on the fused chunk, the XLA mesh
rotations) are injected into the otherwise unchanged native production entry points, so
the Rust and XLA drivers walk the SAME trajectory from the SAME seed and every stored
value becomes comparable.

Two test-only seams, both wrapping the once-built ``NativeObsCtx``:

* ``run_energy_chunk_batch`` (real-Gamma leaves; a force run on a real periodic leaf
  fuses HF/Pulay force and stress into this chunk): the existing
  ``test_injected_inputs=True`` switch consumes XLA normals and rotated meshes.
* ``mcmc_sweep_batch`` / ``mcmc_sweep_complex_batch`` (complex-twist leaves; a force run
  stays on the per-step dispatch): production draws the proposal normals with numpy,
  every other threefry draw is replayed natively.  The seam replaces only the normals
  with ``_draw_injected_normals`` of the same key.  The per-step ECP mesh rotation is
  already drawn by XLA from the shared key on both sides.

The walk must be identical first (accepted-move count, threefry keys and final
positions byte-equal), then ``w_L`` / ``e_L`` / ``e_L2`` / HF+Pulay force / HF+Pulay stress
buffers and the public ``get_E`` / ``get_aF`` / ``get_aT`` estimators are compared with
``get_tolerance_xla_rust_consistency`` at the same ``medium`` level the deterministic
MCMC chain parity (``test_chain_modes``) uses.  Full and mixed are each judged within
their own precision.
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

import contextlib
import os

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

from tests._finite_allclose import assert_finite_allclose
from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc._setting import ATOL_PYTHON_FP64_MEDIUM, MCMC_MIN_BIN_BLOCKS, MCMC_MIN_STEPS, RTOL_PYTHON_FP64_ZERO
from jqmc.atomic_orbital import is_complex_aos
from jqmc.jqmc_mcmc import MCMC, _jit_vmap_draw_injected_normals
from tests.test_rs_kernel.conftest import assert_native_ran, assert_no_native_ran, native_ctx_spy
from tests.test_rs_kernel.test_chain_modes import _draw_chunk_inputs
from tests.test_rs_kernel.test_rust_dispatch_equivalence import (
    _build_ham,
    _build_ham_qe,
    _qe_gamma_ok,
    _qe_twist_ok,
)

_NW = 2
_STEPS = MCMC_MIN_STEPS
_WARMUP = MCMC_MIN_BIN_BLOCKS * 3
_ZONES = ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic", "swct")
_STORED = (
    "w_L",
    "e_L",
    "e_L2",
    "force_HF_stored",
    "force_PP_stored",
    "stress_HF_stored",
    "stress_PP_stored",
)

# (id, builder kind, source).  "qe" rows are the committed QE-projection leaves of
# ``test_rust_dispatch_equivalence._QE_EXOTIC``; "trexio" rows are the committed complex
# pGTO twist files.  Every row is periodic with a Li ccECP or an AE H2 cell.
_LEAVES = [
    ("gamma_pw", "gamma_pw", "qe"),
    ("gamma_bspline", "gamma_bspline", "qe"),
    ("gamma_pgto_pw", "gamma_pgto_pw", "qe"),
    ("gamma_pgto_bspline", "gamma_pgto_bspline", "qe"),
    ("twist_pw", "twist_pw", "qe"),
    ("twist_bspline", "twist_bspline", "qe"),
    ("twist_pgto_pw", "twist_pgto_pw", "qe"),
    ("twist_pgto_bspline", "twist_pgto_bspline", "qe"),
    ("twist_sphe_pgto", "lithium_k025.h5", "trexio"),
    ("twist_cart_pgto", "H2_pbc_ae_k025_cart.h5", "trexio"),
]


def _leaf_ham(kind, source):
    if source == "trexio":
        return _build_ham(kind)
    if kind.startswith("twist_") and not _qe_twist_ok:
        pytest.fail("committed Li_k025 QE fixture is missing")
    if kind.startswith("gamma_") and not _qe_gamma_ok:
        pytest.fail("committed Li_gamma QE fixture is missing")
    return _build_ham_qe(kind)


def _is_complex(ham):
    gd = ham.wavefunction_data.geminal_data
    return bool(is_complex_aos(getattr(gd.orb_data_up_spin, "aos_data", gd.orb_data_up_spin)))


@contextlib.contextmanager
def _inject_xla_inputs(e_L_scale=None):
    """Feed the XLA proposal normals (and chunk meshes) into the native production entries.

    ``e_L_scale`` is the NEGATIVE-CONTROL hook: when set, the native e_L returned by the
    chunk (real leaves) or by the per-step ``e_L*_batch`` kernel (complex leaves) is
    multiplied by it before the driver stores it.  The walk is untouched.
    """
    import jqmc_rs

    original_build = jqmc_rs.build_obs_ctx
    seen = {"chunk": 0, "sweep": 0}

    class _InjectedCtx:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            attr = getattr(self._inner, name)
            if not callable(attr):
                return attr
            if name == "run_energy_chunk_batch":

                def _chunk(*args, **kwargs):
                    assert kwargs.get("native_rng") is True
                    assert not kwargs.get("complex", False), "the injected chunk seam is real-only"
                    keys = jnp.asarray(args[8], dtype=jnp.uint32)
                    _, (normals, rotations) = _draw_chunk_inputs(
                        keys, int(kwargs["n_steps_fast"]), int(args[12]), bool(kwargs.get("random_mesh", False))
                    )
                    base_grid = np.asarray(kwargs["base_grid_points"], dtype=np.float64)
                    call_args = list(args)
                    call_args[9] = np.ascontiguousarray(np.asarray(normals, dtype=np.float64))
                    call_args[10] = np.ascontiguousarray(np.matmul(base_grid, np.asarray(rotations, dtype=np.float64)))
                    kwargs["test_injected_inputs"] = True
                    seen["chunk"] += 1
                    out = attr(*call_args, **kwargs)
                    if e_L_scale is not None:
                        out["e_L"] = np.asarray(out["e_L"]) * e_L_scale
                    return out

                return _chunk
            if name in ("mcmc_sweep_batch", "mcmc_sweep_complex_batch"):

                def _sweep(*args, **kwargs):
                    keys = jnp.asarray(args[8], dtype=jnp.uint32)
                    num_moves = int(np.asarray(args[9]).shape[1])
                    call_args = list(args)
                    call_args[9] = np.ascontiguousarray(
                        np.asarray(_jit_vmap_draw_injected_normals(keys, num_moves), dtype=np.asarray(args[9]).dtype)
                    )
                    seen["sweep"] += 1
                    return attr(*call_args, **kwargs)

                return _sweep
            if e_L_scale is not None and name in ("e_L_batch", "e_L_complex_batch"):

                def _e_L(*args, **kwargs):
                    return np.asarray(attr(*args, **kwargs)) * e_L_scale

                return _e_L
            return attr

    def _build(*args, **kwargs):
        return _InjectedCtx(original_build(*args, **kwargs))

    jqmc_rs.build_obs_ctx = _build
    try:
        yield seen
    finally:
        jqmc_rs.build_obs_ctx = original_build


def _run(ham, rust_on, e_L_scale=None):
    previous = os.environ.get("JQMC_FLAG_RUST_ON_CPU")
    if rust_on:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    else:
        os.environ["JQMC_FLAG_RUST_ON_CPU"] = "0"
    try:
        driver = MCMC(
            hamiltonian_data=ham,
            Dt=2.0,
            mcmc_seed=34356,
            epsilon_AS=1.0e-6,
            num_walkers=_NW,
            comput_position_deriv=True,
            comput_stress=True,
            use_swct=True,
            comput_log_WF_param_deriv=False,
            comput_e_L_param_deriv=False,
            random_discretized_mesh=True,
        )
        if rust_on:
            with _inject_xla_inputs(e_L_scale=e_L_scale) as seen, native_ctx_spy() as counts:
                driver.run(num_mcmc_steps=_STEPS)
            assert_native_ran(counts, "exotic-leaf Rust driver")
            if _is_complex(ham):
                assert seen["sweep"] >= _STEPS and seen["chunk"] == 0, f"complex leaf must use the per-step sweep: {seen}"
            else:
                assert seen["chunk"] >= 1 and seen["sweep"] == 0, f"real leaf must use the fused chunk: {seen}"
        else:
            with native_ctx_spy() as counts:
                driver.run(num_mcmc_steps=_STEPS)
            assert_no_native_ran(counts, "exotic-leaf XLA driver")
        return driver
    finally:
        if previous is None:
            os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
        else:
            os.environ["JQMC_FLAG_RUST_ON_CPU"] = previous


def _assert_value(rust_value, xla_value, label):
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="medium")
    assert_finite_allclose(np.asarray(rust_value), np.asarray(xla_value), atol=atol, rtol=rtol, err_msg=label)


def _assert_same_walk(rust, xla, fixture_id, *, exact_positions=True):
    """Accepted moves and threefry keys exactly; final walkers byte-equal or, with
    ``exact_positions=False``, within the fp64 walker pin the MCMC production-vs-debug
    trajectory witness uses (``ATOL_PYTHON_FP64_MEDIUM``, zero rtol).
    """
    assert rust._MCMC__accepted_moves == xla._MCMC__accepted_moves, f"{fixture_id}: accepted-move count differs"
    np.testing.assert_array_equal(
        np.asarray(rust._MCMC__jax_PRNG_key_list), np.asarray(xla._MCMC__jax_PRNG_key_list), err_msg=f"{fixture_id}: keys"
    )
    for name in ("_MCMC__latest_r_up_carts", "_MCMC__latest_r_dn_carts"):
        r, x = np.asarray(getattr(rust, name)), np.asarray(getattr(xla, name))
        if exact_positions:
            np.testing.assert_array_equal(r, x, err_msg=f"{fixture_id}: {name}")
        else:
            assert_finite_allclose(
                r, x, atol=ATOL_PYTHON_FP64_MEDIUM, rtol=RTOL_PYTHON_FP64_ZERO, err_msg=f"{fixture_id}: {name}"
            )


def _assert_stored_values(rust, xla, fixture_id):
    for name in _STORED:
        r = np.asarray(getattr(rust, name))
        x = np.asarray(getattr(xla, name))
        assert r.shape == x.shape and r.size > 0, f"{fixture_id}: {name} empty or shape mismatch {r.shape} vs {x.shape}"
        _assert_value(r, x, f"{fixture_id}: MCMC stored {name}")
    args = (_WARMUP, MCMC_MIN_BIN_BLOCKS)
    _assert_value(rust.get_E(*args), xla.get_E(*args), f"{fixture_id}: MCMC public energy")
    _assert_value(rust.get_aF(*args), xla.get_aF(*args), f"{fixture_id}: MCMC public force")
    _assert_value(rust.get_aT(*args), xla.get_aT(*args), f"{fixture_id}: MCMC public stress")


@pytest.mark.heavy
@pytest.mark.parametrize(
    "fixture_id, kind, source",
    [pytest.param(fid, kind, src, id=fid) for fid, kind, src in _LEAVES],
)
def test_mcmc_rust_vs_xla_exotic_leaf_values(fixture_id, kind, source):
    """Same seed, same walk: stored MCMC values agree Rust vs XLA on every exotic leaf."""
    ham = _leaf_ham(kind, source)
    assert bool(ham.structure_data.pbc_flag), f"{fixture_id}: every exotic leaf row is periodic"
    assert _is_complex(ham) == fixture_id.startswith("twist_"), f"{fixture_id}: leaf realness does not match the row"
    xla = _run(ham, rust_on=False)
    rust = _run(ham, rust_on=True)
    _assert_same_walk(rust, xla, fixture_id)
    _assert_stored_values(rust, xla, fixture_id)


@pytest.mark.heavy
@pytest.mark.parametrize("fixture_id, kind", [("gamma_bspline", "gamma_bspline"), ("twist_bspline", "twist_bspline")])
def test_mcmc_exotic_leaf_value_parity_rejects_scaled_native_e_L(fixture_id, kind):
    """Negative control: a native e_L scaled by a small factor must fail the e_L comparison.

    The factor is ten times the medium XLA-Rust relative tolerance, so it is small and
    still outside the policy.  The walk is untouched (the seam scales the returned e_L
    only), so the walk check must still pass and the failure must name the stored e_L.
    """
    ham = _leaf_ham(kind, "qe")
    _, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="medium")
    scale = 1.0 + 10.0 * rtol
    xla = _run(ham, rust_on=False)
    rust = _run(ham, rust_on=True, e_L_scale=scale)
    _assert_same_walk(rust, xla, fixture_id)
    with pytest.raises(AssertionError, match=f"{fixture_id}: MCMC stored e_L"):
        _assert_value(rust.e_L, xla.e_L, f"{fixture_id}: MCMC stored e_L")
    _assert_value(rust.w_L, xla.w_L, f"{fixture_id}: MCMC stored w_L")

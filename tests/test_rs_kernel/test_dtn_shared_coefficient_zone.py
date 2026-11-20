"""Issue #774: the shared native DTN coefficient set follows only ``jastrow_grad_lap``.

``JastrowDtnData`` (``jqmc/rs/src/dtn.rs``) inflates the DTN coefficient tensors
(``_dtn_inflate_alpha|beta|gamma`` and the cutoff-policy derivatives) once per
construction. Its shared set is read by the gradient / Laplacian, drift, carried
grad/lap, Pulay (``dJ/dR``) and strain (``dJ/deps``) kernels. #771 gave the
single-move ratio kernels their own ``jastrow_ratio`` copy, but the shared set
still took float32 whenever ``jastrow_eval`` was float32, a zone none of those
consumers belongs to: with only ``jastrow_eval`` float32 the native DTN
gradient / Laplacian, Pulay and strain fields read float32-rounded coefficients
(MCMC/GFMC force/stress drift 1.31e-7 on ``twist_sphepgto_jdtn``). The XLA
carry contract (``jastrow_factor._DTN_RUST_COEFFICIENT_ZONES``) mirrored that
coupling, so the XLA carried grad/lap state rounded its coefficients the same way.

Both checks use the zone-isolation policy: every zone float64 except
``jastrow_eval`` float32, independent of the active mode. Actual and reference
come from independently loaded copies of the same serialized full input under the
full Python policy (the parameter leaves stay float64); only the zone policy
differs, as in the #757 / #766 detectors.

* ``test_native_dtn_shared_coefficient_consumers_ignore_jastrow_eval``: on the
  molecular (``mol_ecp_cart_jdtn``), real-Gamma (``pbc_gamma_sphepgto_jdtn``) and
  complex-twist (``twist_sphepgto_jdtn``) DTN fixtures, the native DTN
  gradient / Laplacian (``compute_grads_and_laplacian_dtn``, the carry seed path
  in float64), Pulay ``grad_J_dtn_wrt_R`` and strain ``_dtn_value_strain`` under the
  isolation policy must equal the all-float64 native evaluation to reduction order.
  The value ``compute_dtn_value`` is a ``jastrow_eval`` quantity: it must equal,
  bit for bit, the value under the policy whose shared set is float32 as well
  (``jastrow_eval`` and ``jastrow_grad_lap`` float32), and must not equal the
  float64 value, so the eval zone still reads float32 coefficients. The float64
  references are anchored on the XLA truth at the strict float64 budget.
* ``test_dtn_carry_coefficients_follow_the_carry_zone`` (issue #781): the carried
  DTN grad/lap state reads coefficients inflated in its own
  ``jastrow_grad_lap_carry`` zone (float64 in both modes), not in the
  ``jastrow_grad_lap`` dtype of the shared set: ``_dtn_carry_coefficient_dtype`` is
  float64 under every policy, and with ``jastrow_eval`` and ``jastrow_grad_lap``
  float32 the XLA carry seed and the native carry (``_dtn_grad_lap_carry_probe``,
  seed and seeded single-electron hops) equal the all-float64 evaluation to reduction
  order. A carry that reads the float32 shared set in mixed is off by a float32-sized
  error (XLA carry seed 5.2e-9 off its float64 rebuild on ``mol_ecp_cart_jdtn``).

Coefficients read in the wrong zone fail both checks with a float32-coefficient error in
either mode.
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jqmc._precision as precision
import jqmc_rs
from jqmc._precision import configure, get_tolerance_min, is_mixed_precision_enabled, precision_policy_snapshot
from jqmc._setting import ATOL_REDUCTION_ORDER, RTOL_REDUCTION_ORDER
from jqmc.jastrow_factor import (
    _compute_grads_and_laplacian_Jastrow_DTN_carry,
    _dtn_carry_coefficient_dtype,
    compute_grads_and_laplacian_Jastrow_DTN,
    compute_Jastrow_DTN,
)
from tests.test_rs_kernel._native_precision_fidelity import full_source_hamiltonian_factory
from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_rs_kernel.test_mcmc_observables import _walker_batch

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.with_chdir]

_NW = 2  # seeded walkers per fixture
# The one zone set to float32: none of the shared set's consumers belongs to it.
_ISOLATED_ZONE = "jastrow_eval"

_PATTERNS = [
    pytest.param("mol_ecp_cart_jdtn", id="mol_ecp_cart_jdtn"),
    pytest.param("pbc_gamma_sphepgto_jdtn", id="pbc_gamma_sphepgto_jdtn"),
    pytest.param("twist_sphepgto_jdtn", id="twist_sphepgto_jdtn"),
]


@contextlib.contextmanager
def _full_python_and_restored_native():
    """Run the body under the full Python policy; restore both backends' entry policies."""
    was_mixed = is_mixed_precision_enabled()
    native_entry = dict(jqmc_rs.native_precision_support())
    configure("full")
    try:
        yield
    finally:
        try:
            configure("mixed" if was_mixed else "full")
        finally:
            jqmc_rs.configure_native_precision(native_entry)


@contextlib.contextmanager
def _python_zone_policy(policy):
    """Install a complete zone policy for the XLA kernels (compiled executables do not key on it)."""
    saved = dict(precision._zone_dtypes)
    jax.clear_caches()
    precision._zone_dtypes.clear()
    precision._zone_dtypes.update(policy)
    try:
        yield
    finally:
        jax.clear_caches()
        precision._zone_dtypes.clear()
        precision._zone_dtypes.update(saved)


def _walkers(ham):
    r_up, r_dn = _walker_batch(ham, nw=_NW, seed=0)
    return np.ascontiguousarray(r_up), np.ascontiguousarray(r_dn)


def _native_fields(ham):
    """Native DTN value, grad/lap, Pulay and strain of every walker under the configured native policy."""
    dtn = ham.wavefunction_data.jastrow_data.jastrow_dtn_data
    assert dtn is not None and dtn.use_u and dtn.use_chi and dtn.use_f
    r_up, r_dn = _walkers(ham)
    out = {"r_up": r_up, "r_dn": r_dn, "value": [], "grad_lap": [], "pulay": [], "strain": []}
    for w in range(_NW):
        ru, rd = r_up[w], r_dn[w]
        out["value"].append(float(jqmc_rs.compute_dtn_value(dtn, ru, rd)))
        gl = jqmc_rs.compute_grads_and_laplacian_dtn(dtn, ru, rd)
        out["grad_lap"].append(np.concatenate([np.ravel(np.asarray(x, float)) for x in gl]))
        out["pulay"].append(np.ravel(np.asarray(jqmc_rs.grad_J_dtn_wrt_R(dtn, ru, rd), float)))
        out["strain"].append(np.ravel(np.asarray(jqmc_rs._dtn_value_strain(ham.wavefunction_data.jastrow_data, ru, rd), float)))
    return {k: np.asarray(v) for k, v in out.items()}


def _xla_truth(ham, r_up, r_dn, grads_and_laplacian=compute_grads_and_laplacian_Jastrow_DTN):
    """XLA DTN value and grad/lap (concatenated like ``_native_fields``) of every walker."""
    dtn = ham.wavefunction_data.jastrow_data.jastrow_dtn_data
    value, grad_lap = [], []
    for w in range(_NW):
        ru, rd = jnp.asarray(r_up[w]), jnp.asarray(r_dn[w])
        value.append(float(compute_Jastrow_DTN(dtn, ru, rd)))
        gl = grads_and_laplacian(dtn, ru, rd)
        grad_lap.append(np.concatenate([np.ravel(np.asarray(x, np.float64)) for x in gl]))
    return np.asarray(value), np.asarray(grad_lap)


def _assert_live_and_finite(ref, key):
    assert ref.size > 0 and np.all(np.isfinite(ref)), f"{key}: non-finite"
    assert np.max(np.abs(ref)) > 1.0e-3, f"{key}: inactive"


@pytest.mark.light
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
@pytest.mark.parametrize("pattern", _PATTERNS)
def test_native_dtn_shared_coefficient_consumers_ignore_jastrow_eval(pattern):
    """Native DTN grad/lap, Pulay and strain with only ``jastrow_eval`` float32 equal the float64 rebuild.

    Two ``_walker_batch(seed=0)`` walkers of the fixture. With every native zone
    float64 except ``jastrow_eval`` float32, ``compute_grads_and_laplacian_dtn``
    (``jastrow_grad_lap``), ``grad_J_dtn_wrt_R`` and ``_dtn_value_strain`` read only the
    shared coefficient set, so they must equal the all-float64 native evaluation of an
    independently loaded copy of the same full input elementwise to reduction order.
    ``compute_dtn_value`` must equal the value under ``jastrow_eval`` and
    ``jastrow_grad_lap`` float32 bit for bit and differ from the float64 value. The
    float64 native value and grad/lap must equal the XLA truth at the strict float64
    budget. Values must be finite and live.
    """
    with _full_python_and_restored_native():
        full_policy = precision_policy_snapshot()
        assert set(full_policy.values()) == {"float64"}
        isolated = {**full_policy, _ISOLATED_ZONE: "float32"}
        shared_f32 = {**isolated, "jastrow_grad_lap": "float32"}
        factory = full_source_hamiltonian_factory(lambda: rs_ham(pattern))
        jqmc_rs.configure_native_precision(full_policy)
        ham_ref = factory()
        reference = _native_fields(ham_ref)
        truth_value, truth_grad_lap = _xla_truth(ham_ref, reference["r_up"], reference["r_dn"])
        jqmc_rs.configure_native_precision(isolated)
        actual = _native_fields(factory())
        jqmc_rs.configure_native_precision(shared_f32)
        eval_f32 = _native_fields(factory())
    for other in (actual, eval_f32):
        np.testing.assert_array_equal(other["r_up"], reference["r_up"])
        np.testing.assert_array_equal(other["r_dn"], reference["r_dn"])

    atol, rtol = get_tolerance_min(("jastrow_eval", "jastrow_grad_lap"), "strict")
    np.testing.assert_allclose(reference["value"], truth_value, atol=atol, rtol=rtol, err_msg=f"{pattern}: value vs XLA")
    np.testing.assert_allclose(
        reference["grad_lap"], truth_grad_lap, atol=atol, rtol=rtol, err_msg=f"{pattern}: grad/lap vs XLA"
    )
    for key in ("grad_lap", "pulay", "strain"):
        got, ref = actual[key], reference[key]
        assert got.shape == ref.shape
        _assert_live_and_finite(ref, key)
        np.testing.assert_allclose(
            got,
            ref,
            atol=ATOL_REDUCTION_ORDER,
            rtol=RTOL_REDUCTION_ORDER,
            err_msg=f"{pattern}: native DTN {key} with only {_ISOLATED_ZONE} float32 vs float64 rebuild",
        )
    _assert_live_and_finite(reference["value"], "value")
    np.testing.assert_array_equal(actual["value"], eval_f32["value"], err_msg=f"{pattern}: value left the eval zone")
    assert not np.allclose(actual["value"], reference["value"], atol=ATOL_REDUCTION_ORDER, rtol=RTOL_REDUCTION_ORDER), (
        f"{pattern}: the {_ISOLATED_ZONE} float32 value equals the float64 value"
    )


def _native_carry(ham, moves):
    """Native ``DtnGradLapCache`` seed and the carry after each seeded single-electron hop."""
    dtn = ham.wavefunction_data.jastrow_data.jastrow_dtn_data
    r_up, r_dn = _walkers(ham)
    rows = []
    for w in range(_NW):
        for is_up, index, new_pos in moves[w]:
            seed, hop = jqmc_rs._dtn_grad_lap_carry_probe(dtn, r_up[w], r_dn[w], is_up, index, new_pos)
            rows.append(np.concatenate([np.ravel(np.asarray(x, float)) for x in (*seed, *hop)]))
    return r_up, r_dn, np.asarray(rows)


def _carry_moves(ham, r_up, r_dn):
    """Two seeded single-electron hops per walker (``default_rng(781)``, sigma = 0.4 bohr)."""
    rng = np.random.default_rng(781)
    moves = []
    for w in range(_NW):
        walker = []
        for is_up, r in ((True, r_up[w]), (False, r_dn[w])):
            if r.shape[0]:
                index = int(rng.integers(0, r.shape[0]))
                walker.append((is_up, index, [float(x) for x in r[index] + rng.normal(scale=0.4, size=3)]))
        moves.append(walker)
    return moves


@pytest.mark.light
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
@pytest.mark.parametrize("pattern", _PATTERNS)
def test_dtn_carry_coefficients_follow_the_carry_zone(pattern):
    """XLA and native carried DTN grad/lap coefficients follow only ``jastrow_grad_lap_carry``.

    The carried DTN grad/lap state consumes its coefficients in its own
    ``jastrow_grad_lap_carry`` zone (float64 in both modes), not in the
    ``jastrow_grad_lap`` dtype of the native shared set.
    ``_dtn_carry_coefficient_dtype`` is float64 under the full and mixed policies and
    under ``jastrow_eval``, ``jastrow_ratio`` or ``jastrow_grad_lap`` alone float32.
    Isolation policy: every zone float64 except ``jastrow_eval`` and ``jastrow_grad_lap``
    float32 (Python and native), i.e. the mixed policy of every zone the carry does not
    belong to. Under it the XLA carry seed ``_compute_grads_and_laplacian_Jastrow_DTN_carry``
    and the native carry (``_dtn_grad_lap_carry_probe``: the ``DtnGradLapCache`` seed and
    its state after two seeded single-electron hops per walker) of the two
    ``_walker_batch(seed=0)`` walkers must each equal the same backend's all-float64
    evaluation of an independently loaded copy of the same full input to reduction
    order, and the float64 native seed must equal the XLA one at the strict float64
    ``jastrow_grad_lap_carry`` budget. Values must be finite and live.
    """
    carry = _compute_grads_and_laplacian_Jastrow_DTN_carry
    with _full_python_and_restored_native():
        full_policy = precision_policy_snapshot()
        isolated = {**full_policy, "jastrow_eval": "float32", "jastrow_grad_lap": "float32"}
        carry_dtypes = {}
        for zone in ("jastrow_eval", "jastrow_ratio", "jastrow_grad_lap"):
            with _python_zone_policy({**full_policy, zone: "float32"}):
                carry_dtypes[zone] = _dtn_carry_coefficient_dtype()
        configure("mixed")
        carry_dtypes["mixed"] = _dtn_carry_coefficient_dtype()
        configure("full")
        carry_dtypes["full"] = _dtn_carry_coefficient_dtype()

        factory = full_source_hamiltonian_factory(lambda: rs_ham(pattern))
        jqmc_rs.configure_native_precision(full_policy)
        ham_ref = factory()
        r_up, r_dn = _walkers(ham_ref)
        moves = _carry_moves(ham_ref, r_up, r_dn)
        _, _, native_ref = _native_carry(ham_ref, moves)
        _, xla_ref = _xla_truth(ham_ref, r_up, r_dn, carry)
        jqmc_rs.configure_native_precision(isolated)
        ham = factory()
        got_up, got_dn, native = _native_carry(ham, moves)
        with _python_zone_policy(isolated):
            _, xla = _xla_truth(ham, r_up, r_dn, carry)
    np.testing.assert_array_equal(got_up, r_up)
    np.testing.assert_array_equal(got_dn, r_dn)
    atol, rtol = get_tolerance_min(("jastrow_grad_lap_carry",), "strict")
    n_seed = xla_ref.shape[1]
    seeds_ref = native_ref.reshape(_NW, -1, native_ref.shape[1])[:, 0, :n_seed]
    np.testing.assert_allclose(seeds_ref, xla_ref, atol=atol, rtol=rtol, err_msg=f"{pattern}: native vs XLA float64 carry seed")
    for label, got, ref in (("XLA carry seed", xla, xla_ref), ("native carry", native, native_ref)):
        _assert_live_and_finite(ref, label)
        np.testing.assert_allclose(
            got,
            ref,
            atol=ATOL_REDUCTION_ORDER,
            rtol=RTOL_REDUCTION_ORDER,
            err_msg=f"{pattern}: {label} with jastrow_eval and jastrow_grad_lap float32 vs float64 rebuild",
        )
    assert carry_dtypes == {
        "jastrow_eval": np.float64,
        "jastrow_ratio": np.float64,
        "jastrow_grad_lap": np.float64,
        "mixed": np.float64,
        "full": np.float64,
    }, "carry coefficient precision must follow only jastrow_grad_lap_carry"

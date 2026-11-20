"""Issue #766: native DTN single-move ratios read coefficients inflated in the jastrow_ratio zone.

The precision policy (``jqmc/_precision.py``) assigns the single-move Jastrow
ratio ``exp(J(R') - J(R))`` to ``jastrow_ratio`` (float64 in both modes). The
native DTN (``JastrowDtnData``) carried ONE set of inflated coefficient tensors
(``_dtn_inflate_alpha|beta|gamma``), inflated in float32 whenever any of
``jastrow_eval`` / ``jastrow_grad_lap`` / ``jastrow_ratio`` is float32. In
mixed (``jastrow_eval`` / ``jastrow_grad_lap`` float32) every native DTN ratio
kernel therefore evaluated J(R')-J(R) with float32-rounded coefficients: the
streaming ratio of ``DtnRatioState`` (``MeshFastStateC::jas_ratio`` and the
real fast state), the grouped / candidate-table ratios, the mesh ratio and the
direct ``compute_dtn_ratio``, on the molecular, real-Gamma periodic and
complex-twist routes alike. XLA's DTN ratio kernels
(``_delta_J_DTN_one_electron_moved``, ``_delta_J_DTN_from_ratio_state``, the
``J_DTN_candidate_mesh_state`` init / advance) inflate in the ``jastrow_ratio``
dtype and are not affected.

The DTN parameter leaves themselves are stored in float64 in both modes (issue
#781; before it, a Hamiltonian built in mixed carried float32 ``*_indep``
leaves), a data-precision choice shared by both backends, not a native zone.
Both detectors build the actual and the reference case from independently
loaded copies of the same serialized full input under the full Python policy and
vary only the NATIVE policy, as the #757 detectors do:

* ``test_native_dtn_single_move_ratio_matches_full_rebuild``: the native DTN
  ratio ``jqmc_rs.compute_dtn_ratio`` on the molecular
  (``mol_ecp_cart_jdtn``), real-Gamma periodic (``pbc_gamma_sphepgto_jdtn``)
  and complex-twist (``twist_sphepgto_jdtn``) DTN fixtures under the active
  mode's whole native policy. The kernel reads only ``jastrow_ratio``, so it
  must equal the all-float64 native evaluation to reduction order; that
  reference must equal the XLA truth ``exp(J_DTN(R') - J_DTN(R))`` at the
  strict float64 budget.
* ``test_native_fast_lrdmc_dtn_ratio_matches_full_rebuild``: the fast LRDMC
  branching sum ``b_L`` (``_gfmc_lrdmc_ecp(..., fast=True)``
  and ``_gfmc_lrdmc_ecp_complex(..., fast=True)``), whose discretized-kinetic
  and ECP-mesh DTN factors come from the fast states' streaming /
  candidate-table ratios, with the all-float64 native policy except
  ``jastrow_eval`` and ``jastrow_grad_lap`` at the active mode's dtype (the two
  zones that made the shared coefficient set float32 in mixed). ``b_L`` reads
  neither zone, so it must equal the all-float64 evaluation to reduction order,
  Re and Im. (The LRDMC ``e_L`` is not used: its continuum kinetic part reads
  ``jastrow_grad_lap``.)

Coefficients read in the wrong zone fail both detectors in mixed with a
float32-coefficient error; full runs are same-case consistency checks.
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

import jqmc_rs
from jqmc._precision import configure, get_tolerance_min, is_mixed_precision_enabled, precision_policy_snapshot
from jqmc._setting import ATOL_REDUCTION_ORDER, RTOL_REDUCTION_ORDER
from jqmc.jastrow_factor import compute_Jastrow_DTN
from tests.test_rs_kernel import test_batch_mesh_per_walker as per_walker
from tests.test_rs_kernel._native_precision_fidelity import full_source_hamiltonian_factory
from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_rs_kernel.test_mcmc_observables import _is_complex, _walker_batch

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.with_chdir]

_N_MOVES = 4  # candidate positions per electron
_STEP = 0.5  # bohr, standard deviation of the seeded move displacement
_MOVE_SEED = 766
_ALAT = 0.3
_NN = 1
# Zones whose float32 dtype made the shared native DTN coefficient set float32 in mixed.
_COEFFICIENT_ZONES = ("jastrow_eval", "jastrow_grad_lap")

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


def _actual_and_reference(pattern, evaluate, actual_policy_of):
    """``(actual, reference)``: ``evaluate(ham)`` on two loads of the full input, two native policies."""
    mode_policy = precision_policy_snapshot()
    with _full_python_and_restored_native():
        full_policy = precision_policy_snapshot()
        assert set(full_policy.values()) == {"float64"}
        actual_policy = actual_policy_of(mode_policy, full_policy)
        factory = full_source_hamiltonian_factory(lambda: rs_ham(pattern))
        jqmc_rs.configure_native_precision(full_policy)
        reference = evaluate(factory())
        jqmc_rs.configure_native_precision(actual_policy)
        actual = evaluate(factory())
    np.testing.assert_array_equal(actual["r_up"], reference["r_up"])
    np.testing.assert_array_equal(actual["r_dn"], reference["r_dn"])
    return actual, reference


def _moves(ham):
    """``(r_up, r_dn, [(is_up, idx, new_pos)])``: one seeded walker and seeded single-electron moves."""
    r_up, r_dn = _walker_batch(ham, nw=1, seed=0)
    r_up, r_dn = np.ascontiguousarray(r_up[0]), np.ascontiguousarray(r_dn[0])
    rng = np.random.default_rng(_MOVE_SEED)
    moves = []
    for is_up, block in ((True, r_up), (False, r_dn)):
        for idx in range(block.shape[0]):
            for _ in range(_N_MOVES):
                moves.append((is_up, idx, block[idx] + _STEP * rng.standard_normal(3)))
    return r_up, r_dn, moves


def _dtn_ratios(ham):
    """Native ``compute_dtn_ratio`` and the XLA truth ``exp(J(R') - J(R))`` of every move."""
    dtn = ham.wavefunction_data.jastrow_data.jastrow_dtn_data
    assert dtn is not None and dtn.use_u and dtn.use_chi and dtn.use_f
    r_up, r_dn, moves = _moves(ham)
    j_base = float(compute_Jastrow_DTN(dtn, jnp.asarray(r_up), jnp.asarray(r_dn)))
    native, truth = [], []
    for is_up, idx, new_pos in moves:
        native.append(float(jqmc_rs.compute_dtn_ratio(dtn, r_up, r_dn, is_up, idx, [float(x) for x in new_pos])))
        new_up, new_dn = r_up.copy(), r_dn.copy()
        (new_up if is_up else new_dn)[idx] = new_pos
        truth.append(np.exp(float(compute_Jastrow_DTN(dtn, jnp.asarray(new_up), jnp.asarray(new_dn))) - j_base))
    return {"r_up": r_up, "r_dn": r_dn, "native": np.asarray(native), "truth": np.asarray(truth)}


@pytest.mark.light
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
@pytest.mark.parametrize("pattern", _PATTERNS)
def test_native_dtn_single_move_ratio_matches_full_rebuild(pattern):
    """Native DTN single-move ratios under the mode's native policy equal the float64 rebuild.

    One ``_walker_batch(seed=0)`` walker of the fixture and four seeded
    (``default_rng(766)``, sigma = 0.5 bohr) single-electron moves of every up
    and down electron. ``jqmc_rs.compute_dtn_ratio`` reads only the
    ``jastrow_ratio`` zone (float64 in both modes), so under the active mode's
    whole native policy it must equal the all-float64 native evaluation of an
    independently loaded copy of the same full input elementwise to reduction
    order. The reference must equal the XLA truth
    ``exp(compute_Jastrow_DTN(R') - compute_Jastrow_DTN(R))`` at the strict
    float64 budget. Values must be finite and live.
    """
    assert precision_policy_snapshot()["jastrow_ratio"] == "float64", "reduction-order budget needs float64 jastrow_ratio"
    actual, reference = _actual_and_reference(pattern, _dtn_ratios, lambda mode, full: dict(mode))
    got, ref = actual["native"], reference["native"]
    assert got.shape == ref.shape and got.size > 0
    assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref))
    atol, rtol = get_tolerance_min(("jastrow_eval", "jastrow_ratio"), "strict")
    assert not np.allclose(ref, 1.0, atol=atol, rtol=rtol), "inactive DTN ratio"
    np.testing.assert_allclose(ref, reference["truth"], atol=atol, rtol=rtol, err_msg=f"{pattern}: native full vs XLA truth")
    np.testing.assert_allclose(
        got,
        ref,
        atol=ATOL_REDUCTION_ORDER,
        rtol=RTOL_REDUCTION_ORDER,
        err_msg=f"{pattern}: native DTN ratio vs float64 rebuild",
    )


def _fast_lrdmc(ham):
    """Fast LRDMC ``b_L`` of the three walkers with the per-walker production rotations."""
    assert bool(ham.coulomb_potential_data.ecp_flag)
    assert ham.wavefunction_data.jastrow_data.jastrow_dtn_data is not None
    assert ham.wavefunction_data.jastrow_data.jastrow_three_body_data is None
    wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
    complex_route = _is_complex(ham)
    entry = jqmc_rs._gfmc_lrdmc_ecp_complex if complex_route else jqmc_rs._gfmc_lrdmc_ecp
    nw = per_walker._NW
    r_up, r_dn = _walker_batch(ham, nw=nw, seed=0)
    r_up, r_dn = np.ascontiguousarray(r_up), np.ascontiguousarray(r_dn)
    rts = per_walker._distinct_rotations(nw)
    b_l = []
    for w in range(nw):
        out = entry(
            wf.geminal_data,
            wf.jastrow_data,
            cp,
            r_up[w],
            r_dn[w],
            _ALAT,
            rts[w],
            per_walker._W,
            per_walker._GP,
            _NN,
            False,
            True,
        )
        b_l.append(complex(out[2], out[3]) if complex_route else complex(out[1]))
    return {"r_up": r_up, "r_dn": r_dn, "lrdmc_b_L": np.asarray(b_l)}


@pytest.mark.light
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
@pytest.mark.parametrize("pattern", _PATTERNS)
def test_native_fast_lrdmc_dtn_ratio_matches_full_rebuild(pattern):
    """Fast LRDMC b_L with the mode's Jastrow-coefficient zones equals the float64 rebuild.

    The fixture's three ``_walker_batch(seed=0)`` walkers and the three production
    per-walker rotations of ``test_batch_mesh_per_walker`` on the default Nv mesh,
    NN = 1, ``alat = 0.3``, through ``_gfmc_lrdmc_ecp(..., fast=True)`` (molecular,
    real Gamma) or ``_gfmc_lrdmc_ecp_complex(..., fast=True)`` (twist). The actual
    native policy is all float64 except ``jastrow_eval`` and ``jastrow_grad_lap`` at
    the active mode's dtype. The discretized-kinetic and ECP-mesh Jastrow factors of
    ``b_L`` are ``jastrow_ratio`` quantities and it reads no whole Jastrow value or
    gradient, so it must equal the all-float64 native evaluation of an independently
    loaded copy of the same full input elementwise to reduction order, Re and Im.
    Values must be finite and live.
    """
    actual, reference = _actual_and_reference(
        pattern, _fast_lrdmc, lambda mode, full: {**full, **{zone: mode[zone] for zone in _COEFFICIENT_ZONES}}
    )
    for key in ("lrdmc_b_L",):
        got, ref = np.asarray(actual[key]), np.asarray(reference[key])
        assert got.shape == ref.shape == (per_walker._NW,)
        assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref)), f"{key}: non-finite"
        assert np.max(np.abs(ref)) > 1.0, f"{key}: inactive"
        for part, f in (("Re", np.real), ("Im", np.imag)):
            np.testing.assert_allclose(
                f(got), f(ref), atol=ATOL_REDUCTION_ORDER, rtol=RTOL_REDUCTION_ORDER, err_msg=f"{pattern} {key} ({part})"
            )

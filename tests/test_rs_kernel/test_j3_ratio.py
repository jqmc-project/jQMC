"""J3 (three-body) rank-1 ratio + streaming state -- SELF-CONTAINED.

This verifies the J3 streaming *algebra* against a pure-numpy
J3 reference on SYNTHETIC orbital matrices -- no jqmc dependency. The numpy J3
value formula here is the same one the Rust `compute_jastrow_three_body_value`
was validated against vs jqmc `compute_Jastrow_three_body` in
test_jastrow.py::test_jastrow_three_body_value_matches_jax.

The test bodies are pure numpy (no h5 fixture, no jqmc import); tolerances come
from the zone-derived helper.
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

pytestmark = pytest.mark.with_chdir

from jqmc._precision import get_tolerance_xla_rust_consistency

# Pure-numpy J3 streaming algebra; the closest owning-module zone is the
# Jastrow ratio-update zone (jastrow_factor.py rank-1 update path).
_ZONES = ("jastrow_ratio",)


def _j3_value(jm, aos_up, aos_dn):
    """Pure-numpy J3 bilinear value (matches compute_Jastrow_three_body)."""
    j3 = jm[:, :-1]
    j1 = jm[:, -1]
    n_up = aos_up.shape[1]
    n_dn = aos_dn.shape[1]
    val = 0.0
    for i in range(n_up):
        val += j1 @ aos_up[:, i]
    for i in range(n_dn):
        val += j1 @ aos_dn[:, i]
    for i in range(n_up):
        for j in range(i + 1, n_up):
            val += aos_up[:, i] @ j3 @ aos_up[:, j]
    for i in range(n_dn):
        for j in range(i + 1, n_dn):
            val += aos_dn[:, i] @ j3 @ aos_dn[:, j]
    for i in range(n_up):
        for j in range(n_dn):
            val += aos_up[:, i] @ j3 @ aos_dn[:, j]
    return val


def _synthetic(seed, n_orb, n_up, n_dn):
    rng = np.random.default_rng(seed)
    jm = (rng.random((n_orb, n_orb + 1)) - 0.5).astype(np.float64)
    # The J3 coupling block is SYMMETRIC in production (bosonic Jastrow; jqmc's
    # Jastrow_three_body_data.sanity_check asserts it, and the native ctx build
    # strict-raises otherwise). The streaming kernels exploit j3^T*x == j3*x, so
    # the contract these entrypoints implement is symmetric-block only -- feed a
    # symmetric block here (the j1 column, index n_orb, stays arbitrary).
    j3 = jm[:, :-1]
    jm[:, :-1] = 0.5 * (j3 + j3.T)
    aos_up = (rng.random((n_orb, n_up)) - 0.5).astype(np.float64)
    aos_dn = (rng.random((n_orb, n_dn)) - 0.5).astype(np.float64)
    return jm, aos_up, aos_dn


SHAPES = [(5, 4, 3), (8, 6, 6), (3, 2, 1), (6, 5, 0)]

# (shape, is_up) pairs with the spinless side EXCLUDED AT COLLECTION: moving a
# dn electron of an n_dn == 0 system is not a case, it is a non-statement, and
# a parametrization that skips in every session leaves a module-receipt pin
# that can never close.
_MOVE_CASES = [(shape, is_up) for shape in SHAPES for is_up in (True, False) if (shape[1] if is_up else shape[2]) > 0]


@pytest.mark.light
@pytest.mark.parametrize("n_orb,n_up,n_dn", SHAPES)
def test_j3_init_state_matches_numpy(n_orb, n_up, n_dn):
    """J3 init state matches numpy."""
    jm, aos_up, aos_dn = _synthetic(0, n_orb, n_up, n_dn)
    j3 = jm[:, :-1]
    fields = jqmc_rs._j3_init_state(jm, aos_up, aos_dn)
    # Symmetric-collapsed layout: (aos_up, aos_dn, w_up=j3@aos_up,
    # w_dn=j3@aos_dn, cross_up=rowsum(w_up), cross_dn=rowsum(w_dn)). Under the
    # symmetric block j3^T@aos == j3@aos, so there are no separate _t_ tables.
    ref = [
        aos_up,
        aos_dn,
        j3 @ aos_up,
        j3 @ aos_dn,
        (j3 @ aos_up).sum(axis=1),
        (j3 @ aos_dn).sum(axis=1),
    ]
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="strict")
    for field, index in (("ao_up", 0), ("w_up", 2), ("cross_up", 4), ("cross_dn", 5)):
        arr, r = fields[index], ref[index]
        assert np.allclose(np.asarray(arr), r, rtol=rtol, atol=atol)
    for field, index in (("ao_dn", 1), ("w_dn", 3)):
        arr, r = fields[index], ref[index]
        assert np.allclose(np.asarray(arr), r, rtol=rtol, atol=atol)


@pytest.mark.light
@pytest.mark.parametrize("shape,is_up", _MOVE_CASES)
@pytest.mark.parametrize("seed", [1, 6])
def test_j3_ratio_matches_value_diff(shape, is_up, seed):
    """J3 ratio matches value diff."""
    n_orb, n_up, n_dn = shape
    jm, aos_up, aos_dn = _synthetic(seed, n_orb, n_up, n_dn)
    rng = np.random.default_rng(seed + 100)
    idx = 0
    aos_new_col = (rng.random(n_orb) - 0.5).astype(np.float64)

    # ground truth: exp(J3(new) - J3(old)) via the full bilinear value
    aos_up_new, aos_dn_new = aos_up.copy(), aos_dn.copy()
    if is_up:
        aos_up_new[:, idx] = aos_new_col
    else:
        aos_dn_new[:, idx] = aos_new_col
    truth = float(np.exp(_j3_value(jm, aos_up_new, aos_dn_new) - _j3_value(jm, aos_up, aos_dn)))

    got = jqmc_rs._j3_ratio(jm, aos_up, aos_dn, is_up, idx, aos_new_col)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="strict")
    assert np.isclose(got, truth, rtol=rtol, atol=atol), (got, truth)


@pytest.mark.light
@pytest.mark.parametrize("shape,is_up", _MOVE_CASES)
def test_j3_advance_equals_reinit(shape, is_up):
    """J3 advance equals reinit."""
    n_orb, n_up, n_dn = shape
    jm, aos_up, aos_dn = _synthetic(2, n_orb, n_up, n_dn)
    rng = np.random.default_rng(202)
    idx = 0
    aos_new_col = (rng.random(n_orb) - 0.5).astype(np.float64)

    advanced = jqmc_rs._j3_advance_state(jm, aos_up, aos_dn, is_up, idx, aos_new_col)
    # re-init at the moved config must reproduce the advanced state exactly
    aos_up_new, aos_dn_new = aos_up.copy(), aos_dn.copy()
    if is_up:
        aos_up_new[:, idx] = aos_new_col
    else:
        aos_dn_new[:, idx] = aos_new_col
    reinit = jqmc_rs._j3_init_state(jm, aos_up_new, aos_dn_new)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="strict")
    for field, index in (("ao_up", 0), ("w_up", 2), ("cross_up", 4), ("cross_dn", 5)):
        a, b = advanced[index], reinit[index]
        assert np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)
    for field, index in (("ao_dn", 1), ("w_dn", 3)):
        a, b = advanced[index], reinit[index]
        assert np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=atol)

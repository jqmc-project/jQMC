"""Layer-1: periodic (pgto) cart-GTO AO derivative kernel (Rust pgto_cart_deriv =
kernel-rank deriv + image fold + wrap) vs jqmc compute_AOs* on the cart pgto leaf.
REAL Gamma-point only (Rust is all-f64; complex twist unsupported).

Fixture: the committed lithium pattern via
``tests.test_rs_kernel._rs_fixtures.rs_ham``.

Precision regime: ``jqmc_rs.pgto_cart_deriv`` is fp64
in both precision modes, so every XLA reference it is judged against is fp64 as
well. The ``compute_AOs*`` references are evaluated with every precision zone at
fp64 (``_fp64_reference_policy``; a no-op in full mode, so the full-mode
references are numerically unchanged), and the AO value is additionally checked
against the fp64-by-construction XLA twin ``jqmc.derivatives.pgto_deriv.pgto_value``.
No fp32 ``ao_eval`` output enters these comparisons.
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

import numpy as np
import pytest

from jqmc.atomic_orbital import (
    compute_AOs,
    compute_AOs_grad,
    compute_AOs_laplacian,
    _aos_sphe_to_cart,
)

import jqmc_rs

from jqmc._precision import configure, get_tolerance_xla_rust_consistency, is_mixed_precision_enabled
from jqmc.derivatives.pgto_deriv import pgto_value
from jqmc._setting import ATOL_RELATIVE_COMPARISON_SCALE_FLOOR

from tests.test_rs_kernel._rs_fixtures import rs_ham

# Every rs test declares its cost tier (light/heavy, per test) and CWD policy
# (with_chdir here -- autouse _isolate_cwd chdirs into a per-test tmp_path for
# xdist safety; committed fixtures load via absolute paths so cwd is irrelevant).
pytestmark = pytest.mark.with_chdir

# Periodic pgto AO eval tolerance (same "medium" ao_eval zone the far-image
# cutoff comparison in test_atomic_orbital_pgto.py uses; this is a single-
# configuration precision check, not accumulated over many electrons, but
# The dedicated XLA-Rust consistency tier gives the backend-parity budget.
_ATOL, _RTOL = get_tolerance_xla_rust_consistency(("ao_eval",))


@contextlib.contextmanager
def _fp64_reference_policy():
    """Evaluate the enclosed XLA reference with every precision zone at fp64.

    The Rust kernel under test is fp64 in both modes; its XLA reference must be
    in the same regime. In mixed mode the full policy is activated for the
    enclosed block and the mixed policy restored afterwards; in full mode this
    is a no-op.
    """
    was_mixed = is_mixed_precision_enabled()
    if was_mixed:
        configure("full")
    try:
        yield
    finally:
        if was_mixed:
            configure("mixed")


def _close(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = np.abs(a - b).max()
    return d <= _ATOL or d <= _RTOL * (np.abs(b).max() + ATOL_RELATIVE_COMPARISON_SCALE_FLOOR)


@pytest.mark.light
def test_pgto_deriv():
    """Pgto deriv."""
    ham = rs_ham("pbc_gamma_sphepgto_jnone")
    sphe = ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data
    cart, _T = _aos_sphe_to_cart(sphe)
    cell = np.asarray(ham.structure_data.cell, float)
    gem = ham.wavefunction_data.geminal_data
    R = np.asarray(ham.structure_data.positions, float)
    rng = np.random.default_rng(1)
    r = R[np.arange(gem.num_electron_up) % len(R)] + rng.normal(scale=0.6, size=(gem.num_electron_up, 3))

    def D(a, b, c):
        return np.asarray(jqmc_rs.pgto_cart_deriv(cart, r, cell, a, b, c))

    # fp64 references for the fp64 Rust kernel: production XLA AOs with
    # every zone at fp64, plus the fp64-by-construction derivative-kernel twin.
    with _fp64_reference_policy():
        ao_ref = np.asarray(compute_AOs(cart, r))
        grad_ref = np.stack([np.asarray(x) for x in compute_AOs_grad(cart, r)], 0)
        lap_ref = np.asarray(compute_AOs_laplacian(cart, r))
    value_twin = np.asarray(pgto_value(cart, r))
    assert ao_ref.dtype == np.float64 and value_twin.dtype == np.float64

    assert _close(D(0, 0, 0), ao_ref)
    assert _close(D(0, 0, 0), value_twin)
    g = np.stack([D(1, 0, 0), D(0, 1, 0), D(0, 0, 1)], 0)
    assert _close(g, grad_ref)
    lap = D(2, 0, 0) + D(0, 2, 0) + D(0, 0, 2)
    assert _close(lap, lap_ref)

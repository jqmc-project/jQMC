"""Axis-wise B-spline stencil contraction off the CPU backend (issue #527).

With the GPU route selected, the real ``AOs_bspline_data._eval_core`` and the complex
   ``AOs_bspline_complex_data._eval_u_core`` return the same value, gradient and Laplacian as the
   ``einsum("bpqr,p,q,r->b")`` contractions, under the ao_eval / ao_grad_lap strict zone tolerances.
   The cases are cubic and quintic, Gamma and a general twist, on a skewed cell, with points outside
   the home cell.
The values are the only claim: the einsum route is the reference.
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

import jqmc.atomic_orbital as ao_mod
from jqmc._precision import get_tolerance
from jqmc.atomic_orbital import AOs_bspline_complex_data, AOs_bspline_data
from tests.conftest import CELL_FIXTURES, make_pbc_structure

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_SPLINE_SHAPE = (7, 8, 9)
_NUM_BAND = 3
_NUM_POINTS = 11
_TWISTS = {"gamma": (0.0, 0.0, 0.0), "general-k": (0.27, -0.1, 0.13)}


def _structure():
    return make_pbc_structure(CELL_FIXTURES["triclinic_skewed"])


def _leaf(kind, order, twist):
    if kind == "real":
        rng = np.random.default_rng(527 + order)
        return AOs_bspline_data.build(_structure(), rng.normal(size=(_NUM_BAND, *_SPLINE_SHAPE)), _SPLINE_SHAPE, order)
    rng = np.random.default_rng(5270 + order)
    coefficients = rng.normal(size=(_NUM_BAND, *_SPLINE_SHAPE)) + 1j * rng.normal(size=(_NUM_BAND, *_SPLINE_SHAPE))
    return AOs_bspline_complex_data.build(_structure(), coefficients, _SPLINE_SHAPE, order, k_point=_TWISTS[twist])


def _points(leaf):
    cell = np.asarray(leaf.structure_data._cell_jnp)
    frac = np.random.default_rng(11).uniform(-0.5, 1.5, size=(_NUM_POINTS, 3))
    return jnp.asarray(frac @ cell)


_CASES = [pytest.param("real", order, None, id=f"real-order{order}") for order in (3, 5)] + [
    pytest.param("complex", order, twist, id=f"complex-order{order}-{twist}") for order in (3, 5) for twist in _TWISTS
]


def _core(leaf):
    return leaf._eval_core if isinstance(leaf, AOs_bspline_data) else leaf._eval_u_core


@pytest.mark.parametrize(("kind", "order", "twist"), _CASES)
def test_axiswise_stencil_matches_einsum_contractions(monkeypatch, kind, order, twist):
    """The axis-wise route reproduces value, gradient and Laplacian of the einsum route."""
    leaf = _leaf(kind, order, twist)
    r = _points(leaf)
    core = _core(leaf)
    results = {}
    for route in (False, True):
        monkeypatch.setattr(ao_mod, "_bspline_axiswise_stencil", lambda route=route: route)
        results[route] = {
            "ao_eval": jax.jit(lambda x: core(x, "ao_eval")[0])(r),
            "ao_grad_lap": jax.jit(lambda x: core(x, "ao_grad_lap"))(r),
        }
    checks = [("value", "ao_eval", results[True]["ao_eval"], results[False]["ao_eval"])]
    for name, got, ref in zip(("value", "gx", "gy", "gz", "lap"), results[True]["ao_grad_lap"], results[False]["ao_grad_lap"]):
        checks.append((name, "ao_grad_lap", got, ref))
    for name, zone, got, ref in checks:
        got, ref = np.asarray(got), np.asarray(ref)
        assert got.shape == ref.shape == (_NUM_BAND, _NUM_POINTS), (name, zone)
        assert got.dtype == ref.dtype, (name, zone)
        assert np.all(np.isfinite(got)) and np.max(np.abs(ref)) > 0.0, (name, zone)
        atol, rtol = get_tolerance(zone, "strict")
        np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, err_msg=f"{name} ({zone})")

"""Closed-form small-displacement checks of the J1 / J2 kernels.

A hydrogen nucleus (or the up electron, for J2) sits at the large common
coordinate ``(5, 5, 5)`` Bohr and the partner electron at ``+h`` along x,
``h in {2**-16, 2**-26, 2**-38}``. In each precision mode the production
value quotient ``J(h) / h``, the gradient and ``h * Laplacian`` are compared
with the independent closed forms of
``tests/test_jqmc_kernel/_jastrow_small_displacement_reference.py`` at the
existing jastrow strict zone tolerances; full and mixed are judged
separately, never against each other. The closed forms are
evaluated in the active zone dtypes (``jastrow_eval`` for the value,
``jastrow_grad_lap`` for gradient / Laplacian; fp32 in mixed, fp64 in full)
from the fp64-reconstructed separation ``h`` cast to that dtype, so mixed is
judged within its own precision.

What this pins:

* ``r - R`` (and ``r_i - r_j``) must be formed from the caller-supplied fp64
  walker coordinates before the downcast to the jastrow zone. Rounding the
  absolute coordinates first (``r.astype(dtype) - R.astype(dtype)``) makes
  ``5 + 2**-26`` equal to ``5`` in fp32, so the mixed lane collapses the
  separation to zero.
* The exp forms must not evaluate ``1 - exp(-x)`` by subtraction: at
  ``x ~ h`` it cancels all digits in fp32 (quotient 0 instead of -1 / 0.5)
  and loses ~1e-6 relative in fp64 at ``h = 2**-38``.
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

import numpy as np
import pytest

from jqmc._precision import get_tolerance, get_tolerance_min
from jqmc.jastrow_factor import (
    Jastrow_one_body_data,
    Jastrow_two_body_data,
    compute_grads_and_laplacian_Jastrow_one_body,
    compute_grads_and_laplacian_Jastrow_two_body,
    compute_Jastrow_one_body,
    compute_Jastrow_two_body,
)
from jqmc.structure import Structure_data
from tests.conftest import CELL_FIXTURES, make_pbc_structure
from tests.test_jqmc_kernel._jastrow_small_displacement_reference import (
    J1_PARAMS,
    J2_PARAMS,
    NUCLEUS,
    SMALL_DISPLACEMENTS,
    closed_form,
)

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

_GEOMETRIES = ("molecular", "cubic_10bohr")
_KINDS = ("J1", "J2")
_FORMS = ("pade", "exp")


def _structure(geometry: str) -> Structure_data:
    labels = {"atomic_numbers": (1,), "element_symbols": ("H",), "atomic_labels": ("H1",)}
    if geometry == "molecular":
        return Structure_data(pbc_flag=False, positions=NUCLEUS[None, :].copy(), **labels)
    return make_pbc_structure(CELL_FIXTURES[geometry], positions=NUCLEUS[None, :].copy(), **labels)


def _data_and_electrons(kind: str, form: str, a: float, h: float, structure: Structure_data):
    """Return the Jastrow data, (r_up, r_dn) and the index/spin of the ``+h`` electron."""
    moved = (NUCLEUS + np.array([h, 0.0, 0.0]))[None, :]
    if kind == "J1":
        data = Jastrow_one_body_data(jastrow_1b_param=a, jastrow_1b_type=form, structure_data=structure, core_electrons=(0.0,))
        return data, moved, np.zeros((0, 3), dtype=np.float64)
    data = Jastrow_two_body_data(jastrow_2b_param=a, jastrow_2b_type=form, structure_data=structure)
    return data, NUCLEUS[None, :].copy(), moved


def _params(kind: str) -> tuple[float, ...]:
    return J1_PARAMS if kind == "J1" else J2_PARAMS


def _labelled(labels: list[str], got: list[float]) -> str:
    return "; ".join(f"{label}={value:.12g}" for label, value in zip(labels, got, strict=True))


@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("geometry", _GEOMETRIES)
def test_jastrow_value_small_displacement_matches_closed_form(geometry, kind, form):
    """``J(h) / h`` of production J1/J2 equals the jastrow_eval-dtype closed form at jastrow_eval strict."""
    atol, rtol = get_tolerance("jastrow_eval", "strict")
    structure = _structure(geometry)
    evaluate = compute_Jastrow_one_body if kind == "J1" else compute_Jastrow_two_body
    labels, value_over_h, closed_value_over_h = [], [], []
    for a in _params(kind):
        for h in SMALL_DISPLACEMENTS:
            data, r_up, r_dn = _data_and_electrons(kind, form, a, h, structure)
            labels.append(f"{kind} {form} a={a} h=2^{np.log2(h):.0f} J/h")
            value_over_h.append(float(evaluate(data, r_up, r_dn)) / h)
            closed_value_over_h.append(closed_form(kind, form, a, h, geometry != "molecular")["value_over_h"])
    assert np.all(np.isfinite(value_over_h)), _labelled(labels, value_over_h)
    np.testing.assert_allclose(value_over_h, closed_value_over_h, atol=atol, rtol=rtol, err_msg=_labelled(labels, value_over_h))


@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("geometry", _GEOMETRIES)
def test_jastrow_grad_lap_small_displacement_matches_closed_form(geometry, kind, form):
    """Production J1/J2 gradient and ``h * Laplacian`` equal the jastrow_grad_lap-dtype closed forms at strict."""
    atol, rtol = get_tolerance_min(["jastrow_grad_lap"], "strict")
    structure = _structure(geometry)
    evaluate = compute_grads_and_laplacian_Jastrow_one_body if kind == "J1" else compute_grads_and_laplacian_Jastrow_two_body
    labels, grad_lap, closed_grad_lap = [], [], []

    def record(label: str, got: float, want: float) -> None:
        labels.append(label)
        grad_lap.append(got)
        closed_grad_lap.append(want)

    for a in _params(kind):
        for h in SMALL_DISPLACEMENTS:
            data, r_up, r_dn = _data_and_electrons(kind, form, a, h, structure)
            want = closed_form(kind, form, a, h, geometry != "molecular")
            grad_up, grad_dn, lap_up, lap_dn = (np.asarray(x, dtype=np.float64) for x in evaluate(data, r_up, r_dn))
            tag = f"{kind} {form} a={a} h=2^{np.log2(h):.0f}"
            # The +h electron: the J1 up electron, the J2 down electron.
            grad_moved, lap_moved = (grad_up[0], lap_up[0]) if kind == "J1" else (grad_dn[0], lap_dn[0])
            record(f"{tag} grad_x", float(grad_moved[0]), want["grad_x"])
            record(f"{tag} grad_y", float(grad_moved[1]), 0.0)
            record(f"{tag} grad_z", float(grad_moved[2]), 0.0)
            record(f"{tag} h*lap", float(lap_moved) * h, want["lap_times_h"])
            if kind == "J2":
                # Pair symmetry: the partner sees the negated gradient and the same Laplacian.
                record(f"{tag} partner grad_x", float(grad_up[0, 0]), -want["grad_x"])
                record(f"{tag} partner h*lap", float(lap_up[0]) * h, want["lap_times_h"])
    assert np.all(np.isfinite(grad_lap)), _labelled(labels, grad_lap)
    np.testing.assert_allclose(grad_lap, closed_grad_lap, atol=atol, rtol=rtol, err_msg=_labelled(labels, grad_lap))

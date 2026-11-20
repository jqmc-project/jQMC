"""Native f32 J1 / J2 kernels must form ``r - R`` before the downcast.

Fixture: the nucleus (J1) or the up electron (J2) at the fp32-unrepresentable
anchor ``(5.1, 5.2, 5.3)`` Bohr, the moved electron at ``+h`` along x,
``h in {2**-16, 2**-26, 2**-38}``, molecular and cubic periodic, Pade and exp.
The closed forms of ``tests/test_jqmc_kernel/_jastrow_small_displacement_reference.py``
are evaluated at the exact fp64 separation ``fl(fl(5.1 + h) - 5.1)``
(``_jastrow_param_derivative_reference.offgrid_separation``), cast to the
active zone dtype (``jastrow_eval`` value, ``jastrow_grad_lap`` gradient /
Laplacian; fp32 in mixed, fp64 in full): the reference is evaluated in the zone dtype.

The XLA kernels form the difference from the fp64 coordinates and downcast
afterwards (pinned for J1 by
``tests/test_jqmc_kernel/test_jastrow_offgrid_nucleus.py``
and for J2 by the small-displacement nodes). The native mixed-lane
kernels ``compute_jastrow_one_body_f32`` (value), ``j1_grad_lap_one_spin_f32``
and ``j2_grad_lap_f32`` (gradient / Laplacian) must likewise not round the nuclear
(or both electron) coordinates to f32 first, which would move the separation by up to
2.4e-7 Bohr (1.6% of ``h = 2**-16``, everything at ``2**-26``). Full
precision runs the f64 kernels; the nodes are the fp64 guard there.
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
from jqmc._precision import get_tolerance, get_tolerance_min
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_two_body_data
from jqmc.structure import Structure_data
from tests.conftest import CELL_FIXTURES, make_pbc_structure
from tests.test_jqmc_kernel._jastrow_param_derivative_reference import OFFGRID_ANCHOR, offgrid_separation
from tests.test_jqmc_kernel._jastrow_small_displacement_reference import (
    J1_PARAMS,
    J2_PARAMS,
    SMALL_DISPLACEMENTS,
    closed_form,
)

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

_EMPTY = np.zeros((0, 3), dtype=np.float64)


def _structure(geometry: str) -> Structure_data:
    labels = {"atomic_numbers": (1,), "element_symbols": ("H",), "atomic_labels": ("H1",)}
    if geometry == "molecular":
        return Structure_data(pbc_flag=False, positions=OFFGRID_ANCHOR[None, :].copy(), **labels)
    return make_pbc_structure(CELL_FIXTURES[geometry], positions=OFFGRID_ANCHOR[None, :].copy(), **labels)


def _j1(form: str, a: float, structure: Structure_data) -> Jastrow_one_body_data:
    return Jastrow_one_body_data(jastrow_1b_param=a, jastrow_1b_type=form, structure_data=structure, core_electrons=(0.0,))


def _labelled(labels: list[str], got: list[float]) -> str:
    return "; ".join(f"{label}={value:.12g}" for label, value in zip(labels, got, strict=True))


@pytest.mark.parametrize("form", ["pade", "exp"])
@pytest.mark.parametrize("geometry", ["molecular", "cubic_10bohr"])
def test_rust_jastrow_one_body_value_offgrid_nucleus_matches_closed_form(geometry, form):
    """Native ``J1(h) / h`` at an fp32-unrepresentable nucleus equals the closed form at jastrow_eval strict."""
    atol, rtol = get_tolerance("jastrow_eval", "strict")
    structure = _structure(geometry)
    labels, value_over_h, closed_value_over_h = [], [], []
    for a in J1_PARAMS:
        for h in SMALL_DISPLACEMENTS:
            sep = offgrid_separation(h)
            moved = (OFFGRID_ANCHOR + np.array([h, 0.0, 0.0]))[None, :]
            labels.append(f"J1 {form} a={a} h=2^{np.log2(h):.0f} J/h")
            value_over_h.append(float(jqmc_rs.compute_Jastrow_one_body(_j1(form, a, structure), moved, _EMPTY)) / sep)
            closed_value_over_h.append(closed_form("J1", form, a, sep, geometry != "molecular")["value_over_h"])
    assert np.all(np.isfinite(value_over_h)), _labelled(labels, value_over_h)
    np.testing.assert_allclose(value_over_h, closed_value_over_h, atol=atol, rtol=rtol, err_msg=_labelled(labels, value_over_h))


@pytest.mark.parametrize("form", ["pade", "exp"])
@pytest.mark.parametrize("kind", ["J1", "J2"])
@pytest.mark.parametrize("geometry", ["molecular", "cubic_10bohr"])
def test_rust_jastrow_grad_lap_offgrid_anchor_matches_closed_form(geometry, kind, form):
    """Native J1/J2 gradient and ``h * Laplacian`` at the off-grid anchor equal the closed forms at jastrow_grad_lap strict."""
    atol, rtol = get_tolerance_min(["jastrow_grad_lap"], "strict")
    structure = _structure(geometry)
    labels, grad_lap, closed_grad_lap = [], [], []
    for a in J1_PARAMS if kind == "J1" else J2_PARAMS:
        for h in SMALL_DISPLACEMENTS:
            sep = offgrid_separation(h)
            moved = (OFFGRID_ANCHOR + np.array([h, 0.0, 0.0]))[None, :]
            if kind == "J1":
                jastrow = Jastrow_data(jastrow_one_body_data=_j1(form, a, structure))
                r_up, r_dn = moved, _EMPTY
            else:
                j2 = Jastrow_two_body_data(jastrow_2b_param=a, jastrow_2b_type=form, structure_data=structure)
                jastrow = Jastrow_data(jastrow_two_body_data=j2)
                r_up, r_dn = OFFGRID_ANCHOR[None, :].copy(), moved
            grad_up, grad_dn, lap_up, lap_dn = (
                np.asarray(x, dtype=np.float64) for x in jqmc_rs.compute_grads_and_laplacian_jastrow_part(jastrow, r_up, r_dn)
            )
            ref = closed_form(kind, form, a, sep, geometry != "molecular")
            tag = f"{kind} {form} a={a} h=2^{np.log2(h):.0f}"
            # The +h electron: the J1 up electron, the J2 down electron.
            grad_moved, lap_moved = (grad_up[0], lap_up[0]) if kind == "J1" else (grad_dn[0], lap_dn[0])
            rows = [
                ("grad_x", grad_moved[0], ref["grad_x"]),
                ("grad_y", grad_moved[1], 0.0),
                ("grad_z", grad_moved[2], 0.0),
                ("h*lap", lap_moved * sep, ref["lap_times_h"]),
            ]
            if kind == "J2":
                # Pair symmetry: the partner sees the negated gradient and the same Laplacian.
                rows += [
                    ("partner grad_x", grad_up[0, 0], -ref["grad_x"]),
                    ("partner h*lap", lap_up[0] * sep, ref["lap_times_h"]),
                ]
            for label, value, expected in rows:
                labels.append(f"{tag} {label}")
                grad_lap.append(float(value))
                closed_grad_lap.append(expected)
    assert np.all(np.isfinite(grad_lap)), _labelled(labels, grad_lap)
    np.testing.assert_allclose(grad_lap, closed_grad_lap, atol=atol, rtol=rtol, err_msg=_labelled(labels, grad_lap))

"""Closed-form small-displacement check of the Rust J1 / J2 values.

Rust mirror of ``tests/test_jqmc_kernel/test_jastrow_small_displacement.py``:
``jqmc_rs.compute_Jastrow_one_body`` / ``compute_Jastrow_two_body`` at a
nucleus (or partner electron) at ``(5, 5, 5)`` Bohr and the moved electron at
``+h`` along x, ``h in {2**-16, 2**-26, 2**-38}``, molecular and cubic
periodic. ``J(h) / h`` must equal the independent closed form of
``tests/test_jqmc_kernel/_jastrow_small_displacement_reference.py`` at the
jastrow_eval strict zone tolerance of the active precision mode (the Rust
f32 kernels run in the mixed lane). The closed form is evaluated in
the active jastrow_eval dtype (fp32 in mixed, fp64 in full) from the
fp64-reconstructed separation cast to that dtype. The exp forms pin the ``exp_m1``
evaluation of ``1 - exp(-x)``; the subtraction form returns 0 for the
fp32 quotient at ``h = 2**-26``.
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

from jqmc._precision import get_tolerance
from jqmc.jastrow_factor import Jastrow_one_body_data, Jastrow_two_body_data
from jqmc.structure import Structure_data
from tests.conftest import CELL_FIXTURES, make_pbc_structure
from tests.test_jqmc_kernel._jastrow_small_displacement_reference import (
    J1_PARAMS,
    J2_PARAMS,
    NUCLEUS,
    SMALL_DISPLACEMENTS,
    SWITCH_SEPARATIONS,
    closed_form,
)

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]


def _structure(geometry: str) -> Structure_data:
    labels = {"atomic_numbers": (1,), "element_symbols": ("H",), "atomic_labels": ("H1",)}
    if geometry == "molecular":
        return Structure_data(pbc_flag=False, positions=NUCLEUS[None, :].copy(), **labels)
    return make_pbc_structure(CELL_FIXTURES[geometry], positions=NUCLEUS[None, :].copy(), **labels)


def _value_quotients(geometry: str, kind: str, form: str, separations) -> tuple[list[float], list[float], str]:
    """Return Rust ``J(h) / h``, its closed form and a labelled message over ``J*_PARAMS x separations``."""
    structure = _structure(geometry)
    labels, value_over_h, closed_value_over_h = [], [], []
    for a in J1_PARAMS if kind == "J1" else J2_PARAMS:
        for h in separations:
            moved = (NUCLEUS + np.array([h, 0.0, 0.0]))[None, :]
            if kind == "J1":
                data = Jastrow_one_body_data(
                    jastrow_1b_param=a, jastrow_1b_type=form, structure_data=structure, core_electrons=(0.0,)
                )
                value = jqmc_rs.compute_Jastrow_one_body(data, moved, np.zeros((0, 3), dtype=np.float64))
            else:
                data = Jastrow_two_body_data(jastrow_2b_param=a, jastrow_2b_type=form, structure_data=structure)
                value = jqmc_rs.compute_Jastrow_two_body(data, NUCLEUS[None, :].copy(), moved)
            labels.append(f"{kind} {form} a={a} h={h:.6g} J/h")
            value_over_h.append(float(value) / h)
            closed_value_over_h.append(closed_form(kind, form, a, h, geometry != "molecular")["value_over_h"])
    message = "; ".join(f"{label}={got:.12g}" for label, got in zip(labels, value_over_h, strict=True))
    return value_over_h, closed_value_over_h, message


@pytest.mark.parametrize("form", ["pade", "exp"])
@pytest.mark.parametrize("kind", ["J1", "J2"])
@pytest.mark.parametrize("geometry", ["molecular", "cubic_10bohr"])
def test_rust_jastrow_value_small_displacement_matches_closed_form(geometry, kind, form):
    """Rust ``J(h) / h`` for J1/J2 equals the jastrow_eval-dtype closed form at jastrow_eval strict."""
    atol, rtol = get_tolerance("jastrow_eval", "strict")
    value_over_h, closed_value_over_h, message = _value_quotients(geometry, kind, form, SMALL_DISPLACEMENTS)
    assert np.all(np.isfinite(value_over_h)), message
    np.testing.assert_allclose(value_over_h, closed_value_over_h, atol=atol, rtol=rtol, err_msg=message)


@pytest.mark.parametrize("kind", ["J1", "J2"])
@pytest.mark.parametrize("geometry", ["molecular", "cubic_10bohr"])
def test_rust_jastrow_exp_value_across_branch_points_matches_closed_form(geometry, kind):
    """Rust exp-form ``J(r) / r`` equals the closed form on both sides of the ``one_minus_exp_neg`` branch points."""
    atol, rtol = get_tolerance("jastrow_eval", "strict")
    value_over_h, closed_value_over_h, message = _value_quotients(geometry, kind, "exp", SWITCH_SEPARATIONS)
    assert np.all(np.isfinite(value_over_h)), message
    np.testing.assert_allclose(value_over_h, closed_value_over_h, atol=atol, rtol=rtol, err_msg=message)

"""Closed-form small-separation check of the native J1 / J2 parameter derivatives.

Rust mirror of ``tests/test_jqmc_kernel/test_jastrow_param_derivative_small_separation.py``:
``jqmc_rs.grad_ln_Psi_jastrow_1b_param`` / ``grad_ln_Psi_jastrow_2b_param``
(the native ``O_k`` of the J1 / J2 parameters, ``param_grad.rs``) at the
small-displacement fixture (nucleus / partner electron at ``(5, 5, 5)`` Bohr, moved
electron at ``+h`` along x, ``h in {2**-16, 2**-26, 2**-38}``, molecular and
cubic periodic, Pade and exp). ``(dJ/da) / h**2`` must equal the exact closed
form of ``tests/test_jqmc_kernel/_jastrow_param_derivative_reference.py`` at
the existing jastrow_eval strict tolerance of the active precision mode. The
exp-form derivative ``-q(x) / (2 a**2)``, ``q(x) = 1 - (1 + x) exp(-x)``, is
O(x**2); evaluating it as ``-(1 - e) + x e`` collapses it at ``h = 2**-26``.
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
from jqmc._precision import get_tolerance_min
from jqmc.jastrow_factor import Jastrow_one_body_data, Jastrow_two_body_data
from jqmc.structure import Structure_data
from tests.conftest import CELL_FIXTURES, make_pbc_structure
from tests.test_jqmc_kernel._jastrow_param_derivative_reference import param_derivative_over_h2
from tests.test_jqmc_kernel._jastrow_small_displacement_reference import (
    J1_PARAMS,
    J2_PARAMS,
    NUCLEUS,
    SMALL_DISPLACEMENTS,
    SWITCH_SEPARATIONS,
)

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

_EMPTY = np.zeros((0, 3), dtype=np.float64)


def _structure(geometry: str) -> Structure_data:
    labels = {"atomic_numbers": (1,), "element_symbols": ("H",), "atomic_labels": ("H1",)}
    if geometry == "molecular":
        return Structure_data(pbc_flag=False, positions=NUCLEUS[None, :].copy(), **labels)
    return make_pbc_structure(CELL_FIXTURES[geometry], positions=NUCLEUS[None, :].copy(), **labels)


def _param_derivative_quotients(geometry: str, kind: str, form: str, separations) -> tuple[list[float], list[float], str]:
    """Return native ``(dJ/da) / h**2``, its closed form and a labelled message over ``J*_PARAMS x separations``."""
    structure = _structure(geometry)
    labels, got, want = [], [], []
    for a in J1_PARAMS if kind == "J1" else J2_PARAMS:
        for h in separations:
            moved = (NUCLEUS + np.array([h, 0.0, 0.0]))[None, :]
            if kind == "J1":
                data = Jastrow_one_body_data(
                    jastrow_1b_param=a, jastrow_1b_type=form, structure_data=structure, core_electrons=(0.0,)
                )
                deriv = jqmc_rs.grad_ln_Psi_jastrow_1b_param(data, moved, _EMPTY)
            else:
                data = Jastrow_two_body_data(jastrow_2b_param=a, jastrow_2b_type=form, structure_data=structure)
                deriv = jqmc_rs.grad_ln_Psi_jastrow_2b_param(data, NUCLEUS[None, :].copy(), moved)
            labels.append(f"{kind} {form} a={a} h={h:.6g} (dJ/da)/h^2")
            got.append(float(deriv) / (h * h))
            want.append(param_derivative_over_h2(kind, form, a, h, geometry != "molecular"))
    message = "; ".join(f"{label}={value:.15g}" for label, value in zip(labels, got, strict=True))
    return got, want, message


@pytest.mark.parametrize("form", ["pade", "exp"])
@pytest.mark.parametrize("kind", ["J1", "J2"])
@pytest.mark.parametrize("geometry", ["molecular", "cubic_10bohr"])
def test_rust_jastrow_param_derivative_small_separation_matches_closed_form(geometry, kind, form):
    """Native ``(dJ/da) / h**2`` for J1/J2 equals the exact closed form at jastrow_eval strict."""
    atol, rtol = get_tolerance_min(("jastrow_eval",), "strict")
    derivative_over_h2, closed_derivative_over_h2, message = _param_derivative_quotients(
        geometry, kind, form, SMALL_DISPLACEMENTS
    )
    assert np.all(np.isfinite(derivative_over_h2)), message
    np.testing.assert_allclose(derivative_over_h2, closed_derivative_over_h2, atol=atol, rtol=rtol, err_msg=message)


@pytest.mark.parametrize("kind", ["J1", "J2"])
@pytest.mark.parametrize("geometry", ["molecular", "cubic_10bohr"])
def test_rust_jastrow_exp_param_derivative_across_branch_points_matches_closed_form(geometry, kind):
    """Native exp-form ``(dJ/da) / r**2`` on both sides of the ``|y| = 1/4`` Taylor / direct switch equals the closed form."""
    atol, rtol = get_tolerance_min(("jastrow_eval",), "strict")
    derivative_over_h2, closed_derivative_over_h2, message = _param_derivative_quotients(
        geometry, kind, "exp", SWITCH_SEPARATIONS
    )
    assert np.all(np.isfinite(derivative_over_h2)), message
    np.testing.assert_allclose(derivative_over_h2, closed_derivative_over_h2, atol=atol, rtol=rtol, err_msg=message)

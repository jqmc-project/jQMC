"""XLA J1 at an fp32-unrepresentable nucleus against its closed forms.

One hydrogen nucleus at ``(5.1, 5.2, 5.3)`` Bohr (not representable in
fp32), the electron at ``+h`` along x, ``h in {2**-16, 2**-26, 2**-38}``,
molecular and cubic periodic, Pade and exp. The production J1 value quotient,
gradient and ``s * Laplacian`` are compared with the closed forms of
``tests/test_jqmc_kernel/_jastrow_small_displacement_reference.py`` at the
exact fp64 separation ``s = fl(fl(5.1 + h) - 5.1)``, at the existing jastrow
strict zone tolerances in each precision mode. The closed forms are
evaluated in the active zone dtypes (``jastrow_eval`` value,
``jastrow_grad_lap`` gradient / Laplacian) from ``s`` cast to that dtype. This is the XLA reference
behaviour the native f32 J1 kernels must match
(``tests/test_rs_kernel/test_jastrow_offgrid_anchor.py``).
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
from jqmc.jastrow_factor import Jastrow_one_body_data, compute_grads_and_laplacian_Jastrow_one_body, compute_Jastrow_one_body
from jqmc.structure import Structure_data
from tests.conftest import CELL_FIXTURES, make_pbc_structure
from tests.test_jqmc_kernel._jastrow_param_derivative_reference import OFFGRID_ANCHOR, offgrid_separation
from tests.test_jqmc_kernel._jastrow_small_displacement_reference import J1_PARAMS, SMALL_DISPLACEMENTS, closed_form

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

_GEOMETRIES = ("molecular", "cubic_10bohr")
_FORMS = ("pade", "exp")
_EMPTY = np.zeros((0, 3), dtype=np.float64)


def _structure(geometry: str) -> Structure_data:
    labels = {"atomic_numbers": (1,), "element_symbols": ("H",), "atomic_labels": ("H1",)}
    if geometry == "molecular":
        return Structure_data(pbc_flag=False, positions=OFFGRID_ANCHOR[None, :].copy(), **labels)
    return make_pbc_structure(CELL_FIXTURES[geometry], positions=OFFGRID_ANCHOR[None, :].copy(), **labels)


def _j1(form: str, a: float, structure: Structure_data) -> Jastrow_one_body_data:
    return Jastrow_one_body_data(jastrow_1b_param=a, jastrow_1b_type=form, structure_data=structure, core_electrons=(0.0,))


def _labelled(labels: list[str], got: list[float]) -> str:
    return "; ".join(f"{label}={value:.15g}" for label, value in zip(labels, got, strict=True))


@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize("geometry", _GEOMETRIES)
def test_jastrow_one_body_offgrid_nucleus_matches_closed_form(geometry, form):
    """XLA J1 value / gradient / Laplacian with the nucleus at an fp32-unrepresentable coordinate.

    ``r - R`` must be formed from the fp64 coordinates before the downcast to
    the jastrow zones: rounding ``R = (5.1, 5.2, 5.3)`` to fp32 first moves the
    separation by up to 2.4e-7 Bohr (1.6% of ``h = 2**-16``). This is the XLA
    side of the native J1 f32 contract pinned in
    ``tests/test_rs_kernel/test_jastrow_offgrid_anchor.py``.
    """
    atol_v, rtol_v = get_tolerance("jastrow_eval", "strict")
    atol_g, rtol_g = get_tolerance_min(["jastrow_grad_lap"], "strict")
    structure = _structure(geometry)
    periodic = geometry != "molecular"
    value_labels, value_over_h, closed_value_over_h = [], [], []
    grad_labels, grad_lap, closed_grad_lap = [], [], []
    for a in J1_PARAMS:
        for h in SMALL_DISPLACEMENTS:
            sep = offgrid_separation(h)
            data = _j1(form, a, structure)
            moved = (OFFGRID_ANCHOR + np.array([h, 0.0, 0.0]))[None, :]
            want = closed_form("J1", form, a, sep, periodic)
            tag = f"J1 {form} a={a} h=2^{np.log2(h):.0f}"
            value_labels.append(f"{tag} J/h")
            value_over_h.append(float(compute_Jastrow_one_body(data, moved, _EMPTY)) / sep)
            closed_value_over_h.append(want["value_over_h"])
            grad_up, _, lap_up, _ = (
                np.asarray(x, dtype=np.float64) for x in compute_grads_and_laplacian_Jastrow_one_body(data, moved, _EMPTY)
            )
            for label, value, ref in (
                ("grad_x", grad_up[0, 0], want["grad_x"]),
                ("grad_y", grad_up[0, 1], 0.0),
                ("grad_z", grad_up[0, 2], 0.0),
                ("h*lap", lap_up[0] * sep, want["lap_times_h"]),
            ):
                grad_labels.append(f"{tag} {label}")
                grad_lap.append(float(value))
                closed_grad_lap.append(ref)
    assert np.all(np.isfinite(value_over_h)), _labelled(value_labels, value_over_h)
    np.testing.assert_allclose(
        value_over_h, closed_value_over_h, atol=atol_v, rtol=rtol_v, err_msg=_labelled(value_labels, value_over_h)
    )
    assert np.all(np.isfinite(grad_lap)), _labelled(grad_labels, grad_lap)
    np.testing.assert_allclose(grad_lap, closed_grad_lap, atol=atol_g, rtol=rtol_g, err_msg=_labelled(grad_labels, grad_lap))

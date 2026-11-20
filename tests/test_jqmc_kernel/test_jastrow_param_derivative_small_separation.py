"""Closed-form small-separation checks of the Jastrow parameter derivatives.

The small-displacement checks applied to ``jqmc.derivatives``: the J1 / J2
parameter derivatives ``grad_J1_wrt_param`` / ``grad_J2_wrt_param``
(``derivatives/assemble.py``, the XLA ``O_k`` of the Jastrow parameters,
also reused per mesh point by ``one_move_mesh.dV_ecp_dc_fast``) and the
one-move mesh J2 value delta ``one_move_mesh.j2_mesh`` at the small-displacement
fixture (nucleus / partner at ``(5, 5, 5)`` Bohr, moved electron at ``+h``,
``h in {2**-16, 2**-26, 2**-38}``, molecular and cubic periodic, Pade and
exp), judged against the independent closed forms of
``tests/test_jqmc_kernel/_jastrow_param_derivative_reference.py`` at the
existing jastrow strict zone tolerances in each precision mode.
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

from decimal import Decimal, localcontext

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_tolerance, get_tolerance_min
from jqmc.derivatives.assemble import grad_J1_wrt_param, grad_J2_wrt_param
from jqmc.derivatives.one_move_mesh import j2_mesh
from jqmc.jastrow_factor import Jastrow_one_body_data, Jastrow_two_body_data
from jqmc.structure import Structure_data
from tests.conftest import CELL_FIXTURES, make_pbc_structure
from tests.test_jqmc_kernel._jastrow_param_derivative_reference import j2_mesh_delta_over_h, param_derivative_over_h2
from tests.test_jqmc_kernel._jastrow_small_displacement_reference import (
    J1_PARAMS,
    J2_PARAMS,
    NUCLEUS,
    SMALL_DISPLACEMENTS,
    SWITCH_SEPARATIONS,
)

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

_GEOMETRIES = ("molecular", "cubic_10bohr")
_KINDS = ("J1", "J2")
_FORMS = ("pade", "exp")
_EMPTY = np.zeros((0, 3), dtype=np.float64)


def _structure(geometry: str) -> Structure_data:
    labels = {"atomic_numbers": (1,), "element_symbols": ("H",), "atomic_labels": ("H1",)}
    if geometry == "molecular":
        return Structure_data(pbc_flag=False, positions=NUCLEUS[None, :].copy(), **labels)
    return make_pbc_structure(CELL_FIXTURES[geometry], positions=NUCLEUS[None, :].copy(), **labels)


def _j1(form: str, a: float, structure: Structure_data) -> Jastrow_one_body_data:
    return Jastrow_one_body_data(jastrow_1b_param=a, jastrow_1b_type=form, structure_data=structure, core_electrons=(0.0,))


def _j2(form: str, a: float, structure: Structure_data) -> Jastrow_two_body_data:
    return Jastrow_two_body_data(jastrow_2b_param=a, jastrow_2b_type=form, structure_data=structure)


def _labelled(labels: list[str], got: list[float]) -> str:
    return "; ".join(f"{label}={value:.15g}" for label, value in zip(labels, got, strict=True))


def _param_derivative_quotients(geometry: str, kind: str, form: str, separations) -> tuple[list[str], list[float], list[float]]:
    """Return labels, production ``(dJ/da) / h**2`` and its closed form over ``J*_PARAMS x separations``."""
    structure = _structure(geometry)
    labels, got, want = [], [], []
    for a in J1_PARAMS if kind == "J1" else J2_PARAMS:
        for h in separations:
            displaced_r_carts = (NUCLEUS + np.array([h, 0.0, 0.0]))[None, :]
            if kind == "J1":
                deriv = grad_J1_wrt_param(_j1(form, a, structure), displaced_r_carts, _EMPTY)
            else:
                deriv = grad_J2_wrt_param(_j2(form, a, structure), NUCLEUS[None, :].copy(), displaced_r_carts)
            labels.append(f"{kind} {form} a={a} h={h:.6g} (dJ/da)/h^2")
            got.append(float(deriv) / (h * h))
            want.append(param_derivative_over_h2(kind, form, a, h, geometry != "molecular"))
    return labels, got, want


@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("geometry", _GEOMETRIES)
def test_jastrow_param_derivative_small_separation_matches_closed_form(geometry, kind, form):
    """``(dJ/da) / h**2`` of ``grad_J1_wrt_param`` / ``grad_J2_wrt_param`` equals the exact closed form.

    The exp-form derivative ``q(x) = 1 - (1 + x) exp(-x)`` is O(x**2) while its
    two terms are O(x); forming it by subtraction loses every digit at
    ``h = 2**-26`` in fp64 (the O_k of a near-coincident pair collapses).
    """
    atol, rtol = get_tolerance_min(("jastrow_eval",), "strict")
    labels, derivative_over_h2, closed_derivative_over_h2 = _param_derivative_quotients(
        geometry, kind, form, SMALL_DISPLACEMENTS
    )
    assert np.all(np.isfinite(derivative_over_h2)), _labelled(labels, derivative_over_h2)
    np.testing.assert_allclose(
        derivative_over_h2, closed_derivative_over_h2, atol=atol, rtol=rtol, err_msg=_labelled(labels, derivative_over_h2)
    )


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("geometry", _GEOMETRIES)
def test_jastrow_exp_param_derivative_across_branch_points_matches_closed_form(geometry, kind):
    """Exp-form ``(dJ/da) / r**2`` on both sides of the ``|x| = 1/4`` Taylor / direct switch equals the closed form."""
    atol, rtol = get_tolerance_min(("jastrow_eval",), "strict")
    labels, derivative_over_h2, closed_derivative_over_h2 = _param_derivative_quotients(
        geometry, kind, "exp", SWITCH_SEPARATIONS
    )
    assert np.all(np.isfinite(derivative_over_h2)), _labelled(labels, derivative_over_h2)
    np.testing.assert_allclose(
        derivative_over_h2, closed_derivative_over_h2, atol=atol, rtol=rtol, err_msg=_labelled(labels, derivative_over_h2)
    )


@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize("geometry", _GEOMETRIES)
def test_j2_mesh_value_delta_small_separation_matches_closed_form(geometry, form):
    """``one_move_mesh.j2_mesh`` value delta ``/ h`` for a move from ``+h x`` to ``+h (x + y)`` equals the closed form.

    Both separations are O(h); the exp pair difference must retain its small
    value (separately subtracting ``1 - exp(-b r)`` loses ~eps/x relative on
    the non-dyadic diagonal separation, ~3e-5 at ``h = 2**-38``).
    """
    atol, rtol = get_tolerance("jastrow_eval", "strict")
    structure = _structure(geometry)
    labels, delta_over_h, closed_delta_over_h = [], [], []
    for a in J2_PARAMS:
        for h in SMALL_DISPLACEMENTS:
            r_carts = np.stack([NUCLEUS, NUCLEUS + np.array([h, 0.0, 0.0])])
            candidate_r_carts = (NUCLEUS + np.array([h, h, 0.0]))[None, :]
            delta, _, _ = j2_mesh(_j2(form, a, structure), r_carts, candidate_r_carts, np.array([1]))
            labels.append(f"J2 {form} a={a} h=2^{np.log2(h):.0f} dJ/h")
            delta_over_h.append(float(np.asarray(delta)[0]) / h)
            closed_delta_over_h.append(j2_mesh_delta_over_h(form, a, h, geometry != "molecular"))
    assert np.all(np.isfinite(delta_over_h)), _labelled(labels, delta_over_h)
    np.testing.assert_allclose(delta_over_h, closed_delta_over_h, atol=atol, rtol=rtol, err_msg=_labelled(labels, delta_over_h))


@pytest.mark.parametrize("point_count", [1, 12])
@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize("geometry", _GEOMETRIES)
def test_j2_mesh_difference_switch_drift_and_autodiff_match_closed_form(geometry, form, point_count):
    """Mesh value differences, spectator/moved drifts and AD match independent closed forms.

    An explicit two-electron component fixture checks tiny diagonal moves and
    signed exponent differences around both sides of the |dx|=1/4 switch,
    zero displacement and larger moves, with one or twelve simultaneous points.
    Tiny value differences are divided by their input separation to retain
    detection sensitivity. The reference uses Decimal exponentials and NumPy
    geometry, never the production Taylor polynomial. Each mode uses its own
    unchanged strict jastrow zones, without finite differences or sampling.
    """
    structure = _structure(geometry)
    b = 0.75
    data = _j2(form, b, structure)
    periodic = geometry != "molecular"
    h = 2.0**-38
    atol_value, rtol_value = get_tolerance("jastrow_eval", "strict")
    atol_drift, rtol_drift = get_tolerance("jastrow_grad_lap", "strict")
    tiny_points = np.tile(np.array([[h, h, 0.0], [h, 1.25 * h, 0.0], [h, 1.5 * h, 0.0]]), (4, 1))[:point_count]
    exponent_changes = np.array(
        [
            -0.250000000001,
            -0.25,
            -0.249999999999,
            -(2.0**-26),
            0.0,
            2.0**-26,
            0.249999999999,
            0.25,
            0.250000000001,
            1.0,
            2.0,
            -0.5,
        ]
    )[:point_count]
    new_distances = 1.0 + exponent_changes / b
    if periodic:
        # Separate larger positive case stays within the cubic radial chart;
        # all six points bracketing +/-1/4 remain unmodified.
        new_distances = np.minimum(new_distances, 2.5)
    old_position = np.arcsin(np.pi / 10.0) * 10.0 / np.pi if periodic else 1.0
    new_positions = np.arcsin(new_distances * np.pi / 10.0) * 10.0 / np.pi if periodic else new_distances
    switch_points = np.column_stack([new_positions, np.zeros(point_count), np.zeros(point_count)])
    for scenario, old_x, candidate_r_carts, value_scale in (
        ("tiny", h, tiny_points, h),
        ("switch", old_position, switch_points, 1.0),
    ):
        r_carts = np.array([[0.0, 0.0, 0.0], [old_x, 0.0, 0.0]])
        indices = np.ones(point_count, dtype=np.int64)
        values, spectator_drift, moved_drift = j2_mesh(data, r_carts, candidate_r_carts, indices)
        distances, gradients = [], []
        # For this two-electron fixture, the moved-minus-spectator vector is
        # exactly the point. Recompute geometry from the encoded inputs.
        for coordinates in (r_carts[1:2], candidate_r_carts):
            if periodic:
                k = np.pi / 10.0
                smooth = np.sin(k * coordinates) / k
                distance = np.linalg.norm(smooth, axis=1)
                gradient = np.sin(2.0 * k * coordinates) / (2.0 * k * distance[:, None])
            else:
                distance = np.linalg.norm(coordinates, axis=1)
                gradient = coordinates / distance[:, None]
            distances.append(distance)
            gradients.append(gradient)
        old_distance, new_distance = distances
        closed_values = []
        for distance in new_distance:
            with localcontext() as context:
                context.prec = 60
                db = Decimal(b)
                dn, do = Decimal(float(distance)), Decimal(float(old_distance[0]))
                if form == "exp":
                    delta = ((-db * do).exp() - (-db * dn).exp()) / (2 * db)
                else:
                    delta = dn / (2 * (1 + db * dn)) - do / (2 * (1 + db * do))
                closed_values.append(float(delta) / value_scale)
        radial_old = np.exp(-b * old_distance) / 2 if form == "exp" else 1 / (2 * (1 + b * old_distance) ** 2)
        radial_new = np.exp(-b * new_distance) / 2 if form == "exp" else 1 / (2 * (1 + b * new_distance) ** 2)
        closed_moved_drift = radial_new[:, None] * gradients[1]
        closed_spectator_drift = np.zeros((point_count, 2, 3))
        closed_spectator_drift[:, 0] = radial_old[:, None] * gradients[0] - closed_moved_drift
        value_quotients = np.asarray(values) / value_scale
        np.testing.assert_allclose(value_quotients, closed_values, atol=atol_value, rtol=rtol_value, equal_nan=False)
        np.testing.assert_allclose(moved_drift, closed_moved_drift, atol=atol_drift, rtol=rtol_drift, equal_nan=False)
        np.testing.assert_allclose(spectator_drift, closed_spectator_drift, atol=atol_drift, rtol=rtol_drift, equal_nan=False)
        mesh_jacobian = jax.jacfwd(lambda p: j2_mesh(data, r_carts, p, indices)[0])(jnp.asarray(candidate_r_carts))
        closed_mesh_jacobian = np.zeros((point_count, point_count, 3))
        closed_mesh_jacobian[np.arange(point_count), np.arange(point_count)] = closed_moved_drift
        assert np.all(np.isfinite(mesh_jacobian))
        np.testing.assert_allclose(mesh_jacobian, closed_mesh_jacobian, atol=atol_drift, rtol=rtol_drift, equal_nan=False)

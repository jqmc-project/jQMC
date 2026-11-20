"""Kinetic contraction algebra and work bounds, with orbital evaluation isolated.

The synthetic tables are local value/gradient/Laplacian jets. They can be
realized by quadratic orbitals around each electron; no AO approximation or
inverse conditioning filter is involved. Physical-orbital controls remain in
TestDeterminantFastUpdate and the canonical fast local-energy assembly tests.
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

from jqmc import determinant as det
from jqmc import jastrow_factor as jas
from jqmc._precision import get_tolerance
from tests.test_jqmc_kernel.test_determinant import _build_minimal_mos_data

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _draw(rng, shape, complex_values=False):
    values = rng.normal(size=shape)
    if complex_values:
        values = values + 0.3j * rng.normal(size=shape)
    return jnp.asarray(values)


def _det_case(kind, shell, complex_values=False, n_orb=7):
    rng = np.random.default_rng(359)
    n_up, n_dn = {"closed": (3, 3), "open": (3, 2), "empty-down": (3, 0)}[shell]
    mos_up = _build_minimal_mos_data(n_orb, n_orb)
    mos_up = mos_up.replace(mo_coefficients=np.asarray(_draw(rng, (n_orb, n_orb), complex_values)))
    n_orb_dn = n_orb - 1 if kind == "agp" else n_orb
    mos_dn = _build_minimal_mos_data(n_orb_dn, n_orb)
    mos_dn = mos_dn.replace(mo_coefficients=np.asarray(_draw(rng, (n_orb_dn, n_orb), complex_values)))
    common = dict(
        num_electron_up=n_up, num_electron_dn=n_dn, orb_data_up_spin=mos_up, orb_data_dn_spin=mos_dn, time_reversal=False
    )
    if kind == "sd":
        common["orb_data_up_spin"] = mos_up.replace(num_mo=n_up, mo_coefficients=mos_up.mo_coefficients[:n_up])
        common["orb_data_dn_spin"] = mos_dn.replace(num_mo=n_dn, mo_coefficients=mos_dn.mo_coefficients[:n_dn])
        data = det.SD_data(**common)
        # The SD API evaluates only occupied rows.
        n_orb, n_orb_dn = n_up, n_dn
        paired = jnp.eye(n_up)[:, :n_dn]
        unpaired = jnp.eye(n_up)[:, n_dn:]
    elif kind == "agpn":
        data = det.AGPn_data(
            **common,
            lambda_diag=np.linspace(0.3, 1.1, n_orb),
            lambda_unpaired=np.asarray(_draw(rng, (n_orb, n_up - n_dn), complex_values)),
        )
        paired, unpaired = jnp.split(data._lambda_matrix_jnp, [n_orb_dn], axis=1)
    else:
        paired = _draw(rng, (n_orb, n_orb_dn), complex_values)
        unpaired = _draw(rng, (n_orb, n_up - n_dn), complex_values)
        data = det.AGP_data(**common, lambda_matrix=np.asarray(jnp.column_stack((paired, unpaired))))
    up = tuple(_draw(rng, (n_orb, n_up), complex_values) for _ in range(5))
    down = tuple(_draw(rng, (n_orb_dn, n_dn), complex_values) for _ in range(5))
    matrix = jnp.column_stack((up[0].T @ paired @ down[0], up[0].T @ unpaired))
    inverse = jnp.linalg.inv(matrix)
    return data, paired, unpaired, up, down, inverse


def _dense_reference(paired, unpaired, up, down, inverse):
    """Explicit derivative matrices and scalar diagonal contractions (no reassociation)."""
    up_fields, down_fields = [], []
    for derivative in range(1, 5):
        d_up = jnp.column_stack((up[derivative].T @ paired @ down[0], up[derivative].T @ unpaired))
        d_down = jnp.column_stack(
            (up[0].T @ paired @ down[derivative], jnp.zeros((up[0].shape[1], unpaired.shape[1]), dtype=paired.dtype))
        )
        up_fields.append(jnp.diag(d_up @ inverse))
        down_fields.append(jnp.diag(inverse @ d_down)[: down[0].shape[1]])
    gu = jnp.stack(up_fields[:3], axis=1)
    gd = jnp.stack(down_fields[:3], axis=1)
    return gu, gd, up_fields[3] - jnp.sum(gu**2, axis=1), down_fields[3] - jnp.sum(gd**2, axis=1)


@pytest.mark.parametrize("kind", ["sd", "agp", "agpn"])
@pytest.mark.parametrize("shell", ["closed", "open", "empty-down"])
@pytest.mark.parametrize("complex_values", [False, True])
def test_fast_contractions_preserve_dense_values_and_directional_derivatives(monkeypatch, kind, shell, complex_values):
    """Actual fast contraction retains complex bilinear values and coordinate-jet JVPs.

    Dense AGP has rectangular spin orbital dimensions; AGPn has nonidentity
    occupations; SD uses occupied rows with no lambda. Closed, open and empty
    down spins each use the same independently constructed inverse. A jet
    perturbation checks differentiation through values, derivative tables and
    inverse together, including imaginary components. Existing strict
    det_grad_lap budgets apply in both precision modes.
    """
    monkeypatch.setattr(det, "compute_MOs_value_grad_lap", lambda _orb, tables: tables)
    data, paired, unpaired, up, down, inverse = _det_case(kind, shell, complex_values)

    def fast(u, d, i):
        return det.compute_grads_and_laplacian_ln_Det_fast.__wrapped__(data, u, d, i)

    def dense(u, d, i):
        return _dense_reference(paired, unpaired, u, d, i)

    tangent = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * (0.1 + (0.07j if complex_values else 0)), (up, down, inverse))
    actual = jax.jvp(fast, (up, down, inverse), tangent)
    reference = jax.jvp(dense, (up, down, inverse), tangent)
    atol, rtol = get_tolerance("det_grad_lap", "strict")
    for index, (observed, expected) in enumerate(zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(reference))):
        comparison_iteration = str(index)
        np.testing.assert_allclose(observed, expected, atol=atol, rtol=rtol, equal_nan=False, err_msg=comparison_iteration)


def _dot_work(closed):
    """Count scalar multiply/add work in traced dot contractions, independent of backend fusion."""
    total = 0
    for eqn in closed.jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            contracted = eqn.params["dimension_numbers"][0][0]
            size = np.prod([eqn.invars[0].aval.shape[i] for i in contracted])
            total += 2 * int(size) * int(np.prod(eqn.outvars[0].aval.shape))
        for value in eqn.params.values():
            if hasattr(value, "jaxpr"):
                total += _dot_work(value)
    return total


def test_fast_determinant_contraction_work_bound(monkeypatch):
    """The actual fast route contracts the inverse before orbital derivatives.

    The bound permits the two paired orbital/inverse chains and linear
    gradient/Laplacian reductions. It excludes the old four full derivative
    tables per spin; no compiler-version-dependent wall-time threshold.
    """
    monkeypatch.setattr(det, "compute_MOs_value_grad_lap", lambda _orb, tables: tables)
    data, _, _, up, down, inverse = _det_case("agp", "open", n_orb=64)
    traced = jax.make_jaxpr(lambda u, d, i: det.compute_grads_and_laplacian_ln_Det_fast.__wrapped__(data, u, d, i))(
        up, down, inverse
    )
    n, m, nu, nd = 64, 63, 3, 2
    allowed = 2 * (2 * n * m * nd + n * nu * nu + n * nu * nd) + 40 * n * nu
    assert _dot_work(traced) <= allowed


def _j3_reference(matrix, up, down):
    """Original ordered scalar pair derivative, including both coupling orientations."""
    values = jnp.concatenate([up[0], down[0]], axis=1)
    fields = []
    for electron in range(values.shape[1]):
        coefficient = matrix[:, -1]
        for other in range(values.shape[1]):
            if other > electron:
                coefficient = coefficient + matrix[:, :-1] @ values[:, other]
            elif other < electron:
                coefficient = coefficient + matrix[:, :-1].T @ values[:, other]
        fields.append(coefficient)
    coefficients = jnp.stack(fields, axis=1)
    gu = jnp.stack([jnp.sum(coefficients[:, : up[0].shape[1]] * table, axis=0) for table in up[1:4]], axis=1)
    gd = jnp.stack([jnp.sum(coefficients[:, up[0].shape[1] :] * table, axis=0) for table in down[1:4]], axis=1)
    lu = jnp.sum(coefficients[:, : up[0].shape[1]] * up[4], axis=0)
    ld = jnp.sum(coefficients[:, up[0].shape[1] :] * down[4], axis=0)
    return gu, gd, lu, ld


@pytest.mark.parametrize("n_down", [0, 2])
@pytest.mark.parametrize("symmetric", [False, True])
def test_j3_ordered_pair_values_and_coupling_derivatives(monkeypatch, n_down, symmetric):
    """Prefix/suffix sums preserve pair orientation and unconstrained coupling JVP.

    Arbitrary asymmetric matrices are an algebraic control; exactly symmetric
    matrices are the supported physical J3 input. Off-symmetry tangent entries
    check the public parameter derivatives at symmetric inputs too.
    """
    monkeypatch.setattr(jas, "compute_AOs_value_grad_lap", lambda _orb, tables: tables)
    rng = np.random.default_rng(359)
    matrix = _draw(rng, (7, 8))
    if symmetric:
        matrix = matrix.at[:, :-1].set((matrix[:, :-1] + matrix[:, :-1].T) / 2)
    up = tuple(_draw(rng, (7, 3)) for _ in range(5))
    down = tuple(_draw(rng, (7, n_down)) for _ in range(5))
    data = jas.Jastrow_three_body_data(orb_data=_build_minimal_mos_data(7, 7).aos_data, j_matrix=matrix)

    def fast(j):
        return jas.compute_grads_and_laplacian_Jastrow_three_body.__wrapped__(data.replace(j_matrix=j), up, down)

    def reference(j):
        return _j3_reference(j, up, down)

    direction = _draw(rng, matrix.shape)
    actual = jax.jvp(fast, (matrix,), (direction,))
    expected = jax.jvp(reference, (matrix,), (direction,))
    atol, rtol = get_tolerance("jastrow_grad_lap", "strict")
    for index, (observed, scalar) in enumerate(zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected))):
        comparison_iteration = str(index)
        np.testing.assert_allclose(observed, scalar, atol=atol, rtol=rtol, equal_nan=False, err_msg=comparison_iteration)


def test_j3_pair_accumulation_has_no_quadratic_electron_contractions(monkeypatch):
    """Tracing real J3 contraction accepts linear prefix sums, rejects dense N-by-N masks."""
    monkeypatch.setattr(jas, "compute_AOs_value_grad_lap", lambda _orb, tables: tables)
    rng = np.random.default_rng(359)
    data = jas.Jastrow_three_body_data(orb_data=_build_minimal_mos_data(7, 7).aos_data, j_matrix=_draw(rng, (7, 8)))
    up = tuple(_draw(rng, (7, 32)) for _ in range(5))
    down = tuple(_draw(rng, (7, 24)) for _ in range(5))
    traced = jax.make_jaxpr(lambda u, d: jas.compute_grads_and_laplacian_Jastrow_three_body.__wrapped__(data, u, d))(up, down)
    # Two oriented orbital contractions, plus four per-electron reductions.
    allowed = 4 * 7 * 7 * (32 + 24) + 8 * 7 * (32 + 24)
    assert _dot_work(traced) <= allowed

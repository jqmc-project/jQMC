"""Independent projected-manifold primal for the DTN cutoff-policy fields.

No production policy tangent, radial derivative, or gradient/Laplacian helper
is used in the reference.  AD differentiates the nearest constrained full
coefficients and the elementary polynomial Jastrow value.
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

from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RCOND_DTN_CONSTRAINT_REFERENCE, RTOL_PYTHON_FP64_ZERO
from jqmc.derivatives.dtn import dtn_cutoff_policy_fields
from tests.test_jqmc_kernel.test_jastrow_dtn import _build_dtn, _draw_electrons, _h2_pbc_structure

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]


def _projected_primal(data):
    """J(t,r), with L(t)=exp(t)L and Euclidean projection onto cusp/null constraints.

    The policy fields themselves deliberately evaluate in fp64 in both modes;
    this independent scalar primal uses that same arithmetic, not a full-mode
    production result as a mixed-mode oracle.
    """
    C = data.cutoff_smoothness
    alpha = jnp.einsum("ski,si->sk", jnp.asarray(data.M_alpha), data.alpha_indep) + jnp.asarray(data.b_alpha)
    beta = jnp.einsum("Iki,Ii->Ik", jnp.asarray(data.M_beta), data.beta_indep) + jnp.asarray(data.b_beta)
    gamma = jnp.einsum("Ilmni,Ii->Ilmn", jnp.asarray(data.M_gamma), data.gamma_indep)
    cell = jnp.asarray(data.structure_data._cell_jnp, dtype=jnp.float64)
    inverse = jnp.linalg.inv(cell)
    ions = jnp.asarray(data.structure_data.positions, dtype=jnp.float64)
    species = np.asarray(data.species_index)
    ne, nee = data.n_order_eN, data.n_order_ee
    size = (ne + 1) ** 2 * (nee + 1)
    eye = jnp.eye(size).reshape((ne + 1, ne + 1, nee + 1, size))

    def gamma_constraints(L):
        rows = [eye[l, m, n] - eye[m, l, n] for l in range(ne + 1) for m in range(l) for n in range(nee + 1)]
        rows += [sum((eye[l, s - l, 1] for l in range(ne + 1) if 0 <= s - l <= ne), jnp.zeros(size)) for s in range(2 * ne + 1)]
        rows += [
            sum((C * eye[0, m, s - m] - L * eye[1, m, s - m] for m in range(ne + 1) if 0 <= s - m <= nee), jnp.zeros(size))
            for s in range(ne + nee + 1)
        ]
        return jnp.stack(rows)

    def cusp_project(coefficients, L, cusp):
        row = jnp.zeros(coefficients.shape[-1]).at[0].set(C).at[1].set(-L)
        # Slope ``cusp`` at 0+ of the evaluators' (L - r)^C form: C c_0 - L c_1 = -cusp / L^(C-1).
        rhs = -cusp / L ** (C - 1)
        return coefficients + (rhs - coefficients @ row)[..., None] * row / (row @ row)

    def distance(a, b):
        fractional = (a - b) @ inverse
        return jnp.linalg.norm((fractional - jnp.round(fractional)) @ cell)

    def polynomial(r, L, order):
        return jnp.maximum(L - r, 0.0) ** C * r ** jnp.arange(order + 1)

    def primal(t, electrons):
        scale = jnp.exp(t)
        lu, lc, lf = data.L_u * scale, jnp.asarray(data.L_chi) * scale, jnp.asarray(data.L_f) * scale
        a = cusp_project(alpha, lu, jnp.asarray([0.25, 0.5]))
        b = jnp.stack([cusp_project(beta[s], lc[s], -data.z_eff_per_species[s]) for s in range(beta.shape[0])])
        g = []
        for s in range(gamma.shape[0]):
            constraints = gamma_constraints(lf[s])
            original = gamma[s].ravel()
            g.append(
                (
                    original - jnp.linalg.pinv(constraints, rtol=RCOND_DTN_CONSTRAINT_REFERENCE) @ (constraints @ original)
                ).reshape(gamma[s].shape)
            )
        value = jnp.asarray(0.0, dtype=jnp.float64)
        for i in range(3):
            for atom, ion in enumerate(ions):
                s = species[atom]
                value += polynomial(distance(electrons[i], ion), lc[s], data.n_order_chi) @ b[s]
            for j in range(i):
                rij = distance(electrons[i], electrons[j])
                channel = 0 if i < 2 else 1
                value += polynomial(rij, lu, data.n_order_u) @ a[channel]
                for atom, ion in enumerate(ions):
                    s = species[atom]
                    value += jnp.einsum(
                        "lmn,l,m,n->",
                        g[s],
                        polynomial(distance(electrons[i], ion), lf[s], ne),
                        polynomial(distance(electrons[j], ion), lf[s], ne),
                        rij ** jnp.arange(nee + 1),
                    )
        return value

    return primal


def _reference_fields(data, up, down):
    primal = _projected_primal(data)
    electrons = jnp.concatenate((up, down))

    def fields(t):
        value = primal(t, electrons)
        gradient = jax.grad(primal, argnums=1)(t, electrons)
        hessian = jax.hessian(primal, argnums=1)(t, electrons).reshape((9, 9))
        laplacian = jnp.diag(hessian).reshape((3, 3)).sum(axis=1)
        return value, gradient[:2], gradient[2:], laplacian[:2], laplacian[2:]

    return jax.jit(lambda: jax.jvp(fields, (jnp.asarray(0.0),), (jnp.asarray(1.0),))[1])()


def test_original_cutoff_fixture_is_outside_support():
    """Preserve the original all-zero fixture as a support/boundary check, not accuracy evidence."""
    structure = _h2_pbc_structure()
    data = _build_dtn(structure, seed=43, L=2.0, N_u=4, N_chi=4, N_eN=2, N_ee=2)
    up, down = _draw_electrons(structure, 2, 1, seed=44)
    electrons = np.concatenate((up, down))
    cell = np.asarray(structure._cell_jnp)
    inv = np.linalg.inv(cell)
    displacement = electrons[:, None] - np.asarray(structure.positions)
    fractional = displacement @ inv
    ion_distances = np.linalg.norm((fractional - np.round(fractional)) @ cell, axis=-1)
    assert np.all(ion_distances > max(np.max(data.L_chi), np.max(data.L_f)))
    for i in range(3):
        for j in range(i):
            fractional = (electrons[i] - electrons[j]) @ inv
            assert np.linalg.norm((fractional - np.round(fractional)) @ cell) > data.L_u
    actual = dtn_cutoff_policy_fields(data, up, down)
    reference = _reference_fields(data, up, down)
    for field, got, ref in zip(("value", "grad-up", "grad-down", "lap-up", "lap-down"), actual, reference):
        assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref))
        np.testing.assert_array_equal(got, np.zeros_like(got))
        np.testing.assert_array_equal(ref, np.zeros_like(ref))


#: Electrons placed inside every DTN support of the original fixture (H2 at
#: (2,2,2) and (3.4,2,2), L_u = L_chi = L_f = 2 Bohr): each electron is within
#: 1.35 Bohr of both nuclei and every electron pair is within 1.56 Bohr.
_IN_SUPPORT_UP = np.array([[2.5, 2.3, 2.1], [3.0, 2.8, 1.6]])
_IN_SUPPORT_DOWN = np.array([[2.7, 1.5, 2.4]])


def test_in_support_fixture_matches_independent_primal_per_field():
    """Nonzero companion of the original all-zero cutoff fixture.

    Same DTN data (H2 PBC, coefficient seed 43, L = 2) as
    ``test_original_cutoff_fixture_is_outside_support``, which is kept unchanged,
    but with all three electrons inside the u, chi and f supports, so every
    element of all five policy fields (value, grad-up, grad-down, lap-up,
    lap-down) is nonzero. Each field of ``dtn_cutoff_policy_fields`` is compared
    separately with the independent projected-manifold primal, not only through
    a contracted scalar. Both sides evaluate in fp64 in both precision modes
    (see ``_projected_primal``), so each mode is judged against its own fp64
    reference with the central fp64 pair; no full-mode result enters the
    mixed-mode verdict.
    """
    structure = _h2_pbc_structure()
    data = _build_dtn(structure, seed=43, L=2.0, N_u=4, N_chi=4, N_eN=2, N_ee=2)
    up, down = _IN_SUPPORT_UP, _IN_SUPPORT_DOWN
    electrons = np.concatenate((up, down))
    cell = np.asarray(structure._cell_jnp)
    inv = np.linalg.inv(cell)
    fractional = (electrons[:, None] - np.asarray(structure.positions)) @ inv
    ion_distances = np.linalg.norm((fractional - np.round(fractional)) @ cell, axis=-1)
    assert np.all(ion_distances < min(np.min(data.L_chi), np.min(data.L_f)))
    for i in range(3):
        for j in range(i):
            fractional = (electrons[i] - electrons[j]) @ inv
            assert np.linalg.norm((fractional - np.round(fractional)) @ cell) < data.L_u
    actual = dtn_cutoff_policy_fields(data, up, down)
    reference = _reference_fields(data, up, down)
    for field, got, ref in zip(("value", "grad-up", "grad-down", "lap-up", "lap-down"), actual, reference):
        got, ref = np.asarray(got), np.asarray(ref)
        assert got.dtype == ref.dtype == np.dtype(np.float64), field
        assert got.shape == ref.shape, field
        assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref)), field
        assert np.all(ref != 0.0), field
        np.testing.assert_allclose(got, ref, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO, err_msg=field)

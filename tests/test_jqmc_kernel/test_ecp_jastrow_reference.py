"""Independent s-Gaussian/J2 reference for the complete nonlocal ECP mesh."""

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

import ast
import inspect
import textwrap

import numpy as np
import pytest

from jqmc import coulomb_potential as coulomb
from jqmc._precision import get_dtype_np, get_tolerance_min
from tests.test_jqmc_kernel.test_ecp_projector_reference import _quadrature_18
from tests.test_jqmc_kernel.test_gfmc_ecp_gaussian_reference import _ecp_j2_hamiltonian

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _reference(nv):
    ham = _ecp_j2_hamiltonian(-1, 0.4)
    r_up = np.array([[0.4, 0.7, -0.2]])
    r_dn = np.array([[-0.6, -0.3, 0.8]])
    points = np.concatenate([r_up, r_dn])
    directions, weights = _quadrature_18() if nv == 18 else (np.concatenate([np.eye(3), -np.eye(3)]), np.full(6, 1 / 6))
    proposed = np.repeat(points[None], 2 * nv, axis=0)
    for spin in range(2):
        proposed[spin * nv : (spin + 1) * nv, spin] = np.linalg.norm(points[spin]) * directions
    dtype = get_dtype_np("jastrow_eval")

    def j_value(configuration):
        distance = np.linalg.norm((configuration[..., 0, :] - configuration[..., 1, :]).astype(dtype), axis=-1)
        return distance / (dtype(2) * (dtype(1) + dtype(0.4) * distance))

    # The radial Gaussian determinant is unchanged by each rotation about its
    # single centre; only the explicitly evaluated J2 ratio varies over angles.
    ratio = np.exp(j_value(proposed).astype(np.float64) - float(j_value(points)))
    assert all(np.ptp(block) > 0 for block in ratio.reshape(2, nv))
    values = -1.3 * np.exp(-0.7 * np.repeat(np.sum(points**2, axis=1), nv)) * np.tile(weights, 2) * ratio
    return ham, r_up, r_dn, proposed, values


def _evaluate(ham, up, down, nv, route):
    args = (ham.coulomb_potential_data, ham.wavefunction_data, up, down)
    if route == "all-pairs":
        return coulomb.compute_ecp_non_local_parts_all_pairs(*args, RT=np.eye(3), Nv=nv)
    if route == "nearest":
        return coulomb.compute_ecp_non_local_parts_nearest_neighbors(*args, RT=np.eye(3), NN=1, Nv=nv)
    if route == "debug-all-pairs":
        return coulomb._compute_ecp_non_local_parts_all_pairs_debug(*args, RT=np.eye(3), Nv=nv)
    assert route == "debug-nearest"
    return coulomb._compute_ecp_non_local_parts_nearest_neighbors_debug(*args, RT=np.eye(3), NN=1, Nv=nv)


@pytest.mark.parametrize("nv", [6, 18])
@pytest.mark.parametrize("route", ["all-pairs", "nearest", "debug-all-pairs", "debug-nearest"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_ecp_jastrow_complete_mesh_matches_independent_formula(nv, route):
    """Both spin blocks of the full ECP mesh equal the independent Gaussian/J2 formula.

    Synthetic molecular one-centre SD, one electron per spin, s Gaussian
    exponent 0.5, nonzero J2 Pade b=0.4 and negative s projector
    -1.3 exp(-0.7 r^2); no TREXIO. Positions are (0.4,0.7,-0.2) and
    (-0.6,-0.3,0.8), with no random sampling or conditioning selection.
    Identity rotation and both Nv=6/18 meshes are required. Since each Gaussian
    is radial about the sole nucleus, its determinant ratio is exactly one.
    An independent active-zone J2 formula supplies angle-dependent factors in
    both spin blocks. Compare every proposed configuration, every contribution,
    and the total using the central Coulomb/AO/Jastrow strict policy within each
    mode, never against a separate full-precision run. This tests the ECP
    assembly and reference down-spin update, not general J3/PBC accuracy.
    """
    ham, r_up, r_dn, proposed, expected_values = _reference(nv)
    result = _evaluate(ham, r_up, r_dn, nv, route)
    actual_points = np.concatenate(result[:2], axis=1)
    distances = np.max(np.abs(actual_points[:, None] - proposed[None]), axis=(2, 3))
    indices = np.argmin(distances, axis=1)
    assert len(indices) == len(set(indices)) == 2 * nv
    atol, rtol = get_tolerance_min(("coulomb", "ao_eval", "jastrow_eval"), "strict")
    reference_points = proposed[indices]
    np.testing.assert_allclose(actual_points, reference_points, atol=atol, rtol=rtol, equal_nan=False)
    actual, reference = np.asarray(result[2]), expected_values[indices]
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)
    actual_total, reference_total = np.asarray(result[3]), np.sum(expected_values)
    np.testing.assert_allclose(actual_total, reference_total, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("nv", [6, 18])
def test_ecp_reference_rejects_stale_down_spin_jastrow(nv):
    """Reintroducing the old down-spin Jastrow reuse breaks the same fixed-input formula."""
    ham, r_up, r_dn, proposed, expected_values = _reference(nv)
    baseline = _evaluate(ham, r_up, r_dn, nv, "debug-all-pairs")
    # Match the independently authored quadrature ordering bijectively, just
    # as in the normal test; neither API promises our six-direction order.
    points = np.concatenate(baseline[:2], axis=1)
    indices = np.argmin(np.max(np.abs(points[:, None] - proposed[None]), axis=(2, 3)), axis=1)
    assert len(indices) == len(set(indices)) == 2 * nv
    expected_values = expected_values[indices]
    atol, rtol = get_tolerance_min(("coulomb", "ao_eval", "jastrow_eval"), "strict")
    actual, reference = np.asarray(baseline[2]), expected_values
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)
    tree = ast.parse(textwrap.dedent(inspect.getsource(coulomb._compute_ecp_non_local_parts_all_pairs_debug)))
    removed = 0
    for parent in ast.walk(tree):
        for field, children in ast.iter_fields(parent):
            if not isinstance(children, list):
                continue
            for index, child in enumerate(children):
                if (
                    isinstance(child, ast.If)
                    and ast.unparse(child.test) == "flag_determinant_only"
                    and "jastrow_numerator = compute_Jastrow_part" in ast.unparse(child)
                    and "r_dn_carts=r_dn_carts_on_mesh" in ast.unparse(child)
                ):
                    children[index] = ast.copy_location(ast.Pass(), child)
                    removed += 1
    assert removed == 1
    namespace = dict(vars(coulomb))
    exec(compile(ast.fix_missing_locations(tree), "<stale-down-spin-jastrow>", "exec"), namespace)
    broken = namespace["_compute_ecp_non_local_parts_all_pairs_debug"]
    damaged = broken(ham.coulomb_potential_data, ham.wavefunction_data, r_up, r_dn, RT=np.eye(3), Nv=nv)
    assert np.all(np.isfinite(damaged[2]))
    assert not np.allclose(damaged[2], expected_values, atol=atol, rtol=rtol, equal_nan=False)

"""Canonical nonlocal ECP consumption, dense definition, and measured activation."""

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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min
from jqmc.wavefunction import evaluate_wavefunction
from gdd.engine.canonical_ecp import CanonicalECPCapture, ecp_activation_errors
from tests.canonical_wavefunction_fixtures import build_canonical_hamiltonian
from tests.test_jqmc_kernel.test_ecp_projector_reference import _quadrature_18


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@jax.jit
def _dense_wavefunctions(wavefunction, up, down):
    return jax.vmap(evaluate_wavefunction, in_axes=(None, 0, 0))(wavefunction, up, down)


def _direct_s_projector(ham, r_up, r_dn):
    """Explicit s-channel radial sum and dense Psi ratios, without fast ratio code."""
    coulomb, wf = ham.coulomb_potential_data, ham.wavefunction_data
    # These are assumptions of this independent reference, not inferred expected
    # output. Refuse a fixture change that needs a more general reference.
    assert tuple(coulomb.nucleus_index) == (0, 0, 0)
    assert tuple(coulomb.ang_moms) == (1, 1, 0)
    assert tuple(coulomb.max_ang_mom_plus_1) == (1,)
    assert tuple(coulomb.powers) == (1, 2, 2)
    assert np.array_equal(np.asarray(ham.structure_data.positions), np.zeros((1, 3)))
    assert float(coulomb.coefficients[2]) == 0.2
    assert float(coulomb.exponents[2]) == 0.9
    directions, weights = _quadrature_18()
    count = (len(r_up) + len(r_dn)) * 18
    up = np.repeat(r_up[None], count, axis=0)
    down = np.repeat(r_dn[None], count, axis=0)
    radial = []
    offset = 0
    for points, trials in ((r_up, up), (r_dn, down)):
        for index, point in enumerate(points):
            radius = np.linalg.norm(point)
            trials[offset : offset + 18, index] = radius * directions
            radial.extend(0.2 * np.exp(-0.9 * radius**2) * weights)
            offset += 18
    numerator = np.asarray(_dense_wavefunctions(wf, jnp.asarray(up), jnp.asarray(down)))
    denominator = np.asarray(evaluate_wavefunction(wf, jnp.asarray(r_up), jnp.asarray(r_dn)))
    return (up, down), np.asarray(radial) * np.real(numerator / denominator)


@pytest.mark.parametrize(
    "pattern_id", ["WF-PATTERN-004", "WF-PATTERN-006", "WF-PATTERN-008", "WF-PATTERN-012", "WF-PATTERN-014"]
)
def test_canonical_nonlocal_ecp_matches_direct_wavefunction_sum(pattern_id):
    """Every core ECP-target fixture's forward projector matches its dense discrete definition.

    Synthetic parameter-set Hamiltonians, no TREXIO: the five core-suite ECP-zero
    target patterns, with their actual SD/AGPn, orbitals, Jastrow/DTN, spin and
    periodic cell/k. One PCG64(seed=864) normal(scale=0.6) electronic configuration;
    cond(G) is measured and finite, never screened. Nearest neighbour NN=1, fixed
    identity rotation, Nv=18. Independently authored quadrature coordinates and
    s-channel radial weights multiply directly re-evaluated many-electron Psi
    ratios; no fast ratio or debug projector supplies this reference. Both routes
    use the same precision mode. The forward API intentionally takes the real
    part, unlike its holomorphic sibling. Every proposed electron configuration
    and each per-grid contribution are compared using the central Coulomb strict
    policy. Zeroing only nonlocal ECP coefficients must produce exactly zero and
    measurably change the same consumed output; local terms, wavefunction, rotation
    and electronic positions stay fixed. This neither claims full Cartesian
    coverage nor substitutes for the separate high-l/component ECP references.
    """
    ham = build_canonical_hamiltonian(pattern_id)
    wf, coulomb = ham.wavefunction_data, ham.coulomb_potential_data
    geminal = wf.geminal_data
    rng = np.random.default_rng(864)
    r_up = rng.normal(scale=0.6, size=(geminal.num_electron_up, 3))
    r_dn = rng.normal(scale=0.6, size=(geminal.num_electron_dn, 3))
    capture = CanonicalECPCapture(ham)
    result = capture.evaluate(coulomb, wf, r_up, r_dn)
    actual = np.asarray(result[-2])
    proposed, reference = _direct_s_projector(ham, r_up, r_dn)
    actual_points = np.concatenate([np.asarray(points).ravel() for points in result[:2]])
    reference_points = np.concatenate([points.ravel() for points in proposed])
    assert actual.shape == reference.shape == ((len(r_up) + len(r_dn)) * 18,)
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
    atol, rtol = get_tolerance_min(("coulomb",), "strict")
    assert ecp_activation_errors(capture.records[0], atol=atol, rtol=rtol) == []
    np.testing.assert_allclose(actual_points, reference_points, atol=atol, rtol=rtol, equal_nan=False)
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)
    jax.clear_caches()


@pytest.mark.parametrize("fault", ["omit-jastrow", "swap-jastrow-spin-blocks"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_ecp_dense_reference_rejects_jastrow_mesh_assembly_fault(fault):
    """The dense projector reference detects omitted or spin-swapped Jastrow weights.

    Synthetic canonical 006, no artifact, has non-Gamma unrestricted pGTO
    orbitals, AGPn and nonzero J2/J3/DTN channels, with both spin sectors.
    Retain seed-864 normal(scale=0.6) positions without conditioning selection,
    NN=1, Nv=18 and identity rotation. The explicit s-projector radial weights
    multiply independently evaluated dense Psi ratios, not the ECP assembler.
    First the actual unmodified assembler must match each mesh contribution.
    Then remove only its Jastrow multiplier or exchange its Jastrow up/down
    blocks; the same reference and unchanged Coulomb strict bound must reject
    that real expression mutation. Both modes are tested independently.
    """
    from jqmc import coulomb_potential as coulomb

    ham = build_canonical_hamiltonian("WF-PATTERN-006")
    wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
    rng = np.random.default_rng(864)
    r_up = rng.normal(scale=0.6, size=(wf.geminal_data.num_electron_up, 3))
    r_dn = rng.normal(scale=0.6, size=(wf.geminal_data.num_electron_dn, 3))
    assert len(r_up) == len(r_dn) > 0
    _, reference = _direct_s_projector(ham, r_up, r_dn)
    args = (cp, wf, jnp.asarray(r_up), jnp.asarray(r_dn), jnp.eye(3))
    actual = np.asarray(coulomb.compute_ecp_non_local_parts_nearest_neighbors(*args, NN=1, Nv=18)[-2])
    atol, rtol = get_tolerance_min(("coulomb",), "strict")
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)
    source = textwrap.dedent(inspect.getsource(coulomb._compute_ecp_non_local_parts_nearest_neighbors_impl))
    old = "jnp.exp(jastrow_xp - jastrow_x)"
    replacement = (
        "jnp.ones_like(jastrow_xp)"
        if fault == "omit-jastrow"
        else "jnp.exp(jnp.roll(jastrow_xp, r_up_carts.shape[0] * NN * Nv) - jastrow_x)"
    )
    assert source.count(old) == 1
    namespace = dict(vars(coulomb))
    exec(compile(ast.parse(source.replace(old, replacement)), "<fault-injected-ecp-jastrow>", "exec"), namespace)
    broken = namespace["_compute_ecp_non_local_parts_nearest_neighbors_impl"]
    damaged = np.asarray(broken(*args, NN=1, Nv=18)[-2])
    assert np.all(np.isfinite(damaged))
    assert not np.allclose(damaged, reference, atol=atol, rtol=rtol, equal_nan=False)
    jax.clear_caches()

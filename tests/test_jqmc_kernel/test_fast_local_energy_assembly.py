"""Fast local-energy assembly on molecular and periodic Jastrow/ECP inputs."""

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

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc import determinant, hamiltonians
from jqmc._precision import get_tolerance_min
from tests.canonical_wavefunction_fixtures import build_canonical_hamiltonian
from tests.test_rs_kernel._rs_fixtures import rs_ham

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_INPUTS = (
    "WF-PATTERN-002",
    "WF-PATTERN-004",
    "WF-PATTERN-006",
    "WF-PATTERN-008",
)


def _inputs(ham):
    geminal = ham.wavefunction_data.geminal_data
    rng = np.random.default_rng(864)
    r_up = jnp.asarray(rng.normal(scale=0.6, size=(geminal.num_electron_up, 3)))
    r_dn = jnp.asarray(rng.normal(scale=0.6, size=(geminal.num_electron_dn, 3)))
    matrix = determinant.compute_geminal_all_elements(geminal, r_up, r_dn)
    cond_g = float(np.linalg.cond(matrix))
    assert np.isfinite(cond_g)
    return ham, r_up, r_dn, jnp.linalg.inv(matrix)


def _compare(ham, r_up, r_dn, inverse, fast):
    actual = np.asarray(fast(ham, r_up, r_dn, jnp.eye(3), inverse))
    reference = np.asarray(jax.jit(hamiltonians.compute_local_energy)(ham, r_up, r_dn, jnp.eye(3)))
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic", "local_energy"), "strict")
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("pattern", _INPUTS)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_fast_local_energy_assembles_jastrow_periodic_ecp_terms(pattern):
    """Fast and full local-energy assembly agree on Jastrow, nonzero ECP and complex periodic inputs.

    Synthetic canonical 002/004/006/008 use parameter-set Hamiltonians without
    any TREXIO file. Retain the old Jastrow-free molecular tests. Rows include open and
    periodic cells, SD/AGPn, unrestricted/empty-down spin, AO/MO J3 and DTN.
    Draw one PCG64(seed=864) normal(scale=0.6) configuration per input, with
    no cond(G) screening. The freshly computed inverse belongs to exactly
    that configuration; this is assembly parity, not a carried-inverse test.
    Compare the entire scalar, including its imaginary part, with the existing
    strict local-energy zone policy in each precision separately. The full
    implementation is a route reference, not independent physics.
    """
    _compare(*_inputs(build_canonical_hamiltonian(pattern)), jax.jit(hamiltonians.compute_local_energy_fast))
    jax.clear_caches()


@pytest.mark.parametrize("pattern", ["mol_ecp_cart_j123", "pbc_gamma_sphepgto_j12", "twist_sphepgto_j12"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_committed_fast_energy_assembles_jastrow_and_ecp(pattern):
    """Fast assembly preserves the full complex local energy on committed H2-ECP/J123 and Gamma/twisted Li/J12.

    Use H2_ecp_ccpvdz_cart.h5, lithium.h5 or lithium_k025.h5 through the
    existing SD/MO fixture builder. J1/J2 are exp; the molecular case has
    Cartesian AO J3. Li has a nonzero nonlocal ECP and Ewald terms. Draw one
    PCG64(seed=864) normal(scale=0.6) configuration; no conditioning screen.
    The full-recompute route and fast route with its same-input fresh inverse
    agree at the existing strict local-energy zone tolerance in each mode.
    """
    _compare(*_inputs(rs_ham(pattern)), jax.jit(hamiltonians.compute_local_energy_fast))
    jax.clear_caches()


def _broken_fast(missing_term):
    """Isolate the real production assembler while corrupting just one component."""
    namespace = dict(vars(hamiltonians))
    original_potential = namespace["compute_coulomb_potential_fast"]
    original_kinetic = namespace["compute_kinetic_energy_all_elements_fast_update"]

    def potential(**kwargs):
        if missing_term == "potential":
            return jnp.asarray(0.0)
        if missing_term == "nonlocal-ecp":
            data = kwargs["coulomb_potential_data"]
            local_channels = np.asarray(data.max_ang_mom_plus_1)[np.asarray(data.nucleus_index)]
            nonlocal_mask = np.asarray(data.ang_moms) != local_channels
            kwargs["coulomb_potential_data"] = data.replace(
                coefficients=tuple(np.where(nonlocal_mask, 0.0, np.asarray(data.coefficients)))
            )
        return original_potential(**kwargs)

    def kinetic(**kwargs):
        up, down = original_kinetic(**kwargs)
        return (jnp.zeros_like(up), jnp.zeros_like(down)) if missing_term == "kinetic" else (up, down)

    namespace["compute_coulomb_potential_fast"] = potential
    namespace["compute_kinetic_energy_all_elements_fast_update"] = kinetic
    exec(
        compile(inspect.getsource(hamiltonians.compute_local_energy_fast), "<isolated-energy-component-fault>", "exec"),
        namespace,
    )
    return jax.jit(namespace["compute_local_energy_fast"])


@pytest.mark.parametrize("missing_term", ["kinetic", "potential", "nonlocal-ecp"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_fast_assembly_reference_rejects_missing_component(missing_term):
    """Removing T, all V, or only the nonlocal ECP from actual fast assembly fails on non-Gamma Li/J12."""
    args = _inputs(rs_ham("twist_sphepgto_j12"))
    _compare(*args, jax.jit(hamiltonians.compute_local_energy_fast))
    with pytest.raises(AssertionError):
        _compare(*args, _broken_fast(missing_term))
    jax.clear_caches()

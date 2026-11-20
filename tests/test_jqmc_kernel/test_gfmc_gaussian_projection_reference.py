"""Independent Gaussian/Coulomb anchors for actual conventional GFMC substeps."""

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

from jqmc import jqmc_gfmc as gfmc
from jqmc._precision import get_tolerance_min
from tests.test_jqmc_kernel.test_mcmc_gaussian_energy_reference import (
    _gaussian_hamiltonian,
    _gaussian_orbitals,
    _local_energy,
    _zone_cast,
)
from tests.test_rs_kernel import test_gfmc_projection as projection

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _compare_substep(charge, exponent, alat, tau, *, core=None):
    ham = _gaussian_hamiltonian(charge, exponent)
    r_up_carts = jnp.asarray([[0.4, 0.7, -0.2]])
    r_dn_carts = jnp.asarray([[-0.6, -0.3, 0.8]])[: charge - 1]
    geminal = projection.compute_geminal_all_elements(ham.wavefunction_data.geminal_data, r_up_carts, r_dn_carts)
    inverse = jnp.linalg.inv(geminal)
    kinetic = projection.compute_kinetic_energy_all_elements_fast_update(ham.wavefunction_data, r_up_carts, r_dn_carts, inverse)
    energy = _local_energy(np.asarray(r_up_carts)[None], np.asarray(r_dn_carts)[None], charge, exponent)[0]
    points = np.concatenate([r_up_carts, r_dn_carts])
    shifts = alat * np.concatenate([np.eye(3), -np.eye(3)])
    # Psi(r')/Psi(r) is the ratio of orbital VALUES, each in the ao_eval
    # dtype (moved positions reconstructed in fp64 first), divided in det_ratio.
    moved_values = _gaussian_orbitals(points[:, None, :] + shifts[None, :, :], exponent, "ao_eval")["s"]
    values = _gaussian_orbitals(points, exponent, "ao_eval")["s"]
    ratios = _zone_cast(moved_values, "det_ratio") / _zone_cast(values, "det_ratio")[:, None]
    rate = np.sum(ratios) / (2 * alat**2)
    key = jax.random.PRNGKey(3)
    after_rotation, _ = jax.random.split(key)
    _, escape_key = jax.random.split(after_rotation)
    uniform = float(jax.random.uniform(escape_key))
    escape = -np.log1p(-uniform) / rate
    elapsed = min(tau, escape)
    core = projection._jit_projection_t_core if core is None else core
    result = core(
        0, tau, 1.0, r_up_carts, r_dn_carts, inverse, key, *kinetic, None, jnp.eye(3), jnp.eye(3), False, "tmove", alat, ham
    )
    atol, rtol = get_tolerance_min(("ao_eval", "wf_kinetic", "coulomb", "local_energy", "det_ratio"), "strict")
    for quantity, actual, reference in (
        ("local-energy", result[0], energy),
        ("remaining-time", result[2], tau - elapsed),
        ("branching-weight", result[3], np.exp(-elapsed * energy)),
    ):
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=quantity)
    assert int(result[1]) == 1
    if tau == 0.0001:
        assert escape > tau
        np.testing.assert_array_equal(result[4], r_up_carts)
        np.testing.assert_array_equal(result[5], r_dn_carts)
    else:
        assert escape < tau
        assert np.any(np.concatenate([result[4], result[5]]) != points)


@pytest.mark.parametrize("charge", [1, 2])
@pytest.mark.parametrize("exponent", [0.5, 1.25])
@pytest.mark.parametrize("alat", [0.15, 0.3])
@pytest.mark.parametrize("tau", [0.0001, 0.2])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_gaussian_projection_substep_matches_physical_formula(charge, exponent, alat, tau):
    """GFMC local energy, escape time and branching weight obey independent closed forms.

    Synthetic molecular H/He, single s Gaussian with exponent 0.5/1.25, SD,
    one up and zero/one down electron, no Jastrow/ECP. Explicit coordinates
    are (0.4,0.7,-0.2) and (-0.6,-0.3,0.8); no conditioning selection or RNG
    for positions. The actual production substep starts from a fresh inverse,
    identity mesh, alat=0.15/0.3 and fixed key 3. The short/long tau exercise
    respectively exhausted-time/no-hop and a genuine hop with remaining time.
    The reference uses Gaussian kinetic plus Coulomb energy and analytical
    Gaussian ratios for the six-neighbour escape rate, not production energy
    or ratio kernels. Each piece is evaluated in its production zone
    dtype (orbital values in ao_eval, derivatives in ao_grad_lap, value ratios
    in det_ratio, Coulomb in coulomb), never as an fp64 reference for mixed. Inverting the exponential CDF on the actual specified
    uniform gives elapsed time; exp(-elapsed*energy) gives the weight.
    Each mode uses its own existing central strict zones. This is deterministic
    finite-input physics/propagation evidence, not a ground-state estimate,
    distribution-fit or stochastic alat-extrapolation acceptance criterion.
    """
    _compare_substep(charge, exponent, alat, tau)


@pytest.mark.parametrize("fault", ["energy", "escape", "weight"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
def test_gaussian_projection_reference_rejects_production_fault(fault):
    """Changing the actual core energy, inverse CDF or weight exponent fails its independent reference."""
    _compare_substep(2, 0.5, 0.3, 0.2)
    old, new, message = {
        "energy": ("        e_L_meas,", "        2 * e_L_meas,", "local-energy"),
        "escape": ("jnp.log(1 - xi)", "jnp.log(xi)", "remaining-time"),
        "weight": ("jnp.exp(-tau_update * jnp.real(e_L))", "jnp.exp(-2 * tau_update * jnp.real(e_L))", "branching-weight"),
    }[fault]
    source = textwrap.dedent(inspect.getsource(gfmc._projection_t_core))
    assert source.count(old) == 1
    namespace = dict(vars(gfmc))
    exec(compile(ast.parse(source.replace(old, new)), "<fault-injected-gfmc-core>", "exec"), namespace)
    broken = jax.jit(namespace["_projection_t_core"], static_argnums=(12, 13))
    with pytest.raises(AssertionError, match=message):
        _compare_substep(2, 0.5, 0.3, 0.2, core=broken)

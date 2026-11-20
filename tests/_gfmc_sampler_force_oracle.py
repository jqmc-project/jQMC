"""Observe actual GFMC inputs and independently differentiate warped primal values."""

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

from functools import partial
import sys

import jax
import jax.numpy as jnp
import numpy as np

from jqmc._precision import get_dtype_jnp
from jqmc.jqmc_mcmc import _inject_active_positions
from jqmc.wavefunction import evaluate_ln_wavefunction
from tests.test_jqmc_kernel.test_gfmc_force_analytic import _e_L_t_ae_jnone


class ObserveBranching:
    """Record the pre-reconfiguration state without changing the RNG draw or stream.

    The caller-frame binding is intentional: the run method has no public
    per-walker output at this boundary. Fail if its expected inputs disappear;
    do not substitute post-resampling coordinates or stored averages.
    """

    def __init__(self, driver, algorithm):
        self.original = driver._branching_rng
        self.driver, self.algorithm, self.rows = driver, algorithm, []

    def random(self, *args, **kwargs):
        values = sys._getframe(1).f_locals
        assert values["self"] is self.driver, "snapshot must come from this driver's branching boundary"
        driver = self.driver
        row = {
            "r_up": np.asarray(driver._latest_r_up_carts).copy(),
            "r_dn": np.asarray(driver._latest_r_dn_carts).copy(),
            "weights": np.asarray(values["w_L_list"]).copy(),
            "energy": np.asarray(values["e_L_list"]).copy(),
        }
        if self.algorithm == "load-balanced":
            row["denominator"] = np.real(np.asarray(values["V_diag_list"])).copy() - driver._E_scf
        else:
            assert self.algorithm == "conventional"
            row["denominator"] = np.ones_like(row["weights"])
        self.rows.append(row)
        return self.original.random(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.original, name)


@partial(jax.jit, static_argnames=("use_swct",))
def primal_force_reference(ham, r_up_carts, r_dn_carts, *, use_swct):
    """Differentiate independent continuum energy and log(Psi * sqrt(Jacobian))."""
    nuclei = jnp.asarray(ham.structure_data.positions)

    def warp(points, delta):
        if not use_swct:
            return points
        differences = (points[:, None, :] - nuclei[None, :, :]).astype(get_dtype_jnp("swct"))
        radial = jnp.sum(differences**2, axis=-1) ** -2
        weights = radial / jnp.sum(radial, axis=1, keepdims=True)
        return points + weights @ delta

    def walker(r_up, r_dn):
        def energy(delta):
            moved = _inject_active_positions(ham, {"positions": nuclei + delta})
            return _e_L_t_ae_jnone(moved, warp(r_up, delta), warp(r_dn, delta), jnp.eye(3), 0.30)

        def log_amplitude(delta):
            moved = _inject_active_positions(ham, {"positions": nuclei + delta})
            result = evaluate_ln_wavefunction(moved.wavefunction_data, warp(r_up, delta), warp(r_dn, delta))
            for points in (r_up, r_dn):
                jacobian = jax.jacfwd(warp)(points, delta).reshape(points.size, points.size)
                result = result + 0.5 * jnp.linalg.slogdet(jacobian)[1]
            return result

        zero = jnp.zeros_like(nuclei)
        local_energy, hf = jax.value_and_grad(energy)(zero)
        pp = jax.grad(log_amplitude)(zero)
        gradients = jax.grad(lambda up, down: evaluate_ln_wavefunction(ham.wavefunction_data, up, down), argnums=(0, 1))(
            r_up, r_dn
        )
        distance = 1 / jnp.sqrt(sum(jnp.sum(value**2) for value in gradients))
        return local_energy, hf, pp, distance

    return jax.vmap(walker)(r_up_carts, r_dn_carts)


def pw_factor(distance, epsilon):
    """Explicit reviewed polynomial, not production's regularization helper."""
    if epsilon == 0:
        return np.ones_like(distance)
    ratio = distance / epsilon
    return np.where(ratio < 1, 7 * ratio**6 - 15 * ratio**4 + 9 * ratio**2, 1)


def force_moments(energy, hf, pp, fractions, factor):
    """Aggregate original walker values; never replace a cross moment by a product of means."""
    return tuple(np.sum((fractions * factor)[:, None, None] * value, axis=0) for value in (hf, pp, energy[:, None, None] * pp))

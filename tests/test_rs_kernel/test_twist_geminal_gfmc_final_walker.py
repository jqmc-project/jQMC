"""From-scratch recomputation of the final GFMC walker for complex AGP and AGPn at k != Gamma.

The driver twins (``test_twist_geminal_driver_twins.py``) and the native/XLA pairs
(``test_twist_geminal_driver_rust_xla.py``) compare two implementations of the same
fast-update chain with each other. This module instead takes the walker each
production GFMC run returns and recomputes, from scratch and without the production
geminal or determinant kernels, the quantities the run carried to that walker:

* the stored complex local energy of the last branching step (measured on the
  carried Sherman-Morrison inverse of the final projection interval) and its stored
  square, against ``-1/2 lap Psi / Psi`` from the complex Hessian of the independent
  ``psi_reference`` plus the production real Coulomb/Ewald potential;
* on the XLA route, the carried geminal inverse against the inverse of the
  independently assembled geminal matrix. The native route keeps its inverse inside
  the interval kernel, so only its stored energies cross the boundary.

One walker is used for both drivers so the stored energy of the last step is the
measurement at exactly the returned positions (with several walkers it is a weighted
mean and reconfiguration reorders walkers after the measurement). The open-shell
cases make the unpaired geminal columns live; the AGP lambda and the open-shell AGPn
unpaired block are complex, so a wrong conjugation in the fast update changes the
carried state.
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

import collections

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min
from jqmc.coulomb_potential import compute_coulomb_potential
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from tests.test_jqmc_kernel._twist_geminal_cases import build_case, kinetic_reference, reference_geminal
from tests.test_rs_kernel.test_twist_geminal_driver_rust_xla import _backend, _native_ctx

pytestmark = [pytest.mark.with_chdir]

jax.config.update("jax_enable_x64", True)

_CASES = ("agp", "agpn", "agp_open", "agpn_open")
_SEED = 3446
_STEPS = 4
_E_L_ZONES = ("ao_eval", "ao_grad_lap", "jastrow_eval", "jastrow_grad_lap", "det_grad_lap", "wf_kinetic", "coulomb")
_INVERSE_ZONES = ("ao_eval", "det_eval")


def _run(driver, hamiltonian, rust, counts):
    common = dict(
        hamiltonian_data=hamiltonian,
        num_walkers=1,
        num_gfmc_collect_steps=1,
        mcmc_seed=_SEED,
        alat=0.30,
        random_discretized_mesh=True,
        comput_position_deriv=False,
    )
    with _backend(rust), _native_ctx(counts):
        if driver == "n":
            run = GFMC_n(**common, num_mcmc_per_measurement=4, E_scf=-2.0)
        else:
            run = GFMC_t(**common, tau=0.05)
        run.run(num_mcmc_steps=_STEPS)
    return run


@pytest.mark.light
@pytest.mark.parametrize("backend", ["xla", "rust"])
@pytest.mark.parametrize("driver", ["n", "t"])
@pytest.mark.parametrize("case", _CASES)
def test_twist_geminal_gfmc_final_walker_matches_independent_recompute(case, driver, backend):
    """Stored e_L, e_L2 and the carried inverse of the final GFMC walker equal a from-scratch recomputation.

    H4_pbc_ae_k025_cart.h5 with J1+J2; complex dense AO AGP and four-MO AGPn, each closed shell
    (2 up / 2 down) and open shell (3 up / 1 down, two complex unpaired columns). Production
    GFMC_n (load-balanced, 4 projections per measurement, E_scf=-2) and GFMC_t (conventional,
    tau=0.05), one walker, seed 3446, alat=0.30, random mesh, 4 branching steps, on the XLA and
    on the native complex route (witnessed by the ctx spy). At the returned walker the stored
    complex e_L of the last step (Im outside the band) and its stored e_L2 equal the Hessian of
    the independent complex psi_reference plus the production Coulomb sum, and on XLA the
    carried geminal inverse equals the inverse of the independently assembled geminal matrix.
    Each comparison is strict in its own precision mode, Re and Im together.
    """
    hamiltonian = build_case(case)
    wavefunction = hamiltonian.wavefunction_data
    geminal = wavefunction.geminal_data
    rust = backend == "rust"
    counts = collections.Counter()
    run = _run(driver, hamiltonian, rust, counts)
    if rust:
        assert counts[f"run_gfmc_projection_{driver}_complex_batch"] == _STEPS, dict(counts)
    else:
        assert sum(counts.values()) == 0, dict(counts)
    if driver == "t":
        # More than the final no-hop sub-step: the last interval advanced the carried state.
        assert np.asarray(run._stored_average_projection_counter)[-1] > 1.0

    r_up = jnp.asarray(np.asarray(run._latest_r_up_carts)[0])
    r_dn = jnp.asarray(np.asarray(run._latest_r_dn_carts)[0])
    assert (r_up.shape[0], r_dn.shape[0]) == (geminal.num_electron_up, geminal.num_electron_dn)
    g_ref = np.asarray(reference_geminal(wavefunction, r_up, r_dn))
    assert np.all(np.isfinite(g_ref))
    if geminal.num_electron_up > geminal.num_electron_dn:
        unpaired_columns = g_ref[:, geminal.num_electron_dn :]
        atol, rtol = get_tolerance_min(_INVERSE_ZONES, "strict")
        assert np.max(np.abs(unpaired_columns)) > atol, f"{case}: unpaired geminal columns are not live"
    potential = compute_coulomb_potential(hamiltonian.coulomb_potential_data, r_up, r_dn, jnp.eye(3), wavefunction)
    e_l_ref = np.asarray([complex(jnp.sum(kinetic_reference(wavefunction, r_up, r_dn))) + float(potential)])

    stored_e_l = np.asarray(run._stored_e_L)[-1]
    stored_e_l2 = np.asarray(run._stored_e_L2)[-1]
    assert np.iscomplexobj(stored_e_l) and np.all(np.isfinite(stored_e_l)) and np.all(np.isfinite(e_l_ref))
    assert np.all(np.isfinite(stored_e_l2))
    atol, rtol = get_tolerance_min(_E_L_ZONES, "strict")
    band = atol + rtol * np.abs(np.real(e_l_ref))
    assert np.max(np.abs(stored_e_l - np.real(e_l_ref)) - band) > 0.0, f"{case}: Im e_L is inside the band"
    label = f"{case}: GFMC_{driver} {backend}"
    np.testing.assert_allclose(stored_e_l, e_l_ref, atol=atol, rtol=rtol, equal_nan=False, err_msg=f"{label} e_L")
    np.testing.assert_allclose(
        stored_e_l2, np.abs(e_l_ref) ** 2, atol=atol, rtol=rtol, equal_nan=False, err_msg=f"{label} e_L2"
    )
    if not rust:
        carried_inverse = np.asarray(getattr(run, f"_GFMC_{driver}__latest_A_old_inv"))[0]
        assert np.iscomplexobj(carried_inverse) and np.all(np.isfinite(carried_inverse))
        reference_inverse = np.linalg.inv(g_ref)
        atol, rtol = get_tolerance_min(_INVERSE_ZONES, "strict")
        np.testing.assert_allclose(
            carried_inverse, reference_inverse, atol=atol, rtol=rtol, equal_nan=False, err_msg=f"{label} inverse"
        )
    jax.clear_caches()

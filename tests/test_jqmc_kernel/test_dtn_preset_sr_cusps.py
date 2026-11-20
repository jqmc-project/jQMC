"""Shipped DTN presets retain their affine cusp constraints after real SR."""

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

from pathlib import Path

import jax
import numpy as np
import pytest

from jqmc import jastrow_factor as jf
from jqmc._precision import get_dtype_jnp, get_tolerance_min
from jqmc._setting import ATOL_NONVACUITY_DISTINCTION, RTOL_NONVACUITY_DISTINCTION
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jqmc_mcmc import MCMC
from jqmc.jqmc_tool import _build_jastrow_dtn_from_cli
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]
ROOT = Path(__file__).resolve().parents[2]


def _cusp_constraints(dtn, expected_en):
    atol, rtol = get_tolerance_min(("jastrow_eval",), level="strict")
    dtype = get_dtype_jnp("jastrow_eval")
    alpha = np.asarray(jf._dtn_inflate_alpha(dtn, dtype))
    beta = np.asarray(jf._dtn_inflate_beta(dtn, dtype))
    gamma = np.asarray(jf._dtn_inflate_gamma(dtn, dtype))
    assert all(np.all(np.isfinite(array)) for array in (alpha, beta, gamma))
    c = dtn.cutoff_smoothness
    length = float(dtn.L_u)
    # Slope at 0+ of the evaluators' (L - r)^C sum_k a_k r^k form.
    slope = length ** (c - 1) * (length * alpha[:, 1] - c * alpha[:, 0])
    np.testing.assert_allclose(slope, [0.25, 0.5], atol=atol, rtol=rtol)
    if dtn.use_chi:
        length = np.asarray(dtn.L_chi)
        slope = length ** (c - 1) * (length * beta[:, 1] - c * beta[:, 0])
        np.testing.assert_allclose(slope, expected_en, atol=atol, rtol=rtol)
    else:
        np.testing.assert_array_equal(beta, np.zeros_like(beta))
    if dtn.use_f:
        np.testing.assert_allclose(gamma, gamma.swapaxes(1, 2), atol=atol, rtol=rtol)
        for species in range(gamma.shape[0]):
            g = gamma[species]
            for k in range(2 * g.shape[0] - 1):
                value = sum(g[l, m, 1] for l in range(g.shape[0]) for m in range(g.shape[1]) if l + m == k)
                np.testing.assert_allclose(value, 0.0, atol=atol, rtol=rtol)
            for k in range(g.shape[1] + g.shape[2] - 1):
                value = sum(
                    c * g[0, m, n] - float(dtn.L_f[species]) * g[1, m, n]
                    for m in range(g.shape[1])
                    for n in range(g.shape[2])
                    if m + n == k
                )
                np.testing.assert_allclose(value, 0.0, atol=atol, rtol=rtol)
    else:
        np.testing.assert_array_equal(gamma, np.zeros_like(gamma))


@pytest.mark.parametrize("preset", ["u", "u_chi", "u_f", "u_chi_f"])
@pytest.mark.parametrize("fixture", ["H2_ae_ccpvdz_cart.h5", "H2_ecp_ccpvdz_cart.h5"])
def test_dtn_shipped_presets_keep_cusps_after_sr(fixture, preset):
    """Every CLI DTN preset makes a live SR step and preserves AE/ECP cusp slopes.

    Two committed H2 Cartesian inputs, seed42, two walkers and thirty actual
    MCMC steps exercise the unchanged SR update. Independent polynomial
    identities check production-inflated u/chi/f coefficients before and after
    the update; ECP chi has zero slope, AE chi has minus the nuclear charge.
    Disabled channels have no parameters or inflated contribution. No energy
    window or statistical test is used.
    """
    sd, _, _, _, gd, cd = read_trexio_file(str(ROOT / "tests/trexio_example_files" / fixture), store_tuple=True)
    dtn = _build_jastrow_dtn_from_cli(
        sd,
        jastrow_dtn=preset,
        coulomb_potential_data=cd,
        dtn_n_u=4,
        dtn_n_chi=4,
        dtn_n_eN=2,
        dtn_n_ee=2,
        dtn_L_u=2.0,
        dtn_L_chi=2.0,
        dtn_L_f=2.0,
    )
    flags = tuple(name in preset.split("_") for name in ("u", "chi", "f"))
    assert (dtn.use_u, dtn.use_chi, dtn.use_f) == flags
    before = [np.array(getattr(dtn, name)) for name in ("alpha_indep", "beta_indep", "gamma_indep")]
    expected_en = -np.where(np.asarray(cd.pseudized_per_atom), 0.0, np.asarray(cd._effective_charges))[:1]
    _cusp_constraints(dtn, expected_en)
    ham = Hamiltonian_data(
        structure_data=sd,
        coulomb_potential_data=cd,
        wavefunction_data=Wavefunction_data(geminal_data=gd, jastrow_data=jf.Jastrow_data(jastrow_dtn_data=dtn)),
    )
    sampler = MCMC(
        hamiltonian_data=ham,
        Dt=2.0,
        mcmc_seed=42,
        epsilon_AS=1e-6,
        num_walkers=2,
        comput_position_deriv=False,
        comput_log_WF_param_deriv=True,
        comput_e_L_param_deriv=True,
        random_discretized_mesh=True,
    )
    sampler.run_optimize(
        num_mcmc_steps=30,
        num_opt_steps=1,
        num_mcmc_warmup_steps=0,
        num_mcmc_bin_blocks=1,
        opt_J1_param=False,
        opt_J2_param=False,
        opt_J3_param=False,
        opt_geminal=False,
        optimizer_kwargs={"method": "sr", "delta": 1e-3, "epsilon": 1e-3, "cg_flag": True},
    )
    after = sampler.hamiltonian_data.wavefunction_data.jastrow_data.jastrow_dtn_data
    for flag, initial, name in zip(flags, before, ("alpha_indep", "beta_indep", "gamma_indep"), strict=True):
        value = np.asarray(getattr(after, name))
        assert np.all(np.isfinite(value))
        if flag:
            assert not np.allclose(initial, value, atol=ATOL_NONVACUITY_DISTINCTION, rtol=RTOL_NONVACUITY_DISTINCTION), name
        else:
            np.testing.assert_array_equal(initial, value)
    _cusp_constraints(after, expected_en)
    jax.clear_caches()

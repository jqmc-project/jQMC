"""Systematic detection of accidental real-in-complex contamination.

The complex-wavefunction contract requires a test mechanism that catches every silent
``jnp.asarray(complex_value, dtype=real_dtype)`` cast on the complex
Bloch path. A Wirtinger sanity test plus an ``|Im<E_L>|`` runtime warning
is INSUFFICIENT: a dtype hardcast inside the kinetic / streaming /
fast-update chain produces ``Im = 0`` walker-by-walker (the ``Im`` is
dropped at the cast), so the ``|Im<E_L>|`` guard reports ``OK`` while the real part is silently wrong. We need a
mechanism that fires at the cast site itself, not downstream.

The mechanism (two halves):

* ``conftest._strict_complex_cast_guard`` (autouse): promotes
  ``ComplexWarning`` to a test failure session-wide. Real-path
  tests cannot trigger it (no complex arrays anywhere); complex-path
  tests trip on any dtype hardcast that drops ``Im``. This gives every
  complex-touching test in the suite -- existing OR new -- automatic
  contamination coverage for free.

* This file: focused tests on the lithium_k025 fixture that exercise the
  five main chains (``compute_local_energy``, ``MCMC.run``,
  ``MCMC.get_E``, ``MCMC.get_aF``, ``MCMC.get_gF``) so a fresh checkout
  has end-to-end coverage even before other tests touch complex.

Fixture: ``lithium_k025.h5`` (single non-Gamma twist, complex Bloch MOs
with ``max |Im mo_coefficients| ~ 0.18``, Convention B
``mo_dn = conj(mo_up)``). The MO is the converged ccECP HF wavefunction;
the deterministic identity-level correctness tests (``psi_dn(r) ==
conj(psi_up(r))`` etc.) cover the Convention-B path. The matching
end-to-end ``VMC(no-Jastrow) ~ E_HF`` stochastic check is intentionally
NOT in the unit-test suite -- it runs as production validation.
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

import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy
from jqmc.jastrow_factor import Jastrow_data
from jqmc.jqmc_mcmc import MCMC
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data


@pytest.fixture(scope="module")
def lithium_k025_ham():
    """Hamiltonian_data for the single-twist non-Gamma lithium fixture.

    This is the consumer-side workhorse fixture for the
    contamination guards: it has a complex Bloch AO leaf with
    ``max |Im mo_coefficients| ~ 0.18`` and a complex Hermitian
    ``lambda_matrix``, so any ``jnp.asarray(complex, dtype=real)`` cast
    on the chain fires ComplexWarning when promoted to an error by
    the conftest autouse fixture.
    """
    fixture = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "trexio_example_files",
        "lithium_k025.h5",
    )
    if not os.path.exists(fixture):
        pytest.skip(f"missing fixture {fixture}")
    out = read_trexio_file(trexio_file=fixture, store_tuple=True)
    structure_data, _, _, _, geminal_mo_data, coulomb_potential_data = out
    ham = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=Wavefunction_data(
            jastrow_data=Jastrow_data(),  # no Jastrow: VMC(HF Slater) == HF
            geminal_data=geminal_mo_data,
        ),
    )
    ham.sanity_check()
    return ham


# ---------- systematic guards (no Im-truncation) ----------


class TestComplexBlochPath:
    @pytest.mark.heavy
    def test_no_complex_truncation_compute_local_energy(self, lithium_k025_ham):
        """``compute_local_energy`` at a few fixed walker positions must not
        raise ``ComplexWarning`` -- promotes every silent complex-to-real
        cast in the kinetic / Coulomb / ECP chain to a test failure.

        Reproducibility: walker positions seeded by ``np.random.default_rng``
        so the test exercises a deterministic subset of trace branches.
        """
        n_up = int(lithium_k025_ham.wavefunction_data.geminal_data.num_electron_up)
        n_dn = int(lithium_k025_ham.wavefunction_data.geminal_data.num_electron_dn)
        rng = np.random.default_rng(0)
        RT = jnp.eye(3)
        for _ in range(3):
            r_up = jnp.asarray(rng.standard_normal((n_up, 3)))
            r_dn = jnp.asarray(rng.standard_normal((n_dn, 3)))
            e_L = compute_local_energy(lithium_k025_ham, r_up, r_dn, RT)
            assert np.isfinite(complex(e_L).real)

    @pytest.mark.heavy
    def test_no_complex_truncation_mcmc_run(self, lithium_k025_ham):
        """``MCMC.run`` exercises the streaming-state advance + fast-update
        chain on top of ``compute_local_energy``. A short run is enough to
        pin every cast site -- the JAX jaxpr trace is built once and reused
        for every step, so the first iteration is the worst-case for
        ComplexWarning emission."""
        mcmc = MCMC(
            hamiltonian_data=lithium_k025_ham,
            num_walkers=2,
            mcmc_seed=12345,
            epsilon_AS=0.0,
        )
        mcmc.run(num_mcmc_steps=3)

    @pytest.mark.heavy
    def test_no_complex_truncation_mcmc_get_E(self, lithium_k025_ham):
        """``MCMC.get_E`` exercises the binning / jackknife reduction on
        complex ``e_L``. The R6.1 ``.real`` projection at the return is
        intentional and lives downstream of the binning; any cast
        UPSTREAM of that projection fires ComplexWarning here."""
        mcmc = MCMC(
            hamiltonian_data=lithium_k025_ham,
            num_walkers=2,
            mcmc_seed=12345,
            epsilon_AS=0.0,
        )
        mcmc.run(num_mcmc_steps=30)
        E, Estd, Var, Vstd = mcmc.get_E(num_mcmc_warmup_steps=5, num_mcmc_bin_blocks=5)
        assert np.isfinite(E)
        assert np.isfinite(Estd)

    @pytest.mark.heavy
    def test_no_complex_truncation_mcmc_get_aF(self, lithium_k025_ham):
        """``MCMC.get_aF`` exercises the atomic-force chain (R6.1 ``.real``
        on the assembled force, ``np.abs(centered)**2`` variance).
        ``comput_position_deriv=True`` is required to enable the force
        grad buffers in ``MCMC.run``."""
        mcmc = MCMC(
            hamiltonian_data=lithium_k025_ham,
            num_walkers=2,
            mcmc_seed=12345,
            epsilon_AS=0.0,
            comput_position_deriv=True,
        )
        mcmc.run(num_mcmc_steps=30)
        f_mean, f_std = mcmc.get_aF(num_mcmc_warmup_steps=5, num_mcmc_bin_blocks=5)
        assert np.all(np.isfinite(f_mean))
        assert np.all(np.isfinite(f_std))

    @pytest.mark.heavy
    def test_no_complex_truncation_mcmc_get_gF(self, lithium_k025_ham):
        """``MCMC.get_gF`` exercises the generalised-force chain
        (``conj(dO)`` einsum + per-column real projection). Requires
        ``comput_log_WF_param_deriv=True`` to populate the per-block grad
        buffers; ``blocks`` is the variational-parameter block list, fetched
        from the wavefunction (matches what the production SR loop passes)."""
        mcmc = MCMC(
            hamiltonian_data=lithium_k025_ham,
            num_walkers=2,
            mcmc_seed=12345,
            epsilon_AS=0.0,
            comput_log_WF_param_deriv=True,
        )
        mcmc.run(num_mcmc_steps=30)
        blocks = lithium_k025_ham.wavefunction_data.get_variational_blocks(
            state_token=("test", 0),
            opt_geminal=True,
        )
        mo_up = next(block for block in blocks if block.name == "mo_coefficients_up")
        assert (mo_up.field, mo_up.derivative_policy, mo_up.sync_scope, mo_up.tangent_spec.kind) == (
            "complex",
            "wirtinger_pair",
            "sampler",
            "s_stiefel_rows",
        )
        gF_mean, gF_std = mcmc.get_gF(num_mcmc_warmup_steps=5, num_mcmc_bin_blocks=5, blocks=blocks)
        assert np.all(np.isfinite(gF_mean))
        assert np.all(np.isfinite(gF_std))


# No stochastic ``VMC(no-Jastrow) ~ E_HF within N*sigma'' check is made here,
# because a Monte-Carlo expectation is non-deterministic by construction: PRNG seed, walker count, step
# count, and autocorrelation tail events all conspire to let a real bug
# hide inside the sigma band (a 0.046 Ha variational violation can pass
# as 1.34 sigma against a 5 sigma tolerance). End-to-end VMC vs reference-E
# checks live in production validation, not in the unit-test suite.
# Correctness of the Convention-B complex-Bloch path is guarded by
# deterministic operator-level identity tests instead.

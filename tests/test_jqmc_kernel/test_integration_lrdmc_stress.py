"""Regression tests for the PBC stress / pressure pipeline (LRDMC).

Mirror of :py:mod:`tests.test_jqmc_kernel.test_integration_mcmc_stress` for the LRDMC sampler
(:py:class:`jqmc.jqmc_gfmc.GFMC_t`). The infrastructure tests
(Ewald JAX vs numpy, Cauchy-Born linearity, translation invariance)
are sampler-independent and live in the MCMC file; here we cover the
five LRDMC-specific contracts:

* ``test_aT_aP_diagonal_consistency_lrdmc``
* ``test_aT_aP_molecular_zeros_lrdmc``
* ``test_get_aT_shape_pbc_vs_molecular_lrdmc``
* ``test_stress_HF_molecular_buffer_zero_lrdmc``
* ``test_stress_PP_buffer_populated_pbc_lrdmc``

All tolerances are sourced from :py:func:`jqmc._precision.get_tolerance`
(no hardcoded numerical thresholds).
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

import jax
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

# Periodic PBC fixture for stress: ``lithium.h5`` (2-atom Li AE cell, num_ao=26,
# num_ao_kernel=7150). silicon.h5 (8-atom ccECP-ccpvtz) was the original choice
# but its force/stress-grad path hits 67+ GB AD-tape peak per walker on a 256 GB
# host, causing OOM in CI. lithium retains
# the PBC stress code-path coverage (kernel branches on ``structure_data.pbc_flag``
# only) at ~1/16 the AD-tape footprint of silicon. Re-evaluate after the AD-tape
# memory architecture is addressed.
PERIODIC_FIXTURE = "lithium.h5"
PERIODIC_NON_LOCAL_MOVE = "tmove"  # Li ccECP -> T-move required
# AE (no ECP) -> the T-move proposal set is empty, so "tmove" is a bit-exact no-op.
# ``None`` is rejected by the _GFMCBase.__init__ validation: the native GFMC
# chunk coerces non_local_move to ``dltmove == (value == "dltmove")``, so an unvalidated
# ``None`` would silently run tmove on the native chunk while XLA's _projection_t_core
# raised -- both backends now reject any value outside {"tmove", "dltmove"}. The sibling
# test_jqmc_gfmc adopts the same convention (omits the kwarg -> the "tmove" default) for
# its AE rows; passing "tmove" explicitly here is equivalent and keeps the molecular
# short-circuit (sigma == 0, gated on pbc_flag) unchanged.
MOLECULAR_FIXTURE = "H2_ae_ccpvdz_cart.h5"
MOLECULAR_NON_LOCAL_MOVE = "tmove"

# LRDMC step counts kept conservative: stress kernels are exercised across
# branching boundaries within a few steps (the buffer-population test is
# the only sampling-dependent check). The collect-steps slice
# (``num_gfmc_collect_steps``) trims the initial transient from
# ``stress_HF`` / ``stress_PP`` accessors, so a 10-step run with 5
# collect-steps leaves 5 usable samples for binning.
_NUM_WALKERS = 2
_NUM_STEPS = 10
_NUM_COLLECT = 5
_NUM_WARMUP = 1
_NUM_BINS = 2


def _read_fixture(trexio_file: str):
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file)
    return read_trexio_file(trexio_file=path, store_tuple=True)


def _build_hamiltonian(trexio_file: str) -> Hamiltonian_data:
    """Build a minimal (no-Jastrow) :py:class:`Hamiltonian_data` for fast smoke runs."""
    (structure_data, _aos_data, _, _, geminal_mo_data, coulomb_potential_data) = _read_fixture(trexio_file)
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=None,
        jastrow_three_body_data=None,
    )
    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    return Hamiltonian_data.build(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )


def _run_lrdmc(trexio_file: str, non_local_move, num_steps: int = _NUM_STEPS) -> GFMC_t:
    """Construct + run a short LRDMC chain with ``comput_position_deriv=True``."""
    hamiltonian_data = _build_hamiltonian(trexio_file)
    gfmc = GFMC_t(
        hamiltonian_data=hamiltonian_data,
        num_walkers=_NUM_WALKERS,
        num_gfmc_collect_steps=_NUM_COLLECT,
        mcmc_seed=34356,
        tau=0.10,
        alat=0.30,
        non_local_move=non_local_move,
        comput_position_deriv=True,
        comput_stress=True,
        epsilon_PW=1.0e-2,
        use_swct=True,
    )
    gfmc.run(num_mcmc_steps=num_steps)
    return gfmc


# Projections per measurement for the GFMC_n (lrdmc-bra) branching chain.
# Kept small so the short smoke run stays fast while still crossing a
# branching boundary (where the per-step stress write happens).
_NUM_PROJ_PER_MEAS = 2


def _run_lrdmc_n(trexio_file: str, non_local_move, num_steps: int = _NUM_STEPS) -> GFMC_n:
    """Construct + run a short GFMC_n (lrdmc-bra) chain with ``comput_position_deriv=True``.

    Mirror of :py:func:`_run_lrdmc` but driving the branching (fixed survived
    walker) GFMC_n sampler instead of the fixed-tau GFMC_t one.
    """
    hamiltonian_data = _build_hamiltonian(trexio_file)
    gfmc = GFMC_n(
        hamiltonian_data=hamiltonian_data,
        num_walkers=_NUM_WALKERS,
        num_mcmc_per_measurement=_NUM_PROJ_PER_MEAS,
        num_gfmc_collect_steps=_NUM_COLLECT,
        mcmc_seed=34356,
        E_scf=0.0,
        alat=0.30,
        non_local_move=non_local_move,
        comput_position_deriv=True,
        comput_stress=True,
        epsilon_PW=1.0e-2,
        use_swct=True,
    )
    gfmc.run(num_mcmc_steps=num_steps)
    return gfmc


# ---------------------------------------------------------------------------
# Mirrors of the MCMC contracts
# ---------------------------------------------------------------------------


class TestGFMCTStress:
    @pytest.mark.heavy
    def test_aT_aP_diagonal_consistency_lrdmc(self):
        """``get_aP()[i] == -get_aT()[i, i]`` (bit-identical definitional check).

        Pure slice + sign-flip relationship between the public LRDMC APIs ---
        no statistics involved; ``np.testing.assert_array_equal`` is the
        appropriate strictest check.
        """
        gfmc = _run_lrdmc(PERIODIC_FIXTURE, PERIODIC_NON_LOCAL_MOVE)
        sigma_mean, sigma_std = gfmc.get_aT(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        P_mean, P_std = gfmc.get_aP(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        assert sigma_mean.shape == (3, 3)
        assert sigma_std.shape == (3, 3)
        assert P_mean.shape == (3,)
        assert P_std.shape == (3,)
        np.testing.assert_array_equal(P_mean, -np.diag(sigma_mean))
        np.testing.assert_array_equal(P_std, np.diag(sigma_std))

    @pytest.mark.heavy
    def test_aT_aP_molecular_zeros_lrdmc(self):
        """Molecular fixture: LRDMC ``get_aT`` / ``get_aP`` return strict zeros.

        Molecular short-circuit. Buffer-level zero is
        checked separately by :py:func:`test_stress_HF_molecular_buffer_zero_lrdmc`.
        """
        gfmc = _run_lrdmc(MOLECULAR_FIXTURE, MOLECULAR_NON_LOCAL_MOVE)
        sigma_mean, sigma_std = gfmc.get_aT(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        P_mean, P_std = gfmc.get_aP(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        assert sigma_mean.shape == (3, 3) and sigma_std.shape == (3, 3)
        assert P_mean.shape == (3,) and P_std.shape == (3,)
        np.testing.assert_array_equal(sigma_mean, np.zeros((3, 3)))
        np.testing.assert_array_equal(sigma_std, np.zeros((3, 3)))
        np.testing.assert_array_equal(P_mean, np.zeros((3,)))
        np.testing.assert_array_equal(P_std, np.zeros((3,)))

    @pytest.mark.heavy
    def test_get_aT_shape_pbc_vs_molecular_lrdmc(self):
        """API contract: LRDMC ``get_aT`` / ``get_aP`` shapes are sampler-uniform.

        Downstream code (logging, HDF5 save, NPT helpers) reads the sigma tensor
        as a fixed-shape ``(3, 3)`` (and ``(3,)`` for pressure) regardless of
        ``pbc_flag``; this test pins that contract for LRDMC.
        """
        gfmc_pbc = _run_lrdmc(PERIODIC_FIXTURE, PERIODIC_NON_LOCAL_MOVE)
        gfmc_mol = _run_lrdmc(MOLECULAR_FIXTURE, MOLECULAR_NON_LOCAL_MOVE)
        sigma_pbc, _ = gfmc_pbc.get_aT(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        sigma_mol, _ = gfmc_mol.get_aT(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        P_pbc, _ = gfmc_pbc.get_aP(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        P_mol, _ = gfmc_mol.get_aP(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        assert sigma_pbc.shape == sigma_mol.shape == (3, 3)
        assert P_pbc.shape == P_mol.shape == (3,)
        assert np.isfinite(sigma_pbc).all(), "periodic sigma must be finite"
        assert (sigma_mol == 0).all(), "molecular sigma must be strict zero"

    @pytest.mark.heavy
    def test_stress_HF_molecular_buffer_zero_lrdmc(self):
        """Molecular LRDMC: per-step sigma_HF / sigma_PP buffers are strict zero.

        Confirms the molecular short-circuit at the storage level
        for LRDMC --- the kernel branch in
        :py:meth:`GFMC_t.run` is gated on ``structure_data.pbc_flag``, and
        the per-step write replaces the zero-padded slot with the
        branching-weighted sigma only on the periodic path.
        """
        gfmc = _run_lrdmc(MOLECULAR_FIXTURE, MOLECULAR_NON_LOCAL_MOVE)
        stress_HF = gfmc.stress_HF
        stress_PP = gfmc.stress_PP
        E_L_stress_PP = gfmc.E_L_stress_PP
        # GFMC_t accessors slice off the first ``num_gfmc_collect_steps``; the
        # remaining buffer carries one branching-weighted sample per recorded step.
        assert stress_HF.ndim == 4 and stress_HF.shape[1:] == (1, 3, 3)
        assert stress_PP.shape == stress_HF.shape
        assert E_L_stress_PP.shape == stress_HF.shape
        np.testing.assert_array_equal(stress_HF, 0)
        np.testing.assert_array_equal(stress_PP, 0)
        np.testing.assert_array_equal(E_L_stress_PP, 0)

    @pytest.mark.heavy
    def test_stress_PP_buffer_populated_pbc_lrdmc(self):
        """Periodic LRDMC: sigma_HF / sigma_PP buffers carry non-trivial cotangents.

        Sanity that the LRDMC strained-energy kernel (the ``_t`` variant
        threaded through :py:meth:`GFMC_t.run` with ``non_local_move`` and
        ``alat``) actually fires on the PBC path --- a silently zero buffer
        would mask an entire half of the covariance-corrected estimator
        despite the construction-time kernel compile succeeding.
        """
        gfmc = _run_lrdmc(PERIODIC_FIXTURE, PERIODIC_NON_LOCAL_MOVE)
        stress_HF = gfmc.stress_HF
        stress_PP = gfmc.stress_PP
        E_L_stress_PP = gfmc.E_L_stress_PP
        assert stress_HF.ndim == 4 and stress_HF.shape[1:] == (1, 3, 3)
        assert stress_PP.shape == stress_HF.shape
        assert E_L_stress_PP.shape == stress_HF.shape
        assert not np.isnan(stress_HF).any(), "sigma_HF has NaN entries"
        assert not np.isnan(stress_PP).any(), "sigma_PP has NaN entries"
        assert not np.isnan(E_L_stress_PP).any(), "E_L sigma_PP has NaN entries"
        assert np.max(np.abs(stress_HF)) > 0.0, "sigma_HF buffer is identically zero on the periodic path"
        assert np.max(np.abs(stress_PP)) > 0.0, "sigma_PP buffer is identically zero on the periodic path"


class TestGFMCNStress:
    """GFMC_n (lrdmc-bra) stress mirror of :py:class:`TestGFMCTStress`.

    Same fixtures, same shape / strict-zero / finite-and-nonzero assertions,
    driving the branching GFMC_n sampler (via :py:func:`_run_lrdmc_n`) so the
    lrdmc-bra ``get_aT`` / ``get_aP`` reach parity with lrdmc-tau.
    """

    @pytest.mark.heavy
    def test_aT_aP_diagonal_consistency_lrdmc_n(self):
        """``get_aP()[i] == -get_aT()[i, i]`` (bit-identical definitional check)."""
        gfmc = _run_lrdmc_n(PERIODIC_FIXTURE, PERIODIC_NON_LOCAL_MOVE)
        sigma_mean, sigma_std = gfmc.get_aT(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        P_mean, P_std = gfmc.get_aP(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        assert sigma_mean.shape == (3, 3)
        assert sigma_std.shape == (3, 3)
        assert P_mean.shape == (3,)
        assert P_std.shape == (3,)
        np.testing.assert_array_equal(P_mean, -np.diag(sigma_mean))
        np.testing.assert_array_equal(P_std, np.diag(sigma_std))

    @pytest.mark.heavy
    def test_aT_aP_molecular_zeros_lrdmc_n(self):
        """Molecular fixture: GFMC_n ``get_aT`` / ``get_aP`` return strict zeros."""
        gfmc = _run_lrdmc_n(MOLECULAR_FIXTURE, MOLECULAR_NON_LOCAL_MOVE)
        sigma_mean, sigma_std = gfmc.get_aT(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        P_mean, P_std = gfmc.get_aP(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        assert sigma_mean.shape == (3, 3) and sigma_std.shape == (3, 3)
        assert P_mean.shape == (3,) and P_std.shape == (3,)
        np.testing.assert_array_equal(sigma_mean, np.zeros((3, 3)))
        np.testing.assert_array_equal(sigma_std, np.zeros((3, 3)))
        np.testing.assert_array_equal(P_mean, np.zeros((3,)))
        np.testing.assert_array_equal(P_std, np.zeros((3,)))

    @pytest.mark.heavy
    def test_get_aT_shape_pbc_vs_molecular_lrdmc_n(self):
        """API contract: GFMC_n ``get_aT`` / ``get_aP`` shapes are sampler-uniform."""
        gfmc_pbc = _run_lrdmc_n(PERIODIC_FIXTURE, PERIODIC_NON_LOCAL_MOVE)
        gfmc_mol = _run_lrdmc_n(MOLECULAR_FIXTURE, MOLECULAR_NON_LOCAL_MOVE)
        sigma_pbc, _ = gfmc_pbc.get_aT(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        sigma_mol, _ = gfmc_mol.get_aT(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        P_pbc, _ = gfmc_pbc.get_aP(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        P_mol, _ = gfmc_mol.get_aP(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        assert sigma_pbc.shape == sigma_mol.shape == (3, 3)
        assert P_pbc.shape == P_mol.shape == (3,)
        assert np.isfinite(sigma_pbc).all(), "periodic sigma must be finite"
        assert (sigma_mol == 0).all(), "molecular sigma must be strict zero"

    @pytest.mark.heavy
    def test_stress_HF_molecular_buffer_zero_lrdmc_n(self):
        """Molecular GFMC_n: per-step sigma_HF / sigma_PP buffers are strict zero.

        Confirms the molecular short-circuit at the storage level for the
        branching sampler --- the kernel branch in :py:meth:`GFMC_n.run` is
        gated on ``structure_data.pbc_flag``.
        """
        gfmc = _run_lrdmc_n(MOLECULAR_FIXTURE, MOLECULAR_NON_LOCAL_MOVE)
        stress_HF = gfmc.stress_HF
        stress_PP = gfmc.stress_PP
        E_L_stress_PP = gfmc.E_L_stress_PP
        assert stress_HF.ndim == 4 and stress_HF.shape[1:] == (1, 3, 3)
        assert stress_PP.shape == stress_HF.shape
        assert E_L_stress_PP.shape == stress_HF.shape
        np.testing.assert_array_equal(stress_HF, 0)
        np.testing.assert_array_equal(stress_PP, 0)
        np.testing.assert_array_equal(E_L_stress_PP, 0)

    @pytest.mark.heavy
    def test_stress_PP_buffer_populated_pbc_lrdmc_n(self):
        """Periodic GFMC_n: sigma_HF / sigma_PP buffers carry non-trivial cotangents.

        Sanity that the LRDMC strained-energy kernel (the ``_n`` variant
        threaded through :py:meth:`GFMC_n.run` with ``non_local_move`` and
        ``alat``) actually fires on the PBC branching path.
        """
        gfmc = _run_lrdmc_n(PERIODIC_FIXTURE, PERIODIC_NON_LOCAL_MOVE)
        stress_HF = gfmc.stress_HF
        stress_PP = gfmc.stress_PP
        E_L_stress_PP = gfmc.E_L_stress_PP
        assert stress_HF.ndim == 4 and stress_HF.shape[1:] == (1, 3, 3)
        assert stress_PP.shape == stress_HF.shape
        assert E_L_stress_PP.shape == stress_HF.shape
        assert not np.isnan(stress_HF).any(), "sigma_HF has NaN entries"
        assert not np.isnan(stress_PP).any(), "sigma_PP has NaN entries"
        assert not np.isnan(E_L_stress_PP).any(), "E_L sigma_PP has NaN entries"
        assert np.max(np.abs(stress_HF)) > 0.0, "sigma_HF buffer is identically zero on the periodic path"
        assert np.max(np.abs(stress_PP)) > 0.0, "sigma_PP buffer is identically zero on the periodic path"

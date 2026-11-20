"""Regression tests for the PBC stress / pressure pipeline (MCMC).

Covers:

* Definitional / API contract: ``test_aT_aP_diagonal_consistency_mcmc``,
  ``test_aT_aP_molecular_zeros_mcmc``,
  ``test_get_aT_shape_pbc_vs_molecular_mcmc``,
  ``test_stress_HF_molecular_buffer_zero_mcmc``,
  ``test_stress_PP_buffer_populated_pbc_mcmc``.

* Cauchy-Born / AD-pipeline correctness:
  ``test_stress_translation_invariance``,
  ``test_stress_HF_fd_vs_ad_per_walker``,
  ``test_ewald_ii_jax_vs_numpy``,
  ``test_accumulate_strain_grad_linearity``.

All tolerances are sourced from :py:func:`jqmc._precision.get_tolerance` /
:py:func:`jqmc._precision.get_tolerance_min` so the suite tracks the active
precision-zone configuration (no hardcoded numerical thresholds).
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
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_jqmc_kernel._gdd_fd_adapters import fd_case_parameters, record_fd_case

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import get_tolerance
from jqmc.coulomb_potential import (
    _compute_ewald_ii_energy,
    _compute_ewald_ii_energy_jax,
)
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy
from jqmc.jastrow_factor import Jastrow_data
from jqmc.derivatives._auto import _jit_vmap_local_stress_HF, _jit_vmap_local_stress_PP
from jqmc.jqmc_mcmc import MCMC
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

SILICON_FIXTURE = "silicon.h5"
MOLECULAR_FIXTURE = "H2_ae_ccpvdz_sphe.h5"

# Conservative MCMC step counts -- the stress tests need the kernel to fire at
# least once. The full-tensor value oracle for the stored stress buffers is
# ``tests/test_jqmc_kernel/test_mcmc_sampler_stress_reference.py``. 2 walkers /
# 5 steps keeps the per-test cost in the tens of seconds for silicon.
_NUM_WALKERS = 2
_NUM_STEPS = 5
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


def _run_mcmc(trexio_file: str, num_steps: int = _NUM_STEPS) -> MCMC:
    """Construct + run a short MCMC chain with ``comput_position_deriv=True``."""
    hamiltonian_data = _build_hamiltonian(trexio_file)
    mcmc = MCMC(
        hamiltonian_data=hamiltonian_data,
        Dt=2.0,
        mcmc_seed=34356,
        num_walkers=_NUM_WALKERS,
        comput_position_deriv=True,
        comput_stress=True,
        comput_log_WF_param_deriv=False,
        comput_e_L_param_deriv=False,
        epsilon_AS=1.0e-2,
    )
    mcmc.run(num_mcmc_steps=num_steps)
    return mcmc


# ---------------------------------------------------------------------------
# Definitional / API contract
# ---------------------------------------------------------------------------


class TestMCMCStressDefinition:
    @pytest.mark.heavy
    def test_aT_aP_diagonal_consistency_mcmc(self):
        """``get_aP()[i] == -get_aT()[i, i]`` to floating-point identity.

        Pure definitional consistency: ``get_aP`` is a thin slice of
        ``get_aT``'s diagonal with a sign flip. No statistics involved --- the
        check is bit-equal up to the round-off of one negation + indexing.
        """
        mcmc = _run_mcmc(SILICON_FIXTURE)
        sigma_mean, sigma_std = mcmc.get_aT(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        P_mean, P_std = mcmc.get_aP(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        assert sigma_mean.shape == (3, 3)
        assert sigma_std.shape == (3, 3)
        assert P_mean.shape == (3,)
        assert P_std.shape == (3,)
        np.testing.assert_array_equal(P_mean, -np.diag(sigma_mean))
        np.testing.assert_array_equal(P_std, np.diag(sigma_std))

    @pytest.mark.heavy
    def test_aT_aP_molecular_zeros_mcmc(self):
        """Molecular fixture: ``get_aT`` / ``get_aP`` return strict zeros of the right shape.

        Molecular short-circuit. The buffer-zero claim is
        covered separately by :py:func:`test_stress_HF_molecular_buffer_zero_mcmc`.
        """
        mcmc = _run_mcmc(MOLECULAR_FIXTURE, num_steps=3)
        sigma_mean, sigma_std = mcmc.get_aT(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        P_mean, P_std = mcmc.get_aP(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        assert sigma_mean.shape == (3, 3) and sigma_std.shape == (3, 3)
        assert P_mean.shape == (3,) and P_std.shape == (3,)
        np.testing.assert_array_equal(sigma_mean, np.zeros((3, 3)))
        np.testing.assert_array_equal(sigma_std, np.zeros((3, 3)))
        np.testing.assert_array_equal(P_mean, np.zeros((3,)))
        np.testing.assert_array_equal(P_std, np.zeros((3,)))

    @pytest.mark.heavy
    def test_get_aT_shape_pbc_vs_molecular_mcmc(self):
        """API contract: shapes are identical across periodic and molecular.

        Downstream consumers (logging, HDF5 save, NPT helpers) read
        ``aT`` / ``aP`` as a fixed-shape tensor regardless of ``pbc_flag``;
        this test pins the contract.
        """
        pbc_mcmc = _run_mcmc(SILICON_FIXTURE)
        mol_mcmc = _run_mcmc(MOLECULAR_FIXTURE, num_steps=3)
        sigma_pbc, _ = pbc_mcmc.get_aT(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        sigma_mol, _ = mol_mcmc.get_aT(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        P_pbc, _ = pbc_mcmc.get_aP(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        P_mol, _ = mol_mcmc.get_aP(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        assert sigma_pbc.shape == sigma_mol.shape == (3, 3)
        assert P_pbc.shape == P_mol.shape == (3,)
        assert np.isfinite(sigma_pbc).all(), "periodic sigma must be finite"
        assert (sigma_mol == 0).all(), "molecular sigma must be strict zero"

    @pytest.mark.heavy
    def test_stress_HF_molecular_buffer_zero_mcmc(self):
        """Molecular run: per-step sigma_HF buffer is strict zero (kernel call is skipped).

        Confirms the molecular short-circuit at the storage level
        --- the kernel branch in ``MCMC.run`` is gated on
        ``structure_data.pbc_flag``, and the buffer is zero-padded at
        allocation, so molecular runs must observably bypass the kernel.
        """
        mcmc = _run_mcmc(MOLECULAR_FIXTURE, num_steps=3)
        stress_HF = mcmc.stress_HF_stored
        stress_PP = mcmc.stress_PP_stored
        E_L_stress_PP = mcmc.E_L_stress_PP_stored
        assert stress_HF.shape == (3, _NUM_WALKERS, 3, 3)
        assert stress_PP.shape == stress_HF.shape
        assert E_L_stress_PP.shape == stress_HF.shape
        np.testing.assert_array_equal(stress_HF, 0)
        np.testing.assert_array_equal(stress_PP, 0)
        np.testing.assert_array_equal(E_L_stress_PP, 0)

    @pytest.mark.heavy
    def test_stress_PP_buffer_populated_pbc_mcmc(self):
        """Periodic run: sigma_PP buffer carries non-trivial cotangents (no NaN, not all zero).

        Sanity that the Pulay AD kernel actually fires on the PBC path --- a
        silently zero sigma_PP buffer would hide an entire half of the
        covariance-corrected estimator (S5).
        """
        mcmc = _run_mcmc(SILICON_FIXTURE)
        stress_HF = mcmc.stress_HF_stored
        stress_PP = mcmc.stress_PP_stored
        E_L_stress_PP = mcmc.E_L_stress_PP_stored
        assert stress_HF.shape == (_NUM_STEPS, _NUM_WALKERS, 3, 3)
        assert stress_PP.shape == stress_HF.shape
        assert E_L_stress_PP.shape == stress_HF.shape
        assert not np.isnan(stress_HF).any(), "sigma_HF has NaN entries"
        assert not np.isnan(stress_PP).any(), "sigma_PP has NaN entries"
        assert not np.isnan(E_L_stress_PP).any(), "E_L sigma_PP has NaN entries"
        assert np.max(np.abs(stress_HF)) > 0.0, "sigma_HF buffer is identically zero on the periodic path"
        assert np.max(np.abs(stress_PP)) > 0.0, "sigma_PP buffer is identically zero on the periodic path"


# ---------------------------------------------------------------------------
# AD-pipeline correctness
# ---------------------------------------------------------------------------


class TestMCMCStressAD:
    @pytest.mark.heavy
    def test_stress_translation_invariance(self):
        """``accumulate_strain_grad`` is invariant under rigid translation of all positions.

        Translating positions + lattice rows by a constant vector ``delta`` leaves
        the Cauchy-Born projection ``cell.T @ grad_cell + positions.T @ grad_positions``
        unchanged when ``grad_positions`` sums to zero (which it does for any
        translation-invariant energy by Noether's theorem applied per walker).

        Operates on a single ``jax.grad(ion-ion Ewald energy)`` call so the test
        is purely deterministic and cheap.
        """
        from jqmc.coulomb_potential import compute_ewald_potential_ii

        hamiltonian_data = _build_hamiltonian(SILICON_FIXTURE)
        structure_data = hamiltonian_data.structure_data

        def ii_only(h):
            return compute_ewald_potential_ii(h.coulomb_potential_data, h.coulomb_potential_data.ewald_data)

        grad_h = jax.grad(ii_only)(hamiltonian_data)
        sigma_orig = np.asarray(hamiltonian_data.accumulate_strain_grad(grad_h))

        # Translate every atomic position by ``shift`` (cell unchanged --- the
        # Cauchy-Born projection has both ``positions.T @ dE/dR`` and
        # ``cell.T @ dE/dcell`` terms; shifting positions alone exercises the
        # ``positions.T @ dE/dR`` cancellation that ``sum_I dE/dR_I = 0``
        # delivers by translation invariance).
        shift = np.array([0.5, -0.25, 0.75], dtype=np.float64)
        new_pos = structure_data.positions + shift[None, :]
        new_sd = structure_data.replace(positions=new_pos)
        new_cpd_sd = hamiltonian_data.coulomb_potential_data.structure_data.replace(positions=new_pos)
        new_cpd = hamiltonian_data.coulomb_potential_data.replace(structure_data=new_cpd_sd)
        new_ewald_sd = hamiltonian_data.coulomb_potential_data.ewald_data.structure_data.replace(positions=new_pos)
        new_ewald = hamiltonian_data.coulomb_potential_data.ewald_data.replace(structure_data=new_ewald_sd)
        new_cpd = new_cpd.replace(ewald_data=new_ewald)
        new_hamiltonian = hamiltonian_data.replace(structure_data=new_sd, coulomb_potential_data=new_cpd)

        grad_h_shift = jax.grad(ii_only)(new_hamiltonian)
        sigma_shift = np.asarray(new_hamiltonian.accumulate_strain_grad(grad_h_shift))

        atol, rtol = get_tolerance("ewald", "strict")
        np.testing.assert_allclose(sigma_shift, sigma_orig, atol=atol, rtol=rtol)

    @pytest.mark.heavy
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="MATH-DERIV-LOCAL-ENERGY-001",
    )
    def test_stress_HF_fd_vs_ad_per_walker(self):
        """Per-walker sigma_HF: jacrev AD vs central finite differences at one fixed (r_up, r_dn).

        Hits the full strained-energy kernel (electrons + cell + positions
        transformed by ``F = I + eps``; the ECP quadrature directions held
        fixed). Uses a single component
        ``eps_{00}`` and compares ``sigma_HF[0, 0]`` against the central FD. This
        FD strains through ``_apply_strain`` like the AD route. The full tensor,
        the Pulay part and a strain rebuilt without ``_apply_strain`` are checked
        in ``test_mcmc_sampler_stress_reference.py``.

        Tolerance: ``get_tolerance("local_energy", "medium")`` -- the FD path
        inherits all the per-walker rotation-noise of the analytic stress
        (MO coefficients are not rotation-equivariant in a fixed AO basis;
        see S4 implementation notes), so a "medium" zone tolerance is the
        appropriate ceiling.
        """
        hamiltonian_data = _build_hamiltonian(SILICON_FIXTURE)
        num_up = hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
        num_dn = hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
        rng = np.random.default_rng(2026)
        r_up = jnp.asarray(rng.uniform(0.0, 5.0, size=(num_up, 3)), dtype=jnp.float64)
        r_dn = jnp.asarray(rng.uniform(0.0, 5.0, size=(num_dn, 3)), dtype=jnp.float64)
        RT = jnp.eye(3, dtype=jnp.float64)

        sigma_HF = np.asarray(
            _jit_vmap_local_stress_HF(
                hamiltonian_data,
                r_up[None, ...],  # add walker axis
                r_dn[None, ...],
                RT[None, ...],
                jnp.zeros((3, 3), dtype=jnp.float64),
            )
        )[0]  # (3, 3)

        h_step = 1.0e-4

        def _e_at(eps_xx):
            eps_mat = jnp.zeros((3, 3), dtype=jnp.float64).at[0, 0].set(eps_xx)
            F = jnp.eye(3, dtype=jnp.float64) + eps_mat
            new_h = hamiltonian_data._apply_strain(F)
            return float(compute_local_energy(new_h, r_up @ F.T, r_dn @ F.T, RT))

        fd_xx = (_e_at(h_step) - _e_at(-h_step)) / (2.0 * h_step)
        ad_xx = float(sigma_HF[0, 0])

        atol, rtol = get_tolerance("local_energy", "medium")
        np.testing.assert_allclose(ad_xx, fd_xx, atol=atol, rtol=rtol)
        record_fd_case(
            "fd-ad-mcmc-stress",
            pytest_node_id="tests/test_jqmc_kernel/test_integration_mcmc_stress.py::TestMCMCStressAD::test_stress_HF_fd_vs_ad_per_walker",
            fixture_parameters=fd_case_parameters("fd-ad-mcmc-stress"),
            source_ids=("WF-SRC-SI-AE-SPHE-GAMMA",),
            transformations=("derivatives.fd.build-no-jastrow-hamiltonian.v1",),
            coordinates=(r_up, r_dn),
            coordinate_generator="numpy.random.Generator.uniform",
            coordinate_seed=2026,
            compared_pairs=((ad_xx, fd_xx),),
            atol=atol,
            rtol=rtol,
            zones=("local_energy",),
            level="medium",
            stencil="central-first-order-2",
            step=h_step,
            direction="eps-00",
        )

    @pytest.mark.heavy
    def test_ewald_ii_jax_vs_numpy(self):
        """``_compute_ewald_ii_energy_jax`` matches the numpy oracle bit-equal (fp64 round-off).

        Cross-validates the in-trace JAX path (S2) against the setup-time
        numpy implementation (legacy fast path / debug oracle). Both consume
        the same integer index sets, so the comparison isolates the
        JAX-vs-numpy reassociation noise.
        """
        hamiltonian_data = _build_hamiltonian(SILICON_FIXTURE)
        structure_data = hamiltonian_data.structure_data
        ewald_data = hamiltonian_data.coulomb_potential_data.ewald_data
        charges_np = np.asarray(hamiltonian_data.coulomb_potential_data._effective_charges, dtype=np.float64)
        positions_np = np.asarray(structure_data._positions_cart_np, dtype=np.float64)
        eta = float(ewald_data.eta)

        volume_np = float(np.abs(np.linalg.det(np.asarray(structure_data._cell_jnp, dtype=np.float64))))
        ii_numpy = _compute_ewald_ii_energy(
            charges_np,
            positions_np,
            eta,
            np.asarray(ewald_data.real_image_list, dtype=np.float64),
            np.asarray(ewald_data.g_vectors, dtype=np.float64),
            np.asarray(ewald_data.g_factors, dtype=np.float64),
            volume_np,
            np.asarray(structure_data._cell_jnp, dtype=np.float64),
        )
        ii_jax = float(
            _compute_ewald_ii_energy_jax(
                jnp.asarray(charges_np, dtype=jnp.float64),
                jnp.asarray(positions_np, dtype=jnp.float64),
                jnp.asarray(eta, dtype=jnp.float64),
                ewald_data._real_image_indices_jnp,
                ewald_data._g_indices_jnp,
                structure_data._cell_jnp.astype(jnp.float64),
            )
        )

        atol, rtol = get_tolerance("ewald", "strict")
        np.testing.assert_allclose(ii_jax, ii_numpy, atol=atol, rtol=rtol)

    @pytest.mark.heavy
    def test_accumulate_strain_grad_linearity(self):
        """``accumulate_strain_grad(scale * g) == scale * accumulate_strain_grad(g)``.

        The accumulator is a JAX-linear contraction
        (``cell.T @ grad_cell + positions.T @ grad_positions``) per leaf;
        linearity in the cotangent ``g`` is a structural property that
        catches accidental non-linearities introduced by future refactors
        (e.g. an absolute-value or square inside the contraction).
        """
        from jqmc.coulomb_potential import compute_ewald_potential_ii

        hamiltonian_data = _build_hamiltonian(SILICON_FIXTURE)

        def ii_only(h):
            return compute_ewald_potential_ii(h.coulomb_potential_data, h.coulomb_potential_data.ewald_data)

        grad_h = jax.grad(ii_only)(hamiltonian_data)
        sigma_single = np.asarray(hamiltonian_data.accumulate_strain_grad(grad_h))

        scale = 3.0
        grad_h_scaled = jax.tree_util.tree_map(lambda x: scale * x if hasattr(x, "shape") else x, grad_h)
        sigma_scaled = np.asarray(hamiltonian_data.accumulate_strain_grad(grad_h_scaled))

        atol, rtol = get_tolerance("ewald", "strict")
        np.testing.assert_allclose(sigma_scaled, scale * sigma_single, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Composite-AOs regression: _apply_strain must traverse AOs_composite_data
# children rather than calling .replace(structure_data=...) on the composite
# itself (which has no such field --- it exposes ``structure_data`` as a
# derived property via :py:attr:`AOs_composite_data.structure_data`).
#
# Calling .replace(structure_data=...) on the composite raises
#   TypeError: AOs_composite_data.__init__() got an unexpected keyword
#   argument 'structure_data'
# in determinant.py::_strain_orb. The other stress fixtures
# (silicon.h5 / H2_ae_ccpvdz_sphe.h5) carry a single pgto / sphe-gto leaf,
# so this branch is never hit by the rest of the suite.
# ---------------------------------------------------------------------------


def _build_composite_hamiltonian(trexio_file: str) -> Hamiltonian_data:
    """Wrap the silicon fixture's AO leaf in an ``AOs_composite_data`` shell.

    A single-child composite is enough to expose the bug: the strain code
    sees an :py:class:`AOs_composite_data` whose ``structure_data`` is a
    derived property, so the original ``.replace(structure_data=...)``
    call fails regardless of how many children the composite holds. Wrapping
    preserves ``num_ao`` and the MO coefficient shape, so no further
    geminal / coulomb surgery is required.
    """
    from jqmc.atomic_orbital import AOs_composite_data

    ham = _build_hamiltonian(trexio_file)
    gem = ham.wavefunction_data.geminal_data

    def _wrap(orb):
        if orb is None:
            return None
        if hasattr(orb, "aos_data"):
            composite = AOs_composite_data(children=(orb.aos_data,))
            composite.sanity_check()
            return orb.replace(aos_data=composite)
        composite = AOs_composite_data(children=(orb,))
        composite.sanity_check()
        return composite

    new_gem = gem.replace(
        orb_data_up_spin=_wrap(gem.orb_data_up_spin),
        orb_data_dn_spin=_wrap(gem.orb_data_dn_spin),
    )
    new_wf = ham.wavefunction_data.replace(geminal_data=new_gem)
    return ham.replace(wavefunction_data=new_wf)


class TestCompositeAO:
    @pytest.mark.heavy
    def test_stress_HF_composite_aos_kernel_runs(self):
        """``_jit_vmap_local_stress_HF`` must accept a composite-AOs geminal.

        Builds the silicon fixture and wraps its pgto AO leaf in a
        single-child :py:class:`AOs_composite_data`.
        ``AOs_composite_data.structure_data`` is a derived property, not a
        dataclass field, so ``determinant.py::_strain_orb`` cannot strain it
        through ``orb.aos_data.replace(structure_data=...)`` (a ``TypeError``);
        the helper recurses into ``children`` and the kernel must return a
        finite ``(3, 3)`` tensor.
        """
        hamiltonian_data = _build_composite_hamiltonian(SILICON_FIXTURE)
        num_up = hamiltonian_data.wavefunction_data.geminal_data.num_electron_up
        num_dn = hamiltonian_data.wavefunction_data.geminal_data.num_electron_dn
        rng = np.random.default_rng(2026)
        r_up = jnp.asarray(rng.uniform(0.0, 5.0, size=(num_up, 3)), dtype=jnp.float64)
        r_dn = jnp.asarray(rng.uniform(0.0, 5.0, size=(num_dn, 3)), dtype=jnp.float64)
        RT = jnp.eye(3, dtype=jnp.float64)

        sigma_HF = np.asarray(
            _jit_vmap_local_stress_HF(
                hamiltonian_data,
                r_up[None, ...],
                r_dn[None, ...],
                RT[None, ...],
                jnp.zeros((3, 3), dtype=jnp.float64),
            )
        )[0]
        assert sigma_HF.shape == (3, 3)
        assert np.isfinite(sigma_HF).all(), "sigma_HF has non-finite entries"

    @pytest.mark.heavy
    def test_stress_PP_composite_aos_kernel_runs(self):
        """``_jit_vmap_local_stress_PP`` must accept a composite-AOs wavefunction.

        Same regression as :py:func:`test_stress_HF_composite_aos_kernel_runs`
        but exercising the Pulay path (``_local_log_psi_strained`` ->
        ``Wavefunction_data._apply_strain`` -> ``Geminal_data._apply_strain``).
        """
        hamiltonian_data = _build_composite_hamiltonian(SILICON_FIXTURE)
        wf = hamiltonian_data.wavefunction_data
        num_up = wf.geminal_data.num_electron_up
        num_dn = wf.geminal_data.num_electron_dn
        rng = np.random.default_rng(7)
        r_up = jnp.asarray(rng.uniform(0.0, 5.0, size=(num_up, 3)), dtype=jnp.float64)
        r_dn = jnp.asarray(rng.uniform(0.0, 5.0, size=(num_dn, 3)), dtype=jnp.float64)

        sigma_PP = np.asarray(
            _jit_vmap_local_stress_PP(
                wf,
                r_up[None, ...],
                r_dn[None, ...],
                jnp.zeros((3, 3), dtype=jnp.float64),
            )
        )[0]
        assert sigma_PP.shape == (3, 3)
        assert np.isfinite(sigma_PP).all(), "sigma_PP has non-finite entries"


# ---------------------------------------------------------------------------
# DTN + position-deriv regression (PBC stress).
#
# ``_dtn_u_term_contributions`` evaluates the analytic-derivative kernel on
# all electron pairs INCLUDING the self-pair ``i == j`` and zeroes the
# self-pair via a downstream ``jnp.where(m > 0, ...)`` mask. At the self-
# pair ``u = r_e_i - r_e_i = 0`` a naive ``sqrt(jnp.dot(c, c))`` in
# ``_smooth_periodic_distance`` returns ``sqrt(0) = 0`` forward (fine) but
# its second-order AD chain (``grad`` and ``hessian`` of ``r_of_u`` taken
# inside ``_dtn_scalar_dist_with_derivs``) NaNs in the backward at
# ``cotangent / (2 * sqrt(0)) = 0 * inf = NaN``. ``jnp.where``-on-output
# cannot rescue this -- it would only mask the FORWARD value while the
# JAX VJP still evaluates the NaN intermediate.
#
# The other PBC fixtures that touch the position-deriv / stress kernels
# (``silicon.h5``, ``H2_ae_ccpvdz_sphe.h5``, ``lithium.h5``) carry no
# ``Jastrow_DTN_data``, and the DTN-bearing tests (``test_jastrow_dtn.py``,
# ``test_param_grad_mask_production_path.py``, ``test_param_grad_mask_matrix.py``)
# run with ``comput_position_deriv=False``, so this class covers the
# cross-product.
#
# ``_smooth_periodic_distance`` therefore uses the standard
# ``jnp.where(d2 > 0, jnp.sqrt(d2_safe), 0.0)`` saddle-routing pattern
# (forward byte-equal at every ``d2``; backward sends the cotangent
# through the zero branch when ``d2 == 0`` instead of the NaN sqrt VJP).
# ---------------------------------------------------------------------------


class TestDTNRegression:
    @pytest.mark.heavy
    def test_stress_HF_PP_pgto_with_DTN_finite(self):
        """``_jit_vmap_local_stress_{HF,PP}`` must not NaN on a DTN-bearing PBC WF.

        Covers the production path
        (``mcmc.run`` with ``comput_position_deriv=True`` on a DTN + pGTO
        wavefunction). The root cause is the inner ``jax.grad`` /
        ``jax.hessian`` of ``_smooth_periodic_distance`` evaluating
        ``sqrt(0)`` backward at the same-spin u-term self-pair
        (``u = r_i - r_i = 0`` is masked downstream but still computed; the
        JAX VJP of ``sqrt`` is ``cotangent / (2 * sqrt(0))`` which produces
        ``0 * inf = NaN`` and propagates through the masked branch).
        Both stress tensors must be finite.

        HF and PP are exercised together in a single test deliberately:
        ``Jastrow_DTN_data`` carries ``M_alpha`` / ``M_beta`` / ``M_gamma`` /
        ``b_alpha`` / ``b_beta`` as ``pytree_node=False`` numpy arrays. JAX's
        pytree cache uses metadata-equality on those fields, which silently
        crashes (``ValueError: arrays cannot be passed as metadata fields``)
        when a second DTN-bearing wavefunction hits the same JIT cache slot
        in the same pytest session. ``_pick_electron_positions`` triggers
        that crash on the second build, so we share one Hamiltonian and one
        electron configuration across both stress kernels until that
        cache-equality issue is fixed separately.
        """
        import os
        import sys

        here = os.path.dirname(__file__)
        if here not in sys.path:
            sys.path.insert(0, here)
        from test_param_grad_mask_production_path import _build_pbc_ham, _pick_electron_positions

        hamiltonian_data = _build_pbc_ham("pgto", include_dtn=True)
        r_up, r_dn = _pick_electron_positions(hamiltonian_data, "pgto", seed=11)
        r_up_b = r_up[None, ...]
        r_dn_b = r_dn[None, ...]
        RT_b = jnp.eye(3, dtype=jnp.float64)[None, ...]
        zero_eps = jnp.zeros((3, 3), dtype=jnp.float64)

        sigma_HF = np.asarray(_jit_vmap_local_stress_HF(hamiltonian_data, r_up_b, r_dn_b, RT_b, zero_eps))[0]
        assert sigma_HF.shape == (3, 3)
        assert np.isfinite(sigma_HF).all(), f"sigma_HF has non-finite entries: {sigma_HF}"

        sigma_PP = np.asarray(_jit_vmap_local_stress_PP(hamiltonian_data.wavefunction_data, r_up_b, r_dn_b, zero_eps))[0]
        assert sigma_PP.shape == (3, 3)
        assert np.isfinite(sigma_PP).all(), f"sigma_PP has non-finite entries: {sigma_PP}"


class TestHydrostaticCovariance:
    """P_hydro must use the trace-first jackknife.

    The three normal stresses are strongly correlated, so propagating their
    separate error bars in quadrature discards the covariance and is wrong in
    both directions (3.76 vs 6.40 on the probe data). The driver keeps
    the replicates, so the correct bar can only be produced there.
    """

    @staticmethod
    def _stub(sigma_jn):
        """A driver stub exposing only what get_P_hydrostatic's math needs."""
        import numpy as _np

        from jqmc.jqmc_mcmc import MCMC

        obj = MCMC.__new__(MCMC)
        obj._replicates = _np.asarray(sigma_jn)
        return obj

    @pytest.mark.light
    def test_trace_first_differs_from_diagonal_quadrature(self):
        """On correlated replicates the two recipes must disagree measurably."""
        rng = np.random.default_rng(11)
        M = 40
        common = rng.standard_normal(M)  # shared isotropic fluctuation
        sigma_jn = np.zeros((M, 3, 3))
        for k in range(3):
            sigma_jn[:, k, k] = -1.0 + 0.5 * common + 0.02 * rng.standard_normal(M)

        p_jn = -np.trace(sigma_jn, axis1=1, axis2=2) / 3.0
        trace_first = float(np.sqrt((M - 1) * np.mean((p_jn - p_jn.mean()) ** 2)))

        diag_std = np.array(
            [float(np.sqrt((M - 1) * np.mean((sigma_jn[:, k, k] - sigma_jn[:, k, k].mean()) ** 2))) for k in range(3)]
        )
        quadrature = float(np.sqrt(np.sum(diag_std**2)) / 3.0)

        # fully correlated diagonals: quadrature UNDER-reports by ~sqrt(3)
        assert trace_first > 1.5 * quadrature, (trace_first, quadrature)

    @pytest.mark.light
    def test_driver_exposes_trace_first_hydrostatic(self):
        """Both production drivers must expose the covariance-aware accessor."""
        from jqmc.jqmc_gfmc import _GFMCBase
        from jqmc.jqmc_mcmc import MCMC

        for cls in (MCMC, _GFMCBase):
            assert hasattr(cls, "get_P_hydrostatic"), cls.__name__
            import inspect

            source = inspect.getsource(cls.get_aT)
            assert "_return_hydrostatic" in source, f"{cls.__name__}.get_aT lost the replicate hook"
            assert "np.trace(" in source, f"{cls.__name__}.get_aT lost the trace-first jackknife"

    @pytest.mark.light
    def test_cli_formatter_refuses_to_invent_a_hydrostatic_bar(self):
        """Without replicates the formatter prints no bar instead of a wrong one."""
        import inspect

        from jqmc import jqmc_cli

        source = inspect.getsource(jqmc_cli._log_stress_section)
        assert "P_std[0] ** 2" not in source, "diagonal quadrature reintroduced"
        assert "no error bar" in source


# ---------------------------------------------------------------------------
# The non-local ECP quadrature directions stay fixed under strain
# ---------------------------------------------------------------------------

#: Precision zones crossed by the stored HF stress of a ccECP periodic SD with J1/J2.
_ECP_HF_ZONES = (
    "ao_eval",
    "ao_grad_lap",
    "det_eval",
    "jastrow_eval",
    "jastrow_grad_lap",
    "wf_eval",
    "wf_kinetic",
    "coulomb",
    "ewald",
)


def _quadrature_strain_ad(ham, r_up, r_dn, carry_quadrature):
    """``d Re e_L / d eps`` by AD of the production forward, quadrature held fixed or carried by ``F``.

    The cell, nuclei (``Hamiltonian_data._apply_strain``) and electrons are
    mapped by ``F = I + eps``.  With ``carry_quadrature=False`` the non-local
    ECP quadrature rotation stays the identity: every strained quadrature
    point lies on the sphere of the strained electron-nucleus distance, so the
    rotation average of this derivative is the exact strain derivative of the
    angular integral.  ``carry_quadrature=True`` passes ``F.T`` as the
    rotation, which moves the points off that sphere; it serves only as a
    negative control.
    """

    def energy(eps):
        deformation = jnp.eye(3) + eps
        strained = ham._apply_strain(deformation)
        rotation = deformation.T if carry_quadrature else jnp.eye(3)
        return jnp.real(compute_local_energy(strained, r_up @ deformation.T, r_dn @ deformation.T, rotation))

    return np.asarray(jax.jacrev(energy)(jnp.zeros((3, 3))))


@pytest.mark.heavy
@pytest.mark.parametrize("backend", ["xla", "rust"])
@pytest.mark.parametrize("stem", ["lithium.h5", "lithium_k025.h5"], ids=["gamma", "twist"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_mcmc_stress_hf_matches_fixed_quadrature_strain_ad(stem, backend):
    """Stored stress_HF equals AD of e_L with the ECP quadrature directions held fixed under strain.

    Committed lithium.h5 (real Gamma) and lithium_k025.h5 (complex Bloch twist
    k = (1/4, 1/4, 1/4)): bcc Li, ccECP, spherical pGTO SD, exponential J1/J2
    (both 1.0).  Two walkers from seed 34 take two moves per measurement at
    Dt=1.0, AS=0, identity quadrature rotation, force and stress on.  At each
    stored walker position the reference is jax AD, over all nine strain
    entries, of the production forward ``Re compute_local_energy`` with the
    cell, nuclei and electrons mapped by ``F = I + eps`` and the quadrature
    rotation held at the identity (see ``_quadrature_strain_ad``).  The same
    AD with the rotation carried as ``F.T`` (a mesh convention that biases
    the non-local ECP stress) must be rejected by the same
    comparison at every walker, so the test discriminates the two
    conventions.  Both sides run in the active precision, so full is judged at
    the strict and mixed at the medium zone tolerance of their own mode; no
    mixed output is compared with an fp64 reference.  XLA and Rust are both
    required.
    """
    from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
    from tests.test_jqmc_kernel._grad_fidelity_harness import _build_ham
    from tests.test_rs_kernel.conftest import assert_native_ran, assert_no_native_ran, native_ctx_spy
    from tests.test_rs_kernel.test_gfmc_projection import _backend_env

    ham = _build_ham(stem, True, None)
    level = "medium" if is_mixed_precision_enabled() else "strict"
    atol, rtol = get_tolerance_min(_ECP_HF_ZONES, level)
    with _backend_env("fast" if backend == "rust" else "jax"):
        driver = MCMC(
            hamiltonian_data=ham,
            num_walkers=2,
            mcmc_seed=34,
            num_mcmc_per_measurement=2,
            Dt=1.0,
            epsilon_AS=0.0,
            random_discretized_mesh=False,
            comput_position_deriv=True,
            comput_stress=True,
        )
        with native_ctx_spy() as counts:
            for step in range(2):
                driver.run(num_mcmc_steps=1)
                r_up_carts = np.asarray(driver._MCMC__latest_r_up_carts)
                r_dn_carts = np.asarray(driver._MCMC__latest_r_dn_carts)
                for walker in range(2):
                    r_up, r_dn = jnp.asarray(r_up_carts[walker]), jnp.asarray(r_dn_carts[walker])
                    label = f"{stem}/{step}/{walker}"
                    reference = _quadrature_strain_ad(ham, r_up, r_dn, carry_quadrature=False)
                    superseded = _quadrature_strain_ad(ham, r_up, r_dn, carry_quadrature=True)
                    actual = np.real(np.asarray(driver._MCMC__stored_stress_HF)[step, walker])
                    assert actual.shape == reference.shape == (3, 3), label
                    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)), label
                    assert np.all(np.isfinite(superseded)), label
                    assert np.any(np.abs(reference) > atol + rtol * np.abs(reference)), f"{label}: vacuous reference"
                    assert np.any(np.abs(superseded - reference) > atol + rtol * np.abs(reference)), (
                        f"{label}: the carried quadrature must be distinguishable from the fixed one"
                    )
                    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=label)
    if backend == "rust":
        assert_native_ran(counts, "MCMC fixed-quadrature stress AD reference")
    else:
        assert_no_native_ran(counts, "MCMC fixed-quadrature stress AD reference")

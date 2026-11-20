"""collections of unit tests."""

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

from jqmc._precision import is_mixed_precision_enabled
from jqmc._setting import get_tolerance_mixed_fidelity
from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block

pytestmark = [pytest.mark.with_chdir]
from jax import numpy as jnp

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import get_tolerance, get_tolerance_min
from jqmc._setting import (
    AO_IMAGE_TRUNCATION_ATOL,
    ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT,
    ATOL_PYTHON_FP64_LOOSE,
    ATOL_PYTHON_FP64_TIGHT,
    RTOL_BIT_EXACT,
    RTOL_PYTHON_FP64_LOOSE,
)
from jqmc.atomic_orbital import (
    AOs_bspline_data,
    AOs_plane_wave_data,
    AOs_sphe_gto_data,
    _apply_image_expansion,
)
from jqmc.determinant import Geminal_data, compute_geminal_all_elements
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.jqmc_optimizer import RealCoordinateMap
from jqmc.molecular_orbital import MOs_data

# Spline prefiltering lives in the projection module (not atomic_orbital);
# only the periodic B-spline leaf builder needs it.
from jqmc.pw_pgto_projection import bspline_prefilter_periodic
from jqmc.structure import Structure_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import (
    Wavefunction_data,
    _advance_kinetic_energy_all_elements_streaming_state,
    _compute_discretized_kinetic_energy_debug,
    _compute_kinetic_energy_all_elements_auto,
    _compute_kinetic_energy_all_elements_fast_update_debug,
    _compute_kinetic_energy_auto,
    _compute_nodal_distance_debug,
    _init_kinetic_energy_all_elements_streaming_state,
    _kinetic_energy_from_streaming_state,
    compute_discretized_kinetic_energy,
    compute_discretized_kinetic_energy_fast_update,
    compute_kinetic_energy,
    compute_kinetic_energy_all_elements,
    compute_kinetic_energy_all_elements_fast_update,
    compute_nodal_distance,
    evaluate_ln_wavefunction,
    evaluate_ln_wavefunction_fast,
    f_epsilon_PW,
)

from tests.test_jqmc_kernel.optimizer_metadata_helpers import make_test_block
from tests._ecp_core_electrons import core_electrons_of

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


def _assert_component_derivative_fidelity(mixed_block, full_block, atol, rtol):
    """Keep the homogeneous-block bound transparent to GDD runtime receipts."""
    assert np.abs(mixed_block - full_block).max() <= atol + rtol * np.abs(full_block).max()


@pytest.mark.light
def test_molecular_wavefunction_is_fermion_antisymmetric_under_same_spin_exchange():
    """Exchanging two up-spin electrons sends ``Psi -> -Psi`` and leaves ``e_L`` unchanged.

    The committed open-shell N fixture has five up and two down electrons and an
    active J1+J2 Jastrow.  The fixed configuration is screened away from a node,
    so the signed-wavefunction assertion is non-vacuous.  This is a same-spin
    exchange; it is deliberately not the older up/down density swap.
    """
    from jqmc.hamiltonians import compute_local_energy
    from jqmc.wavefunction import evaluate_wavefunction
    from tests.test_jqmc_kernel._grad_fidelity_harness import _build_ham, _draw

    ham = _build_ham("N_ae_ccpvdz_cart.h5", True, None)
    r_up, r_dn = _draw(ham, seed=29)
    indices = jnp.asarray([0, 1])
    exchanged_up = r_up.at[indices].set(r_up[jnp.asarray([1, 0])])

    geminal = np.asarray(compute_geminal_all_elements(ham.wavefunction_data.geminal_data, r_up, r_dn))
    condition = np.linalg.cond(geminal)
    assert condition < 1.0e2

    psi = np.asarray(evaluate_wavefunction(ham.wavefunction_data, r_up, r_dn))
    psi_exchanged = np.asarray(evaluate_wavefunction(ham.wavefunction_data, exchanged_up, r_dn))
    assert np.abs(psi) > 1.0e-10

    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "wf_eval"), "strict")
    np.testing.assert_allclose(psi_exchanged, -psi, atol=atol * np.abs(psi), rtol=rtol)

    local_energy = np.asarray(compute_local_energy(ham, r_up, r_dn, jnp.eye(3)))
    exchanged_local_energy = np.asarray(compute_local_energy(ham, exchanged_up, r_dn, jnp.eye(3)))
    np.testing.assert_allclose(exchanged_local_energy, local_energy, atol=atol, rtol=rtol)


_KINETIC_NODE_FLOOR = 1.0e-8  # Existing node-far precondition, unchanged.


def _molecular_kinetic_case(trexio_file, *, per_electron=False, fixed_positions=None):
    """Rebuild the molecular source with deterministic J3 and original positions."""
    from jqmc.wavefunction import evaluate_wavefunction

    (
        _,
        aos_data,
        _,
        _,
        geminal_mo_data,
        _,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
        store_tuple=True,
    )

    jastrow_onebody_data = None
    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="pade")
    jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
        orb_data=aos_data, random_init=True, random_scale=1.0e-3, seed=43 if per_electron else 42
    )
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
    )

    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

    n_up = geminal_mo_data.num_electron_up
    n_dn = geminal_mo_data.num_electron_dn
    if fixed_positions is not None:
        r_up, r_dn = (np.array(value, copy=True) for value in fixed_positions)
        assert r_up.shape == (n_up, 3) and r_dn.shape == (n_dn, 3)
        if not per_electron:
            assert abs(float(evaluate_wavefunction(wavefunction_data, r_up, r_dn))) > _KINETIC_NODE_FLOOR
        return wavefunction_data, r_up, r_dn
    rng = np.random.default_rng(43 if per_electron else 42)
    if per_electron:
        r_up = rng.uniform(-2.0, 2.0, size=(n_up, 3))
        r_dn = rng.uniform(-2.0, 2.0, size=(n_dn, 3))
        return wavefunction_data, r_up, r_dn
    for _ in range(50):
        r_up = 4.0 * rng.random((n_up, 3)) - 2.0
        r_dn = 4.0 * rng.random((n_dn, 3)) - 2.0
        if abs(float(evaluate_wavefunction(wavefunction_data, r_up, r_dn))) > _KINETIC_NODE_FLOOR:
            return wavefunction_data, r_up, r_dn
    pytest.fail(
        "Could not find electron configuration sufficiently far from node with the fixed"
        " seed; this precondition search is deterministic and a failure indicates"
        " evaluate_wavefunction regressed (e.g. returning 0/NaN), not an environment fluke"
    )


def _kinetic_production_modes(case_factory, control_atol, control_rtol):
    """Keep scalar and per-electron spatial second-derivative outputs separate."""

    def analytical(case):
        wavefunction, r_up, r_dn = case
        geminal = wavefunction.geminal_data
        scalar = compute_kinetic_energy(wavefunction, r_up, r_dn)
        up, dn = compute_kinetic_energy_all_elements(wavefunction, r_up, r_dn)
        assert np.asarray(scalar).shape == ()
        assert np.asarray(up).shape == (geminal.num_electron_up,)
        assert np.asarray(dn).shape == (geminal.num_electron_dn,)
        assert np.all(np.isfinite(scalar)) and np.all(np.isfinite(up)) and np.all(np.isfinite(dn))
        # A separate same-policy identity protects the old scalar route while
        # per-electron fidelity blocks expose cancellation in its total.
        np.testing.assert_allclose(scalar, np.sum(up) + np.sum(dn), atol=control_atol, rtol=control_rtol)
        return scalar, up, dn

    return mixed_vs_full(case_factory, analytical)


class TestKineticEnergyConsistency:
    """Analytic vs autodiff kinetic-energy consistency tests."""

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"])
    def test_kinetic_energy_analytic_and_auto(self, trexio_file: str):
        """Full retains strict AD; mixed checks scalar and per-electron production fidelity."""
        if is_mixed_precision_enabled():
            # Freeze the entry-policy source positions, then rebuild the model
            # independently under both policies without another node search.
            _, source_up, source_dn = _molecular_kinetic_case(trexio_file)
            fixed_positions = (source_up.copy(), source_dn.copy())
            control_atol, control_rtol = get_tolerance_min(
                ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "wf_kinetic"), "strict"
            )
            mixed, full = _kinetic_production_modes(
                lambda: _molecular_kinetic_case(trexio_file, fixed_positions=fixed_positions), control_atol, control_rtol
            )
            blocks = mixed_fidelity_blocks(mixed, full)
            atol, rtol = get_tolerance_mixed_fidelity("orbital_derivative")
            for mixed_block, full_block in blocks:
                _assert_component_derivative_fidelity(mixed_block, full_block, atol, rtol)
            corrupted = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
            with pytest.raises(AssertionError):
                _assert_component_derivative_fidelity(corrupted, blocks[0][1], atol, rtol)
            jax.clear_caches()
            return

        wavefunction_data, r_up_carts, r_dn_carts = _molecular_kinetic_case(trexio_file)

        K_analytic = compute_kinetic_energy(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
        K_auto = _compute_kinetic_energy_auto(
            wavefunction_data=wavefunction_data,
            r_up_carts=jnp.asarray(r_up_carts),
            r_dn_carts=jnp.asarray(r_dn_carts),
        )

        # T_L crosses ao_eval/jastrow_eval/jastrow_grad_lap/wf_kinetic zones; the
        # achievable analytic-vs-auto agreement is bounded by the weakest (fp32 in mixed).
        atol, rtol = get_tolerance_min(
            ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "wf_kinetic"),
            "strict",
        )
        assert not np.any(np.isnan(np.asarray(K_analytic))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(K_auto))), "NaN detected in second argument"
        np.testing.assert_allclose(K_analytic, K_auto, atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"])
    def test_auto_and_analytic_kinetic_energy_all_elements(self, trexio_file: str):
        """Full keeps strict per-electron AD; mixed checks fresh analytical production blocks."""
        if is_mixed_precision_enabled():
            _, source_up, source_dn = _molecular_kinetic_case(trexio_file, per_electron=True)
            fixed_positions = (source_up.copy(), source_dn.copy())
            control_atol, control_rtol = get_tolerance_min(
                ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "wf_kinetic"), "strict"
            )
            mixed, full = _kinetic_production_modes(
                lambda: _molecular_kinetic_case(trexio_file, per_electron=True, fixed_positions=fixed_positions),
                control_atol,
                control_rtol,
            )
            blocks = mixed_fidelity_blocks(mixed, full)
            atol, rtol = get_tolerance_mixed_fidelity("orbital_derivative")
            for mixed_block, full_block in blocks:
                _assert_component_derivative_fidelity(mixed_block, full_block, atol, rtol)
            corrupted = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
            with pytest.raises(AssertionError):
                _assert_component_derivative_fidelity(corrupted, blocks[0][1], atol, rtol)
            jax.clear_caches()
            return

        wavefunction_data, r_up_carts_np, r_dn_carts_np = _molecular_kinetic_case(trexio_file, per_electron=True)
        r_up_carts_jnp = jnp.array(r_up_carts_np)
        r_dn_carts_jnp = jnp.array(r_dn_carts_np)

        K_elements_up_auto, K_elements_dn_auto = _compute_kinetic_energy_all_elements_auto(
            wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_jnp, r_dn_carts=r_dn_carts_jnp
        )
        K_elements_up_analytic, K_elements_dn_analytic = compute_kinetic_energy_all_elements(
            wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_jnp, r_dn_carts=r_dn_carts_jnp
        )

        # This fp64-only analytic-vs-autodiff correctness oracle compares two
        # implementations of the same quantity and therefore uses strict.
        atol, rtol = get_tolerance_min(
            ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "wf_kinetic"),
            "strict",
        )
        assert not np.any(np.isnan(np.asarray(K_elements_up_auto))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(K_elements_up_analytic))), "NaN detected in second argument"
        np.testing.assert_allclose(K_elements_up_auto, K_elements_up_analytic, atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(K_elements_dn_auto))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(K_elements_dn_analytic))), "NaN detected in second argument"
        np.testing.assert_allclose(K_elements_dn_auto, K_elements_dn_analytic, atol=atol, rtol=rtol)


class TestKineticEnergyFastUpdate:
    """Fast-update per-electron kinetic-energy consistency."""

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"])
    def test_compute_kinetic_energy_all_elements_fast_update_vs_debug(self, trexio_file: str):
        """Guardrail: compute_kinetic_energy_all_elements_fast_update == _debug.

        Renamed from test_fast_update_kinetic_energy_all_elements so the
        fast_update/_debug pair follows the discoverable ..._vs_debug naming
        convention; the comparison against the standard analytic/_debug path is
        unchanged.
        """
        (
            structure_data,
            aos_data,
            _,
            _,
            geminal_mo_data,
            _,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        jastrow_onebody_data = None
        jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="exp")
        jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=aos_data, random_init=True, random_scale=1.0e-3
        )
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=jastrow_onebody_data,
            jastrow_two_body_data=jastrow_twobody_data,
            jastrow_three_body_data=jastrow_threebody_data,
        )

        wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

        num_ele_up = geminal_mo_data.num_electron_up
        num_ele_dn = geminal_mo_data.num_electron_dn
        rng = np.random.default_rng(42)
        centers = np.asarray(structure_data.positions)
        cond_max = 1.0e4
        max_draw_tries = 300
        for _ in range(max_draw_tries):
            r_up_carts = centers[rng.integers(0, len(centers), size=num_ele_up)] + rng.normal(scale=0.5, size=(num_ele_up, 3))
            r_dn_carts = centers[rng.integers(0, len(centers), size=num_ele_dn)] + rng.normal(scale=0.5, size=(num_ele_dn, 3))
            A = np.asarray(
                compute_geminal_all_elements(
                    geminal_data=geminal_mo_data,
                    r_up_carts=r_up_carts,
                    r_dn_carts=r_dn_carts,
                )
            )
            if np.isfinite(A).all() and np.linalg.cond(A) < cond_max:
                break
        else:
            pytest.fail(f"Could not find cond(G) < {cond_max:g} for {trexio_file} after {max_draw_tries} deterministic draws")

        r_up_carts_jnp = jnp.asarray(r_up_carts)
        r_dn_carts_jnp = jnp.asarray(r_dn_carts)

        # Standard analytic per-electron kinetic energy
        ke_up_debug, ke_dn_debug = _compute_kinetic_energy_all_elements_fast_update_debug(
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts_jnp,
            r_dn_carts=r_dn_carts_jnp,
        )

        # Build the fast-path inverse from the accepted, well-conditioned G.
        A_inv = jnp.asarray(np.linalg.inv(A))

        ke_up_fast, ke_dn_fast = compute_kinetic_energy_all_elements_fast_update(
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts_jnp,
            r_dn_carts=r_dn_carts_jnp,
            geminal_inverse=A_inv,
        )

        # Fast-update path crosses ao_eval/jastrow_eval/jastrow_grad_lap/det_ratio/
        # wf_kinetic zones; the achievable agreement is bounded by the weakest
        # (fp32 in mixed for ao_eval / jastrow_eval / jastrow_grad_lap).
        atol, rtol = get_tolerance_min(
            ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "det_ratio", "wf_kinetic"),
            "strict",
        )
        assert not np.any(np.isnan(np.asarray(ke_up_fast))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(ke_up_debug))), "NaN detected in second argument"
        np.testing.assert_allclose(ke_up_fast, ke_up_debug, atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(ke_dn_fast))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(ke_dn_debug))), "NaN detected in second argument"
        np.testing.assert_allclose(ke_dn_fast, ke_dn_debug, atol=atol, rtol=rtol)


class TestDiscretizedKineticEnergy:
    """Discretized LRDMC kinetic-energy paths and PBC translation invariance."""

    @pytest.mark.parametrize(
        "trexio_file",
        ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"],
    )
    @pytest.mark.light
    def test_debug_and_jax_discretized_kinetic_energy(self, trexio_file: str):
        """Production discretised LRDMC kinetic agrees with ``_debug`` and ``fast_update``.

        Molecular parametrisation only. PBC coverage of the four
        ``_wrap_into_cell_if_pbc`` calls inside
        ``compute_discretized_kinetic_energy{,_fast_update}`` lives in the
        sibling test :py:func:`test_debug_and_jax_discretized_kinetic_energy_pbc`,
        which constructs an inline synthetic 2-atom cubic fixture (compact
        s-basis) so the AO-image-expansion intermediate stays small enough
        to JIT-compile. Real PBC TREXIO fixtures (silicon.h5, lithium.h5)
        blow up the discretised-kinetic JIT trace.
        """
        (
            _,
            aos_data,
            _,
            _,
            geminal_mo_data,
            _,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        jastrow_onebody_data = None
        jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type="pade")
        jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=aos_data, random_init=True, random_scale=1.0e-3
        )

        jastrow_data = Jastrow_data(
            jastrow_one_body_data=jastrow_onebody_data,
            jastrow_two_body_data=jastrow_twobody_data,
            jastrow_three_body_data=jastrow_threebody_data,
        )
        jastrow_data.sanity_check()

        wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)
        wavefunction_data.sanity_check()

        num_ele_up = geminal_mo_data.num_electron_up
        num_ele_dn = geminal_mo_data.num_electron_dn
        rng = np.random.default_rng(44)
        r_up_carts_np = rng.uniform(-2.0, 2.0, size=(num_ele_up, 3))
        r_dn_carts_np = rng.uniform(-2.0, 2.0, size=(num_ele_dn, 3))

        r_up_carts_jnp = jnp.array(r_up_carts_np)
        r_dn_carts_jnp = jnp.array(r_dn_carts_np)

        alat = 0.05
        RT = np.eye(3)
        mesh_kinetic_part_r_up_carts_debug, mesh_kinetic_part_r_dn_carts_debug, elements_kinetic_part_debug = (
            _compute_discretized_kinetic_energy_debug(
                alat=alat, wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_np, r_dn_carts=r_dn_carts_np
            )
        )

        mesh_kinetic_part_r_up_carts_jax, mesh_kinetic_part_r_dn_carts_jax, elements_kinetic_part_jax = (
            compute_discretized_kinetic_energy(
                alat=alat, wavefunction_data=wavefunction_data, r_up_carts=r_up_carts_jnp, r_dn_carts=r_dn_carts_jnp, RT=RT
            )
        )

        A = compute_geminal_all_elements(geminal_data=geminal_mo_data, r_up_carts=r_up_carts_jnp, r_dn_carts=r_dn_carts_jnp)
        A_old_inv = np.linalg.inv(A)
        (
            mesh_kinetic_part_r_up_carts_jax_fast_update,
            mesh_kinetic_part_r_dn_carts_jax_fast_update,
            elements_kinetic_part_jax_fast_update,
        ) = compute_discretized_kinetic_energy_fast_update(
            alat=alat,
            A_old_inv=A_old_inv,
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts_jnp,
            r_dn_carts=r_dn_carts_jnp,
            RT=RT,
        )

        # Discretized kinetic energy (LRDMC) crosses ao_eval/jastrow_eval/
        # jastrow_grad_lap/jastrow_ratio/det_ratio/wf_ratio/wf_kinetic zones; the
        # fast_update path uses Sherman-Morrison ratios. Agreement is bounded by
        # the weakest (fp32 in mixed for ao_eval / jastrow_* zones).
        atol, rtol = get_tolerance_min(
            (
                "ao_eval",
                "jastrow_eval",
                "jastrow_grad_lap",
                "jastrow_ratio",
                "det_ratio",
                "wf_ratio",
                "wf_kinetic",
            ),
            "strict",
        )
        assert not np.any(np.isnan(np.asarray(mesh_kinetic_part_r_up_carts_jax))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(mesh_kinetic_part_r_up_carts_debug))), "NaN detected in second argument"
        np.testing.assert_allclose(
            mesh_kinetic_part_r_up_carts_jax,
            mesh_kinetic_part_r_up_carts_debug,
            atol=atol,
            rtol=rtol,
        )
        assert not np.any(np.isnan(np.asarray(mesh_kinetic_part_r_dn_carts_jax))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(mesh_kinetic_part_r_dn_carts_debug))), "NaN detected in second argument"
        np.testing.assert_allclose(
            mesh_kinetic_part_r_dn_carts_jax,
            mesh_kinetic_part_r_dn_carts_debug,
            atol=atol,
            rtol=rtol,
        )
        assert not np.any(np.isnan(np.asarray(mesh_kinetic_part_r_up_carts_jax_fast_update))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(mesh_kinetic_part_r_up_carts_debug))), "NaN detected in second argument"
        np.testing.assert_allclose(
            mesh_kinetic_part_r_up_carts_jax_fast_update,
            mesh_kinetic_part_r_up_carts_debug,
            atol=atol,
            rtol=rtol,
        )
        assert not np.any(np.isnan(np.asarray(mesh_kinetic_part_r_dn_carts_jax_fast_update))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(mesh_kinetic_part_r_dn_carts_debug))), "NaN detected in second argument"
        np.testing.assert_allclose(
            mesh_kinetic_part_r_dn_carts_jax_fast_update,
            mesh_kinetic_part_r_dn_carts_debug,
            atol=atol,
            rtol=rtol,
        )
        assert not np.any(np.isnan(np.asarray(elements_kinetic_part_jax))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(elements_kinetic_part_debug))), "NaN detected in second argument"
        np.testing.assert_allclose(elements_kinetic_part_jax, elements_kinetic_part_debug, atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(elements_kinetic_part_jax_fast_update))), "NaN detected in first argument"
        assert not np.any(np.isnan(np.asarray(elements_kinetic_part_debug))), "NaN detected in second argument"
        np.testing.assert_allclose(
            elements_kinetic_part_jax_fast_update,
            elements_kinetic_part_debug,
            atol=atol,
            rtol=rtol,
        )

    @pytest.mark.light
    def test_discretized_kinetic_pbc_lattice_translation_invariance(self):
        """Discretised LRDMC kinetic energy is invariant under a lattice translation.

        Kinetic-energy analogue of the ECP non-local lattice-translation test
        (``test_ecp_non_local_pbc_lattice_translation_invariance`` in
        ``tests/test_jqmc_kernel/test_coulomb_potential.py``). The nuclei are kept fixed inside the
        home cell (as a real PBC loader stores them) and *all electrons* are
        shifted by an integer lattice vector ``T``. The (Gamma-point) kinetic
        energy must be unchanged, because ``_wrap_into_cell_if_pbc`` inside
        ``compute_discretized_kinetic_energy`` folds the shifted electrons back
        onto their home-cell positions. A wrap dropped or mis-applied at any of
        the +-alat finite-difference evaluation points would break this
        invariance.

        Uses the real ``silicon.h5`` PBC fixture (8-atom Si conventional cell,
        32 valence electrons, ccECP-ccpvtz). Tractable after the
        ``_primitive_image_radius`` atol-tolerance fix dropped ``n_image`` from
        ~6,900 to ~250 for diffuse Si p-orbitals.
        """
        trexio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "silicon.h5")
        if not os.path.exists(trexio_path):
            pytest.skip("Periodic TREXIO fixture silicon.h5 is not present.")
        structure_data, _aos, _mos_up, _mos_dn, geminal_mo_data, _coulomb = read_trexio_file(
            trexio_file=trexio_path, store_tuple=True
        )
        cell_np = np.asarray(structure_data.cell, dtype=np.float64)
        T = np.array([1, -2, 1], dtype=np.float64) @ cell_np

        alat = 0.30
        RT = np.eye(3)

        wf_base = Wavefunction_data(
            geminal_data=geminal_mo_data,
            jastrow_data=Jastrow_data(
                jastrow_one_body_data=None,
                jastrow_two_body_data=None,
                jastrow_three_body_data=None,
            ),
        )
        # Each ccECP Si carries 4 valence electrons -> assign 2 up + 2 dn per Si.
        # Small Gaussian jitter (sigma=0.4 Bohr) avoids the coincidence singularity
        # without scrambling electrons across atoms. NOTE: ``|Psi|`` for the
        # periodic 16-up/16-dn Si AGP is ~1e-40 even at well-conditioned configs
        # because each Bloch MO is normalised over the unit cell volume V ~ 1050
        # Bohr^3 and the 16x16 Slater determinant compounds the 1/sqrt(V)
        # amplitude. Translation invariance is about the *ratio*
        # ``Psi(r+T)/Psi(r) = 1`` not the absolute magnitude, so no node guard
        # threshold is appropriate here.
        n_up = geminal_mo_data.num_electron_up
        n_dn = geminal_mo_data.num_electron_dn
        atoms = np.asarray(structure_data.positions, dtype=np.float64)
        natom = atoms.shape[0]
        up_per_atom = n_up // natom
        dn_per_atom = n_dn // natom
        assert n_up == up_per_atom * natom and n_dn == dn_per_atom * natom, (
            f"silicon fixture expected to distribute electrons evenly across nuclei (n_up={n_up}, n_dn={n_dn}, natom={natom})"
        )
        rng = np.random.default_rng(0)
        r_up_base = np.repeat(atoms, up_per_atom, axis=0) + rng.normal(scale=0.4, size=(n_up, 3))
        r_dn_base = np.repeat(atoms, dn_per_atom, axis=0) + rng.normal(scale=0.4, size=(n_dn, 3))
        r_up_tr = r_up_base + T
        r_dn_tr = r_dn_base + T
        wf_tr = wf_base  # nuclei unchanged; only the electrons translate

        _, _, elements_base = compute_discretized_kinetic_energy(
            alat=alat, wavefunction_data=wf_base, r_up_carts=jnp.array(r_up_base), r_dn_carts=jnp.array(r_dn_base), RT=RT
        )
        _, _, elements_tr = compute_discretized_kinetic_energy(
            alat=alat, wavefunction_data=wf_tr, r_up_carts=jnp.array(r_up_tr), r_dn_carts=jnp.array(r_dn_tr), RT=RT
        )

        sum_base = float(np.asarray(elements_base).sum())
        sum_tr = float(np.asarray(elements_tr).sum())

        atol, rtol = get_tolerance("wf_kinetic", "strict")
        assert not np.isnan(sum_base) and not np.isnan(sum_tr)
        np.testing.assert_allclose(sum_tr, sum_base, atol=atol, rtol=rtol)

        jax.clear_caches()


class TestNodalDistance:
    """Analytic vs debug nodal-distance consistency."""

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"])
    def test_nodal_distance_analytic_vs_debug(self, trexio_file: str):
        """Analytic compute_nodal_distance should match _compute_nodal_distance_debug."""
        (
            structure_data,
            aos_data,
            _,
            _,
            geminal_mo_data,
            _,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        jastrow_onebody_data = None
        jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="exp")
        jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=aos_data, random_init=True, random_scale=1.0e-3
        )
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=jastrow_onebody_data,
            jastrow_two_body_data=jastrow_twobody_data,
            jastrow_three_body_data=jastrow_threebody_data,
        )

        wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)

        num_ele_up = geminal_mo_data.num_electron_up
        num_ele_dn = geminal_mo_data.num_electron_dn
        r_cart_min, r_cart_max = -3.0, +3.0

        # Reject configurations near the wavefunction node: when |Psi| ~ 0 the
        # quantity |grad ln|Psi|| diverges, which makes the analytic vs autodiff
        # comparison sensitive to floating-point round-off (a few percent in
        # mixed precision). Same node-avoidance pattern as other tests.
        from jqmc.wavefunction import evaluate_wavefunction

        rng = np.random.default_rng(42)
        nd_analytic = None
        r_up_carts_jnp = r_dn_carts_jnp = None
        for _ in range(50):
            r_up_carts = (r_cart_max - r_cart_min) * rng.random((num_ele_up, 3)) + r_cart_min
            r_dn_carts = (r_cart_max - r_cart_min) * rng.random((num_ele_dn, 3)) + r_cart_min
            r_up_carts_jnp = jnp.asarray(r_up_carts)
            r_dn_carts_jnp = jnp.asarray(r_dn_carts)
            psi_val = evaluate_wavefunction(wavefunction_data, r_up_carts_jnp, r_dn_carts_jnp)
            if abs(float(psi_val)) > 1.0e-8:
                nd_analytic = compute_nodal_distance(
                    wavefunction_data=wavefunction_data,
                    r_up_carts=r_up_carts_jnp,
                    r_dn_carts=r_dn_carts_jnp,
                )
                break
        else:
            pytest.fail(
                f"Could not find electron configuration sufficiently far from node for {trexio_file}"
                " with the fixed seed; this precondition search is deterministic and a failure"
                " indicates evaluate_wavefunction regressed, not an environment fluke"
            )

        # Debug path (paper formula)
        nd_debug = _compute_nodal_distance_debug(
            wavefunction_data=wavefunction_data,
            r_up_carts=r_up_carts_jnp,
            r_dn_carts=r_dn_carts_jnp,
        )

        # This is an fp64 analytic-vs-autodiff oracle. Mixed precision skips it;
        # avoiding the node keeps the full-precision comparison well conditioned.
        atol, rtol = get_tolerance_min(
            ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "wf_kinetic"),
            "strict",
        )
        np.testing.assert_allclose(
            np.asarray(nd_analytic),
            np.asarray(nd_debug),
            rtol=rtol,
            atol=atol,
        )

        # Sanity: nodal distance should be positive
        assert float(nd_analytic) > 0.0
        assert float(nd_debug) > 0.0


class TestWavefunctionUtilities:
    """Boundary / property checks for wavefunction utility kernels."""

    @pytest.mark.light
    def test_f_epsilon_PW_boundary_values(self):
        """Test f_epsilon_PW boundary conditions and properties."""
        epsilon_PW = 0.5

        # f(0) = 0
        np.testing.assert_array_equal(float(f_epsilon_PW(jnp.array(0.0), epsilon_PW)), 0.0)

        # f(epsilon) = 1 (i.e., |t|=1)
        np.testing.assert_array_equal(float(f_epsilon_PW(jnp.array(epsilon_PW), epsilon_PW)), 1.0)

        # f(x) = 1 for |x| >= epsilon
        np.testing.assert_array_equal(float(f_epsilon_PW(jnp.array(2.0 * epsilon_PW), epsilon_PW)), 1.0)
        np.testing.assert_array_equal(float(f_epsilon_PW(jnp.array(10.0 * epsilon_PW), epsilon_PW)), 1.0)

        # f'(0) = 0 and f'(1) = 0 (smooth at boundaries)
        # Check numerically: f(small) ~ 0 (derivative at origin is zero)
        delta = 1.0e-8
        f_at_delta = float(f_epsilon_PW(jnp.array(delta), epsilon_PW))
        # Should scale as t^2, so f(delta/eps) ~ 9*(delta/eps)^2
        assert f_at_delta < ATOL_PYTHON_FP64_AGGREGATOR_INVARIANT  # very close to 0

        # f(x) >= 0 for interior
        t_vals = jnp.linspace(0.0, epsilon_PW, 100)
        f_vals = f_epsilon_PW(t_vals, epsilon_PW)
        assert jnp.all(f_vals >= -1e-14)


class TestWavefunctionFastPath:
    """Forward/backward consistency of evaluate_ln_wavefunction_fast."""

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"])
    def test_evaluate_ln_wavefunction_fast_forward(self, trexio_file):
        """Forward value of evaluate_ln_wavefunction_fast must match evaluate_ln_wavefunction."""
        _, _, _, _, geminal_data, _ = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )
        wavefunction_data = Wavefunction_data(geminal_data=geminal_data)
        rng = np.random.default_rng(2)
        n_up = geminal_data.num_electron_up
        n_dn = geminal_data.num_electron_dn

        atol, rtol = get_tolerance_min(("ao_eval", "det_eval", "wf_kinetic"), "strict")
        for sample in range(10):
            sample_label = str(sample)
            r_up = jnp.array(rng.standard_normal((n_up, 3)) * 1.2, dtype=jnp.float64)
            r_dn = jnp.array(rng.standard_normal((n_dn, 3)) * 1.2, dtype=jnp.float64)
            G = compute_geminal_all_elements(geminal_data, r_up, r_dn)
            G_inv = jnp.linalg.inv(G)

            val_ref = float(evaluate_ln_wavefunction(wavefunction_data, r_up, r_dn))
            val_fast = float(evaluate_ln_wavefunction_fast(wavefunction_data, r_up, r_dn, G_inv))

            assert np.isfinite(val_ref), f"Reference value is not finite: {val_ref}"
            assert np.isfinite(val_fast), f"Fast value is not finite: {val_fast}"
            np.testing.assert_allclose(
                val_fast,
                val_ref,
                atol=atol,
                rtol=rtol,
                err_msg=f"Forward mismatch: fast={val_fast:.15f}, ref={val_ref:.15f}",
            )

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"])
    def test_evaluate_ln_wavefunction_fast_backward(self, trexio_file):
        """Gradient of evaluate_ln_wavefunction_fast w.r.t. wavefunction_data must match evaluate_ln_wavefunction."""
        _, _, _, _, geminal_data, _ = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )
        wavefunction_data = Wavefunction_data(geminal_data=geminal_data)
        rng = np.random.default_rng(3)
        n_up = geminal_data.num_electron_up
        n_dn = geminal_data.num_electron_dn

        grad_ref_fn = jax.grad(evaluate_ln_wavefunction, argnums=0)
        grad_fast_fn = jax.grad(evaluate_ln_wavefunction_fast, argnums=0)

        atol, rtol = get_tolerance_min(("ao_eval", "det_eval", "wf_kinetic"), "strict")
        for _ in range(10):
            r_up = jnp.array(rng.standard_normal((n_up, 3)) * 1.2, dtype=jnp.float64)
            r_dn = jnp.array(rng.standard_normal((n_dn, 3)) * 1.2, dtype=jnp.float64)
            G = compute_geminal_all_elements(geminal_data, r_up, r_dn)
            G_inv = jnp.linalg.inv(G)

            grad_ref = grad_ref_fn(wavefunction_data, r_up, r_dn)
            grad_fast = grad_fast_fn(wavefunction_data, r_up, r_dn, G_inv)

            ref_leaves, ref_tree = jax.tree_util.tree_flatten(grad_ref)
            fast_leaves, fast_tree = jax.tree_util.tree_flatten(grad_fast)
            assert fast_tree == ref_tree
            assert len(ref_leaves) > 0
            for a, b in zip(ref_leaves, fast_leaves, strict=True):
                np.testing.assert_allclose(
                    np.asarray(a),
                    np.asarray(b),
                    atol=atol,
                    rtol=rtol,
                    err_msg="Backward mismatch in evaluate_ln_wavefunction_fast",
                )


# ---------------------------------------------------------------------------
# Streaming kinetic-energy state tests (PR1: J3 streaming)
# ---------------------------------------------------------------------------


def _build_A_inv_from_carts(geminal_data, r_up_jnp, r_dn_jnp):
    """Compute A_inv = G(r_up, r_dn)^{-1} via SVD (matches the fast-update warning)."""
    A = compute_geminal_all_elements(
        geminal_data=geminal_data,
        r_up_carts=r_up_jnp,
        r_dn_carts=r_dn_jnp,
    )
    return jnp.linalg.inv(A)


def _streaming_step_consistency_one(wavefunction_data, r_up0, r_dn0, K, atol, rtol, seed=0):
    """Run K random single-electron moves through the streaming state and
    compare the resulting kinetic energies with a fresh fast-update call at
    the final configuration.
    """
    # Every caller declares this exact strict budget. Keep the helper arguments
    # checked so a caller cannot silently request a different comparison policy.
    expected_tolerance = get_tolerance_min(["wf_kinetic", "jastrow_grad_lap"], "strict")
    assert (atol, rtol) == expected_tolerance
    atol, rtol = get_tolerance_min(["wf_kinetic", "jastrow_grad_lap"], "strict")
    rng = np.random.RandomState(seed)
    r_up = np.asarray(r_up0, dtype=np.float64).copy()
    r_dn = np.asarray(r_dn0, dtype=np.float64).copy()
    n_up = r_up.shape[0]
    n_dn = r_dn.shape[0]

    A_inv = _build_A_inv_from_carts(wavefunction_data.geminal_data, jnp.asarray(r_up), jnp.asarray(r_dn))
    state = _init_kinetic_energy_all_elements_streaming_state(
        wavefunction_data=wavefunction_data,
        r_up_carts=jnp.asarray(r_up),
        r_dn_carts=jnp.asarray(r_dn),
        geminal_inverse=A_inv,
    )

    for _ in range(K):
        choices = []
        if n_up > 0:
            choices.append(0)
        if n_dn > 0:
            choices.append(1)
        spin = choices[rng.randint(0, len(choices))]
        if spin == 0:
            idx = rng.randint(0, n_up)
            r_up = r_up.copy()
            r_up[idx] = r_up[idx] + 0.05 * rng.randn(3)
            moved_spin_is_up = True
            moved_index = idx
        else:
            idx = rng.randint(0, n_dn)
            r_dn = r_dn.copy()
            r_dn[idx] = r_dn[idx] + 0.05 * rng.randn(3)
            moved_spin_is_up = False
            moved_index = idx

        # rebuild A_inv at the new configuration (mirrors what Sherman-Morrison
        # produces in the GFMC loop, modulo round-off -- comparing at the same
        # numerical reference here).
        A_inv = _build_A_inv_from_carts(wavefunction_data.geminal_data, jnp.asarray(r_up), jnp.asarray(r_dn))
        state = _advance_kinetic_energy_all_elements_streaming_state(
            wavefunction_data=wavefunction_data,
            state=state,
            moved_spin_is_up=jnp.asarray(moved_spin_is_up),
            moved_index=jnp.asarray(moved_index, dtype=jnp.int32),
            r_up_carts_new=jnp.asarray(r_up),
            r_dn_carts_new=jnp.asarray(r_dn),
            A_new_inv=A_inv,
        )

    ke_up_stream, ke_dn_stream = _kinetic_energy_from_streaming_state(state)
    ke_up_fresh, ke_dn_fresh = compute_kinetic_energy_all_elements_fast_update(
        wavefunction_data=wavefunction_data,
        r_up_carts=jnp.asarray(r_up),
        r_dn_carts=jnp.asarray(r_dn),
        geminal_inverse=A_inv,
    )
    np.testing.assert_allclose(np.asarray(ke_up_stream), np.asarray(ke_up_fresh), atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(ke_dn_stream), np.asarray(ke_dn_fresh), atol=atol, rtol=rtol)


def _build_wavefunction_J3(trexio_file, j2_type="exp", with_J1=False, with_J2=True):
    """Build a Wavefunction_data with J3 + optional J1/J2 from a trexio file.

    PR1 streaming requires J3 to be present (the dispatch demands it).
    """
    from jqmc.jastrow_factor import Jastrow_one_body_data

    (
        structure_data,
        aos_data,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
        store_tuple=True,
    )

    if with_J1:
        jastrow_one_body_data = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=0.5,
            structure_data=structure_data,
            core_electrons=core_electrons_of(coulomb_potential_data, len(structure_data.atomic_numbers)),
            jastrow_1b_type="pade",
        )
    else:
        jastrow_one_body_data = None

    if with_J2:
        jastrow_two_body_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, jastrow_2b_type=j2_type)
    else:
        jastrow_two_body_data = None

    jastrow_three_body_data = Jastrow_three_body_data.init_jastrow_three_body_data(
        orb_data=aos_data, random_init=True, random_scale=1.0e-3
    )

    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_one_body_data,
        jastrow_two_body_data=jastrow_two_body_data,
        jastrow_three_body_data=jastrow_three_body_data,
    )
    wavefunction_data = Wavefunction_data(geminal_data=geminal_mo_data, jastrow_data=jastrow_data)
    return wavefunction_data, geminal_mo_data


class TestStreamingKineticState:
    """Streaming kinetic-energy state advance/consistency tests."""

    @pytest.mark.parametrize(
        "trexio_file",
        ["water_ccecp_ccpvqz.h5", "H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"],
    )
    @pytest.mark.light
    def test_streaming_kinetic_energy_step_consistency(self, trexio_file):
        """K=32 random single-electron moves advanced via the streaming kinetic
        state must reproduce the fresh fast-update kinetic energy at the resulting
        configuration within strict tolerance.
        """
        wf, gem = _build_wavefunction_J3(trexio_file)
        n_up = gem.num_electron_up
        n_dn = gem.num_electron_dn
        rng = np.random.RandomState(0)
        r_up0 = 4.0 * rng.rand(n_up, 3) - 2.0
        r_dn0 = 4.0 * rng.rand(n_dn, 3) - 2.0
        atol, rtol = get_tolerance_min(["wf_kinetic", "jastrow_grad_lap"], "strict")
        _streaming_step_consistency_one(wf, r_up0, r_dn0, K=32, atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("K", [32, 100, 1000])
    def test_streaming_kinetic_drift_accumulation(self, K):
        """Drift accumulation: K-step advance vs fresh init at config_K must stay
        within strict tolerance even at K=1000. Empirically the streaming and
        fresh paths agree to machine precision because the kinetic-energy
        assembly (`wf_kinetic` zone, fp64) is recomputed in both paths from the
        same per-step inputs.
        """
        wf, gem = _build_wavefunction_J3("H2_ae_ccpvdz_cart.h5")
        rng = np.random.RandomState(1)
        r_up0 = 4.0 * rng.rand(gem.num_electron_up, 3) - 2.0
        r_dn0 = 4.0 * rng.rand(gem.num_electron_dn, 3) - 2.0
        atol, rtol = get_tolerance_min(["wf_kinetic", "jastrow_grad_lap"], "strict")
        _streaming_step_consistency_one(wf, r_up0, r_dn0, K=K, atol=atol, rtol=rtol, seed=2)

    @pytest.mark.parametrize(
        "trexio_file",
        ["H2_ae_ccpvdz_cart.h5", "Li_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5"],
    )
    @pytest.mark.light
    def test_streaming_kinetic_edge_cases(self, trexio_file):
        """Edge cases: small electron counts and ``N_up != N_dn`` (Li, N) must
        still match the fresh fast-update result.
        """
        wf, gem = _build_wavefunction_J3(trexio_file)
        rng = np.random.RandomState(3)
        r_up0 = 4.0 * rng.rand(gem.num_electron_up, 3) - 2.0
        r_dn0 = 4.0 * rng.rand(gem.num_electron_dn, 3) - 2.0
        atol, rtol = get_tolerance_min(["wf_kinetic", "jastrow_grad_lap"], "strict")
        _streaming_step_consistency_one(wf, r_up0, r_dn0, K=24, atol=atol, rtol=rtol, seed=4)

    @pytest.mark.light
    def test_streaming_kinetic_dtn_advance_consistency(self):
        """REGRESSION: the streaming advance once dropped the DTN
        grad/lap contribution entirely (totals were J1+J2+J3+det only), so any
        DTN wavefunction on the XLA streaming path lost the whole DTN kinetic
        term after the FIRST accepted move (~0.9 Ha ke error on water). The
        combo matrix below never contained a DTN case, which is why K-step
        consistency stayed green. This test puts DTN into the same
        advance-vs-fresh harness."""
        from jqmc.jastrow_factor import Jastrow_DTN_data, Jastrow_data
        from jqmc.trexio_wrapper import read_trexio_file

        trexio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "water_ccecp_ccpvqz.h5")
        sd, _aos, _mu, _md, gem, _c = read_trexio_file(trexio_file=trexio_path, store_tuple=True)
        dtn = Jastrow_DTN_data.build(
            structure_data=sd,
            z_eff_per_species=[6.0, 1.0],
            pseudized_per_species=[True, True],
            use_u=True,
            use_chi=True,
            use_f=True,
            cutoff_smoothness=4,
            n_order_eN=2,
            n_order_ee=2,
            L_f=2.0,
            random_init=True,
            seed=7,
        )
        wf = Wavefunction_data(geminal_data=gem, jastrow_data=Jastrow_data(jastrow_dtn_data=dtn))
        rng = np.random.RandomState(11)
        r_up0 = 4.0 * rng.rand(gem.num_electron_up, 3) - 2.0
        r_dn0 = 4.0 * rng.rand(gem.num_electron_dn, 3) - 2.0
        atol, rtol = get_tolerance_min(["wf_kinetic", "jastrow_grad_lap"], "strict")
        _streaming_step_consistency_one(wf, r_up0, r_dn0, K=8, atol=atol, rtol=rtol, seed=13)

    @pytest.mark.light
    @pytest.mark.parametrize("jastrow_combo", ["J3_only", "J1_J3", "J2_J3", "J1_J2_J3"])
    def test_streaming_kinetic_jastrow_combinations(self, jastrow_combo):
        """Streaming path must work for every J3-containing Jastrow combination."""
        with_J1 = "J1" in jastrow_combo
        with_J2 = "J2" in jastrow_combo
        wf, gem = _build_wavefunction_J3("water_ccecp_ccpvqz.h5", with_J1=with_J1, with_J2=with_J2)
        rng = np.random.RandomState(5)
        r_up0 = 4.0 * rng.rand(gem.num_electron_up, 3) - 2.0
        r_dn0 = 4.0 * rng.rand(gem.num_electron_dn, 3) - 2.0
        atol, rtol = get_tolerance_min(["wf_kinetic", "jastrow_grad_lap"], "strict")
        _streaming_step_consistency_one(wf, r_up0, r_dn0, K=24, atol=atol, rtol=rtol, seed=6)

    @pytest.mark.light
    def test_streaming_kinetic_walker_axis_vmap(self):
        """``vmap`` over the walker axis must produce results equal to the
        independent per-walker streaming chains. Confirms the state pytree carries
        walkers correctly along the leading axis.
        """
        wf, gem = _build_wavefunction_J3("H2_ae_ccpvdz_cart.h5")
        n_walkers = 4
        rng = np.random.RandomState(7)
        r_up_w = jnp.asarray(4.0 * rng.rand(n_walkers, gem.num_electron_up, 3) - 2.0)
        r_dn_w = jnp.asarray(4.0 * rng.rand(n_walkers, gem.num_electron_dn, 3) - 2.0)

        # Per-walker A_inv and initial state, computed via vmap.
        def _make_init_state(r_up, r_dn):
            A_inv = _build_A_inv_from_carts(wf.geminal_data, r_up, r_dn)
            return _init_kinetic_energy_all_elements_streaming_state(
                wavefunction_data=wf,
                r_up_carts=r_up,
                r_dn_carts=r_dn,
                geminal_inverse=A_inv,
            ), A_inv

        states, A_invs = jax.vmap(_make_init_state, in_axes=(0, 0))(r_up_w, r_dn_w)

        # Single up-electron move on walker 0 only; other walkers see the same
        # advance call but with their own (unchanged) inputs.
        moved_spin_is_up = jnp.asarray([True] * n_walkers)
        moved_index = jnp.asarray([0] * n_walkers, dtype=jnp.int32)

        # Apply the same delta to electron 0 across walkers (just to exercise the
        # vmap; the per-walker chains remain independent because `state` and
        # `r_*_carts_new` are walker-batched).
        delta = 0.05 * rng.randn(3)
        r_up_w_new = r_up_w.at[:, 0, :].add(jnp.asarray(delta))
        A_invs_new = jax.vmap(lambda ru, rd: _build_A_inv_from_carts(wf.geminal_data, ru, rd), in_axes=(0, 0))(
            r_up_w_new, r_dn_w
        )

        advance_vmapped = jax.vmap(
            lambda st, msi, mi, ru, rd, ai: _advance_kinetic_energy_all_elements_streaming_state(
                wavefunction_data=wf,
                state=st,
                moved_spin_is_up=msi,
                moved_index=mi,
                r_up_carts_new=ru,
                r_dn_carts_new=rd,
                A_new_inv=ai,
            ),
            in_axes=(0, 0, 0, 0, 0, 0),
        )
        states_new = advance_vmapped(states, moved_spin_is_up, moved_index, r_up_w_new, r_dn_w, A_invs_new)

        # Reference: fresh evaluation per walker.
        ke_up_v, ke_dn_v = jax.vmap(_kinetic_energy_from_streaming_state)(states_new)
        atol, rtol = get_tolerance_min(["wf_kinetic", "jastrow_grad_lap"], "strict")
        for w in range(n_walkers):
            walker_label = str(w)
            ke_up_ref, ke_dn_ref = compute_kinetic_energy_all_elements_fast_update(
                wavefunction_data=wf,
                r_up_carts=r_up_w_new[w],
                r_dn_carts=r_dn_w[w],
                geminal_inverse=A_invs_new[w],
            )
            np.testing.assert_allclose(np.asarray(ke_up_v[w]), np.asarray(ke_up_ref), atol=atol, rtol=rtol)
            np.testing.assert_allclose(np.asarray(ke_dn_v[w]), np.asarray(ke_dn_ref), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# j3_state-forwarding consistency tests (PR4/PR5: ECP non-local AO reuse +
# discretized kinetic AO reuse)
#
# These verify the Python-static dispatch in
# ``_compute_ratio_Jastrow_part_rank1_update`` and
# ``_compute_ratio_Jastrow_part_split_spin``: the with-state path must produce
# identical Jastrow ratios (and therefore identical kinetic / ECP elements) as
# the no-state path when the streaming state is consistent with
# ``(r_up_carts, r_dn_carts)``.
# ---------------------------------------------------------------------------


class TestStreamingStateDispatches:
    """j3_state-forwarding consistency for ratio-kernel dispatches."""

    @pytest.mark.parametrize(
        "trexio_file,jastrow_combo",
        [
            ("water_ccecp_ccpvqz.h5", "J3_only"),
            ("water_ccecp_ccpvqz.h5", "J1_J2_J3"),
            ("H2_ae_ccpvdz_cart.h5", "J2_J3"),
            ("N_ae_ccpvdz_cart.h5", "J1_J3"),
        ],
    )
    @pytest.mark.light
    def test_streaming_discretized_kinetic_j3_state_consistency(self, trexio_file, jastrow_combo):
        """``compute_discretized_kinetic_energy_fast_update`` must return the same
        kinetic mesh elements whether the J3 streaming state is forwarded or not.

        Validates the Python-static dispatch in
        ``_compute_ratio_Jastrow_part_rank1_update`` (the ratio kernel called by
        the discretized kinetic for the LRDMC mesh).
        """
        with_J1 = "J1" in jastrow_combo
        with_J2 = "J2" in jastrow_combo
        wf, gem = _build_wavefunction_J3(trexio_file, with_J1=with_J1, with_J2=with_J2)
        rng = np.random.RandomState(11)
        r_up = jnp.asarray(4.0 * rng.rand(gem.num_electron_up, 3) - 2.0)
        r_dn = jnp.asarray(4.0 * rng.rand(gem.num_electron_dn, 3) - 2.0)
        A_inv = _build_A_inv_from_carts(wf.geminal_data, r_up, r_dn)
        state = _init_kinetic_energy_all_elements_streaming_state(
            wavefunction_data=wf, r_up_carts=r_up, r_dn_carts=r_dn, geminal_inverse=A_inv
        )

        alat = 0.40
        RT = jnp.eye(3, dtype=jnp.float64)

        rup_ref, rdn_ref, ke_ref = compute_discretized_kinetic_energy_fast_update(
            alat=alat,
            wavefunction_data=wf,
            A_old_inv=A_inv,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
            RT=RT,
            j3_state=None,
        )
        rup_st, rdn_st, ke_st = compute_discretized_kinetic_energy_fast_update(
            alat=alat,
            wavefunction_data=wf,
            A_old_inv=A_inv,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
            RT=RT,
            j3_state=state.j3_state,
        )

        atol, rtol = get_tolerance("wf_kinetic", "strict")
        np.testing.assert_allclose(np.asarray(rup_st), np.asarray(rup_ref), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(rdn_st), np.asarray(rdn_ref), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(ke_st), np.asarray(ke_ref), atol=atol, rtol=rtol)

    @pytest.mark.light
    @pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5"])
    @pytest.mark.parametrize("jastrow_combo", ["J3_only", "J2_J3", "J1_J2_J3"])
    def test_streaming_ecp_nonlocal_j3_state_consistency(self, trexio_file, jastrow_combo):
        """``compute_ecp_non_local_parts_nearest_neighbors_fast_update`` (tmove
        path, ``flag_determinant_only=False``) must return identical V_nonlocal
        whether the J3 streaming state is forwarded or not.

        Validates the Python-static dispatch in
        ``_compute_ratio_Jastrow_part_split_spin`` (the ratio kernel used for the
        block-structured non-local ECP grid).
        """
        from jqmc.coulomb_potential import compute_ecp_non_local_parts_nearest_neighbors_fast_update

        with_J1 = "J1" in jastrow_combo
        with_J2 = "J2" in jastrow_combo
        (_, _, _, _, _, coulomb_potential_data) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )
        wf, gem = _build_wavefunction_J3(trexio_file, with_J1=with_J1, with_J2=with_J2)
        rng = np.random.RandomState(13)
        r_up = jnp.asarray(4.0 * rng.rand(gem.num_electron_up, 3) - 2.0)
        r_dn = jnp.asarray(4.0 * rng.rand(gem.num_electron_dn, 3) - 2.0)
        A_inv = _build_A_inv_from_carts(wf.geminal_data, r_up, r_dn)
        state = _init_kinetic_energy_all_elements_streaming_state(
            wavefunction_data=wf, r_up_carts=r_up, r_dn_carts=r_dn, geminal_inverse=A_inv
        )

        RT = jnp.eye(3, dtype=jnp.float64)

        rup_ref, rdn_ref, V_ref, sV_ref = compute_ecp_non_local_parts_nearest_neighbors_fast_update(
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wf,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
            RT=RT,
            A_old_inv=A_inv,
            flag_determinant_only=False,
            j3_state=None,
        )
        rup_st, rdn_st, V_st, sV_st = compute_ecp_non_local_parts_nearest_neighbors_fast_update(
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wf,
            r_up_carts=r_up,
            r_dn_carts=r_dn,
            RT=RT,
            A_old_inv=A_inv,
            flag_determinant_only=False,
            j3_state=state.j3_state,
        )

        atol, rtol = get_tolerance("wf_kinetic", "strict")
        np.testing.assert_allclose(np.asarray(rup_st), np.asarray(rup_ref), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(rdn_st), np.asarray(rdn_ref), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(V_st), np.asarray(V_ref), atol=atol, rtol=rtol)
        np.testing.assert_allclose(np.asarray(sV_st), np.asarray(sV_ref), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Periodic kinetic-energy tests on top of PW / B-spline / pGTO / composite
# leaves (dummy instances, no external TREXIO / QE file).
#
# evaluate_wavefunction / compute_kinetic_energy / _compute_kinetic_energy_auto
# route through compute_AOs / compute_MOs and their grad / Laplacian siblings,
# so the kinetic-energy path is leaf-agnostic. Building a tiny periodic geminal
# (closed-shell, 2 up + 2 dn) over each periodic leaf / composite mirrors the
# molecular TREXIO-driven tests above without any file dependency.
#
# Only a 2-body Jastrow is attached: the 3-body Jastrow seeds its orbital basis
# from GTO exponents, which the PW / B-spline leaves do not expose, so keeping
# J2-only here lets the SAME setup cover every periodic basis uniformly.
# ---------------------------------------------------------------------------

_PWF_A = 6.0  # cubic cell edge (Bohr)
_PWF_BSPLINE_NS = (12, 12, 12)
_PWF_BSPLINE_ORDER = 5
_PWF_PERIODIC_KINDS = ["pw", "bspline", "pgto", "pgto+pw", "pgto+bspline"]


def _pwf_cubic_structure(a: float = _PWF_A) -> Structure_data:
    """Single-atom cubic periodic cell shared by all periodic WF leaves."""
    cell = np.eye(3, dtype=np.float64) * a
    return Structure_data.build(
        pbc_flag=True,
        positions=np.zeros((1, 3), dtype=np.float64),
        vec_a=tuple(cell[0]),
        vec_b=tuple(cell[1]),
        vec_c=tuple(cell[2]),
        atomic_numbers=(1,),
        element_symbols=("H",),
        atomic_labels=("H1",),
    )


def _pwf_pw_leaf(structure: Structure_data):
    """Real cos/sin plane-wave leaf: constant + 5 (cos, sin) pairs = 11 AOs."""
    recip = structure.recip_cell
    millers = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, -1, 1]],
        dtype=np.float64,
    )
    g_list: list[np.ndarray] = []
    is_sin: list[float] = []
    for m in millers:
        g = m @ recip
        if np.all(m == 0):
            g_list.append(g)
            is_sin.append(0.0)
        else:
            g_list.append(g)
            is_sin.append(0.0)
            g_list.append(g)
            is_sin.append(1.0)
    g_vectors = np.asarray(g_list, dtype=np.float64)
    phase_is_sin = np.asarray(is_sin, dtype=np.float64)
    return AOs_plane_wave_data.build(structure, g_vectors, phase_is_sin)


def _pwf_bspline_bands(frac):
    """Two smooth periodic bands of fractional coords, shape ``(n_band, ...)``."""
    x, y, z = frac[..., 0], frac[..., 1], frac[..., 2]
    tp = 2.0 * np.pi
    b0 = np.sin(tp * x) + 0.5 * np.cos(tp * y) + 0.25 * np.sin(tp * z)
    b1 = np.cos(tp * (x + y)) + 0.3 * np.sin(tp * z)
    return np.stack([b0, b1], axis=0)


def _pwf_bspline_leaf(structure: Structure_data, order: int = _PWF_BSPLINE_ORDER):
    """Interpolating B-spline leaf reproducing two smooth periodic bands."""
    ns = _PWF_BSPLINE_NS
    grids = [np.arange(n) / n for n in ns]
    fx, fy, fz = np.meshgrid(grids[0], grids[1], grids[2], indexing="ij")
    frac = np.stack([fx, fy, fz], axis=-1)
    vals = _pwf_bspline_bands(frac)
    coeffs_cube = bspline_prefilter_periodic(vals.astype(np.complex128), order)
    coeffs_flat = coeffs_cube.reshape(coeffs_cube.shape[0], -1)
    return AOs_bspline_data.build(structure, coeffs_flat, ns, order)


def _pwf_pgto_leaf(structure: Structure_data):
    """Image-expanded periodic pGTO leaf: two moderate-exponent s shells (2 AOs)."""
    base = AOs_sphe_gto_data._from_full_basis(
        structure_data=structure,
        nucleus_index=(0, 0),
        num_ao=2,
        num_ao_prim=2,
        orbital_indices=(0, 1),
        exponents=np.array([0.8, 0.4], dtype=np.float64),
        coefficients=np.array([1.0, 1.0], dtype=np.float64),
        angular_momentums=(0, 0),
        magnetic_quantum_numbers=(0, 0),
    )
    return _apply_image_expansion(base, structure, AO_IMAGE_TRUNCATION_ATOL)


def _periodic_geminal_setup(kind: str, seed: int, fixed_positions=None):
    """Build ``(wavefunction_data, r_up_carts, r_dn_carts)`` for one periodic kind.

    Closed-shell 2-up / 2-dn AGP over a 2-MO periodic determinant plus a
    2-body Jastrow. Picks an electron configuration away from the node so the
    analytic-vs-autodiff kinetic comparison is well conditioned.
    """
    from jqmc.wavefunction import evaluate_wavefunction

    structure = _pwf_cubic_structure()
    if kind == "pw":
        aos_data = _pwf_pw_leaf(structure)
    elif kind == "bspline":
        aos_data = _pwf_bspline_leaf(structure)
    elif kind == "pgto":
        aos_data = _pwf_pgto_leaf(structure)
    elif kind == "pgto+pw":
        aos_data = _pwf_pgto_leaf(structure) + _pwf_pw_leaf(structure)
    elif kind == "pgto+bspline":
        aos_data = _pwf_pgto_leaf(structure) + _pwf_bspline_leaf(structure)
    else:
        raise ValueError(f"unknown periodic WF kind: {kind}")
    aos_data.sanity_check()

    num_mo = 2
    n_up = n_dn = 2
    rng = np.random.default_rng(seed)
    mo_coefficients = rng.standard_normal((num_mo, aos_data.num_ao)).astype(np.float64)
    mos_data = MOs_data(num_mo=num_mo, aos_data=aos_data, mo_coefficients=mo_coefficients)
    mos_data.sanity_check()

    lambda_paired = rng.uniform(-0.2, 0.2, size=(num_mo, num_mo))
    lambda_paired = 0.5 * (lambda_paired + lambda_paired.T) + np.eye(num_mo)
    lambda_matrix = np.hstack([lambda_paired, np.zeros((num_mo, 0), dtype=np.float64)])

    geminal_data = Geminal_data(
        num_electron_up=n_up,
        num_electron_dn=n_dn,
        orb_data_up_spin=mos_data,
        orb_data_dn_spin=mos_data,
        lambda_matrix=lambda_matrix,
    )
    geminal_data.sanity_check()

    jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=1.0, jastrow_2b_type="pade", structure_data=structure
    )
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=None,
    )
    wavefunction_data = Wavefunction_data(geminal_data=geminal_data, jastrow_data=jastrow_data)

    if fixed_positions is not None:
        r_up_carts, r_dn_carts = (np.array(value, copy=True) for value in fixed_positions)
        assert r_up_carts.shape == (n_up, 3) and r_dn_carts.shape == (n_dn, 3)
        assert abs(float(evaluate_wavefunction(wavefunction_data, r_up_carts, r_dn_carts))) > _KINETIC_NODE_FLOOR
        return wavefunction_data, r_up_carts, r_dn_carts

    a = _PWF_A
    r_up_carts = r_dn_carts = None
    # Deterministic node-far search budget. This dummy AGP has an intrinsically
    # tiny |Psi| scale (~1e-8) for random MO coefficients, so the fixed
    # ``|Psi| > 1e-8`` acceptance is marginal: with the (now correct) PERIODIC
    # Pade J2 -- ``structure_data`` is threaded above so J2 uses the Williamson
    # metric, not the Euclidean norm on raw coords -- ``kind="pgto"`` at
    # ``seed=4321`` first clears the threshold at try #289 (vs #20 under the old
    # Euclidean-J2 bug, where exactly 1 of 50 tries passed by a hair). The budget
    # is sized to reliably locate a node-far config for every kind/seed; the
    # acceptance threshold and the downstream kinetic tolerances are unchanged.
    for _ in range(1000):
        r_up_carts = rng.uniform(0.5, a - 0.5, size=(n_up, 3)).astype(np.float64)
        r_dn_carts = rng.uniform(0.5, a - 0.5, size=(n_dn, 3)).astype(np.float64)
        psi_val = evaluate_wavefunction(wavefunction_data, r_up_carts, r_dn_carts)
        if abs(float(psi_val)) > 1.0e-8:
            break
    else:
        pytest.fail(
            f"Could not find a node-far configuration for periodic kind {kind} with the fixed"
            " seed; this precondition search is deterministic and a failure indicates"
            " evaluate_wavefunction regressed, not an environment fluke"
        )
    return wavefunction_data, r_up_carts, r_dn_carts


class TestPeriodicWavefunction:
    """Kinetic, value, and log-derivative checks on periodic AO leaves."""

    @pytest.mark.light
    @pytest.mark.parametrize("kind", _PWF_PERIODIC_KINDS)
    def test_kinetic_energy_periodic_analytic_and_auto(self, kind: str):
        """Analytic vs autodiff local kinetic energy on a periodic geminal.

        PBC sibling of :py:func:`test_kinetic_energy_analytic_and_auto`, exercising
        the periodic PW / B-spline / pGTO / pGTO+PW / pGTO+B-spline bases through a
        dummy 2-up / 2-dn AGP. Confirms the full kinetic path (geminal det + 2-body
        Jastrow + AO grad / Laplacian) stays consistent across every periodic leaf.
        """
        if is_mixed_precision_enabled():
            _, source_up, source_dn = _periodic_geminal_setup(kind, seed=2025)
            fixed_positions = (source_up.copy(), source_dn.copy())
            control_atol, control_rtol = get_tolerance_min(
                ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "wf_kinetic"), "strict"
            )
            mixed, full = _kinetic_production_modes(
                lambda: _periodic_geminal_setup(kind, seed=2025, fixed_positions=fixed_positions), control_atol, control_rtol
            )
            blocks = mixed_fidelity_blocks(mixed, full)
            atol, rtol = get_tolerance_mixed_fidelity("orbital_derivative")
            for mixed_block, full_block in blocks:
                _assert_component_derivative_fidelity(mixed_block, full_block, atol, rtol)
            corrupted = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
            with pytest.raises(AssertionError):
                _assert_component_derivative_fidelity(corrupted, blocks[0][1], atol, rtol)
            jax.clear_caches()
            return

        wavefunction_data, r_up_carts, r_dn_carts = _periodic_geminal_setup(kind, seed=2025)

        K_analytic = compute_kinetic_energy(wavefunction_data=wavefunction_data, r_up_carts=r_up_carts, r_dn_carts=r_dn_carts)
        K_auto = _compute_kinetic_energy_auto(
            wavefunction_data=wavefunction_data,
            r_up_carts=jnp.asarray(r_up_carts),
            r_dn_carts=jnp.asarray(r_dn_carts),
        )

        atol, rtol = get_tolerance_min(
            ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "wf_kinetic"),
            "strict",
        )
        assert not np.any(np.isnan(np.asarray(K_analytic))), f"NaN in analytic K ({kind})"
        assert not np.any(np.isnan(np.asarray(K_auto))), f"NaN in autodiff K ({kind})"
        np.testing.assert_allclose(K_analytic, K_auto, atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("kind", _PWF_PERIODIC_KINDS)
    def test_kinetic_energy_all_elements_periodic_analytic_and_auto(self, kind: str):
        """Per-electron analytic vs autodiff kinetic energy on a periodic geminal."""
        if is_mixed_precision_enabled():
            _, source_up, source_dn = _periodic_geminal_setup(kind, seed=4321)
            fixed_positions = (source_up.copy(), source_dn.copy())
            control_atol, control_rtol = get_tolerance_min(
                ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "wf_kinetic"), "strict"
            )
            mixed, full = _kinetic_production_modes(
                lambda: _periodic_geminal_setup(kind, seed=4321, fixed_positions=fixed_positions), control_atol, control_rtol
            )
            blocks = mixed_fidelity_blocks(mixed, full)
            atol, rtol = get_tolerance_mixed_fidelity("orbital_derivative")
            for mixed_block, full_block in blocks:
                _assert_component_derivative_fidelity(mixed_block, full_block, atol, rtol)
            corrupted = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
            with pytest.raises(AssertionError):
                _assert_component_derivative_fidelity(corrupted, blocks[0][1], atol, rtol)
            jax.clear_caches()
            return

        wavefunction_data, r_up_carts, r_dn_carts = _periodic_geminal_setup(kind, seed=4321)
        r_up_jnp = jnp.asarray(r_up_carts)
        r_dn_jnp = jnp.asarray(r_dn_carts)

        K_up_auto, K_dn_auto = _compute_kinetic_energy_all_elements_auto(
            wavefunction_data=wavefunction_data, r_up_carts=r_up_jnp, r_dn_carts=r_dn_jnp
        )
        K_up_an, K_dn_an = compute_kinetic_energy_all_elements(
            wavefunction_data=wavefunction_data, r_up_carts=r_up_jnp, r_dn_carts=r_dn_jnp
        )

        atol, rtol = get_tolerance_min(
            ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "wf_kinetic"),
            "strict",
        )
        assert not np.any(np.isnan(np.asarray(K_up_auto))), f"NaN in autodiff K_up ({kind})"
        assert not np.any(np.isnan(np.asarray(K_up_an))), f"NaN in analytic K_up ({kind})"
        np.testing.assert_allclose(K_up_auto, K_up_an, atol=atol, rtol=rtol)
        assert not np.any(np.isnan(np.asarray(K_dn_auto))), f"NaN in autodiff K_dn ({kind})"
        assert not np.any(np.isnan(np.asarray(K_dn_an))), f"NaN in analytic K_dn ({kind})"
        np.testing.assert_allclose(K_dn_auto, K_dn_an, atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("kind", _PWF_PERIODIC_KINDS)
    def test_evaluate_wavefunction_periodic_value(self, kind: str):
        """Wavefunction value (``Psi`` and ``ln|Psi|``) is finite and self-consistent.

        Verifies the public ``evaluate_wavefunction`` (returns ``Psi``) and
        ``evaluate_ln_wavefunction`` (returns ``ln|Psi|``) on a periodic dummy
        geminal across every periodic AO leaf / composite kind. Self-consistency
        check: ``|Psi|`` from ``evaluate_wavefunction`` must equal
        ``exp(ln|Psi|)`` from ``evaluate_ln_wavefunction``.
        """
        from jqmc.wavefunction import evaluate_ln_wavefunction, evaluate_wavefunction

        wavefunction_data, r_up_carts, r_dn_carts = _periodic_geminal_setup(kind, seed=11)
        r_up_jnp = jnp.asarray(r_up_carts)
        r_dn_jnp = jnp.asarray(r_dn_carts)

        psi = evaluate_wavefunction(wavefunction_data, r_up_jnp, r_dn_jnp)
        ln_abs_psi = evaluate_ln_wavefunction(wavefunction_data, r_up_jnp, r_dn_jnp)

        psi_np = np.asarray(psi)
        ln_abs_psi_np = np.asarray(ln_abs_psi)
        assert psi_np.shape == (), f"Psi must be scalar, got shape {psi_np.shape} ({kind})"
        assert ln_abs_psi_np.shape == (), f"ln|Psi| must be scalar, got shape {ln_abs_psi_np.shape} ({kind})"
        assert np.isfinite(np.abs(psi_np)), f"|Psi| not finite ({kind}): {psi_np}"
        assert np.isfinite(ln_abs_psi_np), f"ln|Psi| not finite ({kind}): {ln_abs_psi_np}"

        atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "wf_kinetic"), "strict")
        np.testing.assert_allclose(np.log(np.abs(psi_np)), ln_abs_psi_np, atol=atol, rtol=rtol)
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.parametrize("kind", _PWF_PERIODIC_KINDS)
    def test_periodic_log_gradient_shape_and_finiteness_both_precisions(self, kind: str):
        """Both electron log-gradient blocks are finite, nonzero, and have the expected shape and dtype.

        Use the same synthetic periodic geminal, J2 Pade, five orbital kinds,
        and seed-22 node-far configuration as the full-only finite-difference
        anchor. No TREXIO file is used. The existing builder selects by the
        wavefunction magnitude, not by cond(G). Require both spin blocks in
        one verdict so one passing block cannot stand for the other. The
        coordinate-gradient dtype matches the differentiated coordinates;
        both spin blocks must contain a nonzero entry on these fixed inputs.

        This is a shape and NaN/Inf check in each precision mode, not an
        accuracy reference or a replacement for the full-only FD comparison.
        """
        from jqmc.wavefunction import evaluate_ln_wavefunction

        wavefunction_data, r_up_carts, r_dn_carts = _periodic_geminal_setup(kind, seed=22)
        r_up_jnp = jnp.asarray(r_up_carts)
        r_dn_jnp = jnp.asarray(r_dn_carts)
        grad_ln_up = jax.grad(evaluate_ln_wavefunction, argnums=1)(wavefunction_data, r_up_jnp, r_dn_jnp)
        grad_ln_dn = jax.grad(evaluate_ln_wavefunction, argnums=2)(wavefunction_data, r_up_jnp, r_dn_jnp)
        grad_ln_up_np = np.asarray(grad_ln_up)
        grad_ln_dn_np = np.asarray(grad_ln_dn)

        assert (
            grad_ln_up_np.shape == r_up_jnp.shape
            and grad_ln_dn_np.shape == r_dn_jnp.shape
            and grad_ln_up_np.dtype == r_up_jnp.dtype
            and grad_ln_dn_np.dtype == r_dn_jnp.dtype
            and np.all(np.isfinite(grad_ln_up_np))
            and np.all(np.isfinite(grad_ln_dn_np))
            and np.any(grad_ln_up_np != 0)
            and np.any(grad_ln_dn_np != 0)
        ), f"Invalid up/down electron log-gradient shape, dtype, NaN/Inf, or zero block ({kind})"
        jax.clear_caches()

    @pytest.mark.light
    @pytest.mark.skip_mixed_precision(
        reason="FD / numerical quadrature comparison dominated by float32 round-off",
        alternative_guarantee_ref="MATH-DERIV-LOG-WAVEFUNCTION-001",
    )
    @pytest.mark.parametrize("kind", _PWF_PERIODIC_KINDS)
    def test_evaluate_ln_wavefunction_periodic_log_derivative(self, kind: str):
        """``grad ln|Psi|`` w.r.t. electron coordinates is finite, correct shape, and FD-consistent.

        Exercises the autodiff log-derivative path (the same ``jax.grad`` of
        ``evaluate_ln_wavefunction`` used inside ``_compute_nodal_distance_debug``)
        on the periodic dummy geminal across every periodic leaf / composite kind.
        Compares against a central-FD finite difference of ``ln|Psi|`` to confirm
        the gradient is numerically correct, not just finite.
        """
        from jqmc.wavefunction import evaluate_ln_wavefunction

        wavefunction_data, r_up_carts, r_dn_carts = _periodic_geminal_setup(kind, seed=22)
        r_up_jnp = jnp.asarray(r_up_carts)
        r_dn_jnp = jnp.asarray(r_dn_carts)
        n_up = r_up_jnp.shape[0]
        n_dn = r_dn_jnp.shape[0]

        grad_ln_up = jax.grad(evaluate_ln_wavefunction, argnums=1)(wavefunction_data, r_up_jnp, r_dn_jnp)
        grad_ln_dn = jax.grad(evaluate_ln_wavefunction, argnums=2)(wavefunction_data, r_up_jnp, r_dn_jnp)

        grad_ln_up_np = np.asarray(grad_ln_up)
        grad_ln_dn_np = np.asarray(grad_ln_dn)
        assert grad_ln_up_np.shape == (n_up, 3), f"grad ln|Psi| up shape ({kind}): {grad_ln_up_np.shape}"
        assert grad_ln_dn_np.shape == (n_dn, 3), f"grad ln|Psi| dn shape ({kind}): {grad_ln_dn_np.shape}"
        assert np.all(np.isfinite(grad_ln_up_np)), f"NaN/Inf in grad ln|Psi| up ({kind})"
        assert np.all(np.isfinite(grad_ln_dn_np)), f"NaN/Inf in grad ln|Psi| dn ({kind})"

        # Central-FD reference of ln|Psi|. Step size 1e-4 Bohr; the surrounding
        # cell edge is 6 Bohr and the configurations come from
        # _periodic_geminal_setup which already rejects node-close points, so the
        # FD ratio is well behaved.
        h = 1.0e-4
        r_up_np = np.asarray(r_up_jnp, dtype=np.float64).copy()
        r_dn_np = np.asarray(r_dn_jnp, dtype=np.float64).copy()
        fd_up = np.zeros_like(r_up_np)
        fd_dn = np.zeros_like(r_dn_np)
        for i in range(n_up):
            for d in range(3):
                r_plus = r_up_np.copy()
                r_minus = r_up_np.copy()
                r_plus[i, d] += h
                r_minus[i, d] -= h
                f_plus = float(evaluate_ln_wavefunction(wavefunction_data, jnp.asarray(r_plus), r_dn_jnp))
                f_minus = float(evaluate_ln_wavefunction(wavefunction_data, jnp.asarray(r_minus), r_dn_jnp))
                fd_up[i, d] = (f_plus - f_minus) / (2.0 * h)
        for i in range(n_dn):
            for d in range(3):
                r_plus = r_dn_np.copy()
                r_minus = r_dn_np.copy()
                r_plus[i, d] += h
                r_minus[i, d] -= h
                f_plus = float(evaluate_ln_wavefunction(wavefunction_data, r_up_jnp, jnp.asarray(r_plus)))
                f_minus = float(evaluate_ln_wavefunction(wavefunction_data, r_up_jnp, jnp.asarray(r_minus)))
                fd_dn[i, d] = (f_plus - f_minus) / (2.0 * h)

        # Central-FD truncation error is O(h^2) ~ 1e-8 in fp64, well within the
        # "medium" tolerance band shared with the kinetic / log-derivative paths.
        atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "wf_kinetic"), "medium")
        np.testing.assert_allclose(grad_ln_up_np, fd_up, atol=atol, rtol=rtol)
        np.testing.assert_allclose(grad_ln_dn_np, fd_dn, atol=atol, rtol=rtol)
        jax.clear_caches()


# ---------------------------------------------------------------------------
# RHF closed-shell singlet identity (complex Bloch path, non-Gamma)
# ---------------------------------------------------------------------------
#
# For a closed-shell RHF singlet at twist k, both spin sectors occupy the
# SAME +k orbital (the spatial wavefunction is symmetric under r_up <-> r_dn,
# which is the singlet condition). At the orbital level this means
#
#     compute_MOs(mos_dn, r) == compute_MOs(mos_up, r)    for all r.
#
# An earlier loader path applied ``conj(C)`` to the dn-spin coefficients
# under the heading "Convention B", which built a non-singlet trial WF
# that gave VMC energies below the pyscf HF reference at twist (-2 Ha
# for Si primitive cc-pVDZ, -20 Ha for Oxygen kUHF). This test pins the
# correct identity and is deterministic (no MCMC, no statistical noise).


class TestClosedShellIdentity:
    """Closed-shell singlet BCS / TR-pair invariant at non-Gamma twist."""

    @pytest.mark.light
    def test_closed_shell_singlet_dn_orbital_is_TR_partner_of_up(self):
        """Closed-shell singlet at twist k: BCS / TR-pair invariant
        ``psi_dn(r) == conj(psi_up(r))`` (= ``phi^{-k}(r)``).

        The dn channel evaluates the time-reversal partner of the up orbital;
        on a +k complex Bloch AO leaf for up and the matching -k Bloch leaf for
        dn, the kernel produces orbitals related by complex conjugation. This is
        the absolute invariant of the BCS / TR-pair singlet representation jqmc
        uses throughout the loader / builder / engine; an implementation that
        leaves dn at +k (the legacy Convention-A shortcut) violates this identity
        and disagrees with pyscf RHF / kRHF VMC by many Ha for localized systems.

        Operator-level deterministic check; sub-second. Uses the lithium_k025
        fixture (max |Im mo_coefficients| ~ 0.18) so the invariant is non-trivial
        (a Gaussian-decayed H-in-large-box fixture would give effectively real
        Bloch sums and the identity would hold trivially).
        """
        from jqmc.molecular_orbital import compute_MOs

        fixture = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "trexio_example_files",
            "lithium_k025.h5",
        )
        if not os.path.exists(fixture):
            pytest.skip(f"missing fixture {fixture}")
        structure_data, _aos, mos_up, mos_dn, _gem, _cp = read_trexio_file(trexio_file=fixture, store_tuple=True)
        assert np.iscomplexobj(np.asarray(mos_up.mo_coefficients)), (
            "fixture must carry complex MO coefficients (non-Gamma twist)"
        )

        cell = np.asarray(structure_data._cell_np)
        rng = np.random.default_rng(0)
        frac = rng.random((4, 3))
        r = jnp.asarray(frac @ cell)
        psi_up = np.asarray(compute_MOs(mos_up, r))
        psi_dn = np.asarray(compute_MOs(mos_dn, r))

        assert np.abs(psi_up.imag).max() > 1.0e-3, "psi_up must have a non-trivial Im part for this test to be informative"

        np.testing.assert_allclose(
            psi_dn,
            np.conj(psi_up),
            atol=ATOL_PYTHON_FP64_LOOSE,
            rtol=RTOL_PYTHON_FP64_LOOSE,
            err_msg=(
                "BCS / TR-pair singlet invariant psi_dn(r) == conj(psi_up(r)) "
                "violated at non-Gamma. Likely cause: the dn channel is built on "
                "the +k AO leaf instead of the -k Bloch leaf (with_negated_k()), "
                "or its mo_coefficients are not conjugated. The kernel produces a "
                "non-physical mixed-phase orbital that is neither phi^k nor "
                "phi^{-k} and disagrees with pyscf kRHF VMC by tens of Ha for "
                "localized systems such as H2O."
            ),
        )
        jax.clear_caches()


# ---------------------------------------------------------------------------
# VariationalParameterBlock update contracts.
# ---------------------------------------------------------------------------


class TestVariationalParameterBlock:
    """Typed updates after field-coordinate decoding."""

    @pytest.mark.light
    def test_real_coordinate_decode_supplies_a_real_storage_update(self):
        """The coordinate layer, not an update-time cast, owns real storage."""
        blk = make_test_block(
            name="jastrow_two_body",
            values=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            field="real",
            derivative_policy="direct_real",
            sync_scope="world",
        )
        delta = RealCoordinateMap.from_blocks([blk]).encode(np.asarray([0.5, -0.1, 0.2]))
        assert not np.iscomplexobj(delta)
        new = blk.apply_update(delta, learning_rate=0.1)
        np.testing.assert_array_equal(new.values.dtype, np.float64)
        expected = np.array([1.0, 2.0, 3.0]) + 0.1 * delta
        np.testing.assert_allclose(new.values, expected, atol=ATOL_PYTHON_FP64_TIGHT, rtol=RTOL_BIT_EXACT)
        # And the param's identity / shape is preserved.
        assert new.name == "jastrow_two_body"
        assert new.shape == (3,)
        assert new.size == 3

    @pytest.mark.light
    def test_apply_update_complex_param_keeps_complex_delta(self):
        """Complex-storage param + complex SR direction => Im(dX) preserved.

        For complex variational params (paired/unpaired lambda, complex MO
        coeffs) the SR direction is genuinely complex and drives the complex
        parameter step; the apply_update must NOT project to real.
        """
        values = np.array([1.0 + 0.5j, 2.0 - 0.3j], dtype=np.complex128)
        blk = make_test_block(
            name="lambda_matrix",
            values=values,
            field="complex",
            derivative_policy="holomorphic",
            sync_scope="sampler",
        )
        delta = np.array([0.5 + 7.0j, -0.1 + 0.3j], dtype=np.complex128)
        new = blk.apply_update(delta, learning_rate=0.1)
        np.testing.assert_array_equal(new.values.dtype, np.complex128)
        expected = values + 0.1 * delta
        np.testing.assert_array_equal(new.values, expected)

    @pytest.mark.light
    def test_apply_update_real_param_real_delta_bit_identical(self):
        """Real param + real delta => real-path bit-identity (no change from
        the complex projection, which only kicks in when ``dX`` is complex)."""
        blk = make_test_block(
            name="jastrow_one_body",
            values=np.array([0.7, -0.4], dtype=np.float64),
            field="real",
            derivative_policy="direct_real",
            sync_scope="world",
        )
        delta = np.array([0.1, 0.2], dtype=np.float64)
        new = blk.apply_update(delta, learning_rate=0.5)
        expected = np.array([0.7 + 0.05, -0.4 + 0.1])
        np.testing.assert_array_equal(new.values, expected)
        assert new.values.dtype == np.float64


if __name__ == "__main__":
    from logging import Formatter, StreamHandler, getLogger

    logger = getLogger("jqmc")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

"""Rust real-Gamma forces and stress of a PW-first pGTO(+)PW composite J3 orbital against AD.

The native real periodic J3 kernels place the rows of a pGTO(+)PW composite J3 orbital at
offsets that depend on the child order:

* ``forces_pbc.rs::j3_pbc_side_w`` (J3 kinetic force, nuclear axis) scatters only the pGTO
  rows of the composite cotangent, starting at ``let gto0 = if gto_first { 0 } else { num_pw }``;
* ``forces_pbc.rs::grad_j3_wrt_r_pbc`` (J3 value Pulay force) slices the pGTO rows at the same
  ``gto0``;
* ``stress_pbc.rs::stack_rows`` stacks the GTO and PW blocks in the child order for the J3
  kinetic stress and the J3 value strain;
* ``stress_pbc.rs::j3_pbc_kstress_w`` (J3 kinetic stress) places the GTO and PW rows at
  ``gto0`` / ``pw0``, and ``stress_pbc.rs::j3_pbc_value_strain`` (J3 value strain) the GTO rows
  at ``gto0``.

Every committed real composite J3 is pGTO-first, so the PW-first offsets were never reached:
replacing the force ``gto0`` of ``j3_pbc_side_w`` by ``0`` survived a 2985-node Rust selection
in full and mixed precision (issue #102). This is the real-Gamma twin of issue #83.

This module builds the committed ``pbc_composite_j123`` cell (QE Li Gamma ccECP, real
pGTO(+)PW single determinant kept pGTO-first, exp J1/J2 and an identity-MO-wrapped pGTO(+)PW
composite J3) with a seeded nonzero ``j_matrix`` in two J3 child orders:

* ``gto_first``: the J3 orbital as built (pGTO, PW);
* ``pw_first``: the J3 children reordered to (PW, pGTO) and the ``j_matrix`` rows/columns
  (and the J1-like last column) permuted to match, so the wave function, and therefore every
  observable, is unchanged.

For each order, at one seeded electron configuration:

* the Hellmann-Feynman derivatives ``d e_L / dR`` and ``d e_L / dr`` of the native VMC kernel
  ``jqmc_rs.vmc_grad_e_L_pbc``, of the MCMC driver entry ``NativeObsCtx.grad_e_L_pbc_batch``
  and of the GFMC/LRDMC measurement force ``jqmc_rs._gfmc_grad_e_L_ecp_pbc`` (T-move, whose
  measured quantity is the same continuum local energy), and the Pulay derivative
  ``d ln|Psi| / dR`` of ``jqmc_rs.vmc_grad_ln_psi_pbc`` and ``NativeObsCtx.grad_ln_psi_pbc_batch``,
  must equal reverse-mode AD of the XLA ``compute_local_energy`` (identity quadrature rotation,
  the same base mesh) and ``evaluate_ln_wavefunction`` in full precision. The
  nuclear derivative displaces every ``Structure_data.positions`` leaf of the Hamiltonian
  pytree together;
* the VMC Pulay and HF stress ``jqmc_rs.vmc_grad_ln_psi_eps_pbc`` / ``vmc_grad_e_L_eps_pbc``
  and their driver entries ``NativeObsCtx.grad_ln_psi_eps_pbc_batch`` /
  ``grad_e_L_eps_pbc_batch`` must equal reverse-mode AD of the same scalars with the cell,
  nuclei and electrons strained together (``Hamiltonian_data._apply_strain`` and ``r @ F.T``).

The references differentiate the scalar functions themselves and use no Rust code and no
analytic force or stress formula. In full precision the strain AD agrees with a fourth-order
central finite difference of the same functions to 1.8e-10 (recorded in issue #102).

Precision scope (#163): full retains the original independent energy / ln-wavefunction
AD formulas at strict. Mixed compares the same native analytical blocks with independently
rebuilt full native blocks from identical serialized physical input and deterministic
coordinates, under the central parameter-derivative fidelity budget. The original medium
nonzero thresholds are captured on mixed entry and checked in both analytical callbacks.
All native arguments, parameter cases and ordinary public-driver backend comparisons remain.

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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jqmc_rs
from jqmc._setting import get_tolerance_mixed_fidelity
from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block
from tests.test_rs_kernel._native_precision_fidelity import (
    full_source_hamiltonian_factory,
)
from jqmc._precision import get_tolerance_min, precision_policy_snapshot, is_mixed_precision_enabled
from jqmc.atomic_orbital import AOs_composite_data, AOs_plane_wave_data
from jqmc.hamiltonians import compute_local_energy
from jqmc.jqmc_mcmc import _NV_BASE_MESH, Nv_default, _geminal_inv_batched
from jqmc.structure import Structure_data
from jqmc.wavefunction import evaluate_ln_wavefunction
from tests.test_rs_kernel._rs_fixtures import rs_ham

jax.config.update("jax_enable_x64", True)

# The test reads only committed fixtures through absolute paths.
pytestmark = pytest.mark.with_chdir

#: Committed real-Gamma composite J123 pattern of ``_rs_fixtures``.
_PATTERN = "pbc_composite_j123"
_J_SEED = 7
_J_SCALE = 0.05
_SEED = 1
_SCALE = 0.6
#: GFMC lattice spacing; the T-move measurement force does not depend on it.
_ALAT = 0.3
_NN = 1
_WEIGHTS = np.asarray(_NV_BASE_MESH[Nv_default].weights, dtype=np.float64)
_GRID = np.asarray(_NV_BASE_MESH[Nv_default].grid_points, dtype=np.float64)
#: Zones crossed by the real J123 local energy, ln|Psi| and their nuclear / electron / strain derivatives.
_ZONES = (
    "ao_eval",
    "ao_grad_lap",
    "mo_eval",
    "mo_grad",
    "mo_lap",
    "det_eval",
    "det_grad_lap",
    "jastrow_eval",
    "jastrow_grad_lap",
    "wf_kinetic",
    "ewald",
    "coulomb",
)


def _hamiltonian(order):
    """The pbc_composite_j123 cell with a seeded j_matrix, J3 children in ``order``."""
    ham = rs_ham(_PATTERN)
    wf = ham.wavefunction_data
    jastrow = wf.jastrow_data
    j3 = jastrow.jastrow_three_body_data
    shape = np.asarray(j3.j_matrix).shape
    n = shape[0]
    j_matrix = _J_SCALE * np.random.default_rng(_J_SEED).standard_normal(shape)
    j_matrix[:, :n] = 0.5 * (j_matrix[:, :n] + j_matrix[:, :n].T)

    orb = j3.orb_data
    gto, pw = orb.aos_data.children
    assert isinstance(pw, AOs_plane_wave_data), orb.aos_data.children
    num_gto, num_pw = int(gto.num_ao), int(pw.num_ao)
    # Identity MO wrapper: the J3 orbital index is the AO index, so a child reorder is a
    # row/column permutation of j_matrix.
    np.testing.assert_array_equal(np.asarray(orb.mo_coefficients), np.eye(num_gto + num_pw))
    if order == "pw_first":
        perm = np.concatenate([np.arange(num_gto, num_gto + num_pw), np.arange(num_gto)])
        permuted = np.zeros_like(j_matrix)
        permuted[:, :n] = j_matrix[np.ix_(perm, perm)]
        permuted[:, n] = j_matrix[perm, n]
        j_matrix = permuted
        orb = orb.replace(aos_data=AOs_composite_data.build((pw, gto)))
        assert isinstance(orb.aos_data.children[0], AOs_plane_wave_data)
    # A numpy float64 j_matrix: the native driver context reads it as a numpy array.
    j3 = j3.replace(orb_data=orb, j_matrix=np.asarray(j_matrix, dtype=np.float64))
    wf = wf.replace(jastrow_data=jastrow.replace(jastrow_three_body_data=j3))
    return ham.replace(wavefunction_data=wf)


def _full_hamiltonian_source(order):
    """Build full physical input without a mode-agnostic cached Gamma model."""
    assert not is_mixed_precision_enabled()
    from tests.test_jqmc_kernel._grad_fidelity_harness import _qe_composite_pieces

    _qe_composite_pieces.cache_clear()
    return _hamiltonian(order)


def _configuration(ham):
    geminal = ham.wavefunction_data.geminal_data
    num_up, num_dn = int(geminal.num_electron_up), int(geminal.num_electron_dn)
    positions = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(_SEED)
    r_up = positions[np.arange(num_up) % len(positions)] + rng.normal(scale=_SCALE, size=(num_up, 3))
    r_dn = positions[np.arange(num_dn) % len(positions)] + rng.normal(scale=_SCALE, size=(num_dn, 3))
    return np.ascontiguousarray(r_up, dtype=np.float64), np.ascontiguousarray(r_dn, dtype=np.float64)


def _is_structure(x):
    return isinstance(x, Structure_data)


def _displaced(ham, d_positions):
    """Hamiltonian with every Structure_data.positions leaf displaced by ``d_positions``."""
    return jax.tree_util.tree_map(
        lambda x: x.replace(positions=jnp.asarray(x.positions) + d_positions) if _is_structure(x) else x,
        ham,
        is_leaf=_is_structure,
    )


def _local_energy(ham, r_up, r_dn):
    return jnp.real(compute_local_energy(ham, r_up, r_dn, jnp.eye(3)))


def _ln_psi(ham, r_up, r_dn):
    return jnp.real(evaluate_ln_wavefunction(ham.wavefunction_data, r_up, r_dn))


def _strain_ad(ham, r_up, r_dn, scalar):
    """Reverse-mode AD of ``scalar(strained_ham, r_up F^T, r_dn F^T)`` with respect to ``eps`` at 0."""

    def at_strain(eps):
        deformation = jnp.eye(3) + eps
        strained = ham._apply_strain(deformation)
        return scalar(strained, jnp.asarray(r_up) @ deformation.T, jnp.asarray(r_dn) @ deformation.T)

    return np.asarray(jax.jacrev(at_strain)(jnp.zeros((3, 3))), dtype=np.float64)


def _setup(order):
    ham = _hamiltonian(order)
    r_up, r_dn = _configuration(ham)
    wf = ham.wavefunction_data
    geminal = wf.geminal_data
    geminal_inv = np.ascontiguousarray(
        np.asarray(_geminal_inv_batched(geminal, jnp.asarray(r_up)[None], jnp.asarray(r_dn)[None])[1], dtype=np.float64)
    )
    return ham, r_up, r_dn, geminal, wf.jastrow_data, ham.coulomb_potential_data, geminal_inv


def assert_parameter_derivative_mixed_fidelity(mixed, full, atol, rtol):
    """Apply the central block budget at a source-qualified local receipt site."""
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()


def _native_force_outputs(ham, r_up, r_dn):
    """Original native analytical blocks, with unchanged kernel arguments."""
    wf = ham.wavefunction_data
    geminal, jastrow, coulomb = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    geminal_inv = np.ascontiguousarray(
        np.asarray(_geminal_inv_batched(geminal, jnp.asarray(r_up)[None], jnp.asarray(r_dn)[None])[1], dtype=np.float64)
    )
    free_hf = jqmc_rs.vmc_grad_e_L_pbc(geminal, jastrow, coulomb, r_up, r_dn, _WEIGHTS, _GRID, geminal_inv[0])
    free_pulay = jqmc_rs.vmc_grad_ln_psi_pbc(geminal, jastrow, r_up, r_dn, geminal_inv[0])
    ctx = jqmc_rs.build_obs_ctx(geminal, jastrow, coulomb, False, True, False, precision_policy=precision_policy_snapshot())
    ctx_hf = ctx.grad_e_L_pbc_batch(r_up[None], r_dn[None], _WEIGHTS, _GRID[None], geminal_inv)
    ctx_pulay = ctx.grad_ln_psi_pbc_batch(r_up[None], r_dn[None], geminal_inv)
    gfmc_hf = jqmc_rs._gfmc_grad_e_L_ecp_pbc(
        geminal, jastrow, coulomb, r_up, r_dn, _ALAT, np.eye(3), _WEIGHTS, _GRID, _NN, False
    )

    return {
        "HF dR (VMC free kernel)": np.asarray(free_hf[0], dtype=np.float64),
        "HF dR (VMC driver ctx)": np.asarray(np.asarray(ctx_hf[0])[0], dtype=np.float64),
        "HF dR (GFMC T-move)": np.asarray(gfmc_hf[0], dtype=np.float64),
        "HF dr_up (VMC free kernel)": np.asarray(free_hf[1], dtype=np.float64),
        "HF dr_up (VMC driver ctx)": np.asarray(np.asarray(ctx_hf[1])[0], dtype=np.float64),
        "HF dr_up (GFMC T-move)": np.asarray(gfmc_hf[1], dtype=np.float64),
        "HF dr_dn (VMC free kernel)": np.asarray(free_hf[2], dtype=np.float64),
        "HF dr_dn (VMC driver ctx)": np.asarray(np.asarray(ctx_hf[2])[0], dtype=np.float64),
        "HF dr_dn (GFMC T-move)": np.asarray(gfmc_hf[2], dtype=np.float64),
        "Pulay dR (VMC free kernel)": np.asarray(free_pulay[0], dtype=np.float64),
        "Pulay dR (VMC driver ctx)": np.asarray(np.asarray(ctx_pulay[0])[0], dtype=np.float64),
    }


@pytest.mark.light
@pytest.mark.parametrize("order", ["gto_first", "pw_first"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_real_j3_composite_forces_match_ad_in_both_child_orders(order):
    """Native real VMC / GFMC HF and Pulay forces equal AD of e_L and ln|Psi| for a (pGTO, PW) and a (PW, pGTO) J3 orbital."""
    if is_mixed_precision_enabled():
        entry_liveness = get_tolerance_min(_ZONES, "medium")

        def analytical(fresh):
            r_up, r_dn = _configuration(fresh)
            values = _native_force_outputs(fresh, r_up, r_dn)
            expected_fields = {
                "HF dr_dn (VMC free kernel)",
                "HF dr_up (VMC driver ctx)",
                "HF dr_dn (VMC driver ctx)",
                "HF dr_dn (GFMC T-move)",
                "Pulay dR (VMC driver ctx)",
                "Pulay dR (VMC free kernel)",
                "HF dr_up (VMC free kernel)",
                "HF dR (VMC free kernel)",
                "HF dR (VMC driver ctx)",
                "HF dR (GFMC T-move)",
                "HF dr_up (GFMC T-move)",
            }
            assert set(values) == expected_fields, values.keys()
            live_atol, live_rtol = entry_liveness
            for name, value in values.items():
                expected_shape = (
                    (3, 3)
                    if "stress" in name or "strain" in name
                    else r_up.shape
                    if "dr_up" in name
                    else r_dn.shape
                    if "dr_dn" in name
                    else np.asarray(fresh.structure_data.positions).shape
                )
                assert value.shape == expected_shape, (name, value.shape, expected_shape)
                assert np.all(np.isfinite(value)), (name, value)
                assert np.max(np.abs(value)) > live_atol + live_rtol * np.max(np.abs(value)), (name, value)
            return values

        factory = full_source_hamiltonian_factory(lambda: _full_hamiltonian_source(order))
        mixed, full = mixed_vs_full(
            factory,
            analytical,
            configure_native=jqmc_rs.configure_native_precision,
            native_policy_snapshot=jqmc_rs.native_precision_support,
        )
        blocks = mixed_fidelity_blocks(mixed, full)
        atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
        for mixed_block, full_block in blocks:
            assert_parameter_derivative_mixed_fidelity(mixed_block, full_block, atol, rtol)
        mutated = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            assert_parameter_derivative_mixed_fidelity(mutated, blocks[0][1], atol, rtol)
        return

    ham, r_up, r_dn, geminal, jastrow, coulomb, geminal_inv = _setup(order)
    up, dn = jnp.asarray(r_up), jnp.asarray(r_dn)
    zero = jnp.zeros_like(jnp.asarray(ham.structure_data.positions))
    hf_nuclei = np.asarray(jax.grad(lambda d: _local_energy(_displaced(ham, d), up, dn))(zero), dtype=np.float64)
    hf_up, hf_dn = (
        np.asarray(x, dtype=np.float64) for x in jax.grad(lambda a, b: _local_energy(ham, a, b), argnums=(0, 1))(up, dn)
    )
    pulay_nuclei = np.asarray(jax.grad(lambda d: _ln_psi(_displaced(ham, d), up, dn))(zero), dtype=np.float64)

    values = _native_force_outputs(ham, r_up, r_dn)

    checks = [
        ("HF dR (VMC free kernel)", values["HF dR (VMC free kernel)"], hf_nuclei),
        ("HF dR (VMC driver ctx)", values["HF dR (VMC driver ctx)"], hf_nuclei),
        ("HF dR (GFMC T-move)", values["HF dR (GFMC T-move)"], hf_nuclei),
        ("HF dr_up (VMC free kernel)", values["HF dr_up (VMC free kernel)"], hf_up),
        ("HF dr_up (VMC driver ctx)", values["HF dr_up (VMC driver ctx)"], hf_up),
        ("HF dr_up (GFMC T-move)", values["HF dr_up (GFMC T-move)"], hf_up),
        ("HF dr_dn (VMC free kernel)", values["HF dr_dn (VMC free kernel)"], hf_dn),
        ("HF dr_dn (VMC driver ctx)", values["HF dr_dn (VMC driver ctx)"], hf_dn),
        ("HF dr_dn (GFMC T-move)", values["HF dr_dn (GFMC T-move)"], hf_dn),
        ("Pulay dR (VMC free kernel)", values["Pulay dR (VMC free kernel)"], pulay_nuclei),
        ("Pulay dR (VMC driver ctx)", values["Pulay dR (VMC driver ctx)"], pulay_nuclei),
    ]
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    live_atol, live_rtol = get_tolerance_min(_ZONES, "medium")
    for name, value, reference in checks:
        value = np.asarray(value, dtype=np.float64)
        assert value.shape == reference.shape, (name, value.shape, reference.shape)
        assert np.all(np.isfinite(reference)), (name, reference)
        assert np.all(np.isfinite(value)), (name, value)
        # Non-vacuous: the reference is distinguishable from zero at the comparison tolerance.
        assert np.max(np.abs(reference)) > live_atol + live_rtol * np.max(np.abs(reference)), (name, reference)
        np.testing.assert_allclose(
            value,
            reference,
            atol=atol,
            rtol=rtol,
            equal_nan=False,
            err_msg=f"Rust real J3 composite force != AD reference: {name} (J3 {order})",
        )


# About 45 s per row on the CPU reference host (two strain jacrev of e_L), above the light budget.
def _native_stress_outputs(ham, r_up, r_dn):
    """Original native analytical blocks, with unchanged kernel arguments."""
    wf = ham.wavefunction_data
    geminal, jastrow, coulomb = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    geminal_inv = np.ascontiguousarray(
        np.asarray(_geminal_inv_batched(geminal, jnp.asarray(r_up)[None], jnp.asarray(r_dn)[None])[1], dtype=np.float64)
    )
    free_pulay = jqmc_rs.vmc_grad_ln_psi_eps_pbc(geminal, jastrow, r_up, r_dn, geminal_inv[0])
    free_hf = jqmc_rs.vmc_grad_e_L_eps_pbc(geminal, jastrow, coulomb, r_up, r_dn, _WEIGHTS, _GRID, geminal_inv[0])
    ctx = jqmc_rs.build_obs_ctx(geminal, jastrow, coulomb, False, True, True, precision_policy=precision_policy_snapshot())
    ctx_pulay = ctx.grad_ln_psi_eps_pbc_batch(r_up[None], r_dn[None], geminal_inv)
    ctx_hf = ctx.grad_e_L_eps_pbc_batch(r_up[None], r_dn[None], _WEIGHTS, _GRID[None], geminal_inv)

    return {
        "Pulay strain (VMC free kernel)": np.asarray(free_pulay, dtype=np.float64),
        "Pulay strain (VMC driver ctx)": np.asarray(np.asarray(ctx_pulay)[0], dtype=np.float64),
        "HF strain (VMC free kernel)": np.asarray(free_hf, dtype=np.float64),
        "HF strain (VMC driver ctx)": np.asarray(np.asarray(ctx_hf)[0], dtype=np.float64),
    }


@pytest.mark.heavy
@pytest.mark.parametrize("order", ["gto_first", "pw_first"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_real_j3_composite_stress_matches_strain_ad_in_both_child_orders(order):
    """Native real VMC Pulay and HF stress equal strain AD of ln|Psi| and e_L for a (pGTO, PW) and a (PW, pGTO) J3 orbital."""
    if is_mixed_precision_enabled():
        entry_liveness = get_tolerance_min(_ZONES, "medium")

        def analytical(fresh):
            r_up, r_dn = _configuration(fresh)
            values = _native_stress_outputs(fresh, r_up, r_dn)
            expected_fields = {
                "Pulay strain (VMC free kernel)",
                "HF strain (VMC driver ctx)",
                "HF strain (VMC free kernel)",
                "Pulay strain (VMC driver ctx)",
            }
            assert set(values) == expected_fields, values.keys()
            live_atol, live_rtol = entry_liveness
            for name, value in values.items():
                expected_shape = (
                    (3, 3)
                    if "stress" in name or "strain" in name
                    else r_up.shape
                    if "dr_up" in name
                    else r_dn.shape
                    if "dr_dn" in name
                    else np.asarray(fresh.structure_data.positions).shape
                )
                assert value.shape == expected_shape, (name, value.shape, expected_shape)
                assert np.all(np.isfinite(value)), (name, value)
                assert np.max(np.abs(value)) > live_atol + live_rtol * np.max(np.abs(value)), (name, value)
            return values

        factory = full_source_hamiltonian_factory(lambda: _full_hamiltonian_source(order))
        mixed, full = mixed_vs_full(
            factory,
            analytical,
            configure_native=jqmc_rs.configure_native_precision,
            native_policy_snapshot=jqmc_rs.native_precision_support,
        )
        blocks = mixed_fidelity_blocks(mixed, full)
        atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
        for mixed_block, full_block in blocks:
            assert_parameter_derivative_mixed_fidelity(mixed_block, full_block, atol, rtol)
        mutated = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            assert_parameter_derivative_mixed_fidelity(mutated, blocks[0][1], atol, rtol)
        return

    ham, r_up, r_dn, geminal, jastrow, coulomb, geminal_inv = _setup(order)
    pulay_reference = _strain_ad(ham, r_up, r_dn, _ln_psi)
    hf_reference = _strain_ad(ham, r_up, r_dn, _local_energy)

    values = _native_stress_outputs(ham, r_up, r_dn)

    checks = [
        ("Pulay strain (VMC free kernel)", values["Pulay strain (VMC free kernel)"], pulay_reference),
        ("Pulay strain (VMC driver ctx)", values["Pulay strain (VMC driver ctx)"], pulay_reference),
        ("HF strain (VMC free kernel)", values["HF strain (VMC free kernel)"], hf_reference),
        ("HF strain (VMC driver ctx)", values["HF strain (VMC driver ctx)"], hf_reference),
    ]
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    live_atol, live_rtol = get_tolerance_min(_ZONES, "medium")
    for name, value, reference in checks:
        value = np.asarray(value, dtype=np.float64)
        assert value.shape == reference.shape, (name, value.shape, reference.shape)
        assert np.all(np.isfinite(reference)), (name, reference)
        assert np.all(np.isfinite(value)), (name, value)
        # Non-vacuous: the reference is distinguishable from zero at the comparison tolerance.
        assert np.max(np.abs(reference)) > live_atol + live_rtol * np.max(np.abs(reference)), (name, reference)
        np.testing.assert_allclose(
            value,
            reference,
            atol=atol,
            rtol=rtol,
            equal_nan=False,
            err_msg=f"Rust real J3 composite stress != AD reference: {name} (J3 {order})",
        )

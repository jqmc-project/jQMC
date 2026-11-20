"""Rust complex-twist J3 forces and stress for a square, non-identity MO-wrapped J3 orbital.

The native complex J3 force and stress kernels (``j3_side_complex``,
``j3_egrad_side_complex``, ``j3_pbc_kstress_complex``, ``j3_value_strain_complex`` and
``j3_value_force_complex``) chose between an orbital-rank path (``C @ AO``) and a plain
AO-rank path by the SHAPE of the MO matrix (``c.nrows() != c.ncols()``). A square but
non-identity MO therefore took the AO-rank path, which drops the MO coefficients: the
local energy and the drift stayed correct, but every J3 force and stress quantity was
wrong by O(1) with no error (issue #119).

This module builds the committed ``pgto_pw_twist_j123`` cell (QE-projected Li ccECP at
k = 0.25, complex pGTO(+)PW single determinant, exp J1/J2 and an identity-MO-wrapped
pGTO(+)PW composite J3) with a seeded nonzero ``j_matrix``, and its reduction to a bare
complex pGTO J3 leaf (the PW child dropped). The J3 MO is then replaced by a seeded
orthogonal ``Q`` with ``j_matrix -> Q M Q^T`` and the J1-like column ``-> Q j``, so the
J3 orbital matrix is square and non-identity while the wave function is unchanged.

At one seeded electron configuration the native Hellmann-Feynman derivatives of
``Re e_L`` (nuclei, electrons, strain) and the Pulay derivatives of ``Re ln Psi``
(nuclei, strain) from the free VMC kernels, the MCMC driver ctx and the GFMC T-move
kernels must equal reverse-mode AD of the XLA ``Re compute_local_energy`` /
``Re evaluate_ln_wavefunction`` in full precision. The nuclear derivative
displaces every ``Structure_data.positions`` leaf; the strain derivative uses
``Hamiltonian_data._apply_strain(F)`` with electrons ``r F^T``. The reference uses no
Rust code and no analytic force formula. In full precision it agrees with a fourth-order
central finite difference of the same XLA energies to 4.5e-11, and with the AD reference
of the identity-MO wave function to 5e-14 (recorded in issue #119).

A public-driver check runs the shared GFMC_t force/stress driver oracle on the rotated
pGTO(+)PW cell: the native stored forces and stress must agree with the XLA driver.

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
from jqmc.atomic_orbital import AOs_plane_wave_complex_data
from jqmc.hamiltonians import compute_local_energy
from jqmc.jqmc_mcmc import _NV_BASE_MESH, Nv_default, _geminal_inv_batched
from jqmc.structure import Structure_data
from jqmc.wavefunction import evaluate_ln_wavefunction
from tests.test_rs_kernel import _rs_fixtures as fixtures
from tests.test_rs_kernel._rs_fixtures import _build_ham_for_cell
from tests.test_rs_kernel.test_gfmc_chain_modes import _assert_gfmc_force_stress_driver_pair

jax.config.update("jax_enable_x64", True)

# The test reads only committed fixtures through absolute paths.
pytestmark = pytest.mark.with_chdir

#: The ``pgto_pw_twist_j123`` recipe of test_richer_production_drivers.py.
_PATTERN = "pgto_pw_twist_j123"
_CELL = ("QEPROJ:pgto+pw:Li_k025:ECP", True, {"j3": "mo"}, "SD")
_J_SEED = 7
_J_SCALE = 0.05
_Q_SEED = 11
_SEED = 1
_SCALE = 0.6
#: GFMC lattice spacing; the T-move measurement force does not depend on it.
_ALAT = 0.3
_NN = 1
_WEIGHTS = np.asarray(_NV_BASE_MESH[Nv_default].weights, dtype=np.float64)
_GRID = np.asarray(_NV_BASE_MESH[Nv_default].grid_points, dtype=np.float64)
#: Zones crossed by the complex J123 local energy, ln Psi and their derivatives.
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


def _hamiltonian(basis):
    """The pgto_pw_twist_j123 cell (``basis="pgto_pw"``) or its bare-pGTO J3 reduction
    (``basis="pgto"``) with a seeded j_matrix and a square, non-identity orthogonal J3 MO
    ``Q``; ``j_matrix`` is rotated so the wave function equals the identity-MO one.
    """
    ham = _build_ham_for_cell(*_CELL)
    wf = ham.wavefunction_data
    jastrow = wf.jastrow_data
    j3 = jastrow.jastrow_three_body_data
    orb = j3.orb_data
    gto, pw = orb.aos_data.children
    assert isinstance(pw, AOs_plane_wave_complex_data), orb.aos_data.children
    num_gto, num_pw = int(gto.num_ao), int(pw.num_ao)
    # The committed fixture wraps the composite in an identity MO.
    np.testing.assert_array_equal(np.asarray(orb.mo_coefficients), np.eye(num_gto + num_pw))
    # The J3 orbital is over complex AO children, so its MO coefficients must be complex128
    # (``MOs_data.sanity_check``); the committed identity wrapper stores them as float64, so the
    # dtype is set explicitly here (the values are unchanged).
    mo_dtype = np.complex128
    if basis == "pgto":
        # ``num_mo`` follows the reduced basis as well (the composite wrapper has num_gto + num_pw).
        orb = orb.replace(aos_data=gto, num_mo=num_gto, mo_coefficients=jnp.eye(num_gto, dtype=mo_dtype))
    n = int(np.asarray(orb.mo_coefficients).shape[0])
    j_matrix = _J_SCALE * np.random.default_rng(_J_SEED).standard_normal((n, n + 1))
    j_matrix[:, :n] = 0.5 * (j_matrix[:, :n] + j_matrix[:, :n].T)

    q, _ = np.linalg.qr(np.random.default_rng(_Q_SEED).standard_normal((n, n)))
    # Non-vacuous: Q is square and far from the identity (and from any signed permutation).
    assert q.shape == (n, n)
    assert np.max(np.abs(q - np.eye(n))) > 0.5
    assert np.max(np.abs(np.abs(q) - np.eye(n))) > 0.5
    rotated = np.zeros_like(j_matrix)
    rotated[:, :n] = q @ j_matrix[:, :n] @ q.T
    rotated[:, n] = q @ j_matrix[:, n]
    orb = orb.replace(mo_coefficients=jnp.asarray(q.astype(mo_dtype)))
    assert np.asarray(orb.mo_coefficients).dtype == np.complex128
    orb.sanity_check()
    j3 = j3.replace(orb_data=orb, j_matrix=jnp.asarray(rotated))
    j3.sanity_check()
    wf = wf.replace(jastrow_data=jastrow.replace(jastrow_three_body_data=j3))
    ham = ham.replace(wavefunction_data=wf)
    ham.sanity_check()
    return ham


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


def _re_e_l(ham, r_up, r_dn):
    return jnp.real(compute_local_energy(ham, r_up, r_dn, jnp.eye(3)))


def _re_ln_psi(ham, r_up, r_dn):
    return jnp.real(evaluate_ln_wavefunction(ham.wavefunction_data, r_up, r_dn))


def _nuclear_ad(scalar, ham, r_up, r_dn):
    zero = jnp.zeros_like(jnp.asarray(ham.structure_data.positions))
    up, dn = jnp.asarray(r_up), jnp.asarray(r_dn)
    return np.asarray(jax.grad(lambda d: scalar(_displaced(ham, d), up, dn))(zero), dtype=np.float64)


def _strain_ad(scalar, ham, r_up, r_dn):
    def at_strain(eps):
        deformation = jnp.eye(3) + eps
        return scalar(ham._apply_strain(deformation), jnp.asarray(r_up) @ deformation.T, jnp.asarray(r_dn) @ deformation.T)

    return np.asarray(jax.jacrev(at_strain)(jnp.zeros((3, 3))), dtype=np.float64)


def _geminal_inv(geminal, r_up, r_dn):
    return np.asarray(_geminal_inv_batched(geminal, jnp.asarray(r_up)[None], jnp.asarray(r_dn)[None])[1], dtype=np.complex128)


def assert_parameter_derivative_mixed_fidelity(mixed, full, atol, rtol):
    """Apply the central block budget at a source-qualified local receipt site."""
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()


def _native_force_outputs(ham, r_up, r_dn):
    """Original native analytical blocks, with unchanged kernel arguments."""
    wf = ham.wavefunction_data
    geminal, jastrow, coulomb = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    free_hf = jqmc_rs.vmc_grad_e_L_pbc_complex(geminal, jastrow, coulomb, r_up, r_dn, _WEIGHTS, _GRID, None)
    free_pulay = jqmc_rs.vmc_grad_ln_psi_pbc_complex(geminal, jastrow, r_up, r_dn)
    geminal_inv = _geminal_inv(geminal, r_up, r_dn)
    ctx = jqmc_rs.build_obs_ctx(geminal, jastrow, coulomb, True, True, True, precision_policy=precision_policy_snapshot())
    ctx_hf = ctx.grad_e_L_pbc_complex_batch(r_up[None], r_dn[None], _WEIGHTS, _GRID[None], geminal_inv)
    ctx_pulay = ctx.grad_ln_psi_pbc_complex_batch(r_up[None], r_dn[None], geminal_inv)
    gfmc_hf = jqmc_rs._gfmc_grad_e_L_ecp_pbc_complex(
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
@pytest.mark.parametrize("basis", ["pgto_pw", "pgto"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_complex_j3_square_nonidentity_mo_forces_match_ad(basis):
    """Native complex HF and Pulay forces equal AD of Re e_L / Re ln Psi for a square non-identity J3 MO."""
    if is_mixed_precision_enabled():
        entry_liveness = get_tolerance_min(_ZONES, "medium")

        def analytical(fresh):
            r_up, r_dn = _configuration(fresh)
            values = _native_force_outputs(fresh, r_up, r_dn)
            expected_fields = {
                "HF dR (VMC free kernel)",
                "HF dr_up (GFMC T-move)",
                "HF dr_dn (VMC free kernel)",
                "HF dR (GFMC T-move)",
                "Pulay dR (VMC free kernel)",
                "HF dr_up (VMC free kernel)",
                "Pulay dR (VMC driver ctx)",
                "HF dr_dn (GFMC T-move)",
                "HF dR (VMC driver ctx)",
                "HF dr_dn (VMC driver ctx)",
                "HF dr_up (VMC driver ctx)",
            }
            assert set(values) == expected_fields, values.keys()
            live_atol, live_rtol = entry_liveness
            for name, value in values.items():
                expected_shape = (
                    (3, 3)
                    if "stress" in name
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

        factory = full_source_hamiltonian_factory(lambda: _hamiltonian(basis))
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

    ham = _hamiltonian(basis)
    r_up, r_dn = _configuration(ham)

    hf_nuclei = _nuclear_ad(_re_e_l, ham, r_up, r_dn)
    hf_up, hf_dn = (
        np.asarray(x, dtype=np.float64)
        for x in jax.grad(lambda a, b: _re_e_l(ham, a, b), argnums=(0, 1))(jnp.asarray(r_up), jnp.asarray(r_dn))
    )
    pulay_nuclei = _nuclear_ad(_re_ln_psi, ham, r_up, r_dn)

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
            err_msg=f"Rust complex square non-identity MO J3 force != AD reference: {name} (J3 {basis})",
        )


def _native_stress_outputs(ham, r_up, r_dn):
    """Original native analytical blocks, with unchanged kernel arguments."""
    wf = ham.wavefunction_data
    geminal, jastrow, coulomb = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    free_hf = jqmc_rs.vmc_grad_e_L_eps_pbc_complex(geminal, jastrow, coulomb, r_up, r_dn, _WEIGHTS, _GRID)
    free_pulay = jqmc_rs.vmc_grad_ln_psi_eps_pbc_complex(geminal, jastrow, r_up, r_dn)
    geminal_inv = _geminal_inv(geminal, r_up, r_dn)
    ctx = jqmc_rs.build_obs_ctx(geminal, jastrow, coulomb, True, True, True, precision_policy=precision_policy_snapshot())
    ctx_hf = ctx.grad_e_L_eps_pbc_complex_batch(r_up[None], r_dn[None], _WEIGHTS, _GRID[None], geminal_inv)
    ctx_pulay = ctx.grad_ln_psi_eps_pbc_complex_batch(r_up[None], r_dn[None], geminal_inv)
    gfmc_hf = jqmc_rs._gfmc_stress_ecp_pbc_complex(
        geminal, jastrow, coulomb, r_up, r_dn, _ALAT, np.eye(3), _WEIGHTS, _GRID, _NN, False
    )

    return {
        "HF stress (VMC free kernel)": np.asarray(free_hf, dtype=np.float64),
        "HF stress (VMC driver ctx)": np.asarray(np.asarray(ctx_hf)[0], dtype=np.float64),
        "HF stress (GFMC T-move)": np.asarray(gfmc_hf, dtype=np.float64),
        "Pulay stress (VMC free kernel)": np.asarray(free_pulay, dtype=np.float64),
        "Pulay stress (VMC driver ctx)": np.asarray(np.asarray(ctx_pulay)[0], dtype=np.float64),
    }


@pytest.mark.heavy
@pytest.mark.parametrize("basis", ["pgto_pw", "pgto"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_complex_j3_square_nonidentity_mo_stress_matches_strain_ad(basis):
    """Native complex HF and Pulay stress equal strain AD of Re e_L / Re ln Psi for a square non-identity J3 MO."""
    if is_mixed_precision_enabled():
        entry_liveness = get_tolerance_min(_ZONES, "medium")

        def analytical(fresh):
            r_up, r_dn = _configuration(fresh)
            values = _native_stress_outputs(fresh, r_up, r_dn)
            expected_fields = {
                "HF stress (VMC driver ctx)",
                "HF stress (VMC free kernel)",
                "Pulay stress (VMC driver ctx)",
                "HF stress (GFMC T-move)",
                "Pulay stress (VMC free kernel)",
            }
            assert set(values) == expected_fields, values.keys()
            live_atol, live_rtol = entry_liveness
            for name, value in values.items():
                expected_shape = (
                    (3, 3)
                    if "stress" in name
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

        factory = full_source_hamiltonian_factory(lambda: _hamiltonian(basis))
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

    ham = _hamiltonian(basis)
    r_up, r_dn = _configuration(ham)

    hf_strain = _strain_ad(_re_e_l, ham, r_up, r_dn)
    pulay_strain = _strain_ad(_re_ln_psi, ham, r_up, r_dn)

    values = _native_stress_outputs(ham, r_up, r_dn)

    checks = [
        ("HF stress (VMC free kernel)", values["HF stress (VMC free kernel)"], hf_strain),
        ("HF stress (VMC driver ctx)", values["HF stress (VMC driver ctx)"], hf_strain),
        ("HF stress (GFMC T-move)", values["HF stress (GFMC T-move)"], hf_strain),
        ("Pulay stress (VMC free kernel)", values["Pulay stress (VMC free kernel)"], pulay_strain),
        ("Pulay stress (VMC driver ctx)", values["Pulay stress (VMC driver ctx)"], pulay_strain),
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
            err_msg=f"Rust complex square non-identity MO J3 stress != AD reference: {name} (J3 {basis})",
        )


# One-walker forty-step GFMC_t chains through both production backends (several minutes).
@pytest.mark.heavy
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated", seed=34356)
def test_complex_j3_square_nonidentity_mo_gfmc_t_force_stress_driver_matches_xla(monkeypatch):
    """Native GFMC_t stored forces and stress agree with XLA for a square non-identity J3 MO."""
    monkeypatch.setitem(fixtures._PATTERNS, _PATTERN, _CELL)
    _assert_gfmc_force_stress_driver_pair("t", _PATTERN, _hamiltonian("pgto_pw"), monkeypatch)

"""Rust complex-twist HF stress of a PW-first pGTO(+)PW composite J3 orbital against strain AD.

The native complex J3 kinetic stress (``complex_stress.rs``, ``j3_pbc_kstress_complex``)
places the rows of a pGTO(+)PW composite J3 orbital at offsets that depend on the child order:

* the pGTO rows of the coefA / bar_A cotangent fed to the sphe AO field stress start at
  ``let gto0 = if j3.gto_first { 0 } else { num_pw }``;
* the PW rows of the lap / gou cotangent fed to ``pw_lap_stress_complex`` /
  ``pw_gou_stress_complex`` start at ``pw0 = if dense_mo.is_some() || j3.gto_first { num_gto } else { 0 }``.

Every committed complex J3 composite is pGTO-first, so the PW-first offsets were never
reached: replacing them by ``0`` and ``num_gto`` survived every test in full and mixed
precision (issue #101).

This module builds the committed ``pgto_pw_twist_j123`` cell (QE-projected Li ccECP at
k = 0.25, complex pGTO(+)PW single determinant kept pGTO-first, exp J1/J2 and an
identity-MO-wrapped pGTO(+)PW composite J3) with a seeded nonzero ``j_matrix`` in two J3
child orders:

* ``gto_first``: the J3 orbital as built (pGTO, PW);
* ``pw_first``: the J3 children reordered to (PW, pGTO) and the ``j_matrix`` rows/columns
  (and the J1-like last column) permuted to match, so the wave function, and therefore
  every observable, is unchanged.

For each order, at one seeded electron configuration, the Hellmann-Feynman stress
``d Re e_L / d eps`` of the native VMC kernel ``jqmc_rs.vmc_grad_e_L_eps_pbc_complex``, of
the MCMC driver entry ``NativeObsCtx.grad_e_L_eps_pbc_complex_batch`` and of the GFMC/LRDMC
stress ``jqmc_rs._gfmc_stress_ecp_pbc_complex`` (T-move, whose measured quantity is the same
continuum local energy) must equal reverse-mode AD of the XLA ``Re compute_local_energy``
(identity quadrature rotation, the same base mesh) in full precision, with the
cell, nuclei, every basis leaf and the electrons strained together
(``Hamiltonian_data._apply_strain`` and ``r @ F.T``). The reference differentiates the energy
itself and uses no Rust code and no analytic stress formula. In full precision it agrees with
a fourth-order central finite difference of the same energy to 6.2e-11 (pGTO-first) and
4.5e-11 (PW-first) (recorded in issue #101). Mixed precision uses the numerical
fidelity scope described below.

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
from jqmc.atomic_orbital import AOs_composite_data, AOs_plane_wave_complex_data
from jqmc.hamiltonians import compute_local_energy
from jqmc.jqmc_mcmc import _NV_BASE_MESH, Nv_default, _geminal_inv_batched
from tests.test_rs_kernel._rs_fixtures import _build_ham_for_cell

jax.config.update("jax_enable_x64", True)

# The test reads only committed fixtures through absolute paths.
pytestmark = pytest.mark.with_chdir

#: The ``pgto_pw_twist_j123`` recipe of test_richer_production_drivers.py.
_CELL = ("QEPROJ:pgto+pw:Li_k025:ECP", True, {"j3": "mo"}, "SD")
_J_SEED = 7
_J_SCALE = 0.05
_SEED = 1
_SCALE = 0.6
#: GFMC lattice spacing; the T-move measurement energy does not depend on it.
_ALAT = 0.3
_NN = 1
_WEIGHTS = np.asarray(_NV_BASE_MESH[Nv_default].weights, dtype=np.float64)
_GRID = np.asarray(_NV_BASE_MESH[Nv_default].grid_points, dtype=np.float64)
#: Zones crossed by the complex J123 local energy and its strain derivative.
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
    """The pgto_pw_twist_j123 cell with a seeded j_matrix, J3 children in ``order``."""
    ham = _build_ham_for_cell(*_CELL)
    wf = ham.wavefunction_data
    jastrow = wf.jastrow_data
    j3 = jastrow.jastrow_three_body_data
    shape = np.asarray(j3.j_matrix).shape
    n = shape[0]
    j_matrix = _J_SCALE * np.random.default_rng(_J_SEED).standard_normal(shape)
    j_matrix[:, :n] = 0.5 * (j_matrix[:, :n] + j_matrix[:, :n].T)

    orb = j3.orb_data
    gto, pw = orb.aos_data.children
    assert isinstance(pw, AOs_plane_wave_complex_data), orb.aos_data.children
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
        assert isinstance(orb.aos_data.children[0], AOs_plane_wave_complex_data)
    j3 = j3.replace(orb_data=orb, j_matrix=jnp.asarray(j_matrix))
    wf = wf.replace(jastrow_data=jastrow.replace(jastrow_three_body_data=j3))
    return ham.replace(wavefunction_data=wf)


def _configuration(ham):
    geminal = ham.wavefunction_data.geminal_data
    num_up, num_dn = int(geminal.num_electron_up), int(geminal.num_electron_dn)
    positions = np.asarray(ham.structure_data.positions, dtype=np.float64)
    rng = np.random.default_rng(_SEED)
    r_up = positions[np.arange(num_up) % len(positions)] + rng.normal(scale=_SCALE, size=(num_up, 3))
    r_dn = positions[np.arange(num_dn) % len(positions)] + rng.normal(scale=_SCALE, size=(num_dn, 3))
    return np.ascontiguousarray(r_up, dtype=np.float64), np.ascontiguousarray(r_dn, dtype=np.float64)


def _strain_ad_local_energy(ham, r_up, r_dn):
    """Reverse-mode AD of Re e_L(strained ham, r_up F^T, r_dn F^T) with respect to ``eps`` at 0."""

    def at_strain(eps):
        deformation = jnp.eye(3) + eps
        strained = ham._apply_strain(deformation)
        up, dn = jnp.asarray(r_up) @ deformation.T, jnp.asarray(r_dn) @ deformation.T
        return jnp.real(compute_local_energy(strained, up, dn, jnp.eye(3)))

    return np.asarray(jax.jacrev(at_strain)(jnp.zeros((3, 3))), dtype=np.float64)


# About 40 s per row on the CPU reference host (strain AD of the J123 local energy), above the light budget.
def assert_parameter_derivative_mixed_fidelity(mixed, full, atol, rtol):
    """Apply the central block budget at a source-qualified local receipt site."""
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()


def _native_stress_outputs(ham, r_up, r_dn):
    """Original native analytical blocks, with unchanged kernel arguments."""
    wf = ham.wavefunction_data
    geminal, jastrow, coulomb = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    free_hf = jqmc_rs.vmc_grad_e_L_eps_pbc_complex(geminal, jastrow, coulomb, r_up, r_dn, _WEIGHTS, _GRID)
    geminal_inv = np.asarray(
        _geminal_inv_batched(geminal, jnp.asarray(r_up)[None], jnp.asarray(r_dn)[None])[1], dtype=np.complex128
    )
    ctx = jqmc_rs.build_obs_ctx(geminal, jastrow, coulomb, True, True, True, precision_policy=precision_policy_snapshot())
    ctx_hf = ctx.grad_e_L_eps_pbc_complex_batch(r_up[None], r_dn[None], _WEIGHTS, _GRID[None], geminal_inv)
    gfmc_hf = jqmc_rs._gfmc_stress_ecp_pbc_complex(
        geminal, jastrow, coulomb, r_up, r_dn, _ALAT, np.eye(3), _WEIGHTS, _GRID, _NN, False
    )

    return {
        "HF stress (VMC free kernel)": np.asarray(free_hf, dtype=np.float64),
        "HF stress (VMC driver ctx)": np.asarray(np.asarray(ctx_hf)[0], dtype=np.float64),
        "HF stress (GFMC T-move)": np.asarray(gfmc_hf, dtype=np.float64),
    }


@pytest.mark.heavy
@pytest.mark.parametrize("order", ["gto_first", "pw_first"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_complex_j3_composite_hf_stress_matches_strain_ad_in_both_child_orders(order):
    """Native complex VMC / GFMC HF stress equals strain AD of Re e_L for a (pGTO, PW) and a (PW, pGTO) J3 orbital."""
    if is_mixed_precision_enabled():
        entry_liveness = get_tolerance_min(_ZONES, "medium")

        def analytical(fresh):
            r_up, r_dn = _configuration(fresh)
            values = _native_stress_outputs(fresh, r_up, r_dn)
            expected_fields = {"HF stress (GFMC T-move)", "HF stress (VMC driver ctx)", "HF stress (VMC free kernel)"}
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

        factory = full_source_hamiltonian_factory(lambda: _hamiltonian(order))
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

    ham = _hamiltonian(order)
    r_up, r_dn = _configuration(ham)
    reference = _strain_ad_local_energy(ham, r_up, r_dn)

    values = _native_stress_outputs(ham, r_up, r_dn)

    checks = [
        ("HF stress (VMC free kernel)", values["HF stress (VMC free kernel)"], reference),
        ("HF stress (VMC driver ctx)", values["HF stress (VMC driver ctx)"], reference),
        ("HF stress (GFMC T-move)", values["HF stress (GFMC T-move)"], reference),
    ]
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    live_atol, live_rtol = get_tolerance_min(_ZONES, "medium")
    assert reference.shape == (3, 3), reference.shape
    assert np.all(np.isfinite(reference)), reference
    # Non-vacuous: the reference is distinguishable from zero at the comparison tolerance.
    assert np.max(np.abs(reference)) > live_atol + live_rtol * np.max(np.abs(reference)), reference
    for name, value, reference in checks:
        value = np.asarray(value, dtype=np.float64)
        assert value.shape == reference.shape, (name, value.shape, reference.shape)
        assert np.all(np.isfinite(value)), (name, value)
        np.testing.assert_allclose(
            value,
            reference,
            atol=atol,
            rtol=rtol,
            equal_nan=False,
            err_msg=f"Rust complex J3 composite HF stress != strain AD of Re e_L: {name} (J3 {order})",
        )

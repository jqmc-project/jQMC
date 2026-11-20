"""Rust complex-twist VMC forces of an extra-first pGTO(+)PW / pGTO(+)B-spline composite against AD.

The native complex determinant force kernels slice the GTO rows out of the AO-rank
cotangent of a complex composite basis at an offset that depends on the child order
(``complex_det.rs``: ``gto_slice_cm`` used by ``pbc_det_pulay_force_complex``,
``let g0 = if orb.gto_first { 0 } else { num_extra }``, and ``complex_det_kinetic_grads``,
``let gto0 = if orb.gto_first { 0 } else { num_extra }``). Every committed complex composite
fixture orders the pGTO child first, so the extra-first offsets were never reached:
replacing either offset by ``0`` survived every test in full and mixed precision (issue #75).

This module builds the committed ``pgto_pw_twist_jnone`` and ``pgto_bspline_twist_jnone``
cells (QE-projected Li ccECP at k = 0.25, complex composite, single determinant, no Jastrow)
in two child orders:

* ``gto_first``: the fixture as built (pGTO, extra);
* ``extra_first``: children reordered to (extra, pGTO) and the MO coefficient columns permuted
  to match, so the wave function, and therefore every observable, is unchanged.

For each order, at one seeded electron configuration, the native VMC Pulay force
``d Re ln Psi / dR`` (``jqmc_rs.vmc_grad_ln_psi_pbc_complex`` and the MCMC driver entry
``NativeObsCtx.grad_ln_psi_pbc_complex_batch``) and Hellmann-Feynman derivatives
``d Re e_L / dR`` and ``d Re e_L / dr`` (``jqmc_rs.vmc_grad_e_L_pbc_complex`` and
``NativeObsCtx.grad_e_L_pbc_complex_batch``) must equal reverse-mode AD of the XLA
``Re evaluate_ln_wavefunction`` and ``Re compute_local_energy`` (identity quadrature
rotation, the same base mesh) in the active precision policy. The nuclear derivative
displaces every ``Structure_data.positions`` leaf of the Hamiltonian pytree together. The
reference differentiates the quantities themselves and uses no Rust code and no analytic
force formula. In full precision the same reference agrees with a fourth-order central
finite difference of the same quantities to 1.2e-10 (recorded in issue #75); it is not
repeated here because mixed precision has no finite-difference verdict.

Precision scope (#163): the independent AD statements below apply in full at the strict budget.
Mixed evaluates only the native analytical quantities against the identical native route rebuilt
in full. Both policies load fresh Hamiltonians from the same full-precision serialized physical
input and use identical deterministic coordinates. Each homogeneous parameter block meets the
central fidelity budget and rejects an out-of-budget mixed perturbation. No native precision
zone is promoted and no mixed case is skipped.
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
from jqmc.atomic_orbital import (
    AOs_bspline_complex_data,
    AOs_composite_data,
    AOs_plane_wave_complex_data,
)
from jqmc.hamiltonians import compute_local_energy
from jqmc.jqmc_mcmc import _NV_BASE_MESH, Nv_default, _geminal_inv_batched
from jqmc.structure import Structure_data
from jqmc.wavefunction import evaluate_ln_wavefunction
from tests.test_rs_kernel._rs_fixtures import rs_ham


def assert_parameter_derivative_mixed_fidelity(mixed, full, atol, rtol):
    """Bind the central block norm directly to this module source."""
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()


jax.config.update("jax_enable_x64", True)

# The test reads only committed fixtures through absolute paths.
pytestmark = pytest.mark.with_chdir

_EXTRA_TYPES = {
    "pgto_pw_twist_jnone": AOs_plane_wave_complex_data,
    "pgto_bspline_twist_jnone": AOs_bspline_complex_data,
}
_SEED = 1
_SCALE = 0.6
_WEIGHTS = np.asarray(_NV_BASE_MESH[Nv_default].weights, dtype=np.float64)
_GRID = np.asarray(_NV_BASE_MESH[Nv_default].grid_points, dtype=np.float64)
#: Zones crossed by Re ln Psi, the complex no-Jastrow local energy and their derivatives.
_ZONES = (
    "ao_eval",
    "ao_grad_lap",
    "mo_eval",
    "mo_grad",
    "mo_lap",
    "det_eval",
    "det_grad_lap",
    "wf_kinetic",
    "ewald",
    "coulomb",
)


def _extra_first_orb(orb):
    """Reorder a (pGTO, extra) composite MO leaf to (extra, pGTO), permuting MO columns to match."""
    gto, extra = orb.aos_data.children
    num_gto = int(gto.num_ao)
    coefficients = jnp.asarray(orb.mo_coefficients)
    permuted = jnp.concatenate([coefficients[:, num_gto:], coefficients[:, :num_gto]], axis=1)
    return orb.replace(aos_data=AOs_composite_data.build((extra, gto)), mo_coefficients=permuted)


def _hamiltonian(pattern, order):
    ham = rs_ham(pattern)
    extra_type = _EXTRA_TYPES[pattern]
    geminal = ham.wavefunction_data.geminal_data
    for orb in (geminal.orb_data_up_spin, geminal.orb_data_dn_spin):
        children = orb.aos_data.children
        assert len(children) == 2 and isinstance(children[1], extra_type), children
        assert np.iscomplexobj(np.asarray(orb.mo_coefficients))
    if order == "extra_first":
        geminal = geminal.replace(
            orb_data_up_spin=_extra_first_orb(geminal.orb_data_up_spin),
            orb_data_dn_spin=_extra_first_orb(geminal.orb_data_dn_spin),
        )
        ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(geminal_data=geminal))
        for orb in (geminal.orb_data_up_spin, geminal.orb_data_dn_spin):
            assert isinstance(orb.aos_data.children[0], extra_type)
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


def _ad_references(ham, r_up, r_dn):
    """Reverse-mode AD of Re ln Psi (nuclei) and Re e_L (nuclei, up and down electrons)."""
    up, dn = jnp.asarray(r_up), jnp.asarray(r_dn)
    zero = jnp.zeros_like(jnp.asarray(ham.structure_data.positions))

    def ln_psi_nuclei(d_positions):
        return jnp.real(evaluate_ln_wavefunction(_displaced(ham, d_positions).wavefunction_data, up, dn))

    def e_l_nuclei(d_positions):
        return jnp.real(compute_local_energy(_displaced(ham, d_positions), up, dn, jnp.eye(3)))

    def e_l_electrons(a, b):
        return jnp.real(compute_local_energy(ham, a, b, jnp.eye(3)))

    pulay_nuclei = np.asarray(jax.grad(ln_psi_nuclei)(zero), dtype=np.float64)
    hf_nuclei = np.asarray(jax.grad(e_l_nuclei)(zero), dtype=np.float64)
    hf_up, hf_dn = (np.asarray(x, dtype=np.float64) for x in jax.grad(e_l_electrons, argnums=(0, 1))(up, dn))
    return pulay_nuclei, hf_nuclei, hf_up, hf_dn


def _native_derivative_outputs(ham, r_up, r_dn):
    """The original native analytical quantities, with identical kernel arguments."""
    wf = ham.wavefunction_data
    geminal, jastrow, coulomb = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    free_pulay = jqmc_rs.vmc_grad_ln_psi_pbc_complex(geminal, jastrow, r_up, r_dn)
    free_hf = jqmc_rs.vmc_grad_e_L_pbc_complex(geminal, jastrow, coulomb, r_up, r_dn, _WEIGHTS, _GRID, None)
    geminal_inv = np.asarray(
        _geminal_inv_batched(geminal, jnp.asarray(r_up)[None], jnp.asarray(r_dn)[None])[1], dtype=np.complex128
    )
    ctx = jqmc_rs.build_obs_ctx(geminal, jastrow, coulomb, True, True, True, precision_policy=precision_policy_snapshot())
    ctx_pulay = ctx.grad_ln_psi_pbc_complex_batch(r_up[None], r_dn[None], geminal_inv)
    ctx_hf = ctx.grad_e_L_pbc_complex_batch(r_up[None], r_dn[None], _WEIGHTS, _GRID[None], geminal_inv)

    return {
        "Pulay dR (free kernel)": np.asarray(free_pulay[0]),
        "Pulay dR (driver ctx)": np.asarray(np.asarray(ctx_pulay[0])[0]),
        "HF dR (free kernel)": np.asarray(free_hf[0]),
        "HF dR (driver ctx)": np.asarray(np.asarray(ctx_hf[0])[0]),
        "HF dr_up (free kernel)": np.asarray(free_hf[1]),
        "HF dr_up (driver ctx)": np.asarray(np.asarray(ctx_hf[1])[0]),
        "HF dr_dn (free kernel)": np.asarray(free_hf[2]),
        "HF dr_dn (driver ctx)": np.asarray(np.asarray(ctx_hf[2])[0]),
    }


@pytest.mark.parametrize("order", ["gto_first", "extra_first"])
@pytest.mark.parametrize(
    "pattern",
    [
        pytest.param("pgto_pw_twist_jnone", marks=pytest.mark.light),
        # About 30 s per row on one CPU core (B-spline AD of the local energy).
        pytest.param("pgto_bspline_twist_jnone", marks=pytest.mark.heavy),
    ],
)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_vmc_complex_composite_forces_match_ad_in_both_child_orders(pattern, order):
    """Native complex VMC Pulay / HF forces equal AD of Re ln Psi / Re e_L for (pGTO, extra) and (extra, pGTO)."""
    if is_mixed_precision_enabled():
        entry_liveness = get_tolerance_min(_ZONES, "medium")

        def analytical(fresh):
            r_up, r_dn = _configuration(fresh)
            values = _native_derivative_outputs(fresh, r_up, r_dn)
            nuclear_shape = np.asarray(fresh.structure_data.positions).shape
            expected_shapes = {
                "Pulay dR (free kernel)": nuclear_shape,
                "Pulay dR (driver ctx)": nuclear_shape,
                "HF dR (free kernel)": nuclear_shape,
                "HF dR (driver ctx)": nuclear_shape,
                "HF dr_up (free kernel)": r_up.shape,
                "HF dr_up (driver ctx)": r_up.shape,
                "HF dr_dn (free kernel)": r_dn.shape,
                "HF dr_dn (driver ctx)": r_dn.shape,
            }
            assert set(values) == set(expected_shapes), values.keys()
            live_atol, live_rtol = entry_liveness
            for name, value in values.items():
                assert value.shape == expected_shapes[name], (name, value.shape, expected_shapes[name])
                assert np.all(np.isfinite(value)), (name, value)
                assert np.max(np.abs(value)) > live_atol + live_rtol * np.max(np.abs(value)), (name, value)
            return values

        factory = full_source_hamiltonian_factory(lambda: _hamiltonian(pattern, order))
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

    ham = _hamiltonian(pattern, order)
    r_up, r_dn = _configuration(ham)
    wf = ham.wavefunction_data
    geminal, jastrow, coulomb = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    pulay_nuclei, hf_nuclei, hf_up, hf_dn = _ad_references(ham, r_up, r_dn)

    values = _native_derivative_outputs(ham, r_up, r_dn)

    checks = [
        ("Pulay dR (free kernel)", values["Pulay dR (free kernel)"], pulay_nuclei),
        ("Pulay dR (driver ctx)", values["Pulay dR (driver ctx)"], pulay_nuclei),
        ("HF dR (free kernel)", values["HF dR (free kernel)"], hf_nuclei),
        ("HF dR (driver ctx)", values["HF dR (driver ctx)"], hf_nuclei),
        ("HF dr_up (free kernel)", values["HF dr_up (free kernel)"], hf_up),
        ("HF dr_up (driver ctx)", values["HF dr_up (driver ctx)"], hf_up),
        ("HF dr_dn (free kernel)", values["HF dr_dn (free kernel)"], hf_dn),
        ("HF dr_dn (driver ctx)", values["HF dr_dn (driver ctx)"], hf_dn),
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
            err_msg=f"Rust complex composite force != AD reference: {name} ({pattern}, {order})",
        )

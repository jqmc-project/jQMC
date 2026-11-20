"""Public MCMC / GFMC_t HF force and stress of a rectangular dense-MO composite J3 (issue #104, PR #106).

``test_real_gamma_converter_mo_j3_composite.py`` checks the native force / stress kernels of the QE
converter's dense-MO composite J3 (a ONE-row MO over a pGTO(+)extra composite) against AD. Its six
cases do not detect, in mixed precision, a native fault that zeroes only the Laplacian of the
B-spline child of the J3 basis (``forces_pbc.rs`` ``J3Extra::vgl``, values and gradients kept): the
one-row J3 with a 1x2 j_matrix leaves that Laplacian contribution inside the mixed envelope.

This test drives the public production drivers on a case where that contribution is not a residual:

* the J3 orbital is a valid TWO-row real rectangular MO ``C`` (shape (2, n_ao)) over the converter's
  Gamma pGTO(+)B-spline composite: the converted first row plus a seeded second row (standard
  deviation 0.04) whose B-spline coefficient is twice the first row's;
* the J3 copy of the B-spline child is scaled by 1000, as in the authored "amplified" cases (the
  determinant is unchanged);
* the j_matrix has nonzero symmetric off-diagonal and nonzero linear (last column) couplings.

Public ``MCMC`` (one walker, one step) and ``GFMC_t`` (one walker, six branching steps, tmove, fixed
identity ECP mesh, no SWCT) run on the native backend with ``comput_position_deriv`` and
``comput_stress``. With one walker, the last stored HF nuclear derivative ``force_HF`` (``d e_L/dR``)
and HF stress ``stress_HF`` (``d e_L/d eps``) are the values at the final sampled configuration.

Full uses independent reverse-mode AD of the XLA Re local energy at the final sampled configuration,
with the original coordinate and homogeneous-strain definitions and the strict full budget. Mixed
compares the actual stored quantities and freshly rebuilt native HF measurement kernels with the
same kernels rebuilt independently in full at that identical point. Both policies start from the
same full-precision converted physical input. The central parameter-derivative block infinity-norm
budget applies; this does not require matching full/mixed sampling trajectories.
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

import os
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jqmc_rs
from jqmc._precision import configure, get_tolerance_min, is_mixed_precision_enabled, precision_policy_snapshot
from jqmc._setting import get_tolerance_mixed_fidelity
from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block
from jqmc._setting import GFMC_MIN_COLLECT_STEPS
from jqmc.atomic_orbital import AOs_bspline_data, AOs_composite_data
from jqmc.hamiltonians import compute_local_energy
from jqmc.jqmc_gfmc import GFMC_t
from jqmc.jqmc_mcmc import MCMC
from tests.test_rs_kernel.conftest import assert_native_ran, native_ctx_spy
from tests.test_rs_kernel.test_real_gamma_converter_mo_j3_composite import (
    _converter_hamiltonian,
    _converted,
    _native_hf_measurement_outputs,
    _displaced,
    _without_j3,
)


def _assert_parameter_derivative_mixed_fidelity(mixed, full, atol, rtol):
    """Bind the central block norm directly to this module source."""
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()


jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.with_chdir

#: Zones crossed by the real J123 local energy and its nuclear / strain derivatives (the same zones
#: as the converter kernel test). Defined in this module, not imported, so that the GDD comparison
#: binding can resolve the ``get_tolerance_min`` arguments statically from this source.
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

#: Seed and spread of the second J3 MO row.
_ROW_SEED = 191
_ROW_SCALE = 0.04
#: Factor applied to the J3 copy of the B-spline child (as in the authored "amplified" cases).
_BSPLINE_AMPLIFICATION = 1000.0
#: Symmetric (2, 2) block plus linear column; every coupling is nonzero.
_J_MATRIX = np.array([[0.07, 0.035, 0.11], [0.035, -0.04, -0.08]])
_DRIVER_SEED = 113
_MCMC_STEPS = 1
_GFMC_STEPS = GFMC_MIN_COLLECT_STEPS + 1
_ALAT = 0.30
_TAU = 0.05


@functools.lru_cache(maxsize=1)
def _full_precision_source():
    """Freeze one FP64 conversion before either production policy builds its model."""
    original_mode = "mixed" if is_mixed_precision_enabled() else "full"
    original_native = dict(jqmc_rs.native_precision_support())
    try:
        configure("full")
        jqmc_rs.configure_native_precision(precision_policy_snapshot())
        # Bypass the converter helper's mode-agnostic cache: a different test
        # may have populated that cache while running in mixed precision.
        return _converted.__wrapped__("pgto+bspline")
    finally:
        try:
            configure(original_mode)
        finally:
            jqmc_rs.configure_native_precision(original_native)


def _rectangular_j3_hamiltonian(source_path=None):
    """Converter pGTO+B-spline Hamiltonian with a two-row rectangular J3 MO and a nonzero j_matrix."""
    ham = _converter_hamiltonian("pgto+bspline", source_path=source_path)
    wf = ham.wavefunction_data
    jastrow = wf.jastrow_data
    j3 = jastrow.jastrow_three_body_data
    orb = j3.orb_data
    gto, bspline = orb.aos_data.children
    assert isinstance(bspline, AOs_bspline_data), orb.aos_data.children
    assert int(bspline.num_ao) == 1, bspline.num_ao
    bspline = bspline.replace(spline_coefficients=np.asarray(bspline.spline_coefficients) * _BSPLINE_AMPLIFICATION)
    first = np.asarray(orb.mo_coefficients, dtype=np.float64)
    assert first.shape[0] == 1, first.shape
    second = _ROW_SCALE * np.random.default_rng(_ROW_SEED).standard_normal(first.shape[1])
    num_gto = int(gto.num_ao)
    second[num_gto] = 2.0 * first[0, num_gto]
    coefficients = np.ascontiguousarray(np.vstack([first, second[None]]))
    assert coefficients.shape == (2, num_gto + 1) and np.linalg.matrix_rank(coefficients) == 2, coefficients.shape
    assert np.allclose(_J_MATRIX[:, :2], _J_MATRIX[:, :2].T) and np.all(_J_MATRIX != 0.0)
    orb = orb.replace(
        aos_data=AOs_composite_data.build((gto, bspline)),
        num_mo=2,
        mo_coefficients=coefficients,
    )
    j3 = j3.replace(orb_data=orb, j_matrix=np.array(_J_MATRIX))
    return ham.replace(wavefunction_data=wf.replace(jastrow_data=jastrow.replace(jastrow_three_body_data=j3)))


def _hf_references(ham, r_up, r_dn):
    """Reverse-mode AD of Re e_L (identity quadrature rotation): (d/dR, d/deps) at (r_up, r_dn)."""
    up, dn = jnp.asarray(r_up), jnp.asarray(r_dn)

    def e_l(d_positions, eps):
        deformation = jnp.eye(3) + eps
        h = _displaced(ham, d_positions)._apply_strain(deformation)
        return jnp.real(compute_local_energy(h, up @ deformation.T, dn @ deformation.T, jnp.eye(3)))

    zero = jnp.zeros_like(jnp.asarray(ham.structure_data.positions))
    nuclei, strain = (np.asarray(x, dtype=np.float64) for x in jax.grad(e_l, argnums=(0, 1))(zero, jnp.zeros((3, 3))))
    jax.clear_caches()
    return nuclei, strain


def _run_public(driver, ham):
    """One public native production run; returns (stored HF force, stored HF stress, r_up, r_dn)."""
    env_flag = os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    try:
        with native_ctx_spy() as counts:
            if driver == "MCMC":
                run = MCMC(
                    hamiltonian_data=ham,
                    Dt=2.0,
                    mcmc_seed=_DRIVER_SEED,
                    epsilon_AS=1.0e-6,
                    num_walkers=1,
                    comput_position_deriv=True,
                    comput_stress=True,
                    random_discretized_mesh=False,
                    use_swct=False,
                )
                run.run(num_mcmc_steps=_MCMC_STEPS)
                force, stress = run.force_HF_stored, run.stress_HF_stored
                r_up, r_dn = run._MCMC__latest_r_up_carts, run._MCMC__latest_r_dn_carts
                steps = _MCMC_STEPS
            else:
                run = GFMC_t(
                    hamiltonian_data=ham,
                    num_walkers=1,
                    num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
                    mcmc_seed=_DRIVER_SEED,
                    tau=_TAU,
                    alat=_ALAT,
                    random_discretized_mesh=False,
                    non_local_move="tmove",
                    comput_position_deriv=True,
                    comput_stress=True,
                    use_swct=False,
                )
                run.run(num_mcmc_steps=_GFMC_STEPS)
                force, stress = np.asarray(run._stored_force_HF), np.asarray(run._stored_stress_HF)
                r_up, r_dn = run._latest_r_up_carts, run._latest_r_dn_carts
                steps = _GFMC_STEPS
        assert_native_ran(counts, f"public {driver} rectangular composite J3")
    finally:
        if env_flag is not None:
            os.environ["JQMC_FLAG_RUST_ON_CPU"] = env_flag
    force, stress = np.asarray(force, dtype=np.float64), np.asarray(stress, dtype=np.float64)
    n_atoms = int(ham.structure_data.positions.shape[0])
    assert force.shape == (steps, 1, n_atoms, 3), force.shape
    assert stress.shape == (steps, 1, 3, 3), stress.shape
    r_up, r_dn = np.asarray(r_up, dtype=np.float64), np.asarray(r_dn, dtype=np.float64)
    assert r_up.shape[0] == 1 and r_dn.shape[0] == 1, (r_up.shape, r_dn.shape)
    return force[-1, 0], stress[-1, 0], r_up[0], r_dn[0]


@pytest.mark.heavy
@pytest.mark.parametrize("driver", ["MCMC", "GFMC_t"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_public_rectangular_composite_j3_hf_force_and_stress_match_ad(driver):
    """Public native MCMC / GFMC_t HF force and stress equal AD for a rectangular composite J3.

    See the module docstring for the fixture. The stored HF nuclear derivative and HF stress of the
    final measured step (one walker) must equal reverse-mode AD of the XLA Re e_L at the final
    sampled configuration, at the strict full-precision budget. Mixed compares the stored
    quantities with the same native measurement kernels independently rebuilt in full precision
    at that identical configuration; it also checks a fresh mixed kernel reconstruction. Both
    policies retain the nonzero J3 contribution. Full/mixed sampling trajectories are not compared.
    """
    source_path = _full_precision_source()
    ham = _rectangular_j3_hamiltonian(source_path=source_path)
    force, stress, r_up, r_dn = _run_public(driver, ham)

    if is_mixed_precision_enabled():
        # Preserve the original mixed-mode liveness floor even while the
        # independent analytical reference is evaluated under the full policy.
        live_atol, live_rtol = get_tolerance_min(_ZONES, "medium")

        def case_factory():
            # Each policy loads a fresh Hamiltonian from the identical FP64
            # serialized source. Coordinates come from the actual mixed run.
            return _rectangular_j3_hamiltonian(source_path=source_path)

        def analytical(fresh):
            values = _native_hf_measurement_outputs(fresh, r_up, r_dn, driver)
            control = _native_hf_measurement_outputs(_without_j3(fresh), r_up, r_dn, driver)
            expected_shapes = {"force": (int(fresh.structure_data.positions.shape[0]), 3), "stress": (3, 3)}
            for field in ("force", "stress"):
                assert values[field].shape == expected_shapes[field], (field, values[field].shape)
                assert np.all(np.isfinite(values[field])) and np.all(np.isfinite(control[field])), field
                reference_norm = np.max(np.abs(values[field]))
                assert reference_norm > live_atol + live_rtol * reference_norm, (field, values[field])
                assert not np.allclose(values[field], control[field], atol=live_atol, rtol=live_rtol), field
            return values

        mixed, full = mixed_vs_full(
            case_factory,
            analytical,
            configure_native=jqmc_rs.configure_native_precision,
            native_policy_snapshot=jqmc_rs.native_precision_support,
        )
        # MCMC stores raw HF dE_L/dR and dE_L/deps. GFMC_t's one-walker
        # weighted sum / weight is the same final measurement. Keep the original
        # final-step scope and avoid comparing two different sampling trajectories.
        stored = {"force": force, "stress": stress}
        blocks = mixed_fidelity_blocks(
            {"public_stored": stored, "native_rebuilt": mixed},
            {"public_stored": full, "native_rebuilt": full},
        )
        atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
        for mixed_block, full_block in blocks:
            _assert_parameter_derivative_mixed_fidelity(mixed_block, full_block, atol, rtol)
        mutated = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            _assert_parameter_derivative_mixed_fidelity(mutated, blocks[0][1], atol, rtol)
        return

    ref_force, ref_stress = _hf_references(ham, r_up, r_dn)
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    live_atol, live_rtol = get_tolerance_min(_ZONES, "medium")

    no_j3_force, no_j3_stress = _hf_references(_without_j3(ham), r_up, r_dn)
    assert not np.allclose(ref_force, no_j3_force, atol=live_atol, rtol=live_rtol), (
        "J3 share of the HF force is below tolerance"
    )
    assert not np.allclose(ref_stress, no_j3_stress, atol=live_atol, rtol=live_rtol), (
        "J3 share of the HF stress is below tolerance"
    )

    for name, value, reference in (("HF dE_L/dR", force, ref_force), ("HF stress dE_L/deps", stress, ref_stress)):
        assert value.shape == reference.shape, (name, value.shape, reference.shape)
        assert np.all(np.isfinite(reference)), (name, reference)
        assert np.all(np.isfinite(value)), (name, value)
        assert np.max(np.abs(reference)) > live_atol + live_rtol * np.max(np.abs(reference)), (name, reference)
        np.testing.assert_allclose(
            value,
            reference,
            atol=atol,
            rtol=rtol,
            equal_nan=False,
            err_msg=f"public {driver} rectangular dense-MO composite J3 != AD reference: {name}",
        )

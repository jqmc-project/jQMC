"""Value verdicts for the complex quantities the drivers consume at k != Gamma.

``test_integration_complex_contamination.py`` guards the complex Bloch path only
through finiteness and the session-wide ``ComplexWarning`` trap.  That trap sees
implicit NumPy-style casts only: an explicit ``jnp.real``, a ``jax.Array.astype``
to a real dtype and the transpose of a real-to-complex promotion emit no warning
in JAX 0.10.  A real projection of Psi, of the geminal matrix, of the kinetic
energy or of a parameter derivative therefore stays green there.

The nodes below compare each complex quantity, imaginary part included, with an
independent complex recomputation (``_twist_geminal_cases``): the MO contraction,
geminal assembly, determinant and Jastrow exponent are written from the defining
formulas over the production AO values, and its derivatives are XLA
automatic derivatives of that complex ``ln Psi``.  Every comparison also shows
that the real-projected reference is rejected by the same tolerance, i.e. the
imaginary parts are larger than the comparison band.

Fixture: ``H4_pbc_ae_k025_cart.h5`` all-electron (no ECP), J1+J2
(no J3 at the twist), four geminal representations.
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min
from jqmc.coulomb_potential import compute_coulomb_potential
from jqmc.determinant import _compute_ratio_determinant_part_rank1_update
from jqmc.hamiltonians import compute_local_energy
from jqmc.jqmc_mcmc import MCMC
from jqmc.wavefunction import compute_discretized_kinetic_energy, evaluate_ln_wavefunction, evaluate_wavefunction
from tests.test_jqmc_kernel._twist_geminal_cases import (
    CASES,
    build_case,
    kinetic_reference,
    ln_psi_reference,
    psi_reference,
    reference_geminal,
    reference_parameters,
)

pytestmark = [pytest.mark.with_chdir]

jax.config.update("jax_enable_x64", True)

_NUM_CONFIGURATIONS = 3
_ALAT = 0.30
_PSI_ZONES = ("ao_eval", "jastrow_eval", "det_eval", "wf_eval")
_RATIO_ZONES = ("ao_eval", "jastrow_eval", "det_eval", "det_ratio", "wf_ratio")
_E_L_ZONES = ("ao_eval", "ao_grad_lap", "jastrow_eval", "jastrow_grad_lap", "det_grad_lap", "wf_kinetic", "coulomb")
_O_K_ZONES = ("ao_eval", "jastrow_eval", "det_eval", "det_grad_lap", "wf_eval")


def _configurations(hamiltonian):
    geminal = hamiltonian.wavefunction_data.geminal_data
    rng = np.random.Generator(np.random.PCG64(2026))
    return [
        (
            rng.normal(scale=1.5, size=(geminal.num_electron_up, 3)),
            rng.normal(scale=1.5, size=(geminal.num_electron_dn, 3)),
        )
        for _ in range(_NUM_CONFIGURATIONS)
    ]


def _assert_complex_shape_and_phase(actual, reference, zones, label):
    """Check shape, finite complex dtype and sensitivity to a real projection."""
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    assert actual.shape == reference.shape, label
    assert np.iscomplexobj(actual), f"{label}: production value lost its complex dtype ({actual.dtype})"
    assert np.all(np.isfinite(actual)), label
    assert np.all(np.isfinite(reference)), label
    atol, rtol = get_tolerance_min(zones, "strict")
    band = atol + rtol * np.abs(np.real(reference))
    excess = np.max(np.abs(actual - np.real(reference)) - band)
    assert excess > 0.0, f"{label}: Im is inside the comparison band, a real projection would pass (excess {excess:.3e})"


@pytest.mark.light
@pytest.mark.parametrize("case", CASES)
def test_twist_complex_values_match_independent_reference(case):
    """Psi phase, e_L and wavefunction ratios at k != Gamma equal an independent complex recomputation.

    Fixture: H4_pbc_ae_k025_cart.h5 (all-electron, cubic cell, k_frac=(0.25,0,0)) with the reviewed
    J1+J2 recipe, for the committed time-reversal SD, an untied SD (time_reversal=False, perturbed
    down MOs), the full-rank dense AO AGP lift and a four-MO AGPn with occupations 1, 1/2, 1/4, 1/8.
    Inputs: three PCG64(seed=2026) normal(scale=1.5) configurations; no conditioning screen.
    Reference: the MO contraction, geminal assembly, determinant and Jastrow exponent written from
    the defining formulas over the production AO values; kinetic energy from the complex Hessian of
    that Psi; potential from the production real Coulomb/Ewald sum. Compared: the unit phase
    Psi/|Psi|, ln|Psi|, the complex local energy, the six-direction lattice kinetic elements
    -1/(2 a^2) Psi(R')/Psi(R) (a = 0.30) and the rank-1 determinant ratios for the same moves.
    Each comparison is strict in its own precision mode, Re and Im together, and must also reject
    the real-projected reference, so a real projection anywhere upstream fails.
    """
    hamiltonian = build_case(case)
    wavefunction = hamiltonian.wavefunction_data
    geminal = wavefunction.geminal_data
    eye = jnp.eye(3)
    phases, phase_refs, e_ls, e_l_refs = [], [], [], []
    kinetic_elements, kinetic_refs, det_ratios, det_ratio_refs = [], [], [], []
    for r_up, r_dn in _configurations(hamiltonian):
        r_up_j, r_dn_j = jnp.asarray(r_up), jnp.asarray(r_dn)
        psi = complex(evaluate_wavefunction(wavefunction, r_up_j, r_dn_j))
        psi_ref = complex(psi_reference(wavefunction, r_up_j, r_dn_j))
        assert np.isfinite(psi) and np.isfinite(psi_ref) and abs(psi_ref) > 0.0
        phases.append(psi / abs(psi_ref))
        phase_refs.append(psi_ref / abs(psi_ref))
        ln_abs = float(evaluate_ln_wavefunction(wavefunction, r_up_j, r_dn_j))
        atol, rtol = get_tolerance_min(_PSI_ZONES, "strict")
        np.testing.assert_allclose(ln_abs, np.log(abs(psi_ref)), atol=atol, rtol=rtol, equal_nan=False)

        potential = np.asarray(compute_coulomb_potential(hamiltonian.coulomb_potential_data, r_up_j, r_dn_j, eye, wavefunction))
        assert np.isrealobj(potential) and np.all(np.isfinite(potential))
        e_ls.append(complex(compute_local_energy(hamiltonian, r_up_j, r_dn_j, eye)))
        e_l_refs.append(complex(jnp.sum(kinetic_reference(wavefunction, r_up_j, r_dn_j))) + float(potential))

        mesh_up, mesh_dn, elements = compute_discretized_kinetic_energy(_ALAT, wavefunction, r_up_j, r_dn_j, eye)
        psi_mesh = jax.vmap(psi_reference, in_axes=(None, 0, 0))(wavefunction, mesh_up, mesh_dn)
        kinetic_elements.append(np.asarray(elements))
        kinetic_refs.append(-1.0 / (2.0 * _ALAT**2) * np.asarray(psi_mesh) / psi_ref)

        g_ref = np.asarray(reference_geminal(wavefunction, r_up_j, r_dn_j))
        ratios = _compute_ratio_determinant_part_rank1_update(
            geminal, jnp.asarray(np.linalg.inv(g_ref)), r_up_j, r_dn_j, mesh_up, mesh_dn
        )
        det_mesh = jax.vmap(lambda u, d: jnp.linalg.det(reference_geminal(wavefunction, u, d)))(mesh_up, mesh_dn)
        det_ratios.append(np.asarray(ratios))
        det_ratio_refs.append(np.asarray(det_mesh) / np.linalg.det(g_ref))

    _assert_complex_shape_and_phase(phases, phase_refs, _PSI_ZONES, f"{case}: Psi/|Psi|")
    atol, rtol = get_tolerance_min(_PSI_ZONES, "strict")
    np.testing.assert_allclose(phases, phase_refs, atol=atol, rtol=rtol, equal_nan=False, err_msg=f"{case}: Psi/|Psi|")
    _assert_complex_shape_and_phase(e_ls, e_l_refs, _E_L_ZONES, f"{case}: complex e_L")
    atol, rtol = get_tolerance_min(_E_L_ZONES, "strict")
    np.testing.assert_allclose(e_ls, e_l_refs, atol=atol, rtol=rtol, equal_nan=False, err_msg=f"{case}: complex e_L")
    _assert_complex_shape_and_phase(kinetic_elements, kinetic_refs, _RATIO_ZONES, f"{case}: lattice kinetic elements")
    atol, rtol = get_tolerance_min(_RATIO_ZONES, "strict")
    np.testing.assert_allclose(
        kinetic_elements, kinetic_refs, atol=atol, rtol=rtol, equal_nan=False, err_msg=f"{case}: lattice kinetic elements"
    )
    _assert_complex_shape_and_phase(det_ratios, det_ratio_refs, _RATIO_ZONES, f"{case}: rank-1 determinant ratios")
    atol, rtol = get_tolerance_min(_RATIO_ZONES, "strict")
    np.testing.assert_allclose(
        det_ratios, det_ratio_refs, atol=atol, rtol=rtol, equal_nan=False, err_msg=f"{case}: rank-1 determinant ratios"
    )
    jax.clear_caches()


def _reference_sources(wavefunction, r_up, r_dn):
    """Holomorphic derivatives of the complex reference ``ln Psi`` per reference parameter."""
    parameters = reference_parameters(wavefunction)
    return jax.grad(lambda p: ln_psi_reference(wavefunction, r_up, r_dn, p), holomorphic=True)(parameters)


@pytest.mark.light
@pytest.mark.parametrize("case", CASES)
def test_twist_driver_consumed_values_match_independent_reference(case):
    """MCMC-stored complex e_L and geminal O_k sources at k != Gamma equal the independent reference.

    Production XLA MCMC (2 walkers, seed 3446, 3 measurements of 4 moves, Dt=0.5, epsilon_AS=0,
    comput_log_WF_param_deriv=True) on the four H4 twist geminal cases of the sibling node. At the
    final walker positions the stored complex e_L of the last measurement equals the independent
    Hessian-plus-Coulomb reference, and the optimizer derivative sources of every geminal block equal
    the holomorphic derivatives of the independent complex ln Psi: the Wirtinger pair (d_p, d_pbar)
    of a time-reversal MO block as the derivatives through the up and the conjugate down rows, the
    untied MO blocks and the AGP ambient lambda as d_p (with d_pbar = 0 for AGP), and the AGPn
    occupations as the complex direct derivative. Strict per precision mode, Re and Im together; the
    real-projected reference of every geminal source must be rejected. J1/J2 sources (real, geometry
    only) are not part of this claim.
    """
    hamiltonian = build_case(case)
    wavefunction = hamiltonian.wavefunction_data
    geminal = wavefunction.geminal_data
    assert os.environ.get("JQMC_FLAG_RUST_ON_CPU") == "0", "this oracle is the XLA production route"
    mcmc = MCMC(
        hamiltonian_data=hamiltonian,
        num_walkers=2,
        num_mcmc_per_measurement=4,
        mcmc_seed=3446,
        Dt=0.5,
        epsilon_AS=0.0,
        comput_log_WF_param_deriv=True,
    )
    mcmc.run(num_mcmc_steps=3)
    r_up_final = np.asarray(mcmc._MCMC__latest_r_up_carts)
    r_dn_final = np.asarray(mcmc._MCMC__latest_r_dn_carts)
    blocks = wavefunction.get_variational_blocks(state_token=("twist-value-oracle", 0), opt_geminal=True)
    sources = {source.name: source for source in mcmc._optimization_derivative_sources(blocks=blocks)}
    geminal_blocks = [block.name for block in blocks if block.name not in ("j1_param", "j2_param")]
    assert geminal_blocks, f"{case}: no geminal block published"

    e_l_refs = []
    actual_components: dict[str, list] = {}
    reference_components: dict[str, list] = {}
    for walker in range(r_up_final.shape[0]):
        r_up_j, r_dn_j = jnp.asarray(r_up_final[walker]), jnp.asarray(r_dn_final[walker])
        potential = compute_coulomb_potential(hamiltonian.coulomb_potential_data, r_up_j, r_dn_j, jnp.eye(3), wavefunction)
        e_l_refs.append(complex(jnp.sum(kinetic_reference(wavefunction, r_up_j, r_dn_j))) + float(potential))
        derivatives = {key: np.asarray(value) for key, value in _reference_sources(wavefunction, r_up_j, r_dn_j).items()}
        expected = {}
        if "mo_coefficients_up" in geminal_blocks:
            if geminal.time_reversal:
                expected[("mo_coefficients_up", "d_p")] = derivatives["cu"]
                expected[("mo_coefficients_up", "d_pbar")] = derivatives["cd"]
            else:
                expected[("mo_coefficients_up", "d_p")] = derivatives["cu"]
                expected[("mo_coefficients_dn", "d_p")] = derivatives["cd"]
        if "lambda_matrix" in geminal_blocks:
            expected[("lambda_matrix", "d_p")] = derivatives["lam"]
            expected[("lambda_matrix", "d_pbar")] = np.zeros_like(derivatives["lam"])
        if "lambda_diag" in geminal_blocks:
            expected[("lambda_diag", "direct")] = derivatives["lam"]
        for (name, component), reference in expected.items():
            stored = getattr(sources[name], component)
            assert stored is not None, f"{case}: {name}.{component} was not published"
            actual_components.setdefault(f"{name}.{component}", []).append(np.asarray(stored)[-1, walker])
            reference_components.setdefault(f"{name}.{component}", []).append(reference)
    assert {key.split(".")[0] for key in actual_components} == set(geminal_blocks)

    stored_e_l = np.asarray(mcmc.e_L)[-1]
    atol, rtol = get_tolerance_min(_E_L_ZONES, "strict")
    assert np.iscomplexobj(stored_e_l) and np.all(np.isfinite(stored_e_l))
    np.testing.assert_allclose(stored_e_l, np.asarray(e_l_refs), atol=atol, rtol=rtol, equal_nan=False)
    for key, values in actual_components.items():
        reference = np.asarray(reference_components[key])
        if key == "lambda_matrix.d_pbar":
            actual = np.asarray(values)
            assert np.all(np.isfinite(actual))
            np.testing.assert_array_equal(actual, reference)
            continue
        _assert_complex_shape_and_phase(values, reference, _O_K_ZONES, f"{case}: MCMC source {key}")
        atol, rtol = get_tolerance_min(_O_K_ZONES, "strict")
        np.testing.assert_allclose(
            values, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=f"{case}: MCMC source {key}"
        )
    jax.clear_caches()

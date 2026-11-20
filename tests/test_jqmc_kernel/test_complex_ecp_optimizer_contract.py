"""Complex ECP source, finite-sample and actual-sampler optimizer contracts."""

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

import numpy as np
import pytest
import scipy.linalg

from jqmc._setting import OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF, RTOL_FD_VS_ANALYTIC_TIGHT
from jqmc.jqmc_mcmc import MCMC
from jqmc.jqmc_optimizer import (
    OptimizationCoordinateMap,
    build_reduced_lm_real_pencil,
    estimate_coordinate_force_jackknife,
    solve_lm_real_pencil,
)

from ._complex_ecp_optimizer_harness import (
    MASK,
    SELECTION,
    _FD_H,
    assert_algebra,
    assert_fd,
    assert_same_mode,
    build_ham,
    make_case,
)
from ._optimizer_step_reference import reference_lm_update
from .test_optimizer_lm_real import _centered_oracle

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]


@pytest.fixture(autouse=True)
def used_electron_coordinates(request):
    """Expose each test's actual cached sample coordinates to live GDD observation.

    A module-scoped Case may have been built for an earlier test. Read that
    exact requested Case, without rebuilding, resampling or injecting buffers.
    The normal runtime observer records these arrays under coordinate names.
    """
    if "point_case" in request.fixturenames:
        case = request.getfixturevalue("point_case")
    elif "runtime" in request.fixturenames:
        case = request.getfixturevalue("runtime")[1]
    else:
        raise AssertionError("complex ECP test has no declared sample fixture")
    r_up_carts = np.asarray(case.up)
    r_dn_carts = np.asarray(case.dn)
    assert np.all(np.isfinite(r_up_carts)) and np.all(np.isfinite(r_dn_carts))
    return r_up_carts, r_dn_carts


@pytest.fixture(scope="module")
def level(request):
    return "medium" if request.config.getoption("--precision-mode") == "mixed" else "strict"


@pytest.fixture(scope="module", params=("gamma", "twist"))
def point_case(request):
    ham = build_ham(request.param)
    blocks = ham.wavefunction_data.get_variational_blocks(state_token=("complex-ecp-source", 0), **MASK)
    return make_case(ham, blocks)


def test_pointwise_sources_match_same_mode_autodiff(point_case, level):
    """Untied lithium ECP Gamma/twist J2 and first-MO x/y sources equal same-mode AD.

    Eight fixed seed-2026 configurations and identity ECP rotation are reused
    for every direction. The AD forward explicitly sums kinetic, bare/local
    potential and complex nonlocal ECP, without differentiating analytic sources.
    """
    for label, direction, derivative, log_derivative in point_case.probes():
        reference_d, reference_o = point_case.autodiff(direction)
        assert_same_mode(derivative, reference_d, level, label + " energy source vs AD")
        assert_same_mode(log_derivative, reference_o, level, label + " logarithmic source vs AD")


@pytest.mark.skip_mixed_precision(
    reason="Finite differences require full precision; mixed correctness is covered by same-mode complex ECP references.",
    alternative_guarantee_ref="MATH-DERIV-COMPLEX-ECP-AD-001",
)
def test_pointwise_sources_match_independent_finite_difference(point_case):
    """Fixed J2/MO x/y directions match explicit complex-energy and wavefunction FD.

    Gamma and twist committed lithium fixtures use a fixed Richardson stencil,
    central step from the established complex optimizer harness, and unchanged
    central FD tolerance. Both production and separately assembled forward
    energies must refer to the same unprojected complex quantity.
    """
    for label, direction, derivative, log_derivative in point_case.probes():
        production_fd, reference_fd, reference_o = point_case.finite_difference(direction)
        assert_fd(derivative, production_fd, label + " production-forward FD")
        assert_fd(derivative, reference_fd, label + " explicit-energy FD")
        assert_fd(log_derivative, reference_o, label + " logarithmic FD")


@pytest.mark.skip_mixed_precision(
    reason="Finite differences require full precision; mixed correctness is covered by same-mode complex ECP references.",
    alternative_guarantee_ref="MATH-DERIV-COMPLEX-ECP-AD-001",
)
def test_finite_sample_complex_energy_chain_rule(point_case):
    """Reweighted complex energy obeys the exact finite-sample chain rule including mean D.

    The fixed sample identity is mean(D)+2 cov(Re(O),E), not the optimizer's
    covariance-only force and not a stochastic convergence criterion.
    """
    energy, psi0 = point_case.base[1:3]
    for label, direction, _derivative, _log_derivative in point_case.probes():
        _production_fd, derivative, log_derivative = point_case.finite_difference(direction)
        plus, minus = point_case.shifted_values(direction, _FD_H), point_case.shifted_values(direction, -_FD_H)
        wp, wm = abs(plus[2] / psi0) ** 2, abs(minus[2] / psi0) ** 2
        reference = (np.dot(wp, plus[1]) / wp.sum() - np.dot(wm, minus[1]) / wm.sum()) / (2 * _FD_H)
        actual = derivative.mean() + 2 * np.mean(np.real(log_derivative - log_derivative.mean()) * (energy - energy.mean()))
        assert_fd(actual, reference, label + " finite-sample quotient rule")


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    # Module-scoped sampler setup also gets a private output directory; the
    # normal per-test cwd isolation remains enabled for every collected test.
    import os

    previous = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("complex-ecp-optimizer"))
    try:
        ham = build_ham("twist")
        sampler = MCMC(
            hamiltonian_data=ham,
            mcmc_seed=3446,
            num_walkers=8,
            num_mcmc_per_measurement=1,
            Dt=0.5,
            epsilon_AS=0.0,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=True,
            random_discretized_mesh=False,
        )
        sampler.select_parameters(SELECTION.gradient_flag_kwargs())
        sampler.run(num_mcmc_steps=1)
        blocks = sampler.variational_blocks(SELECTION)
        case = make_case(
            ham, blocks, np.asarray(sampler._MCMC__latest_r_up_carts), np.asarray(sampler._MCMC__latest_r_dn_carts)
        )
        batch = sampler.optimization_batch(
            num_warmup_steps=0, num_bin_blocks=1, blocks=blocks, need_local_energy_derivatives=True
        )
        yield sampler, case, batch, OptimizationCoordinateMap.from_blocks(blocks)
    finally:
        os.chdir(previous)


def test_runtime_publication_preserves_complex_sources(runtime, level):
    """Actual twist MCMC samples publish complex energies and exact source components.

    Seed 3446, eight walkers and one measured step use fixed identity ECP
    rotation. No eL/O/D buffer is injected. Actual raw stored values and real
    optimizer coordinates are checked separately from legacy projected storage.
    """
    sampler, case, batch, coordinate_map = runtime
    np.testing.assert_array_equal(batch.local_energies, np.asarray(sampler.e_L).ravel())
    assert_same_mode(batch.local_energies, case.base[0], level, "stored production forward")
    assert_same_mode(batch.local_energies, case.base[1], level, "stored explicit complex energy")
    stored_o = np.asarray(sampler.get_dln_WF(num_mcmc_warmup_steps=0, blocks=case.blocks)).reshape(len(case.up), case.size)
    assert_same_mode(stored_o, case.O, level, "stored logarithmic source")
    encoded = np.asarray(coordinate_map.encode(np.eye(coordinate_map.coordinate_size)))
    assert_same_mode(
        coordinate_map.transform_derivatives(batch.log_derivative_sources), case.O @ encoded.T, level, "coordinate O"
    )
    assert_same_mode(
        coordinate_map.transform_derivatives(batch.local_energy_derivative_sources), case.D @ encoded.T, level, "coordinate D"
    )


def test_runtime_generalized_force_uses_complex_energy(runtime, level):
    """Actual batch force equals complex-energy covariance before real-coordinate projection."""
    _sampler, case, batch, coordinate_map = runtime
    o = np.asarray(coordinate_map.transform_derivatives(batch.log_derivative_sources))
    e, w = case.base[1], np.asarray(batch.weights)
    centered = o - np.average(o, axis=0, weights=w)
    expected = -2 * np.real(centered.conj().T @ (w * (e - np.average(e, weights=w)))) / w.sum()
    actual, _std = estimate_coordinate_force_jackknife(batch, coordinate_map)
    assert_same_mode(actual, expected, level, "coordinate covariance force")


def test_legacy_get_gf_exact_delete_one_bin_consumption(runtime, level):
    """Legacy generalized force consumes actual buffers as exact delete-one-bin covariances."""
    sampler, case, _batch, _coordinate_map = runtime
    e, w = np.asarray(sampler.e_L).ravel(), np.asarray(sampler.w_L).ravel()
    o = np.asarray(sampler.get_dln_WF(num_mcmc_warmup_steps=0, blocks=case.blocks)).reshape(len(w), case.size)
    replicas = []
    for omit in range(len(w)):
        kept = np.arange(len(w)) != omit
        wk, ek, ok = w[kept], e[kept], o[kept]
        centered = ok - np.average(ok, axis=0, weights=wk)
        replicas.append(-2 * centered.conj().T @ (wk * (ek - np.average(ek, weights=wk))) / wk.sum())
    actual, _std = sampler.get_gF(num_mcmc_warmup_steps=0, num_mcmc_bin_blocks=1, blocks=case.blocks)
    assert_algebra(actual, np.mean(replicas, axis=0), "legacy bin covariance")


def test_legacy_get_ah_includes_nonsymmetric_bra(runtime, level):
    """Legacy LM consumes stored O/D including real-J2 projection and the unconjugated bra."""
    sampler, case, _batch, _coordinate_map = runtime
    w, e = np.asarray(sampler.w_L).ravel(), np.asarray(sampler.e_L).ravel()
    o = np.asarray(sampler.get_dln_WF(num_mcmc_warmup_steps=0, blocks=case.blocks)).reshape(len(w), case.size)
    d = np.concatenate([np.asarray(sampler.de_L_dc[b.name]).reshape(len(w), b.size) for b in case.blocks], axis=1)
    co, cd, ce = o - np.average(o, axis=0, weights=w), d - np.average(d, axis=0, weights=w), e - np.average(e, weights=w)
    expected = (
        np.real(np.average(e, weights=w)),
        -2 * co.conj().T @ (w * ce) / w.sum(),
        co.conj().T @ (w[:, None] * co) / w.sum(),
        co.conj().T @ ((w * e)[:, None] * co) / w.sum(),
        cd.conj().T @ (w[:, None] * co) / w.sum(),
        np.average(d, axis=0, weights=w) + co.T @ (w * ce) / w.sum(),
    )
    actual = sampler.get_aH(blocks=case.blocks, num_mcmc_warmup_steps=0, return_matrices=True)
    for observed, reference, name in zip(actual, expected, ("H0", "f", "S", "K", "B", "bra"), strict=True):
        assert_algebra(observed, reference, "legacy LM " + name)


def _lm_reference(runtime, *, finite_difference):
    _sampler, case, batch, coordinate_map = runtime
    count = coordinate_map.coordinate_size
    assert count >= 5
    basis = np.eye(count)[:, [0, 3, 4]]
    pencil, transform = build_reduced_lm_real_pencil(batch, coordinate_map, basis[:, 0], np.asarray([3, 4]))
    np.testing.assert_array_equal(transform, basis)
    encoded = np.asarray(coordinate_map.encode(basis.T))
    columns = [
        case.finite_difference(direction)[1:] if finite_difference else case.autodiff(direction) for direction in encoded
    ]
    d, o = (np.stack([row[index] for row in columns], axis=1) for index in (0, 1))
    expected_h, expected_s = _centered_oracle(o, d, batch.weights, case.base[1])
    metric = expected_s[1:, 1:]
    scales = np.sqrt(np.diag(metric))
    assert np.all(np.isfinite(scales)) and np.all(scales > 0)
    assert np.linalg.eigvalsh(metric / np.outer(scales, scales))[0] > OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF
    roots, vectors = scipy.linalg.eig(expected_h, expected_s)
    norms = np.real(np.einsum("ij,ik,kj->j", vectors.conj(), expected_s, vectors))
    weights = np.where(np.imag(roots) == 0, abs(vectors[0]) ** 2 / norms, -np.inf)
    ordered = np.sort(weights)
    assert np.isfinite(ordered[-1]) and ordered[-1] - ordered[-2] > RTOL_FD_VS_ANALYTIC_TIGHT
    return case, batch, coordinate_map, basis, pencil, o, d, expected_h, expected_s


@pytest.fixture(scope="module")
def lm_fd(runtime):
    return _lm_reference(runtime, finite_difference=True)


@pytest.fixture(scope="module")
def lm_ad(runtime):
    return _lm_reference(runtime, finite_difference=False)


def test_reduced_lm_matches_same_mode_ad_inputs(lm_ad, level):
    """Actual sampler nonsymmetric LM inputs and matrices agree with same-mode AD references."""
    _case, batch, coordinate_map, basis, pencil, o, d, h, s = lm_ad
    assert_same_mode(
        np.asarray(coordinate_map.transform_derivatives(batch.log_derivative_sources)) @ basis, o, level, "AD LM O input"
    )
    assert_same_mode(
        np.asarray(coordinate_map.transform_derivatives(batch.local_energy_derivative_sources)) @ basis,
        d,
        level,
        "AD LM D input",
    )
    assert_same_mode(pencil.H, h, level, "AD-input LM H")
    assert_same_mode(pencil.S, s, level, "AD-input LM S")


def test_reduced_lm_update_matches_same_mode_ad_reference(lm_ad, level):
    """Actual sampler real LM update and selected root match same-mode AD-input dense algebra."""
    case, batch, _map, _basis, pencil, o, d, _h, _s = lm_ad
    q, energy, reference_weight, active = reference_lm_update(
        o, d, case.base[1], batch.weights, OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF
    )
    solved = solve_lm_real_pencil(pencil, condition_cutoff=OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF)
    np.testing.assert_array_equal(solved.active_indices, active)
    assert np.isfinite(reference_weight) and reference_weight > 0
    assert_same_mode(solved.q, q, level, "AD-input LM update")
    assert_same_mode(solved.predicted_energy, energy, level, "AD-input LM eigenvalue")


def test_reduced_lm_rejects_missing_mean_derivative(lm_ad, level):
    """The actual complex-ECP sample rejects omission of the finite-sample mean-D bra term."""
    _case, batch, _map, _basis, pencil, _o, d, h, _s = lm_ad
    assert_same_mode(pencil.H, h, level, "unmodified LM H")
    wrong = np.array(h, copy=True)
    wrong[0, 1:] -= np.real(np.average(d, axis=0, weights=batch.weights))
    with pytest.raises(AssertionError):
        assert_same_mode(wrong, h, level, "missing mean-D bra must fire")


@pytest.mark.skip_mixed_precision(
    reason="Finite differences require full precision; mixed correctness is covered by same-mode complex ECP references.",
    alternative_guarantee_ref="MATH-OPT-COMPLEX-ECP-LM-001",
)
def test_reduced_lm_inputs_match_independent_fd(lm_fd):
    """Actual sampler LM columns in fixed coordinates 0/3/4 match independent ambient FD."""
    _case, batch, coordinate_map, basis, _pencil, o, d, _h, _s = lm_fd
    assert_fd(np.asarray(coordinate_map.transform_derivatives(batch.log_derivative_sources)) @ basis, o, "LM O input")
    assert_fd(np.asarray(coordinate_map.transform_derivatives(batch.local_energy_derivative_sources)) @ basis, d, "LM D input")


@pytest.mark.skip_mixed_precision(
    reason="Finite differences require full precision; mixed correctness is covered by same-mode complex ECP references.",
    alternative_guarantee_ref="MATH-OPT-COMPLEX-ECP-LM-001",
)
def test_reduced_lm_matrices_match_independent_fd_reference(lm_fd):
    """Nonsymmetric actual-sampler LM matrices equal dense assembly from independent FD inputs."""
    _case, _batch, _map, _basis, pencil, _o, _d, h, s = lm_fd
    assert_fd(pencil.H, h, "FD-input LM H")
    assert_fd(pencil.S, s, "FD-input LM S")


@pytest.mark.skip_mixed_precision(
    reason="Finite differences require full precision; mixed correctness is covered by same-mode complex ECP references.",
    alternative_guarantee_ref="MATH-OPT-COMPLEX-ECP-LM-001",
)
def test_reduced_lm_update_matches_independent_fd_reference(lm_fd):
    """Real LM update and eigenvalue agree with an independent FD-input nonsymmetric eigensolve."""
    case, batch, _map, _basis, pencil, o, d, _h, _s = lm_fd
    q, energy, reference_weight, active = reference_lm_update(
        o, d, case.base[1], batch.weights, OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF
    )
    solved = solve_lm_real_pencil(pencil, condition_cutoff=OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF)
    np.testing.assert_array_equal(solved.active_indices, active)
    assert np.isfinite(reference_weight) and reference_weight > 0
    assert_fd(solved.q, q, "FD-input LM update")
    assert_fd(solved.predicted_energy, energy, "FD-input LM eigenvalue")


def test_source_projection_and_conjugation_controls(runtime, level):
    """The actual twist fixture rejects real projection and conjugation of the complex energy source."""
    _sampler, case, _batch, _map = runtime
    direction = np.zeros(case.size, complex)
    direction[0] = 1
    reference, _o = case.autodiff(direction)
    assert_same_mode(case.D[:, 0], reference, level, "unmodified J2 source")
    for wrong in (np.real(case.D[:, 0]), np.conj(case.D[:, 0])):
        with pytest.raises(AssertionError):
            assert_same_mode(wrong, reference, level, "complex source fault must fire")

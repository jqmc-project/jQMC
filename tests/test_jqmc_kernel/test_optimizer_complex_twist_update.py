"""Complex-twist row of the real-sampler optimizer update oracle.

One production ``MCMC.run_optimize`` step (real sampler, real optimizer) on the
complex Bloch fixture ``H2_pbc_ae_k025_cart.h5`` (all-electron, twist
``k = (0.25, 0.25, 0.25)``, time-reversal SD) with Pade J1 = 1.0, J2 = 0.5 and
the MO block optimized together with J1/J2. The MO block is a Wirtinger pair
(``d_p``, ``d_pbar``) on an S-Stiefel tangent, so its optimizer coordinates
are complex directional derivatives. One SR step and one full-space LM step.

Deterministic checks on the stored configurations (see
``_optimizer_step_reference``):

1. the consumed complex ``e_L`` equals ``compute_local_energy``; the production
   pull-back of the published sources (``transform_derivatives``) equals the
   independent per-coordinate derivative of the phase-preserving ``ln Psi``
   (and, for LM, of ``e_L``) along the forward coordinate directions;
2. SR: ``target.q`` equals the dense solve of ``(S + eps diag S) q = f`` with
   ``S = Re<dO* dO>``, ``f = -2 Re<dO* (e - Re<e>)>``; LM: ``target.q``, the
   predicted energy and the reference weight equal the dense eigensolve of the
   real projection of the centred complex pencil after the correlation cutoff;
3. the applied change is ``delta * q``: exactly on the J1/J2 parameters and,
   for the MO block, as the typed storage update ``E (delta q)``.

Negative controls (both precision modes): conjugating the ``d_pbar``
component (a) in the production real adjoint and (b) in the MCMC source
publisher must fail the update verdict and the source verdict respectively.
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

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jqmc.derivatives.bridge as bridge_module
import jqmc.jqmc_optimizer as optimizer_module
from jqmc._precision import get_tolerance, get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import (
    ATOL_PYTHON_FP64_LOOSE,
    ATOL_COMPLEX_GRAD_FD,
    OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF,
    RTOL_PYTHON_FP64_LOOSE,
    RTOL_COMPLEX_GRAD_FD,
    min_S_diag_abs,
)
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.determinant import to_AGP
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_two_body_data
from jqmc.jqmc_optimizer import OptimizationCoordinateMap
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

from tests.test_jqmc_kernel._optimizer_step_reference import (
    LN_PSI_ZONES,
    LOCAL_ENERGY_ZONES,
    StepCapture,
    _complex_ln_psi,
    coordinate_columns,
    encode_matrix,
    independent_local_energies,
    record_one_optimize_step,
    reference_lm_update,
    reference_sr_update,
)

pytestmark = [pytest.mark.with_chdir]

jax.config.update("jax_enable_x64", True)

_TREXIO_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H2_pbc_ae_k025_cart.h5")
_J1_START = 1.0
_J2_START = 0.5
_NUM_WALKERS = 4
_NUM_STEPS = 25
_NUM_BIN_BLOCKS = 5
_DELTA = 0.05
_EPSILON = 1.0e-3
_SEED = 20260930
_BLOCK_NAMES = ["j1_param", "j2_param", "mo_coefficients_up"]
_JASTROW_COORDINATES = [0, 1]
# The optimizer stores the updated J1/J2 scalars in float64 in both modes
# (jastrow_factor._JASTROW_PARAMETER_STORAGE_DTYPE, issue #781), so the applied
# change equals the fp64 step up to fp64 round-off on the same data.
_PARAMETER_TOLERANCE = (ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_LOOSE)
_CASES = ("sr", "lm")
_CAPTURE_CACHE: dict[tuple[str, bool], StepCapture] = {}


def _build_hamiltonian() -> Hamiltonian_data:
    structure_data, _aos, _mos_up, _mos_dn, geminal_data, coulomb_potential_data = read_trexio_file(
        trexio_file=_TREXIO_FILE, store_tuple=True
    )
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=_J1_START,
            structure_data=structure_data,
            core_electrons=tuple(0.0 for _ in structure_data.atomic_numbers),
            jastrow_1b_type="pade",
        ),
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=_J2_START, jastrow_2b_type="pade", structure_data=structure_data
        ),
        jastrow_three_body_data=None,
    )
    return Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data),
    )


def _optimizer_kwargs(case: str) -> dict:
    kwargs = {"method": "sr", "delta": _DELTA, "epsilon": _EPSILON, "cg_flag": False}
    if case == "lm":
        kwargs.update(use_lm=True, lm_subspace_dim=-1)
    return kwargs


def _record(case: str) -> StepCapture:
    return record_one_optimize_step(
        _build_hamiltonian(),
        optimizer_kwargs=_optimizer_kwargs(case),
        selection={"opt_J1_param": True, "opt_J2_param": True, "opt_J3_param": False, "opt_geminal": True},
        use_lm=case == "lm",
        seed=_SEED,
        num_walkers=_NUM_WALKERS,
        num_steps=_NUM_STEPS,
        num_bin_blocks=_NUM_BIN_BLOCKS,
    )


def _capture(case: str) -> StepCapture:
    key = (case, is_mixed_precision_enabled())
    if key not in _CAPTURE_CACHE:
        _CAPTURE_CACHE[key] = _record(case)
    return _CAPTURE_CACHE[key]


def _jastrow_parameters(hamiltonian: Hamiltonian_data) -> np.ndarray:
    jastrow = hamiltonian.wavefunction_data.jastrow_data
    return np.asarray(
        [
            np.asarray(jastrow.jastrow_one_body_data.jastrow_1b_param),
            np.asarray(jastrow.jastrow_two_body_data.jastrow_2b_param),
        ],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------- #
# Verdicts (shared by the positive checks and the negative controls)
# --------------------------------------------------------------------------- #


def _assert_sources_match_independent(capture: StepCapture, log_derivatives, energy_derivatives=None) -> None:
    """The consumed e_L and coordinate columns equal the independent evaluation at the stored configurations."""
    hamiltonian = capture.hamiltonian_before
    atol, rtol = get_tolerance_min(LOCAL_ENERGY_ZONES, "strict")
    np.testing.assert_allclose(
        capture.local_energies,
        independent_local_energies(hamiltonian, capture.r_up, capture.r_dn),
        rtol=rtol,
        atol=atol,
        err_msg="consumed complex e_L differs from the independent evaluation",
    )
    ln_atol, ln_rtol = get_tolerance_min(LN_PSI_ZONES, "strict")
    np.testing.assert_allclose(
        np.asarray(log_derivatives),
        coordinate_columns(hamiltonian, capture.coordinate_map, capture.r_up, capture.r_dn, "ln_psi"),
        rtol=ln_rtol,
        atol=ln_atol,
        err_msg="consumed d ln Psi / dq differs from the independent coordinate derivative",
    )
    if energy_derivatives is not None:
        np.testing.assert_allclose(
            np.asarray(energy_derivatives),
            coordinate_columns(hamiltonian, capture.coordinate_map, capture.r_up, capture.r_dn, "local_energy"),
            rtol=rtol,
            atol=atol,
            err_msg="consumed d e_L / dq differs from the independent coordinate derivative",
        )


def _assert_update_matches_reference(actual, reference, label: str) -> None:
    # Batch arrays and the host solve are fp64 in both precision modes.
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(reference),
        rtol=RTOL_PYTHON_FP64_LOOSE,
        atol=ATOL_PYTHON_FP64_LOOSE,
        err_msg=f"production {label} update differs from the dense reference",
    )


def _sr_reference(capture: StepCapture) -> np.ndarray:
    q, _diagonal, active = reference_sr_update(
        capture.log_derivatives, capture.local_energies, capture.weights, _EPSILON, min_S_diag_abs
    )
    assert np.all(active), "precondition: no coordinate falls under the SR diagonal floor"
    return q


def _lm_reference(capture: StepCapture):
    return reference_lm_update(
        capture.log_derivatives,
        capture.local_energy_derivatives,
        capture.local_energies,
        capture.weights,
        OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF,
    )


def _replay_target(capture: StepCapture):
    kwargs = dict(capture.target_kwargs)
    kwargs["warm_start"] = None
    return optimizer_module.compute_real_coordinate_optimization_target(capture.batch, capture.coordinate_map, **kwargs)


def _mo_source(capture: StepCapture):
    return next(source for source in capture.batch.log_derivative_sources if source.name == "mo_coefficients_up")


# --------------------------------------------------------------------------- #
# Positive checks
# --------------------------------------------------------------------------- #


@pytest.mark.light
@pytest.mark.parametrize("case", _CASES)
def test_complex_twist_batch_matches_independent_coordinate_derivatives(case):
    """Consumed complex e_L, d ln Psi / dq (and d e_L / dq) equal the independent coordinate derivatives."""
    capture = _capture(case)
    assert [block.name for block in capture.coordinate_map.blocks] == _BLOCK_NAMES
    assert capture.coordinate_map.blocks[2].derivative_policy == "wirtinger_pair"
    assert np.all(capture.weights > 0.0)
    # The fixture is genuinely complex: e_L, the MO columns and the d_pbar component carry
    # imaginary parts above the zone tolerance, so a real projection or a conjugation is visible.
    atol, _rtol = get_tolerance_min(LN_PSI_ZONES, "strict")
    assert np.max(np.abs(np.imag(capture.local_energies))) > atol
    assert np.max(np.abs(np.imag(capture.log_derivatives[:, 2:]))) > atol
    assert np.max(np.abs(np.imag(_mo_source(capture).d_pbar))) > atol
    _assert_sources_match_independent(capture, capture.log_derivatives, capture.local_energy_derivatives)


def _assert_applied_change(capture: StepCapture, reference_q) -> None:
    np.testing.assert_allclose(
        np.asarray(capture.step.q), _DELTA * np.asarray(reference_q), rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_PYTHON_FP64_LOOSE
    )
    atol, rtol = _PARAMETER_TOLERANCE
    applied = _jastrow_parameters(capture.hamiltonian_after) - _jastrow_parameters(capture.hamiltonian_before)
    np.testing.assert_allclose(applied, _DELTA * np.asarray(reference_q)[_JASTROW_COORDINATES], rtol=rtol, atol=atol)
    mo_update = next(update for update in capture.step.updates if update.name == "mo_coefficients_up")
    storage = capture.coordinate_map.block_storage_slices[2]
    expected = (_DELTA * np.asarray(reference_q)) @ encode_matrix(capture.coordinate_map)[:, storage]
    np.testing.assert_allclose(
        np.asarray(mo_update.delta).reshape(-1), expected, rtol=RTOL_PYTHON_FP64_LOOSE, atol=ATOL_PYTHON_FP64_LOOSE
    )


@pytest.mark.light
def test_complex_twist_run_optimize_sr_update_matches_dense_recomputation():
    """The complex-twist SR target and the applied change equal the dense (S + eps diag S) q = f solve."""
    capture = _capture("sr")
    assert capture.target.frozen_coordinate_groups == ()
    reference_q = _sr_reference(capture)
    _assert_update_matches_reference(capture.target.q, reference_q, "SR")
    _assert_applied_change(capture, reference_q)


@pytest.mark.light
def test_complex_twist_run_optimize_lm_update_matches_dense_pencil():
    """The complex-twist full-space LM target and the applied change equal the dense centred-pencil eigensolve."""
    capture = _capture("lm")
    assert capture.target.rejection_reason is None
    assert capture.target.accepted_scale == 1.0
    reference_q, reference_energy, reference_weight, _survivors = _lm_reference(capture)
    _assert_update_matches_reference(capture.target.q, reference_q, "LM")
    _assert_update_matches_reference(capture.target.predicted_energy, reference_energy, "LM energy")
    _assert_update_matches_reference(capture.target.lm_reference_weight, reference_weight, "LM reference weight")
    _assert_applied_change(capture, reference_q)


def _independent_hermitian_basis(size):
    """Orthonormal generators of Herm(size), independent of the production tie."""
    units = np.eye(size * size, dtype=np.complex128).reshape(size * size, size, size)
    basis = [units[i * size + i] for i in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            forward, backward = units[i * size + j], units[j * size + i]
            basis.extend(((forward + backward) / np.sqrt(2.0), 1j * (forward - backward) / np.sqrt(2.0)))
    return np.stack(basis)


@pytest.mark.heavy
@pytest.mark.parametrize("use_device_optimizer", [False, True])
def test_complex_twist_agp_lambda_step_preserves_the_whole_hermitian_manifold(use_device_optimizer):
    """Real host/device SR steps consume all independent Hermitian directions.

    A closed-form basis defines the physical displacement without consulting
    the optimizer's forward map or tie builder. Its directional AD columns
    have a separate phase-preserving finite-difference anchor. The unchanged
    J1/J2 sources are held fixed while independently recomputing the dense SR
    solve, isolating this test's lambda-manifold and applied-storage contract.
    """
    ham = _build_hamiltonian()
    agp = to_AGP(ham.wavefunction_data.geminal_data)
    lam0 = np.asarray(agp.lambda_matrix)
    size = lam0.shape[0]
    perturbation = np.random.default_rng(7).standard_normal(lam0.shape)
    lam0 = lam0 + 0.05j * (perturbation - perturbation.T)
    agp = agp.replace(lambda_matrix=jnp.asarray(lam0))
    wf = ham.wavefunction_data.replace(geminal_data=agp)
    ham = ham.replace(wavefunction_data=wf)
    capture = record_one_optimize_step(
        ham,
        optimizer_kwargs=_optimizer_kwargs("sr"),
        selection={"opt_J1_param": True, "opt_J2_param": True, "opt_J3_param": False, "opt_geminal": True},
        use_lm=False,
        seed=20261104,
        num_walkers=_NUM_WALKERS,
        num_steps=_NUM_STEPS,
        num_bin_blocks=_NUM_BIN_BLOCKS,
        use_device_optimizer=use_device_optimizer,
    )
    from jqmc.jqmc_optimizer import DeviceOptimizationBatch

    assert isinstance(capture.batch, DeviceOptimizationBatch) == use_device_optimizer
    coordinate_map = capture.coordinate_map
    assert [block.name for block in coordinate_map.blocks] == ["j1_param", "j2_param", "lambda_matrix"]
    basis = _independent_hermitian_basis(size)
    assert basis.shape == (size * size, size, size)

    def displaced_ln_psi(q, up, dn):
        delta = jnp.einsum("k,kij->ij", q.astype(jnp.complex128), basis)
        displaced = wf.replace(geminal_data=agp.replace(lambda_matrix=jnp.asarray(lam0) + delta))
        return _complex_ln_psi(displaced, up, dn)

    def directional_columns(up, dn):
        zero = jnp.zeros(size * size, dtype=jnp.float64)
        real = jax.jacrev(lambda q: jnp.real(displaced_ln_psi(q, up, dn)))(zero)
        imag = jax.jacrev(lambda q: jnp.imag(displaced_ln_psi(q, up, dn)))(zero)
        return real + 1j * imag

    reference_columns = np.asarray(jax.jit(jax.vmap(directional_columns))(capture.r_up, capture.r_dn))
    actual_columns = np.asarray(capture.log_derivatives)[:, 2:]
    atol, rtol = get_tolerance("det_eval", "strict")
    assert np.isfinite(reference_columns).all() and np.isfinite(actual_columns).all()
    assert np.max(np.abs(reference_columns[:, size + 1 :: 2])) > atol

    # Anchor an imaginary-antisymmetric direction without AD or a tie map.
    direction = np.eye(size * size)[size + 1]
    h = 1.0e-4
    evaluate = jax.jit(jax.vmap(displaced_ln_psi, in_axes=(None, 0, 0)))
    plus = np.asarray(evaluate(h * direction, capture.r_up, capture.r_dn))
    minus = np.asarray(evaluate(-h * direction, capture.r_up, capture.r_dn))
    finite_difference = np.log(np.exp(plus - minus)) / (2 * h)
    np.testing.assert_allclose(
        reference_columns[:, size + 1], finite_difference, atol=ATOL_COMPLEX_GRAD_FD, rtol=RTOL_COMPLEX_GRAD_FD
    )

    reference = np.concatenate((np.asarray(capture.log_derivatives)[:, :2], reference_columns), axis=1)
    q_reference, _diagonal, active = reference_sr_update(
        reference, capture.local_energies, capture.weights, _EPSILON, min_S_diag_abs
    )
    assert np.all(active) and capture.target.frozen_coordinate_groups == ()
    expected_storage = _DELTA * np.einsum("k,kij->ij", q_reference[2:], basis)
    lam1 = np.asarray(capture.hamiltonian_after.wavefunction_data.geminal_data.lambda_matrix)
    assert np.max(np.abs(np.imag(expected_storage))) > atol
    np.testing.assert_allclose(lam1 - lam0, expected_storage, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
    assert coordinate_map.block_coordinate_slices[2].stop - coordinate_map.block_coordinate_slices[2].start == size * size
    np.testing.assert_allclose(actual_columns, reference_columns, atol=atol, rtol=rtol)
    _assert_update_matches_reference(capture.target.q, q_reference, "Hermitian AGP SR")
    np.testing.assert_allclose(lam1, lam1.conj().T, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE)
    jax.clear_caches()


# --------------------------------------------------------------------------- #
# Negative controls: a conjugated d_pbar component
# --------------------------------------------------------------------------- #


def _conjugate_pbar(source):
    if source.d_pbar is None:
        return source
    conjugated = np.conj(source.d_pbar)
    conjugated.setflags(write=False)
    return dataclasses.replace(source, d_pbar=conjugated)


@pytest.mark.light
@pytest.mark.parametrize("case", _CASES)
def test_negative_control_conjugated_pbar_in_adjoint_fails_update_check(case, monkeypatch):
    """Conjugating d_pbar inside the production real adjoint changes the replayed SR/LM target."""
    capture = _capture(case)
    reference_q = _sr_reference(capture) if case == "sr" else _lm_reference(capture)[0]
    _assert_update_matches_reference(_replay_target(capture).q, reference_q, case.upper())

    production_transform = OptimizationCoordinateMap.transform_derivatives

    def conjugated_pbar_transform(self, sources):
        return production_transform(self, [_conjugate_pbar(source) for source in sources])

    monkeypatch.setattr(OptimizationCoordinateMap, "transform_derivatives", conjugated_pbar_transform)
    with pytest.raises(AssertionError, match=f"production {case.upper()} update differs from the dense reference"):
        _assert_update_matches_reference(_replay_target(capture).q, reference_q, case.upper())


@pytest.mark.light
def test_negative_control_conjugated_pbar_in_publisher_fails_source_check(monkeypatch):
    """Conjugating d_pbar in the MCMC source publisher fails the independent coordinate-derivative verdict."""
    production_publisher = bridge_module._collect_c_param_derivative_sources

    def conjugated_pbar_publisher(*args, **kwargs):
        return tuple(_conjugate_pbar(source) for source in production_publisher(*args, **kwargs))

    monkeypatch.setattr(bridge_module, "_collect_c_param_derivative_sources", conjugated_pbar_publisher)
    faulty = _record("sr")
    with pytest.raises(AssertionError, match="consumed d ln Psi / dq differs from the independent coordinate derivative"):
        _assert_sources_match_independent(faulty, faulty.log_derivatives)

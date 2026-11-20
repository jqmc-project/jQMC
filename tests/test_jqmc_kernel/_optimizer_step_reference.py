"""Shared recorder and independent references for one production optimization step.

Used by ``test_optimizer_mask_and_floor.py`` and
``test_optimizer_complex_twist_update.py``.

* ``record_one_optimize_step`` runs ONE ``MCMC.run_optimize`` step with the real
  sampler and the real optimizer. Passive spies record, without changing any
  value, the per-step walker positions, the batch / coordinate map / target the
  production target builder consumed and returned, and the
  ``finalize_optimization_step`` result.
* ``coordinate_columns`` is the independent evaluation of the per-sample
  directional derivatives ``d ln Psi / dq_k`` (complex logarithm, so the phase
  of a complex Bloch wavefunction is kept) or ``d e_L / dq_k`` along every real
  optimizer coordinate ``q_k``. The wavefunction is displaced along the
  ambient storage direction ``E e_k`` (``E`` = ``coordinate_map.encode``, the
  forward coordinate map) by writing the displaced leaves here, including the
  time-reversal tie ``C_dn = conj(C_up)``, and ``jax.jacfwd`` differentiates the
  production forward model. The production side instead publishes Wirtinger
  components ``(d_p, d_pbar)`` and pulls them back with the real ADJOINT
  ``transform_derivatives``, so a conjugation or sign fault in the published
  components or in the adjoint moves only the production side.
* ``reference_sr_update`` / ``reference_lm_update`` are dense NumPy
  recomputations of the documented SR system and centred linear-method pencil
  for complex coordinate columns; they reduce to the real formulas of
  ``test_optimizer_real_batch_update.py`` when every column is real.
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

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

import jqmc.jqmc_optimizer as optimizer_module
from jqmc._setting import get_optimizer_roundoff_tolerance
from jqmc.determinant import compute_det_geminal_all_elements
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy
from jqmc.jastrow_factor import compute_Jastrow_part
from jqmc.jqmc_mcmc import MCMC
from jqmc.jqmc_optimizer import DeviceOptimizationBatch
from tests.test_jqmc_kernel._exact_sr_reference import exact_centred_moments, exact_sr_solve, ext_diag, to_float64

# Zones crossed by ln Psi (AO -> MO -> geminal determinant, Jastrow) and, for
# the local energy, additionally by the kinetic and potential assembly.
LN_PSI_ZONES = ("ao_eval", "mo_eval", "jastrow_eval", "det_eval", "wf_eval")
LOCAL_ENERGY_ZONES = LN_PSI_ZONES + (
    "ao_grad_lap",
    "mo_grad",
    "mo_lap",
    "jastrow_grad_lap",
    "det_grad_lap",
    "wf_kinetic",
    "coulomb",
    "ewald",
    "local_energy",
)


@dataclass(frozen=True)
class StepCapture:
    """Everything one recorded production optimization step consumed and produced."""

    hamiltonian_before: Hamiltonian_data
    hamiltonian_after: Hamiltonian_data
    batch: object
    coordinate_map: object
    target_kwargs: dict
    target: object
    step: object
    r_up: np.ndarray
    r_dn: np.ndarray
    weights: np.ndarray
    local_energies: np.ndarray
    log_derivatives: np.ndarray
    local_energy_derivatives: np.ndarray | None


def host_coordinates(coordinate_map, batch, sources) -> np.ndarray:
    """Production pull-back of published sources to real-coordinate columns."""
    if isinstance(batch, DeviceOptimizationBatch):
        return np.asarray(coordinate_map.transform_derivatives_device(sources))
    return np.asarray(coordinate_map.transform_derivatives(sources))


def record_one_optimize_step(
    hamiltonian: Hamiltonian_data,
    *,
    optimizer_kwargs: dict,
    selection: dict,
    use_lm: bool,
    seed: int,
    num_walkers: int,
    num_steps: int,
    num_bin_blocks: int,
    use_device_optimizer: bool = False,
) -> StepCapture:
    """Run one production ``run_optimize`` step and record its inputs and outputs."""
    mcmc = MCMC(
        hamiltonian_data=hamiltonian,
        Dt=2.0,
        mcmc_seed=seed,
        num_walkers=num_walkers,
        comput_position_deriv=False,
        comput_log_WF_param_deriv=True,
        comput_e_L_param_deriv=use_lm,
    )
    recorded: dict[str, object] = {}
    positions_up: list[np.ndarray] = []
    positions_dn: list[np.ndarray] = []
    production_target = optimizer_module.compute_real_coordinate_optimization_target
    production_finalize = optimizer_module.finalize_optimization_step
    production_sample = mcmc.sample

    def recording_target(batch, coordinate_map, **kwargs):
        target = production_target(batch, coordinate_map, **kwargs)
        recorded.update(batch=batch, coordinate_map=coordinate_map, target_kwargs=dict(kwargs), target=target)
        return target

    def recording_finalize(target, coordinate_map, **kwargs):
        result = production_finalize(target, coordinate_map, **kwargs)
        recorded["step"] = result
        return result

    def recording_sample(*, num_steps, max_time):
        for _ in range(num_steps):
            production_sample(num_steps=1, max_time=max_time)
            positions_up.append(np.asarray(mcmc._MCMC__latest_r_up_carts))
            positions_dn.append(np.asarray(mcmc._MCMC__latest_r_dn_carts))

    mcmc.sample = recording_sample
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(optimizer_module, "compute_real_coordinate_optimization_target", recording_target)
        patch.setattr(optimizer_module, "finalize_optimization_step", recording_finalize)
        mcmc.run_optimize(
            num_mcmc_steps=num_steps,
            num_opt_steps=1,
            num_mcmc_warmup_steps=0,
            num_mcmc_bin_blocks=num_bin_blocks,
            optimizer_kwargs=dict(optimizer_kwargs),
            dump_wavefunction=False,
            use_device_optimizer=use_device_optimizer,
            **selection,
        )
    batch = recorded["batch"]
    coordinate_map = recorded["coordinate_map"]
    energy_sources = batch.local_energy_derivative_sources
    n_samples = num_steps * num_walkers
    return StepCapture(
        hamiltonian_before=hamiltonian,
        hamiltonian_after=mcmc.hamiltonian_data,
        batch=batch,
        coordinate_map=coordinate_map,
        target_kwargs=recorded["target_kwargs"],
        target=recorded["target"],
        step=recorded["step"],
        r_up=np.stack(positions_up).reshape(n_samples, *positions_up[0].shape[1:]),
        r_dn=np.stack(positions_dn).reshape(n_samples, *positions_dn[0].shape[1:]),
        weights=np.asarray(batch.weights),
        local_energies=np.asarray(batch.local_energies),
        log_derivatives=host_coordinates(coordinate_map, batch, batch.log_derivative_sources),
        local_energy_derivatives=(None if energy_sources is None else host_coordinates(coordinate_map, batch, energy_sources)),
    )


# --------------------------------------------------------------------------- #
# Independent directional derivatives along the optimizer coordinates
# --------------------------------------------------------------------------- #


def encode_matrix(coordinate_map) -> np.ndarray:
    """Row k is the ambient storage direction ``E e_k`` of real coordinate k."""
    return np.asarray(coordinate_map.encode(np.eye(coordinate_map.coordinate_size)))


def block_directions(coordinate_map) -> list:
    """Per-block slice of ``E`` in the block's field (a real block's directions must be real)."""
    directions = encode_matrix(coordinate_map)
    per_block = []
    for block, storage_slice in zip(coordinate_map.blocks, coordinate_map.block_storage_slices, strict=True):
        part = directions[:, storage_slice]
        if block.field == "real":
            assert np.array_equal(np.imag(part), np.zeros(part.shape)), f"real block {block.name!r} has complex directions"
            part = np.real(part)
        per_block.append(jnp.asarray(part))
    return per_block


def displace_wavefunction(wavefunction, coordinate_map, block_deltas):
    """Add each block's storage delta to the leaves it names (written here, not the production updater)."""
    jastrow = wavefunction.jastrow_data
    geminal = wavefunction.geminal_data
    for block, block_delta in zip(coordinate_map.blocks, block_deltas, strict=True):
        delta = block_delta.reshape(block.shape)
        if block.name == "j1_param":
            j1 = jastrow.jastrow_one_body_data
            jastrow = jastrow.replace(jastrow_one_body_data=j1.replace(jastrow_1b_param=j1.jastrow_1b_param + delta))
        elif block.name == "j2_param":
            j2 = jastrow.jastrow_two_body_data
            jastrow = jastrow.replace(jastrow_two_body_data=j2.replace(jastrow_2b_param=j2.jastrow_2b_param + delta))
        elif block.name == "j3_matrix":
            j3 = jastrow.jastrow_three_body_data
            jastrow = jastrow.replace(jastrow_three_body_data=j3.replace(j_matrix=j3.j_matrix + delta))
        elif block.name == "lambda_matrix":
            geminal = geminal.replace(lambda_matrix=geminal.lambda_matrix + delta)
        elif block.name == "mo_coefficients_up":
            # Time-reversal tie: the down-spin factor is conj(C_up) (Convention B).
            assert geminal.time_reversal, "this reference writes only the time-reversal-tied MO block"
            up = geminal.orb_data_up_spin
            dn = geminal.orb_data_dn_spin
            num_dn_rows = dn.mo_coefficients.shape[0]
            geminal = geminal.replace(
                orb_data_up_spin=up.replace(mo_coefficients=up.mo_coefficients + delta),
                orb_data_dn_spin=dn.replace(mo_coefficients=dn.mo_coefficients + jnp.conj(delta)[:num_dn_rows]),
            )
        else:
            raise AssertionError(f"no independent displacement written for block {block.name!r}")
    return wavefunction.replace(jastrow_data=jastrow, geminal_data=geminal)


def _complex_ln_psi(wavefunction, r_up, r_dn):
    """Phase-preserving ln Psi = J + log(det) (complex logarithm)."""
    jastrow = compute_Jastrow_part(jastrow_data=wavefunction.jastrow_data, r_up_carts=r_up, r_dn_carts=r_dn)
    determinant = compute_det_geminal_all_elements(geminal_data=wavefunction.geminal_data, r_up_carts=r_up, r_dn_carts=r_dn)
    return jastrow + jnp.log(determinant.astype(jnp.complex128))


def coordinate_columns(hamiltonian: Hamiltonian_data, coordinate_map, r_up, r_dn, quantity: str) -> np.ndarray:
    """Per-sample ``d ln Psi / dq`` (``quantity='ln_psi'``) or ``d e_L / dq`` (``'local_energy'``)."""
    directions = block_directions(coordinate_map)
    size = coordinate_map.coordinate_size
    rotation = jnp.eye(3)

    def displaced(q, sample_up, sample_dn):
        deltas = [q.astype(part.dtype) @ part for part in directions]
        wavefunction = displace_wavefunction(hamiltonian.wavefunction_data, coordinate_map, deltas)
        if quantity == "ln_psi":
            return _complex_ln_psi(wavefunction, sample_up, sample_dn)
        assert quantity == "local_energy"
        value = compute_local_energy(hamiltonian.replace(wavefunction_data=wavefunction), sample_up, sample_dn, rotation)
        return jnp.asarray(value, dtype=jnp.complex128)

    # Reverse mode on the real and imaginary parts separately (the local-energy
    # chain contains custom_vjp kernels, which admit no forward-mode rule).
    def columns(q, sample_up, sample_dn):
        real = jax.jacrev(lambda x: jnp.real(displaced(x, sample_up, sample_dn)))(q)
        imag = jax.jacrev(lambda x: jnp.imag(displaced(x, sample_up, sample_dn)))(q)
        return real + 1.0j * imag

    batched = jax.jit(jax.vmap(columns, in_axes=(None, 0, 0)))
    return np.asarray(batched(jnp.zeros(size, dtype=jnp.float64), jnp.asarray(r_up), jnp.asarray(r_dn)))


def independent_local_energies(hamiltonian: Hamiltonian_data, r_up, r_dn) -> np.ndarray:
    rotation = jnp.eye(3)
    batched = jax.jit(jax.vmap(lambda a, b: compute_local_energy(hamiltonian, a, b, rotation), in_axes=(0, 0)))
    return np.asarray(batched(jnp.asarray(r_up), jnp.asarray(r_dn)))


# --------------------------------------------------------------------------- #
# Dense NumPy references on (possibly complex) coordinate columns
# --------------------------------------------------------------------------- #


def reference_sr_update(log_derivatives, local_energies, weights, epsilon, diagonal_floor=None):
    """Solve ``(S + epsilon diag S) q = f`` on the coordinates whose ``diag S`` exceeds the floor.

    The moments and the solve are exact (``_exact_sr_reference``): an fp64
    reference would carry its own ``cond * u * |q|`` (~1e-12 at
    ``epsilon = 1e-3``) forward error, the size of the comparison bound.
    The floor mask is decided on the fp64-rounded diagonal, as in production.

    Returns ``(q, diag S, active mask)``; frozen coordinates carry ``q = 0``.
    """
    S, f = exact_centred_moments(log_derivatives, local_energies, weights)
    diagonal = to_float64(ext_diag(S))
    active = np.ones(diagonal.size, dtype=bool) if diagonal_floor is None else diagonal > diagonal_floor
    q = np.zeros(diagonal.size)
    index = np.flatnonzero(active)
    q[index] = exact_sr_solve(S[np.ix_(index, index)], f[index], epsilon)
    return q, diagonal, active


def pivoted_cholesky_correlation_survivors(metric, cutoff):
    """Unblocked NumPy pivoted Cholesky of the shifted correlation matrix ``C - cutoff I``.

    Independent of LAPACK ``?pstrf``: a right-looking textbook loop that, at
    each step ``j``, takes the largest residual diagonal over the not yet
    pivoted positions (``np.argmax``: first position among exact ties),
    swaps it into position ``j`` (the LAPACK pivot-order convention) and
    applies the rank-one Schur update.  The
    first pivot is accepted when positive; every later pivot must exceed the
    documented round-off stop ``n * eps`` (``lm_correlation_pivot``).
    Returns the sorted original indices of the accepted pivots: the
    coordinates whose correlation block satisfies ``C_A - cutoff I > 0``.
    """
    metric = np.asarray(metric, dtype=np.float64)
    metric = 0.5 * (metric + metric.T)
    size = metric.shape[0]
    scale = 1.0 / np.sqrt(np.diag(metric))
    residual = scale[:, None] * metric * scale[None, :]
    residual = 0.5 * (residual + residual.T)
    np.fill_diagonal(residual, 1.0)
    residual = residual - cutoff * np.eye(size)
    stop = get_optimizer_roundoff_tolerance("lm_correlation_pivot", np.float64, scale=size)
    order = np.arange(size)
    for step in range(size):
        position = step + int(np.argmax(np.diag(residual)[step:]))
        pivot = residual[position, position]
        if (step == 0 and not pivot > 0.0) or (step > 0 and not pivot > stop):
            return np.sort(order[:step])
        residual[[step, position], :] = residual[[position, step], :]
        residual[:, [step, position]] = residual[:, [position, step]]
        order[[step, position]] = order[[position, step]]
        column = residual[step + 1 :, step] / np.sqrt(pivot)
        residual[step + 1 :, step + 1 :] -= np.outer(column, column)
    return np.sort(order)


def reference_lm_update(log_derivatives, energy_derivatives, local_energies, weights, condition_cutoff):
    """Dense generalized eigensolve of the real projection of the centred non-symmetric complex pencil.

    Estimators (each element by its own estimator, no Hermitian averaging):
    ``H_00 = Re<e>``, ket column ``H_a0 = <dO_a* de>``, bra row
    ``H_0a = <dO_a de> + <D_a>``, ``H_ab = <e dO_a* dO_b> + <dD_b* dO_a>^*``
    (``K + B^dagger`` with ``B_ab = <dD_a* dO_b>``), overlap ``S_ab = <dO_a* dO_b>``;
    the real coordinate pencil is the real part of both. Coordinates are
    first removed by the correlation-conditioning cutoff on ``Re S``
    (``pivoted_cholesky_correlation_survivors``).
    Returns ``(q, predicted energy, reference weight, surviving coordinates)``.
    """
    O = np.asarray(log_derivatives, dtype=np.complex128)
    D = np.asarray(energy_derivatives, dtype=np.complex128)
    e = np.asarray(local_energies, dtype=np.complex128)
    w = np.asarray(weights, dtype=np.float64)
    total = np.sum(w)
    e_bar = (w @ e) / total
    D_bar = (w @ D) / total
    dO = O - (w @ O) / total
    dD = D - D_bar
    de = e - e_bar
    S = np.einsum("n,na,nb->ab", w, np.conj(dO), dO) / total
    ket = np.einsum("n,na,n->a", w, np.conj(dO), de) / total
    bra = np.einsum("n,na,n->a", w, dO, de) / total + D_bar
    K = np.einsum("n,n,na,nb->ab", w, e, np.conj(dO), dO) / total
    B = np.einsum("n,na,nb->ab", w, np.conj(dD), dO) / total
    size = O.shape[1]
    H = np.zeros((size + 1, size + 1), dtype=np.complex128)
    metric = np.zeros_like(H)
    H[0, 0] = np.real(e_bar)
    H[1:, 0] = ket
    H[0, 1:] = bra
    H[1:, 1:] = K + B.conj().T
    metric[0, 0] = 1.0
    metric[1:, 1:] = S
    H = np.real(H)
    metric = np.real(metric)
    metric = 0.5 * (metric + metric.T)
    survivors = pivoted_cholesky_correlation_survivors(metric[1:, 1:], condition_cutoff)
    augmented = np.concatenate(([0], survivors + 1))
    eigenvalues, vectors = scipy.linalg.eig(H[np.ix_(augmented, augmented)], metric[np.ix_(augmented, augmented)])
    vectors = np.real(vectors)
    sub_metric = metric[np.ix_(augmented, augmented)]
    vectors = vectors / np.sqrt(np.einsum("ij,ik,kj->j", vectors, sub_metric, vectors))[None, :]
    weights_ref = np.where(np.imag(eigenvalues) == 0.0, vectors[0] ** 2, -np.inf)
    best = int(np.argmax(weights_ref))
    q = np.zeros(size)
    q[survivors] = vectors[1:, best] / vectors[0, best]
    return q, float(np.real(eigenvalues[best])), float(weights_ref[best]), survivors

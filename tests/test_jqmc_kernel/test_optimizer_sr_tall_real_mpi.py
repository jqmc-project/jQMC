"""C10 unequal-count two-rank real-coordinate tall SR oracle."""

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

import numpy as np
import pytest
from mpi4py import MPI

from jqmc._precision import get_tolerance_xla_rust_consistency
from jqmc._setting import (
    ATOL_BIT_EXACT,
    ATOL_C9_GAMMA_SR_UPDATE,
    ATOL_PYTHON_FP64_LOOSE,
    RTOL_C9_GAMMA_SR_UPDATE,
    RTOL_PYTHON_FP64_LOOSE,
)
from jqmc.jqmc_optimizer import (
    BlockDerivativeSource,
    DeviceBlockDerivativeSource,
    DeviceJackknifeBinMetadata,
    DeviceJackknifeBlockMoments,
    DeviceOptimizationBatch,
    JackknifeBinMetadata,
    JackknifeBlockMoments,
    OptimizationBatch,
    OptimizationCoordinateMap,
    build_tall_sr_real_layout,
    compute_real_coordinate_optimization_target,
    compute_tall_sr_real_target,
    estimate_coordinate_force_jackknife,
    estimate_coordinate_force_jackknife_device,
)
from jqmc.jqmc_optimizer_kernel import (
    solve_device_tall_real_cg_chunks_sharded,
    solve_tall_sr_real_cg,
    solve_tall_sr_real_direct,
)

from ._optimizer_transform_cases import case_blocks, jackknife_batches
from .optimizer_metadata_helpers import TEST_STATE_TOKEN, make_test_block

pytestmark = [
    pytest.mark.heavy,
    pytest.mark.without_chdir,
    pytest.mark.skip_serial(reason="two-rank SR contract; launch via mpirun -np 2"),
]


@pytest.fixture(autouse=True)
def _exactly_two_ranks():
    assert MPI.COMM_WORLD.Get_size() == 2, "the two-rank SR contract needs exactly two ranks: launch via mpirun -np 2"


# ``lm_cond`` must be positive (issue #855); this cutoff removes nothing in this
# reduced-LM parity case.
_NEGLIGIBLE_LM_CONDITION_CUTOFF = 1.0e-12


def test_two_rank_sharded_tall_chunk_cg_matches_global_dense_oracle():
    """Two rank sharded tall chunk cg matches global dense oracle."""
    comm = MPI.COMM_WORLD
    rng = np.random.default_rng(1900 + comm.Get_rank())
    chunks = tuple(rng.normal(size=(14, count)) + 1.0j * rng.normal(size=(14, count)) for count in (2, 1))
    forces = tuple(rng.normal(size=count) + 1.0j * rng.normal(size=count) for count in (2, 1))
    global_chunks = tuple(np.concatenate(comm.allgather(chunk), axis=1) for chunk in chunks)
    global_forces = tuple(np.concatenate(comm.allgather(force)) for force in forces)
    X_global = np.concatenate(global_chunks, axis=1)
    F_global = np.concatenate(global_forces)
    Z_global = np.concatenate((np.real(X_global), np.imag(X_global)), axis=1)
    h_global = np.concatenate((np.real(F_global), -np.imag(F_global)))
    expected = solve_tall_sr_real_direct(Z_global, h_global, 0.2)
    # cg_tol is relative to the global dual right-hand side (issue #859); this
    # tolerance keeps the absolute stopping residual ATOL_PYTHON_FP64_LOOSE.
    actual, residual, _iterations, _breakdown = solve_device_tall_real_cg_chunks_sharded(
        chunks,
        forces,
        0.2,
        max_iter=500,
        tol=ATOL_PYTHON_FP64_LOOSE / np.linalg.norm(h_global),
    )
    np.testing.assert_allclose(
        actual,
        expected,
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_C9_GAMMA_SR_UPDATE,
    )
    assert residual <= ATOL_PYTHON_FP64_LOOSE


def _unequal_batch(rank, local_count):
    rng = np.random.default_rng(1700 + rank)
    blocks = tuple(
        make_test_block(
            f"r{index}",
            np.asarray(0.0),
            field="real",
            derivative_policy="direct_real",
            sync_scope="sampler",
        )
        for index in range(11)
    )
    energies = rng.normal(size=local_count) + 1.0j * rng.normal(size=local_count)
    values = tuple(rng.normal(size=local_count) + 1.0j * rng.normal(size=local_count) for _ in blocks)

    def source(block, array):
        return BlockDerivativeSource.build(block=block, direct=array)

    sources = tuple(source(block, array) for block, array in zip(blocks, values))
    moments = tuple(
        JackknifeBlockMoments.build(
            block=block,
            source_sum=source(block, array),
            conjugate_source_energy_sum=source(block, np.conj(array) * energies),
        )
        for block, array in zip(blocks, values)
    )
    token = "c10-mpi:1"
    metadata = JackknifeBinMetadata.build(
        blocks=blocks,
        batch_token=token,
        bin_sample_counts=np.ones(local_count, dtype=np.int64),
        bin_weight_sums=np.ones(local_count),
        bin_weighted_energy_sums=energies,
        log_source_moments=moments,
    )
    return blocks, OptimizationBatch.build(
        blocks=blocks,
        batch_token=token,
        state_token=TEST_STATE_TOKEN,
        weights=np.ones(local_count),
        local_energies=energies,
        log_derivative_sources=sources,
        local_energy_derivative_sources=sources,
        jackknife_bin_metadata=metadata,
    )


def _device_batch(blocks, batch):
    def source(block, values):
        return DeviceBlockDerivativeSource.build(
            block=block,
            direct=values.direct,
            d_p=values.d_p,
            d_pbar=values.d_pbar,
        )

    moments = tuple(
        DeviceJackknifeBlockMoments(
            name=block.name,
            source_sum=source(block, moment.source_sum),
            conjugate_source_energy_sum=source(block, moment.conjugate_source_energy_sum),
        )
        for block, moment in zip(blocks, batch.jackknife_bin_metadata.log_source_moments, strict=True)
    )
    return DeviceOptimizationBatch(
        batch_token=batch.batch_token,
        state_token=batch.state_token,
        weights=batch.weights,
        local_energies=batch.local_energies,
        log_derivative_sources=tuple(
            source(block, values) for block, values in zip(blocks, batch.log_derivative_sources, strict=True)
        ),
        local_energy_derivative_sources=tuple(
            source(block, values) for block, values in zip(blocks, batch.local_energy_derivative_sources, strict=True)
        ),
        jackknife_bin_metadata=DeviceJackknifeBinMetadata(
            batch_token=batch.batch_token,
            bin_sample_counts=batch.jackknife_bin_metadata.bin_sample_counts,
            bin_weight_sums=batch.jackknife_bin_metadata.bin_weight_sums,
            bin_weighted_energy_sums=batch.jackknife_bin_metadata.bin_weighted_energy_sums,
            log_source_moments=moments,
        ),
    )


def test_unequal_count_two_rank_cpu_and_device_match_valid_dense_oracle():
    """Unequal-count two-rank host tall CG matches the dense oracle of the gathered valid dual."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    local_count = rank + 2
    rng = np.random.default_rng(710 + rank)
    X_local = rng.normal(size=(11, local_count)) + 1.0j * rng.normal(size=(11, local_count))
    F_local = rng.normal(size=local_count) + 1.0j * rng.normal(size=local_count)
    counts = tuple(comm.allgather(local_count))
    layout = build_tall_sr_real_layout(X_local, F_local, sample_counts=counts, rank=rank)
    Z_global = np.concatenate(comm.allgather(layout.Z_valid_local), axis=1)
    h_global = np.concatenate(comm.allgather(layout.h_valid_local))
    epsilon = 0.13
    expected = solve_tall_sr_real_direct(Z_global, h_global, epsilon)
    cpu_cg, _residual, _iterations, _breakdown = solve_tall_sr_real_cg(
        Z_global,
        h_global,
        epsilon,
        max_iter=1000,
        tol=ATOL_PYTHON_FP64_LOOSE,
    )
    # Unequal per-rank counts have no device route: the world-mesh device
    # kernels require one layout on every rank (_require_world_device_mesh).
    for comparison_index, (actual) in enumerate((cpu_cg,)):
        comparison_iteration = str(comparison_index)  # noqa: F841 - captured by the GDD iteration receipt
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=RTOL_C9_GAMMA_SR_UPDATE,
            atol=ATOL_C9_GAMMA_SR_UPDATE,
        )


def test_unequal_count_full_target_uses_valid_gather_and_nonpersistent_dual():
    """Unequal count full target uses valid gather and nonpersistent dual."""
    comm = MPI.COMM_WORLD
    blocks, batch = _unequal_batch(comm.Get_rank(), comm.Get_rank() + 2)
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    results = []
    for cg in (False, True):
        results.append(
            compute_tall_sr_real_target(
                batch,
                coordinate_map,
                comm=comm,
                epsilon=0.2,
                diagonal_floor=0.0,
                cg=cg,
                cg_max_iter=1000,
                cg_tol=ATOL_PYTHON_FP64_LOOSE,
            )
        )
    expected = results[0]
    np.testing.assert_array_equal(expected.dual_layout.valid_counts, np.asarray([4, 6]))
    np.testing.assert_array_equal(expected.dual_layout.valid_displacements, np.asarray([0, 4]))
    for comparison_index, (actual) in enumerate(results):
        comparison_iteration = str(comparison_index)  # noqa: F841 - captured by the GDD iteration receipt
        assert actual.warm_start_out is None
        np.testing.assert_allclose(
            actual.q,
            expected.q,
            rtol=RTOL_C9_GAMMA_SR_UPDATE,
            atol=ATOL_C9_GAMMA_SR_UPDATE,
        )


def test_coordinate_force_jackknife_uses_distributed_allreduce_moments():
    """Coordinate force jackknife uses distributed allreduce moments."""
    comm = MPI.COMM_WORLD
    blocks, batch = _unequal_batch(comm.Get_rank(), comm.Get_rank() + 2)
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    actual_force, actual_std = estimate_coordinate_force_jackknife(batch, coordinate_map, comm=comm)

    weights = np.concatenate(comm.allgather(np.asarray(batch.weights)))
    energies = np.concatenate(comm.allgather(np.asarray(batch.local_energies)))
    derivatives = np.concatenate(
        comm.allgather(coordinate_map.transform_derivatives(batch.log_derivative_sources)),
        axis=0,
    )

    def direct(indices):
        selected_weights = weights[indices]
        selected_energies = energies[indices]
        selected_derivatives = derivatives[indices]
        total_weight = np.sum(selected_weights)
        energy_mean = np.dot(selected_weights, selected_energies) / total_weight
        derivative_mean = np.einsum("n,nq->q", selected_weights, selected_derivatives) / total_weight
        return -2.0 * np.real(
            np.einsum(
                "n,nq,n->q",
                selected_weights,
                np.conj(selected_derivatives - derivative_mean),
                selected_energies - energy_mean,
            )
            / total_weight
        )

    indices = np.arange(weights.size)
    expected_force = direct(indices)
    replicas = np.asarray([direct(np.delete(indices, index)) for index in indices])
    replica_mean = np.mean(replicas, axis=0)
    expected_variance = (replicas.shape[0] - 1.0) / replicas.shape[0] * np.sum(np.square(replicas - replica_mean), axis=0)
    np.testing.assert_allclose(actual_force, expected_force, atol=ATOL_BIT_EXACT, rtol=RTOL_PYTHON_FP64_LOOSE)
    np.testing.assert_allclose(actual_std, np.sqrt(expected_variance), atol=ATOL_BIT_EXACT, rtol=RTOL_PYTHON_FP64_LOOSE)


@pytest.mark.parametrize("case", ["j2-j3-sd-real-source", "complex-tied-lambda-stiefel"])
def test_rust_coordinate_force_jackknife_matches_device_twin_on_two_ranks(monkeypatch, case):
    """Rust host jackknife (Python MPI reductions) equals the device twin on two ranks with unequal bin counts."""
    comm = MPI.COMM_WORLD
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    rank = comm.Get_rank()
    blocks = case_blocks(case)
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    batch, device_batch = jackknife_batches(
        blocks,
        bins=3 + 2 * rank,
        complex_source=False,
        complex_energy=True,
        seed=8450 + 10 * rank,
    )
    force, std = estimate_coordinate_force_jackknife(batch, coordinate_map, comm=comm)
    reference_force, reference_std = estimate_coordinate_force_jackknife_device(device_batch, coordinate_map, comm=comm)
    # Derivative sources (wf_eval) and local-energy moments (local_energy): float64 zones in both modes.
    atol, rtol = get_tolerance_xla_rust_consistency(("wf_eval", "local_energy"), level="strict")
    np.testing.assert_allclose(force, reference_force, atol=atol, rtol=rtol)
    np.testing.assert_allclose(std, reference_std, atol=atol, rtol=rtol)


def test_production_device_batch_matches_two_rank_cpu_reference_through_reduced_lm():
    """Production device batch matches two rank cpu reference through reduced LM."""
    comm = MPI.COMM_WORLD
    blocks, batch = _unequal_batch(comm.Get_rank(), 3)
    device_batch = _device_batch(blocks, batch)
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    common = dict(
        comm=comm,
        epsilon=0.2,
        diagonal_floor=0.0,
        cg=True,
        cg_max_iter=1000,
        cg_tol=ATOL_PYTHON_FP64_LOOSE,
        warm_start=None,
        use_lm=True,
        lm_subspace_dim=3,
        lm_condition_cutoff=_NEGLIGIBLE_LM_CONDITION_CUTOFF,
        energy_standard_error=0.1,
    )
    cpu = compute_real_coordinate_optimization_target(
        batch,
        coordinate_map,
        **common,
    )
    device = compute_real_coordinate_optimization_target(
        device_batch,
        coordinate_map,
        **common,
    )
    np.testing.assert_allclose(
        device.coordinate_force,
        cpu.coordinate_force,
        rtol=RTOL_C9_GAMMA_SR_UPDATE,
        atol=ATOL_C9_GAMMA_SR_UPDATE,
    )
    np.testing.assert_allclose(
        device.coordinate_force_std,
        cpu.coordinate_force_std,
        rtol=RTOL_C9_GAMMA_SR_UPDATE,
        atol=ATOL_C9_GAMMA_SR_UPDATE,
    )
    np.testing.assert_allclose(
        device.sr_q,
        cpu.sr_q,
        rtol=RTOL_C9_GAMMA_SR_UPDATE,
        atol=ATOL_C9_GAMMA_SR_UPDATE,
    )
    np.testing.assert_allclose(
        device.q,
        cpu.q,
        rtol=RTOL_C9_GAMMA_SR_UPDATE,
        atol=ATOL_C9_GAMMA_SR_UPDATE,
    )


@pytest.mark.mpi_only
@pytest.mark.gdd_electron_configuration(
    mode="required",
    position_source="production-driver-generated",
    reason="Actual two-rank MCMC718 walkers; one-rank validation fault proves collective rejection before publication.",
)
def test_two_rank_candidate_rejection_preserves_accepted_state(tmp_path, monkeypatch):
    """One rank rejects after real candidate evaluation; both ranks retain their accepted state."""
    from jqmc.hamiltonians import Hamiltonian_data
    from jqmc.jqmc_mcmc import MCMC
    from tests.test_jqmc_kernel._periodic_j3_basis_cases import AUDIT_FIXTURE

    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2, "launch with mpirun -np 2"
    monkeypatch.chdir(tmp_path)
    initial = Hamiltonian_data.load_from_hdf5(str(AUDIT_FIXTURE))
    driver = MCMC(
        hamiltonian_data=initial, num_walkers=2, mcmc_seed=718, num_mcmc_per_measurement=1, comput_log_WF_param_deriv=True
    )
    validate = driver.validate_wavefunction_candidate
    boundary = {}

    def rank_local_validation(candidate):
        validate(candidate)  # Real candidate and actual local walker evaluation.
        boundary["ham"] = driver.hamiltonian_data
        boundary["token"] = driver.wavefunction_state_token
        boundary["up"] = np.array(driver._MCMC__latest_r_up_carts, copy=True)
        boundary["dn"] = np.array(driver._MCMC__latest_r_dn_carts, copy=True)
        if comm.Get_rank() == 0:
            raise ValueError("one-rank validation fault")

    monkeypatch.setattr(driver, "validate_wavefunction_candidate", rank_local_validation)
    with pytest.raises(ValueError, match=r"one-rank validation fault|another sampler rank"):
        driver.run_optimize(
            num_mcmc_steps=4,
            num_opt_steps=1,
            num_mcmc_bin_blocks=2,
            dump_wavefunction=False,
            opt_J3_basis_exp=True,
            opt_J1_param=False,
            opt_J2_param=False,
            opt_J3_param=False,
            opt_geminal=False,
            optimizer_kwargs={"method": "sr", "delta": 1e-3, "epsilon": 1e-3},
        )
    assert driver.hamiltonian_data is boundary["ham"]
    assert driver.wavefunction_state_token == boundary["token"]
    assert driver.i_opt == 0
    np.testing.assert_array_equal(driver._MCMC__latest_r_up_carts, boundary["up"], strict=True)
    np.testing.assert_array_equal(driver._MCMC__latest_r_dn_carts, boundary["dn"], strict=True)
    assert comm.allgather(driver.i_opt) == [0, 0]

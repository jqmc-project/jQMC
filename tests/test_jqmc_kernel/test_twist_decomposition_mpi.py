"""Twist-decomposition identity tests ("sum check") under MPI.

The BZ-mean machinery must reduce EXACTLY to "independent per-twist chains
+ weighted arithmetic": the per-twist warmup /
binning / jackknife happen inside each twist's ``get_*`` BEFORE the Level-2
combine, and the combine itself is the pure quadrature form of
``bz_mean_energy`` / ``bz_mean_force``. These tests pin that decomposition
for the full estimator quartet (``get_E`` / ``get_aF`` / ``get_aT`` /
``get_gF``) on the MCMC driver and for (``get_E`` / ``get_aF`` /
``get_aT``) on the GFMC driver, by running the SAME seeds twice:

1. standalone: one direct ``MCMC`` / ``GFMC_t`` / ``GFMC_n`` chain per twist on the
   intra communicator that :func:`split_world_for_twists` would assign;
2. driver: ``run_twist_average_from_hamiltonians`` /
   :func:`run_twist_average_gfmc` over both twists.

Identical seeds + identical comms -> identical samples, so every per-twist
estimator must match exactly and the BZ combine must equal the hand-rolled
weighted sum (non-uniform weights on purpose -- a dropped ``W_k`` or a
pooled-sample bin would show up immediately).

Launching convention (same as ``test_integration_qe_twist_average_mpi``):

    mpirun -np 2 pytest -q tests/test_jqmc_kernel/test_twist_decomposition_mpi.py

``world_size`` must be a multiple of ``n_twist = 2``; a serial pytest run
skips (the comm split requires divisibility by design). The two twists are
the committed ``Li_gamma`` (real path) + ``Li_k025`` (complex Bloch path)
fixtures converted in-process, so the identity also crosses the real /
complex dtype boundary.

A twist-averaged translational force sum rule rides on the same run: each
twist's SWCT force sums to zero per sample, so the BZ-mean force must too
-- any weight / transpose / real-projection bug in the combine breaks it.
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

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mpi4py import MPI

from jqmc._precision import get_tolerance_min
from jqmc._setting import ATOL_PYTHON_FP64_BZ_MEAN, ATOL_PYTHON_FP64_LOOSE, RTOL_BIT_EXACT
from jqmc._setting import RTOL_PYTHON_FP64_ZERO
from tests._test_data_cache import qe_example_files_root
from tests.conftest import check_sum_rule_force

_FIX = qe_example_files_root()
_SAVES = {
    0: _FIX / "Li_gamma" / "out" / "li.save",
    1: _FIX / "Li_k025" / "out" / "li_k025.save",
}
_available = all((s / "data-file-schema.xml").is_file() and (s / "wfc1.hdf5").is_file() for s in _SAVES.values())
requires_li_fixtures = pytest.mark.skipif(not _available, reason="Li_gamma / Li_k025 QE fixtures not present")

_world = MPI.COMM_WORLD
_world_size = _world.Get_size()
_world_rank = _world.Get_rank()

_N_TWIST = 2
requires_mpi_world = pytest.mark.skip_serial(reason="twist decomposition over MPI ranks; launch via mpirun -np 2 pytest.")


@pytest.fixture(autouse=True)
def _world_divides_into_twists():
    if _world_size > 1:
        assert _world_size % _N_TWIST == 0, f"world_size={_world_size} must be a multiple of n_twist={_N_TWIST}"


# Non-uniform on purpose: a uniform-weight bug (e.g. mean instead of
# weighted mean) would be invisible at W = (0.5, 0.5).
_RAW_WEIGHTS = [0.75, 0.25]

_SEED = 12345
_NUM_WALKERS = 2
_NUM_STEPS = 30
_NUM_WARMUP = 10
_NUM_BINS = 5


class _TwistRootSumSpy:
    """The real twist-roots comm, recording every Level-2 SUM operand and result."""

    def __init__(self, comm):
        self._comm = comm
        self.records = []

    def __getattr__(self, name):
        return getattr(self._comm, name)

    def allreduce(self, value, op=MPI.SUM):
        result = self._comm.allreduce(value, op=op)
        self.records.append((np.array(value, copy=True), np.array(result, copy=True)))
        return result


def _spy_twist_root_sums(monkeypatch):
    """Wrap the real ``split_world_for_twists`` so a twist root's inter comm records its SUMs.

    Returns one ``(intra_rank, spy)`` entry per split: twist roots (intra rank 0)
    get a recording spy; non-roots keep ``inter_comm is None`` and record ``None``.
    """
    import dataclasses

    import jqmc._twist_average as twist_average

    real_split = twist_average.split_world_for_twists
    splits = []

    def split(world_comm, n_twist):
        result = real_split(world_comm, n_twist)
        intra_rank = result.intra_comm.Get_rank()
        if result.inter_comm is None:
            splits.append((intra_rank, None))
            return result
        spy = _TwistRootSumSpy(result.inter_comm)
        splits.append((intra_rank, spy))
        return dataclasses.replace(result, inter_comm=spy)

    monkeypatch.setattr(twist_average, "split_world_for_twists", split)
    return splits


def _assert_no_twist_reported_stopped(splits):
    """Twist roots send and receive an all-zero ``per_twist_stopped`` SUM; non-roots join no SUM."""
    assert len(splits) == 1, "the driver splits the world exactly once"
    intra_rank, spy = splits[0]
    if intra_rank != 0:
        # A non-root rank of its twist (world_size > n_twist): no twist-roots comm.
        assert spy is None, f"intra rank {intra_rank} must not hold a twist-roots communicator"
        return
    assert spy is not None, "a twist root (intra rank 0) must hold the twist-roots communicator"
    stopped = [(sent, got) for sent, got in spy.records if sent.dtype.kind == "i"]
    assert len(stopped) == 1, "exactly one integer Level-2 SUM: per_twist_stopped"
    sent, got = stopped[0]
    np.testing.assert_array_equal(sent, np.zeros(_N_TWIST, dtype=np.int64))
    np.testing.assert_array_equal(got, np.zeros(_N_TWIST, dtype=np.int64))


@pytest.mark.light
@requires_mpi_world
def test_mpc_bz_aggregate_uses_real_twist_root_mpi_collectives(monkeypatch):
    """MPC Level-1 summaries survive the real inter/intra MPI topology."""
    import jqmc.jqmc_mcmc as mcmc_module
    from jqmc._twist_average import bz_mean_energy, run_twist_average_from_hamiltonians

    per_twist = {
        "E": np.array([-1.0, -2.0]),
        "E_std": np.array([0.4, 0.6]),
        "dE": np.array([0.2, -0.1]),
        "dE_std": np.array([0.3, 0.2]),
        "E_mpc": np.array([-0.8, -2.1]),
        "E_mpc_std": np.array([0.1, 0.25]),
    }

    def make_hamiltonian(kx):
        aos = SimpleNamespace(k_point=(kx, 0.0, 0.0))
        orbital = SimpleNamespace(aos_data=aos)
        geminal = SimpleNamespace(orb_data_up_spin=orbital)
        return SimpleNamespace(wavefunction_data=SimpleNamespace(geminal_data=geminal))

    class Sampler:
        def __init__(self, hamiltonian_data, **kwargs):
            self.index = int(hamiltonian_data.wavefunction_data.geminal_data.orb_data_up_spin.aos_data.k_point[0] > 0)

        def run(self, num_mcmc_steps, **kwargs):
            # A completed MCMC run: the driver reads ``MCMC.mcmc_counter``.
            self.mcmc_counter = num_mcmc_steps

        def get_E(self, **kwargs):
            return per_twist["E"][self.index], per_twist["E_std"][self.index], 0.0, 0.0

        def get_dE_mpc(self, **kwargs):
            return per_twist["dE"][self.index], per_twist["dE_std"][self.index]

        def get_E_mpc(self, **kwargs):
            return per_twist["E_mpc"][self.index], per_twist["E_mpc_std"][self.index]

    monkeypatch.setattr(mcmc_module, "MCMC", Sampler)
    splits = _spy_twist_root_sums(monkeypatch)
    result = run_twist_average_from_hamiltonians(
        hamiltonian_list=[make_hamiltonian(0.0), make_hamiltonian(0.25)],
        weights=_RAW_WEIGHTS,
        num_mcmc_steps=4,
        num_walkers=1,
        num_mcmc_warmup_steps=1,
        num_mcmc_bin_blocks=1,
        world_comm=_world,
        mcmc_kwargs={"mpc": True, "mpc_cutoff": 42.0},
    )

    expected_correction = bz_mean_energy(per_twist["dE"], per_twist["dE_std"], _RAW_WEIGHTS)
    expected_energy = bz_mean_energy(per_twist["E_mpc"], per_twist["E_mpc_std"], _RAW_WEIGHTS)
    assert result.mpc is not None
    np.testing.assert_array_equal(result.mpc.per_twist_correction, per_twist["dE"])
    np.testing.assert_array_equal(result.mpc.per_twist_energy, per_twist["E_mpc"])
    np.testing.assert_allclose(
        result.mpc.correction.E_TA, expected_correction.E_TA, atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO
    )
    np.testing.assert_allclose(
        result.mpc.correction.E_TA_std, expected_correction.E_TA_std, atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO
    )
    np.testing.assert_allclose(
        result.mpc.energy.E_TA, expected_energy.E_TA, atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO
    )
    np.testing.assert_allclose(
        result.mpc.energy.E_TA_std, expected_energy.E_TA_std, atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO
    )
    assert _world.allgather(result.mpc.energy.E_TA) == [expected_energy.E_TA] * _world_size
    _assert_no_twist_reported_stopped(splits)


@pytest.mark.light
@requires_mpi_world
def test_optimization_step_energy_is_broadcast_after_twist_root_gather():
    """Pin the real MPI gather -> inter bcast -> intra bcast topology."""
    from jqmc._twist_average import (
        _aggregate_twist_optimization_energy,
        split_world_for_twists,
    )
    from jqmc.jqmc_optimizer import EnergyEstimate

    split = split_world_for_twists(_world, n_twist=_N_TWIST)
    local = EnergyEstimate(-1.0, 0.2, 2.0, 0.3) if split.twist_index == 0 else EnergyEstimate(-3.0, 0.4, 6.0, 0.5)
    actual = _aggregate_twist_optimization_energy(
        local,
        twist_index=split.twist_index,
        weights=_RAW_WEIGHTS,
        intra_comm=split.intra_comm,
        inter_comm=split.inter_comm,
    )

    np.testing.assert_allclose(actual.mean, -1.5, atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO)
    np.testing.assert_allclose(
        actual.standard_error,
        np.sqrt((0.75 * 0.2) ** 2 + (0.25 * 0.4) ** 2),
        atol=ATOL_PYTHON_FP64_BZ_MEAN,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )
    np.testing.assert_allclose(
        actual.variance, 0.75 * 2.0 + 0.25 * 6.0, atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO
    )
    np.testing.assert_allclose(
        actual.variance_standard_error,
        np.sqrt((0.75 * 0.3) ** 2 + (0.25 * 0.5) ** 2),
        atol=ATOL_PYTHON_FP64_BZ_MEAN,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )
    assert _world.allgather(actual) == [actual] * _world_size

    split.intra_comm.Free()
    if split.inter_comm is not None:
        split.inter_comm.Free()


@pytest.mark.light
@requires_mpi_world
def test_planb_lm_coordinate_selection_has_world_and_per_twist_consensus():
    """Pin independent shared and twist-local LM rankings under real MPI."""
    from jqmc._twist_average import split_world_for_twists
    from jqmc.jqmc_optimizer import (
        OptimizationCoordinateMap,
        _coordinate_scope_indices,
        _select_planb_lm_scope_coordinates,
    )
    from tests.test_jqmc_kernel.optimizer_metadata_helpers import make_test_block

    split = split_world_for_twists(_world, n_twist=_N_TWIST)
    scopes = ("world", "sampler", "world", "sampler")
    blocks = tuple(
        make_test_block(
            f"p{index}",
            np.asarray(0.0),
            field="real",
            derivative_policy="direct_real",
            sync_scope=scope,
        )
        for index, scope in enumerate(scopes)
    )
    coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)
    shared = _coordinate_scope_indices(coordinate_map, "world")
    local = _coordinate_scope_indices(coordinate_map, "sampler")
    scores = np.zeros(coordinate_map.coordinate_size)
    scores[shared] = (9.0, 1.0) if _world_rank == 0 else (1.0, 9.0)
    scores[local] = (9.0, 1.0) if split.twist_index == 0 else (1.0, 9.0)

    shared_selected = _select_planb_lm_scope_coordinates(
        scores,
        coordinate_map,
        shared,
        coordinate_budget=1,
        comm=_world,
    )
    local_selected = _select_planb_lm_scope_coordinates(
        scores,
        coordinate_map,
        local,
        coordinate_budget=1,
        comm=split.intra_comm,
    )

    assert _world.allgather(shared_selected.tolist()) == [[0]] * _world_size
    expected_local = [0] if split.twist_index == 0 else [1]
    assert local_selected.tolist() == expected_local
    assert split.intra_comm.allgather(local_selected.tolist()) == [expected_local] * split.intra_comm.Get_size()

    split.intra_comm.Free()
    if split.inter_comm is not None:
        split.inter_comm.Free()


def _build_hamiltonians(tmp_dir: str):
    """Convert both Li fixtures (rank 0 writes, all ranks load).

    The tiny JAX warmup + Barrier below is the multi-rank gotcha fix
    documented in ``test_integration_qe_twist_average_mpi``: under
    ``mpirun`` the session-wide ``jax.distributed`` coordination service
    registers each rank's CPU backend lazily on its FIRST JAX op; if
    rank 0 enters the rank-0-only conversion section before rank 1 has
    registered, rank 1's topology exchange times out. Force every rank
    through the registration at a common synchronisation point first.
    """
    import jax.numpy as jnp

    from jqmc.hamiltonians import Hamiltonian_data
    from jqmc.jqmc_qe import run_qe2jqmc

    _ = jnp.sum(jnp.arange(2)).block_until_ready()
    _world.barrier()

    paths = [f"{tmp_dir}/li_twist{i}.h5" for i in range(_N_TWIST)]
    if _world_rank == 0:
        for i, p in enumerate(paths):
            run_qe2jqmc(
                {"qe_save_dir": str(_SAVES[i]), "kind": "pgto", "upf_paths": [], "ecutwfc": None, "alpha_min": None},
                {"j1_parameter": None, "j2_parameter": 1.0},
                p,
            )
    _world.barrier()
    return [Hamiltonian_data.load_from_hdf5(p) for p in paths]


@pytest.fixture(scope="module")
def twist_hamiltonians(tmp_path_factory):
    """Both Li fixtures converted ONCE for the whole module.

    ``_build_hamiltonians`` is not just slow, it is a *compiler*: the QE ->
    jQMC conversion traces and lowers its own JAX programs, and every XLA
    executable this process ever produces is pinned for the lifetime of the
    process by JAX's compiled-function caches (see
    ``tests/conftest.py::_bound_xla_executable_address_space`` for why that
    matters -- each kernel costs three VMAs out of ``vm.max_map_count``).
    Four heavy tests each converting the same two committed QE save
    directories paid that four times over for four byte-identical results.

    Module scope is safe because the conversion output is a pure function of
    the committed fixtures and nothing downstream mutates it: the optimizer
    test deep-copies before handing the list to the driver, and the MCMC /
    GFMC drivers treat ``Hamiltonian_data`` as the immutable flax pytree it
    is. Rank 0 still writes the HDF5 and every rank still loads it, exactly
    as before -- only the number of times that happens changes.
    """
    tmp_dir = _world.bcast(str(tmp_path_factory.mktemp("twist_hamiltonians")) if _world_rank == 0 else None, root=0)
    return _build_hamiltonians(tmp_dir)


def _scatter_per_twist(value_by_twist_root, my_twist, intra_rank, shape, *, dtype=np.float64):
    """Assemble the (N_twist, ...) array from per-twist intra-root values.

    Mirrors the driver's inter-reduce: only the intra rank-0 of each twist
    contributes its slice; SUM over the world assembles the full array on
    every rank.
    """
    # All MPI ranks must enter Allreduce with the same dtype.  In particular,
    # the raw parameter force is real at Gamma but complex at a non-Gamma
    # twist, so inferring the dtype independently on each rank is invalid.
    buf = np.zeros((_N_TWIST,) + shape, dtype=dtype)
    if intra_rank == 0:
        buf[my_twist] = value_by_twist_root
    return _world.allreduce(buf, op=MPI.SUM)


class TestTwistDecompositionMCMC:
    @pytest.mark.heavy
    @requires_li_fixtures
    @requires_mpi_world
    def test_twist_optimizer_shared_jastrow_one_multi_and_resume(self, tmp_path, twist_hamiltonians):
        """H1: every accepted shared update is world-identical.

        The real Gamma and complex non-Gamma twists deliberately generate
        different determinant-side statistics.  We require exact shared-J2
        equality after one and multiple steps, then require a one-step
        checkpoint + resume to reproduce the uninterrupted two-step result.
        The explicit unsafe device request is part of the regression: the
        twist driver must override it with the communicator-correct host path.
        """
        import copy

        from jqmc._twist_average import run_twist_average_optimize

        tmp_dir = _world.bcast(str(tmp_path) if _world_rank == 0 else None, root=0)
        hams = twist_hamiltonians

        common = dict(
            num_mcmc_steps=12,
            num_mcmc_warmup_steps=3,
            num_mcmc_bin_blocks=3,
            opt_J1_param=False,
            opt_J2_param=True,
            opt_J3_param=False,
            opt_geminal=False,
            optimizer_kwargs={"method": "sr", "delta": 1.0e-2, "epsilon": 1.0e-3},
            use_device_optimizer=True,
        )

        def _j2(ham):
            return float(ham.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param)

        one = run_twist_average_optimize(
            copy.deepcopy(hams),
            _RAW_WEIGHTS,
            mcmc_seed=_SEED,
            num_walkers=_NUM_WALKERS,
            world_comm=_world,
            optimize_kwargs={**common, "num_opt_steps": 1},
        )
        one_world = _world.allgather(_j2(one))
        np.testing.assert_array_equal(one_world, np.full(_world_size, one_world[0]))

        fresh = run_twist_average_optimize(
            copy.deepcopy(hams),
            _RAW_WEIGHTS,
            mcmc_seed=_SEED,
            num_walkers=_NUM_WALKERS,
            world_comm=_world,
            optimize_kwargs={**common, "num_opt_steps": 2, "wf_dump_freq": 1},
        )
        fresh_world = _world.allgather(_j2(fresh))
        np.testing.assert_array_equal(fresh_world, np.full(_world_size, fresh_world[0]))

        checkpoint_dir = f"{tmp_dir}/twist_opt_h1_resume"
        run_twist_average_optimize(
            copy.deepcopy(hams),
            _RAW_WEIGHTS,
            mcmc_seed=_SEED,
            num_walkers=_NUM_WALKERS,
            world_comm=_world,
            optimize_kwargs={**common, "num_opt_steps": 1, "wf_dump_freq": 1},
            checkpoint_dir=checkpoint_dir,
        )
        resumed = run_twist_average_optimize(
            copy.deepcopy(hams),
            _RAW_WEIGHTS,
            mcmc_seed=_SEED,
            num_walkers=_NUM_WALKERS,
            world_comm=_world,
            # On resume ``num_opt_steps`` counts ADDITIONAL steps:
            # 1 restored + 1 more = the uninterrupted 2-step run.
            optimize_kwargs={**common, "num_opt_steps": 1, "wf_dump_freq": 1},
            checkpoint_dir=checkpoint_dir,
            resume=True,
        )
        resumed_world = _world.allgather(_j2(resumed))
        np.testing.assert_array_equal(resumed_world, np.full(_world_size, resumed_world[0]))
        np.testing.assert_allclose(resumed_world, fresh_world, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_BIT_EXACT)

    @pytest.mark.heavy
    @requires_li_fixtures
    @requires_mpi_world
    def test_mcmc_driver_decomposes_into_per_twist_chains(self, twist_hamiltonians):
        """MCMC twist driver == standalone per-twist chains + weighted arithmetic.

        Covers all four estimators (E / aF / aT / gF) and both combine
        levels: per-twist equality (same seeds -> same samples -> same
        jackknife outputs) and the BZ quadrature combine (values AND
        error bars AND s_twist against hand-rolled numpy).
        """
        from jqmc._twist_average import _twist_sampler_seed, run_twist_average_from_hamiltonians, split_world_for_twists
        from jqmc.jqmc_mcmc import MCMC

        hams = twist_hamiltonians

        # ---- standalone side: one direct chain for THIS rank's twist ----
        split = split_world_for_twists(_world, n_twist=_N_TWIST)
        my_twist = int(split.twist_index)
        intra = split.intra_comm
        ham_k = hams[my_twist]
        mcmc = MCMC(
            hamiltonian_data=ham_k,
            mcmc_seed=_twist_sampler_seed(_SEED, my_twist, _N_TWIST),
            num_walkers=_NUM_WALKERS,
            mpi_comm_in=intra,
            world_comm_in=_world,
            comput_position_deriv=True,
            comput_stress=True,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=True,
        )
        mcmc.run(num_mcmc_steps=_NUM_STEPS)
        E_k, sigma_k, _, _ = mcmc.get_E(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        F_k, F_std_k = mcmc.get_aF(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        T_k, T_std_k = mcmc.get_aT(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        gF_k, gF_std_k = mcmc.get_gF(
            num_mcmc_warmup_steps=_NUM_WARMUP,
            num_mcmc_bin_blocks=_NUM_BINS,
            blocks=ham_k.wavefunction_data.get_variational_blocks(state_token=("test", 0)),
        )
        intra_rank = intra.Get_rank()
        n_atom = int(ham_k.structure_data.natom)
        n_param = int(np.ravel(np.asarray(gF_k)).size)
        E_manual = _scatter_per_twist(float(E_k), my_twist, intra_rank, ())
        sigma_manual = _scatter_per_twist(float(sigma_k), my_twist, intra_rank, ())
        F_manual = _scatter_per_twist(np.asarray(F_k, dtype=np.float64), my_twist, intra_rank, (n_atom, 3))
        F_std_manual = _scatter_per_twist(np.asarray(F_std_k, dtype=np.float64), my_twist, intra_rank, (n_atom, 3))
        T_manual = _scatter_per_twist(np.asarray(T_k, dtype=np.float64), my_twist, intra_rank, (3, 3))
        T_std_manual = _scatter_per_twist(np.asarray(T_std_k, dtype=np.float64), my_twist, intra_rank, (3, 3))
        gF_manual = _scatter_per_twist(
            np.ravel(np.asarray(gF_k)),
            my_twist,
            intra_rank,
            (n_param,),
            dtype=np.complex128,
        )
        gF_std_manual = _scatter_per_twist(np.ravel(np.asarray(gF_std_k)), my_twist, intra_rank, (n_param,))

        # ---- driver side: same seeds, same comm layout ----
        result = run_twist_average_from_hamiltonians(
            hamiltonian_list=hams,
            weights=_RAW_WEIGHTS,
            mcmc_seed=_SEED,
            num_mcmc_steps=_NUM_STEPS,
            num_walkers=_NUM_WALKERS,
            num_mcmc_warmup_steps=_NUM_WARMUP,
            num_mcmc_bin_blocks=_NUM_BINS,
            world_comm=_world,
            compute_force=True,
            compute_stress=True,
            compute_param_grad=True,
        )

        # ---- per-twist equality (identical chains) ----
        np.testing.assert_allclose(result.per_twist_E, E_manual, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        np.testing.assert_allclose(
            result.per_twist_E_std, sigma_manual, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(result.per_twist_F, F_manual, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        np.testing.assert_allclose(
            result.per_twist_F_std, F_std_manual, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(result.per_twist_T, T_manual, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        np.testing.assert_allclose(
            result.per_twist_T_std, T_std_manual, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(result.per_twist_gF, gF_manual, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        np.testing.assert_allclose(
            result.per_twist_gF_std, gF_std_manual, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )

        # ---- BZ combine == hand-rolled weighted arithmetic ----
        W = np.asarray(_RAW_WEIGHTS, dtype=np.float64)
        W = W / W.sum()
        E_TA = float(np.sum(W * E_manual))
        assert result.energy.E_TA == pytest.approx(E_TA, abs=ATOL_PYTHON_FP64_BZ_MEAN)
        assert result.energy.E_TA_std == pytest.approx(
            float(np.sqrt(np.sum(W**2 * sigma_manual**2))), abs=ATOL_PYTHON_FP64_BZ_MEAN
        )
        assert result.energy.s_twist == pytest.approx(
            float(np.sqrt(np.sum(W * (E_manual - E_TA) ** 2))), abs=ATOL_PYTHON_FP64_BZ_MEAN
        )
        np.testing.assert_allclose(
            result.force.F_TA, np.einsum("k,kad->ad", W, F_manual), atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(
            result.force.F_TA_std,
            np.sqrt(np.einsum("k,kad->ad", W**2, F_std_manual**2)),
            atol=ATOL_PYTHON_FP64_BZ_MEAN,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )
        np.testing.assert_allclose(
            result.stress.F_TA, np.einsum("k,kab->ab", W, T_manual), atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(
            result.stress.F_TA_std,
            np.sqrt(np.einsum("k,kab->ab", W**2, T_std_manual**2)),
            atol=ATOL_PYTHON_FP64_BZ_MEAN,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )
        np.testing.assert_allclose(
            result.param_grad.F_TA,
            np.einsum("k,kp->p", W, gF_manual),
            atol=ATOL_PYTHON_FP64_BZ_MEAN,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )
        np.testing.assert_allclose(
            result.param_grad.F_TA_std,
            np.sqrt(np.einsum("k,kp->p", W**2, gF_std_manual**2)),
            atol=ATOL_PYTHON_FP64_BZ_MEAN,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )

        # ---- twist-averaged translational force sum rule ----
        # Each twist's SWCT force sums to zero per sample (partition of
        # unity), so the weighted BZ mean must too. Catches combine-layer
        # weight / transpose / real-projection bugs the per-twist tests
        # cannot see.
        atol, rtol = get_tolerance_min(
            ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic"),
            "strict",
        )
        check_sum_rule_force(result.force.F_TA, atol, rtol)


class TestTwistDecompositionGFMC:
    @pytest.mark.heavy
    @requires_li_fixtures
    @requires_mpi_world
    @pytest.mark.parametrize("gfmc_kind", ["tau", "n"])
    def test_gfmc_driver_decomposes_into_per_twist_chains(self, gfmc_kind, twist_hamiltonians):
        """GFMC twist driver == standalone per-twist GFMC chains + arithmetic.

        Same contract as the MCMC test for the estimators the GFMC driver
        aggregates (E / aF / aT), for BOTH production kinds (``tau`` ->
        GFMC_t, ``n`` -> GFMC_n). The chains are short and NOT physically
        equilibrated -- irrelevant for the identity, which compares the
        same samples through two code paths.
        """
        import jax

        from jqmc._twist_average import _twist_sampler_seed, run_twist_average_gfmc, split_world_for_twists
        from jqmc.jqmc_gfmc import GFMC_n, GFMC_t

        gfmc_cls, gfmc_kwargs = {
            "tau": (GFMC_t, {"tau": 0.1, "alat": 0.1}),
            "n": (GFMC_n, {"alat": 0.1, "E_scf": -0.5, "num_mcmc_per_measurement": 16}),
        }[gfmc_kind]

        hams = twist_hamiltonians

        split = split_world_for_twists(_world, n_twist=_N_TWIST)
        my_twist = int(split.twist_index)
        intra = split.intra_comm
        ham_k = hams[my_twist]
        gfmc = gfmc_cls(
            hamiltonian_data=ham_k,
            mcmc_seed=_twist_sampler_seed(_SEED, my_twist, _N_TWIST),
            num_walkers=_NUM_WALKERS,
            mpi_comm_in=intra,
            world_comm_in=_world,
            comput_position_deriv=True,
            comput_stress=True,
            **gfmc_kwargs,
        )
        gfmc.run(num_mcmc_steps=_NUM_STEPS)
        E_k, sigma_k, _, _ = gfmc.get_E(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        F_k, F_std_k = gfmc.get_aF(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        T_k, T_std_k = gfmc.get_aT(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
        intra_rank = intra.Get_rank()
        n_atom = int(ham_k.structure_data.natom)
        E_manual = _scatter_per_twist(float(E_k), my_twist, intra_rank, ())
        sigma_manual = _scatter_per_twist(float(sigma_k), my_twist, intra_rank, ())
        F_manual = _scatter_per_twist(np.asarray(F_k, dtype=np.float64), my_twist, intra_rank, (n_atom, 3))
        F_std_manual = _scatter_per_twist(np.asarray(F_std_k, dtype=np.float64), my_twist, intra_rank, (n_atom, 3))
        T_manual = _scatter_per_twist(np.asarray(T_k, dtype=np.float64), my_twist, intra_rank, (3, 3))
        T_std_manual = _scatter_per_twist(np.asarray(T_std_k, dtype=np.float64), my_twist, intra_rank, (3, 3))

        # ---- phase boundary: release the standalone chain's executables ----
        #
        # Everything the comparison needs is already NumPy above; from here on
        # the standalone chain is dead. Dropping it matters because this single
        # test is the most expensive one in the repository in XLA address
        # space, and the reason is structural: every jitted callable inside
        # ``GFMC_t.run`` / ``GFMC_n.run`` is a ``run()``-LOCAL closure (see the
        # F3 note at ``jqmc/jqmc_gfmc.py``'s ``_projection_driver_cache``,
        # which caches them on the *instance*). The driver below builds its own
        # ``GFMC_*`` instance over the same Hamiltonian, so it recreates and
        # recompiles the whole set no matter what we do here -- the two phases
        # share nothing, and their costs simply added up inside one test.
        #
        # ``tests/conftest.py::_bound_xla_executable_address_space`` cannot help
        # with that: it reclaims at the test BOUNDARY, and this cost is entirely
        # intra-test. Measured under ``mpirun -np 2 --precision-mode=mixed``,
        # the ``[n]`` row alone cost 32655 of 65530 VMAs (49.8% of the table),
        # leaving the boundary guarantee holding by ~110 VMAs. Clearing here
        # turns "phase 1 + phase 2" into "max(phase 1, phase 2)".
        #
        # It is NOT free: ``clear_caches`` is a sledgehammer, so the driver
        # phase also has to re-TRACE the programs it did legitimately reuse
        # (tracing is Python, and is not what the persistent XLA cache
        # caches). Measured on an idle host with a warm cache, these two rows
        # alone: 1206.4 s -> 1322.1 s, i.e. +9.6% wall clock to more than
        # halve the address space (high water 45612 -> 20080 VMAs). Worth it
        # here and nowhere cheap -- do not copy this into a test that is not
        # actually near the ceiling. Beware measuring it under load: run
        # against three concurrent jobs it looks like a 2x regression.
        #
        # Both ranks execute this line at the same point of the same test, so
        # the clear stays MPI-symmetric -- the same reason the conftest fixture
        # OR-reduces its own decision. The precedent for the shape is
        # ``tests/test_rs_kernel/test_gfmc_forces.py::_clear_jax_caches_between_force_cells``.
        del gfmc
        jax.clear_caches()

        result = run_twist_average_gfmc(
            hamiltonian_list=hams,
            weights=_RAW_WEIGHTS,
            gfmc_kind=gfmc_kind,
            mcmc_seed=_SEED,
            num_mcmc_steps=_NUM_STEPS,
            num_walkers=_NUM_WALKERS,
            num_mcmc_warmup_steps=_NUM_WARMUP,
            num_mcmc_bin_blocks=_NUM_BINS,
            world_comm=_world,
            compute_force=True,
            compute_stress=True,
            gfmc_kwargs=dict(gfmc_kwargs),
        )

        np.testing.assert_allclose(result.per_twist_E, E_manual, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        np.testing.assert_allclose(
            result.per_twist_E_std, sigma_manual, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(result.per_twist_F, F_manual, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        np.testing.assert_allclose(
            result.per_twist_F_std, F_std_manual, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(result.per_twist_T, T_manual, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
        np.testing.assert_allclose(
            result.per_twist_T_std, T_std_manual, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )

        W = np.asarray(_RAW_WEIGHTS, dtype=np.float64)
        W = W / W.sum()
        E_TA = float(np.sum(W * E_manual))
        assert result.energy.E_TA == pytest.approx(E_TA, abs=ATOL_PYTHON_FP64_BZ_MEAN)
        np.testing.assert_allclose(
            result.force.F_TA, np.einsum("k,kad->ad", W, F_manual), atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(
            result.stress.F_TA, np.einsum("k,kab->ab", W, T_manual), atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO
        )


@pytest.mark.light
@requires_mpi_world
@pytest.mark.parametrize("driver_kind", ["mcmc", "gfmc"])
def test_twist_result_keeps_all_k_points_on_every_rank(monkeypatch, driver_kind):
    """Every rank retains the complete k-point table after twist decomposition."""
    from types import SimpleNamespace

    from jqmc._twist_average import run_twist_average_from_hamiltonians, run_twist_average_gfmc

    expected_k_points = [(0.0, 0.0, 0.0), (0.25, 0.0, 0.0)]
    hamiltonians = []
    for k_point in expected_k_points:
        aos_data = SimpleNamespace(k_point=k_point)
        orb_data = SimpleNamespace(aos_data=aos_data)
        geminal_data = SimpleNamespace(orb_data_up_spin=orb_data)
        hamiltonians.append(SimpleNamespace(wavefunction_data=SimpleNamespace(geminal_data=geminal_data)))

    class FakeChain:
        def __init__(self, hamiltonian_data, **kwargs):
            del kwargs
            self.k_point = hamiltonian_data.wavefunction_data.geminal_data.orb_data_up_spin.aos_data.k_point

        def run(self, num_mcmc_steps, max_time=None):
            del max_time
            # A completed MCMC run: the driver reads ``MCMC.mcmc_counter``.
            self.mcmc_counter = num_mcmc_steps

        def get_E(self, num_mcmc_warmup_steps, num_mcmc_bin_blocks):
            del num_mcmc_warmup_steps, num_mcmc_bin_blocks
            return float(self.k_point[0]), 0.01, None, None

    class FakeGFMCChain(FakeChain):
        def run(self, num_mcmc_steps, max_time=None):
            del max_time
            # A completed GFMC run: the driver reads the raw ``GFMC._mcmc_counter``.
            # ``GFMC.mcmc_counter`` leaves out the collect steps, so it is absent here.
            self._mcmc_counter = num_mcmc_steps

    common = {
        "hamiltonian_list": hamiltonians,
        "weights": _RAW_WEIGHTS,
        "mcmc_seed": _SEED,
        "num_mcmc_steps": 1,
        "num_walkers": 1,
        "num_mcmc_warmup_steps": 0,
        "num_mcmc_bin_blocks": 1,
        "world_comm": _world,
    }
    splits = _spy_twist_root_sums(monkeypatch)
    if driver_kind == "mcmc":
        import jqmc.jqmc_mcmc

        monkeypatch.setattr(jqmc.jqmc_mcmc, "MCMC", FakeChain)
        result = run_twist_average_from_hamiltonians(**common)
    else:
        import jqmc.jqmc_gfmc

        monkeypatch.setattr(jqmc.jqmc_gfmc, "GFMC_t", FakeGFMCChain)
        result = run_twist_average_gfmc(gfmc_kind="tau", gfmc_kwargs={}, **common)

    assert result.k_points == expected_k_points
    assert _world.allgather(result.k_points) == [expected_k_points] * _world_size
    _assert_no_twist_reported_stopped(splits)

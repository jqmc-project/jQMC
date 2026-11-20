"""Value oracle for one twist-averaged optimization step.

``run_twist_average_optimize`` with two twists runs one MCMC per twist, reduces
the shared Jastrow block over the all-twist world communicator with the fixed
BZ weights, and applies ONE update to every twist. This file checks the value
of that update against a recomputation from the two per-twist sampled batches.

Single process, real production collectives semantics: the two twists run in
two threads, each with its own ``MCMC`` on ``MPI.COMM_SELF`` (the twist-local
sampler), and share an in-process two-rank world communicator
(``_ThreadWorldComm``) whose collectives exchange the operands of both
threads and reduce them exactly as MPI would for two ranks (``a + b`` is
exact-commutative in IEEE arithmetic). ``split_world_for_twists`` is replaced
by the split a two-rank ``mpirun`` would produce (twist index = world rank,
intra = one-rank communicator, both ranks are twist roots). Seeds, weights,
optimizer and energy aggregation are the production ones.

Fixture: two distinct real Hamiltonians with the same Jastrow structure,
H2 all-electron cc-pVDZ in the Cartesian and in the spherical basis, Pade
J1 = 1.0 and J2 = 0.5, BZ raw weights 1 : 3, 4 walkers x 25 steps each,
seed 20260927, SR (direct) delta 0.05, epsilon 1e-3, J1 and J2 optimized.

Passive spies record, without changing values, each twist's per-step walker
positions and the batch, coordinate map, Plan-B target kwargs and target that
``compute_twist_decomposed_real_coordinate_optimization_target`` consumed / returned.
Checks (deterministic, fixed stored data):

1. each twist's consumed e_L and d ln Psi / dc equal an independent evaluation
   at that twist's stored configurations;
2. the shared update equals the dense solve of (S + eps diag S) q = f with the
   documented fixed-BZ combination S = sum_k W_k S_k, f = sum_k W_k f_k, where
   S_k, f_k are the weighted moments centred inside twist k and
   W = w / sum(w); and both twists apply delta * q (identical parameters).

Sensitivity: the same verdict rejects the uniform-weight combination and the
update computed from twist 0 alone.
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

import copy
import os
import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mpi4py import MPI

import jqmc._twist_average as twist_module
import jqmc.jqmc_optimizer as optimizer_module
from jqmc._precision import get_tolerance, get_tolerance_min
from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_LOOSE, min_S_diag_abs
from jqmc._twist_average import TwistCommSplit, normalize_twist_weights, run_twist_average_optimize
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_two_body_data
from jqmc.jqmc_mcmc import MCMC
from jqmc.jqmc_optimizer import DeviceOptimizationBatch
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data, evaluate_ln_wavefunction
from tests.test_jqmc_kernel._exact_sr_reference import exact_centred_moments, exact_sr_solve

pytestmark = [pytest.mark.with_chdir]

jax.config.update("jax_enable_x64", True)

_TREXIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")
_TWIST_FILES = ("H2_ae_ccpvdz_cart.h5", "H2_ae_ccpvdz_sphe.h5")
_RAW_WEIGHTS = (1.0, 3.0)
_J1_START = 1.0
_J2_START = 0.5
_NUM_WALKERS = 4
_NUM_STEPS = 25
_NUM_BIN_BLOCKS = 5
_DELTA = 0.05
_EPSILON = 1.0e-3
_SEED = 20260927
_SOURCE_ZONES = ("jastrow_eval", "jastrow_grad_lap", "wf_eval", "wf_kinetic", "local_energy")
_PARAMETER_ZONE = "jastrow_eval"
_BARRIER_TIMEOUT = 1800.0


# --------------------------------------------------------------------------- #
# In-process two-rank world communicator
# --------------------------------------------------------------------------- #


class _ThreadWorld:
    """Shared state of an in-process world: one slot per rank and a barrier."""

    def __init__(self, size):
        self.size = size
        self.barrier = threading.Barrier(size, timeout=_BARRIER_TIMEOUT)
        self.slots = [None] * size

    def exchange(self, rank, value):
        self.slots[rank] = value
        self.barrier.wait()
        values = [copy.deepcopy(v) for v in self.slots]
        self.barrier.wait()
        return values


def _combine(values, op):
    if op == MPI.SUM:
        result = values[0]
        for value in values[1:]:
            result = result + value
        return result
    if op == MPI.MIN:
        return (
            values[0] if len(values) == 1 else np.minimum.reduce(values) if isinstance(values[0], np.ndarray) else min(values)
        )
    if op == MPI.MAX:
        return (
            values[0] if len(values) == 1 else np.maximum.reduce(values) if isinstance(values[0], np.ndarray) else max(values)
        )
    raise NotImplementedError(f"in-process world: unsupported reduction {op!r}")


def _buffer(spec):
    return spec[0] if isinstance(spec, (list, tuple)) else spec


class _ThreadWorldComm:
    """Rank view of a ``_ThreadWorld``; collectives follow mpi4py semantics for these call shapes."""

    def __init__(self, world, rank):
        self._world = world
        self.rank = rank
        self.size = world.size

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def allreduce(self, value, op=MPI.SUM):
        return _combine(self._world.exchange(self.rank, value), op)

    def Allreduce(self, sendbuf, recvbuf, op=MPI.SUM):
        receive = _buffer(recvbuf)
        send = receive if sendbuf is MPI.IN_PLACE else _buffer(sendbuf)
        receive[...] = _combine(self._world.exchange(self.rank, np.array(send, copy=True)), op)

    def reduce(self, value, op=MPI.SUM, root=0):
        result = self.allreduce(value, op=op)
        return result if self.rank == root else None

    def bcast(self, value, root=0):
        return self._world.exchange(self.rank, value)[root]

    def gather(self, value, root=0):
        values = self._world.exchange(self.rank, value)
        return values if self.rank == root else None

    def allgather(self, value):
        return self._world.exchange(self.rank, value)

    def Barrier(self):
        self._world.exchange(self.rank, None)

    barrier = Barrier

    def __getattr__(self, name):
        raise AttributeError(f"in-process world communicator does not implement {name!r}")


# --------------------------------------------------------------------------- #
# Fixture and capture
# --------------------------------------------------------------------------- #


def _build_hamiltonian(stem):
    structure_data, _aos, _mos_up, _mos_dn, geminal_mo_data, coulomb_potential_data = read_trexio_file(
        trexio_file=os.path.join(_TREXIO_DIR, stem), store_tuple=True
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
        wavefunction_data=Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data),
    )


def _jastrow_parameters(hamiltonian_data):
    jastrow = hamiltonian_data.wavefunction_data.jastrow_data
    return np.asarray(
        [
            np.asarray(jastrow.jastrow_one_body_data.jastrow_1b_param),
            np.asarray(jastrow.jastrow_two_body_data.jastrow_2b_param),
        ],
        dtype=np.float64,
    )


def _host_coordinates(coordinate_map, batch, sources):
    if isinstance(batch, DeviceOptimizationBatch):
        return np.asarray(coordinate_map.transform_derivatives_device(sources))
    return np.asarray(coordinate_map.transform_derivatives(sources))


_CAPTURE = {}


def _two_twist_capture():
    """Run (once per process and precision mode) one twist-averaged optimization step on two threads."""
    from jqmc._precision import is_mixed_precision_enabled

    key = is_mixed_precision_enabled()
    if key in _CAPTURE:
        return _CAPTURE[key]
    hamiltonians = [_build_hamiltonian(stem) for stem in _TWIST_FILES]
    world = _ThreadWorld(len(hamiltonians))
    records = [dict(positions_up=[], positions_dn=[]) for _ in hamiltonians]
    results = [None] * len(hamiltonians)
    errors = []
    production_target = optimizer_module.compute_twist_decomposed_real_coordinate_optimization_target
    production_sample = MCMC.sample

    def split_like_two_rank_mpirun(world_comm, n_twist):
        assert n_twist == world_comm.Get_size()
        return TwistCommSplit(
            world_comm=world_comm,
            intra_comm=MPI.COMM_SELF,
            inter_comm=world_comm,
            twist_index=world_comm.Get_rank(),
            n_twist=n_twist,
        )

    def recording_target(batch, coordinate_map, **kwargs):
        target = production_target(batch, coordinate_map, **kwargs)
        records[kwargs["world_comm"].Get_rank()].update(
            batch=batch, coordinate_map=coordinate_map, target_kwargs=dict(kwargs), target=target
        )
        return target

    def recording_sample(self, *, num_steps, max_time):
        record = records[self.world_comm.Get_rank()]
        for _ in range(num_steps):
            production_sample(self, num_steps=1, max_time=max_time)
            record["positions_up"].append(np.asarray(self._MCMC__latest_r_up_carts))
            record["positions_dn"].append(np.asarray(self._MCMC__latest_r_dn_carts))

    optimize_kwargs = dict(
        num_mcmc_steps=_NUM_STEPS,
        num_opt_steps=1,
        num_mcmc_warmup_steps=0,
        num_mcmc_bin_blocks=_NUM_BIN_BLOCKS,
        optimizer_kwargs={"method": "sr", "delta": _DELTA, "epsilon": _EPSILON, "cg_flag": False},
        opt_J1_param=True,
        opt_J2_param=True,
        opt_J3_param=False,
        opt_geminal=False,
    )

    def run_rank(rank):
        try:
            results[rank] = run_twist_average_optimize(
                copy.deepcopy(hamiltonians),
                list(_RAW_WEIGHTS),
                mcmc_seed=_SEED,
                num_walkers=_NUM_WALKERS,
                world_comm=_ThreadWorldComm(world, rank),
                mcmc_kwargs={"Dt": 2.0},
                optimize_kwargs=copy.deepcopy(optimize_kwargs),
            )
        except BaseException as error:  # propagate to the main thread, release the partner
            errors.append((rank, error))
            world.barrier.abort()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(twist_module, "split_world_for_twists", split_like_two_rank_mpirun)
        patch.setattr(optimizer_module, "compute_twist_decomposed_real_coordinate_optimization_target", recording_target)
        patch.setattr(MCMC, "sample", recording_sample)
        threads = [threading.Thread(target=run_rank, args=(rank,)) for rank in range(len(hamiltonians))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    if errors:
        rank, error = sorted(errors, key=lambda item: isinstance(item[1], threading.BrokenBarrierError))[0]
        raise AssertionError(f"twist rank {rank} failed") from error

    twists = []
    for rank, record in enumerate(records):
        batch = record["batch"]
        coordinate_map = record["coordinate_map"]
        twists.append(
            dict(
                hamiltonian_before=hamiltonians[rank],
                target_kwargs=record["target_kwargs"],
                target=record["target"],
                batch=batch,
                coordinate_map=coordinate_map,
                r_up=np.stack(record["positions_up"]).reshape(_NUM_STEPS * _NUM_WALKERS, -1, 3),
                r_dn=np.stack(record["positions_dn"]).reshape(_NUM_STEPS * _NUM_WALKERS, -1, 3),
                weights=np.asarray(batch.weights, dtype=np.float64),
                local_energies=np.asarray(batch.local_energies),
                log_derivatives=_host_coordinates(coordinate_map, batch, batch.log_derivative_sources),
                params_before=_jastrow_parameters(hamiltonians[rank]),
                params_after=_jastrow_parameters(results[rank]),
            )
        )
    _CAPTURE[key] = twists
    return twists


# --------------------------------------------------------------------------- #
# Independent evaluation and references
# --------------------------------------------------------------------------- #


def _with_parameters(hamiltonian_data, parameters):
    wavefunction = hamiltonian_data.wavefunction_data
    jastrow = wavefunction.jastrow_data
    jastrow = jastrow.replace(
        jastrow_one_body_data=jastrow.jastrow_one_body_data.replace(jastrow_1b_param=parameters[0]),
        jastrow_two_body_data=jastrow.jastrow_two_body_data.replace(jastrow_2b_param=parameters[1]),
    )
    return hamiltonian_data.replace(wavefunction_data=wavefunction.replace(jastrow_data=jastrow))


def _independent_sources(twist):
    base = twist["hamiltonian_before"]
    rotation = jnp.eye(3)

    def ln_psi(parameters, r_up, r_dn):
        return evaluate_ln_wavefunction(_with_parameters(base, parameters).wavefunction_data, r_up, r_dn)

    def local_energy(parameters, r_up, r_dn):
        return compute_local_energy(_with_parameters(base, parameters), r_up, r_dn, rotation)

    parameters = jnp.asarray(twist["params_before"], dtype=jnp.float64)
    energies = jax.jit(jax.vmap(local_energy, in_axes=(None, 0, 0)))(parameters, twist["r_up"], twist["r_dn"])
    log_derivatives = jax.jit(jax.vmap(jax.grad(ln_psi), in_axes=(None, 0, 0)))(parameters, twist["r_up"], twist["r_dn"])
    return np.asarray(energies), np.asarray(log_derivatives)


def _twist_moments(twist):
    """Extended-precision centred moments of one twist (``_exact_sr_reference``)."""
    O = np.real(np.asarray(twist["log_derivatives"], dtype=np.complex128))
    e = np.real(np.asarray(twist["local_energies"], dtype=np.complex128))
    return exact_centred_moments(O, e, twist["weights"])


def _reference_bz_update(twists, bz_weights):
    """Dense solve of (S + eps diag S) q = f with S = sum_k W_k S_k and f = sum_k W_k f_k."""
    S = f = 0.0
    for twist, weight in zip(twists, bz_weights, strict=True):
        S_k, f_k = _twist_moments(twist)
        S = float(weight) * S_k + S
        f = float(weight) * f_k + f
    return exact_sr_solve(S, f, _EPSILON)


def _assert_update_matches(actual, reference, label):
    # Batches and the host solve are fp64 in both precision modes.
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(reference),
        rtol=RTOL_PYTHON_FP64_LOOSE,
        atol=ATOL_PYTHON_FP64_LOOSE,
        err_msg=f"{label} differs from the fixed-BZ dense reference",
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.heavy
def test_two_twist_batches_match_independent_evaluation():
    """Each twist consumed e_L and d ln Psi / dc equal to an independent evaluation at its stored configurations."""
    twists = _two_twist_capture()
    atol, rtol = get_tolerance_min(_SOURCE_ZONES, "strict")
    for rank, twist in enumerate(twists):
        assert twist["target_kwargs"]["world_comm"].Get_size() == 2
        assert twist["target_kwargs"]["sampler_comm"] is MPI.COMM_SELF
        assert twist["target_kwargs"]["sampler_weight"] == float(normalize_twist_weights(_RAW_WEIGHTS)[rank])
        assert [block.name for block in twist["coordinate_map"].blocks] == ["j1_param", "j2_param"]
        energies, log_derivatives = _independent_sources(twist)
        np.testing.assert_allclose(
            np.real(twist["local_energies"]), energies, rtol=rtol, atol=atol, err_msg=f"twist {rank} e_L"
        )
        np.testing.assert_allclose(
            np.real(twist["log_derivatives"]), log_derivatives, rtol=rtol, atol=atol, err_msg=f"twist {rank} dlnPsi/dc"
        )
        assert np.all(twist["weights"] > 0.0)
    # The twists are physically different Hamiltonians: their batches differ.
    energy_gap = np.max(np.abs(np.real(twists[0]["local_energies"]) - np.real(twists[1]["local_energies"])))
    assert energy_gap > atol + rtol * np.max(np.abs(np.real(twists[1]["local_energies"]))), energy_gap


@pytest.mark.heavy
def test_two_twist_shared_update_equals_fixed_bz_weighted_recomputation():
    """The applied shared-Jastrow update equals the fixed-BZ weighted SR solve from both twists' batches.

    W = (1, 3) / 4. The reference combines the per-twist centred moments with
    W_k and solves (S + epsilon diag S) q = f densely. Both twists report the
    same target q, equal to the reference at fp64 round-off, and both apply
    delta * q to identical starting J1/J2 (identical updated parameters).
    Sensitivity: the uniform-weight combination and the twist-0-only update
    are distinguishable from the reference and fail the same verdict.
    """
    twists = _two_twist_capture()
    bz_weights = normalize_twist_weights(_RAW_WEIGHTS)
    reference_q = _reference_bz_update(twists, bz_weights)
    for rank, twist in enumerate(twists):
        target = twist["target"]
        assert np.all(np.asarray(target.metric_diagonal) > min_S_diag_abs)
        assert target.frozen_coordinate_groups == ()
        _assert_update_matches(target.q, reference_q, f"twist {rank} target q")
    np.testing.assert_array_equal(twists[0]["params_before"], twists[1]["params_before"])
    np.testing.assert_array_equal(twists[0]["params_after"], twists[1]["params_after"])
    atol, rtol = get_tolerance(_PARAMETER_ZONE, "strict")
    for rank, twist in enumerate(twists):
        np.testing.assert_allclose(
            twist["params_after"] - twist["params_before"],
            _DELTA * reference_q,
            rtol=rtol,
            atol=atol,
            err_msg=f"twist {rank} applied shared update",
        )

    uniform_q = _reference_bz_update(twists, np.full(2, 0.5))
    with pytest.raises(AssertionError, match="differs from the fixed-BZ dense reference"):
        _assert_update_matches(uniform_q, reference_q, "uniform-weight combination")
    twist_zero_q = _reference_bz_update(twists[:1], np.ones(1))
    with pytest.raises(AssertionError, match="differs from the fixed-BZ dense reference"):
        _assert_update_matches(twist_zero_q, reference_q, "twist-0-only update")

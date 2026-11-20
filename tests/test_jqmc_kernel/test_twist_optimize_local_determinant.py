"""Value oracle for the twist-local determinant block of a twist-averaged optimization step.

``run_twist_average_optimize`` with ``opt_geminal=True`` optimizes two kinds
of coordinates (doc/notes/wavefunction_optimization.md, "Twist ownership and
parameter coordinates"): the shared Jastrow block (``sync_scope="world"``),
whose metric and force are combined over all twists with the fixed BZ
weights, and the twist-local determinant block (``sync_scope="sampler"``,
here the occupied MO coefficients ``mo_coefficients_up``), whose metric and
force are accumulated only within each twist, with zero shared/local cross
blocks. This file checks the twist-local block against an independent
per-twist recomputation from the sampled batches.

Harness: the in-process two-rank world of ``test_twist_optimize_update_value``
(two threads, one ``MCMC`` per twist on ``MPI.COMM_SELF``, collectives that
reduce both ranks' operands as MPI does, the split a two-rank ``mpirun``
produces). ``MCMC.__init__`` is serialized between the threads because it
seeds and draws the initial walkers from NumPy's process-global RNG (one
process per rank under mpirun); this keeps each twist's sampled batch
reproducible. Fixture: H2 all-electron cc-pVDZ in the Cartesian and in the
spherical basis (distinct real Gamma-point Hamiltonians), Pade J1 = 1.0 and
J2 = 0.5, closed-shell single determinant with ``time_reversal=True``, BZ raw
weights 1 : 3, 4 walkers x 25 steps, seed 20260927, delta 0.05, epsilon 1e-3,
J1, J2 and the geminal optimized. Cases: direct SR, one-dimensional aSR
(``use_lm=True, lm_subspace_dim=0``) and LM with a coordinate budget of 3
(``lm_subspace_dim=3``: full LM in the two-coordinate shared scope, a
three-coordinate selection in the ten-coordinate local scope), each with the
host and the device (JAX batch, not GPU) optimizer batch.

The MO coordinates are tangent coordinates of the S-Stiefel manifold of the
occupied row (C S C^T fixed, S the analytic AO overlap): q moves the single
occupied row C to C + q - (q . u) C with u = S C^T / (C S C^T). The reference
builds this map from the AO overlap alone and differentiates through it.

Checks (deterministic, fixed stored data):

1. each twist's consumed e_L, d ln Psi / dq (aSR/LM also d e_L / dq) over
   (J1, J2, q) equal an independent autodiff evaluation at that twist's
   stored configurations, with the tied down-spin row C_dn = conj(C_up)
   entering by the chain rule;
2. for each twist k the target force on the local coordinates equals f_k,loc
   of twist k's own batch, the local SR direction equals the dense solve
   (S_k,loc + eps diag S_k,loc) q = f_k,loc, and the local target equals it
   (SR), gamma_k times it (aSR, one-dimensional model of twist k's own
   samples) or the nonsymmetric LM solve in the basis of that direction and
   the coordinates selected by twist k's own |f_k| / sigma(f_k) (LM); the
   shared coordinates equal the fixed-BZ weighted reference over the shared
   sub-block (force sum_k W_k f_k,shared); the shared J1/J2 update
   delta * q_shared is applied identically on both twists.

Sensitivity: the cross-twist (pooled) local solve, which a wrong
communicator for the twist-local scope would produce, fails the same verdict,
and the two twists' own LM selections differ.
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
import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mpi4py import MPI

import jqmc._twist_average as twist_module
import jqmc.jqmc_optimizer as optimizer_module
from jqmc._precision import get_tolerance, get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_LOOSE
from jqmc._twist_average import TwistCommSplit, normalize_twist_weights, run_twist_average_optimize
from jqmc.atomic_orbital import compute_overlap_matrix
from jqmc.hamiltonians import compute_local_energy
from jqmc.jqmc_mcmc import MCMC
from jqmc.wavefunction import evaluate_ln_wavefunction
from tests.test_jqmc_kernel import test_optimizer_real_batch_update as reference
from tests.test_jqmc_kernel import test_twist_optimize_update_value as sampled
from tests.test_jqmc_kernel._exact_sr_reference import exact_centred_moments, exact_sr_solve, to_float64

pytestmark = [pytest.mark.with_chdir]

jax.config.update("jax_enable_x64", True)

_CASES = ("sr-host", "sr-device", "asr-host", "asr-device", "lm-host", "lm-device")
_LM_BUDGET = 3
_BLOCK_NAMES = ["j1_param", "j2_param", "mo_coefficients_up"]
_SOURCE_ZONES = (*sampled._SOURCE_ZONES, "ao_eval", "ao_grad_lap", "mo_eval", "mo_grad", "mo_lap", "det_eval", "det_grad_lap")


# --------------------------------------------------------------------------- #
# Capture: one twist-averaged step with the geminal block optimized
# --------------------------------------------------------------------------- #

_CAPTURE = {}


def _capture(case):
    """Run (once per process, precision mode and case) one two-twist step with opt_geminal=True."""
    method, batch_kind = case.split("-")
    key = (is_mixed_precision_enabled(), case)
    if key in _CAPTURE:
        return _CAPTURE[key]
    hamiltonians = [sampled._build_hamiltonian(stem) for stem in sampled._TWIST_FILES]
    world = sampled._ThreadWorld(len(hamiltonians))
    records = [dict(positions_up=[], positions_dn=[]) for _ in hamiltonians]
    results = [None] * len(hamiltonians)
    errors = []
    production_target = optimizer_module.compute_twist_decomposed_real_coordinate_optimization_target
    production_sample = MCMC.sample
    production_init = MCMC.__init__
    init_lock = threading.Lock()

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

    def serialized_init(self, *args, **kwargs):
        # MCMC.__init__ seeds and draws the initial walkers from NumPy's process-global
        # legacy RNG (one process per rank under mpirun). Serialize it so the two
        # in-process twists cannot interleave those draws; it has no collectives.
        with init_lock:
            production_init(self, *args, **kwargs)

    settings = {"method": "sr", "delta": sampled._DELTA, "epsilon": sampled._EPSILON, "cg_flag": False}
    if method == "asr":
        settings.update(use_lm=True, lm_subspace_dim=0)
    elif method == "lm":
        settings.update(use_lm=True, lm_subspace_dim=_LM_BUDGET)
    optimize_kwargs = dict(
        num_mcmc_steps=sampled._NUM_STEPS,
        num_opt_steps=1,
        num_mcmc_warmup_steps=0,
        num_mcmc_bin_blocks=sampled._NUM_BIN_BLOCKS,
        optimizer_kwargs=settings,
        opt_J1_param=True,
        opt_J2_param=True,
        opt_J3_param=False,
        opt_geminal=True,
        dump_wavefunction=False,
        use_device_optimizer=batch_kind == "device",
    )

    def run_rank(rank):
        try:
            results[rank] = run_twist_average_optimize(
                copy.deepcopy(hamiltonians),
                list(sampled._RAW_WEIGHTS),
                mcmc_seed=sampled._SEED,
                num_walkers=sampled._NUM_WALKERS,
                world_comm=sampled._ThreadWorldComm(world, rank),
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
        patch.setattr(MCMC, "__init__", serialized_init)
        threads = [threading.Thread(target=run_rank, args=(rank,)) for rank in range(len(hamiltonians))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    if errors:
        rank, error = sorted(errors, key=lambda item: isinstance(item[1], threading.BrokenBarrierError))[0]
        raise AssertionError(f"twist rank {rank} failed") from error

    n_samples = sampled._NUM_STEPS * sampled._NUM_WALKERS
    twists = []
    for rank, record in enumerate(records):
        batch = record["batch"]
        coordinate_map = record["coordinate_map"]
        energy_derivatives = batch.local_energy_derivative_sources
        twists.append(
            dict(
                hamiltonian_before=hamiltonians[rank],
                target_kwargs=record["target_kwargs"],
                target=record["target"],
                batch=batch,
                coordinate_map=coordinate_map,
                r_up=np.stack(record["positions_up"]).reshape(n_samples, -1, 3),
                r_dn=np.stack(record["positions_dn"]).reshape(n_samples, -1, 3),
                weights=np.asarray(batch.weights, dtype=np.float64),
                local_energies=np.asarray(batch.local_energies),
                log_derivatives=sampled._host_coordinates(coordinate_map, batch, batch.log_derivative_sources),
                energy_derivatives=(
                    None if energy_derivatives is None else sampled._host_coordinates(coordinate_map, batch, energy_derivatives)
                ),
                shared=np.asarray(optimizer_module._coordinate_scope_indices(coordinate_map, "world")),
                local=np.asarray(optimizer_module._coordinate_scope_indices(coordinate_map, "sampler")),
                params_before=sampled._jastrow_parameters(hamiltonians[rank]),
                params_after=sampled._jastrow_parameters(results[rank]),
            )
        )
    _CAPTURE[key] = twists
    return twists


# --------------------------------------------------------------------------- #
# Independent evaluation and references
# --------------------------------------------------------------------------- #


def _occupied_row_and_tangent_basis(hamiltonian_data):
    """Occupied up MO row C and the S-orthogonal tangent map of the MO coordinates.

    The MO coordinates are tangent coordinates of the S-Stiefel manifold of
    occupied rows (C S C^H = const, S the analytic AO overlap): for the single
    occupied row of H2, a coordinate vector q moves C to C + q - (q . u) C with
    u = S C^T / (C S C^T), i.e. the S-orthogonal projection that removes the
    normalisation direction. Built here from the AO overlap alone.
    """
    geminal = hamiltonian_data.wavefunction_data.geminal_data
    mo_up = np.asarray(geminal.orb_data_up_spin.mo_coefficients)
    assert mo_up.shape[0] == 1 and np.isrealobj(mo_up), "this reference is written for one real occupied row"
    overlap = np.asarray(compute_overlap_matrix(geminal.orb_data_up_spin.aos_data), dtype=np.float64)
    row = np.asarray(mo_up[0], dtype=np.float64)
    u = overlap @ row / (row @ overlap @ row)
    return row, u


def _initial_coordinates(hamiltonian_data):
    """(J1, J2, q = 0) in the documented block order."""
    row, _u = _occupied_row_and_tangent_basis(hamiltonian_data)
    return np.concatenate([sampled._jastrow_parameters(hamiltonian_data), np.zeros_like(row)])


def _with_coordinates(hamiltonian_data, coordinates, row, u):
    """Place J1, J2 and the moved occupied up row; the tied down row is C_dn = conj(C_up)."""
    hamiltonian_data = sampled._with_parameters(hamiltonian_data, coordinates[:2])
    wavefunction = hamiltonian_data.wavefunction_data
    geminal = wavefunction.geminal_data
    q = coordinates[2:]
    mo_up = jnp.reshape(row + q - jnp.dot(q, u) * row, geminal.orb_data_up_spin.mo_coefficients.shape)
    geminal = geminal.replace(
        orb_data_up_spin=geminal.orb_data_up_spin.replace(mo_coefficients=mo_up),
        orb_data_dn_spin=geminal.orb_data_dn_spin.replace(mo_coefficients=jnp.conj(mo_up)),
    )
    return hamiltonian_data.replace(wavefunction_data=wavefunction.replace(geminal_data=geminal))


def _independent_sources(twist):
    base = twist["hamiltonian_before"]
    rotation = jnp.eye(3)
    row, u = _occupied_row_and_tangent_basis(base)

    def ln_psi(coordinates, r_up, r_dn):
        return evaluate_ln_wavefunction(_with_coordinates(base, coordinates, row, u).wavefunction_data, r_up, r_dn)

    def local_energy(coordinates, r_up, r_dn):
        return compute_local_energy(_with_coordinates(base, coordinates, row, u), r_up, r_dn, rotation)

    coordinates = jnp.asarray(_initial_coordinates(base), dtype=jnp.float64)
    vmapped = dict(in_axes=(None, 0, 0))
    energies = jax.jit(jax.vmap(local_energy, **vmapped))(coordinates, twist["r_up"], twist["r_dn"])
    log_derivatives = jax.jit(jax.vmap(jax.grad(ln_psi), **vmapped))(coordinates, twist["r_up"], twist["r_dn"])
    energy_derivatives = None
    if twist["energy_derivatives"] is not None:
        energy_derivatives = jax.jit(jax.vmap(jax.grad(local_energy), **vmapped))(coordinates, twist["r_up"], twist["r_dn"])
        energy_derivatives = np.asarray(energy_derivatives)
    return np.asarray(energies), np.asarray(log_derivatives), energy_derivatives


def _moments(twist, indices):
    """Extended-precision moments centred inside one twist, restricted to ``indices``."""
    O = np.real(np.asarray(twist["log_derivatives"], dtype=np.complex128))[:, indices]
    e = np.real(np.asarray(twist["local_energies"], dtype=np.complex128))
    return exact_centred_moments(O, e, twist["weights"])


def _sr_solve(S, f):
    return exact_sr_solve(S, f, sampled._EPSILON)


def _top_scores(scores, budget):
    """Indices of the ``budget`` largest scores (boundary ties kept), the documented LM selection rule."""
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    chosen = order[:budget]
    chosen += [index for index in order[budget:] if scores[index] == scores[chosen[-1]]]
    return np.asarray(sorted(chosen), dtype=np.int64)


def _accelerated_or_linear(O, D, e, w, sr_q, method, kwargs, selected, label):
    """aSR along sr_q, or LM in the basis (sr_q, selected unit vectors); returns (q, energy, LM weight)."""
    if method == "asr":
        gamma, energy, capped = reference._reference_asr_update(
            O, D, e, w, sr_q, kwargs["asr_gamma_max"], kwargs["asr_fallback_gamma"]
        )
        assert not capped, f"the reference-connected {label} aSR root must be live and uncapped"
        return gamma * sr_q, energy, None
    basis = np.eye(sr_q.size)
    if selected is not None:
        basis = np.zeros((sr_q.size, selected.size + 1))
        basis[:, 0] = sr_q
        basis[selected, np.arange(1, selected.size + 1)] = 1.0
    reduced_q, energy, weight = reference._reference_lm_update(O @ basis, D @ basis, e, w)
    return basis @ reduced_q, energy, weight


def _reference_local(twist, method, kwargs):
    """Twist-local block from this twist's own batch only.

    SR: (S_k + eps diag S_k) q = f_k. aSR: gamma_k times it. LM: the
    nonsymmetric LM eigenproblem in the basis (SR direction, coordinates
    selected by this twist's own |f_k| / sigma(f_k)). Returns
    (q, sr_q, force, selected or None, predicted energy, LM reference weight).
    """
    local = twist["local"]
    S, f = _moments(twist, local)
    sr_q = _sr_solve(S, f)
    f = to_float64(f)
    if method == "sr":
        return sr_q, sr_q, f, None, None, None
    selected = None
    if method == "lm":
        std = np.asarray(twist["target"].coordinate_force_std)[local]
        selected = _top_scores(np.abs(f) / std, _LM_BUDGET)
    q, energy, weight = _accelerated_or_linear(
        np.real(twist["log_derivatives"])[:, local],
        np.real(twist["energy_derivatives"])[:, local],
        np.real(twist["local_energies"]),
        twist["weights"],
        sr_q,
        method,
        kwargs,
        selected,
        "local",
    )
    return q, sr_q, f, selected, energy, weight


def _combined_shared_samples(twists, bz_weights, shared):
    """Fixed-BZ objective samples (per-twist centring, weight W_k w / sum w) over the shared coordinates."""
    common_energy = sum(
        weight * np.real(twist["weights"] @ twist["local_energies"]) / twist["weights"].sum()
        for weight, twist in zip(bz_weights, twists, strict=True)
    )
    Os, Ds, es, ws = [], [], [], []
    for weight, twist in zip(bz_weights, twists, strict=True):
        w = twist["weights"]
        norm = w.sum()
        O = np.real(twist["log_derivatives"])[:, shared]
        e = np.real(twist["local_energies"])
        D = np.zeros_like(O) if twist["energy_derivatives"] is None else np.real(twist["energy_derivatives"])[:, shared]
        Os.append(O - w @ O / norm)
        Ds.append(D)
        es.append(e - w @ e / norm + common_energy)
        ws.append(weight * w / norm)
    return tuple(map(np.concatenate, (Os, Ds, es, ws)))


def _reference_shared(twists, bz_weights, method, kwargs):
    """Shared block: fixed-BZ weighted SR (aSR along it, or full LM) over the shared sub-block.

    Returns (q, sr_q, force, predicted energy, LM reference weight).
    """
    shared = twists[0]["shared"]
    S = sum(weight * _moments(twist, shared)[0] for weight, twist in zip(bz_weights, twists, strict=True))
    f = sum(weight * _moments(twist, shared)[1] for weight, twist in zip(bz_weights, twists, strict=True))
    sr_q = _sr_solve(S, f)
    f = to_float64(f)
    if method == "sr":
        return sr_q, sr_q, f, None, None
    assert shared.size <= _LM_BUDGET, "the shared scope fits the LM budget: no shared selection"
    O, D, e, w = _combined_shared_samples(twists, bz_weights, shared)
    q, energy, weight = _accelerated_or_linear(O, D, e, w, sr_q, method, kwargs, None, "shared")
    return q, sr_q, f, energy, weight


def _pooled_local(twists):
    """What a twist-local SR solve over the all-twist communicator would produce (cross-twist coupling)."""
    local = twists[0]["local"]
    moments = [_moments(twist, local) for twist in twists]
    return _sr_solve(sum(S for S, _ in moments), sum(f for _, f in moments))


def _assert_matches(actual, expected, label):
    # Batches, coordinate moments and the host/device solves are fp64 in both precision modes.
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=RTOL_PYTHON_FP64_LOOSE,
        atol=ATOL_PYTHON_FP64_LOOSE,
        err_msg=f"{label} differs from the independent dense reference",
    )


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.heavy
@pytest.mark.parametrize("case", _CASES)
def test_twist_local_determinant_sources_match_independent_evaluation(case):
    """Each twist consumed e_L and d ln Psi / dc (aSR/LM: d e_L / dc) over (J1, J2, C_up) equal an independent evaluation."""
    method, batch_kind = case.split("-")
    twists = _capture(case)
    atol, rtol = get_tolerance_min(_SOURCE_ZONES, "strict")
    for rank, twist in enumerate(twists):
        assert twist["target_kwargs"]["world_comm"].Get_size() == 2
        assert twist["target_kwargs"]["sampler_comm"] is MPI.COMM_SELF
        assert isinstance(twist["batch"], optimizer_module.DeviceOptimizationBatch) == (batch_kind == "device")
        blocks = twist["coordinate_map"].blocks
        assert [block.name for block in blocks] == _BLOCK_NAMES
        assert [block.sync_scope for block in blocks] == ["world", "world", "sampler"]
        assert twist["shared"].tolist() == [0, 1]
        assert twist["local"].tolist() == list(range(2, twist["coordinate_map"].coordinate_size))
        assert twist["hamiltonian_before"].wavefunction_data.geminal_data.time_reversal
        assert (twist["energy_derivatives"] is not None) == (method != "sr")
        energies, log_derivatives, energy_derivatives = _independent_sources(twist)
        assert log_derivatives.shape == twist["log_derivatives"].shape
        np.testing.assert_allclose(
            np.real(twist["local_energies"]), energies, rtol=rtol, atol=atol, err_msg=f"twist {rank} e_L"
        )
        np.testing.assert_allclose(
            np.real(twist["log_derivatives"]), log_derivatives, rtol=rtol, atol=atol, err_msg=f"twist {rank} dlnPsi/dc"
        )
        if energy_derivatives is not None:
            np.testing.assert_allclose(
                np.real(twist["energy_derivatives"]),
                energy_derivatives,
                rtol=rtol,
                atol=atol,
                err_msg=f"twist {rank} de_L/dc",
            )
        assert np.all(twist["weights"] > 0.0)
    # The MO derivatives are live (not all zero) on both twists.
    for twist in twists:
        local_O = np.real(twist["log_derivatives"])[:, twist["local"]]
        assert np.max(np.abs(local_O - local_O.mean(axis=0))) > atol + rtol * np.max(np.abs(local_O))


@pytest.mark.heavy
@pytest.mark.parametrize("case", _CASES)
def test_twist_local_determinant_update_equals_per_twist_reference(case):
    """The twist-local MO update of each twist equals its own per-twist reference; the shared block stays fixed-BZ.

    For twist k the target force on the local coordinates equals f_k,loc of
    twist k's own batch, the local SR direction equals the dense solve
    (S_k,loc + epsilon diag S_k,loc) q = f_k,loc, and the local target equals
    that direction (SR), gamma_k times it (aSR) or the LM solve in the basis
    of that direction and the coordinates selected by twist k's own
    |f_k| / sigma(f_k) (LM, budget 3). The shared J1/J2 coordinates equal the
    fixed-BZ weighted reference over the shared sub-block, and both twists
    apply delta * q_shared to identical starting J1/J2.
    Sensitivity: the pooled (cross-twist) local solve and, for LM, the other
    twist's selection fail the same verdicts.
    """
    method, _batch_kind = case.split("-")
    twists = _capture(case)
    bz_weights = normalize_twist_weights(sampled._RAW_WEIGHTS)
    kwargs = twists[0]["target_kwargs"]
    shared = twists[0]["shared"]
    q_shared, sr_shared, f_shared, energy_shared, weight_shared = _reference_shared(twists, bz_weights, method, kwargs)
    references = []
    for rank, twist in enumerate(twists):
        target = twist["target"]
        local = twist["local"]
        np.testing.assert_array_equal(twist["shared"], shared)
        q_local, sr_local, f_local, selected, energy_local, weight_local = _reference_local(
            twist, method, twist["target_kwargs"]
        )
        references.append((sr_local, selected))
        assert target.frozen_coordinate_groups == ()
        _assert_matches(np.asarray(target.coordinate_force)[local], f_local, f"twist {rank} local force")
        _assert_matches(np.asarray(target.coordinate_force)[shared], f_shared, f"twist {rank} shared force")
        _assert_matches(np.asarray(target.sr_q)[local], sr_local, f"twist {rank} local SR direction")
        _assert_matches(np.asarray(target.sr_q)[shared], sr_shared, f"twist {rank} shared SR direction")
        if method == "lm":
            selected_groups = tuple((int(index),) for index in (*shared.tolist(), *local[selected].tolist()))
            assert target.selected_coordinate_groups == selected_groups, (
                f"twist {rank} LM selection {target.selected_coordinate_groups} != own-force selection {selected_groups}"
            )
        _assert_matches(np.asarray(target.q)[local], q_local, f"twist {rank} local target q")
        _assert_matches(np.asarray(target.q)[shared], q_shared, f"twist {rank} shared target q")
        if method != "sr":
            assert target.rejection_reason is None, target.rejection_reason
            _assert_matches(target.predicted_energy, min(energy_local, energy_shared), f"twist {rank} predicted energy")
        if method == "lm":
            _assert_matches(target.lm_reference_weight, min(weight_local, weight_shared), f"twist {rank} LM weight")

    np.testing.assert_array_equal(twists[0]["params_before"], twists[1]["params_before"])
    np.testing.assert_array_equal(twists[0]["params_after"], twists[1]["params_after"])
    atol, rtol = get_tolerance(sampled._PARAMETER_ZONE, "strict")
    for rank, twist in enumerate(twists):
        np.testing.assert_allclose(
            twist["params_after"] - twist["params_before"],
            sampled._DELTA * q_shared,
            rtol=rtol,
            atol=atol,
            err_msg=f"twist {rank} applied shared update",
        )

    pooled = _pooled_local(twists)
    for rank, (sr_local, _selected) in enumerate(references):
        with pytest.raises(AssertionError, match="differs from the independent dense reference"):
            _assert_matches(pooled, sr_local, f"twist {rank} pooled cross-twist local solve")
    if method == "lm":
        assert not np.array_equal(references[0][1], references[1][1]), "the two twists' own LM selections must differ"

"""GDD regression net for MPC results on twist-averaged routes.

MPC is sampled per twist as a passive observable.  The BZ layer must carry
both the correction and the covariance-aware corrected total through MPI,
checkpoint/resume, and the final CLI output without changing the pure-Ewald
energy contract.
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

import inspect
import logging
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from jqmc._setting import ATOL_PYTHON_FP64_BZ_MEAN, ATOL_PYTHON_FP64_TIGHT
from jqmc._twist_average import TwistAverageEnergy, TwistAverageMPC, TwistAverageResult
from jqmc.coulomb_potential import Coulomb_potential_data


pytestmark = [pytest.mark.light, pytest.mark.with_chdir]


_PER_TWIST = {
    "E": np.array([-1.0, -2.0]),
    "E_std": np.array([0.40, 0.60]),
    "dE": np.array([0.20, -0.10]),
    "dE_std": np.array([0.30, 0.20]),
    # Deliberately covariance-narrowed relative to E_std/dE_std.  Rebuilding
    # this error from their quadrature gives the wrong result.
    "E_mpc": np.array([-0.80, -2.10]),
    "E_mpc_std": np.array([0.10, 0.25]),
}
_RAW_WEIGHTS = np.array([1.0, 3.0])
_MPC_CUTOFF = 42.0


def _hamiltonian_stub(kx: float):
    aos = SimpleNamespace(k_point=(kx, 0.0, 0.0))
    orb = SimpleNamespace(aos_data=aos)
    geminal = SimpleNamespace(orb_data_up_spin=orb, _determinant_dtype_jnp=np.dtype(np.float64))
    wavefunction = SimpleNamespace(geminal_data=geminal)
    # The twist drivers forward ``coulomb_potential_data`` to the resume ECP
    # identity check.  Use the all-electron value the TREXIO loader builds.
    return SimpleNamespace(
        wavefunction_data=wavefunction,
        structure_data=SimpleNamespace(natom=1),
        coulomb_potential_data=Coulomb_potential_data(ecp_flag=False),
    )


class _IntraComm:
    def Get_rank(self):
        return 0

    def bcast(self, value, root=0):
        assert root == 0
        return value


class _InterComm:
    """Inject the second twist into each Level-2 SUM in production order."""

    def __init__(self):
        self._remote = [
            np.array([0.0, _PER_TWIST["E"][1]]),
            np.array([0.0, _PER_TWIST["E_std"][1]]),
            # ``per_twist_stopped``: the remote twist also completed its steps.
            np.zeros(2, dtype=np.int64),
            np.array([0.0, _PER_TWIST["dE"][1]]),
            np.array([0.0, _PER_TWIST["dE_std"][1]]),
            np.array([0.0, _PER_TWIST["E_mpc"][1]]),
            np.array([0.0, _PER_TWIST["E_mpc_std"][1]]),
        ]
        self.calls = 0
        self.sent = []

    def Get_rank(self):
        return 0

    def allreduce(self, value, op=None):
        self.sent.append(np.array(value, copy=True))
        remote = self._remote[self.calls]
        self.calls += 1
        return np.asarray(value) + remote


class _WorldComm:
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 2

    def bcast(self, value, root=0):
        assert root == 0
        return value


class _FakeSampler:
    instances = []

    def __init__(self, hamiltonian_data, **kwargs):
        self.hamiltonian_data = hamiltonian_data
        self.kwargs = kwargs
        self.__class__.instances.append(self)

    def run(self, num_mcmc_steps, max_time=float("inf")):
        self.num_mcmc_steps = num_mcmc_steps
        self.max_time = max_time
        # A completed MCMC run: the driver reads ``MCMC.mcmc_counter``.
        self.mcmc_counter = num_mcmc_steps

    def get_E(self, **kwargs):
        return (_PER_TWIST["E"][0], _PER_TWIST["E_std"][0], 0.0, 0.0)

    def get_dE_mpc(self, **kwargs):
        return (_PER_TWIST["dE"][0], _PER_TWIST["dE_std"][0])

    def get_E_mpc(self, **kwargs):
        return (_PER_TWIST["E_mpc"][0], _PER_TWIST["E_mpc_std"][0])


class _FakeGFMC(_FakeSampler):
    """A completed GFMC run: the driver reads the raw ``GFMC._mcmc_counter``.

    ``GFMC.mcmc_counter`` leaves out the collect steps, so it is deliberately
    absent here: a driver that read it would fail instead of passing.
    """

    def run(self, num_mcmc_steps, max_time=float("inf")):
        self.num_mcmc_steps = num_mcmc_steps
        self.max_time = max_time
        self._mcmc_counter = num_mcmc_steps


def _run_fake_twist_route(monkeypatch, route: str):
    import jqmc._twist_average as twist_average

    intra = _IntraComm()
    inter = _InterComm()
    monkeypatch.setattr(
        twist_average,
        "split_world_for_twists",
        lambda world_comm, n_twist: SimpleNamespace(intra_comm=intra, inter_comm=inter, twist_index=0),
    )
    _FakeSampler.instances = []
    hams = [_hamiltonian_stub(0.0), _hamiltonian_stub(0.25)]
    common = dict(
        hamiltonian_list=hams,
        weights=_RAW_WEIGHTS,
        mcmc_seed=19,
        num_mcmc_steps=8,
        num_walkers=2,
        num_mcmc_warmup_steps=2,
        num_mcmc_bin_blocks=2,
        world_comm=_WorldComm(),
    )
    if route == "mcmc":
        import jqmc.jqmc_mcmc as mcmc_module

        monkeypatch.setattr(mcmc_module, "MCMC", _FakeSampler)
        result = twist_average.run_twist_average_from_hamiltonians(
            **common,
            mcmc_kwargs={"mpc": True, "mpc_cutoff": _MPC_CUTOFF},
        )
    else:
        import jqmc.jqmc_gfmc as gfmc_module

        class_name = "GFMC_t" if route == "gfmc_tau" else "GFMC_n"
        monkeypatch.setattr(gfmc_module, class_name, _FakeGFMC)
        result = twist_average.run_twist_average_gfmc(
            **common,
            gfmc_kind="tau" if route == "gfmc_tau" else "n",
            gfmc_kwargs={"mpc": True, "mpc_cutoff": _MPC_CUTOFF, **_GFMC_N_KWARGS.get(route, {})},
        )
    return result, inter


# GFMC_n has no default projection count (#690); these tests use 360 at the
# class default alat = 0.1.
_GFMC_N_KWARGS = {"gfmc_n": {"num_mcmc_per_measurement": 360}}


@pytest.mark.parametrize("route", ["mcmc", "gfmc_tau", "gfmc_n"])
def test_mpc_bz_aggregate_uses_unequal_weights_and_joint_per_twist_errors(monkeypatch, route):
    """MCMC/GFMC MPC values survive Level-2 reduction with correct errors.

    Each twist first supplies the covariance-aware ``get_E_mpc`` error.  Only
    then may independent twists combine in ``sqrt(sum(W_k^2 sigma_k^2))``.
    Reconstructing the corrected error from BZ Ewald and dE bars must disagree.
    """
    result, inter = _run_fake_twist_route(monkeypatch, route)
    # E, E_std, per_twist_stopped, dE, dE_std, E_mpc, E_mpc_std Level-2 SUMs.
    assert inter.calls == 7
    # The local twist completed, so its ``per_twist_stopped`` operand is an all-zero count vector.
    assert inter.sent[2].dtype == np.int64
    np.testing.assert_array_equal(inter.sent[2], np.zeros(2, dtype=np.int64))
    assert result.mpc is not None
    assert result.mpc.cutoff == pytest.approx(_MPC_CUTOFF, abs=ATOL_PYTHON_FP64_TIGHT)
    np.testing.assert_array_equal(result.mpc.per_twist_correction, _PER_TWIST["dE"])
    np.testing.assert_array_equal(result.mpc.per_twist_correction_std, _PER_TWIST["dE_std"])
    np.testing.assert_array_equal(result.mpc.per_twist_energy, _PER_TWIST["E_mpc"])
    np.testing.assert_array_equal(result.mpc.per_twist_energy_std, _PER_TWIST["E_mpc_std"])

    W = _RAW_WEIGHTS / np.sum(_RAW_WEIGHTS)
    expected_dE = float(np.sum(W * _PER_TWIST["dE"]))
    expected_dE_std = float(np.sqrt(np.sum(W**2 * _PER_TWIST["dE_std"] ** 2)))
    expected_E_mpc = float(np.sum(W * _PER_TWIST["E_mpc"]))
    expected_E_mpc_std = float(np.sqrt(np.sum(W**2 * _PER_TWIST["E_mpc_std"] ** 2)))
    assert pytest.approx(expected_dE, abs=ATOL_PYTHON_FP64_BZ_MEAN) == result.mpc.correction.E_TA
    assert result.mpc.correction.E_TA_std == pytest.approx(expected_dE_std, abs=ATOL_PYTHON_FP64_BZ_MEAN)
    assert pytest.approx(expected_E_mpc, abs=ATOL_PYTHON_FP64_BZ_MEAN) == result.mpc.energy.E_TA
    assert result.mpc.energy.E_TA_std == pytest.approx(expected_E_mpc_std, abs=ATOL_PYTHON_FP64_BZ_MEAN)
    wrong_quadrature = np.hypot(result.energy.E_TA_std, result.mpc.correction.E_TA_std)
    assert result.mpc.energy.E_TA_std != pytest.approx(wrong_quadrature, abs=ATOL_PYTHON_FP64_BZ_MEAN)
    assert _FakeSampler.instances[0].kwargs["mpc"] is True
    from jqmc._twist_average import _twist_sampler_seed

    assert _FakeSampler.instances[0].kwargs["mcmc_seed"] == _twist_sampler_seed(19, 0, 2)
    base_seed = 34456
    assert _twist_sampler_seed(base_seed, 0, 1) == base_seed
    seeds = [_twist_sampler_seed(base_seed, twist_index, 64) for twist_index in range(64)]
    assert seeds == [_twist_sampler_seed(base_seed, twist_index, 64) for twist_index in range(64)]
    assert len(set(seeds)) == len(seeds)
    # Post-SeedSequence-migration contract: multi-twist chain seeds are the
    # bare key tuples (mcmc_seed, twist_index) -- never collapsed to a uint32
    # (the old range assert guarded exactly the collapse that caused the
    # 2^32 seed-overflow bug).
    assert seeds == [(base_seed, twist_index) for twist_index in range(64)]


def test_mpc_false_keeps_result_absent_and_does_not_call_mpc_estimators(monkeypatch):
    """Pure-Ewald twist runs neither evaluate nor publish an MPC result."""
    import jqmc.jqmc_mcmc as mcmc_module
    from jqmc._twist_average import run_twist_average_from_hamiltonians

    class NoMPCSampler(_FakeSampler):
        def get_dE_mpc(self, **kwargs):
            raise AssertionError("mpc=False must not evaluate get_dE_mpc")

        def get_E_mpc(self, **kwargs):
            raise AssertionError("mpc=False must not evaluate get_E_mpc")

    monkeypatch.setattr(mcmc_module, "MCMC", NoMPCSampler)
    result = run_twist_average_from_hamiltonians(
        hamiltonian_list=[_hamiltonian_stub(0.0)],
        weights=[1.0],
        num_mcmc_steps=8,
        num_mcmc_warmup_steps=2,
        num_mcmc_bin_blocks=2,
        world_comm=_WorldComm(),
        mcmc_kwargs={"mpc": False},
    )
    assert result.mpc is None


@pytest.mark.parametrize("route", ["mcmc", "gfmc_tau", "gfmc_n"])
def test_mpc_resume_rehydrates_all_per_twist_summaries_without_resampling(monkeypatch, tmp_path, route):
    """A complete MPC checkpoint rebuilds the result without constructing a sampler."""
    import jqmc._twist_checkpoint as checkpoint
    import jqmc._twist_average as twist_average

    summary = {
        "E_k": _PER_TWIST["E"][0],
        "sigma_k": _PER_TWIST["E_std"][0],
        "dE_mpc_k": _PER_TWIST["dE"][0],
        "sigma_dE_mpc_k": _PER_TWIST["dE_std"][0],
        "E_mpc_k": _PER_TWIST["E_mpc"][0],
        "sigma_E_mpc_k": _PER_TWIST["E_mpc_std"][0],
        "mpc_cutoff": _MPC_CUTOFF,
    }
    monkeypatch.setattr(checkpoint, "is_twist_complete", lambda *args, **kwargs: True)
    monkeypatch.setattr(checkpoint, "load_twist_checkpoint_summary", lambda *args, **kwargs: summary)
    monkeypatch.setattr(checkpoint, "twist_identity_digest", lambda ham: "digest")

    class MustNotRun:
        def __init__(self, *args, **kwargs):
            raise AssertionError("a complete MPC twist must not be resampled")

    common = dict(
        hamiltonian_list=[_hamiltonian_stub(0.0)],
        weights=[1.0],
        num_mcmc_steps=8,
        num_mcmc_warmup_steps=2,
        num_mcmc_bin_blocks=2,
        world_comm=_WorldComm(),
        checkpoint_dir=str(tmp_path),
        resume=True,
    )
    if route == "mcmc":
        import jqmc.jqmc_mcmc as mcmc_module

        MustNotRun.__signature__ = inspect.signature(mcmc_module.MCMC)
        monkeypatch.setattr(mcmc_module, "MCMC", MustNotRun)
        result = twist_average.run_twist_average_from_hamiltonians(
            **common,
            mcmc_kwargs={"mpc": True, "mpc_cutoff": _MPC_CUTOFF},
        )
    else:
        import jqmc.jqmc_gfmc as gfmc_module

        class_name = "GFMC_t" if route == "gfmc_tau" else "GFMC_n"
        MustNotRun.__signature__ = inspect.signature(getattr(gfmc_module, class_name))
        monkeypatch.setattr(gfmc_module, class_name, MustNotRun)
        result = twist_average.run_twist_average_gfmc(
            **common,
            gfmc_kind="tau" if route == "gfmc_tau" else "n",
            gfmc_kwargs={"mpc": True, "mpc_cutoff": _MPC_CUTOFF, **_GFMC_N_KWARGS.get(route, {})},
        )
    assert result.mpc is not None
    assert pytest.approx(_PER_TWIST["dE"][0], abs=ATOL_PYTHON_FP64_TIGHT) == result.mpc.correction.E_TA
    assert result.mpc.energy.E_TA_std == pytest.approx(_PER_TWIST["E_mpc_std"][0], abs=ATOL_PYTHON_FP64_TIGHT)


def test_checkpoint_summary_and_identity_include_mpc_cutoff(monkeypatch, tmp_path):
    """MPC summaries round-trip and a changed cutoff is a stale-result error."""
    import jqmc._checkpoint as checkpoint_base
    from jqmc._twist_checkpoint import (
        is_twist_complete,
        load_twist_checkpoint_summary,
        save_twist_checkpoint,
        twist_checkpoint_path,
    )

    class Comm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def Barrier(self):
            return None

    class Driver:
        hamiltonian_data = None

        def save_to_hdf5(self, filepath):
            with h5py.File(filepath, "w"):
                pass

    def fake_merge(*, output_path, **kwargs):
        with h5py.File(output_path, "w") as handle:
            handle.create_group("_meta")

    monkeypatch.setattr(checkpoint_base, "merge_rank_checkpoints", fake_merge)
    save_twist_checkpoint(
        str(tmp_path),
        twist_idx=0,
        mcmc_or_gfmc=Driver(),
        intra_comm=Comm(),
        inter_comm=None,
        driver_type="MCMC",
        target_steps=8,
        converged=True,
        E_k=_PER_TWIST["E"][0],
        sigma_k=_PER_TWIST["E_std"][0],
        dE_mpc_k=_PER_TWIST["dE"][0],
        sigma_dE_mpc_k=_PER_TWIST["dE_std"][0],
        E_mpc_k=_PER_TWIST["E_mpc"][0],
        sigma_E_mpc_k=_PER_TWIST["E_mpc_std"][0],
        mpc_cutoff=_MPC_CUTOFF,
        ham_digest="digest",
    )

    path = twist_checkpoint_path(str(tmp_path), 0)
    summary = load_twist_checkpoint_summary(str(tmp_path), 0)
    with h5py.File(path, "r") as handle:
        stored = {key: float(handle["_meta"].attrs[key]) for key in summary if key in handle["_meta"].attrs}
    for key in (
        "dE_mpc_k",
        "sigma_dE_mpc_k",
        "E_mpc_k",
        "sigma_E_mpc_k",
        "mpc_cutoff",
    ):
        assert summary[key] == pytest.approx(stored[key], abs=ATOL_PYTHON_FP64_TIGHT)
    assert is_twist_complete(
        str(tmp_path),
        0,
        8,
        ham_digest="digest",
        mpc_cutoff=_MPC_CUTOFF,
    )
    with pytest.raises(ValueError, match="mpc_cutoff"):
        is_twist_complete(
            str(tmp_path),
            0,
            8,
            ham_digest="digest",
            mpc_cutoff=_MPC_CUTOFF + 1.0,
        )
    with h5py.File(path, "a") as handle:
        del handle["_meta"].attrs["mpc_cutoff"]
    assert not is_twist_complete(
        str(tmp_path),
        0,
        8,
        ham_digest="digest",
        mpc_cutoff=_MPC_CUTOFF,
    )


@pytest.mark.parametrize("mpc_enabled", [True, False])
def test_cli_twist_summary_prints_mpc_only_when_enabled(mpc_enabled):
    """BZ CLI output publishes joint MPC values and stays silent when disabled."""
    from jqmc.jqmc_cli import _log_twist_average_result

    mpc = None
    if mpc_enabled:
        mpc = TwistAverageMPC(
            correction=TwistAverageEnergy(0.05, 0.01, 0.02, np.array([0.25, 0.75])),
            energy=TwistAverageEnergy(-1.45, 0.03, 0.04, np.array([0.25, 0.75])),
            per_twist_correction=np.array([0.08, 0.04]),
            per_twist_correction_std=np.array([0.02, 0.01]),
            per_twist_energy=np.array([-0.92, -1.62]),
            per_twist_energy_std=np.array([0.04, 0.03]),
            cutoff=_MPC_CUTOFF,
        )
    result = TwistAverageResult(
        energy=TwistAverageEnergy(-1.5, 0.04, 0.1, np.array([0.25, 0.75])),
        force=None,
        per_twist_E=np.array([-1.0, -1.666666666666]),
        per_twist_E_std=np.array([0.05, 0.05]),
        per_twist_F=None,
        per_twist_F_std=None,
        k_points=[None, (0.25, 0.0, 0.0)],
        mpc=mpc,
    )
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("jqmc")
    handler = Capture()
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        _log_twist_average_result(result, with_force=False)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    text = "\n".join(records)
    if mpc_enabled:
        assert "BZ-averaged MPC correction: dE_MPC = +0.05000 +- 0.01000 Ha (cutoff = 42.0 Ha)." in text
        assert "BZ-averaged Total Energy (MPC-corrected): E_MPC = -1.45000 +- 0.03000 Ha." in text
    else:
        assert "dE_MPC" not in text
        assert "E_MPC" not in text


def test_shared_cli_mpc_collector_forwards_the_requested_gfmc_warmup():
    """The shared collector prevents LRDMC from using an unrelated MCMC warmup local."""
    import ast
    import inspect

    import jqmc.jqmc_cli as cli
    from jqmc._twist_average import run_twist_average_optimize

    _collect_mpc_estimates = cli._collect_mpc_estimates
    _validate_mpc_job_type = cli._validate_mpc_job_type

    calls = []

    class Driver:
        def get_dE_mpc(self, **kwargs):
            calls.append(("dE", kwargs))
            return (0.1, 0.01)

        def get_E_mpc(self, **kwargs):
            calls.append(("E", kwargs))
            return (-1.0, 0.02)

    assert _collect_mpc_estimates(Driver(), num_mcmc_warmup_steps=17, num_mcmc_bin_blocks=6) == (0.1, 0.01, -1.0, 0.02)
    assert calls == [
        ("dE", {"num_mcmc_warmup_steps": 17, "num_mcmc_bin_blocks": 6}),
        ("E", {"num_mcmc_warmup_steps": 17, "num_mcmc_bin_blocks": 6}),
    ]

    # Pin the three single-twist CLI call sites as well: VMC uses its MCMC
    # locals, while both LRDMC branches must supply their GFMC locals.  This
    # guards against a NameError or wrong routing at these call sites.
    tree = ast.parse(inspect.getsource(cli._cli))
    validation_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_validate_mpc_job_type"
    ]
    assert len(validation_calls) == 1
    assert ast.unparse(validation_calls[0].args[0]) == "job_type"
    assert {keyword.arg: ast.unparse(keyword.value) for keyword in validation_calls[0].keywords} == {"mpc": "bool(mpc)"}
    with pytest.raises(ValueError, match="does not publish a final MPC estimate"):
        _validate_mpc_job_type("vmc", mpc=True)
    for job_type in ("mcmc", "lrdmc-tau", "lrdmc-bra"):
        _validate_mpc_job_type(job_type, mpc=True)
    _validate_mpc_job_type("vmc", mpc=False)
    with pytest.raises(ValueError, match="does not publish a final MPC estimate"):
        run_twist_average_optimize(
            hamiltonian_list=[_hamiltonian_stub(0.0)],
            weights=[1.0],
            mcmc_kwargs={"mpc": True},
        )
    callsite_kwargs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_collect_mpc_estimates":
            callsite_kwargs.append({keyword.arg: ast.unparse(keyword.value) for keyword in node.keywords})
    assert callsite_kwargs == [
        {
            "num_mcmc_warmup_steps": "num_mcmc_warmup_steps",
            "num_mcmc_bin_blocks": "num_mcmc_bin_blocks",
        },
        {
            "num_mcmc_warmup_steps": "num_gfmc_warmup_steps",
            "num_mcmc_bin_blocks": "num_gfmc_bin_blocks",
        },
        {
            "num_mcmc_warmup_steps": "num_gfmc_warmup_steps",
            "num_mcmc_bin_blocks": "num_gfmc_bin_blocks",
        },
    ]

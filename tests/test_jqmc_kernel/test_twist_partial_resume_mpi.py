"""Deterministic mixed cached/fresh twist recovery on two real MPI ranks."""

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

from functools import wraps
from pathlib import Path
import re
import shutil

from mpi4py import MPI
import numpy as np
import pytest

from jqmc._setting import GFMC_MIN_BIN_BLOCKS, GFMC_MIN_COLLECT_STEPS, GFMC_MIN_STEPS, GFMC_MIN_WARMUP_STEPS
from jqmc._twist_average import run_twist_average_from_hamiltonians, run_twist_average_gfmc
from jqmc import _twist_checkpoint
from jqmc._twist_checkpoint import twist_checkpoint_path
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data
from jqmc.jqmc_gfmc import GFMC_t
from jqmc.jqmc_mcmc import MCMC
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir, pytest.mark.mpi_only]


def _read_h2_twists():
    inputs = Path(__file__).resolve().parents[1] / "trexio_example_files"
    hamiltonians = []
    for filename in ("H2_pbc_ae_cart.h5", "H2_pbc_ae_k025_cart.h5"):
        structure, _, _, _, geminal, coulomb = read_trexio_file(str(inputs / filename), store_tuple=True)
        hamiltonians.append(
            Hamiltonian_data(
                structure_data=structure,
                coulomb_potential_data=coulomb,
                wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data()),
            )
        )
    return hamiltonians


@pytest.mark.parametrize("cached_index", [0, 1])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_partial_twist_resume_matches_uninterrupted_values(tmp_path, monkeypatch, cached_index):
    """Two unequal-weight twists preserve their indexed values when only one is cached.

    Real MPI ranks run committed H2 Gamma and k025 Cartesian pGTO/MO inputs,
    one up/one down electron and no Jastrow, with weights 1/4 and 3/4.
    Two walkers per twist use seed 12345, twenty one-update measurements, five warmup
    measurements and three bins, without cond(G) screening. Both cached-index
    choices are required. No sampler, communication or reduction is mocked.

    Copy only one completed twist from an uninterrupted run into an otherwise
    empty recovery directory. This models a partial checkpoint set, not a
    process kill. Resume must call the real sampler only for the missing twist,
    preserve the cached file bytes, and reconstruct both indexed energies and
    dispersions exactly. The final weighted mean is checked against an explicit
    1/4,3/4 sum. Sampling error is output data, never an acceptance tolerance.
    """
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2
    rank = comm.Get_rank()
    root = Path(comm.bcast(str(tmp_path) if rank == 0 else None, root=0))
    options = dict(
        hamiltonian_list=_read_h2_twists(),
        weights=[0.25, 0.75],
        mcmc_seed=12345,
        num_walkers=2,
        num_mcmc_steps=20,
        num_mcmc_warmup_steps=5,
        num_mcmc_bin_blocks=3,
        mcmc_kwargs={"num_mcmc_per_measurement": 1},
        world_comm=comm,
    )
    original = root / "uninterrupted"
    recovery = root / "partial"
    expected = run_twist_average_from_hamiltonians(checkpoint_dir=str(original), **options)
    comm.Barrier()
    cached = Path(twist_checkpoint_path(str(recovery), cached_index))
    missing = Path(twist_checkpoint_path(str(recovery), 1 - cached_index))
    if rank == 0:
        recovery.mkdir()
        shutil.copy2(twist_checkpoint_path(str(original), cached_index), cached)
    comm.Barrier()
    saved_bytes = cached.read_bytes()
    assert not missing.exists()
    calls = []
    real_run = MCMC.run

    def counted_run(driver, *args, **kwargs):
        calls.append(rank)
        return real_run(driver, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(MCMC, "run", counted_run)
        actual = run_twist_average_from_hamiltonians(checkpoint_dir=str(recovery), resume=True, **options)
    comm.Barrier()
    assert comm.allgather(len(calls)) == [int(index != cached_index) for index in range(2)]
    assert cached.read_bytes() == saved_bytes
    assert missing.is_file()
    for name in ("per_twist_E", "per_twist_E_std"):
        observed, reference = getattr(actual, name), getattr(expected, name)
        assert observed.shape == reference.shape == (2,)
        assert np.isfinite(observed).all() and np.isfinite(reference).all()
        np.testing.assert_array_equal(observed, reference)
    assert expected.per_twist_E[0] != expected.per_twist_E[1], "the fixture must detect a cached-row permutation"
    observed_aggregate = np.asarray([actual.energy.E_TA, actual.energy.E_TA_std, actual.energy.s_twist])
    expected_aggregate = np.asarray([expected.energy.E_TA, expected.energy.E_TA_std, expected.energy.s_twist])
    assert np.isfinite(observed_aggregate).all() and np.isfinite(expected_aggregate).all()
    np.testing.assert_array_equal(observed_aggregate, expected_aggregate)
    np.testing.assert_array_equal(actual.energy.W, np.asarray([0.25, 0.75]))
    np.testing.assert_array_equal(actual.energy.E_TA, 0.25 * expected.per_twist_E[0] + 0.75 * expected.per_twist_E[1])

    # Only the rank that owns the edited twist can see the conflicting setting
    # during its lookup. Both ranks must stop before either enters the sampler.
    if rank == 0:
        import h5py

        with h5py.File(cached, "r+") as handle:
            handle["rank_0/driver_config"].attrs["Dt"] = 0.25
    comm.Barrier()
    edited_bytes = cached.read_bytes()
    with pytest.raises(ValueError, match=rf"{cached.name}.*has Dt=.*refusing to reuse"):
        run_twist_average_from_hamiltonians(checkpoint_dir=str(recovery), resume=True, **options)
    assert cached.read_bytes() == edited_bytes

    # Put both ranks on twist 0. Its root alone must check the file, and both
    # ranks must reuse the completed estimate using the shared decision.
    checked = []
    real_check = _twist_checkpoint.is_twist_complete

    def counted_check(*args, **kwargs):
        checked.append(rank)
        return real_check(*args, **kwargs)

    # The one-twist resume needs a one-twist directory: the two-twist one is refused (issue 800).
    single_twist = {**options, "hamiltonian_list": options["hamiltonian_list"][:1], "weights": [1.0]}
    single = run_twist_average_from_hamiltonians(checkpoint_dir=str(root / "single"), **single_twist)
    comm.Barrier()
    with monkeypatch.context() as patch:
        patch.setattr(_twist_checkpoint, "is_twist_complete", counted_check)
        resumed = run_twist_average_from_hamiltonians(checkpoint_dir=str(root / "single"), resume=True, **single_twist)
    assert comm.allgather(len(checked)) == [1, 0]
    np.testing.assert_array_equal(resumed.per_twist_E, single.per_twist_E)


@pytest.mark.parametrize("cached_index", [0, 1])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_partial_twist_resume_verdict_rejects_corrupted_cached_energy(tmp_path, monkeypatch, cached_index):
    """The real two-rank recovery oracle rejects a one-Hartree cached-energy corruption.

    First run the unchanged Gamma/k025 H2 test successfully. Then alter only
    the cached summary returned by the real checkpoint loader during recovery,
    leaving both samplers, communicator operations and fixed inputs unchanged.
    Both cached-index choices must fail the per-twist value comparison on both
    ranks, not an unrelated exception or file-existence check.
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        (tmp_path / "baseline").mkdir()
        (tmp_path / "corrupted").mkdir()
    MPI.COMM_WORLD.Barrier()
    test_partial_twist_resume_matches_uninterrupted_values(tmp_path / "baseline", monkeypatch, cached_index)
    real_load = _twist_checkpoint.load_twist_checkpoint_summary

    def corrupt_energy(*args, **kwargs):
        summary = real_load(*args, **kwargs)
        summary["E_k"] += 1.0
        return summary

    with monkeypatch.context() as patch:
        patch.setattr(_twist_checkpoint, "load_twist_checkpoint_summary", corrupt_energy)
        with pytest.raises(AssertionError, match="Arrays are not equal"):
            test_partial_twist_resume_matches_uninterrupted_values(tmp_path / "corrupted", monkeypatch, cached_index)


def _observable_run_options(route, comm):
    options = dict(
        hamiltonian_list=_read_h2_twists(),
        weights=[0.25, 0.75],
        mcmc_seed=12345,
        num_walkers=2,
        world_comm=comm,
        compute_force=True,
        compute_stress=True,
    )
    mpc = {"mpc": True, "mpc_cutoff": 4.0}
    if route == "mcmc":
        options.update(
            num_mcmc_steps=20,
            num_mcmc_warmup_steps=5,
            num_mcmc_bin_blocks=3,
            mcmc_kwargs={"num_mcmc_per_measurement": 1, **mpc},
        )
        return run_twist_average_from_hamiltonians, MCMC, options
    options.update(
        gfmc_kind="tau",
        num_mcmc_steps=GFMC_MIN_STEPS,
        num_mcmc_warmup_steps=GFMC_MIN_WARMUP_STEPS,
        num_mcmc_bin_blocks=GFMC_MIN_BIN_BLOCKS,
        gfmc_kwargs={"alat": 0.5, "num_gfmc_collect_steps": GFMC_MIN_COLLECT_STEPS, **mpc},
    )
    return run_twist_average_gfmc, GFMC_t, options


def _hand_weighted_twist_pair(means, errors):
    mean = 0.25 * means[0] + 0.75 * means[1]
    error = np.sqrt(0.25**2 * errors[0] ** 2 + 0.75**2 * errors[1] ** 2)
    return np.stack((mean, error))


@pytest.mark.parametrize("route", ["mcmc", "gfmc_tau"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_resume_refuses_changed_bz_weights(tmp_path, route):
    """A resume with other BZ weights refuses the twists stored for the old ones.

    jqmc-tool combines a twist directory with the weights stored in it, so
    reusing a twist written for weights 1:3 in a 3:1 resume would leave a
    directory that reports the old average while the run prints the new one.
    The control resumes both H2 twists with the unchanged weights and must
    equal the explicit 1/4,3/4 sum of the per-twist energies of the first run.
    The 3:1 resume must fail on both ranks and leave each twist file unchanged.
    """
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2
    root = Path(comm.bcast(str(tmp_path) if comm.Get_rank() == 0 else None, root=0))
    run, _sampler, options = _observable_run_options(route, comm)
    checkpoint_dir = str(root / "checkpoints")
    first = run(checkpoint_dir=checkpoint_dir, **options)
    comm.Barrier()
    checkpoint = Path(twist_checkpoint_path(checkpoint_dir, comm.Get_rank()))
    saved_bytes = checkpoint.read_bytes()
    same = run(checkpoint_dir=checkpoint_dir, resume=True, **options)
    assert np.isfinite(first.per_twist_E).all()
    np.testing.assert_array_equal(same.energy.E_TA, 0.25 * first.per_twist_E[0] + 0.75 * first.per_twist_E[1])
    assert checkpoint.read_bytes() == saved_bytes
    with pytest.raises(ValueError, match=r"normalized BZ weight .* but resume expects .*: refusing to blend"):
        run(checkpoint_dir=checkpoint_dir, resume=True, **{**options, "weights": [0.75, 0.25]})
    assert checkpoint.read_bytes() == saved_bytes


@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_resume_refuses_a_changed_twist_count(tmp_path):
    """Growing or shrinking the twist set on resume is refused.

    A one-twist H2 Gamma directory resumed with the Gamma/k025 pair would reuse
    twist 0 stored for one twist of weight 1. A Gamma/k025 directory resumed
    with Gamma alone would keep ``twist_001.h5``, which jqmc-tool still
    combines. Both resumes must fail on both ranks and leave every twist file
    unchanged.
    """
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2
    root = Path(comm.bcast(str(tmp_path) if comm.Get_rank() == 0 else None, root=0))
    hamiltonians = _read_h2_twists()
    options = dict(
        mcmc_seed=12345,
        num_walkers=2,
        num_mcmc_steps=20,
        num_mcmc_warmup_steps=5,
        num_mcmc_bin_blocks=3,
        mcmc_kwargs={"num_mcmc_per_measurement": 1},
        world_comm=comm,
    )
    one, two = root / "one-twist", root / "two-twist"
    run_twist_average_from_hamiltonians(hamiltonians[:1], [1.0], checkpoint_dir=str(one), **options)
    run_twist_average_from_hamiltonians(hamiltonians, [0.25, 0.75], checkpoint_dir=str(two), **options)
    comm.Barrier()
    saved = {path: path.read_bytes() for directory in (one, two) for path in sorted(directory.glob("twist_*.h5"))}
    assert len(saved) == 3
    with pytest.raises(ValueError, match=r"num_twists=1 but resume expects num_twists=2: refusing to blend"):
        run_twist_average_from_hamiltonians(hamiltonians, [0.25, 0.75], checkpoint_dir=str(one), resume=True, **options)
    with pytest.raises(ValueError, match=r"twist_001\.h5 beyond the 1 twist\(s\) of this run: refusing to resume"):
        run_twist_average_from_hamiltonians(hamiltonians[:1], [1.0], checkpoint_dir=str(two), resume=True, **options)
    comm.Barrier()
    assert sorted(path for directory in (one, two) for path in directory.glob("twist_*.h5")) == list(saved)
    assert all(path.read_bytes() == content for path, content in saved.items())


@pytest.mark.parametrize("route", ["mcmc", "gfmc_tau"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_resume_refuses_a_cached_twist_analysed_with_other_settings(tmp_path, route):
    """A resume refuses a completed twist analysed with other steps, warmup or bins.

    A reused twist keeps the energy of its stored analysis. Resuming the
    Gamma/k025 H2 pair with twist 1 removed and another warmup, bin count or
    shorter chain would combine that stored analysis of twist 0 with a fresh
    analysis of twist 1 into one BZ average. Each such resume must fail on both
    ranks before sampling, name twist 0 and its stored and requested settings,
    leave twist 0 unchanged and write no twist 1. The control resumes with the
    unchanged settings and must equal the explicit 1/4,3/4 sum of the per-twist
    energies of the uninterrupted run.
    """
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2
    rank = comm.Get_rank()
    root = Path(comm.bcast(str(tmp_path) if rank == 0 else None, root=0))
    options = dict(hamiltonian_list=_read_h2_twists(), weights=[0.25, 0.75], mcmc_seed=12345, num_walkers=2, world_comm=comm)
    if route == "mcmc":
        run = run_twist_average_from_hamiltonians
        steps, warmup, bins = 20, 5, 3
        options["mcmc_kwargs"] = {"num_mcmc_per_measurement": 1}
    else:
        run = run_twist_average_gfmc
        steps, warmup, bins = GFMC_MIN_STEPS + 10, GFMC_MIN_WARMUP_STEPS, GFMC_MIN_BIN_BLOCKS
        options.update(gfmc_kind="tau", gfmc_kwargs={"alat": 0.5, "num_gfmc_collect_steps": GFMC_MIN_COLLECT_STEPS})
    settings = dict(num_mcmc_steps=steps, num_mcmc_warmup_steps=warmup, num_mcmc_bin_blocks=bins)
    checkpoint_dir = str(root / "checkpoints")
    first = run(checkpoint_dir=checkpoint_dir, **options, **settings)
    comm.Barrier()
    cached = Path(twist_checkpoint_path(checkpoint_dir, 0))
    missing = Path(twist_checkpoint_path(checkpoint_dir, 1))
    if rank == 0:
        missing.unlink()
    comm.Barrier()
    saved_bytes = cached.read_bytes()
    stored = ", ".join(f"'{key}': {value}" for key, value in settings.items())
    for changed in (
        {"num_mcmc_warmup_steps": warmup + 1},
        {"num_mcmc_bin_blocks": bins + 1},
        {"num_mcmc_steps": steps - 10},
    ):
        requested = ", ".join(f"'{key}': {value}" for key, value in {**settings, **changed}.items())
        message = (
            rf"{cached.name} \(twist 0\) was analysed with \{{{stored}\}}, but resume requests \{{{requested}\}}: refusing"
        )
        with pytest.raises(ValueError, match=message):
            run(checkpoint_dir=checkpoint_dir, resume=True, **options, **{**settings, **changed})
        comm.Barrier()
        assert cached.read_bytes() == saved_bytes
        assert not missing.exists()
    same = run(checkpoint_dir=checkpoint_dir, resume=True, **options, **settings)
    assert np.isfinite(first.per_twist_E).all() and first.per_twist_E[0] != first.per_twist_E[1]
    np.testing.assert_array_equal(same.per_twist_E, first.per_twist_E)
    np.testing.assert_array_equal(same.energy.E_TA, 0.25 * first.per_twist_E[0] + 0.75 * first.per_twist_E[1])
    assert cached.read_bytes() == saved_bytes


@pytest.mark.parametrize("route", ["mcmc", "gfmc_tau"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_completed_twist_resume_preserves_observables(tmp_path, monkeypatch, route):
    """Both cached H2 twists restore the uninterrupted forces, stress, pressure and MPC."""
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2
    rank = comm.Get_rank()
    root = Path(comm.bcast(str(tmp_path) if rank == 0 else None, root=0))
    run, sampler, options = _observable_run_options(route, comm)
    checkpoint_dir = str(root / "checkpoints")
    expected = run(checkpoint_dir=checkpoint_dir, **options)
    comm.Barrier()
    checkpoint = Path(twist_checkpoint_path(checkpoint_dir, rank))
    saved_bytes = checkpoint.read_bytes()
    saved_mtime = checkpoint.stat().st_mtime_ns
    calls = []
    real_run = sampler.run

    @wraps(real_run)
    def counted_run(driver, *args, **kwargs):
        calls.append(rank)
        return real_run(driver, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(sampler, "run", counted_run)
        actual = run(checkpoint_dir=checkpoint_dir, resume=True, **options)
    assert comm.allgather(len(calls)) == [0, 0]
    assert checkpoint.read_bytes() == saved_bytes and checkpoint.stat().st_mtime_ns == saved_mtime
    assert not np.array_equal(expected.per_twist_F[0], expected.per_twist_F[1])
    assert np.all(np.any(expected.per_twist_F != expected.per_twist_F_std, axis=(1, 2)))

    indexed = {
        "force": ("per_twist_F", (2, 2, 3)),
        "stress": ("per_twist_T", (2, 3, 3)),
        "P_hydro": ("per_twist_P_hydro", (2, 1)),
    }
    for quantity in ("force", "stress", "P_hydro"):
        name, shape = indexed[quantity]
        means, errors = getattr(expected, name), getattr(expected, f"{name}_std")
        reference = np.stack((means, errors))
        observed = np.stack((getattr(actual, name), getattr(actual, f"{name}_std")))
        assert observed.shape == reference.shape == (2, *shape)
        assert np.isfinite(observed).all() and np.isfinite(reference).all()
        np.testing.assert_array_equal(observed, reference, err_msg=f"indexed {quantity}")
        bz = getattr(actual, quantity)
        np.testing.assert_array_equal(np.stack((bz.F_TA, bz.F_TA_std)), _hand_weighted_twist_pair(means, errors))

    assert actual.mpc is not None and expected.mpc is not None
    assert actual.mpc.cutoff == expected.mpc.cutoff == 4.0
    for quantity in ("correction", "energy"):
        name = f"per_twist_{quantity}"
        means, errors = getattr(expected.mpc, name), getattr(expected.mpc, f"{name}_std")
        reference = np.stack((means, errors))
        observed = np.stack((getattr(actual.mpc, name), getattr(actual.mpc, f"{name}_std")))
        assert observed.shape == reference.shape == (2, 2)
        assert np.isfinite(observed).all() and np.isfinite(reference).all()
        np.testing.assert_array_equal(observed, reference, err_msg=f"indexed MPC {quantity}")
        bz = getattr(actual.mpc, quantity)
        np.testing.assert_array_equal(np.asarray([bz.E_TA, bz.E_TA_std]), _hand_weighted_twist_pair(means, errors))


@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_twist_stopped_on_max_time_is_rerun_on_resume(tmp_path, monkeypatch):
    """A twist stopped on max_time is not certified complete, and a resume finishes it.

    Same two-rank Gamma/k025 H2 inputs and options as the partial-resume test.
    Only the sampler's clock is changed: it passes ``max_time`` after twelve of
    the twenty measurements of each twist. That checkpoint must not be marked
    converged (otherwise a resume would skip it, and the requested
    measurements would never be taken). The resume then reruns both twists, and the
    per-twist values and the BZ average equal the uninterrupted run exactly.
    A real LRDMC run stopped the same way, on both projections, must behave the
    same, and a finished one must stay certified complete. No sampler,
    communication or reduction is mocked.
    """
    from types import SimpleNamespace

    import h5py

    from jqmc import _twist_average, jqmc_gfmc, jqmc_mcmc

    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2
    root = Path(comm.bcast(str(tmp_path) if comm.Get_rank() == 0 else None, root=0))
    hamiltonians = _read_h2_twists()
    num_mcmc_steps = 20
    options = dict(
        hamiltonian_list=hamiltonians,
        weights=[0.25, 0.75],
        mcmc_seed=12345,
        num_walkers=2,
        num_mcmc_steps=num_mcmc_steps,
        num_mcmc_warmup_steps=5,
        num_mcmc_bin_blocks=3,
        mcmc_kwargs={"num_mcmc_per_measurement": 1},
        world_comm=comm,
    )
    expected = run_twist_average_from_hamiltonians(checkpoint_dir=str(root / "uninterrupted"), **options)
    recovery = root / "stopped"
    real_run = MCMC.run

    def run_until_twelve(driver, *args, **kwargs):
        ticks = 0

        def clock():
            nonlocal ticks
            ticks += 1
            # Rows of the zero-filled sample buffer that hold a (nonzero) measurement.
            filled = np.count_nonzero(np.any(np.asarray(driver.e_L) != 0, axis=1))
            return (1000 if filled >= 12 else 0) + ticks * 1e-6

        with monkeypatch.context() as patch:
            patch.setattr(jqmc_mcmc, "time", SimpleNamespace(perf_counter=clock))
            return real_run(driver, *args, **kwargs)

    warnings = []
    with monkeypatch.context() as patch:
        patch.setattr(MCMC, "run", run_until_twelve)
        patch.setattr(_twist_average.logger, "warning", lambda message, *args, **kwargs: warnings.append(message))
        run_twist_average_from_hamiltonians(checkpoint_dir=str(recovery), max_time=50.0, **options)
    comm.Barrier()
    # The rank that writes the log reports every stopped twist, not only its own.
    stopped = [re.match(r"Twist (\d+) stopped on max_time after (\d+) of (\d+) ", m) for m in warnings]
    stopped = [(int(m.group(1)), int(m.group(2)), int(m.group(3))) for m in stopped if m]
    if comm.Get_rank() == 0:
        assert [twist for twist, _, _ in stopped] == [0, 1]
        assert all(done < requested == num_mcmc_steps for _, done, requested in stopped)
    else:
        assert stopped == []
    for index in range(2):
        with h5py.File(twist_checkpoint_path(str(recovery), index), "r") as handle:
            assert not bool(handle["_meta"].attrs["converged"]), f"twist {index} stopped early but is certified complete"
            assert int(handle["rank_0/driver_config"].attrs["mcmc_counter"]) < num_mcmc_steps
    comm.Barrier()
    actual = run_twist_average_from_hamiltonians(checkpoint_dir=str(recovery), resume=True, **options)
    comm.Barrier()
    for index in range(2):
        with h5py.File(twist_checkpoint_path(str(recovery), index), "r") as handle:
            assert bool(handle["_meta"].attrs["converged"])
            assert int(handle["rank_0/driver_config"].attrs["mcmc_counter"]) == num_mcmc_steps

    def assert_same_values(resumed, uninterrupted):
        for name in ("per_twist_E", "per_twist_E_std"):
            observed, reference = getattr(resumed, name), getattr(uninterrupted, name)
            assert observed.shape == reference.shape == (2,)
            assert np.isfinite(observed).all() and np.isfinite(reference).all()
            np.testing.assert_array_equal(observed, reference)
        observed_aggregate = np.asarray([resumed.energy.E_TA, resumed.energy.E_TA_std, resumed.energy.s_twist])
        expected_aggregate = np.asarray(
            [uninterrupted.energy.E_TA, uninterrupted.energy.E_TA_std, uninterrupted.energy.s_twist]
        )
        assert np.isfinite(observed_aggregate).all() and np.isfinite(expected_aggregate).all()
        np.testing.assert_array_equal(observed_aggregate, expected_aggregate)

    assert_same_values(actual, expected)

    # A finished LRDMC twist is complete: ``GFMC.mcmc_counter`` leaves out the
    # collect steps, so it must not be what the completeness check compares.
    warnings.clear()
    finished = root / "finished-gfmc"
    with monkeypatch.context() as patch:
        patch.setattr(_twist_average.logger, "warning", lambda message, *args, **kwargs: warnings.append(message))
        run_twist_average_gfmc(
            hamiltonians,
            [0.25, 0.75],
            mcmc_seed=12345,
            num_mcmc_steps=GFMC_MIN_STEPS,
            num_walkers=2,
            num_mcmc_warmup_steps=GFMC_MIN_WARMUP_STEPS,
            num_mcmc_bin_blocks=GFMC_MIN_BIN_BLOCKS,
            max_time=1e6,
            world_comm=comm,
            gfmc_kwargs={"alat": 0.5, "num_gfmc_collect_steps": GFMC_MIN_COLLECT_STEPS},
            checkpoint_dir=str(finished),
        )
    comm.Barrier()
    assert not [m for m in warnings if "stopped on max_time" in m]
    for index in range(2):
        with h5py.File(twist_checkpoint_path(str(finished), index), "r") as handle:
            assert bool(handle["_meta"].attrs["converged"]), f"finished LRDMC twist {index} is not certified complete"

    # A stopped LRDMC twist is not complete, whatever the projection. The clock
    # passes max_time at a fixed number of filled measurement rows of the 2 *
    # GFMC_MIN_STEPS requested, which leaves enough samples for the energy.
    def stopping_gfmc_run(real_run):
        # The twist driver passes max_time only to a ``run`` that names it.
        def run(driver, num_mcmc_steps=50, max_time=86400):
            ticks = 0

            def clock():
                nonlocal ticks
                ticks += 1
                filled = np.count_nonzero(np.asarray(driver._stored_e_L)[:, 0] != 0)
                return (1000 if filled >= GFMC_MIN_STEPS + 20 else 0) + ticks * 1e-6

            with monkeypatch.context() as patch:
                patch.setattr(jqmc_gfmc, "time", SimpleNamespace(perf_counter=clock))
                return real_run(driver, num_mcmc_steps=num_mcmc_steps, max_time=max_time)

        return run

    for gfmc_kind, driver_class in (("tau", jqmc_gfmc.GFMC_t), ("n", jqmc_gfmc.GFMC_n)):
        gfmc_kwargs = {"alat": 0.5, "num_gfmc_collect_steps": GFMC_MIN_COLLECT_STEPS}
        if gfmc_kind == "n":
            gfmc_kwargs["num_mcmc_per_measurement"] = 1
        gfmc_options = dict(
            hamiltonian_list=hamiltonians,
            weights=[0.25, 0.75],
            gfmc_kind=gfmc_kind,
            mcmc_seed=12345,
            num_walkers=2,
            num_mcmc_steps=2 * GFMC_MIN_STEPS,
            num_mcmc_warmup_steps=GFMC_MIN_WARMUP_STEPS,
            num_mcmc_bin_blocks=GFMC_MIN_BIN_BLOCKS,
            gfmc_kwargs=gfmc_kwargs,
            world_comm=comm,
        )
        expected = run_twist_average_gfmc(checkpoint_dir=str(root / f"gfmc-{gfmc_kind}-uninterrupted"), **gfmc_options)
        recovery = root / f"gfmc-{gfmc_kind}-stopped"
        with monkeypatch.context() as patch:
            patch.setattr(driver_class, "run", stopping_gfmc_run(driver_class.run))
            run_twist_average_gfmc(checkpoint_dir=str(recovery), max_time=50.0, **gfmc_options)
        comm.Barrier()
        for index in range(2):
            with h5py.File(twist_checkpoint_path(str(recovery), index), "r") as handle:
                assert not bool(handle["_meta"].attrs["converged"]), (
                    f"{gfmc_kind} twist {index} stopped early but is certified complete"
                )
                assert 0 < int(handle["rank_0/driver_config"].attrs["mcmc_counter"]) < 2 * GFMC_MIN_STEPS
        actual = run_twist_average_gfmc(checkpoint_dir=str(recovery), resume=True, **gfmc_options)
        comm.Barrier()
        for index in range(2):
            with h5py.File(twist_checkpoint_path(str(recovery), index), "r") as handle:
                assert bool(handle["_meta"].attrs["converged"])
        assert_same_values(actual, expected)

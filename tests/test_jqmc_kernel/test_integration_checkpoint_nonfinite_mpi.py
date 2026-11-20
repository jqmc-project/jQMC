"""Two-rank restart boundary for rank-local checkpoint damage.

Non-finite walker coordinates, invalid restored scalars, and (issues #805-#807)
restart state that one rank group lacks: each must fail identically on every
rank instead of leaving a peer blocked in a collective or running a different
algorithm on one rank.
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

import copy
import faulthandler
import hashlib
import logging
from pathlib import Path
import sys

import h5py
from mpi4py import MPI
import numpy as np
import pytest
import toml

from jqmc import jqmc_cli
from jqmc._checkpoint import load_rank_checkpoint, merge_rank_checkpoints
from jqmc._precision import is_mixed_precision_enabled
from jqmc._setting import GFMC_MIN_BIN_BLOCKS, GFMC_MIN_COLLECT_STEPS, GFMC_MIN_STEPS, GFMC_MIN_WARMUP_STEPS
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

pytestmark = [pytest.mark.light, pytest.mark.with_chdir, pytest.mark.mpi_only]

_TREXIO_FILE = Path(__file__).resolve().parents[1] / "trexio_example_files" / "H2_ae_ccpvdz_cart.h5"


def _build_hamiltonian() -> Hamiltonian_data:
    structure, _, _, _, geminal, coulomb = read_trexio_file(str(_TREXIO_FILE), store_tuple=True)
    return Hamiltonian_data(
        structure_data=structure,
        coulomb_potential_data=coulomb,
        wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data()),
    )


def _driver(driver_name: str, hamiltonian: Hamiltonian_data, comm: MPI.Comm):
    common = {"hamiltonian_data": hamiltonian, "num_walkers": 2, "mpi_comm_in": comm, "world_comm_in": comm}
    if driver_name == "MCMC":
        from jqmc.jqmc_mcmc import MCMC

        return MCMC(**common)
    from jqmc.jqmc_gfmc import GFMC_n, GFMC_t

    if driver_name == "GFMC_t":
        return GFMC_t(**common)
    return GFMC_n(**common, num_mcmc_per_measurement=2)


def _write_two_rank_checkpoint(root: Path, driver_name: str, comm: MPI.Comm, *, fault: str = "walker") -> Path:
    rank = comm.Get_rank()
    hamiltonian = _build_hamiltonian()
    driver = _driver(driver_name, hamiltonian, comm)
    directory = root / driver_name
    if rank == 0:
        directory.mkdir(parents=True)
    comm.Barrier()
    driver.save_to_hdf5(str(directory / f"._restart_rank{rank}.h5"))
    comm.Barrier()
    checkpoint = directory / "restart.h5"
    if rank == 0:
        merge_rank_checkpoints(
            output_path=str(checkpoint),
            mpi_size=2,
            driver_type=driver_name,
            hamiltonian_data=hamiltonian,
            tmp_pattern=str(directory / "._restart_rank{rank}.h5"),
        )
        with h5py.File(checkpoint, "r+") as handle:
            if fault in ("tau", "alat"):
                handle["rank_1/driver_config"].attrs[fault] = 0.0
            else:
                handle["rank_1/walker_state/latest_r_up_carts"][0, 0, 0] = np.nan
    comm.Barrier()
    return checkpoint


def test_rank_local_nonfinite_coordinate_fails_collectively_for_all_drivers(tmp_path):
    """MCMC and both GFMC loaders give every rank the same coordinate verdict."""
    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2
    root = Path(comm.bcast(str(tmp_path) if comm.Get_rank() == 0 else None, root=0))
    from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
    from jqmc.jqmc_mcmc import MCMC

    classes = {"MCMC": MCMC, "GFMC_t": GFMC_t, "GFMC_n": GFMC_n}
    for driver_name, driver_class in classes.items():
        checkpoint = _write_two_rank_checkpoint(root, driver_name, comm)
        try:
            driver_class.load_from_hdf5(
                str(checkpoint),
                rank=comm.Get_rank(),
                mpi_comm_in=comm,
                world_comm_in=comm,
            )
        except FloatingPointError as exc:
            message = str(exc)
        except BaseException as exc:  # reported collectively below
            message = f"WRONG_EXCEPTION {type(exc).__name__}: {exc}"
        else:
            message = "NO_EXCEPTION"
        messages = comm.allgather(message)
        required = (
            driver_name,
            "checkpoint load",
            "sampling_rank=1",
            "field=latest_r_up_carts",
            "walker=0",
            "electron=0",
            "component=0",
        )
        assert len(set(messages)) == 1, messages
        assert all(token in messages[0] for token in required), messages


def test_rank_local_invalid_tau_fails_collectively(tmp_path):
    """A bad interval on the non-root rank stops every rank before sampling."""
    from jqmc.jqmc_gfmc import GFMC_t

    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2
    root = Path(comm.bcast(str(tmp_path) if comm.Get_rank() == 0 else None, root=0))
    checkpoint = _write_two_rank_checkpoint(root, "GFMC_t", comm, fault="tau")
    try:
        GFMC_t.load_from_hdf5(str(checkpoint), mpi_comm_in=comm)
    except ValueError as exc:
        message = str(exc)
    else:
        message = "NO_EXCEPTION"
    messages = comm.allgather(message)
    assert len(set(messages)) == 1, messages
    assert "sampling_rank=1" in messages[0], messages
    assert "tau must be finite and > 0" in messages[0], messages


@pytest.mark.parametrize("driver_name", ["GFMC_t", "GFMC_n"])
def test_rank_local_invalid_alat_fails_collectively(tmp_path, driver_name):
    """A zero lattice spacing on the non-root rank stops every rank before sampling."""
    from jqmc.jqmc_gfmc import GFMC_n, GFMC_t

    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2
    root = Path(comm.bcast(str(tmp_path) if comm.Get_rank() == 0 else None, root=0))
    checkpoint = _write_two_rank_checkpoint(root, driver_name, comm, fault="alat")
    driver_cls = GFMC_t if driver_name == "GFMC_t" else GFMC_n
    try:
        driver_cls.load_from_hdf5(str(checkpoint), mpi_comm_in=comm)
    except ValueError as exc:
        message = str(exc)
    else:
        message = "NO_EXCEPTION"
    messages = comm.allgather(message)
    assert len(set(messages)) == 1, messages
    assert f"{driver_name} checkpoint load sampling_rank=1" in messages[0], messages
    assert "alat must be finite and > 0" in messages[0], messages


# ---------------------------------------------------------------------------
# Rank-asymmetric restart state through the actual CLI (issues #805-#807).
# A rank that branches alone on its restored state leaves its peer blocked in a
# collective; the watchdog turns such a hang into a failed run. It writes to the
# real stderr because ``--capture=tee-sys`` (the MPI test argv) has no fd.
# ---------------------------------------------------------------------------

_RESTART_HANG_TIMEOUT_SEC = 300


def _cli_job(root: Path, job_type: str, *, restart: bool, extra: dict | None = None) -> str | None:
    """Run one CLI job on both ranks; return the exception text, or ``None`` on success."""
    if job_type == "mcmc":
        params = {"num_mcmc_steps": 4 if restart else 8, "num_mcmc_per_measurement": 1}
        params.update(num_mcmc_warmup_steps=0, num_mcmc_bin_blocks=2)
    else:
        # A short continuation: a rank that restored a too-small counter would
        # find too few samples for the final output while its peer has enough.
        params = {
            "num_mcmc_steps": GFMC_MIN_STEPS // 2 if restart else GFMC_MIN_STEPS,
            "num_gfmc_warmup_steps": GFMC_MIN_WARMUP_STEPS,
            "num_gfmc_bin_blocks": GFMC_MIN_BIN_BLOCKS,
            "num_gfmc_collect_steps": GFMC_MIN_COLLECT_STEPS,
        }
    params.update(extra or {})
    payload = {
        "control": {
            "job_type": job_type,
            "mcmc_seed": 12345,
            "number_of_walkers": 2,
            "hamiltonian_h5": str(root / "hamiltonian.h5"),
            "restart": restart,
            "restart_chk": str(root / "restart.h5"),
            "verbosity": "low",
        },
        "precision": {"mode": "mixed" if is_mixed_precision_enabled() else "full"},
        job_type: params,
    }
    input_file = root / f"{job_type}-{'restart' if restart else 'fresh'}.toml"
    if MPI.COMM_WORLD.Get_rank() == 0:
        input_file.write_text(toml.dumps(payload), encoding="utf-8")
    MPI.COMM_WORLD.Barrier()
    logger = logging.getLogger("jqmc")
    handlers, level = list(logger.handlers), logger.level
    patch = pytest.MonkeyPatch()
    try:
        patch.chdir(root)
        patch.setattr(sys, "argv", ["jqmc", str(input_file)])
        patch.setattr(jqmc_cli, "cli_parameters", copy.deepcopy(jqmc_cli.cli_parameters))
        jqmc_cli._cli()
    except Exception as exc:  # compared across ranks by the caller
        return f"{type(exc).__name__}: {exc}"
    finally:
        patch.undo()
        for handler in logger.handlers:
            if handler not in handlers and isinstance(handler, logging.FileHandler):
                handler.close()
        logger.handlers, logger.level = handlers, level
    return None


def _restart_damaged_checkpoint(tmp_path, job_type, damage, *, trexio_name="H2_ae_ccpvdz_cart.h5", extra=None):
    """Write a real two-rank checkpoint, damage it on rank 0, restart; return (messages, unchanged)."""
    from tests.test_jqmc_kernel.test_cli_restart_values import _write_hamiltonian

    comm = MPI.COMM_WORLD
    assert comm.Get_size() == 2, "launch this MPI-only contract with mpirun -np 2"
    root = Path(comm.bcast(str(tmp_path) if comm.Get_rank() == 0 else None, root=0))
    if comm.Get_rank() == 0:
        _write_hamiltonian(root, trexio_name)
    comm.Barrier()
    assert _cli_job(root, job_type, restart=False, extra=extra) is None
    checkpoint = root / "restart.h5"
    digest = None
    if comm.Get_rank() == 0:
        with h5py.File(checkpoint, "r+") as handle:
            damage(handle)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    digest = comm.bcast(digest, root=0)
    faulthandler.dump_traceback_later(_RESTART_HANG_TIMEOUT_SEC, exit=True, file=sys.__stderr__)
    try:
        message = _cli_job(root, job_type, restart=True, extra=extra)
    finally:
        faulthandler.cancel_dump_traceback_later()
    messages = comm.allgather(message)
    unchanged = hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    return messages, unchanged, checkpoint


def _assert_rejected_identically(messages, unchanged, *tokens):
    assert messages[0] is not None and messages[0].startswith("ValueError: "), messages
    assert len(set(messages)) == 1, messages
    assert all(token in messages[0] for token in tokens), messages
    assert unchanged, "a rejected restart must leave restart.h5 untouched"


@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
@pytest.mark.parametrize("missing_on", ["rank_1", "all"])
def test_restart_optimizer_state_presence_is_decided_collectively(tmp_path, missing_on):
    """``optimizer_state`` absent on one rank fails on every rank; absent on all ranks starts one fresh lineage.

    Committed H2 AE input with Pade J2, an ``mcmc`` job with two walkers per
    rank and eight measurements, then a four-measurement restart. If only the
    damaged rank broadcast a new lineage id, the run would hang. When
    every rank lacks the group (a legitimate fresh lineage), the restart
    succeeds and both ranks write the same new lineage id.
    """
    ranks = (1,) if missing_on == "rank_1" else (0, 1)

    def damage(handle):
        for rank in ranks:
            del handle[f"rank_{rank}/optimizer_state"]

    messages, unchanged, checkpoint = _restart_damaged_checkpoint(tmp_path, "mcmc", damage)
    if missing_on == "rank_1":
        _assert_rejected_identically(messages, unchanged, "rank_1='absent'", "optimizer_state differs across ranks")
        return
    assert messages == [None, None], messages
    restored = [load_rank_checkpoint(str(checkpoint), rank=rank) for rank in (0, 1)]
    assert [data["driver_config"]["mcmc_counter"] for data in restored] == [12, 12]
    lineages = {data["optimizer_state"]["wavefunction_lineage_id"] for data in restored}
    assert len(lineages) == 1 and all(lineages), lineages


@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_lrdmc_restart_without_rank_counter_fails_on_every_rank(tmp_path):
    """``mcmc_counter`` absent on rank 1 of an ``lrdmc-tau`` checkpoint is rejected on both ranks.

    Same H2 AE input, GFMC_t with the minimum step policy. A damaged
    rank that defaulted its counter to 0 would skip the collective ``get_E``
    and leave rank 0 blocked in it (the continuation is too short for a
    restored counter of 0 to reach the final output).
    """

    def damage(handle):
        del handle["rank_1/driver_config"].attrs["mcmc_counter"]

    messages, unchanged, _ = _restart_damaged_checkpoint(tmp_path, "lrdmc-tau", damage)
    _assert_rejected_identically(messages, unchanged, "rank_1/driver_config/mcmc_counter", "required restart key is missing")


@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_ecp_lrdmc_restart_without_rank_non_local_move_fails_on_every_rank(tmp_path):
    """``non_local_move`` absent on rank 1 of an ECP ``dltmove`` checkpoint is rejected on both ranks.

    Committed H2 ECP input with Pade J2, ``lrdmc-tau`` with an explicit
    ``non_local_move = "dltmove"`` in both TOMLs. Rank 1 must not silently
    run T-moves while the log reports dltmove.
    """

    def damage(handle):
        del handle["rank_1/driver_config"].attrs["non_local_move"]

    messages, unchanged, _ = _restart_damaged_checkpoint(
        tmp_path, "lrdmc-tau", damage, trexio_name="H2_ecp_ccpvdz_cart.h5", extra={"non_local_move": "dltmove"}
    )
    _assert_rejected_identically(messages, unchanged, "rank_1/driver_config", "non_local_move")

"""Deterministic restart-topology checks, without loading numerical payloads."""

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
import logging
import sys

import h5py
import pytest
import toml

from jqmc import _checkpoint
from jqmc._precision import is_mixed_precision_enabled
from jqmc._setting import GFMC_MIN_BIN_BLOCKS, GFMC_MIN_COLLECT_STEPS, GFMC_MIN_WARMUP_STEPS
from jqmc import jqmc_cli
from jqmc.jqmc_mcmc import MCMC
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _checkpoint_topology(path, stored_size, rank_names):
    """Write only the explicit rank groups and metadata to a temporary HDF5 file."""
    with h5py.File(path, "w") as checkpoint:
        meta = checkpoint.create_group("_meta")
        if stored_size is not None:
            meta.attrs["mpi_size"] = stored_size
        for name in rank_names:
            checkpoint.create_group(name)
    return str(path)


class _SamplingCommunicator:
    def __init__(self, size, rank=0):
        self.size, self.rank = size, rank

    def Get_size(self):
        return self.size

    def Get_rank(self):
        return self.rank


@pytest.mark.parametrize("size", [1, 2, 3])
def test_matching_checkpoint_rank_topology_is_accepted(tmp_path, size):
    """One, two and three contiguous rank groups agree with metadata and sampling size.

    This parameter-set fixture contains only temporary HDF5 metadata and groups;
    no wavefunction, electron positions, random sampling or numerical payload is used.
    """
    path = _checkpoint_topology(tmp_path / "restart.h5", size, tuple(f"rank_{rank}" for rank in range(size)))
    assert _checkpoint._validate_restart_rank_topology(path, size) is None


@pytest.mark.parametrize(
    "requested_size,stored_size,rank_names,reason",
    [
        (1, 2, ("rank_0", "rank_1"), "Restart MPI size mismatch"),
        (2, 1, ("rank_0",), "Restart MPI size mismatch"),
        (2, 2, ("rank_0",), "metadata does not match"),
        (2, 2, ("rank_0", "rank_2"), "must be contiguous"),
        (2, 2, ("rank_0", "rank_01"), "must be contiguous"),
        (2, None, ("rank_0", "rank_1"), "mpi_size must be a positive integer"),
        (2, True, ("rank_0", "rank_1"), "mpi_size must be a positive integer"),
        (2, 2.0, ("rank_0", "rank_1"), "mpi_size must be a positive integer"),
        (2, "2", ("rank_0", "rank_1"), "mpi_size must be a positive integer"),
        (2, 0, ("rank_0", "rank_1"), "mpi_size must be a positive integer"),
        (2, 1000000000000, ("rank_0", "rank_1"), "metadata does not match"),
        (0, 2, ("rank_0", "rank_1"), "communicator size must be a positive integer"),
    ],
)
def test_invalid_checkpoint_rank_topology_is_rejected(tmp_path, requested_size, stored_size, rank_names, reason):
    """Reject rank-count changes, missing/noncanonical groups and malformed metadata.

    Each literal tuple defines the requested sampling size, stored mpi_size and
    complete rank-group set in a temporary HDF5 file. Missing, boolean, floating,
    string, nonpositive and enormous metadata cannot silently redefine a restart.
    Only metadata is read; no Hamiltonian, electrons or sampler values are involved.
    """
    path = _checkpoint_topology(tmp_path / "restart.h5", stored_size, rank_names)
    with pytest.raises(ValueError, match=reason):
        _checkpoint._validate_restart_rank_topology(path, requested_size)


@pytest.mark.parametrize("malformation", ["missing-meta", "meta-dataset", "rank-dataset"])
def test_checkpoint_rank_topology_rejects_non_group_entries(tmp_path, malformation):
    """Rank payloads and the metadata container must be HDF5 groups, not datasets."""
    path = _checkpoint_topology(tmp_path / "restart.h5", 1, ("rank_0",))
    with h5py.File(path, "a") as checkpoint:
        name = "rank_0" if malformation == "rank-dataset" else "_meta"
        del checkpoint[name]
        if malformation != "missing-meta":
            checkpoint.create_dataset(name, data=1)
    with pytest.raises(ValueError, match="metadata|HDF5 groups"):
        _checkpoint._validate_restart_rank_topology(path, 1)


@pytest.mark.parametrize("driver", ["MCMC", "GFMC_n", "GFMC_t"])
@pytest.mark.parametrize("stored_size,requested_size", [(2, 1), (1, 2)])
def test_driver_restart_rejects_changed_rank_count_before_payload(tmp_path, monkeypatch, driver, stored_size, requested_size):
    """Every public sampler loader rejects fewer or more ranks before reading state.

    The HDF5 fixture has explicit rank metadata; a communicator double supplies
    only size/rank. A forbidden-payload sentinel proves rejection precedes state
    loading. This is a routing/precondition test, not an MPI numerical restart.
    """
    path = _checkpoint_topology(tmp_path / "restart.h5", stored_size, tuple(f"rank_{rank}" for rank in range(stored_size)))
    driver_class = {"MCMC": MCMC, "GFMC_n": GFMC_n, "GFMC_t": GFMC_t}[driver]

    def forbidden_payload(*args, **kwargs):
        pytest.fail("payload read before rejecting the restart rank mismatch")

    monkeypatch.setattr(_checkpoint, "load_rank_checkpoint", forbidden_payload)
    with pytest.raises(ValueError, match="Restart MPI size mismatch"):
        driver_class.load_from_hdf5(path, mpi_comm_in=_SamplingCommunicator(requested_size))


@pytest.mark.parametrize("driver", ["MCMC", "GFMC_n", "GFMC_t"])
def test_driver_restart_validates_sampling_not_world_communicator(tmp_path, monkeypatch, driver):
    """A two-rank twist checkpoint binds its sampling communicator, not a four-rank world.

    The real loader must reach payload loading with sampling rank one, without
    rejecting the different world size. A sentinel ends the test before numerical
    state loading; real multi-rank restart values require separate integration evidence.
    """
    path = _checkpoint_topology(tmp_path / "restart.h5", 2, ("rank_0", "rank_1"))
    driver_class = {"MCMC": MCMC, "GFMC_n": GFMC_n, "GFMC_t": GFMC_t}[driver]
    seen_ranks = []
    seen_paths = []

    def reached_payload(filepath, rank):
        seen_paths.append(filepath)
        seen_ranks.append(rank)
        raise RuntimeError("reached the validated payload")

    monkeypatch.setattr(_checkpoint, "load_rank_checkpoint", reached_payload)
    with pytest.raises(RuntimeError, match="reached the validated payload"):
        driver_class.load_from_hdf5(path, mpi_comm_in=_SamplingCommunicator(2, 1), world_comm_in=_SamplingCommunicator(4, 3))
    assert seen_ranks == [1]
    assert seen_paths == [path]


@pytest.mark.parametrize("job_type", ["mcmc", "vmc", "lrdmc-bra", "lrdmc-tau"])
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Restart topology is rejected before frozen config or numerical payload loading."
)
def test_cli_restart_rejects_rank_mismatch_before_config_read(tmp_path, monkeypatch, job_type):
    """All four CLI jobs reject a changed rank count before frozen-config loading.

    A temporary checkpoint declares one more rank than COMM_WORLD. The TOML
    requests restart in the current precision mode; no scientific sampling runs.
    This covers real CLI dispatch, not an MPI energy or waveform assertion.
    """
    size = jqmc_cli.MPI.COMM_WORLD.Get_size() + 1
    checkpoint = _checkpoint_topology(tmp_path / "restart.h5", size, tuple(f"rank_{rank}" for rank in range(size)))
    parameters = {
        "control": {
            "job_type": job_type,
            "restart": True,
            "restart_chk": checkpoint,
            "hamiltonian_h5": str(tmp_path / "unused.h5"),
            "verbosity": "low",
        },
        "precision": {"mode": "mixed" if is_mixed_precision_enabled() else "full"},
        job_type: {
            "num_mcmc_steps": 100,
            "num_opt_steps": 1,
            "num_gfmc_warmup_steps": GFMC_MIN_WARMUP_STEPS,
            "num_gfmc_bin_blocks": GFMC_MIN_BIN_BLOCKS,
            "num_gfmc_collect_steps": GFMC_MIN_COLLECT_STEPS,
        },
    }
    toml_path = tmp_path / "restart.toml"
    toml_path.write_text(toml.dumps(parameters))
    monkeypatch.setattr(sys, "argv", ["jqmc", str(toml_path)])
    monkeypatch.setattr(jqmc_cli, "cli_parameters", copy.deepcopy(jqmc_cli.cli_parameters))

    def forbidden_config(*args, **kwargs):
        pytest.fail("configuration read before rejecting the restart rank mismatch")

    monkeypatch.setattr(_checkpoint, "load_driver_config_from_checkpoint", forbidden_config)
    logger = logging.getLogger("jqmc")
    handlers, level = list(logger.handlers), logger.level
    try:
        with pytest.raises(ValueError, match="Restart MPI size mismatch"):
            jqmc_cli._cli()
    finally:
        logger.handlers = handlers
        logger.setLevel(level)

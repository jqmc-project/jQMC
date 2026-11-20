"""Two twists of two ranks each: the twist-local device SR solve stays inside its twist.

The test launches the production twist-decomposed optimizer target under
``mpirun -n 4`` (CPU JAX multi-process, ``jax.distributed`` over the four
ranks) and compares, on every rank, the device-batch update with the host-batch
update of the same samples. The sampler-owned coordinates of a twist with two
ranks take the bounded streaming device CG, so the device solve must reduce
over exactly that twist's sampler communicator. A solve that reduced over every
JAX process would mix the other twist's Gram matrix into this twist's system.

The comparison itself runs here, in the launching pytest process, on the
arrays the four ranks write.
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

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from jqmc._precision import is_mixed_precision_enabled
from jqmc._setting import ATOL_C9_GAMMA_SR_UPDATE, RTOL_C9_GAMMA_SR_UPDATE

pytestmark = [pytest.mark.without_chdir]

_WORLD_SIZE = 4
_RANKS_PER_TWIST = 2
_COORDINATES = 5
_SAMPLES_PER_RANK = 6
_SUBPROCESS_TIMEOUT_S = 900
_MPIRUN = shutil.which("mpirun") or shutil.which("mpiexec")
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _rank_batches(rank):
    """Host and device batches of one rank's real sampler-owned samples."""
    from jqmc.jqmc_optimizer import (
        BlockDerivativeSource,
        JackknifeBinMetadata,
        JackknifeBlockMoments,
        OptimizationBatch,
        OptimizationCoordinateMap,
    )

    from .optimizer_metadata_helpers import TEST_STATE_TOKEN, make_test_block
    from .test_optimizer_lm_real import _device_batch

    rng = np.random.default_rng(8940 + rank)
    blocks = tuple(
        make_test_block(f"local{index}", np.asarray(0.0), field="real", derivative_policy="direct_real", sync_scope="sampler")
        for index in range(_COORDINATES)
    )
    energies = rng.normal(size=_SAMPLES_PER_RANK)
    values = tuple(rng.normal(size=_SAMPLES_PER_RANK) for _ in blocks)

    def source(block, array):
        return BlockDerivativeSource.build(block=block, direct=array)

    moments = tuple(
        JackknifeBlockMoments.build(
            block=block,
            source_sum=source(block, array),
            conjugate_source_energy_sum=source(block, array * energies),
        )
        for block, array in zip(blocks, values, strict=True)
    )
    token = "twist-sampler-device-sr"
    metadata = JackknifeBinMetadata.build(
        blocks=blocks,
        batch_token=token,
        bin_sample_counts=np.ones(_SAMPLES_PER_RANK, dtype=np.int64),
        bin_weight_sums=np.ones(_SAMPLES_PER_RANK),
        bin_weighted_energy_sums=energies,
        log_source_moments=moments,
    )
    sources = tuple(source(block, array) for block, array in zip(blocks, values, strict=True))
    host = OptimizationBatch.build(
        blocks=blocks,
        batch_token=token,
        state_token=TEST_STATE_TOKEN,
        weights=np.ones(_SAMPLES_PER_RANK),
        local_energies=energies,
        log_derivative_sources=sources,
        local_energy_derivative_sources=sources,
        jackknife_bin_metadata=metadata,
    )
    return host, _device_batch(blocks, host), OptimizationCoordinateMap.from_blocks(blocks)


def _worker(output: str, precision: str) -> None:
    """One rank of the four-rank run: device and host targets of this rank's twist."""
    import jax
    from mpi4py import MPI

    from jqmc._driver_helpers import _initialize_jax_distributed
    from jqmc._precision import configure
    from jqmc.jqmc_optimizer import compute_twist_decomposed_real_coordinate_optimization_target

    configure(precision)
    _initialize_jax_distributed()
    world = MPI.COMM_WORLD
    if world.size != _WORLD_SIZE or jax.process_count() != _WORLD_SIZE:
        raise RuntimeError(f"requires {_WORLD_SIZE} MPI ranks and JAX processes; got {world.size}/{jax.process_count()}")
    twist = world.rank // _RANKS_PER_TWIST
    sampler = world.Split(twist, world.rank)
    host, device, coordinate_map = _rank_batches(world.rank)
    targets = [
        compute_twist_decomposed_real_coordinate_optimization_target(
            batch,
            coordinate_map,
            sampler_comm=sampler,
            world_comm=world,
            epsilon=0.05,
            diagonal_floor=0.0,
            cg=True,
            cg_max_iter=200,
            cg_tol=1.0e-14,
            use_lm=False,
            lm_subspace_dim=0,
            lm_condition_cutoff=1.0e-8,
            world_energy_standard_error=0.0,
            sampler_energy_standard_error=0.0,
        )
        for batch in (device, host)
    ]
    gathered = world.gather(
        (twist, sampler.size, *(np.asarray(target.sr_q) for target in targets), targets[0].sr_regime), root=0
    )
    if world.rank == 0:
        twists, sampler_sizes, device_q, host_q, regimes = zip(*gathered, strict=True)
        np.savez(output, twists=twists, sampler_sizes=sampler_sizes, device_q=device_q, host_q=host_q, regimes=regimes)
    sampler.Free()


@pytest.mark.heavy
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Synthetic sampler-owned SR samples; no numerical model or electron positions."
)
def test_multi_rank_twist_local_device_sr_matches_host_route(tmp_path):
    """Each twist's sampler-scope device CG reproduces the host update of that twist alone."""
    assert _MPIRUN is not None, "the four-rank twist run needs mpirun or mpiexec on PATH"
    output = tmp_path / "targets.npz"
    env = dict(os.environ, JQMC_FLAG_RUST_ON_CPU="0", PYTHONUNBUFFERED="1", OMPI_MCA_hwloc_base_binding_policy="none")
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        env.pop(name, None)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(_REPO_ROOT), env.get("PYTHONPATH"))))
    precision = "mixed" if is_mixed_precision_enabled() else "full"
    command = [_MPIRUN, "-n", str(_WORLD_SIZE), sys.executable, "-m", __name__, str(output), precision]
    run = subprocess.run(command, cwd=_REPO_ROOT, env=env, capture_output=True, text=True, timeout=_SUBPROCESS_TIMEOUT_S)
    assert run.returncode == 0, run.stdout + run.stderr
    result = np.load(output)
    np.testing.assert_array_equal(result["twists"], [0, 0, 1, 1])
    np.testing.assert_array_equal(result["sampler_sizes"], [_RANKS_PER_TWIST] * _WORLD_SIZE)
    host_q = result["host_q"]
    # The two twists solve different systems, so a solve that mixed them is visible.
    assert np.max(np.abs(host_q[0] - host_q[2])) > ATOL_C9_GAMMA_SR_UPDATE + RTOL_C9_GAMMA_SR_UPDATE * np.max(np.abs(host_q))
    np.testing.assert_allclose(result["device_q"], host_q, atol=ATOL_C9_GAMMA_SR_UPDATE, rtol=RTOL_C9_GAMMA_SR_UPDATE)
    # The comparison above ran the multi-rank twist-local streaming route.
    assert all("sampler:wide-device-staged-mpi" in regime for regime in result["regimes"]), result["regimes"]


if __name__ == "__main__":
    _worker(*sys.argv[1:])

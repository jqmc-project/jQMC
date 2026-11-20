"""QE multi-twist composite-kind twist-averaged MCMC under MPI.

Every supported
projection ``kind`` (``pgto``, ``pw``, ``bspline``, ``pgto+pw``,
``pgto+bspline``) must round-trip through

    QE save -> ``convert_qe_to_twist_average_input`` (combined HDF5) ->
    ``run_twist_average`` (twist-averaged MCMC under MPI) ->
    finite ``E_TA``

without firing the ``ComplexWarning`` autouse guard or producing
``NaN``. This file exercises that contract under a genuine
``MPI.COMM_WORLD`` (the multi-rank twist split + Bloch composite kernels
must agree).

Launching convention (deliberately NOT subprocess-based; the launcher
command is not guaranteed to be ``mpirun`` on every machine -- HPC
schedulers use ``srun`` / ``mpiexec`` / ``jsrun`` / ``flux run``):

    mpirun -np N pytest -v tests/test_jqmc_kernel/test_integration_qe_twist_average_mpi.py

``world_size`` must be a multiple of every ``n_twist`` exercised here
(equal compute per twist). The current
``Li_k025`` fixture has ``n_twist = 1`` so any ``world_size`` works; a
run on the full ``Li_k_twist`` (8 k-points) fixture needs ``-np N`` a
multiple of 8 (4 ranks/twist at ``-np 32``, 8 ranks/twist at ``-np 64``,
etc.).
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

import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mpi4py import MPI

from jqmc.atomic_orbital import (
    AOs_bspline_complex_data,
    AOs_composite_data,
    AOs_plane_wave_complex_data,
    is_complex_aos,
)
from tests._test_data_cache import qe_example_files_root

_FIXTURE_DIR = qe_example_files_root() / "Li_k025" / "out" / "li_k025.save"
_UPF_PATH = _FIXTURE_DIR / "Li.ccECP.upf"
_available = (
    (_FIXTURE_DIR / "data-file-schema.xml").is_file() and (_FIXTURE_DIR / "wfc1.hdf5").is_file() and _UPF_PATH.is_file()
)

requires_k025 = pytest.mark.skipif(not _available, reason="Li_k025 QE fixture not present")

# Read MPI.COMM_WORLD up front -- the test launcher (regular pytest or
# ``mpirun -np N pytest``) determines the world size. ``n_twist`` for
# the Li_k025 fixture is 1, which divides every positive world_size.
_world = MPI.COMM_WORLD
_world_size = _world.Get_size()
_world_rank = _world.Get_rank()


class TestQETwistAverageMPI:
    @requires_k025
    @pytest.mark.parametrize(
        "kind",
        ["pgto", "pw", "bspline", "pgto+pw", "pgto+bspline"],
    )
    @pytest.mark.heavy
    def test_twist_average_mcmc_smoke(self, tmp_path, kind):
        """Twist-averaged MCMC smoke at every supported ``kind`` under MPI.

        Pipeline (rank 0 writes the combined input; every rank participates in
        the twist-averaged MCMC across the comm split):

        1. Rank 0 calls ``convert_qe_to_twist_average_input`` on the
           Li_k025 fixture for the requested ``kind`` (n_twist=1; works at
           any world_size).
        2. All ranks barrier on the file write.
        3. All ranks call ``run_twist_average`` with a short chain
           (``num_mcmc_steps=30``, ``num_walkers=2``); ``world_comm`` is
           ``MPI.COMM_WORLD`` so the runner's
           :func:`split_world_for_twists` picks up the launched
           ``world_size``.
        4. Rank 0 asserts ``E_TA`` is finite and the per-twist AO leaf
           still satisfies :func:`is_complex_aos`.

        The ``MPI.COMM_WORLD`` use (instead of a subprocess that calls
        ``mpirun`` internally) lets the launcher choice flow from the
        caller without hard-coding ``shutil.which("mpirun")`` inside the
        test. The autouse ``ComplexWarning`` guard in ``conftest.py``
        catches any silent complex -> real cast inside the e_L chain.
        """
        import jax
        import jax.numpy as jnp

        from jqmc._twist_average import is_twist_average_input, load_twist, run_twist_average
        from jqmc.qe_to_jqmc import convert_qe_to_twist_average_input
        from jqmc.upf_wrapper import parse_gaussinfo

        # Multi-rank gotcha: ``jax.distributed.initialize`` (set
        # up session-wide by conftest under ``mpirun -np N``) only opens the
        # JAX coordination service; the per-rank CPU backend topology is
        # registered lazily on the first JAX op. If rank 0 triggers JAX
        # later than rank 1 (or vice versa) -- e.g. because only rank 0
        # runs the heavy ``convert_qe_to_twist_average_input`` step below
        # while rank 1 sits at the next Barrier -- the late rank's
        # ``GetKeyValue(cpu:local_topology/cpu/<rank>)`` blocks for
        # ``coordination_service_timeout_milliseconds`` (default 2 min) and
        # times out. The tiny warmup here forces every rank to register its
        # backend topology BEFORE the rank-0-only critical section, so the
        # coordination exchange happens at a synchronisation point both
        # ranks pass through.
        _ = jnp.sum(jnp.arange(2)).block_until_ready()
        _world.Barrier()

        # Shared per-rank tmp path: every rank must agree on the combined file
        # location. ``tmp_path`` is per-rank under pytest, so broadcast
        # rank-0's choice.
        out_path = str(tmp_path / f"{kind.replace('+', '_')}_mt.h5") if _world_rank == 0 else None
        out_path = _world.bcast(out_path, root=0)

        if _world_rank == 0:
            ecp = parse_gaussinfo(str(_UPF_PATH))
            convert_qe_to_twist_average_input(
                qe_save_dir=str(_FIXTURE_DIR),
                output_path=out_path,
                upf_paths={"Li": str(_UPF_PATH)},
                kind=kind,
                ecp_by_element={"Li": ecp},
                j1_param=None,
                j2_param=None,
            )
            assert is_twist_average_input(out_path), "writer did not produce a combined file"
            # Eagerly load on rank 0 too so we can validate the composite
            # before the MCMC chain starts.
            ham = load_twist(out_path, 0)
            ham.sanity_check()
            aos = ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data
            assert is_complex_aos(aos), f"AO leaf must be complex at non-Gamma; got {type(aos).__name__}"
            if kind in ("pgto+pw", "pgto+bspline"):
                assert isinstance(aos, AOs_composite_data)
            elif kind == "pw":
                assert isinstance(aos, AOs_plane_wave_complex_data)
            elif kind == "bspline":
                assert isinstance(aos, AOs_bspline_complex_data)
        # Barrier so non-zero ranks see the combined file before opening it.
        _world.Barrier()

        # Twist-averaged MCMC across the comm split. ``num_mcmc_steps`` is
        # tiny -- this is a wire / dtype contract test, not a physics
        # benchmark.
        result = run_twist_average(
            twist_input_path=out_path,
            mcmc_seed=12345,
            num_mcmc_steps=30,
            num_walkers=2,
            num_mcmc_warmup_steps=5,
            num_mcmc_bin_blocks=5,
            world_comm=_world,
        )

        # Under N_twist=1 the Level-2 BZ combine returns the same aggregate on
        # every rank.  Assert that postcondition on every participant so a rank
        # cannot issue an assertion-free GDD receipt.
        assert np.isfinite(result.energy.E_TA), f"E_TA must be finite; got {result.energy.E_TA!r}"
        assert np.isfinite(result.energy.E_TA_std), f"E_TA_std must be finite; got {result.energy.E_TA_std!r}"
        assert all(np.isfinite(result.per_twist_E)), "per_twist_E carries non-finite entries"
        assert all(np.isfinite(result.per_twist_E_std)), "per_twist_E_std carries non-finite entries"

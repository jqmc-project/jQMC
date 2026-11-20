"""QE non-pGTO orbital families through the twist-average MCMC driver: value oracle.

``test_integration_qe_twist_average_mpi`` runs every QE projection ``kind``
through ``run_twist_average`` under MPI but asserts only types and finiteness,
and the per-twist decomposition oracle in ``test_twist_decomposition_mpi`` runs
only ``kind="pgto"``.  This module extends that oracle to the plane-wave,
B-spline and composite families (``pw``, ``bspline``, ``pgto+pw``,
``pgto+bspline``).

For each kind, the committed ``Li_gamma`` (Gamma) and ``Li_k025`` (non-Gamma,
complex Bloch leaf) QE saves are converted in-process with
``convert_qe_to_twist_average_input`` and loaded with ``load_twist``.  The SAME
seeds then run twice on the SAME communicator layout:

1. standalone: one direct ``MCMC`` chain for this rank's twist on the intra
   communicator that :func:`split_world_for_twists` assigns;
2. driver: :func:`run_twist_average_from_hamiltonians` over both twists with
   the non-uniform raw weights ``(0.75, 0.25)``.

The per-twist energies and error bars must match exactly, and ``E_TA`` /
``E_TA_std`` / ``s_twist`` must equal the hand-rolled weighted arithmetic on
those per-twist values.  The two twist energies differ, so a weight reversal or
a twist/weight mis-pairing in the BZ combine moves ``E_TA`` by more than its
pin (asserted on the data, not assumed).

Launch (two twists need an even world): ``mpirun -np 2 pytest -q
tests/test_jqmc_kernel/test_qe_twist_decomposition_nonpgto_mpi.py``.  The file
runs only under MPI (``mpi_only``).
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

pytestmark = [pytest.mark.with_chdir, pytest.mark.heavy, pytest.mark.mpi_only]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mpi4py import MPI

from jqmc._setting import ATOL_PYTHON_FP64_BZ_MEAN, ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_ZERO
from jqmc.atomic_orbital import (
    AOs_bspline_complex_data,
    AOs_composite_data,
    AOs_plane_wave_complex_data,
    is_complex_aos,
)
from tests._test_data_cache import qe_example_files_root

_FIX = qe_example_files_root()
_SAVES = (
    _FIX / "Li_gamma" / "out" / "li.save",
    _FIX / "Li_k025" / "out" / "li_k025.save",
)
_UPF = _FIX / "Li.ccECP.upf"

_world = MPI.COMM_WORLD
_world_rank = _world.Get_rank()

_N_TWIST = 2
_RAW_WEIGHTS = [0.75, 0.25]
_SEED = 24680
_NUM_WALKERS = 2
_NUM_STEPS = 30
_NUM_WARMUP = 10
_NUM_BINS = 5

_COMPLEX_LEAF = {
    "pw": AOs_plane_wave_complex_data,
    "bspline": AOs_bspline_complex_data,
    "pgto+pw": AOs_composite_data,
    "pgto+bspline": AOs_composite_data,
}


def _build_hamiltonians(tmp_dir: str, kind: str):
    """Rank 0 converts both saves for ``kind``; every rank loads both twists."""
    import jax.numpy as jnp

    from jqmc._twist_average import load_twist
    from jqmc.qe_to_jqmc import convert_qe_to_twist_average_input
    from jqmc.upf_wrapper import parse_gaussinfo

    # Register every rank's JAX backend before the rank-0-only section (see
    # test_integration_qe_twist_average_mpi for the coordination-service gotcha).
    _ = jnp.sum(jnp.arange(2)).block_until_ready()
    _world.Barrier()
    tag = kind.replace("+", "_")
    paths = [f"{tmp_dir}/li_{tag}_twist{i}.h5" for i in range(_N_TWIST)]
    if _world_rank == 0:
        ecp = parse_gaussinfo(str(_UPF))
        for save, path in zip(_SAVES, paths, strict=True):
            convert_qe_to_twist_average_input(
                qe_save_dir=str(save),
                output_path=path,
                upf_paths={"Li": str(_UPF)},
                kind=kind,
                ecp_by_element={"Li": ecp},
                j1_param=None,
                j2_param=1.0,
            )
    _world.Barrier()
    return [load_twist(path, 0) for path in paths]


@pytest.mark.parametrize("kind", ["pw", "bspline", "pgto+pw", "pgto+bspline"])
def test_qe_nonpgto_twist_driver_decomposes_into_per_twist_chains(tmp_path, kind):
    """Driver per-twist E == standalone chains; E_TA == sum_k W_k E_k with non-uniform W (pw/bspline/composite)."""
    from jqmc._twist_average import (
        _twist_k_point_from_hamiltonian,
        _twist_sampler_seed,
        run_twist_average_from_hamiltonians,
        split_world_for_twists,
    )
    from jqmc.jqmc_mcmc import MCMC

    tmp_dir = _world.bcast(str(tmp_path) if _world_rank == 0 else None, root=0)
    hams = _build_hamiltonians(tmp_dir, kind)

    # The QE route really reached the non-pGTO family on the non-Gamma twist.
    aos_k = hams[1].wavefunction_data.geminal_data.orb_data_up_spin.aos_data
    assert is_complex_aos(aos_k), f"kind={kind}: the k025 AO leaf must be complex, got {type(aos_k).__name__}"
    assert isinstance(aos_k, _COMPLEX_LEAF[kind]), f"kind={kind}: unexpected AO leaf {type(aos_k).__name__}"
    # The two twists really are different k: Gamma and a non-Gamma point.
    k_gamma, k_twist = (_twist_k_point_from_hamiltonian(h) for h in hams)
    assert k_gamma is None or not np.any(np.asarray(k_gamma)), f"kind={kind}: twist 0 must be Gamma, got {k_gamma}"
    assert k_twist is not None and np.any(np.asarray(k_twist)), f"kind={kind}: twist 1 must be non-Gamma, got {k_twist}"

    # ---- standalone side: one direct chain for THIS rank's twist ----
    split = split_world_for_twists(_world, n_twist=_N_TWIST)
    my_twist = int(split.twist_index)
    intra = split.intra_comm
    mcmc = MCMC(
        hamiltonian_data=hams[my_twist],
        mcmc_seed=_twist_sampler_seed(_SEED, my_twist, _N_TWIST),
        num_walkers=_NUM_WALKERS,
        mpi_comm_in=intra,
        world_comm_in=_world,
    )
    mcmc.run(num_mcmc_steps=_NUM_STEPS)
    E_k, sigma_k, _, _ = mcmc.get_E(num_mcmc_warmup_steps=_NUM_WARMUP, num_mcmc_bin_blocks=_NUM_BINS)
    buf = np.zeros((2, _N_TWIST), dtype=np.float64)
    if intra.Get_rank() == 0:
        buf[:, my_twist] = (float(E_k), float(sigma_k))
    E_manual, sigma_manual = _world.allreduce(buf, op=MPI.SUM)
    intra.Free()
    if split.inter_comm is not None:
        split.inter_comm.Free()

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
    )

    np.testing.assert_allclose(result.per_twist_E, E_manual, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
    np.testing.assert_allclose(result.per_twist_E_std, sigma_manual, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)

    W = np.asarray(_RAW_WEIGHTS, dtype=np.float64)
    W = W / W.sum()
    E_TA = float(np.sum(W * E_manual))
    assert pytest.approx(E_TA, abs=ATOL_PYTHON_FP64_BZ_MEAN) == result.energy.E_TA
    assert result.energy.E_TA_std == pytest.approx(float(np.sqrt(np.sum(W**2 * sigma_manual**2))), abs=ATOL_PYTHON_FP64_BZ_MEAN)
    assert result.energy.s_twist == pytest.approx(
        float(np.sqrt(np.sum(W * (E_manual - E_TA) ** 2))), abs=ATOL_PYTHON_FP64_BZ_MEAN
    )

    # Non-vacuity: the two twists carry different energies, so the reversed
    # weight vector (or a twist/weight mis-pairing) gives a different E_TA,
    # separated from the true one by more than the pin used above.
    E_TA_reversed = float(np.sum(W[::-1] * E_manual))
    assert abs(E_TA_reversed - E_TA) > ATOL_PYTHON_FP64_BZ_MEAN, (
        f"kind={kind}: per-twist energies {E_manual} do not separate a weight reversal from the true combine"
    )
    assert pytest.approx(E_TA_reversed, abs=ATOL_PYTHON_FP64_BZ_MEAN) != result.energy.E_TA

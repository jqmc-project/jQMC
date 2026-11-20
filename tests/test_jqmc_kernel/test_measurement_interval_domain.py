"""Rejection of a measurement interval without sampler updates (issue #598).

``num_mcmc_per_measurement`` is the number of MCMC updates (``MCMC``; CLI
``mcmc`` / ``vmc``) or GFMC projections (``GFMC_n``; CLI ``lrdmc-bra``)
between two successive measurements. With zero updates every measurement
re-measures the unmoved initial configuration, so a value below one (or a
non-integer) must be rejected before sampling instead of being published as
an energy or an optimized wavefunction.
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

import os
import subprocess
import sys
from pathlib import Path

import h5py
import pytest

from jqmc._checkpoint import merge_rank_checkpoints
from jqmc._precision import is_mixed_precision_enabled
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

pytestmark = [pytest.mark.with_chdir]

_H2_TREXIO = Path(__file__).resolve().parents[1] / "trexio_example_files" / "H2_ae_ccpvdz_cart.h5"
_MESSAGE = "num_mcmc_per_measurement must be an integer >= 1"


def _h2_hamiltonian() -> Hamiltonian_data:
    """All-electron H2 cc-pVDZ with a two-body Jastrow (so vmc has a parameter)."""
    structure_data, _, _, _, geminal_data, coulomb_potential_data = read_trexio_file(
        trexio_file=str(_H2_TREXIO), store_tuple=True
    )
    jastrow_data = Jastrow_data(
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0),
    )
    return Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_data),
    )


def _make_driver(driver: str, hamiltonian_data: Hamiltonian_data, num_mcmc_per_measurement):
    if driver == "MCMC":
        from jqmc.jqmc_mcmc import MCMC

        return MCMC(
            hamiltonian_data=hamiltonian_data,
            num_walkers=2,
            mcmc_seed=12345,
            num_mcmc_per_measurement=num_mcmc_per_measurement,
        )
    from jqmc.jqmc_gfmc import GFMC_n

    return GFMC_n(
        hamiltonian_data=hamiltonian_data,
        num_walkers=2,
        mcmc_seed=12345,
        alat=0.3,
        E_scf=-1.1,
        num_mcmc_per_measurement=num_mcmc_per_measurement,
    )


@pytest.mark.light
@pytest.mark.parametrize("driver", ["MCMC", "GFMC_n"])
@pytest.mark.parametrize("boundary", ["constructor", "checkpoint"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(0, id="zero"),
        pytest.param(-4, id="negative"),
        pytest.param(2.5, id="non_integer"),
        pytest.param(True, id="boolean"),
    ],
)
def test_drivers_reject_invalid_num_mcmc_per_measurement(tmp_path, driver, boundary, value):
    """Constructor and checkpoint load refuse a count below one or a non-integer count."""
    hamiltonian_data = _h2_hamiltonian()
    if boundary == "constructor":
        with pytest.raises(ValueError, match=_MESSAGE):
            _make_driver(driver, hamiltonian_data, value)
        return

    valid = _make_driver(driver, hamiltonian_data, 2)
    rank_file = tmp_path / "._restart_rank0.h5"
    valid.save_to_hdf5(str(rank_file))
    merged = tmp_path / "restart.h5"
    merge_rank_checkpoints(
        output_path=str(merged),
        mpi_size=1,
        driver_type=driver,
        hamiltonian_data=hamiltonian_data,
        tmp_pattern=str(tmp_path / "._restart_rank{rank}.h5"),
    )
    with h5py.File(merged, "r+") as checkpoint:
        checkpoint["rank_0/driver_config"].attrs["num_mcmc_per_measurement"] = value
    with pytest.raises(ValueError, match=_MESSAGE):
        type(valid).load_from_hdf5(str(merged), rank=0)


_JOB_SECTIONS = {
    "mcmc": "num_mcmc_steps = 12\nnum_mcmc_warmup_steps = 2\nnum_mcmc_bin_blocks = 5\nDt = 1.0\n",
    "vmc": (
        "num_mcmc_steps = 12\nnum_mcmc_warmup_steps = 2\nnum_mcmc_bin_blocks = 5\nDt = 1.0\n"
        "num_opt_steps = 2\nwf_dump_freq = 1\n"
    ),
    "lrdmc-bra": "num_mcmc_steps = 40\nalat = 0.3\nE_scf = -1.1\n",
}


@pytest.mark.light
@pytest.mark.parametrize("backend", ["rust", "xla"])
@pytest.mark.parametrize("job_type", ["mcmc", "vmc", "lrdmc-bra"])
@pytest.mark.parametrize("value", [pytest.param(0, id="zero"), pytest.param(-4, id="negative")])
def test_cli_rejects_nonpositive_num_mcmc_per_measurement(tmp_path, backend, job_type, value):
    """The console entry exits non-zero naming the key and writes no checkpoint or wavefunction."""
    _h2_hamiltonian().save_to_hdf5(str(tmp_path / "hamiltonian.h5"))
    precision = "mixed" if is_mixed_precision_enabled() else "full"
    (tmp_path / "input.toml").write_text(
        f"""
[control]
job_type = "{job_type}"
number_of_walkers = 2
mcmc_seed = 12345
hamiltonian_h5 = "hamiltonian.h5"
restart_chk = "restart.h5"
restart = false

[precision]
mode = "{precision}"

[{job_type}]
num_mcmc_per_measurement = {value}
{_JOB_SECTIONS[job_type]}""",
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[2]
    environment = {
        **os.environ,
        "PYTHONPATH": str(repo_root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "JQMC_FLAG_RUST_ON_CPU": "1" if backend == "rust" else "0",
    }
    completed = subprocess.run(
        [sys.executable, "-c", "from jqmc._cli_entry import main; main()", "input.toml"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode != 0, completed.stdout[-4000:]
    assert f"{_MESSAGE} (got {value})" in completed.stderr, completed.stderr[-4000:]
    assert not (tmp_path / "restart.h5").exists()
    assert not list(tmp_path.glob("hamiltonian_data_opt_step_*.h5"))

"""jqmc-tool inspection commands on twist-averaged inputs, and the module entry point.

* ``hamiltonian show-info`` / ``to-xyz`` must read a twist-averaged
  (combined layout) ``hamiltonian_twists.h5``, which the single-Hamiltonian
  loader cannot.
* ``python -m jqmc.jqmc_tool`` must run the CLI, not exit 0 doing nothing
  (it needs a ``__main__`` guard).
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

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from jqmc.jqmc_tool import hamiltonian_app
from tests._xyz_reference import assert_xyz_matches_trexio

pytestmark = [pytest.mark.with_chdir]

_TREXIO_FIXTURE = Path(__file__).parent.parent / "trexio_example_files" / "H2_ae_ccpvdz_cart.h5"


@pytest.fixture
def combined_twist_file(tmp_path):
    """Two-twist combined input that ``load_twist`` can fully materialise.

    Both twists sit at Gamma with real MOs: a molecular fixture cannot carry a
    complex non-Gamma Bloch leaf, and the inspection commands only need a
    loadable twist, not distinct k points.
    """
    from jqmc._twist_average import save_twist_average_input
    from jqmc.hamiltonians import Hamiltonian_data
    from jqmc.jastrow_factor import Jastrow_data
    from jqmc.trexio_wrapper import read_trexio_file
    from jqmc.wavefunction import Wavefunction_data

    assert _TREXIO_FIXTURE.is_file(), f"committed fixture missing: {_TREXIO_FIXTURE}"
    sd, _, _, _, gem, cpd = read_trexio_file(str(_TREXIO_FIXTURE), store_tuple=True)
    ham = Hamiltonian_data(
        structure_data=sd,
        coulomb_potential_data=cpd,
        wavefunction_data=Wavefunction_data(jastrow_data=Jastrow_data(), geminal_data=gem),
    )
    mo_up = np.asarray(gem.orb_data_up_spin.mo_coefficients, dtype=np.float64)
    mo_dn = np.asarray(gem.orb_data_dn_spin.mo_coefficients, dtype=np.float64)
    out = tmp_path / "h_twists.h5"
    save_twist_average_input(
        filepath=str(out),
        template_hamiltonian=ham,
        k_points=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        weights=[0.5, 0.5],
        mo_coefficients_up_per_twist=[mo_up, mo_up],
        mo_coefficients_dn_per_twist=[mo_dn, mo_dn],
    )
    return str(out)


@pytest.mark.light
def test_show_info_accepts_twist_averaged_input(combined_twist_file):
    """show-info prints the twist table and the selected twist instead of raising."""
    result = CliRunner().invoke(hamiltonian_app, ["show-info", combined_twist_file, "--twist", "1"])
    assert result.exit_code == 0, result.output
    assert "Twist-averaged input: 2 twists" in result.output
    assert "Showing twist 1:" in result.output


@pytest.mark.light
def test_show_info_rejects_out_of_range_twist(combined_twist_file):
    """An out-of-range --twist is a usage error, not a traceback."""
    result = CliRunner().invoke(hamiltonian_app, ["show-info", combined_twist_file, "--twist", "5"])
    assert result.exit_code != 0
    assert "out of range" in result.output


@pytest.mark.light
def test_to_xyz_accepts_twist_averaged_input(combined_twist_file, tmp_path):
    """to-xyz writes the shared structure of a twist-averaged input in angstroms."""
    out = tmp_path / "struct.xyz"
    result = CliRunner().invoke(hamiltonian_app, ["to-xyz", combined_twist_file, "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert_xyz_matches_trexio(out, _TREXIO_FIXTURE)


@pytest.mark.light
def test_module_entry_point_runs_the_cli():
    """``python -m jqmc.jqmc_tool --help`` prints the command list."""
    proc = subprocess.run(
        [sys.executable, "-m", "jqmc.jqmc_tool", "--help"],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "hamiltonian" in proc.stdout

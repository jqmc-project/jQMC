"""Workflow-side 'auto' resolution for num_mcmc_per_measurement.

The x4 policy and the h5 read live in the workflow; the driver only ever sees a
concrete int (guarded in jqmc.jqmc_cli). Light tests cover the pure policy and
the passthrough / missing-h5 paths; the heavy test builds a real H2 Hamiltonian
h5 and checks end-to-end resolution + caching.
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

import pytest

from jqmc._setting import ATOL_WORKFLOW_ACCEPTANCE_FIT
from jqmc_workflow._auto_params import fit_Dt_for_acceptance, num_mcmc_per_measurement_from_electrons
from jqmc_workflow.mcmc_workflow import MCMC_Workflow

pytestmark = [pytest.mark.with_chdir]


@pytest.mark.light
@pytest.mark.parametrize(
    ("nup", "ndn", "expected"),
    [(1, 1, 8), (5, 3, 32), (10, 10, 80), (0, 0, 0)],
)
def test_num_mcmc_per_measurement_from_electrons(nup, ndn, expected):
    """Pure policy: num_mcmc_per_measurement = 4 * (nup + ndn)."""
    assert num_mcmc_per_measurement_from_electrons(nup, ndn) == expected


@pytest.mark.light
def test_non_auto_passthrough(tmp_path):
    """A concrete int / None passes through untouched (no h5 read)."""
    wf_int = MCMC_Workflow(num_mcmc_per_measurement=16)
    assert wf_int._resolve_num_mcmc_per_measurement(str(tmp_path / "input.toml")) == 16

    wf_none = MCMC_Workflow(num_mcmc_per_measurement=None)
    # None flows to resolve_with_defaults downstream (-> misc default 40).
    assert wf_none._resolve_num_mcmc_per_measurement(str(tmp_path / "input.toml")) is None


@pytest.mark.light
def test_auto_missing_h5_raises(tmp_path):
    """'auto' with no staged input h5 fails loudly (not silently)."""
    wf = MCMC_Workflow(num_mcmc_per_measurement="auto")
    with pytest.raises(FileNotFoundError, match="auto"):
        wf._resolve_num_mcmc_per_measurement(str(tmp_path / "input.toml"))


@pytest.mark.light
def test_fit_Dt_for_acceptance_inverse():
    """Fit in 1/Dt recovers the Dt hitting the target acceptance (A ~= k/Dt)."""
    # A = 1/Dt exactly at these points; target 0.4 -> Dt* = 2.5.
    dt = fit_Dt_for_acceptance(dts=[2.0, 4.0], acceptances=[0.5, 0.25], target=0.4)
    assert abs(dt - 2.5) < ATOL_WORKFLOW_ACCEPTANCE_FIT
    # target 0.5 lands back on Dt = 2.0.
    assert abs(fit_Dt_for_acceptance([2.0, 4.0], [0.5, 0.25], target=0.5) - 2.0) < ATOL_WORKFLOW_ACCEPTANCE_FIT


@pytest.mark.light
def test_fit_Dt_needs_two_points():
    """Fewer than two calibration points is rejected (shared with LRDMC)."""
    with pytest.raises(ValueError, match="2 data points"):
        fit_Dt_for_acceptance(dts=[2.0], acceptances=[0.5], target=0.5)


@pytest.mark.light
def test_fit_Dt_degenerate_raises():
    """Identical Dt values give a degenerate fit."""
    with pytest.raises(RuntimeError, match="egenerate"):
        fit_Dt_for_acceptance(dts=[2.0, 2.0], acceptances=[0.5, 0.4], target=0.5)


@pytest.mark.light
def test_fit_Dt_rounds_to_one_decimal():
    """The fitted Dt is rounded to 1 decimal place (spurious precision otherwise)."""
    dt = fit_Dt_for_acceptance(dts=[1.0, 2.0, 3.0], acceptances=[0.60, 0.40, 0.25], target=0.5)
    assert dt == round(dt, 1)
    assert 1.0 <= dt <= 2.0


@pytest.mark.light
def test_dt_auto_config():
    """MCMC_Workflow accepts Dt='auto' and stores target_acceptance_ratio."""
    wf = MCMC_Workflow(Dt="auto", target_acceptance_ratio=0.45)
    assert wf.Dt == "auto"
    assert wf.target_acceptance_ratio == 0.45
    assert MCMC_Workflow().target_acceptance_ratio == 0.5  # default


@pytest.mark.light
def test_vmc_dt_auto_config():
    """VMC_Workflow accepts Dt='auto' (cross-driver pilot_a) and stores the target."""
    from jqmc_workflow.vmc_workflow import VMC_Workflow

    wf = VMC_Workflow(Dt="auto", target_acceptance_ratio=0.55)
    assert wf.Dt == "auto"
    assert wf.target_acceptance_ratio == 0.55
    assert VMC_Workflow().target_acceptance_ratio == 0.5  # default


@pytest.mark.light
def test_generate_input_dt_override(tmp_path):
    """pilot_a writes the trial Dt via dt_override; Dt='auto' never leaks to the TOML."""
    wf = MCMC_Workflow(Dt="auto", num_mcmc_per_measurement=16)
    input_file = str(tmp_path / "input.toml")
    wf._generate_input(10, input_file, dt_override=1.5)
    with open(input_file) as f:
        text = f.read()
    assert "Dt = 1.5" in text
    # Workflow-level 'auto' must not leak into the driver-facing physics keys.
    assert 'Dt = "auto"' not in text
    assert 'num_mcmc_per_measurement = "auto"' not in text
    assert "num_mcmc_per_measurement = 16" in text


@pytest.mark.light
def test_vmc_shares_auto_resolution(tmp_path):
    """VMC_Workflow uses the same 'auto' contract (passthrough + missing-h5)."""
    from jqmc_workflow.vmc_workflow import VMC_Workflow

    assert VMC_Workflow(num_mcmc_per_measurement=24)._resolve_num_mcmc_per_measurement(str(tmp_path / "input.toml")) == 24
    with pytest.raises(FileNotFoundError, match="auto"):
        VMC_Workflow(num_mcmc_per_measurement="auto")._resolve_num_mcmc_per_measurement(str(tmp_path / "input.toml"))


@pytest.mark.heavy
def test_vmc_calibration_input_is_valid_mcmc(tmp_path):
    """VMC pilot_a writes a valid job_type='mcmc' TOML at the trial Dt (cross-driver).

    Exercises the riskiest untested path: VMC generating *mcmc*-type calibration
    input (driver asymmetry) with num_mcmc_per_measurement='auto' resolved from
    the staged h5. H2 -> 4*2 = 8.
    """
    import tomllib

    from jqmc.hamiltonians import Hamiltonian_data
    from jqmc.jastrow_factor import Jastrow_data
    from jqmc.trexio_wrapper import read_trexio_file
    from jqmc.wavefunction import Wavefunction_data
    from jqmc_workflow.vmc_workflow import VMC_Workflow

    trexio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H2_ae_ccpvdz_cart.h5")
    (structure_data, _, _mos, _, geminal_mo_data, coulomb_potential_data) = read_trexio_file(
        trexio_file=trexio_path, store_tuple=True
    )
    ham = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=Wavefunction_data(
            jastrow_data=Jastrow_data(jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None),
            geminal_data=geminal_mo_data,
        ),
    )
    ham.save_to_hdf5(str(tmp_path / "hamiltonian_data.h5"))

    wf = VMC_Workflow(Dt="auto", num_mcmc_per_measurement="auto", pilot_mcmc_steps=50, hamiltonian_file="hamiltonian_data.h5")
    input_path = str(tmp_path / "input.toml")
    wf._write_mcmc_calibration_input(input_path, dt_v=1.5)

    with open(input_path, "rb") as f:
        cfg = tomllib.load(f)
    assert cfg["control"]["job_type"] == "mcmc"
    assert cfg["mcmc"]["Dt"] == 1.5
    assert cfg["mcmc"]["num_mcmc_steps"] == 50
    assert cfg["mcmc"]["num_mcmc_per_measurement"] == 8  # H2: 4 * (1 + 1)


@pytest.mark.heavy
def test_auto_resolves_from_h5(tmp_path):
    """End-to-end: 'auto' -> 4 * total electrons read from the staged h5, cached.

    H2 all-electron has 2 electrons, so 'auto' -> 8. The second call must reuse
    the cached value without touching the h5 (delete it to prove it).
    """
    from jqmc.hamiltonians import Hamiltonian_data
    from jqmc.jastrow_factor import Jastrow_data
    from jqmc.trexio_wrapper import read_trexio_file
    from jqmc.wavefunction import Wavefunction_data

    trexio_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H2_ae_ccpvdz_cart.h5")
    (structure_data, _, _mos, _, geminal_mo_data, coulomb_potential_data) = read_trexio_file(
        trexio_file=trexio_path, store_tuple=True
    )
    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=Wavefunction_data(
            jastrow_data=Jastrow_data(jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None),
            geminal_data=geminal_mo_data,
        ),
    )
    h5_path = tmp_path / "hamiltonian_data.h5"
    hamiltonian_data.save_to_hdf5(str(h5_path))

    wf = MCMC_Workflow(num_mcmc_per_measurement="auto", hamiltonian_file="hamiltonian_data.h5")
    input_file = str(tmp_path / "input.toml")

    assert wf._resolve_num_mcmc_per_measurement(input_file) == 8

    # cached: a second call must not need the h5 anymore.
    h5_path.unlink()
    assert wf._resolve_num_mcmc_per_measurement(input_file) == 8

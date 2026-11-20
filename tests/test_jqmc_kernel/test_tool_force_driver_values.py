"""Deterministic force post-processing from actual serial sampler checkpoints."""

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

import ast
import inspect
from pathlib import Path
import textwrap
import warnings

import numpy as np
from numpy.exceptions import ComplexWarning
import pytest
from typer.testing import CliRunner
from uncertainties import ufloat

from jqmc import jqmc_tool
from jqmc._checkpoint import merge_rank_checkpoints
from jqmc._precision import get_tolerance_min
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from jqmc.jqmc_mcmc import MCMC
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]

_DRIVER_WINDOWS = [("MCMC", 0), ("GFMC_t", 5), ("GFMC_t", 7), ("GFMC_n", 5), ("GFMC_n", 7)]
# Measurements, discarded warmup measurements and bins per driver. GFMC uses the
# documented LRDMC analysis minima (warmup 25, 10 bins), which lrdmc compute-force
# enforces; 44 measurements leave 14 or 12 (uneven bins) after collect and warmup.
_ANALYSIS = {"MCMC": (20, 2, 3), "GFMC_t": (44, 25, 10), "GFMC_n": (44, 25, 10)}


def _run_checkpoint(root, driver_name, collect_steps, fixture="H2_ae_ccpvdz_cart.h5"):
    """Run the real sampler and write the same merged format consumed by the CLI."""
    source = Path(__file__).resolve().parents[1] / "trexio_example_files" / fixture
    structure, _, _, _, geminal, coulomb = read_trexio_file(str(source), store_tuple=True)
    ham = Hamiltonian_data(
        structure_data=structure,
        coulomb_potential_data=coulomb,
        wavefunction_data=Wavefunction_data(
            geminal_data=geminal,
            jastrow_data=Jastrow_data(
                jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
                    jastrow_2b_param=1.0, jastrow_2b_type="pade", structure_data=structure
                )
            ),
        ),
    )
    driver_class = {"MCMC": MCMC, "GFMC_t": GFMC_t, "GFMC_n": GFMC_n}[driver_name]
    options = {"tau": 0.1} if driver_name == "GFMC_t" else {"num_mcmc_per_measurement": 1}
    if driver_name != "MCMC":
        options.update(num_gfmc_collect_steps=collect_steps, alat=0.3)
    driver = driver_class(
        hamiltonian_data=ham,
        num_walkers=2,
        mcmc_seed=12345,
        comput_position_deriv=True,
        use_swct=False,
        **options,
    )
    num_steps, warmup, bins = _ANALYSIS[driver_name]
    driver.run(num_mcmc_steps=num_steps)
    reference = np.asarray(driver.get_aF(num_mcmc_warmup_steps=warmup, num_mcmc_bin_blocks=bins))
    assert reference.shape == (2, 2, 3)
    assert np.isrealobj(reference) and np.isfinite(reference).all()
    assert np.any(reference[0] != 0) and np.any(reference[1] > 0)
    pattern = str(root / "rank-{rank}.h5")
    driver.save_to_hdf5(pattern.format(rank=0))
    checkpoint = str(root / "restart.h5")
    merge_rank_checkpoints(checkpoint, mpi_size=1, driver_type=driver_name, hamiltonian_data=ham, tmp_pattern=pattern)
    return driver, checkpoint, reference


def _assert_tool_force(driver, checkpoint, reference, driver_name, collect_steps, monkeypatch):
    """Observe unrounded values but still execute the real formatter and CLI."""
    captured = []

    def observe_ufloat(value, error, *args, **kwargs):
        captured.append((value, error))
        return ufloat(value, error, *args, **kwargs)

    app = jqmc_tool.mcmc_app if driver_name == "MCMC" else jqmc_tool.lrdmc_app
    _num_steps, warmup, bins = _ANALYSIS[driver_name]
    arguments = ["compute-force", checkpoint, "-b", str(bins), "-w", str(warmup)]
    if driver_name != "MCMC":
        arguments.extend(["-c", str(collect_steps)])
    with monkeypatch.context() as patch:
        patch.setattr(jqmc_tool, "ufloat", observe_ufloat)
        result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, (result.output, result.exception)
    assert "Found 1 MPI ranks" in result.output
    observed = np.asarray(captured).T.reshape(2, 2, 3)
    atol, rtol = get_tolerance_min(("local_energy", "det_grad_lap", "coulomb"), "strict")
    for field in ("mean", "error"):
        index = {"mean": 0, "error": 1}[field]
        actual, expected = observed[index], reference[index]
        assert actual.shape == expected.shape == (2, 3)
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    labels = driver.hamiltonian_data.structure_data.atomic_labels
    rows = [line.split() for line in result.output.splitlines() if line.split() and line.split()[0] in labels]
    expected_rows = [
        [label, *(f"{ufloat(reference[0, i, j], reference[1, i, j]):+2uS}" for j in range(3))] for i, label in enumerate(labels)
    ]
    assert rows == expected_rows


@pytest.mark.parametrize("driver_name,collect_steps", _DRIVER_WINDOWS)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_force_cli_matches_actual_driver(tmp_path, monkeypatch, driver_name, collect_steps):
    """Real MCMC and both LRDMC algorithms reproduce driver forces through a saved checkpoint.

    Committed H2 AE Cartesian GTO/MO has one up/one down electron, open
    boundaries and Pade J2=1. Two walkers use seed 12345, no SWCT or cond(G)
    filter. Conventional GFMC uses tau=0.1; MCMC and load-balanced GFMC use
    one update per measurement, GFMC alat=0.3. MCMC takes twenty
    measurements, discards two and uses three bins; GFMC takes 44
    measurements with collect windows five and seven, discards 25 and uses
    ten bins (the documented LRDMC analysis minima). Both include uneven
    bin sizes.

    The real serial driver writes and merges its checkpoint. The actual
    compute-force command must match every driver get_aF mean/error component
    with the existing same-mode strict policy, and its printed table must
    equal the specified two-significant-digit-error representation. Force
    and error must be nonzero. Error bars are outputs, never tolerances.
    This is a driver/checkpoint/CLI route check, not an independent force law.
    """
    driver, checkpoint, reference = _run_checkpoint(tmp_path, driver_name, collect_steps)
    _assert_tool_force(driver, checkpoint, reference, driver_name, collect_steps, monkeypatch)


@pytest.mark.parametrize("driver_name,collect_steps", _DRIVER_WINDOWS)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_force_cli_oracle_rejects_wrong_coefficients(tmp_path, monkeypatch, driver_name, collect_steps):
    """Actual driver/checkpoint comparisons reject half Pulay and reversed HF terms.

    Use the identical H2/J2 fixed-seed production trajectory and collect/bin
    settings as the normal force-route test. Establish the normal baseline
    first, then change only the live CLI function's code inside a restoring
    patch context. Both coefficient faults must fail the numerical comparison;
    the driver, checkpoint, formatter and tolerance policy remain unchanged.
    """
    driver, checkpoint, reference = _run_checkpoint(tmp_path, driver_name, collect_steps)
    _assert_tool_force(driver, checkpoint, reference, driver_name, collect_steps, monkeypatch)
    function = jqmc_tool.mcmc_compute_force if driver_name == "MCMC" else jqmc_tool.lrdmc_compute_force
    source = textwrap.dedent(inspect.getsource(function))
    for before, after in (
        ("force_Pulay_jn = -2.0 *", "force_Pulay_jn = -1.0 *"),
        ("force_HF_jn = -1.0 *", "force_HF_jn = 1.0 *"),
    ):
        assert source.count(before) == 1
        tree = ast.parse(source.replace(before, after))
        tree.body[0].decorator_list = []
        namespace = dict(vars(jqmc_tool))
        exec(compile(tree, "<force-coefficient-counterexample>", "exec"), namespace)
        with monkeypatch.context() as patch:
            patch.setattr(function, "__code__", namespace[function.__name__].__code__)
            with pytest.raises(AssertionError, match="Not equal to tolerance"):
                _assert_tool_force(driver, checkpoint, reference, driver_name, collect_steps, monkeypatch)


@pytest.mark.parametrize("driver_name,collect_steps", _DRIVER_WINDOWS)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_force_cli_matches_actual_driver_complex_twist(tmp_path, monkeypatch, driver_name, collect_steps):
    """At a complex Bloch twist the CLI reports the driver's real force and real-part error.

    Committed periodic H2 AE Cartesian GTO/MO at the non-Gamma twist
    k = (0.25, 0.25, 0.25) has genuinely complex MOs, so the stored local
    energy and the E_L * Pulay product are complex. Pade J2=1, two walkers,
    seed 12345, no SWCT; the measurement counts, GFMC settings, collect
    windows, warmup and bins are those of the open-boundary route test.

    The physical force is real: the drivers project every jackknife force
    sample onto its real part before the mean and the error. The CLI must do
    the same, so its unrounded mean and error equal the driver get_aF values
    under the same-mode strict policy, and its table equals the specified
    formatting. A CLI that keeps complex jackknife samples inflates the error
    by the imaginary fluctuation (and casts complex to real when printing).
    The fixture must really produce complex samples: the stored E_L * Pulay
    product has a nonzero imaginary part.
    """
    driver, checkpoint, reference = _run_checkpoint(tmp_path, driver_name, collect_steps, fixture="H2_pbc_ae_k025_cart.h5")
    rank_data = jqmc_tool.load_observables_from_checkpoint(checkpoint, ["E_L_force_PP"])
    assert np.iscomplexobj(rank_data[0]["E_L_force_PP"])
    assert np.max(np.abs(np.imag(rank_data[0]["E_L_force_PP"]))) > 0
    _assert_tool_force(driver, checkpoint, reference, driver_name, collect_steps, monkeypatch)


@pytest.mark.parametrize("driver_name,collect_steps", _DRIVER_WINDOWS)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_force_cli_complex_twist_oracle_rejects_complex_jackknife(tmp_path, monkeypatch, driver_name, collect_steps):
    """The complex-twist comparison rejects a CLI that keeps complex jackknife samples.

    Use the identical periodic H2 k=(0.25,0.25,0.25) fixed-seed trajectory and
    settings as the complex-twist route test and establish its baseline first.
    Then remove only the real-part projection of the live CLI's force
    jackknife samples inside a restoring patch context, with the
    complex-to-real print cast allowed, so the fault is judged numerically:
    the error, which then includes the imaginary fluctuation, must fail the
    strict comparison against the driver.
    """
    driver, checkpoint, reference = _run_checkpoint(tmp_path, driver_name, collect_steps, fixture="H2_pbc_ae_k025_cart.h5")
    _assert_tool_force(driver, checkpoint, reference, driver_name, collect_steps, monkeypatch)
    function = jqmc_tool.mcmc_compute_force if driver_name == "MCMC" else jqmc_tool.lrdmc_compute_force
    source = textwrap.dedent(inspect.getsource(function))
    before = "force_jn = np.real(force_HF_jn + force_Pulay_jn)"
    assert source.count(before) == 1
    tree = ast.parse(source.replace(before, "force_jn = force_HF_jn + force_Pulay_jn"))
    tree.body[0].decorator_list = []
    namespace = dict(vars(jqmc_tool))
    exec(compile(tree, "<force-complex-jackknife-counterexample>", "exec"), namespace)
    with monkeypatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        patch.setattr(function, "__code__", namespace[function.__name__].__code__)
        with pytest.raises(AssertionError, match="Not equal to tolerance"):
            _assert_tool_force(driver, checkpoint, reference, driver_name, collect_steps, monkeypatch)

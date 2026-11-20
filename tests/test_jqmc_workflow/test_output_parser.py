"""Tests for jqmc_workflow.output_parser -- regex-based jQMC log parser.

Layer 1 (pure unit): hand-crafted text fixtures (mirroring real jqmc stdout
format, documented in each parser's docstring) + tmp_path-backed fake
workflow directories.

Test class layout (expanded):
    TestParseUfloatShort      -- "+0.123(45)" / scientific notation parser
    TestParseForceTable       -- atomic-force table from jqmc-tool
    TestParseStressTable      -- Stress Tensor (sigma) table
    TestParsePressureTable    -- Pressure (P = -diag(sigma)) table
    TestParseRunMetadata      -- MPI / walkers / JAX backend / devices header
    TestParseVmcLogText       -- per-optimization-step VMC parser
    TestParseVmcOutput        -- file-based VMC wrapper
    TestParseMcmcOutput       -- file-based MCMC wrapper
    TestParseLrdmcOutput      -- file-based LRDMC wrapper
    TestParseLrdmcExtOutput   -- a^2->0 extrapolation parser
    TestParseInputParams      -- TOML input + restart.h5 -> Input_Parameters
    TestRepairForcesFromOutput
    TestInternalHelpers       -- _read_text / _tail / _find_*_files
    TestVmcOptimizerTimerLabel -- solver label of a real one-step VMC run
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

import math
import textwrap
from pathlib import Path

import pytest
import toml

from jqmc.output_parser import parse_jqmc_log

from jqmc_workflow.forces_parser import (
    _parse_labelled_ufloat_table,
    parse_force_table,
    parse_pressure_table,
    parse_stress_table,
    parse_ufloat_short,
    repair_forces_from_output,
)
from jqmc_workflow.output_parser import (
    _find_hamiltonian_h5,
    _find_input_files,
    _find_output_files,
    _parse_run_metadata,
    _parse_vmc_log_text,
    _read_text,
    _tail,
    parse_input_params,
    parse_lrdmc_ext_output,
    parse_lrdmc_output,
    parse_mcmc_output,
    parse_vmc_output,
)

pytestmark = [pytest.mark.with_chdir]


# -- Helpers ---------------------------------------------------------


def _bootstrap_workflow_dir(
    directory: Path,
    output_file: str = "out.o",
    output_text: str = "",
    input_file: str = "input.toml",
    input_text: str = "",
    job_status: str = "fetched",
    step: int = 0,
    result_section: dict | None = None,
) -> None:
    """Create a fake workflow dir with workflow_state.toml + output + input files."""
    directory.mkdir(parents=True, exist_ok=True)
    if output_text:
        (directory / output_file).write_text(output_text)
    if input_text:
        (directory / input_file).write_text(input_text)
    state = {
        "workflow": {"label": "test", "type": "T", "status": "completed"},
        "jobs": [
            {
                "input_file": input_file,
                "output_file": output_file,
                "job_id": "1",
                "server_machine": "local",
                "status": job_status,
                "step": step,
            }
        ],
        "result": result_section or {},
    }
    (directory / "workflow_state.toml").write_text(toml.dumps(state))


# Canonical force-table snippet from the docstring.
_FORCE_TABLE = textwrap.dedent("""
    Some preamble line.
    Atomic Forces:
    ------------------------------------------------
    Label   Fx(Ha/bohr) Fy(Ha/bohr) Fz(Ha/bohr)
    ------------------------------------------------
    O       -0.0114(14)  -0.0082(11)  -0.0173(13)
    H       +0.0112(11)  +0.0050(7)   +0.0054(9)
    H       +0.0002(10)  +0.0032(8)   +0.0119(11)
    ------------------------------------------------
    More text below.
""").strip()


_STRESS_TABLE = textwrap.dedent("""
    Stress Tensor (sigma):
    ------------------------------------------------
    Component   Value (Ha/bohr^3)
    ------------------------------------------------
    sigma_xx    +0.001234(56)
    sigma_yy    +0.002345(67)
    sigma_zz    +0.003456(78)
    sigma_xy    +0.000012(34)
    sigma_xz    +0.000023(45)
    sigma_yz    +0.000034(56)
    ------------------------------------------------
""").strip()


_PRESSURE_TABLE = textwrap.dedent("""
    Pressure (P = -diag(sigma)):
    ------------------------------------------------
    Component   Value (Ha/bohr^3)
    ------------------------------------------------
    P_x         -0.001234(56)
    P_y         -0.002345(67)
    P_z         -0.003456(78)
    P_hydro     -0.002345(67)
    ------------------------------------------------
""").strip()


_VMC_LOG_2STEPS = textwrap.dedent("""
    ==================================================================
    Optimization step = 1/2.
    ==================================================================
    Pre-compilation time for MCMC = 11.35 sec.
    Total elapsed time for MCMC 1000 steps. = 21.07 sec.
    Net total time for MCMC = 9.72 sec.
    Time for MCMC update = 4.50 msec.
    Time for computing e_L = 3.20 msec.
    Time for computing de_L/dR and de_L/dr = 1.10 msec.
    Time for computing dln_Psi/dR and dln_Psi/dr = 0.80 msec.
    Time for computing dln_Psi/dc = 0.50 msec.
    Time for computing de_L/dc = 0.20 msec.
    Time for MPI barrier after MCMC update = 0.10 msec.
    Time for misc. (others) = 0.05 msec.
    Average of walker weights is 0.799. Ideal is ~ 0.800. Adjust epsilon_AS.
    Acceptance ratio is 51.46 %.  Ideal is ~ 50.00%. Adjust Dt.
    Max generalized force = 17.984 +- 0.330 Ha/a.u.
    Max generalized force signal-to-noise = 126.871.
    E = -16.61072 +- 0.00704 Ha
    ==================================================================
    Optimization step = 2/2.
    ==================================================================
    Net total time for MCMC = 8.00 sec.
    Average of walker weights is 0.800.
    Acceptance ratio is 50.00 %.
    E = -16.62000 +- 0.00500 Ha
""").strip()


_VMC_OPT_FOOTER = textwrap.dedent("""
    Total elapsed time for optimization 2 steps. = 50.00 sec.
    Time for MCMC run = 30.0 sec.
    Time for computing E = 5.0 sec.
    Time for computing generalized forces (gF) = 8.0 sec.
    Time for optimizer (SR) = 4.0 sec.
    Time for parameter update = 2.0 sec.
    Time for MPI barrier = 0.5 sec.
    Time for misc. (others) = 0.5 sec.
""").strip()


_METADATA_HEADER = textwrap.dedent("""
    Some preamble.
    The number of MPI process = 4.
    The number of walkers assigned for each MPI process = 8.
    JAX backend = gpu.
    *** XLA Global devices recognized by JAX***
    [CudaDevice(id=0), CudaDevice(id=1), CudaDevice(id=2), CudaDevice(id=3)]
""").strip()


# -- Tests -----------------------------------------------------------


class TestParseUfloatShort:
    @pytest.mark.parametrize(
        "text,expected_val,expected_unc",
        [
            ("+0.0114(14)", 0.0114, 0.0014),
            ("-1.23(4)", -1.23, 0.04),
            ("+0.100(5)", 0.100, 0.005),
            ("3.14(2)", 3.14, 0.02),
            ("+3(8)e-05", 3e-5, 8e-5),
            ("+3.9(3.5)e-05", 3.9e-5, 3.5e-5),
            ("-2.0(1.0)e3", -2000.0, 1000.0),
        ],
    )
    @pytest.mark.light
    def test_canonical_formats(self, text, expected_val, expected_unc):
        val, unc = parse_ufloat_short(text)
        assert val == pytest.approx(expected_val)
        assert unc == pytest.approx(expected_unc)

    @pytest.mark.light
    @pytest.mark.parametrize("bad", ["", "not a number", "1.0+/-0.1", "(5)"])
    def test_malformed_returns_none_pair(self, bad):
        assert parse_ufloat_short(bad) == (None, None)

    @pytest.mark.light
    def test_strips_surrounding_whitespace(self):
        val, unc = parse_ufloat_short("   +1.0(2)   ")
        assert val == 1.0
        assert unc == 0.2


class TestParseForceTable:
    @pytest.mark.light
    def test_canonical_table(self):
        forces = parse_force_table(_FORCE_TABLE)
        assert forces is not None
        assert len(forces) == 3
        assert forces[0] == {
            "label": "O",
            "Fx": pytest.approx(-0.0114),
            "Fx_err": pytest.approx(0.0014),
            "Fy": pytest.approx(-0.0082),
            "Fy_err": pytest.approx(0.0011),
            "Fz": pytest.approx(-0.0173),
            "Fz_err": pytest.approx(0.0013),
        }

    @pytest.mark.light
    def test_returns_none_when_no_table(self):
        assert parse_force_table("just some text\nno table here") is None

    @pytest.mark.light
    def test_handles_extra_columns_gracefully(self):
        """Lines with extra columns (atom index, etc.) still parsed by first 4 tokens."""
        txt = textwrap.dedent("""
            Atomic Forces:
            ------------------------------------------------
            Label   Fx Fy Fz extra
            ------------------------------------------------
            O       -0.0114(14)  -0.0082(11)  -0.0173(13) ignored
            ------------------------------------------------
        """).strip()
        forces = parse_force_table(txt)
        assert forces is not None and len(forces) == 1

        # A diverged run keeps its row, with the non-finite value.
        val, unc = parse_ufloat_short("+0.1(nan)")
        assert val == 0.1 and math.isnan(unc)
        assert parse_ufloat_short("+inf(10)")[0] == math.inf
        forces = parse_force_table(txt.replace("-0.0114(14)", "+nan(nan)"))
        assert len(forces) == 1 and math.isnan(forces[0]["Fx"]) and math.isnan(forces[0]["Fx_err"])
        assert forces[0]["Fy"] == pytest.approx(-0.0082)

        # A max_time given as a float in the TOML is logged as a float.
        assert parse_jqmc_log("Stopping... max_time = 3600.0 sec. exceeds.", "mcmc")["stopped_by_max_time"]


class TestParseStressTable:
    @pytest.mark.light
    def test_canonical_table(self):
        stress = parse_stress_table(_STRESS_TABLE)
        assert stress is not None
        assert stress["sigma_xx"] == pytest.approx(0.001234)
        assert stress["sigma_xx_err"] == pytest.approx(
            0.000056
        )  # tol-ok: golden parsed value from a text fixture, not a tolerance
        assert stress["sigma_yz_err"] == pytest.approx(
            0.000056
        )  # tol-ok: golden parsed value from a text fixture, not a tolerance

    @pytest.mark.light
    def test_returns_none_when_no_table(self):
        assert parse_stress_table("no stress here") is None

    @pytest.mark.light
    def test_returns_none_when_component_missing(self):
        """A table missing one of the 6 required sigma components fails parsing."""
        partial = textwrap.dedent("""
            Stress Tensor (sigma):
            ------------------------------------------------
            sigma_xx    +0.001(2)
            sigma_yy    +0.002(3)
            ------------------------------------------------
        """).strip()
        assert parse_stress_table(partial) is None


class TestParsePressureTable:
    @pytest.mark.light
    def test_canonical_table(self):
        p = parse_pressure_table(_PRESSURE_TABLE)
        assert p is not None
        assert p["Px"] == pytest.approx(-0.001234)
        assert p["Px_err"] == pytest.approx(0.000056)  # tol-ok: golden parsed value from a text fixture, not a tolerance
        assert p["P_hydro"] == pytest.approx(-0.002345)
        assert p["P_hydro_err"] == pytest.approx(0.000067)  # tol-ok: golden parsed value from a text fixture, not a tolerance

    @pytest.mark.light
    def test_returns_none_when_no_table(self):
        assert parse_pressure_table("no pressure data") is None

    @pytest.mark.light
    def test_returns_none_when_component_missing(self):
        partial = textwrap.dedent("""
            Pressure (P = -diag(sigma)):
            ------------------------------------------------
            P_x         -0.001(2)
            P_y         -0.002(3)
            ------------------------------------------------
        """).strip()
        assert parse_pressure_table(partial) is None


class TestLabelledUfloatTable:
    """_parse_labelled_ufloat_table is the shared helper for stress/pressure."""

    @pytest.mark.light
    def test_returns_none_when_header_absent(self):
        assert _parse_labelled_ufloat_table("nothing", "MyHeader", ("a", "b")) is None

    @pytest.mark.light
    def test_skips_unrecognized_labels(self):
        txt = "MyHeader\n---\na  +1.0(1)\nbogus +9.9(9)\nb  +2.0(2)\n---\n"
        result = _parse_labelled_ufloat_table(txt, "MyHeader", ("a", "b"))
        assert result is not None
        assert set(result) == {"a", "b"}


class TestParseRunMetadata:
    @pytest.mark.light
    def test_canonical_header(self):
        meta = _parse_run_metadata(_METADATA_HEADER)
        assert meta["num_mpi_processes"] == 4
        assert meta["num_walkers_per_process"] == 8
        assert meta["jax_backend"] == "gpu"
        assert meta["jax_devices"] == [
            "CudaDevice(id=0)",
            "CudaDevice(id=1)",
            "CudaDevice(id=2)",
            "CudaDevice(id=3)",
        ]

    @pytest.mark.light
    def test_cpu_fallback(self):
        meta = _parse_run_metadata("Running on CPUs or single GPU. Etc.")
        assert meta["jax_backend"] == "cpu"

    @pytest.mark.light
    def test_explicit_backend_wins_over_fallback(self):
        text = "Running on CPUs or single GPU.\nJAX backend = cuda.\n"
        meta = _parse_run_metadata(text)
        assert meta["jax_backend"] == "cuda"

    @pytest.mark.light
    def test_all_none_when_empty(self):
        meta = _parse_run_metadata("")
        assert meta == {
            "num_mpi_processes": None,
            "num_walkers_per_process": None,
            "jax_backend": None,
            "jax_devices": None,
        }


class TestJaxEnvironmentEcho:
    """``jax.print_environment_info`` echoes ``JAX_*``/``XLA_*`` variables (issue #348)."""

    _LOG = textwrap.dedent(
        """\
        All rights reserved.
        jax:    0.7.2
        platform: uname_result(system='Linux', release='5.15.0-43-generic')
        JAX_COMPILATION_CACHE_DIR=/tmp/runner/_work/_temp/jax-cache/gw0
        XLA_FLAGS=--xla_force_host_platform_device_count=2
        Program ends 2026-11-02 22:11:38
        """
    )

    @pytest.mark.light
    def test_environment_variables_are_extracted_not_unparsed(self):
        from jqmc.output_parser import parse_environment, unparsed_numeric_lines

        assert unparsed_numeric_lines(self._LOG) == []
        environment = parse_environment(self._LOG)
        assert environment["jax_environment_variables"] == {
            "JAX_COMPILATION_CACHE_DIR": "/tmp/runner/_work/_temp/jax-cache/gw0",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
        }
        assert environment["xla_flags"] == "--xla_force_host_platform_device_count=2"

    @pytest.mark.light
    def test_logged_platform_omits_host_name_and_kernel(self, monkeypatch):
        """The logged environment keeps OS family and architecture, never the host name or kernel build."""
        import platform

        import jax

        from jqmc.jqmc_cli import _jax_environment_info
        from jqmc.output_parser import parse_environment

        host, release, build = "synthetic-host-7f3a", "9.9.9-synthetic-release", "#42 SMP synthetic-build"
        echoed = (
            "jax:    0.7.2\n"
            f"platform: uname_result(system='Linux', node='{host}', release='{release}', version='{build}', machine='x86_64')\n"
            "XLA_FLAGS=--xla_force_host_platform_device_count=2"
        )
        monkeypatch.setattr(jax, "print_environment_info", lambda return_string=False: echoed)
        logged = _jax_environment_info()
        for secret in (host, release, build):
            assert secret not in logged
        assert "jax:    0.7.2" in logged and "XLA_FLAGS=--xla_force_host_platform_device_count=2" in logged
        assert parse_environment(logged)["platform"] == f"{platform.system()} {platform.machine()}"

    @pytest.mark.light
    def test_numeric_output_next_to_the_echo_is_still_reported(self):
        """Negative control: only the JAX/XLA echo is exempt, not numbers or other assignments."""
        from jqmc.output_parser import unparsed_numeric_lines

        stray = [
            "Mystery estimator = 1.25 Ha",
            "MY_CACHE_DIR=/tmp/42/cache",
            "E_JAX_LIKE=3.5",
            "  JAX_COMPILATION_CACHE_DIR 7",
        ]
        text = self._LOG.replace("Program ends", "\n".join(stray) + "\nProgram ends")
        assert unparsed_numeric_lines(text) == [line.strip() for line in stray]


class TestParseVmcLogText:
    @pytest.mark.light
    def test_every_logged_top_coordinate_is_parsed(self):
        """A verbosity=high top-coordinate block of the configured length is parsed row for row."""
        from jqmc._setting import OPTIMIZER_DIAGNOSTICS_TOP_COORDINATES
        from jqmc.output_parser import parse_vmc_steps

        count = OPTIMIZER_DIAGNOSTICS_TOP_COORDINATES
        rows = [
            f"    j3_matrix[q{index}]: force=+1.0e-03 +- 1.0e-04, signal-to-noise={count - index:.6e}, metric_diag=2.0e-01."
            for index in range(count)
        ]
        text = "\n".join(
            ["Optimization step = 1/1.", f"  Top {count} coordinates by generalized-force signal-to-noise:", *rows]
        )
        (step,) = parse_vmc_steps(text)
        assert [row["coordinate"] for row in step["top_coordinates"]] == [f"j3_matrix[q{index}]" for index in range(count)]
        assert [row["signal_to_noise"] for row in step["top_coordinates"]] == [float(count - index) for index in range(count)]

    @pytest.mark.light
    def test_two_steps_parsed(self):
        steps = _parse_vmc_log_text(_VMC_LOG_2STEPS)
        assert len(steps) == 2
        assert [s.step for s in steps] == [1, 2]

    @pytest.mark.light
    def test_step1_fields(self):
        steps = _parse_vmc_log_text(_VMC_LOG_2STEPS)
        s = steps[0]
        assert s.energy == pytest.approx(-16.61072)
        assert s.energy_error == pytest.approx(0.00704)  # tol-ok: golden parsed value from a text fixture, not a tolerance
        assert s.max_force == pytest.approx(17.984)
        assert s.max_force_error == pytest.approx(0.330)  # tol-ok: golden parsed value from a text fixture, not a tolerance
        assert s.signal_to_noise_ratio == pytest.approx(126.871)
        assert s.avg_walker_weight == pytest.approx(0.799)
        assert s.acceptance_ratio == pytest.approx(0.5146)
        assert s.net_time_sec == pytest.approx(9.72)
        assert s.total_time_sec == pytest.approx(21.07)
        assert s.precompilation_time_sec == pytest.approx(11.35)

    @pytest.mark.light
    def test_step1_timing_breakdown(self):
        s = _parse_vmc_log_text(_VMC_LOG_2STEPS)[0]
        assert s.timing_breakdown["mcmc_update"] == pytest.approx(4.50)
        assert s.timing_breakdown["e_L"] == pytest.approx(3.20)
        assert s.timing_breakdown["misc"] == pytest.approx(0.05)

    @pytest.mark.light
    def test_step2_partial_fields(self):
        """Step 2 in the fixture has fewer fields -- partial data is OK."""
        s = _parse_vmc_log_text(_VMC_LOG_2STEPS)[1]
        assert s.energy == pytest.approx(-16.62000)
        assert s.acceptance_ratio == pytest.approx(0.5000)
        # Fields absent from step 2 stay at defaults.
        assert s.max_force is None
        assert s.signal_to_noise_ratio is None

    @pytest.mark.light
    def test_lines_before_first_step_ignored(self):
        text = "garbage line\nE = -1.0 +- 0.1 Ha\n" + _VMC_LOG_2STEPS
        steps = _parse_vmc_log_text(text)
        assert len(steps) == 2
        assert steps[0].step == 1  # the pre-step E= line did not start a step

    @pytest.mark.light
    def test_nan_energy_parsed(self):
        text = "Optimization step = 1/1.\nE = nan +- inf Ha\n"
        steps = _parse_vmc_log_text(text)
        import math

        assert math.isnan(steps[0].energy)
        assert math.isinf(steps[0].energy_error)

    @pytest.mark.light
    def test_generalized_force_does_not_overwrite_energy(self):
        text = textwrap.dedent("""
            Optimization step = 1/1.
            E = -14.76025 +- 0.27171 Ha
            Max generalized force = 141.257 +- 35.391 Ha/a.u.
        """)
        step = _parse_vmc_log_text(text)[0]

        assert step.energy == -14.76025
        assert step.energy_error == 0.27171
        assert step.max_force == 141.257
        assert step.max_force_error == 35.391

    @pytest.mark.light
    def test_legacy_force_labels_are_rejected(self):
        text = textwrap.dedent("""
            Optimization step = 1/1.
            Max f = 1.0 +- 0.1 Ha/a.u.
            Max coordinate force = 2.0 +- 0.2 Ha/a.u.
            Max of signal-to-noise of f = max(|f|/|std f|) = 3.0.
        """)
        step = _parse_vmc_log_text(text)[0]
        assert step.max_force is None
        assert step.max_force_error is None
        assert step.signal_to_noise_ratio is None

    @pytest.mark.light
    def test_empty_text_returns_empty_list(self):
        assert _parse_vmc_log_text("") == []


class TestParseVmcOutput:
    @pytest.mark.light
    def test_two_step_file_parsed(self, tmp_path):
        _bootstrap_workflow_dir(
            tmp_path,
            output_text=_VMC_LOG_2STEPS + "\n" + _METADATA_HEADER + "\n" + _VMC_OPT_FOOTER,
        )
        result = parse_vmc_output(str(tmp_path))
        assert len(result.steps) == 2
        assert result.total_opt_steps == 2
        assert result.total_opt_time_sec == pytest.approx(50.0)
        assert result.opt_timing_breakdown["mcmc_run"] == pytest.approx(30.0)
        assert result.opt_timing_breakdown["get_E"] == pytest.approx(5.0)
        # Metadata applied
        assert result.num_mpi_processes == 4
        assert result.jax_backend == "gpu"

    @pytest.mark.light
    def test_missing_directory_returns_empty(self, tmp_path):
        result = parse_vmc_output(str(tmp_path / "nope"))
        assert result.steps == []
        assert result.total_opt_steps is None

    @pytest.mark.light
    def test_optimized_hamiltonian_picks_highest_step(self, tmp_path):
        _bootstrap_workflow_dir(tmp_path, output_text=_VMC_LOG_2STEPS)
        for step in (0, 5, 12, 3):
            (tmp_path / f"hamiltonian_data_opt_step_{step}.h5").touch()
        result = parse_vmc_output(str(tmp_path))
        assert result.optimized_hamiltonian.endswith("hamiltonian_data_opt_step_12.h5")

    @pytest.mark.light
    def test_optimized_hamiltonian_none_when_no_file(self, tmp_path):
        _bootstrap_workflow_dir(tmp_path, output_text=_VMC_LOG_2STEPS)
        result = parse_vmc_output(str(tmp_path))
        assert result.optimized_hamiltonian is None

    @pytest.mark.light
    def test_restart_checkpoint_captured(self, tmp_path):
        text = _VMC_LOG_2STEPS + "\nDump restart checkpoint file(s) to restart.h5.\n"
        _bootstrap_workflow_dir(tmp_path, output_text=text)
        result = parse_vmc_output(str(tmp_path))
        assert result.restart_checkpoint == "restart.h5"

    @pytest.mark.light
    def test_stderr_tail_captured(self, tmp_path):
        _bootstrap_workflow_dir(tmp_path, output_text=_VMC_LOG_2STEPS)
        (tmp_path / "stderr").write_text("warning: something\n" * 5)
        result = parse_vmc_output(str(tmp_path))
        assert "warning" in result.stderr_tail


class TestParseMcmcOutput:
    @pytest.mark.light
    def test_basic_fields_from_last_output(self, tmp_path):
        text = (
            _METADATA_HEADER
            + "\n"
            + "Average of walker weights is 0.799.\n"
            + "Acceptance ratio is 51.46 %.\n"
            + "Total elapsed time for MCMC 1000 steps. = 21.07 sec.\n"
            + "Pre-compilation time for MCMC = 11.35 sec.\n"
            + "Net total time for MCMC = 9.72 sec.\n"
            + "Dump restart checkpoint file(s) to mcmc_restart.h5.\n"
        )
        _bootstrap_workflow_dir(tmp_path, output_text=text)
        result = parse_mcmc_output(str(tmp_path))
        assert result.avg_walker_weight == pytest.approx(0.799)
        assert result.acceptance_ratio == pytest.approx(0.5146)
        assert result.total_time_sec == pytest.approx(21.07)
        assert result.net_time_sec == pytest.approx(9.72)
        assert result.precompilation_time_sec == pytest.approx(11.35)
        assert result.restart_checkpoint == "mcmc_restart.h5"
        assert result.num_mpi_processes == 4

    @pytest.mark.light
    def test_energy_from_state_result_section(self, tmp_path):
        _bootstrap_workflow_dir(
            tmp_path,
            output_text="some output",
            result_section={"energy": -1.234, "energy_error": 1e-3},
        )
        result = parse_mcmc_output(str(tmp_path))
        assert result.energy == pytest.approx(-1.234)
        assert result.energy_error == pytest.approx(1e-3)  # tol-ok: golden parsed value from a text fixture, not a tolerance

    @pytest.mark.light
    def test_forces_stress_pressure_from_state(self, tmp_path):
        _bootstrap_workflow_dir(
            tmp_path,
            output_text="x",
            result_section={
                "forces": [{"label": "H", "Fx": 1.0}],
                "stress": {"sigma_xx": 0.1},
                "pressure": {"P_hydro": -0.1},
            },
        )
        result = parse_mcmc_output(str(tmp_path))
        assert result.atomic_forces == [{"label": "H", "Fx": 1.0}]
        assert result.stress_tensor == {"sigma_xx": 0.1}
        assert result.pressure == {"P_hydro": -0.1}

    @pytest.mark.light
    def test_hamiltonian_data_file_from_input_toml(self, tmp_path):
        input_text = '[control]\nhamiltonian_h5 = "ham.h5"\n'
        _bootstrap_workflow_dir(
            tmp_path,
            input_text=input_text,
            output_text="x",
        )
        result = parse_mcmc_output(str(tmp_path))
        assert result.hamiltonian_data_file == "ham.h5"

    @pytest.mark.light
    def test_missing_directory_returns_empty(self, tmp_path):
        result = parse_mcmc_output(str(tmp_path / "nope"))
        assert result.energy is None
        assert result.acceptance_ratio is None


class TestParseLrdmcOutput:
    @pytest.mark.light
    def test_basic_lrdmc_fields(self, tmp_path):
        text = (
            _METADATA_HEADER
            + "\nSurvived walkers ratio = 95.2 %\n"
            + "Average of the number of projections  = 123\n"
            + "Total GFMC time for 1000 branching steps = 3000.0 sec.\n"
            + "Pre-compilation time for GFMC = 167.674 sec.\n"
            + "Net GFMC time without pre-compilations = 2832.326 sec.\n"
            + "Dump restart checkpoint file(s) to lrdmc.h5.\n"
        )
        _bootstrap_workflow_dir(tmp_path, output_text=text)
        result = parse_lrdmc_output(str(tmp_path))
        assert result.survived_walkers_ratio == pytest.approx(0.952)
        assert result.avg_num_projections == pytest.approx(123.0)
        assert result.total_time_sec == pytest.approx(3000.0)
        assert result.precompilation_time_sec == pytest.approx(167.674)
        assert result.net_time_sec == pytest.approx(2832.326)
        assert result.restart_checkpoint == "lrdmc.h5"

    @pytest.mark.light
    def test_missing_dir_returns_empty(self, tmp_path):
        result = parse_lrdmc_output(str(tmp_path / "nope"))
        assert result.net_time_sec is None

    @pytest.mark.light
    def test_energy_from_state(self, tmp_path):
        _bootstrap_workflow_dir(
            tmp_path,
            output_text="x",
            result_section={"energy": -2.5, "energy_error": 1e-4},
        )
        result = parse_lrdmc_output(str(tmp_path))
        assert result.energy == pytest.approx(-2.5)
        assert result.energy_error == pytest.approx(1e-4)  # tol-ok: golden parsed value from a text fixture, not a tolerance


class TestParseLrdmcExtOutput:
    @pytest.mark.light
    def test_extrapolated_energy_parsed(self, tmp_path):
        text = "For a -> 0 bohr: E = -76.4385 +- 0.0123 Ha.\n"
        _bootstrap_workflow_dir(tmp_path, output_text=text)
        result = parse_lrdmc_ext_output(str(tmp_path))
        assert result.extrapolated_energy == pytest.approx(-76.4385)
        assert result.extrapolated_energy_error == pytest.approx(
            0.0123
        )  # tol-ok: golden parsed value from a text fixture, not a tolerance

    @pytest.mark.light
    def test_per_alat_results_from_state(self, tmp_path):
        per_alat = [
            {"alat": 0.1, "energy": -1.0, "energy_error": 0.001},
            {"alat": 0.2, "energy": -0.99, "energy_error": 0.001},
        ]
        _bootstrap_workflow_dir(
            tmp_path,
            output_text="For a -> 0 bohr: E = -1.0 +- 0.001 Ha.\n",
            result_section={"per_alat_results": per_alat},
        )
        result = parse_lrdmc_ext_output(str(tmp_path))
        assert result.per_alat_results == per_alat

    @pytest.mark.light
    def test_missing_dir_returns_empty(self, tmp_path):
        result = parse_lrdmc_ext_output(str(tmp_path / "nope"))
        assert result.extrapolated_energy is None


class TestParseInputParams:
    @pytest.mark.light
    def test_parses_per_input(self, tmp_path):
        input_text = textwrap.dedent("""
            [control]
            job_type = "mcmc"
            hamiltonian_h5 = "ham.h5"

            [mcmc]
            num_mcmc_steps = 500
            Dt = 1.5
        """).strip()
        _bootstrap_workflow_dir(tmp_path, input_text=input_text, output_text="x")
        result = parse_input_params(str(tmp_path))
        assert len(result.per_input) == 1
        entry = result.per_input[0]
        assert entry["job_type"] == "mcmc"
        assert entry["control"]["job_type"] == "mcmc"
        assert entry["mcmc"]["num_mcmc_steps"] == 500
        # Defaults merged: Dt explicit, num_mcmc_warmup_steps default
        assert "num_mcmc_warmup_steps" in entry["mcmc"]

    @pytest.mark.light
    def test_missing_dir_returns_empty(self, tmp_path):
        result = parse_input_params(str(tmp_path / "nope"))
        assert result.per_input == []
        assert result.actual_opt_steps is None

    @pytest.mark.light
    def test_unknown_job_type_keeps_raw_section(self, tmp_path):
        input_text = textwrap.dedent("""
            [control]
            job_type = "custom_unknown"

            [custom_unknown]
            x = 1
        """).strip()
        _bootstrap_workflow_dir(tmp_path, input_text=input_text, output_text="x")
        result = parse_input_params(str(tmp_path))
        assert result.per_input[0]["custom_unknown"] == {"x": 1}


class TestRepairForcesFromOutput:
    @pytest.mark.light
    def test_writes_forces_to_state(self, tmp_path):
        _bootstrap_workflow_dir(tmp_path, output_text=_FORCE_TABLE)
        ok = repair_forces_from_output(str(tmp_path))
        assert ok is True
        state = toml.load(tmp_path / "workflow_state.toml")
        assert "forces" in state["result"]
        assert len(state["result"]["forces"]) == 3

    @pytest.mark.light
    def test_returns_false_when_no_state_toml(self, tmp_path):
        assert repair_forces_from_output(str(tmp_path)) is False

    @pytest.mark.light
    def test_returns_false_when_no_output(self, tmp_path):
        # state.toml exists but no [[jobs]] with status=fetched -> no output files found
        (tmp_path / "workflow_state.toml").write_text("[workflow]\nlabel = 'x'\n")
        assert repair_forces_from_output(str(tmp_path)) is False

    @pytest.mark.light
    def test_returns_false_when_no_force_table(self, tmp_path):
        _bootstrap_workflow_dir(tmp_path, output_text="no forces here")
        assert repair_forces_from_output(str(tmp_path)) is False


class TestInternalHelpers:
    @pytest.mark.light
    def test_read_text_returns_none_on_missing(self, tmp_path):
        assert _read_text(str(tmp_path / "missing")) is None

    @pytest.mark.light
    def test_read_text_round_trip(self, tmp_path):
        p = tmp_path / "x.txt"
        p.write_text("hello\n")
        assert _read_text(str(p)) == "hello\n"

    @pytest.mark.light
    def test_tail_returns_last_n_lines(self):
        text = "\n".join(str(i) for i in range(500))
        out = _tail(text, max_lines=10)
        assert out.splitlines()[-1] == "499"
        assert len(out.splitlines()) == 10

    @pytest.mark.light
    def test_tail_shorter_than_max_returns_full(self):
        text = "a\nb\nc\n"
        assert _tail(text, max_lines=10) == text

    @pytest.mark.light
    def test_find_input_files_filters_unfetched(self, tmp_path):
        (tmp_path / "i1.toml").write_text("x")
        (tmp_path / "i2.toml").write_text("x")
        state = {
            "jobs": [
                {"input_file": "i1.toml", "status": "fetched", "step": 0},
                {"input_file": "i2.toml", "status": "submitted", "step": 1},  # filtered
            ]
        }
        (tmp_path / "workflow_state.toml").write_text(toml.dumps(state))
        files = _find_input_files(str(tmp_path))
        assert len(files) == 1
        assert files[0].endswith("i1.toml")

    @pytest.mark.light
    def test_find_input_files_empty_when_no_state(self, tmp_path):
        assert _find_input_files(str(tmp_path)) == []

    @pytest.mark.light
    def test_find_input_files_sorted_by_step(self, tmp_path):
        (tmp_path / "a.toml").write_text("x")
        (tmp_path / "b.toml").write_text("x")
        (tmp_path / "c.toml").write_text("x")
        state = {
            "jobs": [
                {"input_file": "c.toml", "status": "fetched", "step": 2},
                {"input_file": "a.toml", "status": "fetched", "step": 0},
                {"input_file": "b.toml", "status": "fetched", "step": 1},
            ]
        }
        (tmp_path / "workflow_state.toml").write_text(toml.dumps(state))
        files = _find_input_files(str(tmp_path))
        assert [Path(f).name for f in files] == ["a.toml", "b.toml", "c.toml"]

    @pytest.mark.light
    def test_find_output_files_mirrors_input(self, tmp_path):
        (tmp_path / "o.o").write_text("x")
        state = {"jobs": [{"output_file": "o.o", "status": "fetched", "step": 0}]}
        (tmp_path / "workflow_state.toml").write_text(toml.dumps(state))
        files = _find_output_files(str(tmp_path))
        assert len(files) == 1

    @pytest.mark.light
    def test_find_hamiltonian_h5_extracts_from_input(self, tmp_path):
        (tmp_path / "i.toml").write_text('[control]\nhamiltonian_h5 = "ham.h5"\n')
        state = {"jobs": [{"input_file": "i.toml", "status": "fetched", "step": 0}]}
        (tmp_path / "workflow_state.toml").write_text(toml.dumps(state))
        assert _find_hamiltonian_h5(str(tmp_path)) == "ham.h5"

    @pytest.mark.light
    def test_find_hamiltonian_h5_returns_none_when_missing(self, tmp_path):
        (tmp_path / "i.toml").write_text("[control]\n")  # no hamiltonian_h5
        state = {"jobs": [{"input_file": "i.toml", "status": "fetched", "step": 0}]}
        (tmp_path / "workflow_state.toml").write_text(toml.dumps(state))
        assert _find_hamiltonian_h5(str(tmp_path)) is None


class TestStressTimerBreakdown:
    """The MCMC and GFMC drivers print the periodic stress cost on its own timer line (issue #620)."""

    _MCMC = textwrap.dedent(
        """\
        Elapsed times per MCMC step, averaged over 100 steps.
          Time for MCMC update = 13.22 msec.
          Time for computing e_L = 20.10 msec.
          Time for computing de_L/dR and de_L/dr = 0.00 msec.
          Time for computing dln_Psi/dR and dln_Psi/dr = 0.00 msec.
          Time for computing dln_Psi/dc = 0.00 msec.
          Time for computing de_L/dc = 0.00 msec.
          Time for computing de_L/deps and dln_Psi/deps = 6270.51 msec.
          Time for MPI barrier after MCMC update = 0.00 msec.
          Time for misc. (others) = 15.11 msec.
        """
    )
    _GFMC_T = textwrap.dedent(
        """\
        Elapsed times per branching, averaged over 100 branching steps.
          Projection time per branching =  583.193 msec.
          Time for Observable measurement time per branching =  1.772 msec.
          Time for computing de_L/deps and dln_Psi/deps = 6290.12 msec.
          Time for MPI barrier before branching = 0.00 msec.
          Time for walker observable collections time per branching =  5.832 msec.
          Time for walker reconfiguration time per branching =  4.155 msec.
          Time for misc. (others) = 84.06 msec.
        """
    )

    @pytest.mark.light
    def test_mcmc_stress_timer_is_parsed(self):
        from jqmc.output_parser import parse_mcmc_log

        breakdown = parse_mcmc_log(self._MCMC)["timing_breakdown"]
        assert breakdown["stress"] == pytest.approx(6270.51)
        assert breakdown["de_L_dc"] == pytest.approx(0.0)
        assert breakdown["misc"] == pytest.approx(15.11)

    @pytest.mark.light
    def test_gfmc_stress_timer_is_parsed(self):
        from jqmc.output_parser import parse_lrdmc_log

        breakdown = parse_lrdmc_log(self._GFMC_T)["timing_breakdown"]
        assert breakdown["stress"] == pytest.approx(6290.12)
        assert breakdown["collection"] == pytest.approx(5.832)
        assert breakdown["observable"] == pytest.approx(1.772)


class TestOptimizationCoordinateCount:
    @pytest.mark.light
    def test_both_counts_are_extracted(self):
        """The optimizer prints the stored values and the real coordinates (the SR/LM dimension); both are steps' fields."""
        from jqmc.output_parser import parse_vmc_steps, unparsed_numeric_lines

        text = textwrap.dedent("""
            Optimization step = 1/1.
            E = -61.20306 +- 0.00546 Ha
            Number of variational parameters = 11461.
            Number of real optimization coordinates = 7975.
        """).strip()
        (step,) = parse_vmc_steps(text)
        assert step["num_variational_parameters"] == 11461
        assert step["num_optimization_coordinates"] == 7975
        assert unparsed_numeric_lines("All rights reserved.\n" + text) == []

    @pytest.mark.light
    def test_twist_average_lines_are_extracted(self):
        """A twist run's all-twists counts, per-twist local max S/N, twist-labelled top rows and blocks are steps' fields."""
        from jqmc.output_parser import parse_vmc_steps, unparsed_numeric_lines

        text = textwrap.dedent("""
            Optimization step = 2/2.
            Number of variational parameters = 12.
            Number of variational parameters (all twists) = 22.
            Number of real optimization coordinates = 12.
            Number of real optimization coordinates (all twists) = 22.
              Coordinates: real=12, storage=12, active=12, frozen=0 in 0 atomic group(s).
              Coordinates (all twists): real=22, storage=22, active=21, frozen=1 in 1 atomic group(s).
              Top 2 coordinates by generalized-force signal-to-noise:
                mo_coefficients_up[t1:q7]: force=+5.172833e-01 +- 2.339719e-01, signal-to-noise=2.210878e+00, metric_diag=2.495557e+00.
                j1_param[q0]: force=+1.393904e-01 +- 6.330472e-02, signal-to-noise=2.201896e+00, metric_diag=4.609823e-02.
              [optimizer block] j1_param: ||force||=1.393904e-01, max|force|=1.393904e-01, max(S/N)=2.201896e+00, metric_diag[min=4.609823e-02, max=4.609823e-02].
                update: ||delta_q||=1.0e-02, max|delta_q|=1.0e-02, ||delta_p||=1.0e-02, max|delta_p|=1.0e-02.
              [optimizer block] mo_coefficients_up[t1]: ||force||=6.0e-01, max|force|=5.172833e-01, max(S/N)=2.210878e+00, metric_diag[min=1.0e-01, max=2.495557e+00].
                update: ||delta_q||=2.0e-02, max|delta_q|=1.5e-02, ||delta_p||=2.0e-02, max|delta_p|=1.5e-02.
            Max generalized force = 0.517 +- 0.234 Ha/a.u.
            Max generalized force signal-to-noise = 2.211.
              Twist 0 local max generalized force signal-to-noise = 2.140.
              Twist 1 local max generalized force signal-to-noise = 2.211.
        """).strip()
        (step,) = parse_vmc_steps(text)
        assert (step["num_variational_parameters"], step["num_variational_parameters_all_twists"]) == (12, 22)
        assert (step["num_optimization_coordinates"], step["num_optimization_coordinates_all_twists"]) == (12, 22)
        assert [row["coordinate"] for row in step["top_coordinates"]] == ["mo_coefficients_up[t1:q7]", "j1_param[q0]"]
        assert step["top_coordinates"][0]["signal_to_noise"] == 2.210878
        assert step["diagnostics"]["coordinates_all_twists"] == dict(real=22, storage=22, active=21, frozen=1, atomic_groups=1)
        assert step["diagnostics"]["coordinates"]["real"] == 12
        assert [block["block"] for block in step["blocks"]] == ["j1_param", "mo_coefficients_up[t1]"]
        assert step["blocks"][1]["update"]["max_abs_delta_p"] == 1.5e-02
        assert (step["max_force"], step["max_force_error"], step["signal_to_noise_ratio"]) == (0.517, 0.234, 2.211)
        assert step["twist_local_max_signal_to_noise"] == [
            {"twist": 0, "signal_to_noise": 2.140},
            {"twist": 1, "signal_to_noise": 2.211},
        ]
        assert unparsed_numeric_lines("All rights reserved.\n" + text) == []


class TestAsrFallbackAndStiefelLines:
    @pytest.mark.light
    def test_fallback_clipping_and_stiefel_lines_are_extracted(self):
        """Every aSR fallback, the clipped discriminant and each block's S-Stiefel tangent are steps' fields."""
        from jqmc.output_parser import parse_vmc_steps, unparsed_numeric_lines

        text = textwrap.dedent("""
            Optimization step = 1/3.
            E = -61.20306 +- 0.00546 Ha
            aSR: discriminant is negative (-1.250e-08); setting it to zero.
            aSR: gamma+ = 0.300000, gamma- = 5.000000.
            aSR: selected gamma+ = 0.300000 (smaller-|gamma|, reference-connected root).
            aSR: selected reference-connected root gamma=0.300000 exceeds |gamma_max|=0.200000; falling back to plain SR with the model's sign (gamma=0.100000).
            aSR: selected gamma produced an invalid model denominator; using gamma = 0.
            aSR: final gamma = 0.000000.
              [optimizer block] mo_coefficients_up[t1]: ||force||=7.363022e-01, max|force|=5.172833e-01, max(S/N)=2.210878e+00, metric_diag[min=3.956989e-01, max=2.495557e+00].
                update: ||delta_q||=4.017986e-02, max|delta_q|=2.876777e-02, ||delta_p||=4.121985e-02, max|delta_p|=2.532409e-02.
                S-Stiefel tangent: ||delta_C S^(1/2)||_F=2.547088e-02, ||delta_C S^(1/2)||_2=1.602374e-02.
            Optimization step = 2/3.
            E = -61.21000 +- 0.00500 Ha
            aSR: centered metric contraction S_2=-3.000000e-04 is negative; falling back to plain SR (gamma=0.100000).
            aSR: final gamma = 0.100000.
            Optimization step = 3/3.
            E = -61.22000 +- 0.00400 Ha
            aSR: denom = -2*H_1*S_2 ~ 0 (H_1=1.000e-20, S_2=2.000e-03); no meaningful optimization direction. Returning gamma = 0.
            aSR: selected root is not finite/nonzero (gamma=nan); falling back to plain SR (gamma=0.100000).
        """).strip()
        first, second, third = parse_vmc_steps(text)
        assert first["asr"]["clipped_discriminant"] == -1.25e-8
        assert first["asr"]["fallbacks"] == [
            {"reason": "root_exceeds_gamma_max", "root_gamma": 0.3, "gamma_max": 0.2, "gamma": 0.1},
            {"reason": "invalid_model_denominator", "gamma": 0.0},
        ]
        (block,) = first["blocks"]
        assert block["s_stiefel_tangent"] == {"frobenius_norm": 2.547088e-2, "spectral_norm": 1.602374e-2}
        assert second["asr"]["fallbacks"] == [{"reason": "negative_centered_metric", "S_2": -3.0e-4, "gamma": 0.1}]
        degenerate, not_finite = third["asr"]["fallbacks"]
        assert degenerate == {"reason": "degenerate_denominator", "H_1": 1.0e-20, "S_2": 2.0e-3, "gamma": 0.0}
        assert not_finite["reason"] == "root_not_finite_or_zero" and not_finite["gamma"] == 0.1
        assert math.isnan(not_finite["root_gamma"])
        assert third["asr"]["notes"] == [
            "selected root is not finite/nonzero (gamma=nan); falling back to plain SR (gamma=0.100000)"
        ]
        assert unparsed_numeric_lines("All rights reserved.\n" + text) == []


class TestLinearMethodTiledOrbitalAndDerivativeSourceLines:
    @pytest.mark.light
    def test_lm_and_derivative_source_lines_are_extracted(self):
        """LM conditioning/state/subspace/decision lines and a derivative source without D are steps' fields."""
        from jqmc.output_parser import parse_vmc_steps, unparsed_numeric_lines

        text = textwrap.dedent("""
            Optimization step = 1/2.
            E = -2.27118 +- 0.00321 Ha
              Derivative source j2_param: O shape=(120,), D shape=(120,), policy=direct_real, tangent=euclidean.
              LM conditioning input: coordinates=28, cutoff=1.000000e-03, diag[min=3.729200e-05, max=1.587520e-01].
              LM conditioning: active=24/28, tiny-diagonal removed=0, correlation removed=4, min correlation eigenvalue of the active set=1.391902e-03 (cutoff=1.000e-03).
              LM selected state: reference weight=0.99368148, ||q||=2.016177e+01, max|q|=1.808844e+01.
              LM subspace: SR collective + 28 coordinate(s) in 28 atomic group(s).
              LM decision: accepted (reference weight=0.99368148).
            Optimization step = 2/2.
            E = -2.28000 +- 0.00300 Ha
              Derivative source j3_matrix: O shape=(120, 6, 7), D shape=None, policy=direct_real, tangent=linear_tie.
              LM conditioning: active=3/3, tiny-diagonal removed=0, correlation removed=0, min correlation eigenvalue of the active set=7.951070e-02 (cutoff=1.000e-03).
              LM decision: rejected; fallback scale=1.000000e-01 (reference weight 0.80000000 is below 0.90000000; aSR fallback gamma=0.10000000).
              Rejected LM candidate reference weight=0.80000000.
        """).strip()
        first, second = parse_vmc_steps(text)
        assert first["derivative_sources"][0]["D_shape"] == [120]
        assert first["lm"] == {
            "conditioning_input": {"coordinates": 28, "cutoff": 1.0e-3, "diag_min": 3.7292e-5, "diag_max": 1.58752e-1},
            "conditioning": {
                "active": 24,
                "coordinates": 28,
                "tiny_diagonal_removed": 0,
                "correlation_removed": 4,
                "min_correlation_eigenvalue": 1.391902e-3,
                "cutoff": 1.0e-3,
            },
            "selected_state": {"reference_weight": 0.99368148, "q_norm": 2.016177e1, "max_abs_q": 1.808844e1},
            "subspace": {"coordinates": 28, "atomic_groups": 28},
            "decision": {"accepted": True, "reference_weight": 0.99368148},
        }
        assert second["derivative_sources"] == [
            {"block": "j3_matrix", "O_shape": [120, 6, 7], "D_shape": None, "policy": "direct_real", "tangent": "linear_tie"}
        ]
        assert second["lm"]["decision"] == {
            "accepted": False,
            "fallback_scale": 0.1,
            "reason": "reference weight 0.80000000 is below 0.90000000; aSR fallback gamma=0.10000000",
            "reference_weight": 0.8,
        }
        assert unparsed_numeric_lines(text) == []

    @pytest.mark.light
    def test_tiled_orbital_summary_is_extracted(self):
        """Each spin channel's TiledOrbitalSet_data summary is a system-info entry; np.float64 supertwists parse too."""
        from jqmc.output_parser import parse_system_info, unparsed_numeric_lines

        text = textwrap.dedent("""
            **TiledOrbitalSet_data (frozen)
              spin=up, supertwist=(0.0, 0.0, 0.0), tiling=((2, 0, 0), (0, 1, 0), (0, 0, 1))
              primitive k blocks=2, occupied orbitals=2
              local basis sizes=(26, 26)
              orbital offsets=(0, 1, 2), normalization=primitive-cell-to-supercell
            **TiledOrbitalSet_data (frozen)
              spin=down, supertwist=(np.float64(-0.0), np.float64(-0.0), np.float64(-0.0)), tiling=((2, 0, 0), (0, 1, 0), (0, 0, 1))
              primitive k blocks=2, occupied orbitals=2
              local basis sizes=(26, 26)
              orbital offsets=(0, 1, 2), normalization=primitive-cell-to-supercell
        """).strip()
        up, down = parse_system_info(text)["tiled_orbitals"]
        assert up == {
            "spin": "up",
            "supertwist": [0.0, 0.0, 0.0],
            "tiling": [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
            "num_k_blocks": 2,
            "num_occupied_orbitals": 2,
            "local_basis_sizes": [26, 26],
            "orbital_offsets": [0, 1, 2],
            "normalization": "primitive-cell-to-supercell",
        }
        assert down["spin"] == "down" and down["supertwist"] == [-0.0, -0.0, -0.0]
        assert unparsed_numeric_lines(text) == []
        with pytest.raises(ValueError, match="incomplete TiledOrbitalSet_data summary"):
            parse_system_info(text.rsplit("\n", 1)[0])


class TestParameterUpdateLines:
    @pytest.mark.light
    def test_update_lines_of_every_method_are_extracted(self):
        """The per-block update lines of SR, aSR and LM steps, and the no-update form, are steps' parameter updates."""
        from jqmc.output_parser import parse_vmc_steps, unparsed_numeric_lines

        text = textwrap.dedent("""
            Optimization step = 1/2.
            E = -61.20306 +- 0.00546 Ha
              [SR update] - block=j1_param size=1  ||delta_p||=1.250e-02  max|delta_p|=1.250e-02  max|applied update|=2.500e-04
              [SR update] - block=j3_matrix size=210  theta=ALL ZERO (no update)
            Optimization step = 2/2.
            E = -61.21000 +- 0.00500 Ha
              [aSR update] - block=j1_param size=1  ||delta_p||=3.000e-03  max|delta_p|=3.000e-03  max|applied update|=6.000e-05
              [LM update] - block=lambda_matrix size=36  ||delta_p||=4.000e-01  max|delta_p|=1.000e-01  max|applied update|=1.000e-01
        """).strip()
        first, second = parse_vmc_steps(text)
        assert first["parameter_updates"] == [
            {
                "method": "SR",
                "block": "j1_param",
                "size": 1,
                "delta_p_norm": 1.25e-2,
                "max_abs_delta_p": 1.25e-2,
                "max_abs_applied_update": 2.5e-4,
            },
            {
                "method": "SR",
                "block": "j3_matrix",
                "size": 210,
                "delta_p_norm": 0.0,
                "max_abs_delta_p": 0.0,
                "max_abs_applied_update": 0.0,
            },
        ]
        assert [(row["method"], row["block"], row["delta_p_norm"]) for row in second["parameter_updates"]] == [
            ("aSR", "j1_param", 3.0e-3),
            ("LM", "lambda_matrix", 4.0e-1),
        ]
        assert unparsed_numeric_lines("All rights reserved.\n" + text) == []


class TestVmcOptimizerTimerLabel:
    """A real one-step ``run_optimize`` names the solver that ran in its timer line, and the parser reads it (issue #857).

    The run is H2 (all-electron cc-pVDZ, Pade J1 + J2) with the production
    sampler and optimizer. ``use_lm = true`` with ``lm_subspace_dim = 0`` is
    aSR, with a nonzero ``lm_subspace_dim`` it is the (reduced or full) LM.
    Only the label and its complete extraction are checked; the printed
    duration varies between machines and is not compared with a number.
    """

    _TREXIO = Path(__file__).resolve().parents[1] / "trexio_example_files" / "H2_ae_ccpvdz_cart.h5"
    _RE_LINE = r"Time for optimizer \((\w+)\) = ([0-9.eE+-]+) sec\."

    def _run_and_capture_log(self, optimizer_kwargs, caplog) -> str:
        import logging

        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_two_body_data
        from jqmc.jqmc_mcmc import MCMC
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data

        structure_data, _aos, _mos_up, _mos_dn, geminal_mo_data, coulomb_potential_data = read_trexio_file(
            trexio_file=str(self._TREXIO), store_tuple=True
        )
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=Jastrow_one_body_data.init_jastrow_one_body_data(
                jastrow_1b_param=1.0,
                structure_data=structure_data,
                core_electrons=tuple(0.0 for _ in structure_data.atomic_numbers),
                jastrow_1b_type="pade",
            ),
            jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
                jastrow_2b_param=0.5, jastrow_2b_type="pade"
            ),
            jastrow_three_body_data=None,
        )
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data),
        )
        use_lm = bool(optimizer_kwargs.get("use_lm", False))
        mcmc = MCMC(
            hamiltonian_data=hamiltonian_data,
            Dt=2.0,
            mcmc_seed=20261113,
            num_walkers=4,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=use_lm,
        )
        with caplog.at_level(logging.INFO, logger="jqmc"):
            mcmc.run_optimize(
                num_mcmc_steps=25,
                num_opt_steps=1,
                num_mcmc_warmup_steps=0,
                num_mcmc_bin_blocks=5,
                optimizer_kwargs={"method": "sr", "delta": 0.05, "epsilon": 1.0e-3, "cg_flag": False, **optimizer_kwargs},
                opt_J1_param=True,
                opt_J2_param=True,
                opt_J3_param=False,
                dump_wavefunction=False,
                use_device_optimizer=False,
            )
        assert mcmc.optimization_step_index == 1
        return "\n".join(record.getMessage() for record in caplog.records)

    @pytest.mark.light
    @pytest.mark.parametrize(
        "optimizer_kwargs,expected_label",
        [
            ({}, "SR"),
            ({"use_lm": True, "lm_subspace_dim": 0}, "aSR"),
            ({"use_lm": True, "lm_subspace_dim": 1}, "LM"),
            ({"use_lm": True, "lm_subspace_dim": -1}, "LM"),
        ],
        ids=["sr", "asr", "reduced-lm", "full-lm"],
    )
    def test_timer_line_names_the_method_and_is_extracted(self, optimizer_kwargs, expected_label, caplog):
        import re

        from jqmc.output_parser import parse_jqmc_log, unparsed_numeric_lines

        text = self._run_and_capture_log(optimizer_kwargs, caplog)
        lines = re.findall(self._RE_LINE, text)
        assert len(lines) == 1, lines
        label, printed = lines[0]
        assert label == expected_label
        assert parse_jqmc_log(text, "vmc")["opt_timing_breakdown"]["optimizer"] == float(printed)
        assert not [line for line in unparsed_numeric_lines(text) if "Time for optimizer" in line]


class TestMemoryLines:
    """``[memory]`` lines keep the device peak that GPU runs print after the RSS peak (issue #856)."""

    # Rank-0 lines of an LRDMC and a VMC run on 4 GPUs.
    _GPU_LINES = (
        "[memory] stage=gfmc_t_post_run, RSS_peak_max=4.27 GB, JAX_peak_max=0.24 GB",
        "[memory] stage=mcmc_post_run, RSS_peak_max=6.26 GB, JAX_peak_max=4.45 GB",
    )
    # A CPU run prints the RSS peak only.
    _CPU_LINE = "[memory] stage=optimize_post_loop, RSS_peak_max=1.53 GB"

    @pytest.mark.light
    def test_device_and_rss_peaks_are_extracted(self):
        from jqmc.output_parser import parse_environment, parse_vmc_steps, unparsed_numeric_lines

        text = "\n".join(("All rights reserved.", *self._GPU_LINES, self._CPU_LINE))
        assert parse_environment(text)["memory"] == [
            {"stage": "gfmc_t_post_run", "rss_peak_max_gb": 4.27, "jax_peak_max_gb": 0.24},
            {"stage": "mcmc_post_run", "rss_peak_max_gb": 6.26, "jax_peak_max_gb": 4.45},
            {"stage": "optimize_post_loop", "rss_peak_max_gb": 1.53},
        ]
        assert unparsed_numeric_lines(text) == []

        step_text = "\n".join(("Optimization step = 1/1.", self._GPU_LINES[1], self._CPU_LINE))
        (step,) = parse_vmc_steps(step_text)
        assert step["memory"] == [
            {"stage": "mcmc_post_run", "rss_peak_max_gb": 6.26, "jax_peak_max_gb": 4.45},
            {"stage": "optimize_post_loop", "rss_peak_max_gb": 1.53},
        ]

    @pytest.mark.light
    def test_unknown_field_on_a_memory_line_is_reported(self):
        """Negative control: a number the parser does not extract from a ``[memory]`` line is listed, not dropped."""
        from jqmc.output_parser import parse_environment, unparsed_numeric_lines

        stray = "[memory] stage=mcmc_post_run, RSS_peak_max=6.26 GB, HOST_peak_max=7.00 GB"
        text = "\n".join(("All rights reserved.", self._CPU_LINE, stray))
        assert unparsed_numeric_lines(text) == [stray]
        assert parse_environment(text)["memory"] == [{"stage": "optimize_post_loop", "rss_peak_max_gb": 1.53}]

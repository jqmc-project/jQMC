"""Subprocess-driven end-to-end
CLI test on a combined twist-averaged input file.

The CLI dispatch resolver ``_resolve_twist_dispatch`` is unit-tested in
``tests/test_jqmc_kernel/test_cli_twist_dispatch.py``; the per-driver orchestrators
(``run_twist_average_*``) are tested in their own files. This test
fills the integration gap: spawn the real ``python -m jqmc.jqmc_cli``
on a minimal combined input + toml and assert that

* the auto-dispatch fires (the log contains the
  "Twist-averaged dispatch" banner with the right N_twist);
* the per-twist checkpoint files materialize under
  ``[control].restart_chk`` (treated as a directory in twist-averaged
  mode);
* the BZ-mean ``E_TA`` shows up in the final output block;
* a follow-up ``restart = true`` run skips the completed twists (the
  resume contract also runs through the CLI
  end-to-end, not just through the driver-level test).

The test deliberately uses a tiny configuration (2 twists, 2 walkers,
few MCMC steps) so the subprocess returns in seconds; the goal is
pipeline-wiring coverage, not numerical validation.
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
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import h5py

from jqmc import output_parser
from jqmc._precision import is_mixed_precision_enabled
from tests._test_data_cache import qe_example_files_root

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Wall-clock budget for a single ``mpirun -n N python -m jqmc.jqmc_cli`` call.
# The body of each test runs in seconds once compiled, but the FIRST CLI
# subprocess of a job pays the full cold XLA compile. The composite
# ``pgto+bspline`` AO path compiles markedly slower than ``pgto`` / ``pgto+pw``
# and, on a cold ``JAX_COMPILATION_CACHE_DIR``, can take several hundred seconds
# under ``mpirun -n 2``. With pytest-xdist the two ``[pgto+bspline]``
# parametrizations (auto-dispatch + resume) land on different workers and each
# pays its own cold compile in parallel, so BOTH can blow a short cap on a slow
# machine even though the code is correct. The cap stays finite so a genuine
# hang still fails the test rather than hanging the run.
_CLI_SUBPROCESS_TIMEOUT_S = 1800

_LI_FIXTURE = Path(__file__).parent.parent / "trexio_example_files" / "lithium_twist_222.h5"
_li_available = _LI_FIXTURE.is_file()
requires_h_fixture = pytest.mark.skipif(not _li_available, reason="lithium_twist_222.h5 fixture not present")

# QE-side fixture for the composite-kind combined builder. The single-k Li_k025 fixture
# (n_twist = 1) is the smallest available QE save that exercises the
# non-Gamma complex composite path; the test does not need many
# twists, only a /twists/twist_*/composite_child override under each.
_QE_FIXTURE = qe_example_files_root() / "Li_k025" / "out" / "li_k025.save"
_QE_UPF = qe_example_files_root() / "Li.ccECP.upf"
_qe_available = (_QE_FIXTURE / "data-file-schema.xml").is_file() and (_QE_FIXTURE / "wfc1.hdf5").is_file() and _QE_UPF.is_file()
requires_qe_fixture = pytest.mark.skipif(not _qe_available, reason="Li_k025 QE fixture not present")


# Per-kind combined builder + expected twist count. The pgto combined file is
# built from the TREXIO 8-point fixture (truncated to 2 twists for
# cross-twist orchestration coverage); the composite kinds retain the
# QE fixture's single twist. Both are combined inputs and
# therefore take the same twist-average CLI route.
def _expected_n_twist(kind: str) -> int:
    return 2 if kind == "pgto" else 1


def _build_combined_input(out_path: str, kind: str = "pgto") -> None:
    """Build a combined twist-averaged input file for the given projection ``kind``.

    Two paths:

    * ``kind == "pgto"``: TREXIO-driven. Reads the first 2 of the 8
      lithium_twist_222 twists into the per-twist writer (kept to 2 to
      bound subprocess run time; the test does not benchmark BZ
      sweeps). The shared AO basis is the k-independent pGTO basis;
      no ``composite_child`` overrides are emitted.
    * ``kind in {"pgto+pw", "pgto+bspline"}``: QE-driven. Calls
      :func:`jqmc.qe_to_jqmc.convert_qe_to_twist_average_input` on the
      Li_k025 save (n_twist=1) so the writer emits the per-twist
      ``composite_child`` group. Keeping
      the resulting one-twist combined file also verifies that the schema,
      rather than the number of twists, selects the CLI route.

    The chosen ``kind`` must match the fixture availability checked on
    the test (``requires_h_fixture`` for pgto, ``requires_qe_fixture``
    for composite kinds).
    """
    if kind == "pgto":
        from jqmc._twist_average import save_twist_average_input
        from jqmc.trexio_wrapper import (
            read_trexio_file,
            read_trexio_kpoint_list,
        )

        kpts = read_trexio_kpoint_list(str(_LI_FIXTURE))
        n_keep = 2
        k_points = [kpts[i][0] for i in range(n_keep)]
        weights = [float(kpts[i][1]) for i in range(n_keep)]

        mo_up_list, mo_dn_list = [], []
        template_ham = None
        for ik in range(n_keep):
            sd, _, _, _, gem, cpd = read_trexio_file(
                trexio_file=str(_LI_FIXTURE),
                store_tuple=True,
                twist_ik=ik,
            )
            if template_ham is None:
                from jqmc.hamiltonians import Hamiltonian_data
                from jqmc.jastrow_factor import Jastrow_data
                from jqmc.wavefunction import Wavefunction_data

                template_ham = Hamiltonian_data(
                    structure_data=sd,
                    coulomb_potential_data=cpd,
                    wavefunction_data=Wavefunction_data(jastrow_data=Jastrow_data(), geminal_data=gem),
                )
            mo_up_list.append(np.asarray(gem.orb_data_up_spin.mo_coefficients, dtype=np.complex128))
            mo_dn_list.append(np.asarray(gem.orb_data_dn_spin.mo_coefficients, dtype=np.complex128))

        save_twist_average_input(
            filepath=out_path,
            template_hamiltonian=template_ham,
            k_points=k_points,
            weights=weights,
            mo_coefficients_up_per_twist=mo_up_list,
            mo_coefficients_dn_per_twist=mo_dn_list,
        )
        return
    if kind in ("pgto+pw", "pgto+bspline"):
        from jqmc.qe_to_jqmc import convert_qe_to_twist_average_input
        from jqmc.upf_wrapper import parse_gaussinfo

        ecp = parse_gaussinfo(str(_QE_UPF))
        convert_qe_to_twist_average_input(
            qe_save_dir=str(_QE_FIXTURE),
            output_path=out_path,
            upf_paths={"Li": str(_QE_UPF)},
            kind=kind,
            ecp_by_element={"Li": ecp},
            j1_param=None,
            j2_param=None,
        )
        return
    raise ValueError(f"_build_combined_input: unsupported kind={kind!r}")


_MINIMAL_TOML = """\
[control]
job_type = "mcmc"
hamiltonian_h5 = "{ham_h5}"
restart_chk = "{ckpt_dir}"
restart = {restart}
mcmc_seed = 12345
number_of_walkers = 2
max_time = 86400
verbosity = "low"

[mcmc]
num_mcmc_steps = 20
num_mcmc_warmup_steps = 5
num_mcmc_bin_blocks = 3
num_mcmc_per_measurement = 4
Dt = 2.0
epsilon_AS = 0.0
atomic_force = false
compute_stress = false
use_swct = true
parameter_derivatives = false

[precision]
mode = "{precision}"
"""


def _write_toml(tmp_path: Path, *, ham_h5: str, ckpt_dir: str, restart: bool) -> Path:
    toml_path = tmp_path / "input.toml"
    toml_path.write_text(
        _MINIMAL_TOML.format(
            ham_h5=ham_h5,
            ckpt_dir=ckpt_dir,
            restart="true" if restart else "false",
            precision="mixed" if is_mixed_precision_enabled() else "full",
        )
    )
    return toml_path


import shutil

_MPIRUN = shutil.which("mpirun") or shutil.which("mpiexec")
requires_mpirun = pytest.mark.skipif(
    _MPIRUN is None,
    reason="mpirun/mpiexec not on PATH; multi-rank CLI dispatch test needs MPI runner",
)


def _run_cli(toml_path: Path, cwd: Path, *, n_ranks: int = 2) -> subprocess.CompletedProcess:
    """Spawn ``mpirun -n N python -m jqmc.jqmc_cli`` on the given toml.
    We use the module-execution form rather than the ``jqmc``
    console-script entry so the test is independent of whether the
    package is installed.

    The twist-averaged driver requires ``world_size % n_twist == 0``
    (equal compute per twist), so a 2-twist input needs
    >= 2 ranks. Tests skip when mpirun is unavailable.
    """
    return subprocess.run(
        _cli_command(toml_path, n_ranks),
        cwd=str(cwd),
        env=_cli_env(),
        capture_output=True,
        text=True,
        timeout=_CLI_SUBPROCESS_TIMEOUT_S,
    )


def _mpirun_command(n_ranks: int, argv: list[str]) -> list[str]:
    return [_MPIRUN, "-n", str(n_ranks), *argv]


def _cli_command(toml_path: Path, n_ranks: int) -> list[str]:
    return _mpirun_command(n_ranks, [sys.executable, "-m", "jqmc.jqmc_cli", str(toml_path)])


# Open MPI binds rank i to core i by default, and every ``mpirun`` numbers the
# cores from 0 on its own. Concurrent launches on one host -- the two jobs of
# the auto-dispatch test, the xdist workers of this file and other concurrent
# MPI tests -- would all be pinned to cores 0 and 1 and can hit the subprocess
# timeout (issue #50). Unbound ranks
# inherit the launching process's CPU set instead. Other MPI launchers (MPICH
# Hydra does not bind by default) ignore this Open MPI MCA variable.
_MPI_UNBOUND_ENV = {"OMPI_MCA_hwloc_base_binding_policy": "none"}


def _cli_env(**overrides: str) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    # The subprocess runs from ``tmp_path``. Pin this checkout ahead of any
    # unrelated editable install inherited through PYTHONPATH/site-packages.
    repo_root = str(Path(__file__).resolve().parents[2])
    inherited_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = repo_root if not inherited_pythonpath else repo_root + os.pathsep + inherited_pythonpath
    env.update(_MPI_UNBOUND_ENV)
    env.update(overrides)
    return env


def _run_cli_concurrently(jobs: list[tuple[Path, Path]], *, n_ranks: int) -> list[subprocess.CompletedProcess]:
    """Start every ``(toml_path, cwd)`` CLI job at once on this host and wait for all of them."""
    cli_jobs = [([sys.executable, "-m", "jqmc.jqmc_cli", str(toml_path)], cwd) for toml_path, cwd in jobs]
    return _run_mpirun_concurrently(cli_jobs, n_ranks=n_ranks)


def _run_mpirun_concurrently(jobs: list[tuple[list[str], Path]], *, n_ranks: int) -> list[subprocess.CompletedProcess]:
    """Start every ``(argv, cwd)`` job under ``mpirun -n n_ranks`` at once and wait for all of them.

    All jobs share ``PYTHONHASHSEED=0``. JAX's own ``mpi4py`` coordinator
    port is ``hash(hostname) % 4096 + 61440``, so under the pinned seed every
    job would compute the SAME port: the jobs can only coexist when the CLI
    picks its coordinator port independently of the string hash (two
    concurrent jobs that share a port abort with ``Unexpected task
    registered``).
    """
    env = _cli_env(PYTHONHASHSEED="0")
    # An explicit JAX coordinator pin would make the jobs collide by request.
    for pinned in ("JAX_COORDINATOR_ADDRESS", "JAX_COORDINATOR_PORT"):
        env.pop(pinned, None)
    procs = [
        subprocess.Popen(
            _mpirun_command(n_ranks, argv),
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for argv, cwd in jobs
    ]
    results = []
    try:
        for proc in procs:
            stdout, stderr = proc.communicate(timeout=_CLI_SUBPROCESS_TIMEOUT_S)
            results.append(subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr))
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
    return results


# ---------- end-to-end CLI: schema dispatch on a combined file ----------


def _fixture_marker_for(kind: str):
    """Pick the right ``requires_*`` marker for the combined builder of ``kind``.

    ``pgto`` is sourced from the TREXIO ``lithium_twist_222`` fixture;
    composite kinds use the QE ``Li_k025`` fixture.
    """
    return requires_h_fixture if kind == "pgto" else requires_qe_fixture


def _kind_id(kind: str) -> str:
    """Stable identifier for ``kind`` used in subprocess output paths
    and parametrize ids (the ``+`` char is fine in id strings but we
    keep file paths underscored)."""
    return kind.replace("+", "_")


def _assert_twist_summary_from_checkpoints(outputs, checkpoint_dir, n_twist):
    """Check both displayed summaries against independent arithmetic on saved per-twist values.

    This checks the CLI's five-decimal presentation, not a physical energy
    oracle or statistical convergence. No production BZ aggregator is called.
    """
    assert len(outputs) == 2, "both fresh and resumed output are required"
    rows = []
    for index in range(n_twist):
        with h5py.File(checkpoint_dir / f"twist_{index:03d}.h5", "r") as handle:
            meta = handle["_meta"].attrs
            rows.append([meta["normalized_twist_weight"], meta["E_k"], meta["sigma_k"]])
    weights, energies, errors = np.asarray(rows).T
    assert np.isfinite(rows).all() and np.all(weights > 0) and np.all(errors > 0)
    assert weights.sum() == 1.0 and np.any(energies != 0)
    energy = float(np.sum(weights * energies))
    error = float(np.sqrt(np.sum((weights * errors) ** 2)))
    dispersion = float(np.sqrt(np.sum(weights * (energies - energy) ** 2)))
    rounded = [float(f"{value:.5f}") for value in (energy, error, dispersion)]
    expected = np.asarray([rounded, rounded])
    reported = []
    for output in outputs:
        matches = re.findall(r"E_TA = ([-+\d.eE]+) \+- ([-+\d.eE]+) Ha \(s_twist = ([-+\d.eE]+) Ha\)", output)
        assert len(matches) == 1, "exactly one BZ energy summary is required per CLI run"
        reported.append([float(value) for value in matches[0]])
    actual = np.asarray(reported)
    np.testing.assert_array_equal(actual, expected, err_msg="CLI twist summary disagrees with its checkpoints")


def _assert_twist_checkpoint_inputs(ham_h5, checkpoint_dir, n_twist):
    """Check each completed subprocess twist's saved orbitals and k identity against the input.

    Saved final walker coordinates are observed, not supplied or screened by
    this test. This is input identity, not independent orbital correctness.
    """
    from jqmc._checkpoint import load_hamiltonian_from_checkpoint, load_rank_checkpoint
    from jqmc._twist_average import load_twist, load_twist_average_input

    meta = load_twist_average_input(str(ham_h5))
    for index in range(n_twist):
        checkpoint = checkpoint_dir / f"twist_{index:03d}.h5"
        expected = load_twist(str(ham_h5), index).wavefunction_data.geminal_data
        actual = load_hamiltonian_from_checkpoint(str(checkpoint)).wavefunction_data.geminal_data
        for spin in ("orb_data_up_spin", "orb_data_dn_spin"):
            observed_mo = np.asarray(getattr(actual, spin).mo_coefficients)
            expected_mo = np.asarray(getattr(expected, spin).mo_coefficients)
            np.testing.assert_array_equal(observed_mo, expected_mo, err_msg="CLI loaded the wrong twist orbitals")
        with h5py.File(checkpoint, "r") as handle:
            saved_k = np.asarray(handle["_meta"].attrs["k_point"])
        input_k = np.asarray(meta.k_points[index])
        np.testing.assert_array_equal(saved_k, input_k, err_msg="CLI saved the wrong twist identity")
        state = load_rank_checkpoint(str(checkpoint), rank=0)
        r_up_carts = np.asarray(state["walker_state"]["latest_r_up_carts"])
        r_dn_carts = np.asarray(state["walker_state"]["latest_r_dn_carts"])
        assert np.isfinite(r_up_carts).all() and np.isfinite(r_dn_carts).all()


class TestCLITwistDispatchSubprocess:
    @pytest.mark.heavy
    @requires_mpirun
    @pytest.mark.parametrize("kind", ["pgto", "pgto+pw", "pgto+bspline"])
    def test_cli_auto_dispatch_routes_combined_input_to_twist_average(self, tmp_path, kind):
        """End-to-end smoke: the CLI on a combined Hamiltonian prints the
        "Twist-averaged dispatch: N twists" banner and writes the per-twist
        checkpoint files. Pins that the schema routing is reachable from the actual ``jqmc`` entry-point (not just
        from a direct ``_resolve_twist_dispatch`` import).

        Parametrized over ``kind`` to exercise both the pgto-only combined layout
        (k-independent AO basis; no per-twist override group) and the
        one-twist composite kinds (per-twist ``composite_child`` group must
        be swapped by ``load_twist``).

        Two such CLI jobs run concurrently on this host under one pinned
        ``PYTHONHASHSEED`` and both must complete: with JAX's hashed
        ``mpi4py`` coordinator port they would share one coordination
        service and one job would abort.
        """
        if kind != "pgto":
            if not _qe_available:
                pytest.skip("Li_k025 QE fixture not present")
        elif not _li_available:
            pytest.skip("lithium_twist_222.h5 fixture not present")
        from jqmc._twist_checkpoint import shared_state_path, twist_checkpoint_path

        n_twist = _expected_n_twist(kind)
        ham_h5 = tmp_path / f"{_kind_id(kind)}_twists.h5"
        _build_combined_input(str(ham_h5), kind=kind)
        # Two independent jobs on the same input start together on this host,
        # as on a shared machine or a node running several jqmc jobs.
        jobs = []
        for job in ("job_a", "job_b"):
            job_dir = tmp_path / job
            job_dir.mkdir()
            ckpt_dir = job_dir / f"ckpt_{_kind_id(kind)}"
            toml = _write_toml(job_dir, ham_h5=str(ham_h5), ckpt_dir=str(ckpt_dir), restart=False)
            jobs.append((job, ckpt_dir, toml, job_dir))

        results = _run_cli_concurrently([(toml, job_dir) for _, _, toml, job_dir in jobs], n_ranks=max(2, n_twist))
        for (job, ckpt_dir, _, _), result in zip(jobs, results, strict=True):
            out = result.stdout + result.stderr
            assert result.returncode == 0, f"concurrent jqmc CLI {job} exited {result.returncode}:\n{out}"
            assert "Twist-averaged dispatch" in out, (
                f"dispatch banner not emitted -- dispatch did not fire on the combined input.\n--- stdout+stderr ---\n{out}"
            )
            assert f"{n_twist} twists" in out or (n_twist == 1 and "1 twist" in out), (
                f"expected '{n_twist} twists' banner; full output:\n{out}"
            )
            # Every twist materialised a checkpoint file under restart_chk.
            for ik in range(n_twist):
                assert os.path.isfile(twist_checkpoint_path(str(ckpt_dir), ik)), (
                    f"{job} twist_{ik:03d}.h5 missing.\nstdout+stderr:\n{out}"
                )
            assert os.path.isfile(shared_state_path(str(ckpt_dir)))
            # The BZ-mean E_TA appears in the final output, and jqmc.output_parser
            # extracts it with one per-twist row per twist of the input (#308).
            assert "E_TA" in out
            parsed = output_parser.parse_twist_average_result(result.stdout)
            assert parsed is not None, f"{job}: BZ-averaged energy not extracted"
            for name in ("energy", "energy_error", "s_twist"):
                assert parsed[name] is not None and np.isfinite(parsed[name]), f"{job}: {name}"
            assert [row["twist"] for row in parsed["per_twist"]] == list(range(n_twist)), f"{job}: per-twist rows"
            for row in parsed["per_twist"]:
                assert np.isfinite(row["energy"]) and np.isfinite(row["energy_error"]), f"{job}: twist {row['twist']}"
            summary = output_parser.parse_jqmc_log(result.stdout, "mcmc")
            assert summary["twist_average"] == parsed

    # ---------- end-to-end CLI: resume skips completed twists ----------
    @pytest.mark.heavy
    @requires_mpirun
    @pytest.mark.parametrize("kind", ["pgto", "pgto+pw", "pgto+bspline"])
    @pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
    def test_cli_resume_skips_completed_twists_end_to_end(self, tmp_path, kind):
        """The twist-resume contract holds through the CLI too:
        after a first run the checkpoint dir certifies every twist; a
        second run with ``restart = true`` finishes immediately (every
        twist is skipped), and the per-twist checkpoint files do not get
        re-written (mtime unchanged).

        Parametrized identically to the auto-dispatch test so the resume
        contract is exercised for both pgto and composite kinds (the
        per-twist ``composite_child`` group on disk must survive the
        resume contract intact).

        Require fresh and resumed E_TA, its reported error, and twist dispersion
        to equal independent arithmetic on the saved per-twist summaries at
        the CLI's five displayed decimals. Check each saved up/down MO payload
        and k identity against that input twist. These are deterministic I/O
        checks, not a sigma acceptance test or independent physical energy.
        Both precisions are explicitly passed through TOML to the subprocess.
        Use original Li fixtures, two walkers, seed 12345 and 20 measurements;
        positions are sampler-generated without cond(G) selection. Capture
        saved final coordinates, not every internal subprocess trajectory.
        """
        assert _qe_available if kind != "pgto" else _li_available, "the required committed fixture is missing"
        from jqmc._twist_checkpoint import twist_checkpoint_path

        n_twist = _expected_n_twist(kind)
        ham_h5 = tmp_path / f"{_kind_id(kind)}_twists.h5"
        _build_combined_input(str(ham_h5), kind=kind)
        ckpt_dir = tmp_path / f"ckpt_{_kind_id(kind)}"

        # First run: fresh.
        toml1 = _write_toml(tmp_path, ham_h5=str(ham_h5), ckpt_dir=str(ckpt_dir), restart=False)
        result1 = _run_cli(toml1, tmp_path, n_ranks=max(2, n_twist))
        assert result1.returncode == 0, result1.stdout + result1.stderr
        _assert_twist_checkpoint_inputs(ham_h5, ckpt_dir, n_twist)

        mtimes_before = {ik: os.path.getmtime(twist_checkpoint_path(str(ckpt_dir), ik)) for ik in range(n_twist)}

        # Allow file mtime resolution to distinguish a potential re-write.
        import time

        time.sleep(0.05)

        # Second run: resume should skip every twist.
        toml2 = _write_toml(tmp_path, ham_h5=str(ham_h5), ckpt_dir=str(ckpt_dir), restart=True)
        # A twist-averaged restart takes an omitted frozen key from the default, and says so.
        toml2.write_text(toml2.read_text().replace("mcmc_seed = 12345\n", ""))
        result2 = _run_cli(toml2, tmp_path, n_ranks=max(2, n_twist))
        out2 = result2.stdout + result2.stderr
        assert result2.returncode == 0, out2
        assert "The default value of mcmc_seed" in out2

        mtimes_after = {ik: os.path.getmtime(twist_checkpoint_path(str(ckpt_dir), ik)) for ik in range(n_twist)}
        assert mtimes_after == mtimes_before, (
            "resume re-wrote completed twists -- the checkpoint-mtime invariance failed.\n--- second run ---\n" + out2
        )
        outputs = [result1.stdout + result1.stderr, out2]
        mode = "mixed" if is_mixed_precision_enabled() else "full"
        for output in outputs:
            assert f"**section:precision**\n  mode={mode}\n" in output
        _assert_twist_summary_from_checkpoints(outputs, ckpt_dir, n_twist)

    @pytest.mark.heavy
    @pytest.mark.parametrize("kind", ["pgto", "pgto+pw", "pgto+bspline"])
    @pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
    def test_cli_twist_resume_saved_values_and_input_identity(self, tmp_path, kind):
        """The real MPI CLI preserves each input twist and both five-decimal energy summaries.

        Execute the original fresh/resume journey on the same three Li fixture
        kinds, seed 12345, two walkers and twenty measurements. Assert the
        saved up/down MOs and k match the selected input, and independently
        assemble energy/error/dispersion from raw checkpoint values for both
        displays. Explicit full/mixed TOML is required. The final saved
        sampler coordinates are observed without cond(G) screening; this is
        not an independent physical-energy or statistical-convergence oracle.

        This distinct evidence node retains the existing smoke registration
        unchanged. It adds no distinct physical fixture or coverage sample;
        its purpose is the stronger mandatory input/output verdict.
        """
        assert _MPIRUN is not None, "the required CLI integration must have an MPI launcher"
        assert _qe_available if kind != "pgto" else _li_available, "the required committed fixture is missing"
        self.test_cli_resume_skips_completed_twists_end_to_end(tmp_path, kind)

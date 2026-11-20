"""CLI twist-averaged auto-dispatch.

``jqmc_cli`` looks at the input ``hamiltonian_data.h5`` and decides
between the single-twist driver path
(``MCMC`` / ``GFMC_*`` on ``COMM_WORLD``) and the twist-averaged path
(``run_twist_average_*`` orchestrators). Fresh runs use the Hamiltonian layout. Restarts use
an existing checkpoint file or directory; a single-file checkpoint contains
its Hamiltonian and must not depend on the initialization input.

These tests pin the dispatch resolver (``_resolve_twist_dispatch``) at
the unit-test level -- the resolver returns the selected route name the
CLI's run path then consumes. Pinning at this layer is much cheaper
than spawning the full CLI for every dispatch combination; the
end-to-end "CLI on a combined file actually routes through the TA driver"
behaviour is covered by ``tests/test_jqmc_kernel/test_twist_resume.py`` (the driver
itself uses the same code paths the CLI delegates to).
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

import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jqmc.jqmc_cli as jqmc_cli
from jqmc.jqmc_cli import _resolve_twist_dispatch


@pytest.mark.light
@pytest.mark.parametrize("raises", [False, True])
def test_cli_process_entrypoint_finalizes_distributed_runtimes(monkeypatch, raises):
    """The process entry point owns JAX/MPI teardown on every exit path.

    ``_cli`` remains directly callable by in-process tests without destroying
    their MPI runtime; only the executable/module entry point performs this
    process-global cleanup.
    """
    calls = []

    def fake_cli():
        calls.append("cli")
        if raises:
            raise RuntimeError("sentinel")
        return "result"

    class FakeDistributed:
        @staticmethod
        def is_initialized():
            calls.append("jax.is_initialized")
            return True

        @staticmethod
        def shutdown():
            calls.append("jax.shutdown")

    class FakeMPI:
        @staticmethod
        def Is_initialized():
            calls.append("mpi.Is_initialized")
            return True

        @staticmethod
        def Is_finalized():
            calls.append("mpi.Is_finalized")
            return False

        @staticmethod
        def Finalize():
            calls.append("mpi.Finalize")

    monkeypatch.setattr(jqmc_cli, "_cli", fake_cli)
    monkeypatch.setattr(jqmc_cli.jax, "distributed", FakeDistributed)
    monkeypatch.setattr(jqmc_cli, "MPI", FakeMPI)

    if raises:
        with pytest.raises(RuntimeError, match="sentinel"):
            jqmc_cli.main()
    else:
        assert jqmc_cli.main() == "result"

    assert calls == [
        "cli",
        "jax.is_initialized",
        "jax.shutdown",
        "mpi.Is_initialized",
        "mpi.Is_finalized",
        "mpi.Finalize",
    ]


@pytest.mark.light
@pytest.mark.parametrize(
    "outcome,size,initialized,finalized,expected_abort",
    [
        pytest.param("return", 2, True, False, False, id="success"),
        pytest.param("runtime-error", 2, True, False, True, id="multirank-error"),
        pytest.param("interrupt", 2, True, False, True, id="multirank-interrupt"),
        pytest.param("exit-error", 2, True, False, True, id="multirank-nonzero-exit"),
        pytest.param("exit-zero", 2, True, False, False, id="clean-zero-exit"),
        pytest.param("exit-none", 2, True, False, False, id="clean-none-exit"),
        pytest.param("runtime-error", 1, True, False, False, id="single-rank-error"),
        pytest.param("runtime-error", 2, False, False, False, id="uninitialized-mpi"),
        pytest.param("runtime-error", 2, True, True, False, id="finalized-mpi"),
    ],
)
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Console failure/cleanup control flow uses no Hamiltonian or electronic coordinates."
)
def test_console_entrypoint_handles_mpi_failures_and_cleanup(
    monkeypatch, tmp_path, capsys, outcome, size, initialized, finalized, expected_abort
):
    """The actual console wrapper aborts failed multi-rank jobs before cleanup.

    The typed communicator supplies independent rank/state facts and records
    Abort's argument without terminating pytest. Its return permits observing
    the finally path too; actual MPI abort normally never returns. The payload
    represents a job's return or failure, with no scientific calculation mocked.
    """
    from jqmc import _cli_entry

    events = []
    errors = {
        "runtime-error": RuntimeError("rank-local sentinel"),
        "interrupt": KeyboardInterrupt("rank-local sentinel"),
        "exit-error": SystemExit(1),
        "exit-zero": SystemExit(0),
        "exit-none": SystemExit(None),
    }
    error = errors.get(outcome)

    def payload():
        events.append("payload")
        if error is not None:
            raise error
        return "result"

    class World:
        @staticmethod
        def Get_size():
            return size

        @staticmethod
        def Get_rank():
            return size - 1

        @staticmethod
        def Abort(code):
            events.append(("abort", code))

    class MPI:
        COMM_WORLD = World

        @staticmethod
        def Is_initialized():
            return initialized

        @staticmethod
        def Is_finalized():
            return finalized

        @staticmethod
        def Finalize():
            events.append("mpi-finalize")

    class Distributed:
        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def shutdown():
            events.append("jax-shutdown")

    toml_file = tmp_path / "input.toml"
    toml_file.write_text("[control]\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["jqmc", str(toml_file)])
    monkeypatch.setattr(jqmc_cli, "_cli", payload)
    monkeypatch.setattr(jqmc_cli, "MPI", MPI)
    monkeypatch.setattr(jqmc_cli.jax, "distributed", Distributed)

    if error is None:
        assert _cli_entry.main() == "result"
    else:
        with pytest.raises(type(error)) as caught:
            _cli_entry.main()
        assert caught.value is error

    expected = ["payload"]
    if expected_abort:
        expected.append(("abort", 1))
    expected.append("jax-shutdown")
    if initialized and not finalized:
        expected.append("mpi-finalize")
    assert events == expected
    stderr = capsys.readouterr().err
    if expected_abort:
        assert f"[jqmc rank 1] FATAL: unhandled {type(error).__name__}" in stderr
        assert "Traceback (most recent call last)" in stderr
    else:
        assert stderr == ""


class TestCLITwistDispatch:
    # ---------- single-twist (non-TA) files always route to single-twist ----------

    @pytest.mark.light
    def test_dispatch_unreadable_file_raises(self, tmp_path):
        """An unreadable input is an error, never a single-twist fallback."""
        with pytest.raises(OSError):
            _resolve_twist_dispatch(str(tmp_path / "missing.h5"))

    @pytest.mark.light
    def test_control_schema_omits_twist_dispatch_overrides(self):
        """Generated core inputs cannot override schema-owned dispatch."""
        from jqmc.jqmc_miscs import cli_parameters

        assert "twist_average" not in cli_parameters["control"]
        assert "twist_index" not in cli_parameters["control"]

    @pytest.mark.light
    @pytest.mark.parametrize(
        ("key", "value"),
        [("twist_average", '"auto"'), ("twist_index", "0")],
    )
    @pytest.mark.gdd_electron_configuration(
        mode="not-required", reason="TOML rejected before MPI/JAX or Hamiltonian initialization."
    )
    def test_removed_control_key_is_rejected_before_runtime_initialization(self, tmp_path, monkeypatch, key, value):
        """Legacy dispatch controls fail before MPI or JAX initialization."""
        toml_path = tmp_path / "removed-control.toml"
        toml_path.write_text(f"[control]\n{key} = {value}\n", encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["jqmc", str(toml_path)])

        with pytest.raises(ValueError, match=r"Removed \[control\] key"):
            jqmc_cli._cli()

    # ---------- every combined file routes to twist averaging ----------

    @pytest.fixture(params=[1, 2], ids=["one-twist", "two-twist"])
    def combined_twist_file(self, request, tmp_path):
        """Build a minimal combined input via the production writer
        with one or two fake twists. The one-twist case remains a combined
        twist-averaged Hamiltonian and must not change execution paths."""
        from jqmc._twist_average import save_twist_average_input
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jastrow_factor import Jastrow_data
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data
        import numpy as np

        fixture = Path(__file__).parent.parent / "trexio_example_files" / "H_ae_ccpvdz_cart.h5"
        if not fixture.is_file():
            pytest.skip("H_ae_ccpvdz_cart.h5 fixture not present")
        sd, _, _, _, gem, cpd = read_trexio_file(str(fixture), store_tuple=True)
        ham = Hamiltonian_data(
            structure_data=sd,
            coulomb_potential_data=cpd,
            wavefunction_data=Wavefunction_data(jastrow_data=Jastrow_data(), geminal_data=gem),
        )
        mo_up = np.asarray(gem.orb_data_up_spin.mo_coefficients, dtype=np.complex128)
        mo_dn = np.asarray(gem.orb_data_dn_spin.mo_coefficients, dtype=np.complex128)
        n_twist = int(request.param)
        out = tmp_path / f"ha_{n_twist}_twists.h5"
        save_twist_average_input(
            filepath=str(out),
            template_hamiltonian=ham,
            k_points=[(0.0, 0.0, 0.0), (0.25, 0.0, 0.0)][:n_twist],
            weights=[1.0] if n_twist == 1 else [0.5, 0.5],
            mo_coefficients_up_per_twist=[mo_up] * n_twist,
            mo_coefficients_dn_per_twist=[mo_dn] * n_twist,
        )
        return str(out)

    @pytest.mark.light
    def test_dispatch_combined_routes_to_twist_average(self, combined_twist_file):
        """The combined schema selects twist averaging for any twist count."""
        assert _resolve_twist_dispatch(combined_twist_file) == "twist_average"


def _write_combined_twist_input(directory, n_twist):
    """Write a loadable combined input with the first ``n_twist`` twists of the lithium TREXIO fixture.

    The committed lithium fixture is required (no skip): a missing file is a failure.
    """
    import numpy as np

    from jqmc._twist_average import save_twist_average_input
    from jqmc.hamiltonians import Hamiltonian_data
    from jqmc.jastrow_factor import Jastrow_data
    from jqmc.trexio_wrapper import read_trexio_file, read_trexio_kpoint_list
    from jqmc.wavefunction import Wavefunction_data

    fixture = Path(__file__).parent.parent / "trexio_example_files" / "lithium_twist_222.h5"
    kpts = read_trexio_kpoint_list(str(fixture))[:n_twist]
    mo_up, mo_dn, template = [], [], None
    for ik in range(n_twist):
        sd, _, _, _, gem, cpd = read_trexio_file(trexio_file=str(fixture), store_tuple=True, twist_ik=ik)
        if template is None:
            template = Hamiltonian_data(
                structure_data=sd,
                coulomb_potential_data=cpd,
                wavefunction_data=Wavefunction_data(jastrow_data=Jastrow_data(), geminal_data=gem),
            )
        mo_up.append(np.asarray(gem.orb_data_up_spin.mo_coefficients, dtype=np.complex128))
        mo_dn.append(np.asarray(gem.orb_data_dn_spin.mo_coefficients, dtype=np.complex128))
    out = Path(directory) / f"li_{n_twist}_twists.h5"
    save_twist_average_input(
        filepath=str(out),
        template_hamiltonian=template,
        k_points=[k for k, _ in kpts],
        weights=[1.0 / n_twist] * n_twist,
        mo_coefficients_up_per_twist=mo_up,
        mo_coefficients_dn_per_twist=mo_dn,
    )
    return str(out)


def _dispatch_mcmc(hamiltonian_h5, restart_chk, restart, monkeypatch, reads):
    """Run the real twist dispatch with the MCMC twist driver replaced by a recorder.

    The recorder stops the run at the driver boundary, so no sampling is
    performed. Every combined input read is appended to ``reads``. Returns the
    recorded driver keyword arguments.
    """
    import copy

    import jqmc._twist_average as twist_average
    from jqmc.jqmc_miscs import cli_parameters

    class _ReachedDriver(Exception):
        pass

    reached = {}
    load_input = twist_average.load_twist_average_input

    def recording_load(path):
        reads.append(path)
        return load_input(path)

    def recording_driver(**kwargs):
        reached.update(kwargs)
        raise _ReachedDriver

    monkeypatch.setattr(twist_average, "load_twist_average_input", recording_load)
    monkeypatch.setattr(twist_average, "run_twist_average_from_hamiltonians", recording_driver)
    try:
        jqmc_cli._run_twist_average_dispatch(
            job_type="mcmc",
            hamiltonian_h5=hamiltonian_h5,
            parameters=copy.deepcopy(cli_parameters),
            dict_toml={"mcmc": {"num_mcmc_steps": 2}},
            mcmc_seed=1,
            number_of_walkers=1,
            mpc=False,
            mpc_cutoff=0.0,
            max_time=60,
            restart=restart,
            restart_chk=restart_chk,
            mpi_comm=jqmc_cli.MPI.COMM_WORLD,
            mpi_rank=0,
        )
    except _ReachedDriver:
        return reached
    raise AssertionError("twist dispatch returned without reaching the recorded driver")


class TestTwistCheckpointLayout:
    """Twist-averaged runs use restart_chk as a checkpoint directory."""

    @pytest.fixture(params=[1, 2], ids=["one-twist", "two-twist"])
    def twist_input(self, request, tmp_path):
        """A combined input; the one-twist case also runs the twist-averaged path."""
        return _write_combined_twist_input(tmp_path, int(request.param))

    @pytest.mark.light
    @pytest.mark.parametrize("restart", [False, True], ids=["fresh", "restart"])
    @pytest.mark.parametrize("name", ["restart.h5", "single_twist_checkpoint.h5"], ids=["default-name", "custom-name"])
    def test_existing_regular_restart_file_is_rejected_before_any_twist_runs(
        self, twist_input, tmp_path, monkeypatch, restart, name
    ):
        """A regular file at restart_chk fails at startup with the directory-layout explanation.

        Covers the default ``restart.h5`` left by a single-k run and a custom
        single-file checkpoint, with ``restart`` false and true. Neither the combined
        input nor any twist driver is touched, so no computation happens first,
        and the file itself is left unchanged.
        """
        monkeypatch.chdir(tmp_path)
        (tmp_path / name).write_bytes(b"single-file checkpoint")
        reads = []
        with pytest.raises(ValueError) as raised:
            _dispatch_mcmc(twist_input, name, restart, monkeypatch, reads)
        message = str(raised.value)
        for fragment in (
            f"restart_chk = '{name}' is an existing regular file",
            "checkpoint DIRECTORY",
            "twist_NNN.h5",
            "shared_state.h5",
            "cannot be resumed by, or converted to, this layout",
            "set [control] restart_chk to a directory name",
        ):
            assert fragment in message, (fragment, message)
        assert reads == []
        assert (tmp_path / name).read_bytes() == b"single-file checkpoint"

    @pytest.mark.light
    @pytest.mark.parametrize("restart", [False, True], ids=["fresh", "restart"])
    @pytest.mark.parametrize("state", ["existing-directory", "absent"])
    def test_directory_or_absent_restart_path_reaches_twist_driver(self, twist_input, tmp_path, monkeypatch, restart, state):
        """An existing directory or an absent path is forwarded unchanged to the twist driver."""
        monkeypatch.chdir(tmp_path)
        if state == "existing-directory":
            (tmp_path / "restart.h5").mkdir()
        reads = []
        reached = _dispatch_mcmc(twist_input, "restart.h5", restart, monkeypatch, reads)
        assert reads == [twist_input]
        assert reached["checkpoint_dir"] == "restart.h5"
        assert reached["resume"] is restart
        assert (tmp_path / "restart.h5").is_dir() is (state == "existing-directory")


@pytest.mark.light
@pytest.mark.parametrize("damage", ["meta", "shared", "twists", "count", "count-mismatch", "twist-name"])
def test_combined_loaders_require_current_container(tmp_path, damage):
    """Missing current payloads are errors, never a legacy-format fallback."""
    import h5py

    from jqmc._twist_average import is_twist_average_input, load_twist, load_twist_average_input
    from jqmc.jqmc_cli import _resolve_twist_dispatch

    path = tmp_path / "incomplete_combined.h5"
    with h5py.File(path, "w") as h5:
        meta = h5.create_group("_meta")
        meta.attrs["n_twist"] = 1
        h5.create_group("shared")
        twists = h5.create_group("twists")
        twists.create_group("twist_000")
        if damage in ("meta", "shared", "twists"):
            del h5["_meta" if damage == "meta" else damage]
        elif damage == "count":
            del meta.attrs["n_twist"]
        elif damage == "count-mismatch":
            meta.attrs["n_twist"] = 2
        else:
            twists.move("twist_000", "twist_001")
    for reader in (
        lambda: load_twist_average_input(path),
        lambda: load_twist(path, 0),
        lambda: is_twist_average_input(path),
        lambda: _resolve_twist_dispatch(path),
    ):
        with pytest.raises(ValueError, match="twist-averaged input requires"):
            reader()


@pytest.mark.light
@pytest.mark.parametrize("initial_input", ["omitted", "unreadable", "combined"])
def test_single_file_restart_does_not_read_initial_hamiltonian(tmp_path, monkeypatch, pytestconfig, initial_input):
    """The actual CLI restores a current checkpoint before entering the sampler.

    The initialization input is frozen/unused for this route. No sampling is
    performed; a sentinel observes the restored state and stops at run().
    """
    import hashlib
    import logging

    import toml

    from jqmc.jqmc_mcmc import MCMC
    from tests.test_jqmc_kernel.test_integration_checkpoint_mcmc import _build_hamiltonian, _make_mcmc, _save_merge

    ham = _build_hamiltonian("H2_ae_ccpvdz_cart.h5", "none")
    checkpoint = Path(_save_merge(_make_mcmc(ham, num_walkers=2), ham, tmp_path))
    restored = MCMC.load_from_hdf5(str(checkpoint), rank=0)
    expected_token = restored.wavefunction_state_token
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    control = {"job_type": "mcmc", "restart": True, "restart_chk": str(checkpoint), "verbosity": "low"}
    if initial_input == "unreadable":
        unused = tmp_path / "unreadable_initial.h5"
        unused.write_bytes(b"not an HDF5 file")
        control["hamiltonian_h5"] = str(unused)
    elif initial_input == "combined":
        control["hamiltonian_h5"] = _write_combined_twist_input(tmp_path, 1)
    assert not (tmp_path / "hamiltonian_data.h5").exists()
    payload = {
        "control": control,
        "precision": {"mode": pytestconfig.getoption("--precision-mode")},
        "mcmc": {"num_mcmc_steps": 1, "num_mcmc_warmup_steps": 0, "num_mcmc_bin_blocks": 2},
    }
    toml_path = tmp_path / "resume.toml"
    toml_path.write_text(toml.dumps(payload))

    class SamplerReached(Exception):
        pass

    seen = []

    def stop_before_sampling(driver, *args, **kwargs):
        assert driver.num_walkers == restored.num_walkers == 2
        assert driver.mcmc_counter == restored.mcmc_counter
        assert driver.wavefunction_state_token == expected_token
        seen.append(driver)
        raise SamplerReached

    monkeypatch.setattr(MCMC, "run", stop_before_sampling)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["jqmc", str(toml_path)])
    logger = logging.getLogger("jqmc")
    handlers, level = list(logger.handlers), logger.level
    try:
        with pytest.raises(SamplerReached):
            jqmc_cli._cli()
    finally:
        logger.handlers, logger.level = handlers, level
    assert len(seen) == 1
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash


@pytest.mark.light
def test_bad_restart_checkpoint_is_rejected_without_fresh_initialization(tmp_path, monkeypatch, pytestconfig):
    """A readable initial Hamiltonian cannot rescue a corrupt restart payload."""
    import logging

    import toml

    from jqmc.jqmc_mcmc import MCMC
    from tests.test_jqmc_kernel.test_integration_checkpoint_mcmc import _build_hamiltonian

    initial = tmp_path / "initial.h5"
    _build_hamiltonian("H2_ae_ccpvdz_cart.h5", "none").save_to_hdf5(initial)
    checkpoint = tmp_path / "bad_restart.h5"
    invalid_bytes = b"not an HDF5 checkpoint"
    checkpoint.write_bytes(invalid_bytes)
    toml_path = tmp_path / "resume.toml"
    toml_path.write_text(
        toml.dumps(
            {
                "control": {
                    "job_type": "mcmc",
                    "restart": True,
                    "restart_chk": str(checkpoint),
                    "hamiltonian_h5": str(initial),
                },
                "precision": {"mode": pytestconfig.getoption("--precision-mode")},
                "mcmc": {"num_mcmc_steps": 1, "num_mcmc_warmup_steps": 0, "num_mcmc_bin_blocks": 2},
            }
        )
    )

    def forbidden_sampler(*args, **kwargs):
        pytest.fail("a corrupt checkpoint entered the sampler")

    monkeypatch.setattr(MCMC, "run", forbidden_sampler)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["jqmc", str(toml_path)])
    logger = logging.getLogger("jqmc")
    handlers, level = list(logger.handlers), logger.level
    try:
        with pytest.raises(OSError):
            jqmc_cli._cli()
    finally:
        logger.handlers, logger.level = handlers, level
    assert checkpoint.read_bytes() == invalid_bytes


@pytest.mark.light
def test_restart_directory_selects_twist_dispatch(tmp_path):
    """The existing directory layout selects the driver that validates twist inputs."""
    directory = tmp_path / "restart_twists"
    directory.mkdir()
    assert _resolve_twist_dispatch(str(tmp_path / "initial.h5"), restart=True, restart_chk=str(directory)) == "twist_average"


_TWIST_DRIVERS = {
    "mcmc": ("run_twist_average_from_hamiltonians", {"mcmc": {"num_mcmc_steps": 2}}),
    "vmc": ("run_twist_average_optimize", {"vmc": {"num_mcmc_steps": 2, "num_opt_steps": 1}}),
    "lrdmc-bra": ("run_twist_average_gfmc", {"lrdmc-bra": {"num_mcmc_steps": 50, "num_mcmc_per_measurement": 1}}),
    "lrdmc-tau": ("run_twist_average_gfmc", {"lrdmc-tau": {"num_mcmc_steps": 50}}),
}


@pytest.mark.light
@pytest.mark.parametrize("job_type", list(_TWIST_DRIVERS))
def test_twist_dispatch_prints_the_hamiltonian_summary_once(tmp_path, monkeypatch, caplog, job_type):
    """Every twist-averaged job type prints one Hamiltonian summary (twist 0) before its driver runs."""
    import copy
    import logging

    import jqmc._twist_average as twist_average
    from jqmc.jqmc_miscs import cli_parameters
    from jqmc.output_parser import parse_system_info

    class _ReachedDriver(Exception):
        pass

    def stop_at_driver(**kwargs):
        raise _ReachedDriver

    driver_name, dict_toml = _TWIST_DRIVERS[job_type]
    monkeypatch.setattr(twist_average, driver_name, stop_at_driver)
    hamiltonian_h5 = _write_combined_twist_input(tmp_path, 2)
    caplog.set_level(logging.INFO, logger="jqmc")
    parameters = copy.deepcopy(cli_parameters)
    for section, values in dict_toml.items():
        parameters[section].update(values)
    with pytest.raises(_ReachedDriver):
        jqmc_cli._run_twist_average_dispatch(
            job_type=job_type,
            hamiltonian_h5=hamiltonian_h5,
            parameters=parameters,
            dict_toml=dict_toml,
            mcmc_seed=1,
            number_of_walkers=1,
            mpc=False,
            mpc_cutoff=0.0,
            max_time=60,
            restart=False,
            restart_chk=str(tmp_path / "restart.h5"),
            mpi_comm=jqmc_cli.MPI.COMM_WORLD,
            mpi_rank=0,
        )
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert text.count("Printing out information in hamiltonian_data instance.") == 1
    system = parse_system_info(text)
    assert system["pbc"] is True and system["lattice_vectors"] is not None

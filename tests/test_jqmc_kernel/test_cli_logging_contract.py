"""Exact CLI/TOML logging contracts, without launching a numerical calculation."""

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

import io
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import toml

from jqmc import jqmc_cli

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


class _LoggingReady(Exception):
    """Stop after real handler setup and before distributed initialization."""


def _stop_before_header():
    raise _LoggingReady


def _capture_logging(tmp_path, monkeypatch, verbosity, rank):
    """Use real TOML parsing and real loggers with only the MPI rank supplied."""
    control = {} if verbosity == "default" else {"verbosity": verbosity}
    path = tmp_path / "logging.toml"
    path.write_text(toml.dumps({"control": control}), encoding="utf-8")
    stream = io.StringIO()
    log = logging.Logger("gdd-cli-logging")
    try:
        with monkeypatch.context() as patch:
            patch.chdir(tmp_path)
            patch.setattr(sys, "argv", ["jqmc", str(path)])
            patch.setattr(sys, "stdout", stream)
            patch.setattr(
                jqmc_cli, "MPI", SimpleNamespace(COMM_WORLD=SimpleNamespace(Get_rank=lambda: rank, Get_size=lambda: 2))
            )
            patch.setattr(jqmc_cli, "getLogger", lambda name: log)
            patch.setattr(jqmc_cli, "_print_header", _stop_before_header)
            with pytest.raises(_LoggingReady):
                jqmc_cli._cli()
            assert len(log.handlers) == 1
            handler = log.handlers[0]
            for level, message in [(5, "detail"), (10, "debug"), (20, "info"), (40, "error")]:
                log.log(level, message)
            handler.flush()
            filename = Path(handler.baseFilename).name if isinstance(handler, logging.FileHandler) else "stdout"
            messages = (
                Path(handler.baseFilename).read_text().splitlines() if filename != "stdout" else stream.getvalue().splitlines()
            )
            return [log.level, handler.level], filename, messages, stream.getvalue()
    finally:
        for handler in log.handlers:
            handler.close()


@pytest.mark.gdd_electron_configuration(mode="not-required", reason="TOML and logging setup only; no Hamiltonian or electrons")
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize(
    "verbosity, base_level, allowed_messages",
    [
        ("default", 20, ["info", "error"]),
        ("low", 20, ["info", "error"]),
        ("high", 10, ["debug", "info", "error"]),
        ("devel", 5, ["detail", "debug", "info", "error"]),
        ("mpi-low", 20, ["info", "error"]),
        ("mpi-high", 10, ["debug", "info", "error"]),
        ("mpi-devel", 5, ["detail", "debug", "info", "error"]),
    ],
)
def test_toml_verbosity_controls_levels_and_output(tmp_path, monkeypatch, verbosity, base_level, allowed_messages, rank):
    """Six verbosity values and the omitted default select exact levels, filtering and rank-specific destinations."""
    levels, destination, messages, stdout = _capture_logging(tmp_path, monkeypatch, verbosity, rank)
    mpi_logging = verbosity.startswith("mpi-")
    expected_level = base_level if rank == 0 or mpi_logging else 40
    expected_messages = allowed_messages if rank == 0 or mpi_logging else ["error"]
    expected_destination = f"log_jqmc_mpirank_{rank}.log" if mpi_logging else "stdout"
    message_bytes = np.frombuffer("\n".join(messages).encode("utf-8"), dtype=np.uint8)
    expected_message_bytes = np.frombuffer("\n".join(expected_messages).encode("utf-8"), dtype=np.uint8)
    destination_bytes = np.frombuffer(destination.encode("utf-8"), dtype=np.uint8)
    expected_destination_bytes = np.frombuffer(expected_destination.encode("utf-8"), dtype=np.uint8)
    np.testing.assert_array_equal(levels, [expected_level, expected_level])
    np.testing.assert_array_equal(message_bytes, expected_message_bytes)
    np.testing.assert_array_equal(destination_bytes, expected_destination_bytes)
    if mpi_logging:
        assert stdout == ""
        assert sorted(path.name for path in tmp_path.glob("log_jqmc_mpirank_*.log")) == [expected_destination]


@pytest.mark.gdd_electron_configuration(mode="not-required", reason="invalid TOML rejected before numerical initialization")
@pytest.mark.parametrize("verbosity", ["hgih", "", "INFO", "MPI-low", 3, True])
def test_unknown_toml_verbosity_is_rejected(tmp_path, monkeypatch, verbosity):
    """Unknown or non-string verbosity raises an explicit error instead of silently selecting INFO."""
    with pytest.raises(ValueError, match=r"Invalid control\.verbosity"):
        _capture_logging(tmp_path, monkeypatch, verbosity, 0)


@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="logging corruption controls use no electronic configuration"
)
@pytest.mark.parametrize("fault", ["logger-level", "handler-level", "wrong-rank-file"])
def test_logging_verdict_rejects_wrong_configuration(tmp_path, monkeypatch, fault):
    """The normal exact verdict rejects a wrong logger level, handler level or rank-file routing after a passing baseline."""
    test_toml_verbosity_controls_levels_and_output(
        tmp_path, monkeypatch, "mpi-devel", 5, ["detail", "debug", "info", "error"], 1
    )
    if fault == "logger-level":
        original = logging.Logger.setLevel

        def broken_level(self, level):
            return original(self, "INFO")

        monkeypatch.setattr(logging.Logger, "setLevel", broken_level)
    elif fault == "handler-level":
        original = logging.Handler.setLevel

        def broken_level(self, level):
            return original(self, "INFO")

        monkeypatch.setattr(logging.Handler, "setLevel", broken_level)
    else:
        original = jqmc_cli.FileHandler

        def wrong_file(filename, mode):
            return original("log_jqmc_mpirank_0.log", mode=mode)

        monkeypatch.setattr(jqmc_cli, "FileHandler", wrong_file)
    with pytest.raises(AssertionError):
        test_toml_verbosity_controls_levels_and_output(
            tmp_path, monkeypatch, "mpi-devel", 5, ["detail", "debug", "info", "error"], 1
        )

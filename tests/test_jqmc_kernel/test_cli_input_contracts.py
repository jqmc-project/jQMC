"""CLI input contracts: TOML defaults, merges, minimums and twist forwarding.

Unit-level pins for how ``jqmc_cli`` turns a user TOML into driver
arguments. Each case is a defect that once let a run silently use a value
the user did not ask for:

* a partial ``[vmc].optimizer_kwargs`` table replaced the documented
  defaults instead of merging into them;
* unknown TOML keys and sections were dropped without a word;
* the single-twist ``lrdmc-tau`` route skipped the GFMC minimum checks;
* an unsupported ``[control] job_type`` on an ordinary Hamiltonian ran no job
  and still exited normally;
* the twist-averaged ``mcmc`` / ``vmc`` routes did not forward
  ``num_mcmc_per_measurement`` (the driver default 16 was used instead).
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

import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

import jqmc.jqmc_cli as jqmc_cli
from jqmc.jqmc_miscs import cli_parameters

pytestmark = [pytest.mark.with_chdir]


class _StopDispatch(Exception):
    """Raised by the fake twist drivers once the forwarded kwargs are captured."""


@pytest.fixture
def single_input_header(tmp_path):
    """Readable single-input header; parameter errors must precede body loading.

    Unreadable files now raise during dispatch. Leave the Hamiltonian body
    absent so these parameter tests still detect premature deserialization.
    """
    import h5py

    path = tmp_path / "single_input.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["_class_name"] = "Hamiltonian_data"
    return path


@pytest.fixture
def combined_twist_file(tmp_path):
    """Two-twist combined input of the lithium fixture, loadable by the twist dispatch."""
    from tests.test_jqmc_kernel.test_cli_twist_dispatch import _write_combined_twist_input

    return _write_combined_twist_input(tmp_path, 2)


@pytest.mark.light
def test_partial_optimizer_kwargs_merge_onto_documented_defaults():
    """A partial table overrides only the keys it names; the rest keep the documented defaults."""
    documented = copy.deepcopy(cli_parameters["vmc"]["optimizer_kwargs"])

    merged = jqmc_cli._merged_optimizer_kwargs({"delta": 0.05})

    assert merged["delta"] == 0.05
    for key, value in documented.items():
        if key != "delta":
            assert merged[key] == value, key
    # The merge must not leak into the module-level defaults.
    assert cli_parameters["vmc"]["optimizer_kwargs"] == documented
    assert jqmc_cli._merged_optimizer_kwargs(None) == documented
    with pytest.raises(TypeError, match="optimizer_kwargs"):
        jqmc_cli._merged_optimizer_kwargs(0.1)


@pytest.mark.light
def test_unknown_toml_keys_and_sections_warn(caplog):
    """Typos in keys or section names are reported instead of silently ignored."""
    dict_toml = {
        "control": {"job_type": "mcmc", "hamiltonian_h5": "h.h5", "mcmc_sed": 1},
        "mcmc": {"num_mcmc_steps": 10, "num_mcmc_per_measuremnt": 4},
        "lrdmc": {"alat": 0.3},
        "vmc": {"num_opt_steps": 1, "optimizer_kwargs": {"num_opt_param": -1, "epsilo": 10.0, "epsilon": 0.5}},
    }
    with caplog.at_level(logging.WARNING, logger="jqmc"):
        jqmc_cli._warn_unknown_toml_keys(dict_toml, "mcmc")
    text = caplog.text
    assert "Unknown TOML sections ignored: ['lrdmc']" in text
    assert "Unknown [control] keys ignored: ['mcmc_sed']" in text
    assert "Unknown [mcmc] keys ignored: ['num_mcmc_per_measuremnt']" in text
    assert "[vmc] is ignored because job_type = 'mcmc'" in text
    assert "optimizer_kwargs" not in text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="jqmc"):
        jqmc_cli._warn_unknown_toml_keys(dict_toml, "vmc")
    nested_warnings = [record.message for record in caplog.records if "[vmc].optimizer_kwargs" in record.message]
    assert len(nested_warnings) == 1
    assert "epsilo" in nested_warnings[0] and "num_opt_param" in nested_warnings[0]
    assert "epsilon" not in nested_warnings[0]


@pytest.mark.light
def test_unknown_toml_keys_silent_for_valid_input(caplog):
    """A fully valid input produces no unknown-key warnings."""
    dict_toml = {
        "control": {"job_type": "vmc", "hamiltonian_h5": "h.h5"},
        "vmc": {
            "num_mcmc_steps": 10,
            "num_opt_steps": 2,
            "optimizer_kwargs": {
                "method": "sr",
                "delta": 0.1,
                "epsilon": 0.5,
                "cg_flag": False,
                "cg_max_iter": 100,
                "cg_tol": 0.0001,
                "use_lm": True,
                "lm_subspace_dim": -1,
                "lm_cond": 0.01,
                "asr_gamma_max": 0.75,
                "asr_fallback_gamma": 0.25,
                "lm_reference_weight_min": 0.8,
            },
        },
        "precision": {"mode": "full"},
    }
    with caplog.at_level(logging.WARNING, logger="jqmc"):
        jqmc_cli._warn_unknown_toml_keys(dict_toml, "vmc")
    assert "ignored" not in caplog.text


@pytest.mark.light
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="TOML input validation rejects the job before any Hamiltonian or electron is loaded"
)
@pytest.mark.parametrize("lm_cond", ["0.0", "-0.001", "1.0", "2.5"])
def test_vmc_rejects_lm_condition_cutoff_outside_open_unit_interval(tmp_path, monkeypatch, single_input_header, lm_cond):
    """The VMC route rejects ``optimizer_kwargs.lm_cond`` outside (0, 1) before loading the Hamiltonian."""
    restart = tmp_path / "restart.h5"
    lines = [
        "[control]",
        'job_type = "vmc"',
        f'hamiltonian_h5 = "{single_input_header}"',
        f'restart_chk = "{restart}"',
        "",
        "[vmc]",
        "num_mcmc_steps = 100",
        "num_opt_steps = 1",
        f'optimizer_kwargs = {{ method = "sr", use_lm = true, lm_subspace_dim = -1, lm_cond = {lm_cond} }}',
    ]
    toml_path = tmp_path / "vmc.toml"
    toml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["jqmc", str(toml_path)])

    with pytest.raises(ValueError, match=r"lm_cond must be .*open interval \(0, 1\)"):
        jqmc_cli._cli()
    assert not restart.exists()


@pytest.mark.light
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="TOML input validation rejects the job before any Hamiltonian or electron is loaded"
)
@pytest.mark.parametrize(
    ("key", "value"),
    [("num_gfmc_warmup_steps", 0), ("num_gfmc_bin_blocks", 5), ("num_gfmc_collect_steps", 0)],
)
def test_single_twist_lrdmc_tau_enforces_gfmc_minima(tmp_path, monkeypatch, single_input_header, key, value):
    """The single-twist lrdmc-tau route rejects sub-minimum GFMC settings like lrdmc-bra does."""
    section = {"num_mcmc_steps": 100, "num_gfmc_warmup_steps": 25, "num_gfmc_bin_blocks": 10, "num_gfmc_collect_steps": 5}
    section[key] = value
    lines = ["[control]", 'job_type = "lrdmc-tau"', f'hamiltonian_h5 = "{single_input_header}"', "", "[lrdmc-tau]"]
    lines += [f"{k} = {v}" for k, v in section.items()]
    toml_path = tmp_path / "lrdmc_tau.toml"
    toml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["jqmc", str(toml_path)])

    with pytest.raises(ValueError, match=key):
        jqmc_cli._cli()


@pytest.mark.light
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="TOML input validation rejects the job before any Hamiltonian or electron is loaded"
)
@pytest.mark.parametrize("job_type", ["mcmcc", "MCMC", "mcmc ", "lrdmc_bra", 1])
def test_single_twist_rejects_unsupported_job_type(tmp_path, monkeypatch, job_type):
    """An unsupported [control] job_type is rejected before any input file is read.

    The documented values are "mcmc", "vmc", "lrdmc-bra" and "lrdmc-tau"
    (``jqmc_miscs.control_comments["job_type"]``); for any other value no job can
    run, so the CLI must raise instead of printing ``Program ends`` with exit 0.
    The Hamiltonian and restart checkpoint do not exist, so the error has to come
    from the job_type check itself.
    """
    lines = [
        "[control]",
        f"job_type = {json.dumps(job_type)}",
        f'hamiltonian_h5 = "{tmp_path / "missing.h5"}"',
        "restart = true",
        f'restart_chk = "{tmp_path / "missing_restart.h5"}"',
        "",
        "[mcmc]",
        "num_mcmc_steps = 12",
    ]
    toml_path = tmp_path / "bad_job_type.toml"
    toml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["jqmc", str(toml_path)])

    with pytest.raises(ValueError, match=r"job_type") as excinfo:
        jqmc_cli._cli()
    message = str(excinfo.value)
    for allowed in ("mcmc", "vmc", "lrdmc-bra", "lrdmc-tau"):
        assert allowed in message


@pytest.mark.light
@pytest.mark.parametrize("job_type", ["mcmc", "vmc"])
def test_twist_dispatch_forwards_num_mcmc_per_measurement(combined_twist_file, monkeypatch, job_type):
    """The twist route passes [job].num_mcmc_per_measurement to MCMC (not the driver default 16)."""
    import jqmc._twist_average as twist_average

    captured = {}

    def fake_driver(**kwargs):
        captured.update(kwargs)
        raise _StopDispatch

    monkeypatch.setattr(twist_average, "run_twist_average_from_hamiltonians", fake_driver)
    monkeypatch.setattr(twist_average, "run_twist_average_optimize", fake_driver)

    parameters = copy.deepcopy(cli_parameters)
    section = {"num_mcmc_steps": 50, "num_mcmc_per_measurement": 7}
    if job_type == "vmc":
        section.update(num_opt_steps=1, optimizer_kwargs={"delta": 0.05})
    with pytest.raises(_StopDispatch):
        jqmc_cli._run_twist_average_dispatch(
            job_type=job_type,
            hamiltonian_h5=combined_twist_file,
            parameters=parameters,
            dict_toml={job_type: section},
            mcmc_seed=1,
            number_of_walkers=1,
            mpc=False,
            mpc_cutoff=0.0,
            max_time=60,
            restart=False,
            restart_chk=None,
            mpi_comm=None,
            mpi_rank=0,
        )
    assert captured["mcmc_kwargs"]["num_mcmc_per_measurement"] == 7
    if job_type == "vmc":
        opt = captured["optimize_kwargs"]["optimizer_kwargs"]
        assert opt["delta"] == 0.05
        assert opt["use_lm"] == cli_parameters["vmc"]["optimizer_kwargs"]["use_lm"]

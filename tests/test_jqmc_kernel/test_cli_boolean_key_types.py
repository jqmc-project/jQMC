"""CLI input contract: boolean-valued TOML keys must be TOML booleans (#599).

Read by Python truthiness, a quoted ``"false"`` or a number in a boolean key
would make ``[control] restart = "false"`` continue a stale checkpoint and
``[vmc] opt_J2_param = "false"`` optimize J2, both with exit 0. The CLI
rejects such a value, naming the section, key and value, before any
Hamiltonian or checkpoint file is read.
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
import re
import sys

import pytest
import toml

import jqmc.jqmc_cli as jqmc_cli
from jqmc.jqmc_miscs import cli_parameters

pytestmark = [pytest.mark.with_chdir]

# Minimal job sections; the Hamiltonian file named in [control] does not exist,
# so any route that gets past input validation fails on reading it.
_JOB_SECTIONS = {
    "mcmc": {"num_mcmc_steps": 10},
    "vmc": {"num_mcmc_steps": 10, "num_opt_steps": 1},
}
_STALE_CHECKPOINT_BYTES = b"stale checkpoint of an earlier run\n"


@pytest.mark.light
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="TOML input validation rejects the job before any Hamiltonian or electron is loaded"
)
@pytest.mark.parametrize("value", ["false", "true", 0, 1], ids=["str-false", "str-true", "int-0", "int-1"])
@pytest.mark.parametrize(
    ("job_type", "section", "key"),
    [
        ("mcmc", "control", "restart"),
        ("vmc", "vmc", "opt_J2_param"),
        ("mcmc", "mcmc", "atomic_force"),
        ("mcmc", "coulomb", "mpc"),
        ("vmc", "vmc", "optimizer_kwargs.use_lm"),
    ],
)
def test_non_boolean_value_for_boolean_key_is_rejected(tmp_path, monkeypatch, job_type, section, key, value):
    """A string or number in a boolean key raises ValueError naming it; no file is read or written."""
    tables = {
        "control": {"job_type": job_type, "hamiltonian_h5": str(tmp_path / "missing_hamiltonian.h5")},
        job_type: dict(_JOB_SECTIONS[job_type]),
    }
    target = tables.setdefault(section, {})
    *parents, leaf = key.split(".")
    for parent in parents:
        target = target.setdefault(parent, {})
    target[leaf] = value
    toml_path = tmp_path / "input.toml"
    toml_path.write_text(toml.dumps(tables), encoding="utf-8")
    # A leftover checkpoint of an earlier run under the default restart_chk name.
    checkpoint = tmp_path / "restart.h5"
    checkpoint.write_bytes(_STALE_CHECKPOINT_BYTES)
    before = sorted(p.name for p in tmp_path.iterdir())
    monkeypatch.setattr(sys, "argv", ["jqmc", str(toml_path)])

    expected = rf"\[{re.escape(section)}\] {re.escape(key)} must be a TOML boolean .*got {re.escape(repr(value))}"
    with pytest.raises(ValueError, match=expected):
        jqmc_cli._cli()

    assert checkpoint.read_bytes() == _STALE_CHECKPOINT_BYTES
    assert sorted(p.name for p in tmp_path.iterdir()) == before


@pytest.mark.light
@pytest.mark.parametrize("job_type", ["mcmc", "vmc", "lrdmc-bra", "lrdmc-tau"])
def test_toml_boolean_values_pass_the_type_check(job_type):
    """Every boolean key of the sections a job reads accepts both TOML booleans."""
    for flag in (True, False):
        dict_toml = {"control": {"job_type": job_type}, "coulomb": {}, job_type: {}}
        for section, table in dict_toml.items():
            for key, default in cli_parameters[section].items():
                if isinstance(default, bool):
                    table[key] = flag
                elif isinstance(default, dict):
                    table[key] = {sub: flag for sub, d in default.items() if isinstance(d, bool)}
        snapshot = copy.deepcopy(dict_toml)
        jqmc_cli._validate_toml_boolean_keys(dict_toml)
        assert dict_toml == snapshot

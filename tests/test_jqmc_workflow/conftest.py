"""Shared fixtures for jqmc_workflow tests.

Provides a tmp_path-backed ``~/.jqmc_setting/`` substitute and a Machine
instance configured for local-mode operation (no SSH).
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

import textwrap

import pytest


@pytest.fixture
def tmp_jqmc_setting_dir(tmp_path, monkeypatch):
    """Redirect get_config_dir() to a tmp_path-backed config directory.

    Resets the module-level _resolved_config_dir cache before and after
    each test so that no production ~/.jqmc_setting/ leakage occurs.
    """
    from jqmc_workflow import _config

    setting = tmp_path / "jqmc_setting"
    setting.mkdir()
    _config.reset_config_dir()
    monkeypatch.setattr(_config, "_resolved_config_dir", str(setting))
    yield setting
    _config.reset_config_dir()


@pytest.fixture
def local_machine_yaml(tmp_jqmc_setting_dir, tmp_path):
    """Write a minimal machine_data.yaml defining 'local-test' and 'localhost'.

    'local-test' uses subprocess on localhost (machine_type=local) with
    no-op scheduler commands so jobsubmit/jobcheck/jobdel succeed in
    tests without spinning up a real queue.

    'localhost' is required by Data_transfer for local-root path mapping.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    yaml_path = tmp_jqmc_setting_dir / "machine_data.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            f"""
            local-test:
              machine_type: local
              workspace_root: {workspace}
              queuing: false
              jobsubmit: /bin/echo
              jobcheck: /usr/bin/true
              jobdel: /usr/bin/true
              jobnum_index: 1
            localhost:
              machine_type: local
              workspace_root: {workspace}
              queuing: false
              jobsubmit: /bin/echo
              jobcheck: /usr/bin/true
              jobdel: /usr/bin/true
              jobnum_index: 1
            """
        ).strip()
        + "\n"
    )
    # Clear cached Machines from a prior test so this yaml is freshly loaded.
    from jqmc_workflow.machine import close_all_machines

    close_all_machines()
    yield yaml_path
    close_all_machines()


@pytest.fixture
def local_machine_with_queue(local_machine_yaml, tmp_jqmc_setting_dir):
    """Set up queue_data.toml + submit template under the local-test config dir.

    The setup mirrors what ``~/.jqmc_setting/<machine>/`` contains for a real
    machine: a queue_data.toml with at least one queue label, and a submit
    template file referenced by that label.
    """
    machine_dir = tmp_jqmc_setting_dir / "local-test"
    machine_dir.mkdir(exist_ok=True)

    (machine_dir / "queue_data.toml").write_text(
        textwrap.dedent(
            """
            [default]
            submit_template = "submit_template.sh"
            num_cores = 1
            max_job_submit = 100
            queue = "default"
            """
        ).strip()
        + "\n"
    )

    (machine_dir / "submit_template.sh").write_text(
        textwrap.dedent(
            """
            #!/bin/bash
            # _JOBNAME_ submit template
            _COMMAND_ > _OUTPUT_ 2>_JOB_STDERR_
            """
        ).strip()
        + "\n"
    )
    return local_machine_yaml


@pytest.fixture
def local_machine(local_machine_yaml):
    """A Machine instance configured as local, ready for subprocess-backed tests.

    ssh_retry_time is compressed to 0 so that failing-command tests do not
    sleep the default 120 s between retries (which would blow the
    light-tier 90 s budget on the first failure).
    """
    from jqmc_workflow.machine import Machine, close_all_machines

    machine = Machine("local-test")
    machine.ssh_retry_time = 0
    yield machine
    close_all_machines()  # ensure registry is clean for the next test


@pytest.fixture
def fresh_machine_registry():
    """Clear the process-wide _MACHINE_REGISTRY before and after the test."""
    from jqmc_workflow.machine import close_all_machines

    close_all_machines()
    yield
    close_all_machines()

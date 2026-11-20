"""Request an external stop the way a user or a workflow engine does.

Every production driver loop (``MCMC.run``, ``GFMC_t.run``, ``GFMC_n.run`` and
the optimizer ``run_optimize``) creates its ``external_control_*.toml`` file with
``stop = false`` through ``_init_external_control_file`` and reads it at every
step boundary through ``_read_external_stop_flag``. A user ends a running job
early by rewriting that file to ``[external_control] stop = true``.

:func:`request_external_stop` reproduces exactly that, deterministically and
without a helper thread: it wraps the driver module's
``_init_external_control_file`` so that, right after the real function has
created the file, the world root rewrites it with ``stop = true``. The real
reader then finds the request at the first step boundary. Nothing else is
replaced: the file, its reader and the stop decision are the production ones.
The helper is shared by the driver tests (issue #354 for the twist-averaged
optimizer; issue #371 for MCMC/GFMC).
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

from types import ModuleType

import pytest
import toml


def request_external_stop(
    monkeypatch: pytest.MonkeyPatch,
    driver_module: ModuleType,
    toml_filename: str,
    *,
    requests: int = 1,
) -> dict[str, int]:
    """Set ``stop = true`` in the first ``requests`` control files the driver creates.

    Args:
        monkeypatch: The test's monkeypatch; the wrapper is undone with it.
        driver_module: Module whose ``_init_external_control_file`` name the
            driver loop calls (``jqmc.jqmc_optimizer``, ``jqmc.jqmc_mcmc`` or
            ``jqmc.jqmc_gfmc``).
        toml_filename: Only creations of this control file are answered with a
            stop request (for example ``"external_control_opt.toml"``).
        requests: How many creations of that file get a stop request; later
            creations keep the driver's ``stop = false``.

    Returns:
        A live counter dict: ``created`` (creations of ``toml_filename`` seen by
        this process) and ``requested`` (stop requests written for them).
    """
    real_init = driver_module._init_external_control_file
    counter = {"created": 0, "requested": 0}

    def init_then_request_stop(filename, world_comm):
        real_init(filename, world_comm)
        if filename != toml_filename:
            return
        counter["created"] += 1
        if counter["requested"] >= requests:
            return
        counter["requested"] += 1
        # The real function writes the file on the world root only; the stop
        # request is written by the same process, after it.
        if world_comm.Get_rank() == 0:
            with open(filename, "w") as f:
                toml.dump({"external_control": {"stop": True}}, f)

    monkeypatch.setattr(driver_module, "_init_external_control_file", init_then_request_stop)
    return counter

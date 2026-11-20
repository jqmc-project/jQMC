"""End-of-run peak-memory lines reach the default driver output."""

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

import io
import logging
import re
import resource
import sys

import pytest

from jqmc import _driver_helpers as helpers

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

# Line format consumed by benchmark collectors; keep in sync with _log_memory_usage.
MEMORY_LINE = re.compile(
    r"^\[memory\] stage=(?P<stage>[A-Za-z0-9_]+), RSS_peak_max=(?P<rss>[0-9]+\.[0-9]{2}|nan) GB"
    r"(?:, JAX_peak_max=(?P<jax>[0-9]+\.[0-9]{2}) GB)?$"
)

# Rank-0 handler thresholds that jqmc_cli installs for each verbosity.
VERBOSITY_LEVEL = {"low": "INFO", "high": "DEBUG"}

END_OF_RUN_STAGES = ("mcmc_post_run", "gfmc_t_post_run", "gfmc_n_post_run", "optimize_post_loop")
INTERMEDIATE_STAGES = ("mcmc_post_warmup", "gfmc_t_post_warmup", "gfmc_n_post_warmup", "optimize_pre_loop")

DEVICE_PEAK_BYTES = 3 * 1024**3


class _FakeDevice:
    def memory_stats(self):
        return {"peak_bytes_in_use": DEVICE_PEAK_BYTES}


def _peak_rss_gb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024**3) if sys.platform == "darwin" else ru / (1024**2)


@pytest.mark.parametrize("backend", ["cpu", "device"])
@pytest.mark.parametrize("verbosity", ["low", "high"])
@pytest.mark.parametrize(
    "stage",
    [pytest.param(stage, id=stage) for stage in END_OF_RUN_STAGES + INTERMEDIATE_STAGES],
)
def test_end_of_run_memory_line_is_reported_at_default_verbosity(stage, verbosity, backend, monkeypatch):
    """The CLI's rank-0 logger set-up receives one complete line per end-of-run stage.

    End-of-run stages must appear under both ``low`` (INFO) and ``high`` (DEBUG);
    intermediate stages only under ``high``. The RSS value is checked against the
    process high-water mark read independently before and after the call.
    """
    if backend == "device":
        monkeypatch.setattr(helpers.jax, "default_backend", lambda: "gpu")
        monkeypatch.setattr(helpers.jax, "devices", lambda *args, **kwargs: [_FakeDevice()])
    else:
        monkeypatch.setattr(helpers.jax, "default_backend", lambda: "cpu")

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(VERBOSITY_LEVEL[verbosity])
    handler.setFormatter(logging.Formatter("%(message)s"))
    jqmc_logger = logging.getLogger("jqmc")
    saved = (jqmc_logger.level, list(jqmc_logger.handlers), jqmc_logger.propagate, helpers.logger.level)
    jqmc_logger.handlers.clear()
    jqmc_logger.addHandler(handler)
    jqmc_logger.setLevel(VERBOSITY_LEVEL[verbosity])
    jqmc_logger.propagate = False
    helpers.logger.setLevel(logging.NOTSET)
    try:
        rss_before = _peak_rss_gb()
        result = helpers._log_memory_usage(stage, comm=None)
        rss_after = _peak_rss_gb()
    finally:
        jqmc_logger.handlers.clear()
        for h in saved[1]:
            jqmc_logger.addHandler(h)
        jqmc_logger.setLevel(saved[0])
        jqmc_logger.propagate = saved[2]
        helpers.logger.setLevel(saved[3])

    lines = [line for line in stream.getvalue().splitlines() if line.startswith("[memory]")]
    expected_visible = stage in END_OF_RUN_STAGES or verbosity == "high"
    if not expected_visible:
        assert lines == []
        return

    assert len(lines) == 1, lines
    match = MEMORY_LINE.match(lines[0])
    assert match is not None, lines[0]
    assert match["stage"] == stage
    assert rss_before <= result["rss_peak_max_gb"] <= rss_after
    assert match["rss"] == f"{result['rss_peak_max_gb']:.2f}"
    if backend == "device":
        assert match["jax"] == f"{DEVICE_PEAK_BYTES / 1024**3:.2f}"
        assert result["jax_peak_max_gb"] == DEVICE_PEAK_BYTES / 1024**3
    else:
        assert match["jax"] is None

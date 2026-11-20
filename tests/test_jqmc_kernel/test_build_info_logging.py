"""The real stdlib logger decides whether the API banner needs stderr fallback."""

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

import pytest

from jqmc import _build_info

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.mark.parametrize(
    "route",
    [
        "disabled-info",
        "stopped-propagation",
        "parent-stops-propagation",
        "warning-handler",
        "enabled-parent-handler",
        "local-handler-stops-propagation",
        "local-warning-stops-propagation",
        "no-handlers",
    ],
)
def test_api_banner_uses_the_real_logging_route_or_stderr_once(route, monkeypatch, capsys):
    """Logger enablement, handler thresholds and propagation jointly govern delivery.

    Probe the actual stdlib logger before invoking the production banner helper;
    the reference never calls or reproduces _emits_info. Private loggers and
    streams isolate the cases from pytest's own logging handlers.
    """
    stream = io.StringIO()
    root = logging.RootLogger(logging.WARNING if route in ("disabled-info", "no-handlers") else logging.INFO)
    logger = logging.Logger("jqmc-private-banner-route", level=logging.NOTSET)
    logger.parent = root
    if route == "parent-stops-propagation":
        parent = logging.Logger("jqmc-private-banner-parent", level=logging.INFO)
        parent.parent = root
        parent.propagate = False
        logger.parent = parent
    if route != "no-handlers":
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(logging.WARNING if route == "warning-handler" else logging.NOTSET)
        root.addHandler(handler)
    if route in ("stopped-propagation", "local-handler-stops-propagation", "local-warning-stops-propagation"):
        logger.propagate = False
    if route.startswith("local-"):
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(logging.WARNING if route == "local-warning-stops-propagation" else logging.INFO)
        logger.addHandler(handler)

    logger.info("independent routing probe")
    emitted_by_logger = stream.getvalue() == "independent routing probe\n"
    assert emitted_by_logger == (route in ("enabled-parent-handler", "local-handler-stops-propagation"))
    stream.seek(0)
    stream.truncate()
    monkeypatch.setattr(_build_info, "build_info", lambda: _build_info.BuildInfo("fixture-version", "a" * 40))
    monkeypatch.setattr(_build_info, "_banner_logged", False)
    expected = ["version = fixture-version", "git commit = " + "a" * 40]

    _build_info.log_build_info_once(logger)
    _build_info.log_build_info_once(logger)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert stream.getvalue().splitlines() == (expected if emitted_by_logger else [])
    assert captured.err.splitlines() == ([] if emitted_by_logger else expected)

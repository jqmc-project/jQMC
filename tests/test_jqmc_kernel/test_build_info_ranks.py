"""The stderr banner fallback follows the supplied MPI lifecycle and rank."""

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

import logging
import sys
from types import SimpleNamespace

import pytest

from jqmc import _build_info

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


@pytest.mark.parametrize(
    "rank,initialized,finalized",
    [(0, True, False), (1, True, False), (3, True, False), (1, False, False), (1, True, True)],
)
def test_api_banner_stderr_fallback_obeys_mpi_rank_and_lifecycle(rank, initialized, finalized, monkeypatch, capsys):
    """A typed communicator supplies rank; only initialized live rank zero prints.

    Before MPI initialization and after finalization the helper intentionally
    behaves as a standalone process. No collective or numerical kernel is
    mocked: this tests the local rank-selection protocol itself.
    """
    rank_reads = []

    def supplied_rank():
        rank_reads.append(rank)
        return rank

    mpi = SimpleNamespace(
        Is_initialized=lambda: initialized,
        Is_finalized=lambda: finalized,
        COMM_WORLD=SimpleNamespace(Get_rank=supplied_rank),
    )
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", mpi)
    monkeypatch.setattr(_build_info, "build_info", lambda: _build_info.BuildInfo("fixture-version", "b" * 40))
    monkeypatch.setattr(_build_info, "_banner_logged", False)
    logger = logging.Logger("jqmc-private-rank-banner", level=logging.WARNING)
    logger.propagate = False
    expected = ["version = fixture-version", "git commit = " + "b" * 40]

    _build_info.log_build_info_once(logger)
    _build_info.log_build_info_once(logger)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines() == (expected if not initialized or finalized or rank == 0 else [])
    assert rank_reads == ([rank] if initialized and not finalized else [])

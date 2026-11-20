"""``Program ends`` must be found even when shutdown noise follows it."""

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

import pytest

from jqmc_workflow.state import _PROGRAM_ENDS_TAIL_BYTES, _check_normal_termination, _has_program_ends

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]

_UCX_NOISE_LINE = (
    "[1794146802.467843] [node:107  :0]          rcache.c:268  UCX  WARN  "
    "munmap(0x14b4054ee000, 151552) failed: Cannot allocate memory\n"
)


def _ucx_noise(min_bytes: int) -> str:
    """MPI-finalize UCX warnings totalling more than *min_bytes*."""
    return _UCX_NOISE_LINE * (min_bytes // len(_UCX_NOISE_LINE) + 1)


def test_program_ends_survives_trailing_grpc_noise(tmp_path):
    out = tmp_path / "output.out"
    noise = ':UNKNOWN:Error received from peer {grpc_message:"CANCELLED"} WatchTasksAsync failed\n' * 600  # ~50 KiB
    out.write_text("E = -17.4 +- 0.001 Ha\nProgram ends 2026-09-06 18:52:00\n" + noise)
    assert _has_program_ends(str(out)) is True


def test_program_ends_survives_more_noise_than_the_tail(tmp_path):
    # More than the scanned tail of stderr noise follows the marker (UCX warnings at MPI finalize).
    out = tmp_path / "output.out"
    noise = _ucx_noise(2 * _PROGRAM_ENDS_TAIL_BYTES)
    out.write_text("E = -31.52 +- 0.0016 Ha\nProgram ends 2026-11-08 23:06:14\n" + noise)
    assert len(noise) > _PROGRAM_ENDS_TAIL_BYTES
    assert _has_program_ends(str(out)) is True
    assert _check_normal_termination(str(tmp_path), [{"status": "fetched", "output_file": out.name}]) == []


def test_program_ends_absent_is_false(tmp_path):
    out = tmp_path / "output.out"
    out.write_text("E = -17.4 +- 0.001 Ha\n" + "x" * 100)
    assert _has_program_ends(str(out)) is False
    assert _has_program_ends(str(tmp_path / "missing.out")) is None


def test_program_ends_absent_under_more_noise_than_the_tail_is_false(tmp_path):
    out = tmp_path / "output.out"
    out.write_text("E = -17.4 +- 0.001 Ha\n" + _ucx_noise(2 * _PROGRAM_ENDS_TAIL_BYTES))
    assert _has_program_ends(str(out)) is False
    assert _check_normal_termination(str(tmp_path), [{"status": "fetched", "output_file": out.name}]) == [out.name]

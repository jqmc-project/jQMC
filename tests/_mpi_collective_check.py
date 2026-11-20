"""Turn a rank-local test failure into a COLLECTIVE test failure.

Why this module exists
----------------------
Several kernel tests compare a production driver against its ``_*_debug``
oracle and guard the per-walker comparisons with ``if mpi_rank == 0:`` --
the oracle gathers its per-walker arrays to rank 0, so only rank 0 holds
the data to assert on.  The *rest* of the test body then calls driver
methods (``get_E``, ``get_aF``, ...) that are MPI **collectives**:
``allreduce`` / ``bcast`` over ``COMM_WORLD``.

That combination is a deadlock waiting to happen.  When an assertion inside
the rank-0-only block fails, rank 0 unwinds straight out of the test
function while every other rank walks on into ``get_E`` and blocks in an
``allreduce`` that rank 0 will never call.  Nothing crashes, nothing is
reported: the job simply stops making progress until its wall-clock cap
kills it, with ZERO ordinary test failures reported, hiding a single
marginal tolerance miss on rank 0.

The fix shape
-------------
Wrap the rank-divergent region in :func:`collective_check`.  Every rank
enters the ``with`` block (the ranks that hold no data simply execute
nothing inside it), the failure -- if any -- is ``allgather``-ed, and then
EVERY rank raises.  All ranks therefore leave the test at the same point,
no rank is left alone in a collective, and the failure is reported in every
rank's pytest output instead of being swallowed by a hang.

This is the "MPI-reduced pass/fail flag at the test boundary" shape: it is
preferred over ``MPI_Abort`` for ordinary assertion failures because it
keeps the pytest report on all ranks and lets the session continue to the
next test.  ``MPI_Abort`` remains the backstop for *unexpected* divergence
(see ``_mpi_test_boundary_sync`` in ``tests/conftest.py``), which no test
can anticipate by construction.
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

import traceback
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

__all__ = ["collective_check"]

# ``pytest.skip`` raises this; a rank-local SKIP diverges exactly as badly as a
# rank-local failure, so it has to travel through the same reduction.
_SKIPPED = pytest.skip.Exception


@contextmanager
def collective_check(comm, label: str = "rank-divergent check") -> Iterator[None]:
    """Run a rank-divergent block so that any rank's failure fails ALL ranks.

    Args:
        comm: The MPI communicator every rank of the test runs under
            (normally ``MPI.COMM_WORLD``).
        label: Short description of the guarded block, prefixed to the
            aggregated failure message.

    Yields:
        Control to the guarded block.

    Raises:
        AssertionError: On **every** rank as soon as the block failed on at
            least one of them, carrying each offending rank's traceback.
        Skipped: On every rank when the block requested a skip on at least
            one rank and raised nothing else anywhere -- a rank-local skip
            diverges just as badly as a rank-local failure.

    Note:
        A single-rank world short-circuits to a plain ``yield``: there is
        nothing to synchronise and the original exception keeps its own
        traceback and type, so single-process runs report exactly what they
        reported before this helper existed.
    """
    if comm.Get_size() == 1:
        yield
        return

    failure: tuple[bool, str] | None = None
    try:
        yield
    except (KeyboardInterrupt, SystemExit):
        # Interpreter-level control flow: propagating it immediately is more
        # important than staying in step with the other ranks, and the
        # boundary watchdog will tear the world down if they get stuck.
        raise
    except BaseException as exc:  # re-raised on every rank below
        failure = (
            isinstance(exc, _SKIPPED),
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )

    reports = comm.allgather(failure)
    offenders = [(rank, report) for rank, report in enumerate(reports) if report is not None]
    if not offenders:
        return

    ranks = [rank for rank, _ in offenders]
    detail = "\n\n".join(f"---- rank {rank} ----\n{report[1]}" for rank, report in offenders)
    header = (
        f"{label}: raised on rank(s) {ranks} of {comm.Get_size()}. Re-raised on every rank "
        "so all of them leave the test at the same point -- a rank-local raise would strand "
        "the other ranks in the next MPI collective (get_E / get_aF) forever."
    )
    if all(report[0] for _, report in offenders):
        pytest.skip(f"{header}\n\n{detail}")
    raise AssertionError(f"{header}\n\n{detail}")

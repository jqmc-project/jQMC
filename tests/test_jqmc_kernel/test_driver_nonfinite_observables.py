"""Collective fail-loud checks shared by the production QMC drivers."""

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

import numpy as np
import pytest

from jqmc._driver_helpers import _raise_collective_nonfinite_observables


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


class _CollectiveComm:
    """Single-process model of the two collectives used by the failure gate."""

    def __init__(self, *, rank=0, size=1, remote_bad_count=0, remote_records=()):
        self.rank = rank
        self.size = size
        self.remote_bad_count = remote_bad_count
        self.remote_records = tuple(remote_records)
        self.allgather_calls = 0

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def allreduce(self, value, op=None):
        del op
        return int(value) + self.remote_bad_count

    def allgather(self, value):
        self.allgather_calls += 1
        return [tuple(value), self.remote_records]


def test_finite_driver_observables_do_not_collect_diagnostics():
    """Finite real/complex samples return without the failure-only allgather."""
    comm = _CollectiveComm()
    _raise_collective_nonfinite_observables(
        comm,
        driver="MCMC",
        stage="measurement",
        first_step=12,
        observables={
            "e_L": np.array([[1.0 + 2.0j, -3.0j], [4.0, 5.0]]),
            "w_L": np.ones((2, 2)),
        },
        r_up_carts=np.zeros((2, 1, 3)),
        r_dn_carts=np.zeros((2, 1, 3)),
    )
    assert comm.allgather_calls == 0


@pytest.mark.parametrize("bad_case", ["nan", "positive-infinity", "negative-infinity", "complex-imaginary-nan"])
def test_local_nonfinite_observable_names_rank_step_walker_and_field(bad_case):
    """A local failure is never omitted or reduced to an anonymous global NaN."""
    bad_value = {
        "nan": np.nan,
        "positive-infinity": np.inf,
        "negative-infinity": -np.inf,
        "complex-imaginary-nan": np.complex128("1+nanj"),
    }[bad_case]
    field = "e_L2" if bad_case == "positive-infinity" else "e_L"
    comm = _CollectiveComm(rank=3, size=8)
    energies = np.zeros((3, 2), dtype=np.complex128)
    energies[1, 1] = bad_value
    with pytest.raises(FloatingPointError) as exc_info:
        _raise_collective_nonfinite_observables(
            comm,
            driver="GFMC_t",
            stage="pre-aggregation",
            first_step=40,
            observables={field: energies, "w_L": np.ones((3, 2))},
            r_up_carts=np.array([[[0.0, 0.0, 0.0]], [[123.0, 0.0, 0.0]]]),
            r_dn_carts=np.zeros((2, 0, 3)),
        )
    message = str(exc_info.value)
    assert "GFMC_t" in message
    assert "pre-aggregation" in message
    assert "precision=" in message
    assert "backend=" in message
    assert "sampling_rank=3" in message
    assert "step=41" in message
    assert "walker=1" in message
    assert f"field={field}" in message
    assert "max_abs_position=123" in message
    assert "refusing to publish or checkpoint" in message
    assert comm.allgather_calls == 1


def test_remote_nonfinite_observable_raises_on_a_locally_finite_rank():
    """Every rank fails together when only another rank owns the bad walker."""
    remote = ("sampling_rank=6, world_rank=6, step=77, walker=0, field=e_L, value=nan, max_abs_position=120",)
    comm = _CollectiveComm(rank=2, size=8, remote_bad_count=1, remote_records=remote)
    with pytest.raises(FloatingPointError, match="sampling_rank=6.*step=77.*walker=0"):
        _raise_collective_nonfinite_observables(
            comm,
            driver="MCMC",
            stage="measurement",
            first_step=77,
            observables={"e_L": np.array([[-31.4]])},
            r_up_carts=np.zeros((1, 1, 3)),
            r_dn_carts=np.zeros((1, 1, 3)),
        )
    assert comm.allgather_calls == 1

"""The FD derivative obligation set is expanded from authored selectors, never shrunk silently."""

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

from gdd.engine.derivative_closure import required_fd_ids

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _fd_contract(values):
    return {
        "fixture_sets": [
            {
                "id": "DERIV-FIX-FD-AUTODIFF-001",
                "reviewed_evidence": {
                    "expected_cases": 999,  # Archival numbers have no authority.
                    "nodes": [
                        {
                            "id": "fixture",
                            "expected_cases": 999,
                            "id_dimensions": ["basis"],
                            "fixture": {"dimensions": {"basis": values}},
                        }
                    ],
                },
            }
        ]
    }


@pytest.mark.parametrize("values", [[], ["gto", "gto"]])
def test_fd_empty_or_duplicate_selectors_fail(values):
    """Selector mistakes cannot silently shrink the required FD case set."""
    with pytest.raises(ValueError):
        required_fd_ids(_fd_contract(values))


def test_authored_universes_equal_their_pins():
    """The exact universes computed from the authored contract are the pinned ones."""
    from gdd.engine import derivative_closure as closure

    assert closure.universe_pin_errors(closure.lane_universes()) == []


KEY = "private|leaf|P01"


def _collection(rows):
    return [
        {
            "precision": "full",
            "execution": {"name": "serial", "world_size": 1, "rank": 0},
            "selected": rows,
        }
    ]


def _emitting(nodeid, keys=(KEY,), **row):
    return {"nodeid": nodeid, "required": True, **row, "derivative_keys": {"lane": "xla-full", "keys": list(keys)}}


@pytest.mark.parametrize("case", ["none", "no-node", "excluded", "two-nodes"])
def test_universe_key_without_exactly_one_collected_required_case_is_rejected(case):
    """Each universe key must be registered by exactly one collected required case.

    ``none`` is the control: one collected case closes the universe. A key no
    collected case registers, a key on an excluded case, or a key on two
    collected cases leaves the closure failing.
    """
    from gdd.engine import derivative_closure as closure

    rows = {
        "none": [_emitting("tests/test_toy.py::test_emit")],
        "no-node": [],
        "excluded": [_emitting("tests/test_toy.py::test_emit", required=False, excluded="needs QE")],
        "two-nodes": [_emitting("tests/test_toy.py::test_emit"), _emitting("tests/test_toy.py::test_other")],
    }[case]
    errors = closure.collection_closure_errors(
        _collection(rows), universes={"xla-full": {KEY}}, pins={"xla-full": closure.digest({KEY})}
    )
    expected = {
        "none": None,
        "no-node": "xla-full derivative universe is not closed",
        "excluded": "is not a required collected case",
        "two-nodes": "belongs to 2 collected nodes",
    }[case]
    if expected is None:
        assert errors == []
    else:
        assert any(expected in error for error in errors), errors

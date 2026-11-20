"""The wired medium-tolerance site gate on the real tree (``gdd_sync --check`` runs it)."""

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

import pytest

from gdd.engine import medium_tolerance_allowlist as allowlist
from scripts import check_gdd_medium_tolerance_sites as checker

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def test_medium_gate_rejects_an_unreviewed_real_site():
    """Every real medium-capable call is reviewed; the same call without its review turns the gate red."""
    assert checker.validation_errors() == []
    reviewed = dict(allowlist.load_medium_tolerance_allowlist()["MEDIUM_SITES"])
    # A real medium-tolerance call site whose
    # review is missing is exactly what an unreviewed new use looks like.
    site = "tests/test_rs_kernel/test_chain_modes.py::_assert_value_parity"
    del reviewed[site]
    errors = checker.validation_errors(reviewed=reviewed)
    assert len(errors) == 1 and f"unreviewed site {site} " in errors[0], errors

"""External stop requests also control the production native MCMC chunk boundary."""

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

from tests.test_jqmc_kernel.test_external_stop_drivers import _assert_stop_continuation
from tests.test_rs_kernel.conftest import native_ctx_spy

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]


@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_native_chunk_external_stop_checkpoint_continues(tmp_path, monkeypatch):
    """H2 native MCMC stops at its real 500-step boundary and resumes 100 steps.

    Two walkers, seed 12345, committed H2 AE Cartesian MOs and Pade J2=1 use the
    production 600-measurement energy-only route. The native entry spy forbids
    XLA fallback, and no chunk cap, reader, physics or tolerance is modified.
    """
    monkeypatch.delenv("JQMC_FLAG_RUST_ON_CPU", raising=False)
    with native_ctx_spy() as calls:
        _assert_stop_continuation(tmp_path, monkeypatch, "mcmc", total=600, expected_stop=500)
    assert calls["run_energy_chunk_batch"] > 0, "the production native energy chunk must actually execute"

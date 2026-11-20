"""Resource check of the production DTN derivative kernels: no pair chunk wider than its pair list (issue #483).

The closed-form DTN derivatives stream the electron pairs in ``lax.scan`` chunks built by
``derivatives.dtn._chunk_pairs``. A chunk width of ``budget // per_pair`` without a clamp pads a
short pair list up to one full chunk and computes it in full (masked to zero, so values stay
correct): 28 Si2 pairs become 16,777 computed pairs, and under the GPU walker ``vmap`` the padded
operand reaches 12.9 GB and aborts XLA autotuning. The check records every
``_chunk_pairs`` call of the production kernels and is structural, so no tolerance constants are
involved. It imports only names that exist with and without the clamp, so removing the clamp makes
it fail on the recorded widths rather than on an import.
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

import jax
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

jax.config.update("jax_enable_x64", True)


@pytest.mark.light
def test_production_dtn_derivatives_never_pad_pairs_beyond_the_pair_count(monkeypatch):
    """The production DTN derivative kernels never chunk a pair list wider than its pair count.

    An unclamped pair-chunk width ``budget // per_pair`` pads a short pair list up to one full
    chunk (28 Si2 pairs -> 16,777 computed pairs; under the GPU walker ``vmap`` the padded
    operand reaches 12.9 GB and aborts XLA autotuning). Every call of
    ``derivatives.dtn._chunk_pairs`` made while tracing ``grad_dtn_wrt_R``, ``grad_dtn_wrt_strain``
    and ``grad_dtn_wrt_param`` on a periodic H2 J_DTN (u, chi and f terms, 3 + 2 electrons) is
    recorded, and each must use a chunk no wider than its pair list. The JAX caches are cleared so
    the kernels are really traced (non-vacuity: a pair list longer than one pair is chunked).
    """
    import jqmc.derivatives.dtn as dtn_mod
    from tests.test_jqmc_kernel.test_jastrow_dtn import _build_dtn, _draw_near_nuclei, _h2_pbc_structure

    calls = []
    original = dtn_mod._chunk_pairs

    def spy(ia, ib, chunk):
        calls.append((int(np.asarray(ia).shape[0]), int(chunk)))
        return original(ia, ib, chunk)

    monkeypatch.setattr(dtn_mod, "_chunk_pairs", spy)
    sd = _h2_pbc_structure()
    dtn = _build_dtn(sd, preset="u_chi_f", cutoff_smoothness=4, seed=37)
    r_up, r_dn = _draw_near_nuclei(sd, 3, 2, seed=41, spread=0.7)
    jax.clear_caches()
    for kernel in (dtn_mod.grad_dtn_wrt_R, dtn_mod.grad_dtn_wrt_strain, dtn_mod.grad_dtn_wrt_param):
        jax.block_until_ready(kernel(dtn, r_up, r_dn))
    assert any(n > 1 for n, _ in calls), calls
    padded = [(n, chunk) for n, chunk in calls if chunk > n]
    assert padded == [], f"pair lists chunked wider than their pair count (n_pair, chunk): {padded}"

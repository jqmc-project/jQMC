"""DTN kinetic shared-inverse SVD-lowering check, ISOLATED in its own file.

This test lowers SIX large production graphs in one test (``bridge._jit_e_L_giv``
for r/R/eps as StableHLO text plus both ``batched_grad_e_L`` variants per axis).
Run late in a shared xdist worker it is the single most reliable trigger of the
documented XLA-accumulation worker segfault (a crash at exactly this test
in a shared worker of ``test_jqmc_derivatives``, while every in-isolation rerun
passes). Housing it alone in its own test FILE gives it a fresh process -- the same remedy the
sibling DTN ``_auto`` oracle rows use via per-axis parametrization (see the
``test_pbc_dtn_ecp_e_L_closed_form_vs_auto`` docstring in
``test_jqmc_derivatives.py``). Helpers are imported from that module (single
source of truth); the assertion body is moved verbatim.
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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled  # noqa: E402
from tests.test_jqmc_kernel.test_jqmc_derivatives import _build_pbc_dtn_ham, _draw  # noqa: E402


@pytest.mark.light
@pytest.mark.parametrize("stem", ["lithium.h5"])
def test_dtn_kinetic_shared_inverse_no_svd(stem):
    """The DTN kinetic cotangent must reuse the running geminal inverse (no fresh pinv SVD). With the
    per-walker ``geminal_inv`` threaded into the production e_L/{r,R,eps} closures, the lowered
    StableHLO carries ZERO ``lapack_dgesdd_ffi`` for ALL of e_L/{r,R,eps}. On the eps axis
    ``one_move_mesh.mesh_stress_wf`` uses the closed-form periodic Pulay-stress delta
    (``sum_g W_g (stress_lnPsi_pbc(config) - stress_lnPsi_pbc(base))``) streamed with per-config
    Sherman-Morrison inverses off the base ``G^{-1}``, not autodiff of the strained log-ratio through a
    fresh strained-geminal pinv -- no fresh pinv on any config.
    Numerically, providing the inverse is inert: the r/R/eps DTN kinetic gradients agree with the
    SVD-pinv fallback to fp."""
    from jqmc.derivatives import bridge
    from jqmc.derivatives.fast import geminal_pinv
    from jqmc.derivatives.hoist import cartesianize_orbitals
    from jqmc.determinant import compute_geminal_all_elements

    ham = _build_pbc_dtn_ham(stem)
    assert ham.structure_data.pbc_flag and ham.wavefunction_data.jastrow_data.jastrow_dtn_data is not None
    gd = ham.wavefunction_data.geminal_data
    ru0, rd0 = _draw(ham, seed=0)
    ru1, rd1 = _draw(ham, seed=1)
    rub = jnp.stack([ru0, ru1])
    rdb = jnp.stack([rd0, rd1])
    RTb = jnp.tile(jnp.eye(3)[None], (rub.shape[0], 1, 1))
    gi_b = jax.jit(jax.vmap(lambda a, b: geminal_pinv(compute_geminal_all_elements(gd, a, b))))(rub, rdb)
    jax.block_until_ready(gi_b)

    hc = cartesianize_orbitals(ham)
    expected_svd = {"r": 0, "R": 0, "eps": 0}  # eps 1->0: mesh_stress_wf autodiff replaced by closed form
    level = "medium" if is_mixed_precision_enabled() else "strict"
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), level)
    for wrt in ("r", "R", "eps"):
        txt = bridge._jit_e_L_giv(wrt, False).lower(hc, rub, rdb, RTb, gi_b).as_text()
        assert txt.count("lapack_dgesdd_ffi") == expected_svd[wrt], (
            f"e_L/{wrt}: {txt.count('lapack_dgesdd_ffi')} SVD FFI calls, expected {expected_svd[wrt]}"
        )
        a = np.asarray(bridge.batched_grad_e_L(wrt=wrt, with_geminal_inv=False)(ham, rub, rdb, RTb))
        b = np.asarray(bridge.batched_grad_e_L(wrt=wrt, with_geminal_inv=True)(ham, rub, rdb, RTb, gi_b))
        np.testing.assert_allclose(b, a, atol=atol, rtol=rtol)

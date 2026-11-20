"""The ``compute_grads_and_laplacian_ln_Det`` backward pass emits no ``ComplexWarning`` at k != Gamma.

``_grads_lap_bwd`` differentiates ``_grads_lap_body`` with ``jax.vjp``.  For a complex
Bloch geminal the cotangents of the real input slots (electron coordinates) are the
real part of complex intermediate cotangents.  Under the pinned JAX, the VJP transpose
of a real -> complex promotion does not warn, so the backward pass needs no
``warnings.catch_warnings`` + ``filterwarnings("ignore", ComplexWarning)`` around that
``vjp_fn`` call.

This module is the evidence: it takes the reverse-mode
derivative of every ``compute_grads_and_laplacian_ln_Det`` output with respect to the
geminal parameters and both electron-coordinate slots, eagerly and under ``jax.jit``,
for the complex twist SD (time reversal), dense AO AGP and AGPn cases, with
``ComplexWarning`` promoted to an error (both here and by the session-wide conftest
trap).  A spy on ``_grads_lap_body`` (called only from the backward pass) proves the
backward pass ran on complex data.
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

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.exceptions import ComplexWarning

from jqmc import determinant
from tests.test_jqmc_kernel._twist_geminal_cases import build_case

pytestmark = [pytest.mark.with_chdir]

jax.config.update("jax_enable_x64", True)


def _scalar_of_outputs(geminal_data, r_up, r_dn):
    outputs = determinant.compute_grads_and_laplacian_ln_Det(geminal_data, r_up, r_dn)
    return sum(jnp.sum(jnp.real(value)) + jnp.sum(jnp.imag(value)) for value in outputs)


@pytest.mark.light
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("case", ["sd_tr", "agp", "agpn"])
def test_grads_lap_bwd_at_twist_emits_no_complex_warning(case, jit, monkeypatch):
    """Reverse mode through compute_grads_and_laplacian_ln_Det at k=(0.25,0,0) raises no ComplexWarning.

    H4_pbc_ae_k025_cart.h5 (all-electron, complex Bloch) with the twist geminal cases of
    _twist_geminal_cases; PCG64(2026) configuration.  The backward pass must run (spy on
    _grads_lap_body, complex outputs), and the electron-coordinate cotangents must be real,
    finite and non-zero.
    """
    geminal_data = build_case(case).wavefunction_data.geminal_data
    rng = np.random.Generator(np.random.PCG64(2026))
    r_up = jnp.asarray(rng.normal(scale=1.5, size=(int(geminal_data.num_electron_up), 3)))
    r_dn = jnp.asarray(rng.normal(scale=1.5, size=(int(geminal_data.num_electron_dn), 3)))

    seen_dtypes = []
    original_body = determinant._grads_lap_body

    def spy_body(*args):
        outputs = original_body(*args)
        seen_dtypes.extend(value.dtype for value in outputs)
        return outputs

    monkeypatch.setattr(determinant, "_grads_lap_body", spy_body)
    gradient = jax.grad(_scalar_of_outputs, argnums=(0, 1, 2))
    if jit:
        gradient = jax.jit(gradient)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ComplexWarning)
        _grad_geminal, grad_r_up, grad_r_dn = gradient(geminal_data, r_up, r_dn)
        grad_r_up = np.asarray(grad_r_up)
        grad_r_dn = np.asarray(grad_r_dn)

    assert seen_dtypes, "the custom-VJP backward pass did not run"
    assert any(np.issubdtype(dtype, np.complexfloating) for dtype in seen_dtypes), seen_dtypes
    for label, value in (("r_up", grad_r_up), ("r_dn", grad_r_dn)):
        assert not np.iscomplexobj(value), (label, value.dtype)
        assert np.all(np.isfinite(value)), label
        assert np.max(np.abs(value)) > 0.0, label
    jax.clear_caches()

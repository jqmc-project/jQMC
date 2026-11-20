"""AO-level PERMUTATION-EQUIVARIANCE property tests (non-shell-major basis support).

Permuting the stored AO ordering yields a PHYSICALLY IDENTICAL basis, so every per-AO
computation ``f`` must satisfy ``f(P(ao)) == P_axis(f(ao))`` elementwise -- the AO axis
of the output permutes with the input and NO value changes. This is a same-engine
same-code identity, asserted at the tightest achievable floor.

The permutation ``P`` (``_ragged_permute.interleave_perm``) INTERLEAVES same-``l`` shells
so a shell's ``2l+1`` / Cartesian components are NON-contiguous -- the layout of a QE-converted basis.
Coverage: value / grad / laplacian and the analytic Cartesian-GTO AO-derivative kernels,
over Cartesian AND spherical classes, molecular AND periodic (ragged pGTO, ``n_image>1``).

Cross-validation: every per-AO kernel is layout-FREE; the ONE layout-sensitive consumer is the
traced-ham spherical->Cartesian twin
(``_aos_sphe_to_cart`` -> ``_aos_sphe_to_cart_traced`` -> ``_ragged_structural_shell_proxy``)
on a periodic spherical-pGTO leaf -- ``test_traced_c2s_reconstruction`` -- which fails with the
ragged-proxy ``RuntimeError`` unless the twin handles a non-shell-major layout.
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

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.with_chdir, pytest.mark.light]

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import get_tolerance_min  # noqa: E402
from jqmc.atomic_orbital import (  # noqa: E402
    _aos_sphe_to_cart,
    compute_AOs,
    compute_AOs_grad,
    compute_AOs_laplacian,
)
from jqmc.derivatives.ao_deriv import ao_grad, ao_laplacian, ao_hessian, ao_gradlap, j3_basis_derivs  # noqa: E402
from jqmc.trexio_wrapper import read_trexio_file  # noqa: E402
from tests.test_jqmc_kernel._ragged_permute import (  # noqa: E402
    has_interleavable_shell,
    interleave_perm,
    molecular_view,
    permute_aos,
    random_perm,
)

_TREXIO = os.path.join(os.path.dirname(__file__), "..", "trexio_example_files")


def _tols():
    """Precision-aware tolerances, resolved at CALL time (not import time).

    The pytest ``--precision-mode`` plugin configures the mode after module
    import, so a module-level ``get_tolerance_min`` would bake in the fp64
    tolerances and fail the mixed[fp64/fp32] lanes at fp32 roundoff (the
    ``ao_eval`` zone is float32 under mixed). Same convention as the sibling
    ``test_atomic_orbital.py`` (in-function ``get_tolerance`` lookups).
    """
    return get_tolerance_min(("ao_eval", "ao_grad_lap"), "strict")


# (leaf-id -> trexio stem): Cartesian AND spherical, molecular AND periodic (ragged pGTO).
_LEAVES = {
    "mol_cart": "H2_ae_ccpvdz_cart.h5",
    "mol_sphe": "H2_ae_ccpvdz_sphe.h5",
    "pbc_cart_pgto": "lithium_cart.h5",
    "pbc_sphe_pgto": "lithium.h5",
}


def _load_aos(stem):
    aos = read_trexio_file(os.path.join(_TREXIO, stem))[1]
    return aos


def _perm_for(aos):
    """Prefer the shell-INTERLEAVING permutation (breaks a shell's contiguity -- the layout of a
    QE-converted basis); fall back to a deterministic full shuffle when the basis has no width>1 same-``l``
    duplicate to interleave (still non-trivial; the per-AO property is layout-free)."""
    mol = molecular_view(aos)
    p = interleave_perm(mol) if has_interleavable_shell(mol) else []
    if p == list(range(mol.num_ao)) or not p:
        p = random_perm(mol.num_ao, seed=3)
    assert p != list(range(len(p))), "permutation must be non-trivial"
    return p


def _r_carts(aos, n=4, seed=0):
    rng = np.random.default_rng(seed)
    if aos.structure_data.pbc_flag:
        return jnp.asarray(rng.random((n, 3)) @ np.asarray(aos.structure_data._cell_np))
    return jnp.asarray(rng.normal(scale=1.0, size=(n, 3)))


# --------------------------------------------------------------------------- #
# value / grad / laplacian: per-AO equivariance (layout-free everywhere).
# --------------------------------------------------------------------------- #
def _stack(out):
    return np.stack([np.asarray(x) for x in out]) if isinstance(out, tuple) else np.asarray(out)


_FUNCS = {
    "value": compute_AOs,
    "grad": compute_AOs_grad,
    "lap": compute_AOs_laplacian,
}


@pytest.mark.parametrize("leaf_id", list(_LEAVES))
@pytest.mark.parametrize("func_id", list(_FUNCS))
def test_compute_aos_equivariance(leaf_id, func_id):
    """Compute AOs equivariance."""
    aos = _load_aos(_LEAVES[leaf_id])
    perm = _perm_for(aos)
    r = _r_carts(aos)
    f = _FUNCS[func_id]
    ref = _stack(f(aos, r))  # (..., num_ao, Ne)
    got = _stack(f(permute_aos(aos, perm), r))  # (..., num_ao, Ne) in permuted AO order
    # AO axis is the second-to-last; f(P(ao)) must equal f(ao) gathered by perm on that axis.
    expected = np.take(ref, np.asarray(perm), axis=-2)
    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap"), "strict")
    np.testing.assert_allclose(got, expected, atol=atol, rtol=rtol, err_msg=f"{leaf_id}/{func_id} not equivariant")


# --------------------------------------------------------------------------- #
# analytic AO-derivative kernels (jqmc.derivatives.ao_deriv) -- carry an AO axis.
# ao_grad / ao_laplacian / ao_hessian / ao_gradlap consume a Cartesian-GTO leaf; each
# returns a molecular-rank array with the AO axis at ``num_ao``. j3_basis_derivs is
# sphe->cart aware (it invokes the c2s transform inside the kernel), so it also covers
# the SPHERICAL molecular leaf. Molecular leaves: the periodic per-AO axis is exercised
# by the value/grad/lap rows above; the periodic sphe c2s by test_traced_c2s_reconstruction.
# --------------------------------------------------------------------------- #
def _assert_axis_equivariant(out_ref, out_got, num_ao, perm, tag):
    outs_ref = out_ref if isinstance(out_ref, tuple) else (out_ref,)
    outs_got = out_got if isinstance(out_got, tuple) else (out_got,)
    atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap"), "strict")
    assert len(outs_ref) == len(outs_got)
    for k, (a, b) in enumerate(zip(outs_ref, outs_got)):
        output_label = str(k)
        a, b = np.asarray(a), np.asarray(b)
        ax = next(i for i, s in enumerate(a.shape) if s == num_ao)
        expected = np.take(a, np.asarray(perm), axis=ax)
        np.testing.assert_allclose(b, expected, atol=atol, rtol=rtol, err_msg=f"{tag}[out{output_label}] not equivariant")


@pytest.mark.parametrize("kernel_id", ["ao_grad", "ao_laplacian", "ao_hessian", "ao_gradlap"])
def test_cart_ao_deriv_equivariance(kernel_id):
    """Cart AO deriv equivariance."""
    kern = {"ao_grad": ao_grad, "ao_laplacian": ao_laplacian, "ao_hessian": ao_hessian, "ao_gradlap": ao_gradlap}[kernel_id]
    aos = _load_aos(_LEAVES["mol_cart"])
    perm = _perm_for(aos)
    r = _r_carts(aos)
    _assert_axis_equivariant(kern(aos, r), kern(permute_aos(aos, perm), r), aos.num_ao, perm, f"mol_cart/{kernel_id}")


@pytest.mark.parametrize("leaf_id", ["mol_cart", "mol_sphe"])
def test_j3_basis_derivs_equivariance(leaf_id):
    """The analytic J3-basis derivative tower (value/grad/lap/hessian/gradlap); the ``mol_sphe``
    case drives the spherical->Cartesian transform INSIDE the derivative kernel."""
    aos = _load_aos(_LEAVES[leaf_id])
    perm = _perm_for(aos)
    r = _r_carts(aos)
    _assert_axis_equivariant(
        j3_basis_derivs(aos, r), j3_basis_derivs(permute_aos(aos, perm), r), aos.num_ao, perm, f"{leaf_id}/j3_basis_derivs"
    )


# --------------------------------------------------------------------------- #
# The layout-SENSITIVE consumer: traced-ham spherical->Cartesian twin.
# Under a jit the basis exponents are TRACERS, so `_aos_sphe_to_cart` routes to
# `_aos_sphe_to_cart_traced` -> `_ragged_structural_shell_proxy`. On a permuted
# (non-shell-major) periodic sphe-pGTO leaf the reconstructed spherical values
# `T @ cart_vals == sphe_vals` must hold.
# --------------------------------------------------------------------------- #
@jax.jit
def _traced_recon(aos, r):
    cart, T = _aos_sphe_to_cart(aos)  # exponents are tracers here -> traced twin -> proxy
    sphe_vals = compute_AOs(aos, r)  # (num_sphe, Ne)
    cart_vals = compute_AOs(cart, r)  # (num_cart, Ne)
    return sphe_vals, jnp.asarray(T) @ cart_vals


@pytest.mark.parametrize("leaf_id", ["mol_sphe", "pbc_sphe_pgto"])
def test_traced_c2s_reconstruction(leaf_id):
    """Traced c2s reconstruction."""
    aos = _load_aos(_LEAVES[leaf_id])
    perm = _perm_for(aos)
    r = _r_carts(aos)
    sphe_vals, recon = _traced_recon(permute_aos(aos, perm), r)
    np.testing.assert_allclose(
        np.asarray(recon),
        np.asarray(sphe_vals),
        atol=_tols()[0],
        rtol=_tols()[1],
        err_msg=f"{leaf_id} traced c2s reconstruction",
    )

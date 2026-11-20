"""Periodic pGTO gradients and Laplacians on non-orthogonal cells against an independent AD lattice sum.

The value, overlap, seam and ECP oracles of
``test_noncubic_cell_oracles.py`` do not cover the analytic periodic gradient and
Laplacian on a skewed cell. Here the reference is automatic differentiation of the
test's own brute-force lattice sum

    F_mu(r) = sum_n exp(2 pi i k.n) chi_mu(r - n @ cell)

over every image inside the Gaussian cutoff that ``_noncubic_oracles`` derives
itself; the production image list, per-atom image radius and fold are not used.
``chi`` is the open-boundary (molecular) AO kernel, pinned separately to a
plain-numpy Cartesian GTO by ``test_periodic_pgto_values_match_bruteforce_image_sum``.

The production leaf is built with the tighter truncation used by the existing
face-crossing Laplacian check (AO_IMAGE_TRUNCATION_ATOL**2), because the
production truncation bound is stated for AO values; the comparison budget is the
existing ``ao_eval``/``ao_grad_lap`` strict policy (the AD reference differentiates
the ``ao_eval`` kernel, as the cubic analytic-vs-AD rows do) plus that truncation.
A halved per-atom image radius on the triclinic cell must fail in full and mixed.
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

from jqmc._precision import get_tolerance_min
from jqmc.atomic_orbital import compute_AOs, compute_AOs_grad, compute_AOs_laplacian
from tests.test_jqmc_kernel import _noncubic_oracles as ref
from tests.test_jqmc_kernel.test_noncubic_cell_oracles import (
    _CELLS,
    _FACE_LAPLACIAN_TOL,
    _TWIST,
    _assert_finite,
    _electrons,
    _placed_leaf,
)

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

_AD_CELLS = ("primitive_fcc_si", "triclinic_skewed")


def ad_bruteforce_grad_lap(cell, positions, kind, r, k):
    """AD gradient (3, num_ao, N_e) and Laplacian (num_ao, N_e) of the brute-force lattice sum."""
    molecular = ref.build_aos(ref.build_structure(None, positions), kind)
    alpha_min = min(min(e) for _, e, _ in ref.SHELLS)
    l_max = max(l for l, _, _ in ref.SHELLS)
    radius = ref._gaussian_cutoff(alpha_min, l_max)
    displacements = (np.asarray(r)[:, None, :] - np.asarray(positions)[None, :, :]).reshape(-1, 3)
    images = ref.lattice_images_within(cell, displacements, radius)
    shifts = jnp.asarray(images.astype(np.float64) @ cell)
    phase = np.ones(len(images)) if k is None else np.exp(2j * np.pi * (images @ np.asarray(k)))
    phase = jnp.asarray(phase)

    def lattice_sum(point):
        values = compute_AOs(aos_data=molecular, r_carts=point[None, :] - shifts)  # (num_ao, n_images)
        return jnp.sum(values.astype(phase.dtype) * phase[None, :], axis=1)

    def split(f):
        if k is None:
            return lambda point: jnp.real(f(point))
        return lambda point: jnp.stack([jnp.real(f(point)), jnp.imag(f(point))])

    def derivatives(point):
        parts = split(lattice_sum)
        grad = jax.jacfwd(parts)(point)  # (..., num_ao, 3)
        hess = jax.jacfwd(jax.jacfwd(parts))(point)  # (..., num_ao, 3, 3)
        lap = jnp.trace(hess, axis1=-2, axis2=-1)
        return grad, lap

    grad, lap = jax.jit(jax.vmap(derivatives))(jnp.asarray(r))
    grad, lap = np.asarray(grad), np.asarray(lap)
    if k is not None:  # (N_e, 2, num_ao, ...) -> complex
        grad = grad[:, 0] + 1j * grad[:, 1]
        lap = lap[:, 0] + 1j * lap[:, 1]
    # (N_e, num_ao, 3) -> (3, num_ao, N_e); (N_e, num_ao) -> (num_ao, N_e)
    return np.transpose(grad, (2, 1, 0)), lap.T, len(images)


def check_grad_lap_against_ad(cell_name, kind, k):
    """Raise ``AssertionError('periodic AO grad/lap AD mismatch ...')`` on disagreement."""
    cell = _CELLS[cell_name]
    positions = ref.ATOM_FRACTIONS @ cell
    leaf = _placed_leaf(cell, positions, kind, k, _FACE_LAPLACIAN_TOL)
    r = _electrons(cell, 20260928)
    prod_grad = np.stack([np.asarray(g) for g in compute_AOs_grad(aos_data=leaf, r_carts=jnp.asarray(r))])
    prod_lap = np.asarray(compute_AOs_laplacian(aos_data=leaf, r_carts=jnp.asarray(r)))
    ref_grad, ref_lap, n_images = ad_bruteforce_grad_lap(cell, positions, kind, r, k)
    _assert_finite(prod_grad, prod_lap, ref_grad, ref_lap)
    assert n_images > 27, f"brute-force sum is not a lattice sum ({n_images} images)"
    assert prod_grad.shape == ref_grad.shape and prod_lap.shape == ref_lap.shape
    zone_atol, rtol = get_tolerance_min(("ao_eval", "ao_grad_lap"), "strict")
    atol = _FACE_LAPLACIAN_TOL + zone_atol
    for quantity, prod, brute in (("grad", prod_grad, ref_grad), ("laplacian", prod_lap, ref_lap)):
        if not np.allclose(prod, brute, atol=atol, rtol=rtol, equal_nan=False):
            raise AssertionError(
                f"periodic AO grad/lap AD mismatch on {cell_name}/{kind}/k={k}/{quantity}: "
                f"max|prod-ref|={np.max(np.abs(prod - brute)):.3e}, max|ref|={np.max(np.abs(brute)):.3e}"
            )
    return ref_grad, ref_lap


@pytest.mark.parametrize("kind", ["cart", "sphe"])
@pytest.mark.parametrize("cell_name", _AD_CELLS)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_periodic_pgto_grad_and_laplacian_match_ad_of_bruteforce_lattice_sum(cell_name, kind):
    """Analytic periodic pGTO gradients and Laplacians on FCC and triclinic cells equal AD of a lattice sum.

    Synthetic parameter set, no artifact: two H centres at fractional
    (0.1,0.2,0.15) and (0.7,0.55,0.8), s(1.3,0.35)/p/d/f shells, Cartesian or
    spherical. Six electrons from PCG64(seed=20260928) at fractional positions in
    [-0.3, 1.3), so production wraps them. Gamma and k=(0.25,-0.1,0.4). The
    reference is jax.jacfwd (gradient) and the trace of the nested jacfwd Hessian
    (Laplacian) of the test's own brute-force image sum of the molecular kernel,
    with its own Gaussian cutoff and at least 28 images; the production image
    list is not used. Tolerance: existing ao_eval/ao_grad_lap strict policy plus
    the leaf truncation AO_IMAGE_TRUNCATION_ATOL**2, full and mixed separately.
    """
    cell = _CELLS[cell_name]
    gram = cell @ cell.T
    assert np.max(np.abs(gram - np.diag(np.diag(gram)))) > 1.0, f"{cell_name} is orthogonal"
    for k in (None, _TWIST):
        ref_grad, ref_lap = check_grad_lap_against_ad(cell_name, kind, k)
        assert np.max(np.abs(ref_grad)) > 1.0e-2 and np.max(np.abs(ref_lap)) > 1.0e-2, "derivatives are vacuous"
        if k is not None:
            assert np.max(np.abs(ref_grad.imag)) > 1.0e-2, "twist reference is effectively real"
    jax.clear_caches()


@pytest.mark.parametrize("fault", ["halve-per-atom-image-radius"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_noncubic_grad_lap_ad_oracle_rejects_halved_image_radius(fault, monkeypatch):
    """Halving the per-atom image radius drops needed triclinic images and fails the AD check.

    Same inputs as the triclinic Cartesian Gamma row. ``_primitive_image_radius``
    is halved (the acceptance fault "halve the per-atom box"), so production keeps
    fewer images than the brute-force sum; the unmutated row passes above. The
    check must raise its own message in full and mixed at unchanged tolerances.
    """
    from jqmc import structure as st

    assert fault == "halve-per-atom-image-radius"
    original = st._primitive_image_radius
    monkeypatch.setattr(st, "_primitive_image_radius", lambda e, L, tol: 0.5 * original(e, L, tol))
    jax.clear_caches()
    with pytest.raises(AssertionError, match="periodic AO grad/lap AD mismatch on triclinic_skewed/cart"):
        check_grad_lap_against_ad("triclinic_skewed", "cart", None)
    monkeypatch.undo()
    jax.clear_caches()

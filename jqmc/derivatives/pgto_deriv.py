"""Periodic (pgto) Cartesian-GTO AO derivatives, autograd-free -- the higher-order AO
kernels the PBC kinetic force/stress need (grad, grad-of-laplacian, Hessian), folded to
molecular rank.

A periodic AO is an image-tiled molecular cart AO: phi(r) = Sum_n phi_kernel(r; R_I + T_n)
(times the Bloch phase for a twisted leaf). The image index n is independent of the nuclear
position R, so ANY spatial derivative folds trivially: evaluate the molecular cart-GTO
derivative (``ao_deriv.ao_deriv``) at kernel rank (n_image * num_ao) on the wrapped
electrons, then fold the image axis with the leaf's own image-phase operation and
apply the wrap-restore phase for complex leaves. The implementation uses JAX arrays
throughout so it composes with the walker-batch JIT.
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

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from jqmc.atomic_orbital import (
    AOs_base_data,
    AOs_cart_pgto_complex_data,
    AOs_cart_pgto_data,
    AOs_sphe_pgto_complex_data,
    AOs_sphe_pgto_data,
)
from jqmc._precision import get_complex_dtype_jnp
from jqmc.molecular_orbital import MOs_data
from jqmc.structure import _wrap_into_cell_if_pbc

from .ao_deriv import (
    GRAD_GRADLAP_HESS_ORDERS,
    GRAD_ORDERS,
    GRADLAP_ORDERS,
    HESS_ORDERS,
    LAP_ORDERS,
    AxisOrder,
    ao_deriv_set,
    fields_from_deriv_set,
)


def _is_complex_pgto(aos: AOs_base_data) -> bool:
    return hasattr(aos, "_image_phase_factors") and hasattr(aos, "_restore_phase_np")


def pgto_deriv_set(
    aos: AOs_cart_pgto_data | AOs_cart_pgto_complex_data,
    r_carts: jax.Array,
    orders: tuple[AxisOrder, ...],
) -> jax.Array:
    """Folded periodic cart-GTO AO derivatives of every order in ``orders``; shape
    ``(len(orders), num_ao, N)``.

    Wraps the electrons into the home cell (matching jqmc's ``_compute_AOs``) once, evaluates all
    orders at kernel rank in ONE primitive sweep (:func:`ao_deriv.ao_deriv_set`) on the image-tiled
    AOs, folds the image axis of all orders with one product against the kernel-AO -> AO map
    (Bloch phase for complex leaves) and applies the wrap-restore phase once.

    Pure ``jnp`` (the cart core and the ``_fold_image_axis`` / ``_restore_phase`` helpers are all
    jnp twins of the ``_np`` debug variants), so it composes inside the ``jit(vmap(...))`` walker
    batch -- no host sync."""
    r_j = jnp.asarray(r_carts, jnp.float64)
    r_w = _wrap_into_cell_if_pbc(aos.structure_data, r_j)
    raw = ao_deriv_set(aos, r_w, orders)  # (n_ord, n_image*num_ao, N) kernel rank
    # Image fold of all orders as ONE matrix product with the static kernel-AO -> AO map
    # (one-hot, Bloch phase e^{ik.T} for a complex leaf): the same sum as the leaf's
    # ``_fold_image_axis``, for a stacked table.
    _, phase, home = _pgto_kernel_image(aos)
    fold = (home[:, None] == jnp.arange(int(aos.num_ao))[None, :]) * phase[:, None]  # (n_kernel, num_ao)
    if _is_complex_pgto(aos):
        fold = fold.astype(get_complex_dtype_jnp("ao_grad_lap"))
        raw = raw.astype(fold.dtype)
    folded = jnp.stack([fold.T @ raw[i] for i in range(raw.shape[0])], 0)  # (n_ord, num_ao, N)
    if _is_complex_pgto(aos):
        folded = folded * aos._restore_phase(r_j, "ao_grad_lap")[None, None, :]
    return folded


def pgto_deriv(
    aos: AOs_cart_pgto_data | AOs_cart_pgto_complex_data,
    r_carts: jax.Array,
    order: tuple[int, int, int],
) -> jax.Array:
    """Folded periodic cart-GTO AO derivative of ``order=(ax,ay,az)``; shape (num_ao, N)
    (one order of :func:`pgto_deriv_set`)."""
    return pgto_deriv_set(aos, r_carts, (order,))[0]


def _pgto_kernel_image(aos: AOs_base_data) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-kernel-AO (T (M,3), phase (M,), home (M,)) for the PBC stress image-T correction.
    T_k = image_index_k . cell; phase_k = e^{ik.T_k} (complex leaf) or 1; home_k = the folded
    (molecular) AO that kernel-AO k folds into."""
    # cell from the jnp twin (_cell_jnp), NOT _cell_np: under the production bridge the hamiltonian
    # is a TRACED jit arg, so the lattice (vec_a..) are tracers and _cell_np would raise. The image
    # indices / phase / home map are genuinely static (pytree_node=False) -> jnp.asarray = constants.
    cell = jnp.asarray(aos.structure_data._cell_jnp, jnp.float64)
    n = jnp.asarray(np.asarray(aos._image_indices, np.float64).reshape(-1, 3))  # (M, 3) static
    T = n @ cell
    phase = jnp.asarray(aos._image_phase_factors) if _is_complex_pgto(aos) else jnp.ones(n.shape[0])
    home = (
        jnp.asarray(np.asarray(aos._ragged_kernel_to_home))
        if getattr(aos, "_ragged_kernel_to_home", ())
        else jnp.asarray(np.tile(np.arange(int(aos.num_ao)), int(aos._n_image)))
    )
    return T, phase, home


def sphe_leaf_to_cart(
    aos: AOs_cart_pgto_data | AOs_cart_pgto_complex_data | AOs_sphe_pgto_data | AOs_sphe_pgto_complex_data,
) -> tuple[AOs_cart_pgto_data | AOs_cart_pgto_complex_data, npt.NDArray]:
    """(cart_aos, T_mol) for a sphe-pgto AO leaf: sphe_AO = T_mol @ cart_AO (molecular rank).

    For a ragged periodic basis, ``_aos_sphe_to_cart`` de-tiles, converts the molecular
    spherical basis to Cartesian form, and then re-expands it; a
    twisted leaf is promoted to its complex sibling (per-kernel Bloch phase). cart -> (aos, I)."""
    from jqmc.atomic_orbital import _aos_sphe_to_cart, _promote_pgto_to_complex

    if "sphe" not in type(aos).__name__:
        return aos, np.eye(int(aos.num_ao))
    aos_cart, T = _aos_sphe_to_cart(aos)
    n_sph, n_cart = int(aos.num_ao), int(aos_cart.num_ao)
    T_mol = np.asarray(T)[:n_sph, :n_cart]
    if _is_complex_pgto(aos):
        aos_cart = _promote_pgto_to_complex(aos_cart, aos.k_point)
    return aos_cart, T_mol


def _orb_aos_and_coeff(orb_data: AOs_base_data | MOs_data) -> tuple[AOs_base_data, jax.Array]:
    """(aos_container, C) for a geminal orbital leaf: ``orbital = C @ AO`` at the container's AO rank.

    ``MOs_data`` -> its ``aos_data`` + the MO coefficient matrix (``(num_mo, num_ao)``, possibly
    traced/complex). A BARE dense AO-lambda AGP leaf (any raw AO leaf / composite) -> the leaf itself
    + an identity coefficient matrix at its total AO rank (the orbital IS the AO; the dense
    ``lambda_matrix`` carries the coupling, exactly as ``basis._orb_pieces`` treats a bare leaf).
    Keeping ``C`` a jnp array lets the leaf split below trace cleanly inside the jit bridge."""
    if type(orb_data).__name__ == "MOs_data":
        return orb_data.aos_data, orb_data._mo_coefficients_jnp  # keep jnp (traced under the jit bridge)
    return orb_data, jnp.eye(int(orb_data.num_ao), dtype=jnp.float64)


def sphe_pgto_to_cart(
    orb_data: AOs_base_data | MOs_data,
) -> tuple[AOs_base_data, jax.Array]:
    """(cart_aos, C) for a sphe-pgto orbital: orb = C @ cart_AO_folded (c2s folded into C).
    cart / composite orbitals are returned unchanged.

    Handles both an MO-backed orbital (``MOs_data`` -- ``C`` is the MO coefficient matrix) and a
    BARE dense AO-lambda AGP leaf (the raw AO leaf: the "orbital" IS the AO, so ``C`` is identity at
    the leaf's AO rank). Bare callers use the returned ``aos``/``C`` exactly like the MO ones."""
    aos, C_mo = _orb_aos_and_coeff(orb_data)
    if "sphe" not in type(aos).__name__:
        return aos, C_mo
    aos_cart, T_mol = sphe_leaf_to_cart(aos)
    return aos_cart, C_mo @ jnp.asarray(T_mol)


def orbital_leaves(
    orb_data: AOs_base_data | MOs_data,
) -> list[tuple[AOs_base_data, jax.Array, Literal["gto", "pgto", "pw", "bspline"]]]:
    """Per-AO-leaf decomposition of a geminal orbital for the analytic derivative dispatch.

    Returns a list of ``(leaf_aos, C_leaf, kind)`` -- one entry per AO leaf (a composite's children,
    else the single leaf), in their NATIVE representation (no sphe->cart). ``kind`` in {"gto",
    "pgto", "pw", "bspline"} (molecular GTO / periodic pGTO / plane-wave / B-spline). ``C_leaf`` maps
    that leaf's native AO rank to the orbitals (``(num_orb, leaf.num_ao)``). The composite concatenates
    children along the AO axis (axis 0), so the coefficient columns split by child ``num_ao`` in
    order. Callers that need the analytic pgto kernels cart-convert a "pgto"/"gto" leaf themselves
    (sphe_leaf_to_cart); callers on the forward (compute_AOs) use the native leaf directly.

    Handles an MO-backed orbital (``C_leaf`` = the MO coefficient columns) AND a BARE dense AO-lambda
    AGP leaf (``C_leaf`` = the identity columns selecting that child's AO block -- the orbital IS the
    AO), via the shared ``_orb_aos_and_coeff`` decomposition."""
    aos, C_mo = _orb_aos_and_coeff(orb_data)  # (num_orb, num_ao_total); identity for a bare AGP leaf
    # The leaf split is STRUCTURAL (child types + num_ao = static), so the jnp column slices below
    # trace cleanly inside the jit bridge; eager callers np.asarray the concrete slice themselves.
    children = aos.children if type(aos).__name__ == "AOs_composite_data" else (aos,)
    out = []
    off = 0
    for ch in children:
        n = int(ch.num_ao)
        C_ch = C_mo[:, off : off + n]
        off += n
        tn = type(ch).__name__
        if "pgto" in tn:
            kind = "pgto"
        elif "plane_wave" in tn:
            kind = "pw"
        elif "bspline" in tn:
            kind = "bspline"
        elif "gto" in tn:  # molecular cart / sphe GTO (open boundary)
            kind = "gto"
        else:
            # Defensive guard (currently unreachable): the gto/pgto/pw/bspline substring tests above
            # cover all 10 concrete AO leaf types (incl. the complex-twist variants). Kept for any
            # future AO leaf type.
            raise NotImplementedError(f"orbital_leaves: unsupported AO leaf {tn}")
        out.append((ch, C_ch, kind))
    return out


def pgto_value(aos: AOs_cart_pgto_data | AOs_cart_pgto_complex_data, r: jax.Array) -> jax.Array:
    return pgto_deriv(aos, r, (0, 0, 0))


def pgto_grad(aos: AOs_cart_pgto_data | AOs_cart_pgto_complex_data, r: jax.Array) -> jax.Array:
    """(3, num_ao, N) AO gradient."""
    return pgto_deriv_set(aos, r, GRAD_ORDERS)


def pgto_laplacian(aos: AOs_cart_pgto_data | AOs_cart_pgto_complex_data, r: jax.Array) -> jax.Array:
    return jnp.sum(pgto_deriv_set(aos, r, LAP_ORDERS), axis=0)


def pgto_gradlap(aos: AOs_cart_pgto_data | AOs_cart_pgto_complex_data, r: jax.Array) -> jax.Array:
    """grad(laplacian AO): (3, num_ao, N) -- the 3rd-order term the kinetic force needs."""
    terms = pgto_deriv_set(aos, r, sum(GRADLAP_ORDERS, ()))  # (9, nao, N), row-major (c, term)
    return jnp.sum(terms.reshape((3, 3) + terms.shape[1:]), axis=1)


def pgto_hessian(aos: AOs_cart_pgto_data | AOs_cart_pgto_complex_data, r: jax.Array) -> jax.Array:
    """(3, 3, num_ao, N) symmetric AO Hessian."""
    orders = HESS_ORDERS[0] + (HESS_ORDERS[1][1], HESS_ORDERS[1][2], HESS_ORDERS[2][2])
    at = dict(zip(orders, pgto_deriv_set(aos, r, orders)))
    return jnp.stack([jnp.stack([at[o] for o in row], 0) for row in HESS_ORDERS], 0)


def pgto_grad_gradlap_hessian(
    aos: AOs_cart_pgto_data | AOs_cart_pgto_complex_data, r: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """(grad (3, nao, N), gradlap (3, nao, N), hessian (3, 3, nao, N)) in one primitive sweep
    (the 18 distinct orders of :data:`ao_deriv.GRAD_GRADLAP_HESS_ORDERS`)."""
    return fields_from_deriv_set(pgto_deriv_set(aos, r, GRAD_GRADLAP_HESS_ORDERS), GRAD_GRADLAP_HESS_ORDERS)

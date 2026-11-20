"""SWCT module.

Precision Zones:
    - ``swct``: all functions in this module.

See :mod:`jqmc._precision` for details.
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

# python modules
from logging import getLogger

# JAX
import jax
import numpy as np
import numpy.typing as npt
from jax import jit, vmap
from jax import numpy as jnp

# jQMC modules
from ._precision import get_dtype_jnp, get_dtype_np
from ._setting import (
    SWCT_OMEGA_BATCH_BUDGET_BYTES,
    measured_stream_batch,
)
from .structure import Structure_data, _inv_3x3, _minimum_image_displacement_debug

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)


def _swct_minimum_image(structure_data: Structure_data, diff: jax.Array) -> jax.Array:
    r"""Wigner-Seitz minimum image of electron-nucleus displacements ``r - R``.

    The SWCT weights must be functions on the torus: a nuclear displacement
    moves every image of the nucleus, and the drivers keep walker positions
    UNWRAPPED, so ``omega_{a i}`` has to use the nearest image of ``R_a`` seen
    from ``r_i``. The selected image is the one
    :py:func:`jqmc.structure._minimum_image_displacement` selects (parallelepiped
    round, then the shortest candidate in the static ``_n_max`` box, first
    candidate on exact ties). The candidates are scanned with a running minimum
    so that the ``(..., n_image, 3)`` candidate block is never materialized for
    the ``(B, N_e, N_a)`` driver batches.

    Molecules (``pbc_flag=False``, static) return ``diff`` unchanged.

    Args:
        structure_data: Structure carrying ``pbc_flag``, the cell and ``_n_max``.
        diff: Displacements ``r - R`` with shape ``(..., 3)`` in caller precision.

    Returns:
        jax.Array: Minimum-image displacements, same shape and dtype as ``diff``.
    """
    if not structure_data.pbc_flag:
        return diff
    cell = structure_data._cell_jnp.astype(diff.dtype)
    inv_cell = _inv_3x3(cell)
    frac = diff @ inv_cell
    wrapped = (frac - jnp.round(frac)) @ cell
    n_max = structure_data._n_max
    n_range = jnp.arange(-n_max, n_max + 1)
    na, nb, nc = jnp.meshgrid(n_range, n_range, n_range, indexing="ij")
    n_grid = jnp.stack([na.reshape(-1), nb.reshape(-1), nc.reshape(-1)], axis=-1).astype(cell.dtype)
    translations = n_grid @ cell  # (n_image, 3)

    def _step(carry, translation):
        best, best_d2 = carry
        candidate = wrapped + translation
        d2 = jnp.sum(candidate * candidate, axis=-1)
        better = d2 < best_d2
        return (jnp.where(better[..., None], candidate, best), jnp.where(better, d2, best_d2)), None

    init = (wrapped, jnp.full(wrapped.shape[:-1], jnp.inf, dtype=wrapped.dtype))
    (best, _), _ = jax.lax.scan(_step, init, translations)
    return best


def _swct_minimum_image_debug(structure_data: Structure_data, diff: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """NumPy twin of :func:`_swct_minimum_image` for one ``(3,)`` displacement."""
    if not structure_data.pbc_flag:
        return diff
    cell = structure_data.cell
    return _minimum_image_displacement_debug(diff, cell, np.linalg.inv(cell), structure_data._n_max)


@jit
def evaluate_swct_omega(
    structure_data: Structure_data,
    r_carts: jax.Array,
) -> jax.Array:
    r"""Compute SWCT weights :math:`\omega_{\alpha i}` for each atom/electron pair.

    Args:
        structure_data (Structure_data): Nuclear geometry information.
        r_carts (jax.Array): Electron Cartesian coordinates with shape ``(N_e, 3)`` and ``float64`` dtype.

    Returns:
        jax.Array: Normalized weights with shape ``(N_a, N_e)``, summing to 1 over atoms for each electron.
    """
    dtype_jnp = get_dtype_jnp("swct")
    R_carts = structure_data._positions_cart_jnp  # fp64 storage accessor

    def compute_omega(R_cart: jax.Array, r_cart: jax.Array) -> jax.Array:
        # Reconstruct r - R in caller-supplied precision (fp64 from MCMC walker
        # state) via JAX promotion, then downcast to the swct zone at the use
        # site (Principle 3b -- cast operands to this function's own zone
        # immediately before consuming them as arithmetic operands).
        # Under PBC the nearest image of each nucleus is used (the weights
        # must be lattice periodic); molecules keep the raw difference.
        diff_one = _swct_minimum_image(structure_data, r_cart - R_cart).astype(dtype_jnp)
        diff_all = _swct_minimum_image(structure_data, r_cart - R_carts).astype(dtype_jnp)
        kappa = 1.0 / jnp.linalg.norm(diff_one) ** 4
        kappa_sum = jnp.sum(1.0 / jnp.linalg.norm(diff_all, axis=1) ** 4)
        return kappa / kappa_sum

    vmap_compute_omega = vmap(
        vmap(
            compute_omega,
            in_axes=(None, 0),
        ),
        in_axes=(0, None),
    )

    omega = vmap_compute_omega(R_carts, r_carts)

    return omega


@jit
def evaluate_swct_omega_dot(
    structure_data: Structure_data,
    r_carts: jax.Array,
    field: jax.Array,
) -> jax.Array:
    r"""Contract the SWCT weights with a per-electron field: ``sum_i omega_{a i} f_{i c}``.

    Args:
        structure_data (Structure_data): Nuclear geometry information.
        r_carts (jax.Array): Electron Cartesian coordinates, shape ``(N_e, 3)``.
        field (jax.Array): Per-electron field to contract, shape ``(N_e, C)``
            (the drift / force gradients, ``C = 3``).

    Returns:
        jax.Array: ``(N_a, C)`` -- the SWCT-warped sum.

    The weight matrix is only a contraction operand. Keeping the contraction
    inside one jitted function lets XLA fuse the ``(N_a, N_e)`` intermediate
    instead of materializing one copy per walker. The fused device reduction
    may sum electrons in a different order from a host BLAS contraction;
    :func:`evaluate_swct_omega` remains the independent reference.
    """
    dtype_jnp = get_dtype_jnp("swct")
    R_carts = structure_data._positions_cart_jnp  # fp64 storage accessor
    # Same kappa / normalisation as ``evaluate_swct_omega``, written batched so
    # the weights stay a fused intermediate instead of a materialised output.
    diff = _swct_minimum_image(structure_data, r_carts[:, None, :] - R_carts[None, :, :]).astype(dtype_jnp)  # (N_e, N_a, 3)
    d2 = jnp.sum(diff * diff, axis=-1)  # (N_e, N_a)
    kappa = 1.0 / (d2 * d2)  # (N_e, N_a) == 1 / |r - R|^4
    omega_t = kappa / jnp.sum(kappa, axis=1, keepdims=True)  # (N_e, N_a), normalised over atoms
    # The contraction is written as broadcast-multiply-reduce, NOT as
    # einsum/dot_general: some XLA:GPU versions miscompile the fp64 GEMM
    # consumer of this fused kappa/normalizer producer. The explicit reduction
    # keeps the weights fused and avoids that backend-specific GEMM path.
    return jnp.sum(omega_t[:, :, None] * jnp.asarray(field, dtype=dtype_jnp)[:, None, :], axis=0)


def evaluate_swct_omega_dot_batched(
    structure_data: Structure_data,
    r_carts_batch: jax.Array,
    field_batch: jax.Array,
) -> jax.Array:
    """Batched :func:`evaluate_swct_omega_dot` with the batch axis STREAMED.

    Args:
        structure_data (Structure_data): Nuclear geometry information.
        r_carts_batch (jax.Array): ``(B, N_e, 3)`` electron coordinates.
        field_batch (jax.Array): ``(B, N_e, C)`` per-electron field.

    Returns:
        jax.Array: ``(B, N_a, C)``.

    A plain ``vmap`` over ``B`` materializes ``B`` copies of the
    ``(N_e, N_a)`` kappa block. The batch is therefore streamed against
    ``SWCT_OMEGA_BATCH_BUDGET_BYTES``;
    ``batch >= B`` degenerates to the plain vmap.
    """
    r_carts_batch = jnp.asarray(r_carts_batch)
    field_batch = jnp.asarray(field_batch)
    n_b = int(r_carts_batch.shape[0])
    n_e = int(r_carts_batch.shape[1])
    # Per-batch-element bytes are measured rather than hand-modeled: the
    # streamed element IS ``evaluate_swct_omega_dot`` (a plain
    # ``(structure_data, r_carts, field) -> (N_a, C)`` kernel), so probing it on
    # one-element avals captures the fused kappa/omega workspace exactly. A
    # hand count ``8 * n_e * n_a`` covers only the (N_e, N_a) kappa block; the fused kernel
    # also holds its normaliser, the field contraction and the (N_a, C) output.
    _r_spec = jax.ShapeDtypeStruct((n_e, 3), r_carts_batch.dtype)
    _f_spec = jax.ShapeDtypeStruct(tuple(field_batch.shape[1:]), field_batch.dtype)
    batch = measured_stream_batch(
        evaluate_swct_omega_dot, structure_data, (_r_spec, _f_spec), n_b, SWCT_OMEGA_BATCH_BUDGET_BYTES
    )
    fn = vmap(evaluate_swct_omega_dot, in_axes=(None, 0, 0))
    if batch >= n_b:
        return fn(structure_data, r_carts_batch, field_batch)
    n_pad = (-n_b) % batch
    if n_pad:
        r_carts_batch = jnp.concatenate([r_carts_batch, jnp.broadcast_to(r_carts_batch[:1], (n_pad, *r_carts_batch.shape[1:]))])
        field_batch = jnp.concatenate([field_batch, jnp.broadcast_to(field_batch[:1], (n_pad, *field_batch.shape[1:]))])
    n_ch = (n_b + n_pad) // batch
    rs = r_carts_batch.reshape((n_ch, batch, *r_carts_batch.shape[1:]))
    fs = field_batch.reshape((n_ch, batch, *field_batch.shape[1:]))
    _, ys = jax.lax.scan(lambda carry, xy: (carry, fn(structure_data, xy[0], xy[1])), None, (rs, fs))
    return ys.reshape((n_ch * batch, *ys.shape[2:]))[:n_b]


def _evaluate_swct_omega_debug(
    structure_data: Structure_data,
    r_carts: npt.NDArray[np.float64],
) -> np.ndarray:
    """NumPy fallback for ``evaluate_swct_omega`` used in debug paths.

    Return dtype matches the swct zone (selectable precision); declared as a
    plain ``np.ndarray`` rather than ``npt.NDArray[np.float64]``.
    """
    dtype_np = get_dtype_np("swct")
    R_carts = structure_data._positions_cart_np  # fp64 storage accessor
    omega = np.zeros((len(R_carts), len(r_carts)), dtype=dtype_np)

    for alpha in range(len(R_carts)):
        for i in range(len(r_carts)):
            # Reconstruct r - R in caller-supplied precision (fp64 from MCMC
            # walker state), then downcast to the swct zone at the use site
            # (Principle 3b).
            diff_one = _swct_minimum_image_debug(structure_data, r_carts[i] - R_carts[alpha]).astype(dtype_np)
            kappa = 1.0 / np.linalg.norm(diff_one) ** 4
            kappa_sum = np.sum(
                [
                    1.0
                    / np.linalg.norm(_swct_minimum_image_debug(structure_data, r_carts[i] - R_carts[beta]).astype(dtype_np))
                    ** 4
                    for beta in range(len(R_carts))
                ]
            )
            omega[alpha, i] = kappa / kappa_sum

    return omega


@jit
def evaluate_swct_domega(
    structure_data: Structure_data,
    r_carts: jax.Array,
) -> jax.Array:
    r"""Evaluate :math:`\sum_i \nabla_{r_i} \omega_{\alpha i}` for each atom.

    Args:
        structure_data (Structure_data): Nuclear geometry information.
        r_carts (jax.Array): Electron Cartesian coordinates with shape ``(N_e, 3)`` and ``float64`` dtype.

    Returns:
        jax.Array: Sum of gradients per atom with shape ``(N_a, 3)``.
    """
    # CLOSED FORM (no autodiff -- production must not trace generic AD).
    # ``omega_{alpha,i}``
    # depends only on ``r_i``, so the (N_e, N_e) block of the jacobian is
    # diagonal and ``sum_j nabla_{r_j} omega_{alpha,i} = nabla_{r_i}
    # omega_{alpha,i}``. A ``jacrev`` would materialise the FULL
    # ``(N_a, N_e, N_e, 3)`` jacobian (non-zero fraction exactly 1/N_e) plus
    # the ``kappa_sum`` backward residual: measured on an A30, ``temp = 0.343
    # * nw * N_a^2 * N_e^2 * 24 B`` -- 2.29 TB at N_a = 108, N_e = 432 and 17.9 GB
    # even at nw = 1. For molecules the closed form is 1:1 with the Rust
    # reference ``gfmc_forces::swct_domega`` (``jqmc/rs/src/gfmc_forces.rs``).
    # Native drivers use the Rust SWCT implementation, including periodic images::
    #
    #     kappa_a      = 1 / d_a^4,   d_a = |r_i - R_a|  (minimum image under PBC)
    #     grad kappa_a = -4 (r_i - R_a) / d_a^6
    #     grad omega_a = (grad kappa_a * S - kappa_a * grad S) / S^2,
    #     S = sum_b kappa_b
    #
    # Intermediates are ``(N_e, N_a, 3)`` -- output-sized, amplification ~1.
    dtype_jnp = get_dtype_jnp("swct")
    R_carts = structure_data._positions_cart_jnp  # fp64 storage accessor
    # Reconstruct r - R in caller-supplied precision, downcast at the use site
    # (Principle 3b), exactly as ``evaluate_swct_omega`` does.
    diff = _swct_minimum_image(structure_data, r_carts[:, None, :] - R_carts[None, :, :]).astype(dtype_jnp)  # (N_e, N_a, 3)
    d2 = jnp.sum(diff * diff, axis=-1)  # (N_e, N_a)
    kappa = 1.0 / (d2 * d2)  # (N_e, N_a)
    s_sum = jnp.sum(kappa, axis=1)  # (N_e,)
    grad_kappa = -4.0 * diff * (kappa / d2)[..., None]  # (N_e, N_a, 3)
    grad_s = jnp.sum(grad_kappa, axis=1)  # (N_e, 3)
    grad_omega = (grad_kappa * s_sum[:, None, None] - kappa[..., None] * grad_s[:, None, :]) / (s_sum * s_sum)[:, None, None]
    domega = jnp.sum(grad_omega, axis=0)  # (N_a, 3)

    return domega


def _evaluate_swct_domega_debug(
    structure_data: Structure_data,
    r_carts: npt.NDArray[np.float64],
) -> np.ndarray:
    """NumPy fallback for ``evaluate_swct_domega`` used in debug paths.

    Return dtype matches the swct zone (selectable precision); declared as a
    plain ``np.ndarray`` rather than ``npt.NDArray[np.float64]``.
    """
    dtype_np = get_dtype_np("swct")
    R_carts = structure_data._positions_cart_np  # fp64 storage accessor
    domega = np.zeros((len(R_carts), 3), dtype=dtype_np)

    def compute_omega(
        R_cart: npt.NDArray[np.float64],
        r_cart: npt.NDArray[np.float64],
        R_carts: npt.NDArray[np.float64],
    ) -> np.floating:
        # Reconstruct r - R in caller-supplied precision then downcast to the
        # swct zone at the use site (Principle 3b).
        diff_one = _swct_minimum_image_debug(structure_data, r_cart - R_cart).astype(dtype_np)
        kappa = 1.0 / np.linalg.norm(diff_one) ** 4
        kappa_sum = np.sum(
            [
                1.0 / np.linalg.norm(_swct_minimum_image_debug(structure_data, r_cart - R_carts[beta]).astype(dtype_np)) ** 4
                for beta in range(len(R_carts))
            ]
        )
        omega = kappa / kappa_sum
        return omega

    diff_h = 1.0e-5

    for i, R_cart in enumerate(R_carts):
        for r_cart in r_carts:
            r_cart_p_dx = r_cart.copy()
            r_cart_m_dx = r_cart.copy()
            r_cart_p_dx[0] += diff_h
            r_cart_m_dx[0] -= diff_h
            omega_dx = (compute_omega(R_cart, r_cart_p_dx, R_carts) - compute_omega(R_cart, r_cart_m_dx, R_carts)) / (
                2 * diff_h
            )

            r_cart_p_dy = r_cart.copy()
            r_cart_m_dy = r_cart.copy()
            r_cart_p_dy[1] += diff_h
            r_cart_m_dy[1] -= diff_h
            omega_dy = (compute_omega(R_cart, r_cart_p_dy, R_carts) - compute_omega(R_cart, r_cart_m_dy, R_carts)) / (
                2 * diff_h
            )

            r_cart_p_dz = r_cart.copy()
            r_cart_m_dz = r_cart.copy()
            r_cart_p_dz[2] += diff_h
            r_cart_m_dz[2] -= diff_h
            omega_dz = (compute_omega(R_cart, r_cart_p_dz, R_carts) - compute_omega(R_cart, r_cart_m_dz, R_carts)) / (
                2 * diff_h
            )

            domega[i, 0] += omega_dx
            domega[i, 1] += omega_dy
            domega[i, 2] += omega_dz

    return domega

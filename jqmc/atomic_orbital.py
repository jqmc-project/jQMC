"""Atomic Orbitals module.

Module containing classes and methods related to Atomic Orbitals.

Precision Zones:
    - ``ao_eval``: real GTO, plane-wave and spline forward evaluation.
    - ``ao_bloch_eval``: complex periodic GTO values, including image sums and
      phases. Pinned to fp64 even in mixed mode: rounding individual Bloch AOs
      can amplify through local MO contractions and inverse derivatives.
    - ``ao_grad_lap``: AO gradient and Laplacian (compute_AOs_grad,
      compute_AOs_laplacian). Pinned to fp64 even in mixed mode -- the
      shared kernel must avoid catastrophic cancellation in the
      Laplacian arithmetic (e.g. ``4 Z^2 r^2 - 6 Z`` for s-type AOs).

The fused :func:`compute_AOs_value_grad_lap` API returns ``(val, gx, gy,
gz, lap)`` from a single dispatch -- the heavy block (``exp``, polynomial
chain, ``S_l_m``) is shared across val/grad/lap instead of recomputed
three times. ``val`` follows the leaf value zone (``ao_eval`` or
``ao_bloch_eval``), while grad/lap stay in
``ao_grad_lap``. Use it when value, gradient, and Laplacian are all
needed at the same call site (kinetic-energy estimators, GFMC streaming
advance); otherwise prefer the standalone APIs (:func:`compute_AOs`,
:func:`compute_AOs_grad`, :func:`compute_AOs_laplacian`).

See :mod:`jqmc._precision` for details.
"""

from __future__ import annotations

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
import dataclasses
import weakref
from collections.abc import Callable, Sequence
from functools import cache, partial
from logging import getLogger
from math import comb, factorial
from operator import itemgetter
from types import NotImplementedType
from typing import TYPE_CHECKING, Self

import jax
import jax.numpy as jnp
import jax.scipy as jscipy
import numpy as np
import numpy.typing as npt
import scipy
from flax import struct
from jax import hessian, jacrev, jit, vmap
from jax import typing as jnpt
from numpy import linalg as LA

from ._precision import (
    cast_to_zone,
    get_complex_dtype_jnp,
    get_complex_dtype_np,
    get_dtype_jnp,
    get_dtype_np,
)
from . import _setting
from ._setting import (
    AO_IMAGE_TRUNCATION_ATOL,
    ATOL_TILING_INTEGER_RESIDUE,
    ATOL_TWIST_CHECKPOINT_KPOINT_MATCH,
    COMPOSITE_CROSS_OVERLAP_SPECTRAL_TERM_TOLERANCE,
    atol_consistency,
    get_pgto_lattice_overlap_term_tolerance,
    rtol_consistency,
)
from .structure import (
    Structure_data,
    _inv_3x3,
    _primitive_image_list_indices,
    _primitive_image_list_indices_per_atom,
    _strain_structure_data,
    _wrap_into_cell_if_pbc,
    normalize_tiling_matrix,
)

if TYPE_CHECKING:
    # Imported only for static type-checkers (pyright / mypy). The runtime
    # path uses a lazy ``from .wavefunction import VariationalParameterBlock``
    # inside the methods that need it, to avoid the
    # ``wavefunction <-> atomic_orbital`` circular import.
    from .wavefunction import VariationalParameterBlock

# set logger
logger = getLogger("jqmc").getChild(__name__)

# JAX float64
jax.config.update("jax_enable_x64", True)

# Tolerances for comparing float values
rtol = rtol_consistency
atol = atol_consistency


@cache
def _cart_to_spherical_matrix(l: int) -> np.ndarray:
    """Precomputed cart -> real-spherical transform for angular momentum ``l`` (0-6).

    The matrix has shape ``((l+1)(l+2)/2, 2l+1)`` and satisfies
    ``A_sph = A_cart @ T`` under the normalization used in the codebase. Values
    are deterministic and cached to avoid runtime fitting.
    """
    precomputed: dict[int, np.ndarray] = {
        0: np.array([[1.0]], dtype=np.float64),
        1: np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        2: np.array(
            [
                [0.0, 0.0, -0.5, 0.0, np.sqrt(3) / 2.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, -0.5, 0.0, -np.sqrt(3) / 2.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        3: np.array(
            [
                [0.0, 0.0, 0.0, 0.0, -np.sqrt(6) / 4.0, 0.0, np.sqrt(10) / 4.0],
                [3.0 * np.sqrt(2) / 4.0, 0.0, -np.sqrt(30) / 20.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -3.0 * np.sqrt(5) / 10.0, 0.0, np.sqrt(3) / 2.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -np.sqrt(30) / 20.0, 0.0, -3.0 * np.sqrt(2) / 4.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, np.sqrt(30) / 5.0, 0.0, 0.0],
                [-np.sqrt(10) / 4.0, 0.0, -np.sqrt(6) / 4.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -3.0 * np.sqrt(5) / 10.0, 0.0, -np.sqrt(3) / 2.0, 0.0],
                [0.0, 0.0, np.sqrt(30) / 5.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        4: np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 3.0 / 8.0, 0.0, -np.sqrt(5) / 4.0, 0.0, np.sqrt(35) / 8.0],
                [np.sqrt(5) / 2.0, 0.0, -np.sqrt(35) / 14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -3.0 * np.sqrt(70) / 28.0, 0.0, np.sqrt(10) / 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 3.0 * np.sqrt(105) / 140.0, 0.0, 0.0, 0.0, -3.0 * np.sqrt(3) / 4.0],
                [0.0, 3.0 * np.sqrt(2) / 4.0, 0.0, -3.0 * np.sqrt(14) / 28.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -3.0 * np.sqrt(105) / 35.0, 0.0, 3.0 * np.sqrt(21) / 14.0, 0.0, 0.0],
                [-np.sqrt(5) / 2.0, 0.0, -np.sqrt(35) / 14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -3.0 * np.sqrt(14) / 28.0, 0.0, -3.0 * np.sqrt(2) / 4.0, 0.0],
                [0.0, 0.0, 3.0 * np.sqrt(7) / 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(70) / 7.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 3.0 / 8.0, 0.0, np.sqrt(5) / 4.0, 0.0, np.sqrt(35) / 8.0],
                [0.0, -np.sqrt(10) / 4.0, 0.0, -3.0 * np.sqrt(70) / 28.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -3.0 * np.sqrt(105) / 35.0, 0.0, -3.0 * np.sqrt(21) / 14.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, np.sqrt(70) / 7.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        5: np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(15) / 8.0, 0.0, -np.sqrt(70) / 16.0, 0.0, 3.0 * np.sqrt(14) / 16.0],
                [5.0 * np.sqrt(14) / 16.0, 0.0, -np.sqrt(70) / 16.0, 0.0, np.sqrt(15) / 24.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 5.0 / 8.0, 0.0, -np.sqrt(105) / 12.0, 0.0, np.sqrt(35) / 8.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(35) / 28.0, 0.0, np.sqrt(30) / 24.0, 0.0, -5.0 * np.sqrt(6) / 8.0],
                [0.0, np.sqrt(5) / 2.0, 0.0, -np.sqrt(15) / 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0 * np.sqrt(35) / 14.0, 0.0, np.sqrt(30) / 6.0, 0.0, 0.0],
                [-5.0 * np.sqrt(6) / 8.0, 0.0, -np.sqrt(30) / 24.0, 0.0, np.sqrt(35) / 28.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(105) / 28.0, 0.0, 0.0, 0.0, -3.0 * np.sqrt(3) / 4.0, 0.0],
                [0.0, 0.0, np.sqrt(6) / 2.0, 0.0, -3.0 * np.sqrt(7) / 14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -5.0 * np.sqrt(21) / 21.0, 0.0, np.sqrt(5) / 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(15) / 24.0, 0.0, np.sqrt(70) / 16.0, 0.0, 5.0 * np.sqrt(14) / 16.0],
                [0.0, -np.sqrt(5) / 2.0, 0.0, -np.sqrt(15) / 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0 * np.sqrt(7) / 14.0, 0.0, -np.sqrt(6) / 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, np.sqrt(15) / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(15) / 3.0, 0.0, 0.0, 0.0, 0.0],
                [3.0 * np.sqrt(14) / 16.0, 0.0, np.sqrt(70) / 16.0, 0.0, np.sqrt(15) / 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 5.0 / 8.0, 0.0, np.sqrt(105) / 12.0, 0.0, np.sqrt(35) / 8.0, 0.0],
                [0.0, 0.0, -np.sqrt(30) / 6.0, 0.0, -3.0 * np.sqrt(35) / 14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -5.0 * np.sqrt(21) / 21.0, 0.0, -np.sqrt(5) / 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, np.sqrt(15) / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        6: np.array(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -5.0 / 16.0,
                    0.0,
                    np.sqrt(210) / 32.0,
                    0.0,
                    -3.0 * np.sqrt(7) / 16.0,
                    0.0,
                    np.sqrt(462) / 32.0,
                ],
                [
                    3.0 * np.sqrt(42) / 16.0,
                    0.0,
                    -np.sqrt(693) / 44.0,
                    0.0,
                    np.sqrt(2310) / 176.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    5.0 * np.sqrt(231) / 88.0,
                    0.0,
                    -3.0 * np.sqrt(2310) / 176.0,
                    0.0,
                    3.0 * np.sqrt(14) / 16.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -5.0 * np.sqrt(33) / 176.0,
                    0.0,
                    np.sqrt(770) / 352.0,
                    0.0,
                    5.0 * np.sqrt(231) / 176.0,
                    0.0,
                    -15.0 * np.sqrt(14) / 32.0,
                ],
                [
                    0.0,
                    5.0 * np.sqrt(14) / 16.0,
                    0.0,
                    -3.0 * np.sqrt(2310) / 176.0,
                    0.0,
                    5.0 * np.sqrt(231) / 264.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    15.0 * np.sqrt(33) / 88.0,
                    0.0,
                    -np.sqrt(770) / 22.0,
                    0.0,
                    5.0 * np.sqrt(231) / 88.0,
                    0.0,
                    0.0,
                ],
                [-5.0 * np.sqrt(10) / 8.0, 0.0, 0.0, 0.0, 5.0 * np.sqrt(22) / 88.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    5.0 * np.sqrt(11) / 44.0,
                    0.0,
                    3.0 * np.sqrt(110) / 88.0,
                    0.0,
                    -5.0 * np.sqrt(6) / 8.0,
                    0.0,
                ],
                [0.0, 0.0, 5.0 * np.sqrt(33) / 22.0, 0.0, -np.sqrt(110) / 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0 * np.sqrt(55) / 22.0, 0.0, 5.0 * np.sqrt(22) / 22.0, 0.0, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -5.0 * np.sqrt(33) / 176.0,
                    0.0,
                    -np.sqrt(770) / 352.0,
                    0.0,
                    5.0 * np.sqrt(231) / 176.0,
                    0.0,
                    15.0 * np.sqrt(14) / 32.0,
                ],
                [
                    0.0,
                    -5.0 * np.sqrt(6) / 8.0,
                    0.0,
                    -3.0 * np.sqrt(110) / 88.0,
                    0.0,
                    5.0 * np.sqrt(11) / 44.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0 * np.sqrt(385) / 308.0, 0.0, 0.0, 0.0, -9.0 * np.sqrt(55) / 44.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 3.0 * np.sqrt(110) / 22.0, 0.0, -5.0 * np.sqrt(11) / 22.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0 * np.sqrt(33) / 22.0, 0.0, np.sqrt(770) / 22.0, 0.0, 0.0, 0.0, 0.0],
                [
                    3.0 * np.sqrt(42) / 16.0,
                    0.0,
                    np.sqrt(693) / 44.0,
                    0.0,
                    np.sqrt(2310) / 176.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    5.0 * np.sqrt(231) / 264.0,
                    0.0,
                    3.0 * np.sqrt(2310) / 176.0,
                    0.0,
                    5.0 * np.sqrt(14) / 16.0,
                    0.0,
                ],
                [0.0, 0.0, -5.0 * np.sqrt(33) / 22.0, 0.0, -np.sqrt(110) / 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0 * np.sqrt(11) / 22.0, 0.0, -3.0 * np.sqrt(110) / 22.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, np.sqrt(2310) / 33.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(231) / 11.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -5.0 / 16.0,
                    0.0,
                    -np.sqrt(210) / 32.0,
                    0.0,
                    -3.0 * np.sqrt(7) / 16.0,
                    0.0,
                    -np.sqrt(462) / 32.0,
                ],
                [
                    0.0,
                    3.0 * np.sqrt(14) / 16.0,
                    0.0,
                    3.0 * np.sqrt(2310) / 176.0,
                    0.0,
                    5.0 * np.sqrt(231) / 88.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    15.0 * np.sqrt(33) / 88.0,
                    0.0,
                    np.sqrt(770) / 22.0,
                    0.0,
                    5.0 * np.sqrt(231) / 88.0,
                    0.0,
                    0.0,
                ],
                [0.0, 0.0, 0.0, -5.0 * np.sqrt(22) / 22.0, 0.0, -5.0 * np.sqrt(55) / 22.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0 * np.sqrt(33) / 22.0, 0.0, -np.sqrt(770) / 22.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, np.sqrt(231) / 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    }

    if l not in precomputed:
        raise NotImplementedError("Cartesian conversion is implemented for l up to 6.")

    return precomputed[l]


def _spherical_to_cart_matrix(l: int) -> np.ndarray:
    """Spherical -> Cartesian transform as the transpose of the precomputed table.

    Only ``_cart_to_spherical_matrix`` stores the full analytic values; this helper
    exposes the inverse direction for readability and reuse.
    """
    return _cart_to_spherical_matrix(l).T


@struct.dataclass
class AO_radial_shell_data:
    """Contracted-Gaussian radial basis, stored once per shell.

    Hierarchy: this is the radial level beneath a GTO AO leaf. Every
    m-component (px / py / pz, real spherical harmonics) of one shell shares
    the same radial part, so the Gaussian exponents and contraction
    coefficients live here exactly once -- no per-m duplication. The AO leaf
    nests this dataclass and references it through a *static* per-AO-primitive
    -> shell-primitive map; the standalone :func:`expand_radial_to_ao_prims`
    is the only interface the kernel / optimizer use to read per-AO-primitive
    radial values. This dataclass knows nothing about m-components (loose
    coupling): the angular identity stays on the AO leaf.

    Attributes:
        exponents: Gaussian exponents per shell-primitive (pytree leaf;
            this is the basis-exponent optimisation DOF). dtype float64.
        coefficients: Contraction coefficients per shell-primitive (pytree
            leaf; the basis-coefficient optimisation DOF). dtype float64.
        shell_prim_to_shell: For every shell-primitive, the index of its
            parent shell (``len == num_shell_prim``). Static.
        nucleus_index: Atom index per shell (``len == num_shell``). Static.
        angular_momentum: Angular momentum ``l`` per shell
            (``len == num_shell``). Static.
    """

    exponents: npt.NDArray[np.float64] = struct.field(pytree_node=True, default_factory=lambda: np.array([], dtype=np.float64))
    coefficients: npt.NDArray[np.float64] = struct.field(
        pytree_node=True, default_factory=lambda: np.array([], dtype=np.float64)
    )
    shell_prim_to_shell: tuple[int, ...] = struct.field(pytree_node=False, default_factory=tuple)
    nucleus_index: tuple[int, ...] = struct.field(pytree_node=False, default_factory=tuple)
    angular_momentum: tuple[int, ...] = struct.field(pytree_node=False, default_factory=tuple)

    @property
    def num_shell(self) -> int:
        """Number of shells."""
        return len(self.nucleus_index)

    @property
    def num_shell_prim(self) -> int:
        """Number of shell-primitives (length of the radial leaves)."""
        return len(self.exponents)

    @classmethod
    def from_aos_data(cls, aos_data: AOs_GTO_base_data) -> tuple[AO_radial_shell_data, npt.NDArray[np.int32]]:
        """Build the radial level from an existing per-AO GTO leaf.

        Returns ``(radial, ao_prim_to_shell_prim)``: the de-duplicated radial
        dataclass plus the static per-AO-primitive -> shell-primitive index
        map the AO leaf stores. Thin wrapper over
        :func:`_build_radial_shell_from_full` reading the leaf's full
        per-AO-primitive arrays.
        """
        return _build_radial_shell_from_full(
            exponents=aos_data.exponents,
            coefficients=aos_data.coefficients,
            orbital_indices=aos_data.orbital_indices,
            nucleus_index=aos_data.nucleus_index,
            angular_momentums=aos_data.angular_momentums,
            num_ao_prim=aos_data.num_ao_prim,
            num_ao_kernel=aos_data._num_ao_kernel,
        )


def _build_radial_shell_from_full(
    *,
    exponents: npt.ArrayLike,
    coefficients: npt.ArrayLike,
    orbital_indices: Sequence[int],
    nucleus_index: Sequence[int],
    angular_momentums: Sequence[int],
    num_ao_prim: int,
    num_ao_kernel: int,
) -> tuple[AO_radial_shell_data, npt.NDArray[np.int32]]:
    """De-duplicate FULL per-AO-primitive radial arrays into the shell level.

    Shell identity is ``(nucleus, l, exponents, coefficients)`` per AO (the
    convention used for shell sharing). Returns ``(radial, ao_prim_to_shell_prim)``
    where ``ao_prim_to_shell_prim`` maps every full per-AO-primitive index to
    its shell-primitive index (so ``expand_radial_to_ao_prims`` reconstructs the
    full arrays exactly).
    """
    dtype_np = np.float64
    exps_np = np.asarray(exponents, dtype=dtype_np)
    coefs_np = np.asarray(coefficients, dtype=dtype_np)

    ao_prims: dict[int, list[int]] = {}
    for prim_idx, ao_idx in enumerate(orbital_indices):
        ao_prims.setdefault(int(ao_idx), []).append(prim_idx)

    shells: list[tuple[int, list[int]]] = []  # (representative ao, [ao indices])
    shell_keys: list[tuple] = []
    shell_nuc: list[int] = []
    shell_l: list[int] = []
    for ao_idx in range(num_ao_kernel):
        prims = ao_prims[ao_idx]
        nuc = int(nucleus_index[ao_idx])
        l_val = int(angular_momentums[ao_idx])
        e_tuple = tuple(float(exps_np[p]) for p in prims)
        c_tuple = tuple(float(coefs_np[p]) for p in prims)
        key = (nuc, l_val, len(prims), e_tuple, c_tuple)
        for s_idx, s_key in enumerate(shell_keys):
            if s_key == key:
                shells[s_idx][1].append(ao_idx)
                break
        else:
            shell_keys.append(key)
            shells.append((ao_idx, [ao_idx]))
            shell_nuc.append(nuc)
            shell_l.append(l_val)

    unique_exps: list[float] = []
    unique_coefs: list[float] = []
    shell_prim_to_shell: list[int] = []
    ao_prim_to_shell_prim = np.empty(num_ao_prim, dtype=np.int32)
    for shell_idx, (rep_ao, shell_aos) in enumerate(shells):
        base = len(unique_exps)
        for p in ao_prims[rep_ao]:
            unique_exps.append(float(exps_np[p]))
            unique_coefs.append(float(coefs_np[p]))
            shell_prim_to_shell.append(shell_idx)
        for ao_idx in shell_aos:
            for k, prim_idx in enumerate(ao_prims[ao_idx]):
                ao_prim_to_shell_prim[prim_idx] = base + k

    radial = AO_radial_shell_data(
        exponents=np.asarray(unique_exps, dtype=dtype_np),
        coefficients=np.asarray(unique_coefs, dtype=dtype_np),
        shell_prim_to_shell=tuple(shell_prim_to_shell),
        nucleus_index=tuple(shell_nuc),
        angular_momentum=tuple(shell_l),
    )
    return radial, ao_prim_to_shell_prim


def expand_radial_to_ao_prims(
    radial: AO_radial_shell_data, ao_prim_to_shell_prim: npt.NDArray[np.int32]
) -> tuple[jax.Array, jax.Array]:
    """Per-AO-primitive ``(exponents, coefficients)`` for the AO kernel.

    The single interface the AO kernel uses to read radial values from the
    nested :class:`AO_radial_shell_data` (loose coupling). It is an in-trace
    ``jnp.take``, so it is differentiable: the gradient of any downstream
    quantity flows back to the per-shell radial leaves, giving exactly one
    degree of freedom per shell-primitive (the m-components of a shell sum
    into their shared parameter automatically -- no symmetrisation step).
    """
    # Convert the static index tuple through NumPy: ``jnp.asarray`` on a long
    # Python tuple walks it element by element in Python (thousands of frames
    # per call for image-tiled pgto bases); the NumPy route yields the same
    # dtype and values in one C-level conversion.
    idx = jnp.asarray(
        np.asarray(ao_prim_to_shell_prim) if isinstance(ao_prim_to_shell_prim, (tuple, list)) else ao_prim_to_shell_prim
    )
    return jnp.asarray(radial.exponents)[idx], jnp.asarray(radial.coefficients)[idx]


class AOs_base_data:
    """Interface root for AO data dataclasses.

    DO NOT INSTANTIATE DIRECTLY. This class is the type root that
    concrete leaves (``AOs_cart_gto_data``, ``AOs_sphe_gto_data``,
    ``AOs_cart_pgto_data``, ``AOs_sphe_pgto_data``,
    ``AOs_composite_data``, ``AOs_plane_wave_data``, and
    ``AOs_bspline_data``) implement.

    Use this class for:

    - Type hints: ``def f(aos: AOs_base_data) -> ...``
    - ``isinstance`` checks: ``isinstance(obj, AOs_base_data)``
    - Composition: ``children: tuple[AOs_base_data, ...]``

    Plain class (no ``@struct.dataclass`` decorator) to avoid the
    ABCMeta-vs-flax.struct metaclass conflict. Holds no fields. The
    method stubs below raise ``NotImplementedError``; the ``__add__``
    operator returns an ``AOs_composite_data`` (resolved by name at
    call time, since ``AOs_composite_data`` is defined later in the
    module).
    """

    def _compute_AOs(self, r_carts: jax.Array) -> jax.Array:
        raise NotImplementedError(
            "AOs_base_data is an interface root. Instantiate one of the "
            "concrete leaf classes (AOs_cart_gto_data, AOs_sphe_gto_data, "
            "AOs_cart_pgto_data, AOs_sphe_pgto_data) instead."
        )

    def _compute_AOs_grad(self, r_carts: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        raise NotImplementedError("AOs_base_data is an interface root; call on a concrete leaf.")

    def _compute_AOs_laplacian(self, r_carts: jax.Array) -> jax.Array:
        raise NotImplementedError("AOs_base_data is an interface root; call on a concrete leaf.")

    def _compute_AOs_value_grad_lap(
        self,
        r_carts: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        raise NotImplementedError("AOs_base_data is an interface root; call on a concrete leaf.")

    def _compute_AOs_debug(
        self,
        r_carts: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        raise NotImplementedError("AOs_base_data is an interface root; call on a concrete leaf.")

    def _compute_overlap_matrix(self) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        raise NotImplementedError(f"compute_overlap_matrix: unsupported AOs type {type(self).__name__}.")

    def _overlap_gram(
        self,
        bra: npt.ArrayLike,
        ket: npt.ArrayLike,
        dense_overlap: Callable[["AOs_base_data"], npt.ArrayLike],
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        r"""Return :math:`\overline{B}\,S\,K^{T}` for this leaf's overlap ``S``, in fp64 / complex128.

        ``bra`` / ``ket`` have shape ``(n, num_ao)`` (rows are MOs) and
        ``dense_overlap(leaf)`` returns the leaf's :py:func:`compute_overlap_matrix`
        (the caller may memoize it). Leaves whose overlap has a closed form
        override this so that no ``num_ao x num_ao`` array is formed (issue #247).
        """
        return _gram_with_overlap(bra, dense_overlap(self), ket)

    def sanity_check(self) -> None:
        raise NotImplementedError("AOs_base_data is an interface root; call on a concrete leaf.")

    def _get_info(self) -> list[str]:
        """Default info renderer: just the leaf class name and num_ao.

        GTO leaves override this with a richer per-shell table. The default
        keeps the contract universal so MOs_data._get_info works on every
        leaf type (PW, Bspline, composite, etc.).
        """
        return ["**" + type(self).__name__, f"  num_ao = {getattr(self, 'num_ao', '?')}"]

    # ``num_ao`` and ``structure_data`` are NOT declared as @property stubs
    # on this root because @struct.dataclass uses ``object.__setattr__`` in
    # ``__init__`` (via ``frozen=True``), and ``object.__setattr__`` honours
    # data descriptors on parent classes --- an inherited @property without
    # a setter therefore blocks subclass field assignment. The concrete
    # leaves (AOs_GTO_base_data field, AOs_composite_data property,
    # AOs_plane_wave_data field) provide these names directly via MRO.

    # ---- Variational-parameter protocol ------------------------------------
    # Per-leaf hooks for collecting and applying basis-level variational
    # parameters. The default on this root is "no basis-level variational
    # params" --- the natural answer for PW and Spline leaves. The GTO base
    # (:class:`AOs_GTO_base_data`) overrides these to return / apply the
    # exponent and contraction-coefficient blocks (one degree of freedom per
    # shell primitive). The composite container dispatches to
    # children with a per-child name prefix.

    def get_basis_variational_payloads(
        self,
        *,
        opt_basis_exp: bool = False,
        opt_basis_coeff: bool = False,
        block_name_prefix: str = "",
    ) -> list[tuple[str, np.ndarray]]:
        """Return optimizer-independent ``(name, values)`` basis payloads."""
        return []

    def get_basis_variational_blocks(
        self,
        *,
        state_token: tuple[str, int],
        sync_scope: str,
        opt_basis_exp: bool = False,
        opt_basis_coeff: bool = False,
        block_name_prefix: str = "",
    ) -> list[VariationalParameterBlock]:
        """Return basis-level variational parameter blocks owned by this leaf.

        Default: empty list (the leaf has no basis-level optimisable
        parameters). GTO leaves override with exponent / coefficient
        blocks. PW / Spline leaves inherit the empty default --- their
        basis is determined by ``kinetic_cutoff`` / spline grid, not by
        per-primitive free parameters.

        Args:
            state_token: accepted-wavefunction publication identity.
            sync_scope: optimizer synchronization ownership for these blocks.
            opt_basis_exp: include the exponents block.
            opt_basis_coeff: include the contraction-coefficients block.
            block_name_prefix: prepended to the block ``name`` (e.g.
                ``"lambda_"`` or ``"j3_"`` from the upper layer; further
                prefixed by ``AOs_composite_data`` with ``"childN_"``).
        """
        from .wavefunction import ParameterTangentSpec, VariationalParameterBlock

        return [
            VariationalParameterBlock.build(
                name=name,
                values=values,
                field="real",
                derivative_policy="direct_real",
                sync_scope=sync_scope,
                tangent_spec=ParameterTangentSpec.euclidean(state_token=state_token),
                state_token=state_token,
            )
            for name, values in self.get_basis_variational_payloads(
                opt_basis_exp=opt_basis_exp,
                opt_basis_coeff=opt_basis_coeff,
                block_name_prefix=block_name_prefix,
            )
        ]

    def apply_basis_block_update(self, block: VariationalParameterBlock) -> AOs_base_data:
        """Return a new leaf with ``block`` applied to its basis-level params.

        Default: identity (this leaf does not own basis-level params).
        GTO leaves override to dispatch on the block-name suffix
        (``"basis_exp"`` / ``"basis_coeff"``). Unknown block names are
        ignored (return self) so that an upper-layer block routed to the
        wrong leaf is a no-op rather than a crash --- the composite's
        child-prefix dispatch already routes correctly.
        """
        return self

    def __add__(self, other: AOs_base_data) -> AOs_composite_data | NotImplementedType:
        if not isinstance(other, AOs_base_data):
            return NotImplemented
        left = self.children if isinstance(self, AOs_composite_data) else (self,)
        right = other.children if isinstance(other, AOs_composite_data) else (other,)
        return AOs_composite_data(children=left + right)


@struct.dataclass
class AOs_GTO_base_data(AOs_base_data):
    """Shared concrete fields and properties for all GTO AO leaves.

    Holds the dataclass fields common to Cartesian and real-spherical
    GTOs, both molecular and periodic. Not instantiated directly; always
    via one of the concrete leaves: the molecular ``AOs_cart_gto_data`` /
    ``AOs_sphe_gto_data``, or the periodic ``AOs_cart_pgto_data`` /
    ``AOs_sphe_pgto_data`` (which share the periodic base
    :class:`AOs_pGTO_base_data`).

    The ``_n_image`` field lives here because common GTO validation,
    conversion, and debug kernels obtain their working AO rank through
    ``_num_ao_kernel``. Molecular leaves use the static sentinel value ``1``;
    periodic leaves override the rank calculation for their expanded layout.
    Keeping the field static also gives the two layouts separate JIT caches.
    """

    @property
    def _value_zone(self) -> str:
        """Precision owner for forward values of this Gaussian leaf."""
        return "ao_eval"

    #: Molecular structure and atomic positions used to place AOs.
    structure_data: Structure_data = struct.field(pytree_node=True, default_factory=lambda: Structure_data())
    #: Fractional crystal momentum at which this AO leaf evaluates Bloch states.
    #: ``None`` selects the real / Gamma path (molecular or periodic at
    #: k = Gamma). A non-None length-3 tuple selects a complex Bloch leaf at the given
    #: twist. The down-spin sibling of a Bloch leaf carries the negated tuple
    #: (BCS / TR-pair convention; the up channel evaluates at ``+k`` and the
    #: dn channel at ``-k`` so the geminal pair ``phi^k(r_up) * conj(phi^k)(r_dn)``
    #: forms without spin-asymmetric conj at the kernel level).
    k_point: tuple[float, float, float] | None = struct.field(pytree_node=False, default=None)
    #: AO -> atom index mapping (``len == num_ao``).
    nucleus_index: list[int] | tuple[int, ...] = struct.field(pytree_node=False, default_factory=tuple)
    #: Number of contracted AOs.
    num_ao: int = struct.field(pytree_node=False, default=0)
    #: Number of primitive Gaussians.
    num_ao_prim: int = struct.field(pytree_node=False, default=0)
    #: For each primitive, the parent AO index (``len == num_ao_prim``).
    orbital_indices: list[int] | tuple[int, ...] = struct.field(pytree_node=False, default_factory=tuple)
    #: Radial basis stored once per shell (the nested radial level; its
    #: ``exponents`` / ``coefficients`` are the basis-parameter optimisation
    #: DOF). Per-AO ``exponents`` / ``coefficients`` are reconstructed on the
    #: fly via the :attr:`exponents` / :attr:`coefficients` properties (numpy)
    #: and the ``_exponents_jnp`` / ``_coefficients_jnp`` accessors (jnp,
    #: differentiable). See :class:`AO_radial_shell_data`.
    radial: AO_radial_shell_data = struct.field(pytree_node=True, default_factory=AO_radial_shell_data)
    #: For each full per-AO primitive, the index of its shell-primitive in
    #: ``radial`` (``len == num_ao_prim``). Static; reconstructs the per-AO
    #: layout from the de-duplicated radial leaf.
    ao_prim_to_shell_prim: tuple[int, ...] = struct.field(pytree_node=False, default_factory=tuple)
    #: When the input provides physical shell identity, each AO has a fixed
    #: coefficient scale relative to its common radial contraction. The empty
    #: tuple is the legacy layout, whose gathered coefficients are effective.
    ao_coefficient_scales: tuple[float, ...] = struct.field(pytree_node=False, default_factory=tuple)
    #: Angular momentum quantum numbers ``l`` per AO (``len == num_ao``).
    angular_momentums: list[int] | tuple[int, ...] = struct.field(pytree_node=False, default_factory=tuple)
    #: Number of lattice images used by the periodic AO image expansion.
    #: ``1`` is the molecular instance: the public AO wrappers emit HLO
    #: byte-identical to the pre-PBC AO kernel. ``> 1`` is a periodic
    #: instance produced by :py:func:`_apply_image_expansion`; the pgto
    #: leaves (``AOs_pGTO_base_data``) collapse the kernel-rank output
    #: back to ``(num_ao, N_e)``. Static (``pytree_node=False``) so
    #: molecular and periodic instances JIT-cache separately.
    _n_image: int = struct.field(pytree_node=False, default=1)

    # -- Shared property views ------------------------------------------------

    @property
    def k_frac_np(self) -> npt.NDArray[np.float64]:
        r"""Fractional crystal momentum as a ``(3,)`` float64 array.

        Returns zeros when :py:attr:`k_point` is ``None`` (Gamma / molecular).
        The Bloch image-sum phase reduces to ``exp(2 pi i k_frac . n)`` for
        integer image index ``n``; the per-electron wrap-restore phase uses
        ``exp(2 pi i k_frac . floor(r @ inv_cell))``.
        """
        kp = self.k_point
        if kp is None:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(kp, dtype=np.float64)

    @property
    def _nucleus_index_np(self) -> npt.NDArray[np.int32]:
        """nucleus_index."""
        return np.array(self.nucleus_index, dtype=np.int32)

    @property
    def _nucleus_index_jnp(self) -> jax.Array:
        """nucleus_index."""
        # Via NumPy: identical int32 array without a per-element Python walk.
        return jnp.array(np.asarray(self.nucleus_index, dtype=np.int32), dtype=jnp.int32)

    @property
    def _nucleus_index_prim_np(self) -> npt.NDArray[np.int32]:
        """nucleus_index per primitive."""
        return np.array(self.nucleus_index)[self._orbital_indices_np]

    @property
    def _nucleus_index_prim_jnp(self) -> jax.Array:
        """nucleus_index per primitive."""
        return jnp.array(self._nucleus_index_prim_np, dtype=jnp.int32)

    @property
    def _orbital_indices_np(self) -> npt.NDArray[np.int32]:
        """orbital_indices."""
        return np.array(self.orbital_indices, dtype=np.int32)

    @property
    def _orbital_indices_jnp(self) -> jax.Array:
        """orbital_indices."""
        # Via NumPy: identical int32 array without a per-element Python walk.
        return jnp.array(np.asarray(self.orbital_indices, dtype=np.int32), dtype=jnp.int32)

    @property
    def _prim_groups_by_K(self) -> tuple:
        """Group AOs by contraction depth K (primitives per AO).

        Used by :func:`_reduce_primitives_to_aos` to replace
        ``segment_sum`` (which falls back to a scatter-add while-loop
        in XLA) with a small, fixed number of dense ``reduce_sum`` ops
        -- one per unique contraction depth K.

        Returns:
            tuple of ``(K, ao_idx_np, prim_idx_np, is_identity_perm)``.
        """
        # _num_ao_kernel is the kernel-rank AO count that the orbital_indices
        # array partitions into (= num_ao for gto; = n_image * num_ao for pgto,
        # whose ``num_ao`` is the molecular view).
        return _build_prim_groups_by_K(self.orbital_indices, self._num_ao_kernel)

    @property
    def _expanded_positions_cart_np(self) -> npt.NDArray[np.float64]:
        """Per-center cartesian positions the AO kernel places AOs on.

        For molecular leaves this is exactly ``structure_data`` positions
        (one row per atom). The pgto leaves override this to reconstruct
        the image-tiled centers (``R_mol[I] + T_n``) *in-trace* from the
        molecular ``structure_data`` positions plus the static image
        translations, so the per-image centers stay differentiable w.r.t.
        the single molecular atom positions (image contributions auto-sum
        under AD -> molecular-rank force gradients).
        """
        return self.structure_data._positions_cart_np

    @property
    def _expanded_positions_cart_jnp(self) -> jax.Array:
        """JAX view of :py:attr:`_expanded_positions_cart_np` (see there)."""
        return self.structure_data._positions_cart_jnp

    @property
    def _atomic_center_carts_np(self) -> npt.NDArray[np.float64]:
        """Atomic positions in cartesian (per AO)."""
        return self._expanded_positions_cart_np[self._nucleus_index_np]

    @property
    def _atomic_center_carts_jnp(self) -> jax.Array:
        """Atomic positions in cartesian (per AO)."""
        return self._expanded_positions_cart_jnp[self._nucleus_index_jnp]

    @property
    def _atomic_center_carts_unique_jnp(self) -> jax.Array:
        """Unique atomic positions in cartesian."""
        return self._expanded_positions_cart_jnp

    # ---- Distinct-centre table for the angular / radial evaluation --------
    # The angular block costs ``(num_centres, n_lm, N_e[, 3])``, so it must run
    # on the DISTINCT centres with a per-primitive gather. Molecular leaves are
    # already deduplicated (one table row per atom, ``nucleus_index`` maps
    # AO -> atom), so these are the existing accessors unchanged; the periodic
    # pgto leaf overrides them.

    @property
    def _uq_center_carts_jnp(self) -> jax.Array:
        """Distinct centres the angular table is evaluated at, ``(N_uq, 3)``."""
        return self._atomic_center_carts_unique_jnp

    @property
    def _uq_center_index_prim_jnp(self) -> jax.Array:
        """Row of :py:attr:`_uq_center_carts_jnp` for each primitive."""
        return self._nucleus_index_prim_jnp

    @property
    def _uq_center_index_ao_jnp(self) -> jax.Array:
        """Row of :py:attr:`_uq_center_carts_jnp` for each AO."""
        return self._nucleus_index_jnp

    @property
    def _atomic_center_carts_prim_np(self) -> npt.NDArray[np.float64]:
        """Atomic positions in cartesian for primitve orbitals."""
        return self._atomic_center_carts_np[self.orbital_indices]

    @property
    def _atomic_center_carts_prim_jnp(self) -> jax.Array:
        """Atomic positions in cartesian for primitve orbitals."""
        return self._atomic_center_carts_jnp[self._orbital_indices_jnp]

    @property
    def _angular_momentums_prim_np(self) -> npt.NDArray[np.int32]:
        """Angular momentums for primitive orbitals."""
        return np.array(self.angular_momentums, dtype=np.int32)[self._orbital_indices_np]

    @property
    def _angular_momentums_prim_jnp(self) -> jax.Array:
        """Angular momentums for primitive orbitals."""
        return jnp.array(self._angular_momentums_prim_np, dtype=jnp.int32)

    @property
    def exponents(self) -> npt.NDArray[np.float64]:
        """Full per-AO-primitive Gaussian exponents, expanded from the per-shell radial leaf.

        Numpy view (``len == num_ao_prim``) for non-differentiated readers
        (sanity checks, debug paths, HDF5, shell maps). The differentiable
        kernel path uses :attr:`_exponents_jnp`.
        """
        exps = np.asarray(self.radial.exponents, dtype=np.float64)
        idx = np.asarray(self.ao_prim_to_shell_prim, dtype=np.int64)
        return exps[idx]

    @property
    def coefficients(self) -> npt.NDArray[np.float64]:
        """Full per-AO-primitive contraction coefficients, expanded from the per-shell radial leaf."""
        coefs = np.asarray(self.radial.coefficients, dtype=np.float64)
        idx = np.asarray(self.ao_prim_to_shell_prim, dtype=np.int64)
        if self.ao_coefficient_scales:
            return coefs[idx] * np.asarray(self.ao_coefficient_scales, dtype=np.float64)[self._orbital_indices_np]
        return coefs[idx]

    @property
    def _coefficient_scales_prim_jnp(self) -> jax.Array | float:
        """Fixed derivative of effective primitive coefficients with respect to radial coefficients."""
        if self.ao_coefficient_scales:
            return jnp.asarray(np.asarray(self.ao_coefficient_scales, dtype=np.float64)[self._orbital_indices_np])
        return 1.0

    @property
    def _exponents_jnp(self) -> jax.Array:
        """Full per-AO-primitive exponents (differentiable; grad flows to the per-shell radial leaf)."""
        # Lift-only fp64 basis-data storage accessor (see _precision.py exemption);
        # consumer casts to its own zone at use site.
        e, _ = expand_radial_to_ao_prims(self.radial, self.ao_prim_to_shell_prim)
        return e.astype(jnp.float64)

    @property
    def _coefficients_jnp(self) -> jax.Array:
        """Full per-AO-primitive coefficients (differentiable; grad flows to the per-shell radial leaf)."""
        _, c = expand_radial_to_ao_prims(self.radial, self.ao_prim_to_shell_prim)
        c = c.astype(jnp.float64)
        if self.ao_coefficient_scales:
            return c * self._coefficient_scales_prim_jnp
        return c

    @classmethod
    def _from_full_basis(
        cls,
        *,
        exponents: npt.ArrayLike,
        coefficients: npt.ArrayLike,
        **fields: object,
    ) -> Self:
        """Construct a GTO leaf from FULL per-AO-primitive ``exponents`` / ``coefficients``.

        The single construction entry point for build sites that carry full
        per-AO radial arrays (TREXIO loader, ``build`` factories, image
        tiling). De-duplicates the radial basis to the per-shell level (the
        nested :class:`AO_radial_shell_data` leaf) and records the static
        ``ao_prim_to_shell_prim`` map; ``**fields`` are the remaining
        constructor fields (``orbital_indices``, ``nucleus_index``,
        ``angular_momentums``, ``num_ao``, ``num_ao_prim``, leaf-specific
        angular fields, ...).
        """
        # Kernel-rank AO count = len(nucleus_index): equals n_image * num_ao for
        # uniform (A) tiling and M (= sum_A m_A * k_A) for the ragged (B) layout.
        num_ao_kernel = len(fields["nucleus_index"])
        # An image expansion carries the physical home-cell radial and its
        # structural gather. Kernel centres are deliberately distinct, so
        # rediscovering shells from their nucleus_index would turn every
        # angular component and image into an independent optimizer variable.
        radial = fields.pop("radial", None)
        ao_prim_to_shell_prim = fields.pop("ao_prim_to_shell_prim", None)
        if radial is None:
            if ao_prim_to_shell_prim is not None:
                raise ValueError("A radial primitive map requires its radial leaf.")
            radial, ao_prim_to_shell_prim = _build_radial_shell_from_full(
                exponents=exponents,
                coefficients=coefficients,
                orbital_indices=fields["orbital_indices"],
                nucleus_index=fields["nucleus_index"],
                angular_momentums=fields["angular_momentums"],
                num_ao_prim=int(fields["num_ao_prim"]),
                num_ao_kernel=num_ao_kernel,
            )
        elif ao_prim_to_shell_prim is None:
            raise ValueError("A supplied radial leaf requires its primitive map.")
        return cls(radial=radial, ao_prim_to_shell_prim=tuple(int(x) for x in ao_prim_to_shell_prim), **fields)

    @property
    def _num_orb(self) -> int:
        """Return the number of orbitals (the molecular view)."""
        return self.num_ao

    @property
    def _num_ao_kernel(self) -> int:
        """Kernel-rank AO count (``_n_image * num_ao``).

        For molecular instances (``_n_image == 1``), equals ``num_ao``.
        For periodic instances, equals the tiled rank used by the @jit
        kernels --- the pgto leaves (``AOs_pGTO_base_data``) then
        collapse the kernel output back to ``num_ao`` (the molecular
        view) via the post-kernel reshape-sum.

        Used by internal kernel-rank consumers: ``sanity_check`` length
        validations on per-AO arrays (``nucleus_index``,
        ``angular_momentums``, ``polynominal_order_{x,y,z}``,
        ``magnetic_quantum_numbers``); ``_prim_groups_by_K``; the
        ``_aos_sphe_to_cart`` / ``_aos_cart_to_sphe`` per-AO shell loop;
        the analytic overlap matrix kernel.
        """
        return self._n_image * self.num_ao

    # ---- Variational-parameter protocol override for GTO leaves ------------
    # GTO leaves own ``exponents`` and ``coefficients`` as basis-level
    # variational parameters (one per shell primitive, shared structurally by
    # the shell's m-components). PW / Spline leaves inherit the empty default from
    # ``AOs_base_data``. The upper-layer ``Geminal_data`` /
    # ``Jastrow_data`` / ``wavefunction`` delegate to these methods with
    # an appropriate ``block_name_prefix`` (e.g. ``"lambda_"`` or
    # ``"j3_"``) and slice the up / dn shares before calling
    # ``apply_basis_block_update``.

    def _shell_component_count(self, l: int) -> int:
        """Number of AOs (angular components) in one shell of angular momentum ``l``."""
        raise NotImplementedError

    def _listing_coefficient_scales(self) -> npt.NDArray[np.float64]:
        """Per-AO factor that makes the stored coefficients uniform within a shell."""
        raise NotImplementedError

    def _basis_shell_listing_lines(self) -> list[str]:
        """Return the per-atom basis listing: one entry per contracted shell, in AO order.

        Periodic leaves are first de-tiled to their molecular (home) base, so
        each shell is listed once, on its home atom, in the stored (TREXIO)
        AO order rather than in the bucket-major order of the ragged image
        layout. Consecutive AOs of one atom with the same angular momentum and
        the same (exponent, coefficient) primitives form a shell of at most
        :py:meth:`_shell_component_count` components. Coefficients are the
        stored ones times :py:meth:`_listing_coefficient_scales`, which
        removes the per-component Cartesian factor so that, e.g., the six
        components of a Cartesian d shell print as one shell; for s/p and
        real-spherical shells the stored coefficient is printed unchanged.
        """
        l_map = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h", 6: "i"}
        src = _detile_ragged_to_molecular(self) if isinstance(self, AOs_pGTO_base_data) else self
        exponents = np.asarray(src.exponents, dtype=np.float64)
        coefficients = np.asarray(src.coefficients, dtype=np.float64)
        scales = src._listing_coefficient_scales()
        prim_per_ao: dict[int, list[int]] = {}
        for prim_idx, ao_idx in enumerate(src.orbital_indices):
            prim_per_ao.setdefault(int(ao_idx), []).append(prim_idx)
        ao_per_atom: dict[int, list[int]] = {}
        for ao_idx, atom_idx in enumerate(src.nucleus_index):
            ao_per_atom.setdefault(int(atom_idx), []).append(ao_idx)

        lines: list[str] = []
        for atom_idx in sorted(ao_per_atom):
            symbol = src.structure_data.atomic_labels[atom_idx]
            lines.append("  " + "-" * 36)
            lines.append(f"  **basis set for atom index {atom_idx + 1}: {symbol}**")
            lines.append("  " + "-" * 36)
            shells: list[tuple[int, npt.NDArray[np.float64], npt.NDArray[np.float64], int]] = []
            for ao_idx in ao_per_atom[atom_idx]:
                l = int(src.angular_momentums[ao_idx])
                prims = prim_per_ao.get(ao_idx, [])
                exps = exponents[prims]
                coefs = coefficients[prims] * scales[ao_idx]
                if shells:
                    last_l, last_exps, last_coefs, count = shells[-1]
                    if (
                        last_l == l
                        and count < src._shell_component_count(l)
                        and last_exps.shape == exps.shape
                        and np.allclose(last_exps, exps, rtol=rtol, atol=atol)
                        and np.allclose(last_coefs, coefs, rtol=rtol, atol=atol)
                    ):
                        shells[-1] = (last_l, last_exps, last_coefs, count + 1)
                        continue
                shells.append((l, exps, coefs, 1))
            for l, exps, coefs, _count in shells:
                lines.append(f"  {symbol} {l_map.get(l, 'l > i')}")
                for Z, coef in zip(exps, coefs, strict=True):
                    lines.append(f"    {Z:.6f} {coef:.7f}")
        return lines

    def get_basis_variational_payloads(
        self,
        *,
        opt_basis_exp: bool = False,
        opt_basis_coeff: bool = False,
        block_name_prefix: str = "",
    ) -> list[tuple[str, np.ndarray]]:
        """Return exponent / coefficient payloads for this GTO leaf.

        Block names are ``{block_name_prefix}basis_exp`` and
        ``{block_name_prefix}basis_coeff``. The block exposes the nested
        ``radial`` leaf directly -- one degree of freedom per shell-primitive
        -- so it needs NO shell-sharing symmetriser: the de-duplication is
        structural (a single radial parameter per shell).
        """
        payloads: list[tuple[str, np.ndarray]] = []
        if not (opt_basis_exp or opt_basis_coeff):
            return payloads

        def _payload(values: object) -> np.ndarray:
            # Parameter values are real and published as float64. The same
            # accessor also reads a gradient pytree (``collect_param_grads``);
            # the masked-autograd derivative source of a real parameter is
            # complex (d ln Psi / dc with Im kept), so never cast it to real.
            arr = np.asarray(values)
            return arr if np.iscomplexobj(arr) else arr.astype(np.float64, copy=False)

        if opt_basis_exp:
            payloads.append((f"{block_name_prefix}basis_exp", _payload(self.radial.exponents)))
        if opt_basis_coeff:
            payloads.append((f"{block_name_prefix}basis_coeff", _payload(self.radial.coefficients)))
        return payloads

    def apply_basis_block_update(self, block: VariationalParameterBlock) -> AOs_GTO_base_data:
        """Apply a basis-level block update on THIS GTO leaf.

        Inspects ``block.name`` for the ``"basis_exp"`` / ``"basis_coeff"``
        suffix and ``replace``s the corresponding field with the
        block's values. Unknown suffixes return ``self`` (the upper
        layer's prefix-stripping must have produced one of the known
        suffixes).
        """
        name = block.name
        if name.endswith("basis_exp"):
            field = "exponents"
        elif name.endswith("basis_coeff"):
            field = "coefficients"
        else:
            return self
        return self._with_updated_basis_parameter(field, block.values)

    def _with_updated_basis_parameter(self, field: str, values: object) -> AOs_GTO_base_data:
        """Replace one physical radial array, rebuilding a concrete periodic candidate.

        During AD the static image map is held fixed: infinitesimal radial
        derivatives are those of the certified finite lattice sum. A concrete
        accepted update is validated and re-enumerated at the same tolerance.
        """
        if isinstance(values, jax.core.Tracer):
            return self.replace(radial=self.radial.replace(**{field: values}))
        # Optimizer coordinates remain the physical exponents/coefficients;
        # an invalid additive candidate is rejected, never clipped or accepted
        # as a different basis. Keep fp64 storage in both precision modes.
        values = np.asarray(values)
        current = np.asarray(getattr(self.radial, field))
        if np.iscomplexobj(values) or values.shape != current.shape:
            raise ValueError(f"GTO {field} update must have real shape {current.shape}.")
        values = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(values)) or (field == "exponents" and np.any(values <= 0)):
            raise ValueError(
                f"GTO {field} update must be finite" + (" and strictly positive." if field == "exponents" else ".")
            )
        candidate = self.replace(radial=self.radial.replace(**{field: values}))
        if isinstance(candidate, AOs_pGTO_base_data):
            # Both exponent and coefficient changes alter the certified image
            # envelope. Rebuild before publishing, using the same truncation
            # policy as HDF5 load. The home radial/map retain parameter order
            # across sequential exponent and coefficient blocks.
            home = _detile_ragged_to_molecular(candidate)
            candidate = _apply_image_expansion_complex(
                home, candidate.structure_data, candidate.k_point, tol=candidate._image_truncation_tol
            )
        candidate.sanity_check()
        return candidate

    def _check_radial_domain(self) -> None:
        """Validate the physical Gaussian domain on concrete model data."""
        exponents = np.asarray(self.radial.exponents)
        coefficients = np.asarray(self.radial.coefficients)
        if not np.all(np.isfinite(exponents)) or np.any(exponents <= 0):
            raise ValueError("GTO exponents must be finite and strictly positive.")
        if coefficients.shape != exponents.shape or not np.all(np.isfinite(coefficients)):
            raise ValueError("GTO coefficients must be finite and match the exponent shape.")
        if self.ao_coefficient_scales:
            if len(self.ao_coefficient_scales) != self._num_ao_kernel or not np.all(np.isfinite(self.ao_coefficient_scales)):
                raise ValueError("GTO coefficient scales must contain one finite value per kernel AO.")
            if not np.all(np.isfinite(self.coefficients)):
                raise ValueError("Expanded GTO coefficients must be finite.")


@struct.dataclass
class AOs_cart_gto_data(AOs_GTO_base_data):
    """Atomic orbital definitions in Cartesian form.

    Stores contracted Gaussian basis data used to evaluate atomic orbitals (AOs) on a grid.
    The angular part is represented by Cartesian polynomials (:math:`x^{n_x} y^{n_y} z^{n_z}`) with
    :math:`n_x + n_y + n_z = l` for each AO.

    Attributes:
        structure_data (Structure_data): Molecular structure and atomic positions used to place AOs.
        nucleus_index (list[int] | tuple[int]): AO -> atom index mapping (``len == num_ao``).
        num_ao (int): Number of contracted AOs.
        num_ao_prim (int): Number of primitive Gaussians.
        orbital_indices (list[int] | tuple[int]): For each primitive, the parent AO index (``len == num_ao_prim``).
        exponents (npt.NDArray[np.float64]): Gaussian exponents for primitives (``len == num_ao_prim``). dtype: float64.
        coefficients (npt.NDArray[np.float64]): Contraction coefficients per primitive (``len == num_ao_prim``). dtype: float64.
        angular_momentums (list[int] | tuple[int]): Angular momentum quantum numbers ``l`` per AO (``len == num_ao``).
        polynominal_order_x (list[int] | tuple[int]): Cartesian power ``n_x`` for each AO (``len == num_ao``).
        polynominal_order_y (list[int] | tuple[int]): Cartesian power ``n_y`` for each AO (``len == num_ao``).
        polynominal_order_z (list[int] | tuple[int]): Cartesian power ``n_z`` for each AO (``len == num_ao``).

    Examples:
        Minimal hydrogen dimer (bohr) with all-electron cc-pVTZ (Gaussian format) in Cartesian form::

            from jqmc.structure import Structure_data
            from jqmc.atomic_orbital import AOs_cart_gto_data

            structure = Structure_data(
                positions=[[0.0, 0.0, -0.70], [0.0, 0.0, 0.70]],
                pbc_flag=False,
                atomic_numbers=[1, 1],
                element_symbols=["H", "H"],
                atomic_labels=["H1", "H2"],
            )

            # cc-pVTZ primitives duplicated per Cartesian component; counts:
            # per atom -> 15 AOs, 19 primitives; for two atoms
            num_ao=30; num_ao_prim=38

            exponents = [
                0.3258,
                33.87, 5.095, 1.159, 0.3258, 0.1027,
                0.1027,
                1.407, 1.407, 1.407,
                0.388, 0.388, 0.388,
                1.057, 1.057, 1.057, 1.057, 1.057, 1.057,
                0.3258,
                33.87, 5.095, 1.159, 0.3258, 0.1027,
                0.1027,
                1.407, 1.407, 1.407,
                0.388, 0.388, 0.388,
                1.057, 1.057, 1.057, 1.057, 1.057, 1.057,
            ]

            coefficients = [
                1.0,
                0.006068, 0.045308, 0.202822, 0.503903, 0.383421,
                1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0,
                0.006068, 0.045308, 0.202822, 0.503903, 0.383421,
                1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ]

            orbital_indices = [
                0,                      # S shell (uncontracted)
                1, 1, 1, 1, 1,          # S shell (contracted)
                2,                      # S shell (uncontracted)
                3, 4, 5,                # P shell (uncontracted)
                6, 7, 8,                # P shell (uncontracted)
                9, 10, 11, 12, 13, 14,  # D shell (uncontracted)
                15,                     # S shell (uncontracted)
                16, 16, 16, 16, 16,     # S shell (contracted)
                17,                     # S shell (uncontracted)
                18, 19, 20,             # P shell (uncontracted)
                21, 22, 23,             # P shell (uncontracted)
                24, 25, 26, 27, 28, 29, # D shell (uncontracted)
            ]

            nucleus_index = [
                *([0] * 15),
                *([1] * 15),
            ]

            angular_momentums = [
                0, 0, 0,                # S shells (1 orbital each)
                1, 1, 1, 1, 1, 1,       # two P shells (3 orbitals each)
                2, 2, 2, 2, 2, 2,       # one D shell (6 orbitals each)
                0, 0, 0,                # S shells (1 orbital each)
                1, 1, 1, 1, 1, 1,       # two P shells (3 orbitals each)
                2, 2, 2, 2, 2, 2,       # one D shell (6 orbitals each)
            ]

            polynominal_order_x = [
                0, 0, 0,
                1, 0, 0,
                1, 0, 0,
                2, 1, 1, 0, 0, 0,
                0, 0, 0,
                1, 0, 0,
                1, 0, 0,
                2, 1, 1, 0, 0, 0,
            ]

            polynominal_order_y = [
                0, 0, 0,
                0, 1, 0,
                0, 1, 0,
                0, 1, 0, 2, 1, 0,
                0, 0, 0,
                0, 1, 0,
                0, 1, 0,
                0, 1, 0, 2, 1, 0,
            ]

            polynominal_order_z = [
                0, 0, 0,
                0, 0, 1,
                0, 0, 1,
                0, 0, 1, 0, 1, 2,
                0, 0, 0,
                0, 0, 1,
                0, 0, 1,
                0, 0, 1, 0, 1, 2,
            ]

            aos = AOs_cart_gto_data(
                structure_data=structure,
                nucleus_index=nucleus_index,
                num_ao=num_ao,
                num_ao_prim=num_ao_prim,
                orbital_indices=orbital_indices,
                exponents=exponents,
                coefficients=coefficients,
                angular_momentums=angular_momentums,
                polynominal_order_x=polynominal_order_x,
                polynominal_order_y=polynominal_order_y,
                polynominal_order_z=polynominal_order_z,
            )

            aos.sanity_check()
    """

    #: Cartesian power ``n_x`` for each AO (``len == num_ao``).
    polynominal_order_x: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)
    #: Cartesian power ``n_y`` for each AO (``len == num_ao``).
    polynominal_order_y: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)
    #: Cartesian power ``n_z`` for each AO (``len == num_ao``).
    polynominal_order_z: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)

    def sanity_check(self) -> None:
        """Validate AO array shapes and basic types.

        Ensures per-AO array lengths match the kernel-rank AO count
        (``_num_ao_kernel`` --- equals ``num_ao`` for molecular gto
        leaves, and ``_n_image * num_ao_molecular`` for pgto leaves).
        The pgto leaves' tiled per-AO arrays therefore validate against
        the kernel rank, not the externally visible ``num_ao`` (which is
        the molecular view).

        Raises:
            ValueError: When any of the following holds:
                - ``len(nucleus_index) != _num_ao_kernel``
                - ``len(unique(orbital_indices)) != _num_ao_kernel``
                - ``len(exponents) != num_ao_prim`` or ``len(coefficients) != num_ao_prim``
                - ``len(angular_momentums) != _num_ao_kernel``
                - ``len(polynominal_order_{x,y,z}) != _num_ao_kernel``
                - any attribute has an unexpected Python type
        """
        self._check_radial_domain()
        ao_dim = self._num_ao_kernel
        if len(self.nucleus_index) != ao_dim:
            raise ValueError(
                f"dim. of self.nucleus_index ({len(self.nucleus_index)}) does not match _num_ao_kernel ({ao_dim})."
            )
        if len(np.unique(self.orbital_indices)) != ao_dim:
            raise ValueError(
                f"unique orbital_indices count ({len(np.unique(self.orbital_indices))}) does not "
                f"match _num_ao_kernel ({ao_dim}). num_ao={self.num_ao}, num_ao_prim={self.num_ao_prim}."
            )
        if len(self.exponents) != self.num_ao_prim:
            raise ValueError("dim. of self.exponents is wrong")
        if len(self.coefficients) != self.num_ao_prim:
            raise ValueError("dim. of self.coefficients is wrong")
        if len(self.angular_momentums) != ao_dim:
            raise ValueError(
                f"dim. of self.angular_momentums ({len(self.angular_momentums)}) does not match _num_ao_kernel ({ao_dim})."
            )
        if len(self.polynominal_order_x) != ao_dim:
            raise ValueError(
                f"dim. of self.polynominal_order_x ({len(self.polynominal_order_x)}) does not match _num_ao_kernel ({ao_dim})."
            )
        if len(self.polynominal_order_y) != ao_dim:
            raise ValueError(
                f"dim. of self.polynominal_order_y ({len(self.polynominal_order_y)}) does not match _num_ao_kernel ({ao_dim})."
            )
        if len(self.polynominal_order_z) != ao_dim:
            raise ValueError(
                f"dim. of self.polynominal_order_z ({len(self.polynominal_order_z)}) does not match _num_ao_kernel ({ao_dim})."
            )

        # Post-initialization method to check the types of the attributes.
        # tuple is very slow in practice!! So, we must use list for production runs!!

        if not isinstance(self.nucleus_index, (tuple, list)):
            raise ValueError(f"nucleus_index = {type(self.nucleus_index)} must be a list or tuple.")
        if not isinstance(self.num_ao, (int, np.integer)):
            raise ValueError(f"num_ao = {type(self.num_ao)} must be an int.")
        if not isinstance(self.num_ao_prim, (int, np.integer)):
            raise ValueError(f"num_ao_prim = {type(self.num_ao_prim)} must be an int.")
        if not isinstance(self.orbital_indices, (tuple, list)):
            raise ValueError(f"orbital_indices = {type(self.orbital_indices)} must be a list or tuple.")
        if not isinstance(self.exponents, np.ndarray):
            raise ValueError(f"exponents = {type(self.exponents)} must be an np.ndarray (float64).")
        if not isinstance(self.coefficients, np.ndarray):
            raise ValueError(f"coefficients = {type(self.coefficients)} must be an np.ndarray (float64).")
        if not isinstance(self.angular_momentums, (tuple, list)):
            raise ValueError(f"angular_momentums = {type(self.angular_momentums)} must be a list or tuple.")
        if not isinstance(self.polynominal_order_x, (tuple, list)):
            raise ValueError(f"polynominal_order_x = {type(self.polynominal_order_x)} must be a list or tuple.")
        if not isinstance(self.polynominal_order_y, (tuple, list)):
            raise ValueError(f"polynominal_order_y = {type(self.polynominal_order_y)} must be a list or tuple.")
        if not isinstance(self.polynominal_order_z, (tuple, list)):
            raise ValueError(f"polynominal_order_z = {type(self.polynominal_order_z)} must be a list or tuple.")

        self.structure_data.sanity_check()

    @classmethod
    def build(
        cls,
        structure_data: Structure_data,
        nucleus_index: list[int] | tuple[int, ...],
        num_ao: int,
        num_ao_prim: int,
        orbital_indices: list[int] | tuple[int, ...],
        exponents: npt.NDArray[np.float64],
        coefficients: npt.NDArray[np.float64],
        angular_momentums: list[int] | tuple[int, ...],
        polynominal_order_x: list[int] | tuple[int, ...],
        polynominal_order_y: list[int] | tuple[int, ...],
        polynominal_order_z: list[int] | tuple[int, ...],
    ) -> AOs_cart_gto_data:
        """Build a validated :class:`AOs_cart_gto_data` (molecular Cartesian GTO).

        The blessed factory: it coerces ``exponents`` / ``coefficients`` to
        float64 ndarrays, constructs the instance, and runs
        :py:meth:`sanity_check`. Arguments mirror the dataclass fields.
        Direct construction (``AOs_cart_gto_data(...)``) remains supported.
        """
        data = cls._from_full_basis(
            structure_data=structure_data,
            nucleus_index=nucleus_index,
            num_ao=num_ao,
            num_ao_prim=num_ao_prim,
            orbital_indices=orbital_indices,
            exponents=np.asarray(exponents, dtype=np.float64),
            coefficients=np.asarray(coefficients, dtype=np.float64),
            angular_momentums=angular_momentums,
            polynominal_order_x=polynominal_order_x,
            polynominal_order_y=polynominal_order_y,
            polynominal_order_z=polynominal_order_z,
        )
        data.sanity_check()
        return data

    def _build_uncontracted_aos(self) -> AOs_cart_gto_data:
        """Return the uncontracted AO dataset: each distinct primitive once.

        Every distinct primitive ``(nucleus, l, (nx, ny, nz), exponent)`` of
        the contracted basis becomes one AO with coefficient 1, in order of
        first appearance. A primitive shared by several contracted shells
        (general contraction, e.g. Dunning cc-pVXZ) is emitted once.

        For periodic (pgto) leaves only the home-cell primitives are
        scanned (the first ``num_ao_prim / _n_image`` entries -- pgto
        storage tiles per-primitive arrays image-major, so the home
        cell sits at the start), and the result is re-tiled via
        :py:func:`_apply_image_expansion_complex` so the returned leaf
        keeps the pgto storage convention (``num_ao`` = home count,
        ``_n_image`` / ``_image_truncation_tol`` / ``k_point`` carried
        forward, complex Bloch sibling selected when ``k_point`` is
        non-None). Without this branch the inherited
        ``type(self)(...)`` falls through to the gto kwargs only,
        dropping the pgto-only fields to their dataclass defaults
        (``_n_image=1``, ``_image_truncation_tol=0.0``) -- a downstream
        ``Hamiltonian_data.load_from_hdf5`` then feeds the zero tol to
        ``_primitive_image_radius``, which computes
        ``log_inv_tol = -log(0) = +inf`` and crashes with
        ``OverflowError``.
        """
        is_pgto = isinstance(self, AOs_pGTO_base_data)
        # Periodic (per-shell ragged) leaves: recover the molecular base and
        # uncontract THAT (the kernel-AO arrays are bucket-major, not a uniform
        # home-first tile). Molecular leaves uncontract themselves.
        src = _detile_ragged_to_molecular(self) if is_pgto else self

        seen = set()
        new_nucleus_index = []
        new_angular_momentums = []
        new_orbital_indices = []
        new_exponents = []
        new_coefficients = []
        new_polynominal_order_x = []
        new_polynominal_order_y = []
        new_polynominal_order_z = []

        # Basis parameters have fp64 storage independently of the evaluation
        # policy (issue #167). Deduplicate only exact stored primitives: two
        # exponents that coincide during fp32 evaluation still represent distinct
        # basis parameters and must survive construction in mixed precision.
        # The coefficient is the constant 1, exact in fp64.
        store_dtype = np.float64

        for orb, raw_exp in zip(src.orbital_indices, src.exponents, strict=True):
            # One uncontracted AO per physical primitive. A general-contraction
            # basis lists the same primitive (nucleus, l, component, exponent)
            # in several contracted shells; keying on the contracted AO index
            # would emit a bit-identical copy per shell (issue #112). The
            # exponents are compared exactly in fp64: a shared primitive is the
            # same stored value in every shell.
            exp = store_dtype(raw_exp)
            key = (
                int(src.nucleus_index[orb]),
                int(src.angular_momentums[orb]),
                int(src.polynominal_order_x[orb]),
                int(src.polynominal_order_y[orb]),
                int(src.polynominal_order_z[orb]),
                float(exp),
            )
            if key in seen:
                continue
            seen.add(key)

            new_idx = len(new_nucleus_index)
            new_nucleus_index.append(src.nucleus_index[orb])
            new_angular_momentums.append(src.angular_momentums[orb])
            new_orbital_indices.append(new_idx)
            new_exponents.append(exp)
            new_coefficients.append(1.0)
            new_polynominal_order_x.append(src.polynominal_order_x[orb])
            new_polynominal_order_y.append(src.polynominal_order_y[orb])
            new_polynominal_order_z.append(src.polynominal_order_z[orb])

        home_gto = AOs_cart_gto_data._from_full_basis(
            structure_data=src.structure_data,
            nucleus_index=new_nucleus_index,
            num_ao=len(new_nucleus_index),
            num_ao_prim=len(new_orbital_indices),
            orbital_indices=new_orbital_indices,
            exponents=np.array(new_exponents, dtype=store_dtype),
            coefficients=np.array(new_coefficients, dtype=store_dtype),
            angular_momentums=new_angular_momentums,
            polynominal_order_x=new_polynominal_order_x,
            polynominal_order_y=new_polynominal_order_y,
            polynominal_order_z=new_polynominal_order_z,
        )
        if not is_pgto:
            return home_gto
        # Re-tile to the same pgto family (real or complex Bloch) as ``self``.
        # ``_apply_image_expansion_complex`` falls back to the real expansion
        # when ``k_point is None``, so the same call handles both paths.
        return _apply_image_expansion_complex(
            home_gto, self.structure_data, k_point=self.k_point, tol=float(self._image_truncation_tol)
        )

    def _shell_component_count(self, l: int) -> int:
        """Number of Cartesian components ``(l + 1)(l + 2) / 2`` of one shell."""
        return (l + 1) * (l + 2) // 2

    def _listing_coefficient_scales(self) -> npt.NDArray[np.float64]:
        """``sqrt(f(nx, ny, nz) / f(l, 0, 0))`` with ``f = nx! ny! nz! / ((2nx)! (2ny)! (2nz)!)``.

        The stored Cartesian coefficient carries ``1 / sqrt(f(nx, ny, nz))``
        (the evaluator multiplies it back), so this factor maps every
        component of a shell onto the coefficient of its ``x^l`` component.
        """
        nx = np.asarray(self.polynominal_order_x, dtype=np.int64)
        ny = np.asarray(self.polynominal_order_y, dtype=np.int64)
        nz = np.asarray(self.polynominal_order_z, dtype=np.int64)
        ang = np.asarray(self.angular_momentums, dtype=np.int64)
        fact = scipy.special.factorial

        def component_factor(a, b, c):
            return fact(a) * fact(b) * fact(c) / (fact(2 * a) * fact(2 * b) * fact(2 * c))

        return np.sqrt(component_factor(nx, ny, nz) / component_factor(ang, 0 * ang, 0 * ang))

    def _get_info(self) -> list[str]:
        """Return a list of strings containing information about the class attributes."""
        info_lines = []
        info_lines.append(f"**{self.__class__.__name__}**")
        info_lines.append(f"  Number of AOs = {self.num_ao}")
        info_lines.append(f"  Number of primitive AOs = {self.num_ao_prim}")
        info_lines.append("  Angular part is the polynomial (Cartesian) function.")

        info_lines.extend(self._basis_shell_listing_lines())
        return info_lines

    def _logger_info(self) -> None:
        """Output the information from get_info using logger.info."""
        for line in self._get_info():
            logger.info(line)

    @property
    def _polynominal_order_x_prim_np(self) -> npt.NDArray[np.int32]:
        """Polynominal order of x for primitive orbitals.

        Returns Polynominal order of x for primitive orbitals

        Returns:
            jax.Array: Polynominal order of x for primitive orbitals
        """
        return np.array(self.polynominal_order_x, dtype=np.int32)[self._orbital_indices_np]

    @property
    def _polynominal_order_x_prim_jnp(self) -> jax.Array:
        """Polynominal order of x for primitive orbitals.

        Returns Polynominal order of x for primitive orbitals

        Returns:
            jax.Array: Polynominal order of x for primitive orbitals
        """
        return jnp.array(self._polynominal_order_x_prim_np, dtype=np.int32)

    @property
    def _polynominal_order_y_prim_np(self) -> npt.NDArray[np.int32]:
        """Polynominal order of y for primitive orbitals.

        Returns Polynominal order of y for primitive orbitals

        Returns:
            jax.Array: Polynominal order of y for primitive orbitals
        """
        return np.array(self.polynominal_order_y, dtype=np.int32)[self._orbital_indices_np]

    @property
    def _polynominal_order_y_prim_jnp(self) -> jax.Array:
        """Polynominal order of y for primitive orbitals.

        Returns Polynominal order of y for primitive orbitals

        Returns:
            jax.Array: Polynominal order of y for primitive orbitals
        """
        return jnp.array(self._polynominal_order_y_prim_np, dtype=np.int32)

    @property
    def _polynominal_order_z_prim_np(self) -> npt.NDArray[np.int32]:
        """Polynominal order of z for primitive orbitals.

        Returns Polynominal order of z for primitive orbitals

        Returns:
            jax.Array: Polynominal order of z for primitive orbitals
        """
        return np.array(self.polynominal_order_z, dtype=np.int32)[self._orbital_indices_np]

    @property
    def _polynominal_order_z_prim_jnp(self) -> jax.Array:
        """Polynominal order of z for primitive orbitals.

        Returns Polynominal order of z for primitive orbitals

        Returns:
            jax.Array: Polynominal order of z for primitive orbitals
        """
        return jnp.array(self._polynominal_order_z_prim_np, dtype=np.int32)

    @property
    def _normalization_factorial_ratio_prim_jnp(self) -> jax.Array:
        """Return factorial ratio used in AO normalization (primitive-wise)."""
        # Lift-only fp64 basis-data storage accessor (see _precision.py exemption);
        # consumer casts to its own zone at use site.
        dtype_jnp = jnp.float64
        # prod_i n_i! / (2 n_i)! per primitive in exact Python integers, then one
        # correctly rounded division. scipy.special.factorial(exact=True) on an
        # integer array returns int64 when every argument is <= 20, so the
        # product (2nx)! (2ny)! (2nz)! wrapped around int64 for components such
        # as (7,6,0) or (5,5,5) whenever no power in the leaf exceeded 10
        # (issue #148).
        dtype_np = np.float64
        ratio = np.asarray(
            [
                (factorial(nx) * factorial(ny) * factorial(nz)) / (factorial(2 * nx) * factorial(2 * ny) * factorial(2 * nz))
                for nx, ny, nz in zip(
                    self._polynominal_order_x_prim_np.tolist(),
                    self._polynominal_order_y_prim_np.tolist(),
                    self._polynominal_order_z_prim_np.tolist(),
                )
            ],
            dtype=dtype_np,
        )
        return jnp.array(ratio, dtype=dtype_jnp)

    # ---- Per-leaf AO evaluation methods ------------------------------------
    # Thin wrappers around the JIT'd kernels. These molecular (gto) leaves
    # emit kernel-rank AOs directly; the periodic (pgto) leaves override
    # these methods (via ``AOs_pGTO_base_data``) to collapse the
    # lattice-image axis, so the image fold is hidden entirely inside the
    # pgto compute path and never appears in this shared molecular code.
    # Composites (``AOs_composite_data``) dispatch directly via
    # ``c._compute_AOs(r_carts)``.

    @jit
    def _compute_AOs(self, r_carts: jax.Array) -> jax.Array:
        r"""AO values at electron coordinates. Shape ``(num_ao, N_e)``.

        Implementation note (perf):
            The angular polynomial part :math:`x^{n_x} y^{n_y} z^{n_z}` is
            identical for all primitives belonging to the same contracted
            AO (the angular quantum numbers are an AO property, not a
            primitive property). By the distributive law

            .. math::
               \sum_{p \in \mathrm{AO}} N_p R_p \cdot P
                     = P \cdot \sum_{p \in \mathrm{AO}} N_p R_p ,

            we can apply :func:`_reduce_primitives_to_aos` to the radial
            product (``N R``) FIRST, then multiply by the AO-level
            polynomial. This (1) shrinks the materialised pre-reduction
            buffer from ``(num_ao_prim, n_elec)`` to ``(num_ao, n_elec)``,
            and (2) runs the static-unrolled :func:`_int_pow_unrolled_cart`
            loops at AO rank rather than primitive rank.
        """
        # Periodicity guarantor: wrap electron coordinates into the home cell
        # Omega before evaluating the truncated lattice-image sum. The AO image
        # list is enumerated relative to Omega, so an unwrapped r_cart outside
        # Omega would fall on the wrong side of the truncation and break the
        # sum. Wrapping here makes the "electrons in Omega" precondition an
        # internal guarantee instead of an implicit caller contract.
        # Twist/Bloch signpost: at the Gamma point the removed lattice vector
        # T_0 = floor(frac) @ cell carries phase e^{-i k.T_0} = 1. For k != 0
        # the compensating phase e^{+i k.T_0} must be reintroduced here, using
        # the SAME T_0 that the wrap removes.
        r_carts = _wrap_into_cell_if_pbc(self.structure_data, r_carts)
        dtype_jnp = get_dtype_jnp(self._value_zone)
        # Reconstruct r-R in caller-supplied precision (fp64 from MCMC walker state)
        # via JAX promotion when one operand is fp64, then downcast to the ao_eval
        # zone (Principle 3b -- local cast at point of arithmetic). r_carts is
        # forwarded as-is (Principle 3a) and R_carts is read from the fp64 storage
        # accessor on the basis-data dataclass.
        R_carts = self._atomic_center_carts_prim_jnp
        r_R_diffs = (r_carts[None, :, :] - R_carts[:, None, :]).astype(dtype_jnp)
        c_jnp = self._coefficients_jnp.astype(dtype_jnp)
        Z_jnp = self._exponents_jnp.astype(dtype_jnp)
        l_jnp = self._angular_momentums_prim_jnp

        # The primitive normalization N = sqrt((2Z/pi)^(3/2) (8Z)^l prod n_i!/(2n_i)!)
        # does not depend on the electron coordinates. Form it from the fp64
        # storage values (exponents, factorial ratio) and round it ONCE to the
        # ao_eval zone, as the fused value/grad/lap kernel and the molecular
        # spherical prefactor (``_sphe_primitive_prefactors``) do. Assembling it
        # from zone-dtype pieces overflowed (8Z)^l or underflowed the factorial
        # ratio in float32 although N itself and the AO value are representable
        # (issue #111). Full precision is unchanged (the cast is a no-op there).
        Z_storage = self._exponents_jnp
        N_n_dup_fuctorial_part = self._normalization_factorial_ratio_prim_jnp
        # Static-unrolled (8 Z)**l avoids the XLA repeated-squaring while-loop
        # emitted by ``base ** l`` when ``l`` is a traced int array. The unroll
        # bound for the normalization power is the largest TOTAL degree
        # l = nx + ny + nz (``L_NORM``), not the largest single coordinate power
        # ``L_MAX``: the two coincide only for complete Cartesian shells, and an
        # incomplete shell (e.g. a lone d_xy, l = 2, L_MAX = 1) would otherwise
        # truncate (8 Z)**l to (8 Z)**1 (issue #110). ``L_MAX`` still bounds the
        # polynomial powers below.
        L_MAX = _cart_max_polynomial_order(self)
        L_NORM = _cart_max_angular_momentum(self)
        N_n_dup_Z_part = (2.0 * Z_storage / jnp.pi) ** (3.0 / 2.0) * _int_pow_unrolled_cart(8.0 * Z_storage, l_jnp, L_NORM)
        r_squared = jnp.sum(r_R_diffs**2, axis=-1)
        if self.structure_data.pbc_flag:
            # Periodic leaves keep the whole electron-independent radial factor
            # -- N, the contraction coefficient c and the exponent Z -- at
            # storage precision, as the periodic spherical kernel does (issue
            # #145). Only r^2 and exp(.) of the once-rounded exponent -Z r^2
            # are zone arithmetic, and the primitive product N (c exp(-Z r^2))
            # is rounded ONCE to ao_eval. Rounding N, c and Z to float32
            # separately leaves a COHERENT (per-primitive, image-independent)
            # error in every value, e.g. fl32(Z) r^2 shifts exp(-Z r^2) by
            # Z r^2 eps(Z), which grows over the lattice-image range (issue
            # #718). The AO value stays in the ao_eval zone. Full precision is
            # bit-identical (every cast is a no-op and the association
            # N (c exp(-Z r^2)) is unchanged). The molecular path keeps its
            # zone-dtype factors below.
            N_n_storage = jnp.sqrt(N_n_dup_Z_part * N_n_dup_fuctorial_part)
            exponent = (-Z_storage[:, None] * r_squared).astype(dtype_jnp)
            R_n_dup = self._coefficients_jnp[:, None] * jnp.exp(exponent)
            NR_dup = (N_n_storage[:, None] * R_n_dup).astype(dtype_jnp)  # (num_ao_prim, n_elec)
        else:
            N_n_dup = jnp.sqrt(N_n_dup_Z_part * N_n_dup_fuctorial_part).astype(dtype_jnp)
            R_n_dup = c_jnp[:, None] * jnp.exp(-Z_jnp[:, None] * r_squared)
            NR_dup = N_n_dup[:, None] * R_n_dup  # (num_ao_prim, n_elec)

        # Radial part at primitive level -> reduce to AO level (see docstring).
        NR_ao = _reduce_primitives_to_aos(NR_dup, self)  # (num_ao, n_elec)

        # AO-level coordinates: each AO sits on exactly one atom, so we use the
        # AO->atom mapping (``_atomic_center_carts_jnp``, length num_ao) rather than
        # the prim->atom mapping (``_atomic_center_carts_prim_jnp``, length num_ao_prim).
        R_carts_ao = self._atomic_center_carts_jnp

        # ---- AO-rank polynomial: backend-dependent dispatch -----------
        # CPU (x86 + AVX-512 in particular): the `_int_pow_unrolled_cart`
        # `jnp.where(e == k, p, out)` chain blocks LLVM's loop vectoriser
        # and forces XLA to emit scalar fp arithmetic for the surrounding
        # loop. Permuting AOs into `(nx, ny, nz)`-bucket contiguous order
        # at trace time and expanding the per-bucket polynomial as a
        # chain of multiplies (via :func:`_bucket_pow`) replaces every
        # `where` with a static integer literal exponent, lets LLVM
        # vectorise the inner electron loop, and gives a ~3x speedup
        # measured on the water-cluster Hamiltonian on Sapphire Rapids.
        #
        # GPU / TPU: the SIMT backend handles the `jnp.where` chain
        # efficiently via predication, and the bucket variant's
        # `jnp.concatenate` + trailing gather introduce synchronisation
        # points that cost ~10% per-step, so non-CPU backends use the
        # `jnp.where` path.
        #
        # The backend check is a python-level conditional evaluated once
        # per JIT compile (`jax.default_backend()` is fixed for the
        # process), so no `lax.cond` enters the HLO -- each backend
        # compiles exactly one of the two branches. The CPU branch is
        # entirely local to this function.
        if jax.default_backend() == "cpu":
            # Build the `(nx, ny, nz)` bucket structure at trace time.
            # `self.polynominal_order_*` are static python lists, so
            # this runs concretely (no JAX tracers) and the resulting
            # python ints become compile-time constants in the unrolled
            # bucket loop below.
            nx_np = np.asarray(self.polynominal_order_x, dtype=np.int32)
            ny_np = np.asarray(self.polynominal_order_y, dtype=np.int32)
            nz_np = np.asarray(self.polynominal_order_z, dtype=np.int32)
            n_ao = int(len(nx_np))
            order = sorted(
                range(n_ao),
                key=lambda i: (int(nx_np[i]), int(ny_np[i]), int(nz_np[i])),
            )
            perm = np.asarray(order, dtype=np.int32)
            inv_perm = np.empty(n_ao, dtype=np.int32)
            for i, p in enumerate(perm):
                inv_perm[p] = i
            bucket_off = []  # list of (start, end, (nx, ny, nz))
            s = 0
            while s < n_ao:
                key = (int(nx_np[perm[s]]), int(ny_np[perm[s]]), int(nz_np[perm[s]]))
                e = s
                while (
                    e < n_ao
                    and (
                        int(nx_np[perm[e]]),
                        int(ny_np[perm[e]]),
                        int(nz_np[perm[e]]),
                    )
                    == key
                ):
                    e += 1
                bucket_off.append((int(s), int(e), key))
                s = e
            R_ao_perm = R_carts_ao[jnp.asarray(perm)].astype(dtype_jnp)
            r_R_diffs_ao_perm = (r_carts[None, :, :] - R_ao_perm[:, None, :]).astype(dtype_jnp)
            poly_pieces = []
            for start, end, (nx_v, ny_v, nz_v) in bucket_off:
                sub = r_R_diffs_ao_perm[start:end]
                x = sub[..., 0]
                y = sub[..., 1]
                z = sub[..., 2]
                poly_pieces.append(_bucket_pow(x, nx_v) * _bucket_pow(y, ny_v) * _bucket_pow(z, nz_v))
            P_perm = jnp.concatenate(poly_pieces, axis=0)
            P_perm = P_perm[jnp.asarray(inv_perm)]
            AOs = NR_ao * P_perm
            if get_dtype_jnp(self._value_zone) == jnp.float32:
                # A negligible tail may have zero radial weight and an overflowed
                # polynomial. Preserve every finite product, including signed zero,
                # and the radial (basis-coefficient) tangent. Select on NR_ao rather
                # than on isfinite(AOs): reading the product splits the fused AO
                # kernel on GPU.
                finite_poly = jnp.where(jnp.isfinite(P_perm), P_perm, jnp.zeros_like(P_perm))
                AOs = jnp.where(NR_ao == 0, NR_ao * finite_poly, AOs)
            return AOs

        # ---- Default (GPU / TPU / other): original where-chain path ----
        r_R_diffs_ao = (r_carts[None, :, :] - R_carts_ao[:, None, :]).astype(dtype_jnp)
        x_ao, y_ao, z_ao = r_R_diffs_ao[..., 0], r_R_diffs_ao[..., 1], r_R_diffs_ao[..., 2]
        nx_ao = jnp.asarray(self.polynominal_order_x, dtype=jnp.int32)
        ny_ao = jnp.asarray(self.polynominal_order_y, dtype=jnp.int32)
        nz_ao = jnp.asarray(self.polynominal_order_z, dtype=jnp.int32)
        P_l_nx_ny_nz_ao = (
            _int_pow_unrolled_cart(x_ao, nx_ao, L_MAX)
            * _int_pow_unrolled_cart(y_ao, ny_ao, L_MAX)
            * _int_pow_unrolled_cart(z_ao, nz_ao, L_MAX)
        )
        AOs = NR_ao * P_l_nx_ny_nz_ao
        if get_dtype_jnp(self._value_zone) == jnp.float32:
            # A negligible tail may have zero radial weight and an overflowed
            # polynomial. Preserve every finite product, including signed zero,
            # and the radial (basis-coefficient) tangent. Select on NR_ao rather
            # than on isfinite(AOs): reading the product splits the fused AO
            # kernel on GPU.
            finite_poly = jnp.where(jnp.isfinite(P_l_nx_ny_nz_ao), P_l_nx_ny_nz_ao, jnp.zeros_like(P_l_nx_ny_nz_ao))
            AOs = jnp.where(NR_ao == 0, NR_ao * finite_poly, AOs)
        return AOs

    def _compute_AOs_grad(self, r_carts: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Analytic AO gradients. Each shape ``(num_ao, N_e)``."""
        # Periodicity guarantor: wrap into the home cell (see compute_AOs).
        r_carts = _wrap_into_cell_if_pbc(self.structure_data, r_carts)
        return _compute_AOs_grad_analytic_cart(self, r_carts)

    def _compute_AOs_laplacian(self, r_carts: jax.Array) -> jax.Array:
        """Analytic AO Laplacian. Shape ``(num_ao, N_e)``."""
        # Periodicity guarantor: wrap into the home cell (see compute_AOs).
        r_carts = _wrap_into_cell_if_pbc(self.structure_data, r_carts)
        return _compute_AOs_laplacian_analytic_cart(self, r_carts)

    def _compute_AOs_value_grad_lap(self, r_carts: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Fused AO value/grad/laplacian. Each shape ``(num_ao, N_e)``."""
        # Periodicity guarantor: wrap into the home cell (see compute_AOs).
        r_carts = _wrap_into_cell_if_pbc(self.structure_data, r_carts)
        return _compute_AOs_value_grad_lap_cart(self, r_carts)

    def _compute_AOs_debug(self, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Plain-numpy oracle for :py:func:`compute_AOs` (matches its shape).

        Routes to the kernel-level Cartesian oracle
        (:py:func:`_compute_AOs_cart_debug`). The periodic (pgto) leaves
        override this to fold the lattice-image axis; this molecular form
        returns the kernel result directly (``num_ao == kernel rank``).
        """
        return _compute_AOs_cart_debug(self, r_carts)

    def _compute_overlap_matrix(self) -> npt.NDArray[np.float64]:
        """Analytic Cartesian GTO overlap matrix. Shape ``(num_ao, num_ao)``.

        This method owns the molecular path. Periodic pGTO subclasses
        override it with the independently truncated analytic one-lattice
        sum; they do not fold their AO-evaluation image table here.
        """
        overlap_matrix = _compute_overlap_matrix_cart_analytic(self)
        if self._n_image > 1:
            # Periodic: fold the kernel-AO column axis (length M) onto the
            # molecular ket via the per-shell ragged fold.
            overlap_matrix = np.asarray(self._fold_image_axis_np(np.asarray(overlap_matrix).T)).T  # (M_c, M_c)
        return 0.5 * (overlap_matrix + overlap_matrix.T)


@struct.dataclass
class AOs_sphe_gto_data(AOs_GTO_base_data):
    r"""Atomic orbital definitions in real spherical-harmonic form.

    Stores contracted Gaussian basis data for atomic orbitals (AOs) whose angular part is
    a real spherical harmonic :math:`Y_{l}^{m}` with :math:`m \in \{-l,\dots,+l\}`.

    Attributes:
        structure_data (Structure_data): Molecular structure and atomic positions used to place AOs.
        nucleus_index (list[int] | tuple[int]): AO -> atom index mapping (``len == num_ao``).
        num_ao (int): Number of contracted AOs.
        num_ao_prim (int): Number of primitive Gaussians.
        orbital_indices (list[int] | tuple[int]): For each primitive, the parent AO index (``len == num_ao_prim``).
        exponents (npt.NDArray[np.float64]): Gaussian exponents for primitives (``len == num_ao_prim``). dtype: float64.
        coefficients (npt.NDArray[np.float64]): Contraction coefficients per primitive (``len == num_ao_prim``). dtype: float64.
        angular_momentums (list[int] | tuple[int]): Angular momentum quantum numbers ``l`` per AO (``len == num_ao``).
        magnetic_quantum_numbers (list[int] | tuple[int]): Magnetic quantum numbers ``m`` per AO (``len == num_ao``),
            satisfying ``-l <= m <= l``.

    Examples:
        Hydrogen dimer (bohr) with all-electron cc-pVTZ (Gaussian format), real spherical harmonics::

            from jqmc.structure import Structure_data
            from jqmc.atomic_orbital import AOs_sphe_gto_data

            structure = Structure_data(
                positions=[[0.0, 0.0, -0.70], [0.0, 0.0, 0.70]],
                pbc_flag=False,
                atomic_numbers=[1, 1],
                element_symbols=["H", "H"],
                atomic_labels=["H1", "H2"],
            )

            # Per atom: 14 AOs (3 S, 2xP shells -> 6, 1xD shell -> 5); 18 primitives.
            # Two atoms -> num_ao=28, num_ao_prim=36.
            exponents = [
                # atom 1
                0.3258,
                33.87, 5.095, 1.159, 0.3258, 0.1027,
                0.1027,
                1.407, 1.407, 1.407,
                0.388, 0.388, 0.388,
                1.057, 1.057, 1.057, 1.057, 1.057,
                # atom 2 (same order)
                0.3258,
                33.87, 5.095, 1.159, 0.3258, 0.1027,
                0.1027,
                1.407, 1.407, 1.407,
                0.388, 0.388, 0.388,
                1.057, 1.057, 1.057, 1.057, 1.057,
            ]

            coefficients = [
                # atom 1
                1.0,
                0.006068, 0.045308, 0.202822, 0.503903, 0.383421,
                1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0,
                # atom 2 (same order)
                1.0,
                0.006068, 0.045308, 0.202822, 0.503903, 0.383421,
                1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0,
            ]

            orbital_indices = [
                # atom 1 AOs 0-13
                0,
                1, 1, 1, 1, 1,
                2,
                3, 4, 5,
                6, 7, 8,
                9, 10, 11, 12, 13,
                # atom 2 AOs 14-27
                14,
                15, 15, 15, 15, 15,
                16,
                17, 18, 19,
                20, 21, 22,
                23, 24, 25, 26, 27,
            ]

            nucleus_index = [
                *([0] * 14),
                *([1] * 14),
            ]

            angular_momentums = [
                # atom 1
                0, 0, 0,                # S shells
                1, 1, 1, 1, 1, 1,       # two P shells (3 each)
                2, 2, 2, 2, 2,          # one D shell (5)
                # atom 2
                0, 0, 0,
                1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2,
            ]

            magnetic_quantum_numbers = [
                # atom 1 (S: m=0; P: -1,0,1; D: -2..2)
                0, 0, 0,
                -1, 0, 1,
                -1, 0, 1,
                -2, -1, 0, 1, 2,
                # atom 2
                0, 0, 0,
                -1, 0, 1,
                -1, 0, 1,
                -2, -1, 0, 1, 2,
            ]

            aos = AOs_sphe_gto_data(
                structure_data=structure,
                nucleus_index=nucleus_index,
                num_ao=len(angular_momentums),
                num_ao_prim=len(exponents),
                orbital_indices=orbital_indices,
                exponents=exponents,
                coefficients=coefficients,
                angular_momentums=angular_momentums,
                magnetic_quantum_numbers=magnetic_quantum_numbers,
            )

            aos.sanity_check()
    """

    #: Magnetic quantum numbers ``m`` per AO (``len == num_ao``; ``-l <= m <= l``).
    magnetic_quantum_numbers: list[int] | tuple[int] = struct.field(pytree_node=False, default_factory=tuple)

    def sanity_check(self) -> None:
        """Validate AO array shapes and basic types.

        Ensures per-AO array lengths match ``_num_ao_kernel`` (= ``num_ao``
        for molecular gto leaves, ``_n_image * num_ao_molecular`` for pgto
        leaves). See :py:meth:`AOs_cart_gto_data.sanity_check` for the
        full rationale.

        Raises:
            ValueError: When any of the following holds:
                - ``len(nucleus_index) != _num_ao_kernel``
                - ``len(unique(orbital_indices)) != _num_ao_kernel``
                - ``len(exponents) != num_ao_prim`` or ``len(coefficients) != num_ao_prim``
                - ``len(angular_momentums) != _num_ao_kernel``
                - ``len(magnetic_quantum_numbers) != _num_ao_kernel``
                - any attribute has an unexpected Python type
                - any ``angular_momentums`` entry exceeds ``l = 6`` (the XLA
                  solid-harmonic tables cover l <= 6 only)
        """
        self._check_radial_domain()
        ao_dim = self._num_ao_kernel
        if len(self.nucleus_index) != ao_dim:
            raise ValueError(
                f"dim. of self.nucleus_index ({len(self.nucleus_index)}) does not match _num_ao_kernel ({ao_dim})."
            )
        if len(np.unique(self.orbital_indices)) != ao_dim:
            raise ValueError(
                f"unique orbital_indices count ({len(np.unique(self.orbital_indices))}) does not "
                f"match _num_ao_kernel ({ao_dim}). num_ao={self.num_ao}, num_ao_prim={self.num_ao_prim}."
            )
        if len(self.exponents) != self.num_ao_prim:
            raise ValueError("dim. of self.exponents is wrong")
        if len(self.coefficients) != self.num_ao_prim:
            raise ValueError("dim. of self.coefficients is wrong")
        if len(self.angular_momentums) != ao_dim:
            raise ValueError(
                f"dim. of self.angular_momentums ({len(self.angular_momentums)}) does not match _num_ao_kernel ({ao_dim})."
            )
        if len(self.magnetic_quantum_numbers) != ao_dim:
            raise ValueError("dim. of self.magnetic_quantum_numbers is wrong")

        if not isinstance(self.nucleus_index, (list, tuple)):
            raise ValueError(f"nucleus_index = {type(self.nucleus_index)} must be a list or tuple.")
        if not isinstance(self.num_ao, (int, np.integer)):
            raise ValueError(f"num_ao = {type(self.num_ao)} must be an int.")
        if not isinstance(self.num_ao_prim, (int, np.integer)):
            raise ValueError(f"num_ao_prim = {type(self.num_ao_prim)} must be an int.")
        if not isinstance(self.orbital_indices, (list, tuple)):
            raise ValueError(f"orbital_indices = {type(self.orbital_indices)} must be a list or tuple.")
        if not isinstance(self.exponents, np.ndarray):
            raise ValueError(f"exponents = {type(self.exponents)} must be an np.ndarray (float64).")
        if not isinstance(self.coefficients, np.ndarray):
            raise ValueError(f"coefficients = {type(self.coefficients)} must be an np.ndarray (float64).")
        if not isinstance(self.angular_momentums, (list, tuple)):
            raise ValueError(f"angular_momentums = {type(self.angular_momentums)} must be a list or tuple.")
        if not isinstance(self.magnetic_quantum_numbers, (list, tuple)):
            raise ValueError(f"magnetic_quantum_numbers = {type(self.magnetic_quantum_numbers)} must be a list or tuple.")
        if len(self.angular_momentums) > 0 and max(self.angular_momentums) > _SPHE_XLA_L_MAX:
            raise ValueError(
                f"spherical GTO angular momentum l = {max(self.angular_momentums)} is not supported: "
                f"the XLA solid-harmonic kernels are tabulated for l <= {_SPHE_XLA_L_MAX} only."
            )

        self.structure_data.sanity_check()

    @classmethod
    def build(
        cls,
        structure_data: Structure_data,
        nucleus_index: list[int] | tuple[int, ...],
        num_ao: int,
        num_ao_prim: int,
        orbital_indices: list[int] | tuple[int, ...],
        exponents: npt.NDArray[np.float64],
        coefficients: npt.NDArray[np.float64],
        angular_momentums: list[int] | tuple[int, ...],
        magnetic_quantum_numbers: list[int] | tuple[int, ...],
    ) -> AOs_sphe_gto_data:
        """Build a validated :class:`AOs_sphe_gto_data` (molecular spherical GTO).

        The blessed factory: it coerces ``exponents`` / ``coefficients`` to
        float64 ndarrays, constructs the instance, and runs
        :py:meth:`sanity_check`. Arguments mirror the dataclass fields.
        Direct construction (``AOs_sphe_gto_data(...)``) remains supported.
        """
        data = cls._from_full_basis(
            structure_data=structure_data,
            nucleus_index=nucleus_index,
            num_ao=num_ao,
            num_ao_prim=num_ao_prim,
            orbital_indices=orbital_indices,
            exponents=np.asarray(exponents, dtype=np.float64),
            coefficients=np.asarray(coefficients, dtype=np.float64),
            angular_momentums=angular_momentums,
            magnetic_quantum_numbers=magnetic_quantum_numbers,
        )
        data.sanity_check()
        return data

    def _build_uncontracted_aos(self) -> AOs_sphe_gto_data:
        """Return the uncontracted AO dataset: each distinct primitive once.

        Every distinct primitive ``(nucleus, l, m, exponent)`` becomes one AO
        with coefficient 1, in order of first appearance. Mirror of :meth:`AOs_cart_gto_data._build_uncontracted_aos`: for
        periodic (pgto) leaves only the home-cell primitives are scanned
        and the result is re-tiled via
        :py:func:`_apply_image_expansion_complex` so the returned leaf
        keeps the pgto storage convention (``num_ao`` = home count,
        ``_n_image`` / ``_image_truncation_tol`` / ``k_point`` carried
        forward). See the cart docstring for why the pgto branch is
        needed.
        """
        is_pgto = isinstance(self, AOs_pGTO_base_data)
        # Periodic (per-shell ragged) leaves: uncontract the recovered molecular
        # base (bucket-major kernel arrays have no uniform home-first tile).
        src = _detile_ragged_to_molecular(self) if is_pgto else self

        seen = set()
        new_nucleus_index = []
        new_angular_momentums = []
        new_orbital_indices = []
        new_exponents = []
        new_coefficients = []
        new_magnetic_quantum_numbers = []

        # Match the Cartesian twin: storage and exact duplicate removal use
        # fp64, independent of the active ao_eval evaluation policy (issue #167).
        store_dtype = np.float64

        for orb, raw_exp in zip(src.orbital_indices, src.exponents, strict=True):
            # One uncontracted AO per physical primitive (nucleus, l, m,
            # exponent); see the cart twin for why the contracted AO index
            # must not be part of the key (issue #112).
            exp = store_dtype(raw_exp)
            key = (
                int(src.nucleus_index[orb]),
                int(src.angular_momentums[orb]),
                int(src.magnetic_quantum_numbers[orb]),
                float(exp),
            )
            if key in seen:
                continue
            seen.add(key)

            new_idx = len(new_nucleus_index)
            new_nucleus_index.append(src.nucleus_index[orb])
            new_angular_momentums.append(src.angular_momentums[orb])
            new_orbital_indices.append(new_idx)
            new_exponents.append(exp)
            new_coefficients.append(1.0)
            new_magnetic_quantum_numbers.append(src.magnetic_quantum_numbers[orb])

        home_gto = AOs_sphe_gto_data._from_full_basis(
            structure_data=src.structure_data,
            nucleus_index=new_nucleus_index,
            num_ao=len(new_nucleus_index),
            num_ao_prim=len(new_orbital_indices),
            orbital_indices=new_orbital_indices,
            exponents=np.array(new_exponents, dtype=store_dtype),
            coefficients=np.array(new_coefficients, dtype=store_dtype),
            angular_momentums=new_angular_momentums,
            magnetic_quantum_numbers=new_magnetic_quantum_numbers,
        )
        if not is_pgto:
            return home_gto
        return _apply_image_expansion_complex(
            home_gto, self.structure_data, k_point=self.k_point, tol=float(self._image_truncation_tol)
        )

    def _shell_component_count(self, l: int) -> int:
        """Number of real-spherical components ``2 l + 1`` of one shell."""
        return 2 * l + 1

    def _listing_coefficient_scales(self) -> npt.NDArray[np.float64]:
        """Real-spherical coefficients are already uniform within a shell."""
        return np.ones(len(self.angular_momentums), dtype=np.float64)

    def _get_info(self) -> list[str]:
        """Return a list of strings containing information about the class attributes."""
        info_lines = []
        info_lines.extend(["**" + self.__class__.__name__])
        info_lines.extend([f"  Number of AOs = {self.num_ao}"])
        info_lines.extend([f"  Number of primitive AOs = {self.num_ao_prim}"])
        info_lines.extend(["  Angular part is the real spherical (solid) Harmonics."])

        info_lines.extend(self._basis_shell_listing_lines())
        return info_lines

    def _logger_info(self) -> None:
        """Output the information from get_info using logger.info."""
        for line in self._get_info():
            logger.info(line)

    @property
    def _magnetic_quantum_numbers_prim_np(self) -> npt.NDArray[np.int32]:
        """Magnetic quantum numbers for primitive orbitals."""
        return np.array(self.magnetic_quantum_numbers, dtype=np.int32)[self._orbital_indices_np]

    @property
    def _magnetic_quantum_numbers_prim_jnp(self) -> jax.Array:
        """Magnetic quantum numbers for primitive orbitals."""
        return jnp.array(self._magnetic_quantum_numbers_prim_np, dtype=jnp.int32)

    # ---- Per-leaf AO evaluation methods ------------------------------------
    # Sphe-side mirror of the cart methods above; see those for rationale.

    @jit
    def _compute_AOs(self, r_carts: jax.Array) -> jax.Array:
        """AO values at electron coordinates. Shape ``(num_ao, N_e)``."""
        # Keep the molecular mixed geometry accurate before AO-zone arithmetic.
        # The shared shell layout and polynomial harmonics offset that precision
        # cost; full precision, periodic leaves and GPU keep their existing path.
        if (
            get_dtype_jnp(self._value_zone) == jnp.float32
            and jax.default_backend() == "cpu"
            and not self.structure_data.pbc_flag
        ):
            return _compute_AOs_sphe_molecular_guarded(self, r_carts)
        # Periodicity guarantor: wrap into the home cell (see cart counterpart).
        r_carts = _wrap_into_cell_if_pbc(self.structure_data, r_carts)
        dtype_jnp = get_dtype_jnp(self._value_zone)
        # Reconstruct r-R in caller-supplied precision (fp64 from MCMC walker state)
        # via JAX promotion when one operand is fp64, then downcast to the ao_eval
        # zone (Principle 3b -- local cast at point of arithmetic). r_carts is
        # forwarded as-is (Principle 3a) and R_carts is read from the fp64 storage
        # accessor on the basis-data dataclass.
        R_carts_unique = self._uq_center_carts_jnp
        r_R_diffs_uq = (r_carts[None, :, :] - R_carts_unique[:, None, :]).astype(dtype_jnp)
        l_jnp = self._angular_momentums_prim_jnp

        # The radial normalization
        # N_n = sqrt(2^(2l+3) (l+1)! (2Z)^(l+3/2) / ((2l+2)! sqrt(pi)))
        # does not depend on the electron coordinates. Form it from the fp64
        # storage values, as the Cartesian kernel and the molecular spherical
        # prefactor (``_sphe_primitive_prefactors``) do; it enters the
        # ao_eval zone only through the once-rounded primitive product below.
        # Assembling it from zone-dtype pieces overflowed 2^(2l+3) (2Z)^(l+3/2)
        # or the factorials in float32 and returned inf although N_n and the
        # AO value are representable (issue #111).
        Z_storage = self._exponents_jnp
        l_storage = l_jnp.astype(Z_storage.dtype)
        factorial_l_plus_1 = jnp.exp(jscipy.special.gammaln(l_storage + 2.0))
        factorial_2l_plus_2 = jnp.exp(jscipy.special.gammaln(2.0 * l_storage + 3.0))

        N_n_dup = jnp.sqrt(
            (2.0 ** (2 * l_storage + 3) * factorial_l_plus_1 * (2 * Z_storage) ** (l_storage + 1.5))
            / (factorial_2l_plus_2 * jnp.sqrt(jnp.asarray(jnp.pi, dtype=Z_storage.dtype)))
        )
        N_l_m_dup = jnp.sqrt((2 * l_storage + 1) / (4 * jnp.asarray(jnp.pi, dtype=Z_storage.dtype)))
        # The whole electron-independent radial factor -- N_n N_lm, the
        # contraction coefficient c and the exponent Z -- stays at storage
        # precision, exactly as in the fp64 grad/lap and fused value/grad/lap
        # kernels. Only the coordinate-dependent pieces are zone arithmetic:
        # r^2 and exp(.) of the once-rounded exponent -Z r^2. The primitive
        # product N_n N_lm (c exp(-Z r^2)) is then rounded ONCE to ao_eval.
        # Rounding N_n, N_lm, c and Z to float32 separately leaves a COHERENT
        # (per-primitive, image-independent) error in every value: fl32(Z) r^2
        # alone shifts exp(-Z r^2) by Z r^2 * eps(Z), which grows over the
        # periodic image range. The fast-update kinetic energy pairs G^-1 of
        # these values with the fp64 grad/lap, so the coherent value bias
        # appeared as a systematic e_L shift in mixed precision (issue #145).
        # Full precision is bit-identical (every cast is a no-op there and the
        # association N_n N_lm (c exp(-Z r^2)) is unchanged).
        N_dup = N_n_dup * N_l_m_dup
        c_storage = self._coefficients_jnp

        # Radial distances are shared by every primitive on the same atom-image:
        # ``R_carts(prim) == R_carts_unique[nucleus_index_prim]`` (e.g. 38232
        # prim-images over only 972 unique centers for a 36-atom periodic cell,
        # ~39x). On CPU (memory-bound)
        # reduce r^2 at the unique-center level -- reusing ``r_R_diffs_uq`` already
        # built for the angular ``S_l_m`` below -- and gather to primitive layout,
        # avoiding the ``(num_ao_prim, n_elec, 3)`` per-primitive r-R tensor and
        # its 39x-redundant square. Deduplication only (no reassociation), so the
        # result is bit-identical to ``sum((r-R_prim)**2)``. This is gated to CPU:
        # on GPU the prim-axis gather breaks XLA's fusion of the broadcast-subtract
        # into the radial ``exp`` (measured +20% GPU Net), so GPU keeps the fused
        # per-primitive path. (Same cpu/gpu split rationale as the cart branch
        # above.)
        # Radial part at primitive level -> reduce to AO level. Same distributive-law
        # rationale as in the cart counterpart: the angular factor
        # :math:`Y_{lm}` is constant across primitives of a given AO, so we can
        # reduce ``N R`` first and then multiply by the AO-level ``S_{lm}``.
        #
        # On CPU (memory-bound) build NR from the UNIQUE-center r-R diffs reduced to
        # kernel-rank AO (e.g. 38232 prim-images over only 972 unique centers, ~39x)
        # -- the dedup restructure reuses ``r_R_diffs_uq`` already built for the
        # angular ``S_l_m`` below and is bit-identical to ``sum((r-R_prim)**2)``. On
        # GPU the prim-axis gather breaks XLA's fusion of the broadcast-subtract into
        # the radial ``exp`` (measured +20% GPU Net), so GPU keeps the fused per-prim
        # path. (Same cpu/gpu split rationale as the cart branch above.)
        if jax.default_backend() == "cpu":
            r_squared_uq = jnp.sum(r_R_diffs_uq**2, axis=-1)  # (n_unique, n_elec)
            r_squared = r_squared_uq[self._uq_center_index_prim_jnp]  # (num_ao_prim, n_elec)
            exponent = (-Z_storage[:, None] * r_squared).astype(dtype_jnp)
            R_n_dup = c_storage[:, None] * jnp.exp(exponent)
            NR_dup = (N_dup[:, None] * R_n_dup).astype(dtype_jnp)  # (num_ao_prim, n_elec)
            NR_ao = _reduce_primitives_to_aos(NR_dup, self)  # (num_ao_kernel, n_elec)
        else:
            R_carts = self._atomic_center_carts_prim_jnp
            r_R_diffs = (r_carts[None, :, :] - R_carts[:, None, :]).astype(dtype_jnp)
            r_squared = jnp.sum(r_R_diffs**2, axis=-1)
            exponent = (-Z_storage[:, None] * r_squared).astype(dtype_jnp)
            R_n_dup = c_storage[:, None] * jnp.exp(exponent)
            NR_dup = (N_dup[:, None] * R_n_dup).astype(dtype_jnp)
            NR_ao = _reduce_primitives_to_aos(NR_dup, self)

        # Solid harmonics tabulated at unique-atom level (49, n_atoms_unique, n_elec),
        # then gathered at AO level (not at primitive level via
        # ``nucleus_index_prim_jnp`` / ``l_jnp`` / ``m_jnp`` of length num_ao_prim).
        # Truncate to (l_max+1)**2: harmonics above the basis max-l are never
        # gathered below, so computing the full l=0..6 set (49) is pure waste.
        n_lm = (int(self._angular_momentums_prim_np.max()) + 1) ** 2
        max_ml, S_l_m_dup_all_l_m = _compute_S_l_m(r_R_diffs_uq, n_lm=n_lm)
        S_l_m_dup_all_l_m_reshaped = S_l_m_dup_all_l_m.reshape(
            (S_l_m_dup_all_l_m.shape[0] * S_l_m_dup_all_l_m.shape[1], S_l_m_dup_all_l_m.shape[2]), order="F"
        )
        nucleus_index_ao = self._uq_center_index_ao_jnp
        l_ao = jnp.asarray(self.angular_momentums, dtype=jnp.int32)
        m_ao = jnp.asarray(self.magnetic_quantum_numbers, dtype=jnp.int32)
        global_l_m_index_ao = l_ao**2 + (m_ao + l_ao)
        global_R_l_m_index_ao = nucleus_index_ao * max_ml + global_l_m_index_ao
        S_l_m_ao = S_l_m_dup_all_l_m_reshaped[global_R_l_m_index_ao]  # (num_ao, n_elec)

        AOs = NR_ao * S_l_m_ao
        if get_dtype_jnp(self._value_zone) == jnp.float32:
            # A negligible tail may have zero radial weight and an overflowed
            # polynomial. Preserve every finite product, including signed zero,
            # and the radial (basis-coefficient) tangent. Select on NR_ao rather
            # than on isfinite(AOs): reading the product splits the fused AO
            # kernel on GPU.
            finite_angular = jnp.where(jnp.isfinite(S_l_m_ao), S_l_m_ao, jnp.zeros_like(S_l_m_ao))
            AOs = jnp.where(NR_ao == 0, NR_ao * finite_angular, AOs)
        return AOs

    def _compute_AOs_grad(self, r_carts: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Analytic AO gradients. Each shape ``(num_ao, N_e)``."""
        # Periodicity guarantor: wrap into the home cell (see compute_AOs).
        r_carts = _wrap_into_cell_if_pbc(self.structure_data, r_carts)
        return _compute_AOs_grad_analytic_sphe(self, r_carts)

    def _compute_AOs_laplacian(self, r_carts: jax.Array) -> jax.Array:
        """Analytic AO Laplacian. Shape ``(num_ao, N_e)``."""
        # Periodicity guarantor: wrap into the home cell (see compute_AOs).
        r_carts = _wrap_into_cell_if_pbc(self.structure_data, r_carts)
        return _compute_AOs_laplacian_analytic_sphe(self, r_carts)

    def _compute_AOs_value_grad_lap(self, r_carts: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Fused AO value/grad/laplacian. Each shape ``(num_ao, N_e)``."""
        # Periodicity guarantor: wrap into the home cell (see compute_AOs).
        r_carts = _wrap_into_cell_if_pbc(self.structure_data, r_carts)
        return _compute_AOs_value_grad_lap_sphe(self, r_carts)

    def _compute_AOs_debug(self, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Plain-numpy oracle for :py:func:`compute_AOs` (matches its shape).

        Routes to the kernel-level spherical oracle
        (:py:func:`_compute_AOs_sphe_debug`). The periodic (pgto) leaves
        override this to fold the lattice-image axis; this molecular form
        returns the kernel result directly (``num_ao == kernel rank``).
        """
        return _compute_AOs_sphe_debug(self, r_carts)

    def _compute_overlap_matrix(self) -> npt.NDArray[np.float64]:
        """Analytic spherical GTO overlap matrix. Shape ``(num_ao, num_ao)``.

        Evaluates the overlap by sphe -> cart conversion, applying the
        Cartesian analytic Gaussian integrals, then transforming back via
        the sphe -> cart matrix. This method owns the molecular path.
        Periodic pGTO subclasses override it with the independently
        truncated analytic one-lattice sum.
        """
        aos_cart_data, transform_matrix = _aos_sphe_to_cart(self)
        cart_overlap_matrix = _compute_overlap_matrix_cart_analytic(aos_cart_data)
        if self._n_image > 1:
            M_sphe = self.num_ao
            M_cart = aos_cart_data.num_ao
            T_mol = transform_matrix[:M_sphe, :M_cart]
            S_left = T_mol @ cart_overlap_matrix  # (M_sphe, M_cart_kernel)
            del cart_overlap_matrix
            # Fold the cart kernel-AO column axis to the molecular cart ket via
            # the ragged cart fold, then right-multiply by T_mol.
            S_left_folded = np.asarray(aos_cart_data._fold_image_axis_np(np.asarray(S_left).T)).T  # (M_sphe, M_cart)
            overlap_matrix = S_left_folded @ T_mol.T  # (M_sphe, M_sphe)
        else:
            overlap_matrix = transform_matrix @ cart_overlap_matrix @ transform_matrix.T
        return 0.5 * (overlap_matrix + overlap_matrix.T)


# --- Pgto leaves -----------------------------------------------------------
# The gto leaves (``AOs_cart_gto_data`` / ``AOs_sphe_gto_data``) are the big
# dataclasses defined above; this section adds the periodic siblings, which
# inherit the gto fields and add ``_image_truncation_tol``.


@struct.dataclass
class AOs_pGTO_base_data(AOs_GTO_base_data):
    r"""Shared concrete fields and properties for all pGTO AO leaves.

    Periodic counterpart of :class:`AOs_GTO_base_data`. The periodic pGTO
    leaves (``AOs_cart_pgto_data`` / ``AOs_sphe_pgto_data``) share this base.
    It carries the molecular ``structure_data`` (not image-tiled) plus static
    lattice-image metadata.

    A pGTO leaf evaluates the periodic AO as the sum over lattice images,
    ``chi_mu^PBC(r) = sum_n chi_mu(r + T_n)``.

    Not instantiated directly; always via one of the concrete leaves
    (``AOs_cart_pgto_data`` / ``AOs_sphe_pgto_data``), which are built from
    molecular inputs through their ``build`` factory (symmetric with
    constructing a GTO leaf) or via :py:func:`_apply_image_expansion`.
    """

    def _compute_overlap_matrix(self) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        """Converged analytic one-lattice overlap with its own tail cutoff."""
        return _compute_overlap_matrix_periodic_pgto_lattice(self)

    #: Truncation tolerance used when enumerating the lattice-image set.
    _image_truncation_tol: float = struct.field(pytree_node=False, default=0.0)
    #: Integer lattice-image indices ``(n_a, n_b, n_c)``, shape ``(n_image, 3)``
    #: stored as a hashable tuple-of-triples. ``T_n = n . cell`` is derived
    #: in-trace by :py:attr:`_expanded_positions_cart_jnp` so the lattice
    #: gradient chain through ``cell`` lights up automatically.
    #:
    #: Integer indices are stored rather than Cartesian translations: the
    #: lattice-independent metadata is what the image enumerator already
    #: produces, and deriving the Cartesian form per call rather than storing
    #: it is the prerequisite for ``cell``-active stress AD.
    _image_indices: tuple = struct.field(pytree_node=False, default_factory=tuple)
    #: (B) ragged layout. When ``_ragged_buckets`` is
    #: non-empty the leaf is in PER-ATOM RAGGED mode: ``_image_indices`` holds
    #: ONE image triple per kernel-AO (length ``M = sum_A m_A * k_A``, not
    #: ``n_image``), ``nucleus_index`` is the identity ``range(M)`` into the
    #: per-kernel-AO centres, and the image fold is the bucketed dense
    #: :func:`_fold_image_axis_ragged` instead of the uniform reshape-sum.
    #: ``_ragged_kernel_atom`` gives each kernel-AO's molecular atom (for the
    #: in-trace centre ``R_atom + T_image``); ``_ragged_buckets`` is the static
    #: ``(m, K, offset)`` multiplicity table; ``_ragged_fold_perm`` restores the
    #: original AO order after the per-bucket fold. Empty -> uniform (A) mode.
    _ragged_kernel_atom: tuple = struct.field(pytree_node=False, default_factory=tuple)
    _ragged_buckets: tuple = struct.field(pytree_node=False, default_factory=tuple)
    _ragged_fold_perm: tuple = struct.field(pytree_node=False, default_factory=tuple)
    #: (B) per kernel-AO, its home (molecular) AO index. Used by the h5 de-tile
    #: to recover the molecular base from the bucket-major ragged layout.
    _ragged_kernel_to_home: tuple = struct.field(pytree_node=False, default_factory=tuple)
    #: (B) per HOME (molecular) AO, the home-AO index of its SHELL's representative
    #: (the first AO of the shell it belongs to). Computed ONCE, EAGERLY, from the
    #: canonical molecular primitive gather (:func:`_build_image_expanded_arrays_ragged`)
    #: by order-independent structural identity, without comparing parameter values -- so
    #: it is correct for a NON-shell-major (e.g. QE-converted) basis whose shells'
    #: ``2l+1`` magnetic components are interleaved, not contiguous. Consumed only by
    #: :func:`_ragged_structural_shell_proxy` (the traced-ham c2s twin) to recover
    #: the shell grouping WITHOUT reading the then-traced exponents. Empty tuple ->
    #: the proxy falls back to its original shell-major contiguity walk.
    _ragged_home_shell_rep: tuple = struct.field(pytree_node=False, default_factory=tuple)

    @property
    def _num_ao_kernel(self) -> int:
        """Kernel-rank AO count = ``len(nucleus_index)`` (``M`` for the ragged layout)."""
        return len(self.nucleus_index)

    @property
    def _expanded_positions_cart_np(self) -> npt.NDArray[np.float64]:
        """Per-kernel-AO centres ``(M, 3)``: row ``k`` is ``R_{atom(k)} + T_{image(k)}``.

        ``nucleus_index`` is the identity ``range(M)`` so the AO kernel gathers
        one centre per kernel-AO. ``T = image_index @ cell`` is derived in-trace
        from the static integer image indices and the runtime lattice, so the
        expansion stays differentiable w.r.t. molecular positions (force) and
        ``cell`` (stress).
        """
        R = self.structure_data._positions_cart_np
        n = np.asarray(self._image_indices, dtype=np.float64).reshape(-1, 3)
        T = n @ self.structure_data._cell_np  # (M, 3)
        atom = np.asarray(self._ragged_kernel_atom, dtype=np.int64)
        return R[atom] + T  # (M, 3)

    @property
    def _expanded_positions_cart_jnp(self) -> jax.Array:
        """JAX view of :py:attr:`_expanded_positions_cart_np` (in-trace, differentiable)."""
        R = self.structure_data._positions_cart_jnp
        n = jnp.asarray(np.asarray(self._image_indices, dtype=np.float64).reshape(-1, 3), dtype=R.dtype)
        T = n @ self.structure_data._cell_jnp.astype(R.dtype)  # (M, 3)
        atom = jnp.asarray(np.asarray(self._ragged_kernel_atom, dtype=np.int64))
        return R[atom] + T  # (M, 3)

    @property
    def _uq_center_rows_np(self) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """``(rows, inverse)`` deduplicating the per-kernel-AO centre table.

        Keyed on the STATIC ``(atom, image)`` pair, not on the centre
        coordinates: the coordinates are ``structure_data.positions`` (a traced
        pytree leaf under jit), so a value-based dedup raises
        ``TracerArrayConversionError``. ``_ragged_kernel_atom`` and
        ``_image_indices`` are build-time integers, and the table is exactly
        ``R[atom] + image @ cell``, so the two keys agree.
        """
        atom = np.asarray(self._ragged_kernel_atom, dtype=np.int64).reshape(-1)
        image = np.asarray(self._image_indices, dtype=np.int64).reshape(-1, 3)
        key = np.concatenate([atom[:, None], image], axis=1)  # (M, 4)
        _, rows, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
        return np.asarray(rows, dtype=np.int64), np.asarray(inverse, dtype=np.int64).reshape(-1)

    @property
    def _uq_center_carts_jnp(self) -> jax.Array:
        """Distinct ``R_atom + T_image`` centres, ``(N_uq, 3)``.

        ``nucleus_index`` is the identity over the M KERNEL AOs, so the
        inherited table has one row per kernel AO: every angular component and
        every shell sharing an atom+image repeats the same centre. Measured
        5316 -> 556 rows on silicon (9.6x) and 2207 -> 339 on lithium_k025
        (6.5x).
        """
        rows, _ = self._uq_center_rows_np
        return self._expanded_positions_cart_jnp[jnp.asarray(rows)]

    @property
    def _uq_center_index_prim_jnp(self) -> jax.Array:
        """Row of :py:attr:`_uq_center_carts_jnp` for each primitive."""
        _, inverse = self._uq_center_rows_np
        return jnp.asarray(inverse[np.asarray(self._nucleus_index_prim_np, dtype=np.int64)], dtype=jnp.int32)

    @property
    def _uq_center_index_ao_jnp(self) -> jax.Array:
        """Row of :py:attr:`_uq_center_carts_jnp` for each AO.

        The AO-level twin of the primitive map. Both are needed: the value-only
        kernel strides the flat ``(centre, l_m)`` table with an AO-level
        nucleus index, the value+grad+lap kernels with a primitive-level one.
        Remapping only one of them silently strides the deduplicated table with
        un-deduplicated rows -- which is exactly how the first attempt at this
        change broke the spherical <-> Cartesian roundtrip.
        """
        _, inverse = self._uq_center_rows_np
        return jnp.asarray(inverse[np.asarray(self._nucleus_index_np, dtype=np.int64)], dtype=jnp.int32)

    def _fold_image_axis(self, arr: jax.Array, zone: str) -> jax.Array:
        """Collapse the per-shell ragged kernel image axis to ``(num_ao, ...)``."""
        del zone  # Real folds preserve the dtype selected by the owning kernel.
        return _fold_image_axis_ragged(arr, self._ragged_buckets_dicts, np.asarray(self._ragged_fold_perm))

    def _fold_image_axis_np(self, arr: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Numpy twin of :py:meth:`_fold_image_axis` for the plain-numpy oracle."""
        return np.asarray(
            _fold_image_axis_ragged(jnp.asarray(arr), self._ragged_buckets_dicts, np.asarray(self._ragged_fold_perm))
        )

    @property
    def _ragged_buckets_dicts(self) -> list[dict]:
        """``_ragged_buckets`` tuples re-expanded to the dict form the fold expects."""
        return [{"m": int(m), "K": int(K), "kernel_offset": int(off)} for (m, K, off) in self._ragged_buckets]

    def _compute_AOs(self, r_carts: jax.Array) -> jax.Array:
        """AO values, image-folded to shape ``(num_ao, N_e)``."""
        return self._fold_image_axis(super()._compute_AOs(r_carts), self._value_zone)

    def _compute_AOs_grad(self, r_carts: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Analytic AO gradients, image-folded. Each shape ``(num_ao, N_e)``."""
        gx, gy, gz = super()._compute_AOs_grad(r_carts)
        return (
            self._fold_image_axis(gx, "ao_grad_lap"),
            self._fold_image_axis(gy, "ao_grad_lap"),
            self._fold_image_axis(gz, "ao_grad_lap"),
        )

    def _compute_AOs_laplacian(self, r_carts: jax.Array) -> jax.Array:
        """Analytic AO Laplacian, image-folded. Shape ``(num_ao, N_e)``."""
        return self._fold_image_axis(super()._compute_AOs_laplacian(r_carts), "ao_grad_lap")

    def _compute_AOs_value_grad_lap(self, r_carts: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Fused AO value/grad/laplacian, image-folded. Each shape ``(num_ao, N_e)``."""
        val, gx, gy, gz, lap = super()._compute_AOs_value_grad_lap(r_carts)
        return (
            self._fold_image_axis(val, self._value_zone),
            self._fold_image_axis(gx, "ao_grad_lap"),
            self._fold_image_axis(gy, "ao_grad_lap"),
            self._fold_image_axis(gz, "ao_grad_lap"),
            self._fold_image_axis(lap, "ao_grad_lap"),
        )

    def _compute_AOs_debug(self, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Independent home-cell image sum, with the value zone's arithmetic.

        The Bloch mixin restores the removed translation phase after this
        evaluation. Do not share the production ragged reduction: scatter
        each image value to its declared home AO instead.
        """
        cell = np.asarray(self.structure_data.cell, dtype=np.float64)
        fractional = np.asarray(r_carts) @ np.linalg.inv(cell)
        canonical = (fractional - np.floor(fractional)) @ cell
        values = np.asarray(super()._compute_AOs_debug(canonical))
        phase = getattr(self, "_image_phase_factors", None)
        if phase is not None:
            dtype = get_complex_dtype_np(self._value_zone)
            values = values.astype(dtype) * np.asarray(phase, dtype=dtype)[:, None]
        home = np.asarray(self._ragged_kernel_to_home, dtype=np.int64)
        if home.shape != (values.shape[0],):
            raise ValueError("pGTO debug image sum requires one home AO for every image row")
        result = np.zeros((self.num_ao, *values.shape[1:]), dtype=values.dtype)
        np.add.at(result, home, values)
        return result

    def _get_info(self) -> list[str]:
        """Extend the GTO info with periodic-image metadata.

        Inserts (right after the angular-form description line, before the
        per-atom basis tables) the number of lattice image cells summed by
        the AO image expansion, the max integer image index along each
        lattice vector, and the truncation tolerance used when enumerating
        the image set.
        """
        info_lines = super()._get_info()
        image_lines: list[str] = []
        image_lines.append(f"  Number of lattice image cells = {int(self._n_image)}")
        if self._image_indices:
            n = np.asarray(self._image_indices, dtype=np.int64).reshape(-1, 3)
            n_abs_max = int(np.max(np.abs(n))) if n.size else 0
            image_lines.append(f"  max |n_a|, |n_b|, |n_c| over image indices = {n_abs_max}")
        image_lines.append(f"  image truncation tolerance = {float(self._image_truncation_tol):.3e}")
        insert_idx = len(info_lines)
        for i, line in enumerate(info_lines):
            if line.startswith("  Angular part"):
                insert_idx = i + 1
                break
        return info_lines[:insert_idx] + image_lines + info_lines[insert_idx:]


@struct.dataclass
class AOs_cart_pgto_data(AOs_pGTO_base_data, AOs_cart_gto_data):
    r"""Cartesian periodic Gaussian-type orbital basis (the pgto leaf).

    Concrete Cartesian pGTO leaf built from molecular Cartesian-GTO inputs.
    Holds the molecular ``structure_data`` (not image-tiled) and the
    Cartesian ``polynominal_order_{x,y,z}`` fields; periodic AO values
    include the lattice-image sum.

    Built via :meth:`build` from molecular Cartesian-GTO inputs, symmetric
    with constructing an :class:`AOs_cart_gto_data`.
    """

    @classmethod
    def build(
        cls,
        structure_data: Structure_data,
        nucleus_index: list[int] | tuple[int, ...],
        num_ao: int,
        num_ao_prim: int,
        orbital_indices: list[int] | tuple[int, ...],
        exponents: npt.NDArray[np.float64],
        coefficients: npt.NDArray[np.float64],
        angular_momentums: list[int] | tuple[int, ...],
        polynominal_order_x: list[int] | tuple[int, ...],
        polynominal_order_y: list[int] | tuple[int, ...],
        polynominal_order_z: list[int] | tuple[int, ...],
        tol: float = AO_IMAGE_TRUNCATION_ATOL,
    ) -> AOs_cart_pgto_data:
        r"""Build an :class:`AOs_cart_pgto_data` directly from molecular Cartesian-GTO inputs.

        ``structure_data`` must satisfy ``pbc_flag=True``. The keyword
        arguments are the same as :class:`AOs_cart_gto_data`.
        """
        mol = AOs_cart_gto_data._from_full_basis(
            structure_data=structure_data,
            nucleus_index=nucleus_index,
            num_ao=num_ao,
            num_ao_prim=num_ao_prim,
            orbital_indices=orbital_indices,
            exponents=exponents,
            coefficients=coefficients,
            angular_momentums=angular_momentums,
            polynominal_order_x=polynominal_order_x,
            polynominal_order_y=polynominal_order_y,
            polynominal_order_z=polynominal_order_z,
        )
        return _apply_image_expansion(mol, structure_data, tol=tol)


@struct.dataclass
class AOs_sphe_pgto_data(AOs_pGTO_base_data, AOs_sphe_gto_data):
    """Real-spherical periodic Gaussian-type orbital basis (the pgto leaf).

    Mirror of :class:`AOs_cart_pgto_data` for the real-spherical angular
    form: combines the periodic compute path and lattice-image metadata from
    :class:`AOs_pGTO_base_data` with the real-spherical angular kernel and
    ``magnetic_quantum_numbers`` field from :class:`AOs_sphe_gto_data`. Built
    via :meth:`build` from molecular inputs, symmetric with constructing an
    :class:`AOs_sphe_gto_data`.
    """

    @classmethod
    def build(
        cls,
        structure_data: Structure_data,
        nucleus_index: list[int] | tuple[int, ...],
        num_ao: int,
        num_ao_prim: int,
        orbital_indices: list[int] | tuple[int, ...],
        exponents: npt.NDArray[np.float64],
        coefficients: npt.NDArray[np.float64],
        angular_momentums: list[int] | tuple[int, ...],
        magnetic_quantum_numbers: list[int] | tuple[int, ...],
        tol: float = AO_IMAGE_TRUNCATION_ATOL,
    ) -> AOs_sphe_pgto_data:
        """Build an :class:`AOs_sphe_pgto_data` directly from molecular spherical-GTO inputs.

        Mirror of :meth:`AOs_cart_pgto_data.build` for the real-spherical
        angular form.
        """
        mol = AOs_sphe_gto_data._from_full_basis(
            structure_data=structure_data,
            nucleus_index=nucleus_index,
            num_ao=num_ao,
            num_ao_prim=num_ao_prim,
            orbital_indices=orbital_indices,
            exponents=exponents,
            coefficients=coefficients,
            angular_momentums=angular_momentums,
            magnetic_quantum_numbers=magnetic_quantum_numbers,
        )
        return _apply_image_expansion(mol, structure_data, tol=tol)


def _apply_image_expansion(
    base: AOs_cart_gto_data | AOs_sphe_gto_data,
    structure_data: Structure_data,
    tol: float = AO_IMAGE_TRUNCATION_ATOL,
) -> AOs_cart_pgto_data | AOs_sphe_pgto_data | AOs_cart_gto_data | AOs_sphe_gto_data:
    """Apply periodic image expansion to a base GTO leaf and return the pgto leaf.

    The base GTO leaf (``AOs_*_gto_data``) is the canonical / storage
    representation: it is what TREXIO ships, what HDF5 checkpoints persist
    after :py:func:`_detile_aos_data_for_save`, and what the trexio loader
    constructs directly. This helper expands that base form into the pgto
    leaf (``AOs_*_pgto_data``) consumed at AO evaluation time, using the
    per-SHELL ragged image layout: each shell keeps only its own radial's
    lattice images (a tight shell few, a diffuse shell many) under the
    atom-to-cell (``Omega - R_A``) truncation, and the kernel folds with the
    dense bucketed :func:`_fold_image_axis_ragged`. This is the only supported
    image layout: AO values are unchanged to ``tol`` (each dropped image is
    ``<= tol`` in normalised AO value everywhere in the home cell by
    construction, including the primitive normalisation and the contraction
    coefficient; see :func:`_primitive_image_truncation_tolerances`) while the
    AO kernel shrinks from
    ``n_image * num_ao`` to ``M = sum_shell m_shell * k_shell``.

    Args:
        base: Molecular base GTO leaf (``_n_image == 1``).
        structure_data: PBC structure (``pbc_flag=True``) supplying the cell.
        tol: Image-envelope truncation tolerance.

    If ``structure_data.pbc_flag`` is ``False`` the base leaf is returned
    unchanged --- the caller may invoke this defensively without first
    inspecting ``pbc_flag``.
    """
    if not structure_data.pbc_flag:
        return base
    if base._n_image != 1:
        raise ValueError(f"_apply_image_expansion requires a base AO leaf (_n_image == 1), got _n_image == {base._n_image}.")
    common_kwargs, kth = _build_image_expanded_arrays_ragged(base, structure_data, tol)
    if isinstance(base, AOs_cart_gto_data):
        px, py, pz = base.polynominal_order_x, base.polynominal_order_y, base.polynominal_order_z
        return AOs_cart_pgto_data._from_full_basis(
            **common_kwargs,
            polynominal_order_x=tuple(int(px[h]) for h in kth),
            polynominal_order_y=tuple(int(py[h]) for h in kth),
            polynominal_order_z=tuple(int(pz[h]) for h in kth),
            _image_truncation_tol=float(tol),
        )
    if isinstance(base, AOs_sphe_gto_data):
        mag = base.magnetic_quantum_numbers
        return AOs_sphe_pgto_data._from_full_basis(
            **common_kwargs,
            magnetic_quantum_numbers=tuple(int(mag[h]) for h in kth),
            _image_truncation_tol=float(tol),
        )
    raise TypeError(f"_apply_image_expansion: unsupported AO leaf type {type(base).__name__}")


# --- Complex Bloch-pGTO leaves (k != Gamma) --------------------------------
# Complex AO leaves carrying a single twist ``k != Gamma``. The Bloch AO
# image expansion ``sum_n e^{i k . T_n} chi_mu(r - T_n)`` requires TWO
# unit-modulus phase factors:
#
# 1. **Image-sum phase** ``e^{i k . T_n}`` -- static, one per lattice image
#    (shape ``(_n_image,)``). Stored in :py:attr:`_image_phase_factors` and
#    applied during the image fold by :py:meth:`_fold_image_axis`.
# 2. **Wrap-restore phase** ``e^{+i k . T_0(r)}`` -- dynamic, one per
#    electron (shape ``(N_e,)``). Recomputed per call from
#    ``floor(r @ inv_cell)`` and applied per-electron column after the fold
#    by :py:meth:`_restore_phase`.
#
# With both factors :py:meth:`_compute_AOs` returns the true Bloch amplitude
# ``chi_tilde_k(r)`` for any ``r`` (wrapped or not), so the downstream
# MCMC ratio ``|Psi_k(r')/Psi_k(r)|^2`` and the ECP non-local quadrature
# pick up the Bloch boundary condition automatically -- no per-quadrature
# phase machinery is needed.
#
# Both factors reduce to a fractional identity (no Cartesian k needed):
# ``k . T = 2 pi k_frac . n`` for integer image ``n``, and
# ``k . T_0 = 2 pi k_frac . floor(r @ inv_cell)`` since
# ``G_i . a_j = 2 pi delta_ij``.
#
# Spin convention ``k_up = -k_dn`` for every input: each spin sector runs
# through its own complex leaf. The down-spin leaf is the ``with_negated_k``
# partner (negated ``k_frac`` and conjugated image phases) and carries the
# conjugated coefficients of its own channel, so together they evaluate
# ``conj(phi^{+k})``; ``SD_data`` / ``AGP_data`` / ``AGPn_data`` sanity checks
# enforce the pairing.


@struct.dataclass
class _ComplexBlochPGTOMixin:
    """Bloch-phase operations shared by both complex pGTO leaves (cart + sphe).

    Mixed in FIRST in the MRO of each concrete complex leaf so its
    :py:meth:`_fold_image_axis` and the five :py:meth:`_compute_AOs*`
    overrides take precedence over the real-pgto inherited versions; ``super()``
    calls inside the overrides resolve to the real-pgto path, which means
    the AO kernel itself stays real-valued. The fold and per-electron phase
    introduce complex64 for an f32 zone and complex128 for an f64 zone.
    """

    @property
    def _value_zone(self) -> str:
        """Keep Bloch sums accurate before local MO and inverse contractions.

        A final complex64 rounding alone is insufficient for real-source stress
        parity. This is the same AO-owned policy in ordinary and tiled cells;
        it is independent of the consumer and does not change real J3 values.
        """
        return "ao_bloch_eval"

    #: ``e^{i k . T_n}`` per lattice image, shape ``(_n_image,)``, complex128.
    #: ``pytree_node=True`` so the array is a
    #: dynamic leaf rather than pytree metadata. JAX forbids arrays in
    #: metadata position (``__eq__`` on numpy arrays returns an array, so
    #: the JIT cache equality check raises ``Exception raised while
    #: checking equality of metadata fields of pytree``). A freshly-built
    #: leaf and an HDF5-loaded leaf at the same twist hold identically-
    #: valued but distinct ``ndarray`` objects, which would trip the
    #: cache on a save -> load -> MCMC round trip.
    #: As a dynamic leaf the value is JIT-traced on each compile -- at
    #: one twist per process this caches exactly like metadata storage;
    #: in a twist-averaged run each twist needs its own trace anyway.
    _image_phase_factors: npt.NDArray[np.complex128] = struct.field(
        pytree_node=True,
        default_factory=lambda: np.array([], dtype=np.complex128),
    )

    # ---- Image-fold override (factor #1: image-sum phase) -----------------
    # Replaces the real-pgto plain sum with the Bloch-weighted sum
    # ``Sum_n e^{i k . T_n} arr_n``. Same reshape pattern as the real fold
    # (2D input ``(n_image*num_ao, N_e) -> (num_ao, N_e)``), only the
    # per-image weight is non-trivial; the trailing electron axis is left as
    # the kernel produced it via the ``-1`` reshape.
    def _fold_image_axis(self, arr: jax.Array, zone: str) -> jax.Array:
        """Phase-weighted image fold: ``Sum_n e^{i k . T_n} arr_n -> (num_ao, ...)``.

        Ragged (B): the image-sum phase is per kernel-AO (``_image_phase_factors``
        is length ``M``, built from the per-kernel ``_image_indices``), so the
        Bloch weight is baked into the kernel values before the dense bucketed
        :func:`_fold_image_axis_ragged` (which then sums each AO's own images).
        """
        complex_dtype = get_complex_dtype_jnp(zone)
        phase = jnp.asarray(self._image_phase_factors, dtype=complex_dtype)
        a = jnp.asarray(arr, dtype=complex_dtype)
        return _fold_image_axis_ragged(phase[:, None] * a, self._ragged_buckets_dicts, np.asarray(self._ragged_fold_perm))

    def _fold_image_axis_np(self, arr: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        """Numpy twin of :py:meth:`_fold_image_axis` for the debug oracle path."""
        phase = np.asarray(self._image_phase_factors)
        a = jnp.asarray(np.asarray(arr).astype(phase.dtype))
        return np.asarray(
            _fold_image_axis_ragged(
                jnp.asarray(phase)[:, None] * a, self._ragged_buckets_dicts, np.asarray(self._ragged_fold_perm)
            )
        )

    # ---- Wrap-restore phase (factor #2: per-electron column) --------------
    def _restore_phase(self, r_carts: jax.Array, zone: str) -> jax.Array:
        r"""``exp(2 pi i k_frac . floor(r @ inv_cell))``, shape ``(N_e,)``.

        Locally constant in ``r`` (``floor`` jumps only at cell faces), so
        the SAME column factor multiplies the value, all 3 grad components,
        and the Laplacian -- no extra derivative-of-the-phase term is needed.
        """
        sd = self.structure_data
        inv_cell = _inv_3x3(sd._cell_jnp.astype(jnp.float64))
        # BCS dn-leaf: read k from ``self.k_frac_np`` (the leaf's
        # own k, which honours the ``_k_point_override`` flip used by the dn
        # channel). ``sd.k_frac_np`` would always return the up-channel +k
        # and silently break the dn-channel wrap-restore phase.
        k_frac = jnp.asarray(self.k_frac_np, dtype=jnp.float64)
        n0 = jnp.floor(jnp.asarray(r_carts, dtype=jnp.float64) @ inv_cell)
        turns = n0 @ k_frac
        if get_dtype_jnp(zone) != jnp.float64:
            # k_frac . n0 grows with the distance from the home cell. Reduce it
            # modulo 1 in fp64 so the fp32 angle stays in [-pi, pi].
            turns = turns - jnp.round(turns)
        angle = jnp.asarray(2.0 * jnp.pi * turns, dtype=get_dtype_jnp(zone))
        return jnp.exp(jnp.asarray(1j, dtype=get_complex_dtype_jnp(zone)) * angle)

    def _restore_phase_np(self, r_carts: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        """Numpy twin of :py:meth:`_restore_phase`."""
        inv_cell = np.linalg.inv(np.asarray(self.structure_data._cell_np, dtype=np.float64))
        k_frac = np.asarray(self.k_frac_np, dtype=np.float64)
        n0 = np.floor(np.asarray(r_carts, dtype=np.float64) @ inv_cell)
        turns = n0 @ k_frac
        if get_dtype_np(self._value_zone) != np.float64:
            turns = turns - np.round(turns)
        angle = np.asarray(2 * np.pi * turns, dtype=get_dtype_np(self._value_zone))
        return np.exp(np.asarray(1j, dtype=get_complex_dtype_np(self._value_zone)) * angle)

    def with_negated_k(self) -> Self:
        """Return the BCS / TR-pair sibling of this complex pgto leaf.

        Builds the ``-k`` AO leaf used by the dn-spin channel: flips
        :py:attr:`k_point` to ``-k`` and conjugates the static
        :py:attr:`_image_phase_factors` (since
        ``exp(-2 pi i k . R) = conj(exp(2 pi i k . R))``). The dynamic
        wrap-restore phase (:py:meth:`_restore_phase`) auto-flips via
        :py:attr:`k_frac_np`. Applying twice returns to the original leaf
        byte-for-byte.
        """
        kp = self.k_point
        if kp is None:
            return self
        return self.replace(
            k_point=tuple(-float(c) for c in kp),
            _image_phase_factors=np.conj(np.asarray(self._image_phase_factors)),
        )

    # ---- Five _compute_AOs* overrides: multiply the per-electron column ---
    # super() resolves to the real-pgto base (AOs_pGTO_base_data), whose
    # method calls self._fold_image_axis (already overridden above to apply
    # the image-sum phase). The wrap-restore phase is then multiplied here.

    def _compute_AOs(self, r_carts: jax.Array) -> jax.Array:
        return super()._compute_AOs(r_carts) * self._restore_phase(r_carts, self._value_zone)[None, :]

    def _compute_AOs_grad(self, r_carts: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        ph = self._restore_phase(r_carts, "ao_grad_lap")[None, :]
        gx, gy, gz = super()._compute_AOs_grad(r_carts)
        return gx * ph, gy * ph, gz * ph

    def _compute_AOs_laplacian(self, r_carts: jax.Array) -> jax.Array:
        return super()._compute_AOs_laplacian(r_carts) * self._restore_phase(r_carts, "ao_grad_lap")[None, :]

    def _compute_AOs_value_grad_lap(self, r_carts: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        val, gx, gy, gz, lap = super()._compute_AOs_value_grad_lap(r_carts)
        ph_eval = self._restore_phase(r_carts, self._value_zone)[None, :]
        ph_grad_lap = self._restore_phase(r_carts, "ao_grad_lap")[None, :]
        return val * ph_eval, gx * ph_grad_lap, gy * ph_grad_lap, gz * ph_grad_lap, lap * ph_grad_lap

    def _compute_AOs_debug(self, r_carts: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        ph = self._restore_phase_np(r_carts)[None, :]
        return super()._compute_AOs_debug(r_carts) * ph


@struct.dataclass
class AOs_cart_pgto_complex_data(_ComplexBlochPGTOMixin, AOs_cart_pgto_data):
    """Cartesian Bloch-pGTO leaf at a single non-Gamma twist.

    MRO: ``self -> _ComplexBlochPGTOMixin -> AOs_cart_pgto_data ->
    AOs_pGTO_base_data -> AOs_cart_gto_data -> AOs_GTO_base_data -> ...``.
    Mixin first so its overrides take precedence; ``super()`` calls inside
    the mixin reach the real-pgto code path (the gto kernel + the inherited
    static fields are reused unchanged).
    """

    def _compute_overlap_matrix(self) -> npt.NDArray[np.complex128]:
        """Converged analytic Bloch one-lattice overlap."""
        return _compute_overlap_matrix_periodic_pgto_lattice(self)


@struct.dataclass
class AOs_sphe_pgto_complex_data(_ComplexBlochPGTOMixin, AOs_sphe_pgto_data):
    """Real-spherical Bloch-pGTO leaf at a single non-Gamma twist (mirror of cart)."""

    def _compute_overlap_matrix(self) -> npt.NDArray[np.complex128]:
        """Converged analytic Bloch one-lattice overlap."""
        return _compute_overlap_matrix_periodic_pgto_lattice(self)


def is_complex_aos(aos_data: AOs_base_data) -> bool:
    """True iff ``aos_data`` produces complex-valued AO evaluations.

    Matches the complex Bloch-pGTO leaves
    (:class:`AOs_cart_pgto_complex_data`, :class:`AOs_sphe_pgto_complex_data`)
    and the complex plane-wave leaf (:class:`AOs_plane_wave_complex_data`)
    that carry the non-Gamma twist. Recurses into
    :class:`AOs_composite_data`: a composite is complex iff any of its
    children is complex (a mixed real-PW + complex-Bloch-pGTO composite is
    forbidden by the real-PW leaf's sanity check, so in practice all
    children of a complex composite are complex).

    Used everywhere the dispatcher needs to flip a real-vs-complex switch
    (MCMC ``__stored_e_L`` dtype, ECP non-local outer ``Re(...)``,
    :py:meth:`MOs_data.sanity_check` dtype gate). Mirrors the role of
    ``isinstance(..., AOs_*_pgto_data)`` for the periodic-vs-molecular
    dispatch.
    """
    if isinstance(
        aos_data,
        (
            AOs_cart_pgto_complex_data,
            AOs_sphe_pgto_complex_data,
            AOs_plane_wave_complex_data,
            AOs_bspline_complex_data,
            AOs_bspline_tiled_complex_data,
        ),
    ):
        return True
    # Composite: complex iff any child is. Defined later in this module, so
    # use isinstance via the module-global name (resolved at call time).
    if isinstance(aos_data, AOs_composite_data):
        return any(is_complex_aos(c) for c in aos_data.children)
    return False


def _promote_pgto_to_complex(
    real_leaf: AOs_cart_pgto_data | AOs_sphe_pgto_data,
    k_point: tuple[float, float, float] | None,
) -> AOs_cart_pgto_complex_data | AOs_sphe_pgto_complex_data:
    """Copy a real pgto leaf into the complex subclass + attach the image-sum phase.

    Builds ``_image_phase_factors[n] = exp(2 pi i k_frac . image_indices[n])``
    from the real leaf's existing ``_image_indices`` table (one row per
    image, aligned 1:1 with axis-0 of the image fold). ``k_point`` becomes
    the returned complex leaf's :py:attr:`AOs_GTO_base_data.k_point`.

    For the BCS / TR-pair dn-channel sibling, pass the negated twist so the
    leaf evaluates at ``-k`` and pairs with the ``+k`` up-channel leaf.
    """
    n_image = int(real_leaf._n_image)
    if k_point is None:
        k_frac = np.zeros(3, dtype=np.float64)
    else:
        k_frac = np.asarray(k_point, dtype=np.float64)
    if not real_leaf._image_indices:
        phase = np.ones(n_image, dtype=np.complex128)
    else:
        idx = np.asarray(real_leaf._image_indices, dtype=np.float64).reshape(-1, 3)
        phase = np.exp(2j * np.pi * (idx @ k_frac)).astype(np.complex128)
    if isinstance(real_leaf, AOs_cart_pgto_data):
        cls = AOs_cart_pgto_complex_data
    elif isinstance(real_leaf, AOs_sphe_pgto_data):
        cls = AOs_sphe_pgto_complex_data
    else:
        raise TypeError(f"_promote_pgto_to_complex: expected an AOs_*_pgto_data leaf, got {type(real_leaf).__name__}.")
    # Exclude fields owned by the complex mixin so the copied real-leaf state
    # cannot overwrite the phase array constructed above for this k point.
    complex_mixin_field_names = {field.name for field in dataclasses.fields(_ComplexBlochPGTOMixin)}
    fields = {
        field.name: getattr(real_leaf, field.name)
        for field in dataclasses.fields(real_leaf)
        if field.name not in complex_mixin_field_names
    }
    fields["k_point"] = tuple(float(c) for c in k_point) if k_point is not None else None
    return cls(**fields, _image_phase_factors=phase)


def build_AOs_cart_pgto_complex_data(
    structure_data: Structure_data,
    *,
    k_point: tuple[float, float, float] | None,
    nucleus_index: list[int] | tuple[int, ...],
    num_ao: int,
    num_ao_prim: int,
    orbital_indices: list[int] | tuple[int, ...],
    exponents: npt.NDArray[np.float64],
    coefficients: npt.NDArray[np.float64],
    angular_momentums: list[int] | tuple[int, ...],
    polynominal_order_x: list[int] | tuple[int, ...],
    polynominal_order_y: list[int] | tuple[int, ...],
    polynominal_order_z: list[int] | tuple[int, ...],
    tol: float = AO_IMAGE_TRUNCATION_ATOL,
) -> AOs_cart_pgto_complex_data:
    """Build a Cartesian complex Bloch-pGTO leaf in one shot.

    ``k_point`` must be a length-3 fractional tuple (``(0,0,0)`` is allowed
    and gives a complex leaf with all-ones image phases, which lets the
    Gamma-identity test compare the complex pipeline against the real
    pipeline). Mirror of :py:meth:`AOs_cart_pgto_data.build`;
    under the hood it routes through the real build +
    :py:func:`_promote_pgto_to_complex`, so the kernel/storage path stays
    single-sourced.
    """
    real_leaf = AOs_cart_pgto_data.build(
        structure_data=structure_data,
        nucleus_index=nucleus_index,
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        polynominal_order_x=polynominal_order_x,
        polynominal_order_y=polynominal_order_y,
        polynominal_order_z=polynominal_order_z,
        tol=tol,
    )
    return _promote_pgto_to_complex(real_leaf, k_point)


def build_AOs_sphe_pgto_complex_data(
    structure_data: Structure_data,
    *,
    k_point: tuple[float, float, float] | None,
    nucleus_index: list[int] | tuple[int, ...],
    num_ao: int,
    num_ao_prim: int,
    orbital_indices: list[int] | tuple[int, ...],
    exponents: npt.NDArray[np.float64],
    coefficients: npt.NDArray[np.float64],
    angular_momentums: list[int] | tuple[int, ...],
    magnetic_quantum_numbers: list[int] | tuple[int, ...],
    tol: float = AO_IMAGE_TRUNCATION_ATOL,
) -> AOs_sphe_pgto_complex_data:
    """Build a real-spherical complex Bloch-pGTO leaf in one shot.

    ``k_point``: fractional length-3 tuple at which the Bloch leaf evaluates.
    The dn-spin sibling carries the negated tuple (BCS / TR-pair convention).
    """
    real_leaf = AOs_sphe_pgto_data.build(
        structure_data=structure_data,
        nucleus_index=nucleus_index,
        num_ao=num_ao,
        num_ao_prim=num_ao_prim,
        orbital_indices=orbital_indices,
        exponents=exponents,
        coefficients=coefficients,
        angular_momentums=angular_momentums,
        magnetic_quantum_numbers=magnetic_quantum_numbers,
        tol=tol,
    )
    return _promote_pgto_to_complex(real_leaf, k_point)


def _apply_image_expansion_complex(
    base: AOs_cart_gto_data | AOs_sphe_gto_data,
    structure_data: Structure_data,
    k_point: tuple[float, float, float] | None,
    tol: float = AO_IMAGE_TRUNCATION_ATOL,
) -> AOs_cart_pgto_complex_data | AOs_sphe_pgto_complex_data | AOs_cart_gto_data | AOs_sphe_gto_data:
    """Twin of :py:func:`_apply_image_expansion` that promotes to the complex leaf.

    When ``k_point is not None`` and ``pbc_flag=True``: routes through the
    real expansion and then promotes via :py:func:`_promote_pgto_to_complex`
    so the returned leaf carries the image-sum phase array. Otherwise
    (k_point None or molecular): falls back to :py:func:`_apply_image_expansion`
    unchanged. Used by the on-load rebuild path in :py:mod:`hamiltonians`
    so a checkpoint's determinant orbital leaves come back as complex pgto
    at non-Gamma twists and as real pgto / molecular gto at Gamma / molecules.
    """
    expanded = _apply_image_expansion(base, structure_data, tol=tol)
    if k_point is None or not isinstance(expanded, (AOs_cart_pgto_data, AOs_sphe_pgto_data)):
        return expanded
    return _promote_pgto_to_complex(expanded, k_point)


# --- GTO numpy debug oracles -----------------------------------------------
# Plain-numpy forward oracles for the Cartesian / spherical GTO leaves. The
# production ``@jit`` value kernels live inside the leaf classes above
# (``AOs_cart_gto_data._compute_AOs`` / ``AOs_sphe_gto_data._compute_AOs``);
# these debug oracles stay module-level because they are simple readability
# references and share no JIT cache with the production path.
def _compute_AOs_cart_debug(aos_data: AOs_cart_gto_data, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute AO values at the given r_carts.

    The method is for computing the value of the given atomic orbital at r_carts
    for debugging purpose. See :py:func:`compute_AOs`. Iterates over kernel-rank AOs
    (``_num_ao_kernel``) so periodic pgto inputs are processed at the tiled rank
    before the caller's image-axis reduction.
    """
    dtype = get_dtype_np(aos_data._value_zone)
    aos_values = []

    for ao_index in range(aos_data._num_ao_kernel):
        R_cart = aos_data._atomic_center_carts_np[ao_index]
        l = aos_data.angular_momentums[ao_index]
        shell_indices = [i for i, v in enumerate(aos_data.orbital_indices) if v == ao_index]
        exponents = np.asarray([aos_data.exponents[i] for i in shell_indices], dtype=dtype)
        coefficients = np.asarray([aos_data.coefficients[i] for i in shell_indices], dtype=dtype)
        nx = aos_data.polynominal_order_x[ao_index]
        ny = aos_data.polynominal_order_y[ao_index]
        nz = aos_data.polynominal_order_z[ao_index]

        ao_value = []
        for r_cart in r_carts:
            # Preserve stored geometry until the displacement is formed, then
            # perform value arithmetic in ao_eval, including mixed references.
            displacement = np.asarray(np.asarray(r_cart) - np.asarray(R_cart), dtype=dtype)
            R_n = coefficients * np.exp(-exponents * np.dot(displacement, displacement))
            # normalization part
            factorial_ratio = dtype(
                scipy.special.factorial(nx)
                * scipy.special.factorial(ny)
                * scipy.special.factorial(nz)
                / (scipy.special.factorial(2 * nx) * scipy.special.factorial(2 * ny) * scipy.special.factorial(2 * nz))
            )
            N_n_l = np.sqrt(
                np.power(dtype(2) * exponents / dtype(np.pi), dtype(1.5))
                * np.power(dtype(8) * exponents, int(l), dtype=dtype)
                * factorial_ratio
            )
            # angular part
            P_l_nx_ny_nz = np.prod(np.power(displacement, (nx, ny, nz), dtype=dtype), dtype=dtype)

            ao_value.append(np.sum(N_n_l * R_n) * P_l_nx_ny_nz)

        aos_values.append(ao_value)

    return np.asarray(aos_values, dtype=dtype)


def _compute_AOs_sphe_debug(aos_data: AOs_sphe_gto_data, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute AO values at the given r_carts.

    The method is for computing the value of the given atomic orbital at r_carts
    for debugging purpose. See :py:func:`compute_AOs`. Iterates over kernel-rank AOs
    (``_num_ao_kernel``) so periodic pgto inputs are processed at the tiled rank
    before the caller's image-axis reduction.
    """
    dtype = get_dtype_np(aos_data._value_zone)
    aos_values = []

    for ao_index in range(aos_data._num_ao_kernel):
        atomic_center_cart = aos_data._atomic_center_carts_np[ao_index]
        shell_indices = [i for i, v in enumerate(aos_data.orbital_indices) if v == ao_index]
        exponents = np.asarray([aos_data.exponents[i] for i in shell_indices], dtype=dtype)
        coefficients = np.asarray([aos_data.coefficients[i] for i in shell_indices], dtype=dtype)
        angular_momentum = aos_data.angular_momentums[ao_index]
        magnetic_quantum_number = aos_data.magnetic_quantum_numbers[ao_index]
        ao_value = []
        for r_cart in r_carts:
            # radial part
            displacement = np.asarray(np.asarray(r_cart) - np.asarray(atomic_center_cart), dtype=dtype)
            R_n = coefficients * np.exp(-exponents * np.dot(displacement, displacement))
            # normalization part
            factor = dtype(
                2.0 ** (2 * angular_momentum + 3)
                * scipy.special.factorial(angular_momentum + 1)
                / scipy.special.factorial(2 * angular_momentum + 2)
            )
            N_n_l = np.sqrt(factor * np.power(dtype(2) * exponents, dtype(angular_momentum + 1.5)) / np.sqrt(dtype(np.pi)))
            # angular part
            S_l_m = _compute_S_l_m_debug(
                atomic_center_cart=np.zeros(3, dtype=dtype),
                angular_momentum=angular_momentum,
                magnetic_quantum_number=magnetic_quantum_number,
                r_cart=displacement,
                zone=aos_data._value_zone,
            )

            ao_value.append(
                np.sum(N_n_l * R_n, dtype=dtype) * np.sqrt(dtype(2 * angular_momentum + 1) / (dtype(4) * dtype(np.pi))) * S_l_m
            )

        aos_values.append(ao_value)

    return np.asarray(aos_values, dtype=dtype)


# --- AOs_composite_data + helpers ------------------------------------------
# Heterogeneous container letting a wavefunction span multiple AO data types:
# ``pGTO + PW`` for a mixed basis, or ``GTO_cart + GTO_sphe`` etc.


def _structure_data_equal(
    s1: Structure_data,
    s2: Structure_data,
    rtol: float = rtol_consistency,
    atol: float = atol_consistency,
) -> bool:
    """Tolerance-aware ``Structure_data`` equality used by composite sanity-check.

    Exact equality on categorical / integer fields (``pbc_flag``,
    ``atomic_numbers``, ``element_symbols``, ``atomic_labels``,
    ``_n_max``); ``np.allclose`` on float-valued ``positions`` and
    ``vec_a/b/c`` (two structures built independently --- e.g. one
    from TREXIO, one re-derived --- may agree to ``O(1e-15)`` but not
    bitwise, especially for ``vec_a/b/c`` from a unit-cell spec).
    """
    if s1.pbc_flag != s2.pbc_flag:
        return False
    if tuple(s1.atomic_numbers) != tuple(s2.atomic_numbers):
        return False
    if tuple(s1.element_symbols) != tuple(s2.element_symbols):
        return False
    if tuple(s1.atomic_labels) != tuple(s2.atomic_labels):
        return False
    if s1._n_max != s2._n_max:
        return False
    if not np.allclose(
        np.asarray(s1.positions, dtype=np.float64),
        np.asarray(s2.positions, dtype=np.float64),
        rtol=rtol,
        atol=atol,
    ):
        return False
    for a, b in [(s1.vec_a, s2.vec_a), (s1.vec_b, s2.vec_b), (s1.vec_c, s2.vec_c)]:
        if len(a) != len(b):
            return False
        if len(a) > 0 and not np.allclose(
            np.asarray(a, dtype=np.float64),
            np.asarray(b, dtype=np.float64),
            rtol=rtol,
            atol=atol,
        ):
            return False
    return True


@struct.dataclass
class AOs_composite_data(AOs_base_data):
    """Heterogeneous AO container.

    Holds a tuple of :class:`AOs_base_data` children. ``compute_AOs``
    (and the grad / laplacian / value_grad_lap variants) dispatches
    into each child and concatenates along the AO axis (axis 0). The
    composite's ``num_ao`` is the sum of the children's ``num_ao``.

    Constructed via ``a + b`` (the ``__add__`` operator defined on
    :class:`AOs_base_data`), which flattens nested composites so that
    ``(a + b) + c == a + (b + c) == a + b + c`` --- the final
    ``children`` tuple is always flat. The child ordering is
    user-meaningful: downstream consumers (``mo_coefficients``, the
    Slater determinant matrix, the Jastrow ``J``) index AOs by
    position, so flipping the operand order produces a different
    indexing.

    The composite does not store its own ``structure_data``; it exposes
    ``self.children[0].structure_data`` via a property. ``sanity_check``
    validates that every child agrees on ``structure_data`` (the canonical
    untiled molecular structure --- pgto leaves already carry the molecular
    structure, with image tiling reconstructed in-trace), so a
    ``(pgto + pgto)`` or ``(pgto + PW)`` composite over the same molecule
    passes the check.

    Open / periodic mixing rule: a molecular (open-boundary) GTO leaf
    (``AOs_cart_gto_data`` / ``AOs_sphe_gto_data``) describes a
    non-periodic system and may be combined ONLY with other molecular
    GTO leaves. The periodic leaves --- pGTO
    (``AOs_cart_pgto_data`` / ``AOs_sphe_pgto_data``), plane-wave
    (``AOs_plane_wave_data``), B-spline (``AOs_bspline_data``) --- may be
    freely combined with one another in any combination. Mixing an open
    GTO leaf with any periodic leaf is rejected by ``sanity_check``.
    """

    children: tuple[AOs_base_data, ...] = struct.field(pytree_node=True, default_factory=tuple)

    @classmethod
    def build(cls, children: tuple[AOs_base_data, ...] | list[AOs_base_data]) -> AOs_composite_data:
        """Build a validated :class:`AOs_composite_data` from a flat child sequence.

        The blessed factory: it stores ``children`` as a tuple, then runs
        :py:meth:`sanity_check` (which enforces the shared-structure and
        open / periodic mixing rules). The canonical ``a + b`` operator
        remains the idiomatic constructor and flattens nested composites;
        ``build`` is the explicit form for a pre-assembled child list.
        """
        data = cls(children=tuple(children))
        data.sanity_check()
        return data

    @property
    def num_ao(self) -> int:
        return sum(int(c.num_ao) for c in self.children)

    @property
    def structure_data(self) -> Structure_data:
        if not self.children:
            return Structure_data()
        return self.children[0].structure_data

    def with_negated_k(self) -> Self:
        """Return the BCS / TR-pair sibling: delegate the -k flip to each child.

        Children that carry a Bloch twist (complex pGTO / PW / B-spline
        leaves) are flipped via their own ``with_negated_k``; real /
        Gamma / molecular children pass through unchanged (their flip is
        the identity). Keeps the child order so the caller's conjugated
        MO block stays aligned with the concatenated AO index.
        """
        new_children = tuple(c.with_negated_k() if hasattr(c, "with_negated_k") else c for c in self.children)
        return self.replace(children=new_children)

    def sanity_check(self) -> None:
        if not self.children:
            raise ValueError("AOs_composite_data must have at least one child")
        ref = self.children[0].structure_data
        for c in self.children[1:]:
            if not _structure_data_equal(c.structure_data, ref):
                raise ValueError("AOs_composite_data children must share one molecular structure_data; mismatch detected.")

        # Open / periodic mixing rule: an open-boundary (molecular) GTO leaf
        # is a GTO leaf that is NOT one of the periodic pGTO leaves. Periodic
        # leaves are pGTO / plane-wave / B-spline. Mixing the two families in
        # one composite is unphysical and rejected here.
        has_open_gto = any(
            isinstance(child, AOs_GTO_base_data) and not isinstance(child, AOs_pGTO_base_data) for child in self.children
        )
        has_periodic = any(
            isinstance(
                c,
                (
                    AOs_pGTO_base_data,
                    AOs_plane_wave_data,
                    AOs_plane_wave_complex_data,
                    AOs_bspline_data,
                    AOs_bspline_complex_data,
                    AOs_bspline_tiled_complex_data,
                ),
            )
            for c in self.children
        )
        if has_open_gto and has_periodic:
            raise ValueError(
                "AOs_composite_data cannot mix an open-boundary (molecular) GTO leaf "
                "(AOs_cart_gto_data / AOs_sphe_gto_data) with a periodic leaf "
                "(pGTO / plane-wave / B-spline). Molecular GTO leaves may only be "
                "combined with other molecular GTO leaves; periodic leaves may be "
                "combined with one another in any combination."
            )
        for c in self.children:
            c.sanity_check()

    def _compute_AOs(self, r_carts: jax.Array) -> jax.Array:
        return jnp.concatenate([c._compute_AOs(r_carts) for c in self.children], axis=0)

    def _compute_AOs_grad(self, r_carts: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        parts = [c._compute_AOs_grad(r_carts) for c in self.children]
        return (
            jnp.concatenate([p[0] for p in parts], axis=0),
            jnp.concatenate([p[1] for p in parts], axis=0),
            jnp.concatenate([p[2] for p in parts], axis=0),
        )

    def _compute_AOs_laplacian(self, r_carts: jax.Array) -> jax.Array:
        return jnp.concatenate([c._compute_AOs_laplacian(r_carts) for c in self.children], axis=0)

    def _compute_AOs_value_grad_lap(self, r_carts: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        parts = [c._compute_AOs_value_grad_lap(r_carts) for c in self.children]
        return tuple(jnp.concatenate([p[i] for p in parts], axis=0) for i in range(5))

    def _compute_AOs_debug(self, r_carts: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Plain-numpy oracle for :py:func:`compute_AOs` on a composite.

        Concatenates each child's molecular-view oracle along the AO axis,
        matching :py:meth:`_compute_AOs`.
        """
        return np.concatenate(
            [np.asarray(c._compute_AOs_debug(r_carts)) for c in self.children],
            axis=0,
        )

    def _compute_overlap_matrix(self) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        """Numerical cell-domain Hermitian Gram matrix for periodic composites.

        Composites with multiple basis families (e.g. ``pGTO + plane-wave``)
        require a shared cell to integrate over; a molecular (open-boundary)
        GTO composite has no such cell and is rejected here.
        """
        if not self.structure_data.pbc_flag:
            raise NotImplementedError(
                "compute_overlap_matrix supports only periodic AOs_composite_data "
                "(a shared cell defines the integration domain); a molecular "
                "(open-boundary) GTO composite has no cell -- compute the overlap "
                "of each leaf child separately."
            )
        return _compute_overlap_matrix_composite(self)

    def _overlap_gram(
        self,
        bra: npt.ArrayLike,
        ket: npt.ArrayLike,
        dense_overlap: Callable[["AOs_base_data"], npt.ArrayLike],
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        r"""``conj(bra) S ket^T`` assembled from the child blocks of :py:func:`_compute_overlap_matrix_composite`.

        Diagonal blocks use each child's own :py:meth:`_overlap_gram` (a
        plane-wave child stays diagonal); cross blocks are the exact
        :py:func:`_composite_cross_overlap_block`. The ``num_ao x num_ao``
        composite overlap is never formed (issue #247).
        """
        if not self.structure_data.pbc_flag:
            return super()._overlap_gram(bra, ket, dense_overlap)
        bra = np.asarray(bra)
        ket = np.asarray(ket)
        children = self.children
        offsets = np.concatenate(([0], np.cumsum([int(child.num_ao) for child in children]))).astype(int)
        columns = [slice(offsets[i], offsets[i + 1]) for i in range(len(children))]
        real_composite = not is_complex_aos(self)
        gram = None
        for i, child in enumerate(children):
            term = child._overlap_gram(bra[:, columns[i]], ket[:, columns[i]], dense_overlap)
            for j in range(i + 1, len(children)):
                block = np.asarray(_composite_cross_overlap_block(child, children[j]))
                if real_composite and np.iscomplexobj(block):
                    block = np.real(block)  # as _compute_overlap_matrix_composite stores it
                term = term + _gram_with_overlap(bra[:, columns[i]], block, ket[:, columns[j]])
                term = term + _gram_with_overlap(bra[:, columns[j]], block.conj().T, ket[:, columns[i]])
            gram = term if gram is None else gram + term
        return gram

    def _get_info(self) -> list[str]:
        """Return a list of strings describing every child AO leaf.

        The default ``AOs_base_data._get_info`` only emits the composite
        class name and total ``num_ao``, which hides per-child basis
        details (GTO exponents/coefficients, PW G-count and cutoff,
        B-spline grid). This override dispatches into every child and
        indents the child's own ``_get_info`` lines so the full
        composition is visible.
        """
        info_lines: list[str] = []
        info_lines.append(f"**{self.__class__.__name__}")
        info_lines.append(f"  total num_ao = {self.num_ao}")
        info_lines.append(f"  Number of children = {len(self.children)}")
        for i, c in enumerate(self.children):
            info_lines.append("  " + "=" * 60)
            info_lines.append(f"  child {i}: {type(c).__name__} (num_ao = {getattr(c, 'num_ao', '?')})")
            info_lines.append("  " + "=" * 60)
            for line in c._get_info():
                info_lines.append("  " + line)
        return info_lines

    def _logger_info(self) -> None:
        """Output the information from :py:meth:`_get_info` via ``logger.info``."""
        for line in self._get_info():
            logger.info(line)

    # ---- Variational-parameter protocol on composites ----------------------
    # Per-child dispatch: each child contributes its own blocks under the
    # prefix ``{block_name_prefix}child{i}_``. ``apply_basis_block_update``
    # parses the ``child<i>_`` prefix off the incoming block name, locates
    # the matching child, and substitutes children[i] with the update.

    def get_basis_variational_payloads(
        self,
        *,
        opt_basis_exp: bool = False,
        opt_basis_coeff: bool = False,
        block_name_prefix: str = "",
    ) -> list[tuple[str, np.ndarray]]:
        payloads: list[tuple[str, np.ndarray]] = []
        for i, child in enumerate(self.children):
            child_prefix = f"{block_name_prefix}child{i}_"
            payloads.extend(
                child.get_basis_variational_payloads(
                    opt_basis_exp=opt_basis_exp,
                    opt_basis_coeff=opt_basis_coeff,
                    block_name_prefix=child_prefix,
                )
            )
        return payloads

    def apply_basis_block_update(self, block: VariationalParameterBlock) -> AOs_composite_data:
        # Parse the leading "...child<i>_" segment to locate the target.
        # The full name is of the form ``{outer_prefix}child<i>_<rest>``;
        # everything up to and including the LAST ``child<i>_`` belongs
        # to the routing prefix, ``<rest>`` is what the leaf sees.
        name = block.name
        marker = "child"
        # Find the rightmost "child<digits>_" segment.
        idx_child = name.rfind(marker)
        if idx_child < 0:
            return self  # Not a per-child block: nothing to dispatch.
        # Extract the digits after "child":
        after_marker = name[idx_child + len(marker) :]
        underscore_idx = after_marker.find("_")
        if underscore_idx < 0:
            return self
        digits = after_marker[:underscore_idx]
        if not digits.isdigit():
            return self
        i = int(digits)
        if i < 0 or i >= len(self.children):
            return self
        rest = after_marker[underscore_idx + 1 :]
        # Construct a new block with just the leaf-visible suffix as the
        # name and identical payload; let the child's protocol consume it.
        leaf_block = block.replace(name=rest)
        new_child = self.children[i].apply_basis_block_update(leaf_block)
        new_children = tuple((new_child if j == i else c) for j, c in enumerate(self.children))
        return self.replace(children=new_children)


def _apply_strain_to_aos(aos: AOs_base_data, F: jax.Array) -> AOs_base_data:
    """Return a new AOs leaf / container with ``F = I + eps`` pushed through every owned ``Structure_data``.

    AO leaves (``AOs_*gto_data``, ``AOs_plane_wave_data``, ``AOs_bspline_data``)
    each own a ``structure_data`` *field*, so the leaf's ``structure_data``
    is strained and ``.replace(structure_data=...)`` re-wraps it.
    :class:`AOs_composite_data`, by contrast, exposes ``structure_data`` as a
    *property* delegating to ``children[0].structure_data`` (the composite
    does not store its own structure_data --- all children share the same
    molecular structure by sanity-check). Calling
    ``.replace(structure_data=...)`` on a composite therefore raises
    ``TypeError`` because the dataclass has no such init kwarg; we must
    recurse into ``children`` and rebuild via the ``children=`` field
    instead. The stress kernels
    (:py:func:`jqmc.derivatives._auto._local_energy_strained` /
    :py:func:`jqmc.derivatives._auto._local_log_psi_strained`) call this on every
    AO container reachable from the geminal and the J3 orbital so the
    strain cotangent flows uniformly through every ``Structure_data`` leaf
    in the tree.
    """
    if isinstance(aos, AOs_composite_data):
        new_children = tuple(_apply_strain_to_aos(c, F) for c in aos.children)
        return aos.replace(children=new_children)
    return aos.replace(structure_data=_strain_structure_data(aos.structure_data, F))


@struct.dataclass
class AOs_plane_wave_data(AOs_base_data):
    r"""Plane-wave AO leaf (Gamma-point real cos/sin basis).

    Each "AO" is one real plane-wave basis function over the home cell.
    The basis is the real, Gamma-point form of the complex plane-wave set
    :math:`\{e^{i G \cdot r}\}`: for every reciprocal vector ``G`` we keep
    either ``cos(G . r)`` or ``sin(G . r)`` (the constant ``G = 0`` term is
    ``cos(0) = 1``). A band's PW Fourier amplitudes ``c_G`` (with
    ``c_{-G} = conj(c_G)`` at Gamma) map onto these real basis functions at
    the MO layer

    .. math::
       \psi(r) = c_0 + \sum_{G > 0} \big[ 2\,\mathrm{Re}(c_G)\cos(G.r)
                 - 2\,\mathrm{Im}(c_G)\sin(G.r) \big],

    so the per-band coefficients live in ``MOs_data.mo_coefficients`` and this
    leaf only stores the shared basis (its ``G`` vectors and cos/sin kind).
    ``num_ao`` is therefore the number of real basis functions, NOT the number
    of bands.

    Complex (twist / ``k != 0`` Bloch) case:
        At Gamma the wavefunction is real, so the real cos/sin basis emits
        ``float64`` and the whole downstream determinant / Jastrow path stays
        real. For a general ``k`` point the basis is the complex
        ``e^{i (k + G) . r}``, which is the separate
        :class:`AOs_plane_wave_complex_data` leaf. This class has no complex
        branch: ``_complex = True`` raises in ``sanity_check``.

    Stress AD note:
        Cartesian ``G`` vectors are DERIVED from integer indices ``m``
        and the runtime lattice ``cell`` via
        ``G = m . (2*pi * inv(cell).T)``. Storage is the static integer
        triplets ``_g_indices`` (``pytree_node=False``); the public
        ``g_vectors`` accessor remains as a numpy property for backward
        compatibility, and the JAX-active ``_g_vectors_jnp`` carries the
        ``cell`` chain that stress AD relies on.

    Attributes:
        structure_data: Home-cell ``Structure_data`` (``pbc_flag = True``);
            supplies the lattice for wrapping electrons into the cell and
            the ``cell`` chain for ``G = m . recip``.
        num_ao: Number of real plane-wave basis functions.
        _g_indices: Integer reciprocal triplets ``(m_a, m_b, m_c)`` per basis
            function, shape ``(num_ao, 3)`` stored as a hashable tuple of
            triples (static; never a differentiable leaf).
        phase_is_sin: ``(num_ao,)`` mask, ``1.0`` for a ``sin(G.r)`` basis
            function and ``0.0`` for ``cos(G.r)``. Stored as ``float64`` so the
            ``compute_*`` kernels stay branch-free under ``jit``.
        _complex: static flag; must be ``False`` (the ``k != 0`` basis is
            :class:`AOs_plane_wave_complex_data`).
    """

    #: Home-cell structure (supplies the lattice for cell wrapping).
    structure_data: Structure_data = struct.field(pytree_node=True, default_factory=lambda: Structure_data())
    #: Number of real plane-wave basis functions.
    num_ao: int = struct.field(pytree_node=False, default=0)
    #: Integer reciprocal indices ``(m_a, m_b, m_c)`` per basis function,
    #: stored as a *flat* hashable tuple of length ``3 * num_ao``: row-major
    #: order ``(m_a_0, m_b_0, m_c_0, m_a_1, ...)``. Reshape to
    #: ``(num_ao, 3)`` at use time. Integer indices are stored in place of
    #: Cartesian ``g_vectors`` so the lattice chain through ``cell`` lights
    #: up automatically under AD.
    #: The flat shape (vs the more natural tuple-of-triples) is dictated
    #: by the HDF5 round-trip path -- the shallow ``tuple()`` coercion in
    #: :py:func:`_load_dataclass_from_hdf5` deep-shallows a tuple-of-triples
    #: back into a tuple-of-lists, which fails JAX metadata equality;
    #: a flat tuple of ints reloads as a hashable tuple-of-ints with no
    #: extra shim needed.
    _g_indices: tuple = struct.field(pytree_node=False, default_factory=tuple)
    #: Per-basis cos/sin selector ``(num_ao,)``: 1.0 -> sin, 0.0 -> cos. dtype: float64.
    phase_is_sin: npt.NDArray[np.float64] = struct.field(
        pytree_node=True, default_factory=lambda: np.zeros((0,), dtype=np.float64)
    )
    #: Static flag; must be False (the ``k != 0`` leaf is a separate class).
    _complex: bool = struct.field(pytree_node=False, default=False)

    @property
    def g_vectors(self) -> npt.NDArray[np.float64]:
        """Cartesian reciprocal vectors derived from :py:attr:`_g_indices`.

        ``G = m . (2*pi * inv(cell).T)``. Numpy property kept for backward
        compatibility with callers that read ``aos.g_vectors`` directly
        (e.g. existing overlap matrix code paths). New code that needs the
        AD-active form should use :py:attr:`_g_vectors_jnp`.
        """
        if not self._g_indices:
            return np.zeros((0, 3), dtype=np.float64)
        m = np.asarray(self._g_indices, dtype=np.float64).reshape(-1, 3)
        cell = self.structure_data._cell_np  # (3, 3)
        recip = 2.0 * np.pi * np.linalg.inv(cell).T
        return m @ recip

    @property
    def _g_vectors_jnp(self) -> jax.Array:
        """``G = m . (2*pi * inv(cell).T)`` as a ``float64`` JAX array.

        Stress AD prerequisite: the lattice chain flows through
        ``structure_data._cell_jnp``, so once vec_{a,b,c} become pytree
        leaves the ``cell``-rank cotangent reaches every basis function
        automatically.
        """
        if not self._g_indices:
            return jnp.zeros((0, 3), dtype=jnp.float64)
        m = jnp.asarray(np.asarray(self._g_indices, dtype=np.float64).reshape(-1, 3), dtype=jnp.float64)
        cell = self.structure_data._cell_jnp  # (3, 3)
        recip = 2.0 * jnp.pi * _inv_3x3(cell).T
        return m @ recip

    @property
    def _phase_is_sin_jnp(self) -> jax.Array:
        """``phase_is_sin`` as a ``float64`` JAX array, shape ``(num_ao,)``."""
        return jnp.asarray(self.phase_is_sin, dtype=jnp.float64)

    def _theta_in_zone(self, r_carts: jax.Array, zone: str) -> jax.Array:
        r"""Return the Gamma-PW phase in fractional coordinates.

        For integer reciprocal indices ``m``, the phase is

        ``G.r = 2*pi * (r @ inv(cell)).m``.

        Evaluating that identity directly is materially stronger than first
        constructing wrapped Cartesian coordinates and Cartesian reciprocal
        vectors and then dotting the two.  Under a joint affine strain,
        ``r -> r F.T`` and ``cell -> cell F.T``: the fractional coordinate is
        invariant, so its strain derivative cancels while the cell inverse is
        still fp64.  In a Cartesian formulation, ``r`` and ``G`` would be
        down-cast separately to the ``ao_eval`` zone before their dot product;
        mixed-precision round-off would then leave a spurious O(1e-7) phase
        derivative which an ill-conditioned determinant could amplify to
        O(1e-5) in ``d ln|Psi| / d eps``.

        The modulo reduction is valid because every row of ``m`` is integer.
        It also preserves the existing home-cell convention for arbitrarily
        translated electron coordinates.  Only the local phase arithmetic is
        cast to ``zone``; the affine cancellation and floor decision retain
        the caller/cell precision.
        """
        dtype = get_dtype_jnp(zone)
        cell = self.structure_data._cell_jnp
        frac = jnp.asarray(r_carts) @ _inv_3x3(cell)
        frac = frac - jnp.floor(frac)
        m = jnp.asarray(self._g_indices, dtype=dtype).reshape(-1, 3)
        two_pi = jnp.asarray(2.0 * jnp.pi, dtype=dtype)
        # The length-3 contraction is an explicit sum, not ``frac @ m.T``: XLA:GPU
        # lowers that dot to a GEMM whose algorithm it autotunes per process, so the
        # fp32 phase of the same AO row could round differently from run to run (#546).
        f = frac.astype(dtype)
        return (f[:, 0:1] * m[:, 0] + f[:, 1:2] * m[:, 1] + f[:, 2:3] * m[:, 2]) * two_pi

    def sanity_check(self) -> None:
        if self._complex:
            raise NotImplementedError(
                "AOs_plane_wave_data has no complex (k != 0) path; "
                "set _complex=False (Gamma-point real cos/sin basis), or use the "
                "dedicated AOs_plane_wave_complex_data leaf for a non-Gamma twist."
            )
        if not self.structure_data.pbc_flag:
            raise ValueError("AOs_plane_wave_data requires a periodic structure_data (pbc_flag=True).")
        # Storage is a flat tuple of length 3 * num_ao; reshape to
        # (num_ao, 3) at use. Cartesian ``g_vectors`` is a derived property.
        if len(self._g_indices) != 3 * self.num_ao:
            raise ValueError(f"len(_g_indices) = {len(self._g_indices)} must equal 3 * num_ao = {3 * self.num_ao}.")
        s = np.asarray(self.phase_is_sin)
        if s.shape != (self.num_ao,):
            raise ValueError(f"phase_is_sin must have shape (num_ao,) = ({self.num_ao},); got {s.shape}.")

    def _value_in_zone(self, r_carts: jax.Array, zone: str) -> jax.Array:
        """Evaluate cos/sin values with all local arithmetic in ``zone``."""
        dtype = get_dtype_jnp(zone)
        is_sin = self._phase_is_sin_jnp.astype(dtype)
        theta = self._theta_in_zone(r_carts, zone)
        return jnp.cos(theta - (is_sin * (jnp.asarray(jnp.pi, dtype=dtype) / 2.0))[None, :]).T

    @jit
    def _compute_AOs(self, r_carts: jax.Array) -> jax.Array:
        r"""Evaluate the real plane-wave basis at electron coordinates.

        Args:
            r_carts: Electron Cartesian coordinates ``(N_e, 3)`` in Bohr.

        Returns:
            jax.Array: Basis values ``(num_ao, N_e)`` in the ``ao_eval`` dtype.
        """
        return self._value_in_zone(r_carts, "ao_eval")

    def _compute_AOs_grad(self, r_carts: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        r"""Analytic Cartesian gradients of the real plane-wave basis.

        ``grad cos(G.r) = -sin(G.r) G`` and ``grad sin(G.r) = cos(G.r) G``.

        Returns:
            tuple: ``(gx, gy, gz)``, each in the ``ao_grad_lap`` dtype.
        """
        dtype = get_dtype_jnp("ao_grad_lap")
        g = self._g_vectors_jnp.astype(dtype)
        theta = self._theta_in_zone(r_carts, "ao_grad_lap")
        is_sin = self._phase_is_sin_jnp.astype(dtype)
        # d value / d theta : cos basis -> -sin(theta); sin basis -> cos(theta).
        dval_dtheta = jnp.where(is_sin[None, :] > 0.5, jnp.cos(theta), -jnp.sin(theta))  # (N_e, num_ao)
        # grad_k = dval_dtheta * G_k  -> (num_ao, N_e) per component.
        gx = (dval_dtheta * g[:, 0][None, :]).T
        gy = (dval_dtheta * g[:, 1][None, :]).T
        gz = (dval_dtheta * g[:, 2][None, :]).T
        return gx, gy, gz

    def _compute_AOs_laplacian(self, r_carts: jax.Array) -> jax.Array:
        r"""Analytic Laplacian of the real plane-wave basis.

        ``lap cos(G.r) = -|G|^2 cos(G.r)`` and likewise for ``sin``, so the
        Laplacian is ``-|G|^2`` times the value.

        Returns:
            jax.Array: ``(num_ao, N_e)`` in the ``ao_grad_lap`` dtype.
        """
        dtype = get_dtype_jnp("ao_grad_lap")
        g = self._g_vectors_jnp.astype(dtype)
        g2 = jnp.sum(g * g, axis=1)  # (num_ao,)
        val = self._value_in_zone(r_carts, "ao_grad_lap")
        return -g2[:, None] * val

    def _compute_AOs_value_grad_lap(self, r_carts: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        r"""Fused value, Cartesian gradients, and Laplacian.

        Returns:
            tuple: value in ``ao_eval`` and derivatives in ``ao_grad_lap``.
        """
        grad_dtype = get_dtype_jnp("ao_grad_lap")
        eval_dtype = get_dtype_jnp("ao_eval")
        g = self._g_vectors_jnp.astype(grad_dtype)
        is_sin = self._phase_is_sin_jnp.astype(grad_dtype)
        theta = self._theta_in_zone(r_carts, "ao_grad_lap")

        # Phase-shift form val = cos(theta - is_sin*pi/2), dval = -sin(theta - phase):
        # same cos(phi)+sin(phi) transcendental count as the where form but NO
        # where-select. On GPU the two (N_e, num_ao) where-selects materialize/cost
        # more than the subtraction -> phase form is ~33% faster at nw=128 (MEASURED:
        # PW vgl 11.9 -> 8.0 ms). g / is_sin are function args (not closure):
        # g depends on the reciprocal lattice (cell), so atomic forces / stress
        # differentiate through it.
        def _vgl_jax(
            theta_: jax.Array,
            g_: jax.Array,
            is_sin_: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            phi = theta_ - (is_sin_ * (jnp.pi / 2.0))[None, :]  # (N_e, num_ao)
            val = jnp.cos(phi)
            dval_dtheta = -jnp.sin(phi)
            gx = (dval_dtheta * g_[:, 0][None, :]).T
            gy = (dval_dtheta * g_[:, 1][None, :]).T
            gz = (dval_dtheta * g_[:, 2][None, :]).T
            g2 = jnp.sum(g_ * g_, axis=1)  # (num_ao,)
            lap = (-g2[None, :] * val).T
            return val.T.astype(eval_dtype), gx, gy, gz, lap

        return _vgl_jax(theta, g, is_sin)

    def _compute_AOs_debug(self, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Plain-numpy oracle for :py:func:`compute_AOs` on a plane-wave leaf.

        Routes to the kernel-level oracle
        (:py:func:`_compute_AOs_plane_wave_debug`); plane waves carry no image
        axis, so the result is returned directly as ``(num_ao, N_e)``.
        """
        return _compute_AOs_plane_wave_debug(self, r_carts)

    def _compute_overlap_matrix(self) -> npt.NDArray[np.float64]:
        """Analytic plane-wave cell-domain overlap matrix. Shape ``(num_ao, num_ao)``."""
        return _compute_overlap_matrix_plane_wave(self)

    def _half_space_overlap_weights(self) -> npt.NDArray[np.float64] | None:
        r"""Diagonal of the overlap when it is diagonal, else ``None``.

        By the closed form of :py:func:`_compute_overlap_matrix_plane_wave`, two
        distinct ``cos`` (or two ``sin``) functions overlap only when their
        integer triplets are equal or opposite. Without such a pair the matrix
        is diagonal: ``V`` for ``cos`` at ``G = 0``, ``V/2`` for every other
        ``cos`` or ``sin``, ``0`` for ``sin`` at ``G = 0``.
        """
        indices = [tuple(int(m) for m in triplet) for triplet in np.asarray(self._g_indices, dtype=np.int64).reshape(-1, 3)]
        is_sin = np.asarray(self.phase_is_sin, dtype=np.float64) > 0.5
        for phase in (False, True):
            group = [key for key, sin in zip(indices, is_sin, strict=True) if sin == phase]
            keys = set(group)
            if len(keys) != len(group) or any(key != (0, 0, 0) and (-key[0], -key[1], -key[2]) in keys for key in keys):
                return None
        volume = float(np.abs(np.linalg.det(np.asarray(self.structure_data.cell, dtype=np.float64))))
        at_origin = np.array([key == (0, 0, 0) for key in indices])
        return 0.5 * volume * np.where(is_sin, 1.0 - at_origin, 1.0 + at_origin)

    def _overlap_gram(
        self,
        bra: npt.ArrayLike,
        ket: npt.ArrayLike,
        dense_overlap: Callable[["AOs_base_data"], npt.ArrayLike],
    ) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
        """``conj(bra) S ket^T`` with the diagonal ``S`` of a half-space basis, else the dense overlap (issue #247)."""
        weights = self._half_space_overlap_weights()
        if weights is None:
            return super()._overlap_gram(bra, ket, dense_overlap)
        return _gram_with_overlap(bra, weights, ket)

    def _get_info(self) -> list[str]:
        """Return a list of strings describing the plane-wave basis.

        Reports the real cos/sin basis size, the cos/sin split, the
        ``|G|_max`` and the implied kinetic cutoff (``max |G|^2 / 2``,
        Hartree), and the number of stored integer reciprocal triplets.
        """
        info_lines: list[str] = []
        info_lines.append(f"**{self.__class__.__name__}")
        info_lines.append(f"  Number of AOs = {self.num_ao}")
        is_sin = np.asarray(self.phase_is_sin)
        n_sin = int(np.sum(is_sin > 0.5))
        n_cos = int(self.num_ao - n_sin)
        info_lines.append(f"  cos basis functions = {n_cos}")
        info_lines.append(f"  sin basis functions = {n_sin}")
        info_lines.append(f"  complex (k != 0) path = {self._complex}")
        if self.num_ao > 0:
            g = self.g_vectors  # (num_ao, 3) in Bohr^-1
            g2 = np.sum(g * g, axis=1)
            g_max = float(np.sqrt(np.max(g2)))
            ke_cutoff_Ha = 0.5 * float(np.max(g2))
            info_lines.append(f"  |G|_max = {g_max:.6f} Bohr^-1")
            info_lines.append(f"  kinetic cutoff (max |G|^2 / 2) = {ke_cutoff_Ha:.6f} Hartree")
            info_lines.append(f"  kinetic cutoff (max |G|^2) = {2.0 * ke_cutoff_Ha:.6f} Ry")
            m = np.asarray(self._g_indices, dtype=np.int64).reshape(-1, 3)
            m_abs_max = int(np.max(np.abs(m))) if m.size else 0
            info_lines.append(f"  Number of stored integer G triplets = {m.shape[0]}")
            info_lines.append(f"  max |m_a|, |m_b|, |m_c| over stored triplets = {m_abs_max}")
        return info_lines

    def _logger_info(self) -> None:
        """Output the information from :py:meth:`_get_info` via ``logger.info``."""
        for line in self._get_info():
            logger.info(line)

    @classmethod
    def build(
        cls,
        structure_data: Structure_data,
        g_vectors: npt.NDArray[np.float64],
        phase_is_sin: npt.NDArray[np.float64],
    ) -> AOs_plane_wave_data:
        """Construct an :class:`AOs_plane_wave_data` real plane-wave AO leaf.

        The public interface accepts Cartesian
        ``g_vectors`` for backward compatibility with QE / projection
        pipelines (which emit Cartesian ``G``). Internally the builder
        converts to integer reciprocal indices via
        ``m = round(G . cell.T / (2*pi))`` and stores those as the static
        ``_g_indices`` field. Round-trip recovers the Cartesian to fp64
        round-off via ``G = m . (2*pi * inv(cell).T)``.

        Args:
            structure_data: Periodic home-cell structure (``pbc_flag = True``).
            g_vectors: Cartesian reciprocal vectors ``(num_ao, 3)`` in Bohr^-1.
                The constant term uses ``G = 0``.
            phase_is_sin: ``(num_ao,)`` cos/sin selector (1.0 -> sin, 0.0 -> cos).

        Returns:
            AOs_plane_wave_data: leaf with ``num_ao`` inferred from ``g_vectors``.

        Raises:
            ValueError: If the round-trip ``round(g . cell.T / (2*pi))`` does
                not recover an integer triplet within tolerance 1e-9. This
                catches malformed PW inputs (Cartesian vectors that are not
                on the reciprocal lattice).
        """
        g = np.asarray(g_vectors, dtype=np.float64)
        s = np.asarray(phase_is_sin, dtype=np.float64)
        if g.ndim != 2 or g.shape[1] != 3:
            raise ValueError(f"g_vectors must have shape (num_ao, 3); got {g.shape}.")
        # Convert Cartesian G to integer reciprocal indices m via
        # m = G . cell.T / (2*pi).
        cell = np.asarray(structure_data.cell, dtype=np.float64)  # (3, 3)
        m_float = (g @ cell.T) / (2.0 * np.pi)  # (num_ao, 3)
        m_int = np.round(m_float).astype(np.int64)
        max_drift = float(np.max(np.abs(m_float - m_int))) if g.shape[0] > 0 else 0.0
        if max_drift > 1e-9:
            raise ValueError(
                "build_AOs_plane_wave_data: incoming Cartesian g_vectors are not "
                f"on the reciprocal lattice (max |m - round(m)| = {max_drift:.3e}). "
                "Check the cell vs G-vector orientation."
            )
        # Flat tuple of length 3 * num_ao (see _g_indices field docstring for
        # the HDF5 round-trip rationale).
        indices_tuple = tuple(int(c) for c in m_int.flatten().tolist())
        data = cls(
            structure_data=structure_data,
            num_ao=int(g.shape[0]),
            _g_indices=indices_tuple,
            phase_is_sin=s,
        )
        data.sanity_check()
        return data


def _compute_AOs_plane_wave_debug(aos_data: AOs_plane_wave_data, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Plain-numpy forward oracle for :class:`AOs_plane_wave_data`.

    Readable numpy twin of ``AOs_plane_wave_data.compute_AOs``: each basis
    function is ``cos(G.r)`` (``phase_is_sin == 0``) or ``sin(G.r)``
    (``phase_is_sin == 1``). Plane waves over reciprocal vectors are invariant
    under lattice translations. Reduce in stored geometry precision, then
    evaluate the phase in the active AO zone, including in mixed mode.
    Returns ``(num_ao, N_e)``.
    """
    r = np.asarray(r_carts, dtype=np.float64)  # (N_e, 3)
    dtype = get_dtype_np("ao_eval")
    cell = np.asarray(aos_data.structure_data.cell, dtype=np.float64)
    fractional = r @ np.linalg.inv(cell)
    fractional = (fractional - np.floor(fractional)).astype(dtype)
    indices = np.asarray(aos_data._g_indices, dtype=dtype).reshape((-1, 3))
    is_sin = np.asarray(aos_data.phase_is_sin)
    theta = (fractional @ indices.T) * dtype(2.0 * np.pi)
    val = np.where(is_sin[None, :] > 0.5, np.sin(theta), np.cos(theta))  # (N_e, num_ao)
    return val.T  # (num_ao, N_e)


@struct.dataclass
class AOs_plane_wave_complex_data(AOs_base_data):
    r"""Complex plane-wave AO leaf at a single non-Gamma twist.

    Each AO is one complex plane wave ``e^{i(G + k) . r}`` over the home cell.
    The basis is the genuinely complex Bloch form of the plane-wave set: at a
    non-Gamma twist the QE wfc Fourier amplitudes ``c_G`` do NOT obey the
    Hermitian pairing ``c_{-G} = conj(c_G)`` (that pairing folds the basis to
    the real Gamma cos/sin pair :class:`AOs_plane_wave_data` uses), so each
    ``G`` carries its own complex coefficient and the basis function must be
    complex too. ``num_ao`` equals the number of stored ``G`` vectors (no
    cos/sin doubling).

    The MO layer holds the raw QE Fourier amplitudes ``c_G`` (complex128),
    and the band reconstructs as ``psi_k(r) = sum_G c_G e^{i(G + k) . r}``.

    Cell wrap intentionally OMITTED: unlike ``cos / sin`` which are
    lattice-invariant, ``e^{i k . r}`` picks up a non-trivial phase
    ``e^{i k . T}`` under a lattice translation -- the very Bloch phase the
    consumer (geminal, MCMC, ECP non-local outer ``Re(...)``) needs to see.
    Wrapping ``r`` into the home cell before the exponential would strip that
    phase and corrupt the wavefunction at electrons sitting outside the home
    cell; the kernel evaluates ``e^{i (G + k) . r}`` on the raw Cartesian
    ``r`` so the Bloch phase propagates naturally. This is the same reason
    :class:`_ComplexBlochPGTOMixin` carries its ``_restore_phase`` factor; for
    plane waves the phase is built into the basis function directly.

    Attributes:
        structure_data: Periodic home-cell ``Structure_data`` with a non-trivial
            ``k_point`` (fractional). Supplies the lattice for ``G = m . recip``
            and the Cartesian twist ``k = k_frac . recip``.
        num_ao: Number of complex plane-wave basis functions (= number of G's).
        _g_indices: Integer reciprocal triplets ``(m_a, m_b, m_c)`` stored as a
            flat hashable tuple of length ``3 * num_ao`` (same HDF5 round-trip
            convention as :class:`AOs_plane_wave_data`).
    """

    #: Home-cell structure (PBC lattice / atom positions).
    structure_data: Structure_data = struct.field(pytree_node=True, default_factory=lambda: Structure_data())
    #: Fractional crystal momentum at which this complex Bloch PW leaf
    #: evaluates. Must be a non-None length-3 tuple; the dn-spin sibling
    #: carries the negated tuple (BCS / TR-pair convention).
    k_point: tuple[float, float, float] | None = struct.field(pytree_node=False, default=None)
    #: Number of complex plane-wave basis functions (= number of G's).
    num_ao: int = struct.field(pytree_node=False, default=0)
    #: Integer reciprocal indices ``(m_a, m_b, m_c)`` per basis function,
    #: flat hashable tuple of length ``3 * num_ao`` (row-major); see the
    #: ``AOs_plane_wave_data._g_indices`` field docstring for the HDF5
    #: round-trip rationale.
    _g_indices: tuple = struct.field(pytree_node=False, default_factory=tuple)

    @property
    def k_frac_np(self) -> npt.NDArray[np.float64]:
        r"""Fractional crystal momentum as a ``(3,)`` float64 array.

        Returns zeros when :py:attr:`k_point` is ``None``.
        """
        if self.k_point is None:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(self.k_point, dtype=np.float64)

    @property
    def g_vectors(self) -> npt.NDArray[np.float64]:
        """Cartesian reciprocal vectors derived from :py:attr:`_g_indices`."""
        if not self._g_indices:
            return np.zeros((0, 3), dtype=np.float64)
        m = np.asarray(self._g_indices, dtype=np.float64).reshape(-1, 3)
        cell = self.structure_data._cell_np  # (3, 3)
        recip = 2.0 * np.pi * np.linalg.inv(cell).T
        return m @ recip

    @property
    def _gk_vectors_jnp(self) -> jax.Array:
        r"""``(G + k)`` per AO as a ``float64`` JAX array, shape ``(num_ao, 3)``.

        ``G = m . (2 pi inv(cell).T)`` (lattice chain) plus the Cartesian
        twist ``k_cart = k_frac . (2 pi inv(cell).T)``. Sharing the same
        ``2 pi inv(cell).T`` reciprocal builder keeps ``G`` and ``k`` on a
        single numerical basis (the lattice cotangent flows through
        ``structure_data._cell_jnp`` for both terms).
        """
        cell = self.structure_data._cell_jnp  # (3, 3)
        recip = 2.0 * jnp.pi * _inv_3x3(cell).T
        if not self._g_indices:
            g = jnp.zeros((0, 3), dtype=jnp.float64)
        else:
            m = jnp.asarray(np.asarray(self._g_indices, dtype=np.float64).reshape(-1, 3), dtype=jnp.float64)
            g = m @ recip
        k_frac = jnp.asarray(self.k_frac_np, dtype=jnp.float64)
        k_cart = k_frac @ recip  # (3,)
        return g + k_cart[None, :]

    def sanity_check(self) -> None:
        if not self.structure_data.pbc_flag:
            raise ValueError("AOs_plane_wave_complex_data requires a periodic structure_data (pbc_flag=True).")
        if self.k_point is None:
            raise ValueError(
                "AOs_plane_wave_complex_data requires a non-None k_point "
                "(this leaf carries the genuine Bloch phase e^{i k . r}; for Gamma use the "
                "real AOs_plane_wave_data leaf)."
            )
        if len(self._g_indices) != 3 * self.num_ao:
            raise ValueError(f"len(_g_indices) = {len(self._g_indices)} must equal 3 * num_ao = {3 * self.num_ao}.")

    def with_negated_k(self) -> Self:
        r"""Return the BCS / TR-pair sibling of this complex PW leaf.

        The dn-channel orbital is ``phi^{-k} = conj(phi^{+k})``. With
        ``phi^{+k}(r) = sum_G c_G e^{i (G + k) . r}`` the conjugate is
        ``sum_G conj(c_G) e^{i ((-G) + (-k)) . r}``: the caller conjugates
        the MO coefficients (the ``k_up = -k_dn`` convention), so the leaf must map basis
        function ``mu`` to ``e^{i ((-G_mu) + (-k)) . r}`` -- i.e. negate
        BOTH ``k_point`` and every Miller triple, keeping the AO order so
        the conjugated coefficients stay aligned. Applying twice returns
        the original leaf byte-for-byte.
        """
        kp = self.k_point
        if kp is None:
            return self
        return self.replace(
            k_point=tuple(-float(c) for c in kp),
            _g_indices=tuple(-int(m) for m in self._g_indices),
        )

    def _value_in_zone(self, r_carts: jax.Array, zone: str, *, phase_batch_floor: int = 0) -> jax.Array:
        """Evaluate ``exp(i (G+k).r)`` in the zone's component precision."""
        real_dtype = get_dtype_jnp(zone)
        complex_dtype = get_complex_dtype_jnp(zone)
        # An fp32 r.(G+k) loses about |r| |G+k| 6e-8 rad, which grows with the
        # distance from the home cell (MCMC walkers are not wrapped). An fp32
        # zone therefore forms the phase in fp64 and reduces it modulo 2 pi
        # before the cast. An fp64 zone keeps the direct product.
        phase_dtype = jnp.float64
        rc = jnp.asarray(r_carts, dtype=phase_dtype)
        gk = self._gk_vectors_jnp.astype(phase_dtype)
        padding = max(0, phase_batch_floor - rc.shape[0])
        if padding:
            # Overlap-only tiny chunks retain bulk GEMM reduction arithmetic.
            # Pad the real phase calculation, never the AO/image slab; slice
            # before complex exponentiation. The barrier preserves this layout.
            padded = jnp.pad(rc, ((0, padding), (0, 0)))
            theta = jax.lax.optimization_barrier(jnp.dot(padded, gk.T))[: rc.shape[0]]
        else:
            theta = jnp.dot(rc, gk.T)
        if real_dtype != jnp.float64:
            two_pi = 2.0 * jnp.pi
            theta = (theta - two_pi * jnp.round(theta / two_pi)).astype(real_dtype)
        return jnp.exp(jnp.asarray(1j, dtype=complex_dtype) * theta).T

    @jit
    def _compute_AOs(self, r_carts: jax.Array) -> jax.Array:
        r"""Evaluate the complex plane-wave basis at electron coordinates.

        Returns:
            jax.Array: Basis values in the complex dtype corresponding to
            ``ao_eval`` (complex64 in mixed mode, complex128 in full mode).
        """
        # No cell wrap: see the class docstring -- e^{i k . r} carries the
        # Bloch phase that consumers need; wrapping would strip it.
        return self._value_in_zone(r_carts, "ao_eval")

    def _compute_AOs_grad(self, r_carts: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        r"""Analytic gradient of ``e^{i (G + k) . r}``: ``i (G + k) e^{...}``."""
        zone = "ao_grad_lap"
        real_dtype = get_dtype_jnp(zone)
        imag_unit = jnp.asarray(1j, dtype=get_complex_dtype_jnp(zone))
        val = self._value_in_zone(r_carts, zone)
        gk = self._gk_vectors_jnp.astype(real_dtype)
        gx = (imag_unit * gk[:, 0])[:, None] * val
        gy = (imag_unit * gk[:, 1])[:, None] * val
        gz = (imag_unit * gk[:, 2])[:, None] * val
        return gx, gy, gz

    def _compute_AOs_laplacian(self, r_carts: jax.Array) -> jax.Array:
        r"""Analytic Laplacian: ``-|G + k|^2 e^{i (G + k) . r}``."""
        zone = "ao_grad_lap"
        val = self._value_in_zone(r_carts, zone)
        gk = self._gk_vectors_jnp.astype(get_dtype_jnp(zone))
        g2 = jnp.sum(gk * gk, axis=1)  # (num_ao,)
        return -g2[:, None] * val

    def _compute_AOs_value_grad_lap(self, r_carts: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        r"""Fused value, Cartesian gradients, and Laplacian (complex)."""
        zone = "ao_grad_lap"
        real_dtype = get_dtype_jnp(zone)
        complex_dtype = get_complex_dtype_jnp(zone)
        eval_complex_dtype = get_complex_dtype_jnp("ao_eval")
        gk = self._gk_vectors_jnp.astype(real_dtype)

        # phi = exp(i theta); grad = i gk phi; lap = -|gk|^2 phi. gk is a function arg
        # (not closure): it depends on the reciprocal lattice (cell), so forces/stress
        # differentiate through it.
        def _vgl_jax(
            rc: jax.Array,
            gk_: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            theta = jnp.dot(jnp.asarray(rc, dtype=real_dtype), gk_.T)
            imag_unit = jnp.asarray(1j, dtype=complex_dtype)
            val = jnp.exp(imag_unit * theta).T
            gx = (imag_unit * gk_[:, 0])[:, None] * val
            gy = (imag_unit * gk_[:, 1])[:, None] * val
            gz = (imag_unit * gk_[:, 2])[:, None] * val
            g2 = jnp.sum(gk_ * gk_, axis=1)  # (num_ao,)
            lap = -g2[:, None] * val
            return val.astype(eval_complex_dtype), gx, gy, gz, lap

        return _vgl_jax(r_carts, gk)

    def _compute_AOs_debug(self, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        """Plain-numpy oracle: ``phi_G(r) = e^{i (G + k) . r}``."""
        dtype = get_dtype_np("ao_eval")
        complex_dtype = get_complex_dtype_np("ao_eval")
        r = np.asarray(r_carts, dtype=np.float64)  # (N_e, 3)
        g = np.asarray(self.g_vectors, dtype=np.float64)  # (num_ao, 3)
        cell = np.asarray(self.structure_data._cell_np, dtype=np.float64)
        recip = 2.0 * np.pi * np.linalg.inv(cell).T
        k_cart = np.asarray(self.k_frac_np, dtype=np.float64) @ recip  # (3,)
        gk = g + k_cart[None, :]  # (num_ao, 3)
        # Form (G+k).r from the caller's float64 coordinates and reduce it
        # modulo 2 pi before the AO dtype cast: a rounded absolute coordinate
        # loses ~|r||G+k| eps32 rad, growing with the distance from the cell.
        theta = r @ gk.T  # (N_e, num_ao)
        theta = (theta - 2.0 * np.pi * np.round(theta / (2.0 * np.pi))).astype(dtype)
        return np.exp(complex_dtype(1j) * theta).T  # (num_ao, N_e)

    def _compute_overlap_matrix(self) -> npt.NDArray[np.complex128]:
        r"""Analytic complex plane-wave overlap.

        ``<phi_G | phi_G'> = V * delta_{G, G'}`` (orthogonal on the home cell
        of volume ``V``; the ``k`` shift is shared so it drops out of
        ``e^{-i(G+k).r} e^{i(G'+k).r} = e^{i(G'-G).r}``).
        """
        n = int(self.num_ao)
        V = float(np.abs(np.linalg.det(np.asarray(self.structure_data._cell_np, dtype=np.float64))))
        return V * np.eye(n, dtype=np.complex128)

    def _overlap_gram(
        self,
        bra: npt.ArrayLike,
        ket: npt.ArrayLike,
        dense_overlap: Callable[["AOs_base_data"], npt.ArrayLike],
    ) -> npt.NDArray[np.complex128]:
        """``conj(bra) S ket^T`` with the analytic ``S = V I``; no ``num_ao^2`` array (issue #247)."""
        volume = float(np.abs(np.linalg.det(np.asarray(self.structure_data._cell_np, dtype=np.float64))))
        return _gram_with_overlap(bra, np.full(int(self.num_ao), volume), ket)

    def _get_info(self) -> list[str]:
        info_lines: list[str] = []
        info_lines.append(f"**{self.__class__.__name__}")
        info_lines.append(f"  Number of AOs (complex plane waves) = {self.num_ao}")
        info_lines.append(f"  k_point (fractional) = {self.k_point}")
        if self.num_ao > 0:
            g = self.g_vectors  # (num_ao, 3)
            g2 = np.sum(g * g, axis=1)
            g_max = float(np.sqrt(np.max(g2)))
            ke_cutoff_Ha = 0.5 * float(np.max(g2))
            info_lines.append(f"  |G|_max = {g_max:.6f} Bohr^-1")
            info_lines.append(f"  kinetic cutoff (max |G|^2 / 2) = {ke_cutoff_Ha:.6f} Hartree")
        return info_lines

    def _logger_info(self) -> None:
        for line in self._get_info():
            logger.info(line)

    @classmethod
    def build(
        cls,
        structure_data: Structure_data,
        g_vectors: npt.NDArray[np.float64],
        *,
        k_point: tuple[float, float, float] | None = None,
    ) -> "AOs_plane_wave_complex_data":
        """Construct an :class:`AOs_plane_wave_complex_data` complex PW AO leaf.

        Mirrors :py:meth:`AOs_plane_wave_data.build` but takes only Cartesian
        ``g_vectors`` (no ``phase_is_sin`` because every AO is complex) plus
        the leaf's fractional ``k_point``. The per-G complex coefficients
        live in the MO layer.
        """
        g = np.asarray(g_vectors, dtype=np.float64)
        if g.ndim != 2 or g.shape[1] != 3:
            raise ValueError(f"g_vectors must have shape (num_ao, 3); got {g.shape}.")
        cell = np.asarray(structure_data.cell, dtype=np.float64)
        m_float = (g @ cell.T) / (2.0 * np.pi)
        m_int = np.round(m_float).astype(np.int64)
        max_drift = float(np.max(np.abs(m_float - m_int))) if g.shape[0] > 0 else 0.0
        if max_drift > 1e-9:
            raise ValueError(
                "AOs_plane_wave_complex_data.build: incoming Cartesian g_vectors are not "
                f"on the reciprocal lattice (max |m - round(m)| = {max_drift:.3e})."
            )
        indices_tuple = tuple(int(c) for c in m_int.flatten().tolist())
        data = cls(
            structure_data=structure_data,
            k_point=tuple(float(c) for c in k_point) if k_point is not None else None,
            num_ao=int(g.shape[0]),
            _g_indices=indices_tuple,
        )
        data.sanity_check()
        return data


def _bspline_axiswise_stencil() -> bool:
    """Whether B-spline evaluation contracts the control-point stencil one axis at a time (issue #527).

    Each value and derivative of a separable spline is ``sum_pqr cg[b, p, q, r] wx[p] wy[q] wz[r]``
    over the gathered ``(n_band, K, K, K)`` control-point stencil. As one
    ``einsum("bpqr,p,q,r->b")`` the weight vectors are contracted into an outer product first and the
    stencil is then contracted over several axes in one matrix product, so XLA:GPU materializes the
    gathered stencil for every walker and point (``n_band x K^3`` complex: 221 KB per point for 64
    quintic bands). Measured on a GH200 for Si32: the value-only evaluation at 1,024 walkers x 768
    ECP points needs 164 GiB (the out-of-memory abort of MCMC and LRDMC), against 0.27 GiB contracted
    over z, then y, then x (:func:`_bspline_axiswise_contraction`). The value, gradient and Laplacian
    kernel at 256 walkers x 128 electrons needs 7.3 GiB against 2.8 GiB, at the same speed. On the CPU
    backend the existing contractions are kept: there the fp32 reduction order is pinned
    (``AOs_bspline_data._contract_control_points``) so that cached and freshly evaluated geminal
    matrices agree exactly.
    """
    return jax.default_backend() != "cpu"


def _bspline_axiswise_contraction(cg: jax.Array, wx: jax.Array, wy: jax.Array, wz: jax.Array) -> jax.Array:
    """``sum_pqr cg[b, p, q, r] wx[p] wy[q] wz[r]`` contracted over z, then y, then x.

    ``cg`` is a real ``(n_band, K, K, K)`` control-point stencil and ``wx``, ``wy``, ``wz`` are real
    ``(K,)`` weights. Each step contracts exactly one stencil axis with one weight vector, so no product
    over the whole stencil is formed. Mathematically equal to ``einsum("bpqr,p,q,r->b", cg, wx, wy,
    wz)``; the summation order may differ, so values agree to round-off.
    """
    t = jnp.einsum("bpqr,r->bpq", cg, wz)
    t = jnp.einsum("bpq,q->bp", t, wy)
    return jnp.einsum("bp,p->b", t, wx)


@struct.dataclass
class AOs_bspline_data(AOs_base_data):
    r"""Per-band B-spline AO leaf (Gamma-point, real).

    ``num_ao`` is the number of bands (one spline AO per band).

    Attributes:
        structure_data: Home-cell ``Structure_data`` (``pbc_flag = True``);
            supplies the lattice for the fractional-coordinate map.
        num_ao: Number of bands (one spline AO per band).
        spline_coefficients: ``(num_ao, prod(spline_shape))`` control-point
            coefficients, row-major over ``spline_shape``. dtype: float64.
            Held as a JAX array.
        spline_shape: ``(N0, N1, N2)`` spline grid sizes (static).
        spline_order: spline degree (3 or 5, static).
        _complex: static flag; it must be ``False`` (``True`` raises). The
            ``k != 0`` leaf is :class:`AOs_bspline_complex_data`.
    """

    #: Home-cell structure (supplies the lattice for the fractional map).
    structure_data: Structure_data = struct.field(pytree_node=True, default_factory=lambda: Structure_data())
    #: Number of bands (one spline AO per band).
    num_ao: int = struct.field(pytree_node=False, default=0)
    #: Control-point coefficients ``(num_ao, prod(spline_shape))``. dtype: float64.
    #: A NumPy table passed at construction is held as a JAX array.
    spline_coefficients: jax.Array = struct.field(pytree_node=True, default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    #: Spline grid sizes ``(N0, N1, N2)`` (static).
    spline_shape: tuple[int, ...] = struct.field(pytree_node=False, default=())
    #: Spline degree (3 or 5, static).
    spline_order: int = struct.field(pytree_node=False, default=5)
    #: Static flag; must be False (the ``k != 0`` leaf is a separate class).
    _complex: bool = struct.field(pytree_node=False, default=False)

    @staticmethod
    def _device_resident_table(table: object) -> object:
        """Return a concrete NumPy spline table as a JAX array of the same dtype.

        A B-spline AO leaf carries its control points as a ``pytree_node=True``
        leaf. A NumPy leaf is copied to the device on EVERY call of a jitted
        function that receives the AO leaf as an argument (the transfer happens
        at the jit boundary, before the kernel runs): about 40 ms per call for the
        47 MB complex H8 table, and the full table again for every outer jitted
        driver call. Holding the table as a JAX array makes that a one-time
        transfer at construction.

        Only a concrete ``np.ndarray`` whose dtype JAX represents exactly is
        converted. Tracers, the pytree sentinels JAX passes through ``__init__``
        while unflattening, ``None`` and existing JAX arrays are returned as is,
        and a ``float64`` / ``complex128`` table is kept as NumPy when
        ``jax_enable_x64`` is off, so no conversion ever changes a value.
        """
        if type(table) is not np.ndarray or table.dtype.kind not in "fc":
            return table
        if jax.dtypes.canonicalize_dtype(table.dtype) != table.dtype:
            return table
        return jnp.asarray(table)

    def __post_init__(self) -> None:
        """Hold a concrete NumPy coefficient table as a JAX array (values unchanged)."""
        object.__setattr__(self, "spline_coefficients", self._device_resident_table(self.spline_coefficients))

    @property
    def _coeffs_cube_jnp(self) -> jax.Array:
        """Coefficients reshaped to ``(num_ao, N0, N1, N2)`` as float64 JAX."""
        return jnp.asarray(self.spline_coefficients, dtype=jnp.float64).reshape((self.num_ao, *self.spline_shape))

    def sanity_check(self) -> None:
        if self._complex:
            raise NotImplementedError(
                "AOs_bspline_data has no complex (k != 0) path; "
                "set _complex=False (Gamma-point real B-spline), or use the dedicated "
                "AOs_bspline_complex_data leaf for a non-Gamma twist."
            )
        # Real B-spline leaf is Gamma-only by class identity (the complex
        # Bloch phase lives on AOs_bspline_complex_data).
        if self.spline_order not in (3, 5):
            raise ValueError(f"spline_order must be 3 or 5; got {self.spline_order}.")
        if len(self.spline_shape) != 3:
            raise ValueError(f"spline_shape must be a 3-tuple; got {self.spline_shape}.")
        if any(int(n) < self.spline_order + 1 for n in self.spline_shape):
            raise ValueError(f"each spline_shape entry must be >= order+1 = {self.spline_order + 1}; got {self.spline_shape}.")
        c_shape = tuple(np.shape(self.spline_coefficients))
        n_pts = int(np.prod(self.spline_shape))
        if c_shape != (self.num_ao, n_pts):
            raise ValueError(
                f"spline_coefficients must have shape (num_ao, prod(spline_shape)) = ({self.num_ao}, {n_pts}); got {c_shape}."
            )
        if not self.structure_data.pbc_flag:
            raise ValueError("AOs_bspline_data requires a periodic structure_data (pbc_flag=True).")

    @staticmethod
    def _bspline_weights_jax(f: jax.Array, order: int) -> tuple[jax.Array, jax.Array, jax.Array]:
        r"""Centred uniform B-spline weights and their first/second t-derivatives.

        JAX port of :func:`jqmc.pw_pgto_projection._bspline_weights` extended with
        analytic derivatives. For a scalar fractional offset ``f`` in ``[0, 1)``,
        returns ``(w, dw, d2w)``, each shape ``(order + 1,)``: the cardinal
        B-spline ``B_order`` (and its first / second derivatives) sampled at the
        ``order + 1`` control points around ``floor(t)``. Because ``f = t - floor(t)``
        has ``df/dt = 1``, the t-derivatives equal the f-derivatives.

        Args:
            f: scalar fractional offset (JAX array).
            order: spline degree, 3 (cubic, C2) or 5 (quintic, C4).

        Returns:
            tuple[jax.Array, jax.Array, jax.Array]: value, first-derivative and
            second-derivative weight vectors.
        """
        if order == 3:
            w = (
                jnp.stack(
                    [
                        (1.0 - f) ** 3,
                        3.0 * f**3 - 6.0 * f**2 + 4.0,
                        -3.0 * f**3 + 3.0 * f**2 + 3.0 * f + 1.0,
                        f**3,
                    ]
                )
                / 6.0
            )
            dw = (
                jnp.stack(
                    [
                        -3.0 * (1.0 - f) ** 2,
                        9.0 * f**2 - 12.0 * f,
                        -9.0 * f**2 + 6.0 * f + 3.0,
                        3.0 * f**2,
                    ]
                )
                / 6.0
            )
            d2w = (
                jnp.stack(
                    [
                        6.0 * (1.0 - f),
                        18.0 * f - 12.0,
                        -18.0 * f + 6.0,
                        6.0 * f,
                    ]
                )
                / 6.0
            )
            return w, dw, d2w
        if order == 5:
            w = (
                jnp.stack(
                    [
                        (1.0 - f) ** 5,
                        26.0 - 50.0 * f + 20.0 * f**2 + 20.0 * f**3 - 20.0 * f**4 + 5.0 * f**5,
                        66.0 - 60.0 * f**2 + 30.0 * f**4 - 10.0 * f**5,
                        26.0 + 50.0 * f + 20.0 * f**2 - 20.0 * f**3 - 20.0 * f**4 + 10.0 * f**5,
                        1.0 + 5.0 * f + 10.0 * f**2 + 10.0 * f**3 + 5.0 * f**4 - 5.0 * f**5,
                        f**5,
                    ]
                )
                / 120.0
            )
            dw = (
                jnp.stack(
                    [
                        -5.0 * (1.0 - f) ** 4,
                        -50.0 + 40.0 * f + 60.0 * f**2 - 80.0 * f**3 + 25.0 * f**4,
                        -120.0 * f + 120.0 * f**3 - 50.0 * f**4,
                        50.0 + 40.0 * f - 60.0 * f**2 - 80.0 * f**3 + 50.0 * f**4,
                        5.0 + 20.0 * f + 30.0 * f**2 + 20.0 * f**3 - 25.0 * f**4,
                        5.0 * f**4,
                    ]
                )
                / 120.0
            )
            d2w = (
                jnp.stack(
                    [
                        20.0 * (1.0 - f) ** 3,
                        40.0 + 120.0 * f - 240.0 * f**2 + 100.0 * f**3,
                        -120.0 + 360.0 * f**2 - 200.0 * f**3,
                        40.0 - 120.0 * f - 240.0 * f**2 + 200.0 * f**3,
                        20.0 + 60.0 * f + 60.0 * f**2 - 100.0 * f**3,
                        20.0 * f**3,
                    ]
                )
                / 120.0
            )
            return w, dw, d2w
        raise ValueError(f"Unsupported B-spline order {order!r} (use 3 or 5).")

    @staticmethod
    def _bspline_weights_np(f: float, order: int) -> npt.NDArray[np.float64]:
        """Numpy centred uniform B-spline value weights at scalar offset ``f``.

        Plain-numpy twin of :meth:`_bspline_weights_jax` (value weights only),
        returning the ``order + 1`` cardinal B-spline weights for the local
        control points around ``floor(t)``. Lives on the leaf because only this
        leaf's numpy forward oracle (:py:func:`_compute_AOs_bspline_debug`)
        needs it.
        """
        # An array operand preserves fp32 arithmetic on both NumPy 1.x and
        # 2.x; Python scalar promotion rules differ between those versions.
        dtype = np.asarray(f).dtype
        f = np.asarray(f, dtype=dtype).reshape(1)
        if order == 3:
            return (
                np.array(
                    [
                        (1.0 - f) ** 3,
                        3.0 * f**3 - 6.0 * f**2 + 4.0,
                        -3.0 * f**3 + 3.0 * f**2 + 3.0 * f + 1.0,
                        f**3,
                    ],
                    dtype=dtype,
                ).reshape(-1)
                / 6.0
            )
        if order == 5:
            return (
                np.array(
                    [
                        (1.0 - f) ** 5,
                        26.0 - 50.0 * f + 20.0 * f**2 + 20.0 * f**3 - 20.0 * f**4 + 5.0 * f**5,
                        66.0 - 60.0 * f**2 + 30.0 * f**4 - 10.0 * f**5,
                        26.0 + 50.0 * f + 20.0 * f**2 - 20.0 * f**3 - 20.0 * f**4 + 10.0 * f**5,
                        1.0 + 5.0 * f + 10.0 * f**2 + 10.0 * f**3 + 5.0 * f**4 - 5.0 * f**5,
                        f**5,
                    ],
                    dtype=dtype,
                ).reshape(-1)
                / 120.0
            )
        raise ValueError(f"Unsupported B-spline order {order!r} (use 3 or 5).")

    @staticmethod
    def _contract_control_points(cg: jax.Array, wx: jax.Array, wy: jax.Array, wz: jax.Array) -> jax.Array:
        """Contract real control points with three real spline-weight vectors.

        CPU fp32 quintic dot reductions can round differently for a batch of
        electrons and a single moved electron. Their cached and freshly
        evaluated geminal matrices then differ before any inverse update.
        Use static pairwise sums on the three short local axes in this case.
        The loops expand at trace time; no device loop or dtype promotion is
        introduced. Cubic, fp64, and non-CPU kernels retain their einsum path.

        The complex spline leaf keeps its separate evaluation kernel. Off the CPU backend the
        stencil is contracted one axis at a time (:func:`_bspline_axiswise_stencil`, issue #527).
        """
        if _bspline_axiswise_stencil():
            return _bspline_axiswise_contraction(cg, wx, wy, wz)
        if jax.default_backend() != "cpu" or cg.dtype != jnp.float32 or cg.shape[-1] != 6:
            return jnp.einsum("bpqr,p,q,r->b", cg, wx, wy, wz)

        value = cg
        for weight in (wz, wy, wx):
            terms = [value[..., index] * weight[index] for index in range(weight.shape[0])]
            while len(terms) > 1:
                terms = [terms[index] + terms[index + 1] for index in range(0, len(terms) - 1, 2)] + (
                    terms[-1:] if len(terms) % 2 else []
                )
            value = terms[0]
        return value

    def _eval_core(self, r_carts: jax.Array, zone: str) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        r"""Shared separable spline kernel: value, gradient, Laplacian.

        Returns ``(val, gx, gy, gz, lap)``, each ``(num_ao, N_e)``.
        """
        dtype = get_dtype_jnp(zone)
        stable_geometry = dtype == jnp.float32 and jax.default_backend() == "cpu"
        r_points = (
            jnp.asarray(r_carts) if stable_geometry else _wrap_into_cell_if_pbc(self.structure_data, r_carts).astype(dtype)
        )
        order = self.spline_order
        # ``spline_coefficients`` is a pytree_node=True leaf (so it survives the
        # HDF5 round-trip and tree maps), but it is NEVER a variational
        # parameter -- the B-spline is a post-optimization "bake" of an
        # already-converged orbital, with no optimize flag anywhere. It also
        # has no analytic dependence on nuclear positions or the cell (the
        # control points encode the orbital on the fixed fractional grid).
        # Stop the gradient at the
        # point of use so ``grad(compute_local_energy, argnums=0)`` (parameter
        # derivatives, de_L/dc) does not materialise a dense gradient w.r.t.
        # the full (num_ao, prod(spline_shape)) control-point tensor -- on a
        # silicon cell that buffer is tens of GiB and OOMs the device. This is
        # exact, not an approximation: d(value)/dr (kinetic / drift) and
        # d(value)/dcell (stress) flow through the spline weights w_p(frac(r,
        # cell)), not through the coefficients, so only the (always-zero in
        # practice) d/d(coeff) channel is stopped.
        coeffs = jax.lax.stop_gradient(self._coeffs_cube_jnp.astype(dtype))  # (n_band, N0, N1, N2)
        cell = self.structure_data._cell_jnp.astype(dtype)
        inv_cell = _inv_3x3(cell)  # frac = r @ inv_cell
        # Preserve stored cell precision while reconstructing the local grid offset.
        # A rounded Cartesian home-cell detour moves spline nodes before any
        # weight arithmetic; rounding frac before multiplying by N also amplifies
        # that error. Weights and control-point contractions stay in the AO zone.
        inv_geometry = _inv_3x3(self.structure_data._cell_jnp) if stable_geometry else None
        ns_geometry = jnp.asarray(self.spline_shape, dtype=inv_geometry.dtype) if stable_geometry else None
        ns_vec = jnp.asarray(self.spline_shape, dtype=dtype)  # (3,)
        metric = inv_cell.T @ inv_cell  # (3, 3); sum_k inv_cell[k,a] inv_cell[k,b]
        # A[a,b] = N_a N_b M[a,b]: maps the t-Hessian to the Cartesian Laplacian.
        a_mat = (ns_vec[:, None] * ns_vec[None, :]) * metric  # (3, 3)
        start = -((order - 1) // 2)  # -1 (cubic) / -2 (quintic)
        offsets = jnp.arange(start, start + order + 1, dtype=jnp.int32)  # (order+1,)

        def eval_one(r: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            # Keep the per-point matrix-vector layout for batch/single parity.
            if stable_geometry:
                frac = inv_geometry.T @ r
                frac = frac - jnp.floor(frac)
                t = frac * ns_geometry
            else:
                frac = r @ inv_cell  # (3,)
                frac = frac - jnp.floor(frac)  # into [0, 1)
                t = frac * ns_vec  # (3,)
            i = jnp.floor(t)
            f = (t - i).astype(dtype)  # (3,)
            i_int = i.astype(jnp.int32)
            w0, dw0, d20 = self._bspline_weights_jax(f[0], order)
            w1, dw1, d21 = self._bspline_weights_jax(f[1], order)
            w2, dw2, d22 = self._bspline_weights_jax(f[2], order)
            # Local control-point gather (modulo wrap via mode="wrap").
            cg = jnp.take(coeffs, i_int[0] + offsets, axis=1, mode="wrap")
            cg = jnp.take(cg, i_int[1] + offsets, axis=2, mode="wrap")
            cg = jnp.take(cg, i_int[2] + offsets, axis=3, mode="wrap")  # (n_band, K, K, K)
            val = self._contract_control_points(cg, w0, w1, w2)
            dt0 = self._contract_control_points(cg, dw0, w1, w2)
            dt1 = self._contract_control_points(cg, w0, dw1, w2)
            dt2 = self._contract_control_points(cg, w0, w1, dw2)
            h00 = self._contract_control_points(cg, d20, w1, w2)
            h11 = self._contract_control_points(cg, w0, d21, w2)
            h22 = self._contract_control_points(cg, w0, w1, d22)
            h01 = self._contract_control_points(cg, dw0, dw1, w2)
            h02 = self._contract_control_points(cg, dw0, w1, dw2)
            h12 = self._contract_control_points(cg, w0, dw1, dw2)
            grad_t = jnp.stack([dt0, dt1, dt2], axis=1)  # (n_band, 3)
            grad_r = (grad_t * ns_vec[None, :]) @ inv_cell.T  # (n_band, 3)
            lap = (
                h00 * a_mat[0, 0]
                + h11 * a_mat[1, 1]
                + h22 * a_mat[2, 2]
                + 2.0 * (h01 * a_mat[0, 1] + h02 * a_mat[0, 2] + h12 * a_mat[1, 2])
            )
            return val, grad_r[:, 0], grad_r[:, 1], grad_r[:, 2], lap

        val, gx, gy, gz, lap = jax.vmap(eval_one)(r_points)  # each (N_e, n_band)
        return val.T, gx.T, gy.T, gz.T, lap.T

    @jit
    def _compute_AOs(self, r_carts: jax.Array) -> jax.Array:
        r"""Evaluate the per-band splines at electron coordinates.

        Returns:
            jax.Array: ``(num_ao, N_e)`` band values in ``ao_eval``.
        """
        return self._eval_core(r_carts, "ao_eval")[0]

    def _compute_AOs_grad(self, r_carts: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        r"""Analytic Cartesian gradients of the per-band splines.

        Returns:
            tuple: ``(gx, gy, gz)``, each in ``ao_grad_lap``.
        """
        _, gx, gy, gz, _ = self._eval_core(r_carts, "ao_grad_lap")
        return gx, gy, gz

    def _compute_AOs_laplacian(self, r_carts: jax.Array) -> jax.Array:
        r"""Analytic Laplacian of the per-band splines.

        Returns:
            jax.Array: ``(num_ao, N_e)`` in ``ao_grad_lap``.
        """
        return self._eval_core(r_carts, "ao_grad_lap")[4]

    def _compute_AOs_value_grad_lap(self, r_carts: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        r"""Fused value, Cartesian gradients, and Laplacian.

        Returns:
            tuple: value in ``ao_eval`` and derivatives in ``ao_grad_lap``.
        """
        val, gx, gy, gz, lap = self._eval_core(r_carts, "ao_grad_lap")
        return val.astype(get_dtype_jnp("ao_eval")), gx, gy, gz, lap

    def _compute_AOs_debug(self, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Plain-numpy oracle for :py:func:`compute_AOs` on a B-spline leaf.

        Routes to the kernel-level oracle
        (:py:func:`_compute_AOs_bspline_debug`); the spline leaf carries no
        image axis, so the result is returned directly as ``(num_ao, N_e)``.
        """
        return _compute_AOs_bspline_debug(self, r_carts)

    def _compute_overlap_matrix(self) -> npt.NDArray[np.float64]:
        """Exact cell-domain overlap ``S_ab = int_cell u_a u_b dr``. Shape ``(num_ao, num_ao)``.

        See :py:func:`_compute_overlap_matrix_bspline`.
        """
        return _compute_overlap_matrix_bspline(self)

    def _get_info(self) -> list[str]:
        """Return a list of strings describing the B-spline basis.

        Reports num_ao (= number of bands, one spline AO per band), the
        spline grid shape ``(N0, N1, N2)``, the spline degree, the
        per-band control-point count ``prod(spline_shape)`` and the total
        coefficient-tensor size.
        """
        info_lines: list[str] = []
        info_lines.append(f"**{self.__class__.__name__}")
        info_lines.append(f"  Number of AOs = {self.num_ao}")
        info_lines.append(f"  spline grid shape (N0, N1, N2) = {tuple(int(n) for n in self.spline_shape)}")
        info_lines.append(f"  spline order (degree) = {self.spline_order}")
        cp_per_ao = int(np.prod(self.spline_shape)) if self.spline_shape else 0
        info_lines.append(f"  control points per AO = {cp_per_ao}")
        info_lines.append(f"  total stored coefficients (num_ao * cp_per_ao) = {int(self.num_ao) * cp_per_ao}")
        info_lines.append(f"  complex (k != 0) path = {self._complex}")
        return info_lines

    def _logger_info(self) -> None:
        """Output the information from :py:meth:`_get_info` via ``logger.info``."""
        for line in self._get_info():
            logger.info(line)

    @classmethod
    def build(
        cls,
        structure_data: Structure_data,
        spline_coefficients: npt.NDArray,
        spline_shape: tuple[int, int, int],
        spline_order: int = 5,
        image_tol: float = 1.0e-6,
    ) -> AOs_bspline_data:
        """Construct an :class:`AOs_bspline_data` per-band B-spline AO leaf.

        Bridges :func:`jqmc.pw_pgto_projection.project_pw_to_spline` output to the
        jQMC AO layer. At Gamma the band coefficients are real; if complex
        coefficients are passed their imaginary part must be negligible (the leaf
        keeps the real part). Accepts either flattened ``(num_ao, prod(shape))`` or
        cube ``(num_ao, *shape)`` coefficients.

        Args:
            structure_data: Periodic home-cell structure (``pbc_flag = True``).
            spline_coefficients: control-point coefficients, real or complex.
            spline_shape: ``(N0, N1, N2)`` spline grid sizes.
            spline_order: spline degree (3 or 5).
            image_tol: tolerance on the imaginary part of complex coefficients,
                interpreted both as an absolute floor and as a fraction of
                ``max |Re|``: ``threshold = image_tol * max(1, max(|Re|))``.
                Gamma-point bands are real up to a global phase; the
                PW->spline (or PW->pGTO+spline) projection is ill-conditioned
                (``cond(S)`` can reach ``1e8``) so the roundoff floor on
                ``|Im|`` is ``|Re| * eps_machine * cond ~ |Re| * 1e-8``.
                Default ``1e-6`` keeps a 10-100x margin over the floor at
                ``cond(S) ~ 1e8`` while still rejecting a misrouted twist
                (where ``|Im|/|Re| ~ O(1)``, 6+ orders above threshold).

        Returns:
            AOs_bspline_data: leaf with ``num_ao`` inferred from the coefficients.
        """
        c = np.asarray(spline_coefficients)
        n_band = int(c.shape[0])
        n_pts = int(np.prod(spline_shape))
        c = c.reshape(n_band, n_pts)
        if np.iscomplexobj(c):
            max_imag = float(np.max(np.abs(c.imag))) if c.size else 0.0
            max_real = float(np.max(np.abs(c.real))) if c.size else 0.0
            # Scale tol with |Re|_max so a Gamma projection with O(10) coefficients
            # gets a 10x larger threshold than one with O(1) coefficients (the
            # roundoff floor scales the same way). Floor at the absolute
            # ``image_tol`` so tiny coefficients still get a sane bound.
            threshold = image_tol * max(1.0, max_real)
            if max_imag > threshold:
                raise ValueError(
                    f"AOs_bspline_data.build: spline coefficients carry a non-negligible "
                    f"imaginary part (max |Im| = {max_imag:.3e} > threshold = {threshold:.3e}; "
                    f"image_tol = {image_tol:.3e}, max |Re| = {max_real:.3e}). The "
                    f"real B-spline leaf is for Gamma-point bands only; for a non-Gamma "
                    f"twist use AOs_bspline_complex_data.build (which keeps complex u_k "
                    f"control points and applies the e^{{i k . r}} Bloch phase). The "
                    f"caller (typically _build_bspline_leaf_and_mo) should dispatch on "
                    f"its k_point argument and route to the complex leaf at non-Gamma."
                )
            c = c.real
        c = np.ascontiguousarray(c, dtype=np.float64)
        data = cls(
            structure_data=structure_data,
            num_ao=n_band,
            spline_coefficients=c,
            spline_shape=tuple(int(n) for n in spline_shape),
            spline_order=int(spline_order),
        )
        data.sanity_check()
        return data


@struct.dataclass
class AOs_bspline_complex_data(AOs_base_data):
    r"""Complex B-spline AO leaf at a single non-Gamma twist.

    ``num_ao`` equals the number of bands (one spline AO per band). This leaf
    carries complex control points and the per-electron Bloch phase; value,
    gradient, and Laplacian evaluation include that phase consistently.

    Attributes:
        structure_data: Home-cell ``Structure_data`` with a non-trivial
            ``k_point``.
        num_ao: Number of bands (one spline AO per band).
        spline_coefficients: ``(num_ao, prod(spline_shape))`` complex128
            control-point coefficients, held as a JAX array.
        spline_shape: ``(N0, N1, N2)`` spline grid sizes (static).
        spline_order: spline degree (3 or 5, static).
    """

    structure_data: Structure_data = struct.field(pytree_node=True, default_factory=lambda: Structure_data())
    #: Fractional crystal momentum at which this complex Bloch B-spline leaf
    #: evaluates. Must be a non-None length-3 tuple; the dn-spin sibling
    #: carries the negated tuple (BCS / TR-pair convention).
    k_point: tuple[float, float, float] | None = struct.field(pytree_node=False, default=None)
    num_ao: int = struct.field(pytree_node=False, default=0)
    spline_coefficients: jax.Array = struct.field(
        pytree_node=True, default_factory=lambda: np.zeros((0, 0), dtype=np.complex128)
    )
    spline_shape: tuple[int, ...] = struct.field(pytree_node=False, default=())
    spline_order: int = struct.field(pytree_node=False, default=5)
    #: Logical conjugation flag used by time-reversal siblings.  The large
    #: coefficient table remains shared instead of allocating ``conj(table)``.
    conjugate_coefficients: bool = struct.field(pytree_node=False, default=False)

    def __post_init__(self) -> None:
        """Hold a concrete NumPy coefficient table as a JAX array (values unchanged)."""
        object.__setattr__(self, "spline_coefficients", AOs_bspline_data._device_resident_table(self.spline_coefficients))

    @property
    def k_frac_np(self) -> npt.NDArray[np.float64]:
        r"""Fractional crystal momentum as a ``(3,)`` float64 array.

        Returns zeros when :py:attr:`k_point` is ``None``.
        """
        if self.k_point is None:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(self.k_point, dtype=np.float64)

    @property
    def _coeffs_cube_jnp(self) -> jax.Array:
        """Coefficients reshaped to ``(num_ao, N0, N1, N2)`` as complex128 JAX."""
        coefficients = jnp.asarray(self.spline_coefficients, dtype=jnp.complex128)
        if self.conjugate_coefficients:
            coefficients = jnp.conj(coefficients)
        return coefficients.reshape((self.num_ao, *self.spline_shape))

    @property
    def _spline_coefficients_np(self) -> npt.NDArray[np.complex128]:
        """Logical coefficient table, including a lazy TR conjugation."""
        coefficients = np.asarray(self.spline_coefficients, dtype=np.complex128)
        return np.conj(coefficients) if self.conjugate_coefficients else coefficients

    def sanity_check(self) -> None:
        if not self.structure_data.pbc_flag:
            raise ValueError("AOs_bspline_complex_data requires a periodic structure_data (pbc_flag=True).")
        if self.k_point is None:
            raise ValueError(
                "AOs_bspline_complex_data requires a non-None k_point "
                "(this leaf carries the genuine Bloch phase e^{i k . r}; for Gamma use the "
                "real AOs_bspline_data leaf)."
            )
        if self.spline_order not in (3, 5):
            raise ValueError(f"spline_order must be 3 or 5; got {self.spline_order}.")
        if len(self.spline_shape) != 3:
            raise ValueError(f"spline_shape must be a 3-tuple; got {self.spline_shape}.")
        if any(int(n) < self.spline_order + 1 for n in self.spline_shape):
            raise ValueError(f"each spline_shape entry must be >= order+1 = {self.spline_order + 1}; got {self.spline_shape}.")
        c_shape = tuple(np.shape(self.spline_coefficients))
        n_pts = int(np.prod(self.spline_shape))
        if c_shape != (self.num_ao, n_pts):
            raise ValueError(f"spline_coefficients shape {c_shape} must be (num_ao={self.num_ao}, prod(spline_shape)={n_pts}).")

    def with_negated_k(self) -> Self:
        r"""Return the BCS / TR-pair sibling of this complex B-spline leaf.

        The dn-channel orbital is ``phi^{-k} = conj(phi^{+k})``. With
        ``phi^{+k}(r) = e^{i k . r} u_k(r)`` the conjugate is
        ``e^{i (-k) . r} conj(u_k)(r)``: negate ``k_point`` AND conjugate
        the (per-band) complex spline control points. The caller's MO
        conjugation (the ``k_up = -k_dn`` convention) is a no-op here -- the per-band MO
        matrix is the real identity -- so the whole TR flip lives on the
        leaf. Applying twice returns the original leaf byte-for-byte.
        """
        kp = self.k_point
        if kp is None:
            return self
        return self.replace(
            k_point=tuple(-float(c) for c in kp),
            conjugate_coefficients=not self.conjugate_coefficients,
        )

    @property
    def _k_cart_jnp(self) -> jax.Array:
        """Cartesian twist ``k = k_frac . (2 pi inv(cell).T)``, shape ``(3,)``."""
        cell = self.structure_data._cell_jnp
        recip = 2.0 * jnp.pi * _inv_3x3(cell).T
        k_frac = jnp.asarray(self.k_frac_np, dtype=jnp.float64)
        return k_frac @ recip

    @property
    def _spline_cell_jnp(self) -> jax.Array:
        """Cell on which the periodic spline table is defined.

        The ordinary complex leaf stores a whole-cell table, so this is the
        Hamiltonian cell.  The tiled subclass overrides the property with its
        primitive cell while keeping ``structure_data`` equal to the physical
        supercell used by the rest of the Hamiltonian.
        """
        return self.structure_data._cell_jnp

    @property
    def _orbital_k_cart_jnp(self) -> jax.Array:
        """Cartesian Bloch vectors, broadcastable over the AO axis."""
        return self._k_cart_jnp[None, :]

    def _eval_u_core(self, r_carts: jax.Array, zone: str) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        r"""Complex separable spline kernel for ``u_k`` (cell-periodic part).

        Mirrors :py:meth:`AOs_bspline_data._eval_core` but with complex
        coefficients; returns ``(u, du_x, du_y, du_z, lap u)`` each
        ``(num_ao, N_e)`` in the complex dtype corresponding to ``zone``. No
        Bloch phase is applied here -- that is layered on top in the public
        ``_compute_AOs*`` methods.
        """
        real_dtype = get_dtype_jnp(zone)
        complex_dtype = get_complex_dtype_jnp(zone)
        order = self.spline_order
        coeffs = jax.lax.stop_gradient(self._coeffs_cube_jnp.astype(complex_dtype))
        cell = self._spline_cell_jnp.astype(real_dtype)
        inv_cell = _inv_3x3(cell)
        # Geometry already enters in fp64. Keep fractional wrapping in that
        # zone before converting spline weights to the active AO precision.
        # A Cartesian fp32 round-trip before inv(cell) makes almost identical
        # ECP mesh coordinates select different rounded interpolation weights,
        # and loses the stencil position in proportion to |r| (unwrapped
        # walkers: 5e-3 at 1e5 bohr). This holds on every backend, including
        # the GPU route (issue #543).
        # The grid coordinate ``t = frac N`` and its local offset ``f = t - floor(t)``
        # are formed in the same geometry precision; only ``f`` enters the AO zone,
        # as in the real leaf and the native kernel.  Rounding ``frac`` first leaves
        # ``f`` with an absolute error ~ eps32 * N instead of ~ eps32.
        stable_geometry = real_dtype == jnp.float32
        inv_geometry = _inv_3x3(self._spline_cell_jnp) if stable_geometry else None
        ns_geometry = jnp.asarray(self.spline_shape, dtype=inv_geometry.dtype) if stable_geometry else None
        ns_vec = jnp.asarray(self.spline_shape, dtype=real_dtype)
        metric = inv_cell.T @ inv_cell
        a_mat = (ns_vec[:, None] * ns_vec[None, :]) * metric
        start = -((order - 1) // 2)
        offsets = jnp.arange(start, start + order + 1, dtype=jnp.int32)

        def eval_one(r: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            # Evaluate the short geometry dot per point so scalar and batched
            # paths keep the same layout. Full precision is unchanged.
            if stable_geometry:
                frac = inv_geometry.T @ r
                frac = frac - jnp.floor(frac)
                t = frac * ns_geometry
            else:
                frac = r @ inv_cell
                frac = frac - jnp.floor(frac)
                t = frac * ns_vec
            i = jnp.floor(t)
            f = (t - i).astype(real_dtype)
            i_int = i.astype(jnp.int32)
            w0, dw0, d20 = AOs_bspline_data._bspline_weights_jax(f[0], order)
            w1, dw1, d21 = AOs_bspline_data._bspline_weights_jax(f[1], order)
            w2, dw2, d22 = AOs_bspline_data._bspline_weights_jax(f[2], order)
            cg = jnp.take(coeffs, i_int[0] + offsets, axis=1, mode="wrap")
            cg = jnp.take(cg, i_int[1] + offsets, axis=2, mode="wrap")
            cg = jnp.take(cg, i_int[2] + offsets, axis=3, mode="wrap")  # (n_band, K, K, K) complex
            # A single einsum mixing the complex control points ``cg`` with the REAL,
            # r-differentiable spline weights makes the weights' VJP cotangent complex,
            # which JAX casts to real (ComplexWarning: "Casting complex values to real
            # discards the imaginary part") whenever this forward is auto-differentiated
            # w.r.t. r/cell. Splitting cg into its real / imaginary parts and combining as
            # ``re + 1j*im`` keeps every einsum real-operand (real cotangents), moving the
            # real->complex boundary onto the ``1j * real`` multiply -- exactly the clean
            # structure the plane-wave leaf gets from ``exp(1j * theta)``. Bit-identical
            # values (einsum is linear in cg), holomorphic-clean gradients.
            cg_re, cg_im = jnp.real(cg), jnp.imag(cg)

            imag_unit = jnp.asarray(1j, dtype=complex_dtype)

            def _E(wa: jax.Array, wb: jax.Array, wc: jax.Array) -> jax.Array:
                if _bspline_axiswise_stencil():  # issue #527: never materialize the gathered stencil
                    return _bspline_axiswise_contraction(cg_re, wa, wb, wc) + imag_unit * _bspline_axiswise_contraction(
                        cg_im, wa, wb, wc
                    )
                return jnp.einsum("bpqr,p,q,r->b", cg_re, wa, wb, wc) + imag_unit * jnp.einsum(
                    "bpqr,p,q,r->b", cg_im, wa, wb, wc
                )

            val = _E(w0, w1, w2)
            dt0 = _E(dw0, w1, w2)
            dt1 = _E(w0, dw1, w2)
            dt2 = _E(w0, w1, dw2)
            h00 = _E(d20, w1, w2)
            h11 = _E(w0, d21, w2)
            h22 = _E(w0, w1, d22)
            h01 = _E(dw0, dw1, w2)
            h02 = _E(dw0, w1, dw2)
            h12 = _E(w0, dw1, dw2)
            grad_t = jnp.stack([dt0, dt1, dt2], axis=1)  # complex t-space gradient

            # Contract the complex spline fields with the REAL metric (inv_cell / a_mat) via the
            # real/imag split so the metric's VJP cotangent stays real -- otherwise a complex cotangent
            # on the real cell casts to real (ComplexWarning) whenever this forward is auto-diffed for
            # grad/lap (the J3 kinetic path uses these; the value-only geminal path DCE's them). The
            # combined value is bit-identical to ``(grad_t * ns) @ inv_cell.T`` / the a_mat contraction.
            def _re_im_lap(
                f00: jax.Array,
                f11: jax.Array,
                f22: jax.Array,
                f01: jax.Array,
                f02: jax.Array,
                f12: jax.Array,
            ) -> jax.Array:
                def _c(
                    g00: jax.Array,
                    g11: jax.Array,
                    g22: jax.Array,
                    g01: jax.Array,
                    g02: jax.Array,
                    g12: jax.Array,
                ) -> jax.Array:
                    return (
                        g00 * a_mat[0, 0]
                        + g11 * a_mat[1, 1]
                        + g22 * a_mat[2, 2]
                        + 2.0 * (g01 * a_mat[0, 1] + g02 * a_mat[0, 2] + g12 * a_mat[1, 2])
                    )

                re = _c(jnp.real(f00), jnp.real(f11), jnp.real(f22), jnp.real(f01), jnp.real(f02), jnp.real(f12))
                im = _c(jnp.imag(f00), jnp.imag(f11), jnp.imag(f22), jnp.imag(f01), jnp.imag(f02), jnp.imag(f12))
                return re + imag_unit * im

            grad_r = ((jnp.real(grad_t) * ns_vec[None, :]) @ inv_cell.T) + imag_unit * (
                (jnp.imag(grad_t) * ns_vec[None, :]) @ inv_cell.T
            )  # (n_band, 3) complex
            lap = _re_im_lap(h00, h11, h22, h01, h02, h12)
            return val, grad_r[:, 0], grad_r[:, 1], grad_r[:, 2], lap

        r_canonical = jnp.asarray(r_carts) if stable_geometry else jnp.asarray(r_carts, dtype=real_dtype)
        u, ux, uy, uz, ulap = jax.vmap(eval_one)(r_canonical)
        return u.T, ux.T, uy.T, uz.T, ulap.T  # each (num_ao, N_e) complex

    def _phase_in_zone(self, r_carts: jax.Array, zone: str) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return ``(phase, k_cart, i)`` in the selected component precision.

        ``phase`` has shape ``(1, N_e)`` for this single-k leaf.  The tiled
        subclass supplies one Cartesian k vector per AO, yielding
        ``(num_ao, N_e)`` without changing the product-rule kernels below.
        """
        real_dtype = get_dtype_jnp(zone)
        complex_dtype = get_complex_dtype_jnp(zone)
        k_cart = self._orbital_k_cart_jnp.astype(real_dtype)
        if real_dtype == jnp.float64:
            theta = jnp.asarray(r_carts, dtype=real_dtype) @ k_cart.T
        else:
            # k.r grows with the distance from the home cell. Form it in fp64
            # and reduce it modulo 2 pi before the fp32 cast.
            two_pi = 2.0 * jnp.pi
            theta = jnp.asarray(r_carts, dtype=jnp.float64) @ self._orbital_k_cart_jnp.T
            theta = (theta - two_pi * jnp.round(theta / two_pi)).astype(real_dtype)
        imag_unit = jnp.asarray(1j, dtype=complex_dtype)
        return jnp.exp(imag_unit * theta).T, k_cart, imag_unit

    @jit
    def _compute_AOs(self, r_carts: jax.Array) -> jax.Array:
        r"""``psi_k(r) = e^{i k . r} u_k(r)`` in the complex ``ao_eval`` dtype."""
        u, _, _, _, _ = self._eval_u_core(r_carts, "ao_eval")
        # ``e^{i k . r}`` per electron uses the FULL r (not wrapped) so the
        # Bloch phase across cells is reproduced exactly. Cell-periodicity of
        # ``u_k`` means u(r) = u(r_canonical), so the wrap inside _eval_u_core
        # is consistent with using r here.
        phase, _, _ = self._phase_in_zone(r_carts, "ao_eval")
        return u * phase

    def _compute_AOs_grad(self, r_carts: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        r"""``grad psi = e^{i k . r} (i k u + grad u)`` (product rule)."""
        zone = "ao_grad_lap"
        u, ux, uy, uz, _ = self._eval_u_core(r_carts, zone)
        phase, k_cart, imag_unit = self._phase_in_zone(r_carts, zone)
        gx = (ux + imag_unit * k_cart[:, 0, None] * u) * phase
        gy = (uy + imag_unit * k_cart[:, 1, None] * u) * phase
        gz = (uz + imag_unit * k_cart[:, 2, None] * u) * phase
        return gx, gy, gz

    def _compute_AOs_laplacian(self, r_carts: jax.Array) -> jax.Array:
        r"""``lap psi = e^{i k . r} (- |k|^2 u + 2 i k . grad u + lap u)``."""
        zone = "ao_grad_lap"
        u, ux, uy, uz, ulap = self._eval_u_core(r_carts, zone)
        phase, k_cart, imag_unit = self._phase_in_zone(r_carts, zone)
        k_dot_grad = k_cart[:, 0, None] * ux + k_cart[:, 1, None] * uy + k_cart[:, 2, None] * uz
        k2 = jnp.sum(k_cart * k_cart, axis=1)[:, None]
        return (ulap + (2.0 * imag_unit) * k_dot_grad - k2 * u) * phase

    def _compute_AOs_value_grad_lap(self, r_carts: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        zone = "ao_grad_lap"
        u, ux, uy, uz, ulap = self._eval_u_core(r_carts, zone)
        phase, k_cart, imag_unit = self._phase_in_zone(r_carts, zone)
        val = (u * phase).astype(get_complex_dtype_jnp("ao_eval"))
        gx = (ux + imag_unit * k_cart[:, 0, None] * u) * phase
        gy = (uy + imag_unit * k_cart[:, 1, None] * u) * phase
        gz = (uz + imag_unit * k_cart[:, 2, None] * u) * phase
        k_dot_grad = k_cart[:, 0, None] * ux + k_cart[:, 1, None] * uy + k_cart[:, 2, None] * uz
        k2 = jnp.sum(k_cart * k_cart, axis=1)[:, None]
        lap = (ulap + (2.0 * imag_unit) * k_dot_grad - k2 * u) * phase
        return val, gx, gy, gz, lap

    def _compute_AOs_debug(self, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        """Plain-numpy oracle for the complex-Bloch B-spline leaf."""
        r = np.asarray(r_carts, dtype=np.float64)
        dtype = get_dtype_np("ao_eval")
        complex_dtype = get_complex_dtype_np("ao_eval")
        order = self.spline_order
        ns = tuple(int(n) for n in self.spline_shape)
        n_band = self.num_ao
        coeffs = np.asarray(self._spline_coefficients_np, dtype=complex_dtype).reshape((n_band, *ns))
        cell = np.asarray(self.structure_data.cell, dtype=np.float64)
        inv_cell = np.linalg.inv(cell)
        k_cart = np.asarray(self.k_frac_np, dtype=np.float64) @ (2.0 * np.pi * inv_cell.T)
        # Independent NumPy geometry in the caller's float64: wrap, grid
        # coordinate and local offset; only the offset, the reduced Bloch
        # phase and the weight/contraction arithmetic use the AO value dtype.
        fractional = r @ inv_cell
        fractional = fractional - np.floor(fractional)
        start = -((order - 1) // 2)
        offsets = np.arange(start, start + order + 1)
        out = np.zeros((n_band, r.shape[0]), dtype=complex_dtype)
        for p in range(r.shape[0]):
            t = fractional[p] * np.asarray(ns, dtype=np.float64)
            floor_t = np.floor(t)
            i = floor_t.astype(int)
            f = (t - floor_t).astype(dtype)
            w0 = AOs_bspline_data._bspline_weights_np(f[0], order)
            w1 = AOs_bspline_data._bspline_weights_np(f[1], order)
            w2 = AOs_bspline_data._bspline_weights_np(f[2], order)
            idx0 = (i[0] + offsets) % ns[0]
            idx1 = (i[1] + offsets) % ns[1]
            idx2 = (i[2] + offsets) % ns[2]
            cg = coeffs[:, idx0][:, :, idx1][:, :, :, idx2]
            u_val = np.einsum("bpqr,p,q,r->b", cg, w0, w1, w2)
            theta = float(r[p] @ k_cart)
            theta = dtype(theta - 2.0 * np.pi * np.round(theta / (2.0 * np.pi)))
            phase = np.exp(complex_dtype(1j) * theta)
            out[:, p] = u_val * phase
        return out

    def _compute_overlap_matrix(self) -> npt.NDArray[np.complex128]:
        r"""Exact cell-domain overlap ``S_ab = int_cell conj(psi_a) psi_b dr``.

        Every AO of this leaf is ``psi_a(r) = e^{i k . r} u_a(r)`` with ONE
        leaf-level twist ``k`` (the static ``k_point``) and ``u_a`` evaluated
        on the cell-wrapped coordinate (:py:meth:`_compute_AOs`; there is no
        separate wrap-restore phase on this leaf). Since ``k`` is real,
        ``conj(psi_a) psi_b = conj(u_a) u_b`` pointwise, so the Bloch phase
        cancels and ``S`` is the Gram matrix of the periodic B-spline basis
        applied to the complex control points; see
        :py:func:`_compute_overlap_matrix_bspline_exact`. Returned as
        ``complex128`` (the ``det_eval`` overlap zone), independent of the
        AO-evaluation precision.
        """
        return _compute_overlap_matrix_bspline_exact(
            cell=np.asarray(self.structure_data.cell, dtype=np.float64),
            spline_coefficients=self._spline_coefficients_np,
            spline_shape=tuple(int(n) for n in self.spline_shape),
            spline_order=int(self.spline_order),
        )

    def _get_info(self) -> list[str]:
        info_lines: list[str] = []
        info_lines.append(f"**{self.__class__.__name__}")
        info_lines.append(f"  Number of AOs (bands) = {self.num_ao}")
        info_lines.append(f"  k_point (fractional) = {self.k_point}")
        info_lines.append(f"  spline grid shape (N0, N1, N2) = {tuple(int(n) for n in self.spline_shape)}")
        info_lines.append(f"  spline order (degree) = {self.spline_order}")
        return info_lines

    def _logger_info(self) -> None:
        for line in self._get_info():
            logger.info(line)

    @classmethod
    def build(
        cls,
        structure_data: Structure_data,
        spline_coefficients: npt.NDArray,
        spline_shape: tuple[int, int, int],
        spline_order: int = 5,
        *,
        k_point: tuple[float, float, float] | None = None,
    ) -> "AOs_bspline_complex_data":
        """Construct an :class:`AOs_bspline_complex_data` leaf at a non-Gamma twist.

        Accepts complex (or real) ``spline_coefficients`` and keeps them as
        complex128; no imag-part dropping happens here. The caller is
        responsible for dispatching on its k_point argument (Gamma ->
        :class:`AOs_bspline_data`; non-Gamma -> this class).
        """
        c = np.asarray(spline_coefficients)
        n_band = int(c.shape[0])
        n_pts = int(np.prod(spline_shape))
        c = c.reshape(n_band, n_pts).astype(np.complex128)
        data = cls(
            structure_data=structure_data,
            k_point=tuple(float(c0) for c0 in k_point) if k_point is not None else None,
            num_ao=n_band,
            spline_coefficients=c,
            spline_shape=tuple(int(n) for n in spline_shape),
            spline_order=int(spline_order),
        )
        data.sanity_check()
        return data


@struct.dataclass
class AOs_bspline_tiled_complex_data(AOs_bspline_complex_data):
    r"""Primitive-cell B-spline bands unfolded into a physical supercell.

    ``structure_data`` is the physical supercell; the primitive cell is derived
    in-trace so stress derivatives remain consistent with the Hamiltonian.

    ``k_points`` are fractional in the primitive reciprocal basis and
    ``orbital_k_indices[a]`` selects the k point of AO row ``a``. They must
    contain exactly ``|det(S)|`` distinct representatives and satisfy
    ``S @ k_primitive = k_supercell (mod Z^3)``. The builder applies the
    supercell normalization; direct construction is reserved for already
    supercell-normalized tables.

    This representation is a frozen orbital basis. Control points are stopped
    at evaluation time; electron and cell derivatives remain active, while
    nodal optimization of the spline coefficients is not exposed.
    """

    tiling_matrix: tuple[tuple[int, int, int], ...] = struct.field(
        pytree_node=False,
        default_factory=lambda: ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    )
    k_points: tuple[tuple[float, float, float], ...] = struct.field(pytree_node=False, default_factory=tuple)
    orbital_k_indices: tuple[int, ...] = struct.field(pytree_node=False, default_factory=tuple)

    @property
    def tiling_determinant(self) -> int:
        """Number of primitive cells in the supercell."""
        return abs(int(round(np.linalg.det(np.asarray(self.tiling_matrix, dtype=np.int64)))))

    @property
    def primitive_cell_np(self) -> npt.NDArray[np.float64]:
        """Primitive lattice rows, ``inv(S) @ supercell`` (Bohr)."""
        return np.linalg.solve(
            np.asarray(self.tiling_matrix, dtype=np.float64),
            np.asarray(self.structure_data.cell, dtype=np.float64),
        )

    @property
    def _spline_cell_jnp(self) -> jax.Array:
        """JAX-active primitive cell derived from the physical supercell."""
        matrix = jnp.asarray(self.tiling_matrix, dtype=self.structure_data._cell_jnp.dtype)
        return _inv_3x3(matrix) @ self.structure_data._cell_jnp

    @property
    def _orbital_k_cart_jnp(self) -> jax.Array:
        """One primitive-cell Cartesian Bloch vector per AO row."""
        primitive_cell = self._spline_cell_jnp
        reciprocal = 2.0 * jnp.pi * _inv_3x3(primitive_cell).T
        k_table = jnp.asarray(self.k_points, dtype=jnp.float64) @ reciprocal
        indices = jnp.asarray(self.orbital_k_indices, dtype=jnp.int32)
        return k_table[indices]

    @property
    def orbital_k_cart_np(self) -> npt.NDArray[np.float64]:
        """NumPy mirror of :attr:`_orbital_k_cart_jnp` for native kernels."""
        reciprocal = 2.0 * np.pi * np.linalg.inv(self.primitive_cell_np).T
        table = np.asarray(self.k_points, dtype=np.float64) @ reciprocal
        return table[np.asarray(self.orbital_k_indices, dtype=np.int64)]

    def sanity_check(self) -> None:
        # The inherited validation owns the common coefficient/grid contract.
        super().sanity_check()
        matrix = np.asarray(self.tiling_matrix)
        if matrix.shape != (3, 3) or not np.issubdtype(matrix.dtype, np.integer):
            raise ValueError(f"tiling_matrix must be an integer 3x3 matrix; got shape={matrix.shape}, dtype={matrix.dtype}.")
        det_signed = int(round(np.linalg.det(matrix.astype(np.float64))))
        if det_signed == 0:
            raise ValueError("tiling_matrix must be nonsingular.")
        if not np.isclose(
            np.linalg.det(matrix.astype(np.float64)),
            det_signed,
            rtol=_setting.RTOL_BIT_EXACT,
            atol=ATOL_TILING_INTEGER_RESIDUE,
        ):
            raise ValueError("tiling_matrix determinant is not an exact integer within roundoff.")
        det = abs(det_signed)
        if len(self.k_points) != det:
            raise ValueError(f"k_points must contain |det(tiling_matrix)|={det} primitive twists; got {len(self.k_points)}.")
        k_points = np.asarray(self.k_points, dtype=np.float64)
        if k_points.shape != (det, 3) or not np.all(np.isfinite(k_points)):
            raise ValueError(f"k_points must be finite with shape ({det}, 3); got {k_points.shape}.")
        if len(self.orbital_k_indices) != self.num_ao:
            raise ValueError(
                f"orbital_k_indices must contain one entry per AO ({self.num_ao}); got {len(self.orbital_k_indices)}."
            )
        indices = np.asarray(self.orbital_k_indices, dtype=np.int64)
        if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= det):
            raise ValueError(f"orbital_k_indices entries must lie in [0, {det}); got {self.orbital_k_indices}.")
        if set(indices.tolist()) != set(range(det)):
            raise ValueError("every primitive k point must own at least one tiled orbital row.")

        # Distinct representatives modulo the primitive reciprocal lattice.
        residues = k_points - np.floor(k_points)
        for i in range(det):
            for j in range(i):
                difference = residues[i] - residues[j]
                if float(np.max(np.abs(difference - np.rint(difference)))) <= ATOL_TWIST_CHECKPOINT_KPOINT_MATCH:
                    raise ValueError(f"k_points[{i}] and k_points[{j}] are equivalent modulo an integer reciprocal vector.")

        super_twist = np.asarray(self.k_point, dtype=np.float64)
        unfolded = (matrix @ k_points.T).T - super_twist[None, :]
        mismatch = unfolded - np.rint(unfolded)
        if float(np.max(np.abs(mismatch))) > ATOL_TWIST_CHECKPOINT_KPOINT_MATCH:
            raise ValueError(
                "primitive k points do not unfold to the declared supercell k_point: "
                "require S @ k_primitive = k_supercell (mod integer reciprocal vectors)."
            )

    def with_negated_k(self) -> Self:
        """Return the time-reversal sibling without expanding the spline grid."""
        return self.replace(
            k_point=tuple(-float(value) for value in self.k_point),
            k_points=tuple(tuple(-float(value) for value in point) for point in self.k_points),
            conjugate_coefficients=not self.conjugate_coefficients,
        )

    def _compute_AOs_debug(self, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
        """Plain-NumPy forward oracle using primitive wrapping and per-AO k."""
        r = np.asarray(r_carts, dtype=np.float64)
        dtype = get_dtype_np("ao_eval")
        complex_dtype = get_complex_dtype_np("ao_eval")
        order = self.spline_order
        ns = tuple(int(n) for n in self.spline_shape)
        coeffs = np.asarray(self._spline_coefficients_np, dtype=complex_dtype).reshape((self.num_ao, *ns))
        inv_cell = np.linalg.inv(self.primitive_cell_np)
        k_cart = self.orbital_k_cart_np
        start = -((order - 1) // 2)
        offsets = np.arange(start, start + order + 1)
        out = np.zeros((self.num_ao, r.shape[0]), dtype=complex_dtype)
        for electron, position in enumerate(r):
            # float64 primitive wrap, grid coordinate and local offset; only the
            # offset, the reduced phase and the value arithmetic use the AO dtype.
            frac = position @ inv_cell
            frac -= np.floor(frac)
            t = frac * np.asarray(ns, dtype=np.float64)
            floor_t = np.floor(t)
            index = floor_t.astype(int)
            f = (t - floor_t).astype(dtype)
            weights = [AOs_bspline_data._bspline_weights_np(f[axis], order) for axis in range(3)]
            grid_indices = [(index[axis] + offsets) % ns[axis] for axis in range(3)]
            local = coeffs[:, grid_indices[0]][:, :, grid_indices[1]][:, :, :, grid_indices[2]]
            periodic = np.einsum("bpqr,p,q,r->b", local, weights[0], weights[1], weights[2])
            theta = k_cart @ position
            theta = (theta - 2.0 * np.pi * np.round(theta / (2.0 * np.pi))).astype(dtype)
            phase = np.exp(complex_dtype(1j) * theta)
            out[:, electron] = periodic * phase
        return out

    def _compute_overlap_matrix(self) -> npt.NDArray[np.complex128]:
        r"""Exact supercell Gram matrix of the unfolded spline orbitals.

        Different primitive k representatives are exactly orthogonal over the
        supercell.  Within a k block the primitive-cell spline Gram is
        multiplied by ``|det(S)|``; the builder's ``1/sqrt(|det(S)|)``
        coefficient normalization makes unit-normalized primitive bands stay
        unit normalized in the supercell.
        """
        overlap = np.zeros((self.num_ao, self.num_ao), dtype=np.complex128)
        indices = np.asarray(self.orbital_k_indices, dtype=np.int64)
        coefficients = self._spline_coefficients_np
        for k_index in range(len(self.k_points)):
            rows = np.flatnonzero(indices == k_index)
            block = _compute_overlap_matrix_bspline_exact(
                cell=self.primitive_cell_np,
                spline_coefficients=coefficients[rows],
                spline_shape=tuple(int(n) for n in self.spline_shape),
                spline_order=int(self.spline_order),
            )
            overlap[np.ix_(rows, rows)] = float(self.tiling_determinant) * block
        return overlap

    def _get_info(self) -> list[str]:
        stored = int(self.num_ao) * int(np.prod(self.spline_shape))
        return [
            f"**{self.__class__.__name__}",
            f"  Number of unfolded AOs (bands) = {self.num_ao}",
            f"  tiling matrix S = {self.tiling_matrix}",
            f"  primitive cells |det(S)| = {self.tiling_determinant}",
            f"  supercell k_point (fractional) = {self.k_point}",
            f"  primitive k_points = {self.k_points}",
            f"  primitive spline grid shape = {tuple(int(n) for n in self.spline_shape)}",
            f"  spline order (degree) = {self.spline_order}",
            f"  stored coefficients = {stored} (about {self.tiling_determinant}x fewer than a same-density explicit supercell table)",
        ]

    @classmethod
    def build(
        cls,
        structure_data: Structure_data,
        spline_coefficients_by_k: Sequence[npt.ArrayLike],
        spline_shape: tuple[int, int, int],
        tiling_matrix: npt.ArrayLike,
        k_points: Sequence[Sequence[float]],
        spline_order: int = 5,
        *,
        supercell_k_point: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> Self:
        """Build a supercell-normalized tiled leaf from primitive k blocks.

        Each coefficient block has shape ``(num_band_at_k, prod(shape))`` (or
        its cube form) and is assumed normalized in one primitive cell.  The
        stored rows are divided by ``sqrt(|det(S)|)`` exactly once.
        """
        matrix = normalize_tiling_matrix(tiling_matrix)
        det = abs(int(round(np.linalg.det(matrix.astype(np.float64)))))
        if len(spline_coefficients_by_k) != len(k_points):
            raise ValueError("spline_coefficients_by_k and k_points must have the same length.")
        n_points = int(np.prod(spline_shape))
        blocks: list[np.ndarray] = []
        orbital_k_indices: list[int] = []
        for k_index, block_like in enumerate(spline_coefficients_by_k):
            block = np.asarray(block_like, dtype=np.complex128)
            if block.ndim < 2:
                raise ValueError(f"spline coefficient block {k_index} must have a band axis and grid axes.")
            block = block.reshape(block.shape[0], -1)
            if block.shape[1] != n_points:
                raise ValueError(
                    f"spline coefficient block {k_index} has {block.shape[1]} grid values per band; expected {n_points}."
                )
            blocks.append(block / np.sqrt(float(det)))
            orbital_k_indices.extend([k_index] * block.shape[0])
        coefficients = np.ascontiguousarray(np.concatenate(blocks, axis=0), dtype=np.complex128)
        data = cls(
            structure_data=structure_data,
            k_point=tuple(float(value) for value in supercell_k_point),
            num_ao=int(coefficients.shape[0]),
            spline_coefficients=coefficients,
            spline_shape=tuple(int(n) for n in spline_shape),
            spline_order=int(spline_order),
            tiling_matrix=tuple(tuple(int(value) for value in row) for row in matrix),
            k_points=tuple(tuple(float(value) for value in point) for point in k_points),
            orbital_k_indices=tuple(orbital_k_indices),
        )
        data.sanity_check()
        return data


def _compute_AOs_bspline_debug(aos_data: AOs_bspline_data, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Plain-numpy forward oracle for :class:`AOs_bspline_data`.

    Readable, per-electron numpy version of the separable evaluation in
    ``AOs_bspline_data.compute_AOs``: map ``r`` to fractional coordinates
    ``frac = r . cell^-1`` (wrapped into ``[0, 1)``), scale by the grid
    ``t = frac N``, then contract the ``order + 1`` local control points along
    each axis with the cardinal B-spline weights. Returns ``(num_ao, N_e)``.
    """
    r = np.asarray(r_carts, dtype=np.float64)  # (N_e, 3)
    dtype = get_dtype_np("ao_eval")
    order = aos_data.spline_order
    ns = tuple(int(n) for n in aos_data.spline_shape)
    n_band = aos_data.num_ao
    coeffs = np.asarray(aos_data.spline_coefficients, dtype=dtype).reshape((n_band, *ns))
    cell = np.asarray(aos_data.structure_data.cell, dtype=np.float64)
    fractional = r @ np.linalg.inv(cell)
    stable_geometry = dtype == np.float32 and jax.default_backend() == "cpu"
    canonical = ((fractional - np.floor(fractional)) @ cell).astype(dtype)
    inv_cell = np.linalg.inv(cell.astype(dtype))
    start = -((order - 1) // 2)  # -1 (cubic) / -2 (quintic)
    offsets = np.arange(start, start + order + 1)  # (order+1,)
    out = np.zeros((n_band, r.shape[0]), dtype=dtype)
    for p in range(r.shape[0]):
        if stable_geometry:
            # Independently reconstruct the local offset with NumPy geometry;
            # the weight polynomials and einsum still use the AO value dtype.
            frac = fractional[p] - np.floor(fractional[p])
            t = frac * np.asarray(ns, dtype=cell.dtype)
        else:
            frac = canonical[p] @ inv_cell
            frac -= np.floor(frac)
            t = frac * np.asarray(ns, dtype=dtype)
        floor_t = np.floor(t)
        i = floor_t.astype(int)
        f = (t - floor_t).astype(dtype)
        w0 = AOs_bspline_data._bspline_weights_np(f[0], order)
        w1 = AOs_bspline_data._bspline_weights_np(f[1], order)
        w2 = AOs_bspline_data._bspline_weights_np(f[2], order)
        idx0 = (i[0] + offsets) % ns[0]
        idx1 = (i[1] + offsets) % ns[1]
        idx2 = (i[2] + offsets) % ns[2]
        cg = coeffs[:, idx0][:, :, idx1][:, :, :, idx2]  # (n_band, K, K, K)
        out[:, p] = np.einsum("bpqr,p,q,r->b", cg, w0, w1, w2)
    return out


def _preserve_complex_pgto_kind(
    source: AOs_GTO_base_data,
    converted: AOs_GTO_base_data,
) -> AOs_GTO_base_data:
    """Reattach Bloch metadata after a real angular-basis conversion.

    Spherical/Cartesian conversion changes only the angular basis.  A complex
    periodic source must therefore remain a complex periodic leaf at the same
    twist.  Rebuilding the phases from the converted leaf's image layout also
    handles per-shell ragged layouts whose kernel ordering changes.
    """
    if isinstance(source, (AOs_cart_pgto_complex_data, AOs_sphe_pgto_complex_data)):
        if not isinstance(converted, (AOs_cart_pgto_data, AOs_sphe_pgto_data)):
            raise TypeError(
                f"Complex Bloch basis conversion must produce a periodic pGTO leaf; got {type(converted).__name__}."
            )
        return _promote_pgto_to_complex(converted, source.k_point)
    return converted


def _aos_sphe_to_cart(
    aos_data: AOs_sphe_gto_data | AOs_cart_gto_data,
) -> tuple[AOs_cart_gto_data | AOs_cart_pgto_data, np.ndarray]:
    """Convert spherical AOs to Cartesian AOs and return the transform matrix.

    Returns:
        tuple: (AOs_cart_gto_data, transform_matrix) where transform_matrix maps
        spherical -> Cartesian coefficients with shape (num_ao_sph, num_ao_cart).
    """
    # I/O / setup-time basis conversion: hardcode fp64 (no precision-zone
    # involvement). The transform matrix carries pure mathematical conversion
    # constants and feeds the MO coefficient transform; downcasting it to a
    # mixed-mode fp32 zone leaks fp32 noise into fp64 mo_coefficients.
    dtype_np = np.float64
    if isinstance(aos_data, AOs_cart_gto_data):
        transform_matrix = np.eye(aos_data.num_ao, dtype=dtype_np)
        return aos_data, transform_matrix
    if not isinstance(aos_data, AOs_sphe_gto_data):
        raise ValueError("Cartesian conversion is only available from spherical AOs.")

    # Traced-ham bridge: when the leaf's radial exponents are tracers (production R/eps force/stress
    # ``jit(vmap(...))`` over the whole hamiltonian), the host-side shell grouping / image truncation
    # below would ``np.asarray`` a tracer and raise ``TracerArrayConversionError``. Route to the
    # jit-safe structural twin (bit-identical on the concrete/eager path). Dispatching HERE covers
    # EVERY caller (``sphe_leaf_to_cart``, ``ao_deriv.orb_to_cart_pieces``, ``molecular_orbital``,
    # ``determinant``, ...), not just the pgto force/stress entry.
    if _radial_is_traced(aos_data):
        aos_cart, transform_matrix = _aos_sphe_to_cart_traced(aos_data)
        return _preserve_complex_pgto_kind(aos_data, aos_cart), transform_matrix

    if isinstance(aos_data, AOs_sphe_pgto_data) and aos_data._ragged_buckets:
        # Per-shell ragged periodic: convert at the MOLECULAR level (the
        # kernel-rank shell regrouping below assumes a uniform tile and the
        # identity nucleus_index breaks per-shell matching), then re-expand the
        # cart base to a ragged cart leaf. The returned transform is the
        # molecular sphe->cart block -- exactly the ``T_mol`` the overlap fusion
        # uses; the ragged cart leaf supplies the cart kernel + its fold.
        mol_sphe = _detile_ragged_to_molecular(aos_data)
        mol_cart, transform_matrix = _aos_sphe_to_cart(mol_sphe)
        ragged_cart = _apply_image_expansion(mol_cart, aos_data.structure_data, tol=float(aos_data._image_truncation_tol))
        ragged_cart = _preserve_complex_pgto_kind(aos_data, ragged_cart)
        return ragged_cart, transform_matrix

    aos_sphe = aos_data
    shells: list[dict] = []
    shells_by_identity: dict[tuple[int, int, tuple[int, ...]], dict] = {}
    physical_shells = bool(aos_sphe.ao_coefficient_scales)

    def _match_shell(existing: dict, nucleus: int, l: int, exps: np.ndarray, coefs: np.ndarray) -> bool:
        return (
            existing["nucleus"] == nucleus
            and existing["l"] == l
            and len(existing["exps"]) == len(exps)
            and np.allclose(existing["exps"], exps, rtol=rtol, atol=atol)
            and np.allclose(existing["coefs"], coefs, rtol=rtol, atol=atol)
        )

    # Iterate over the kernel-rank AO count (_num_ao_kernel) so that pgto
    # inputs (whose per-AO arrays are tiled) are processed correctly. For
    # gto inputs _num_ao_kernel == num_ao, so behaviour is unchanged.
    ao_exponents = aos_sphe.exponents
    ao_coefficients = aos_sphe.coefficients
    for ao_idx in range(aos_sphe._num_ao_kernel):
        prim_indices = [i for i, v in enumerate(aos_sphe.orbital_indices) if v == ao_idx]
        exps = ao_exponents[prim_indices]
        coefs = ao_coefficients[prim_indices]
        nucleus = aos_sphe.nucleus_index[ao_idx]
        l = aos_sphe.angular_momentums[ao_idx]

        if physical_shells:
            radial_indices = tuple(aos_sphe.ao_prim_to_shell_prim[i] for i in prim_indices)
            identity = (nucleus, l, radial_indices)
            shell = shells_by_identity.get(identity)
            if shell is None:
                shell = dict(nucleus=nucleus, l=l, exps=exps, coefs=coefs, ao_indices=[], radial_indices=radial_indices)
                shells.append(shell)
                shells_by_identity[identity] = shell
            shell["ao_indices"].append(ao_idx)
            continue

        for shell in shells:
            if _match_shell(shell, nucleus, l, exps, coefs):
                shell["ao_indices"].append(ao_idx)
                break
        else:
            shells.append({"nucleus": nucleus, "l": l, "exps": exps, "coefs": coefs, "ao_indices": [ao_idx]})

    total_cart = sum((shell["l"] + 1) * (shell["l"] + 2) // 2 for shell in shells)
    total_sph = aos_sphe._num_ao_kernel
    transform_matrix = np.zeros((total_sph, total_cart), dtype=dtype_np)

    new_nucleus_index: list[int] = []
    new_angular_momentums: list[int] = []
    new_polynominal_order_x: list[int] = []
    new_polynominal_order_y: list[int] = []
    new_polynominal_order_z: list[int] = []
    new_orbital_indices: list[int] = []
    new_exponents: list[float] = []
    new_coefficients: list[float] = []
    new_radial_indices: list[int] = []

    cart_col_start = 0
    for shell in shells:
        l = shell["l"]
        ao_indices = shell["ao_indices"]
        if len(ao_indices) != 2 * l + 1:
            raise ValueError(f"Shell with l={l} at nucleus {shell['nucleus']} is incomplete: {len(ao_indices)} AOs found.")

        orders = [(nx, ny, l - nx - ny) for nx in range(l, -1, -1) for ny in range(l - nx, -1, -1)]
        n_cart = len(orders)
        cart_indices = list(range(cart_col_start, cart_col_start + n_cart))
        cart_col_start += n_cart

        shell_sph_indices = sorted(ao_indices, key=lambda idx: aos_sphe.magnetic_quantum_numbers[idx])
        transform_matrix[np.ix_(shell_sph_indices, cart_indices)] = _spherical_to_cart_matrix(l)

        for nx, ny, nz in orders:
            ao_new_idx = len(new_nucleus_index)
            new_nucleus_index.append(shell["nucleus"])
            new_angular_momentums.append(l)
            new_polynominal_order_x.append(nx)
            new_polynominal_order_y.append(ny)
            new_polynominal_order_z.append(nz)
            if physical_shells:
                new_radial_indices.extend(shell["radial_indices"])

            for exp, coef in zip(shell["exps"], shell["coefs"], strict=True):
                new_orbital_indices.append(ao_new_idx)
                new_exponents.append(float(exp))
                new_coefficients.append(float(coef))

    common_kwargs = dict(
        structure_data=aos_sphe.structure_data,
        nucleus_index=new_nucleus_index,
        num_ao_prim=len(new_exponents),
        orbital_indices=new_orbital_indices,
        # Emit at the function's setup-time fp64 (see dtype_np above): an
        # "ao_eval"-zone cast here would round the radial values through fp32
        # under mixed precision before the fp64 shell-leaf storage.
        exponents=np.array(new_exponents, dtype=dtype_np),
        coefficients=np.array(new_coefficients, dtype=dtype_np),
        angular_momentums=new_angular_momentums,
        polynominal_order_x=new_polynominal_order_x,
        polynominal_order_y=new_polynominal_order_y,
        polynominal_order_z=new_polynominal_order_z,
    )
    if physical_shells:
        common_kwargs.update(
            radial=aos_sphe.radial,
            ao_prim_to_shell_prim=new_radial_indices,
            ao_coefficient_scales=(1.0,) * total_cart,
        )

    if isinstance(aos_sphe, AOs_sphe_pgto_data):
        # Keep the periodic pgto layout in the converted cart leaf. The per-AO
        # arrays above are kernel-rank (image-tiled), while num_ao stores the
        # molecular view and _num_ao_kernel is reconstructed from
        # _n_image * num_ao.
        n_image = int(aos_sphe._n_image)
        if n_image <= 0 or total_cart % n_image != 0:
            raise RuntimeError(
                "_aos_sphe_to_cart: periodic shell conversion produced an invalid "
                f"cart rank (total_cart={total_cart}, _n_image={n_image})."
            )
        num_ao_mol_cart = total_cart // n_image
        aos_cart: AOs_cart_gto_data | AOs_cart_pgto_data = AOs_cart_pgto_data._from_full_basis(
            **common_kwargs,
            num_ao=num_ao_mol_cart,
            _n_image=n_image,
            _image_truncation_tol=float(aos_sphe._image_truncation_tol),
            _image_indices=tuple(aos_sphe._image_indices),
        )
    else:
        aos_cart = AOs_cart_gto_data._from_full_basis(
            **common_kwargs,
            num_ao=total_cart,
        )

    return _preserve_complex_pgto_kind(aos_data, aos_cart), transform_matrix


def _radial_is_traced(aos_data: AOs_GTO_base_data) -> bool:
    """True iff the leaf's radial exponents are a JAX tracer (basis leaf traced under the
    production ``jit(vmap)``-over-the-hamiltonian bridge). The host-side ``_aos_sphe_to_cart``
    reads ``radial.exponents`` as concrete numpy (shell grouping + image truncation), which
    raises ``TracerArrayConversionError`` when the leaf is traced; ``_aos_sphe_to_cart_traced``
    is the jit-safe twin for that case."""
    rad = getattr(aos_data, "radial", None)
    exps = None if rad is None else getattr(rad, "exponents", None)
    return isinstance(exps, jax.core.Tracer)


def _ragged_structural_shell_proxy(aos_data: AOs_sphe_pgto_data) -> np.ndarray:
    """Structural proxy radial values (one per shell-primitive) for a RAGGED periodic pgto leaf.

    A ragged pgto leaf carries an identity ``nucleus_index = range(M_kernel)``
    for kernel centres but retains one physical home-cell radial shared by
    angular components and images. The structural primitive gather, rather
    than kernel-centre identity or current radial values, owns this sharing.
    This proxy also supports older explicitly constructed expanded leaves.

    The shell grouping (which home AOs share a radial) is precomputed from the
    canonical molecular primitive gather, independently of current parameter values
    (:func:`_home_shell_rep_from_molecular`, stored in ``_ragged_home_shell_rep``) -- so it is correct
    even for a NON-shell-major (QE-converted) basis whose shells' magnetic components are INTERLEAVED.
    Every AO's proxy radial is set to its shell REPRESENTATIVE's source shell-primitive indices. This
    (a) makes all AOs of a shell share a radial (correct grouping) and (b) leaves each Cartesian
    shell-primitive carrying the SOURCE shell-primitive index, so the real (traced) radial is restored
    by ``radial.exponents[proxy]``. Value-independent: uses no (then-traced) exponents. If the static
    map is absent (a leaf built without it), fall back to the shell-major contiguity walk."""
    apsp = np.asarray(aos_data.ao_prim_to_shell_prim, dtype=np.int64)
    n_shell_prim = int(np.asarray(aos_data.radial.shell_prim_to_shell).shape[0])
    orb_k = np.asarray(aos_data.orbital_indices, dtype=np.int64)  # kernel-primitive -> kernel-AO
    M_k = int(aos_data._num_ao_kernel)
    prims_of: list[list[int]] = [[] for _ in range(M_k)]
    for i in range(orb_k.shape[0]):
        prims_of[int(orb_k[i])].append(i)

    kimg = np.asarray(aos_data._image_indices, dtype=np.int64).reshape(-1, 3)
    kth = np.asarray(aos_data._ragged_kernel_to_home, dtype=np.int64)  # kernel-AO -> home (molecular) AO
    M_home = int(aos_data.num_ao)
    is_t0 = (kimg == 0).all(axis=1)
    rep = np.full(M_home, -1, dtype=np.int64)  # home AO -> its T_0 kernel-AO
    rep[kth[is_t0]] = np.nonzero(is_t0)[0]
    if (rep < 0).any():
        raise RuntimeError("_ragged_structural_shell_proxy: a home AO has no T_0 kernel-AO.")

    # Home AO -> its shell's representative home AO. Prefer the precomputed order-independent map
    # (supports non-shell-major bases); else fall back to the shell-major contiguity walk.
    home_shell_rep = tuple(aos_data._ragged_home_shell_rep)
    if home_shell_rep:
        shell_rep = np.asarray(home_shell_rep, dtype=np.int64)
        if shell_rep.shape[0] != M_home:
            raise RuntimeError("_ragged_structural_shell_proxy: _ragged_home_shell_rep length mismatch.")
    else:
        katom = np.asarray(aos_data._ragged_kernel_atom, dtype=np.int64)
        ang = np.asarray(aos_data.angular_momentums, dtype=np.int64)
        mag = np.asarray(aos_data.magnetic_quantum_numbers, dtype=np.int64)
        atom_h = katom[rep]
        l_h = ang[rep]
        m_h = mag[rep]
        # Structural shells over the home-AO order (shell-major layout): a shell is a maximal run of
        # same-(atom, l) home AOs with DISTINCT magnetic numbers, closing at 2l+1 members.
        shell_rep = np.empty(M_home, dtype=np.int64)
        start = 0
        while start < M_home:
            a0, l0 = int(atom_h[start]), int(l_h[start])
            need = 2 * l0 + 1
            seen: set[int] = set()
            j = start
            while j < M_home and int(atom_h[j]) == a0 and int(l_h[j]) == l0 and int(m_h[j]) not in seen and (j - start) < need:
                seen.add(int(m_h[j]))
                j += 1
            if (j - start) != need:
                raise RuntimeError(
                    f"_ragged_structural_shell_proxy: incomplete shell (atom={a0}, l={l0}) -- expected {need} "
                    f"contiguous m-components, found {j - start}. Non-shell-major basis layout is unsupported."
                )
            shell_rep[start:j] = start
            start = j

    proxy = np.arange(n_shell_prim, dtype=np.float64)  # identity default (covers any stray shell-prim)
    for a in range(M_k):
        h = int(kth[a])
        rep_kao = int(rep[int(shell_rep[h])])  # T_0 kernel-AO of this shell's representative home
        pa, pr = prims_of[a], prims_of[rep_kao]
        if len(pa) != len(pr):
            raise RuntimeError("_ragged_structural_shell_proxy: contraction length mismatch within a shell.")
        for k in range(len(pa)):
            proxy[int(apsp[pa[k]])] = float(apsp[pr[k]])
    return proxy


def _aos_sphe_to_cart_traced(
    aos_data: AOs_sphe_gto_data | AOs_sphe_pgto_data,
) -> tuple[AOs_cart_gto_data | AOs_cart_pgto_data, np.ndarray]:
    """jit-safe twin of :func:`_aos_sphe_to_cart` for a leaf whose radial exponents are TRACED
    (the production R/eps force/stress bridge traces the whole hamiltonian, so the basis
    exponents become tracers -- the host ``np.asarray(exponents)`` grouping / image-truncation
    of ``_aos_sphe_to_cart`` then raises ``TracerArrayConversionError``).

    The spherical->Cartesian STRUCTURE (shell grouping, the ``_spherical_to_cart_matrix``
    transform ``T``, the Cartesian angular layout, and -- for a periodic leaf -- the per-shell
    ragged image layout) is a function of the STATIC basis only: it is position/strain- AND
    value-independent. So we run the concrete host conversion on a structural PROXY whose radial
    values are the shell-primitive INDICES (which reproduce the exact shell grouping, since
    ``ao_prim_to_shell_prim`` IS that grouping) and, for a periodic leaf, REUSE the input leaf's
    own (correct, static) image sets instead of recomputing them from the traced exponents.
    The proxy's radial VALUES then carry the source shell-primitive index at every Cartesian
    shell-primitive, so the real (traced) radial is restored by a single ``jnp`` gather -- the
    transform ``T`` and the Cartesian layout are exact constants, the exponents flow as tracers.

    Bit-identical to :func:`_aos_sphe_to_cart` on the concrete path (the caller only routes here
    when the leaf is traced); the eager path is untouched. Complex-twist leaves are handled as
    their real base here (identical to the eager path, which promotes afterwards)."""
    if not isinstance(aos_data, AOs_sphe_gto_data):
        raise ValueError("_aos_sphe_to_cart_traced is only available from spherical AOs.")
    radial = aos_data.radial
    n_shell_prim = int(np.asarray(radial.shell_prim_to_shell).shape[0])
    is_ragged_pgto = isinstance(aos_data, AOs_sphe_pgto_data) and bool(aos_data._ragged_buckets)

    # Proxy radial VALUES that reproduce the concrete shell grouping (value-independent). For a
    # molecular sphe-gto leaf the radial is already de-duplicated per shell, so the shell-prim
    # INDEX itself is a valid grouping key. The ragged structural proxy also
    # supports older expanded leaves with independent kernel-radial entries.
    proxy_vals = _ragged_structural_shell_proxy(aos_data) if is_ragged_pgto else np.arange(n_shell_prim, dtype=np.float64)
    proxy_radial = radial.replace(exponents=proxy_vals.copy(), coefficients=proxy_vals.copy())
    aos_proxy = aos_data.replace(radial=proxy_radial)

    if not is_ragged_pgto:
        # Molecular sphe-gto (no image layout): the host conversion is purely structural given
        # the proxy grouping. Run it, then restore the traced radial by the proxy->source gather.
        cart_proxy, transform_matrix = _aos_sphe_to_cart(aos_proxy)
        idx = np.asarray(cart_proxy.radial.exponents).astype(np.int64)  # proxy value == source shell-prim
        cart = cart_proxy.replace(
            radial=cart_proxy.radial.replace(
                exponents=jnp.asarray(radial.exponents)[idx],
                coefficients=jnp.asarray(radial.coefficients)[idx],
            )
        )
        return cart, transform_matrix

    # --- Periodic ragged sphe-pgto: molecular structure + T (proxy), image layout REUSED. ---
    mol_sphe = _detile_ragged_to_molecular(aos_proxy)  # AOs_sphe_gto_data, proxy radial, correct structure
    mol_cart, transform_matrix = _aos_sphe_to_cart(mol_sphe)  # molecular c2s -> cart base (proxy) + T

    # Per-(sphe) molecular-AO image set, extracted from the INPUT leaf's static ragged layout.
    kimg = np.asarray(aos_data._image_indices, dtype=np.int64).reshape(-1, 3)  # (M_kernel, 3)
    kth = np.asarray(aos_data._ragged_kernel_to_home, dtype=np.int64)  # (M_kernel,) kernel-AO -> home
    M_sphe = int(aos_data.num_ao)
    sphe_ao_images: list[set] = [set() for _ in range(M_sphe)]
    for k in range(kth.shape[0]):
        sphe_ao_images[int(kth[k])].add((int(kimg[k, 0]), int(kimg[k, 1]), int(kimg[k, 2])))

    # Shell signature (nucleus, l, proxy-radial-tuple) uniquely identifies a shell (radial); the
    # image set is shared by all angular AOs of the shell, so match cart AOs to sphe shells by it.
    def _shell_sig(
        mol: AOs_sphe_gto_data | AOs_cart_gto_data,
        ao: int,
    ) -> tuple[int, int, tuple[int, ...]]:
        orb = np.asarray(mol.orbital_indices, dtype=np.int64)
        exps = np.asarray(mol.exponents, dtype=np.float64)
        pr = np.nonzero(orb == ao)[0]
        return (
            int(np.asarray(mol.nucleus_index)[ao]),
            int(np.asarray(mol.angular_momentums)[ao]),
            tuple(int(round(float(exps[p]))) for p in pr),
        )

    sig_to_images: dict[tuple, set] = {}
    for ao in range(M_sphe):
        sig_to_images[_shell_sig(mol_sphe, ao)] = sphe_ao_images[ao]
    M_cart = int(mol_cart.num_ao)
    ao_images_cart = [sig_to_images[_shell_sig(mol_cart, ao)] for ao in range(M_cart)]

    common_kwargs, kth_cart = _build_image_expanded_arrays_ragged(
        mol_cart, aos_data.structure_data, float(aos_data._image_truncation_tol), ao_images_override=ao_images_cart
    )
    px = mol_cart.polynominal_order_x
    py = mol_cart.polynominal_order_y
    pz = mol_cart.polynominal_order_z
    cart_proxy = AOs_cart_pgto_data._from_full_basis(
        **common_kwargs,
        polynominal_order_x=tuple(int(px[h]) for h in kth_cart),
        polynominal_order_y=tuple(int(py[h]) for h in kth_cart),
        polynominal_order_z=tuple(int(pz[h]) for h in kth_cart),
        _image_truncation_tol=float(aos_data._image_truncation_tol),
    )
    idx = np.asarray(cart_proxy.radial.exponents).astype(np.int64)  # proxy value == source shell-prim index
    cart = cart_proxy.replace(
        radial=cart_proxy.radial.replace(
            exponents=jnp.asarray(radial.exponents)[idx],
            coefficients=jnp.asarray(radial.coefficients)[idx],
        )
    )
    return cart, transform_matrix


def _global_cart_to_sphe_metric_map(transform_matrix: np.ndarray, overlap_matrix: np.ndarray) -> np.ndarray:
    r"""Return the global AO-metric reverse map for a spherical embedding.

    ``transform_matrix`` is ``T`` with shape ``(M_sphe, M_cart)`` and maps
    row-vector spherical coefficients to Cartesian coefficients.  The overlap
    is the complete molecular ``S`` or periodic ``S(k)``.  For the complex
    row-vector convention the normal equation is

    ``(conj(T) @ S @ T.T) @ R.T = conj(T) @ S``.

    Solving this equation, rather than forming an inverse, preserves the
    transpose-versus-adjoint convention explicitly.
    """
    transform = np.asarray(transform_matrix)
    overlap = np.asarray(overlap_matrix)
    if transform.ndim != 2 or overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
        raise ValueError("Global Cartesian-to-spherical projection requires rank-2 T and square S.")
    if transform.shape[1] != overlap.shape[0]:
        raise ValueError(
            "Global Cartesian-to-spherical projection dimension mismatch: "
            f"T has {transform.shape[1]} Cartesian columns but S has shape {overlap.shape}."
        )
    dtype = np.result_type(transform.dtype, overlap.dtype, np.float64)
    transform = transform.astype(dtype, copy=False)
    overlap = overlap.astype(dtype, copy=False)
    metric_rhs = transform.conj() @ overlap
    gram = metric_rhs @ transform.T
    gram = 0.5 * (gram + gram.conj().T)
    return scipy.linalg.solve(
        gram,
        metric_rhs,
        assume_a="her",
        check_finite=True,
    ).T


def _aos_cart_to_sphe_with_metric(
    aos_data: AOs_cart_gto_data | AOs_sphe_gto_data,
) -> tuple[AOs_sphe_gto_data, np.ndarray, np.ndarray, np.ndarray | None]:
    r"""Project Cartesian AOs onto spherical AOs in the global AO metric.

    The angular embedding ``T`` uses the same solid-harmonic convention as
    CP2K's ``c2s`` matrix, but the reverse map is a global extension: ``S`` is
    the complete molecular AO overlap, or the complete periodic Bloch overlap
    ``S(k)``, rather than an isolated shell block.  For row-vector expansion
    coefficients, the returned Cartesian-to-spherical map is

    .. math::

        G = T^* S T^T,\qquad
        R = [G^{-1}T^*S]^T.

    Consequently ``T R = I`` and the discarded Cartesian residual is
    globally ``S``-orthogonal to the spherical subspace.  At a non-Gamma
    twist, ``S(k)`` and ``R(k)`` are complex; the last operation above is a
    transpose, not an adjoint.  A Euclidean pseudoinverse or an isolated-shell
    metric does not minimize the residual norm of the complete represented
    function.

    TREXIO folds ``ao_normalization`` into contracted coefficients.  The
    Cartesian components of one shell therefore carry proportional, not
    necessarily identical, coefficient vectors.  Shell discovery factors out
    those component scales and includes them in ``T``; the emitted spherical
    shell retains one common radial contraction.

    Returns:
        tuple: ``(aos_sphe, cart_to_sphe, transform_matrix, overlap)``.
        ``cart_to_sphe`` (``R``) maps Cartesian -> spherical coefficients with
        shape (num_ao_cart, num_ao_sph); ``transform_matrix`` (``T``) is the
        spherical -> native-Cartesian embedding with shape
        (num_ao_sph, num_ao_cart); ``overlap`` is the Cartesian metric ``S``
        used for ``R`` (``None`` for a spherical input, where ``R = T = I``).
    """
    # I/O / setup-time basis conversion: hardcode fp64 (no precision-zone
    # involvement). See ``_aos_sphe_to_cart`` for the rationale.
    dtype_np = np.float64
    if isinstance(aos_data, AOs_sphe_gto_data):
        cart_to_sphe = np.eye(aos_data.num_ao, dtype=dtype_np)
        return aos_data, cart_to_sphe, cart_to_sphe, None
    if not isinstance(aos_data, AOs_cart_gto_data):
        raise ValueError("Spherical conversion is only available from Cartesian AOs.")

    periodic_source = None
    if isinstance(aos_data, AOs_cart_pgto_data) and aos_data._ragged_buckets:
        # Per-shell ragged periodic: discover shells and construct T from the
        # molecular home-cell representation, because the expanded kernel AO
        # arrays use identity nucleus indices.  Keep the original leaf so the
        # reverse map below uses its complete periodic S(k).
        periodic_source = aos_data
        aos_cart = _detile_ragged_to_molecular(aos_data)
    else:
        aos_cart = aos_data
    shells: list[dict] = []
    shells_by_identity: dict[tuple[int, int, tuple[int, ...]], dict] = {}
    physical_shells = bool(aos_cart.ao_coefficient_scales)

    def _component_scale(reference: np.ndarray, candidate: np.ndarray) -> float | None:
        """Return ``s`` when ``candidate == s * reference`` within I/O tolerance."""
        reference_norm_sq = float(np.dot(reference, reference))
        if reference_norm_sq == 0.0:
            return None
        scale = float(np.dot(reference, candidate) / reference_norm_sq)
        if scale == 0.0 or not np.allclose(candidate, scale * reference, rtol=rtol, atol=atol):
            return None
        return scale

    def _match_shell(
        existing: dict,
        nucleus: int,
        l: int,
        exps: np.ndarray,
        coefs: np.ndarray,
    ) -> float | None:
        if (
            existing["nucleus"] != nucleus
            or existing["l"] != l
            or len(existing["exps"]) != len(exps)
            or not np.allclose(existing["exps"], exps, rtol=rtol, atol=atol)
        ):
            return None
        return _component_scale(existing["coefs"], coefs)

    # Iterate over the kernel-rank AO count (_num_ao_kernel) so that pgto
    # inputs are processed correctly; for gto inputs this is equivalent to
    # iterating over num_ao.
    ao_exponents = aos_cart.exponents
    ao_coefficients = aos_cart.coefficients
    for ao_idx in range(aos_cart._num_ao_kernel):
        prim_indices = [i for i, v in enumerate(aos_cart.orbital_indices) if v == ao_idx]
        exps = ao_exponents[prim_indices]
        coefs = ao_coefficients[prim_indices]
        nucleus = aos_cart.nucleus_index[ao_idx]
        l = aos_cart.angular_momentums[ao_idx]
        order = (
            aos_cart.polynominal_order_x[ao_idx],
            aos_cart.polynominal_order_y[ao_idx],
            aos_cart.polynominal_order_z[ao_idx],
        )

        if physical_shells:
            radial_indices = tuple(aos_cart.ao_prim_to_shell_prim[i] for i in prim_indices)
            identity = (nucleus, l, radial_indices)
            shell = shells_by_identity.get(identity)
            if shell is None:
                shell = {
                    "nucleus": nucleus,
                    "l": l,
                    "exps": exps,
                    "coefs": np.asarray(aos_cart.radial.coefficients)[list(radial_indices)],
                    "orders_to_idx": {},
                    "orders_to_scale": {},
                    "radial_indices": radial_indices,
                }
                shells.append(shell)
                shells_by_identity[identity] = shell
            scale = aos_cart.ao_coefficient_scales[ao_idx]
            if scale == 0 or order in shell["orders_to_idx"]:
                raise ValueError(f"Invalid Cartesian component {order} for shell l={l} at nucleus {nucleus}.")
            shell["orders_to_idx"][order] = ao_idx
            shell["orders_to_scale"][order] = scale
            continue

        for shell in shells:
            scale = _match_shell(shell, nucleus, l, exps, coefs)
            if scale is not None:
                if order in shell["orders_to_idx"]:
                    raise ValueError(f"Duplicate Cartesian order {order} for shell l={l} at nucleus {nucleus}.")
                shell["orders_to_idx"][order] = ao_idx
                shell["orders_to_scale"][order] = scale
                break
        else:
            if not np.any(coefs):
                raise ValueError(f"Cartesian shell l={l} at nucleus {nucleus} has a zero radial contraction.")
            shells.append(
                {
                    "nucleus": nucleus,
                    "l": l,
                    "exps": exps,
                    "coefs": coefs,
                    "orders_to_idx": {order: ao_idx},
                    "orders_to_scale": {order: 1.0},
                }
            )

    total_sph = sum(2 * shell["l"] + 1 for shell in shells)
    total_cart = aos_cart._num_ao_kernel
    transform_matrix = np.zeros((total_sph, total_cart), dtype=dtype_np)

    new_nucleus_index: list[int] = []
    new_angular_momentums: list[int] = []
    new_magnetic_quantum_numbers: list[int] = []
    new_orbital_indices: list[int] = []
    new_exponents: list[float] = []
    new_coefficients: list[float] = []
    new_radial_indices: list[int] = []

    sph_row_start = 0
    for shell in shells:
        l = shell["l"]
        orders = [(nx, ny, l - nx - ny) for nx in range(l, -1, -1) for ny in range(l - nx, -1, -1)]
        if len(shell["orders_to_idx"]) != len(orders):
            raise ValueError(
                f"Shell with l={l} at nucleus {shell['nucleus']} is incomplete: {len(shell['orders_to_idx'])} Cartesian AOs found."
            )

        cart_indices = []
        component_scales = []
        for order in orders:
            if order not in shell["orders_to_idx"]:
                raise ValueError(f"Missing Cartesian order {order} for shell l={l} at nucleus {shell['nucleus']}.")
            cart_indices.append(shell["orders_to_idx"][order])
            component_scales.append(shell["orders_to_scale"][order])

        n_sph = 2 * l + 1
        sph_indices = list(range(sph_row_start, sph_row_start + n_sph))
        sph_row_start += n_sph
        # Native Cartesian AO values equal the common-radial Cartesian values
        # times ``component_scales``.  Divide the canonical embedding columns
        # by those scales so that the emitted common-radial spherical AOs are
        # represented exactly in the native Cartesian basis.
        transform_shell = _spherical_to_cart_matrix(l) / np.asarray(component_scales, dtype=dtype_np)[None, :]
        transform_matrix[np.ix_(sph_indices, cart_indices)] = transform_shell

        for m in range(-l, l + 1):
            ao_new_idx = len(new_nucleus_index)
            new_nucleus_index.append(shell["nucleus"])
            new_angular_momentums.append(l)
            new_magnetic_quantum_numbers.append(m)
            if physical_shells:
                new_radial_indices.extend(shell["radial_indices"])

            for exp, coef in zip(shell["exps"], shell["coefs"], strict=True):
                new_orbital_indices.append(ao_new_idx)
                new_exponents.append(float(exp))
                new_coefficients.append(float(coef))

    # Retain every cross-shell and cross-centre overlap block.  Periodic
    # inputs use the original expanded leaf's independently truncated
    # one-lattice S(k); molecular inputs use whole-space analytic integrals.
    if periodic_source is None:
        overlap = np.asarray(_compute_overlap_matrix_cart_analytic(aos_cart), dtype=dtype_np)
    else:
        overlap = np.asarray(compute_overlap_matrix(periodic_source))
    cart_to_sphe = _global_cart_to_sphe_metric_map(transform_matrix, overlap)

    physical_fields = (
        dict(
            radial=aos_cart.radial,
            ao_prim_to_shell_prim=new_radial_indices,
            ao_coefficient_scales=(1.0,) * total_sph,
        )
        if physical_shells
        else {}
    )
    aos_sphe = AOs_sphe_gto_data._from_full_basis(
        structure_data=aos_cart.structure_data,
        nucleus_index=new_nucleus_index,
        num_ao=total_sph,
        num_ao_prim=len(new_exponents),
        orbital_indices=new_orbital_indices,
        # Emit at the function's setup-time fp64 (see dtype_np above): an
        # "ao_eval"-zone cast here would round the radial values through fp32
        # under mixed precision before the fp64 shell-leaf storage.
        exponents=np.array(new_exponents, dtype=dtype_np),
        coefficients=np.array(new_coefficients, dtype=dtype_np),
        angular_momentums=new_angular_momentums,
        magnetic_quantum_numbers=new_magnetic_quantum_numbers,
        **physical_fields,
    )

    if periodic_source is not None:
        ragged_sphe = _apply_image_expansion(
            aos_sphe,
            periodic_source.structure_data,
            tol=float(periodic_source._image_truncation_tol),
        )
        return _preserve_complex_pgto_kind(periodic_source, ragged_sphe), cart_to_sphe, transform_matrix, overlap
    return _preserve_complex_pgto_kind(aos_data, aos_sphe), cart_to_sphe, transform_matrix, overlap


def _aos_cart_to_sphe(aos_data: AOs_cart_gto_data | AOs_sphe_gto_data) -> tuple[AOs_sphe_gto_data, np.ndarray]:
    """Return ``(aos_sphe, cart_to_sphe)`` of :func:`_aos_cart_to_sphe_with_metric`."""
    aos_sphe, cart_to_sphe, _, _ = _aos_cart_to_sphe_with_metric(aos_data)
    return aos_sphe, cart_to_sphe


def _primitive_value_envelope_prefactor(exponent: float, angular_L: int) -> float:
    r"""Bound ``P`` with ``|N A(r)| <= P r^L`` for one normalised primitive, Cartesian or spherical.

    A normalised primitive of an AO is ``c N A(r) exp(-alpha r^2)``. For the
    real-spherical leaf ``N A(r) = N_sph sqrt((2L+1)/(4 pi)) S_LM(r)`` with the
    Racah-normalised solid harmonic ``|S_LM(r)| <= r^L``. For the Cartesian leaf
    ``N A(r) = N_cart(n) x^nx y^ny z^nz`` with
    ``max_angle |x^nx y^ny z^nz| / r^L = sqrt(prod_i n_i^n_i / L^L)``. ``P`` is
    the maximum of both forms over all components of the shell, so it depends
    only on ``(alpha, L)`` and the two angular forms of one shell (and the
    sphe <-> cart conversions that reuse each other's image sets) select the
    same images. Both maxima equal ``N_cart(L, 0, 0)``; the explicit maximum is
    kept so the bound does not rely on that identity. Host-side fp64.
    """
    alpha = float(exponent)
    l = int(angular_L)
    fact = scipy.special.factorial
    spherical = float(
        np.sqrt(2.0 ** (2 * l + 3) * fact(l + 1) / fact(2 * l + 2) * (2.0 * alpha) ** (l + 1.5) / np.sqrt(np.pi))
        * np.sqrt((2 * l + 1) / (4.0 * np.pi))
    )
    cartesian = 0.0
    for nx in range(l + 1):
        for ny in range(l + 1 - nx):
            powers = (nx, ny, l - nx - ny)
            ratio = float(np.prod([fact(n) / fact(2 * n) for n in powers]))
            norm = float(np.sqrt((2.0 * alpha / np.pi) ** 1.5 * (8.0 * alpha) ** l * ratio))
            angular = float(np.sqrt(np.prod([float(n) ** n for n in powers]) / float(l) ** l)) if l > 0 else 1.0
            cartesian = max(cartesian, norm * angular)
    return max(spherical, cartesian)


class AOEvalRangeError(ValueError):
    """A GTO basis cannot be evaluated in the active ``ao_eval`` dtype (mixed precision)."""


#: Maximum number of offending primitives listed in an :class:`AOEvalRangeError` message.
_AO_EVAL_RANGE_REPORT_LIMIT = 8


def _ao_eval_significant_shell(ell: np.ndarray, ln_depth: float) -> tuple[np.ndarray, np.ndarray]:
    r"""Return ``(t_in, t_out)``, the reduced radii ``t = a r^2`` bounding the significant shell.

    The envelope ``r^l exp(-a r^2)`` peaks at ``t0 = l / 2``. The significant
    shell is ``{t : (l/2) ln(t/t0) - (t - t0) >= -ln_depth}``, i.e. where the
    envelope is within ``exp(-ln_depth)`` of its peak. It does not depend on
    ``a``. With ``u = t / t0`` the boundary solves ``u exp(-u) = exp(-(1 + ln_depth / t0))``,
    so ``t = -t0 W_k(-exp(-(1 + ln_depth/t0)))`` on the principal (inner) and
    ``-1`` (outer) Lambert-W branches. For ``l = 0`` the envelope is
    ``exp(-t)``, and the shell is ``0 <= t <= ln_depth``.
    """
    ell = np.asarray(ell)
    t_in = np.zeros(ell.shape)
    t_out = np.full(ell.shape, ln_depth)
    positive = ell > 0
    if np.any(positive):
        t0 = 0.5 * ell[positive]
        z = -np.exp(-(1.0 + ln_depth / t0))
        t_in[positive] = -t0 * scipy.special.lambertw(z, 0).real
        t_out[positive] = -t0 * scipy.special.lambertw(z, -1).real
    return t_in, t_out


def _ao_eval_logsumexp_per_ao(values: np.ndarray, ao_of_prim: np.ndarray, n_ao: int) -> np.ndarray:
    """Return ``ln sum_{p in AO} exp(values_p)`` for each AO (``-inf`` for an empty or all ``-inf`` AO)."""
    peak = np.full(n_ao, -np.inf)
    np.maximum.at(peak, ao_of_prim, values)
    shift = np.where(np.isfinite(peak), peak, 0.0)
    total = np.zeros(n_ao)
    np.add.at(total, ao_of_prim, np.exp(values - shift[ao_of_prim]))
    with np.errstate(divide="ignore"):
        return shift + np.log(total)


def check_ao_eval_value_range(aos_data: AOs_base_data) -> None:
    r"""Reject a GTO basis whose AO values the ``ao_eval`` dtype cannot represent.

    Loud precondition for mixed precision. It is a no-op unless the ``ao_eval`` zone is narrower than the
    basis storage dtype (i.e. in mixed precision), and for non-GTO leaves.
    Composite leaves are checked child by child. It runs where a basis is
    configured into a wavefunction: ``MOs_data.sanity_check`` (hence MO /
    geminal / Hamiltonian validation, TREXIO and HDF5 loads and the MCMC / GFMC
    driver constructors, after the precision mode is set) and
    ``Jastrow_three_body_data.sanity_check``. A bare AO leaf stays
    constructible, because its float64 gradient / Laplacian / fused kernels
    are unaffected; call this function directly before evaluating a bare leaf's
    standalone values in mixed precision.

    The standalone AO value kernels (XLA ``_compute_AOs`` and the Rust f32
    value kernel) form, in the ``ao_eval`` dtype, the per-primitive
    normalization ``N`` (rounded once from storage precision), the radial factor
    ``N exp(-a r^2)``, the angular polynomial (Cartesian ``x^nx y^ny z^nz`` or
    the Racah-normalised solid harmonic, ``|S_lm| <= r^l``) and their product.
    For each primitive (unit coefficient, exponent rounded to the ``ao_eval``
    dtype) this function evaluates these quantities analytically in log space
    and requires, with the bounds of
    :func:`jqmc._setting.get_ao_eval_value_range_bounds`:

    1. ``N`` (and the spherical radial ``N_n`` alone) is a normal finite value;
    2. the peak AO value ``max_r |phi|`` is finite and stays a normal value
       ``ln_depth`` below the peak (the significant shell is representable);
    3. inside the significant shell ``t_in <= a r^2 <= t_out``
       (:func:`_ao_eval_significant_shell`) the radial factor does not
       underflow, ``r^l`` does not overflow and the polynomial on the peak ray
       does not underflow;
    4. wherever a factor still underflows (polynomial near a nodal plane, or
       the radial factor far outside the shell), the dropped AO value is below
       the negligible absolute bound: ``tiny * max(N, max_r N r^l exp(-a r^2))``
       and ``r_u^l * tiny`` at the radius ``r_u`` where ``N exp(-a r_u^2)``
       reaches ``tiny``.

    The bounds include the contraction coefficient ``c`` of every primitive
    with the kernels' order of operations: the XLA and Rust f32
    Cartesian kernels round ``c`` to the ``ao_eval`` dtype and form
    ``exp(-a r^2)``, ``c exp(-a r^2)`` and ``N (c exp(-a r^2))``; the periodic
    spherical kernels keep ``N``, ``c`` and ``a`` at storage precision, form
    ``exp(-a r^2)`` in the zone and round ``N (c exp(-a r^2))`` once; the
    molecular spherical kernels round the prefactor ``N c`` once and
    form ``exp(-a r^2)`` and ``(N c) exp(-a r^2)``; every kernel then sums the
    primitives of a contracted AO before multiplying the angular polynomial.
    The conditions below bound the union of these partial products, so they
    are sufficient for every kernel. For every primitive with ``c != 0`` it additionally
    requires:

    5. ``c`` is a normal finite value;
    6. the radial product ``|N c|`` (its value at ``r = 0``) is finite, and so
       is the contracted radial bound ``sum_p |N_p c_p|`` of each AO;
    7. the peak value ``|c| max_r |phi|`` is finite (also summed over the
       primitives of each AO) and stays a normal value ``ln_depth`` below the
       peak;
    8. none of the partial products ``exp(-a r^2)``, ``|c| exp(-a r^2)``,
       ``|N c| exp(-a r^2)`` underflows inside the significant shell;
    9. the AO value an underflow drops is below the negligible absolute bound:
       ``tiny * max(|N c|, |c| max_r N r^l exp(-a r^2))`` for the polynomial
       (``l > 0``), and ``|N c| r_u^l exp(-a r_u^2)`` at the first radius
       ``r_u`` where one of the radial partial products reaches ``tiny`` (also
       for ``l = 0``).

    Conditions 1 to 4 are the unit-coefficient case of 5 to 9 and are kept.
    A primitive with ``c == 0`` contributes exactly zero; only its ``N`` is
    checked.

    Raises:
        AOEvalRangeError: listing the offending primitives (atom, AO index,
            ``l``, Cartesian powers or ``m``, exponent and failed condition).
    """
    if isinstance(aos_data, AOs_composite_data):
        for child in aos_data.children:
            check_ao_eval_value_range(child)
        return
    is_cart = isinstance(aos_data, AOs_cart_gto_data)
    if not (is_cart or isinstance(aos_data, AOs_sphe_gto_data)):
        return
    if _radial_is_traced(aos_data):
        return
    exponents_storage = np.asarray(aos_data.exponents)
    ao_dtype = np.dtype(get_dtype_np(aos_data._value_zone))
    if np.finfo(ao_dtype).bits >= np.finfo(exponents_storage.dtype).bits:
        return
    if exponents_storage.size == 0:
        return
    ln_min, ln_max, ln_depth, ln_negligible = _setting.get_ao_eval_value_range_bounds(ao_dtype)

    # The kernels evaluate the Gaussian (and, since #129, N) at the zone-rounded exponent.
    a = exponents_storage.astype(ao_dtype).astype(exponents_storage.dtype)
    ao_of_prim = np.asarray(aos_data.orbital_indices, dtype=np.int64)
    ell = np.asarray(aos_data.angular_momentums, dtype=np.int64)[ao_of_prim]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ln_a = np.log(a)
        ln_2a = np.log(2.0) + ln_a
        t_in, t_out = _ao_eval_significant_shell(ell, ln_depth)
        half_l = 0.5 * ell
        has_poly = ell > 0
        # max_r r^l exp(-a r^2) at a r^2 = l/2 (1 for l = 0).
        ln_env_peak_shape = np.where(has_poly, half_l * (np.log(np.maximum(ell, 1)) - ln_2a) - half_l, 0.0)
        if is_cart:
            powers = np.stack(
                [
                    np.asarray(aos_data.polynominal_order_x, dtype=np.int64)[ao_of_prim],
                    np.asarray(aos_data.polynominal_order_y, dtype=np.int64)[ao_of_prim],
                    np.asarray(aos_data.polynominal_order_z, dtype=np.int64)[ao_of_prim],
                ]
            )
            ln_norm = -0.5 * np.sum(scipy.special.gammaln(powers + 0.5) - (powers + 0.5) * ln_2a[None, :], axis=0)
            ln_norm_radial = ln_norm
            nonzero = powers > 0
            # Polynomial at the peak point x_i^2 = n_i / (2a), and its factors on the
            # peak ray at the inner shell edge a r^2 = t_in (scale t_in / (l/2)).
            ln_factor_peak = np.where(nonzero, 0.5 * powers * (np.log(np.maximum(powers, 1)) - ln_2a[None, :]), 0.0)
            ln_scale_in = np.where(has_poly, np.log(t_in) - np.log(np.maximum(half_l, 0.5)), 0.0)
            ln_factor_in = np.where(nonzero, ln_factor_peak + 0.5 * powers * ln_scale_in[None, :], np.inf)
            ln_poly_peak = np.sum(ln_factor_peak, axis=0)
            ln_poly_in_min = np.minimum(np.min(ln_factor_in, axis=0), ln_poly_peak + half_l * ln_scale_in)
            component = [f"(nx,ny,nz)=({x},{y},{z})" for x, y, z in powers.T]
        else:
            ln_norm_radial = 0.5 * (np.log(2.0) + (ell + 1.5) * ln_2a - scipy.special.gammaln(ell + 1.5))
            ln_norm = ln_norm_radial + 0.5 * np.log((2.0 * ell + 1.0) / (4.0 * np.pi))
            ln_poly_peak = ln_env_peak_shape + half_l
            ln_poly_in_min = np.where(has_poly, half_l * (np.log(t_in) - ln_a), 0.0)
            m_of_prim = np.asarray(aos_data.magnetic_quantum_numbers, dtype=np.int64)[ao_of_prim]
            component = [f"m={m}" for m in m_of_prim]
        ln_peak = ln_norm + ln_poly_peak - half_l
        ln_env_peak = ln_norm + ln_env_peak_shape
        ln_poly_out = np.where(has_poly, half_l * (np.log(t_out) - ln_a), -np.inf)
        t_underflow = ln_norm - ln_min
        ln_drop_far = np.where(
            has_poly & (t_underflow > 0), half_l * (np.log(np.maximum(t_underflow, t_out)) - ln_a) + ln_min, -np.inf
        )
        conditions = (
            ((ln_norm_radial < ln_min) | (ln_norm < ln_min), "normalization N underflows"),
            ((ln_norm_radial > ln_max) | (ln_norm > ln_max), "normalization N overflows"),
            (ln_peak > ln_max, "peak AO value overflows"),
            (ln_peak - ln_depth < ln_min, "AO values within the significant shell underflow"),
            (ln_norm - t_out < ln_min, "radial factor N exp(-a r^2) underflows within the significant shell"),
            (ln_poly_out > ln_max, "angular polynomial (<= r^l) overflows within the significant shell"),
            (ln_poly_in_min < ln_min, "angular polynomial underflows within the significant shell"),
            (
                np.maximum(ln_norm, ln_env_peak) + ln_min > ln_negligible,
                "an underflowing polynomial factor would drop AO values above the negligible bound",
            ),
            (ln_drop_far > ln_negligible, "the radial factor underflows where the AO value is not negligible"),
        )
        # Contraction coefficients. The kernels multiply |c| into the radial
        # product before the angular polynomial; a zero coefficient contributes 0.
        coefficients = np.abs(np.asarray(aos_data.coefficients))
        has_c = coefficients != 0.0
        ln_c = np.where(has_c, np.log(np.where(has_c, coefficients, 1.0)), -np.inf)
        ln_nc = ln_norm + ln_c
        ln_peak_c = ln_peak + ln_c
        n_ao = int(ao_of_prim.max()) + 1
        # Partial products exp(-t), |c| exp(-t), |N c| exp(-t) reach tiny first at t_u.
        t_u = np.minimum(np.minimum(-ln_min, ln_c - ln_min), ln_nc - ln_min)
        t_u = np.maximum(t_u, t_out)
        ln_drop_far_c = ln_nc + np.where(has_poly, half_l * (np.log(t_u) - ln_a), 0.0) - t_u
        coefficient_conditions = (
            ((ln_c > ln_max) | (ln_c < ln_min), "contraction coefficient c leaves the range"),
            (ln_nc > ln_max, "radial product N c exp(-a r^2) overflows"),
            (
                _ao_eval_logsumexp_per_ao(ln_nc, ao_of_prim, n_ao)[ao_of_prim] > ln_max,
                "contracted radial sum over the AO's primitives overflows",
            ),
            (ln_peak_c > ln_max, "peak AO value c phi overflows"),
            (
                _ao_eval_logsumexp_per_ao(ln_peak_c, ao_of_prim, n_ao)[ao_of_prim] > ln_max,
                "contracted peak AO value overflows",
            ),
            (ln_peak_c - ln_depth < ln_min, "AO values c phi within the significant shell underflow"),
            (
                np.minimum(np.minimum(0.0, ln_c), ln_nc) - t_out < ln_min,
                "a radial partial product exp(-a r^2), c exp(-a r^2) or N c exp(-a r^2) underflows within the significant shell",
            ),
            (
                has_poly & (np.maximum(ln_nc, ln_env_peak + ln_c) + ln_min > ln_negligible),
                "an underflowing polynomial factor would drop c phi above the negligible bound",
            ),
            (ln_drop_far_c > ln_negligible, "a radial partial product underflows where c phi is not negligible"),
        )
        conditions = conditions + tuple(
            (np.asarray(mask, dtype=bool) & has_c, reason) for mask, reason in coefficient_conditions
        )
    bad = np.zeros(a.shape, dtype=bool)
    for mask, _ in conditions:
        bad |= np.asarray(mask, dtype=bool)
    if not np.any(bad):
        return

    labels = tuple(getattr(aos_data.structure_data, "atomic_labels", ()) or ())
    nucleus_index = np.asarray(aos_data.nucleus_index, dtype=np.int64)
    entries = []
    seen = set()
    for p in np.flatnonzero(bad):
        ao = int(ao_of_prim[p])
        nucleus = int(nucleus_index[ao])
        key = (nucleus, int(ell[p]), component[p], float(a[p]))
        if key in seen:
            continue
        seen.add(key)
        reasons = "; ".join(reason for mask, reason in conditions if bool(np.asarray(mask)[p]))
        atom = labels[nucleus] if nucleus < len(labels) else f"#{nucleus}"
        entries.append(
            f"  atom {atom} (nucleus index {nucleus}), AO {ao}, l={int(ell[p])}, {component[p]}, "
            f"exponent={float(a[p]):.6g}: {reasons}"
        )
    shown = entries[:_AO_EVAL_RANGE_REPORT_LIMIT]
    more = len(entries) - len(shown)
    raise AOEvalRangeError(
        f"{type(aos_data).__name__}: {len(entries)} Gaussian primitive(s) cannot be evaluated in mixed precision: "
        f"the AO evaluation zone 'ao_eval' is {ao_dtype.name}, and these primitives' normalization or AO "
        f"values leave the {ao_dtype.name} range, so the AO values, and the local energy "
        "computed from them, would be silently wrong (Inf, NaN, 0 or rescaled).\n"
        + "\n".join(shown)
        + (f"\n  ... and {more} more" if more > 0 else "")
        + '\nUse full precision for this basis ([precision] mode = "full" in the input TOML, or jqmc._precision.configure("full")), '
        "or remove or replace these shells (very high l or extreme exponents)."
    )


def _primitive_image_truncation_tolerances(aos_data: AOs_cart_gto_data | AOs_sphe_gto_data, tol: float) -> np.ndarray:
    r"""Per-primitive envelope tolerance so that each dropped image is below ``tol`` in AO value.

    The image radius (:py:func:`jqmc.structure._primitive_image_radius`) bounds the
    bare envelope ``r^L exp(-alpha r^2)``. One image of AO ``mu`` contributes
    ``sum_p c_p N_p A(r) exp(-alpha_p r^2)`` over the ``K_mu`` primitives of ``mu``,
    and primitive ``p`` is at most ``|c_p| P(alpha_p, L) r^L exp(-alpha_p r^2)``
    (:py:func:`_primitive_value_envelope_prefactor`). Selecting primitive ``p`` at
    the envelope tolerance ``tol / max(1, K_mu |c_p| P(alpha_p, L))`` bounds its
    dropped term by ``tol / K_mu``, so the dropped image's AO value (the sum over
    the ``K_mu`` primitives) is at most ``tol`` everywhere in the home cell
    (issue #120). The share is per primitive, so a diffuse primitive is not
    tightened by the large prefactor of a tight one in the same contraction. The
    ``max(1, .)`` never loosens the previous bare-envelope selection. Depends only
    on ``(alpha_p, c_p, L, K_mu)``, so all angular components of a shell, in either
    angular form, get the same tolerances. Host-side fp64.

    Returns:
        ndarray: ``(num_ao_prim,)`` fp64 envelope tolerances, primitive order.
    """
    orb_idx = np.asarray(aos_data.orbital_indices, dtype=np.int64)
    L_per_ao = np.asarray(aos_data.angular_momentums, dtype=np.int64)
    exps = np.asarray(aos_data.exponents, dtype=np.float64)
    coefs = np.asarray(aos_data.coefficients, dtype=np.float64)
    n_prim_per_ao = np.bincount(orb_idx, minlength=int(aos_data.num_ao))
    prefactor_cache: dict[tuple[float, int], float] = {}
    prim_tol = np.empty(len(orb_idx), dtype=np.float64)
    for p in range(len(orb_idx)):
        ao = int(orb_idx[p])
        key = (float(exps[p]), int(L_per_ao[ao]))
        if key not in prefactor_cache:
            prefactor_cache[key] = _primitive_value_envelope_prefactor(*key)
        prim_tol[p] = float(tol) / max(1.0, float(n_prim_per_ao[ao]) * abs(float(coefs[p])) * prefactor_cache[key])
    return prim_tol


def _build_ragged_image_layout(
    aos_data: AOs_cart_gto_data | AOs_sphe_gto_data,
    structure_data: Structure_data,
    tol: float = AO_IMAGE_TRUNCATION_ATOL,
    ao_images_override: list[set] | None = None,
) -> dict:
    r"""(B) Host-side per-SHELL ragged image layout.

    Tiles each home AO by only ITS OWN radial's image set (the union over that
    AO's own primitives), shrinking the kernel from ``n_image * num_ao`` to
    ``M = sum_shell m_shell * k_shell``. Because the ``2l+1`` m-components of a
    shell share their radial, a per-AO image set is automatically per-SHELL: a
    TIGHT shell keeps few images even when a DIFFUSE shell on the same atom
    keeps many (~3.78 images/shell -> ~6x kernel reduction on a 36-atom
    periodic cell). This is the
    sole image-expansion granularity.

    To keep the device-side fold dense (a few ``reduce_sum`` ops, never
    ``segment_sum`` -- the fragmentation jQMC avoids in
    :func:`_reduce_primitives_to_aos`), home AOs are bucketed by multiplicity
    ``m``: within a bucket all kernel-AOs form a dense ``(m, K_bucket)`` block
    folded by ``reshape(m, K_bucket, -1).sum(0)``. The bucket outputs
    concatenate and an inverse permutation restores the original AO order.
    Numpy / fp64, setup-time only.

    Returns a ``dict``:
      - ``num_ao`` (int), ``num_kernel`` (int, ``M``),
      - ``n_image_union`` (int): the global-union image count (diagnostic),
      - ``ao_n_image`` (``(num_ao,)`` int): ``m`` per home AO,
      - ``kernel_to_home`` (``(M,)`` int): kernel-AO -> home-AO (original order),
      - ``kernel_image_index`` (``(M, 3)`` int): lattice image triple per kernel-AO,
      - ``buckets``: list of ``{m, home_aos (K,), kernel_offset, K}`` (bucket-major),
      - ``fold_perm`` (``(num_ao,)`` int): bucket-concatenated home-AO order ->
        original AO order (inverse permutation applied after the per-bucket fold).
    """
    if not structure_data.pbc_flag:
        raise ValueError("_build_ragged_image_layout requires structure_data.pbc_flag=True.")

    num_ao = int(aos_data.num_ao)

    if ao_images_override is not None:
        # Traced-ham path (sphe->cart under the production jit bridge): the image
        # truncation ``_primitive_image_radius`` reads exponent VALUES, which are
        # tracers under the bridge. The image sets are position/strain-independent
        # per-shell statics, IDENTICAL to the input (sphe) leaf's, so the caller
        # supplies them here instead of recomputing from the (traced) exponents.
        # Everything below is purely combinatorial on the image sets -> jit-safe.
        if len(ao_images_override) != num_ao:
            raise ValueError(f"ao_images_override length {len(ao_images_override)} != num_ao {num_ao}.")
        ao_images: list[set[tuple[int, int, int]]] = [set(s) for s in ao_images_override]
    else:
        cell = np.asarray(structure_data.cell, dtype=np.float64)
        inv_cell = np.linalg.inv(cell)
        # UNWRAPPED fractional positions: the kernel centres are R + n @ cell with
        # the unwrapped R, so the image set must be chosen for that same R (a
        # wrapped choice shifts the kept set for a nucleus outside the home cell
        # or on its lower boundary).
        frac_positions = np.asarray(structure_data._positions_cart_np, dtype=np.float64) @ inv_cell

        nucleus_of_ao = np.asarray(aos_data.nucleus_index, dtype=np.int64)  # (num_ao,)
        orb_idx = np.asarray(aos_data.orbital_indices, dtype=np.int64)  # per prim -> AO
        L_per_ao = np.asarray(aos_data.angular_momentums, dtype=np.int64)
        exps = np.asarray(aos_data.exponents, dtype=np.float64)
        # Envelope tolerance per primitive: tol shared over the AO's primitives and
        # divided by each one's normalised prefactor, so each dropped image is
        # < tol in AO VALUE (issue #120).
        prim_tol = _primitive_image_truncation_tolerances(aos_data, tol)

        # Per-primitive image lists keyed by (alpha, L, atom, envelope tol); cached.
        prim_cache: dict[tuple[float, int, int, float], np.ndarray] = {}

        # Per-HOME-AO (per-shell) image set: union over that AO's own primitives only.
        ao_images = [set() for _ in range(num_ao)]
        for p in range(len(orb_idx)):
            ao = int(orb_idx[p])
            alpha = float(exps[p])
            angular_momentum = int(L_per_ao[ao])
            atom = int(nucleus_of_ao[ao])
            primitive_key = (alpha, angular_momentum, atom, float(prim_tol[p]))
            if primitive_key not in prim_cache:
                prim_cache[primitive_key] = _primitive_image_list_indices_per_atom(
                    cell,
                    alpha,
                    angular_momentum,
                    frac_positions[atom],
                    float(prim_tol[p]),
                )
            for row in prim_cache[primitive_key]:
                ao_images[ao].add((int(row[0]), int(row[1]), int(row[2])))
        # T_0 is always kept (the de-tile picks each home AO's T_0 kernel-AO). For a
        # nucleus inside the home cell it is already selected; outside, it is an
        # extra (at worst negligible) term of the same lattice sum.
        for image_set in ao_images:
            image_set.add((0, 0, 0))

    ao_sorted: list[list[tuple[int, int, int]]] = []
    for image_set in ao_images:
        ordered = sorted(image_set, key=lambda image: (abs(image[0]) + abs(image[1]) + abs(image[2]), image))
        if (0, 0, 0) in image_set:
            ordered.remove((0, 0, 0))
            ordered.insert(0, (0, 0, 0))
        ao_sorted.append(ordered)
    ao_n_image = np.array([len(ao_sorted[ao]) for ao in range(num_ao)], dtype=np.int64)

    union: set[tuple[int, int, int]] = set()
    for ao in range(num_ao):
        union |= ao_images[ao]
    n_image_union = len(union)

    # Bucket HOME-AOs by their own multiplicity ``m`` so each bucket is a dense
    # ``(m, K)`` block (image-major, then home-AO-major within image).
    distinct_m = sorted({int(x) for x in ao_n_image})
    buckets: list[dict] = []
    kernel_to_home_parts: list[np.ndarray] = []
    kernel_image_parts: list[np.ndarray] = []
    fold_home_order: list[int] = []
    offset = 0
    for m in distinct_m:
        home_aos = [ao for ao in range(num_ao) if int(ao_n_image[ao]) == m]
        K = len(home_aos)
        home_arr = np.array(home_aos, dtype=np.int64)
        block_home = np.empty(m * K, dtype=np.int64)
        block_image = np.empty((m * K, 3), dtype=np.int64)
        for i in range(m):
            block_home[i * K : (i + 1) * K] = home_arr
            block_image[i * K : (i + 1) * K] = np.array([ao_sorted[int(h)][i] for h in home_aos], dtype=np.int64)
        kernel_to_home_parts.append(block_home)
        kernel_image_parts.append(block_image)
        buckets.append({"m": m, "home_aos": home_arr, "kernel_offset": offset, "K": K})
        fold_home_order.extend(home_aos)
        offset += m * K

    kernel_to_home = np.concatenate(kernel_to_home_parts) if kernel_to_home_parts else np.zeros(0, dtype=np.int64)
    kernel_image_index = np.concatenate(kernel_image_parts) if kernel_image_parts else np.zeros((0, 3), dtype=np.int64)
    M = int(kernel_to_home.shape[0])

    # fold_perm: bucket-concatenated home-AO order -> original AO order.
    # fold_home_order[b] = original AO index sitting at position b of the
    # concatenated bucket outputs; inverting it scatters bucket outputs back.
    fold_perm = np.empty(num_ao, dtype=np.int64)
    fold_perm[np.array(fold_home_order, dtype=np.int64)] = np.arange(num_ao, dtype=np.int64)

    return {
        "num_ao": num_ao,
        "num_kernel": M,
        "n_image_union": n_image_union,
        "ao_n_image": ao_n_image,
        "kernel_to_home": kernel_to_home,
        "kernel_image_index": kernel_image_index,
        "buckets": buckets,
        "fold_perm": fold_perm,
    }


def _fold_image_axis_ragged(kernel: jax.Array, buckets: list[dict], fold_perm: np.ndarray) -> jax.Array:
    r"""(B) Device-side ragged image fold.

    Collapses the ragged per-atom kernel axis ``(M, ...) -> (num_ao, ...)``
    (``M = sum_A m_A * k_A``) using the multiplicity buckets from
    :func:`_build_ragged_image_layout`. Each bucket is a dense
    ``(m, K_bucket)`` block folded by ``reshape(m, K_bucket, -1).sum(0)``;
    the bucket outputs concatenate (bucket-major home-AO order) and a single
    gather by ``fold_perm`` restores the original AO order.

    The bucket geometry (``m``, ``K``, ``offset``) is STATIC (host-computed),
    so the Python loop unrolls at trace time into a handful of dense
    reductions -- the :func:`_reduce_primitives_to_aos` /
    ``_build_prim_groups_by_K`` discipline at atom granularity, with **no**
    ``segment_sum`` and hence no scatter-add fragmentation. This is what
    :meth:`AOs_pGTO_base_data._fold_image_axis` calls.

    CPU float32 values use one explicit adjacent-pair image tree for every
    electron batch width. At an odd level the last row is set aside and the
    set-aside rows are added to the tree root, latest level first. The
    layout depends only on the static bucket multiplicity, and the tree has
    only elementwise adds, so folding a full-width float32 kernel table
    gives, per column, bitwise the same values as folding that column alone
    (the fold itself adds no batch-width dependence). Accumulation depth
    stays logarithmic in the image count. Primitive contraction, other
    dtypes (whose sequential sum is pinned bit-exact elsewhere) and other
    backends are unchanged.

    Args:
        kernel: ragged kernel-rank values, shape ``(M, ...)``.
        buckets: list of ``{m, K, kernel_offset}`` (static; bucket-major).
        fold_perm: ``(num_ao,)`` int, bucket-concat home-AO order -> original.

    Returns:
        jax.Array: folded AO values, shape ``(num_ao, ...)``.
    """
    trailing = kernel.shape[1:]
    parts = []
    for b in buckets:
        m, K, off = int(b["m"]), int(b["K"]), int(b["kernel_offset"])
        block = jax.lax.slice_in_dim(kernel, off, off + m * K, axis=0)
        if kernel.dtype == jnp.float32 and jax.default_backend() == "cpu":
            # Long fp32 dots can lose several ulps even for positive image
            # terms. Bound accumulation depth without changing precision.
            # The add tree depends only on the static bucket size m, never on
            # the electron width: only elementwise adds, so each column is
            # rounded identically in single-electron and full-table calls
            # (a carried Sherman-Morrison inverse must match a fresh G).
            # An odd level sets its last row aside instead of concatenating.
            # A single-electron column keeps images on the minor axis (as
            # adjacent pairs); this changes memory order only, not the adds.
            single = trailing == (1,)
            block = block.reshape((m, K)).T if single else block.reshape((m, K) + trailing)
            carries = []
            while m > 1:
                pairs = m // 2
                if single:
                    if m % 2:
                        carries.append(block[:, m - 1])
                    paired = block[:, : 2 * pairs].reshape((K, pairs, 2))
                    block = paired[:, :, 0] + paired[:, :, 1]
                else:
                    if m % 2:
                        carries.append(block[m - 1])
                    block = block[: 2 * pairs : 2] + block[1 : 2 * pairs : 2]
                m = pairs
            total = block[:, 0] if single else block[0]
            for carry in reversed(carries):
                total = total + carry
            parts.append(total.reshape((K,) + trailing))
        else:
            parts.append(block.reshape(m, K, *trailing).sum(axis=0))
    concat = jnp.concatenate(parts, axis=0)  # (num_ao, ...) in bucket-concat order
    return concat[jnp.asarray(fold_perm, dtype=jnp.int32)]


def _home_shell_rep_from_molecular(base: AOs_cart_gto_data | AOs_sphe_gto_data) -> tuple[int, ...] | None:
    """Recover shell identity from the static physical primitive gather.

    Values may coincide after optimization and may be JAX tracers. Neither
    changes which angular components own the same independent parameters.
    Using the stored gather also handles non-shell-major/interleaved bases
    without assuming a spherical component count for Cartesian shells.
    """
    orbital = np.asarray(base.orbital_indices, dtype=np.int64)
    primitive_map = np.asarray(base.ao_prim_to_shell_prim, dtype=np.int64)
    representatives: dict[tuple, int] = {}
    result: list[int] = []
    for home in range(int(base.num_ao)):
        key = (
            int(base.nucleus_index[home]),
            int(base.angular_momentums[home]),
            tuple(int(p) for p in primitive_map[orbital == home]),
        )
        result.append(representatives.setdefault(key, home))
    return tuple(result)


def _build_image_expanded_arrays_ragged(
    aos_data: AOs_cart_gto_data | AOs_sphe_gto_data,
    structure_data: Structure_data,
    tol: float,
    ao_images_override: list[set] | None = None,
) -> tuple[dict, np.ndarray]:
    r"""(B) Ragged image expansion.

    Tiles each atom by ONLY its own image set (per
    :func:`_build_ragged_image_layout`) instead of a uniform global union.
    Emits kernel-AO / kernel-primitive arrays at the
    ragged rank ``M = sum_A m_A * k_A`` plus the ragged-leaf statics
    (``_image_indices`` as one triple per kernel-AO, ``_ragged_kernel_atom``,
    ``_ragged_buckets``, ``_ragged_fold_perm``). ``nucleus_index`` is the
    identity ``range(M)`` into the per-kernel-AO centres reconstructed in-trace.

    Returns ``(common_kwargs, kernel_to_home)``; the caller appends the
    per-type angular arrays (cart ``polynominal_order_*`` / sphe
    ``magnetic_quantum_numbers``) indexed by ``kernel_to_home``.
    """
    if not structure_data.pbc_flag:
        raise ValueError("_build_image_expanded_arrays_ragged requires structure_data.pbc_flag=True.")
    # Per-SHELL granularity is the lever: a tight shell keeps few images even
    # when a diffuse shell on the same atom keeps many.
    lay = _build_ragged_image_layout(aos_data, structure_data, tol, ao_images_override=ao_images_override)
    M = int(lay["num_kernel"])
    kth = np.asarray(lay["kernel_to_home"], dtype=np.int64)  # (M,) kernel-AO -> home AO
    kimg = np.asarray(lay["kernel_image_index"], dtype=np.int64)  # (M, 3)

    M_c = int(aos_data.num_ao)
    nuc_mol = np.asarray(aos_data.nucleus_index, dtype=np.int64)
    orb_mol = np.asarray(aos_data.orbital_indices, dtype=np.int64)
    exps_mol = np.asarray(aos_data.exponents, dtype=np.float64)
    coef_mol = np.asarray(aos_data.coefficients, dtype=np.float64)
    ang_mol = np.asarray(aos_data.angular_momentums, dtype=np.int64)

    # Per-home-AO primitive slices (contiguous in the canonical group-by-AO layout,
    # but gather generically to be safe).
    prims_of_ao = [np.nonzero(orb_mol == h)[0] for h in range(M_c)]
    orbital_indices_r: list[int] = []
    exps_r: list[float] = []
    coef_r: list[float] = []
    radial_map: list[int] = []
    home_map = np.asarray(aos_data.ao_prim_to_shell_prim, dtype=np.int64)
    for k in range(M):
        pr = prims_of_ao[int(kth[k])]
        orbital_indices_r.extend([k] * len(pr))
        exps_r.extend(float(x) for x in exps_mol[pr])
        coef_r.extend(float(x) for x in coef_mol[pr])
        radial_map.extend(int(x) for x in home_map[pr])

    kernel_atom = nuc_mol[kth]  # (M,) molecular atom of each kernel-AO

    common_kwargs = dict(
        structure_data=structure_data,  # molecular (centres reconstructed in-trace)
        nucleus_index=tuple(range(M)),  # identity into the per-kernel-AO centres
        num_ao=M_c,
        num_ao_prim=len(orbital_indices_r),
        orbital_indices=tuple(orbital_indices_r),
        exponents=np.asarray(exps_r, dtype=np.float64),
        coefficients=np.asarray(coef_r, dtype=np.float64),
        radial=aos_data.radial,
        ao_prim_to_shell_prim=tuple(radial_map),
        ao_coefficient_scales=(tuple(aos_data.ao_coefficient_scales[h] for h in kth) if aos_data.ao_coefficient_scales else ()),
        angular_momentums=tuple(int(ang_mol[h]) for h in kth),
        _n_image=int(lay["n_image_union"]),  # display only (info line)
        _image_indices=tuple(tuple(int(c) for c in row) for row in kimg),  # (M, 3) per kernel-AO
        _ragged_kernel_atom=tuple(int(a) for a in kernel_atom),
        _ragged_buckets=tuple((int(b["m"]), int(b["K"]), int(b["kernel_offset"])) for b in lay["buckets"]),
        _ragged_fold_perm=tuple(int(x) for x in lay["fold_perm"]),
        _ragged_kernel_to_home=tuple(int(h) for h in kth),
    )
    # Order-independent shell grouping from the CONCRETE molecular base (correct for
    # non-shell-major layouts). Stored static so the traced-ham c2s twin
    # (_ragged_structural_shell_proxy) need not read the then-traced exponents.
    home_shell_rep = _home_shell_rep_from_molecular(aos_data)
    if home_shell_rep is not None:
        common_kwargs["_ragged_home_shell_rep"] = home_shell_rep
    return common_kwargs, kth


def _detile_ragged_to_molecular(
    aos_data: AOs_cart_pgto_data | AOs_sphe_pgto_data,
) -> AOs_cart_gto_data | AOs_sphe_gto_data:
    """Recover the molecular GTO base from a per-shell ragged pgto leaf.

    The ragged kernel-AO arrays are bucket-major; for each molecular (home) AO
    pick its ``T_0 = (0,0,0)`` kernel-AO (always present) as the representative
    and gather that kernel-AO's angular identity, atom, and contracted
    primitives. Inverse of :func:`_apply_image_expansion`; reused by the HDF5
    de-tile and by :func:`_aos_sphe_to_cart` (which converts at the molecular
    level then re-expands). Numpy / fp64, setup-time only.
    """
    M_c = int(aos_data.num_ao)
    kth = np.asarray(aos_data._ragged_kernel_to_home, dtype=np.int64)
    img = np.asarray(aos_data._image_indices, dtype=np.int64).reshape(-1, 3)
    katom = np.asarray(aos_data._ragged_kernel_atom, dtype=np.int64)
    ang_full = np.asarray(aos_data.angular_momentums, dtype=np.int64)
    orb_full = np.asarray(aos_data.orbital_indices, dtype=np.int64)
    exps_full = np.asarray(aos_data.exponents, dtype=np.float64)
    coef_full = np.asarray(aos_data.coefficients, dtype=np.float64)

    is_t0 = (img == 0).all(axis=1)
    rep = np.full(M_c, -1, dtype=np.int64)  # home AO -> its T_0 kernel-AO
    rep[kth[is_t0]] = np.nonzero(is_t0)[0]
    if (rep < 0).any():
        raise RuntimeError("_detile_ragged_to_molecular: a home AO has no T_0 kernel-AO (ragged invariant broken).")

    mol_orb: list[int] = []
    mol_exp: list[float] = []
    mol_coef: list[float] = []
    mol_map: list[int] = []
    radial_map = np.asarray(aos_data.ao_prim_to_shell_prim, dtype=np.int64)
    for h in range(M_c):
        pr = np.nonzero(orb_full == int(rep[h]))[0]
        mol_orb.extend([h] * len(pr))
        mol_exp.extend(float(x) for x in exps_full[pr])
        mol_coef.extend(float(x) for x in coef_full[pr])
        mol_map.extend(int(x) for x in radial_map[pr])

    common = dict(
        structure_data=aos_data.structure_data,
        k_point=aos_data.k_point,
        nucleus_index=tuple(int(katom[rep[h]]) for h in range(M_c)),
        num_ao=M_c,
        num_ao_prim=len(mol_orb),
        orbital_indices=tuple(mol_orb),
        exponents=np.asarray(mol_exp, dtype=np.float64),
        coefficients=np.asarray(mol_coef, dtype=np.float64),
        radial=aos_data.radial,
        ao_prim_to_shell_prim=tuple(mol_map),
        ao_coefficient_scales=(tuple(aos_data.ao_coefficient_scales[k] for k in rep) if aos_data.ao_coefficient_scales else ()),
        angular_momentums=tuple(int(ang_full[rep[h]]) for h in range(M_c)),
    )
    if isinstance(aos_data, AOs_cart_pgto_data):
        px = np.asarray(aos_data.polynominal_order_x, dtype=np.int64)
        py = np.asarray(aos_data.polynominal_order_y, dtype=np.int64)
        pz = np.asarray(aos_data.polynominal_order_z, dtype=np.int64)
        return AOs_cart_gto_data._from_full_basis(
            **common,
            polynominal_order_x=tuple(int(px[rep[h]]) for h in range(M_c)),
            polynominal_order_y=tuple(int(py[rep[h]]) for h in range(M_c)),
            polynominal_order_z=tuple(int(pz[rep[h]]) for h in range(M_c)),
        )
    if isinstance(aos_data, AOs_sphe_pgto_data):
        mag = np.asarray(aos_data.magnetic_quantum_numbers, dtype=np.int64)
        return AOs_sphe_gto_data._from_full_basis(
            **common,
            magnetic_quantum_numbers=tuple(int(mag[rep[h]]) for h in range(M_c)),
        )
    raise TypeError(f"_detile_ragged_to_molecular: unsupported type {type(aos_data)}")


def _with_periodic_image_expansion_debug(
    aos_data: AOs_cart_gto_data | AOs_sphe_gto_data,
    structure_data: Structure_data,
    tol: float = AO_IMAGE_TRUNCATION_ATOL,
) -> AOs_cart_gto_data | AOs_sphe_gto_data:
    r"""Self-contained reference for :py:func:`_apply_image_expansion`.

    Re-implements the structure + AO + primitive triple expansion from
    scratch with **no calls to the production helpers it is meant to
    validate**:

      * the production image-expansion array builders,
      * :py:func:`_primitive_image_list_indices` (and Cartesian sibling),
      * :py:func:`_primitive_image_radius`,
      * :py:meth:`Structure_data.cell_inradius`,
      * :py:meth:`Structure_data.primitive_image_list`.

    All formulas (cell inradius, the simpler upper-bound r_cut, image-set
    enumeration and union, deterministic ordering with
    ``T_0 = (0, 0, 0)`` first, three-level tile arithmetic) are inlined
    with explicit Python loops. fp64 numpy throughout; no JIT, no
    ``vmap``, no ``lax.scan``.

    Cross-checking the production helper-based path against this oracle
    in one bit-identical comparison validates the entire helper chain in
    a single test; that is why this ``_debug`` reference stays helper-free
    and at entry-point granularity.
    """
    if not structure_data.pbc_flag:
        return aos_data
    if aos_data._n_image != 1:
        raise ValueError(f"_apply_image_expansion is not idempotent; expected _n_image == 1, got {aos_data._n_image}.")

    # Cell vectors and inline inradius (no Structure_data.cell_inradius).
    cell = np.asarray(structure_data.cell, dtype=np.float64)
    volume = float(np.abs(np.dot(cell[0], np.cross(cell[1], cell[2]))))
    face_areas = [
        float(np.linalg.norm(np.cross(cell[1], cell[2]))),
        float(np.linalg.norm(np.cross(cell[2], cell[0]))),
        float(np.linalg.norm(np.cross(cell[0], cell[1]))),
    ]
    L_min = volume / max(face_areas)

    angular_momentums_mol = list(aos_data.angular_momentums)
    orbital_indices_mol = list(aos_data.orbital_indices)
    exponents_mol = np.asarray(aos_data.exponents, dtype=np.float64).tolist()
    coefficients_mol = np.asarray(aos_data.coefficients, dtype=np.float64).tolist()
    L_per_prim = [angular_momentums_mol[i] for i in orbital_indices_mol]

    # Per-ATOM image-set union (inline r_cut, inline atom-to-cell
    # closest-approach test). The r_cut solves r^L e^{-alpha r^2} <= tol as an
    # absolute envelope bound via fixed-point iteration from the L = 0
    # Gaussian-only root (mirrors _primitive_image_radius; the earlier closed-
    # form ``r_peak * log_inv_tol / L`` upper bound over-estimated r_cut by
    # ~4-5x and is intentionally NOT used here).
    #
    # Atom-to-cell metric (Omega - R_A): keep image n of a primitive on atom A
    # iff that copy can reach within r_cut of some electron in the home cell.
    # The per-axis fractional displacement set is [-n - s_A, 1 - n - s_A]; its
    # closest point to 0 is clamp(0, lo, hi). This per-axis clamp is exact for
    # ORTHOGONAL cells (the cubic test fixtures); the production oblique-safe
    # coordinate descent agrees there. s_A is the atom fractional position in
    # the home cell Omega = [0, 1)^3 (electrons are wrapped into Omega).
    # s_A is the UNWRAPPED fractional position (the centres below are R + T_n
    # with the unwrapped R); the per-axis test runs on its wrapped part w and the
    # image index is shifted back by floor(s_A).
    inv_cell = np.linalg.inv(cell)
    frac_positions = np.asarray(structure_data.positions, dtype=np.float64) @ inv_cell
    nucleus_of_ao = [int(i) for i in aos_data.nucleus_index]
    atom_of_prim = [nucleus_of_ao[int(orbital_indices_mol[p])] for p in range(len(orbital_indices_mol))]

    def _clamp0(lo: float, hi: float) -> float:
        """Closest point to 0 in the interval ``[lo, hi]`` (lo <= hi)."""
        if lo > 0.0:
            return lo
        if hi < 0.0:
            return hi
        return 0.0

    # Value-aware envelope tolerance (issue #120): the AO value of one image is
    # sum_p c_p N_p A(r) exp(-alpha_p r^2) with |N_p A(r)| <= P_p r^L, P_p the
    # largest normalised angular prefactor over the spherical (Racah solid
    # harmonic, |S_LM| <= r^L) and Cartesian (max |x^nx y^ny z^nz| / r^L =
    # sqrt(prod n_i^n_i / L^L)) forms. Primitive p of AO mu (K_mu primitives) is
    # selected at the envelope tolerance tol / max(1, K_mu |c_p| P_p), inlined here.
    def _prefactor(alpha: float, L: int) -> float:
        fact = scipy.special.factorial
        best = np.sqrt(
            2.0 ** (2 * L + 3) * fact(L + 1) / fact(2 * L + 2) * (2.0 * alpha) ** (L + 1.5) / np.sqrt(np.pi)
        ) * np.sqrt((2 * L + 1) / (4.0 * np.pi))
        for nx in range(L + 1):
            for ny in range(L + 1 - nx):
                nz = L - nx - ny
                ratio = fact(nx) * fact(ny) * fact(nz) / (fact(2 * nx) * fact(2 * ny) * fact(2 * nz))
                norm = np.sqrt((2.0 * alpha / np.pi) ** 1.5 * (8.0 * alpha) ** L * ratio)
                angular = np.sqrt(float(nx) ** nx * float(ny) ** ny * float(nz) ** nz / float(L) ** L) if L > 0 else 1.0
                best = max(best, norm * angular)
        return float(best)

    prims_of_ao = [0] * int(aos_data.num_ao)
    for orb in orbital_indices_mol:
        prims_of_ao[int(orb)] += 1

    global_images: set[tuple[int, int, int]] = set()
    for p, (alpha_p, L_p) in enumerate(zip(exponents_mol, L_per_prim, strict=True)):
        share = prims_of_ao[int(orbital_indices_mol[p])] * abs(coefficients_mol[p]) * _prefactor(alpha_p, L_p)
        tol_p = float(tol) / max(1.0, share)
        log_inv_tol = max(-float(np.log(tol_p)), 1.0)
        if L_p <= 0:
            r_cut = float(np.sqrt(log_inv_tol / alpha_p))
        else:
            r_cut = float(np.sqrt(log_inv_tol / alpha_p))
            for _ in range(5):
                r_cut = float(np.sqrt((log_inv_tol + L_p * np.log(max(r_cut, 1.0))) / alpha_p))
        s_full = [float(x) for x in frac_positions[atom_of_prim[p]]]
        f_a, f_b, f_c = (int(np.floor(x)) for x in s_full)
        s_a, s_b, s_c = s_full[0] - f_a, s_full[1] - f_b, s_full[2] - f_c
        bound = int(np.ceil(r_cut / L_min)) + 1
        for na in range(-bound, bound + 1):
            cx = _clamp0(-na - s_a, 1.0 - na - s_a)
            for nb in range(-bound, bound + 1):
                cy = _clamp0(-nb - s_b, 1.0 - nb - s_b)
                for nc in range(-bound, bound + 1):
                    cz = _clamp0(-nc - s_c, 1.0 - nc - s_c)
                    disp = cx * cell[0] + cy * cell[1] + cz * cell[2]
                    if float(np.linalg.norm(disp)) <= r_cut:
                        global_images.add((int(na) - f_a, int(nb) - f_b, int(nc) - f_c))
    global_images.add((0, 0, 0))

    # Deterministic ordering with T_0 = (0, 0, 0) first.
    sorted_images = sorted(
        global_images,
        key=lambda t: (abs(t[0]) + abs(t[1]) + abs(t[2]), t),
    )
    if (0, 0, 0) in global_images:
        sorted_images.remove((0, 0, 0))
        sorted_images.insert(0, (0, 0, 0))
    n_image = len(sorted_images)

    # Tile structure positions and labels (explicit Python loop).
    num_atoms = len(structure_data.atomic_numbers)
    R = np.asarray(structure_data.positions, dtype=np.float64)
    new_positions_list = []
    for na, nb, nc in sorted_images:
        T_n = na * cell[0] + nb * cell[1] + nc * cell[2]
        for I in range(num_atoms):
            new_positions_list.append((R[I] + T_n).tolist())
    new_positions = np.array(new_positions_list, dtype=np.float64)

    expanded_structure = Structure_data.build(
        pbc_flag=True,
        positions=new_positions,
        vec_a=structure_data.vec_a,
        vec_b=structure_data.vec_b,
        vec_c=structure_data.vec_c,
        atomic_numbers=tuple(structure_data.atomic_numbers) * n_image,
        element_symbols=tuple(structure_data.element_symbols) * n_image,
        atomic_labels=tuple(structure_data.atomic_labels) * n_image,
    )

    # Tile per-AO arrays with explicit image-shift offsets.
    M_c = int(aos_data.num_ao)
    nucleus_index_mol = [int(i) for i in aos_data.nucleus_index]
    new_nucleus_index: list[int] = []
    for n_idx in range(n_image):
        for orig_nuc in nucleus_index_mol:
            new_nucleus_index.append(orig_nuc + n_idx * num_atoms)

    # Tile per-primitive arrays with explicit image-shift offsets.
    N_p = int(aos_data.num_ao_prim)
    new_orbital_indices: list[int] = []
    for n_idx in range(n_image):
        for orig_orb in orbital_indices_mol:
            new_orbital_indices.append(int(orig_orb) + n_idx * M_c)
    new_exponents_list: list[float] = []
    new_coefficients_list: list[float] = []
    for _ in range(n_image):
        for p in range(N_p):
            new_exponents_list.append(float(exponents_mol[p]))
            new_coefficients_list.append(float(coefficients_mol[p]))
    new_exponents = np.array(new_exponents_list, dtype=np.float64)
    new_coefficients = np.array(new_coefficients_list, dtype=np.float64)

    common_kwargs = dict(
        structure_data=expanded_structure,
        nucleus_index=tuple(new_nucleus_index),
        # num_ao is the molecular view; tiled arrays validate against
        # _num_ao_kernel = n_image * num_ao.
        num_ao=M_c,
        num_ao_prim=n_image * N_p,
        orbital_indices=tuple(new_orbital_indices),
        exponents=new_exponents,
        coefficients=new_coefficients,
        angular_momentums=tuple(angular_momentums_mol) * n_image,
        _n_image=n_image,
    )
    if isinstance(aos_data, AOs_cart_gto_data):
        new_instance: AOs_cart_gto_data | AOs_sphe_gto_data = AOs_cart_gto_data._from_full_basis(
            **common_kwargs,
            polynominal_order_x=tuple(aos_data.polynominal_order_x) * n_image,
            polynominal_order_y=tuple(aos_data.polynominal_order_y) * n_image,
            polynominal_order_z=tuple(aos_data.polynominal_order_z) * n_image,
        )
    elif isinstance(aos_data, AOs_sphe_gto_data):
        new_instance = AOs_sphe_gto_data._from_full_basis(
            **common_kwargs,
            magnetic_quantum_numbers=tuple(aos_data.magnetic_quantum_numbers) * n_image,
        )
    else:
        raise TypeError(f"_with_periodic_image_expansion_debug: unsupported type {type(aos_data)}")
    object.__setattr__(new_instance, "_image_truncation_tol", float(tol))
    return new_instance


def _gaussian_moment_even(order: int, exponent_sum: float) -> float:
    r"""Return :math:`\int t^{order} e^{-p t^2} dt` over :math:`(-\infty,\infty)` for even order."""
    if order % 2 != 0:
        return 0.0

    k = order // 2
    if k == 0:
        double_factorial = 1.0
    else:
        double_factorial = float(scipy.special.factorial2(2 * k - 1, exact=False))

    return double_factorial * np.sqrt(np.pi) / (2.0**k * (exponent_sum ** (k + 0.5)))


def _compute_overlap_1d_cart_pairs(
    alpha: npt.NDArray[np.float64],
    beta: npt.NDArray[np.float64],
    center_a: npt.NDArray[np.float64],
    center_b: npt.NDArray[np.float64],
    order_a: npt.NDArray[np.int64],
    order_b: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    r"""Closed-form 1D overlaps of Cartesian polynomial Gaussians for a block of primitive pairs.

    Bra arrays have shape ``(P, 1)`` and ket arrays ``(1, Q)``; the result is
    ``(P, Q)``. For each pair, with ``p = alpha + beta`` and product centre
    ``X_P``, the binomial expansion of ``(x - A)^{l_a} (x - B)^{l_b}`` about
    ``X_P`` gives

    .. math::
       \sum_{i \le l_a} \sum_{j \le l_b,\, i + j\ \mathrm{even}}
       \binom{l_a}{i}\binom{l_b}{j} (X_P - A)^{l_a - i} (X_P - B)^{l_b - j}
       \int t^{i+j} e^{-p t^2} dt .

    Terms with ``i > l_a`` or ``j > l_b`` are masked to zero, so mixed angular
    momenta share one loop over ``(i, j)``.
    """
    exponent_sum = alpha + beta
    product_center = (alpha * center_a + beta * center_b) / exponent_sum
    shift_a = product_center - center_a
    shift_b = product_center - center_b
    value = np.zeros(np.broadcast_shapes(alpha.shape, beta.shape), dtype=np.float64)
    for i in range(int(np.max(order_a)) + 1):
        coeff_i = np.where(
            order_a >= i,
            scipy.special.comb(order_a, i) * shift_a ** np.maximum(order_a - i, 0),
            0.0,
        )
        for j in range(int(np.max(order_b)) + 1):
            if (i + j) % 2 != 0:
                continue
            coeff_j = np.where(
                order_b >= j,
                scipy.special.comb(order_b, j) * shift_b ** np.maximum(order_b - j, 0),
                0.0,
            )
            value += coeff_i * coeff_j * _gaussian_moment_even(i + j, exponent_sum)
    return value


def _compute_overlap_1d_cart(
    alpha: float,
    beta: float,
    center_a: float,
    center_b: float,
    order_a: int,
    order_b: int,
) -> float:
    """Closed-form 1D overlap integral for Cartesian polynomial Gaussian factors (one primitive pair).

    Scalar form of :py:func:`_compute_overlap_1d_cart_pairs`, which owns the
    formula; kept as the single-pair entry point.
    """
    value = _compute_overlap_1d_cart_pairs(
        np.full((1, 1), alpha, dtype=np.float64),
        np.full((1, 1), beta, dtype=np.float64),
        np.full((1, 1), center_a, dtype=np.float64),
        np.full((1, 1), center_b, dtype=np.float64),
        np.full((1, 1), order_a, dtype=np.int64),
        np.full((1, 1), order_b, dtype=np.int64),
    )
    return float(value[0, 0])


def _group_aos_by_primitive_budget(primitive_counts: npt.NDArray[np.int64], target: int) -> list[tuple[int, int]]:
    """Split a sequence of AOs into consecutive ``[start, end)`` groups of at most ``target`` primitives.

    A single AO with more than ``target`` primitives forms its own group.
    """
    groups: list[tuple[int, int]] = []
    start = 0
    running = 0
    for index, count in enumerate(np.asarray(primitive_counts, dtype=np.int64)):
        if index > start and running + int(count) > target:
            groups.append((start, index))
            start = index
            running = 0
        running += int(count)
    if start < len(primitive_counts):
        groups.append((start, len(primitive_counts)))
    return groups


def _segment_sum(values: npt.NDArray[np.float64], counts: npt.NDArray[np.int64], axis: int) -> npt.NDArray[np.float64]:
    """Sum consecutive segments of ``counts[k]`` entries along ``axis`` (0 or 1); empty segments give zero."""
    shape = list(values.shape)
    shape[axis] = counts.size
    out = np.zeros(shape, dtype=values.dtype)
    nonempty = counts > 0
    if np.any(nonempty):
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))[nonempty]
        reduced = np.add.reduceat(values, starts, axis=axis)
        if axis == 0:
            out[nonempty, :] = reduced
        else:
            out[:, nonempty] = reduced
    return out


def _compute_overlap_matrix_cart_analytic(aos_cart_data: AOs_cart_gto_data) -> npt.NDArray[np.float64]:
    """Compute AO overlap matrix analytically for Cartesian contracted GTOs.

    For a molecular GTO leaf this returns the full ``(M_c, M_c)`` matrix.
    The helper also retains support for explicitly supplied image-expanded
    data in low-level diagnostics: that form returns the home-cell rows
    ``S[0:M_c, :]`` at shape ``(M_c, _num_ao_kernel)``. Production periodic
    pGTO overlap does not use that diagnostic form; it calls
    :py:func:`_compute_overlap_matrix_periodic_pgto_lattice` with an
    overlap-owned primitive-pair image set. The full
    ``(_num_ao_kernel, _num_ao_kernel)`` matrix is not built: the periodic
    image fold uses only the home-cell rows, so computing the remaining
    ``n_image - 1`` row blocks would be wasted work and -- more importantly --
    wasted memory (the allocation scales as ``n_image^2`` instead of
    ``n_image``).
    """
    # Zone is ``det_eval``: the overlap matrix is consumed by det/SVD math
    # (S^{1/2}, polar retraction, geminal AO->MO conversion). Tagging the
    # internals ``ao_eval`` ran the analytic kernel at fp32 under mixed
    # precision, baking fp32 damage into S that no downstream re-cast can
    # undo. This kernel is one-shot at setup -- there is no hot-loop cost
    # benefit to fp32 here.
    dtype_np = get_dtype_np("det_eval")
    num_ao = aos_cart_data._num_ao_kernel
    num_ao_home = aos_cart_data.num_ao  # molecular rank M_c (== num_ao for molecular gto)
    overlap_matrix = np.zeros((num_ao_home, num_ao), dtype=dtype_np)

    centers = np.asarray(aos_cart_data._atomic_center_carts_np, dtype=dtype_np)
    orbital_indices = list(aos_cart_data.orbital_indices)

    exponents = np.asarray(aos_cart_data.exponents, dtype=dtype_np)
    coefficients = np.asarray(aos_cart_data.coefficients, dtype=dtype_np)

    nx = np.asarray(aos_cart_data.polynominal_order_x, dtype=np.int32)
    ny = np.asarray(aos_cart_data.polynominal_order_y, dtype=np.int32)
    nz = np.asarray(aos_cart_data.polynominal_order_z, dtype=np.int32)

    normalization = np.sqrt(
        (2.0 * exponents / np.pi) ** (3.0 / 2.0)
        * (8.0 * exponents) ** np.asarray(aos_cart_data._angular_momentums_prim_np, dtype=np.int32)
        * np.asarray(aos_cart_data._normalization_factorial_ratio_prim_jnp, dtype=dtype_np)
    )

    # Bra rows are the HOME-cell AOs. Under the per-shell ragged layout the home
    # AOs are NOT the first ``M_c`` kernel-AOs (the kernel is bucket-major), so
    # map each home AO to its ``T_0 = (0,0,0)`` kernel-AO; for molecular /
    # single-image leaves the first ``M_c`` kernel-AOs already ARE the home cell
    # (``rep == arange``). Columns range over ALL kernel-AOs (the image fold
    # downstream collapses them); no upper-triangle shortcut since the home rows
    # are not a prefix in the ragged order.
    if getattr(aos_cart_data, "_ragged_buckets", ()):
        kth = np.asarray(aos_cart_data._ragged_kernel_to_home, dtype=np.int64)
        img = np.asarray(aos_cart_data._image_indices, dtype=np.int64).reshape(-1, 3)
        is_t0 = (img == 0).all(axis=1)
        rep = np.full(num_ao_home, -1, dtype=np.int64)
        rep[kth[is_t0]] = np.nonzero(is_t0)[0]
    else:
        rep = np.arange(num_ao_home, dtype=np.int64)

    # Primitive-pair blocks (vectorized). Primitives are sorted by kernel AO so
    # each AO owns one contiguous primitive segment; a block is a group of bra
    # home AOs times a group of ket kernel AOs, cut at AO boundaries so that
    # about ``ANALYTIC_OVERLAP_PAIR_WORK_ARRAYS`` live (bra, ket) float64
    # arrays stay within ``jqmc._setting.ANALYTIC_OVERLAP_PAIR_BUDGET_BYTES``
    # (read at call time so a caller can lower it).
    orbital_index_array = np.asarray(orbital_indices, dtype=np.int64)
    primitive_order = np.argsort(orbital_index_array, kind="stable")
    primitive_counts = np.bincount(orbital_index_array, minlength=num_ao).astype(np.int64)
    primitive_offsets = np.concatenate(([0], np.cumsum(primitive_counts)))
    primitive_weight = coefficients * normalization
    primitive_centers = centers[orbital_index_array]
    primitive_orders = np.stack([nx, ny, nz], axis=1).astype(np.int64)[orbital_index_array]

    def _primitives_of(aos: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        return np.concatenate(
            [primitive_order[primitive_offsets[a] : primitive_offsets[a + 1]] for a in aos] + [np.zeros(0, dtype=np.int64)]
        )

    target = max(
        1, int(np.sqrt(_setting.ANALYTIC_OVERLAP_PAIR_BUDGET_BYTES / (8.0 * _setting.ANALYTIC_OVERLAP_PAIR_WORK_ARRAYS)))
    )
    bra_aos_all = rep.astype(np.int64)
    ket_aos_all = np.arange(num_ao, dtype=np.int64)
    for bra_start, bra_end in _group_aos_by_primitive_budget(primitive_counts[bra_aos_all], target):
        bra_aos = bra_aos_all[bra_start:bra_end]
        bra = _primitives_of(bra_aos)
        if bra.size == 0:
            continue
        for ket_start, ket_end in _group_aos_by_primitive_budget(primitive_counts[ket_aos_all], target):
            ket_aos = ket_aos_all[ket_start:ket_end]
            ket = _primitives_of(ket_aos)
            if ket.size == 0:
                continue
            alpha = exponents[bra][:, None]
            beta = exponents[ket][None, :]
            center_bra = primitive_centers[bra]
            center_ket = primitive_centers[ket]
            rab2 = np.sum((center_bra[:, None, :] - center_ket[None, :, :]) ** 2, axis=-1)
            pair_values = np.exp(-alpha * beta / (alpha + beta) * rab2)
            for axis in range(3):
                pair_values *= _compute_overlap_1d_cart_pairs(
                    alpha,
                    beta,
                    center_bra[:, axis][:, None],
                    center_ket[:, axis][None, :],
                    primitive_orders[bra, axis][:, None],
                    primitive_orders[ket, axis][None, :],
                )
            pair_values *= primitive_weight[bra][:, None] * primitive_weight[ket][None, :]
            block = _segment_sum(pair_values, primitive_counts[ket_aos], axis=1)
            overlap_matrix[bra_start:bra_end, ket_start:ket_end] = _segment_sum(block, primitive_counts[bra_aos], axis=0)

    return overlap_matrix


def _compute_overlap_matrix_debug(
    aos_data: AOs_sphe_gto_data | AOs_cart_gto_data,
    num_grid_points: int = 35,
    tail_tolerance: float = 1.0e-11,
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    """Numerically compute the AO overlap matrix by 3D midpoint integration.

    For molecular ``aos_data`` (``_n_image == 1``): integrate over the
    Cartesian box spanning the AO centres plus a margin derived from
    ``tail_tolerance`` and the smallest positive exponent. The margin is
    clipped to 6--20 Bohr; the tolerance therefore sets an estimated extent,
    not a guaranteed integration error. Evaluate the uniform midpoint grid
    through the Python-loop oracle :py:func:`_compute_AOs_debug`.

    For periodic ``aos_data`` (``_n_image > 1``): integrate over a single
    cell of the parallelepiped lattice. The grid is the
    ``num_grid_points ** 3`` fractional midpoints ``((i + 0.5) / N,
    (j + 0.5) / N, (k + 0.5) / N)`` mapped to Cartesian by the cell
    matrix, with ``volume_element = V_cell / num_grid_points ** 3``.
    AOs are evaluated through the JIT-compiled public
    :py:func:`compute_AOs` rather than the Python-loop
    :py:func:`_compute_AOs_debug` because the latter scales as
    ``O(num_ao_expanded * N_g)`` with high Python-numpy overhead per
    inner iteration -- intractable at ``num_ao = n_image * M_c`` of
    several hundred or more. This mirrors the pattern already used by
    :py:func:`_compute_AOs_grad_debug` /
    :py:func:`_compute_AOs_laplacian_debug` (finite-difference on top
    of ``compute_AOs``). ``compute_AOs`` is itself public-API-rank for
    periodic input (post-kernel reshape-sum), so the overlap output is
    the molecular-rank ``(M_c, M_c)`` matrix expected by callers
    without any further reduction.
    """
    dtype_np = np.result_type(
        get_dtype_np("ao_eval"),
        np.complex64 if is_complex_aos(aos_data) else np.float32,
    )
    if aos_data._n_image > 1:
        cell = np.asarray(aos_data.structure_data.cell, dtype=np.float64)
        steps = (np.arange(num_grid_points, dtype=np.float64) + 0.5) / num_grid_points
        fa, fb, fc = np.meshgrid(steps, steps, steps, indexing="ij")
        r_frac = np.stack([fa.ravel(), fb.ravel(), fc.ravel()], axis=-1)
        r_carts = r_frac @ cell  # (N_g, 3) Cartesian
        volume_element = float(np.abs(np.linalg.det(cell))) / float(num_grid_points**3)
        ao_values = np.asarray(
            compute_AOs(aos_data, jnp.asarray(r_carts, dtype=jnp.float64)),
            dtype=dtype_np,
        )
    else:
        grid_dtype_np = get_dtype_np("ao_eval")
        if num_grid_points < 3:
            raise ValueError(f"num_grid_points must be >= 3. Got {num_grid_points}.")
        if tail_tolerance <= 0.0 or tail_tolerance >= 1.0:
            raise ValueError(f"tail_tolerance must satisfy 0 < tail_tolerance < 1. Got {tail_tolerance}.")

        centers = np.asarray(aos_data.structure_data.positions, dtype=grid_dtype_np)
        exponents = np.asarray(aos_data.exponents, dtype=grid_dtype_np)
        positive_exponents = exponents[exponents > 0.0]
        if positive_exponents.size == 0:
            raise ValueError("All AO exponents are non-positive. Cannot estimate integration bounds.")

        exponent_min = float(np.min(positive_exponents))
        margin = np.sqrt(max(-np.log(tail_tolerance), 1.0) / exponent_min)
        margin = float(np.clip(margin, 6.0, 20.0))
        lower = np.min(centers, axis=0) - margin
        upper = np.max(centers, axis=0) + margin
        lengths = upper - lower
        dx, dy, dz = lengths / float(num_grid_points)

        x = lower[0] + (np.arange(num_grid_points, dtype=grid_dtype_np) + 0.5) * dx
        y = lower[1] + (np.arange(num_grid_points, dtype=grid_dtype_np) + 0.5) * dy
        z = lower[2] + (np.arange(num_grid_points, dtype=grid_dtype_np) + 0.5) * dz
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        r_carts = np.stack([xx, yy, zz], axis=-1).reshape((-1, 3))
        volume_element = float(dx * dy * dz)

        ao_values = np.asarray(
            _compute_AOs_debug(aos_data=aos_data, r_carts=r_carts),
            dtype=dtype_np,
        )

    overlap_matrix = (ao_values.conj() @ ao_values.T) * volume_element
    overlap_matrix = 0.5 * (overlap_matrix + overlap_matrix.conj().T)
    return overlap_matrix


def _gram_with_overlap(
    bra: npt.ArrayLike,
    overlap: npt.ArrayLike,
    ket: npt.ArrayLike,
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    r"""Return :math:`\overline{B}\,S\,K^{T}` in fp64 / complex128.

    ``overlap`` is either the dense ``(n_bra_ao, n_ket_ao)`` block or, for a
    diagonal overlap, its ``(num_ao,)`` diagonal. Every input is promoted to
    float64, or to complex128 when any of them is complex.
    """
    bra, overlap, ket = np.asarray(bra), np.asarray(overlap), np.asarray(ket)
    dtype = np.complex128 if any(np.iscomplexobj(a) for a in (bra, overlap, ket)) else np.float64
    bra, overlap, ket = bra.astype(dtype), overlap.astype(dtype), ket.astype(dtype)
    if overlap.ndim == 1:
        return (bra.conj() * overlap[None, :]) @ ket.T
    return bra.conj() @ overlap @ ket.T


def _compute_overlap_matrix_plane_wave(aos_data: AOs_plane_wave_data) -> npt.NDArray[np.float64]:
    r"""Analytic cell-domain overlap of the real Gamma-point plane-wave basis.

    The basis functions are :math:`\cos(G.r)` / :math:`\sin(G.r)` for the
    reciprocal vectors ``G`` stored in ``g_vectors`` (the constant term is
    ``G = 0``, ``cos``). Over one cell :math:`\Omega` of volume
    :math:`V = |\det(\mathrm{cell})|`, using
    :math:`\int_\Omega \cos(K.r)\,dr = V\,\delta_{K,0}` and
    :math:`\int_\Omega \sin(K.r)\,dr = 0` for any reciprocal vector ``K``,

    .. math::
       \int_\Omega \cos(G_a.r)\cos(G_b.r)\,dr
            &= \tfrac{V}{2}\,(\delta_{G_a,G_b} + \delta_{G_a,-G_b}), \\
       \int_\Omega \sin(G_a.r)\sin(G_b.r)\,dr
            &= \tfrac{V}{2}\,(\delta_{G_a,G_b} - \delta_{G_a,-G_b}), \\
       \int_\Omega \cos(G_a.r)\sin(G_b.r)\,dr &= 0 .

    The :math:`G_a = -G_b` term collapses the ``G = 0`` constant to ``V`` and
    leaves the standard half-space basis exactly diagonal (``V`` for the
    constant, ``V/2`` otherwise). Returned at molecular rank ``(num_ao,
    num_ao)`` (plane-wave leaves carry no periodic image expansion).
    """
    dtype_np = get_dtype_np("ao_eval")
    cell = np.asarray(aos_data.structure_data.cell, dtype=np.float64)
    cell_volume = float(np.abs(np.linalg.det(cell)))

    g = np.asarray(aos_data.g_vectors, dtype=np.float64)  # (num_ao, 3)
    is_sin = np.asarray(aos_data.phase_is_sin, dtype=np.float64) > 0.5  # (num_ao,)

    match_tol = 1.0e-8
    diff = g[:, None, :] - g[None, :, :]
    summ = g[:, None, :] + g[None, :, :]
    equal_plus = np.all(np.abs(diff) < match_tol, axis=-1).astype(np.float64)  # G_a == G_b
    equal_minus = np.all(np.abs(summ) < match_tol, axis=-1).astype(np.float64)  # G_a == -G_b

    cos_mask = (~is_sin).astype(np.float64)  # (num_ao,)
    sin_mask = is_sin.astype(np.float64)
    cos_cos = cos_mask[:, None] * cos_mask[None, :]
    sin_sin = sin_mask[:, None] * sin_mask[None, :]

    overlap_matrix = 0.5 * cell_volume * (cos_cos * (equal_plus + equal_minus) + sin_sin * (equal_plus - equal_minus))
    return overlap_matrix.astype(dtype_np)


def _compute_overlap_matrix_bspline(aos_data: AOs_bspline_data) -> npt.NDArray[np.float64]:
    r"""Exact cell-domain overlap of the real (Gamma) periodic B-spline band basis.

    The real bands are the same periodic B-spline basis as
    :class:`AOs_bspline_complex_data` with real control points, so the Gram
    matrix ``S_ab = int_cell u_a u_b dr`` is
    :py:func:`_compute_overlap_matrix_bspline_exact` (per-interval weight
    products integrated exactly, per-axis circulant diagonalised by FFT); its
    imaginary part is round-off of the real input and is dropped. No
    quadrature error; returned as ``float64`` (the ``det_eval`` overlap zone),
    independent of the AO-evaluation precision, at molecular rank
    ``(num_ao, num_ao)``. A ``2 * spline_shape`` fractional midpoint rule
    would instead carry a relative quadrature error up to about ``4e-3``
    (cubic) / ``6e-5`` (quintic) for rough bands.
    """
    exact = _compute_overlap_matrix_bspline_exact(
        cell=np.asarray(aos_data.structure_data.cell, dtype=np.float64),
        spline_coefficients=np.asarray(aos_data.spline_coefficients, dtype=np.float64),
        spline_shape=tuple(int(n) for n in aos_data.spline_shape),
        spline_order=int(aos_data.spline_order),
    )
    overlap = np.ascontiguousarray(np.real(exact), dtype=np.float64)
    return 0.5 * (overlap + overlap.T)


def _bspline_offset_autocorrelation(spline_order: int) -> npt.NDArray[np.float64]:
    r"""Exact overlaps ``m(d) = int B(t) B(t - d) dt`` of the leaf's cardinal B-spline.

    On one knot interval ``t in [i, i + 1)`` the spline is
    ``u(t) = sum_p c_{i + o_p} w_p(f)`` with the ``order + 1`` weight
    polynomials ``w_p`` of :py:meth:`AOs_bspline_data._bspline_weights_np`
    (degree ``order``, ``f = t - i``) and consecutive offsets ``o_p``. The
    per-interval Gram ``G_pq = int_0^1 w_p w_q df`` has degree ``2 * order``
    integrands, so the ``order + 1``-node Gauss-Legendre rule (exact through
    degree ``2 * order + 1``) gives it exactly. Summed over intervals, a
    control-point pair ``(k, l)`` couples through ``m(k - l)`` with
    ``m(d) = sum_{p - q = d} G_pq``. Returns ``m`` for ``d = -order..order``
    (index ``d + order``), in fp64 regardless of the precision mode.
    """
    nodes, weights = np.polynomial.legendre.leggauss(spline_order + 1)
    offsets_f = 0.5 * (nodes + 1.0)
    weights_f = 0.5 * weights
    values = np.stack([AOs_bspline_data._bspline_weights_np(np.float64(f), spline_order) for f in offsets_f])
    interval_gram = (values * weights_f[:, None]).T @ values  # (order + 1, order + 1)
    autocorrelation = np.zeros(2 * spline_order + 1, dtype=np.float64)
    for p in range(spline_order + 1):
        for q in range(spline_order + 1):
            autocorrelation[p - q + spline_order] += interval_gram[p, q]
    return autocorrelation


def _compute_overlap_matrix_bspline_exact(
    *,
    cell: npt.NDArray[np.float64],
    spline_coefficients: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
    spline_shape: tuple[int, int, int],
    spline_order: int,
) -> npt.NDArray[np.complex128]:
    r"""Exact cell Gram matrix of periodic B-spline bands with (complex) control points.

    For ``u_a(r) = sum_k c_{a,k} prod_x B(N_x s_x(r) - k_x)`` (``s = r cell^{-1}``
    the fractional coordinate, periodic wrap of ``k`` modulo ``N``) the cell
    integral factorises per axis:

    .. math::
       S_{ab} = \int_\Omega \overline{u_a}\,u_b\,dr
              = \frac{|\det \mathrm{cell}|}{N_0 N_1 N_2}
                \sum_{k,l} \overline{c_{a,k}}\,c_{b,l}
                \prod_{x} m\big((k_x - l_x) \bmod N_x\big),

    with ``m`` from :py:func:`_bspline_offset_autocorrelation` (the factor
    ``1/N_x`` is ``ds_x = dt_x / N_x``). Each per-axis operator is circulant,
    with first column ``kernel_x[d mod N_x] += m(d)`` (the accumulation also
    covers a wrap-around alias ``d = d' (mod N_x)`` on short grids), so the
    discrete Fourier transform diagonalises it with real eigenvalues
    ``lambda_x(q) = sum_d m(d) e^{-2 pi i q d / N_x}``; ``lambda_x(q) > 0``
    because it is the norm of the spline built from the Fourier mode ``q``.
    Parseval then gives ``S = V / (N_prod)^2 * conj(W) W^T`` with
    ``W = sqrt(lambda) * fftn(c)``, Hermitian positive semidefinite by
    construction. The bra is conjugated (:py:func:`compute_overlap_matrix`
    convention). There is no quadrature error: the result is exact up to
    fp64 / complex128 round-off and does not depend on the AO-evaluation
    precision.

    Returns:
        ``(num_ao, num_ao)`` complex128 Hermitian matrix.
    """
    num_ao = int(np.asarray(spline_coefficients).shape[0])
    shape = tuple(int(n) for n in spline_shape)
    order = int(spline_order)
    autocorrelation = _bspline_offset_autocorrelation(order)
    eigenvalues = np.ones((1, 1, 1), dtype=np.float64)
    for axis, n_axis in enumerate(shape):
        kernel = np.zeros(n_axis, dtype=np.float64)
        np.add.at(kernel, np.mod(np.arange(-order, order + 1), n_axis), autocorrelation)
        eigenvalues_axis = np.real(np.fft.fft(kernel))
        if not np.all(eigenvalues_axis > 0.0):
            raise ValueError(
                f"B-spline Gram eigenvalues must be positive; got min {eigenvalues_axis.min():.3e} on axis {axis}."
            )
        eigenvalues = eigenvalues * eigenvalues_axis.reshape([n_axis if a == axis else 1 for a in range(3)])
    weighted = np.fft.fftn(
        np.asarray(spline_coefficients, dtype=np.complex128).reshape((num_ao, *shape)),
        axes=(1, 2, 3),
    )
    weighted *= np.sqrt(eigenvalues)[None, ...]
    weighted = weighted.reshape(num_ao, -1)
    n_prod = float(np.prod(shape))
    cell_volume = float(np.abs(np.linalg.det(np.asarray(cell, dtype=np.float64))))
    overlap = (cell_volume / (n_prod * n_prod)) * (weighted.conj() @ weighted.T)
    return 0.5 * (overlap + overlap.conj().T)


def _compute_overlap_1d_cart_vectorized(
    alpha: float,
    beta: float,
    center_a: float,
    center_b: np.ndarray,
    order_a: int,
    order_b: int,
) -> np.ndarray:
    """Vectorized closed-form 1D overlap for a translated ket centre."""
    exponent_sum = alpha + beta
    product_center = (alpha * center_a + beta * center_b) / exponent_sum
    shift_a = product_center - center_a
    shift_b = product_center - center_b
    value = np.zeros_like(center_b, dtype=np.float64)
    for i in range(order_a + 1):
        coeff_i = comb(order_a, i) * shift_a ** (order_a - i)
        for j in range(order_b + 1):
            total_order = i + j
            if total_order % 2 != 0:
                continue
            coeff_j = comb(order_b, j) * shift_b ** (order_b - j)
            value += coeff_i * coeff_j * _gaussian_moment_even(total_order, exponent_sum)
    return value


def _lattice_cell_offsets(cell: np.ndarray, centers: np.ndarray) -> np.ndarray:
    r"""Integer cell index ``floor(R inv(cell))`` of each centre, shape ``(N, 3)``.

    The cell-cell image set (:py:func:`_primitive_image_list_indices`) covers a
    separation whose fractional part lies in ``(-1, 1)^3``, i.e. two centres
    in the SAME cell. For centres ``R_mu``, ``R_nu`` in cells ``F_mu``, ``F_nu``
    the image ``n`` of ``R_nu`` matches the image ``n - (F_mu - F_nu)`` of the
    wrapped pair, so the set is shifted by ``F_mu - F_nu``.
    """
    return np.floor(np.asarray(centers, dtype=np.float64) @ np.linalg.inv(np.asarray(cell, dtype=np.float64))).astype(np.int64)


def _compute_overlap_matrix_periodic_cart_lattice(
    aos_cart_data: AOs_cart_gto_data,
    *,
    k_point: tuple[float, float, float] | None,
    term_tolerance: float,
) -> np.ndarray:
    r"""Analytic one-lattice overlap with primitive-pair-specific tails.

    The lattice cutoff uses ``rho = alpha*beta/(alpha+beta)`` and total
    angular degree, rather than reusing the faster-decaying AO-value image
    cutoff.  Each retained term is still an ordinary closed-form whole-space
    Cartesian Gaussian overlap. Components sharing the same centre and radial
    exponents reuse their primitive-pair products; only one Hermitian half
    is evaluated. Coefficients, Cartesian normalizations and pair cutoffs
    retain their original definitions.
    """
    if term_tolerance <= 0.0 or term_tolerance >= 1.0:
        raise ValueError("periodic-pGTO overlap term_tolerance must satisfy 0 < tol < 1.")
    if aos_cart_data._n_image != 1:
        raise ValueError("periodic Cartesian lattice overlap requires a molecular-rank base leaf.")

    dtype = np.complex128 if k_point is not None else np.float64
    num_ao = int(aos_cart_data.num_ao)
    overlap = np.zeros((num_ao, num_ao), dtype=dtype)
    cell = np.asarray(aos_cart_data.structure_data.cell, dtype=np.float64)
    centers = np.asarray(aos_cart_data._atomic_center_carts_np, dtype=np.float64)
    orbital_indices = np.asarray(aos_cart_data.orbital_indices, dtype=np.int64)
    primitive_indices = [np.nonzero(orbital_indices == ao)[0] for ao in range(num_ao)]
    exponents = np.asarray(aos_cart_data.exponents, dtype=np.float64)
    coefficients = np.asarray(aos_cart_data.coefficients, dtype=np.float64)
    angular_primitive = np.asarray(aos_cart_data._angular_momentums_prim_np, dtype=np.int32)
    normalization = np.sqrt(
        (2.0 * exponents / np.pi) ** 1.5
        * (8.0 * exponents) ** angular_primitive
        * np.asarray(aos_cart_data._normalization_factorial_ratio_prim_jnp, dtype=np.float64)
    )
    nx = np.asarray(aos_cart_data.polynominal_order_x, dtype=np.int32)
    ny = np.asarray(aos_cart_data.polynominal_order_y, dtype=np.int32)
    nz = np.asarray(aos_cart_data.polynominal_order_z, dtype=np.int32)
    k_frac = None if k_point is None else np.asarray(k_point, dtype=np.float64)
    image_cache: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]] = {}
    cell_offsets = _lattice_cell_offsets(cell, centers)

    # Components with the same centre and ordered primitive exponents share
    # every image list and translated Gaussian product. Their coefficients and
    # Cartesian normalization factors remain component-specific.
    radial_groups = {}
    for mu, primitive_mu in enumerate(primitive_indices):
        key = (
            tuple(centers[mu]),
            tuple(exponents[primitive_mu]),
            tuple(angular_primitive[primitive_mu]),
        )
        radial_groups.setdefault(key, []).append(mu)
    groups = [np.asarray(group, dtype=np.int64) for group in radial_groups.values()]

    for group_mu, mus in enumerate(groups):
        center_mu = centers[mus[0]]
        prim_mu = np.stack([primitive_indices[mu] for mu in mus])
        for group_nu in range(group_mu, len(groups)):
            nus = groups[group_nu]
            center_nu = centers[nus[0]]
            prim_nu = np.stack([primitive_indices[nu] for nu in nus])
            if group_mu == group_nu:
                component_mu, component_nu = np.triu_indices(len(mus))
            else:
                component_mu, component_nu = np.indices((len(mus), len(nus)))
                component_mu, component_nu = component_mu.ravel(), component_nu.ravel()
            ao_mu, ao_nu = mus[component_mu], nus[component_nu]
            axis_orders = [
                np.unique(np.stack((orders[ao_mu], orders[ao_nu]), axis=1), axis=0, return_inverse=True)
                for orders in (nx, ny, nz)
            ]
            pair_offset = cell_offsets[mus[0]] - cell_offsets[nus[0]]
            shifted_pair = np.any(pair_offset != 0)
            values = np.zeros(len(ao_mu), dtype=dtype)
            for radial_mu in range(prim_mu.shape[1]):
                primitive_mu = prim_mu[0, radial_mu]
                alpha = float(exponents[primitive_mu])
                for radial_nu in range(prim_nu.shape[1]):
                    primitive_nu = prim_nu[0, radial_nu]
                    beta = float(exponents[primitive_nu])
                    rho = alpha * beta / (alpha + beta)
                    total_angular = int(angular_primitive[primitive_mu] + angular_primitive[primitive_nu])
                    cache_key = (rho, total_angular)
                    cached = image_cache.get(cache_key)
                    if cached is None:
                        indices = _primitive_image_list_indices(cell, rho, total_angular, term_tolerance)
                        translations = np.asarray(indices, dtype=np.float64) @ cell
                        image_cache[cache_key] = (indices, translations)
                    else:
                        indices, translations = cached
                    if shifted_pair:
                        indices = indices + pair_offset[None, :]
                        translations = np.asarray(indices, dtype=np.float64) @ cell
                    translated_nu = center_nu[None, :] + translations
                    rab2 = np.sum((center_mu[None, :] - translated_nu) ** 2, axis=1)
                    primitive_values = np.broadcast_to(np.exp(-rho * rab2), (len(ao_mu), len(indices))).copy()
                    for axis, (orders, inverse) in enumerate(axis_orders):
                        factors = np.stack(
                            [
                                _compute_overlap_1d_cart_vectorized(
                                    alpha, beta, center_mu[axis], translated_nu[:, axis], int(order_a), int(order_b)
                                )
                                for order_a, order_b in orders
                            ]
                        )
                        primitive_values *= factors[inverse]
                    if k_frac is not None:
                        primitive_values = primitive_values * np.exp(2j * np.pi * (indices @ k_frac))[None, :]
                    p_mu = prim_mu[component_mu, radial_mu]
                    p_nu = prim_nu[component_nu, radial_nu]
                    prefactor = coefficients[p_mu] * coefficients[p_nu] * normalization[p_mu] * normalization[p_nu]
                    values += prefactor * np.sum(primitive_values, axis=1)
            overlap[ao_mu, ao_nu] = values
            overlap[ao_nu, ao_mu] = values.conj()
    return 0.5 * (overlap + overlap.conj().T)


def _compute_overlap_matrix_periodic_cart_lattice_debug(
    aos_cart_data: AOs_cart_gto_data,
    *,
    k_point: tuple[float, float, float] | None,
    term_tolerance: float,
) -> np.ndarray:
    r"""Readable AO-pair reference for the periodic analytic lattice overlap.

    Runs the complete AO-pair loop independently of production
    shell grouping and Hermitian-half evaluation. The closed-form 1D terms
    and pair-specific image selector are unchanged mathematical primitives.

    The lattice cutoff uses ``rho = alpha*beta/(alpha+beta)`` and total
    angular degree, rather than reusing the faster-decaying AO-value image
    cutoff.  Each retained term is still an ordinary closed-form whole-space
    Cartesian Gaussian overlap.
    """
    if term_tolerance <= 0.0 or term_tolerance >= 1.0:
        raise ValueError("periodic-pGTO overlap term_tolerance must satisfy 0 < tol < 1.")
    if aos_cart_data._n_image != 1:
        raise ValueError("periodic Cartesian lattice overlap requires a molecular-rank base leaf.")

    dtype = np.complex128 if k_point is not None else np.float64
    num_ao = int(aos_cart_data.num_ao)
    overlap = np.zeros((num_ao, num_ao), dtype=dtype)
    cell = np.asarray(aos_cart_data.structure_data.cell, dtype=np.float64)
    centers = np.asarray(aos_cart_data._atomic_center_carts_np, dtype=np.float64)
    orbital_indices = np.asarray(aos_cart_data.orbital_indices, dtype=np.int64)
    primitive_indices = [np.nonzero(orbital_indices == ao)[0] for ao in range(num_ao)]
    exponents = np.asarray(aos_cart_data.exponents, dtype=np.float64)
    coefficients = np.asarray(aos_cart_data.coefficients, dtype=np.float64)
    angular_primitive = np.asarray(aos_cart_data._angular_momentums_prim_np, dtype=np.int32)
    normalization = np.sqrt(
        (2.0 * exponents / np.pi) ** 1.5
        * (8.0 * exponents) ** angular_primitive
        * np.asarray(aos_cart_data._normalization_factorial_ratio_prim_jnp, dtype=np.float64)
    )
    nx = np.asarray(aos_cart_data.polynominal_order_x, dtype=np.int32)
    ny = np.asarray(aos_cart_data.polynominal_order_y, dtype=np.int32)
    nz = np.asarray(aos_cart_data.polynominal_order_z, dtype=np.int32)
    k_frac = None if k_point is None else np.asarray(k_point, dtype=np.float64)
    image_cache: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]] = {}
    cell_offsets = _lattice_cell_offsets(cell, centers)

    for mu in range(num_ao):
        center_mu = centers[mu]
        for nu in range(num_ao):
            center_nu = centers[nu]
            value = np.asarray(0.0, dtype=dtype)
            for primitive_mu in primitive_indices[mu]:
                alpha = float(exponents[primitive_mu])
                for primitive_nu in primitive_indices[nu]:
                    beta = float(exponents[primitive_nu])
                    rho = alpha * beta / (alpha + beta)
                    total_angular = int(angular_primitive[primitive_mu] + angular_primitive[primitive_nu])
                    cache_key = (rho, total_angular)
                    cached = image_cache.get(cache_key)
                    if cached is None:
                        indices = _primitive_image_list_indices(cell, rho, total_angular, term_tolerance)
                        translations = np.asarray(indices, dtype=np.float64) @ cell
                        image_cache[cache_key] = (indices, translations)
                    else:
                        indices, translations = cached
                    pair_offset = cell_offsets[mu] - cell_offsets[nu]
                    if np.any(pair_offset != 0):
                        # Centres in different cells: the cell-cell set covers the
                        # WRAPPED separation; shift it to the actual one.
                        indices = indices + pair_offset[None, :]
                        translations = np.asarray(indices, dtype=np.float64) @ cell
                    translated_nu = center_nu[None, :] + translations
                    rab2 = np.sum((center_mu[None, :] - translated_nu) ** 2, axis=1)
                    primitive_values = np.exp(-rho * rab2)
                    primitive_values *= _compute_overlap_1d_cart_vectorized(
                        alpha, beta, center_mu[0], translated_nu[:, 0], int(nx[mu]), int(nx[nu])
                    )
                    primitive_values *= _compute_overlap_1d_cart_vectorized(
                        alpha, beta, center_mu[1], translated_nu[:, 1], int(ny[mu]), int(ny[nu])
                    )
                    primitive_values *= _compute_overlap_1d_cart_vectorized(
                        alpha, beta, center_mu[2], translated_nu[:, 2], int(nz[mu]), int(nz[nu])
                    )
                    if k_frac is not None:
                        primitive_values = primitive_values * np.exp(2j * np.pi * (indices @ k_frac))
                    prefactor = (
                        coefficients[primitive_mu]
                        * coefficients[primitive_nu]
                        * normalization[primitive_mu]
                        * normalization[primitive_nu]
                    )
                    value = value + prefactor * np.sum(primitive_values)
            overlap[mu, nu] = value
    return 0.5 * (overlap + overlap.conj().T)


_PERIODIC_PGTO_LATTICE_OVERLAP_CACHE: dict[tuple[int, float], tuple[weakref.ReferenceType, np.ndarray]] = {}


def _compute_overlap_matrix_periodic_pgto_lattice(
    aos_data: AOs_pGTO_base_data,
    *,
    term_tolerance: float | None = None,
) -> np.ndarray:
    """Return the analytic periodic overlap with an overlap-owned cutoff."""
    if term_tolerance is None:
        # The overlap is constructed and consumed in the fp64 ``det_eval``
        # zone, outside the hot AO-value loop. Its lattice truncation must
        # therefore retain det-evaluation precision even when the forward AO
        # kernel runs in mixed precision; an ao_eval-sized cutoff can make an
        # otherwise valid Gram matrix spuriously indefinite.
        term_tolerance = get_pgto_lattice_overlap_term_tolerance(get_dtype_np("det_eval"))
    term_tolerance = float(term_tolerance)
    cache_key = (id(aos_data), term_tolerance)
    cached = _PERIODIC_PGTO_LATTICE_OVERLAP_CACHE.get(cache_key)
    if cached is not None and cached[0]() is aos_data:
        return np.array(cached[1], copy=True)
    molecular = _detile_ragged_to_molecular(aos_data)
    k_point = aos_data.k_point if is_complex_aos(aos_data) else None
    if isinstance(molecular, AOs_sphe_gto_data):
        molecular_cart, transform = _aos_sphe_to_cart(molecular)
        cart_overlap = _compute_overlap_matrix_periodic_cart_lattice(
            molecular_cart,
            k_point=k_point,
            term_tolerance=term_tolerance,
        )
        overlap = transform @ cart_overlap @ transform.T
        overlap = 0.5 * (overlap + overlap.conj().T)
    elif isinstance(molecular, AOs_cart_gto_data):
        overlap = _compute_overlap_matrix_periodic_cart_lattice(
            molecular,
            k_point=k_point,
            term_tolerance=term_tolerance,
        )
    else:
        raise TypeError(f"Unsupported periodic pGTO leaf: {type(aos_data).__name__}.")
    overlap = np.asarray(overlap)
    overlap.setflags(write=False)

    def _drop(
        _reference: weakref.ReferenceType[AOs_pGTO_base_data],
        *,
        key: tuple[int, float] = cache_key,
    ) -> None:
        _PERIODIC_PGTO_LATTICE_OVERLAP_CACHE.pop(key, None)

    _PERIODIC_PGTO_LATTICE_OVERLAP_CACHE[cache_key] = (weakref.ref(aos_data, _drop), overlap)
    return np.array(overlap, copy=True)


def _composite_overlap_default_grid(aos_data: AOs_composite_data) -> tuple[int, int, int]:
    """Per-axis cell-integration grid that resolves every periodic child.

    The composite cell-domain overlap (:py:func:`_compute_overlap_matrix_composite`)
    is a midpoint quadrature, so the per-axis point count must out-resolve the
    finest oscillation any child carries:

    * plane-wave children: a ``cos(G.r)`` / ``sin(G.r)`` basis function with
      reciprocal vector ``G`` completes ``m_i = (a_i . G) / 2pi`` integer
      periods along fractional axis ``i``; the product of two such functions
      reaches ``2 |m_i|`` periods, so ``> 4 |m_i|`` midpoints are used.
    * B-spline children: ``2 * spline_shape`` (the midpoint rule the real
      B-spline leaf used before its overlap became exact).
    * pGTO children: smooth, no extra requirement beyond the shared floor.

    A floor of 16 per axis is applied so a trivial child never produces a
    degenerate grid.
    """
    cell = np.asarray(aos_data.structure_data.cell, dtype=np.float64)
    n_axis = np.full(3, 16, dtype=np.int64)
    for c in aos_data.children:
        if isinstance(c, AOs_plane_wave_data):
            g = np.asarray(c.g_vectors, dtype=np.float64)  # (n_pw, 3), rad/Bohr
            if g.size:
                # m[k, i] = (cell[i] . g[k]) / 2pi  -> integer periods of g[k]
                # along fractional axis i. cell rows are (vec_a, vec_b, vec_c).
                m = np.abs(g @ cell.T) / (2.0 * np.pi)  # (n_pw, 3)
                need = np.ceil(4.0 * m.max(axis=0)).astype(np.int64) + 2
                n_axis = np.maximum(n_axis, need)
        elif isinstance(c, AOs_bspline_data):
            need = np.array([2 * int(s) for s in c.spline_shape], dtype=np.int64)
            n_axis = np.maximum(n_axis, need)
    return (int(n_axis[0]), int(n_axis[1]), int(n_axis[2]))


_COMPOSITE_OVERLAP_VALS_BUDGET_BYTES = 256 * 1024 * 1024


@jax.jit
def _compute_composite_overlap_AOs(aos_data: AOs_base_data, r_carts: jax.Array) -> jax.Array:
    """Evaluate overlap samples with live parameters and stable tiny PW batches.

    Eight phase rows retain bulk CPU GEMM arithmetic for small mixed chunks.
    This is a reduction-layout choice, not a quadrature or tolerance change.
    Public AO calls keep their zero-padding default. Only real PW
    phase workspace is padded; returned samples and pGTO image slabs keep
    the actual requested point count.
    """
    if get_dtype_np("ao_eval") != np.float32 or jax.default_backend() != "cpu":
        return compute_AOs(aos_data, r_carts)
    if isinstance(aos_data, AOs_composite_data):
        return jnp.concatenate([_compute_composite_overlap_AOs(child, r_carts) for child in aos_data.children], axis=0)
    if isinstance(aos_data, AOs_plane_wave_complex_data):
        return aos_data._value_in_zone(r_carts, "ao_eval", phase_batch_floor=8)
    return compute_AOs(aos_data, r_carts)


def _compute_overlap_matrix_composite_debug(
    aos_data: AOs_composite_data,
    num_grid_points: tuple[int, int, int] | None = None,
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    r"""Numerical cell-quadrature reference for the periodic composite overlap.

    Independent reference for :py:func:`_compute_overlap_matrix_composite`
    (which assembles the exact matrix from child blocks). The midpoint rule does not resolve hard
    pGTO exponents unless ``num_grid_points`` is chosen fine enough.

    A composite over a shared periodic cell (e.g. ``pGTO + plane-wave``,
    ``pGTO + B-spline``) has a well-defined Gram matrix

    .. math::
       S_{\mu\nu} = \int_\Omega \chi_\mu(r)^*\,\chi_\nu(r)\,dr ,

    where ``chi`` ranges over the concatenated AO list of all children. A
    homogeneous complex-plane-wave composite at a shared twist is assembled
    analytically. With no single closed form across mixed bases, other
    composites are evaluated by a uniform fractional midpoint rule over one cell: fractional midpoints
    ``((i + 0.5)/N_a)`` mapped to Cartesian by the cell matrix,
    ``volume_element = V_cell / prod(N)``, and
    ``S = volume_element * (vals.conj() @ vals.T)`` with ``vals`` from the
    JIT-compiled :py:func:`compute_AOs` over the whole composite. The
    grid is processed in ``N_g``-axis chunks of at most
    ``_COMPOSITE_OVERLAP_VALS_BUDGET_BYTES`` so the
    ``(num_ao, chunk)`` real or complex slab is bounded; on systems with
    ``num_ao`` in the thousands and a default grid in the 10^5-point range
    a single-pass evaluation would hit several GB per MPI rank and would be
    the dominant per-rank allocation. Returned at molecular rank
    ``(num_ao, num_ao)`` (the pGTO children already collapse their image
    axis inside ``compute_AOs``).
    """
    cell = np.asarray(aos_data.structure_data.cell, dtype=np.float64)
    cell_volume = float(np.abs(np.linalg.det(cell)))

    # A concatenation of complex plane-wave children at one shared twist has
    # an exact analytic Gram matrix.  In particular, cross-child entries are
    # zero unless the children contain the same reciprocal vector.  Do not
    # route this case through the fp32 AO-value quadrature in mixed mode: that
    # turns the mathematically exact sin(pi*n) cancellation into O(1e-5)
    # off-diagonal noise before the result is stored in the fp64 det_eval zone.
    children = aos_data.children
    if children and all(isinstance(child, AOs_plane_wave_complex_data) for child in children):
        k_ref = np.asarray(children[0].k_frac_np, dtype=np.float64)
        if all(np.array_equal(np.asarray(child.k_frac_np, dtype=np.float64), k_ref) for child in children[1:]):
            g_vectors = np.concatenate([np.asarray(child.g_vectors, dtype=np.float64) for child in children], axis=0)
            same_g = np.all(np.abs(g_vectors[:, None, :] - g_vectors[None, :, :]) < 1.0e-10, axis=-1)
            return cell_volume * same_g.astype(np.complex128)

    real_dtype_np = get_dtype_np("ao_eval")
    dtype_np = np.result_type(real_dtype_np, np.complex64) if is_complex_aos(aos_data) else np.dtype(real_dtype_np)

    # Preserve ao_eval samples; overlap contraction follows existing det_eval
    # ownership declared by compute_overlap_matrix and the analytic GTO path.
    zone_repair = np.dtype(dtype_np) == np.dtype(np.complex64) and jax.default_backend() == "cpu"
    accumulation_dtype = get_complex_dtype_np("det_eval") if zone_repair else dtype_np
    evaluate_AOs = _compute_composite_overlap_AOs if zone_repair else compute_AOs

    grid = _composite_overlap_default_grid(aos_data) if num_grid_points is None else tuple(int(n) for n in num_grid_points)
    n0, n1, n2 = grid

    s0 = (np.arange(n0, dtype=np.float64) + 0.5) / n0
    s1 = (np.arange(n1, dtype=np.float64) + 0.5) / n1
    s2 = (np.arange(n2, dtype=np.float64) + 0.5) / n2
    fa, fb, fc = np.meshgrid(s0, s1, s2, indexing="ij")
    r_frac = np.stack([fa.ravel(), fb.ravel(), fc.ravel()], axis=-1)
    r_carts = r_frac @ cell  # (N_g, 3) Cartesian
    volume_element = cell_volume / float(n0 * n1 * n2)

    num_ao = int(aos_data.num_ao)
    n_g = int(r_carts.shape[0])
    # Budget against the largest pre-fold child rank, not the molecular
    # composite rank: pGTO children evaluate at the image-tiled kernel
    # rank ``n_image * num_ao_mol`` before the image fold reduces them
    # back to ``num_ao_mol``, and that pre-fold ``(rank_kernel, chunk)``
    # slab dominates JAX-device memory inside ``compute_AOs``. On a 128-atom
    # system the pGTO child has ``_num_ao_kernel = 17280`` (27 images x 640 mol
    # AOs), an order of magnitude above the composite ``num_ao = 2743``.
    worst_rank = num_ao
    for child in children:
        worst_rank = max(worst_rank, int(getattr(child, "_num_ao_kernel", getattr(child, "num_ao", 0))))
    bytes_per_grid_col = worst_rank * np.dtype(dtype_np).itemsize
    if zone_repair:
        # Promoted host samples and their conjugated copy are additional to
        # the pre-fold child AO slab. Tiny PW phase padding is fixed O(7*M_pw)
        # real workspace, not extra AO columns or image-expanded samples.
        bytes_per_grid_col += 2 * num_ao * np.dtype(accumulation_dtype).itemsize
    chunk = max(1, _COMPOSITE_OVERLAP_VALS_BUDGET_BYTES // max(1, bytes_per_grid_col))
    if chunk >= n_g:
        chunk = n_g

    overlap_matrix = np.zeros((num_ao, num_ao), dtype=accumulation_dtype)
    for start in range(0, n_g, chunk):
        end = min(start + chunk, n_g)
        vals_chunk = np.asarray(
            evaluate_AOs(aos_data, jnp.asarray(r_carts[start:end], dtype=jnp.float64)),
            dtype=dtype_np,
        )  # (num_ao, end-start)
        vals_work = np.asarray(vals_chunk, dtype=accumulation_dtype)
        overlap_matrix += vals_work.conj() @ vals_work.T
        del vals_work
        del vals_chunk
    overlap_matrix *= volume_element
    overlap_matrix = 0.5 * (overlap_matrix + overlap_matrix.conj().T)
    return overlap_matrix.astype(dtype_np)


def _compute_overlap_matrix_composite(
    aos_data: AOs_composite_data,
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    r"""Exact cell-domain overlap of a periodic AO composite, assembled by child blocks.

    .. math::
       S_{\mu\nu} = \int_\Omega \overline{\chi_\mu(r)}\,\chi_\nu(r)\,dr

    over the concatenated AO list of the children (bra conjugated):

    * diagonal block ``(i, i)``: the child's own overlap
      (``children[i]._compute_overlap_matrix()``: analytic lattice pGTO overlap,
      analytic plane-wave overlap, exact B-spline Gram);
    * cross block ``(i, j)``, ``i < j``: :py:func:`_composite_cross_overlap_block`
      (closed forms and an exact reciprocal-space sum, no real-space quadrature);
      block ``(j, i)`` is its conjugate transpose.

    Every block is an exact integral of the same Bloch functions, so ``S`` is
    their Gram matrix (Hermitian positive semidefinite up to round-off and the
    stated spectral truncation). Returned in the ``det_eval`` real / complex
    dtype. The numerical cell quadrature of the whole matrix is kept as the
    independent reference :py:func:`_compute_overlap_matrix_composite_debug`.
    """
    children = aos_data.children
    dtype = np.dtype(get_complex_dtype_np("det_eval") if is_complex_aos(aos_data) else get_dtype_np("det_eval"))
    offsets = np.concatenate(([0], np.cumsum([int(child.num_ao) for child in children]))).astype(int)
    overlap = np.zeros((offsets[-1], offsets[-1]), dtype=dtype)
    for i, child in enumerate(children):
        rows = slice(offsets[i], offsets[i + 1])
        overlap[rows, rows] = np.asarray(child._compute_overlap_matrix())
        for j in range(i + 1, len(children)):
            cols = slice(offsets[j], offsets[j + 1])
            block = _composite_cross_overlap_block(child, children[j])
            if not np.iscomplexobj(overlap) and np.iscomplexobj(block):
                block = np.real(block)
            overlap[rows, cols] = block
            overlap[cols, rows] = block.conj().T
    return overlap


def _composite_child_family(child: AOs_base_data) -> str:
    """Basis family of a periodic composite child: ``pgto``, ``plane-wave`` or ``bspline``."""
    if isinstance(child, AOs_pGTO_base_data):
        return "pgto"
    if isinstance(child, (AOs_plane_wave_data, AOs_plane_wave_complex_data)):
        return "plane-wave"
    if isinstance(child, (AOs_bspline_data, AOs_bspline_complex_data)):
        return "bspline"
    raise NotImplementedError(f"composite overlap: no cross-block formula for a {type(child).__name__} child.")


def _require_shared_twist(child_a: AOs_base_data, child_b: AOs_base_data) -> None:
    """Both children must be Bloch functions of the same crystal momentum (modulo the reciprocal lattice)."""
    difference = np.asarray(getattr(child_a, "k_frac_np", np.zeros(3)), dtype=np.float64) - np.asarray(
        getattr(child_b, "k_frac_np", np.zeros(3)), dtype=np.float64
    )
    if float(np.max(np.abs(difference - np.rint(difference)))) > ATOL_TWIST_CHECKPOINT_KPOINT_MATCH:
        raise ValueError(
            "composite overlap: children carry different twists "
            f"({type(child_a).__name__} k={tuple(getattr(child_a, 'k_frac_np', np.zeros(3)))}, "
            f"{type(child_b).__name__} k={tuple(getattr(child_b, 'k_frac_np', np.zeros(3)))}); "
            "the cell overlap of Bloch functions at different k is not a Gram block."
        )


def _composite_cross_overlap_block(
    child_a: AOs_base_data,
    child_b: AOs_base_data,
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    r"""Exact cross block ``S_ab = int_cell conj(chi_a) chi_b dr`` between two composite children.

    Both children must share the twist ``k`` (modulo the reciprocal lattice).
    Every pair of periodic families has an exact formula:

    * pGTO x plane wave: :py:func:`_pgto_fourier_transform` at ``q = G + k``
      (``cos`` / ``sin`` take the real / imaginary part at Gamma);
    * pGTO x B-spline: :py:func:`_pgto_bspline_cross_overlap` (reciprocal-space
      sum with the exact spline Fourier coefficients);
    * plane wave x B-spline: the exact spline Fourier coefficient
      (:py:func:`_bspline_fourier_coefficients`);
    * plane wave x plane wave: ``V delta_{G,G'}`` (complex) or the cos/sin
      closed form (real);
    * B-spline x B-spline: the exact Gram of the concatenated control points
      (same knots, order and twist; otherwise ``NotImplementedError``);
    * pGTO x pGTO: the analytic one-lattice overlap of the merged molecular leaf.

    A real child paired with a complex one (or a real and a complex plane-wave
    leaf) raises ``NotImplementedError``. Returns
    ``(child_a.num_ao, child_b.num_ao)`` in complex128 or float64.
    """
    family_a, family_b = _composite_child_family(child_a), _composite_child_family(child_b)
    order = ("pgto", "plane-wave", "bspline")
    if order.index(family_a) > order.index(family_b):
        return np.conj(_composite_cross_overlap_block(child_b, child_a)).T
    _require_shared_twist(child_a, child_b)
    cell = np.asarray(child_a.structure_data.cell, dtype=np.float64)
    cell_volume = float(np.abs(np.linalg.det(cell)))
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    if family_a == "pgto" and family_b == "plane-wave":
        if isinstance(child_b, AOs_plane_wave_complex_data):
            k_cart = np.asarray(child_b.k_frac_np, dtype=np.float64) @ reciprocal
            return _pgto_fourier_transform(child_a, np.asarray(child_b.g_vectors, dtype=np.float64) + k_cart[None, :])
        fourier = _pgto_fourier_transform(child_a, np.asarray(child_b.g_vectors, dtype=np.float64))
        if is_complex_aos(child_a):
            raise NotImplementedError("composite overlap: a complex pGTO child with a real cos/sin plane-wave child.")
        is_sin = np.asarray(child_b.phase_is_sin, dtype=np.float64) > 0.5
        return np.where(is_sin[None, :], fourier.imag, fourier.real)
    if family_a == "pgto" and family_b == "bspline":
        return _pgto_bspline_cross_overlap(child_a, child_b)
    if family_a == "pgto" and family_b == "pgto":
        return _pgto_pgto_cross_overlap(child_a, child_b)
    if family_a == "plane-wave" and family_b == "bspline":
        indices = np.rint(np.asarray(child_a.g_vectors, dtype=np.float64) @ cell.T / (2.0 * np.pi)).astype(np.int64)
        coefficients = cell_volume * _bspline_fourier_coefficients(child_b, indices)  # (n_pw, n_band)
        if isinstance(child_a, AOs_plane_wave_complex_data):
            return coefficients
        if is_complex_aos(child_b):
            raise NotImplementedError("composite overlap: a real cos/sin plane-wave child with a complex B-spline child.")
        is_sin = np.asarray(child_a.phase_is_sin, dtype=np.float64) > 0.5
        return np.where(is_sin[:, None], -coefficients.imag, coefficients.real)
    if family_a == "plane-wave" and family_b == "plane-wave":
        g_a = np.asarray(child_a.g_vectors, dtype=np.float64)
        g_b = np.asarray(child_b.g_vectors, dtype=np.float64)
        index_a = np.rint(g_a @ cell.T / (2.0 * np.pi)).astype(np.int64)
        index_b = np.rint(g_b @ cell.T / (2.0 * np.pi)).astype(np.int64)
        same = np.all(index_a[:, None, :] == index_b[None, :, :], axis=-1)
        if isinstance(child_a, AOs_plane_wave_complex_data) and isinstance(child_b, AOs_plane_wave_complex_data):
            return cell_volume * same.astype(np.complex128)
        if isinstance(child_a, AOs_plane_wave_complex_data) or isinstance(child_b, AOs_plane_wave_complex_data):
            raise NotImplementedError("composite overlap: a real cos/sin plane-wave child with a complex plane-wave child.")
        opposite = np.all(index_a[:, None, :] == -index_b[None, :, :], axis=-1)
        sin_a = (np.asarray(child_a.phase_is_sin, dtype=np.float64) > 0.5)[:, None]
        sin_b = (np.asarray(child_b.phase_is_sin, dtype=np.float64) > 0.5)[None, :]
        same_f, opposite_f = same.astype(np.float64), opposite.astype(np.float64)
        cos_cos = ((~sin_a) & (~sin_b)).astype(np.float64)
        sin_sin = (sin_a & sin_b).astype(np.float64)
        return 0.5 * cell_volume * (cos_cos * (same_f + opposite_f) + sin_sin * (same_f - opposite_f))
    # bspline x bspline
    if (
        tuple(child_a.spline_shape) != tuple(child_b.spline_shape)
        or int(child_a.spline_order) != int(child_b.spline_order)
        or is_complex_aos(child_a) != is_complex_aos(child_b)
    ):
        raise NotImplementedError(
            "composite overlap: two B-spline children need the same knots, order and real/complex kind "
            "for the exact cross block."
        )
    # The stored table can be shared with a lazily conjugated time-reversal sibling.
    stacked = np.concatenate(
        [
            np.asarray(getattr(child_a, "_spline_coefficients_np", child_a.spline_coefficients), dtype=np.complex128),
            np.asarray(getattr(child_b, "_spline_coefficients_np", child_b.spline_coefficients), dtype=np.complex128),
        ],
        axis=0,
    )
    gram = _compute_overlap_matrix_bspline_exact(
        cell=cell,
        spline_coefficients=stacked,
        spline_shape=tuple(int(n) for n in child_a.spline_shape),
        spline_order=int(child_a.spline_order),
    )
    block = gram[: int(child_a.num_ao), int(child_a.num_ao) :]
    return block if is_complex_aos(child_a) else np.real(block)


def _pgto_molecular_cartesian(
    pgto: AOs_pGTO_base_data,
) -> tuple[AOs_cart_gto_data, npt.NDArray[np.float64]]:
    """Molecular Cartesian base of a pGTO leaf and ``T`` with ``chi = T @ chi_cart`` (identity for Cartesian)."""
    molecular = _detile_ragged_to_molecular(pgto)
    molecular_cart, transform = _aos_sphe_to_cart(molecular)
    return molecular_cart, np.asarray(transform, dtype=np.float64)


def _cartesian_primitive_table(
    molecular_cart: AOs_cart_gto_data,
) -> tuple[
    npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.float64]
]:
    """Per primitive: owner AO, exponent, normalised weight, Cartesian powers ``(n, 3)``, centre ``(n, 3)``."""
    owner = np.asarray(molecular_cart.orbital_indices, dtype=np.int64)
    exponents = np.asarray(molecular_cart.exponents, dtype=np.float64)
    angular = np.asarray(molecular_cart._angular_momentums_prim_np, dtype=np.float64)
    normalization = np.sqrt(
        (2.0 * exponents / np.pi) ** 1.5
        * (8.0 * exponents) ** angular
        * np.asarray(molecular_cart._normalization_factorial_ratio_prim_jnp, dtype=np.float64)
    )
    weights = np.asarray(molecular_cart.coefficients, dtype=np.float64) * normalization
    powers = np.stack(
        [
            np.asarray(molecular_cart.polynominal_order_x, dtype=np.int64)[owner],
            np.asarray(molecular_cart.polynominal_order_y, dtype=np.int64)[owner],
            np.asarray(molecular_cart.polynominal_order_z, dtype=np.int64)[owner],
        ],
        axis=1,
    )
    centers = np.asarray(molecular_cart._atomic_center_carts_np, dtype=np.float64)[owner]
    return owner, exponents, weights, powers, centers


def _gaussian_moment_fourier_factor(power: int, exponent: float, q: npt.NDArray[np.float64]) -> npt.NDArray[np.complex128]:
    r"""``(int x^n e^{-a x^2 + i q x} dx) / (sqrt(pi / a) e^{-q^2 / 4a}) = i^n (2 sqrt a)^{-n} H_n(q / 2 sqrt a)``."""
    scale = 2.0 * np.sqrt(exponent)
    return (1j**power) * scale ** (-power) * scipy.special.eval_hermite(power, q / scale)


def _pgto_fourier_transform(pgto: AOs_pGTO_base_data, q_carts: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    r"""``F_a(q) = int_{R^3} g_a(r) e^{i q . r} dr`` for the molecular AOs of a pGTO leaf.

    For a Bloch pGTO ``chi_a = sum_T e^{i k.T} g_a(r - T)`` (``g_a`` the real
    contracted Gaussian at its stored centre) and a partner ``phi`` with
    ``phi(r + T) = e^{i k.T} phi(r)``, unfolding the lattice sum gives
    ``int_cell conj(chi_a) phi = int_{R^3} g_a phi``; for ``phi = e^{i q.r}``,
    ``q = G + k``, that is ``F_a(q)``. Per Cartesian primitive
    ``F = w e^{i q.R} prod_x i^{n_x} (2 sqrt a)^{-n_x} H_{n_x}(q_x / 2 sqrt a)
    (pi / a)^{3/2} e^{-|q|^2 / 4a}``; spherical leaves use the leaf's
    spherical-to-Cartesian transform. Returns ``(num_ao, n_q)`` complex128.
    """
    q_carts = np.asarray(q_carts, dtype=np.float64).reshape(-1, 3)
    molecular_cart, transform = _pgto_molecular_cartesian(pgto)
    owner, exponents, weights, powers, centers = _cartesian_primitive_table(molecular_cart)
    q_norm2 = np.sum(q_carts * q_carts, axis=1)
    cartesian = np.zeros((int(molecular_cart.num_ao), q_carts.shape[0]), dtype=np.complex128)
    for p in range(exponents.size):
        alpha = float(exponents[p])
        value = (weights[p] * (np.pi / alpha) ** 1.5) * np.exp(-q_norm2 / (4.0 * alpha) + 1j * (q_carts @ centers[p]))
        for axis in range(3):
            if powers[p, axis]:
                value = value * _gaussian_moment_fourier_factor(int(powers[p, axis]), alpha, q_carts[:, axis])
        cartesian[owner[p]] += value
    return transform @ cartesian


def _bspline_fourier_coefficients(
    spline: AOs_bspline_data | AOs_bspline_complex_data,
    indices: npt.ArrayLike,
) -> npt.NDArray[np.complex128]:
    r"""Exact Fourier coefficients ``u_hat_b(m) = (1/V) int_cell u_b(r) e^{-i G_m . r} dr``.

    For ``u_b = sum_j c_{b,j} prod_x M(N_x s_x - j_x)`` (``M`` the centred cardinal
    B-spline of degree ``n``, periodic in ``j``),
    ``u_hat_b(m) = C_b(m mod N) prod_x sinc(m_x / N_x)^{n+1} / prod(N)`` with
    ``C_b = fftn(c_b)``. ``indices`` are integer reciprocal-lattice coordinates
    ``(n_m, 3)``; returns ``(n_m, n_band)``. The logical control points
    include any lazy time-reversal conjugation of the stored table.
    """
    indices = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    shape = np.asarray(spline.spline_shape, dtype=np.int64)
    order = int(spline.spline_order)
    spectrum = np.fft.fftn(
        np.asarray(getattr(spline, "_spline_coefficients_np", spline.spline_coefficients), dtype=np.complex128).reshape(
            (int(spline.num_ao), *shape)
        ),
        axes=(1, 2, 3),
    )
    envelope = np.prod(np.sinc(indices / shape[None, :]) ** (order + 1), axis=1)
    wrapped = np.mod(indices, shape[None, :])
    gathered = spectrum[:, wrapped[:, 0], wrapped[:, 1], wrapped[:, 2]]  # (n_band, n_m)
    return (gathered * envelope[None, :]).T / float(np.prod(shape))


def _pgto_bspline_cross_overlap(
    pgto: AOs_pGTO_base_data,
    spline: AOs_bspline_data | AOs_bspline_complex_data,
) -> npt.NDArray[np.complex128]:
    r"""Exact ``S_ab = int_cell conj(chi_a) e^{i k.r} u_b dr`` by a reciprocal-space sum.

    With ``e^{i k.r} u_b(r) = sum_m u_hat_b(m) e^{i (G_m + k).r}``
    (:py:func:`_bspline_fourier_coefficients`) and the unfolded Bloch sum
    (:py:func:`_pgto_fourier_transform`),
    ``S_ab = sum_m u_hat_b(m) F_a(G_m + k)``. The Cartesian primitives sharing
    an exponent ``alpha`` (any centre) sum over the reciprocal points with

    .. math::
       \prod_x \mathrm{sinc}(m_x / N_x)^{n+1}\,
       e^{-|q|^2 / 4\alpha}\,(1 + |q| / 2\sqrt\alpha)^{l}
       \ \ge\ \epsilon ,

    ``q = G_m + k``, ``l`` the group's largest angular momentum and
    ``epsilon`` = :py:data:`jqmc._setting.COMPOSITE_CROSS_OVERLAP_SPECTRAL_TERM_TOLERANCE`;
    a dropped term is at most ``epsilon`` times the ``m = 0`` scale of its
    group (both factors are at most 1 there). The candidate ball is bounded by
    the Gaussian factor alone. The group shares the criterion, the envelope and
    the gathered spline spectrum; the centre phase ``e^{i q.R}`` is factorised
    per reciprocal axis. Returns ``(pgto.num_ao, spline.num_ao)`` complex128.
    """
    cell = np.asarray(spline.structure_data.cell, dtype=np.float64)
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T  # rows b_i
    k_cart = np.asarray(getattr(spline, "k_frac_np", np.zeros(3)), dtype=np.float64) @ reciprocal
    shape = np.asarray(spline.spline_shape, dtype=np.int64)
    order = int(spline.spline_order)
    num_band = int(spline.num_ao)
    # Match AO evaluation: transform the logical table, including lazy conjugation.
    spectrum = np.fft.fftn(
        np.asarray(getattr(spline, "_spline_coefficients_np", spline.spline_coefficients), dtype=np.complex128).reshape(
            (num_band, *shape)
        ),
        axes=(1, 2, 3),
    ) / float(np.prod(shape))
    log_tolerance = float(np.log(COMPOSITE_CROSS_OVERLAP_SPECTRAL_TERM_TOLERANCE))
    axis_lengths = np.linalg.norm(cell, axis=1)

    molecular_cart, transform = _pgto_molecular_cartesian(pgto)
    owner, exponents, weights, powers, centers = _cartesian_primitive_table(molecular_cart)
    cartesian = np.zeros((int(molecular_cart.num_ao), num_band), dtype=np.complex128)
    groups: dict[float, list[int]] = {}
    for p in range(exponents.size):
        groups.setdefault(float(exponents[p]), []).append(p)

    for alpha, members in groups.items():
        members = np.asarray(members, dtype=np.int64)
        angular = int(powers[members].sum(axis=1).max())
        # Gaussian-only radius: e^{-u^2} (1 + u)^l < eps beyond u_max, u = |q| / (2 sqrt alpha).
        u_max = float(np.sqrt(-log_tolerance))
        for _ in range(16):
            u_max = float(np.sqrt(-log_tolerance + angular * np.log1p(u_max)))
        radius = 2.0 * np.sqrt(alpha) * u_max
        bound = np.ceil((radius + float(np.linalg.norm(k_cart))) * axis_lengths / (2.0 * np.pi)).astype(np.int64)
        axes = [np.arange(-b, b + 1, dtype=np.int64) for b in bound]
        log_envelope = [(order + 1) * np.log(np.abs(np.sinc(a / n)) + np.finfo(np.float64).tiny) for a, n in zip(axes, shape)]
        m12 = np.stack(np.meshgrid(axes[1], axes[2], indexing="ij"), axis=-1).reshape(-1, 2)
        log_env12 = (log_envelope[1][:, None] + log_envelope[2][None, :]).ravel()
        q12 = m12 @ reciprocal[1:] + k_cart[None, :]
        q12_norm2 = np.sum(q12 * q12, axis=1)
        q12_dot0 = q12 @ reciprocal[0]
        b0_norm2 = float(reciprocal[0] @ reciprocal[0])
        radius2 = radius * radius
        prefactor = weights[members] * (np.pi / alpha) ** 1.5  # (n_member,)
        # e^{i q.R} is separable in m: e^{i k.R} e^{i m0 b0.R} e^{i (m1 b1 + m2 b2).R}.
        theta = centers[members] @ reciprocal.T  # (n_member, 3), b_i . R
        phase12 = np.exp(1j * (centers[members] @ k_cart)[:, None] + 1j * (theta[:, 1:] @ m12.T))  # (n_member, n12)
        for i0, m0 in enumerate(axes[0]):
            # Ball |q| <= radius first (cheap, from the precomputed quadratic form),
            # then the exact per-term criterion on the ball points only.
            q_norm2_all = q12_norm2 + (2.0 * m0) * q12_dot0 + (m0 * m0) * b0_norm2
            ball = np.flatnonzero(q_norm2_all <= radius2)
            if ball.size == 0:
                continue
            q_norm2 = q_norm2_all[ball]
            u = np.sqrt(q_norm2) / (2.0 * np.sqrt(alpha))
            log_term = log_envelope[0][i0] + log_env12[ball] - u * u + angular * np.log1p(u)
            keep_local = log_term >= log_tolerance
            if not np.any(keep_local):
                continue
            keep = ball[keep_local]
            base = np.exp(log_envelope[0][i0] + log_env12[keep] - q_norm2[keep_local] / (4.0 * alpha))  # real
            wrapped0 = int(np.mod(m0, shape[0]))
            wrapped12 = np.mod(m12[keep], shape[1:][None, :])
            gathered = spectrum[:, wrapped0, wrapped12[:, 0], wrapped12[:, 1]]  # (n_band, n_kept)
            values = phase12[:, keep] * (prefactor * np.exp(1j * m0 * theta[:, 0]))[:, None]
            values *= base[None, :]
            if angular:
                q_kept = q12[keep] + m0 * reciprocal[0][None, :]
                for row, p in enumerate(members):
                    for axis in range(3):
                        if powers[p, axis]:
                            values[row] *= _gaussian_moment_fourier_factor(int(powers[p, axis]), alpha, q_kept[:, axis])
            contribution = values @ gathered.T  # (n_member, n_band)
            np.add.at(cartesian, owner[members], contribution)
    return transform @ cartesian


def _pgto_pgto_cross_overlap(pgto_a: AOs_pGTO_base_data, pgto_b: AOs_pGTO_base_data) -> npt.NDArray[np.complex128]:
    """Analytic one-lattice cross overlap of two pGTO children (merged molecular Cartesian leaf)."""
    cart_a, transform_a = _pgto_molecular_cartesian(pgto_a)
    cart_b, transform_b = _pgto_molecular_cartesian(pgto_b)
    n_a = int(cart_a.num_ao)
    prim_a = int(cart_a.num_ao_prim)

    def _field(name: str) -> tuple:
        return tuple(getattr(cart_a, name)) + tuple(getattr(cart_b, name))

    merged = AOs_cart_gto_data._from_full_basis(
        structure_data=cart_a.structure_data,
        nucleus_index=_field("nucleus_index"),
        num_ao=n_a + int(cart_b.num_ao),
        num_ao_prim=prim_a + int(cart_b.num_ao_prim),
        orbital_indices=tuple(int(x) for x in cart_a.orbital_indices) + tuple(int(x) + n_a for x in cart_b.orbital_indices),
        exponents=np.concatenate([np.asarray(cart_a.exponents), np.asarray(cart_b.exponents)]),
        coefficients=np.concatenate([np.asarray(cart_a.coefficients), np.asarray(cart_b.coefficients)]),
        angular_momentums=_field("angular_momentums"),
        polynominal_order_x=_field("polynominal_order_x"),
        polynominal_order_y=_field("polynominal_order_y"),
        polynominal_order_z=_field("polynominal_order_z"),
    )
    k_point = pgto_a.k_point if is_complex_aos(pgto_a) else None
    overlap = _compute_overlap_matrix_periodic_cart_lattice(
        merged,
        k_point=k_point,
        term_tolerance=get_pgto_lattice_overlap_term_tolerance(get_dtype_np("det_eval")),
    )
    return transform_a @ overlap[:n_a, n_a:] @ transform_b.T


def _build_prim_groups_by_K(
    orbital_indices: Sequence[int],
    num_ao: int,
) -> tuple[tuple[tuple[int, npt.NDArray[np.int32]], ...], npt.NDArray[np.int32], bool]:
    """Group AOs by contraction depth K (number of primitives per AO).

    This is a pure-Python (numpy) preprocessor invoked at trace time
    by the cached property :attr:`AOs_cart_gto_data._prim_groups_by_K` /
    :attr:`AOs_sphe_gto_data._prim_groups_by_K`. It produces the static
    bucket descriptors consumed by :func:`_reduce_primitives_to_aos`,
    which replaces the AO-side ``segment_sum`` (XLA scatter-add
    fallback) with one fixed-shape ``reduce_sum`` per unique K plus a
    final inverse-permutation gather (no scatter at all).

    Args:
        orbital_indices: Sequence of length ``num_ao_prim``; entry
            ``j`` is the parent AO index of primitive ``j``.
        num_ao: Total number of contracted AOs.

    Returns:
        Tuple of ``(groups, inv_perm_np, is_identity_perm)`` where:
            - ``groups`` is a tuple of ``(K, prim_idx_np)`` for each
              unique contraction depth K, with ``prim_idx_np`` of
              shape ``(n_ao_K, K)``.
            - ``inv_perm_np`` (int32, shape ``(num_ao,)``) maps the
              concatenated bucket-order layout back to the original
              AO ordering.
            - ``is_identity_perm`` (bool): True iff ``inv_perm_np``
              equals ``arange(num_ao)`` (lets callers skip the final
              gather).
    """
    prim_per_ao: dict[int, list[int]] = {}
    for prim_idx, ao_idx in enumerate(orbital_indices):
        prim_per_ao.setdefault(int(ao_idx), []).append(int(prim_idx))

    by_K: dict[int, list[tuple[int, list[int]]]] = {}
    for ao_idx, prims in prim_per_ao.items():
        by_K.setdefault(len(prims), []).append((ao_idx, prims))

    # Bucket order: by ascending K, and inside each bucket by ascending
    # AO index. ``concat_order`` lists AO indices in this layout.
    groups = []
    concat_order: list[int] = []
    # ``itemgetter`` and a list comprehension keep this per-AO bookkeeping in C
    # (no Python frame per AO). The property calling this runs on every trace
    # of the AO kernel, and a per-element ``lambda`` / generator frame here
    # dominated the per-call cost of any Python-level tracer.
    for K in sorted(by_K.keys()):
        items = sorted(by_K[K], key=itemgetter(0))
        prim_idx_np = np.asarray([p for _, p in items], dtype=np.int32)
        groups.append((K, prim_idx_np))
        concat_order.extend([a for a, _ in items])

    # Inverse permutation: out[ao_idx] = concatenated[inv_perm[ao_idx]]
    inv_perm_np = np.empty(num_ao, dtype=np.int32)
    for pos, ao_idx in enumerate(concat_order):
        inv_perm_np[ao_idx] = pos
    is_identity_perm = bool(np.array_equal(inv_perm_np, np.arange(num_ao, dtype=np.int32)))
    return tuple(groups), inv_perm_np, is_identity_perm


def _sphe_primitive_prefactors(
    exponents: jax.Array, coefficients: jax.Array, angular_momenta: jax.Array, dtype: jnp.dtype
) -> jax.Array:
    """Per-primitive ``N_n(l, Z) * N_lm(l) * c`` at storage precision, rounded once to ``dtype``.

    ``N_n = sqrt(2^(2l+3) (l+1)! (2Z)^(l+3/2) / ((2l+2)! sqrt(pi)))`` and
    ``N_lm = sqrt((2l+1) / (4 pi))``. The factorials (``exp(gammaln)``) and the
    power evaluated in float32 lose up to 23 float32 ulp at l = 6. These
    constants do not depend on electron coordinates, so they are formed in the
    precision of ``exponents`` (fp64 storage) and cast to the AO zone once.

    Args:
        exponents: Primitive exponents ``Z`` in storage precision.
        coefficients: Primitive contraction coefficients ``c``.
        angular_momenta: Primitive angular momenta ``l`` (integers).
        dtype: Output dtype (the ``ao_eval`` zone dtype of the caller).

    Returns:
        The prefactors, shape ``exponents.shape``, dtype ``dtype``.
    """
    exponents = jnp.asarray(exponents)
    ell = jnp.asarray(angular_momenta).astype(exponents.dtype)
    factorial_l = jnp.exp(jscipy.special.gammaln(ell + 2.0))
    factorial_2l = jnp.exp(jscipy.special.gammaln(2.0 * ell + 3.0))
    normalization = jnp.sqrt(
        (2.0 ** (2 * ell + 3) * factorial_l * (2 * exponents) ** (ell + 1.5)) / (factorial_2l * jnp.sqrt(jnp.pi))
    )
    angular_normalization = jnp.sqrt((2 * ell + 1) / (4 * jnp.pi))
    return ((normalization * angular_normalization) * jnp.asarray(coefficients)).astype(dtype)


@jax.jit
def _compute_AOs_sphe_molecular_guarded(aos: AOs_sphe_gto_data, points: jax.Array) -> jax.Array:
    """Evaluate molecular mixed AOs with guarded geometry and tied radial reuse.

    Shell identity uses static parameter indices, never equality of current
    exponent/coefficient values. All grouping is trace-time NumPy work.
    Contraction and returned values retain the AO evaluation zone.
    """
    dtype = get_dtype_jnp("ao_eval")
    differences = points[None, :, :] - aos._uq_center_carts_jnp[:, None, :]
    shell_map = np.asarray(aos.ao_prim_to_shell_prim)
    orbital_indices = np.asarray(aos.orbital_indices)
    representatives = []
    ao_shell = []
    identities = {}
    for orbital, (centre, ell) in enumerate(zip(aos.nucleus_index, aos.angular_momentums)):
        primitives = np.flatnonzero(orbital_indices == orbital)
        identity = (int(centre), int(ell), tuple(shell_map[primitives].tolist()))
        if identity not in identities:
            identities[identity] = len(representatives)
            representatives.append(primitives)
        ao_shell.append(identities[identity])
    buckets = {}
    for shell, primitives in enumerate(representatives):
        buckets.setdefault(len(primitives), []).append((shell, primitives))
    primitive_indices = np.concatenate([primitives for entries in buckets.values() for _, primitives in entries])
    shell_indices = jnp.asarray(shell_map[primitive_indices])
    primitive_indices_jax = jnp.asarray(primitive_indices)
    exponent_storage = jnp.asarray(aos.radial.exponents)[shell_indices]
    coefficients = jnp.asarray(aos.radial.coefficients)[shell_indices]
    centres = aos._uq_center_index_prim_jnp[primitive_indices_jax]
    ell = aos._angular_momentums_prim_jnp[primitive_indices_jax]
    prefactor = _sphe_primitive_prefactors(exponent_storage, coefficients, ell, dtype)
    radius_squared = jnp.sum(differences**2, axis=-1)
    exponent = (-exponent_storage[:, None] * radius_squared[centres]).astype(dtype)
    normalized = prefactor[:, None] * jnp.exp(exponent)
    pieces = []
    ordering = []
    offset = 0
    for depth, entries in buckets.items():
        length = depth * len(entries)
        pieces.append(jnp.sum(normalized[offset : offset + length].reshape((len(entries), depth, points.shape[0])), axis=1))
        ordering.extend(shell for shell, _ in entries)
        offset += length
    shell_values = jnp.concatenate(pieces, axis=0)
    radial_values = shell_values[jnp.asarray(np.argsort(ordering)[np.asarray(ao_shell)])]
    count = (int(aos._angular_momentums_prim_np.max()) + 1) ** 2
    _, harmonics = _compute_S_l_m(differences, n_lm=count, cast_to_ao_eval=True)
    stride = harmonics.shape[0]  # per-centre stride = rows actually tabulated
    flat_harmonics = harmonics.reshape((harmonics.shape[0] * harmonics.shape[1], harmonics.shape[2]), order="F")
    angular_l = jnp.asarray(aos.angular_momentums, dtype=jnp.int32)
    angular_m = jnp.asarray(aos.magnetic_quantum_numbers, dtype=jnp.int32)
    indices = aos._uq_center_index_ao_jnp * stride + angular_l**2 + angular_l + angular_m
    values = radial_values * flat_harmonics[indices]
    return jnp.where((radial_values == 0) & ~jnp.isfinite(values), jnp.zeros_like(values), values)


def _reduce_primitives_to_aos(values: jax.Array, aos_data: AOs_GTO_base_data) -> jax.Array:
    """Sum primitives into contracted AOs without scatter / while loop.

    Functionally equivalent to::

        jax.ops.segment_sum(values,
                            aos_data._orbital_indices_jnp,
                            num_segments=aos_data._num_ao_kernel)

    but emits, for each unique contraction depth K, a single dense
    reduction over ``(n_ao_K, K, ...)``; the per-K outputs are
    concatenated in bucket order and finally permuted into the
    original AO ordering by a single ``gather``. There is no
    ``scatter`` and no XLA ``while`` loop: unlike a ``segment_sum``, the
    kernel chain consists only of fusable gather/reduce/gather ops.

    Periodic CPU float32 values use a transposed all-ones dot instead of
    ``reduce_sum`` to avoid shape-dependent contraction kernels in carried
    candidate caches. The dtype is unchanged; this does not require bitwise
    agreement between arbitrary batch widths.

    For pgto inputs the kernel emits at kernel rank
    (``_num_ao_kernel = n_image * num_ao_molecular``) and the caller
    (the pgto leaf's ``compute_AOs*`` method) reshapes back to molecular
    rank.

    Args:
        values: Primitive values, shape ``(num_ao_prim, ...)``.
        aos_data: ``AOs_cart_gto_data`` or ``AOs_sphe_gto_data`` providing
            ``_num_ao_kernel`` and ``_prim_groups_by_K``.

    Returns:
        Reduced values, shape ``(_num_ao_kernel, ...)`` and same dtype
        as ``values``.
    """
    groups, inv_perm_np, is_identity_perm = aos_data._prim_groups_by_K

    n_image = int(getattr(aos_data, "_n_image", 1))
    n_prim_total = int(values.shape[0])
    n_prim_mol = n_prim_total // n_image if n_image > 0 else n_prim_total
    rest = tuple(values.shape[1:])

    def _bucket_sum(prim_idx_np: np.ndarray) -> jax.Array:
        # If the bucket's primitives form the SAME contiguous molecular-prim
        # slice in every image block (the layout guaranteed by the build-time
        # (K, AO)-sort reorder), reduce it with
        # a strided slice + reshape-sum -- no prim-axis gather. Detection runs
        # on the static numpy index, so the fast path enters the HLO directly
        # for periodic leaves and falls back to the gather for any other order.
        # The image-tiled strided-slice fast path is valid ONLY for uniform
        # tiling, where ``n_prim_total == n_image * n_prim_mol`` exactly. Ragged
        # leaves keep PER-SHELL image counts, so ``n_prim_total`` is
        # generally NOT a multiple of ``n_image`` (``n_prim_mol`` floor-divides);
        # the ``values.reshape((n_image, n_prim_mol) ...)`` below would then be
        # malformed (e.g. (22, ...) -> (14, 1, ...) raises). Skip straight to the
        # gather fallback, which is always correct, whenever the tiling is not
        # exactly uniform.
        slice_info: tuple[int, int, int] | None = None
        if n_image * n_prim_mol == n_prim_total:
            n_ao_K, K = prim_idx_np.shape
            effective_n_image = max(n_image, 1)
            if n_ao_K % effective_n_image == 0:
                n_ao_K_mol = n_ao_K // effective_n_image
                image_primitive_indices = prim_idx_np.reshape(effective_n_image, n_ao_K_mol, K)
                start = int(image_primitive_indices[0].min())
                first_image_expected = start + np.arange(
                    n_ao_K_mol * K,
                    dtype=prim_idx_np.dtype,
                ).reshape(n_ao_K_mol, K)
                if all(
                    np.array_equal(
                        image_primitive_indices[image],
                        first_image_expected + image * n_prim_mol,
                    )
                    for image in range(effective_n_image)
                ):
                    slice_info = (start, n_ao_K_mol, K)
        if slice_info is not None:
            start, n_ao_K_mol, K = slice_info
            blk = values.reshape((n_image, n_prim_mol) + rest)[:, start : start + n_ao_K_mol * K]
            blk = blk.reshape((n_image, n_ao_K_mol, K) + rest)
            if values.dtype == jnp.float32 and aos_data.structure_data.pbc_flag and jax.default_backend() == "cpu":
                # The transposed primitive contraction retains fusion at large
                # batch widths; the opposite dot layout materializes each slice.
                leading = jnp.moveaxis(blk, 2, 0).reshape(K, -1)
                summed = leading.T @ jnp.ones((K,), dtype=values.dtype)
                return summed.reshape((n_image * n_ao_K_mol,) + rest)
            return jnp.sum(blk, axis=2).reshape((n_image * n_ao_K_mol,) + rest)
        tile = values[jnp.asarray(prim_idx_np)]  # (n_ao_K, K, ...)
        if values.dtype == jnp.float32 and aos_data.structure_data.pbc_flag and jax.default_backend() == "cpu":
            depth = tile.shape[1]
            leading = jnp.moveaxis(tile, 1, 0).reshape(depth, -1)
            summed = leading.T @ jnp.ones((depth,), dtype=values.dtype)
            return summed.reshape((tile.shape[0],) + rest)
        return jnp.sum(tile, axis=1)

    # Single-bucket fast path: one reduce, no concat.
    if len(groups) == 1:
        summed = _bucket_sum(groups[0][1])
        if is_identity_perm:
            return summed
        return summed[jnp.asarray(inv_perm_np)]

    # General path: per-K reduce -> concat in bucket order -> final
    # inverse-permutation gather.
    pieces = [_bucket_sum(prim_idx_np) for _K, prim_idx_np in groups]
    concatenated = jnp.concatenate(pieces, axis=0)
    if is_identity_perm:
        return concatenated
    return concatenated[jnp.asarray(inv_perm_np)]


def _cart_max_polynomial_order(aos_data: AOs_cart_gto_data) -> int:
    """Return L_MAX = max(nx, ny, nz) over all AOs as a Python int.

    ``polynominal_order_{x,y,z}`` are ``pytree_node=False`` (static), so this
    is JIT-trace-time available and feeds the static-unrolled integer power
    helper (:func:`_int_pow_unrolled_cart`) below.
    """
    return int(
        max(
            max(aos_data.polynominal_order_x, default=0),
            max(aos_data.polynominal_order_y, default=0),
            max(aos_data.polynominal_order_z, default=0),
        )
    )


def _cart_max_angular_momentum(aos_data: AOs_cart_gto_data) -> int:
    """Return the largest total degree l = nx + ny + nz over all AOs as a Python int.

    Static unroll bound for the normalization power ``(8 Z)**l`` in
    :func:`_int_pow_unrolled_cart`. It is NOT interchangeable with
    :func:`_cart_max_polynomial_order` (the largest single coordinate power):
    ``max(nx, ny, nz) <= l``, with equality over the AO set only when every
    Cartesian shell of the top degree is complete (it then contains
    ``(l, 0, 0)``). An incomplete shell such as a lone d_xy has l = 2 but a
    coordinate-power bound of 1 (issue #110).
    """
    return int(max(aos_data.angular_momentums, default=0))


def _bucket_pow(base: jax.Array, exponent: int) -> jax.Array:
    """Compute ``base ** exponent`` with ``exponent`` a Python int.

    Used by :py:meth:`AOs_cart_gto_data._compute_AOs` after the AO
    permutation has been applied, so the exponent is loop-invariant
    over the bucket and fully unrolled by the python interpreter at
    trace time. Unlike :func:`_int_pow_unrolled_cart` there is NO
    ``jnp.where`` chain, so LLVM's loop vectoriser keeps the inner
    SIMD form through the multiply chain.

    Small-exponent shortcuts (0..6) cover the standard cc-pVNZ basis
    range; the generic fallback handles arbitrary integer exponents.
    """
    if exponent == 0:
        return jnp.ones_like(base)
    if exponent == 1:
        return base
    if exponent == 2:
        return base * base
    if exponent == 3:
        return base * base * base
    if exponent == 4:
        b2 = base * base
        return b2 * b2
    if exponent == 5:
        b2 = base * base
        return b2 * b2 * base
    if exponent == 6:
        b2 = base * base
        b4 = b2 * b2
        return b4 * b2
    out = base
    for _ in range(exponent - 1):
        out = out * base
    return out


def _int_pow_unrolled_cart(base: jax.Array, exp_arr: jax.Array, L_MAX: int) -> jax.Array:
    """Compute ``base ** exp_arr`` (broadcast over base) via a static O(L_MAX) unroll.

    Args:
        base: shape ``(num_ao_prim, ...)`` floating values. May be 1D
            (e.g. normalization factor ``(8 Z)**l``) or higher-rank
            (e.g. ``(num_ao_prim, N_e)`` polynomial part ``(x+eps)**nx``).
        exp_arr: shape ``(num_ao_prim,)`` integer exponents in ``[0, L_MAX]``.
            Trailing axes are added to broadcast against ``base``.
        L_MAX: Python int upper bound on ``exp_arr`` (basis-dependent and
            static). Exponents above it are NOT handled (they silently
            evaluate to ``base ** 1``), so the bound must match the exponent
            array: :func:`_cart_max_polynomial_order` for the polynomial
            powers ``nx``, ``ny``, ``nz`` and :func:`_cart_max_angular_momentum`
            for the normalization power ``l = nx + ny + nz``, which can exceed
            the coordinate-power bound for incomplete Cartesian shells.

    Returns:
        Same shape as ``base`` with ``out[i, ...] = base[i, ...] ** exp_arr[i]``.

    Rationale:
        The naive ``base ** exp[:, None]`` lowers to an XLA repeated-squaring
        ``while_loop`` (because ``exp`` is a traced integer array), which
        emits 4 small kernel launches per iteration and dominates host-side
        launch overhead for kinetic-energy evaluation. Unrolling collapses
        the power into a single fused elementwise kernel.

        Numerically equivalent to
        ``jnp.where(exp == 0, 1.0, base ** exp_b)`` for ``exp in [0, L_MAX]``
        (bitwise-identical: same left-to-right multiplication tree).
    """
    # Broadcast exp_arr against base: prepend trailing singleton axes.
    e = exp_arr.reshape(exp_arr.shape + (1,) * (base.ndim - exp_arr.ndim))
    one = jnp.ones_like(base)
    if L_MAX <= 0:
        return one
    out = jnp.where(e == 0, one, base)  # base^0 -> 1, else base^1
    p = base
    for k in range(2, L_MAX + 1):
        p = p * base  # base^k
        out = jnp.where(e == k, p, out)
    return out


# The XLA real-solid-harmonic tables (``_compute_S_l_m`` and
# ``_compute_S_l_m_and_grad_lap``) are hand-written for l = 0..6 only, i.e.
# (6 + 1)**2 = 49 (l, m) entries. A larger ``n_lm`` cannot be served: the
# truncation ``[:n_lm]`` would silently return 49 rows while the callers'
# per-nucleus stride assumed ``n_lm``, so l = 7 AOs and every AO on later
# nuclei were gathered from the wrong rows. Reject it loudly instead.
_SPHE_XLA_L_MAX = 6
_SPHE_XLA_N_LM_MAX = (_SPHE_XLA_L_MAX + 1) ** 2


def _check_sphe_xla_n_lm(n_lm: int) -> None:
    """Raise if ``n_lm`` exceeds the l <= 6 XLA solid-harmonic tables."""
    if n_lm > _SPHE_XLA_N_LM_MAX:
        raise ValueError(
            f"spherical GTO angular momentum l_max = {round(n_lm**0.5) - 1} is not supported by the XLA "
            f"solid-harmonic kernels, which are tabulated for l <= {_SPHE_XLA_L_MAX} only "
            f"(n_lm = {n_lm} > {_SPHE_XLA_N_LM_MAX})."
        )


@partial(jit, static_argnames=("n_lm", "cast_to_ao_eval"))
def _compute_S_l_m(
    r_R_diffs: jnpt.ArrayLike,
    n_lm: int = 49,
    cast_to_ao_eval: bool = False,
) -> jax.Array:
    r"""Solid harmonics part of a primitve AO.

    ``n_lm`` truncates the returned solid-harmonic stack to the first
    ``n_lm`` entries (global index ``l**2 + (m + l)``), i.e.
    ``(l_max + 1)**2`` for a basis with maximum angular momentum ``l_max``.
    Harmonics above the basis max-l are never gathered by the caller, so
    the unused named-scalar terms become dead code (XLA DCE drops them).
    Default ``49`` (l=0..6) preserves the untruncated behaviour.

    Radius-dependent terms are polynomials in x, y, z, including at the
    nucleus, so their autodiff gradient and Hessian have no norm singularity.
    The molecular mixed value path may cast completed terms to the AO zone
    before stacking. The optional cast defaults off for other callers.

    Compute the solid harmonics, i.e., r^l * spherical hamonics part (c.f., regular solid harmonics) of a given AO

    Args:
        r_R_diffs ( jnpt.ArrayLike): Cartesian coordinate of N electrons - Cartesian corrdinates of M nuclei. dim: (N,M,3)
        n_lm (int): number of leading (l, m) entries returned (at most 49).
        cast_to_ao_eval (bool): if True, cast each completed term to the ``ao_eval`` zone dtype
            before stacking.

    Returns:
        tuple[int, jax.Array]: ``(n_lm, S_l_m)``, where ``S_l_m`` of dim ``(n_lm,N,M)`` holds the
        spherical harmonics part * r^l (i.e., regular solid harmonics) for the first ``n_lm`` (l,m) pairs.

    Raises:
        ValueError: If ``n_lm > 49`` (the tables cover l <= 6 only).
    """
    _check_sphe_xla_n_lm(n_lm)
    x, y, z = r_R_diffs[..., 0], r_R_diffs[..., 1], r_R_diffs[..., 2]
    # Solid harmonics are Cartesian polynomials. Reconstructing r**2/r**4
    # through a regularized sqrt gives the wrong Hessian at r=0, and r**0.0
    # has a nonfinite JVP there. Use the polynomial invariants at every point.
    r2 = x**2 + y**2 + z**2

    def lnorm(l: int) -> jax.Array:
        return jnp.sqrt((4 * jnp.pi) / (2 * l + 1))

    """see https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics (l=0-4)"""
    """Useful tool to generate spherical harmonics generator [https://github.com/elerac/sh_table]"""
    # s orbital
    s_0 = lnorm(l=0) * 1.0 / 2.0 * jnp.sqrt(1.0 / jnp.pi) * jnp.ones_like(r2)  # (l, m) == (0, 0)
    # p orbitals
    p_m1 = lnorm(l=1) * jnp.sqrt(3.0 / (4 * jnp.pi)) * y  # (l, m) == (1, -1)
    p_0 = lnorm(l=1) * jnp.sqrt(3.0 / (4 * jnp.pi)) * z  # (l, m) == (1, 0)
    p_p1 = lnorm(l=1) * jnp.sqrt(3.0 / (4 * jnp.pi)) * x  # (l, m) == (1, 1)
    # d orbitals
    d_m2 = lnorm(l=2) * 1.0 / 2.0 * jnp.sqrt(15.0 / (jnp.pi)) * x * y  # (l, m) == (2, -2)
    d_m1 = lnorm(l=2) * 1.0 / 2.0 * jnp.sqrt(15.0 / (jnp.pi)) * y * z  # (l, m) == (2, -1)
    d_0 = lnorm(l=2) * 1.0 / 4.0 * jnp.sqrt(5.0 / (jnp.pi)) * (3 * z**2 - r2)  # (l, m) == (2, 0):
    d_p1 = lnorm(l=2) * 1.0 / 2.0 * jnp.sqrt(15.0 / (jnp.pi)) * x * z  # (l, m) == (2, 1)
    d_p2 = lnorm(l=2) * 1.0 / 4.0 * jnp.sqrt(15.0 / (jnp.pi)) * (x**2 - y**2)  # (l, m) == (2, 2)
    # f orbitals
    f_m3 = lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(35.0 / (2 * jnp.pi)) * y * (3 * x**2 - y**2)  # (l, m) == (3, -3)
    f_m2 = lnorm(l=3) * 1.0 / 2.0 * jnp.sqrt(105.0 / (jnp.pi)) * x * y * z  # (l, m) == (3, -2)
    f_m1 = lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(21.0 / (2 * jnp.pi)) * y * (5 * z**2 - r2)  # (l, m) == (3, -1)
    f_0 = lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(7.0 / (jnp.pi)) * (5 * z**3 - 3 * z * r2)  # (l, m) == (3, 0)
    f_p1 = lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(21.0 / (2 * jnp.pi)) * x * (5 * z**2 - r2)  # (l, m) == (3, 1)
    f_p2 = lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(105.0 / (jnp.pi)) * (x**2 - y**2) * z  # (l, m) == (3, 2)
    f_p3 = lnorm(l=3) * 1.0 / 4.0 * jnp.sqrt(35.0 / (2 * jnp.pi)) * x * (x**2 - 3 * y**2)  # (l, m) == (3, 3)
    # g orbitals
    g_m4 = lnorm(l=4) * 3.0 / 4.0 * jnp.sqrt(35.0 / (jnp.pi)) * x * y * (x**2 - y**2)  # (l, m) == (4, -4)
    g_m3 = lnorm(l=4) * 3.0 / 4.0 * jnp.sqrt(35.0 / (2 * jnp.pi)) * y * z * (3 * x**2 - y**2)  # (l, m) == (4, -3)
    g_m2 = lnorm(l=4) * 3.0 / 4.0 * jnp.sqrt(5.0 / (jnp.pi)) * x * y * (7 * z**2 - r2)  # (l, m) == (4, -2)
    g_m1 = lnorm(l=4) * 3.0 / 4.0 * jnp.sqrt(5.0 / (2 * jnp.pi)) * y * (7 * z**3 - 3 * z * r2)  # (l, m) == (4, -1)
    g_0 = lnorm(l=4) * 3.0 / 16.0 * jnp.sqrt(1.0 / (jnp.pi)) * (35 * z**4 - 30 * z**2 * r2 + 3 * r2**2)  # (l, m) == (4, 0)
    g_p1 = lnorm(l=4) * 3.0 / 4.0 * jnp.sqrt(5.0 / (2 * jnp.pi)) * x * (7 * z**3 - 3 * z * r2)  # (l, m) == (4, 1)
    g_p2 = lnorm(l=4) * 3.0 / 8.0 * jnp.sqrt(5.0 / (jnp.pi)) * (x**2 - y**2) * (7 * z**2 - r2)  # (l, m) == (4, 2)
    g_p3 = lnorm(l=4) * (3.0 / 4.0 * jnp.sqrt(35.0 / (2 * jnp.pi)) * x * z * (x**2 - 3 * y**2))  # (l, m) == (4, 3)
    g_p4 = (
        lnorm(l=4) * 3.0 / 16.0 * jnp.sqrt(35.0 / (jnp.pi)) * (x**2 * (x**2 - 3 * y**2) - y**2 * (3 * x**2 - y**2))
    )  # (l, m) == (4, 4)
    # h orbitals
    h_m5 = lnorm(5) * 3.0 / 16.0 * jnp.sqrt(77.0 / (2 * jnp.pi)) * (5 * x**4 * y - 10 * x**2 * y**3 + y**5)  # (l, m) == (5, -5)
    h_m4 = lnorm(5) * 3.0 / 16.0 * jnp.sqrt(385.0 / jnp.pi) * 4 * x * y * z * (x**2 - y**2)  # (l, m) == (5, -4)
    h_m3 = (
        lnorm(5) * 1.0 / 16.0 * jnp.sqrt(385.0 / (2 * jnp.pi)) * -1 * (y**3 - 3 * x**2 * y) * (9 * z**2 - (x**2 + y**2 + z**2))
    )  # (l, m) == (5, -3)
    h_m2 = (
        lnorm(5) * 1.0 / 8.0 * jnp.sqrt(1155 / jnp.pi) * 2 * x * y * (3 * z**3 - z * (x**2 + y**2 + z**2))
    )  # (l, m) == (5, -2)
    h_m1 = (
        lnorm(5)
        * 1.0
        / 16.0
        * jnp.sqrt(165 / jnp.pi)
        * y
        * (21 * z**4 - 14 * z**2 * (x**2 + y**2 + z**2) + (x**2 + y**2 + z**2) ** 2)
    )  # (l, m) == (5, -1)
    h_0 = (
        lnorm(5)
        * 1.0
        / 16.0
        * jnp.sqrt(11 / jnp.pi)
        * (63 * z**5 - 70 * z**3 * (x**2 + y**2 + z**2) + 15 * z * (x**2 + y**2 + z**2) ** 2)
    )  # (l, m) == (5, 0)
    h_p1 = (
        lnorm(5)
        * 1.0
        / 16.0
        * jnp.sqrt(165 / jnp.pi)
        * x
        * (21 * z**4 - 14 * z**2 * (x**2 + y**2 + z**2) + (x**2 + y**2 + z**2) ** 2)
    )  # (l, m) == (5, 1)
    h_p2 = (
        lnorm(5) * 1.0 / 8.0 * jnp.sqrt(1155 / jnp.pi) * (x**2 - y**2) * (3 * z**3 - z * (x**2 + y**2 + z**2))
    )  # (l, m) == (5, 2)
    h_p3 = (
        lnorm(5) * 1.0 / 16.0 * jnp.sqrt(385.0 / (2 * jnp.pi)) * (x**3 - 3 * x * y**2) * (9 * z**2 - (x**2 + y**2 + z**2))
    )  # (l, m) == (5, 3)
    h_p4 = (
        lnorm(5) * 3.0 / 16.0 * jnp.sqrt(385.0 / jnp.pi) * (x**2 * z * (x**2 - 3 * y**2) - y**2 * z * (3 * x**2 - y**2))
    )  # (l, m) == (5, 4)
    h_p5 = lnorm(5) * 3.0 / 16.0 * jnp.sqrt(77.0 / (2 * jnp.pi)) * (x**5 - 10 * x**3 * y**2 + 5 * x * y**4)  # (l, m) == (5, 5)
    # i orbitals
    i_m6 = (
        lnorm(6) * 1.0 / 64.0 * jnp.sqrt(6006.0 / jnp.pi) * (6 * x**5 * y - 20 * x**3 * y**3 + 6 * x * y**5)
    )  # (l, m) == (6, -6)
    i_m5 = lnorm(6) * 3.0 / 32.0 * jnp.sqrt(2002.0 / jnp.pi) * z * (5 * x**4 * y - 10 * x**2 * y**3 + y**5)  # (l, m) == (6, -5)
    i_m4 = (
        lnorm(6) * 3.0 / 32.0 * jnp.sqrt(91.0 / jnp.pi) * 4 * x * y * (11 * z**2 - (x**2 + y**2 + z**2)) * (x**2 - y**2)
    )  # (l, m) == (6, -4)
    i_m3 = (
        lnorm(6)
        * 1.0
        / 32.0
        * jnp.sqrt(2730.0 / jnp.pi)
        * -1
        * (11 * z**3 - 3 * z * (x**2 + y**2 + z**2))
        * (y**3 - 3 * x**2 * y)
    )  # (l, m) == (6, -3)
    i_m2 = (
        lnorm(6)
        * 1.0
        / 64.0
        * jnp.sqrt(2730.0 / jnp.pi)
        * 2
        * x
        * y
        * (33 * z**4 - 18 * z**2 * (x**2 + y**2 + z**2) + (x**2 + y**2 + z**2) ** 2)
    )  # (l, m) == (6, -2)
    i_m1 = (
        lnorm(6)
        * 1.0
        / 16.0
        * jnp.sqrt(273.0 / jnp.pi)
        * y
        * (33 * z**5 - 30 * z**3 * (x**2 + y**2 + z**2) + 5 * z * (x**2 + y**2 + z**2) ** 2)
    )  # (l, m) == (6, -1)
    i_0 = (
        lnorm(6)
        * 1.0
        / 32.0
        * jnp.sqrt(13.0 / jnp.pi)
        * (
            231 * z**6
            - 315 * z**4 * (x**2 + y**2 + z**2)
            + 105 * z**2 * (x**2 + y**2 + z**2) ** 2
            - 5 * (x**2 + y**2 + z**2) ** 3
        )
    )  # (l, m) == (6, 0)
    i_p1 = (
        lnorm(6)
        * 1.0
        / 16.0
        * jnp.sqrt(273.0 / jnp.pi)
        * x
        * (33 * z**5 - 30 * z**3 * (x**2 + y**2 + z**2) + 5 * z * (x**2 + y**2 + z**2) ** 2)
    )  # (l, m) == (6, 1)
    i_p2 = (
        lnorm(6)
        * 1.0
        / 64.0
        * jnp.sqrt(2730.0 / jnp.pi)
        * (x**2 - y**2)
        * (33 * z**4 - 18 * z**2 * (x**2 + y**2 + z**2) + (x**2 + y**2 + z**2) ** 2)
    )  # (l, m) == (6, 2)
    i_p3 = (
        lnorm(6) * 1.0 / 32.0 * jnp.sqrt(2730.0 / jnp.pi) * (11 * z**3 - 3 * z * (x**2 + y**2 + z**2)) * (x**3 - 3 * x * y**2)
    )  # (l, m) == (6, 3)
    i_p4 = (
        lnorm(6)
        * 3.0
        / 32.0
        * jnp.sqrt(91.0 / jnp.pi)
        * (11 * z**2 - (x**2 + y**2 + z**2))
        * (x**2 * (x**2 - 3 * y**2) + y**2 * (y**2 - 3 * x**2))
    )  # (l, m) == (6, 4)
    i_p5 = lnorm(6) * 3.0 / 32.0 * jnp.sqrt(2002.0 / jnp.pi) * z * (x**5 - 10 * x**3 * y**2 + 5 * x * y**4)  # (l, m) == (6, 5)
    i_p6 = (
        lnorm(6) * 1.0 / 64.0 * jnp.sqrt(6006.0 / jnp.pi) * (x**6 - 15 * x**4 * y**2 + 15 * x**2 * y**4 - y**6)
    )  # (l, m) == (6, 6)

    terms = [
        s_0,
        p_m1,
        p_0,
        p_p1,
        d_m2,
        d_m1,
        d_0,
        d_p1,
        d_p2,
        f_m3,
        f_m2,
        f_m1,
        f_0,
        f_p1,
        f_p2,
        f_p3,
        g_m4,
        g_m3,
        g_m2,
        g_m1,
        g_0,
        g_p1,
        g_p2,
        g_p3,
        g_p4,
        h_m5,
        h_m4,
        h_m3,
        h_m2,
        h_m1,
        h_0,
        h_p1,
        h_p2,
        h_p3,
        h_p4,
        h_p5,
        i_m6,
        i_m5,
        i_m4,
        i_m3,
        i_m2,
        i_m1,
        i_0,
        i_p1,
        i_p2,
        i_p3,
        i_p4,
        i_p5,
        i_p6,
    ][:n_lm]
    if cast_to_ao_eval:
        # Cast each completed polynomial before stacking, allowing CPU fusion
        # to avoid materializing an expanded float64 harmonic tensor.
        terms = [term.astype(get_dtype_jnp("ao_eval")) for term in terms]
    S_l_m_values = jnp.stack(terms, axis=0)
    # Per-nucleus stride of the caller's gather = rows actually tabulated.
    max_ml = S_l_m_values.shape[0]
    return max_ml, S_l_m_values


@partial(jit, static_argnames=("n_lm",))
def _compute_S_l_m_and_grad_lap(r_R_diffs_uq: jnp.ndarray, n_lm: int = 49) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Vectorized solid harmonics values, gradients, and Laplacians.

    ``n_lm`` truncates the solid-harmonic set to the first ``n_lm`` entries
    (global index ``l**2 + (m + l)``), i.e. ``(l_max + 1)**2`` for a basis
    whose maximum angular momentum is ``l_max``. The full table covers
    ``l = 0..6`` (``n_lm = 49``); a basis with ``l_max < 6`` references only
    the first ``(l_max+1)**2`` harmonics, so computing the rest is pure waste.
    Passing the basis-derived ``n_lm`` makes the output ``(n_lm, num_R, num_r)``
    and is bitwise-identical to ``full[:n_lm]``. Default ``49`` preserves the
    untruncated behaviour for callers that do not specialise.

    Pinned to the ``ao_grad_lap`` zone (fp64 in both full and mixed mode).
    Both grad and lap consumers (``_compute_AOs_grad_analytic_sphe`` and
    ``_compute_AOs_laplacian_analytic_sphe``) live in the same
    ``ao_grad_lap`` zone, so no further down-cast is required at the
    consumer site. Running the helper at fp64 is mandated by the
    catastrophic cancellation in the laplacian arithmetic
    (``4 Z^2 r^2 - 6 Z``); the solid-harmonics polynomial expansion is
    cheap (49 x num_R x num_e) compared to the contracted AO formulas,
    so the cost of fp64 here is small.

    Returns:
        tuple: (values, grads, laps) where values has shape (num_R, n_lm, num_r), grads has shape
        (num_R, n_lm, num_r, 3), and laps has shape (num_R, n_lm, num_r); n_lm = the truncated
        harmonic count (default 49). NOTE the leading axis is num_R (not n_lm) -- the caller flattens
        the (num_R, n_lm) axes in C-order so the prim-rank gather index nucleus*n_lm + global_l_m
        addresses the right row without a transpose.

    Raises:
        ValueError: If ``n_lm > 49`` (the tables cover l <= 6 only).
    """
    _check_sphe_xla_n_lm(n_lm)
    dtype_jnp = get_dtype_jnp("ao_grad_lap")
    S_L_M_COEFFS = (
        jnp.array([1.0], dtype=dtype_jnp),
        jnp.array([1.0], dtype=dtype_jnp),
        jnp.array([1.0], dtype=dtype_jnp),
        jnp.array([1.0], dtype=dtype_jnp),
        jnp.array([1.7320508075688774], dtype=dtype_jnp),
        jnp.array([1.7320508075688774], dtype=dtype_jnp),
        jnp.array([1.0, -0.5, -0.5], dtype=dtype_jnp),
        jnp.array([1.7320508075688774], dtype=dtype_jnp),
        jnp.array([-0.8660254037844387, 0.8660254037844387], dtype=dtype_jnp),
        jnp.array([-0.7905694150420949, 2.3717082451262845], dtype=dtype_jnp),
        jnp.array([3.8729833462074166], dtype=dtype_jnp),
        jnp.array([2.4494897427831783, -0.6123724356957946, -0.6123724356957946], dtype=dtype_jnp),
        jnp.array([1.0, -1.5, -1.5], dtype=dtype_jnp),
        jnp.array([2.4494897427831783, -0.6123724356957946, -0.6123724356957946], dtype=dtype_jnp),
        jnp.array([-1.9364916731037083, 1.9364916731037083], dtype=dtype_jnp),
        jnp.array([-2.3717082451262845, 0.7905694150420949], dtype=dtype_jnp),
        jnp.array([-2.958039891549808, 2.958039891549808], dtype=dtype_jnp),
        jnp.array([-2.091650066335189, 6.274950199005566], dtype=dtype_jnp),
        jnp.array([6.708203932499371, -1.1180339887498951, -1.1180339887498951], dtype=dtype_jnp),
        jnp.array([3.1622776601683795, -2.3717082451262845, -2.3717082451262845], dtype=dtype_jnp),
        jnp.array([1.0, -3.0, 0.375, -3.0, 0.75, 0.375], dtype=dtype_jnp),
        jnp.array([3.1622776601683795, -2.3717082451262845, -2.3717082451262845], dtype=dtype_jnp),
        jnp.array([-3.3541019662496856, 0.5590169943749476, 3.3541019662496856, -0.5590169943749476], dtype=dtype_jnp),
        jnp.array([-6.274950199005566, 2.091650066335189], dtype=dtype_jnp),
        jnp.array([0.739509972887452, -4.437059837324712, 0.739509972887452], dtype=dtype_jnp),
        jnp.array([0.7015607600201141, -7.015607600201141, 3.5078038001005707], dtype=dtype_jnp),
        jnp.array([-8.874119674649426, 8.874119674649426], dtype=dtype_jnp),
        jnp.array(
            [-4.183300132670378, 0.5229125165837972, 12.549900398011133, -1.0458250331675945, -1.5687375497513916],
            dtype=dtype_jnp,
        ),
        jnp.array([10.2469507659596, -5.1234753829798, -5.1234753829798], dtype=dtype_jnp),
        jnp.array(
            [
                3.872983346207417,
                -5.809475019311126,
                0.4841229182759271,
                -5.809475019311126,
                0.9682458365518543,
                0.4841229182759271,
            ],
            dtype=dtype_jnp,
        ),
        jnp.array([1.0, -5.0, 1.875, -5.0, 3.75, 1.875], dtype=dtype_jnp),
        jnp.array(
            [
                3.872983346207417,
                -5.809475019311126,
                0.4841229182759271,
                -5.809475019311126,
                0.9682458365518543,
                0.4841229182759271,
            ],
            dtype=dtype_jnp,
        ),
        jnp.array([-5.1234753829798, 2.5617376914899, 5.1234753829798, -2.5617376914899], dtype=dtype_jnp),
        jnp.array(
            [-12.549900398011133, 1.5687375497513916, 4.183300132670378, 1.0458250331675945, -0.5229125165837972],
            dtype=dtype_jnp,
        ),
        jnp.array([2.2185299186623566, -13.311179511974139, 2.2185299186623566], dtype=dtype_jnp),
        jnp.array([3.5078038001005707, -7.015607600201141, 0.7015607600201141], dtype=dtype_jnp),
        jnp.array([4.030159736288377, -13.433865787627923, 4.030159736288377], dtype=dtype_jnp),
        jnp.array([2.3268138086232857, -23.268138086232856, 11.634069043116428], dtype=dtype_jnp),
        jnp.array([-19.843134832984433, 1.9843134832984433, 19.843134832984433, -1.9843134832984433], dtype=dtype_jnp),
        jnp.array(
            [-7.245688373094719, 2.7171331399105196, 21.737065119284157, -5.434266279821039, -8.15139941973156],
            dtype=dtype_jnp,
        ),
        jnp.array(
            [
                14.491376746189438,
                -14.491376746189438,
                0.9057110466368399,
                -14.491376746189438,
                1.8114220932736798,
                0.9057110466368399,
            ],
            dtype=dtype_jnp,
        ),
        jnp.array(
            [4.58257569495584, -11.4564392373896, 2.8641098093474, -11.4564392373896, 5.7282196186948, 2.8641098093474],
            dtype=dtype_jnp,
        ),
        jnp.array([1.0, -7.5, 5.625, -0.3125, -7.5, 11.25, -0.9375, 5.625, -0.9375, -0.3125], dtype=dtype_jnp),
        jnp.array(
            [4.58257569495584, -11.4564392373896, 2.8641098093474, -11.4564392373896, 5.7282196186948, 2.8641098093474],
            dtype=dtype_jnp,
        ),
        jnp.array(
            [
                -7.245688373094719,
                7.245688373094719,
                -0.45285552331841994,
                7.245688373094719,
                -0.45285552331841994,
                -7.245688373094719,
                0.45285552331841994,
                0.45285552331841994,
            ],
            dtype=dtype_jnp,
        ),
        jnp.array(
            [-21.737065119284157, 8.15139941973156, 7.245688373094719, 5.434266279821039, -2.7171331399105196],
            dtype=dtype_jnp,
        ),
        jnp.array(
            [
                4.960783708246108,
                -0.4960783708246108,
                -29.76470224947665,
                2.480391854123054,
                4.960783708246108,
                2.480391854123054,
                -0.4960783708246108,
            ],
            dtype=dtype_jnp,
        ),
        jnp.array([11.634069043116428, -23.268138086232856, 2.3268138086232857], dtype=dtype_jnp),
        jnp.array([-0.6716932893813962, 10.075399340720942, -10.075399340720942, 0.6716932893813962], dtype=dtype_jnp),
    )

    S_L_M_EXPS = (
        jnp.array([[0, 0, 0]], dtype=jnp.int32),
        jnp.array([[0, 1, 0]], dtype=jnp.int32),
        jnp.array([[0, 0, 1]], dtype=jnp.int32),
        jnp.array([[1, 0, 0]], dtype=jnp.int32),
        jnp.array([[1, 1, 0]], dtype=jnp.int32),
        jnp.array([[0, 1, 1]], dtype=jnp.int32),
        jnp.array([[0, 0, 2], [0, 2, 0], [2, 0, 0]], dtype=jnp.int32),
        jnp.array([[1, 0, 1]], dtype=jnp.int32),
        jnp.array([[0, 2, 0], [2, 0, 0]], dtype=jnp.int32),
        jnp.array([[0, 3, 0], [2, 1, 0]], dtype=jnp.int32),
        jnp.array([[1, 1, 1]], dtype=jnp.int32),
        jnp.array([[0, 1, 2], [0, 3, 0], [2, 1, 0]], dtype=jnp.int32),
        jnp.array([[0, 0, 3], [0, 2, 1], [2, 0, 1]], dtype=jnp.int32),
        jnp.array([[1, 0, 2], [1, 2, 0], [3, 0, 0]], dtype=jnp.int32),
        jnp.array([[0, 2, 1], [2, 0, 1]], dtype=jnp.int32),
        jnp.array([[1, 2, 0], [3, 0, 0]], dtype=jnp.int32),
        jnp.array([[1, 3, 0], [3, 1, 0]], dtype=jnp.int32),
        jnp.array([[0, 3, 1], [2, 1, 1]], dtype=jnp.int32),
        jnp.array([[1, 1, 2], [1, 3, 0], [3, 1, 0]], dtype=jnp.int32),
        jnp.array([[0, 1, 3], [0, 3, 1], [2, 1, 1]], dtype=jnp.int32),
        jnp.array([[0, 0, 4], [0, 2, 2], [0, 4, 0], [2, 0, 2], [2, 2, 0], [4, 0, 0]], dtype=jnp.int32),
        jnp.array([[1, 0, 3], [1, 2, 1], [3, 0, 1]], dtype=jnp.int32),
        jnp.array([[0, 2, 2], [0, 4, 0], [2, 0, 2], [4, 0, 0]], dtype=jnp.int32),
        jnp.array([[1, 2, 1], [3, 0, 1]], dtype=jnp.int32),
        jnp.array([[0, 4, 0], [2, 2, 0], [4, 0, 0]], dtype=jnp.int32),
        jnp.array([[0, 5, 0], [2, 3, 0], [4, 1, 0]], dtype=jnp.int32),
        jnp.array([[1, 3, 1], [3, 1, 1]], dtype=jnp.int32),
        jnp.array([[0, 3, 2], [0, 5, 0], [2, 1, 2], [2, 3, 0], [4, 1, 0]], dtype=jnp.int32),
        jnp.array([[1, 1, 3], [1, 3, 1], [3, 1, 1]], dtype=jnp.int32),
        jnp.array([[0, 1, 4], [0, 3, 2], [0, 5, 0], [2, 1, 2], [2, 3, 0], [4, 1, 0]], dtype=jnp.int32),
        jnp.array([[0, 0, 5], [0, 2, 3], [0, 4, 1], [2, 0, 3], [2, 2, 1], [4, 0, 1]], dtype=jnp.int32),
        jnp.array([[1, 0, 4], [1, 2, 2], [1, 4, 0], [3, 0, 2], [3, 2, 0], [5, 0, 0]], dtype=jnp.int32),
        jnp.array([[0, 2, 3], [0, 4, 1], [2, 0, 3], [4, 0, 1]], dtype=jnp.int32),
        jnp.array([[1, 2, 2], [1, 4, 0], [3, 0, 2], [3, 2, 0], [5, 0, 0]], dtype=jnp.int32),
        jnp.array([[0, 4, 1], [2, 2, 1], [4, 0, 1]], dtype=jnp.int32),
        jnp.array([[1, 4, 0], [3, 2, 0], [5, 0, 0]], dtype=jnp.int32),
        jnp.array([[1, 5, 0], [3, 3, 0], [5, 1, 0]], dtype=jnp.int32),
        jnp.array([[0, 5, 1], [2, 3, 1], [4, 1, 1]], dtype=jnp.int32),
        jnp.array([[1, 3, 2], [1, 5, 0], [3, 1, 2], [5, 1, 0]], dtype=jnp.int32),
        jnp.array([[0, 3, 3], [0, 5, 1], [2, 1, 3], [2, 3, 1], [4, 1, 1]], dtype=jnp.int32),
        jnp.array([[1, 1, 4], [1, 3, 2], [1, 5, 0], [3, 1, 2], [3, 3, 0], [5, 1, 0]], dtype=jnp.int32),
        jnp.array([[0, 1, 5], [0, 3, 3], [0, 5, 1], [2, 1, 3], [2, 3, 1], [4, 1, 1]], dtype=jnp.int32),
        jnp.array(
            [[0, 0, 6], [0, 2, 4], [0, 4, 2], [0, 6, 0], [2, 0, 4], [2, 2, 2], [2, 4, 0], [4, 0, 2], [4, 2, 0], [6, 0, 0]],
            dtype=jnp.int32,
        ),
        jnp.array([[1, 0, 5], [1, 2, 3], [1, 4, 1], [3, 0, 3], [3, 2, 1], [5, 0, 1]], dtype=jnp.int32),
        jnp.array([[0, 2, 4], [0, 4, 2], [0, 6, 0], [2, 0, 4], [2, 4, 0], [4, 0, 2], [4, 2, 0], [6, 0, 0]], dtype=jnp.int32),
        jnp.array([[1, 2, 3], [1, 4, 1], [3, 0, 3], [3, 2, 1], [5, 0, 1]], dtype=jnp.int32),
        jnp.array([[0, 4, 2], [0, 6, 0], [2, 2, 2], [2, 4, 0], [4, 0, 2], [4, 2, 0], [6, 0, 0]], dtype=jnp.int32),
        jnp.array([[1, 4, 1], [3, 2, 1], [5, 0, 1]], dtype=jnp.int32),
        jnp.array([[0, 6, 0], [2, 4, 0], [4, 2, 0], [6, 0, 0]], dtype=jnp.int32),
    )

    def _eval_poly_value_grad_lap(
        coeffs: jax.Array,
        exps: jax.Array,
        x_pows: jax.Array,
        y_pows: jax.Array,
        z_pows: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # Electron-batched: x/y/z pows are (7, num_e); each term is
        # (n_terms, num_e) and we sum over the poly-term axis (0), keeping the
        # electron axis. Returns val (num_e,), grad (num_e, 3), lap (num_e,).
        ex = exps[:, 0]
        ey = exps[:, 1]
        ez = exps[:, 2]

        val_terms = coeffs[:, None] * x_pows[ex] * y_pows[ey] * z_pows[ez]
        val = jnp.sum(val_terms, axis=0)

        ex_m1 = jnp.where(ex > 0, ex - 1, 0)
        ey_m1 = jnp.where(ey > 0, ey - 1, 0)
        ez_m1 = jnp.where(ez > 0, ez - 1, 0)

        gx_terms = (coeffs * ex)[:, None] * x_pows[ex_m1] * y_pows[ey] * z_pows[ez]
        gy_terms = (coeffs * ey)[:, None] * x_pows[ex] * y_pows[ey_m1] * z_pows[ez]
        gz_terms = (coeffs * ez)[:, None] * x_pows[ex] * y_pows[ey] * z_pows[ez_m1]
        grad = jnp.stack([jnp.sum(gx_terms, axis=0), jnp.sum(gy_terms, axis=0), jnp.sum(gz_terms, axis=0)], axis=-1)

        ex_m2 = jnp.where(ex > 1, ex - 2, 0)
        ey_m2 = jnp.where(ey > 1, ey - 2, 0)
        ez_m2 = jnp.where(ez > 1, ez - 2, 0)

        lap_x = (coeffs * ex * (ex - 1))[:, None] * x_pows[ex_m2] * y_pows[ey] * z_pows[ez]
        lap_y = (coeffs * ey * (ey - 1))[:, None] * x_pows[ex] * y_pows[ey_m2] * z_pows[ez]
        lap_z = (coeffs * ez * (ez - 1))[:, None] * x_pows[ex] * y_pows[ey] * z_pows[ez_m2]
        lap = jnp.sum(lap_x + lap_y + lap_z, axis=0)

        return val, grad, lap

    def _batch_val_grad_lap(diffs_R: jnp.ndarray) -> tuple[jax.Array, jax.Array, jax.Array]:
        # diffs_R: (num_e, 3) -- all electrons for one unique R-center.
        x, y, z = diffs_R[:, 0], diffs_R[:, 1], diffs_R[:, 2]
        x_pows = jnp.stack([x**0, x, x**2, x**3, x**4, x**5, x**6], axis=0)  # (7, num_e)
        y_pows = jnp.stack([y**0, y, y**2, y**3, y**4, y**5, y**6], axis=0)
        z_pows = jnp.stack([z**0, z, z**2, z**3, z**4, z**5, z**6], axis=0)

        vals_list = []
        grads_list = []
        laps_list = []
        for coeffs, exps in zip(S_L_M_COEFFS[:n_lm], S_L_M_EXPS[:n_lm], strict=True):
            val, grad, lap = _eval_poly_value_grad_lap(coeffs, exps, x_pows, y_pows, z_pows)
            vals_list.append(val)
            grads_list.append(grad)
            laps_list.append(lap)

        vals = jnp.stack(vals_list, axis=0)  # (n_lm, num_e)
        grads = jnp.stack(grads_list, axis=0)  # (n_lm, num_e, 3)
        laps = jnp.stack(laps_list, axis=0)  # (n_lm, num_e)
        return vals, grads, laps

    # vmap over the R-center axis only (electrons are batched inside the
    # kernel), so the angular table is produced natively in (num_R, n_lm,
    # num_e[, 3]) order -- the exact layout the caller's prim-rank gather
    # wants, with no `.T` / `transpose(1,0,2)` + caller-side `order="F"`
    # flatten, i.e. no standalone 5D transpose kernel.
    vals, grads, laps = vmap(_batch_val_grad_lap)(r_R_diffs_uq)
    return vals, grads, laps


def _compute_S_l_m_debug(
    angular_momentum: int,
    magnetic_quantum_number: int,
    atomic_center_cart: list[float],
    r_cart: list[float],
    *,
    zone: str = "ao_eval",
) -> float:
    r"""Solid harmonics part of a primitve AO.

    Compute the solid harmonics, i.e., r^l * spherical hamonics part (c.f., regular solid harmonics) of a given AO

    Args:
        angular_momentum (int): Angular momentum of the AO, i.e., l
        magnetic_quantum_number (int): Magnetic quantum number of the AO, i.e m = -l .... +l
        atomic_center_cart (list[float]): Center of the nucleus associated to the AO.
        r_cart (list[float]): Cartesian coordinate of an electron
        zone (str): Precision Zone whose dtype is used for the evaluation. Default ``"ao_eval"``.

    Returns:
        float: Value of the spherical harmonics part * r^l (i.e., regular solid harmonics).

    Note:
        A real basis of spherical harmonics Y_{l,m} : S^2 -> R can be defined in terms of
        their complex analogues  Y_{l}^{m} : S^2 -> C by setting:
        Y_{l,m}(theta, phi) =
                sqrt(2) * (-1)^m * \Im[Y_l^{|m|}] (if m < 0)
                Y_l^{0} (if m = 0)
                sqrt(2) * (-1)^m * \Re[Y_l^{|m|}] (if m > 0)

        A conversion from cartesian to spherical coordinate is:
                r = sqrt(x**2 + y**2 + z**2)
                theta = arccos(z/r)
                phi = sgn(y)arccos(x/sqrt(x**2+y**2))

        It indicates that there are two singular points
                1) the origin (x,y,z) = (0,0,0)
                2) points on the z axis (0,0,z)

        Therefore, instead, the so-called solid harmonics function is computed, which is defined as
        S_{l,\pm|m|} = \sqrt(\cfrac{4 * np.pi}{2 * l + 1}) * |\vec{R} - \vec{r}|^l [Y_{l,m,\alpha}(\phi, \theta) +- Y_{l,-m,\alpha}(\phi, \theta)].

        The real solid harmonics function are tabulated in many textbooks and websites such as Wikipedia.
        They can be hardcoded into a code, or they can be computed analytically (e.g., https://en.wikipedia.org/wiki/Solid_harmonics).
        The latter one is the strategy employed in this code,
    """
    dtype_np = get_dtype_np(zone)
    displacement = np.asarray(np.asarray(r_cart) - np.asarray(atomic_center_cart), dtype=dtype_np)
    x, y, z = displacement
    r_norm = LA.norm(displacement)
    l, m = int(angular_momentum), int(magnetic_quantum_number)
    m_abs = abs(m)

    # solid harmonics for (x,y) dependent part:
    def A_m(x: float, y: float) -> float:
        return np.sum(
            [
                dtype_np(scipy.special.binom(m_abs, p) * (1, 0, -1, 0)[(m_abs - p) % 4])
                * np.power(x, p, dtype=dtype_np)
                * np.power(y, m_abs - p, dtype=dtype_np)
                for p in range(m_abs + 1)
            ],
            dtype=dtype_np,
        )

    def B_m(x: float, y: float) -> float:
        return np.sum(
            [
                dtype_np(scipy.special.binom(m_abs, p) * (0, 1, 0, -1)[(m_abs - p) % 4])
                * np.power(x, p, dtype=dtype_np)
                * np.power(y, m_abs - p, dtype=dtype_np)
                for p in range(m_abs + 1)
            ],
            dtype=dtype_np,
        )

    # solid harmonics for (z) dependent part:
    def lambda_lm(k: int) -> float:
        return dtype_np(
            (-1.0) ** (k)
            * 2.0 ** (-l)
            * scipy.special.binom(l, k)
            * scipy.special.binom(2 * l - 2 * k, l)
            * scipy.special.factorial(l - 2 * k)
            / scipy.special.factorial(l - 2 * k - m_abs)
        )

    # solid harmonics for (z) dependent part:
    def Lambda_lm(r_norm: float, z: float) -> float:
        return np.sqrt(
            dtype_np((2 - int(m_abs == 0)) * scipy.special.factorial(l - m_abs) / scipy.special.factorial(l + m_abs))
        ) * np.sum(
            [
                lambda_lm(k) * np.power(r_norm, 2 * k, dtype=dtype_np) * np.power(z, l - 2 * k - m_abs, dtype=dtype_np)
                for k in range((l - m_abs) // 2 + 1)
            ],
            dtype=dtype_np,
        )

    # solid harmonics eveluated in Cartesian coord. (x,y,z):
    if m >= 0:
        gamma = Lambda_lm(r_norm, z) * A_m(x, y)
    if m < 0:
        gamma = Lambda_lm(r_norm, z) * B_m(x, y)
    return gamma


@jit
def _compute_AOs_laplacian_analytic_cart(aos_data: AOs_cart_gto_data, r_carts: jnp.ndarray) -> jax.Array:
    r"""Analytic Laplacian for Cartesian AOs (contracted).

    Implementation note (perf, poly-after-reduce):
        Same AO-rank assembly as ``_compute_AOs_value_grad_lap_cart``.
        Three radial moments ``NR``, ``ZNR``, ``Z^2NR`` are reduced from
        primitive rank to AO rank, then

        .. math::
           \nabla^2 \phi = NR \cdot \nabla^2 P
                            - 4\, ZNR \cdot (x \partial_x P + y \partial_y P + z \partial_z P)
                            + (4 r^2\, Z^2NR - 6\, ZNR) \cdot P

        is assembled at AO rank, so no
        ``(n_walker, num_ao_prim, n_elec)`` prim-rank Laplacian
        intermediate is formed.
    """
    dtype_jnp = get_dtype_jnp("ao_grad_lap")
    # Reconstruct r-R in caller-supplied precision (fp64 from MCMC walker state)
    # via JAX promotion, then downcast to the ao_grad_lap zone (Principle 3b).
    # r_carts forwarded as-is (Principle 3a); R_carts read from fp64 storage
    # accessor on the basis-data dataclass.
    R_carts = aos_data._atomic_center_carts_prim_jnp
    diff = (r_carts[None, :, :] - R_carts[:, None, :]).astype(dtype_jnp)
    c = aos_data._coefficients_jnp.astype(dtype_jnp)
    Z = aos_data._exponents_jnp.astype(dtype_jnp)
    l = aos_data._angular_momentums_prim_jnp

    N_fact = aos_data._normalization_factorial_ratio_prim_jnp.astype(dtype_jnp)
    # Static-unrolled (8 Z)**l avoids the XLA repeated-squaring while-loop
    # emitted by ``base ** l`` when ``l`` is a traced int array. The bound is the
    # largest total degree (``L_NORM``), not the coordinate-power bound ``L_MAX``
    # (they differ for incomplete Cartesian shells, issue #110).
    L_MAX = _cart_max_polynomial_order(aos_data)
    L_NORM = _cart_max_angular_momentum(aos_data)
    N_Z = (2.0 * Z / jnp.pi) ** (3.0 / 2.0) * _int_pow_unrolled_cart(8.0 * Z, l, L_NORM)
    N = jnp.sqrt(N_Z * N_fact)

    r2_prim = jnp.sum(diff**2, axis=-1)
    pref_prim = N[:, None] * c[:, None] * jnp.exp(-Z[:, None] * r2_prim)

    # Three radial moments contracted at primitive rank -> reduced to AO rank.
    Z_b = Z[:, None]
    NR_ao = _reduce_primitives_to_aos(pref_prim, aos_data)
    ZNR_ao = _reduce_primitives_to_aos(Z_b * pref_prim, aos_data)
    Z2NR_ao = _reduce_primitives_to_aos(Z_b * Z_b * pref_prim, aos_data)

    # AO-level coordinates.
    R_carts_ao = aos_data._atomic_center_carts_jnp
    diff_ao = (r_carts[None, :, :] - R_carts_ao[:, None, :]).astype(dtype_jnp)
    x, y, z = diff_ao[..., 0], diff_ao[..., 1], diff_ao[..., 2]
    r2_ao = jnp.sum(diff_ao**2, axis=-1)

    nx_ao = jnp.asarray(aos_data.polynominal_order_x, dtype=jnp.int32)
    ny_ao = jnp.asarray(aos_data.polynominal_order_y, dtype=jnp.int32)
    nz_ao = jnp.asarray(aos_data.polynominal_order_z, dtype=jnp.int32)

    # Static-unrolled integer powers at AO rank. See grad analog above for
    # rationale of the shifted-exponent formulation; here we additionally
    # need ``d/d^2_x x^n = n(n-1) x^(n-2)``.
    px = _int_pow_unrolled_cart(x, nx_ao, L_MAX)
    py = _int_pow_unrolled_cart(y, ny_ao, L_MAX)
    pz = _int_pow_unrolled_cart(z, nz_ao, L_MAX)
    qpx = _int_pow_unrolled_cart(x, jnp.maximum(nx_ao - 1, 0), L_MAX)
    qpy = _int_pow_unrolled_cart(y, jnp.maximum(ny_ao - 1, 0), L_MAX)
    qpz = _int_pow_unrolled_cart(z, jnp.maximum(nz_ao - 1, 0), L_MAX)
    qppx = _int_pow_unrolled_cart(x, jnp.maximum(nx_ao - 2, 0), L_MAX)
    qppy = _int_pow_unrolled_cart(y, jnp.maximum(ny_ao - 2, 0), L_MAX)
    qppz = _int_pow_unrolled_cart(z, jnp.maximum(nz_ao - 2, 0), L_MAX)

    nx_b = nx_ao[:, None].astype(dtype_jnp)
    ny_b = ny_ao[:, None].astype(dtype_jnp)
    nz_b = nz_ao[:, None].astype(dtype_jnp)

    P = px * py * pz
    lapP = (
        (nx_b * (nx_b - 1.0) * qppx) * py * pz + px * (ny_b * (ny_b - 1.0) * qppy) * pz + px * py * (nz_b * (nz_b - 1.0) * qppz)
    )
    rdotgradP = x * (nx_b * qpx) * py * pz + y * px * (ny_b * qpy) * pz + z * px * py * (nz_b * qpz)
    lap = NR_ao * lapP - 4.0 * ZNR_ao * rdotgradP + (4.0 * r2_ao * Z2NR_ao - 6.0 * ZNR_ao) * P
    return lap


@jit
def _compute_AOs_laplacian_analytic_sphe(aos_data: AOs_sphe_gto_data, r_carts: jnp.ndarray) -> jax.Array:
    """Analytic Laplacian for spherical AOs (contracted)."""
    dtype_jnp = get_dtype_jnp("ao_grad_lap")
    # Reconstruct r-R in caller-supplied precision (fp64 from MCMC walker state)
    # via JAX promotion, then downcast to the ao_grad_lap zone (Principle 3b).
    # r_carts forwarded as-is (Principle 3a); R_carts read from fp64 storage
    # accessor on the basis-data dataclass.
    R_carts = aos_data._atomic_center_carts_prim_jnp
    R_carts_unique = aos_data._uq_center_carts_jnp
    r_R_diffs = (r_carts[None, :, :] - R_carts[:, None, :]).astype(dtype_jnp)
    r_R_diffs_uq = (r_carts[None, :, :] - R_carts_unique[:, None, :]).astype(dtype_jnp)
    nucleus_index_prim_jnp = aos_data._uq_center_index_prim_jnp
    c_jnp = aos_data._coefficients_jnp.astype(dtype_jnp)
    Z_jnp = aos_data._exponents_jnp.astype(dtype_jnp)
    l_jnp = aos_data._angular_momentums_prim_jnp
    m_jnp = aos_data._magnetic_quantum_numbers_prim_jnp

    l_f64 = l_jnp.astype(dtype_jnp)
    Z_f64 = Z_jnp.astype(dtype_jnp)
    factorial_l_plus_1 = jnp.exp(jscipy.special.gammaln(l_f64 + 2.0))
    factorial_2l_plus_2 = jnp.exp(jscipy.special.gammaln(2.0 * l_f64 + 3.0))

    N_n_dup = jnp.sqrt(
        (2.0 ** (2 * l_f64 + 3) * factorial_l_plus_1 * (2 * Z_f64) ** (l_f64 + 1.5)) / (factorial_2l_plus_2 * jnp.sqrt(jnp.pi))
    )
    N_l_m_dup = jnp.sqrt((2 * l_f64 + 1) / (4 * jnp.pi))

    r_squared = jnp.sum(r_R_diffs**2, axis=-1)
    R_n_dup = c_jnp[:, None] * jnp.exp(-Z_jnp[:, None] * r_squared)

    # Truncate the solid-harmonic table to (l_max+1)**2: the basis never
    # references harmonics above its max angular momentum, so computing the
    # full l=0..6 set (49) is pure waste. Basis-derived, trace-time static.
    n_lm = (int(aos_data._angular_momentums_prim_np.max()) + 1) ** 2
    S_l_m_vals_all, S_l_m_grads_all, S_l_m_laps_all = _compute_S_l_m_and_grad_lap(r_R_diffs_uq, n_lm=n_lm)
    max_ml = S_l_m_vals_all.shape[1]  # per-nucleus stride = rows actually tabulated

    # Angular table is (num_R, n_lm, num_e[, 3]); C-order flatten of the
    # (num_R, n_lm) axes so global_R_l_m_index = nucleus*n_lm + global_l_m
    # addresses the right row without a standalone transpose.
    S_l_m_vals_flat = S_l_m_vals_all.reshape((S_l_m_vals_all.shape[0] * S_l_m_vals_all.shape[1], S_l_m_vals_all.shape[2]))
    S_l_m_grads_flat = S_l_m_grads_all.reshape(
        (S_l_m_grads_all.shape[0] * S_l_m_grads_all.shape[1], S_l_m_grads_all.shape[2], 3)
    )
    S_l_m_laps_flat = S_l_m_laps_all.reshape((S_l_m_laps_all.shape[0] * S_l_m_laps_all.shape[1], S_l_m_laps_all.shape[2]))

    global_l_m_index = l_jnp**2 + (m_jnp + l_jnp)
    global_R_l_m_index = nucleus_index_prim_jnp * max_ml + global_l_m_index
    S_l_m_dup = S_l_m_vals_flat[global_R_l_m_index]
    S_l_m_grad_dup = S_l_m_grads_flat[global_R_l_m_index]
    S_l_m_lap_dup = S_l_m_laps_flat[global_R_l_m_index]

    pref = N_n_dup[:, None] * R_n_dup * N_l_m_dup[:, None]
    AOs_dup = pref * S_l_m_dup

    grad_S_dot_r = jnp.sum(S_l_m_grad_dup * r_R_diffs, axis=-1)
    lap_dup = (
        pref * S_l_m_lap_dup
        + AOs_dup * (4.0 * (Z_jnp[:, None] ** 2) * r_squared - 6.0 * Z_jnp[:, None])
        - 4.0 * Z_jnp[:, None] * pref * grad_S_dot_r
    )

    lap = _reduce_primitives_to_aos(lap_dup, aos_data)
    return lap


@jit
def _compute_AOs_grad_analytic_cart(
    aos_data: AOs_cart_gto_data, r_carts: jnp.ndarray
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Analytic gradients for Cartesian AOs (contracted).

    Implementation note (perf, poly-after-reduce):
        Same AO-rank assembly as ``_compute_AOs_value_grad_lap_cart``.
        Two radial moments ``NR = Sum_p N_p c_p e^{-Z_p r^2}`` and
        ``ZNR = Sum_p Z_p N_p c_p e^{-Z_p r^2}`` are reduced from
        primitive rank to AO rank, then

        .. math::
           \partial_a \phi = NR \cdot \partial_a P - 2 x_a\, ZNR \cdot P

        is assembled at AO rank. Eliminates the
        ``(n_walker, num_ao_prim, n_elec, 3)`` prim-rank gradient
        intermediate that would otherwise dominate the kinetic_disc /
        kinetic_continuum HLO when the standalone grad API is reached
        (e.g. via ``compute_AOs_grad`` callers).
    """
    dtype_jnp = get_dtype_jnp("ao_grad_lap")
    # Reconstruct r-R in caller-supplied precision (fp64 from MCMC walker state)
    # via JAX promotion, then downcast to the ao_grad_lap zone (Principle 3b).
    # r_carts forwarded as-is (Principle 3a); R_carts read from fp64 storage
    # accessor on the basis-data dataclass.
    R_carts = aos_data._atomic_center_carts_prim_jnp
    diff = (r_carts[None, :, :] - R_carts[:, None, :]).astype(dtype_jnp)
    c = aos_data._coefficients_jnp.astype(dtype_jnp)
    Z = aos_data._exponents_jnp.astype(dtype_jnp)
    l = aos_data._angular_momentums_prim_jnp

    N_fact = aos_data._normalization_factorial_ratio_prim_jnp.astype(dtype_jnp)
    # Static-unrolled (8 Z)**l avoids the XLA repeated-squaring while-loop.
    # Bound: largest total degree ``L_NORM`` (not ``L_MAX``; issue #110).
    L_MAX = _cart_max_polynomial_order(aos_data)
    L_NORM = _cart_max_angular_momentum(aos_data)
    N_Z = (2.0 * Z / jnp.pi) ** (3.0 / 2.0) * _int_pow_unrolled_cart(8.0 * Z, l, L_NORM)
    N = jnp.sqrt(N_Z * N_fact)

    r2_prim = jnp.sum(diff**2, axis=-1)
    pref_prim = N[:, None] * c[:, None] * jnp.exp(-Z[:, None] * r2_prim)

    # Two radial moments contracted at primitive rank -> reduced to AO rank.
    Z_b = Z[:, None]
    NR_ao = _reduce_primitives_to_aos(pref_prim, aos_data)  # Sum_p pref_p
    ZNR_ao = _reduce_primitives_to_aos(Z_b * pref_prim, aos_data)  # Sum_p Z_p pref_p

    # AO-level coordinates: each AO sits on exactly one atom.
    R_carts_ao = aos_data._atomic_center_carts_jnp
    diff_ao = (r_carts[None, :, :] - R_carts_ao[:, None, :]).astype(dtype_jnp)
    x, y, z = diff_ao[..., 0], diff_ao[..., 1], diff_ao[..., 2]

    # AO-rank polynomial orders (length num_ao). Static, constant-folded.
    nx_ao = jnp.asarray(aos_data.polynominal_order_x, dtype=jnp.int32)
    ny_ao = jnp.asarray(aos_data.polynominal_order_y, dtype=jnp.int32)
    nz_ao = jnp.asarray(aos_data.polynominal_order_z, dtype=jnp.int32)

    # Static-unrolled integer powers at AO rank. Shifted exponent
    # ``max(n - 1, 0)`` combined with prefactor ``n`` expresses
    # ``d/d_x x^n = n x^(n-1)`` directly; for ``n == 0`` the prefactor zeros
    # the term, matching the analytic limit (no eps, no divisions).
    px = _int_pow_unrolled_cart(x, nx_ao, L_MAX)
    py = _int_pow_unrolled_cart(y, ny_ao, L_MAX)
    pz = _int_pow_unrolled_cart(z, nz_ao, L_MAX)
    qpx = _int_pow_unrolled_cart(x, jnp.maximum(nx_ao - 1, 0), L_MAX)
    qpy = _int_pow_unrolled_cart(y, jnp.maximum(ny_ao - 1, 0), L_MAX)
    qpz = _int_pow_unrolled_cart(z, jnp.maximum(nz_ao - 1, 0), L_MAX)

    nx_b = nx_ao[:, None].astype(dtype_jnp)
    ny_b = ny_ao[:, None].astype(dtype_jnp)
    nz_b = nz_ao[:, None].astype(dtype_jnp)

    P = px * py * pz  # (num_ao, n_elec)

    # d/d_a phi = NR * d/d_a P - 2 x_a * ZNR * P
    gx = NR_ao * ((nx_b * qpx) * py * pz) - 2.0 * x * ZNR_ao * P
    gy = NR_ao * (px * (ny_b * qpy) * pz) - 2.0 * y * ZNR_ao * P
    gz = NR_ao * (px * py * (nz_b * qpz)) - 2.0 * z * ZNR_ao * P

    return gx, gy, gz


@jit
def _compute_AOs_grad_analytic_sphe(
    aos_data: AOs_sphe_gto_data, r_carts: jnp.ndarray
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Analytic gradients for spherical AOs (contracted)."""
    dtype_jnp = get_dtype_jnp("ao_grad_lap")
    # Reconstruct r-R in caller-supplied precision (fp64 from MCMC walker state)
    # via JAX promotion, then downcast to the ao_grad_lap zone (Principle 3b).
    # r_carts forwarded as-is (Principle 3a); R_carts read from fp64 storage
    # accessor on the basis-data dataclass.
    R_carts = aos_data._atomic_center_carts_prim_jnp
    R_carts_unique = aos_data._uq_center_carts_jnp
    r_R_diffs = (r_carts[None, :, :] - R_carts[:, None, :]).astype(dtype_jnp)
    r_R_diffs_uq = (r_carts[None, :, :] - R_carts_unique[:, None, :]).astype(dtype_jnp)
    nucleus_index_prim_jnp = aos_data._uq_center_index_prim_jnp
    c_jnp = aos_data._coefficients_jnp.astype(dtype_jnp)
    Z_jnp = aos_data._exponents_jnp.astype(dtype_jnp)
    l_jnp = aos_data._angular_momentums_prim_jnp
    m_jnp = aos_data._magnetic_quantum_numbers_prim_jnp

    l_f64 = l_jnp.astype(dtype_jnp)
    Z_f64 = Z_jnp.astype(dtype_jnp)
    factorial_l_plus_1 = jnp.exp(jscipy.special.gammaln(l_f64 + 2.0))
    factorial_2l_plus_2 = jnp.exp(jscipy.special.gammaln(2.0 * l_f64 + 3.0))

    N_n_dup = jnp.sqrt(
        (2.0 ** (2 * l_f64 + 3) * factorial_l_plus_1 * (2 * Z_f64) ** (l_f64 + 1.5)) / (factorial_2l_plus_2 * jnp.sqrt(jnp.pi))
    )
    N_l_m_dup = jnp.sqrt((2 * l_f64 + 1) / (4 * jnp.pi))

    r_squared = jnp.sum(r_R_diffs**2, axis=-1)
    R_n_dup = c_jnp[:, None] * jnp.exp(-Z_jnp[:, None] * r_squared)

    # Use a single ``_compute_S_l_m_and_grad_lap`` call and reuse its value
    # output instead of re-running ``_compute_S_l_m``. The helper is pinned
    # to the ``ao_grad_lap`` zone (fp64), and this caller is also in the
    # ``ao_grad_lap`` zone -- so the ``.astype(dtype_jnp)`` calls
    # below are no-ops at runtime but kept for explicit Principle 3b
    # documentation. Lap output is unused -- JAX DCE eliminates it because
    # this whole function is inlined inside the caller's @jit and the lap
    # branch never reaches a sink.
    # Truncate the solid-harmonic table to (l_max+1)**2 (see fused/lap kernels).
    n_lm = (int(aos_data._angular_momentums_prim_np.max()) + 1) ** 2
    S_l_m_dup_all_l_m, S_l_m_grad_all_l_m, _ = _compute_S_l_m_and_grad_lap(r_R_diffs_uq, n_lm=n_lm)
    max_ml = S_l_m_dup_all_l_m.shape[1]  # per-nucleus stride = rows actually tabulated
    S_l_m_dup_all_l_m = S_l_m_dup_all_l_m.astype(dtype_jnp)
    S_l_m_grad_all_l_m = S_l_m_grad_all_l_m.astype(dtype_jnp)

    # Angular table is (num_R, n_lm, num_e[, 3]); C-order flatten of the
    # (num_R, n_lm) axes (max_ml = n_lm) -- no transpose. See fused kernel.
    S_l_m_dup_all_l_m_reshaped = S_l_m_dup_all_l_m.reshape(
        (S_l_m_dup_all_l_m.shape[0] * S_l_m_dup_all_l_m.shape[1], S_l_m_dup_all_l_m.shape[2])
    )
    S_l_m_grad_all_l_m_reshaped = S_l_m_grad_all_l_m.reshape(
        (
            S_l_m_grad_all_l_m.shape[0] * S_l_m_grad_all_l_m.shape[1],
            S_l_m_grad_all_l_m.shape[2],
            S_l_m_grad_all_l_m.shape[3],
        )
    )

    global_l_m_index = l_jnp**2 + (m_jnp + l_jnp)
    global_R_l_m_index = nucleus_index_prim_jnp * max_ml + global_l_m_index
    S_l_m_dup = S_l_m_dup_all_l_m_reshaped[global_R_l_m_index]
    S_l_m_grad_dup = S_l_m_grad_all_l_m_reshaped[global_R_l_m_index]

    pref = N_n_dup[:, None] * R_n_dup * N_l_m_dup[:, None]
    AOs_dup = pref * S_l_m_dup

    grad_from_R = AOs_dup[..., None] * (-2.0 * Z_jnp[:, None, None] * r_R_diffs)
    grad_from_S = pref[..., None] * S_l_m_grad_dup
    grad_dup = grad_from_R + grad_from_S

    gx = _reduce_primitives_to_aos(grad_dup[..., 0], aos_data)
    gy = _reduce_primitives_to_aos(grad_dup[..., 1], aos_data)
    gz = _reduce_primitives_to_aos(grad_dup[..., 2], aos_data)

    return gx, gy, gz


@jit
def _compute_AOs_value_grad_lap_cart(
    aos_data: AOs_cart_gto_data, r_carts: jnp.ndarray
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""Fused value/grad/lap for Cartesian AOs (contracted).

    Implementation note (perf, poly-after-reduce for grad/lap):
        The angular polynomial part :math:`x^{n_x} y^{n_y} z^{n_z}` and
        its derivatives depend only on AO-level quantum numbers, not on
        the primitive index. Within an AO, every primitive shares the
        same ``(n_x, n_y, n_z)`` and the same Cartesian center. Writing

        .. math::
           \mathrm{pref}_p = N_p c_p e^{-Z_p r^2}, \qquad
           NR    = \sum_p \mathrm{pref}_p, \qquad
           ZNR   = \sum_p Z_p\, \mathrm{pref}_p, \qquad
           Z^2NR = \sum_p Z_p^2\, \mathrm{pref}_p,

        all primitive-rank arrays disappear after three radial reductions
        -- value, grad and Laplacian then reduce to AO-rank polynomial
        algebra:

        .. math::
           \phi    &= NR \cdot P                                              \\
           \partial_a \phi &= NR \cdot \partial_a P - 2 x_a\, ZNR \cdot P   \\
           \nabla^2 \phi   &= NR \cdot \nabla^2 P
                              - 4\, ZNR \cdot (x \partial_x P + y \partial_y P + z \partial_z P)
                              + (4 r^2\, Z^2NR - 6\, ZNR) \cdot P

        The kernel performs one prim-rank reduce per radial moment
        (NR / ZNR / Z^2NR) and runs all polynomial work at AO rank.
        This avoids the four ``(n_walker, num_ao_prim, n_elec)``
        prim-rank tuples (``phi``, ``Kx``, ``Ky``, ``Kz``) that would
        otherwise dominate the kinetic energy / local energy DRAM
        traffic.
    """
    dtype_eval = get_dtype_jnp(aos_data._value_zone)
    dtype_jnp = get_dtype_jnp("ao_grad_lap")
    # Reconstruct r-R in caller-supplied precision (fp64) via JAX promotion
    # then downcast to the shared ao_grad_lap zone (Principle 3b). Mirrors
    # _compute_AOs_grad_analytic_cart / _compute_AOs_laplacian_analytic_cart
    # so grad/lap parity vs the standalone APIs is bitwise in full mode.
    R_carts = aos_data._atomic_center_carts_prim_jnp
    diff = (r_carts[None, :, :] - R_carts[:, None, :]).astype(dtype_jnp)
    c = aos_data._coefficients_jnp.astype(dtype_jnp)
    Z = aos_data._exponents_jnp.astype(dtype_jnp)
    l = aos_data._angular_momentums_prim_jnp

    N_fact = aos_data._normalization_factorial_ratio_prim_jnp.astype(dtype_jnp)
    # Static-unrolled (8 Z)**l avoids the XLA repeated-squaring while-loop.
    # Bound: largest total degree ``L_NORM`` (not ``L_MAX``; issue #110).
    L_MAX = _cart_max_polynomial_order(aos_data)
    L_NORM = _cart_max_angular_momentum(aos_data)
    N_Z = (2.0 * Z / jnp.pi) ** (3.0 / 2.0) * _int_pow_unrolled_cart(8.0 * Z, l, L_NORM)
    N = jnp.sqrt(N_Z * N_fact)

    r2_prim = jnp.sum(diff**2, axis=-1)
    pref_prim = N[:, None] * c[:, None] * jnp.exp(-Z[:, None] * r2_prim)

    # Three radial moments contracted at primitive rank -> reduced to AO rank
    # via a single _reduce_primitives_to_aos each. Everything downstream
    # operates at AO rank (num_ao << num_ao_prim), eliminating the
    # (n_walker, num_ao_prim, n_elec)-sized intermediate tuple that
    # dominated the kinetic energy HLO.
    Z_b = Z[:, None]
    NR_ao = _reduce_primitives_to_aos(pref_prim, aos_data)  # Sum_p pref_p
    ZNR_ao = _reduce_primitives_to_aos(Z_b * pref_prim, aos_data)  # Sum_p Z_p pref_p
    Z2NR_ao = _reduce_primitives_to_aos(Z_b * Z_b * pref_prim, aos_data)  # Sum_p Z_p^2 pref_p

    # AO-level coordinates: each AO sits on exactly one atom, so we use the
    # AO->atom mapping (length num_ao). r-R is reconstructed in fp64 then
    # cast to the ao_grad_lap zone -- same protocol as the prim-rank diff
    # above, ensuring bit-exact match between the two coordinate sources
    # whenever an electron sits on a nucleus.
    R_carts_ao = aos_data._atomic_center_carts_jnp
    diff_ao = (r_carts[None, :, :] - R_carts_ao[:, None, :]).astype(dtype_jnp)
    x, y, z = diff_ao[..., 0], diff_ao[..., 1], diff_ao[..., 2]
    r2_ao = jnp.sum(diff_ao**2, axis=-1)

    # AO-rank polynomial orders (length num_ao). These are static lists on
    # the dataclass; ``jnp.asarray`` is constant-folded into the JIT.
    nx_ao = jnp.asarray(aos_data.polynominal_order_x, dtype=jnp.int32)
    ny_ao = jnp.asarray(aos_data.polynominal_order_y, dtype=jnp.int32)
    nz_ao = jnp.asarray(aos_data.polynominal_order_z, dtype=jnp.int32)

    # Static-unrolled integer powers at AO rank. Shifted exponents
    # (``max(n - 1, 0)``, ``max(n - 2, 0)``) combined with the integer
    # prefactors (``n``, ``n(n-1)``) express the first and second
    # derivatives of ``x^n`` in product form, matching the layout used
    # by the standalone analytic grad/lap kernels.
    px = _int_pow_unrolled_cart(x, nx_ao, L_MAX)
    py = _int_pow_unrolled_cart(y, ny_ao, L_MAX)
    pz = _int_pow_unrolled_cart(z, nz_ao, L_MAX)
    qpx = _int_pow_unrolled_cart(x, jnp.maximum(nx_ao - 1, 0), L_MAX)
    qpy = _int_pow_unrolled_cart(y, jnp.maximum(ny_ao - 1, 0), L_MAX)
    qpz = _int_pow_unrolled_cart(z, jnp.maximum(nz_ao - 1, 0), L_MAX)
    qppx = _int_pow_unrolled_cart(x, jnp.maximum(nx_ao - 2, 0), L_MAX)
    qppy = _int_pow_unrolled_cart(y, jnp.maximum(ny_ao - 2, 0), L_MAX)
    qppz = _int_pow_unrolled_cart(z, jnp.maximum(nz_ao - 2, 0), L_MAX)

    nx_b = nx_ao[:, None].astype(dtype_jnp)
    ny_b = ny_ao[:, None].astype(dtype_jnp)
    nz_b = nz_ao[:, None].astype(dtype_jnp)

    P = px * py * pz  # (num_ao, n_elec)

    # value finalize: only downcast site (Principle 3b).
    val = (NR_ao * P).astype(dtype_eval)

    # grad finalize (kept in ao_grad_lap zone -- no cast).
    # d/d_a phi = NR * d/d_a P - 2 x_a * ZNR * P, with
    # d/d_x P = n_x x^{n_x-1} y^{n_y} z^{n_z}, etc.
    gx = NR_ao * ((nx_b * qpx) * py * pz) - 2.0 * x * ZNR_ao * P
    gy = NR_ao * (px * (ny_b * qpy) * pz) - 2.0 * y * ZNR_ao * P
    gz = NR_ao * (px * py * (nz_b * qpz)) - 2.0 * z * ZNR_ao * P

    # lap finalize (kept in ao_grad_lap zone -- no cast).
    # nabla^2P = Sum_a n_a(n_a-1) x_a^{n_a-2} Prod_{b!=a} x_b^{n_b}
    lapP = (
        (nx_b * (nx_b - 1.0) * qppx) * py * pz + px * (ny_b * (ny_b - 1.0) * qppy) * pz + px * py * (nz_b * (nz_b - 1.0) * qppz)
    )
    # x*d/d_xP + y*d/d_yP + z*d/d_zP = (n_x + n_y + n_z) * P  (Euler identity);
    # the explicit form is kept for a bit-exact match with the prim-rank
    # form when arithmetic order matters.
    rdotgradP = x * (nx_b * qpx) * py * pz + y * px * (ny_b * qpy) * pz + z * px * py * (nz_b * qpz)
    lap = NR_ao * lapP - 4.0 * ZNR_ao * rdotgradP + (4.0 * r2_ao * Z2NR_ao - 6.0 * ZNR_ao) * P

    return val, gx, gy, gz, lap


@jit
def _compute_AOs_value_grad_lap_sphe(
    aos_data: AOs_sphe_gto_data, r_carts: jnp.ndarray
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fused value/grad/lap for spherical AOs (contracted).

    Shared heavy block (``exp(-Z r^2)``, ``_compute_S_l_m_and_grad_lap``,
    ``pref = N_n * R_n * N_l_m``) is evaluated once in the ``ao_grad_lap``
    zone (fp64); only the value output is downcast to ``ao_eval`` at the
    segment-sum site. The single ``_compute_S_l_m_and_grad_lap`` call
    avoids the 2-3x duplicate evaluations of separate eval / grad / lap
    kernels.
    """
    dtype_eval = get_dtype_jnp(aos_data._value_zone)
    dtype_jnp = get_dtype_jnp("ao_grad_lap")
    # Reconstruct r-R in caller-supplied precision (fp64) via JAX promotion
    # then downcast to the shared ao_grad_lap zone (Principle 3b). Mirrors
    # _compute_AOs_laplacian_analytic_sphe so grad/lap parity vs the
    # standalone APIs is bitwise in full mode.
    R_carts = aos_data._atomic_center_carts_prim_jnp
    R_carts_unique = aos_data._uq_center_carts_jnp
    r_R_diffs = (r_carts[None, :, :] - R_carts[:, None, :]).astype(dtype_jnp)
    r_R_diffs_uq = (r_carts[None, :, :] - R_carts_unique[:, None, :]).astype(dtype_jnp)
    nucleus_index_prim_jnp = aos_data._uq_center_index_prim_jnp
    c_jnp = aos_data._coefficients_jnp.astype(dtype_jnp)
    Z_jnp = aos_data._exponents_jnp.astype(dtype_jnp)
    l_jnp = aos_data._angular_momentums_prim_jnp
    m_jnp = aos_data._magnetic_quantum_numbers_prim_jnp

    l_f64 = l_jnp.astype(dtype_jnp)
    Z_f64 = Z_jnp.astype(dtype_jnp)
    factorial_l_plus_1 = jnp.exp(jscipy.special.gammaln(l_f64 + 2.0))
    factorial_2l_plus_2 = jnp.exp(jscipy.special.gammaln(2.0 * l_f64 + 3.0))

    N_n_dup = jnp.sqrt(
        (2.0 ** (2 * l_f64 + 3) * factorial_l_plus_1 * (2 * Z_f64) ** (l_f64 + 1.5)) / (factorial_2l_plus_2 * jnp.sqrt(jnp.pi))
    )
    N_l_m_dup = jnp.sqrt((2 * l_f64 + 1) / (4 * jnp.pi))

    r_squared = jnp.sum(r_R_diffs**2, axis=-1)
    R_n_dup = c_jnp[:, None] * jnp.exp(-Z_jnp[:, None] * r_squared)

    # Single S_l_m call returning (vals, grads, laps) -- avoids the
    # 2-3x duplicate evaluations of separate eval/grad/lap kernels.
    # Truncate to (l_max+1)**2: harmonics above the basis max-l are never
    # gathered, so computing the full l=0..6 set (49) is pure waste.
    n_lm = (int(aos_data._angular_momentums_prim_np.max()) + 1) ** 2
    S_l_m_vals_all, S_l_m_grads_all, S_l_m_laps_all = _compute_S_l_m_and_grad_lap(r_R_diffs_uq, n_lm=n_lm)
    max_ml = S_l_m_vals_all.shape[1]  # per-nucleus stride = rows actually tabulated

    # Angular table is (num_R, n_lm, num_e[, 3]); C-order flatten of the
    # (num_R, n_lm) axes so global_R_l_m_index = nucleus*n_lm + global_l_m
    # addresses the right row without a standalone transpose.
    S_l_m_vals_flat = S_l_m_vals_all.reshape((S_l_m_vals_all.shape[0] * S_l_m_vals_all.shape[1], S_l_m_vals_all.shape[2]))
    S_l_m_grads_flat = S_l_m_grads_all.reshape(
        (S_l_m_grads_all.shape[0] * S_l_m_grads_all.shape[1], S_l_m_grads_all.shape[2], 3)
    )
    S_l_m_laps_flat = S_l_m_laps_all.reshape((S_l_m_laps_all.shape[0] * S_l_m_laps_all.shape[1], S_l_m_laps_all.shape[2]))

    global_l_m_index = l_jnp**2 + (m_jnp + l_jnp)
    global_R_l_m_index = nucleus_index_prim_jnp * max_ml + global_l_m_index
    S_l_m_dup = S_l_m_vals_flat[global_R_l_m_index]
    S_l_m_grad_dup = S_l_m_grads_flat[global_R_l_m_index]
    S_l_m_lap_dup = S_l_m_laps_flat[global_R_l_m_index]

    # Shared body identical to the standalone grad/lap kernels.
    pref = N_n_dup[:, None] * R_n_dup * N_l_m_dup[:, None]
    AOs_dup = pref * S_l_m_dup

    # value finalize: only downcast site (Principle 3b).
    val = _reduce_primitives_to_aos(AOs_dup.astype(dtype_eval), aos_data)

    # grad finalize (kept in ao_grad_lap zone -- no cast).
    grad_from_R = AOs_dup[..., None] * (-2.0 * Z_jnp[:, None, None] * r_R_diffs)
    grad_from_S = pref[..., None] * S_l_m_grad_dup
    grad_dup = grad_from_R + grad_from_S
    gx = _reduce_primitives_to_aos(grad_dup[..., 0], aos_data)
    gy = _reduce_primitives_to_aos(grad_dup[..., 1], aos_data)
    gz = _reduce_primitives_to_aos(grad_dup[..., 2], aos_data)

    # lap finalize (kept in ao_grad_lap zone -- no cast).
    grad_S_dot_r = jnp.sum(S_l_m_grad_dup * r_R_diffs, axis=-1)
    lap_dup = (
        pref * S_l_m_lap_dup
        + AOs_dup * (4.0 * (Z_jnp[:, None] ** 2) * r_squared - 6.0 * Z_jnp[:, None])
        - 4.0 * Z_jnp[:, None] * pref * grad_S_dot_r
    )
    lap = _reduce_primitives_to_aos(lap_dup, aos_data)

    return val, gx, gy, gz, lap


# ============================================================================
# Public API
# ============================================================================
# Polymorphic dispatchers for AO evaluation (value / gradient / Laplacian /
# fused value-grad-lap) and the AO overlap matrix. Each delegates to per-leaf
# methods defined on the AO data classes above; this section is the public
# surface consumed by the rest of jQMC (determinant.py, molecular_orbital.py,
# etc.). Paired debug oracles (FDM-based numerical cross-checks) and autodiff
# alternatives are co-located with their dispatcher for symmetry.


def compute_AOs(aos_data: AOs_base_data, r_carts: jax.Array) -> jax.Array:
    """Evaluate contracted atomic orbitals (AOs) at electron coordinates.

    Dispatches to Cartesian or real-spherical backends and returns float64 JAX arrays
    (ensure ``jax_enable_x64=True``). Call ``aos_data.sanity_check()`` before use.

    For periodic ``aos_data`` (built via
    :py:func:`_apply_image_expansion`), the per-leaf method collapses the
    kernel output ``(n_image * num_ao, N_e)`` back to ``(num_ao, N_e)``
    via a post-kernel reshape-sum hidden inside the pgto leaf. Molecular
    instances emit HLO byte-identical to the pre-PBC molecular kernel.

    Args:
        aos_data: ``AOs_cart_gto_data`` or ``AOs_sphe_gto_data`` describing centers, primitive parameters,
            angular data, and contraction mapping.
        r_carts (jax.Array): Electron Cartesian coordinates, shape ``(N_e, 3)`` in Bohr. Casts to
            ``float64`` internally via ``jnp.asarray``.

    Returns:
        jax.Array: AO values, shape ``(num_ao, N_e)``.

    Raises:
        NotImplementedError: If ``aos_data`` is neither Cartesian nor spherical.
    """
    # NOTE: do not pre-cast r_carts here. The per-leaf ``_compute_AOs``
    # methods reconstruct ``r - R`` in float64 to avoid catastrophic
    # cancellation (positions can be ~50 Bohr; an fp32 difference loses
    # ~6 digits). Downcasting r_carts in this wrapper would destroy that
    # precision *before* the fp64 reconstruction can use it.
    # Every leaf owns its own home-cell wrap and (for pgto) the lattice-image
    # fold via ``_compute_AOs``; delegate so this free-function API stays
    # agnostic (the contract consumed by ``compute_MOs``).
    return aos_data._compute_AOs(r_carts)


def _compute_AOs_debug(aos_data: AOs_base_data, r_carts: jnpt.ArrayLike) -> jax.Array:
    """Reference oracle for :py:func:`compute_AOs` (matches its return shape).

    Data-type-agnostic dispatcher: forwards to the per-leaf
    :py:meth:`AOs_base_data._compute_AOs_debug` method, mirroring the
    production :py:func:`compute_AOs` dispatcher. Each leaf owns its own
    plain-numpy oracle and (for periodic pgto leaves) the post-kernel
    image-axis reduction, so this wrapper never reaches into the
    ``_n_image`` static.

    All leaves return molecular-rank ``(num_ao, N_e)``: pgto leaves
    fold the kernel-rank ``(n_image * num_ao, N_e)`` oracle output internally
    (preserving the molecular golden-value oracle byte-for-byte);
    plane-wave / B-spline leaves carry
    no image axis and return ``(num_ao, N_e)`` directly.

    Callers needing the unfolded ``(n_image * num_ao, N_e)`` raw
    GTO output should invoke :py:func:`_compute_AOs_cart_debug` /
    :py:func:`_compute_AOs_sphe_debug` directly.
    """
    return aos_data._compute_AOs_debug(r_carts)


def compute_AOs_grad(aos_data: AOs_base_data, r_carts: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return analytic Cartesian gradients of contracted atomic orbitals.

    Public gradient API used for drift vectors and kinetic terms. Dispatches to Cartesian or real-spherical
    backends; returns float64 JAX arrays (ensure ``jax_enable_x64=True``).

    For periodic ``aos_data``, each of the three gradient components is
    collapsed across the image axis by a post-kernel reshape-sum; molecular
    instances (``_n_image == 1``) take a trace-time short-circuit so the
    emitted HLO is byte-identical to the pre-PBC kernel. See
    :py:func:`compute_AOs` for the rationale.

    Args:
        aos_data: ``AOs_cart_gto_data`` or ``AOs_sphe_gto_data`` describing primitive parameters, angular info,
            contraction mapping, and centers (run ``sanity_check()`` beforehand).
        r_carts (jax.Array): Electron Cartesian coordinates, shape ``(N_e, 3)`` (Bohr). Casts to ``float64``
            internally via ``jnp.asarray``.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: Gradients w.r.t. x, y, z, each of shape ``(num_ao, N_e)``.
        Order is ``(gx, gy, gz)``.

    Raises:
        NotImplementedError: If ``aos_data`` is neither Cartesian nor spherical.
    """
    # Data-type-agnostic dispatcher: forwards to the per-leaf
    # ``_compute_AOs_grad`` method. Each leaf owns wrap + kernel and (for
    # periodic pgto leaves) the post-kernel image-axis reduction, so this
    # wrapper never reaches into the ``_n_image`` static.
    return aos_data._compute_AOs_grad(r_carts)


def _compute_AOs_grad_autodiff(aos_data: AOs_base_data, r_carts: jnpt.ArrayLike) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute Cartesian Gradients of AOs.

    The method is for computing the Carteisan gradients (x,y,z) of
    the given atomic orbital at r_carts

    Args:
        aos_data (AOs_base_data): an instance of AOs_base_data
        r_carts (jnpt.ArrayLike): Cartesian coordinates of electrons (dim: N_e, 3)

    Returns:
        tuple: tuple containing gradients of the AOs at r_carts. (grad_x, grad_y, grad_z).
        The dim. of each matrix is (num_ao, N_e)

    """
    grad_full = jacrev(compute_AOs, argnums=1)(aos_data, r_carts)
    grad_diag = jnp.diagonal(grad_full, axis1=1, axis2=2)
    grad_diag = jnp.swapaxes(grad_diag, 1, 2)
    ao_matrix_grad_x = grad_diag[..., 0]  # (M, N)
    ao_matrix_grad_y = grad_diag[..., 1]  # (M, N)
    ao_matrix_grad_z = grad_diag[..., 2]  # (M, N)

    return ao_matrix_grad_x, ao_matrix_grad_y, ao_matrix_grad_z


# debug reference, not used in the production code
def _compute_AOs_grad_debug(
    aos_data: AOs_base_data,
    r_carts: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute Cartesian Gradients of AOs.

    The method is for computing the Carteisan gradients (x,y,z) of
    the given atomic orbital at r_carts using FDM for debugging JAX
    implementations. See :py:func:`compute_AOs_grad`.
    """
    # Gradients of AOs (numerical)
    diff_h = 1.0e-5

    # grad x
    diff_p_x_r_carts = r_carts.copy()
    diff_p_x_r_carts[:, 0] += diff_h
    ao_matrix_diff_p_x = compute_AOs(aos_data, diff_p_x_r_carts)
    diff_m_x_r_carts = r_carts.copy()
    diff_m_x_r_carts[:, 0] -= diff_h
    ao_matrix_diff_m_x = compute_AOs(aos_data, diff_m_x_r_carts)

    # grad y
    diff_p_y_r_carts = r_carts.copy()
    diff_p_y_r_carts[:, 1] += diff_h
    ao_matrix_diff_p_y = compute_AOs(aos_data, diff_p_y_r_carts)
    diff_m_y_r_carts = r_carts.copy()
    diff_m_y_r_carts[:, 1] -= diff_h
    ao_matrix_diff_m_y = compute_AOs(aos_data, diff_m_y_r_carts)

    # grad z
    diff_p_z_r_carts = r_carts.copy()
    diff_p_z_r_carts[:, 2] += diff_h
    ao_matrix_diff_p_z = compute_AOs(aos_data, diff_p_z_r_carts)
    diff_m_z_r_carts = r_carts.copy()
    diff_m_z_r_carts[:, 2] -= diff_h
    ao_matrix_diff_m_z = compute_AOs(aos_data, diff_m_z_r_carts)

    ao_matrix_grad_x = (ao_matrix_diff_p_x - ao_matrix_diff_m_x) / (2.0 * diff_h)
    ao_matrix_grad_y = (ao_matrix_diff_p_y - ao_matrix_diff_m_y) / (2.0 * diff_h)
    ao_matrix_grad_z = (ao_matrix_diff_p_z - ao_matrix_diff_m_z) / (2.0 * diff_h)

    return ao_matrix_grad_x, ao_matrix_grad_y, ao_matrix_grad_z


def compute_AOs_laplacian(aos_data: AOs_base_data, r_carts: jax.Array) -> jax.Array:
    """Return analytic Laplacians of contracted atomic orbitals.

    Dispatches to Cartesian or real-spherical implementations; returns float64 JAX arrays
    (ensure ``jax_enable_x64=True``).

    For periodic ``aos_data``, the kernel output is collapsed across the
    image axis by a post-kernel reshape-sum; molecular instances
    (``_n_image == 1``) take a trace-time short-circuit so the emitted
    HLO is byte-identical to the pre-PBC kernel. See :py:func:`compute_AOs`
    for the rationale.

    Args:
        aos_data: ``AOs_cart_gto_data`` or ``AOs_sphe_gto_data`` describing centers, primitive exponents/coefficients,
            angular data, and contraction mapping (run ``sanity_check()`` beforehand).
        r_carts (jax.Array): Electron Cartesian coordinates, shape ``(N_e, 3)`` (Bohr). Casts to ``float64``
            internally via ``jnp.asarray``.

    Returns:
        jax.Array: Laplacians of all contracted AOs, shape ``(num_ao, N_e)``.

    Raises:
        NotImplementedError: If ``aos_data`` is not Cartesian or spherical.
    """
    # Data-type-agnostic dispatcher: forwards to the per-leaf
    # ``_compute_AOs_laplacian`` method. Each leaf owns wrap + kernel and (for
    # periodic pgto leaves) the post-kernel image-axis reduction, so this
    # wrapper never reaches into the ``_n_image`` static.
    return aos_data._compute_AOs_laplacian(r_carts)


@jit
def _compute_AOs_laplacian_autodiff(aos_data: AOs_base_data, r_carts: jnpt.ArrayLike) -> jax.Array:
    """Compute laplacians of the give AOs at r_carts.

    See :py:func:`compute_AOs_laplacian`.

    """
    # Forward r_carts as-is (Principle 3a -- no parameter rebind). compute_AOs's
    # inner kernels reconstruct r-R in caller-supplied precision and downcast to
    # the ao_eval zone at the use site; the hessian inherits that dtype.
    # not very fast, but it works.
    ao_matrix_hessian = hessian(compute_AOs, argnums=1)(aos_data, r_carts)
    ao_matrix_laplacian = jnp.einsum("m i i u i u -> mi", ao_matrix_hessian)

    return ao_matrix_laplacian


def _compute_AOs_laplacian_debug(aos_data: AOs_base_data, r_carts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute laplacians of the give AOs at r_carts.

    The method is for computing the laplacians of the given atomic orbital at r_carts
    using the FDM method for debugging purposes.

    Args:
        aos_data (AOs_base_data): an instance of AOs_base_data
        r_carts (npt.NDArray[np.float64]): Cartesian coordinates of electrons (dim: N_e, 3)

    Returns:
        npt.NDArray[np.float64]:
            Array containing laplacians of the AOs at r_carts. The dim. is (num_ao, N_e)

    """
    # Laplacians of AOs (numerical, 4th-order central FD)
    # f''(x) ~= (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12h^2)
    # Larger h is viable with the 4th-order stencil (O(h^4) truncation).
    diff_h = 1.0e-3

    ao_matrix = compute_AOs(aos_data, r_carts)

    def _shifted(dim: int, mult: int) -> jax.Array:
        rs = r_carts.copy()
        rs[:, dim] += mult * diff_h
        return compute_AOs(aos_data, rs)

    def _grad2_along(dim: int) -> jax.Array:
        f_p1 = _shifted(dim, +1)
        f_p2 = _shifted(dim, +2)
        f_m1 = _shifted(dim, -1)
        f_m2 = _shifted(dim, -2)
        return (-f_p2 + 16 * f_p1 - 30 * ao_matrix + 16 * f_m1 - f_m2) / (12 * diff_h**2)

    ao_matrix_grad2_x = _grad2_along(0)
    ao_matrix_grad2_y = _grad2_along(1)
    ao_matrix_grad2_z = _grad2_along(2)

    ao_matrix_laplacian = ao_matrix_grad2_x + ao_matrix_grad2_y + ao_matrix_grad2_z

    # ``compute_AOs`` returns the molecular-rank ``num_ao`` (the per-leaf
    # methods, and for periodic pgto leaves their internal image reduction,
    # collapse the kernel rank down to ``num_ao``). Use that so the check
    # tracks the production dispatcher.
    expected_num_ao = aos_data.num_ao
    if ao_matrix_laplacian.shape != (expected_num_ao, len(r_carts)):
        logger.error(
            f"ao_matrix_laplacian.shape = {ao_matrix_laplacian.shape} is \
                inconsistent with the expected one = {expected_num_ao, len(r_carts)}"
        )
        raise ValueError

    return ao_matrix_laplacian


def compute_AOs_value_grad_lap(
    aos_data: AOs_base_data, r_carts: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fused evaluation of AO values, Cartesian gradients, and Laplacians.

    Returns ``(val, gx, gy, gz, lap)``. ``val`` is in the ``ao_eval`` zone
    (fp32 in mixed mode, fp64 in full mode); ``gx``, ``gy``, ``gz``, and
    ``lap`` are in the ``ao_grad_lap`` zone (fp64 in both modes). All
    arrays have shape ``(num_ao, N_e)``.

    Use this when value, gradient, and Laplacian are all needed at the
    same call site (kinetic energy, streaming-state initialisation /
    advance). For value-only, grad-only, or lap-only call sites, prefer
    the standalone APIs (``compute_AOs`` / ``compute_AOs_grad`` /
    ``compute_AOs_laplacian``) -- JAX DCE does not reliably eliminate the
    unused outputs of this function across its ``@jit`` boundary.

    Mixed-precision design: shared body (``exp(-Z r^2)``, polynomial /
    ``S_l_m``, ``phi`` / ``pref``) is computed once in ``ao_grad_lap``
    (fp64); ``val`` is downcast to ``ao_eval`` only at the segment-sum
    site. ``gx`` / ``gy`` / ``gz`` / ``lap`` are kept in fp64 to protect
    the laplacian's ``4 Z^2 r^2 - 6 Z`` cancellation. The val downcast
    matches the standalone ``compute_AOs`` zone semantics so that
    forward (det_eval) and grad/lap (det_grad_lap) consumers see val of
    the same fp32 quality and remain variationally consistent.

    Args:
        aos_data: ``AOs_cart_gto_data`` or ``AOs_sphe_gto_data`` describing primitive
            parameters, angular info, contraction mapping, and centers (run
            ``sanity_check()`` beforehand).
        r_carts (jax.Array): Electron Cartesian coordinates, shape ``(N_e, 3)``
            (Bohr). Forwarded as-is (Principle 3a); the kernels reconstruct
            ``r - R`` in fp64 internally to avoid catastrophic cancellation.

    Periodic ``aos_data`` (built via
    :py:func:`_apply_image_expansion`) emits each of
    the five outputs at kernel-rank ``(n_image * num_ao, N_e)``; a single
    post-kernel reshape-sum (internal to the periodic pgto leaf) collapses
    the image axis on every output so downstream consumers see molecular
    shape ``(num_ao, N_e)``. Molecular instances take a trace-time
    short-circuit, so the emitted HLO is byte-identical to the pre-PBC kernel.

    Returns:
        tuple: ``(val, gx, gy, gz, lap)``, each of shape ``(num_ao, N_e)``.

    Raises:
        NotImplementedError: If ``aos_data`` is neither Cartesian nor spherical.
    """
    # Data-type-agnostic dispatcher: forwards to the per-leaf
    # ``_compute_AOs_value_grad_lap`` method. Each leaf owns wrap + kernel and
    # (for periodic pgto leaves) the post-kernel image-axis reduction, so this
    # wrapper never reaches into the ``_n_image`` static.
    return aos_data._compute_AOs_value_grad_lap(r_carts)


def compute_overlap_matrix(aos_data: AOs_base_data) -> jax.Array:
    r"""Compute the AO overlap matrix.

    Returns an overlap matrix with shape
    ``(num_ao_molecular_view, num_ao_molecular_view)`` regardless of
    ``pbc_flag``.

    Raises:
        NotImplementedError: For a molecular (open-boundary) GTO composite,
            which has no cell to integrate over.
    """
    # Zone is ``det_eval`` (not ``ao_eval``): the overlap matrix is never on
    # the AO-eval forward hot loop. Known consumers are det/SVD math
    # (Loewdin S^{1/2}, S-metric polar retraction, geminal AO->MO conversion,
    # S-orthonormality checks); ``det_eval`` is fp64 in both modes and matches
    # what those consumers expect. Tagging this wrapper ``ao_eval`` would drop
    # S to fp32 under mixed precision, and a downstream fp64 re-cast cannot
    # recover the loss.
    #
    # AO primitive values may be stored at the forward AO precision, but the
    # analytic overlap arithmetic and its independent lattice-tail cutoff use
    # ``det_eval`` precision. Rounded primitive values still define a valid
    # basis and hence a positive-semidefinite Gram matrix; their storage width
    # is not a justification for a looser lattice truncation.
    #
    # Complex promotion: a complex Bloch overlap
    # (from a non-Gamma Bloch-pGTO leaf) lifts to ``complex128`` so ``Im``
    # is not silently discarded by a hard-coded real-zone cast.
    inner = aos_data._compute_overlap_matrix()
    return cast_to_zone(inner, "det_eval")

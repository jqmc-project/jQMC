r"""Canonical Model Periodic Coulomb (MPC) finite-size correction.

MPC is applied as a passive observable: walkers are propagated under Ewald and the correction :math:`\Delta V = \langle V^\text{MPC}_{ee} - V^\text{Ewald}_{ee} \rangle` is accumulated on the same configurations (:py:func:`compute_mpc_correction_ee`). This is the QMCPACK ``physical="no"`` convention and the CASINO ``interaction = ewald mpc`` recommendation. Under this convention forces and parameter gradients remain on the Ewald channel; this module exposes no derivative entry points.

The explicit MPC two-body interaction uses the minimum-image :math:`1/r`; there is no Ewald split parameter. The reciprocal-space cutoff ``cutoff_ha`` (CASINO ``mpc_cutoff``, QMCPACK ``cutoff``) defaults to 30 a.u.; converge by doubling it and checking that the correction stops moving. ``V_LR`` is tabulated on a real-space grid and evaluated by periodic tricubic B-spline interpolation.

Detailed derivations of the reciprocal-space construction, density approximation, grid/tabulation choices, and interpolation are work in progress and are not reproduced here. See the public API docstrings for the supported inputs, outputs, and usage.
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

from dataclasses import dataclass
import time
from logging import getLogger
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import struct

from ._precision import cast_to_zone
from ._setting import MPC_GRID_MIN_POINTS, MPC_GRID_POINTS_PER_MAX_FREQUENCY
from .atomic_orbital import compute_AOs
from .coulomb_potential import Ewald_data, compute_ewald_potential_ee
from .molecular_orbital import MOs_data
from .orbital_tiling import TiledOrbitalSet_data, compute_tiled_orbitals
from .structure import Structure_data, _det_3x3, _inv_3x3, _minimum_image_displacement, _shortest_lattice_vector_length

if TYPE_CHECKING:
    # Annotation only: importing mpi4py.MPI initializes MPI (issue #778).
    from mpi4py import MPI

    from .hamiltonians import Hamiltonian_data

logger = getLogger("jqmc").getChild(__name__)


# ---------------------------------------------------------------------------
# Lattice geometry
# ---------------------------------------------------------------------------
def wigner_seitz_radius(cell: npt.NDArray[np.float64], n_search: int = 2) -> float:
    r"""Radius of the largest sphere inscribed in the Wigner-Seitz cell.

    The inscribed radius is half the shortest non-zero lattice-vector length,
    ``min_{T != 0} |T| / 2``. The result depends only on the lattice, not on
    the choice of cell basis.

    Static helper: takes a concrete ``np.ndarray``, returns a Python float.
    ``L`` is a property of the lattice and must be resolved once outside any
    JAX trace.

    Args:
        cell: Lattice vectors as rows, ``(3, 3)`` Bohr.
        n_search: Minimum half-width of the integer search range.

    Returns:
        float: Inscribed-sphere radius ``L`` (Bohr).
    """
    # Same arithmetic as before; the search lives in
    # ``structure._shortest_lattice_vector_length`` so the Structure_data WS
    # radius (DTN cutoffs, issue #68) shares it.
    return 0.5 * _shortest_lattice_vector_length(cell, n_search)


def _validate_mpc_cutoff(cutoff_ha: object) -> float:
    """Return ``cutoff_ha`` as a float, or raise if it is not a positive finite real scalar.

    ``mpc_cutoff`` keeps the reciprocal vectors with ``G^2/2 <= cutoff_ha``.
    A zero cutoff keeps only ``G = 0``, so none of the ``G != 0`` long-range
    model correction enters and the result is not an MPC number. Negative and
    non-finite values have no meaning. Booleans are rejected rather than read
    as 0 or 1. ``jqmc_tool``'s MPC extrapolation enforces the same contract
    on stored cutoffs.
    """
    value = np.asarray(cutoff_ha)
    if (
        isinstance(cutoff_ha, (bool, np.bool_))
        or value.ndim != 0
        or value.dtype.kind not in "fiu"
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"mpc_cutoff must be a positive finite real scalar (Ha); got {cutoff_ha!r}.")
    return float(value)


def resolve_grid_shape(cell: npt.NDArray[np.float64], cutoff_ha: float) -> tuple[int, int, int]:
    r"""FFT / interpolation grid shape for a reciprocal cutoff ``cutoff_ha``.

    The retained reciprocal set is ``G^2/2 <= Ecut`` (QMCPACK ``init_gvecs``),
    i.e. ``|G| <= sqrt(2 Ecut)``. The dual relation ``G . a_d = 2*pi*m_d``
    bounds each integer frequency by ``ceil(|G|_max * |a_d| / (2*pi))``,
    including cancellation between reciprocal basis vectors in skew cells.
    Thus ``2 m_d + 1`` points suffice to *represent* the band-limited
    fields. The grid returned here is deliberately finer:

    * ``MPC_GRID_POINTS_PER_MAX_FREQUENCY * m_d`` points per axis (QMCPACK
      ``init_spline``: ``SplineDim = 4 MaxDim``) -- the tricubic B-spline
      of :py:func:`compute_mpc_potential_ee` is only accurate enough near
      the nuclei when the tabulation oversamples the highest retained
      frequency by this factor;
    * never fewer than ``MPC_GRID_MIN_POINTS`` (QMCPACK ``init_f_G``:
      ``N = max(64, 2 MaxDim + 1)``), which also keeps the ``g_G`` FFT
      extrapolation well resolved for small cells.

    Args:
        cell: Lattice vectors as rows, ``(3, 3)`` Bohr, concrete.
        cutoff_ha: Reciprocal-space kinetic-energy cutoff (Hartree).

    Returns:
        tuple[int, int, int]: Grid shape, each entry even (FFT-friendly).
    """
    cell = np.asarray(cell, dtype=np.float64)
    np.linalg.inv(cell)  # Preserve singular-cell rejection; the bound uses the direct lattice.
    g_max = np.sqrt(2.0 * float(cutoff_ha))
    shape = []
    for d in range(3):
        a_norm = float(np.linalg.norm(cell[d]))
        m = int(np.ceil(g_max * a_norm / (2.0 * np.pi)))
        n = max(int(MPC_GRID_MIN_POINTS), int(MPC_GRID_POINTS_PER_MAX_FREQUENCY) * m)
        shape.append(n + (n % 2))  # round up to even
    return (shape[0], shape[1], shape[2])


# ---------------------------------------------------------------------------
# Hood g + h splitting of f(r) = 1 / |r_MI|
# ---------------------------------------------------------------------------
def _hood_y(r: jax.Array, L: float) -> jax.Array:
    r"""``y(r) = -r^2 / (2 L^3) + 3 / (2 L)`` (CASINO Eq. 192)."""
    return -(r * r) / (2.0 * L**3) + 3.0 / (2.0 * L)


def _hood_g_real(r: jax.Array, L: float) -> jax.Array:
    r"""Smooth part ``g(r) = y(r)`` for ``r < L``, ``1/r`` for ``r >= L``.

    ``C^1`` at ``r = L`` (``y(L) = 1/L``, ``y'(L) = -1/L^2``), which is
    what makes the FFT of ``g`` converge quickly enough to extrapolate.
    """
    r_safe = jnp.where(r > 0.0, r, 1.0)
    return jnp.where(r < L, _hood_y(r, L), 1.0 / r_safe)


def _hood_h_G(g_norm: jax.Array, L: float, volume: jax.Array) -> jax.Array:
    r"""Closed-form Fourier transform of the compact part ``h``.

    ``h(r) = 1/r - y(r)`` inside ``r < L`` and zero outside, so

    .. math::
       h_G = \frac{4\pi}{\Omega G}\int_0^L
             \Big[1 + \frac{r^3}{2L^3} - \frac{3r}{2L}\Big]\sin(Gr)\,dr

    which integrates to the expression below with ``x = G L``. The
    ``G -> 0`` limit is ``h_0 = 2 pi L^2 / (5 Omega)`` (CASINO Eq. 194)
    and is substituted explicitly at ``G = 0``.

    No small-``x`` instability arises in practice: the shortest non-zero
    ``G`` satisfies ``|G| >= 2 pi / L_cell`` while ``L <= L_cell / 2``
    for a cubic cell, so ``x >= pi``.
    """
    G = jnp.where(g_norm > 0.0, g_norm, 1.0)
    x = G * L

    sin_x = jnp.sin(x)
    cos_x = jnp.cos(x)

    I0 = (1.0 - cos_x) / G
    I1 = sin_x / G**2 - L * cos_x / G
    I3 = -(L**3) * cos_x / G + 3.0 * L**2 * sin_x / G**2 + 6.0 * L * cos_x / G**3 - 6.0 * sin_x / G**4

    integral = I0 + I3 / (2.0 * L**3) - (3.0 / (2.0 * L)) * I1
    h_G = (4.0 * jnp.pi / (volume * G)) * integral

    h_0 = 2.0 * jnp.pi * L**2 / (5.0 * volume)
    return jnp.where(g_norm > 0.0, h_G, h_0)


def _fft_int_freqs(n: int) -> np.ndarray:
    """Signed FFT frequency indices ``[0, 1, ..., -2, -1]``, EXACTLY.

    The obvious ``fft.fftfreq(n) * n`` is a float round-trip and does not
    land on integers: at ``n = 94`` it returns ``2.9999999999999996`` for
    index 3, and ``.astype(int)`` truncates that to 2. Sub-sampling a
    finer grid with such indices silently reads the wrong frequency --
    it cost a factor-60 error in the MPC correction at one cell size,
    with the coarse grid's ``g_G`` at frequency 3 being compared against
    the fine grid's at frequency 2 (opposite signs, so the extrapolation
    diverged rather than merely drifting).

    Affected ``n`` are those where ``k / n * n`` is inexact -- 56, 68 and
    94 all are; 40 is not -- so the failure is silent and
    size-dependent, which is exactly the kind that survives testing on
    one cell and appears on another.
    """
    return np.concatenate([np.arange(0, (n + 1) // 2), np.arange(-(n // 2), 0)])


def _reciprocal_vectors(cell: jax.Array, grid_shape: tuple[int, int, int]) -> tuple[jax.Array, jax.Array]:
    """Return ``(|G|, |G|^2)`` on the FFT frequency grid, ``G = 0`` entry zero."""
    n1, n2, n3 = (int(x) for x in grid_shape)
    recip = 2.0 * jnp.pi * _inv_3x3(cell).T
    m1 = jnp.fft.fftfreq(n1) * n1
    m2 = jnp.fft.fftfreq(n2) * n2
    m3 = jnp.fft.fftfreq(n3) * n3
    M1, M2, M3 = jnp.meshgrid(m1, m2, m3, indexing="ij")
    Gx = M1 * recip[0, 0] + M2 * recip[1, 0] + M3 * recip[2, 0]
    Gy = M1 * recip[0, 1] + M2 * recip[1, 1] + M3 * recip[2, 1]
    Gz = M1 * recip[0, 2] + M2 * recip[1, 2] + M3 * recip[2, 2]
    g2 = Gx * Gx + Gy * Gy + Gz * Gz
    return jnp.sqrt(g2), g2


#: Node budget per minimum-image chunk. ``_minimum_image_displacement``
#: materialises a ``(chunk, (2 n_max + 1)^3, 3)`` candidate tensor, so the
#: peak allocation is ``chunk * 27 * 3 * 8`` bytes at ``n_max = 1``
#: (~440 MB here). Without chunking the 4N extrapolation grid at
#: ``cutoff = 120`` reaches 312^3 = 30M nodes and the candidate tensor
#: alone is ~19 GB, which OOMs a 15 GB box.
_MIN_IMAGE_CHUNK = 250_000

#: Relative safety margin on the ``|T| <= 2 R`` image-pruning bound of
#: :py:func:`_cell_grid_min_image_r`. Pruning is exact for any margin
#: ``>= 0`` in exact arithmetic; the margin absorbs the round-off of the
#: wrap and of the norms, and an over-kept image cannot change the minimum.
_MIN_IMAGE_PRUNE_MARGIN = 1.0e-6

#: Sleep between completion checks while non-root ranks wait for rank 0's
#: MPC build (see :py:func:`build_mpc_data`). Negligible against a build
#: of seconds to minutes.
_BROADCAST_IDLE_POLL_SEC = 0.05

#: Grid points per ``compute_AOs`` call when tabulating the density.
#: ``compute_AOs`` peaks at roughly 140 kB per point regardless of how
#: few AOs come back, so this is the knob that decides whether a large
#: cell fits in memory at all.
_AO_CHUNK = 4096


@jax.jit
def _min_image_r_sweep(disp: jax.Array, T_images: jax.Array) -> jax.Array:
    """``min_T |disp + T|`` over the image translations ``T_images``, one compiled pass.

    The image sweep is a ``lax.fori_loop`` carrying the running minimum, so
    XLA fuses the shift, square, sum and minimum into one pass per image
    instead of dispatching four eager operations per image per chunk (a
    Python loop makes ~23k eager dispatches on the 4N grid of a 64^3 cell
    at ``n_max = 3``). The expression is the one such a loop evaluates, and at
    the production chunk lengths the result is bitwise identical to it
    (measured on the 64^3/128^3/256^3 grids of jqmc-example07). XLA rounds
    the three-term ``sum(d * d)`` of a fused kernel and of an eager op
    differently for some short arrays (one ulp at a few hundred rows), so
    the guarantee is stated at the fp64 ulp level. The
    ``(chunk, 3) @ (3, 3)`` products that produce ``disp`` stay outside
    for the same reason.
    """

    def _sweep(t, best):
        d = disp + T_images[t]
        return jnp.minimum(best, jnp.sum(d * d, axis=-1))

    best = jax.lax.fori_loop(0, T_images.shape[0], _sweep, jnp.full((disp.shape[0],), jnp.inf, dtype=jnp.float64))
    return jnp.sqrt(best)


def _cell_grid_min_image_r(cell: jax.Array, n_fft: tuple[int, int, int], n_max: int) -> jax.Array:
    """Minimum-image ``|r|`` at every node of an ``n_fft`` grid over the cell.

    Chunked over the flattened node list -- see ``_MIN_IMAGE_CHUNK``; the
    image sweep of each chunk is one call of the compiled
    :py:func:`_min_image_r_sweep`.
    Coordinates are generated *inside* each chunk from flat indices rather
    than built once for the whole grid: at the 4N extrapolation size
    (312^3 = 30M nodes) a materialised ``frac`` and ``disp`` are 730 MB
    each, plus 243 MB per meshgrid axis. Only ``|r|`` is wanted, so the
    images are swept with a running minimum instead of materialising the
    ``(chunk, N_img, 3)`` candidate tensor of the shared
    ``_minimum_image_displacement`` -- ``O(chunk)`` footprint.
    """
    n1, n2, n3 = (int(x) for x in n_fft)
    inv_cell = _inv_3x3(cell)
    n_nodes = n1 * n2 * n3

    # Image translations, built once: the ``(2 n_max + 1)^3`` box, minus
    # the images that provably cannot be the nearest. After the wrap below
    # every ``disp`` lies in the half-cell parallelepiped, so
    # ``|disp| <= R`` with ``R`` its largest corner norm, and the ``T = 0``
    # candidate is ``<= R``. An image with ``|T| > 2 R`` gives
    # ``|disp + T| >= |T| - |disp| > R`` and is strictly worse. Dropping it
    # leaves the minimum -- and, since ``min`` is exact and each kept
    # candidate is computed as before, every bit of ``|r|`` -- unchanged;
    # the relative margin only keeps a few extra candidates. For the fcc
    # cell of jqmc-example07 this keeps 87 of the 343 images.
    rng = np.arange(-n_max, n_max + 1, dtype=np.float64)
    ia, ib, ic = np.meshgrid(rng, rng, rng, indexing="ij")
    n_grid_int = np.stack([ia.ravel(), ib.ravel(), ic.ravel()], axis=-1)
    cell_np = np.asarray(cell, dtype=np.float64)
    corners = np.stack(np.meshgrid([-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], indexing="ij"), axis=-1).reshape(-1, 3)
    r_wrapped_max = float(np.max(np.linalg.norm(corners @ cell_np, axis=1)))
    keep = np.linalg.norm(n_grid_int @ cell_np, axis=1) <= 2.0 * r_wrapped_max * (1.0 + _MIN_IMAGE_PRUNE_MARGIN)
    T_images = jnp.asarray(n_grid_int[keep], dtype=jnp.float64) @ cell  # (N_img, 3)

    pieces = []
    for start in range(0, n_nodes, _MIN_IMAGE_CHUNK):
        stop = min(start + _MIN_IMAGE_CHUNK, n_nodes)
        flat = jnp.arange(start, stop, dtype=jnp.int64)
        frac = jnp.stack(
            [
                (flat // (n2 * n3)).astype(jnp.float64) / n1,
                ((flat // n3) % n2).astype(jnp.float64) / n2,
                (flat % n3).astype(jnp.float64) / n3,
            ],
            axis=-1,
        )
        disp = frac @ cell
        # Wrap once to the parallelepiped so the image box brackets the
        # nearest copy (same pre-step as _minimum_image_displacement).
        disp_frac = disp @ inv_cell
        disp = (disp_frac - jnp.round(disp_frac)) @ cell
        pieces.append(_min_image_r_sweep(disp, T_images))

    r = pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces, axis=0)
    return r.reshape(n1, n2, n3)


def _g_G_on_grid(cell: jax.Array, n_fft: tuple[int, int, int], L: float, n_max: int) -> jax.Array:
    r"""``g_G`` by FFT on an ``n_fft`` grid, returned on that grid's frequencies.

    ``f_G = Omega^{-1} \int_WS f(r) e^{iGr} dr`` discretises to
    ``(1/N_grid) sum_nodes g(r) e^{iGr}``. ``g`` is real and even under
    minimum image, so ``g_G`` is real and the sign of the exponent is
    immaterial -- ``jnp.fft.fftn`` (which carries ``e^{-iGr}``) is used
    directly and the real part taken.
    """
    r_grid = _cell_grid_min_image_r(cell, n_fft, n_max)
    g_real = _hood_g_real(r_grid, L)
    n_grid = int(n_fft[0]) * int(n_fft[1]) * int(n_fft[2])
    return jnp.real(jnp.fft.fftn(g_real)) / n_grid


def _extrapolate_g_G(
    cell: jax.Array,
    grid_shape: tuple[int, int, int],
    L: float,
    n_max: int,
) -> tuple[jax.Array, float]:
    r"""``g_G`` extrapolated to the infinite-grid limit.

    Evaluates ``g_G`` on ``N``, ``2N`` and ``4N`` grids (QMCPACK
    ``init_f_G``) and extrapolates in ``x = 1/N``:

    * *linear* -- two finest points, ``g = g_inf + a x``
    * *quadratic* -- all three, ``g = g_inf + a x + b x^2``

    The quadratic value is returned; the largest absolute difference
    between the two is returned alongside as a convergence diagnostic
    (CASINO prints the same "worst discrepancy"). Extrapolating in
    ``1/N`` without assuming a convergence order is deliberate -- the
    order for the ``C^1`` function ``g`` is not obvious a priori, so the
    discrepancy is reported rather than a rate being presumed.

    The coarser grids' coefficients are sub-sampled onto the target
    frequency set by index, which is exact: the ``2N`` grid contains
    every frequency of the ``N`` grid.
    """
    n1, n2, n3 = (int(x) for x in grid_shape)

    def sub_sample(arr: jax.Array, factor: int) -> jax.Array:
        """Pick the frequencies of the target grid out of a ``factor * N`` grid."""
        i1 = _fft_int_freqs(n1) % (factor * n1)
        i2 = _fft_int_freqs(n2) % (factor * n2)
        i3 = _fft_int_freqs(n3) % (factor * n3)
        return arr[jnp.ix_(i1, i2, i3)]

    def timed_level(factor: int) -> jax.Array:
        """``g_G`` on the ``factor * N`` grid, with one progress line per level."""
        shape = (factor * n1, factor * n2, factor * n3)
        t0 = time.perf_counter()
        g = _g_G_on_grid(cell, shape, L, n_max)
        g = g if factor == 1 else sub_sample(g, factor)
        g.block_until_ready()
        logger.info(f"  MPC setup: g_G on the {factor}N grid {shape} done in {time.perf_counter() - t0:.1f} sec.")
        return g

    g1 = timed_level(1)
    g2 = timed_level(2)
    g4 = timed_level(4)

    # x = 1/N in units of 1/n1: x1 = 1, x2 = 1/2, x4 = 1/4.
    x1, x2, x4 = 1.0, 0.5, 0.25

    # Linear through the two finest.
    lin = g4 + (g4 - g2) * (0.0 - x4) / (x4 - x2)

    # Quadratic through all three (Lagrange at x = 0).
    l1 = ((0.0 - x2) * (0.0 - x4)) / ((x1 - x2) * (x1 - x4))
    l2 = ((0.0 - x1) * (0.0 - x4)) / ((x2 - x1) * (x2 - x4))
    l4 = ((0.0 - x1) * (0.0 - x2)) / ((x4 - x1) * (x4 - x2))
    quad = l1 * g1 + l2 * g2 + l4 * g4

    discrepancy = float(jnp.max(jnp.abs(quad - lin)))
    return quad, discrepancy


# ---------------------------------------------------------------------------
# MPC data
# ---------------------------------------------------------------------------
@struct.dataclass
class MPC_data:
    r"""Precomputed tables for the canonical MPC e-e observable.

    Built once per (structure, trial density, cutoff) by
    :py:func:`build_mpc_data` and consumed by
    :py:func:`compute_mpc_potential_ee`. Under the ``physical = no``
    convention this never enters the propagated Hamiltonian, so there is
    no requirement that ``jax.grad`` flow through it -- the fields are
    plain arrays and the module exposes no derivative entry points.
    """

    # Only the three lattice quantities the evaluation actually needs are
    # kept, as plain leaves -- NOT the whole ``Structure_data``. Holding it
    # as ``pytree_node=False`` metadata makes JAX compare it with ``==``
    # when it decides whether a cached jit trace can be reused, and
    # ``Structure_data.__eq__`` raises on the ambiguous truth value of its
    # arrays. That surfaces only on the *second* distinct ``MPC_data`` to
    # cross a jit boundary in one process, i.e. exactly on resume and on
    # every wavefunction-optimisation iteration.
    #: Lattice vectors as rows, ``(3, 3)`` Bohr.
    cell: jax.Array = struct.field(pytree_node=True)
    #: Grid origin (nuclear centroid), ``(3,)`` Bohr. Anchoring here rather
    #: than at the cell corner keeps the tabulated fields translationally
    #: covariant with the nuclei.
    origin: jax.Array = struct.field(pytree_node=True)
    #: Static half-width of the minimum-image enumeration box.
    n_max: int = struct.field(pytree_node=False)
    #: Static ``(N1, N2, N3)`` FFT grid shape.
    grid_shape: tuple = struct.field(pytree_node=False)
    #: Long-range one-body potential at the grid nodes, ``(N1, N2, N3)`` Ha.
    #: Kept for diagnostics and tests; the evaluators read ``v_lr_coef``.
    v_lr_grid: jax.Array = struct.field(pytree_node=True, default=None)
    #: Periodic tricubic B-spline coefficients of ``v_lr_grid``, same shape.
    #: This -- not the node values -- is what :py:func:`evaluate_v_lr`
    #: interpolates.
    v_lr_coef: jax.Array = struct.field(pytree_node=True, default=None)
    #: Configuration-independent constant, Ha.
    v_const: jax.Array = struct.field(pytree_node=True, default=None)
    #: Inscribed WS-sphere radius ``L`` (Bohr), lattice property.
    L: float = struct.field(pytree_node=False, default=0.0)
    #: Reciprocal cutoff the tables were built at (Ha).
    cutoff_ha: float = struct.field(pytree_node=False, default=0.0)
    #: ``|quadratic - linear|`` of the ``g_G`` extrapolation; convergence
    #: diagnostic, not used in evaluation.
    f_G_discrepancy: float = struct.field(pytree_node=False, default=0.0)


def build_mpc_data(
    hamiltonian_data: Hamiltonian_data,
    cutoff_ha: float = 30.0,
    grid: str | tuple[int, int, int] = "auto",
    frozen_overlap: tuple[jax.Array, jax.Array] | None = None,
    comm: MPI.Intracomm | None = None,
) -> MPC_data:
    r"""Build the canonical MPC tables from a trial wavefunction + cell.

    Constructs the tables consumed by :py:func:`compute_mpc_potential_ee` and :py:func:`compute_mpc_correction_ee`. The detailed reciprocal-space and interpolation construction is work in progress and is not reproduced here.

    Args:
        hamiltonian_data: Periodic problem definition (trial WF + cell).
        cutoff_ha: Reciprocal kinetic-energy cutoff, Hartree. CASINO
            ``mpc_cutoff`` / QMCPACK ``cutoff``; both default to 30.
        grid: ``"auto"`` resolves from ``cutoff_ha``, or an explicit
            ``(N1, N2, N3)``. An explicit grid only sets the tabulation and
            must still be fine enough for the retained ``G`` set.
        frozen_overlap: Forwarded to the natural-orbital density.
        comm: Optional MPI communicator whose ranks all hold the same
            ``hamiltonian_data`` (a sampler's per-twist sampling comm). With
            more than one rank, the tables are built once on rank 0 and
            broadcast, so every rank receives rank 0's arrays bit for bit
            instead of repeating the build. Collective: every rank of
            ``comm`` must call this. ``None`` or a one-rank comm builds
            locally.

    Returns:
        MPC_data: Tables ready for :py:func:`compute_mpc_potential_ee`.

    Raises:
        ValueError: If ``cutoff_ha`` is not a positive finite real scalar.
            Every rank of ``comm`` checks its own argument before the build.
    """
    _validate_mpc_cutoff(cutoff_ha)
    if comm is None or comm.Get_size() == 1:
        return _build_mpc_data_local(hamiltonian_data, cutoff_ha, grid, frozen_overlap)

    t0 = time.perf_counter()
    rank = comm.Get_rank()
    logger.info(f"MPC setup: building the tables on rank 0 and broadcasting them to {comm.Get_size()} ranks.")
    built: MPC_data | None = None
    header = None
    failure: Exception | None = None
    if rank == 0:
        try:
            built = _build_mpc_data_local(hamiltonian_data, cutoff_ha, grid, frozen_overlap)
            header = {
                "cell": np.asarray(built.cell),
                "origin": np.asarray(built.origin),
                "n_max": built.n_max,
                "grid_shape": built.grid_shape,
                "v_const": np.asarray(built.v_const),
                "L": built.L,
                "cutoff_ha": built.cutoff_ha,
                "f_G_discrepancy": built.f_G_discrepancy,
            }
        except Exception as exc:
            failure = exc
            header = {"error": f"{type(exc).__name__}: {exc}"}
        comm.Ibarrier().Wait()
    else:
        # Wait idle rather than inside a blocking collective: MPI
        # implementations busy-poll there, and 127 spinning cores cut
        # rank 0's build speed several-fold (measured on jqmc-example07,
        # 128 ranks: 4N g_G 73 s spinning vs 12 s single-process).
        request = comm.Ibarrier()
        while not request.Test():
            time.sleep(_BROADCAST_IDLE_POLL_SEC)
    header = comm.bcast(header, root=0)
    if failure is not None:
        raise failure
    if "error" in header:
        raise RuntimeError(f"build_mpc_data failed on rank 0: {header['error']}")

    # The two grids go as raw float64 buffers (``Bcast``), not pickles: at
    # large cutoffs they exceed the 2 GiB pickle limit of ``bcast``.
    grids = {}
    for name in ("v_lr_grid", "v_lr_coef"):
        buf = np.ascontiguousarray(getattr(built, name), dtype=np.float64) if rank == 0 else None
        if buf is None:
            buf = np.empty(tuple(header["grid_shape"]), dtype=np.float64)
        comm.Bcast(buf, root=0)
        grids[name] = buf
    if rank == 0:
        result = built
    else:
        result = MPC_data(
            cell=jnp.asarray(header["cell"], dtype=jnp.float64),
            origin=jnp.asarray(header["origin"], dtype=jnp.float64),
            n_max=int(header["n_max"]),
            grid_shape=tuple(header["grid_shape"]),
            v_lr_grid=jnp.asarray(grids["v_lr_grid"], dtype=jnp.float64),
            v_lr_coef=jnp.asarray(grids["v_lr_coef"], dtype=jnp.float64),
            v_const=jnp.asarray(header["v_const"], dtype=jnp.float64),
            L=float(header["L"]),
            cutoff_ha=float(header["cutoff_ha"]),
            f_G_discrepancy=float(header["f_G_discrepancy"]),
        )
    logger.info(f"MPC setup: done in {time.perf_counter() - t0:.1f} sec.")
    return result


def _build_mpc_data_local(
    hamiltonian_data: Hamiltonian_data,
    cutoff_ha: float,
    grid: str | tuple[int, int, int],
    frozen_overlap: tuple[jax.Array, jax.Array] | None,
) -> MPC_data:
    """Build the MPC tables in this process; see :py:func:`build_mpc_data`."""
    coulomb_potential_data = getattr(hamiltonian_data, "coulomb_potential_data", None)
    if coulomb_potential_data is not None and getattr(coulomb_potential_data, "structure_data", None) is not None:
        structure_data = coulomb_potential_data.structure_data
    else:
        structure_data = hamiltonian_data.structure_data
    if not structure_data.pbc_flag:
        raise ValueError("build_mpc_data requires a periodic (pbc_flag=True) structure.")

    cell_np = np.asarray(structure_data.cell, dtype=np.float64)
    L = wigner_seitz_radius(cell_np)

    if isinstance(grid, str):
        if grid != "auto":
            raise ValueError(f"grid must be 'auto' or a (N1, N2, N3) tuple, got {grid!r}.")
        grid_shape = resolve_grid_shape(cell_np, cutoff_ha)
    else:
        grid_shape = (int(grid[0]), int(grid[1]), int(grid[2]))
    n1, n2, n3 = grid_shape
    n_grid = n1 * n2 * n3

    cell = structure_data._cell_jnp.astype(jnp.float64)
    volume = jnp.abs(_det_3x3(cell))
    n_max = structure_data._n_max

    logger.info(
        f"MPC setup: cutoff = {float(cutoff_ha)} Ha, grid = {grid_shape}, n_max = {int(n_max)}; "
        "g_G is extrapolated from the N, 2N and 4N grids."
    )

    # -- f_G = g_G + h_G ------------------------------------------------
    g_G, discrepancy = _extrapolate_g_G(cell, grid_shape, L, n_max)
    g_norm, g2 = _reciprocal_vectors(cell, grid_shape)
    h_G = _hood_h_G(g_norm, L, volume)
    f_G = g_G + h_G
    f_0 = f_G[0, 0, 0]

    # -- rho_{A,G} ------------------------------------------------------
    geminal_data = hamiltonian_data.wavefunction_data.geminal_data
    identity_mos = (
        isinstance(geminal_data.orb_data_up_spin, MOs_data)
        and isinstance(geminal_data.orb_data_dn_spin, MOs_data)
        and geminal_data.orb_data_up_spin.identity
        and geminal_data.orb_data_dn_spin.identity
    )
    evaluate_density_orbitals = compute_AOs
    if isinstance(geminal_data.orb_data_up_spin, TiledOrbitalSet_data):
        # The provider already returns occupied orbitals after each local C
        # block. MPC needs their density, never a global coefficient matrix.
        aos_up = geminal_data.orb_data_up_spin
        aos_dn = geminal_data.orb_data_dn_spin
        coeff_up = coeff_dn = None
        occ_up = jnp.ones((geminal_data.num_electron_up,), dtype=jnp.float64)
        occ_dn = jnp.ones((geminal_data.num_electron_dn,), dtype=jnp.float64)
        evaluate_density_orbitals = compute_tiled_orbitals
    elif identity_mos:
        # Tiled B-spline orbitals are already the occupied supercell MOs.  Do
        # not materialise their O(N^2) identity coefficient matrices merely
        # to build the MPC trial density; pass the AO values through directly.
        aos_up = geminal_data.orb_data_up_spin.aos_data
        aos_dn = geminal_data.orb_data_dn_spin.aos_data
        if hasattr(geminal_data, "lambda_matrix"):
            lambda_matrix = cast_to_zone(geminal_data.lambda_matrix, "det_eval")
            paired = lambda_matrix[:, : geminal_data.orb_num_dn]
            unpaired = lambda_matrix[:, geminal_data.orb_num_dn :]
            occ_dn = jnp.abs(jnp.diagonal(paired))
            occ_up = occ_dn + jnp.sum(jnp.abs(unpaired), axis=1)
        else:
            # SD_data has no lambda leaf: every stored row is occupied once.
            occ_up = jnp.ones((geminal_data.num_electron_up,), dtype=jnp.float64)
            occ_dn = jnp.ones((geminal_data.num_electron_dn,), dtype=jnp.float64)
        coeff_up = coeff_dn = None
    else:
        (aos_up, coeff_up, occ_up, aos_dn, coeff_dn, occ_dn) = geminal_data.natural_orbital_density_coefficients_jnp(
            frozen_overlap=frozen_overlap
        )
    # Chunked over grid points: ``compute_AOs`` costs ~140 kB of peak
    # working memory per point (measured: 65 AOs x 64k points returns a
    # 33 MB array but peaks near 9 GB), so evaluating the density grid in
    # one shot is what actually dominates this function -- not the FFT
    # extrapolation. Chunking caps it at roughly
    # ``_AO_CHUNK * 140 kB``. The reduction to ``rho`` is per-point, so
    # chunking is exact.
    grid_coords = _grid_coords(structure_data, grid_shape)
    t_density = time.perf_counter()
    rho_pieces = []
    for start in range(0, grid_coords.shape[0], _AO_CHUNK):
        block = grid_coords[start : start + _AO_CHUNK]
        ao_up = evaluate_density_orbitals(aos_up, block)
        ao_dn = evaluate_density_orbitals(aos_dn, block)
        psi_up = ao_up if coeff_up is None else coeff_up @ ao_up
        psi_dn = ao_dn if coeff_dn is None else coeff_dn @ ao_dn
        rho_pieces.append(
            (occ_up[:, None] * jnp.abs(psi_up) ** 2).sum(axis=0) + (occ_dn[:, None] * jnp.abs(psi_dn) ** 2).sum(axis=0)
        )
    rho_flat = rho_pieces[0] if len(rho_pieces) == 1 else jnp.concatenate(rho_pieces, axis=0)
    n_electron = int(geminal_data.num_electron_up + geminal_data.num_electron_dn)
    dV = volume / n_grid
    rho_flat = rho_flat * (jnp.asarray(n_electron, dtype=jnp.float64) / (jnp.sum(rho_flat) * dV))
    rho_grid = rho_flat.reshape(n1, n2, n3)
    rho_grid.block_until_ready()
    logger.info(f"  MPC setup: trial density on the {grid_shape} grid done in {time.perf_counter() - t_density:.1f} sec.")

    # rho_{A,G} in the same convention as f_G: (1/N_grid) sum_nodes rho e^{iGr}.
    # This makes rho_{A,0} = <rho> = N_e / Omega exactly.
    rho_G = jnp.fft.fftn(rho_grid) / n_grid

    # -- K_G, V_LR, V_const ---------------------------------------------
    # G != 0: the Ewald Hartree kernel minus the model kernel.
    # G  = 0: 4 pi / (Omega G^2) is cancelled by the neutralising
    #         background, leaving -f_0. Together with V_const this
    #         reproduces the -N^2 f_0 / 2 of CASINO Eqs. (186)-(187) --
    #         see the module docstring.
    safe_g2 = jnp.where(g2 > 0.0, g2, 1.0)
    K_G = jnp.where(g2 > 0.0, 4.0 * jnp.pi / (volume * safe_g2) - f_G, -f_0)
    # Retain only the cutoff sphere (QMCPACK ``init_gvecs``). The grid is
    # finer than the sphere needs, so without this mask the corners of the
    # reciprocal box would enter the sum and the answer would depend on the
    # tabulation grid rather than on ``cutoff_ha`` alone.
    K_G = jnp.where(0.5 * g2 <= float(cutoff_ha), K_G, 0.0)

    # V_LR(r) = Omega sum_G K_G rho_G e^{-iGr}. ``ifftn`` carries
    # e^{+iGr} and a 1/N_grid factor; both rho and V_LR are real and the
    # kernel is even, so multiplying back by N_grid and taking the real
    # part gives the sum above.
    v_lr_grid = volume * jnp.real(jnp.fft.ifftn(K_G * rho_G) * n_grid)
    v_lr_coef = _periodic_cubic_bspline_coefficients(v_lr_grid)

    v_const = -0.5 * volume**2 * jnp.sum(K_G * jnp.abs(rho_G) ** 2)

    return MPC_data(
        cell=cell,
        origin=jnp.mean(structure_data._positions_cart_jnp.astype(jnp.float64), axis=0),
        n_max=int(n_max),
        grid_shape=grid_shape,
        v_lr_grid=v_lr_grid,
        v_lr_coef=v_lr_coef,
        v_const=v_const,
        L=float(L),
        cutoff_ha=float(cutoff_ha),
        f_G_discrepancy=discrepancy,
    )


def _grid_coords(structure_data: Structure_data, grid_shape: tuple[int, int, int]) -> jax.Array:
    """Cartesian coordinates of the FFT grid nodes, origin at the nuclear centroid.

    Anchoring on the centroid (rather than the cell corner) keeps every
    tabulated field translationally covariant with the nuclei. The energy
    is unaffected -- the density is periodic, so shifting the origin only
    relabels nodes -- but it keeps the convention identical to the
    interpolation lookup in :py:func:`_tricubic_bspline_interp`.
    """
    n1, n2, n3 = (int(x) for x in grid_shape)
    cell = structure_data._cell_jnp.astype(jnp.float64)
    origin = jnp.mean(structure_data._positions_cart_jnp.astype(jnp.float64), axis=0)
    f1 = jnp.arange(n1, dtype=jnp.float64) / n1
    f2 = jnp.arange(n2, dtype=jnp.float64) / n2
    f3 = jnp.arange(n3, dtype=jnp.float64) / n3
    G1, G2, G3 = jnp.meshgrid(f1, f2, f3, indexing="ij")
    frac = jnp.stack([G1.reshape(-1), G2.reshape(-1), G3.reshape(-1)], axis=-1)
    return origin + frac @ cell


def _periodic_cubic_bspline_coefficients(values_grid: jax.Array) -> jax.Array:
    r"""Coefficients of the periodic uniform cubic B-spline interpolating ``values_grid``.

    A uniform cubic B-spline ``S(x) = sum_j c_j B(x - j)`` takes the value
    ``(c_{j-1} + 4 c_j + c_{j+1}) / 6`` at node ``j``. Requiring ``S`` to
    interpolate the node values is therefore a cyclic tridiagonal system
    per axis, which under periodic boundaries is the circular convolution
    ``v = c * [1, 4, 1] / 6``. It is inverted exactly (to round-off) by
    dividing the DFT of ``v`` by the kernel's DFT
    ``(4 + 2 cos(2 pi k / N)) / 6``, separably along the three axes. The
    kernel never vanishes (its DFT lies in ``[1/3, 1]``), so the
    deconvolution is well conditioned.

    This is what einspline's ``create_UBspline_3d_d`` with ``PERIODIC``
    boundary conditions solves for QMCPACK's MPC spline.
    """
    n1, n2, n3 = (int(x) for x in values_grid.shape)

    def kernel_dft(n: int) -> jax.Array:
        k = jnp.asarray(_fft_int_freqs(n), dtype=jnp.float64)
        return (4.0 + 2.0 * jnp.cos(2.0 * jnp.pi * k / n)) / 6.0

    denom = kernel_dft(n1)[:, None, None] * kernel_dft(n2)[None, :, None] * kernel_dft(n3)[None, None, :]
    return jnp.real(jnp.fft.ifftn(jnp.fft.fftn(values_grid.astype(jnp.float64)) / denom))


def _cubic_bspline_weights(t: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    r"""The four uniform cubic B-spline basis weights at fractional offset ``t``.

    For a point between nodes ``i0`` and ``i0 + 1`` (``t in [0, 1)``) the
    non-zero basis functions are those centred on ``i0 - 1 .. i0 + 2``, with
    weights (in that order)

    .. math::
       \frac{(1-t)^3}{6},\quad \frac{3t^3 - 6t^2 + 4}{6},\quad
       \frac{-3t^3 + 3t^2 + 3t + 1}{6},\quad \frac{t^3}{6}

    which sum to one. The arithmetic is written in exactly this order in
    the Rust twin (``jqmc/rs/src/mpc.rs::cubic_bspline_weights``) so the two
    backends agree to round-off.
    """
    t2 = t * t
    t3 = t2 * t
    omt = 1.0 - t
    w0 = omt * omt * omt / 6.0
    w1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0
    w2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0
    w3 = t3 / 6.0
    return w0, w1, w2, w3


def _tricubic_bspline_interp(coef_grid: jax.Array, cell: jax.Array, origin: jax.Array, r_carts: jax.Array) -> jax.Array:
    """Periodic tricubic B-spline evaluation of a coefficient grid at Cartesian points.

    ``coef_grid`` must be the output of
    :py:func:`_periodic_cubic_bspline_coefficients`; the result then
    reproduces the tabulated node values exactly and is ``C^2`` between
    them. The lookup convention (``frac = ((r - origin) @ inv_cell) mod 1``,
    node ``i`` at fractional ``i / N``) is the one :py:func:`_grid_coords`
    tabulates on.
    """
    n1, n2, n3 = (int(x) for x in coef_grid.shape)
    inv_cell = _inv_3x3(cell)
    frac = ((r_carts - origin) @ inv_cell) % 1.0
    gx = frac[:, 0] * n1
    gy = frac[:, 1] * n2
    gz = frac[:, 2] * n3
    i0 = jnp.floor(gx).astype(jnp.int32)
    j0 = jnp.floor(gy).astype(jnp.int32)
    k0 = jnp.floor(gz).astype(jnp.int32)
    tx, ty, tz = gx - i0, gy - j0, gz - k0

    offs = jnp.arange(-1, 3, dtype=jnp.int32)
    ii = (i0[:, None] + offs[None, :]) % n1  # (P, 4)
    jj = (j0[:, None] + offs[None, :]) % n2
    kk = (k0[:, None] + offs[None, :]) % n3
    block = coef_grid[ii[:, :, None, None], jj[:, None, :, None], kk[:, None, None, :]]  # (P, 4, 4, 4)

    wx = jnp.stack(_cubic_bspline_weights(tx), axis=-1)  # (P, 4)
    wy = jnp.stack(_cubic_bspline_weights(ty), axis=-1)
    wz = jnp.stack(_cubic_bspline_weights(tz), axis=-1)
    # Contract z, then y, then x -- the same nesting as the Rust loops.
    return jnp.einsum("pa,pabc,pb,pc->p", wx, block, wy, wz)


def evaluate_v_lr(mpc_data: MPC_data, r_carts: jax.Array) -> jax.Array:
    r"""``V_LR`` at Cartesian points ``(P, 3)`` -- the production interpolator.

    Exactly the one-body lookup :py:func:`compute_mpc_potential_ee` sums
    over the electrons, exposed so its interpolation accuracy can be
    tested against the exact band-limited field it tabulates.
    """
    return _tricubic_bspline_interp(mpc_data.v_lr_coef, mpc_data.cell, mpc_data.origin, r_carts.astype(jnp.float64))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def compute_mpc_potential_ee(
    mpc_data: MPC_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> jax.Array:
    r"""Canonical MPC electron-electron energy for one configuration.

    .. math::
       V^\text{MPC}_{ee} = \sum_{i<j} \frac{1}{|r^\text{MI}_{ij}|}
                         + \sum_i V_\text{LR}(r_i) + V_\text{const}

    matching QMCPACK ``MPC::evaluate``. The pair term is the bare
    Coulomb at the **minimum image** -- a single nearest copy, no lattice
    sum and no screening.

    Args:
        mpc_data: Tables from :py:func:`build_mpc_data`.
        r_up_carts: ``(N_up, 3)`` fp64.
        r_dn_carts: ``(N_dn, 3)`` fp64.

    Returns:
        jax.Array: Scalar e-e energy (Ha), fp64.
    """
    cell = mpc_data.cell
    inv_cell = _inv_3x3(cell)
    n_max = mpc_data.n_max

    r_all = jnp.concatenate([r_up_carts.astype(jnp.float64), r_dn_carts.astype(jnp.float64)], axis=0)

    # Term 1: minimum-image pair sum. The i = j diagonal is masked out;
    # unlike an Ewald short-range sum there are no self-image terms to
    # keep, because f carries no lattice sum.
    d_ij = r_all[:, None, :] - r_all[None, :, :]
    d_ij = jax.vmap(jax.vmap(lambda d: _minimum_image_displacement(d, cell, inv_cell, n_max)))(d_ij)
    r2_ij = jnp.sum(d_ij**2, axis=-1)
    safe_r2 = jnp.where(r2_ij > 0.0, r2_ij, 1.0)
    inv_r = jnp.where(r2_ij > 0.0, 1.0 / jnp.sqrt(safe_r2), 0.0)
    term_pair = 0.5 * jnp.sum(inv_r)

    # Term 2: one-body long-range potential (periodic tricubic B-spline).
    term_lr = jnp.sum(_tricubic_bspline_interp(mpc_data.v_lr_coef, cell, mpc_data.origin, r_all))

    return term_pair + term_lr + mpc_data.v_const


def compute_mpc_correction_ee(
    mpc_data: MPC_data,
    ewald_data: Ewald_data,
    r_up_carts: jax.Array,
    r_dn_carts: jax.Array,
) -> jax.Array:
    r"""Finite-size correction ``V^MPC_ee - V^Ewald_ee`` for one configuration.

    This is the quantity to accumulate under the ``physical = no``
    convention (QMCPACK's ``Delta_MPC``): walkers are propagated under
    Ewald and this difference is averaged over the same configurations,
    giving ``E_MPC = E_Ewald + <Delta_MPC>``.

    Evaluating it as a difference on identical configurations -- rather
    than averaging the two channels separately -- makes the two terms
    almost perfectly correlated, so the correction carries a far smaller
    statistical error than either channel alone, and both numbers come
    out of a single run.

    Args:
        mpc_data: Tables from :py:func:`build_mpc_data`.
        ewald_data: ``coulomb_potential_data.ewald_data``.
        r_up_carts: ``(N_up, 3)`` fp64.
        r_dn_carts: ``(N_dn, 3)`` fp64.

    Returns:
        jax.Array: Scalar correction (Ha), fp64.
    """
    v_mpc = compute_mpc_potential_ee(mpc_data, r_up_carts, r_dn_carts)
    v_ewald = compute_ewald_potential_ee(r_up_carts, r_dn_carts, ewald_data)
    return v_mpc - v_ewald
